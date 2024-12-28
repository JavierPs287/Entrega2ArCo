#include "LBL_FAD_Transform_Operations.h"

#include <CL/sycl.hpp>
using namespace sycl;

#ifdef VERBOSE
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
#endif

queue q{default_selector_v};

void LBL_FAD_Stage1(int blockIndex, unsigned short *ImgRef,
		    unsigned int &n_indexes, unsigned short *bg_indexes,
		    unsigned short *bg_block)
{
  short Img[BLOCK_SIZE * BANDS];
  int centroid[BANDS];
  int qVector[BANDS];
  int uVector[BANDS];
  int projection[BLOCK_SIZE];

  int maxIndex;
  long long maxBrightness;
  long long brightness_iter1[BLOCK_SIZE];
  unsigned short out_index;

  long long tau_nu; // Not used
  unsigned char numQU_nu = 0; // Not used

  bool stop;

  // Calculating the centroid pixel
  averagePixel(ImgRef, centroid, BLOCK_SIZE);

  // Cast ImgRef to int values copy it in Img and subtract the centroid pixel
  duplicateAndCentralizeImg(ImgRef, Img, centroid, BLOCK_SIZE);

  // Extracting the representative pixels and their projections
  for(int iter=0; iter<PMAX; iter++){
    // Calculating the brightness of each pixel
    brightness(Img, maxIndex, maxBrightness, iter, brightness_iter1, BLOCK_SIZE);

    stop_condition(tau_nu, numQU_nu, maxIndex, maxBrightness, brightness_iter1, stop, out_index, ALPHA);

    if (stop)
      break;
    else{
      bg_indexes[n_indexes] = out_index;
      bg_block[n_indexes++] = blockIndex;
    }

    // Calculating "qVector" and "uVector"
    quVectors(Img, maxIndex, maxBrightness, qVector, uVector);

#ifdef VERBOSE_DBG
    if (blockIndex == BLOCK_DBG || BLOCK_DBG==-1){
      stringstream outputFilename;
      outputFilename << OUTPUT_DIR << "/Stage1.txt";
      ofstream outputFile;
      outputFile.open(outputFilename.str(), std::ios::app);
      outputFile <<  maxIndex << endl;
      outputFile.close();
    }
#endif

    // Calculating the projection of "Img" into "uVector"
    projectingImg(Img, projection, uVector, BLOCK_SIZE);

    // Subtracting the information contained in projection from the image
    subtractingInformation(Img, projection, qVector, BLOCK_SIZE);
  }
}


void LBL_FAD_Stage2(unsigned short *ImgRef, int *centroid,
		    int qMatrix[][BANDS], int uMatrix[][BANDS],
		    unsigned char &numQU, long long &tau, int blockSize)
{
  short Img[BLOCK_SIZE * BANDS];	// 14.02 bits
  int qVector[BANDS];					// 20.12 bits
  int uVector[BANDS];					// 02.30 bits
  int projection[BLOCK_SIZE];				// 02.30 bits

  int maxIndex;				// 32.00 bits
  long long maxBrightness;  	// 48.16 bits
  long long brightness_iter1[BLOCK_SIZE];
  unsigned short out_index;

  bool stop;

  // Calculating the centroid pixel
  averagePixel(ImgRef, centroid, blockSize);

  // Cast ImgRef to int values (16.16), copy it in Img and subtract the centroid pixel
  duplicateAndCentralizeImg(ImgRef, Img, centroid, blockSize);

  // Extracting the representative pixels and their projections
  for(int iter=0; iter<PMAX; iter++){
    // Calculating the brightness of each pixel
    brightness(Img, maxIndex, maxBrightness, iter, brightness_iter1, blockSize);

    stop_condition(tau, numQU, maxIndex, maxBrightness, brightness_iter1, stop, out_index, ALPHA);

    if (stop)
      break;

    // Calculating "qVector" and "uVector"
    quVectors(Img, maxIndex, maxBrightness, qVector, uVector);

    for(unsigned char _it=0; _it<BANDS; _it++){
      qMatrix[numQU-1][_it] = qVector[_it];
      uMatrix[numQU-1][_it] = uVector[_it];
    }


#ifdef VERBOSE_DBG
    stringstream outputFilename;
    outputFilename << OUTPUT_DIR << "/Stage2.txt";
    ofstream outputFile;
    outputFile.open(outputFilename.str(), std::ios::app);
    outputFile <<  maxIndex << endl;
    outputFile.close();
#endif

    // Calculating the projection of "Img" into "uVector"
    projectingImg(Img, projection, uVector, blockSize);

    // Subtracting the information contained in projection from the image
    subtractingInformation(Img, projection, qVector, blockSize);
  }
}


void LBL_FAD_Stage3_4(int blockIndex, unsigned short *ImgRef,
		  unsigned char numQU, long long tau,
		  int *centroid,
		  int qMatrix[][BANDS], int uMatrix[][BANDS],
		  bool *block_ad_map)
{
  short Img[BLOCK_SIZE * BANDS];
  int projection[BLOCK_SIZE];

  // Cast ImgRef to int values, copy it in Img and subtract the centroid pixel
  duplicateAndCentralizeImg(ImgRef, Img, centroid, BLOCK_SIZE);

  // Extracting the representative pixels and their projections
  for(unsigned char iter=0; iter<numQU; iter++){
    // Calculating the projection of "Img" into "uVector"
    projectingImg(Img, projection, uMatrix[iter], BLOCK_SIZE);

    // Subtracting the information contained in projection from the image
    subtractingInformation(Img, projection, qMatrix[iter], BLOCK_SIZE);
  }

  brightnessAD(Img, tau, block_ad_map);
}





// ---------- HyperLCA Operators ---------- //


// Calculating the average pixel of the frame (centroid pixel)
void averagePixel(unsigned short* ImgRef, int* centroid, int blockSize) {
    buffer<unsigned short, 1> imgRefBuf(ImgRef, range<1>(blockSize * BANDS));
    buffer<int, 1> centroidBuf(centroid, range<1>(BANDS));

    q.submit([&](handler& h) {
        auto imgRef = imgRefBuf.get_access<access::mode::read>(h);
        auto centroid = centroidBuf.get_access<access::mode::write>(h);

        h.parallel_for(range<1>(BANDS), [=](id<1> band) {
            int sum = 0;
            for (int pixel = 0; pixel < blockSize; ++pixel) {
                sum += imgRef[pixel * BANDS + band];
            }
            centroid[band] = sum / blockSize;
        });
    });
}




// Subtracting the centroid pixel and create the Auxiliary Img
void duplicateAndCentralizeImg(unsigned short* ImgRef, short* Img, int* centroid, int blockSize) {
    buffer<unsigned short, 1> imgRefBuf(ImgRef, range<1>(blockSize * BANDS));
    buffer<short, 1> imgBuf(Img, range<1>(blockSize * BANDS));
    buffer<int, 1> centroidBuf(centroid, range<1>(BANDS));

    q.submit([&](handler& h) {
        auto imgRef = imgRefBuf.get_access<access::mode::read>(h);
        auto img = imgBuf.get_access<access::mode::write>(h);
        auto centroid = centroidBuf.get_access<access::mode::read>(h);

        h.parallel_for(range<1>(blockSize * BANDS), [=](id<1> idx) {
            int pixel = idx[0] / BANDS;
            int band = idx[0] % BANDS;
            img[idx] = ((short)imgRef[idx] - (short)centroid[band]) << 2;
        });
    });
}





// Calculating the brightness of each pixel
void brightness(short *Img, int &maxIndex, long long &maxBrightness, unsigned char iter, long long *brightness_iter1, int blockSize)
{
  maxBrightness = 0;
  maxIndex = 0;

  //unsigned long long actualBrightness;
  long long actualBrightness;
  long long ImgValueLong;

  for(int pixel=0; pixel<blockSize; pixel++){
    // Computing the brightness of one pixel
    actualBrightness = 0;
    for(int band=0; band<BANDS; band++){
      ImgValueLong = Img[pixel*BANDS + band];
      actualBrightness += (ImgValueLong * ImgValueLong)<<12;
    }

    // Comparing with the maximum value obtained
    if(actualBrightness > maxBrightness){
      maxIndex = pixel;
      maxBrightness = actualBrightness;
    }
    if (iter==0)
      brightness_iter1[pixel] = actualBrightness;
  }
}







// Calculating "qVector" and "uVector"
void quVectors(short* Img, int &maxIndex, long long &maxBrightness, int* qVector, int* uVector) {
    buffer<short, 1> imgBuf(Img, range<1>(maxIndex * BANDS + BANDS)); // Aseguramos que Img tiene espacio suficiente
    buffer<int, 1> qVectorBuf(qVector, range<1>(BANDS));
    buffer<int, 1> uVectorBuf(uVector, range<1>(BANDS));

    q.submit([&](handler& h) {
        auto img = imgBuf.get_access<access::mode::read>(h);
        auto qVec = qVectorBuf.get_access<access::mode::write>(h);
        auto uVec = uVectorBuf.get_access<access::mode::write>(h);

        h.parallel_for(range<1>(BANDS), [=](id<1> band) {
            // qVector
            qVec[band] = img[maxIndex * BANDS + band];

            // uVector
            long long ImgValueLong = img[maxIndex * BANDS + band];
            ImgValueLong = ImgValueLong << 28;
            uVec[band] = ImgValueLong / (maxBrightness >> 16);
        });
    });

    q.wait();
}





// Calculating the projection of "Img" into "uVector"
void projectingImg(short* Img, int* projection, int* uVector, int blockSize) {
    buffer<short, 1> imgBuf(Img, range<1>(blockSize * BANDS));
    buffer<int, 1> projectionBuf(projection, range<1>(blockSize));
    buffer<int, 1> uVectorBuf(uVector, range<1>(BANDS));

    q.submit([&](handler& h) {
        auto img = imgBuf.get_access<access::mode::read>(h);
        auto proj = projectionBuf.get_access<access::mode::write>(h);
        auto uV = uVectorBuf.get_access<access::mode::read>(h);

        h.parallel_for(range<1>(blockSize), [=](id<1> pixel) {
            long long projectionValue = 0;
            for (int band = 0; band < BANDS; ++band) {
                projectionValue += uV[band] * img[pixel[0] * BANDS + band];
            }
            proj[pixel] = projectionValue >> 4;
        });
    });
}





// Subtracting the information contained in projection from the image
void subtractingInformation(short *Img, int *projection, int *qVector, int blockSize) {
    buffer<short, 1> imgBuffer(Img, range<1>(blockSize * BANDS));
    buffer<int, 1> projectionBuffer(projection, range<1>(blockSize));
    buffer<int, 1> qVectorBuffer(qVector, range<1>(BANDS));

    queue q{default_selector_v};

    q.submit([&](handler &h) {
        auto img = imgBuffer.get_access<access::mode::read_write>(h);
        auto proj = projectionBuffer.get_access<access::mode::read>(h);
        auto qV = qVectorBuffer.get_access<access::mode::read>(h);

        h.parallel_for(range<2>(blockSize, BANDS), [=](id<2> idx) {
            int pixel = idx[0];
            int band = idx[1];

            long long projectionValue = proj[pixel];
            long long qValue = qV[band];
            long long valueToSubtract = qValue * projectionValue;

            img[pixel * BANDS + band] -= valueToSubtract >> 28;
        });
    });
    q.wait();
}



void stop_condition(long long& tau, unsigned char& numQU, unsigned short maxIndex, long long maxBrightness, long long* brightnessIter1, bool& stop, unsigned short& out_index, int ALPHA) {
    buffer<long long, 1> brightnessBuf(brightnessIter1, range<1>(BLOCK_SIZE));
    buffer<bool, 1> stopBuf(&stop, range<1>(1));
    buffer<unsigned short, 1> outIndexBuf(&out_index, range<1>(1));
    buffer<long long, 1> tauBuf(&tau, range<1>(1));

    const int _alpha = ALPHA << 30;

    queue q{default_selector_v};

    q.submit([&](handler& h) {
        auto brightness = brightnessBuf.get_access<access::mode::read>(h);
        auto stopAcc = stopBuf.get_access<access::mode::write>(h);
        auto outIndex = outIndexBuf.get_access<access::mode::write>(h);
        auto tauAcc = tauBuf.get_access<access::mode::write>(h);

        h.single_task([=]() {
            long long imaxBrightness = brightness[maxIndex] >> 16;
            unsigned long long sf = (maxBrightness << 14) / imaxBrightness;

            if (sf * 100 < _alpha) {
                stopAcc[0] = true;
            } else {
                stopAcc[0] = false;
                tauAcc[0] = maxBrightness;
                outIndex[0] = maxIndex;
            }
        });
    });
    q.wait();
}





// Anomaly Detection
void brightnessAD(short *Img, long long tau, bool *outAnomaly) {
    buffer<short, 1> imgBuffer(Img, range<1>(BLOCK_SIZE * BANDS));
    buffer<bool, 1> anomalyBuffer(outAnomaly, range<1>(BLOCK_SIZE));

    long long threshold = (tau << 1) - (tau >> 1);

    queue q{default_selector_v};

    q.submit([&](handler &h) {
        auto img = imgBuffer.get_access<access::mode::read>(h);
        auto anomaly = anomalyBuffer.get_access<access::mode::write>(h);

        h.parallel_for(range<1>(BLOCK_SIZE), [=](id<1> pixel) {
            long long actualBrightness = 0;
            for (int band = 0; band < BANDS; ++band) {
                long long value = (long long)img[pixel * BANDS + band];
                actualBrightness += (value * value) << 12;
            }

            anomaly[pixel] = (actualBrightness > threshold);
        });
    });
    q.wait();
}

