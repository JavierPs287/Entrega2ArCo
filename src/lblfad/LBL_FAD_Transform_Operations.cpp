#include "LBL_FAD_Transform_Operations.h"

#ifdef VERBOSE
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
#endif


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
void averagePixel(unsigned short *ImgRef, int *centroid, int blockSize)
{
  for(int band=0; band<BANDS; band++){
    centroid[band] = 0;

    for(int pixel=0; pixel<blockSize; pixel++){
      centroid[band] += ImgRef[pixel*BANDS + band];
    }

    centroid[band] = centroid[band] / blockSize;
  }
}



// Subtracting the centroid pixel and create the Auxiliary Img
void duplicateAndCentralizeImg(unsigned short *ImgRef, short *Img, int *centroid, int blockSize)
{
  for(int pixel=0; pixel<blockSize; pixel++){
    for(int band=0; band<BANDS; band++){
      Img[pixel*BANDS + band] = ((short)(ImgRef[pixel*BANDS + band]) - (short)(centroid[band]))<<2;
    }
  }
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
void quVectors(short *Img, int &maxIndex, long long &maxBrightness, int *qVector, int *uVector)
{

  for(int band=0; band<BANDS; band++){
    // qVector
    qVector[band] = Img[maxIndex*BANDS + band];

    // uVector
    long long ImgValueLong = Img[maxIndex*BANDS + band];
    ImgValueLong = ImgValueLong << 28; // -1 Considering the sign
    uVector[band] = ImgValueLong / (maxBrightness >> 16);
  }
}



// Calculating the projection of "Img" into "uVector"
void projectingImg(short *Img, int *projection, int *uVector, int blockSize)
{
  long long uValueLong;
  long long ImgValueLong;
  long long projectionValueLong;

  for(int pixel=0; pixel<blockSize; pixel++){
    projectionValueLong = 0;
    for(int band=0; band<BANDS; band++){
      uValueLong = uVector[band];
      ImgValueLong = Img[pixel*BANDS + band];

      projectionValueLong += uValueLong * ImgValueLong;
    }
    projection[pixel] = projectionValueLong>>4;
  }
}



// Subtracting the information contained in projection from the image
void subtractingInformation(short *Img, int *projection, int *qVector, int blockSize)
{
  long long longValueToSubtract;
  long long qValueLong;
  long long projectionValueLong;

  for(int pixel=0; pixel<blockSize; pixel++) {
    for(int band=0; band<BANDS; band++)	{
      qValueLong = qVector[band];
      projectionValueLong = projection[pixel];

      longValueToSubtract   = qValueLong * projectionValueLong;
      Img[pixel*BANDS + band] -= longValueToSubtract >> 28;
    }
  }
}


void stop_condition(long long &tau, unsigned char &numQU, unsigned short maxIndex, long long maxBrightness, long long *brightnessIter1, bool &stop, unsigned short &out_index, const int ALPHA)
{
  unsigned long long sf;
  const int _alpha = ALPHA<<30;
  long long imaxBrightness;
  stop = false;

  imaxBrightness = (brightnessIter1[maxIndex])>>16;
  sf = (unsigned long long) (maxBrightness<<14)/imaxBrightness;


  if (sf*100 < (_alpha))
    {
      stop = true;
    }else{
    numQU++; //Number of QU extracted in Stage 2 for being used in Stage 3.
    tau = maxBrightness; //Tau as threshold in Stage 4 for anomaly detection.
    out_index = maxIndex;
  }
}


// Anomaly Detection
void brightnessAD(short *Img, long long tau, bool *outAnomaly)
{
  long long ImgValueLong;
  long long threshold = (tau<<1) - (tau>>1);
  for(int pixel=0; pixel<BLOCK_SIZE; pixel++) {
    long long actualBrightness = 0;
    for(int band=0; band<BANDS; band++)	{
      ImgValueLong = (long long) Img[pixel*BANDS + band];
      long long aux = (ImgValueLong * ImgValueLong)<<12;
      actualBrightness += aux;
    }

    if(actualBrightness > threshold)
      outAnomaly[pixel] = true;
    else
      outAnomaly[pixel] = false;
  }
}
