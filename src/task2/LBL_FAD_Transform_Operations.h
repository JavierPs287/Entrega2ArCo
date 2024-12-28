#ifndef __LBL_FAD_OPERATIONS__
#define __LBL_FAD_OPERATIONS__

#include "Configurations.h"


void LBL_FAD_Stage1(int blockIndex, unsigned short *ImgRef,
		    unsigned int &n_indexes, unsigned short *bg_indexes,
		    unsigned short * bg_block);

void LBL_FAD_Stage2(unsigned short *ImgRef, int *centroid,
		    int qMatrix[][BANDS], int uMatrix[][BANDS],
		    unsigned char &numQU, long long &tau, int blockSize);

void LBL_FAD_Stage3_4(int blockIndex, unsigned short *ImgRef,
		      unsigned char numQU, long long tau,
		      int *centroid,
		      int qMatrix[][BANDS], int uMatrix[][BANDS],
		      bool *block_ad_map);



void averagePixel(unsigned short *ImgRef, int *centroid, int blockSize);

void duplicateAndCentralizeImg(unsigned short *ImgRef, short *Img, int *centroid, int blockSize);



void brightness(short *Img, int &maxIndex, long long &maxBrightness, unsigned char iter, long long *brightness_iter1, int blockSize);

void quVectors(short *Img, int &maxIndex, long long &maxBrightness, int *qVector, int *uVector);

void projectingImg(short *Img, int *projection, int *uVector, int blockSize);

void subtractingInformation(short *Img, int *projection, int *qVector, int blockSize);


void stop_condition(long long &tau, unsigned char &numQU, unsigned short maxIndex, long long maxBrightness, long long *brightnessIter1, bool &stop, unsigned short &out_index, const int ALPHA);


void brightnessAD(short *Img, long long tau, bool *outAnomaly);


#endif
