#include "Configurations.h"
#include "Utils.h"



// Function for swaping between little and big endian
void BigEndianToLittleEndian(unsigned short *input_img)
{
  for(int i=0; i<COLUMNS*LINES*BANDS; i++)
    {
      input_img[i] = swap(input_img[i]);
    }
}

short unsigned int swap(unsigned short input)
{
  unsigned short int output;
  unsigned char *outputPointer = (unsigned char *)&output;
  unsigned char *inputPointer = (unsigned char *)&input;
  outputPointer[0] = inputPointer[1];
  outputPointer[1] = inputPointer[0];
  return output;
}



// Function for reordering the image from BIL to BIP
void BILtoBIP(unsigned short *input_img)
{
  short unsigned int *Img_BIP = (short unsigned int *)malloc(sizeof(short unsigned int)*COLUMNS*BANDS*LINES);

  unsigned int pixel;
  unsigned int elementIndex = 0;

  for(int pixelR=0; pixelR<LINES; pixelR++){
    for(int band=0; band<BANDS; band++){
      for(int pixelC=0; pixelC<COLUMNS; pixelC++){	
	pixel = pixelR*COLUMNS + pixelC;
	Img_BIP[pixel*BANDS + band] = input_img[elementIndex];
	elementIndex++;
      }
    }
  }

  for(int elementIndex=0; elementIndex<COLUMNS*BANDS*LINES; elementIndex++){
    input_img[elementIndex] = Img_BIP[elementIndex];
  }
}



// Function for reordering the image from BSQ to BIP
void BSQtoBIP(unsigned short *input_img)
{
  unsigned int nPixels = COLUMNS * LINES;
  short unsigned int *Img_BIP 	= (short unsigned int *)malloc(sizeof(short unsigned int)*COLUMNS*BANDS*LINES);

  for(int band=0; band<BANDS; band++){
    for(unsigned int pixel=0; pixel<nPixels; pixel++){
      Img_BIP[pixel*BANDS + band] = input_img[band*nPixels + pixel];
    }
  }

  for(int elementIndex=0; elementIndex<COLUMNS*LINES*BANDS; elementIndex++){
    input_img[elementIndex] = Img_BIP[elementIndex];
  }
}




