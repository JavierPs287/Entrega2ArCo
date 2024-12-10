#ifndef __UTILS__
#define __UTILS__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;


// Function for swaping between little and big endian
void BigEndianToLittleEndian(unsigned short *input_img);
short unsigned int swap(unsigned short input);

// Function for reordering the image from BIL to BIP
void BILtoBIP(unsigned short *input_img);

// Function for reordering the image from BSQ to BIP
void BSQtoBIP(unsigned short *input_img);

#endif
