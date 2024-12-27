#ifndef __CFG_LBL_FAD__
#define __CFG_LBL_FAD__

// Input parameters
#define OUTPUT_DIR "output"
#define OUTPUT_FILE_NAME "admap.txt"
#define INPUT_FILE_NAME "mission.bin"

#define DR_PROJECTIONS 16
#define BLOCK_SIZE 1024

#define BLOCKS_BACKGROUND 100
#define MAX_ENDMEMBERS_PER_BLOCK 8

#define COLUMNS 1024
#define LINES 825
#define BANDS 160
#define PMAX (BANDS*BLOCK_SIZE)
#define MAX_NUMQU 32

const int ALPHA=1;


// Variables values obtained from the input parameters
#define NUMBER_OF_PIXELS (COLUMNS*LINES)
#define IMAGE_SIZE (COLUMNS*LINES*BANDS)


// libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#endif
