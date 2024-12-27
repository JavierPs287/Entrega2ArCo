PRJ = LBL_FAD
CC = icpx

SRCS  = Main_Detector_HW.cpp LBL_FAD_Transform_Operations.cpp Utils.cpp
OBJS = $(SRCS:.c=.o)


# Please, check that -I path is correct
CPPFLAGS += -O0 -g -I. -I$(HOME)/intel/oneapi/vtune/latest/sdk/include/ -fsycl
#LDFLAGS += -D VERBOSE -D VERBOSE_DBG -D BLOCK_DBG=-1


all : $(PRJ)

%.o : %.c
	$(CC) $(CPPFLAGS) $(LDFLAGS) -c $^

$(PRJ): $(OBJS)
	$(CC) $(LDFLAGS) $(CPPFLAGS) $^ -o $@

run: $(PRJ)
	./$(PRJ)


clean:
	$(RM) -rf $(PRJ) *~ *.o admap.bin output

cleanall: clean
	$(RM) -rf mission.bin
