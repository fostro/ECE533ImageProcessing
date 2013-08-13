OBJDIR=$(PREFIX)/.objs
INCLUDES=
#LIBS=`pkg-config --cflags --libs MagickWand`
LIBS=-DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -I/usr/include/ImageMagick-6 -lMagickWand-6.Q16 -lMagickCore-6.Q16 -lm -lMagickCore-6.Q16
#CFLAGS=-g -Wall $(INCLUDES) $(LIBS)
CFLAGS=-g $(INCLUDES) $(LIBS)
NFLAGS=$(CFLAGS) -arch sm_20
#CC=g++
CC=nvcc

OBJS=$(patsubst %.cpp,$(OBJDIR)/%.o,$(SRCS))

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CC) $(CFLAGS) -MM $^ >> ./.depend;

include .depend
