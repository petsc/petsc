PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) -I.. $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = draw.c dline.c dlinew.c dlinegw.c dtext.c dtextv.c dtexts.c \
           dtextgs.c dpoint.c dpoints.c dviewp.c dcoor.c dpause.c dgpause.c \
           dgcoor.c dbuff.c dsetpause.c dflush.c dsflush.c dclear.c \
           dsclear.c drect.c dtri.c dmouse.c drawv.c
SOURCEF  =
SOURCEH  = 
OBJSC    = draw.o dline.o dlinew.o dlinegw.o dtext.o dtextv.o dtexts.o \
           dtextgs.o  dpoint.o dpoints.o dviewp.o dcoor.o dpause.o dgpause.o \
           dgcoor.o dbuff.o dsetpause.o dflush.o dsflush.o dclear.o \
           dsclear.o drect.o dtri.o dmouse.o drawv.o
OBJSF    =
LIBBASE  = libpetscdraw
MANSEC   = x
LOCDIR   = src/draw/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
