ITOOLSDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = ij.c sp1wd.c spinver.c spnd.c spqmd.c sprcm.c
SOURCEF  = degree.f  fnroot.f genqmd.f qmdqt.f rcm.f fn1wd.f gen1wd.f \
           genrcm.f qmdrch.f rootls.f fndsep.f gennd.f qmdmrg.f qmdupd.f
WSOURCEC = 
SOURCEH  = 
OBJSC    = ij.o  sp1wd.o spinver.o spnd.o spqmd.o sprcm.o
WOBJS    = 
OBJSF    = degree.o  fnroot.o genqmd.o qmdqt.o rcm.o fn1wd.o gen1wd.o \
           genrcm.o qmdrch.o rootls.o fndsep.o gennd.o qmdmrg.o qmdupd.o
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
MANSEC   = 2

include $(ITOOLSDIR)/bmake/$(PARCH)
