IPETSCDIR = ../../../../../../
#
#  All the Sparspak routines have been converted to C with the F2C 
# converter. We do not include in the distribution the original Fortran 77 
# code in the interest of saving disk space. To link with that instead of
# the C versions, uncomment SOURCEF and OBJSF and comment the appropriate 
# parts of the corresponding C lines. The Fortran 77 Sparsepack source 
# may be obtained from Netlib or petsc-maint@mcs.anl.gov, ask for 
# petsc/surplus/sparsepak.tar.Z
#
ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. \
           -I$(IPETSCDIR)/pinclude $(CONF) \
           -D__DIR__='"mat/impls/aij/seq/order/"'
SOURCEC  = sp1wd.c spinver.c spnd.c spqmd.c sprcm.c \
           degree.c  fnroot.c genqmd.c qmdqt.c rcm.c fn1wd.c gen1wd.c \
           genrcm.c qmdrch.c rootls.c fndsep.c gennd.c qmdmrg.c qmdupd.c
#SOURCEF  = degree.f  fnroot.f genqmd.f qmdqt.f rcm.f fn1wd.f gen1wd.f \
#           genrcm.f qmdrch.f rootls.f fndsep.f gennd.f qmdmrg.f qmdupd.f
WSOURCEC = 
SOURCEH  = order.h 
OBJSC    = sp1wd.o spinver.o spnd.o spqmd.o sprcm.o \
           degree.o  fnroot.o genqmd.o qmdqt.o rcm.o fn1wd.o gen1wd.o \
           genrcm.o qmdrch.o rootls.o fndsep.o gennd.o qmdmrg.o qmdupd.o
WOBJS    = 
#OBJSF    = degree.o  fnroot.o genqmd.o qmdqt.o rcm.o fn1wd.o gen1wd.o \
#           genrcm.o qmdrch.o rootls.o fndsep.o gennd.o qmdmrg.o qmdupd.o
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
MANSEC   = 2

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
