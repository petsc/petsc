# $Id: makefile,v 1.33 1999/02/02 21:26:49 balay Exp bsmith $ 
#
#  All the Sparspak routines have been converted to C with the F2C 
# converter. We do not include in the distribution the original Fortran 77 
# code in the interest of saving disk space. 
#
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = sp1wd.c spinver.c spnd.c spqmd.c sprcm.c sorder.c sregis.c\
           degree.c  fnroot.c genqmd.c qmdqt.c rcm.c fn1wd.c gen1wd.c \
           genrcm.c qmdrch.c rootls.c fndsep.c gennd.c qmdmrg.c qmdupd.c
SOURCEH  = order.h 
OBJSC    = sp1wd.o spinver.o spnd.o spqmd.o sprcm.o sorder.o sregis.o\
           degree.o  fnroot.o genqmd.o qmdqt.o rcm.o fn1wd.o gen1wd.o \
           genrcm.o qmdrch.o rootls.o fndsep.o gennd.o qmdmrg.o qmdupd.o
LIBBASE  = libpetscmat
DOCS     = readme
LOCDIR   = src/mat/impls/order/
MANSEC   = MatOrderings

runexamples:

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
