#
#  All the Sparspack routines have been converted to C with the F2C 
# converter. We do not include in the distribution the original Fortran 77 
# code in the interest of saving disk space. 
#
#  All Sparsepack routines are prepended with SPARSEPACK to reduce
# name space pollution.
#
#  This code uses the original SPARSEPACK software that was released 
# into the public domain. Thus it can be used for any purpose. 
# IT IS NOT THE VERSION OF SPARSEPACK THAT REQUIRES A LICENSE

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = sp1wd.c spnd.c spqmd.c sprcm.c sorder.c sregis.c\
           degree.c  fnroot.c genqmd.c qmdqt.c rcm.c fn1wd.c gen1wd.c \
           genrcm.c qmdrch.c rootls.c fndsep.c gennd.c qmdmrg.c qmdupd.c
SOURCEH  = order.h 
OBJSC    = sp1wd.o spnd.o spqmd.o sprcm.o sorder.o sregis.o\
           degree.o  fnroot.o genqmd.o qmdqt.o rcm.o fn1wd.o gen1wd.o \
           genrcm.o qmdrch.o rootls.o fndsep.o gennd.o qmdmrg.o qmdupd.o
LIBBASE  = libpetscmat
DIRS     = amd
LOCDIR   = src/mat/order/
MANSEC   = MatOrderings

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
