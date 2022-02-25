-include ../../../petscdir.mk
#
#  All the Sparspack routines have been converted to C with the F2C
# converter.
#
#  All Sparsepack routines are prepended with SPARSEPACK to reduce
# name space pollution.
#
#  This code uses the original SPARSEPACK software that was released
# into the public domain. Thus it can be used for any purpose.
# IT IS NOT THE VERSION OF SPARSEPACK THAT REQUIRES A LICENSE

ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   = sp1wd.c spnd.c spqmd.c sprcm.c sorder.c spectral.c sregis.c degree.c  fnroot.c genqmd.c qmdqt.c rcm.c fn1wd.c gen1wd.c genrcm.c qmdrch.c rootls.c fndsep.c gennd.c qmdmrg.c qmdupd.c wbm.c
SOURCEH   = ../../../include/petsc/private/matorderimpl.h
LIBBASE   = libpetscmat
DIRS      = amd metisnd
LOCDIR    = src/mat/order/
MANSEC    = Mat
SUBMANSEC = MatOrderings

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
