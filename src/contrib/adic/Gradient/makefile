# $Id: makefile,v 1.14 2001/08/22 18:02:45 balay Exp $ 

ALL: lib

CFLAGS   = $(ADIC_DEFINES) -I$(PETSC_DIR)/include/adic
FFLAGS   = 
SOURCEC  = ad_grad.c ad_grad_daxpy.c ad_utils.c
SOURCEF  =
SOURCEH  = ../../../../include/adic/ad_grad_macro_axpys.h ../../../../include/adic/ad_grad_daxpy.h \
           ../../../../include/adic/ad_grad.h ../../../../include/adic/ad_utils.h
OBJSC    = ad_grad.o ad_grad_daxpy.o ad_utils.o
OBJSF    =
LIBBASE  = libpetsc
DIRS     = 
LOCDIR   = src/contrib/adic/adic/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

