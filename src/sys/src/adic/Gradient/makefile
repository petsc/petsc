#
#    ad_grad.c and ad_utils.c are automatically included into application codes that
#  use adiC. They are ALSO compiled into the PETSc libraries so that apps that do not 
#  use adiC can still link.
#
ALL: lib

CFLAGS   = $(ADIC_DEFINES) -I$(PETSC_DIR)/include/adic
FFLAGS   = 
SOURCEC  = ad_grad_daxpy.c ad_grad.c ad_utils.c
SOURCEF  =
OBJSC    = ad_grad_daxpy.o ad_grad.o ad_utils.o
OBJSF    =
LIBBASE  = libpetsc
DIRS     = 
LOCDIR   = src/sys/src/adic/Gradient/
MANSEC   = Sys

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

