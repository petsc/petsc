# $Id: makefile,v 1.3 2000/01/10 03:09:38 knepley Exp $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = bilinear.c bilinearreg.c bilinearserialize.c bilinearregall.c initializepackage.c dlregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ${SOURCEC:.c=.o}
OBJSF    =
LIBBASE  = libpetscbilinear
MANSEC   = Bilinear
LOCDIR   = src/bilinear/interface/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
