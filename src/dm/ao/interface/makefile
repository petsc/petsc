# $Id: makefile,v 1.21 2001/08/22 18:02:26 balay Exp $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = ao.c aodata.c aodataalias.c dlregis.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = ${SOURCEC:.c=.o}
OBJSF    =
LIBBASE  = libpetscdm
MANSEC   = AO
LOCDIR   = src/dm/ao/interface/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
