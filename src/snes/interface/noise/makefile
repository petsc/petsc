# $Id: makefile,v 1.8 2001/08/22 18:01:58 balay Exp $ 

ALL: lib

#requiresscalar real 
#requiresprecision double

CFLAGS   =
FFLAGS   =
SOURCEC  = snesmfj2.c snesnoise.c snesdnest.c
SOURCEF  =
SOURCEH  = 
OBJSC    = snesmfj2.o snesnoise.o snesdnest.o
OBJSF    =
DIRS     = 
LIBBASE  = libpetscsnes
MANSEC   = SNES
LOCDIR   = src/snes/interface/noise

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

