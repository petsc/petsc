# $Id: makefile,v 1.27 1998/03/19 02:10:59 balay Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = axis.c lg.c dscatter.c hists.c zoom.c
SOURCEF  =
SOURCEH  = 
OBJSC    = axis.o lg.o dscatter.o hists.o zoom.o
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = x
LOCDIR   = src/draw/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
