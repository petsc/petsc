# $Id: makefile,v 1.26 1997/11/09 03:59:46 bsmith Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = axis.c lg.c dscatter.c hists.c zoom.c
SOURCEF  =
SOURCEH  = 
OBJSC    = axis.o lg.o dscatter.o hists.o zoom.o
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = x
LOCDIR   = src/draw/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
