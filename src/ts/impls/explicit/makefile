# $Id: makefile,v 1.8 1997/09/10 19:04:45 balay Exp balay $ 

ALL: lib

CFLAGS   =  ${PETSC_INCLUDE} ${PCONF} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = 
SOURCEF  =
SOURCEH  = 
OBJSC    = 
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/explicit/
DIRS     = euler

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

