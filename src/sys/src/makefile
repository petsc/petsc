# $Id: makefile,v 1.75 1999/01/28 01:59:55 bsmith Exp balay $ 

ALL: lib 

DIRS	  = viewer draw error dll fileio memory objects time utils plog mpiuni
LOCDIR	  = src/sys/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

rs6000_asmtime: rs6000_asmtime.s
	-@if [ "${PARCH}" = "rs6000" ]; then \
	${CC} -c rs6000_asmtime.s ; \
	${AR} cr ${LIBNAME} rs6000_asmtime.o ; \
	${RM} -f rs6000_asmtime.o; \
	fi


