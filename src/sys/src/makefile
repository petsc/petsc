# $Id: makefile,v 1.76 1999/04/18 07:12:01 balay Exp bsmith $ 

ALL: lib 

DIRS	  = viewer draw error dll fileio memory objects time utils plog mpiuni matlabengine
LOCDIR	  = src/sys/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

rs6000_asmtime: rs6000_asmtime.s
	-@if [ "${PARCH}" = "rs6000" ]; then \
	${CC} -c rs6000_asmtime.s ; \
	${AR} cr ${LIBNAME} rs6000_asmtime.o ; \
	${RM} -f rs6000_asmtime.o; \
	fi


