# $Id: makefile,v 1.74 1999/01/27 19:46:16 bsmith Exp bsmith $ 

ALL: lib 

DIRS	  = viewer draw error dll fileio memory objects time utils plog mpiuni
LOCDIR	  = src/sys/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

rs6000_time: rs6000_time.s
	-@if [ "${PARCH}" = "rs6000" ]; then \
	${CC} -c rs6000_time.s ; \
	${AR} cr ${LIBNAME} rs6000_time.o ; \
	${RM} -f rs6000_time.o; \
	fi


