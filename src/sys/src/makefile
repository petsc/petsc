# $Id: makefile,v 1.73 1999/01/12 23:14:08 bsmith Exp bsmith $ 

ALL: lib 

DIRS	  = error dll fileio memory objects time utils plog mpiuni
LOCDIR	  = src/sys/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

rs6000_time: rs6000_time.s
	-@if [ "${PARCH}" = "rs6000" ]; then \
	${CC} -c rs6000_time.s ; \
	${AR} cr ${LIBNAME} rs6000_time.o ; \
	${RM} -f rs6000_time.o; \
	fi


