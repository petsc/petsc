# $Id: makefile,v 1.45 2001/08/22 18:00:07 balay Exp $ 
#
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = 
LOCDIR		 = src/contrib/dvr
EXAMPLESC	 = 
EXAMPLESF	 = dvr.F

include ${PETSC_DIR}/bmake/common/base


dvr: dvr.o  chkopts
	-${FLINKER} -o dvr dvr.o  ${PETSC_FORTRAN_LIB} ${PETSC_VEC_LIB}
	${RM} -f dvr.o


TESTEXAMPLES_3	 = dvr.F

include ${PETSC_DIR}/bmake/common/test
