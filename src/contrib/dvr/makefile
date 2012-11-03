# $Id: makefile,v 1.45 2001/08/22 18:00:07 balay Exp $ 
#
All:dvr
CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = 
LOCDIR		 = src/contrib/dvr/
EXAMPLESC	 = 
EXAMPLESF	 = dvr.F mxm44f2g.F basic.F
DVR_OBJSF        = $(EXAMPLESF:.F=.o)


include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

dvr: $(DVR_OBJSF) chkopts
	-${FLINKER} -o dvr $(DVR_OBJSF) ${PETSC_VEC_LIB}
	${RM} -f dvr.o


TESTEXAMPLES_FORTRAN	 = dvr.F

include ${PETSC_DIR}/conf/test
