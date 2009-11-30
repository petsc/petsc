
CFLAGS           = ${BLOPEX_INCLUDE} 
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/contrib/blopex/driver_fiedler/
EXAMPLESC        = driver_fiedler.c
EXAMPLESF        = 
MANSEC           = KSP


include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

driver_fiedler: driver_fiedler.o  chkopts
	-${CLINKER} -o driver_fiedler driver_fiedler.o  ${PETSC_KSP_LIB}
	${RM} driver_fiedler.o


