
CFLAGS           = ${BLOPEX_INCLUDE} 
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/contrib/blopex/driver_diag/
EXAMPLESC        = driver_diag.c
EXAMPLESF        = 
MANSEC           = KSP


include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

driver_diag: driver_diag.o  chkopts
	-${CLINKER} -o driver_diag driver_diag.o  ${PETSC_KSP_LIB}
	${RM} driver_diag.o


