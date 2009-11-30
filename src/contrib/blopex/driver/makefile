
CFLAGS           = ${BLOPEX_INCLUDE} 
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/contrib/blopex/driver/
EXAMPLESC        = driver.c
EXAMPLESF        = 
MANSEC           = KSP


include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

driver: driver.o  chkopts
	-${CLINKER} -o driver driver.o  ${PETSC_KSP_LIB}
	${RM} driver.o


