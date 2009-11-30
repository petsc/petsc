CFLAGS           = ${BLOPEX_INCLUDE} 
FFLAGS           =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/contrib/blopex/driver_fortran/
EXAMPLESC        =
EXAMPLESF        = ex2f_blopex.F
MANSEC           = KSP

ex2f_blopex: ex2f_blopex.o blopex_c.o chkopts
	-${FLINKER} -o ex2f_blopex ex2f_blopex.o blopex_c.o ${PETSC_KSP_LIB}
	${RM} ex2f_blopex.o

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
