CFLAGS           =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/tao/pde_constrained/examples/tutorials/
MANSEC           = PdeConstrained
EXAMPLESC	 = elliptic.c parabolic.c hyperbolic.c
TUTORIALS        = elliptic.c parabolic.c hyperbolic.c
EXAMPLESF	 = 
EXAMPLESCH	 = 
EXAMPLESFH       = 
EXAMPLESMATLAB   = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

parabolic: parabolic.o tao_chkopts
	-${CLINKER} -o parabolic parabolic.o ${TAO_LIB} ${PETSC_LIB}
	${RM} parabolic.o

hyperbolic: hyperbolic.o tao_chkopts
	-${CLINKER} -o hyperbolic hyperbolic.o ${TAO_LIB} ${PETSC_LIB}
	${RM} hyperbolic.o

elliptic: elliptic.o tao_chkopts
	-${CLINKER} -o elliptic elliptic.o ${TAO_LIB} ${PETSC_LIB}
	${RM} elliptic.o
