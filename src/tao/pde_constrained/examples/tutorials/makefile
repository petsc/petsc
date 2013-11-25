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

parabolic: parabolic.o chkopts
	-${CLINKER} -o parabolic parabolic.o ${PETSC_TAO_LIB}
	${RM} parabolic.o

hyperbolic: hyperbolic.o chkopts
	-${CLINKER} -o hyperbolic hyperbolic.o ${PETSC_TAO_LIB}
	${RM} hyperbolic.o

elliptic: elliptic.o chkopts
	-${CLINKER} -o elliptic elliptic.o ${PETSC_TAO_LIB}
	${RM} elliptic.o
