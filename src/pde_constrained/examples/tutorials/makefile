CFLAGS           =
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/pde_constrained/examples/tutorials/
MANSEC           = 
EXAMPLESC	 = elliptic.c parabolic.c hyperbolic.c
EXAMPLESF	 = 
EXAMPLESCH	 = 
EXAMPLESFH       = 
EXAMPLESMATLAB   = 

include ${TAO_DIR}/conf/tao_base

parabolic: parabolic.o tao_chkopts
	-${CLINKER} -o parabolic parabolic.o ${TAO_LIB} ${PETSC_LIB}
	${RM} parabolic.o

hyperbolic: hyperbolic.o tao_chkopts
	-${CLINKER} -o hyperbolic hyperbolic.o ${TAO_LIB} ${PETSC_LIB}
	${RM} hyperbolic.o

elliptic: elliptic.o tao_chkopts
	-${CLINKER} -o elliptic elliptic.o ${TAO_LIB} ${PETSC_LIB}
	${RM} elliptic.o
