ALL: 

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/bound/examples/tutorials/
MANSEC		 = Bound
DOCS		 = 
DIRS		 = 

include ${TAO_DIR}/conf/tao_base

plate2: plate2.o tao_chkopts
	-${CLINKER} -o plate2 plate2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} plate2.o

plate2f: plate2f.o tao_chkopts
	-${FLINKER} -o plate2f plate2f.o ${TAO_LIB} ${PETSC_LIB}
	${RM} plate2f.o

jbearing2: jbearing2.o tao_chkopts
	-${CLINKER} -o jbearing2 jbearing2.o ${TAO_LIB} ${PETSC_LIB}
	${RM} jbearing2.o
