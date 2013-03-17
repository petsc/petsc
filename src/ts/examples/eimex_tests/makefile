
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/
EXAMPLESC       = allen_cahn.c
MANSEC          = TS

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

allen_cahn: allen_cahn.o chkopts
	-${CLINKER} -o allen_cahn allen_cahn.o ${PETSC_TS_LIB}
	${RM} allen_cahn.o

all: allen_cahn
