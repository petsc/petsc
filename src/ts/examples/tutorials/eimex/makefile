
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/eimex/
EXAMPLESC       = allen_cahn.c ct_vdp_imex.c
MANSEC          = TS
DIRS            =
CLEANFILES      = ct_vdp_imex allen_cahn eimex_nonstiff_vdp.txt

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ct_vdp_imex: ct_vdp_imex.o  chkopts
	-${CLINKER} -o ct_vdp_imex ct_vdp_imex.o ${PETSC_TS_LIB}
	${RM} ct_vdp_imex.o

allen_cahn: allen_cahn.o  chkopts
	-${CLINKER} -o allen_cahn allen_cahn.o ${PETSC_TS_LIB}
	${RM} allen_cahn.o

include ${PETSC_DIR}/lib/petsc/conf/test
