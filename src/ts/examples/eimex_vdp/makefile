
include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ct_vdp_imex: ct_vdp_imex.o  chkopts
	-${CLINKER} -o ct_vdp_imex ct_vdp_imex.o ${PETSC_TS_LIB}
	${RM} ct_vdp_imex.o

all: ct_vdp_imex
