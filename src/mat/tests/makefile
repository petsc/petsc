-include ../../../petscdir.mk

CLEANFILES   = bench_spmv

ex72: mmloader.o mmio.o
bench_spmv: mmloader.o mmio.o cJSON.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
