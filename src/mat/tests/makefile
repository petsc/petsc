-include ../../../petscdir.mk

CLEANFILES   = bench_spmv

ex72: ex72.o mmloader.o mmio.o
bench_spmv: bench_spmv.o mmloader.o mmio.o cJSON.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
