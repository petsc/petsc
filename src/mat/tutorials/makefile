-include ../../../petscdir.mk

LOCDIR          = src/mat/tutorials/
EXAMPLESC       = ex1.c ex2.c ex4.c ex8.c ex9.c ex10.c ex11.c ex12.c ex15.c ex16.c ex17.c ex18.c
EXAMPLESF       = ex4f.F90 ex6f.F90 ex15f.F90 ex17f.F90
EXAMPLESCU      = ex5cu.cu
MANSEC          = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# set the conditional prerequisites after including the 'variables' file, so we
# can reference make macros defined by it, but before including the 'test' file,
# so that ex18 has correct full prerequisites (instead of the default ex18.o)
ex18objs := ex18.o
ifneq ($(CUDAC),)
  ex18objs += ex18cu.o
endif
ifneq ($(KOKKOS_LIB),)
  ex18objs += ex18kok.o
endif
ex18 : ${ex18objs}

include ${PETSC_DIR}/lib/petsc/conf/test
