-include ../../../../petscdir.mk

LOCDIR           = src/ksp/ksp/tutorials/
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk bench_pcsetup
NP               = 1
DIRS             = network amrex

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

testex100: ex100.PETSc
	-@OMPI_MCA_mpi_warn_on_fork=0 ${MPIEXEC} -n 1 ./ex100 -test  > ex100_1.tmp 2>&1; \
	 if (${DIFF} output/ex100_1.testout ex100_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ Python example src/ksp/ksp/tutorials/ex100 run successfully with 1 MPI process"; \
	 else \
           echo "Possible error running C/C++ Python src/ksp/ksp/tutorials/ex100 with 1 MPI process"; \
           echo "See https://petsc.org/release/faq/";\
           cat ex100_1.tmp;\
           touch ${PETSC_DIR}/check_error;\
         fi; \
         ${RM} -f ex100_1.tmp

include ${PETSC_DIR}/lib/petsc/conf/test
