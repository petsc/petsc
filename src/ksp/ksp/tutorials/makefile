-include ../../../../petscdir.mk

LOCDIR           = src/ksp/ksp/tutorials/
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1
DIRS             = network amrex

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

testex100: ex100.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/ksp/ksp/tutorials/ex100 with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/ksp/ksp/tutorials/ex100 manually"; \
	elif [ -f ex100 ]; then \
           OMPI_MCA_mpi_warn_on_fork=0 ${MPIEXEC} -n 1 ./ex100 -test  > ex100_1.tmp 2>&1; \
	   if (${DIFF} output/ex100_1.testout ex100_1.tmp > /dev/null 2>&1) then \
           echo "C/C++ Python example src/ksp/ksp/tutorials/ex100 run successfully with 1 MPI process"; \
	   else echo "Possible error running C/C++ Python src/ksp/ksp/tutorials/ex100 with 1 MPI process"; \
           echo "See http://www.mcs.anl.gov/petsc/documentation/faq.html";\
           cat ex100_1.tmp; fi; \
        ${RM} -f ex100_1.tmp; fi

include ${PETSC_DIR}/lib/petsc/conf/test
