-include ../../../../../petscdir.mk

CXXFLAGS         = ${AMREX_INCLUDE}
OBJECTS          = main.o MyTest.o initEB.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# TODO: have configure determine  -Wl,-undefined,dynamic_lookup for all OS

amrex: $(OBJECTS)
	-${CXXLINKER} -o amrex $(OBJECTS) ${AMREX_LIB} -Wl,-undefined,dynamic_lookup ${PETSC_KSP_LIB}
	${RM} $(OBJECTS)

#-------------------------------------------------------------------------

#  these tests are used by the makefile in PETSC_DIR for basic tests of the install and should not be removed
testamrex: amrex
	-@${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./amrex inputs.rt.2d.petsc | grep -E -v "(AMReX|Timers)" > amrex_1.tmp 2>&1; \
	 if (${DIFF} output/amrex_1.testout amrex_1.tmp > /dev/null 2>&1) then \
           echo "example src/ksp/ksp/tutorials/amrex/amrex run successfully with 1 MPI process"; \
	   else echo "Possible error running  example src/ksp/ksp/tutorials/amrex/amrex with 1 MPI process"; \
           echo "See https://petsc.org/release/faq/";\
           cat amrex_1.tmp; fi;  \
         ${RM} -f amrex_1.tmp

