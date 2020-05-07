-include ../../../../../petscdir.mk
CFLAGS           =
CXXFLAGS         = ${AMREX_INCLUDE}
FFLAGS		 =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR		 = src/ksp/ksp/tutorials/amrex
MANSEC           = KSP
EXAMPLESC	 =
EXAMPLESF	 =
EXAMPLESCH	 = MyEB.H MyTest.H
EXAMPLESFH       =
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
	@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
           echo "Running with batch filesystem; to test run src/ksp/ksp/tutorials/amrex/amrex with" ; \
           echo "your systems batch system"; \
        elif [ "${MPIEXEC}" = "/bin/false" ]; then \
           echo "*mpiexec not found*. Please run src/ksp/ksp/tutorials/amrex/amrex manually"; \
        elif [ -f amrex ]; then \
	   ${MPIEXEC} -n 1 ./amrex inputs.rt.2d.petsc | egrep -v "(AMReX|Timers)" > amrex_1.tmp 2>&1; \
	   if (${DIFF} output/amrex_1.testout amrex_1.tmp > /dev/null 2>&1) then \
           echo "example src/ksp/ksp/tutorials/amrex/amrex run successfully with 1 MPI process"; \
	   else echo "Possible error running  example src/ksp/ksp/tutorials/amrex/amrex with 1 MPI process"; \
           echo "See http://www.mcs.anl.gov/petsc/documentation/faq.html";\
           cat amrex_1.tmp; fi;  \
         ${RM} -f amrex_1.tmp; fi

include ${PETSC_DIR}/lib/petsc/conf/test
