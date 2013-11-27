LOCDIR = src/tao/tests/nlstests

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

# Directories
PPATH = problems

CSRC = wallclock.c

PFSRC = $(PPATH)/dfoxs.f $(PPATH)/dfovec.f $(PPATH)/dfonm.f \
	$(PPATH)/nlsfun.f driver_pounders.f surn01.f

calfun: dfovec.o calfun.f
	${FLINKER} -o calfun calfun.f dfovec.o

runtests: wallclock.o driver.o dfovec.o dfoxs.o chkopts
	${CLINKER} -o runtests driver.o wallclock.o dfovec.o dfoxs.o ${PETSC_TAO_LIB}

runmatlab: wallclock.o mdriver.o dfovec.o dfoxs.o calfun chkopts 
	${CLINKER} -o runmatlab mdriver.o wallclock.o dfovec.o dfoxs.o ${PETSC_TAO_LIB}


