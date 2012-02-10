LOCDIR = tests/nlstests
include ${TAO_DIR}/conf/tao_base


# Directories
PPATH = problems

CSRC = wallclock.c

PFSRC = $(PPATH)/dfoxs.f $(PPATH)/dfovec.f $(PPATH)/dfonm.f \
	$(PPATH)/nlsfun.f driver_pounders.f surn01.f

calfun: dfovec.o calfun.f
	${FLINKER} -o calfun calfun.f dfovec.o

runtests: wallclock.o driver.o dfovec.o dfoxs.o tao_chkopts
	${CLINKER} -o runtests driver.o wallclock.o dfovec.o dfoxs.o ${TAO_LIB} ${PETSC_LIB}

runmatlab: wallclock.o mdriver.o dfovec.o dfoxs.o calfun tao_chkopts 
	${CLINKER} -o runmatlab mdriver.o wallclock.o dfovec.o dfoxs.o ${TAO_LIB} ${PETSC_LIB}


