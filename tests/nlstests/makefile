LOCDIR = tests/nlstests
include ${TAO_DIR}/conf/tao_base


# Directories
PPATH = problems

CSRC = wallclock.c

PFSRC = $(PPATH)/dfoxs.f $(PPATH)/dfovec.f $(PPATH)/dfonm.f \
	$(PPATH)/nlsfun.f driver_pounders.f surn01.f


runtests: wallclock.o driver.o dfovec.o dfoxs.o tao_chkopts
	${CLINKER} -o runtests driver.o wallclock.o dfovec.o dfoxs.o ${TAO_LIB} ${PETSC_LIB}


