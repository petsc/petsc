-include ../../../../petscdir.mk
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC	  = arch.c fhost.c fuser.c memc.c mpiu.c psleep.c sortd.c sorti.c sortso.c str.c sortip.c pbarrier.c pdisplay.c ctable.c psplit.c mpimesg.c sseenabled.c mpitr.c  mpilong.c mathinf.c matheq.c mathclose.c mathfit.c mpits.c segbuffer.c mpishm.c
SOURCEF	  =
SOURCEH	  = ../../../include/petscctable.h ../../../include/petsc/private/generated_hashset.txt
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  = ftn-kernels
LOCDIR	  = src/sys/utils/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

