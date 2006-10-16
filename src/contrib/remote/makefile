
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = ex1.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ex1.o
OBJSF    =
DIRS     = noise
LOCDIR   = src/contrib/remote
OUTPUT   = ex1
DATAFILES = 
DATAFILESRETURN =


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


ex1: chkopts ex1.o
	-${CLINKER} -o ex1 ex1.o ${PETSC_SNES_LIB}
	${RM} ex1.o
