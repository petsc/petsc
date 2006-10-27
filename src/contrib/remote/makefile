
ALL: lib

CFLAGS          =
FFLAGS          =
SOURCEC         = ex1.c
SOURCEF         =
SOURCEH         = hi/hi
OBJSC           = ex1.o
OBJSF           =
EXECUTABLE      = ex1
INPUTDATAFILES  =   
OUTPUTDATAFILES = 

DIRS     = noise
LOCDIR   = src/contrib/remote

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


${EXECUTABLE}: chkopts ${OBJSC} ${OBJSF}
	-${CLINKER} -o ${EXECUTABLE} ${OBJSC} ${OBJSF} ${PETSC_SNES_LIB}
	${RM} ${OBJSC} ${OBJSF}
