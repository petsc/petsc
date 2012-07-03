
ALL: lib

CFLAGS          =
FFLAGS          =
SOURCEC         = ex1.c
SOURCEF         =
SOURCEH         = 
OBJSC           = ex1.o
OBJSF           =
EXECUTABLE      = ex1
INPUTDATAFILES  =   
OUTPUTDATAFILES = 
PETSCPORTABLE   = true

DIRS     = 
LOCDIR   = src/contrib/remote/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test


${EXECUTABLE}: chkopts ${OBJSC} ${OBJSF}
	-${CLINKER} -o ${EXECUTABLE} ${OBJSC} ${OBJSF} ${PETSC_SNES_LIB}
	${RM} ${OBJSC} ${OBJSF}
