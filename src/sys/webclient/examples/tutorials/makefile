
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
EXAMPLESC = googledriveupload.c urlshorten.c googledriverefreshtoken.c
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/webclient/examples/tutorials

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

urlshorten: urlshorten.o  chkopts
	${CLINKER} -o urlshorten2 urlshorten.o ${PETSC_SYS_LIB}
	${RM} -f urlshorten.o

googledriverefreshtoken: googledriverefreshtoken.o  chkopts
	${CLINKER} -o googledriverefreshtoken googledriverefreshtoken.o ${PETSC_SYS_LIB}
	${RM} -f googledriverefreshtoken.o

googledriveupload: googledriveupload.o  chkopts
	${CLINKER} -o googledriveupload googledriveupload.o ${PETSC_SYS_LIB}
	${RM} -f googledriveupload.o


