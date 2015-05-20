
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
EXAMPLESC = googledriveupload.c urlshorten.c googleobtainrefreshtoken.c boxobtainrefreshtoken.c boxupload.c
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/webclient/examples/tutorials

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

urlshorten: urlshorten.o  chkopts
	${CLINKER} -o urlshorten urlshorten.o ${PETSC_SYS_LIB}
	${RM} -f urlshorten.o

googleobtainrefreshtoken: googleobtainrefreshtoken.o  chkopts
	${CLINKER} -o googleobtainrefreshtoken googleobtainrefreshtoken.o ${PETSC_SYS_LIB}
	${RM} -f googleobtainrefreshtoken.o

boxobtainrefreshtoken: boxobtainrefreshtoken.o  chkopts
	${CLINKER} -o boxobtainrefreshtoken boxobtainrefreshtoken.o ${PETSC_SYS_LIB}
	${RM} -f boxobtainrefreshtoken.o

globusobtainaccesstoken: globusobtainaccesstoken.o  chkopts
	${CLINKER} -o globusobtainaccesstoken globusobtainaccesstoken.o ${PETSC_SYS_LIB}
	${RM} -f globusobtainaccesstoken.o

googledriveupload: googledriveupload.o  chkopts
	${CLINKER} -o googledriveupload googledriveupload.o ${PETSC_SYS_LIB}
	${RM} -f googledriveupload.o

boxupload: boxupload.o  chkopts
	${CLINKER} -o boxupload boxupload.o ${PETSC_SYS_LIB}
	${RM} -f boxupload.o

globusobtaintransfers: globusobtaintransfers.o  chkopts
	${CLINKER} -o globusobtaintransfers globusobtaintransfers.o ${PETSC_SYS_LIB}
	${RM} -f globusobtaintransfers.o

globusupload: globusupload.o  chkopts
	${CLINKER} -o globusupload globusupload.o ${PETSC_SYS_LIB}
	${RM} -f globusupload.o

