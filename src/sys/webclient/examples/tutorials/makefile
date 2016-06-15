
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

urlshorten: urlshorten.o  chkopts
	${CLINKER} -o urlshorten urlshorten.o ${PETSC_SYS_LIB}
	${RM} -f urlshorten.o

googleobtainrefreshtoken: googleobtainrefreshtoken.o  chkopts
	${CLINKER} -o googleobtainrefreshtoken googleobtainrefreshtoken.o ${PETSC_SYS_LIB}
	${RM} -f googleobtainrefreshtoken.o

googledriveupload: googledriveupload.o  chkopts
	${CLINKER} -o googledriveupload googledriveupload.o ${PETSC_SYS_LIB}
	${RM} -f googledriveupload.o

boxobtainrefreshtoken: boxobtainrefreshtoken.o  chkopts
	${CLINKER} -o boxobtainrefreshtoken boxobtainrefreshtoken.o ${PETSC_SYS_LIB}
	${RM} -f boxobtainrefreshtoken.o

boxupload: boxupload.o  chkopts
	${CLINKER} -o boxupload boxupload.o ${PETSC_SYS_LIB}
	${RM} -f boxupload.o

globusobtaintransfers: globusobtaintransfers.o  chkopts
	${CLINKER} -o globusobtaintransfers globusobtaintransfers.o ${PETSC_SYS_LIB}
	${RM} -f globusobtaintransfers.o

# the next two files don't exist in the repository; did I forget to git add them when I wrote them?
globusobtainaccesstoken: globusobtainaccesstoken.o  chkopts
	${CLINKER} -o globusobtainaccesstoken globusobtainaccesstoken.o ${PETSC_SYS_LIB}
	${RM} -f globusobtainaccesstoken.o

globusupload: globusupload.o  chkopts
	${CLINKER} -o globusupload globusupload.o ${PETSC_SYS_LIB}
	${RM} -f globusupload.o

runurlshorten:
	-@${MPIEXEC} -n 1 ./urlshorten > urlshorten_1.tmp 2>&1;   \
	   if (${DIFF} output/urlshorten_1.out urlshorten_1.tmp) then true; \
	   else printf "${PWD}\nPossible problem with urlshorten_1, diffs above\n=========================================\n"; fi; \
	   ${RM} -f urlshorten_1.tmp

# we need a test for -textbelt but I don't want to hardwire Matt's cellphone number in the repository

# we need to run this test regularly but I don't want to hardwire Matt's Google refresh token in the repository
rungoogledriveupload:
	-@${MPIEXEC} -n 1 ./googledriveupload -google_refresh_token XXX

# We currently have no tests for the Globus file management

TESTEXAMPLES_C	          = urlshorten.PETSc runurlshorten urlshorten.rm googleobtainrefreshtoken.PETSc googleobtainrefreshtoken.rm \
                            googledriveupload.PETSc googledriveupload.rm 

TESTEXAMPLES_SAWS	   = boxobtainrefreshtoken.PETSc boxobtainrefreshtoken.rm \
                             boxupload.PETSc boxupload.rm

include ${PETSC_DIR}/lib/petsc/conf/test
