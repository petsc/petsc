# $Id: makefile,v 1.7 1999/02/02 21:24:03 balay Exp balay $ 

ALL: lib 

CFLAGS    =
FFLAGS    =
SOURCEC	  = cputime.c fdate.c nt_time.c rs6000_time.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = cputime.o fdate.o nt_time.o rs6000_time.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/time/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

rs6000_asmtime: rs6000_asmtime.s
	-@if [ "${PARCH}" = "rs6000" ]; then \
	${CC} -c rs6000_asmtime.s ; \
	${AR} cr ${LIBNAME} rs6000_asmtime.o ; \
	${RM} -f rs6000_asmtime.o; \
	fi


