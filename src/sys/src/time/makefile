# $Id: makefile,v 1.6 1999/01/12 23:14:29 bsmith Exp balay $ 

ALL: lib 

CFLAGS    =
FFLAGS    =
SOURCEC	  = cputime.c fdate.c nt_time.c 
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = cputime.o fdate.o nt_time.o 
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/time/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

rs6000_time: rs6000_time.s
	-@if [ "${PARCH}" = "rs6000" ]; then \
	${CC} -c rs6000_time.s ; \
	${AR} cr ${LIBNAME} rs6000_time.o ; \
	${RM} -f rs6000_time.o; \
	fi


