# $Id: makefile,v 1.66 1998/04/23 23:28:33 balay Exp bsmith $ 

ALL: lib 
CFLAGS	  = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' -DPETSC_BOPT='"${BOPT}"' \
            -DPETSC_LDIR='"${LDIR}"' -DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'

SOURCEC	  = 
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = 
OBJSF	  =
MANSEC	  = 7
LIBBASE	  = libpetscsys
DIRS	  = dll error fileio memory objects time utils plog mpiuni
LOCDIR	  = src/sys/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

rs6000_time: rs6000_time.s
	-@if [ "${PARCH}" = "rs6000" ]; then \
	${CC} -c rs6000_time.s ; \
	${AR} cr ${LIBNAME} rs6000_time.o ; \
	${RM} -f rs6000_time.o; \
	fi


