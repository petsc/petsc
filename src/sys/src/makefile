# $Id: makefile,v 1.68 1998/06/04 22:34:46 balay Exp balay $ 

ALL: lib 
CFLAGS	  = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' -DPETSC_BOPT='"${BOPT}"' \
            -DPETSC_LDIR='"${LDIR}"' -DPETSC_DIR='"${PETSC_DIR}"' -DPETSC_SLSUFFIX='"${SLSUFFIX}"'

SOURCEC	  = 
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = 
OBJSF	  =
LIBBASE	  = libpetscsys
DIRS	  = error dll fileio memory objects time utils plog mpiuni
LOCDIR	  = src/sys/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

rs6000_time: rs6000_time.s
	-@if [ "${PARCH}" = "rs6000" ]; then \
	${CC} -c rs6000_time.s ; \
	${AR} cr ${LIBNAME} rs6000_time.o ; \
	${RM} -f rs6000_time.o; \
	fi


