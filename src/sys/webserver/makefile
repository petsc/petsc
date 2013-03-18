#requirespackage  'PETSC_HAVE_SERVER'

ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC	  =
SOURCEF	  =
SOURCEH	  =
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/objects/webserver

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

petscwebserver: petscwebserver.o webserve.o  chkopts
	@-lib=`echo  ${AMS_LIB} | sed "s?amspub?amsacc?g"`; \
        ${CLINKER} -o petscwebserver petscwebserver.o webserve.o $$lib ${PETSC_SYS_LIB}
	${RM} -f petscwebserver.o webserve.o
	${MV} -f petscwebserver ${PETSC_DIR}/${PETSC_ARCH}/bin

petscwebclient: petscwebclient.o  chkopts
	${CLINKER} -o petscwebclient petscwebclient.o ${PETSC_SYS_LIB}
	${RM} -f petscwebclient.o
