# $Id: makefile,v 1.3 2000/02/16 03:53:05 bsmith Exp bsmith $ 

#requirespackage  '-DPETSC_HAVE_AMS'

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC	  = pams.c aoptions.c
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = pams.o aoptions.o
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/objects/ams/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

JAVAC       = /usr/java1.2/bin/javac
JAVA        = /usr/java1.2/bin/java
TCLASSPATH  = ".:${AMS_HOME}/java/src:${AMS_HOME}/java/client/AMSBean.jar:${CLASSPATH}"

amsoptions:
	-@${JAVAC} -classpath ${TCLASSPATH} amsoptions.java

runamsoptions:
	-@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AMS_HOME}/lib/lib$1/`${AMS_HOME}/bin/arch.sh`/ ; export LD_LIBRARY_PATH ; ${JAVA} -cp ${TCLASSPATH} amsoptions 

