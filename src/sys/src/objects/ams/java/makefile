# $Id: makefile,v 1.1 2000/02/23 03:15:55 bsmith Exp bsmith $ 

#requirespackage  '-DPETSC_HAVE_JAVA'

lib: amsoptions
libfast: amsoptions

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC	  = 
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = 
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/objects/ams/java/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

TCLASSPATH  = ".:${AMS_HOME}/java/src:${AMS_HOME}/java/client/AMSBean.jar:${CLASSPATH}"

amsoptions:
	-@${JAVAC} -classpath ${TCLASSPATH} amsoptions.java

runamsoptions:
	-@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AMS_HOME}/lib/lib$1/`${AMS_HOME}/bin/arch.sh`/ ; export LD_LIBRARY_PATH ; ${JAVA} -cp ${TCLASSPATH} amsoptions 

