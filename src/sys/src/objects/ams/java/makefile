# $Id: makefile,v 1.5 2000/08/17 04:50:47 bsmith Exp bsmith $ 

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
DOCS      = amsoptions.java

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

TCLASSPATH  = ".:${AMS_HOME}/java/src:${AMS_HOME}/java/client/AMSBean.jar:${CLASSPATH}"

amsoptions:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} amsoptions.java

amsoptionsapplet:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} amsoptionsapplet.java

AMSPETScOptions:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} AMSPETScOptions.java

runamsoptions:
	-@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AMS_LIBDIR}/; export LD_LIBRARY_PATH; ${JAVA} -cp ${TCLASSPATH} amsoptions ${AMS_OPTIONS} 2>&1 | grep -v font.properties
