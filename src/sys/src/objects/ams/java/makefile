# $Id: makefile,v 1.8 2000/09/13 03:10:40 bsmith Exp bsmith $ 

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

AMSPETScOptions:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} AMSPETScOptions.java
	-@jar cf AMSPETScOptions.jar AMSPETScOptions*.class
	-@${RM} *.class

runamsoptions:
	-@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AMS_LIBDIR}/; export LD_LIBRARY_PATH; ${JAVA} -cp ${TCLASSPATH} amsoptions ${AMS_OPTIONS} 2>&1 | grep -v font.properties
