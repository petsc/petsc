# $Id: makefile,v 1.9 2000/09/26 02:56:05 bsmith Exp bsmith $ 

#requirespackage  'PETSC_HAVE_JAVA'

lib:
libfast:

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
DOCS      = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

TCLASSPATH  = ".:${AMS_HOME}/java/src:${AMS_HOME}/java/client/AMSBean.jar:${CLASSPATH}"

PETScRun:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScRun.java

AMSPETScOptions:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} AMSPETScOptions.java
	-@jar cf AMSPETScOptions.jar AMSPETScOptions*.class
	-@${RM} *.class

runPETScRun:
	-@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AMS_LIBDIR}/; export LD_LIBRARY_PATH; ${JAVA} -cp ${TCLASSPATH} PETScRun ${AMS_OPTIONS} 2>&1 | grep -v font.properties
