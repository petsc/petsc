# $Id: makefile,v 1.10 2000/10/24 20:24:39 bsmith Exp bsmith $ 

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
	-@jar cf PETScRun.jar PETScRun*.class
	-@mv -f PETScRun.jar ../../../../../../bin
	-@cp -f PETScRun.html ../../../../../../bin
	-@${RM} *.class

PETScView:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScView.java
	-@mv PETScView*.class AMSBean
	-@cd AMSBean; jar cf PETScView.jar *
	-@mv -f AMSBean/PETScView.jar ../../../../../../bin
	-@cp -f PETScView.html ../../../../../../bin
	-@${RM} AMSBean/*.class

PETScRund:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScRund.java PumpStream.java Servers.java
	-@jar cf PETScRund.jar PETScRund*.class PumpStream*.class Servers*.class
	-@mv -f PETScRund.jar ../../../../../../bin
	-@${RM} *.class

AMSPETScOptions:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} AMSPETScOptions.java
	-@mv AMSPETScOptions*.class AMSBean
	-@cd AMSBean; jar cf AMSPETScOptions.jar *
	-@mv -f AMSBean/AMSPETScOptions.jar ../../../../../../bin
	-@cp -f AMSPETScOptions.html ../../../../../../bin
	-@${RM} AMSBean/*.class

runPETScRun:
	-@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AMS_LIBDIR}/; export LD_LIBRARY_PATH; ${JAVA} -cp ${TCLASSPATH} PETScRun ${AMS_OPTIONS} 2>&1 | grep -v font.properties
