# $Id: makefile,v 1.11 2000/11/19 00:44:24 bsmith Exp bsmith $ 

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
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScView.java PutAMSACC.java
	-@mv PETScView*.class PutAMSACC*.class AMSBean
	-@cd AMSBean; jar cf PETScView.jar *
	-@mv -f AMSBean/PETScView.jar ../../../../../../bin
	-@cp -f PETScView.html ../../../../../../bin
	-@${RM} AMSBean/*.class

PETScRund:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScRund.java PumpStream.java
	-@jar cf PETScRund.jar PETScRund*.class PumpStream*.class
	-@mv -f PETScRund.jar ../../../../../../bin
	-@${RM} *.class

PETScOptions:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScOptions.java  PutAMSACC.java
	-@mv PETScOptions*.class PutAMSACC*.class AMSBean
	-@cd AMSBean; jar cf PETScOptions.jar *
	-@mv -f AMSBean/PETScOptions.jar ../../../../../../bin
	-@cp -f PETScOptions.html ../../../../../../bin
	-@${RM} AMSBean/*.class

runPETScRun:
	-@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AMS_LIBDIR}/; export LD_LIBRARY_PATH; ${JAVA} -cp ${TCLASSPATH} PETScRun ${AMS_OPTIONS} 2>&1 | grep -v font.properties
