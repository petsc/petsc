# $Id: makefile,v 1.12 2000/11/28 17:27:53 bsmith Exp bsmith $ 

#requirespackage  'PETSC_HAVE_JAVA'

lib:
libfast:
all:      PETScRun PETScView PETScOptions PETScRund

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC	  = 
SOURCEJ   = PETScRun.java PETScOptions.java PETScRund.java PETScView.java PutAMSACC.java \
            Pumpstream.java
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
	-@rm -f /mcs/www-unix/petsc/bin/PETScRun.*
	-@mv -f PETScRun.jar /mcs/www-unix/petsc/bin
	-@cp -f PETScRun.html /mcs/www-unix/petsc/bin
	-@${RM} *.class
	-@chmod g+w /mcs/www-unix/petsc/bin/PETScRun.*

PETScView:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScView.java PutAMSACC.java
	-@mv PETScView*.class PutAMSACC*.class AMSBean
	-@cd AMSBean; jar cf PETScView.jar *
	-@rm -f /mcs/www-unix/petsc/bin/PETScView.*
	-@mv -f AMSBean/PETScView.jar /mcs/www-unix/petsc/bin
	-@cp -f PETScView.html /mcs/www-unix/petsc/bin
	-@${RM} AMSBean/*.class
	-@chmod g+w /mcs/www-unix/petsc/bin/PETScView.*

PETScOptions:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScOptions.java  PutAMSACC.java
	-@mv PETScOptions*.class PutAMSACC*.class AMSBean
	-@cd AMSBean; jar cf PETScOptions.jar *
	-@rm -f /mcs/www-unix/petsc/bin/PETScOptions.*
	-@mv -f AMSBean/PETScOptions.jar /mcs/www-unix/petsc/bin
	-@cp -f PETScOptions.html /mcs/www-unix/petsc/bin
	-@${RM} AMSBean/*.class
	-@chmod g+w /mcs/www-unix/petsc/bin/PETScOptions.*

PETScRund:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScRund.java PumpStream.java
	-@jar cf PETScRund.jar PETScRund*.class PumpStream*.class
	-@rm  ../../../../../../bin/PETScRund.jar
	-@mv -f PETScRund.jar ../../../../../../bin
	-@${RM} *.class
	-@chmod g+w  ../../../../../../bin/PETScRund.jar

runPETScRun:
	-@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AMS_LIBDIR}/; export LD_LIBRARY_PATH; ${JAVA} -cp ${TCLASSPATH} PETScRun ${AMS_OPTIONS} 2>&1 | grep -v font.properties
