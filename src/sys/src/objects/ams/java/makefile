# $Id: makefile,v 1.19 2001/03/22 20:29:05 bsmith Exp bsmith $ 

#requirespackage  'PETSC_HAVE_JAVA'

SPECIALLIB     = yes
SPECIALFASTLIB = yes
speciallib:      PETScRun PETScView PETScOptions PETScRund PETScPlugins
specialfastlib: speciallib

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC	  = 
SOURCEJ   = PETScRun.java PETScOptions.java PETScRund.java PETScView.java PumpStream.java PETScViewKSP.java \
            PETScViewSNES.java
SOURCEF	  =
SOURCEH	  = 
OBJSC	  = 
OBJSF	  =
MANSEC	  = Sys
LIBBASE	  = libpetsc
DIRS	  = 
LOCDIR	  = src/sys/src/objects/ams/java/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

TCLASSPATH    = ".:./AMSBean"
PETSCBINPATH  = ${PETSC_DIR}

PETScRun:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScRun.java
	-@${JAR} cf PETScRun.jar PETScRun*.class
	-@rm -f ${PETSCBINPATH}/bin/PETScRun.*
	-@mv -f PETScRun.jar ${PETSCBINPATH}/bin
	-@cp -f PETScRun.html ${PETSCBINPATH}/bin
	-@${RM} *.class
	-@chmod g+w ${PETSCBINPATH}/bin/PETScRun.*

PETScView:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScView*.java 
	-@mv PETScView*.class AMSBean
	-@cd AMSBean; ${JAR} cf PETScView.jar *
	-@rm -f ${PETSCBINPATH}/bin/PETScView.*
	-@mv -f AMSBean/PETScView.jar ${PETSCBINPATH}/bin
	-@cp -f PETScView.html ${PETSCBINPATH}/bin
	-@${RM} AMSBean/*.class
	-@chmod g+w ${PETSCBINPATH}/bin/PETScView.*

PETScOptions:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScOptions.java 
	-@mv PETScOptions*.class AMSBean
	-@cd AMSBean; ${JAR} cf PETScOptions.jar *
	-@rm -f ${PETSCBINPATH}/bin/PETScOptions.*
	-@mv -f AMSBean/PETScOptions.jar ${PETSCBINPATH}/bin
	-@cp -f PETScOptions.html ${PETSCBINPATH}/bin
	-@${RM} AMSBean/*.class
	-@chmod g+w ${PETSCBINPATH}/bin/PETScOptions.*

PETScPlugins:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScPlugins*.java 
	-@${JAR} cf PETScPlugins.jar *.class
	-@rm -f ${PETSCBINPATH}/bin/PETScPlugins.*
	-@mv -f PETScPlugins.jar ${PETSCBINPATH}/bin
	-@cp PETScPlugins.html ${PETSCBINPATH}/bin
	-@${RM} *.class
	-@chmod g+w ${PETSCBINPATH}/bin/PETScPlugins.*

PETScRund:
	-@${JAVAC} -deprecation PETScRund.java PumpStream.java
	-@${JAR} cf PETScRund.jar PETScRund*.class PumpStream*.class
	-@rm  ../../../../../../bin/PETScRund.jar
	-@mv -f PETScRund.jar ../../../../../../bin
	-@${RM} *.class
	-@chmod g+w  ../../../../../../bin/PETScRund.jar

