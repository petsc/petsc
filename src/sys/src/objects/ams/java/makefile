# $Id: makefile,v 1.24 2001/08/22 17:59:28 balay Exp $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

TCLASSPATH    = "."
PETSCBINPATH  = ${PETSC_DIR}

PETScRun:
	-@${JAVAC} -deprecation  PETScRun.java
	-@${JAR} cf PETScRun.jar PETScRun*.class
	-@${RM} ${PETSCBINPATH}/bin/PETScRun.*
	-@${MV} -f PETScRun.jar ${PETSCBINPATH}/bin
	-@${CP} -f PETScRun.html ${PETSCBINPATH}/bin
	-@${RM} *.class
	-@chmod g+w ${PETSCBINPATH}/bin/PETScRun.*

PETScView:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScView*.java 
	-@mkdir tmp
	-@${MV} PETScView*.class tmp
	-@cd tmp; ${JAR} cf PETScView.jar *
	-@${RM} ${PETSCBINPATH}/bin/PETScView.*
	-@${MV} -f tmp/PETScView.jar ${PETSCBINPATH}/bin
	-@${CP} -f PETScView.html ${PETSCBINPATH}/bin
	-@${RM} tmp/*.class
	-@rmdir tmp
	-@chmod g+w ${PETSCBINPATH}/bin/PETScView.*

PETScOptions:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScOptions.java 
	-@mkdir tmp
	-@${MV} PETScOptions*.class tmp
	-@cd tmp; ${JAR} cf PETScOptions.jar *
	-@${RM} ${PETSCBINPATH}/bin/PETScOptions.*
	-@${MV} -f tmp/PETScOptions.jar ${PETSCBINPATH}/bin
	-@${CP} -f PETScOptions.html ${PETSCBINPATH}/bin
	-@${RM} tmp/*.class
	-@rmdir tmp
	-@chmod g+w ${PETSCBINPATH}/bin/PETScOptions.*

PETScPlugins:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScPlugins*.java 
	-@${JAR} cf PETScPlugins.jar *.class
	-@${RM} ${PETSCBINPATH}/bin/PETScPlugins.*
	-@${MV} -f PETScPlugins.jar ${PETSCBINPATH}/bin
	-@${CP} PETScPlugins.html ${PETSCBINPATH}/bin
	-@${RM} *.class
	-@chmod g+w ${PETSCBINPATH}/bin/PETScPlugins.*

PETScRund:
	-@${JAVAC} -deprecation PETScRund.java PumpStream.java
	-@${JAR} cf PETScRund.jar PETScRund*.class PumpStream*.class
	-@${RM}  ../../../../../../bin/PETScRund.jar
	-@${MV} -f PETScRund.jar ../../../../../../bin
	-@${RM} *.class
	-@chmod g+w  ../../../../../../bin/PETScRund.jar

