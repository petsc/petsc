# $Id: makefile,v 1.16 2001/03/09 21:01:04 bsmith Exp bsmith $ 

#requirespackage  'PETSC_HAVE_JAVA'
#requirespackage  'PETSC_HAVE_AMS'

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
DOCS      = 

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test

TCLASSPATH  = ".:${AMS_HOME}/java/src:./AMSBean:${CLASSPATH}"

PETScRun:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScRun.java
	-@${JAR} cf PETScRun.jar PETScRun*.class
	-@rm -f /mcs/www-unix/petsc/bin/PETScRun.*
	-@mv -f PETScRun.jar /mcs/www-unix/petsc/bin
	-@cp -f PETScRun.html /mcs/www-unix/petsc/bin
	-@${RM} *.class
	-@chmod g+w /mcs/www-unix/petsc/bin/PETScRun.*

PETScView:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScView*.java 
	-@mv PETScView*.class AMSBean
	-@cd AMSBean; ${JAR} cf PETScView.jar *
	-@rm -f /mcs/www-unix/petsc/bin/PETScView.*
	-@mv -f AMSBean/PETScView.jar /mcs/www-unix/petsc/bin
	-@cp -f PETScView.html /mcs/www-unix/petsc/bin
	-@${RM} AMSBean/*.class
	-@chmod g+w /mcs/www-unix/petsc/bin/PETScView.*

PETScOptions:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScOptions.java 
	-@mv PETScOptions*.class AMSBean
	-@cd AMSBean; ${JAR} cf PETScOptions.jar *
	-@rm -f /mcs/www-unix/petsc/bin/PETScOptions.*
	-@mv -f AMSBean/PETScOptions.jar /mcs/www-unix/petsc/bin
	-@cp -f PETScOptions.html /mcs/www-unix/petsc/bin
	-@${RM} AMSBean/*.class
	-@chmod g+w /mcs/www-unix/petsc/bin/PETScOptions.*

PETScPlugins:
	-@${JAVAC} -deprecation -classpath ${TCLASSPATH} PETScPlugins*.java 
	-@${JAR} cf PETScPlugins.jar *.class
	-@rm -f /mcs/www-unix/petsc/bin/PETScPlugins.*
	-@mv -f PETScPlugins.jar /mcs/www-unix/petsc/bin
	-@cp PETScPlugins.html /mcs/www-unix/petsc/bin
	-@${RM} *.class
	-@chmod g+w /mcs/www-unix/petsc/bin/PETScPlugins.*

PETScRund:
	-@${JAVAC} -deprecation PETScRund.java PumpStream.java
	-@${JAR} cf PETScRund.jar PETScRund*.class PumpStream*.class
	-@rm  ../../../../../../bin/PETScRund.jar
	-@mv -f PETScRund.jar ../../../../../../bin
	-@${RM} *.class
	-@chmod g+w  ../../../../../../bin/PETScRund.jar

runPETScRun:
	-@LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AMS_LIBDIR}/; export LD_LIBRARY_PATH; ${JAVA} -cp ${TCLASSPATH} PETScRun ${AMS_OPTIONS} 2>&1 | grep -v font.properties
