
#requiresdefine 'PETSC_HAVE_JAVA'
SPECIALLIB     = yes
SPECIALFASTLIB = yes
speciallib:
	-@echo "BEGINNING TO COMPILE AMSOptions"
	-@${JAVAC} -classpath .:${AMS_DIR}/java amsoptions.java
	-@${MV} -f *.class ${PETSC_DIR}/${PETSC_ARCH}/bin
specialfastlib: speciallib

ALL: lib 
SOURCEH   = 
SOURCEC   = 
OBJSC     = 
DIRS      = 
LOCDIR    = src/sys/ams/java
MANSEC    = Sys

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
