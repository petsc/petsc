#
# This code to build f90 modules is obsoleted by the sources
# in src/sys/f90-mod
#

#requirespackage   'PETSC_USING_F90'

#SPECIALLIB     = yes
#SPECIALFASTLIB = yes
speciallib:
	-@${OMAKE}  makemodules
	-@${OMAKE}  clean
specialfastlib: speciallib

CFLAGS    =
FFLAGS    =
SOURCEC   =
SOURCEF   =
SOURCEH   = petsc.h petscdef.h petsclog.h petscvec.h petscsnes.h petscda.h petscdraw.h petscmat.h \
	    petscksp.h petscpc.h petscviewer.h petscis.h petscao.h petscmg.h \
	    petscsys.h petscts.h petscis.h90 petscvec.h90 petscmat.h90 petscda.h90
OBJSC     =
OBJSF     =
LIBBASE   = libpetscvec
DIRS      = 
MANSEC    = Sys
LOCDIR    = include/finclude/
NOADIC    = true
CLEANFILES = *.mod ipath __tmp *.xinc *.f90.h *.f90 *.use *.inc modorder makefile.tmp allinone.h *.o *.f90

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test

.SUFFIXES:
.SUFFIXES: .mod .inc .use

# remake anly if we havn't allinone.h
all: makemodules allinone.h

makefile.tmp: moddeps
	@${CP} makefile makefile.tmp;\
	cat moddeps|grep -v '^ *#'|grep ':'|\
	sed 's/^ *\([^ ]*\)\(.*\)/\1\2 \1.f90.h/' >> makefile.tmp;\
	MODS=`cat moddeps|grep -v '^ *#'|grep ':'|sed 's/^ *\([^ ]*\).*/\1 /'`;\
	echo modules : $$MODS >> makefile.tmp

makemodules: clean makefile.tmp
	@if [ ! -e allinone.h ]; then\
	  ${RM} -f modorder;\
	  ${OMAKE} -f makefile.tmp modules;\
	fi
	@${MV} -f *.mod ${PETSC_DIR}/${PETSC_ARCH}/include

allinone.h: modorder
	LAST=`tail -n 1 modorder`;if [ -e $$LAST.xinc ]; then cat $$LAST.xinc >>allinone.h;fi;\

# ---------------------------------- for "modules" target

# rule to make *.mod and *.xinc - cpp directives only file
# can not use %.mod rule as make is too clever :-(
%.f90.h : %.use %.inc
	@echo "$*" >>modorder        
	@echo "      module $*" >$*.f90.h
	@cat $*.use >>$*.f90.h
	@cat $*.inc >>$*.f90.h
	@echo "      end module $*" >>$*.f90.h
	@${CPP} -DPETSC_USE_FORTRAN_MODULES ${FCPPFLAGS} $*.f90.h |grep -v "^ *#" >$*.f90
	@${FC} -c ${FC_FLAGS} ${FFLAGS} $*.f90
	@cat $*.inc|grep "^ *#">$*.xinc

%.use : moddeps
	@cat moddeps|grep "^ *$*[ :]"|sed 's/^[^:]*: *//'|sed 's/  */\n/g'| grep -v '^ *$$'|sed 's/^/      use /' >$*.use 

# find header file and all what it includes, merge it together, prepend by
# previous *.xinc
%.inc : ipath
	@for dir in `cat ipath`;\
	do if [ -e "$${dir}/${RPATH}/$*.h" ];\
	  then cat "$${dir}/${RPATH}/$*.h" >$*.xinc;\
	    if [ -e "$${dir}/${RPATH}/$*.h90" ];\
	      then cat "$${dir}/${RPATH}/$*.h90" >>$*.xinc;\
	    fi;\
	    break;\
	  fi;\
	done;\
	INCLUDED=`cat $*.xinc|grep "^ *#include"|sed 's/^[^"<]*["<]\([^">]*\).*/\1 /'`;\
	for i in $$INCLUDED;\
	do name="$${i##*/}";rpath="$${i%$$name}";\
	  ${OMAKE} -f makefile.tmp RPATH=$$rpath "$${name%.h}.inc";\
	  cat $*.xinc|sed "\%^ *#include *[\"<]$$i%r $${name%.h}.inc">__tmp;\
	  ${MV} -f __tmp $*.xinc;\
	done
	@${RM} -f $*.inc;touch $*.inc;\
	if [ -e modorder ]; then LAST=`tail -n 1 modorder`;fi;\
	if [ -e $$LAST.xinc ]; then cat $$LAST.xinc >>$*.inc;fi;\
	cat $*.xinc|grep -v "^ *#include" >>$*.inc;\


ipath:
	@echo "${PETSC_DIR}/${PETSC_ARCH}/include ${PETSC_DIR}/include/finclude ${PETSC_DIR}/include ${PETSC_DIR} ${PACKAGES_INCLUDES}  "|sed 's/-I//g' > ipath

