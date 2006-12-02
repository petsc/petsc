
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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test







.SUFFIXES:
.SUFFIXES: .mod .inc .use

# don't remake moddeps, makefile
moddeps:
	touch moddeps
makefile:
	touch makefile

# remake all
forceall: clean_all all

# remake anly if we havn't allinone.h
all: makemodules allinone.h tidy_up

# clean all
clean_all: tidy_up
	rm -f *.mod

# leave only final files
tidy_up:
	rm -f ipath __tmp *.xinc *.f90.h *.f90 *.use *.inc modorder makefile.tmp allinone.h *.o *.f90

makefile.tmp: moddeps
	cp makefile makefile.tmp;\
	cat moddeps|grep -v '^ *#'|grep ':'|\
	sed 's/^ *\([^ ]*\)\(.*\)/\1\2 \1.f90.h/' >> makefile.tmp;\
	MODS=`cat moddeps|grep -v '^ *#'|grep ':'|sed 's/^ *\([^ ]*\).*/\1 /'`;\
	echo modules : $$MODS >> makefile.tmp

makemodules: makefile.tmp
	if [ ! -e allinone.h ]; then\
	  rm -f modorder;\
	  make -f makefile.tmp modules;\
	fi

allinone.h: modorder
	LAST=`tail -n 1 modorder`;if [ -e $$LAST.xinc ]; then cat $$LAST.xinc >>allinone.h;fi;\

# ---------------------------------- for "modules" target

# rule to make *.mod and *.xinc - cpp directives only file
# can not use %.mod rule as make is too clever :-(
%.f90.h : %.use %.inc
	echo "$*" >>modorder        
	# create source
	echo "      module $*" >$*.f90.h
	cat $*.use >>$*.f90.h
	cat $*.inc >>$*.f90.h
	echo "      end module $*" >>$*.f90.h
	${CPP} ${FCPPFLAGS} $*.f90.h |grep -v "^ *#" >$*.f90
	${FC} -c ${FC_FLAGS} ${FFLAGS} $*.f90
	# leave just preprocessor's directives in *.xinc
	cat $*.inc|grep "^ *#">$*.xinc

%.use : moddeps
	echo "USE: $*"
	cat moddeps|grep "^ *$*[ :]"|sed 's/^[^:]*: *//'|sed 's/  */\n/g'|\
	grep -v '^ *$$'|sed 's/^/      use /' >$*.use 

# find header file and all what it includes, merge it together, prepend by
# previous *.xinc
%.inc : ipath
	# search and copy header files
	for dir in `cat ipath`;\
	do if [ -e "$${dir}/${RPATH}/$*.h" ];\
	  then cat "$${dir}/${RPATH}/$*.h" >$*.xinc;\
	    if [ -e "$${dir}/${RPATH}/$*.h90" ];\
	      then cat "$${dir}/${RPATH}/$*.h90" >>$*.xinc;\
	    fi;\
	    break;\
	  fi;\
	done;\
	# make *.inc for all included headers and insert them
	INCLUDED=`cat $*.xinc|grep "^ *#include"|sed 's/^[^"<]*["<]\([^">]*\).*/\1 /'`;\
	for i in $$INCLUDED;\
	do name="$${i##*/}";rpath="$${i%$$name}";\
	  make -f makefile.tmp RPATH=$$rpath "$${name%.h}.inc";\
	  cat $*.xinc|sed "\%^ *#include *[\"<]$$i%r $${name%.h}.inc">__tmp;\
	  mv -f __tmp $*.xinc;\
	done
	# insert *.xinc of previous modules
	rm -f $*.inc;touch $*.inc;\
	if [ -e modorder ]; then LAST=`tail -n 1 modorder`;fi;\
	if [ -e $$LAST.xinc ]; then cat $$LAST.xinc >>$*.inc;fi;\
	cat $*.xinc|grep -v "^ *#include" >>$*.inc;\


ipath:
	echo "-I/usr/include ${PETSC_DIR}/bmake/${PETSC_ARCH} ${PETSC_DIR}/include/finclude ${PETSC_DIR}/include ${PETSC_DIR} ${PACKAGES_INCLUDES}  "|sed 's/-I//g' > ipath

