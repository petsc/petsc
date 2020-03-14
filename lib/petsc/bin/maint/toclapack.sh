#!/bin/bash 

#
# Author: Eloy Romero (slepc-maint@upv.es)
#         Universidad Politecnica de Valencia, Spain
#

###############################################################################
#  This script generates the PETSc external package f2cblaslapack from the
#  LAPACK and BLAS fortran sources.
#
#  It is based on the CLAPACK scripts, see http://www.netlib.org/clapack/
#
#  Usage: toclapack.sh [blas-src-dir] [lapack-src-dir]
#  where blas-src-dir and lapack-src-dir are the SRC directories that
#  contain the fortran sources for BLAS and LAPACK, respectively.
#
#  This script needs the following tools:
#	*) f2c available from http://www.netlib.org/f2c or as a tarball at http://pkgs.fedoraproject.org/repo/pkgs/f2c/
#	*) lex
#	*) c compiler (cc)
#	*) awk
#	*) sed - the sed on Apple does not work with this script so you must do
#          brew install gnu-sed
#          to install a suitable sed. This script will automatically use that version once you have installed it
#       *) the BLAS and LAPACK source code. Download the tarball from http://www.netlib.org/lapack/lapack-3.4.2.tgz
#             blas-src-dir is lapack-version/BLAS/SRC
#             lapack-src-dir is lapack-version/SRC
#             newer releases of LAPACK will not work
#
#  This script should be run with bash, because it uses the 'echo -n' option. 

if [ $# -lt 2 ]
then
  echo Usage: toclapack.sh [blas-src-dir] [lapack-src-dir]
  exit
fi

# Path tools and temp directory
F2C=f2c
LEX=lex
CC=cc
AWK=awk
TMP=${PWD}/toclapack.$$
LEXFLAGS=-lfl
SED=sed
TAR=tar

# Some vars
FBLASLAPACK=f2cblaslapack-3.4.2.q3
BIN=${TMP}/bin
PAC=${TMP}/${FBLASLAPACK}
BLASDIR=${PAC}/blas
LAPACKDIR=${PAC}/lapack
BLASSRC="$1"
LAPACKSRC="$2"
ORIG="$PWD"
MAXPROCS="16"
TESTING="0"   # 0: don't include second, dsecnd and qsecnd

if [ `uname` = Darwin ]
then
  SED=gsed
fi

# 0) Create the temp directory and compile some scripts
mkdir $TMP
mkdir $BIN
mkdir $PAC
mkdir $BLASDIR
mkdir $LAPACKDIR

cat <<EOF > $BIN/lenscrub.l
/* {definitions} */
iofun	"("[^;]*";"
decl	"("[^)]*")"[,;\n]
rdecl	"("[^)]*")"[^{]*";"
any	[.]*
S	[ \t\n]*
cS	","{S}
len	[a-z][a-z0-9]*_len

%%
"s_cat"{iofun}          |
"s_copy"{iofun}         |
"s_stop"{iofun}         |
"s_cmp"{iofun}          |
"i_len"{iofun}          |
"do_fio"{iofun}         |
"do_lio"{iofun}         { printf("%s", yytext); /* unchanged */ }
{any}"ilaenv_("         |
[a-z]"tim"[a-z0-9]*"_(" |
[a-z]"prtb"[a-z0-9]"_(" { 
                          register int c, paran_count = 1;
                          printf("%s", yytext); /* unchanged */ 	
			  /* Loop until the correct closing paranthesis */
                          while (paran_count != 0) {
                              c = input();
                              if (c == '(') ++paran_count;
                              else if (c == ')') --paran_count;
                              putchar(c);
                          } 
                        }
{cS}[1-9]([0-9])*L	{ ; /* omit */ }
{cS}"("{S}"ftnlen"{S}")"{S}([0-9])*{S}	{ ; /* omit */ }
{cS}ftnlen({S}{len})?	{ ; /* omit -- f2c -A */ }
^ftnlen" "{len}";\n"	{ ; /* omit -- f2c without -A or -C++ */ }
{cS}{len}		{ ; }
.			{ printf("%s", yytext); /* unchanged */ }
EOF

${LEX} -o${BIN}/lenscrub.c ${BIN}/lenscrub.l
${CC} -o ${BIN}/lenscrub ${BIN}/lenscrub.c ${LEXFLAGS}

# 1) Write down the package makefile

echo '

########################################################################################
# f2cblaslapack: BLAS/LAPACK in C for being linked with PETSc.
# Created by $PETSC_DIR/lib/petsc/bin/maint/petsc/toclapack.sh script
# You may obtain PETSc at http://www.mcs.anl.gov/petsc
########################################################################################

ALL: blas_lib lapack_lib

########################################################################################
# Specify options to compile and create libraries
########################################################################################
CC         = cc
COPTFLAGS  = -O
CNOOPT     = -O0
RM         = /bin/rm
AR         = ar
AR_FLAGS   = cr
LIB_SUFFIX = a
RANLIB     = ranlib
TAR        = tar
########################################################################################
# By default, pick up the options from the PETSc configuration files
########################################################################################
BLASLAPACK_TYPE  = F2CBLASLAPACK
include ${PETSC_DIR}/lib/petsc/conf/base

########################################################################################
# compile the source files and create the blas and lapack libs
########################################################################################

BLAS_LIB_NAME       = libf2cblas.$(LIB_SUFFIX)
LAPACK_LIB_NAME     = libf2clapack.$(LIB_SUFFIX)
MAKE_OPTIONS        =  CC="$(CC)" COPTFLAGS="$(COPTFLAGS)" CNOOPT="$(CNOOPT)" AR="$(AR)" AR_FLAGS="$(AR_FLAGS)" RM="$(RM)"
MAKE_OPTIONS_BLAS   = $(MAKE_OPTIONS) LIBNAME="$(BLAS_LIB_NAME)"
MAKE_OPTIONS_LAPACK = $(MAKE_OPTIONS) LIBNAME="$(LAPACK_LIB_NAME)"

blas_lib:
	-@cd blas;   $(MAKE) lib $(MAKE_OPTIONS_BLAS)
	-@$(RANLIB) $(BLAS_LIB_NAME)

lapack_lib:
	-@cd lapack; $(MAKE) lib $(MAKE_OPTIONS_LAPACK)
	-@$(RANLIB) $(LAPACK_LIB_NAME)

single:
	-@cd blas;   $(MAKE) single $(MAKE_OPTIONS_BLAS)
	-@cd lapack; $(MAKE) single $(MAKE_OPTIONS_LAPACK)
	-@$(RANLIB) $(BLAS_LIB_NAME) $(LAPACK_LIB_NAME)

double:
	-@cd blas;   $(MAKE) double $(MAKE_OPTIONS_BLAS)
	-@cd lapack; $(MAKE) double $(MAKE_OPTIONS_LAPACK)
	-@$(RANLIB) $(BLAS_LIB_NAME) $(LAPACK_LIB_NAME)

quad:
	-@cd blas;   $(MAKE) quad $(MAKE_OPTIONS_BLAS)
	-@cd lapack; $(MAKE) quad $(MAKE_OPTIONS_LAPACK)
	-@$(RANLIB) $(BLAS_LIB_NAME) $(LAPACK_LIB_NAME)

half:
	-@cd blas;   $(MAKE) half $(MAKE_OPTIONS_BLAS)
	-@cd lapack; $(MAKE) half $(MAKE_OPTIONS_LAPACK)
	-@$(RANLIB) $(BLAS_LIB_NAME) $(LAPACK_LIB_NAME)

cleanblaslapck:
	$(RM) */*.o

cleanlib:
	$(RM) ./*.a ./*.lib

########################################################################################
# Target to create the f2cblaslapack distribution - using gnu-tar
########################################################################################
dist: cleanblaslapck cleanlib
	cd ..; 	$(RM) f2cblaslapack.tar.gz; \
	$(TAR) --create --gzip --file f2cblaslapack.tar.gz f2cblaslapack
' > ${PAC}/makefile

# 2) Transform fortran source to c from blas and lapack

# Create blacklist of files that won't be compiled
# Those functions correspond to extra precision routines
cat > ${TMP}/black.list << EOF
SXLASRC = sgesvxx.o sgerfsx.o sla_gerfsx_extended.o sla_geamv.o		\
   sla_gercond.o sla_rpvgrw.o ssysvxx.o ssyrfsx.o			\
   sla_syrfsx_extended.o sla_syamv.o sla_syrcond.o sla_syrpvgrw.o	\
   sposvxx.o sporfsx.o sla_porfsx_extended.o sla_porcond.o		\
   sla_porpvgrw.o sgbsvxx.o sgbrfsx.o sla_gbrfsx_extended.o		\
   sla_gbamv.o sla_gbrcond.o sla_gbrpvgrw.o sla_lin_berr.o slarscl2.o	\
   slascl2.o sla_wwaddw.o
DXLASRC = dgesvxx.o dgerfsx.o dla_gerfsx_extended.o dla_geamv.o		\
   dla_gercond.o dla_rpvgrw.o dsysvxx.o dsyrfsx.o			\
   dla_syrfsx_extended.o dla_syamv.o dla_syrcond.o dla_syrpvgrw.o	\
   dposvxx.o dporfsx.o dla_porfsx_extended.o dla_porcond.o		\
   dla_porpvgrw.o dgbsvxx.o dgbrfsx.o dla_gbrfsx_extended.o		\
   dla_gbamv.o dla_gbrcond.o dla_gbrpvgrw.o dla_lin_berr.o dlarscl2.o	\
   dlascl2.o dla_wwaddw.o
CXLASRC =    cgesvxx.o cgerfsx.o cla_gerfsx_extended.o cla_geamv.o \
   cla_gercond_c.o cla_gercond_x.o cla_rpvgrw.o \
   csysvxx.o csyrfsx.o cla_syrfsx_extended.o cla_syamv.o \
   cla_syrcond_c.o cla_syrcond_x.o cla_syrpvgrw.o \
   cposvxx.o cporfsx.o cla_porfsx_extended.o \
   cla_porcond_c.o cla_porcond_x.o cla_porpvgrw.o \
   cgbsvxx.o cgbrfsx.o cla_gbrfsx_extended.o cla_gbamv.o \
   cla_gbrcond_c.o cla_gbrcond_x.o cla_gbrpvgrw.o \
   chesvxx.o cherfsx.o cla_herfsx_extended.o cla_heamv.o \
   cla_hercond_c.o cla_hercond_x.o cla_herpvgrw.o \
   cla_lin_berr.o clarscl2.o clascl2.o cla_wwaddw.o
ZXLASRC = zgesvxx.o zgerfsx.o zla_gerfsx_extended.o zla_geamv.o		\
   zla_gercond_c.o zla_gercond_x.o zla_rpvgrw.o zsysvxx.o zsyrfsx.o	\
   zla_syrfsx_extended.o zla_syamv.o zla_syrcond_c.o zla_syrcond_x.o	\
   zla_syrpvgrw.o zposvxx.o zporfsx.o zla_porfsx_extended.o		\
   zla_porcond_c.o zla_porcond_x.o zla_porpvgrw.o zgbsvxx.o zgbrfsx.o	\
   zla_gbrfsx_extended.o zla_gbamv.o zla_gbrcond_c.o zla_gbrcond_x.o	\
   zla_gbrpvgrw.o zhesvxx.o zherfsx.o zla_herfsx_extended.o		\
   zla_heamv.o zla_hercond_c.o zla_hercond_x.o zla_herpvgrw.o		\
   zla_lin_berr.o zlarscl2.o zlascl2.o zla_wwaddw.o
EOF

QL=${TMP}/ql.sed
echo "
	s/doublereal/quadreal/g;
	s/doublecomplex/quadcomplex/g;
	s/([^a-zA-Z_]+)real/\\1doublereal/g;
	s/([^a-zA-Z_1-9]+)dlamch_([^a-zA-Z_1-9]+)/\\1qlamch_\\2/g;
	s/([^a-zA-Z_1-9]+)dlamc1_([^a-zA-Z_1-9]+)/\\1qlamc1_\\2/g;
	s/([^a-zA-Z_1-9]+)dlamc2_([^a-zA-Z_1-9]+)/\\1qlamc2_\\2/g;
	s/([^a-zA-Z_1-9]+)dlamc3_([^a-zA-Z_1-9]+)/\\1qlamc3_\\2/g;" > $QL

HL=${TMP}/hl.sed
echo "
	s/doublereal/halfreal/g;
	s/doublecomplex/halfcomplex/g;
	s/([^a-zA-Z_]+)real/\\1doublereal/g;
	s/([^a-zA-Z_1-9]+)dlamch_([^a-zA-Z_1-9]+)/\\1hlamch_\\2/g;
	s/([^a-zA-Z_1-9]+)dlamc1_([^a-zA-Z_1-9]+)/\\1hlamc1_\\2/g;
	s/([^a-zA-Z_1-9]+)dlamc2_([^a-zA-Z_1-9]+)/\\1hlamc2_\\2/g;
	s/([^a-zA-Z_1-9]+)dlamc3_([^a-zA-Z_1-9]+)/\\1hlamc3_\\2/g;" > $HL

for p in blas qblas hblas lapack qlapack hlapack; do
	case $p in
	blas) 
		SRC="$BLASSRC"
		DES="$BLASDIR"
		NOOP=""
		echo "pow_ii" > ${TMP}/AUX.list
		echo $'pow_si\nsmaxloc\nsf__cabs' > ${TMP}/SINGLE.list
		echo $'pow_di\ndmaxloc\ndf__cabs' > ${TMP}/DOUBLE.list
		cd $SRC
		files="`ls *.f`"
		cd -
		;;
	qblas) 
		SRC="$TMP"
		DES="$BLASDIR"
		NOOP=""
		echo $'pow_qi\nqmaxloc\nqf__cabs' > ${TMP}/QUAD.list
		files="`cat ${TMP}/ql.list`"
		;;
	hblas) 
		SRC="$TMP"
		DES="$BLASDIR"
		NOOP=""
		echo $'pow_hi\nhmaxloc\nhf__cabs' > ${TMP}/HALF.list
		files="`cat ${TMP}/hl.list`"
		;;

	lapack)
		SRC="$LAPACKSRC"
		DES="$LAPACKDIR"
		NOOP="slaruv dlaruv slamch dlamch"
		rm ${TMP}/AUX.list
		echo 'slamch' > ${TMP}/SINGLE.list
		echo 'dlamch' > ${TMP}/DOUBLE.list
		if [[ ${TESTING} != "0" ]]; then
			echo 'second' >> ${TMP}/SINGLE.list
			echo 'dsecnd' >> ${TMP}/DOUBLE.list
		fi
		rm ${TMP}/QUAD.list
		rm ${TMP}/HALF.list
		rm ${TMP}/ql.list
		rm ${TMP}/hl.list
		cd $SRC
		files="`ls *.f`"
		cd -
		;;
	qlapack)
		NOOP="qlaruv qlamch"
		echo $'qlamch' > ${TMP}/QUAD.list
		SRC="$TMP"
		DES="$LAPACKDIR"
		files="`cat ${TMP}/ql.list`"
		;;

        hlapack)
		NOOP="hlaruv hlamch"
		echo $'hlamch' > ${TMP}/HALF.list
		SRC="$TMP"
		DES="$LAPACKDIR"
		files="`cat ${TMP}/hl.list`"
	esac

	# Transform sources
	BACK="${PWD}"
	cd $SRC
	NPROC="0"
	for file in $files; do
		base=`echo $file | $SED -e 's/\.f//g'`
		[[ ${p} = lapack || ${p} = qlapack || ${p} = hlapack ]] && grep -q ${base}.o ${TMP}/black.list && continue

		# Get the precision of the BLAS and LAPACK routines
		case $base in
		chla_transtype)	PR="AUX";;
		sdsdot)		PR="DOUBLE";;
		dqddot)		PR="QUAD";;
		dhddot)		PR="HALF";;
		i[sc]amax)	PR="SINGLE";;
		i[dz]amax)	PR="DOUBLE";;
		i[qw]amax)	PR="QUAD";;
		i[hk]amax)	PR="HALF";;
		[sc]*)		PR="SINGLE";;
		[dz]*)		PR="DOUBLE";;
		[qw]*)		PR="QUAD";;
		[hk]*)		PR="HALF";;
		icmax1)		PR="SINGLE";;
		izmax1)		PR="DOUBLE";;
		iwmax1)		PR="QUAD";;
		ikmax1)		PR="HALF";;
		ila[sc]l[rc])	PR="SINGLE";;
		ila[dz]l[rc])	PR="DOUBLE";;
		ila[qw]l[rc])	PR="QUAD";;
		ila[hk]l[rc])	PR="HALF";;
		*)		PR="AUX";;
		esac

		# Due to limitations of f2c the next changes are performed in the Fortran code
		# - Remove the RECURSIVE Fortran keyword
		# - Replace CHARACTER(1) by CHARACTER
		# - Replace the intrinsic functions exit and maxloc by the macros myexit and mymaxloc
		# - Replace sqrt, sin, cos, log and exp by M(*)
		# - Replace max and min by f2cmax and f2cmin
		$SED -r -e "
			s/RECURSIVE//g;
			s/CHARACTER\\(1\\)/CHARACTER/g;
			s/EXIT/CALL MYEXIT/g;
			s/MAXLOC/MYMAXLOC/g;
			s/(^ *SUBROUTINE[^(]+\\([^)]+\\))/\\1\\n      EXTERNAL MYEXIT, MYMAXLOC\\n       INTEGER MYMAXLOC\\n/g;
			s/(INTRINSIC [^\\n]*)MYMAXLOC/\\1 MAX/g;
			s/MAXLOC\\(([^:]+):/MAXLOC(\\1,/g;
			s/MAXLOC\\(([^(]+)\\((.+)\\),/MAXLOC( \\1, \\2,/g;
		" ${base}.f |
		$F2C -a -A -R | ${BIN}/lenscrub |
		$SED -r -e "
			/\\/\\*  *\\.\\. .*\\*\\//d;
			s/extern integer mymaxloc_\\([^)]*\\);//g;
			s/myexit_\\(void\\)/mecago_()/g;
			$( for i in sqrt sin cos log exp; do
				echo "s/([^a-zA-Z_1-9]+)${i}([^a-zA-Z_1-9]+)/\\1M(${i})\\2/g;"
			done )
			s/([^a-zA-Z_1-9]+)max([^a-zA-Z_1-9]+)/\\1f2cmax\\2/g;
			s/([^a-zA-Z_1-9]+)min([^a-zA-Z_1-9]+)/\\1f2cmin\\2/g;" |
		$AWK '
			BEGIN {	a=1; }
			{
				i=index($0, "/* Builtin functions */");
				if((i==0) && (a==1)) print;
				else a=0;
			}
			/^$/ {	a=1; }' |
		$SED -e "
			s/#include \"f2c.h\"/#define __LAPACK_PRECISION_$PR\\n&/g" |
		if [[ $p = qblas || $p = qlapack ]]; then
			$SED -r -f $QL
		elif [[ $p = hblas || $p = hlapack ]]; then
			$SED -r -f $HL
		else
			cat
		fi > ${DES}/${base}.c &

		# Quick way to parallelize this loop
		NPROC="$(( NPROC+1 ))"
		if [ "$NPROC" -ge "$MAXPROCS" ]; then wait; NPROC="0"; fi

		# Create the routines with quad precision from the double ones
		if [[ $PR = DOUBLE ]]; then
			qbase="$( echo $base | $SED -r -e '
				s/^dcabs1/qcabs1/
				s/^dsdot/qddot/
				s/^dz/qw/
				/^i?d[^z]/ { s/d/q/ }
				s/^zd/wq/
				/^i?z[^d]/ { s/z/w/ }
				s/^sdsdot/dqddot/
				/^ila[dz]l[rc]/ { y/dz/qw/; }' )";
			echo "s/([^a-zA-Z_1-9]+)${base}_([^a-zA-Z_1-9]+)/\\1${qbase}_\\2/g;" >> $QL
			cp $base.f ${TMP}/${qbase}.f
			echo ${qbase}.f >> ${TMP}/ql.list
		fi

		# Create the routines with half precision from the double ones  h - half-real k half-complex
		if [[ $PR = DOUBLE ]]; then
			hbase="$( echo $base | $SED -r -e '
				s/^dcabs1/hcabs1/
				s/^dsdot/hddot/
				s/^dz/hw/
				/^i?d[^z]/ { s/d/h/ }
				s/^zd/wh/
				/^i?z[^d]/ { s/z/k/ }
				s/^sdsdot/dhddot/
				/^ila[dz]l[rc]/ { y/dz/hk/; }' )";
			echo "s/([^a-zA-Z_1-9]+)${base}_([^a-zA-Z_1-9]+)/\\1${hbase}_\\2/g;" >> $HL
			cp $base.f ${TMP}/${hbase}.f
			echo ${hbase}.f >> ${TMP}/hl.list
		fi

		# Separate the files by precision
		echo $base >> ${TMP}/${PR}.list
	done
	wait
        cd $BACK

	# Create the makefile
	case $p in
	blas|lapack)
		cat >> ${DES}/makefile << EOF
AUXO = `cat ${TMP}/AUX.list | $AWK '{printf("%s.o ", $1)}'`
SINGLEO = `cat ${TMP}/SINGLE.list | $AWK '{printf("%s.o ", $1)}'`
DOUBLEO = `cat ${TMP}/DOUBLE.list | $AWK '{printf("%s.o ", $1)}'`

lib: \$(SINGLEO) \$(DOUBLEO) \$(AUXO)
	\$(AR) \$(AR_FLAGS) ../\$(LIBNAME) \$(SINGLEO) \$(DOUBLEO) \$(AUXO)

single: \$(SINGLEO) \$(AUXO)
	\$(AR) \$(AR_FLAGS) ../\$(LIBNAME) \$(SINGLEO) \$(AUXO)

double: \$(DOUBLEO) \$(AUXO)
	\$(AR) \$(AR_FLAGS) ../\$(LIBNAME) \$(DOUBLEO) \$(AUXO)

.c.o:
	\$(CC) \$(COPTFLAGS) -c \$< -o \$@
EOF
		;;

	qblas|qlapack)
		cat >> ${DES}/makefile << EOF
QUADO = `cat ${TMP}/QUAD.list | $AWK '{printf("%s.o ", $1)}'`

quad: \$(QUADO) \$(AUXO)
	\$(AR) \$(AR_FLAGS) ../\$(LIBNAME) \$(QUADO) \$(AUXO)

qlib: \$(SINGLEO) \$(DOUBLEO) \$(QUADO) \$(AUXO)
	\$(AR) \$(AR_FLAGS) ../\$(LIBNAME) \$(SINGLEO) \$(DOUBLEO) \$(QUADO) \$(AUXO)


EOF
		;;
	hblas|hlapack)
		cat >> ${DES}/makefile << EOF
HALFO = `cat ${TMP}/HALF.list | $AWK '{printf("%s.o ", $1)}'`

half: \$(HALFO) \$(AUXO)
	\$(AR) \$(AR_FLAGS) ../\$(LIBNAME) \$(HALFO) \$(AUXO)

hlib: \$(SINGLEO) \$(DOUBLEO) \$(HALFO) \$(AUXO)
	\$(AR) \$(AR_FLAGS) ../\$(LIBNAME) \$(SINGLEO) \$(DOUBLEO) \$(HALFO) \$(AUXO)

qhlib: \$(SINGLEO) \$(DOUBLEO) \$(QUADO) \$(HALFO) \$(AUXO)
	\$(AR) \$(AR_FLAGS) ../\$(LIBNAME) \$(SINGLEO) \$(DOUBLEO) \$(QUADO) \$(HALFO \$(AUXO)

EOF
		;;
	esac

	# Add to the makefile the files that should be built without optimizations
	for f in $NOOP; do
		echo "${f}.o: ${f}.c ; \$(CC) \$(CNOOPT) -c \$< -o \$@" >> ${DES}/makefile
	done
done
        # remove duplicate xerbla.o xerbla_array.o from lapack makefile
        $SED -i -e 's/xerbla.o//; s/xerbla_array.o//' ${LAPACKDIR}/makefile

	# Take care some special source files
	cat << EOF > ${TMP}/xerbla.c
#include "f2c.h"

int xerbla_(char *srname, integer *info) {
    printf("** On entry to %6s, parameter number %2i had an illegal value\n",
		srname, *info);
    return 0;
}
EOF
	cp ${TMP}/xerbla.c ${BLASDIR}
#	cp ${TMP}/xerbla.c ${LAPACKDIR}

	cat << EOF > ${TMP}/xerbla_array.c
#include "f2c.h"
#include <stdlib.h>

int xerbla_array_(char *srname, integer *info, int len) {
    char *n = (char*)malloc(sizeof(char)*(len+1));
    memcpy(n, srname, len*sizeof(char));
    n[len]=0;
    printf("** On entry to %6s, parameter number %2i had an illegal value\n",
		n, *info);
    free(n);
    return 0;
}
EOF
	cp ${TMP}/xerbla_array.c ${BLASDIR}
#	cp ${TMP}/xerbla_array.c ${LAPACKDIR}

	cat << EOF > ${LAPACKDIR}/slamch.c
/*  -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#define __LAPACK_PRECISION_SINGLE
#include "f2c.h"

/* Table of constant values */

static integer c__1 = 1;
static real c_b32 = 0.f;

real slamch_(char *cmach)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    integer i__1;
    real ret_val;

    /* Local variables */
    static real t;
    static integer it;
    static real rnd, eps, base;
    static integer beta;
    static real emin, prec, emax;
    static integer imin, imax;
    static logical lrnd;
    static real rmin, rmax, rmach;
    extern logical lsame_(char *, char *);
    static real small, sfmin;
    extern /* Subroutine */ int slamc2_(integer *, integer *, logical *, real 
	    *, integer *, real *, integer *, real *);


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAMCH determines single precision machine parameters. */

/*  Arguments */
/*  ========= */

/*  CMACH   (input) CHARACTER*1 */
/*          Specifies the value to be returned by SLAMCH: */
/*          = 'E' or 'e',   SLAMCH := eps */
/*          = 'S' or 's ,   SLAMCH := sfmin */
/*          = 'B' or 'b',   SLAMCH := base */
/*          = 'P' or 'p',   SLAMCH := eps*base */
/*          = 'N' or 'n',   SLAMCH := t */
/*          = 'R' or 'r',   SLAMCH := rnd */
/*          = 'M' or 'm',   SLAMCH := emin */
/*          = 'U' or 'u',   SLAMCH := rmin */
/*          = 'L' or 'l',   SLAMCH := emax */
/*          = 'O' or 'o',   SLAMCH := rmax */

/*          where */

/*          eps   = relative machine precision */
/*          sfmin = safe minimum, such that 1/sfmin does not overflow */
/*          base  = base of the machine */
/*          prec  = eps*base */
/*          t     = number of (base) digits in the mantissa */
/*          rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise */
/*          emin  = minimum exponent before (gradual) underflow */
/*          rmin  = underflow threshold - base**(emin-1) */
/*          emax  = largest exponent before overflow */
/*          rmax  = overflow threshold  - (base**emax)*(1-eps) */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	slamc2_(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
	base = (real) beta;
	t = (real) it;
	if (lrnd) {
	    rnd = 1.f;
	    i__1 = 1 - it;
	    eps = pow_ri(&base, &i__1) / 2;
	} else {
	    rnd = 0.f;
	    i__1 = 1 - it;
	    eps = pow_ri(&base, &i__1);
	}
	prec = eps * base;
	emin = (real) imin;
	emax = (real) imax;
	sfmin = rmin;
	small = 1.f / rmax;
	if (small >= sfmin) {

/*           Use SMALL plus a bit, to avoid the possibility of rounding */
/*           causing overflow when computing  1/sfmin. */

	    sfmin = small * (eps + 1.f);
	}
    }

    if (lsame_(cmach, "E")) {
	rmach = eps;
    } else if (lsame_(cmach, "S")) {
	rmach = sfmin;
    } else if (lsame_(cmach, "B")) {
	rmach = base;
    } else if (lsame_(cmach, "P")) {
	rmach = prec;
    } else if (lsame_(cmach, "N")) {
	rmach = t;
    } else if (lsame_(cmach, "R")) {
	rmach = rnd;
    } else if (lsame_(cmach, "M")) {
	rmach = emin;
    } else if (lsame_(cmach, "U")) {
	rmach = rmin;
    } else if (lsame_(cmach, "L")) {
	rmach = emax;
    } else if (lsame_(cmach, "O")) {
	rmach = rmax;
    }

    ret_val = rmach;
    first = FALSE_;
    return ret_val;

/*     End of SLAMCH */

} /* slamch_ */


/* *********************************************************************** */

/* Subroutine */ int slamc1_(integer *beta, integer *t, logical *rnd, logical 
	*ieee1)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    real r__1, r__2;

    /* Local variables */
    static real a, b, c__, f, t1, t2;
    static integer lt;
    static real one, qtr;
    static logical lrnd;
    static integer lbeta;
    static real savec;
    static logical lieee1;
    extern real slamc3_(real *, real *);


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAMC1 determines the machine parameters given by BETA, T, RND, and */
/*  IEEE1. */

/*  Arguments */
/*  ========= */

/*  BETA    (output) INTEGER */
/*          The base of the machine. */

/*  T       (output) INTEGER */
/*          The number of ( BETA ) digits in the mantissa. */

/*  RND     (output) LOGICAL */
/*          Specifies whether proper rounding  ( RND = .TRUE. )  or */
/*          chopping  ( RND = .FALSE. )  occurs in addition. This may not */
/*          be a reliable guide to the way in which the machine performs */
/*          its arithmetic. */

/*  IEEE1   (output) LOGICAL */
/*          Specifies whether rounding appears to be done in the IEEE */
/*          'round to nearest' style. */

/*  Further Details */
/*  =============== */

/*  The routine is based on the routine  ENVRON  by Malcolm and */
/*  incorporates suggestions by Gentleman and Marovich. See */

/*     Malcolm M. A. (1972) Algorithms to reveal properties of */
/*        floating-point arithmetic. Comms. of the ACM, 15, 949-951. */

/*     Gentleman W. M. and Marovich S. B. (1974) More on algorithms */
/*        that reveal properties of floating point arithmetic units. */
/*        Comms. of the ACM, 17, 276-277. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	one = 1.f;

/*        LBETA,  LIEEE1,  LT and  LRND  are the  local values  of  BETA, */
/*        IEEE1, T and RND. */

/*        Throughout this routine  we use the function  SLAMC3  to ensure */
/*        that relevant values are  stored and not held in registers,  or */
/*        are not affected by optimizers. */

/*        Compute  a = 2.0**m  with the  smallest positive integer m such */
/*        that */

/*           fl( a + 1.0 ) = a. */

	a = 1.f;
	c__ = 1.f;

/* +       WHILE( C.EQ.ONE )LOOP */
L10:
	if (c__ == one) {
	    a *= 2;
	    c__ = slamc3_(&a, &one);
	    r__1 = -a;
	    c__ = slamc3_(&c__, &r__1);
	    goto L10;
	}
/* +       END WHILE */

/*        Now compute  b = 2.0**m  with the smallest positive integer m */
/*        such that */

/*           fl( a + b ) .gt. a. */

	b = 1.f;
	c__ = slamc3_(&a, &b);

/* +       WHILE( C.EQ.A )LOOP */
L20:
	if (c__ == a) {
	    b *= 2;
	    c__ = slamc3_(&a, &b);
	    goto L20;
	}
/* +       END WHILE */

/*        Now compute the base.  a and c  are neighbouring floating point */
/*        numbers  in the  interval  ( beta**t, beta**( t + 1 ) )  and so */
/*        their difference is beta. Adding 0.25 to c is to ensure that it */
/*        is truncated to beta and not ( beta - 1 ). */

	qtr = one / 4;
	savec = c__;
	r__1 = -a;
	c__ = slamc3_(&c__, &r__1);
	lbeta = c__ + qtr;

/*        Now determine whether rounding or chopping occurs,  by adding a */
/*        bit  less  than  beta/2  and a  bit  more  than  beta/2  to  a. */

	b = (real) lbeta;
	r__1 = b / 2;
	r__2 = -b / 100;
	f = slamc3_(&r__1, &r__2);
	c__ = slamc3_(&f, &a);
	if (c__ == a) {
	    lrnd = TRUE_;
	} else {
	    lrnd = FALSE_;
	}
	r__1 = b / 2;
	r__2 = b / 100;
	f = slamc3_(&r__1, &r__2);
	c__ = slamc3_(&f, &a);
	if (lrnd && c__ == a) {
	    lrnd = FALSE_;
	}

/*        Try and decide whether rounding is done in the  IEEE  'round to */
/*        nearest' style. B/2 is half a unit in the last place of the two */
/*        numbers A and SAVEC. Furthermore, A is even, i.e. has last  bit */
/*        zero, and SAVEC is odd. Thus adding B/2 to A should not  change */
/*        A, but adding B/2 to SAVEC should change SAVEC. */

	r__1 = b / 2;
	t1 = slamc3_(&r__1, &a);
	r__1 = b / 2;
	t2 = slamc3_(&r__1, &savec);
	lieee1 = t1 == a && t2 > savec && lrnd;

/*        Now find  the  mantissa, t.  It should  be the  integer part of */
/*        log to the base beta of a,  however it is safer to determine  t */
/*        by powering.  So we find t as the smallest positive integer for */
/*        which */

/*           fl( beta**t + 1.0 ) = 1.0. */

	lt = 0;
	a = 1.f;
	c__ = 1.f;

/* +       WHILE( C.EQ.ONE )LOOP */
L30:
	if (c__ == one) {
	    ++lt;
	    a *= lbeta;
	    c__ = slamc3_(&a, &one);
	    r__1 = -a;
	    c__ = slamc3_(&c__, &r__1);
	    goto L30;
	}
/* +       END WHILE */

    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *ieee1 = lieee1;
    first = FALSE_;
    return 0;

/*     End of SLAMC1 */

} /* slamc1_ */


/* *********************************************************************** */

/* Subroutine */ int slamc2_(integer *beta, integer *t, logical *rnd, real *
	eps, integer *emin, real *rmin, integer *emax, real *rmax)
{
    /* Initialized data */

    static logical first = TRUE_;
    static logical iwarn = FALSE_;

    /* Format strings */
    static char fmt_9999[] = "(//\002 WARNING. The value EMIN may be incorre"
	    "ct:-\002,\002  EMIN = \002,i8,/\002 If, after inspection, the va"
	    "lue EMIN looks\002,\002 acceptable please comment out \002,/\002"
	    " the IF block as marked within the code of routine\002,\002 SLAM"
	    "C2,\002,/\002 otherwise supply EMIN explicitly.\002,/)";

    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3, r__4, r__5;

    /* Local variables */
    static real a, b, c__;
    static integer i__, lt;
    static real one, two;
    static logical ieee;
    static real half;
    static logical lrnd;
    static real leps, zero;
    static integer lbeta;
    static real rbase;
    static integer lemin, lemax, gnmin;
    static real small;
    static integer gpmin;
    static real third, lrmin, lrmax, sixth;
    static logical lieee1;
    extern /* Subroutine */ int slamc1_(integer *, integer *, logical *, 
	    logical *);
    extern real slamc3_(real *, real *);
    extern /* Subroutine */ int slamc4_(integer *, real *, integer *), 
	    slamc5_(integer *, integer *, integer *, logical *, integer *, 
	    real *);
    static integer ngnmin, ngpmin;

    /* Fortran I/O blocks */
    static cilist io___58 = { 0, 6, 0, fmt_9999, 0 };



/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAMC2 determines the machine parameters specified in its argument */
/*  list. */

/*  Arguments */
/*  ========= */

/*  BETA    (output) INTEGER */
/*          The base of the machine. */

/*  T       (output) INTEGER */
/*          The number of ( BETA ) digits in the mantissa. */

/*  RND     (output) LOGICAL */
/*          Specifies whether proper rounding  ( RND = .TRUE. )  or */
/*          chopping  ( RND = .FALSE. )  occurs in addition. This may not */
/*          be a reliable guide to the way in which the machine performs */
/*          its arithmetic. */

/*  EPS     (output) REAL */
/*          The smallest positive number such that */

/*             fl( 1.0 - EPS ) .LT. 1.0, */

/*          where fl denotes the computed value. */

/*  EMIN    (output) INTEGER */
/*          The minimum exponent before (gradual) underflow occurs. */

/*  RMIN    (output) REAL */
/*          The smallest normalized number for the machine, given by */
/*          BASE**( EMIN - 1 ), where  BASE  is the floating point value */
/*          of BETA. */

/*  EMAX    (output) INTEGER */
/*          The maximum exponent before overflow occurs. */

/*  RMAX    (output) REAL */
/*          The largest positive number for the machine, given by */
/*          BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating point */
/*          value of BETA. */

/*  Further Details */
/*  =============== */

/*  The computation of  EPS  is based on a routine PARANOIA by */
/*  W. Kahan of the University of California at Berkeley. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	zero = 0.f;
	one = 1.f;
	two = 2.f;

/*        LBETA, LT, LRND, LEPS, LEMIN and LRMIN  are the local values of */
/*        BETA, T, RND, EPS, EMIN and RMIN. */

/*        Throughout this routine  we use the function  SLAMC3  to ensure */
/*        that relevant values are stored  and not held in registers,  or */
/*        are not affected by optimizers. */

/*        SLAMC1 returns the parameters  LBETA, LT, LRND and LIEEE1. */

	slamc1_(&lbeta, &lt, &lrnd, &lieee1);

/*        Start to find EPS. */

	b = (real) lbeta;
	i__1 = -lt;
	a = pow_ri(&b, &i__1);
	leps = a;

/*        Try some tricks to see whether or not this is the correct  EPS. */

	b = two / 3;
	half = one / 2;
	r__1 = -half;
	sixth = slamc3_(&b, &r__1);
	third = slamc3_(&sixth, &sixth);
	r__1 = -half;
	b = slamc3_(&third, &r__1);
	b = slamc3_(&b, &sixth);
	b = dabs(b);
	if (b < leps) {
	    b = leps;
	}

	leps = 1.f;

/* +       WHILE( ( LEPS.GT.B ).AND.( B.GT.ZERO ) )LOOP */
L10:
	if (leps > b && b > zero) {
	    leps = b;
	    r__1 = half * leps;
/* Computing 5th power */
	    r__3 = two, r__4 = r__3, r__3 *= r__3;
/* Computing 2nd power */
	    r__5 = leps;
	    r__2 = r__4 * (r__3 * r__3) * (r__5 * r__5);
	    c__ = slamc3_(&r__1, &r__2);
	    r__1 = -c__;
	    c__ = slamc3_(&half, &r__1);
	    b = slamc3_(&half, &c__);
	    r__1 = -b;
	    c__ = slamc3_(&half, &r__1);
	    b = slamc3_(&half, &c__);
	    goto L10;
	}
/* +       END WHILE */

	if (a < leps) {
	    leps = a;
	}

/*        Computation of EPS complete. */

/*        Now find  EMIN.  Let A = + or - 1, and + or - (1 + BASE**(-3)). */
/*        Keep dividing  A by BETA until (gradual) underflow occurs. This */
/*        is detected when we cannot recover the previous A. */

	rbase = one / lbeta;
	small = one;
	for (i__ = 1; i__ <= 3; ++i__) {
	    r__1 = small * rbase;
	    small = slamc3_(&r__1, &zero);
/* L20: */
	}
	a = slamc3_(&one, &small);
	slamc4_(&ngpmin, &one, &lbeta);
	r__1 = -one;
	slamc4_(&ngnmin, &r__1, &lbeta);
	slamc4_(&gpmin, &a, &lbeta);
	r__1 = -a;
	slamc4_(&gnmin, &r__1, &lbeta);
	ieee = FALSE_;

	if (ngpmin == ngnmin && gpmin == gnmin) {
	    if (ngpmin == gpmin) {
		lemin = ngpmin;
/*            ( Non twos-complement machines, no gradual underflow; */
/*              e.g.,  VAX ) */
	    } else if (gpmin - ngpmin == 3) {
		lemin = ngpmin - 1 + lt;
		ieee = TRUE_;
/*            ( Non twos-complement machines, with gradual underflow; */
/*              e.g., IEEE standard followers ) */
	    } else {
		lemin = f2cmin(ngpmin,gpmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if (ngpmin == gpmin && ngnmin == gnmin) {
	    if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1) {
		lemin = f2cmax(ngpmin,ngnmin);
/*            ( Twos-complement machines, no gradual underflow; */
/*              e.g., CYBER 205 ) */
	    } else {
		lemin = f2cmin(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1 && gpmin == gnmin)
		 {
	    if (gpmin - f2cmin(ngpmin,ngnmin) == 3) {
		lemin = f2cmax(ngpmin,ngnmin) - 1 + lt;
/*            ( Twos-complement machines with gradual underflow; */
/*              no known machine ) */
	    } else {
		lemin = f2cmin(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else {
/* Computing MIN */
	    i__1 = f2cmin(ngpmin,ngnmin), i__1 = f2cmin(i__1,gpmin);
	    lemin = f2cmin(i__1,gnmin);
/*         ( A guess; no known machine ) */
	    iwarn = TRUE_;
	}
	first = FALSE_;
/* ** */
/* Comment out this if block if EMIN is ok */
	if (iwarn) {
	    first = TRUE_;
	    printf("\n\n WARNING. The value EMIN may be incorrect:- ");
	    printf("EMIN = %8i\n",lemin);
	    printf("If, after inspection, the value EMIN looks acceptable");
            printf("please comment out \n the IF block as marked within the"); 
            printf("code of routine SLAMC2, \n otherwise supply EMIN"); 
            printf("explicitly.\n");
	}
/* **   

          Assume IEEE arithmetic if we found denormalised  numbers abo
ve,   
          or if arithmetic seems to round in the  IEEE style,  determi
ned   
          in routine SLAMC1. A true IEEE machine should have both  thi
ngs   
          true; however, faulty machines may have one or the other. */

	ieee = ieee || lieee1;

/*        Compute  RMIN by successive division by  BETA. We could compute */
/*        RMIN as BASE**( EMIN - 1 ),  but some machines underflow during */
/*        this computation. */

	lrmin = 1.f;
	i__1 = 1 - lemin;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    r__1 = lrmin * rbase;
	    lrmin = slamc3_(&r__1, &zero);
/* L30: */
	}

/*        Finally, call SLAMC5 to compute EMAX and RMAX. */

	slamc5_(&lbeta, &lt, &lemin, &ieee, &lemax, &lrmax);
    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *eps = leps;
    *emin = lemin;
    *rmin = lrmin;
    *emax = lemax;
    *rmax = lrmax;

    return 0;


/*     End of SLAMC2 */

} /* slamc2_ */


/* *********************************************************************** */

real slamc3_(real *a, real *b)
{
    /* System generated locals */
    real ret_val;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAMC3  is intended to force  A  and  B  to be stored prior to doing */
/*  the addition of  A  and  B ,  for use in situations where optimizers */
/*  might hold one of these in a register. */

/*  Arguments */
/*  ========= */

/*  A       (input) REAL */
/*  B       (input) REAL */
/*          The values A and B. */

/* ===================================================================== */

/*     .. Executable Statements .. */

    ret_val = *a + *b;

    return ret_val;

/*     End of SLAMC3 */

} /* slamc3_ */


/* *********************************************************************** */

/* Subroutine */ int slamc4_(integer *emin, real *start, integer *base)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Local variables */
    static real a;
    static integer i__;
    static real b1, b2, c1, c2, d1, d2, one, zero, rbase;
    extern real slamc3_(real *, real *);


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAMC4 is a service routine for SLAMC2. */

/*  Arguments */
/*  ========= */

/*  EMIN    (output) INTEGER */
/*          The minimum exponent before (gradual) underflow, computed by */
/*          setting A = START and dividing by BASE until the previous A */
/*          can not be recovered. */

/*  START   (input) REAL */
/*          The starting point for determining EMIN. */

/*  BASE    (input) INTEGER */
/*          The base of the machine. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    a = *start;
    one = 1.f;
    rbase = one / *base;
    zero = 0.f;
    *emin = 1;
    r__1 = a * rbase;
    b1 = slamc3_(&r__1, &zero);
    c1 = a;
    c2 = a;
    d1 = a;
    d2 = a;
/* +    WHILE( ( C1.EQ.A ).AND.( C2.EQ.A ).AND. */
/*    $       ( D1.EQ.A ).AND.( D2.EQ.A )      )LOOP */
L10:
    if (c1 == a && c2 == a && d1 == a && d2 == a) {
	--(*emin);
	a = b1;
	r__1 = a / *base;
	b1 = slamc3_(&r__1, &zero);
	r__1 = b1 * *base;
	c1 = slamc3_(&r__1, &zero);
	d1 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d1 += b1;
/* L20: */
	}
	r__1 = a * rbase;
	b2 = slamc3_(&r__1, &zero);
	r__1 = b2 / rbase;
	c2 = slamc3_(&r__1, &zero);
	d2 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d2 += b2;
/* L30: */
	}
	goto L10;
    }
/* +    END WHILE */

    return 0;

/*     End of SLAMC4 */

} /* slamc4_ */


/* *********************************************************************** */

/* Subroutine */ int slamc5_(integer *beta, integer *p, integer *emin, 
	logical *ieee, integer *emax, real *rmax)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Local variables */
    static integer i__;
    static real y, z__;
    static integer try__, lexp;
    static real oldy;
    static integer uexp, nbits;
    extern real slamc3_(real *, real *);
    static real recbas;
    static integer exbits, expsum;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAMC5 attempts to compute RMAX, the largest machine floating-point */
/*  number, without overflow.  It assumes that EMAX + abs(EMIN) sum */
/*  approximately to a power of 2.  It will fail on machines where this */
/*  assumption does not hold, for example, the Cyber 205 (EMIN = -28625, */
/*  EMAX = 28718).  It will also fail if the value supplied for EMIN is */
/*  too large (i.e. too close to zero), probably with overflow. */

/*  Arguments */
/*  ========= */

/*  BETA    (input) INTEGER */
/*          The base of floating-point arithmetic. */

/*  P       (input) INTEGER */
/*          The number of base BETA digits in the mantissa of a */
/*          floating-point value. */

/*  EMIN    (input) INTEGER */
/*          The minimum exponent before (gradual) underflow. */

/*  IEEE    (input) LOGICAL */
/*          A logical flag specifying whether or not the arithmetic */
/*          system is thought to comply with the IEEE standard. */

/*  EMAX    (output) INTEGER */
/*          The largest exponent before overflow */

/*  RMAX    (output) REAL */
/*          The largest machine floating-point number. */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     First compute LEXP and UEXP, two powers of 2 that bound */
/*     abs(EMIN). We then assume that EMAX + abs(EMIN) will sum */
/*     approximately to the bound that is closest to abs(EMIN). */
/*     (EMAX is the exponent of the required number RMAX). */

    lexp = 1;
    exbits = 1;
L10:
    try__ = lexp << 1;
    if (try__ <= -(*emin)) {
	lexp = try__;
	++exbits;
	goto L10;
    }
    if (lexp == -(*emin)) {
	uexp = lexp;
    } else {
	uexp = try__;
	++exbits;
    }

/*     Now -LEXP is less than or equal to EMIN, and -UEXP is greater */
/*     than or equal to EMIN. EXBITS is the number of bits needed to */
/*     store the exponent. */

    if (uexp + *emin > -lexp - *emin) {
	expsum = lexp << 1;
    } else {
	expsum = uexp << 1;
    }

/*     EXPSUM is the exponent range, approximately equal to */
/*     EMAX - EMIN + 1 . */

    *emax = expsum + *emin - 1;
    nbits = exbits + 1 + *p;

/*     NBITS is the total number of bits needed to store a */
/*     floating-point number. */

    if (nbits % 2 == 1 && *beta == 2) {

/*        Either there are an odd number of bits used to store a */
/*        floating-point number, which is unlikely, or some bits are */
/*        not used in the representation of numbers, which is possible, */
/*        (e.g. Cray machines) or the mantissa has an implicit bit, */
/*        (e.g. IEEE machines, Dec Vax machines), which is perhaps the */
/*        most likely. We have to assume the last alternative. */
/*        If this is true, then we need to reduce EMAX by one because */
/*        there must be some way of representing zero in an implicit-bit */
/*        system. On machines like Cray, we are reducing EMAX by one */
/*        unnecessarily. */

	--(*emax);
    }

    if (*ieee) {

/*        Assume we are on an IEEE machine which reserves one exponent */
/*        for infinity and NaN. */

	--(*emax);
    }

/*     Now create RMAX, the largest machine number, which should */
/*     be equal to (1.0 - BETA**(-P)) * BETA**EMAX . */

/*     First compute 1.0 - BETA**(-P), being careful that the */
/*     result is less than 1.0 . */

    recbas = 1.f / *beta;
    z__ = *beta - 1.f;
    y = 0.f;
    i__1 = *p;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__ *= recbas;
	if (y < 1.f) {
	    oldy = y;
	}
	y = slamc3_(&y, &z__);
/* L20: */
    }
    if (y >= 1.f) {
	y = oldy;
    }

/*     Now multiply by BETA**EMAX to get RMAX. */

    i__1 = *emax;
    for (i__ = 1; i__ <= i__1; ++i__) {
	r__1 = y * *beta;
	y = slamc3_(&r__1, &c_b32);
/* L30: */
    }

    *rmax = y;
    return 0;

/*     End of SLAMC5 */

} /* slamc5_ */
EOF

	cat << EOF > ${LAPACKDIR}/dlamch.c
/*  -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Table of constant values */

static integer c__1 = 1;
static doublereal c_b32 = 0.;

doublereal dlamch_(char *cmach)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    integer i__1;
    doublereal ret_val;

    /* Local variables */
    static doublereal t;
    static integer it;
    static doublereal rnd, eps, base;
    static integer beta;
    static doublereal emin, prec, emax;
    static integer imin, imax;
    static logical lrnd;
    static doublereal rmin, rmax, rmach;
    extern logical lsame_(char *, char *);
    static doublereal small, sfmin;
    extern /* Subroutine */ int dlamc2_(integer *, integer *, logical *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *);


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMCH determines double precision machine parameters. */

/*  Arguments */
/*  ========= */

/*  CMACH   (input) CHARACTER*1 */
/*          Specifies the value to be returned by DLAMCH: */
/*          = 'E' or 'e',   DLAMCH := eps */
/*          = 'S' or 's ,   DLAMCH := sfmin */
/*          = 'B' or 'b',   DLAMCH := base */
/*          = 'P' or 'p',   DLAMCH := eps*base */
/*          = 'N' or 'n',   DLAMCH := t */
/*          = 'R' or 'r',   DLAMCH := rnd */
/*          = 'M' or 'm',   DLAMCH := emin */
/*          = 'U' or 'u',   DLAMCH := rmin */
/*          = 'L' or 'l',   DLAMCH := emax */
/*          = 'O' or 'o',   DLAMCH := rmax */

/*          where */

/*          eps   = relative machine precision */
/*          sfmin = safe minimum, such that 1/sfmin does not overflow */
/*          base  = base of the machine */
/*          prec  = eps*base */
/*          t     = number of (base) digits in the mantissa */
/*          rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise */
/*          emin  = minimum exponent before (gradual) underflow */
/*          rmin  = underflow threshold - base**(emin-1) */
/*          emax  = largest exponent before overflow */
/*          rmax  = overflow threshold  - (base**emax)*(1-eps) */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	dlamc2_(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
	base = (doublereal) beta;
	t = (doublereal) it;
	if (lrnd) {
	    rnd = 1.;
	    i__1 = 1 - it;
	    eps = pow_di(&base, &i__1) / 2;
	} else {
	    rnd = 0.;
	    i__1 = 1 - it;
	    eps = pow_di(&base, &i__1);
	}
	prec = eps * base;
	emin = (doublereal) imin;
	emax = (doublereal) imax;
	sfmin = rmin;
	small = 1. / rmax;
	if (small >= sfmin) {

/*           Use SMALL plus a bit, to avoid the possibility of rounding */
/*           causing overflow when computing  1/sfmin. */

	    sfmin = small * (eps + 1.);
	}
    }

    if (lsame_(cmach, "E")) {
	rmach = eps;
    } else if (lsame_(cmach, "S")) {
	rmach = sfmin;
    } else if (lsame_(cmach, "B")) {
	rmach = base;
    } else if (lsame_(cmach, "P")) {
	rmach = prec;
    } else if (lsame_(cmach, "N")) {
	rmach = t;
    } else if (lsame_(cmach, "R")) {
	rmach = rnd;
    } else if (lsame_(cmach, "M")) {
	rmach = emin;
    } else if (lsame_(cmach, "U")) {
	rmach = rmin;
    } else if (lsame_(cmach, "L")) {
	rmach = emax;
    } else if (lsame_(cmach, "O")) {
	rmach = rmax;
    }

    ret_val = rmach;
    first = FALSE_;
    return ret_val;

/*     End of DLAMCH */

} /* dlamch_ */


/* *********************************************************************** */

/* Subroutine */ int dlamc1_(integer *beta, integer *t, logical *rnd, logical 
	*ieee1)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    doublereal d__1, d__2;

    /* Local variables */
    static doublereal a, b, c__, f, t1, t2;
    static integer lt;
    static doublereal one, qtr;
    static logical lrnd;
    static integer lbeta;
    static doublereal savec;
    extern doublereal dlamc3_(doublereal *, doublereal *);
    static logical lieee1;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC1 determines the machine parameters given by BETA, T, RND, and */
/*  IEEE1. */

/*  Arguments */
/*  ========= */

/*  BETA    (output) INTEGER */
/*          The base of the machine. */

/*  T       (output) INTEGER */
/*          The number of ( BETA ) digits in the mantissa. */

/*  RND     (output) LOGICAL */
/*          Specifies whether proper rounding  ( RND = .TRUE. )  or */
/*          chopping  ( RND = .FALSE. )  occurs in addition. This may not */
/*          be a reliable guide to the way in which the machine performs */
/*          its arithmetic. */

/*  IEEE1   (output) LOGICAL */
/*          Specifies whether rounding appears to be done in the IEEE */
/*          'round to nearest' style. */

/*  Further Details */
/*  =============== */

/*  The routine is based on the routine  ENVRON  by Malcolm and */
/*  incorporates suggestions by Gentleman and Marovich. See */

/*     Malcolm M. A. (1972) Algorithms to reveal properties of */
/*        floating-point arithmetic. Comms. of the ACM, 15, 949-951. */

/*     Gentleman W. M. and Marovich S. B. (1974) More on algorithms */
/*        that reveal properties of floating point arithmetic units. */
/*        Comms. of the ACM, 17, 276-277. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	one = 1.;

/*        LBETA,  LIEEE1,  LT and  LRND  are the  local values  of  BETA, */
/*        IEEE1, T and RND. */

/*        Throughout this routine  we use the function  DLAMC3  to ensure */
/*        that relevant values are  stored and not held in registers,  or */
/*        are not affected by optimizers. */

/*        Compute  a = 2.0**m  with the  smallest positive integer m such */
/*        that */

/*           fl( a + 1.0 ) = a. */

	a = 1.;
	c__ = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L10:
	if (c__ == one) {
	    a *= 2;
	    c__ = dlamc3_(&a, &one);
	    d__1 = -a;
	    c__ = dlamc3_(&c__, &d__1);
	    goto L10;
	}
/* +       END WHILE */

/*        Now compute  b = 2.0**m  with the smallest positive integer m */
/*        such that */

/*           fl( a + b ) .gt. a. */

	b = 1.;
	c__ = dlamc3_(&a, &b);

/* +       WHILE( C.EQ.A )LOOP */
L20:
	if (c__ == a) {
	    b *= 2;
	    c__ = dlamc3_(&a, &b);
	    goto L20;
	}
/* +       END WHILE */

/*        Now compute the base.  a and c  are neighbouring floating point */
/*        numbers  in the  interval  ( beta**t, beta**( t + 1 ) )  and so */
/*        their difference is beta. Adding 0.25 to c is to ensure that it */
/*        is truncated to beta and not ( beta - 1 ). */

	qtr = one / 4;
	savec = c__;
	d__1 = -a;
	c__ = dlamc3_(&c__, &d__1);
	lbeta = (integer) (c__ + qtr);

/*        Now determine whether rounding or chopping occurs,  by adding a */
/*        bit  less  than  beta/2  and a  bit  more  than  beta/2  to  a. */

	b = (doublereal) lbeta;
	d__1 = b / 2;
	d__2 = -b / 100;
	f = dlamc3_(&d__1, &d__2);
	c__ = dlamc3_(&f, &a);
	if (c__ == a) {
	    lrnd = TRUE_;
	} else {
	    lrnd = FALSE_;
	}
	d__1 = b / 2;
	d__2 = b / 100;
	f = dlamc3_(&d__1, &d__2);
	c__ = dlamc3_(&f, &a);
	if (lrnd && c__ == a) {
	    lrnd = FALSE_;
	}

/*        Try and decide whether rounding is done in the  IEEE  'round to */
/*        nearest' style. B/2 is half a unit in the last place of the two */
/*        numbers A and SAVEC. Furthermore, A is even, i.e. has last  bit */
/*        zero, and SAVEC is odd. Thus adding B/2 to A should not  change */
/*        A, but adding B/2 to SAVEC should change SAVEC. */

	d__1 = b / 2;
	t1 = dlamc3_(&d__1, &a);
	d__1 = b / 2;
	t2 = dlamc3_(&d__1, &savec);
	lieee1 = t1 == a && t2 > savec && lrnd;

/*        Now find  the  mantissa, t.  It should  be the  integer part of */
/*        log to the base beta of a,  however it is safer to determine  t */
/*        by powering.  So we find t as the smallest positive integer for */
/*        which */

/*           fl( beta**t + 1.0 ) = 1.0. */

	lt = 0;
	a = 1.;
	c__ = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L30:
	if (c__ == one) {
	    ++lt;
	    a *= lbeta;
	    c__ = dlamc3_(&a, &one);
	    d__1 = -a;
	    c__ = dlamc3_(&c__, &d__1);
	    goto L30;
	}
/* +       END WHILE */

    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *ieee1 = lieee1;
    first = FALSE_;
    return 0;

/*     End of DLAMC1 */

} /* dlamc1_ */


/* *********************************************************************** */

/* Subroutine */ int dlamc2_(integer *beta, integer *t, logical *rnd, 
	doublereal *eps, integer *emin, doublereal *rmin, integer *emax, 
	doublereal *rmax)
{
    /* Initialized data */

    static logical first = TRUE_;
    static logical iwarn = FALSE_;

    /* Format strings */
    static char fmt_9999[] = "(//\002 WARNING. The value EMIN may be incorre"
	    "ct:-\002,\002  EMIN = \002,i8,/\002 If, after inspection, the va"
	    "lue EMIN looks\002,\002 acceptable please comment out \002,/\002"
	    " the IF block as marked within the code of routine\002,\002 DLAM"
	    "C2,\002,/\002 otherwise supply EMIN explicitly.\002,/)";

    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2, d__3, d__4, d__5;

    /* Local variables */
    static doublereal a, b, c__;
    static integer i__, lt;
    static doublereal one, two;
    static logical ieee;
    static doublereal half;
    static logical lrnd;
    static doublereal leps, zero;
    static integer lbeta;
    static doublereal rbase;
    static integer lemin, lemax, gnmin;
    static doublereal small;
    static integer gpmin;
    static doublereal third, lrmin, lrmax, sixth;
    extern /* Subroutine */ int dlamc1_(integer *, integer *, logical *, 
	    logical *);
    extern doublereal dlamc3_(doublereal *, doublereal *);
    static logical lieee1;
    extern /* Subroutine */ int dlamc4_(integer *, doublereal *, integer *), 
	    dlamc5_(integer *, integer *, integer *, logical *, integer *, 
	    doublereal *);
    static integer ngnmin, ngpmin;

    /* Fortran I/O blocks */
    static cilist io___58 = { 0, 6, 0, fmt_9999, 0 };



/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC2 determines the machine parameters specified in its argument */
/*  list. */

/*  Arguments */
/*  ========= */

/*  BETA    (output) INTEGER */
/*          The base of the machine. */

/*  T       (output) INTEGER */
/*          The number of ( BETA ) digits in the mantissa. */

/*  RND     (output) LOGICAL */
/*          Specifies whether proper rounding  ( RND = .TRUE. )  or */
/*          chopping  ( RND = .FALSE. )  occurs in addition. This may not */
/*          be a reliable guide to the way in which the machine performs */
/*          its arithmetic. */

/*  EPS     (output) DOUBLE PRECISION */
/*          The smallest positive number such that */

/*             fl( 1.0 - EPS ) .LT. 1.0, */

/*          where fl denotes the computed value. */

/*  EMIN    (output) INTEGER */
/*          The minimum exponent before (gradual) underflow occurs. */

/*  RMIN    (output) DOUBLE PRECISION */
/*          The smallest normalized number for the machine, given by */
/*          BASE**( EMIN - 1 ), where  BASE  is the floating point value */
/*          of BETA. */

/*  EMAX    (output) INTEGER */
/*          The maximum exponent before overflow occurs. */

/*  RMAX    (output) DOUBLE PRECISION */
/*          The largest positive number for the machine, given by */
/*          BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating point */
/*          value of BETA. */

/*  Further Details */
/*  =============== */

/*  The computation of  EPS  is based on a routine PARANOIA by */
/*  W. Kahan of the University of California at Berkeley. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	zero = 0.;
	one = 1.;
	two = 2.;

/*        LBETA, LT, LRND, LEPS, LEMIN and LRMIN  are the local values of */
/*        BETA, T, RND, EPS, EMIN and RMIN. */

/*        Throughout this routine  we use the function  DLAMC3  to ensure */
/*        that relevant values are stored  and not held in registers,  or */
/*        are not affected by optimizers. */

/*        DLAMC1 returns the parameters  LBETA, LT, LRND and LIEEE1. */

	dlamc1_(&lbeta, &lt, &lrnd, &lieee1);

/*        Start to find EPS. */

	b = (doublereal) lbeta;
	i__1 = -lt;
	a = pow_di(&b, &i__1);
	leps = a;

/*        Try some tricks to see whether or not this is the correct  EPS. */

	b = two / 3;
	half = one / 2;
	d__1 = -half;
	sixth = dlamc3_(&b, &d__1);
	third = dlamc3_(&sixth, &sixth);
	d__1 = -half;
	b = dlamc3_(&third, &d__1);
	b = dlamc3_(&b, &sixth);
	b = abs(b);
	if (b < leps) {
	    b = leps;
	}

	leps = 1.;

/* +       WHILE( ( LEPS.GT.B ).AND.( B.GT.ZERO ) )LOOP */
L10:
	if (leps > b && b > zero) {
	    leps = b;
	    d__1 = half * leps;
/* Computing 5th power */
	    d__3 = two, d__4 = d__3, d__3 *= d__3;
/* Computing 2nd power */
	    d__5 = leps;
	    d__2 = d__4 * (d__3 * d__3) * (d__5 * d__5);
	    c__ = dlamc3_(&d__1, &d__2);
	    d__1 = -c__;
	    c__ = dlamc3_(&half, &d__1);
	    b = dlamc3_(&half, &c__);
	    d__1 = -b;
	    c__ = dlamc3_(&half, &d__1);
	    b = dlamc3_(&half, &c__);
	    goto L10;
	}
/* +       END WHILE */

	if (a < leps) {
	    leps = a;
	}

/*        Computation of EPS complete. */

/*        Now find  EMIN.  Let A = + or - 1, and + or - (1 + BASE**(-3)). */
/*        Keep dividing  A by BETA until (gradual) underflow occurs. This */
/*        is detected when we cannot recover the previous A. */

	rbase = one / lbeta;
	small = one;
	for (i__ = 1; i__ <= 3; ++i__) {
	    d__1 = small * rbase;
	    small = dlamc3_(&d__1, &zero);
/* L20: */
	}
	a = dlamc3_(&one, &small);
	dlamc4_(&ngpmin, &one, &lbeta);
	d__1 = -one;
	dlamc4_(&ngnmin, &d__1, &lbeta);
	dlamc4_(&gpmin, &a, &lbeta);
	d__1 = -a;
	dlamc4_(&gnmin, &d__1, &lbeta);
	ieee = FALSE_;

	if (ngpmin == ngnmin && gpmin == gnmin) {
	    if (ngpmin == gpmin) {
		lemin = ngpmin;
/*            ( Non twos-complement machines, no gradual underflow; */
/*              e.g.,  VAX ) */
	    } else if (gpmin - ngpmin == 3) {
		lemin = ngpmin - 1 + lt;
		ieee = TRUE_;
/*            ( Non twos-complement machines, with gradual underflow; */
/*              e.g., IEEE standard followers ) */
	    } else {
		lemin = f2cmin(ngpmin,gpmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if (ngpmin == gpmin && ngnmin == gnmin) {
	    if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1) {
		lemin = f2cmax(ngpmin,ngnmin);
/*            ( Twos-complement machines, no gradual underflow; */
/*              e.g., CYBER 205 ) */
	    } else {
		lemin = f2cmin(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1 && gpmin == gnmin)
		 {
	    if (gpmin - f2cmin(ngpmin,ngnmin) == 3) {
		lemin = f2cmax(ngpmin,ngnmin) - 1 + lt;
/*            ( Twos-complement machines with gradual underflow; */
/*              no known machine ) */
	    } else {
		lemin = f2cmin(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else {
/* Computing MIN */
	    i__1 = f2cmin(ngpmin,ngnmin), i__1 = f2cmin(i__1,gpmin);
	    lemin = f2cmin(i__1,gnmin);
/*         ( A guess; no known machine ) */
	    iwarn = TRUE_;
	}
	first = FALSE_;
/* ** */
/* Comment out this if block if EMIN is ok */
	if (iwarn) {
	    first = TRUE_;
	    printf("\n\n WARNING. The value EMIN may be incorrect:- ");
	    printf("EMIN = %8i\n",lemin);
	    printf("If, after inspection, the value EMIN looks acceptable");
            printf("please comment out \n the IF block as marked within the"); 
            printf("code of routine DLAMC2, \n otherwise supply EMIN"); 
            printf("explicitly.\n");
	}
/* **   

          Assume IEEE arithmetic if we found denormalised  numbers abo
ve,   
          or if arithmetic seems to round in the  IEEE style,  determi
ned   
          in routine DLAMC1. A true IEEE machine should have both  thi
ngs   
          true; however, faulty machines may have one or the other. */

	ieee = ieee || lieee1;

/*        Compute  RMIN by successive division by  BETA. We could compute */
/*        RMIN as BASE**( EMIN - 1 ),  but some machines underflow during */
/*        this computation. */

	lrmin = 1.;
	i__1 = 1 - lemin;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__1 = lrmin * rbase;
	    lrmin = dlamc3_(&d__1, &zero);
/* L30: */
	}

/*        Finally, call DLAMC5 to compute EMAX and RMAX. */

	dlamc5_(&lbeta, &lt, &lemin, &ieee, &lemax, &lrmax);
    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *eps = leps;
    *emin = lemin;
    *rmin = lrmin;
    *emax = lemax;
    *rmax = lrmax;

    return 0;


/*     End of DLAMC2 */

} /* dlamc2_ */


/* *********************************************************************** */

doublereal dlamc3_(doublereal *a, doublereal *b)
{
    /* System generated locals */
    doublereal ret_val;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC3  is intended to force  A  and  B  to be stored prior to doing */
/*  the addition of  A  and  B ,  for use in situations where optimizers */
/*  might hold one of these in a register. */

/*  Arguments */
/*  ========= */

/*  A       (input) DOUBLE PRECISION */
/*  B       (input) DOUBLE PRECISION */
/*          The values A and B. */

/* ===================================================================== */

/*     .. Executable Statements .. */

    ret_val = *a + *b;

    return ret_val;

/*     End of DLAMC3 */

} /* dlamc3_ */


/* *********************************************************************** */

/* Subroutine */ int dlamc4_(integer *emin, doublereal *start, integer *base)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Local variables */
    static doublereal a;
    static integer i__;
    static doublereal b1, b2, c1, c2, d1, d2, one, zero, rbase;
    extern doublereal dlamc3_(doublereal *, doublereal *);


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC4 is a service routine for DLAMC2. */

/*  Arguments */
/*  ========= */

/*  EMIN    (output) INTEGER */
/*          The minimum exponent before (gradual) underflow, computed by */
/*          setting A = START and dividing by BASE until the previous A */
/*          can not be recovered. */

/*  START   (input) DOUBLE PRECISION */
/*          The starting point for determining EMIN. */

/*  BASE    (input) INTEGER */
/*          The base of the machine. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    a = *start;
    one = 1.;
    rbase = one / *base;
    zero = 0.;
    *emin = 1;
    d__1 = a * rbase;
    b1 = dlamc3_(&d__1, &zero);
    c1 = a;
    c2 = a;
    d1 = a;
    d2 = a;
/* +    WHILE( ( C1.EQ.A ).AND.( C2.EQ.A ).AND. */
/*    $       ( D1.EQ.A ).AND.( D2.EQ.A )      )LOOP */
L10:
    if (c1 == a && c2 == a && d1 == a && d2 == a) {
	--(*emin);
	a = b1;
	d__1 = a / *base;
	b1 = dlamc3_(&d__1, &zero);
	d__1 = b1 * *base;
	c1 = dlamc3_(&d__1, &zero);
	d1 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d1 += b1;
/* L20: */
	}
	d__1 = a * rbase;
	b2 = dlamc3_(&d__1, &zero);
	d__1 = b2 / rbase;
	c2 = dlamc3_(&d__1, &zero);
	d2 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d2 += b2;
/* L30: */
	}
	goto L10;
    }
/* +    END WHILE */

    return 0;

/*     End of DLAMC4 */

} /* dlamc4_ */


/* *********************************************************************** */

/* Subroutine */ int dlamc5_(integer *beta, integer *p, integer *emin, 
	logical *ieee, integer *emax, doublereal *rmax)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Local variables */
    static integer i__;
    static doublereal y, z__;
    static integer try__, lexp;
    static doublereal oldy;
    static integer uexp, nbits;
    extern doublereal dlamc3_(doublereal *, doublereal *);
    static doublereal recbas;
    static integer exbits, expsum;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC5 attempts to compute RMAX, the largest machine floating-point */
/*  number, without overflow.  It assumes that EMAX + abs(EMIN) sum */
/*  approximately to a power of 2.  It will fail on machines where this */
/*  assumption does not hold, for example, the Cyber 205 (EMIN = -28625, */
/*  EMAX = 28718).  It will also fail if the value supplied for EMIN is */
/*  too large (i.e. too close to zero), probably with overflow. */

/*  Arguments */
/*  ========= */

/*  BETA    (input) INTEGER */
/*          The base of floating-point arithmetic. */

/*  P       (input) INTEGER */
/*          The number of base BETA digits in the mantissa of a */
/*          floating-point value. */

/*  EMIN    (input) INTEGER */
/*          The minimum exponent before (gradual) underflow. */

/*  IEEE    (input) LOGICAL */
/*          A logical flag specifying whether or not the arithmetic */
/*          system is thought to comply with the IEEE standard. */

/*  EMAX    (output) INTEGER */
/*          The largest exponent before overflow */

/*  RMAX    (output) DOUBLE PRECISION */
/*          The largest machine floating-point number. */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     First compute LEXP and UEXP, two powers of 2 that bound */
/*     abs(EMIN). We then assume that EMAX + abs(EMIN) will sum */
/*     approximately to the bound that is closest to abs(EMIN). */
/*     (EMAX is the exponent of the required number RMAX). */

    lexp = 1;
    exbits = 1;
L10:
    try__ = lexp << 1;
    if (try__ <= -(*emin)) {
	lexp = try__;
	++exbits;
	goto L10;
    }
    if (lexp == -(*emin)) {
	uexp = lexp;
    } else {
	uexp = try__;
	++exbits;
    }

/*     Now -LEXP is less than or equal to EMIN, and -UEXP is greater */
/*     than or equal to EMIN. EXBITS is the number of bits needed to */
/*     store the exponent. */

    if (uexp + *emin > -lexp - *emin) {
	expsum = lexp << 1;
    } else {
	expsum = uexp << 1;
    }

/*     EXPSUM is the exponent range, approximately equal to */
/*     EMAX - EMIN + 1 . */

    *emax = expsum + *emin - 1;
    nbits = exbits + 1 + *p;

/*     NBITS is the total number of bits needed to store a */
/*     floating-point number. */

    if (nbits % 2 == 1 && *beta == 2) {

/*        Either there are an odd number of bits used to store a */
/*        floating-point number, which is unlikely, or some bits are */
/*        not used in the representation of numbers, which is possible, */
/*        (e.g. Cray machines) or the mantissa has an implicit bit, */
/*        (e.g. IEEE machines, Dec Vax machines), which is perhaps the */
/*        most likely. We have to assume the last alternative. */
/*        If this is true, then we need to reduce EMAX by one because */
/*        there must be some way of representing zero in an implicit-bit */
/*        system. On machines like Cray, we are reducing EMAX by one */
/*        unnecessarily. */

	--(*emax);
    }

    if (*ieee) {

/*        Assume we are on an IEEE machine which reserves one exponent */
/*        for infinity and NaN. */

	--(*emax);
    }

/*     Now create RMAX, the largest machine number, which should */
/*     be equal to (1.0 - BETA**(-P)) * BETA**EMAX . */

/*     First compute 1.0 - BETA**(-P), being careful that the */
/*     result is less than 1.0 . */

    recbas = 1. / *beta;
    z__ = *beta - 1.;
    y = 0.;
    i__1 = *p;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__ *= recbas;
	if (y < 1.) {
	    oldy = y;
	}
	y = dlamc3_(&y, &z__);
/* L20: */
    }
    if (y >= 1.) {
	y = oldy;
    }

/*     Now multiply by BETA**EMAX to get RMAX. */

    i__1 = *emax;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__1 = y * *beta;
	y = dlamc3_(&d__1, &c_b32);
/* L30: */
    }

    *rmax = y;
    return 0;

/*     End of DLAMC5 */

} /* dlamc5_ */

EOF
	cat << EOF > ${LAPACKDIR}/qlamch.c
/*  -- translated by f2c (version 20090411).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#define __LAPACK_PRECISION_QUAD
#include "f2c.h"

/* Table of constant values */

static integer c__1 = 1;
static quadreal c_b32 = 0.q;

quadreal qlamch_(char *cmach)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    integer i__1;
    quadreal ret_val;

    /* Local variables */
    static quadreal t;
    static integer it;
    static quadreal rnd, eps, base;
    static integer beta;
    static quadreal emin, prec, emax;
    static integer imin, imax;
    static logical lrnd;
    static quadreal rmin, rmax, rmach;
    extern logical lsame_(char *, char *);
    static quadreal small, sfmin;
    extern /* Subroutine */ int qlamc2_(integer *, integer *, logical *, 
	    quadreal *, integer *, quadreal *, integer *, quadreal *);


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMCH determines __float128 precision machine parameters. */

/*  Arguments */
/*  ========= */

/*  CMACH   (input) CHARACTER*1 */
/*          Specifies the value to be returned by DLAMCH: */
/*          = 'E' or 'e',   DLAMCH := eps */
/*          = 'S' or 's ,   DLAMCH := sfmin */
/*          = 'B' or 'b',   DLAMCH := base */
/*          = 'P' or 'p',   DLAMCH := eps*base */
/*          = 'N' or 'n',   DLAMCH := t */
/*          = 'R' or 'r',   DLAMCH := rnd */
/*          = 'M' or 'm',   DLAMCH := emin */
/*          = 'U' or 'u',   DLAMCH := rmin */
/*          = 'L' or 'l',   DLAMCH := emax */
/*          = 'O' or 'o',   DLAMCH := rmax */

/*          where */

/*          eps   = relative machine precision */
/*          sfmin = safe minimum, such that 1/sfmin does not overflow */
/*          base  = base of the machine */
/*          prec  = eps*base */
/*          t     = number of (base) digits in the mantissa */
/*          rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise */
/*          emin  = minimum exponent before (gradual) underflow */
/*          rmin  = underflow threshold - base**(emin-1) */
/*          emax  = largest exponent before overflow */
/*          rmax  = overflow threshold  - (base**emax)*(1-eps) */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	qlamc2_(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
	base = (quadreal) beta;
	t = (quadreal) it;
	if (lrnd) {
	    rnd = 1.q;
	    i__1 = 1 - it;
	    eps = pow_di(&base, &i__1) / 2;
	} else {
	    rnd = 0.q;
	    i__1 = 1 - it;
	    eps = pow_di(&base, &i__1);
	}
	prec = eps * base;
	emin = (quadreal) imin;
	emax = (quadreal) imax;
	sfmin = rmin;
	small = 1.q / rmax;
	if (small >= sfmin) {

/*           Use SMALL plus a bit, to avoid the possibility of rounding */
/*           causing overflow when computing  1/sfmin. */

	    sfmin = small * (eps + 1.q);
	}
    }

    if (lsame_(cmach, "E")) {
	rmach = eps;
    } else if (lsame_(cmach, "S")) {
	rmach = sfmin;
    } else if (lsame_(cmach, "B")) {
	rmach = base;
    } else if (lsame_(cmach, "P")) {
	rmach = prec;
    } else if (lsame_(cmach, "N")) {
	rmach = t;
    } else if (lsame_(cmach, "R")) {
	rmach = rnd;
    } else if (lsame_(cmach, "M")) {
	rmach = emin;
    } else if (lsame_(cmach, "U")) {
	rmach = rmin;
    } else if (lsame_(cmach, "L")) {
	rmach = emax;
    } else if (lsame_(cmach, "O")) {
	rmach = rmax;
    }

    ret_val = rmach;
    first = FALSE_;
    return ret_val;

/*     End of DLAMCH */

} /* qlamch_ */


/* *********************************************************************** */

/* Subroutine */ int qlamc1_(integer *beta, integer *t, logical *rnd, logical 
	*ieee1)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    quadreal d__1, d__2;

    /* Local variables */
    static quadreal a, b, c__, f, t1, t2;
    static integer lt;
    static quadreal one, qtr;
    static logical lrnd;
    static integer lbeta;
    static quadreal savec;
    extern quadreal qlamc3_(quadreal *, quadreal *);
    static logical lieee1;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC1 determines the machine parameters given by BETA, T, RND, and */
/*  IEEE1. */

/*  Arguments */
/*  ========= */

/*  BETA    (output) INTEGER */
/*          The base of the machine. */

/*  T       (output) INTEGER */
/*          The number of ( BETA ) digits in the mantissa. */

/*  RND     (output) LOGICAL */
/*          Specifies whether proper rounding  ( RND = .TRUE. )  or */
/*          chopping  ( RND = .FALSE. )  occurs in addition. This may not */
/*          be a reliable guide to the way in which the machine performs */
/*          its arithmetic. */

/*  IEEE1   (output) LOGICAL */
/*          Specifies whether rounding appears to be done in the IEEE */
/*          'round to nearest' style. */

/*  Further Details */
/*  =============== */

/*  The routine is based on the routine  ENVRON  by Malcolm and */
/*  incorporates suggestions by Gentleman and Marovich. See */

/*     Malcolm M. A. (1972) Algorithms to reveal properties of */
/*        floating-point arithmetic. Comms. of the ACM, 15, 949-951. */

/*     Gentleman W. M. and Marovich S. B. (1974) More on algorithms */
/*        that reveal properties of floating point arithmetic units. */
/*        Comms. of the ACM, 17, 276-277. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	one = 1.q;

/*        LBETA,  LIEEE1,  LT and  LRND  are the  local values  of  BETA, */
/*        IEEE1, T and RND. */

/*        Throughout this routine  we use the function  DLAMC3  to ensure */
/*        that relevant values are  stored and not held in registers,  or */
/*        are not affected by optimizers. */

/*        Compute  a = 2.0**m  with the  smallest positive integer m such */
/*        that */

/*           fl( a + 1.0 ) = a. */

	a = 1.;
	c__ = 1.q;

/* +       WHILE( C.EQ.ONE )LOOP */
L10:
	if (c__ == one) {
	    a *= 2;
	    c__ = qlamc3_(&a, &one);
	    d__1 = -a;
	    c__ = qlamc3_(&c__, &d__1);
	    goto L10;
	}
/* +       END WHILE */

/*        Now compute  b = 2.0**m  with the smallest positive integer m */
/*        such that */

/*           fl( a + b ) .gt. a. */

	b = 1.q;
	c__ = qlamc3_(&a, &b);

/* +       WHILE( C.EQ.A )LOOP */
L20:
	if (c__ == a) {
	    b *= 2;
	    c__ = qlamc3_(&a, &b);
	    goto L20;
	}
/* +       END WHILE */

/*        Now compute the base.  a and c  are neighbouring floating point */
/*        numbers  in the  interval  ( beta**t, beta**( t + 1 ) )  and so */
/*        their difference is beta. Adding 0.25 to c is to ensure that it */
/*        is truncated to beta and not ( beta - 1 ). */

	qtr = one / 4;
	savec = c__;
	d__1 = -a;
	c__ = qlamc3_(&c__, &d__1);
	lbeta = (integer) (c__ + qtr);

/*        Now determine whether rounding or chopping occurs,  by adding a */
/*        bit  less  than  beta/2  and a  bit  more  than  beta/2  to  a. */

	b = (quadreal) lbeta;
	d__1 = b / 2;
	d__2 = -b / 100;
	f = qlamc3_(&d__1, &d__2);
	c__ = qlamc3_(&f, &a);
	if (c__ == a) {
	    lrnd = TRUE_;
	} else {
	    lrnd = FALSE_;
	}
	d__1 = b / 2;
	d__2 = b / 100;
	f = qlamc3_(&d__1, &d__2);
	c__ = qlamc3_(&f, &a);
	if (lrnd && c__ == a) {
	    lrnd = FALSE_;
	}

/*        Try and decide whether rounding is done in the  IEEE  'round to */
/*        nearest' style. B/2 is half a unit in the last place of the two */
/*        numbers A and SAVEC. Furthermore, A is even, i.e. has last  bit */
/*        zero, and SAVEC is odd. Thus adding B/2 to A should not  change */
/*        A, but adding B/2 to SAVEC should change SAVEC. */

	d__1 = b / 2;
	t1 = qlamc3_(&d__1, &a);
	d__1 = b / 2;
	t2 = qlamc3_(&d__1, &savec);
	lieee1 = t1 == a && t2 > savec && lrnd;

/*        Now find  the  mantissa, t.  It should  be the  integer part of */
/*        logq to the base beta of a,  however it is safer to determine  t */
/*        by powering.  So we find t as the smallest positive integer for */
/*        which */

/*           fl( beta**t + 1.0 ) = 1.0. */

	lt = 0;
	a = 1.q;
	c__ = 1.q;

/* +       WHILE( C.EQ.ONE )LOOP */
L30:
	if (c__ == one) {
	    ++lt;
	    a *= lbeta;
	    c__ = qlamc3_(&a, &one);
	    d__1 = -a;
	    c__ = qlamc3_(&c__, &d__1);
	    goto L30;
	}
/* +       END WHILE */

    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *ieee1 = lieee1;
    first = FALSE_;
    return 0;

/*     End of DLAMC1 */

} /* qlamc1_ */


/* *********************************************************************** */

/* Subroutine */ int qlamc2_(integer *beta, integer *t, logical *rnd, 
	quadreal *eps, integer *emin, quadreal *rmin, integer *emax, 
	quadreal *rmax)
{
    /* Initialized data */

    static logical first = TRUE_;
    static logical iwarn = FALSE_;

    /* Format strings */
    static char fmt_9999[] = "(//\002 WARNING. The value EMIN may be incorre"
	    "ct:-\002,\002  EMIN = \002,i8,/\002 If, after inspection, the va"
	    "lue EMIN looks\002,\002 acceptable please comment out \002,/\002"
	    " the IF block as marked within the code of routine\002,\002 DLAM"
	    "C2,\002,/\002 otherwise supply EMIN explicitly.\002,/)";

    /* System generated locals */
    integer i__1;
    quadreal d__1, d__2, d__3, d__4, d__5;

    /* Local variables */
    static quadreal a, b, c__;
    static integer i__, lt;
    static quadreal one, two;
    static logical ieee;
    static quadreal half;
    static logical lrnd;
    static quadreal leps, zero;
    static integer lbeta;
    static quadreal rbase;
    static integer lemin, lemax, gnmin;
    static quadreal small;
    static integer gpmin;
    static quadreal third, lrmin, lrmax, sixth;
    extern /* Subroutine */ int qlamc1_(integer *, integer *, logical *, 
	    logical *);
    extern quadreal qlamc3_(quadreal *, quadreal *);
    static logical lieee1;
    extern /* Subroutine */ int qlamc4_(integer *, quadreal *, integer *), 
	    qlamc5_(integer *, integer *, integer *, logical *, integer *, 
	    quadreal *);
    static integer ngnmin, ngpmin;

    /* Fortran I/O blocks */
    static cilist io___58 = { 0, 6, 0, fmt_9999, 0 };



/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC2 determines the machine parameters specified in its argument */
/*  list. */

/*  Arguments */
/*  ========= */

/*  BETA    (output) INTEGER */
/*          The base of the machine. */

/*  T       (output) INTEGER */
/*          The number of ( BETA ) digits in the mantissa. */

/*  RND     (output) LOGICAL */
/*          Specifies whether proper rounding  ( RND = .TRUE. )  or */
/*          chopping  ( RND = .FALSE. )  occurs in addition. This may not */
/*          be a reliable guide to the way in which the machine performs */
/*          its arithmetic. */

/*  EPS     (output) DOUBLE PRECISION */
/*          The smallest positive number such that */

/*             fl( 1.0 - EPS ) .LT. 1.0, */

/*          where fl denotes the computed value. */

/*  EMIN    (output) INTEGER */
/*          The minimum exponent before (gradual) underflow occurs. */

/*  RMIN    (output) DOUBLE PRECISION */
/*          The smallest normalized number for the machine, given by */
/*          BASE**( EMIN - 1 ), where  BASE  is the floating point value */
/*          of BETA. */

/*  EMAX    (output) INTEGER */
/*          The maximum exponent before overflow occurs. */

/*  RMAX    (output) DOUBLE PRECISION */
/*          The largest positive number for the machine, given by */
/*          BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating point */
/*          value of BETA. */

/*  Further Details */
/*  =============== */

/*  The computation of  EPS  is based on a routine PARANOIA by */
/*  W. Kahan of the University of California at Berkeley. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	zero = 0.q;
	one = 1.q;
	two = 2.q;

/*        LBETA, LT, LRND, LEPS, LEMIN and LRMIN  are the local values of */
/*        BETA, T, RND, EPS, EMIN and RMIN. */

/*        Throughout this routine  we use the function  DLAMC3  to ensure */
/*        that relevant values are stored  and not held in registers,  or */
/*        are not affected by optimizers. */

/*        DLAMC1 returns the parameters  LBETA, LT, LRND and LIEEE1. */

	qlamc1_(&lbeta, &lt, &lrnd, &lieee1);

/*        Start to find EPS. */

	b = (quadreal) lbeta;
	i__1 = -lt;
	a = pow_di(&b, &i__1);
	leps = a;

/*        Try some tricks to see whether or not this is the correct  EPS. */

	b = two / 3;
	half = one / 2;
	d__1 = -half;
	sixth = qlamc3_(&b, &d__1);
	third = qlamc3_(&sixth, &sixth);
	d__1 = -half;
	b = qlamc3_(&third, &d__1);
	b = qlamc3_(&b, &sixth);
	b = abs(b);
	if (b < leps) {
	    b = leps;
	}

	leps = 1.q;

/* +       WHILE( ( LEPS.GT.B ).AND.( B.GT.ZERO ) )LOOP */
L10:
	if (leps > b && b > zero) {
	    leps = b;
	    d__1 = half * leps;
/* Computing 5th power */
	    d__3 = two, d__4 = d__3, d__3 *= d__3;
/* Computing 2nd power */
	    d__5 = leps;
	    d__2 = d__4 * (d__3 * d__3) * (d__5 * d__5);
	    c__ = qlamc3_(&d__1, &d__2);
	    d__1 = -c__;
	    c__ = qlamc3_(&half, &d__1);
	    b = qlamc3_(&half, &c__);
	    d__1 = -b;
	    c__ = qlamc3_(&half, &d__1);
	    b = qlamc3_(&half, &c__);
	    goto L10;
	}
/* +       END WHILE */

	if (a < leps) {
	    leps = a;
	}

/*        Computation of EPS complete. */

/*        Now find  EMIN.  Let A = + or - 1, and + or - (1 + BASE**(-3)). */
/*        Keep dividing  A by BETA until (gradual) underflow occurs. This */
/*        is detected when we cannot recover the previous A. */

	rbase = one / lbeta;
	small = one;
	for (i__ = 1; i__ <= 3; ++i__) {
	    d__1 = small * rbase;
	    small = qlamc3_(&d__1, &zero);
/* L20: */
	}
	a = qlamc3_(&one, &small);
	qlamc4_(&ngpmin, &one, &lbeta);
	d__1 = -one;
	qlamc4_(&ngnmin, &d__1, &lbeta);
	qlamc4_(&gpmin, &a, &lbeta);
	d__1 = -a;
	qlamc4_(&gnmin, &d__1, &lbeta);
	ieee = FALSE_;

	if (ngpmin == ngnmin && gpmin == gnmin) {
	    if (ngpmin == gpmin) {
		lemin = ngpmin;
/*            ( Non twos-complement machines, no gradual underflow; */
/*              e.g.,  VAX ) */
	    } else if (gpmin - ngpmin == 3) {
		lemin = ngpmin - 1 + lt;
		ieee = TRUE_;
/*            ( Non twos-complement machines, with gradual underflow; */
/*              e.g., IEEE standard followers ) */
	    } else {
		lemin = f2cmin(ngpmin,gpmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if (ngpmin == gpmin && ngnmin == gnmin) {
	    if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1) {
		lemin = f2cmax(ngpmin,ngnmin);
/*            ( Twos-complement machines, no gradual underflow; */
/*              e.g., CYBER 205 ) */
	    } else {
		lemin = f2cmin(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1 && gpmin == gnmin)
		 {
	    if (gpmin - f2cmin(ngpmin,ngnmin) == 3) {
		lemin = f2cmax(ngpmin,ngnmin) - 1 + lt;
/*            ( Twos-complement machines with gradual underflow; */
/*              no known machine ) */
	    } else {
		lemin = f2cmin(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else {
/* Computing MIN */
	    i__1 = f2cmin(ngpmin,ngnmin), i__1 = f2cmin(i__1,gpmin);
	    lemin = f2cmin(i__1,gnmin);
/*         ( A guess; no known machine ) */
	    iwarn = TRUE_;
	}
	first = FALSE_;
/* ** */
/* Comment out this if block if EMIN is ok */
	if (iwarn) {
	    first = TRUE_;
	    printf("\n\n WARNING. The value EMIN may be incorrect:- ");
	    printf("EMIN = %8i\n",lemin);
	    printf("If, after inspection, the value EMIN looks acceptable");
            printf("please comment out \n the IF block as marked within the"); 
            printf("code of routine DLAMC2, \n otherwise supply EMIN"); 
            printf("explicitly.\n");
	}
/* **   

          Assume IEEE arithmetic if we found denormalised  numbers abo
ve,   
          or if arithmetic seems to round in the  IEEE style,  determi
ned   
          in routine DLAMC1.q A true IEEE machine should have both  thi
ngs   
          true; however, faulty machines may have one or the other. */

	ieee = ieee || lieee1;

/*        Compute  RMIN by successive division by  BETA. We could compute */
/*        RMIN as BASE**( EMIN - 1 ),  but some machines underflow during */
/*        this computation. */

	lrmin = 1.q;
	i__1 = 1 - lemin;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__1 = lrmin * rbase;
	    lrmin = qlamc3_(&d__1, &zero);
/* L30: */
	}

/*        Finally, call DLAMC5 to compute EMAX and RMAX. */

	qlamc5_(&lbeta, &lt, &lemin, &ieee, &lemax, &lrmax);
    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *eps = leps;
    *emin = lemin;
    *rmin = lrmin;
    *emax = lemax;
    *rmax = lrmax;

    return 0;


/*     End of DLAMC2 */

} /* qlamc2_ */


/* *********************************************************************** */

quadreal qlamc3_(quadreal *a, quadreal *b)
{
    /* System generated locals */
    quadreal ret_val;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC3  is intended to force  A  and  B  to be stored prior to doing */
/*  the addition of  A  and  B ,  for use in situations where optimizers */
/*  might hold one of these in a register. */

/*  Arguments */
/*  ========= */

/*  A       (input) DOUBLE PRECISION */
/*  B       (input) DOUBLE PRECISION */
/*          The values A and B. */

/* ===================================================================== */

/*     .. Executable Statements .. */

    ret_val = *a + *b;

    return ret_val;

/*     End of DLAMC3 */

} /* qlamc3_ */


/* *********************************************************************** */

/* Subroutine */ int qlamc4_(integer *emin, quadreal *start, integer *base)
{
    /* System generated locals */
    integer i__1;
    quadreal d__1;

    /* Local variables */
    static quadreal a;
    static integer i__;
    static quadreal b1, b2, c1, c2, d1, d2, one, zero, rbase;
    extern quadreal qlamc3_(quadreal *, quadreal *);


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC4 is a service routine for DLAMC2. */

/*  Arguments */
/*  ========= */

/*  EMIN    (output) INTEGER */
/*          The minimum exponent before (gradual) underflow, computed by */
/*          setting A = START and dividing by BASE until the previous A */
/*          can not be recovered. */

/*  START   (input) DOUBLE PRECISION */
/*          The starting point for determining EMIN. */

/*  BASE    (input) INTEGER */
/*          The base of the machine. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    a = *start;
    one = 1.q;
    rbase = one / *base;
    zero = 0.q;
    *emin = 1;
    d__1 = a * rbase;
    b1 = qlamc3_(&d__1, &zero);
    c1 = a;
    c2 = a;
    d1 = a;
    d2 = a;
/* +    WHILE( ( C1.EQ.A ).AND.( C2.EQ.A ).AND. */
/*    $       ( D1.EQ.A ).AND.( D2.EQ.A )      )LOOP */
L10:
    if (c1 == a && c2 == a && d1 == a && d2 == a) {
	--(*emin);
	a = b1;
	d__1 = a / *base;
	b1 = qlamc3_(&d__1, &zero);
	d__1 = b1 * *base;
	c1 = qlamc3_(&d__1, &zero);
	d1 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d1 += b1;
/* L20: */
	}
	d__1 = a * rbase;
	b2 = qlamc3_(&d__1, &zero);
	d__1 = b2 / rbase;
	c2 = qlamc3_(&d__1, &zero);
	d2 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d2 += b2;
/* L30: */
	}
	goto L10;
    }
/* +    END WHILE */

    return 0;

/*     End of DLAMC4 */

} /* qlamc4_ */


/* *********************************************************************** */

/* Subroutine */ int qlamc5_(integer *beta, integer *p, integer *emin, 
	logical *ieee, integer *emax, quadreal *rmax)
{
    /* System generated locals */
    integer i__1;
    quadreal d__1;

    /* Local variables */
    static integer i__;
    static quadreal y, z__;
    static integer try__, lexp;
    static quadreal oldy;
    static integer uexp, nbits;
    extern quadreal qlamc3_(quadreal *, quadreal *);
    static quadreal recbas;
    static integer exbits, expsum;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC5 attempts to compute RMAX, the largest machine floating-point */
/*  number, without overflow.  It assumes that EMAX + abs(EMIN) sum */
/*  approximately to a power of 2.  It will fail on machines where this */
/*  assumption does not hold, for example, the Cyber 205 (EMIN = -28625, */
/*  EMAX = 28718).  It will also fail if the value supplied for EMIN is */
/*  too large (i.e. too close to zero), probably with overflow. */

/*  Arguments */
/*  ========= */

/*  BETA    (input) INTEGER */
/*          The base of floating-point arithmetic. */

/*  P       (input) INTEGER */
/*          The number of base BETA digits in the mantissa of a */
/*          floating-point value. */

/*  EMIN    (input) INTEGER */
/*          The minimum exponent before (gradual) underflow. */

/*  IEEE    (input) LOGICAL */
/*          A logical flag specifying whether or not the arithmetic */
/*          system is thought to comply with the IEEE standard. */

/*  EMAX    (output) INTEGER */
/*          The largest exponent before overflow */

/*  RMAX    (output) DOUBLE PRECISION */
/*          The largest machine floating-point number. */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     First compute LEXP and UEXP, two powers of 2 that bound */
/*     abs(EMIN). We then assume that EMAX + abs(EMIN) will sum */
/*     approximately to the bound that is closest to abs(EMIN). */
/*     (EMAX is the exponent of the required number RMAX). */

    lexp = 1;
    exbits = 1;
L10:
    try__ = lexp << 1;
    if (try__ <= -(*emin)) {
	lexp = try__;
	++exbits;
	goto L10;
    }
    if (lexp == -(*emin)) {
	uexp = lexp;
    } else {
	uexp = try__;
	++exbits;
    }

/*     Now -LEXP is less than or equal to EMIN, and -UEXP is greater */
/*     than or equal to EMIN. EXBITS is the number of bits needed to */
/*     store the exponent. */

    if (uexp + *emin > -lexp - *emin) {
	expsum = lexp << 1;
    } else {
	expsum = uexp << 1;
    }

/*     EXPSUM is the exponent range, approximately equal to */
/*     EMAX - EMIN + 1 . */

    *emax = expsum + *emin - 1;
    nbits = exbits + 1 + *p;

/*     NBITS is the total number of bits needed to store a */
/*     floating-point number. */

    if (nbits % 2 == 1 && *beta == 2) {

/*        Either there are an odd number of bits used to store a */
/*        floating-point number, which is unlikely, or some bits are */
/*        not used in the representation of numbers, which is possible, */
/*        (e.g. Cray machines) or the mantissa has an implicit bit, */
/*        (e.g. IEEE machines, Dec Vax machines), which is perhaps the */
/*        most likely. We have to assume the last alternative. */
/*        If this is true, then we need to reduce EMAX by one because */
/*        there must be some way of representing zero in an implicit-bit */
/*        system. On machines like Cray, we are reducing EMAX by one */
/*        unnecessarily. */

	--(*emax);
    }

    if (*ieee) {

/*        Assume we are on an IEEE machine which reserves one exponent */
/*        for infinity and NaN. */

	--(*emax);
    }

/*     Now create RMAX, the largest machine number, which should */
/*     be equal to (1.0 - BETA**(-P)) * BETA**EMAX . */

/*     First compute 1.0 - BETA**(-P), being careful that the */
/*     result is less than 1.0 . */

    recbas = 1.q / *beta;
    z__ = *beta - 1.q;
    y = 0.q;
    i__1 = *p;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__ *= recbas;
	if (y < 1.q) {
	    oldy = y;
	}
	y = qlamc3_(&y, &z__);
/* L20: */
    }
    if (y >= 1.q) {
	y = oldy;
    }

/*     Now multiply by BETA**EMAX to get RMAX. */

    i__1 = *emax;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__1 = y * *beta;
	y = qlamc3_(&d__1, &c_b32);
/* L30: */
    }

    *rmax = y;
    return 0;

/*     End of DLAMC5 */

} /* qlamc5_ */
EOF
	cat << EOF > ${LAPACKDIR}/hlamch.c
/*  -- translated by f2c (version 20090411).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#define __LAPACK_PRECISION_HALF
#include "f2c.h"

/* Table of constant values */

static integer c__1 = 1;
static halfreal c_b32 = 0.;

halfreal hlamch_(char *cmach)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    integer i__1;
    halfreal ret_val;

    /* Local variables */
    static halfreal t;
    static integer it;
    static halfreal rnd, eps, base;
    static integer beta;
    static halfreal emin, prec, emax;
    static integer imin, imax;
    static logical lrnd;
    static halfreal rmin, rmax, rmach;
    extern logical lsame_(char *, char *);
    static halfreal small, sfmin;
    extern /* Subroutine */ int hlamc2_(integer *, integer *, logical *, 
	    halfreal *, integer *, halfreal *, integer *, halfreal *);


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMCH determines __float128 precision machine parameters. */

/*  Arguments */
/*  ========= */

/*  CMACH   (input) CHARACTER*1 */
/*          Specifies the value to be returned by DLAMCH: */
/*          = 'E' or 'e',   DLAMCH := eps */
/*          = 'S' or 's ,   DLAMCH := sfmin */
/*          = 'B' or 'b',   DLAMCH := base */
/*          = 'P' or 'p',   DLAMCH := eps*base */
/*          = 'N' or 'n',   DLAMCH := t */
/*          = 'R' or 'r',   DLAMCH := rnd */
/*          = 'M' or 'm',   DLAMCH := emin */
/*          = 'U' or 'u',   DLAMCH := rmin */
/*          = 'L' or 'l',   DLAMCH := emax */
/*          = 'O' or 'o',   DLAMCH := rmax */

/*          where */

/*          eps   = relative machine precision */
/*          sfmin = safe minimum, such that 1/sfmin does not overflow */
/*          base  = base of the machine */
/*          prec  = eps*base */
/*          t     = number of (base) digits in the mantissa */
/*          rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise */
/*          emin  = minimum exponent before (gradual) underflow */
/*          rmin  = underflow threshold - base**(emin-1) */
/*          emax  = largest exponent before overflow */
/*          rmax  = overflow threshold  - (base**emax)*(1-eps) */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	hlamc2_(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
	base = (halfreal) beta;
	t = (halfreal) it;
	if (lrnd) {
	    rnd = 1.;
	    i__1 = 1 - it;
	    eps = pow_di(&base, &i__1) / 2;
	} else {
	    rnd = 0.;
	    i__1 = 1 - it;
	    eps = pow_di(&base, &i__1);
	}
	prec = eps * base;
	emin = (halfreal) imin;
	emax = (halfreal) imax;
	sfmin = rmin;
	small = 1. / rmax;
	if (small >= sfmin) {

/*           Use SMALL plus a bit, to avoid the possibility of rounding */
/*           causing overflow when computing  1/sfmin. */

	    sfmin = small * (eps + 1.);
	}
    }

    if (lsame_(cmach, "E")) {
	rmach = eps;
    } else if (lsame_(cmach, "S")) {
	rmach = sfmin;
    } else if (lsame_(cmach, "B")) {
	rmach = base;
    } else if (lsame_(cmach, "P")) {
	rmach = prec;
    } else if (lsame_(cmach, "N")) {
	rmach = t;
    } else if (lsame_(cmach, "R")) {
	rmach = rnd;
    } else if (lsame_(cmach, "M")) {
	rmach = emin;
    } else if (lsame_(cmach, "U")) {
	rmach = rmin;
    } else if (lsame_(cmach, "L")) {
	rmach = emax;
    } else if (lsame_(cmach, "O")) {
	rmach = rmax;
    }

    ret_val = rmach;
    first = FALSE_;
    return ret_val;

/*     End of DLAMCH */

} /* hlamch_ */


/* *********************************************************************** */

/* Subroutine */ int hlamc1_(integer *beta, integer *t, logical *rnd, logical 
	*ieee1)
{
    /* Initialized data */

    static logical first = TRUE_;

    /* System generated locals */
    halfreal d__1, d__2;

    /* Local variables */
    static halfreal a, b, c__, f, t1, t2;
    static integer lt;
    static halfreal one, qtr;
    static logical lrnd;
    static integer lbeta;
    static halfreal savec;
    extern halfreal hlamc3_(halfreal *, halfreal *);
    static logical lieee1;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC1 determines the machine parameters given by BETA, T, RND, and */
/*  IEEE1. */

/*  Arguments */
/*  ========= */

/*  BETA    (output) INTEGER */
/*          The base of the machine. */

/*  T       (output) INTEGER */
/*          The number of ( BETA ) digits in the mantissa. */

/*  RND     (output) LOGICAL */
/*          Specifies whether proper rounding  ( RND = .TRUE. )  or */
/*          chopping  ( RND = .FALSE. )  occurs in addition. This may not */
/*          be a reliable guide to the way in which the machine performs */
/*          its arithmetic. */

/*  IEEE1   (output) LOGICAL */
/*          Specifies whether rounding appears to be done in the IEEE */
/*          'round to nearest' style. */

/*  Further Details */
/*  =============== */

/*  The routine is based on the routine  ENVRON  by Malcolm and */
/*  incorporates suggestions by Gentleman and Marovich. See */

/*     Malcolm M. A. (1972) Algorithms to reveal properties of */
/*        floating-point arithmetic. Comms. of the ACM, 15, 949-951. */

/*     Gentleman W. M. and Marovich S. B. (1974) More on algorithms */
/*        that reveal properties of floating point arithmetic units. */
/*        Comms. of the ACM, 17, 276-277. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	one = 1.;

/*        LBETA,  LIEEE1,  LT and  LRND  are the  local values  of  BETA, */
/*        IEEE1, T and RND. */

/*        Throughout this routine  we use the function  DLAMC3  to ensure */
/*        that relevant values are  stored and not held in registers,  or */
/*        are not affected by optimizers. */

/*        Compute  a = 2.0**m  with the  smallest positive integer m such */
/*        that */

/*           fl( a + 1.0 ) = a. */

	a = 1.;
	c__ = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L10:
	if (c__ == one) {
	    a *= 2;
	    c__ = hlamc3_(&a, &one);
	    d__1 = -a;
	    c__ = hlamc3_(&c__, &d__1);
	    goto L10;
	}
/* +       END WHILE */

/*        Now compute  b = 2.0**m  with the smallest positive integer m */
/*        such that */

/*           fl( a + b ) .gt. a. */

	b = 1.;
	c__ = hlamc3_(&a, &b);

/* +       WHILE( C.EQ.A )LOOP */
L20:
	if (c__ == a) {
	    b *= 2;
	    c__ = hlamc3_(&a, &b);
	    goto L20;
	}
/* +       END WHILE */

/*        Now compute the base.  a and c  are neighbouring floating point */
/*        numbers  in the  interval  ( beta**t, beta**( t + 1 ) )  and so */
/*        their difference is beta. Adding 0.25 to c is to ensure that it */
/*        is truncated to beta and not ( beta - 1 ). */

	qtr = one / 4;
	savec = c__;
	d__1 = -a;
	c__ = hlamc3_(&c__, &d__1);
	lbeta = (integer) (c__ + qtr);

/*        Now determine whether rounding or chopping occurs,  by adding a */
/*        bit  less  than  beta/2  and a  bit  more  than  beta/2  to  a. */

	b = (halfreal) lbeta;
	d__1 = b / 2;
	d__2 = -b / 100;
	f = hlamc3_(&d__1, &d__2);
	c__ = hlamc3_(&f, &a);
	if (c__ == a) {
	    lrnd = TRUE_;
	} else {
	    lrnd = FALSE_;
	}
	d__1 = b / 2;
	d__2 = b / 100;
	f = hlamc3_(&d__1, &d__2);
	c__ = hlamc3_(&f, &a);
	if (lrnd && c__ == a) {
	    lrnd = FALSE_;
	}

/*        Try and decide whether rounding is done in the  IEEE  'round to */
/*        nearest' style. B/2 is half a unit in the last place of the two */
/*        numbers A and SAVEC. Furthermore, A is even, i.e. has last  bit */
/*        zero, and SAVEC is odd. Thus adding B/2 to A should not  change */
/*        A, but adding B/2 to SAVEC should change SAVEC. */

	d__1 = b / 2;
	t1 = hlamc3_(&d__1, &a);
	d__1 = b / 2;
	t2 = hlamc3_(&d__1, &savec);
	lieee1 = t1 == a && t2 > savec && lrnd;

/*        Now find  the  mantissa, t.  It should  be the  integer part of */
/*        logq to the base beta of a,  however it is safer to determine  t */
/*        by powering.  So we find t as the smallest positive integer for */
/*        which */

/*           fl( beta**t + 1.0 ) = 1.0. */

	lt = 0;
	a = 1.;
	c__ = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L30:
	if (c__ == one) {
	    ++lt;
	    a *= lbeta;
	    c__ = hlamc3_(&a, &one);
	    d__1 = -a;
	    c__ = hlamc3_(&c__, &d__1);
	    goto L30;
	}
/* +       END WHILE */

    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *ieee1 = lieee1;
    first = FALSE_;
    return 0;

/*     End of DLAMC1 */

} /* hlamc1_ */


/* *********************************************************************** */

/* Subroutine */ int hlamc2_(integer *beta, integer *t, logical *rnd, 
	halfreal *eps, integer *emin, halfreal *rmin, integer *emax, 
	halfreal *rmax)
{
    /* Initialized data */

    static logical first = TRUE_;
    static logical iwarn = FALSE_;

    /* Format strings */
    static char fmt_9999[] = "(//\002 WARNING. The value EMIN may be incorre"
	    "ct:-\002,\002  EMIN = \002,i8,/\002 If, after inspection, the va"
	    "lue EMIN looks\002,\002 acceptable please comment out \002,/\002"
	    " the IF block as marked within the code of routine\002,\002 DLAM"
	    "C2,\002,/\002 otherwise supply EMIN explicitly.\002,/)";

    /* System generated locals */
    integer i__1;
    halfreal d__1, d__2, d__3, d__4, d__5;

    /* Local variables */
    static halfreal a, b, c__;
    static integer i__, lt;
    static halfreal one, two;
    static logical ieee;
    static halfreal half;
    static logical lrnd;
    static halfreal leps, zero;
    static integer lbeta;
    static halfreal rbase;
    static integer lemin, lemax, gnmin;
    static halfreal small;
    static integer gpmin;
    static halfreal third, lrmin, lrmax, sixth;
    extern /* Subroutine */ int hlamc1_(integer *, integer *, logical *, 
	    logical *);
    extern halfreal hlamc3_(halfreal *, halfreal *);
    static logical lieee1;
    extern /* Subroutine */ int hlamc4_(integer *, halfreal *, integer *), 
	    hlamc5_(integer *, integer *, integer *, logical *, integer *, 
	    halfreal *);
    static integer ngnmin, ngpmin;

    /* Fortran I/O blocks */
    static cilist io___58 = { 0, 6, 0, fmt_9999, 0 };



/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC2 determines the machine parameters specified in its argument */
/*  list. */

/*  Arguments */
/*  ========= */

/*  BETA    (output) INTEGER */
/*          The base of the machine. */

/*  T       (output) INTEGER */
/*          The number of ( BETA ) digits in the mantissa. */

/*  RND     (output) LOGICAL */
/*          Specifies whether proper rounding  ( RND = .TRUE. )  or */
/*          chopping  ( RND = .FALSE. )  occurs in addition. This may not */
/*          be a reliable guide to the way in which the machine performs */
/*          its arithmetic. */

/*  EPS     (output) DOUBLE PRECISION */
/*          The smallest positive number such that */

/*             fl( 1.0 - EPS ) .LT. 1.0, */

/*          where fl denotes the computed value. */

/*  EMIN    (output) INTEGER */
/*          The minimum exponent before (gradual) underflow occurs. */

/*  RMIN    (output) DOUBLE PRECISION */
/*          The smallest normalized number for the machine, given by */
/*          BASE**( EMIN - 1 ), where  BASE  is the floating point value */
/*          of BETA. */

/*  EMAX    (output) INTEGER */
/*          The maximum exponent before overflow occurs. */

/*  RMAX    (output) DOUBLE PRECISION */
/*          The largest positive number for the machine, given by */
/*          BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating point */
/*          value of BETA. */

/*  Further Details */
/*  =============== */

/*  The computation of  EPS  is based on a routine PARANOIA by */
/*  W. Kahan of the University of California at Berkeley. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     .. */
/*     .. Data statements .. */
/*     .. */
/*     .. Executable Statements .. */

    if (first) {
	zero = 0.;
	one = 1.;
	two = 2.;

/*        LBETA, LT, LRND, LEPS, LEMIN and LRMIN  are the local values of */
/*        BETA, T, RND, EPS, EMIN and RMIN. */

/*        Throughout this routine  we use the function  DLAMC3  to ensure */
/*        that relevant values are stored  and not held in registers,  or */
/*        are not affected by optimizers. */

/*        DLAMC1 returns the parameters  LBETA, LT, LRND and LIEEE1. */

	hlamc1_(&lbeta, &lt, &lrnd, &lieee1);

/*        Start to find EPS. */

	b = (halfreal) lbeta;
	i__1 = -lt;
	a = pow_di(&b, &i__1);
	leps = a;

/*        Try some tricks to see whether or not this is the correct  EPS. */

	b = two / 3;
	half = one / 2;
	d__1 = -half;
	sixth = hlamc3_(&b, &d__1);
	third = hlamc3_(&sixth, &sixth);
	d__1 = -half;
	b = hlamc3_(&third, &d__1);
	b = hlamc3_(&b, &sixth);
	b = abs(b);
	if (b < leps) {
	    b = leps;
	}

	leps = 1.;

/* +       WHILE( ( LEPS.GT.B ).AND.( B.GT.ZERO ) )LOOP */
L10:
	if (leps > b && b > zero) {
	    leps = b;
	    d__1 = half * leps;
/* Computing 5th power */
	    d__3 = two, d__4 = d__3, d__3 *= d__3;
/* Computing 2nd power */
	    d__5 = leps;
	    d__2 = d__4 * (d__3 * d__3) * (d__5 * d__5);
	    c__ = hlamc3_(&d__1, &d__2);
	    d__1 = -c__;
	    c__ = hlamc3_(&half, &d__1);
	    b = hlamc3_(&half, &c__);
	    d__1 = -b;
	    c__ = hlamc3_(&half, &d__1);
	    b = hlamc3_(&half, &c__);
	    goto L10;
	}
/* +       END WHILE */

	if (a < leps) {
	    leps = a;
	}

/*        Computation of EPS complete. */

/*        Now find  EMIN.  Let A = + or - 1, and + or - (1 + BASE**(-3)). */
/*        Keep dividing  A by BETA until (gradual) underflow occurs. This */
/*        is detected when we cannot recover the previous A. */

	rbase = one / lbeta;
	small = one;
	for (i__ = 1; i__ <= 3; ++i__) {
	    d__1 = small * rbase;
	    small = hlamc3_(&d__1, &zero);
/* L20: */
	}
	a = hlamc3_(&one, &small);
	hlamc4_(&ngpmin, &one, &lbeta);
	d__1 = -one;
	hlamc4_(&ngnmin, &d__1, &lbeta);
	hlamc4_(&gpmin, &a, &lbeta);
	d__1 = -a;
	hlamc4_(&gnmin, &d__1, &lbeta);
	ieee = FALSE_;

	if (ngpmin == ngnmin && gpmin == gnmin) {
	    if (ngpmin == gpmin) {
		lemin = ngpmin;
/*            ( Non twos-complement machines, no gradual underflow; */
/*              e.g.,  VAX ) */
	    } else if (gpmin - ngpmin == 3) {
		lemin = ngpmin - 1 + lt;
		ieee = TRUE_;
/*            ( Non twos-complement machines, with gradual underflow; */
/*              e.g., IEEE standard followers ) */
	    } else {
		lemin = f2cmin(ngpmin,gpmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if (ngpmin == gpmin && ngnmin == gnmin) {
	    if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1) {
		lemin = f2cmax(ngpmin,ngnmin);
/*            ( Twos-complement machines, no gradual underflow; */
/*              e.g., CYBER 205 ) */
	    } else {
		lemin = f2cmin(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1 && gpmin == gnmin)
		 {
	    if (gpmin - f2cmin(ngpmin,ngnmin) == 3) {
		lemin = f2cmax(ngpmin,ngnmin) - 1 + lt;
/*            ( Twos-complement machines with gradual underflow; */
/*              no known machine ) */
	    } else {
		lemin = f2cmin(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else {
/* Computing MIN */
	    i__1 = f2cmin(ngpmin,ngnmin), i__1 = f2cmin(i__1,gpmin);
	    lemin = f2cmin(i__1,gnmin);
/*         ( A guess; no known machine ) */
	    iwarn = TRUE_;
	}
	first = FALSE_;
/* ** */
/* Comment out this if block if EMIN is ok */
	if (iwarn) {
	    first = TRUE_;
	    printf("\n\n WARNING. The value EMIN may be incorrect:- ");
	    printf("EMIN = %8i\n",lemin);
	    printf("If, after inspection, the value EMIN looks acceptable");
            printf("please comment out \n the IF block as marked within the"); 
            printf("code of routine DLAMC2, \n otherwise supply EMIN"); 
            printf("explicitly.\n");
	}
/* **   

          Assume IEEE arithmetic if we found denormalised  numbers abo
ve,   
          or if arithmetic seems to round in the  IEEE style,  determi
ned   
          in routine DLAMC1.q A true IEEE machine should have both  thi
ngs   
          true; however, faulty machines may have one or the other. */

	ieee = ieee || lieee1;

/*        Compute  RMIN by successive division by  BETA. We could compute */
/*        RMIN as BASE**( EMIN - 1 ),  but some machines underflow during */
/*        this computation. */

	lrmin = 1.;
	i__1 = 1 - lemin;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__1 = lrmin * rbase;
	    lrmin = hlamc3_(&d__1, &zero);
/* L30: */
	}

/*        Finally, call DLAMC5 to compute EMAX and RMAX. */

	hlamc5_(&lbeta, &lt, &lemin, &ieee, &lemax, &lrmax);
    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *eps = leps;
    *emin = lemin;
    *rmin = lrmin;
    *emax = lemax;
    *rmax = lrmax;

    return 0;


/*     End of DLAMC2 */

} /* hlamc2_ */


/* *********************************************************************** */

halfreal hlamc3_(halfreal *a, halfreal *b)
{
    /* System generated locals */
    halfreal ret_val;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC3  is intended to force  A  and  B  to be stored prior to doing */
/*  the addition of  A  and  B ,  for use in situations where optimizers */
/*  might hold one of these in a register. */

/*  Arguments */
/*  ========= */

/*  A       (input) DOUBLE PRECISION */
/*  B       (input) DOUBLE PRECISION */
/*          The values A and B. */

/* ===================================================================== */

/*     .. Executable Statements .. */

    ret_val = *a + *b;

    return ret_val;

/*     End of DLAMC3 */

} /* hlamc3_ */


/* *********************************************************************** */

/* Subroutine */ int hlamc4_(integer *emin, halfreal *start, integer *base)
{
    /* System generated locals */
    integer i__1;
    halfreal d__1;

    /* Local variables */
    static halfreal a;
    static integer i__;
    static halfreal b1, b2, c1, c2, d1, d2, one, zero, rbase;
    extern halfreal hlamc3_(halfreal *, halfreal *);


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC4 is a service routine for DLAMC2. */

/*  Arguments */
/*  ========= */

/*  EMIN    (output) INTEGER */
/*          The minimum exponent before (gradual) underflow, computed by */
/*          setting A = START and dividing by BASE until the previous A */
/*          can not be recovered. */

/*  START   (input) DOUBLE PRECISION */
/*          The starting point for determining EMIN. */

/*  BASE    (input) INTEGER */
/*          The base of the machine. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    a = *start;
    one = 1.;
    rbase = one / *base;
    zero = 0.;
    *emin = 1;
    d__1 = a * rbase;
    b1 = hlamc3_(&d__1, &zero);
    c1 = a;
    c2 = a;
    d1 = a;
    d2 = a;
/* +    WHILE( ( C1.EQ.A ).AND.( C2.EQ.A ).AND. */
/*    $       ( D1.EQ.A ).AND.( D2.EQ.A )      )LOOP */
L10:
    if (c1 == a && c2 == a && d1 == a && d2 == a) {
	--(*emin);
	a = b1;
	d__1 = a / *base;
	b1 = hlamc3_(&d__1, &zero);
	d__1 = b1 * *base;
	c1 = hlamc3_(&d__1, &zero);
	d1 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d1 += b1;
/* L20: */
	}
	d__1 = a * rbase;
	b2 = hlamc3_(&d__1, &zero);
	d__1 = b2 / rbase;
	c2 = hlamc3_(&d__1, &zero);
	d2 = zero;
	i__1 = *base;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d2 += b2;
/* L30: */
	}
	goto L10;
    }
/* +    END WHILE */

    return 0;

/*     End of DLAMC4 */

} /* hlamc4_ */


/* *********************************************************************** */

/* Subroutine */ int hlamc5_(integer *beta, integer *p, integer *emin, 
	logical *ieee, integer *emax, halfreal *rmax)
{
    /* System generated locals */
    integer i__1;
    halfreal d__1;

    /* Local variables */
    static integer i__;
    static halfreal y, z__;
    static integer try__, lexp;
    static halfreal oldy;
    static integer uexp, nbits;
    extern halfreal hlamc3_(halfreal *, halfreal *);
    static halfreal recbas;
    static integer exbits, expsum;


/*  -- LAPACK auxiliary routine (version 3.3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2010 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAMC5 attempts to compute RMAX, the largest machine floating-point */
/*  number, without overflow.  It assumes that EMAX + abs(EMIN) sum */
/*  approximately to a power of 2.  It will fail on machines where this */
/*  assumption does not hold, for example, the Cyber 205 (EMIN = -28625, */
/*  EMAX = 28718).  It will also fail if the value supplied for EMIN is */
/*  too large (i.e. too close to zero), probably with overflow. */

/*  Arguments */
/*  ========= */

/*  BETA    (input) INTEGER */
/*          The base of floating-point arithmetic. */

/*  P       (input) INTEGER */
/*          The number of base BETA digits in the mantissa of a */
/*          floating-point value. */

/*  EMIN    (input) INTEGER */
/*          The minimum exponent before (gradual) underflow. */

/*  IEEE    (input) LOGICAL */
/*          A logical flag specifying whether or not the arithmetic */
/*          system is thought to comply with the IEEE standard. */

/*  EMAX    (output) INTEGER */
/*          The largest exponent before overflow */

/*  RMAX    (output) DOUBLE PRECISION */
/*          The largest machine floating-point number. */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     First compute LEXP and UEXP, two powers of 2 that bound */
/*     abs(EMIN). We then assume that EMAX + abs(EMIN) will sum */
/*     approximately to the bound that is closest to abs(EMIN). */
/*     (EMAX is the exponent of the required number RMAX). */

    lexp = 1;
    exbits = 1;
L10:
    try__ = lexp << 1;
    if (try__ <= -(*emin)) {
	lexp = try__;
	++exbits;
	goto L10;
    }
    if (lexp == -(*emin)) {
	uexp = lexp;
    } else {
	uexp = try__;
	++exbits;
    }

/*     Now -LEXP is less than or equal to EMIN, and -UEXP is greater */
/*     than or equal to EMIN. EXBITS is the number of bits needed to */
/*     store the exponent. */

    if (uexp + *emin > -lexp - *emin) {
	expsum = lexp << 1;
    } else {
	expsum = uexp << 1;
    }

/*     EXPSUM is the exponent range, approximately equal to */
/*     EMAX - EMIN + 1 . */

    *emax = expsum + *emin - 1;
    nbits = exbits + 1 + *p;

/*     NBITS is the total number of bits needed to store a */
/*     floating-point number. */

    if (nbits % 2 == 1 && *beta == 2) {

/*        Either there are an odd number of bits used to store a */
/*        floating-point number, which is unlikely, or some bits are */
/*        not used in the representation of numbers, which is possible, */
/*        (e.g. Cray machines) or the mantissa has an implicit bit, */
/*        (e.g. IEEE machines, Dec Vax machines), which is perhaps the */
/*        most likely. We have to assume the last alternative. */
/*        If this is true, then we need to reduce EMAX by one because */
/*        there must be some way of representing zero in an implicit-bit */
/*        system. On machines like Cray, we are reducing EMAX by one */
/*        unnecessarily. */

	--(*emax);
    }

    if (*ieee) {

/*        Assume we are on an IEEE machine which reserves one exponent */
/*        for infinity and NaN. */

	--(*emax);
    }

/*     Now create RMAX, the largest machine number, which should */
/*     be equal to (1.0 - BETA**(-P)) * BETA**EMAX . */

/*     First compute 1.0 - BETA**(-P), being careful that the */
/*     result is less than 1.0 . */

    recbas = 1. / *beta;
    z__ = *beta - 1.;
    y = 0.;
    i__1 = *p;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__ *= recbas;
	if (y < 1.) {
	    oldy = y;
	}
	y = hlamc3_(&y, &z__);
/* L20: */
    }
    if (y >= 1.) {
	y = oldy;
    }

/*     Now multiply by BETA**EMAX to get RMAX. */

    i__1 = *emax;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__1 = y * *beta;
	y = hlamc3_(&d__1, &c_b32);
/* L30: */
    }

    *rmax = y;
    return 0;

/*     End of DLAMC5 */

} /* hlamc5_ */
EOF
	if [[ $TESTING != 0 ]]; then
		cat << EOF > ${LAPACKDIR}/second.c
#include "f2c.h"
#include <sys/times.h>
#include <sys/types.h>
#include <time.h>

#ifndef CLK_TCK
#define CLK_TCK 60
#endif

real second_()
{
  struct tms rusage;

  times(&rusage);
  return (real)(rusage.tms_utime) / CLK_TCK;

} /* second_ */
EOF

		cat << EOF > ${LAPACKDIR}/dsecnd.c
#include "f2c.h"
#include <sys/times.h>
#include <sys/types.h>
#include <time.h>

#ifndef CLK_TCK
#define CLK_TCK 60
#endif

doublereal dsecnd_()
{
  struct tms rusage;

  times(&rusage);
  return (doublereal)(rusage.tms_utime) / CLK_TCK;

} /* dsecnd_ */
EOF
	fi

	cat << EOF > ${BLASDIR}/lsame.c
#include "f2c.h"

logical lsame_(char *ca, char *cb)
{
/*  -- LAPACK auxiliary routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       September 30, 1994   


    Purpose   
    =======   

    LSAME returns .TRUE. if CA is the same letter as CB regardless of   
    case.   

    Arguments   
    =========   

    CA      (input) CHARACTER*1   
    CB      (input) CHARACTER*1   
            CA and CB specify the single characters to be compared.   

   ===================================================================== 
  


       Test if the characters are equal */
    /* System generated locals */
    logical ret_val;
    /* Local variables */
    static integer inta, intb, zcode;


    ret_val = *(unsigned char *)ca == *(unsigned char *)cb;
    if (ret_val) {
	return ret_val;
    }

/*     Now test for equivalence if both characters are alphabetic. */

    zcode = 'Z';

/*     Use 'Z' rather than 'A' so that ASCII can be detected on Prime   
       machines, on which ICHAR returns a value with bit 8 set.   
       ICHAR('A') on Prime machines returns 193 which is the same as   
       ICHAR('A') on an EBCDIC machine. */

    inta = *(unsigned char *)ca;
    intb = *(unsigned char *)cb;

    if (zcode == 90 || zcode == 122) {

/*        ASCII is assumed - ZCODE is the ASCII code of either lower o
r   
          upper case 'Z'. */

	if (inta >= 97 && inta <= 122) {
	    inta += -32;
	}
	if (intb >= 97 && intb <= 122) {
	    intb += -32;
	}

    } else if (zcode == 233 || zcode == 169) {

/*        EBCDIC is assumed - ZCODE is the EBCDIC code of either lower
 or   
          upper case 'Z'. */

	if (inta >= 129 && inta <= 137 || inta >= 145 && inta <= 153 || inta 
		>= 162 && inta <= 169) {
	    inta += 64;
	}
	if (intb >= 129 && intb <= 137 || intb >= 145 && intb <= 153 || intb 
		>= 162 && intb <= 169) {
	    intb += 64;
	}

    } else if (zcode == 218 || zcode == 250) {

/*        ASCII is assumed, on Prime machines - ZCODE is the ASCII cod
e   
          plus 128 of either lower or upper case 'Z'. */

	if (inta >= 225 && inta <= 250) {
	    inta += -32;
	}
	if (intb >= 225 && intb <= 250) {
	    intb += -32;
	}
    }
    ret_val = inta == intb;

/*     RETURN   

       End of LSAME */

    return ret_val;
} /* lsame_ */
EOF

	cat << EOF > ${LAPACKDIR}/lsamen.c
#include "f2c.h"
#include <string.h>

logical lsamen_(integer *n, char *ca, char *cb)
{
/*  -- LAPACK auxiliary routine (version 3.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       September 30, 1994   


    Purpose   
    =======   

    LSAMEN  tests if the first N letters of CA are the same as the   
    first N letters of CB, regardless of case.   
    LSAMEN returns .TRUE. if CA and CB are equivalent except for case   
    and .FALSE. otherwise.  LSAMEN also returns .FALSE. if LEN( CA )   
    or LEN( CB ) is less than N.   

    Arguments   
    =========   

    N       (input) INTEGER   
            The number of characters in CA and CB to be compared.   

    CA      (input) CHARACTER*(*)   
    CB      (input) CHARACTER*(*)   
            CA and CB specify two character strings of length at least N. 
  
            Only the first N characters of each string will be accessed. 
  

   ===================================================================== 
*/
    /* System generated locals */
    integer i__1;
    logical ret_val;
    /* Local variables */
    static integer i;
    extern logical lsame_(char *, char *);


    ret_val = FALSE_;
    if (strlen(ca) < *n || strlen(cb) < *n) {
	goto L20;
    }

/*     Do for each character in the two strings. */

    i__1 = *n;
    for (i = 1; i <= *n; ++i) {

/*        Test if the characters are equal using LSAME. */

	if (! lsame_(ca + (i - 1), cb + (i - 1))) {
	    goto L20;
	}

/* L10: */
    }
    ret_val = TRUE_;

L20:
    return ret_val;

/*     End of LSAMEN */

} /* lsamen_ */
EOF

	cat << EOF > ${LAPACKDIR}/chla_transtype.c
/*  -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Character */ VOID chla_transtype__(char *ret_val, 
	integer *trans)
{

/*  -- LAPACK routine (version 3.2.2) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     October 2008 */


/*  Purpose */
/*  ======= */

/*  This subroutine translates from a BLAST-specified integer constant to */
/*  the character string specifying a transposition operation. */

/*  CHLA_TRANSTYPE returns an CHARACTER*1.  If CHLA_TRANSTYPE is 'X', */
/*  then input is not an integer indicating a transposition operator. */
/*  Otherwise CHLA_TRANSTYPE returns the constant value corresponding to */
/*  TRANS. */

/*  Arguments */
/*  ========= */
/*  TRANS   (input) INTEGER */
/*          Specifies the form of the system of equations: */
/*          = BLAS_NO_TRANS   = 111 :  No Transpose */
/*          = BLAS_TRANS      = 112 :  Transpose */
/*          = BLAS_CONJ_TRANS = 113 :  Conjugate Transpose */
/*  ===================================================================== */

    if (*trans == 111) {
	*(unsigned char *)ret_val = 'N';
    } else if (*trans == 112) {
	*(unsigned char *)ret_val = 'T';
    } else if (*trans == 113) {
	*(unsigned char *)ret_val = 'C';
    } else {
	*(unsigned char *)ret_val = 'X';
    }
    return ;

/*     End of CHLA_TRANSTYPE */

} /* chla_transtype__ */
EOF

	cat <<EOF > ${TMP}/f2c.h
/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef F2C_INCLUDE
#define F2C_INCLUDE

#include <math.h>
#if defined(__LAPACK_PRECISION_QUAD)
#	include <quadmath.h>
#	define M(A) A##q
	typedef __float128 quadreal;
	typedef struct { quadreal r, i; } quadcomplex;
#	define scalar __float128
#	define scalarcomplex quadcomplex
#	define dscalar __float128
#elif defined(__LAPACK_PRECISION_HALF)
#	define M(A) A##f
	typedef __fp16 halfreal;
	typedef struct { halfreal r, i; } halfcomplex;
#	define scalar __fp16
#	define scalarcomplex halfcomplex
#	define dscalar __fp16
#elif defined( __LAPACK_PRECISION_SINGLE)
#	define M(A) A##f
#	define scalar float
#	define scalarcomplex complex
#	define dscalar double
#else
#	define M(A) A
#	define scalar double
#	define scalarcomplex doublecomplex
#	define dscalar double
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef int integer;
typedef unsigned int uinteger;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
typedef int logical;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;

#define TRUE_ (1)
#define FALSE_ (0)

/* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

/* I/O stuff */

typedef int flag;
typedef int ftnlen;
typedef int ftnint;

/*external read, write*/
typedef struct
{	flag cierr;
	ftnint ciunit;
	flag ciend;
	char *cifmt;
	ftnint cirec;
} cilist;

/*internal read, write*/
typedef struct
{	flag icierr;
	char *iciunit;
	flag iciend;
	char *icifmt;
	ftnint icirlen;
	ftnint icirnum;
} icilist;

/*open*/
typedef struct
{	flag oerr;
	ftnint ounit;
	char *ofnm;
	ftnlen ofnmlen;
	char *osta;
	char *oacc;
	char *ofm;
	ftnint orl;
	char *oblnk;
} olist;

/*close*/
typedef struct
{	flag cerr;
	ftnint cunit;
	char *csta;
} cllist;

/*rewind, backspace, endfile*/
typedef struct
{	flag aerr;
	ftnint aunit;
} alist;

/* inquire */
typedef struct
{	flag inerr;
	ftnint inunit;
	char *infile;
	ftnlen infilen;
	ftnint	*inex;	/*parameters in standard's order*/
	ftnint	*inopen;
	ftnint	*innum;
	ftnint	*innamed;
	char	*inname;
	ftnlen	innamlen;
	char	*inacc;
	ftnlen	inacclen;
	char	*inseq;
	ftnlen	inseqlen;
	char 	*indir;
	ftnlen	indirlen;
	char	*infmt;
	ftnlen	infmtlen;
	char	*inform;
	ftnint	informlen;
	char	*inunf;
	ftnlen	inunflen;
	ftnint	*inrecl;
	ftnint	*innrec;
	char	*inblank;
	ftnlen	inblanklen;
} inlist;

#define VOID void

union Multitype {	/* for multiple entry points */
	integer1 g;
	shortint h;
	integer i;
	/* longint j; */
	real r;
	doublereal d;
	complex c;
	doublecomplex z;
	};

typedef union Multitype Multitype;

struct Vardesc {	/* for Namelist */
	char *name;
	char *addr;
	ftnlen *dims;
	int  type;
	};
typedef struct Vardesc Vardesc;

struct Namelist {
	char *name;
	Vardesc **vars;
	int nvars;
	};
typedef struct Namelist Namelist;

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (abs(x))
#define f2cmin(a,b) ((a) <= (b) ? (a) : (b))
#define f2cmax(a,b) ((a) >= (b) ? (a) : (b))
#define dmin(a,b) (f2cmin(a,b))
#define dmax(a,b) (f2cmax(a,b))
#define bit_test(a,b)	((a) >> (b) & 1)
#define bit_clear(a,b)	((a) & ~((uinteger)1 << (b)))
#define bit_set(a,b)	((a) |  ((uinteger)1 << (b)))

#define abort_() { \
sig_die("Fortran abort routine called", 1); \
}
#if defined(__LAPACK_PRECISION_QUAD)
#	define f__cabs(r,i) qf__cabs((r),(i))
	extern scalar qf__cabs(scalar r, scalar i);
#elif defined(__LAPACK_PRECISION_HALF)
#	define f__cabs(r,i) hf__cabs((r),(i))
	extern scalar hf__cabs(scalar r, scalar i);
#elif defined( __LAPACK_PRECISION_SINGLE)
#	define f__cabs(r,i) sf__cabs((r),(i))
	extern scalar sf__cabs(scalar r, scalar i);
#else
#	define f__cabs(r,i) df__cabs((r),(i))
	extern scalar df__cabs(scalar r, scalar i);
#endif
#define c_abs(z) ( f__cabs( (z)->r, (z)->i ) )
#define c_cos(R,Z) {(R)->r = (M(cos)((Z)->r) * M(cosh)((Z)->i)); (R)->i = (-M(sin)((Z)->r) * M(sinh)((Z)->i));}
#define c_div(c, a, b) { \
	scalar __ratio, __den, __abr, __abi, __cr; \
	if( (__abr = (b)->r) < 0.) \
		__abr = - __abr; \
	if( (__abi = (b)->i) < 0.) \
		__abi = - __abi; \
	if( __abr <= __abi ) \
		{ \
		if(__abi == 0) \
			sig_die("complex division by zero", 1); \
		__ratio = (b)->r / (b)->i ; \
		__den = (b)->i * (M(1.0) + __ratio*__ratio); \
		__cr = ((a)->r*__ratio + (a)->i) / __den; \
		(c)->i = ((a)->i*__ratio - (a)->r) / __den; \
		} \
	else \
		{ \
		__ratio = (b)->i / (b)->r ; \
		__den = (b)->r * (1 + __ratio*__ratio); \
		__cr = ((a)->r + (a)->i*__ratio) / __den; \
		(c)->i = ((a)->i - (a)->r*__ratio) / __den; \
		} \
	(c)->r = __cr; \
	}
#define z_div(c, a, b) c_div(c, a, b)
#define c_exp(R, Z) { (R)->r = (M(exp)((Z)->r) * M(cos)((Z)->i)); (R)->i = (M(exp)((Z)->r) * M(sin)((Z)->i)); }
#define c_log(R, Z) { (R)->i = M(atan2)((Z)->i, (Z)->r); (R)->r = M(log)( f__cabs((Z)->r, (Z)->i) ); }
#define c_sin(R, Z) { (R)->r = (M(sin)((Z)->r) * M(cosh)((Z)->i)); (R)->i = (M(cos)((Z)->r) * M(sinh)((Z)->i)); }
#define c_sqrt(R, Z) { \
	scalar __mag, __t, __zi = (Z)->i, __zr = (Z)->r; \
	if( (__mag = f__cabs(__zr, __zi)) == 0.) (R)->r = (R)->i = 0.; \
	else if(__zr > 0) { \
		(R)->r = __t = M(sqrt)(M(0.5) * (__mag + __zr) ); \
		__t = __zi / __t; \
		(R)->i = M(0.5) * __t; \
	} else { \
		__t = M(sqrt)(M(0.5) * (__mag - __zr) ); \
		if(__zi < 0) __t = -__t; \
		(R)->i = __t; \
		__t = __zi / __t; \
		(R)->r = M(0.5) * __t; \
	} \
}
#define d_abs(x) abs(*(x))
#define d_acos(x) (M(acos)(*(x)))
#define d_asin(x) (M(asin)(*(x)))
#define d_atan(x) (M(atan)(*(x)))
#define d_atn2(x, y) (M(atan2)(*(x),*(y)))
#define d_cnjg(R, Z) { (R)->r = (Z)->r;	(R)->i = -((Z)->i); }
#define d_cos(x) (M(cos)(*(x)))
#define d_cosh(x) (M(cosh)(*(x)))
#define d_dim(__a, __b) ( *(__a) > *(__b) ? *(__a) - *(__b) : 0.0 )
#define d_exp(x) (M(exp)(*(x)))
#define d_imag(z) ((z)->i)
#define d_int(__x) (*(__x)>0 ? M(floor)(*(__x)) : -M(floor)(- *(__x)))
#define d_lg10(x) ( M(0.43429448190325182765) * M(log)(*(x)) )
#define d_log(x) (M(log)(*(x)))
#define d_mod(x, y) (M(fmod)(*(x), *(y)))
#define u_nint(__x) ((__x)>=0 ? M(floor)((__x) + M(.5)) : -M(floor)(M(.5) - (__x)))
#define d_nint(x) u_nint(*(x))
#define u_sign(__a,__b) ((__b) >= 0 ? ((__a) >= 0 ? (__a) : -(__a)) : -((__a) >= 0 ? (__a) : -(__a)))
#define d_sign(a,b) u_sign(*(a),*(b))
#define d_sin(x) (M(sin)(*(x)))
#define d_sinh(x) (M(sinh)(*(x)))
#define d_sqrt(x) (M(sqrt)(*(x)))
#define d_tan(x) (M(tan)(*(x)))
#define d_tanh(x) (M(tanh)(*(x)))
#define i_abs(x) abs(*(x))
#define i_dnnt(x) ((integer)u_nint(*(x)))
#define i_len(s, n) (n)
#define i_nint(x) ((integer)u_nint(*(x)))
#define i_sign(a,b) ((integer)u_sign((integer)*(a),(integer)*(b)))
#define pow_ci(p, a, b) { pow_zi((p), (a), (b)); }
#define pow_dd(ap, bp) (M(pow)(*(ap), *(bp)))
#if defined(__LAPACK_PRECISION_QUAD)
#	define pow_di(B,E) qpow_ui((B),*(E))
	extern dscalar qpow_ui(scalar *_x, integer n);
#elif defined(__LAPACK_PRECISION_HALF)
#	define pow_di(B,E) hpow_ui((B),*(E))
	extern dscalar hpow_ui(scalar *_x, integer n);
#elif defined( __LAPACK_PRECISION_SINGLE)
#	define pow_ri(B,E) spow_ui((B),*(E))
	extern dscalar spow_ui(scalar *_x, integer n);
#else
#	define pow_di(B,E) dpow_ui((B),*(E))
	extern dscalar dpow_ui(scalar *_x, integer n);
#endif
extern integer pow_ii(integer*,integer*);
#define pow_zi(p, a, b) { \
	integer __n=*(b); unsigned long __u; scalar __t; scalarcomplex __x; \
	static scalarcomplex one = {1.0, 0.0}; \
	(p)->r = 1; (p)->i = 0; \
	if(__n != 0) { \
		if(__n < 0) { \
			__n = -__n; \
			z_div(&__x, &one, (a)); \
		} else { \
			__x.r = (a)->r; __x.i = (a)->i; \
		} \
		for(__u = __n; ; ) { \
			if(__u & 01) { \
				__t = (p)->r * __x.r - (p)->i * __x.i; \
				(p)->i = (p)->r * __x.i + (p)->i * __x.r; \
				(p)->r = __t; \
			} \
			if(__u >>= 1) { \
				__t = __x.r * __x.r - __x.i * __x.i; \
				__x.i = 2 * __x.r * __x.i; \
				__x.r = __t; \
			} else break; \
		} \
	} \
}
#define pow_zz(R,A,B) { \
	scalar __logr, __logi, __x, __y; \
	__logr = M(log)( f__cabs((A)->r, (A)->i) ); \
	__logi = M(atan2)((A)->i, (A)->r); \
	__x = M(exp)( __logr * (B)->r - __logi * (B)->i ); \
	__y = __logr * (B)->i + __logi * (B)->r; \
	(R)->r = __x * M(cos)(__y); \
	(R)->i = __x * M(sin)(__y); \
}
#define r_cnjg(R, Z) d_cnjg(R,Z)
#define r_imag(z) d_imag(z)
#define r_lg10(x) d_lg10(x)
#define r_sign(a,b) d_sign(a,b)
#define s_cat(lpp, rpp, rnp, np, llp) { \
	ftnlen i, nc, ll; char *f__rp, *lp; \
	ll = (llp); lp = (lpp); \
	for(i=0; i < (int)*(np); ++i) { \
        	nc = ll; \
	        if((rnp)[i] < nc) nc = (rnp)[i]; \
	        ll -= nc; \
        	f__rp = (rpp)[i]; \
	        while(--nc >= 0) *lp++ = *(f__rp)++; \
        } \
	while(--ll >= 0) *lp++ = ' '; \
}
#define s_cmp(a,b,c,d) ((integer)strncmp((a),(b),f2cmin((c),(d))))
#define s_copy(A,B,C,D) { strncpy((A),(B),f2cmin((C),(D))); }
#define sig_die(s, kill) { exit(1); }
#define s_stop(s, n) {exit(0);}
static char junk[] = "\n@(#)LIBF77 VERSION 19990503\n";
#define z_abs(z) c_abs(z)
#define z_exp(R, Z) c_exp(R, Z)
#define z_sqrt(R, Z) c_sqrt(R, Z)
#define myexit_() break;
#if defined(__LAPACK_PRECISION_QUAD)
#	define mymaxloc_(w,s,e,n) qmaxloc_((w),*(s),*(e),n)
	extern integer qmaxloc_(scalar *w, integer s, integer e, integer *n);
#elif defined(__LAPACK_PRECISION_HALF)
#	define mymaxloc_(w,s,e,n) hmaxloc_((w),*(s),*(e),n)
	extern integer hmaxloc_(scalar *w, integer s, integer e, integer *n);
#elif defined( __LAPACK_PRECISION_SINGLE)
#	define mymaxloc_(w,s,e,n) smaxloc_((w),*(s),*(e),n)
	extern integer smaxloc_(scalar *w, integer s, integer e, integer *n);
#else
#	define mymaxloc_(w,s,e,n) dmaxloc_((w),*(s),*(e),n)
	extern integer dmaxloc_(scalar *w, integer s, integer e, integer *n);
#endif

/* procedure parameter types for -A and -C++ */

#define F2C_proc_par_types 1
#ifdef __cplusplus
typedef logical (*L_fp)(...);
#else
typedef logical (*L_fp)();
#endif
#endif
EOF
	cp ${TMP}/f2c.h ${BLASDIR}
	cp ${TMP}/f2c.h ${LAPACKDIR}


	cat <<EOF > ${BLASDIR}/pow_ii.c
#include "f2c.h"
integer pow_ii(integer *_x, integer *_n) {
	integer x=*_x, n=*_n, pow; unsigned long int u;
	if (n <= 0) {
		if (n == 0 || x == 1) pow = 1;
		else if (x != -1) pow = x == 0 ? 1/x : 0;
		else n = -n;
	}
	if ((n > 0) || !(n == 0 || x == 1 || x != -1)) {
		u = n;
		for(pow = 1; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
EOF

	for i in s d q h; do
		case $i in
		s) P="SINGLE";;
		d) P="DOUBLE";;
		q) P="QUAD";;
		h) P="HALF";;
		esac

		cat <<EOF > ${BLASDIR}/pow_${i}i.c
#define __LAPACK_PRECISION_${P}
#include "f2c.h"
dscalar ${i}pow_ui(scalar *_x, integer n) {
	dscalar x = *_x; dscalar pow=1.0; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x = 1/x;
		for(u = n; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
EOF
		cat <<EOF > ${BLASDIR}/${i}maxloc.c
#define __LAPACK_PRECISION_${P}
#include "f2c.h"
integer ${i}maxloc_(scalar *w, integer s, integer e, integer *n)
{
	scalar m; integer i, mi;
	for(m=w[s-1], mi=s, i=s+1; i<=e; i++)
		if (w[i-1]>m) mi=i ,m=w[i-1];
	return mi-s+1;
}
EOF

		cat <<EOF > ${BLASDIR}/${i}f__cabs.c
#define __LAPACK_PRECISION_${P}
#include "f2c.h"
scalar ${i}f__cabs(scalar r, scalar i) {
	scalar temp;
	if(r < 0) r = -r;
	if(i < 0) i = -i;
	if(i > r){
		temp = r;
		r = i;
		i = temp;
	}
	if((r+i) == r)
		temp = r;
	else {
		temp = i/r;
		temp = r*M(sqrt)(M(1.0) + temp*temp);  /*overflow!!*/
	}
	return temp;
}
EOF
	done

		
# 3) Make the package, copy it to the current directory
#	 and remove temp directory
cd $TMP
$TAR --create --gzip --file $ORIG/${FBLASLAPACK}.tar.gz ${FBLASLAPACK}
cd $ORIG 
rm -r $TMP

# 4) Testing the single and double routines in BLAS and LAPACK
# In order to execute the available tests in blas and lapack I have to
# transform the fortran test files to C also. For that I usedlibf2c, that
# is available in http://www.netlib.org/f2c/libf2c.zip or in CLAPACK
# http://www.netlib.org/clapack/clapack.tgz
#
# 4.1) Create a script that tranlate the code to c before compiling:
#
# cat << EOF > cfortran
#!/bin/bash
#RM=""
#for f in `echo $@ | tr ' ' '\n' | sed -n -r -e '/[^ ]+\.f/ {p}'`; do
#	sed -r -e "
#		s/(SUBROUTINE[^(]+\\([^)]+\\))/\\1\\n      EXTERNAL LEN_TRIM, CEILING\\n       INTEGER LEN_TRIM, CEILING\\n/g;
#		s/(INTRINSIC [^\\n]*)LEN_TRIM/\\1 MAX/g;
#		s/(INTRINSIC [^\\n]*)CEILING/\\1 MIN/g;
#	" < $f | \
#	f2c -a -A -R | \
#	sed -e "
#		1 i\
#		#define len_trim__(cad,len) ({ \
#			integer _r=0,i; \
#			for(i=0; i<(len) && (cad)[i]; i++) \
#				if((cad)[i] != ' ') _r=i; \
#			_r+1; })
#		1 i\
#		#define ceiling_(a) (myceil(*(a)))
#		1 i\
#		#define myceil(a) (sizeof(a) == sizeof(float) ? ceilf(a) : ceil(a))
#		1 i\
#		#include <math.h>
#		s/extern integer len_trim__([^)]*);//g
#		s/extern [^ ]* ceiling_([^)]*);//g
#	" > ${f/.f/.c}
#	RM="$RM ${f/.f/.c}"
#done
#
#gcc `echo $@ | sed -r -e 's/([^ ]+)\.f/\1.c/g'` -I$HOME/local/include -L$HOME/local/lib -lf2c -lm
#[ -z $DDD ] && rm $RM
#EOF
#
## 4.2) Modify make.inc from an original LAPACK distribution in order to change the
## compiler and to link properly. I modified the next variables in main.inc:
#
#FORTRAN = cfortran
#LOADER  = mygcc gcc "-L/home/eloy/local/lib -lf2c -lm"
#LOADOPTS = -lf2c -lm
#
# where
# cat << EOF > mygcc
##!/bin/bash
#C="$1"; A="$2"; shift; shift; $C $@ $A
#EOF
#
## 4.3) Fix xerbla.f from TESTING/{LIN,EIG}, replacing the fortran code by someone similar to this
#cat << EOF > xerbla.c
##define len_trim__(cad,len) ({ 			integer _r=0,i; 			for(i=0; i<(len) && (cad)[i]; i++) 				if((cad)[i] != ' ') _r=i; 			_r+1; })
##define ceiling_(a) (ceil(*(a)))
#/*  -- translated by f2c (version 20100827).
#   You must link the resulting object file with libf2c:
#	on Microsoft Windows system, link with libf2c.lib;
#	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
#	or, if you install libf2c.a in a standard place, with -lf2c -lm
#	-- in that order, at the end of the command line, as in
#		cc *.o -lf2c -lm
#	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,
#
#		http://www.netlib.org/f2c/libf2c.zip
#*/
#
##include "f2c.h"
#
#/* Common Block Declarations */
#
#struct {
#    integer infot, nout;
#    logical ok, lerr;
#} infoc_;
#
##define infoc_1 infoc_
#
#struct {
#    char srnamt[32];
#} srnamc_;
#
##define srnamc_1 srnamc_
#
#/* Table of constant values */
#
#static integer c__1 = 1;
#
#/* Subroutine */ int xerbla_(char *srname, integer *info)
#{
#    ftnlen srname_len = strlen(srname);
#
#    /* Format strings */
#    static char fmt_9999[] = "(\002 *** XERBLA was called from \002,a,\002 w"
#	    "ith INFO = \002,i6,\002 instead of \002,i2,\002 ***\002)";
#    static char fmt_9997[] = "(\002 *** On entry to \002,a,\002 parameter nu"
#	    "mber \002,i6,\002 had an illegal value ***\002)";
#    static char fmt_9998[] = "(\002 *** XERBLA was called with SRNAME = \002"
#	    ",a,\002 instead of \002,a6,\002 ***\002)";
#
#    /* Builtin functions */
#    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void),
#	     s_cmp(char *, char *, ftnlen, ftnlen);
#
#    /* Local variables */
#    
#
#    /* Fortran I/O blocks */
#    static cilist io___1 = { 0, 0, 0, fmt_9999, 0 };
#    static cilist io___2 = { 0, 0, 0, fmt_9997, 0 };
#    static cilist io___3 = { 0, 0, 0, fmt_9998, 0 };
#
#
#
#/*  -- LAPACK auxiliary routine (version 3.1) -- */
#/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
#/*     November 2006 */
#
#/*     .. Scalar Arguments .. */
#/*     .. */
#
#/*  Purpose */
#/*  ======= */
#
#/*  This is a special version of XERBLA to be used only as part of */
#/*  the test program for testing error exits from the LAPACK routines. */
#/*  Error messages are printed if INFO.NE.INFOT or if SRNAME.NE.SRMANT, */
#/*  where INFOT and SRNAMT are values stored in COMMON. */
#
#/*  Arguments */
#/*  ========= */
#
#/*  SRNAME  (input) CHARACTER*(*) */
#/*          The name of the subroutine calling XERBLA.  This name should */
#/*          match the COMMON variable SRNAMT. */
#
#/*  INFO    (input) INTEGER */
#/*          The error return code from the calling subroutine.  INFO */
#/*          should equal the COMMON variable INFOT. */
#
#/*  Further Details */
#/*  ======= ======= */
#
#/*  The following variables are passed via the common blocks INFOC and */
#/*  SRNAMC: */
#
#/*  INFOT   INTEGER      Expected integer return code */
#/*  NOUT    INTEGER      Unit number for printing error messages */
#/*  OK      LOGICAL      Set to .TRUE. if INFO = INFOT and */
#/*                       SRNAME = SRNAMT, otherwise set to .FALSE. */
#/*  LERR    LOGICAL      Set to .TRUE., indicating that XERBLA was called */
#/*  SRNAMT  CHARACTER*(*) Expected name of calling subroutine */
#
#
#/*     .. Scalars in Common .. */
#/*     .. */
#/*     .. Intrinsic Functions .. */
#/*     .. */
#/*     .. Common blocks .. */
#/*     .. */
#/*     .. Executable Statements .. */
#
#    infoc_1.lerr = TRUE_;
#    if (*info != infoc_1.infot) {
#	if (infoc_1.infot != 0) {
#	    io___1.ciunit = infoc_1.nout;
#	    s_wsfe(&io___1);
#	    do_fio(&c__1, srnamc_1.srnamt, len_trim__(srnamc_1.srnamt, (
#		    ftnlen)32));
#	    do_fio(&c__1, (char *)&(*info), (ftnlen)sizeof(integer));
#	    do_fio(&c__1, (char *)&infoc_1.infot, (ftnlen)sizeof(integer));
#	    e_wsfe();
#	} else {
#	    io___2.ciunit = infoc_1.nout;
#	    s_wsfe(&io___2);
#	    do_fio(&c__1, srname, len_trim__(srname, srname_len));
#	    do_fio(&c__1, (char *)&(*info), (ftnlen)sizeof(integer));
#	    e_wsfe();
#	}
#	infoc_1.ok = FALSE_;
#    }
#    if (s_cmp(srname, srnamc_1.srnamt, srname_len, (ftnlen)32) != 0) {
#	io___3.ciunit = infoc_1.nout;
#	s_wsfe(&io___3);
#	do_fio(&c__1, srname, len_trim__(srname, srname_len));
#	do_fio(&c__1, srnamc_1.srnamt, len_trim__(srnamc_1.srnamt, (ftnlen)32)
#		);
#	e_wsfe();
#	infoc_1.ok = FALSE_;
#    }
#    return 0;
#
#
#/*     End of XERBLA */
#
#} /* xerbla_ */
#EOF

# 5) Testing the quad routines in BLAS and LAPACK
# In that case, I have to modify the double routines in libf2c, that is,
# the d_*.c, pow_d*.c and cabs.c files, changing double by doublereal and the
# math functions by the quad version (log to logq, sin to sinq...). Then I set
# doublereal as __float128 and real as double in f2c.h.
#
# The cfortran script is modified in order to apply the ql.sed sed script
# genereted by this bash script:
#
# cat << EOF > cfortran
##!/bin/bash
#
#FUN=""
#for i in sqrt sin cos log exp; do
#	FUN="$FUN s/([^a-zA-Z_]+)$i([^a-zA-Z_]+)/\\1${i}q\\2/g;"
#done
#FUN="$FUN s/([^a-zA-Z_]+)double([^a-zA-Z_]+)/\\1__float128\\2/g;"
#
#RM=""
#for f in `echo $@ | tr ' ' '\n' | sed -n -r -e '/[^ ]+\.f/ {p}'`; do
#	sed -r -e "
#		s/(^ *SUBROUTINE[^(]+\\([^)]+\\))/\\1\\n      EXTERNAL LEN_TRIM, CEILING\\n       INTEGER LEN_TRIM, CEILING\\n/g;
#		s/(INTRINSIC [^\\n]*)LEN_TRIM/\\1 MAX/g;
#		s/(INTRINSIC [^\\n]*)CEILING/\\1 MIN/g;
#	" < $f | \
#	f2c -a -A -R | \
#	sed -e "
#		1 i\
#		#define len_trim__(cad,len) ({ \
#			integer _r=0,i; \
#			for(i=0; i<(len) && (cad)[i]; i++) \
#				if((cad)[i] != ' ') _r=i; \
#			_r+1; })
#		1 i\
#		#define ceiling_(a) (myceil(*(a)))
#		1 i\
#		#define myceil(a) (sizeof(a) == sizeof(float) ? ceilf(a) : ceil(a))
#		1 i\
#		#include <math.h>
#		1 i\
#		#define __LAPACK_PRECISION_QUAD
#		s/extern integer len_trim__([^)]*);//g
#		s/extern [^ ]* ceiling_([^)]*);//g" |
#	sed -r -f $HOME/local/bin/ql.sed |
#	sed -r -e "$FUN" > ${f/.f/.c}
#	RM="$RM ${f/.f/.c}"
#done
#
#gcc -I$HOME/local/include `echo $@ | sed -r -e 's/([^ ]+)\.f/\1.c/g'` -I$HOME/local/include -L$HOME/local/lib -lf2c -lm
#if [ -z $DDD ]; then rm $RM; fi
#EOF
#
# Finally you can test the quad routines by making the "double" precision tests
# in BLAS and LAPACK
