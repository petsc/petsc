dnl
dnl Special configure macros
dnl
define(AC_FORTRAN_NAMES_IN_C,[
AC_MSG_CHECKING(for Fortran external names)
# First, compile a Fortran program:
/bin/rm -f conff.f
cat > conff.f <<EOF
        subroutine d1chk()
        return
        end
EOF
if $FC -c conff.f >/dev/null 2>&1 ; then 
    :
else
    echo "Could not compile Fortran routine"
fi
# Now, build a C program and try to link with it
cat > conf.c <<EOF
main() {
d1chk_();
return 0;
}
EOF
if $CC -o conf conf.c conff.o >/dev/null 2>&1 ; then
    AC_DEFINE(HAVE_FORTRAN_UNDERSCORE)
    FORTRAN_NAMING="-DHAVE_FORTRAN_UNDERSCORE"
    /bin/rm -f conf conf.c conff.f conff.o conf.o
    AC_MSG_RESULT(trailing underscore)
else
    /bin/rm -f conf conf.c conf.o
    cat > conf.c <<EOF
main() {
d1chk();
return 0;
}
EOF
    if $CC -o conf conf.c conff.o >/dev/null 2>&1 ; then 
        AC_DEFINE(HAVE_FORTRAN_NOUNDERSCORE)
        FORTRAN_NAMING="-DHAVE_FORTRAN_NOUNDERSCORE"
        /bin/rm -f conf conf.c conff.f conff.o conf.o
        AC_MSG_RESULT(no underscore)
    else
        /bin/rm -f conf conf.c conf.o
        cat > conf.c <<EOF
main() {
D1CHK();
return 0;
}
EOF
        if $CC -o conf conf.c conff.o >/dev/null 2>&1 ; then 
            AC_DEFINE(HAVE_FORTRAN_CAPS)
            FORTRAN_NAMING="-DHAVE_FORTRAN_CAPS"
            /bin/rm -f conf conf.c conff.f conff.o conf.o
            AC_MSG_RESULT(uppercase)
        else
	    FORTRAN_NAMING=""
            AC_MSG_RESULT(unknown!)
        fi
    fi
fi
/bin/rm -f conf conf.c conff.f conff.o conf.o
#
AC_SUBST(FORTRAN_NAMING)])
dnl
dnl Check that a function prototype will work.  This is used to handle
dnl the various variants in such things as "select" and "connect" (!)
dnl The need to do this is one of the reasons that Window NT will win.
dnl
dnl AC_CHECK_PROTOTYPE(foo(int,...),action-if-corrent,action-if-fails)
define(AC_CHECK_PROTOTYPE,[
AC_TRY_COMPILE($1,return 1;,$2,$3)
])
dnl
dnl
dnl PAC_CHECK_COMPILER_OPTION(optionname,action-if-ok,action-if-fail)
dnl This should actually check that compiler doesn't complain about it either,
dnl by compiling the same program with two options, and diff'ing the output.
dnl
define([PAC_CHECK_COMPILER_OPTION],[
AC_MSG_CHECKING([that C compiler accepts option $1])
CFLAGSSAV="$CFLAGS"
CFLAGS="$1 $CFLAGS"
echo 'void f(){}' > conftest.c
if test -z "`${CC-cc} $CFLAGS -c conftest.c 2>&1`"; then
  AC_MSG_RESULT(yes)
  $2
else
  AC_MSG_RESULT(no)
  $3
fi
rm -f conftest*
CFLAGS="$CFLAGSSAV"
])
dnl
dnl PAC_CHECK_FC_COMPILER_OPTION is like PAC_CHECK_COMPILER_OPTION,
dnl except for Fortran 
define([PAC_CHECK_FC_COMPILER_OPTION],[
AC_MSG_CHECKING([that Fortran compiler accepts option $1])
FFLAGSSAV="$FFLAGS"
FFLAGS="$1 $FFLAGS"
cat >conftest.f <<EOF
        program main
        end
EOF
/bin/rm -f conftest1.out conftest2.out
if $FC $FFLAGS -c conftest.f > conftest1.out 2>&1 ; then
    if $FC $FFLAGSSAV -c conftest.f > conftest2.out 2>&1 ; then
        if diff conftest2.out conftest1.out ; then
            AC_MSG_RESULT(yes)
            $2
	else
            AC_MSG_RESULT(no)
            cat conftest2.out >> config.log
            $3
	fi
    else
        AC_MSG_RESULT(no)
        cat conftest2.out >> config.log
        $3
    fi
else
    AC_MSG_RESULT(no)
    cat conftest1.out >> config.log
    $3
fi
rm -f conftest*
FFLAGS="$FFLAGSSAV"
])
dnl PAC_MACRO_NAME_IN_MACRO([action if ok],[action if failed])
dnl
dnl Note that we can't put a pound sign into the msg_checking macro because
dnl it confuses autoconf
AC_DEFUN([PAC_MACRO_NAME_IN_MACRO],
[AC_REQUIRE([AC_PROG_CC])dnl
AC_CACHE_CHECK([that compiler allows recursive definitions],
ac_cv_prog_cpp_recursive,
[AC_TRY_COMPILE([
void a(i,j)int i,j;{}
#define a(b) a(b,__LINE__)],[
a(0);return 0;],ac_cv_prog_cpp_recursive="yes",ac_cv_prog_cpp_recursive="no")])
if test $ac_cv_prog_cpp_recursive = "yes" ; then
    ifelse([$1],,:,[$1])
else
    ifelse([$2],,:,[$2])
fi
])
