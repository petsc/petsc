#!/bin/sh
function usage () {
 	echo "$0: usage: <mode> <libpath> <language> [dynamic <scope> <resolution>] <babelComponentName space separated list>" > /dev/stderr
	echo "This script generates to stdout .scl or .cca files which are installed next to the" > /dev/stderr
	echo "library in the same directory under some name xxxxxx.depl.cca. or libpath.scl." > /dev/stderr
	echo "The script creates scl and cca info in cca mode or just babel info in scl mode." > /dev/stderr
	echo " Arguments: mode is cca or scl." > /dev/stderr
	echo "            libpath is the full path of the .la, .o, .so, or .a file" > /dev/stderr
	echo "            language is the implementation language (c c++ f77 f90 python)" > /dev/stderr
	echo "            babelComponentName is a space separated list of the full dot-qualified" > /dev/stderr
        echo "                           babel.class.name of the component. The class implementing" > /dev/stderr
        echo "                           gov.cca.Component should appear first for CCA files." > /dev/stderr
	echo " Optional arguments (if library is dynamically loadable)" > /dev/stderr
	echo "            dynamic -- required literal. just put it there." > /dev/stderr
	echo "            scope is global or private" > /dev/stderr
	echo "            resolution is now or lazy" > /dev/stderr
	echo " If optional arguments are not given, static is assumed." > /dev/stderr
	echo "e.g.: $0 scl /somewhere/lib/libComponent3.a Comp3 c++ test3.Component1" > /dev/stderr
	echo "e.g.: $0 cca /somewhere/lib/libComponent2.so Comp2 python dynamic global lazy test2.Component1" > /dev/stderr
	echo "e.g.: $0 scl /somewhere/lib/libComponent1.la Comp1 c dynamic private now test1.Component1" > /dev/stderr
	echo "e.g.: $0 cca /somewhere/lib/libComponent0.la test0.Component1 f77 test0.Component1" > /dev/stderr
}
if test $# -lt 5; then
	usage
	exit 1;
fi
execstring="$0 $*"
dstring=`date`
pstring=`pwd`/
mode=$1
libpath=$2
language=$4
classes=""
dynamic="static"
scope="global"
resolution="now"
shift 4
if test $# -ge 1 && test "dynamic" = "$1"; then
   dynamic="dynamic"
   shift
   if test $# -ge 1 && test $1 = "global" -o $1 = "private"; then
     scope=$1
     shift
     if test $# -ge 1 && test $1 = "now" -o $1 = "lazy"; then
       resolution=$1
       shift
     fi
   fi
fi
while test $# -ge 1; do
  classes="$classes $1"
  shift
done
#echo $mode
#echo $libpath
#echo $className
#echo $language
#echo $dynamic
#echo $scope
#echo $resolution
#exit 0
if [ "$language" == python ] ; then
  pythonImplLine="<class name=\"$className\" desc=\"python/impl\" />"
fi;  
if test "x$mode" = "xscl"; then
cat << __EOF1
<?xml version="1.0"?> 
<!-- # generated scl index. -->
<!-- date=$dstring -->
<!-- builder=$USER@$HOST -->
<!-- $execstring -->
<scl>
  <library uri="$libpath" 
	scope="$scope" 
	resolution="$resolution" > 
__EOF1
for className in $classes; do
    echo "    <class name=\"$className\" desc=\"ior/impl\" />"
    if [ "$language" == python ] ; then
      echo "    <class name=\"$className\" desc=\"python/impl\" />"
    fi
done
cat << __EOF2
  </library>
</scl>
__EOF2

exit 0
fi

if test "x$mode" = "xcca"; then
cat << __EOF3
<?xml version="1.0"?> 
<libInfo>
<!-- # generated component index. -->
<!-- date=$dstring -->
<!-- builder=$USER@$HOST -->
<!-- $0 $* -->
<scl>
  <library uri="$libpath" 
	scope="$scope" 
	resolution="$resolution" > 
__EOF3
for className in $classes; do
    echo "    <class name=\"$className\" desc=\"ior/impl\" />"
    if [ "$language" == python ] ; then
      echo "    <class name=\"$className\" desc=\"python/impl\" />"
    fi
done
className=`echo $classes | sed 's/ .*//'`
cat << __EOF4
  </library>
</scl>
__EOF4
for className in $classes; do
echo "<componentDeployment "
echo "  name=\"$className\" "
echo "  paletteClassAlias=\"$className\" "
echo "> "
echo "    <environment> "
echo "        <ccaSpec binding=\"babel\" /> "
echo "        <library loading=\"$dynamic\" />"
echo "    </environment>"
echo "</componentDeployment>"
done
cat << __EOF5
</libInfo>
__EOF5
exit 0
fi

echo "$0: Unrecognized mode" > /dev/stderr
usage
exit 1
