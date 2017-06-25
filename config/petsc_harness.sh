

scriptname=`basename $0`
rundir=${scriptname%.sh}

if test "$PWD"!=`dirname $0`; then
  cd `dirname $0`
fi
mkdir -p ${rundir}
if test -n "${runfiles}"; then 
  cp ${runfiles} ${rundir}
fi
cd ${rundir}

#
# Method to print out general and script specific options
#
print_usage() {

cat >&2 <<EOF
Usage: $0 [options]

OPTIONS
  -a <args> ......... Override default arguments
  -c <cleanup> ...... Cleanup (remove generated files)
  -d ................ Launch in debugger
  -e <args> ......... Add extra arguments to default
  -f ................ force attempt to run test that would otherwise be skipped
  -h ................ help: print this message
  -n <integer> ...... Override the number of processors to use
  -j ................ Pass -j to petscdiff (just use diff)
  -J <arg> .......... Pass -J to petscdiff (just use diff with arg)
  -m ................ Update results using petscdiff
  -V ................ run Valgrind
  -v ................ Verbose: Print commands
EOF

  if declare -f extrausage > /dev/null; then extrausage; fi
  exit $1
}
###
##  Arguments for overriding things
#
verbose=false
cleanup=false
debugger=false
force=false
diff_flags=""
while getopts "a:cde:fhjJ:mn:vV" arg
do
  case $arg in
    a ) args="$OPTARG"       ;;  
    c ) cleanup=true         ;;  
    d ) debugger=true        ;;  
    e ) extra_args="$OPTARG" ;;  
    f ) force=true           ;;
    h ) print_usage; exit    ;;  
    n ) nsize="$OPTARG"      ;;  
    j ) diff_flags="-j"      ;;  
    J ) diff_flags="-J $OPTARG" ;;  
    m ) diff_flags="-m"      ;;  
    V ) mpiexec="petsc_mpiexec_valgrind $mpiexec" ;;  
    v ) verbose=true         ;;  
    *)  # To take care of any extra args
      if test -n "$OPTARG"; then
        eval $arg=\"$OPTARG\"
      else
        eval $arg=found
      fi
      ;;
  esac
done
shift $(( $OPTIND - 1 ))

if test -n "$extra_args"; then
  args="$args $extra_args"
fi
if $debugger; then
  args="-start_in_debugger $args"
fi


# Init
success=0; failed=0; failures=""; rmfiles=""
total=0
todo=-1; skip=-1

function petsc_testrun() {
  # First arg = Basic command
  # Second arg = stdout file
  # Third arg = stderr file
  # Fourth arg = label for reporting
  # Fifth arg = Filter
  rmfiles="${rmfiles} $2 $3"
  tlabel=$4
  filter=$5
  # Determining whether this test passes or fails is tricky because of filters
  # and eval.  Use sum of all parts of a potential pipe to determine status. See:
  #  https://stackoverflow.com/questions/24734850/how-to-get-the-exit-status-of-the-first-command-in-a-pipe
  #  http://www.unix.com/shell-programming-and-scripting/128869-creating-run-script-getting-pipestatus-eval.html

  if test -z "$filter"; then
    cmd="$1 > $2 2> $3"
  else
    cmd="$1 2> $3 | $filter > $2 2>> $3"
  fi
  echo $cmd > ${tlabel}.sh; chmod 755 ${tlabel}.sh
  eval "$cmd; typeset -a cmd_errstat=(\${PIPESTATUS[@]})"
  let cmd_res=0
  for i in ${cmd_errstat[@]}; do let cmd_res+=$i; done

  if test $cmd_res == 0; then
    if "${verbose}"; then
     printf "ok $tlabel $cmd\n"
    else
     printf "ok $tlabel\n"
    fi
    let success=$success+1
  else
    if "${verbose}"; then 
      printf "not ok $tlabel $cmd\n"
    else
      printf "not ok $tlabel\n"
    fi
    awk '{print "#\t" $0}' < $3
    let failed=$failed+1
    failures="$failures $tlabel"
  fi
  let total=$success+$failed
  return $cmd_res
}

function petsc_testend() {
  logfile=$1/counts/${label}.counts
  logdir=`dirname $logfile`
  if ! test -d "$logdir"; then
    mkdir -p $logdir
  fi
  if ! test -e "$logfile"; then
    touch $logfile
  fi
  printf "total $total\n" > $logfile
  printf "success $success\n" >> $logfile
  printf "failed $failed\n" >> $logfile
  printf "failures $failures\n" >> $logfile
  if test ${todo} -gt 0; then
    printf "todo $todo\n" >> $logfile
  fi
  if test ${skip} -gt 0; then
    printf "skip $skip\n" >> $logfile
  fi
  if $cleanup; then
    echo "Cleaning up"
    /bin/rm -f $rmfiles
  fi
}

function petsc_mpiexec_valgrind() {
  mpiexec=$1;shift
  npopt=$1;shift
  np=$1;shift

  valgrind="valgrind -q --tool=memcheck --leak-check=yes --num-callers=20 --track-origins=yes --suppressions=$petsc_dir/bin/maint/petsc-val.supp"
  $mpiexec $npopt $np $valgrind $*
}
export LC_ALL=C
