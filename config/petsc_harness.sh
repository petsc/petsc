

scriptname=`basename $0`
rundir=${scriptname%.sh}
TIMEOUT=60

if test "$PWD"!=`dirname $0`; then
  cd `dirname $0`
fi
if test -d "${rundir}" && test -n "${rundir}"; then
  rm ${rundir}/*.tmp ${rundir}/*.err ${rundir}/*.out
fi
mkdir -p ${rundir}
if test -d "${runfiles}"; then
  cp -r ${runfiles} ${rundir}
elif test -n "${runfiles}"; then
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
  -M ................ Update alt files using petscdiff
  -t ................ Override the default timeout (default=$TIMEOUT sec)
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
while getopts "a:cde:fhjJ:mMn:t:vV" arg
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
    M ) diff_flags="-M"      ;;  
    t ) TIMEOUT=$OPTARG      ;;  
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

# Individual tests can extend the default
TIMEOUT=$((TIMEOUT*timeoutfactor))

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
job_level=0

function petsc_testrun() {
  # First arg = Basic command
  # Second arg = stdout file
  # Third arg = stderr file
  # Fourth arg = label for reporting
  # Fifth arg = Filter
  rmfiles="${rmfiles} $2 $3"
  tlabel=$4
  filter=$5
  job_control=true
  cmd="$1 > $2 2> $3"
  if test -n "$filter"; then
    if test "${filter:0:6}"=="Error:"; then
      job_control=false      # redirection error method causes job control probs
      filter=${filter##Error:}
      cmd="$1 2>&1 | cat > $2"
    fi
  fi
  echo "$cmd" > ${tlabel}.sh; chmod 755 ${tlabel}.sh

  if $job_control; then
    # The action:
    eval "($cmd) &"
    pid=$!
    # Put a watcher process in that will kill a job that exceeds limit
    $config_dir/watchtime.sh $pid $TIMEOUT &
    watcher=$!

    # See if the job we want finishes
    wait $pid 2> /dev/null
    cmd_res=$?
    if ps -p $watcher > /dev/null; then
      # Keep processes tidy by killing watcher
      pkill -13 -P $watcher
      wait $watcher 2>/dev/null  # Wait used here to capture the kill message
    else
      # Timeout
      cmd_res=1
      echo "Exceeded timeout limit of $TIMEOUT s" > $3
    fi
  else
    # The action -- assume no timeout needed
    eval "$cmd"
    # We are testing error codes so just make it pass
    cmd_res=0
  fi

  # Handle filters separately and assume no timeout check needed
  if test -n "$filter"; then
    cmd="cat $2 | $filter > $2.tmp 2>> $3 && mv $2.tmp $2"
    echo "$cmd" >> ${tlabel}.sh
    eval "$cmd"
  fi

  # Report errors
  if test $cmd_res == 0; then
    if "${verbose}"; then
     printf "ok $tlabel $cmd\n" | tee -a ${testlogfile}
    else
     printf "ok $tlabel\n" | tee -a ${testlogfile}
    fi
    let success=$success+1
  else
    if "${verbose}"; then 
      printf "not ok $tlabel $cmd\n" | tee -a ${testlogfile}
    else
      printf "not ok $tlabel\n" | tee -a ${testlogfile}
    fi
    awk '{print "#\t" $0}' < $3 | tee -a ${testlogfile}
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

  valgrind="valgrind -q --tool=memcheck --leak-check=yes --num-callers=20 --track-origins=yes --suppressions=$petsc_bindir/maint/petsc-val.supp"

  $mpiexec $npopt $np $valgrind $*
}
export LC_ALL=C
