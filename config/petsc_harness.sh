

scriptname=`basename $0`
rundir=${scriptname%.sh}
TIMEOUT=60

if test "$PWD"!=`dirname $0`; then
  cd `dirname $0`
  abspath_scriptdir=$PWD
fi
if test -d "${rundir}" && test -n "${rundir}"; then
  rm -f ${rundir}/*.tmp ${rundir}/*.err ${rundir}/*.out
fi
mkdir -p ${rundir}
if test -n "${runfiles}"; then
  for runfile in ${runfiles}; do
      subdir=`dirname ${runfile}`
      mkdir -p ${rundir}/${subdir}
      cp -r ${runfile} ${rundir}/${subdir}
  done
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
  -c ................ Cleanup (remove generated files)
  -C ................ Compile
  -d ................ Launch in debugger
  -e <args> ......... Add extra arguments to default
  -f ................ force attempt to run test that would otherwise be skipped
  -h ................ help: print this message
  -n <integer> ...... Override the number of processors to use
  -j ................ Pass -j to petscdiff (just use diff)
  -J <arg> .......... Pass -J to petscdiff (just use diff with arg)
  -m ................ Update results using petscdiff
  -M ................ Update alt files using petscdiff
  -o <arg> .......... Output format: 'interactive', 'err_only'
  -p ................ Print command:  Print first command and exit
  -t ................ Override the default timeout (default=$TIMEOUT sec)
  -U ................ run cUda-memcheck
  -V ................ run Valgrind
  -v ................ Verbose: Print commands
EOF

  if declare -f extrausage > /dev/null; then extrausage; fi
  exit $1
}
###
##  Arguments for overriding things
#
output_fmt="interactive"
verbose=false
cleanup=false
compile=false
debugger=false
printcmd=false
mpiexec_function=false
force=false
diff_flags=""
while getopts "a:cCde:fhjJ:mMn:o:pt:UvV" arg
do
  case $arg in
    a ) args="$OPTARG"       ;;  
    c ) cleanup=true         ;;  
    C ) compile=true         ;;  
    d ) debugger=true        ;;  
    e ) extra_args="$OPTARG" ;;  
    f ) force=true           ;;
    h ) print_usage; exit    ;;  
    n ) nsize="$OPTARG"      ;;  
    j ) diff_flags=$diff_flags" -j"      ;;  
    J ) diff_flags=$diff_flags" -J $OPTARG" ;;  
    m ) diff_flags=$diff_flags" -m"      ;;  
    M ) diff_flags=$diff_flags" -M"      ;;  
    o ) output_fmt=$OPTARG   ;;  
    p ) printcmd=true        ;;
    t ) TIMEOUT=$OPTARG      ;;  
    U ) mpiexec="petsc_mpiexec_cudamemcheck $mpiexec" 
        mpiexec_function=true
        ;;  
    V ) mpiexec="petsc_mpiexec_valgrind $mpiexec"
        mpiexec_function=true
        ;;  
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
export MPIEXEC_TIMEOUT=$((TIMEOUT*timeoutfactor))
STARTTIME=`date +%s`

if test -n "$extra_args"; then
  args="$args $extra_args"
fi
if $debugger; then
  args="-start_in_debugger $args"
fi
if test -n "$filter"; then
  diff_flags=$diff_flags" -F \$'$filter'"
fi
if test -n "$filter_output"; then
  diff_flags=$diff_flags" -f \$'$filter_output'"
fi


# Init
success=0; failed=0; failures=""; rmfiles=""
total=0
todo=-1; skip=-1
job_level=0

if $compile; then
   curexec=`basename ${exec}`
   fullexec=${abspath_scriptdir}/${curexec}
   maketarget=`echo ${fullexec} | sed "s#${petsc_dir}/*##"`
   (cd $petsc_dir && make -f gmakefile.test ${maketarget})
fi

###
##   Rest of code is functions
#
function petsc_report_tapoutput() {
  notornot=$1
  test_label=$2
  comment=$3
  if test -n "$comment"; then
    comment=" # ${comment}"
  fi

  tap_message="${notornot} ok ${test_label}${comment}"

  # Log messages
  printf "${tap_message}\n" >> ${testlogtapfile}
  
  if test ${output_fmt} == "err_only"; then
     if test -n "${notornot}"; then 
        printf "${tap_message}\n" | tee -a ${testlogerrfile}
     fi
  else 
     printf "${tap_message}\n"
  fi
}

function printcmd() {
  # Print command that can be run from PETSC_DIR
  cmd="$1"
  basedir=`dirname ${PWD} | sed "s#${petsc_dir}/##"`
  modcmd=`echo ${cmd} | sed -e "s#\.\.#${basedir}#" | sed s#\>.*## | sed s#\%#\%\%#`
  if $mpiexec_function; then
     # Have to expand valgrind/cudamemchk
     modcmd=`eval "$modcmd"`
  fi
  printf "${modcmd}\n" 
  exit
}

function petsc_testrun() {
  # First arg = Basic command
  # Second arg = stdout file
  # Third arg = stderr file
  # Fourth arg = label for reporting
  rmfiles="${rmfiles} $2 $3"
  tlabel=$4
  error=$5
  cmd="$1 > $2 2> $3"
  if test -n "$error"; then
    cmd="$1 2>&1 | cat > $2"
  fi
  echo "$cmd" > ${tlabel}.sh; chmod 755 ${tlabel}.sh
  if $printcmd; then
     printcmd "$cmd"
  fi

  eval "{ time -p $cmd ; } 2>> timing.out"
  cmd_res=$?
  #  If it is a lack of GPU resources or MPI failure (Intel) then try once more
  #  See: src/sys/error/err.c
  if [ $cmd_res -eq 96 -o $cmd_res -eq 97 -o $cmd_res -eq 98 ]; then
    printf "# retrying ${tlabel}\n" | tee -a ${testlogerrfile}
    sleep 3
    eval "{ time -p $cmd ; } 2>> timing.out"
    cmd_res=$?
  fi
  touch "$2" "$3"
  # It appears current MPICH and OpenMPI just shut down the job executation and do not return an error code to the executable
  # ETIMEDOUT=110 was used by OpenMPI 3.0.  MPICH used 255
  # Earlier OpenMPI versions returned 1 and the error string
  if [ $cmd_res -eq 110 -o $cmd_res -eq 255 ] || \
        fgrep -q -s 'APPLICATION TIMED OUT' "$2" "$3" || \
        fgrep -q -s MPIEXEC_TIMEOUT "$2" "$3" || \
        fgrep -q -s 'APPLICATION TERMINATED WITH THE EXIT STRING: job ending due to timeout' "$2" "$3" || \
        grep -q -s "Timeout after [0-9]* seconds. Terminating job" "$2" "$3"; then
    timed_out=1
    # If timed out, then ensure non-zero error code
    if [ $cmd_res -eq 0 ]; then
      cmd_res=1
    fi
  fi

  # Report errors
  comment=""
  if test $cmd_res == 0; then
     if "${verbose}"; then
        comment="${cmd}"
     fi
    petsc_report_tapoutput "" "$tlabel" "$comment"
    let success=$success+1
  else
    if [ -n "$timed_out" ]; then
      comment="Exceeded timeout limit of $MPIEXEC_TIMEOUT s"
    else
      comment="Error code: ${cmd_res}"
    fi
    petsc_report_tapoutput "not" "$tlabel" "$comment"

    # Report errors in detail
    if [ -z "$timed_out" ]; then
      # We've had tests fail but stderr->stdout, as well as having
      # mpi_abort go to stderr which throws this test off.  Show both
      # with stdout first
      awk '{print "#\t" $0}' < $2 | tee -a ${testlogerrfile}
      # if statement is for diff tests
      if test "$2" != "$3"; then
        awk '{print "#\t" $0}' < $3 | tee -a ${testlogerrfile}
      fi
    fi
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
  ENDTIME=`date +%s`
  timing=`touch timing.out && egrep '(user|sys)' timing.out | awk '{if( sum1 == "" || $2 > sum1 ) { sum1=sprintf("%.2f",$2) } ; sum2 += sprintf("%.2f",$2)} END {printf "%.2f %.2f\n",sum1,sum2}'`
  printf "time $timing\n" >> $logfile
  if $cleanup; then
    echo "Cleaning up"
    /bin/rm -f $rmfiles
  fi
}

function petsc_mpiexec_cudamemcheck() {
  _mpiexec=$1;shift
  npopt=$1;shift
  np=$1;shift

  cudamemchk="cuda-memcheck"

  $_mpiexec $npopt $np $cudamemchk $*
}

function petsc_mpiexec_valgrind() {
  # some systems set $1 to be the function name
  if [[ $1 == 'petsc_mpiexec_valgrind' ]]; then
    shift
  fi
  _mpiexec=$1;shift
  npopt=$1;shift
  np=$1;shift

  valgrind="valgrind -q --tool=memcheck --leak-check=yes --num-callers=20 --track-origins=yes --suppressions=$petsc_bindir/maint/petsc-val.supp --error-exitcode=10"

  if $printcmd; then
     echo $_mpiexec $npopt $np $valgrind "$@"
  else
     $_mpiexec $npopt $np $valgrind "$@"
  fi
}
export LC_ALL=C
