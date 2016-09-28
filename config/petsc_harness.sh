

scriptname=`basename $0`
rundir=${scriptname%.sh}

if test "$PWD"!=`dirname $0`; then
  cd `dirname $0`
fi
mkdir -p ${rundir}
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
  -e <args> ......... Add extra arguments to default
  -h ................ help: print this message
  -n <integer> ...... Override the number of processors to use
  -o <output file> .. Override default output file to diff with
  -t <testname> ..... Override test name
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
while getopts "a:c:e:hn:o:t:v" arg
do
  case $arg in
    a ) args=$OPTARG     ;;  
    c ) cleanup=true     ;;  
    e ) extra_args=$OPTARG     ;;  
    h ) print_usage; exit ;;  
    n ) nsize=$OPTARG     ;;  
    o ) output_file=$OPTARG     ;;  
    t ) testname=$OPTARG     ;;  
    v ) verbose=true     ;;  
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

  if test -z "$filter"; then
    cmd="$1 > $2 2> $3"
  else
    cmd="$1 | $filter > $2 2> $3"
  fi
  if "${verbose}"; then 
    printf "${cmd}\n"
  fi
  eval $cmd
  if test $? == 0; then
      printf "ok $tlabel $cmd\n"
      let success=$success+1
  else
      printf "not ok $tlabel\n"
      awk '{print "#\t" $0}' < $3
      let failed=$failed+1
      failures="$failures $tlabel"
  fi
  let total=$success+$failed
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
