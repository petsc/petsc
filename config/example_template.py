
header="""#!/usr/bin/env bash
# This script was created by gmakegentest.py

# PATH for DLLs on windows
PATH="$PATH:@PETSC_LIB_DIR@"
mpiexec='@MPIEXEC@'
exec='@EXEC@'
testname='@TESTNAME@'
label='@LABEL@'
runfiles='@LOCALRUNFILES@'
wPETSC_DIR='@WPETSC_DIR@'
petsc_dir='@PETSC_DIR@'

. "${petsc_dir}/config/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${petsc_dir}/bin/petscdiff ${diff_flags}"

"""

datfilespath="@DATAFILESPATH@"
footer='petsc_testend "@TESTROOT@" '

todoline='printf "ok ${label} # TODO @TODOCOMMENT@\\n"'
skipline='printf "ok ${label} # SKIP @SKIPCOMMENT@\\n"'
mpitest='petsc_testrun "${mpiexec} -n @NSIZE@ ${exec} ${args}" @REDIRECT_FILE@ ${testname}.err "${label}" @FILTER@'
subargsline='subargs="@SUBARGS@"'
argsline='args="@ARGS@"'
mpisubtest='petsc_testrun "${mpiexec} -n @NSIZE@ ${exec} ${args} ${subargs}" @REDIRECT_FILE@ ${testname}.err "${label}" @FILTER@'
#Better labelling
#mpitest='petsc_testrun "${mpiexec} -n @NSIZE@ ${exec} ${args}" @REDIRECT_FILE@ ${testname}.err "${label}-${args}" @FILTER@'
difftest='petsc_testrun "${diff_exe} @OUTPUT_FILE@ @REDIRECT_FILE@" diff-${testname}.out diff-${testname}.out diff-${label} ""'
filterdifftest='petsc_testrun "@FILTER_OUTPUT@ @OUTPUT_FILE@ | ${diff_exe} - @REDIRECT_FILE@" diff-${testname}.out diff-${testname}.out diff-${label} ""'
commandtest='petsc_testrun "@COMMAND@" @REDIRECT_FILE@ ${testname}.err cmd-${label} @FILTER@'
