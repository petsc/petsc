
header="""#!/usr/bin/env bash
# This script was created by gmakegentest.py

# PATH for DLLs on windows
PATH=$PATH:@PETSC_LIB_DIR@
mpiexec='@MPIEXEC@'
exec=@EXEC@
testname='@TESTNAME@'
label=@LABEL@
runfiles=@LOCALRUNFILES@
petsc_dir=@PETSC_DIR@

. ${petsc_dir}/config/petsc_harness.sh
"""

datfilespath="@DATAFILESPATH@"
footer='petsc_testend "@TESTROOT@" '

todoline='printf "ok ${label} # TODO @TODOCOMMENT@\\n"'
skipline='printf "ok ${label} # SKIP @SKIPCOMMENT@\\n"'
mpitest='petsc_testrun "${mpiexec} -n @NSIZE@ ${exec} @ARGS@" @REDIRECT_FILE@ ${testname}.err "${label}" @FILTER@'
#Better labelling
#mpitest='petsc_testrun "${mpiexec} -n @NSIZE@ ${exec} @ARGS@" @REDIRECT_FILE@ ${testname}.err "${label}-@ARGS@" @FILTER@'
difftest='petsc_testrun "${petsc_dir}/bin/petscdiff @OUTPUT_FILE@ @REDIRECT_FILE@" diff-${testname}.out diff-${testname}.out diff-${label} ""'
filterdifftest='petsc_testrun "@FILTER_OUTPUT@ @OUTPUT_FILE@ | ${petsc_dir}/bin/petscdiff - @REDIRECT_FILE@" diff-${testname}.out diff-${testname}.out diff-${label} ""'
commandtest='petsc_testrun "@COMMAND@" @REDIRECT_FILE@ ${testname}.err cmd-${label} @FILTER@'
