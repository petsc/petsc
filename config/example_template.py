
header="""#!/bin/bash
# This script was created by gmakegentest.py

mpiexec='@MPIEXEC@'
exec=@EXEC@
testname='@TESTNAME@'
label=@LABEL@
runfiles=@LOCALRUNFILES@

. @TESTROOT@/petsc_harness.sh
"""

datfilepath="@DATAFILEPATH@"
footer='petsc_testend "@TESTSROOT@" '

todoline='printf "ok ${label} # TODO @TODOCOMMENT@\\n"'
skipline='printf "ok ${label} # SKIP @SKIPCOMMENT@\\n"'
mpitest='petsc_testrun "${mpiexec} -n @NSIZE@ ${exec} @ARGS@" @REDIRECT_FILE@ ${testname}.err "${label}" @FILTER@'
#Better labelling
#mpitest='petsc_testrun "${mpiexec} -n @NSIZE@ ${exec} @ARGS@" @REDIRECT_FILE@ ${testname}.err "${label}-@ARGS@" @FILTER@'
difftest='petsc_testrun "${PETSC_DIR}/bin/petscdiff @REDIRECT_FILE@ @OUTPUT_FILE@" diff-${testname}.out diff-${testname}.out diff-${label} ""'
filterdifftest='petsc_testrun "@FILTER_OUTPUT@ @OUTPUT_FILE@ | diff @REDIRECT_FILE@ -" diff-${testname}.out diff-${testname}.out diff-${label} ""'
commandtest='petsc_testrun "@COMMAND@" @REDIRECT_FILE@ ${testname}.err cmd-${label} @FILTER@'
