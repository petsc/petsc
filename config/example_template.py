
header="""#!/bin/sh
# This script was created by gmakegentest.py

mpiexec='@MPIEXEC@'
exec=@EXEC@
testname='@TESTNAME@'
label=@LABEL@

. @TESTROOT@/petsc_harness.sh
"""

datfilepath="@DATAFILEPATH@"
footer='petsc_testend "@TESTSROOT@" '

todoline='printf "ok ${label} # TODO @TODOCOMMENT@\\n"'
skipline='printf "not ok ${label} # SKIP @SKIPCOMMENT@\\n"'
mpitest='petsc_testrun "${mpiexec} -n @NSIZE@ ${exec} @ARGS@" @REDIRECT_FILE@ ${testname}.err "${label}" @FILTER@'
#Better labelling
#mpitest='petsc_testrun "${mpiexec} -n @NSIZE@ ${exec} @ARGS@" @REDIRECT_FILE@ ${testname}.err "${label}-@ARGS@" @FILTER@'
difftest='petsc_testrun "diff @REDIRECT_FILE@ @OUTPUT_FILE@" diff-${testname}.out diff-${testname}.out diff-${label} ""'
commandtest='petsc_testrun "@COMMAND@" @REDIRECT_FILE@ ${testname}.err cmd-${label} @FILTER@'
