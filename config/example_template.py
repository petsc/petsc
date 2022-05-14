
header="""#!/usr/bin/env bash
# This script was created by gmakegentest.py

@COMMENTS@

# PATH for DLLs on windows
PATH="$PATH":@PETSC_LIB_DIR@
exec='@EXEC@'
testname='@TESTNAME@'
label='@LABEL@'
runfiles='@LOCALRUNFILES@'
wPETSC_DIR='@WPETSC_DIR@'
@PKG_NAME@_dir='@PKG_DIR@'
@PKG_NAME@_arch='@PKG_ARCH@'
# Must be consistent with gmakefile.test
testlogtapfile=@TESTROOT@/test_${@PKG_NAME@_arch}_tap.log
testlogerrfile=@TESTROOT@/test_${@PKG_NAME@_arch}_err.log
config_dir='@CONFIG_DIR@'
filter='@FILTER@'
filter_output='@FILTER_OUTPUT@'
petsc_bindir='@PETSC_BINDIR@'
@DATAFILESPATH_LINE@
args='@ARGS@'
diff_args='@DIFF_ARGS@'
timeoutfactor=@TIMEOUTFACTOR@
export PETSC_OPTIONS="${PETSC_OPTIONS} -check_pointer_intensity 0 -error_output_stdout -malloc_dump @PETSC_TEST_OPTIONS@"

mpiexec=${PETSCMPIEXEC:-"@MPIEXEC@"}
diffexec=${PETSCDIFF:-"${petsc_bindir}/petscdiff"}

. "${config_dir}/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags} ${diff_args}"
mpiexec="${mpiexec} ${mpiexec_flags}"
"""

footer='petsc_testend "@TESTROOT@" '

todoline='petsc_report_tapoutput "" "${label}" "TODO @TODOCOMMENT@"'
skipline='petsc_report_tapoutput "" "${label}" "SKIP @SKIPCOMMENT@"'
mpitest='petsc_testrun "${mpiexec} -n ${insize} ${exec} ${args} @SUBARGS@" @REDIRECT_FILE@ ${testname}.err "${label}@LABEL_SUFFIX@" @ERROR@'
difftest='petsc_testrun "${diff_exe} @OUTPUT_FILE@ @REDIRECT_FILE@" diff-${testname}.out diff-${testname}.out diff-${label}@LABEL_SUFFIX@ ""'
commandtest='petsc_testrun "@COMMAND@" @REDIRECT_FILE@ ${testname}.err cmd-${label}@LABEL_SUFFIX@ @ERROR@'
