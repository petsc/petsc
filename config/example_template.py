
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
petsc_dir='@PETSC_DIR@'
petsc_arch='@PETSC_ARCH@'
# Must be consistent with gmakefile.test
testlogfile=@TESTROOT@/examples_${petsc_arch}.log
config_dir='@CONFIG_DIR@'
petsc_bindir='@PETSC_BINDIR@'
@DATAFILESPATH_LINE@
args='@ARGS@'
timeoutfactor=@TIMEOUTFACTOR@

mpiexec=${PETSCMPIEXEC:-"@MPIEXEC@"}
diffexec=${PETSCDIFF:-"${petsc_bindir}/petscdiff"}

. "${config_dir}/petsc_harness.sh"

# The diff flags come from script arguments
diff_exe="${diffexec} ${diff_flags}"
mpiexec="${mpiexec} ${mpiexec_flags}"
nsize=${nsize:-@NSIZE@}
"""

footer='petsc_testend "@TESTROOT@" '

todoline='printf "ok ${label} # TODO @TODOCOMMENT@\\n"'
skipline='printf "ok ${label} # SKIP @SKIPCOMMENT@\\n"'
mpitest='petsc_testrun "${mpiexec} -n ${nsize} ${exec} ${args} @SUBARGS@" @REDIRECT_FILE@ ${testname}.err "${label}@LABEL_SUFFIX@" @FILTER@'
difftest='petsc_testrun "${diff_exe} @OUTPUT_FILE@ @REDIRECT_FILE@" diff-${testname}.out diff-${testname}.out diff-${label}@LABEL_SUFFIX@ ""'
filterdifftest='petsc_testrun "@FILTER_OUTPUT@ @OUTPUT_FILE@ | ${diff_exe} - @REDIRECT_FILE@" diff-${testname}.out diff-${testname}.out diff-${label}@LABEL_SUFFIX@ ""'
commandtest='petsc_testrun "@COMMAND@" @REDIRECT_FILE@ ${testname}.err cmd-${label}@LABEL_SUFFIX@ @FILTER@'
