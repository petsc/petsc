#!/bin/bash
# Runs selected checkers from checks/-directory. Only checks for which the source tree is clean are considered.
#
# All directories/files to be checked are passed as parameter(s). Automatic FORTRAN-stubs are ALWAYS ignored.
#
# Examples:
#
#  - Check everything in src/ts:
#    $> stylecheck path/to/src/ts/
#
#  - Check only src/ts/examples/tutorials/ex1.c:
#    $> stylecheck path/to/src/ts/examples/tutorials/ex1.c
#


checkers=("assert.sh" "bool-condition.sh" "chkxxx-space.sh" "closing-bracket.sh" "cpp-comments.sh" "else-indentation.sh" \
          "funct.sh" "function-bracket.sh" "if0.sh" "ifdef.sh" "if-indentation.sh" "impl-include.sh" \
          "mpi3-removed.sh" "null.sh" "parentheses-space.sh" "parenthesis-curly-brace.sh" "PetscFunctionReturnFunction.sh" \
          "space-in-cast.sh" "tabs.sh")

script_args="$@"
echo $script_args

for check in "${checkers[@]}";
do
  ruledesc=`grep "Rule" $(dirname $0)/checks/${check}`

  # Status message
  echo -e "\\E[32m -- Checking $ruledesc\\E[m "

  # Find all files below the provided location
  find $script_args -name "*.[ch]" | grep -v "ftn-auto" | xargs $(dirname $0)/checks/${check}

done

