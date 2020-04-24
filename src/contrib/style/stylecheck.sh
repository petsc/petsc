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
#  - Check only src/ts/tutorials/ex1.c:
#    $> stylecheck path/to/src/ts/tutorials/ex1.c
#

usage() {
    cat >&2 <<EOF
usage: $0 [options] PATH ...

Runs style checkers in specified directories.
Options:
    -h    --help        shows this message
    --skip-tests        skip tests
    --skip-examples     skips tests and tutorials
EOF
}


while true; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --skip-examples)
            echo 'Skipping examples'
            exclude="$exclude"' ! -path *examples*'
            shift
            ;;
        --skip-tests)
            echo 'Skipping tests'
            exclude="$exclude"' ! -path *tests*'
            shift
            ;;
        -*)
            echo "Error: unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)                      # No more options
            break;
    esac
done

checkers=("assert.sh" "bool-condition.sh" "chkxxx-space.sh" "closing-bracket.sh" "cpp-comments.sh" "else-indentation.sh" \
          "funct.sh" "function-bracket.sh" "if0.sh" "ifdef.sh" "if-indentation.sh" "impl-include.sh" \
          "mpi3-removed.sh" "null.sh" "parentheses-space.sh" "parenthesis-curly-brace.sh" "PetscFunctionReturnFunction.sh" \
          "space-in-cast.sh" "tabs.sh")

script_args="$@"
#echo $script_args

for check in "${checkers[@]}";
do
  ruledesc=`grep "Rule" $(dirname $0)/checks/${check}`

  # Status message
  printf "\033[32m -- Checking $ruledesc\033[m \n"

  # Find all files below the provided location
  find $script_args -name "*.[ch]" $exclude ! -path '*ftn-auto*' | xargs $(dirname $0)/checks/${check}

done

