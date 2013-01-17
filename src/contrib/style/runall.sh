#!/bin/bash

#Runs all checks defined in src/contrib/style/checks. Run from $PETSC_DIR.

for f in `ls src/contrib/style/checks/`
do
  grep "# Rule" $f
  `$f`
done

#echo "Violations of rule 'No space before CHKXXX();':"
#src/contrib/style/checks/chkxxx-space.sh | wc -l

#echo "Violations of rule 'The closing bracket for an if/for/do/while-block should be on its own line':"
#src/contrib/style/checks/closing-bracket.sh | wc -l

#echo "Violations of rule 'No C++ comments':"
#src/contrib/style/checks/cpp-comments.sh | wc -l

#echo "Violations of rule 'Indentation of '} else {':"
#src/contrib/style/checks/else-indentation.sh | wc -l

#echo "Violations of rule 'Opening { for a function on a new line':"
#src/contrib/style/checks/function-bracket.sh | wc -l

#echo "Violations of rule 'No #ifdef or #ifndef':"
#src/contrib/style/checks/ifdef.sh | wc -l

#echo "Violations of rule 'Indentation of if (...) {':"
#src/contrib/style/checks/if-indentation.sh | wc -l

#echo "Violations of rule 'No space after '(' and no space before ')'':"
#src/contrib/style/checks/parentheses-space.sh | wc -l

#echo "Violations of rule 'No blank line after PetscFunctionBegin;':"
#src/contrib/style/checks/PetscFunctionBegin.sh | wc -l

#echo "Violations of rule 'No blank line before PetscFunctionReturn;':"
#src/contrib/style/checks/PetscFunctionReturn.sh | wc -l

#echo "Violations of rule 'No function call inside PetscFunctionReturn':"
#src/contrib/style/checks/PetscFunctionReturnFunction.sh | wc -l

#echo "Violations of rule 'No tabs':"
#src/contrib/style/checks/tabs.sh | wc -l

