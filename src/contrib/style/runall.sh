#!/bin/bash

#Runs all checks defined in src/contrib/style/checks. Run from $PETSC_DIR.

for f in `ls src/contrib/style/checks/*-all.sh`
do
  grep "# Rule" $f
  echo `$f | wc -l`
done

