#!/bin/bash

# Checks for compliance with 
# Rule: 'Lines with ierr = ... should end with CHKERRQ();'


# Steps:
# - find any line with a ierr =
# - ignore lines with CHKERR.*
# - ignore lines ending with ,
# - ignore ierr = PetscFinalize();


grep -n -H "ierr *=" "$@" \
 | grep -v "CHKERR" \
 | grep -v ",$" \
 | grep -v "ierr = PetscFinalize();" \
 | grep -v -F "*ierr"


