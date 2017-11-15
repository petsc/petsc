#!/bin/bash

find . -name *c-example.html -type f | xargs sed -i '/ierr;/d'
find . -name *c-example.html -type f | xargs sed -i '/__FUNCT__/d'
find . -name *c-example.html -type f | xargs sed -i '/PetscFunctionBeginUser;/d'
find . -name *c-example.html -type f | xargs sed -i 's/PetscFunctionReturn/return/g'

