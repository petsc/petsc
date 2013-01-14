#!/bin/bash

##### The following commands do the following:
# Remove 'ierr ='
# Remove 'CHKERRQ(ierr);'
  sed -e 's/ierr = //g' $1 \
| sed -e 's/CHKERRQ(ierr);//g'

