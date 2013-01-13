#!/bin/bash

rm -rf src2/
cp -R src/ src2/
find src2/*/examples/tutorials/*.c | xargs sed -i 's/ierr = //g'
find src2/*/examples/tutorials/*.c | xargs sed -i 's/CHKERRQ(ierr);//g'

