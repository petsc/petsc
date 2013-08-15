#!/bin/bash
#
#  Returns the SAWS published PETSc stack one function per line
#
${PETSC_DIR}/bin/saws/getSAWs.bash PETSc/Stack | jshon -e directories -e Stack -e variable -e functions -e data 
#


