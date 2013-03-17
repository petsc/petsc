#!/bin/bash
#
# Script for extracting information about modified files from Mercurial and running the style checker on it
#
# Requires PETSC_DIR to be set correctly. Script does not take any arguments.
#

# Get status from Mercurial:
cd ${PETSC_DIR}
updated_files=`hg status | grep "^[AM]"`

# Crop off first letter in each line and store in array:
updated_file_array=()
while read -r f; do
  updated_file_array=("${updated_file_array[@]}" ${f:2})
done <<< "$updated_files"

# Now run style-checker on all files (note that we are in ${PETSC_DIR}:
src/contrib/style/stylecheck.sh "${updated_file_array[@]}"


