#!/bin/bash

# Checks for compliance with 
# Rule: 'Deprecated MPI routines require special care.'
# See http://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf for hints on fixing the issues.

# Steps:
# - find lines with deprecated MPI routines
# 

mpideprecated=("MPI_Keyval_create\s*(" \
               "MPI_Keyval_free\s*(" \
               "MPI_Attr_put\s*(" \
               "MPI_Attr_get\s*(" \
               "MPI_Attr_delete\s*(" \
               "MPI_Comm_errhandler_fn" \
               "MPI_File_errhandler_fn" \
               "MPI_Win_errhandler_fn")

for f in "$@"
do
 isvalid=`echo "$f" | grep -v "/ftn-auto/\|src/docs/";`
 if [ -n "$isvalid" ]; then
   for mpiidentifier in "${mpideprecated[@]}";
   do
     #grep -H -B 3 -A 3 "${mpiidentifier}" $f  #use this for additional context
     grep -n -H "${mpiidentifier}" $f
   done
 fi
done

