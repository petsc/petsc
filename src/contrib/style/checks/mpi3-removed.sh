#!/bin/bash

# Checks for compliance with 
# Rule: 'Removed MPI routines require special care.'
# See http://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf for hints on fixing the issues.

# Steps:
# - find lines with removed MPI routines
# 

mpideprecated=("MPI_Type_hvector\s*(" \
               "MPI_Type_hindexed\s*(" \
               "MPI_Type_struct\s*(" \
               "MPI_Type_extent\s*("  \
               "MPI_Type_lb\s*("  \
               "MPI_Type_ub\s*("  \
               "MPI_Address\s*(" \
               "MPI_Errhandler_create\s*(" \
               "MPI_Errhandler_get\s*(" \
               "MPI_Errhandler_set\s*(" \
               "MPI_Handler_function\s*("  \
               "MPI_Comm_errhandler_fn" \
               "MPI_File_errhandler_fn" \
               "MPI_Win_errhandler_fn" \
               "MPI_COMBINER_HINDEXED_INTEGER" \
               "MPI_COMBINER_HVECTOR_INTEGER" \
               "MPI_COMBINER_STRUCT_INTEGER" \
              )

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

