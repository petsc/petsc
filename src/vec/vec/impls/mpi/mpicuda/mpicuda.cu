#define PETSCVEC_DLL
/*
   This file contains routines for Parallel vector operations.
 */
#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "../src/vec/vec/impls/mpi/pvecimpl.h"   /*I  "petscvec.h"   I*/
PETSC_CUDA_EXTERN_C_END
#include "../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h"


/*MC
   VECMPI - VECMPI = "mpi" - The basic parallel vector

   Options Database Keys:
. -vec_type mpi - sets the vector type to VECMPI during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMpiWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateMpi()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_MPICUDA"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_MPICUDA(Vec vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(vv,PETSC_TRUE,0,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)vv,VECMPICUDA);CHKERRQ(ierr);
  vv->valid_GPU_array = PETSC_CUDA_UNALLOCATED;
  /* finish filling these in */
  vv->ops->scale      = VecScale_SeqCUDA;
  PetscFunctionReturn(0);
}
EXTERN_C_END







