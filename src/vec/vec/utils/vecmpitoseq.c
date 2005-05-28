#define PETSCVEC_DLL

#include "vecimpl.h"    /*I "petscvec.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreateToAll"
/*@
      VecScatterCreateToAll - Creates a scatter context that copies all 
          vector values to each processor

  Collective

  Input Parameter: 
.  vin  - input MPIVEC

  Output Parameter:
+  ctx - scatter context
-  vout - output SEQVEC that is large enough to scatter into

  Level: intermediate

   Usage:
$        VecScatterCreateToAll(vin,&ctx,&vout);
$
$        // scatter as many times as you need 
$        VecScatterBegin(vin,vout,INSERT_VALUES,SCATTER_FORWARD,ctx);
$        VecScatterEnd(vin,vout,INSERT_VALUES,SCATTER_FORWARD,ctx);
$
$        // destroy scatter context and local vector when no longer needed
$        VecScatterDestroy(ctx);
$        VecDestroy(vout);

.seealso VecScatterCreate(), VecScatterCreateToZero(), VecScatterBegin(), VecScatterEnd()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecScatterCreateToAll(Vec vin,VecScatter *ctx,Vec *vout)
{

  PetscErrorCode ierr;
  PetscInt       N;
  IS             is;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_COOKIE,1);
  PetscValidType(vin,1);
  PetscValidPointer(ctx,2);
  PetscValidPointer(vout,3);

  /* Create seq vec on each proc, with the same size of the original mpi vec */
  ierr = VecGetSize(vin,&N);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,vout);CHKERRQ(ierr);
  /* Create the VecScatter ctx with the communication info */
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(vin,is,*vout,is,ctx);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreateToZero"
/*@
      VecScatterCreateToZero - Creates a scatter context that copies all 
          vector values to a vector on the zeroth processor

  Collective

  Input Parameter: 
.  vin  - input MPIVEC

  Output Parameter:
+  ctx - scatter context
-  vout - output MPIVEC that is large enough to scatter into on processor 0 and
          of length zero on all other processors

  Level: intermediate

   Usage:
$        VecScatterCreateToZero(vin,&ctx,&vout);
$
$        // scatter as many times as you need 
$        VecScatterBegin(vin,vout,INSERT_VALUES,SCATTER_FORWARD,ctx);
$        VecScatterEnd(vin,vout,INSERT_VALUES,SCATTER_FORWARD,ctx);
$
$        // destroy scatter context and local vector when no longer needed
$        VecScatterDestroy(ctx);
$        VecDestroy(vout);

   Note: If you want to treat the vector on processor zero as a sequential vector call
         VecGetArray() on it and create a sequential vector with VecCreateSeqWithArray().

.seealso VecScatterCreate(), VecScatterCreateToAll(), VecScatterBegin(), VecScatterEnd()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecScatterCreateToZero(Vec vin,VecScatter *ctx,Vec *vout)
{

  PetscErrorCode ierr;
  PetscInt       N;
  PetscMPIInt    rank;
  IS             is;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_COOKIE,1);
  PetscValidType(vin,1);
  PetscValidPointer(ctx,2);
  PetscValidPointer(vout,3);

  /* Create seq vec on each proc, with the same size of the original mpi vec */
  ierr = VecGetSize(vin,&N);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(vin->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = VecCreateMPI(vin->comm,N,N,vout);CHKERRQ(ierr);
  } else {
    ierr = VecCreateMPI(vin->comm,0,N,vout);CHKERRQ(ierr);
  }
  /* Create the VecScatter ctx with the communication info */
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(vin,is,*vout,is,ctx);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

