
#include <petsc-private/vecimpl.h>    /*I "petscvec.h" I*/

#undef __FUNCT__
#define __FUNCT__ "VecScatterCreateToAll"
/*@
      VecScatterCreateToAll - Creates a vector and a scatter context that copies all
          vector values to each processor

  Collective on Vec

  Input Parameter:
.  vin  - input MPIVEC

  Output Parameter:
+  ctx - scatter context
-  vout - output SEQVEC that is large enough to scatter into

  Level: intermediate

   Note: vout may be PETSC_NULL [PETSC_NULL_OBJECT from fortran] if you do not
   need to have it created

   Usage:
$        VecScatterCreateToAll(vin,&ctx,&vout);
$
$        // scatter as many times as you need
$        VecScatterBegin(ctx,vin,vout,INSERT_VALUES,SCATTER_FORWARD);
$        VecScatterEnd(ctx,vin,vout,INSERT_VALUES,SCATTER_FORWARD);
$
$        // destroy scatter context and local vector when no longer needed
$        VecScatterDestroy(ctx);
$        VecDestroy(vout);

    Do NOT create a vector and then pass it in as the final argument vout! vout is created by this routine
  automatically (unless you pass PETSC_NULL in for that argument if you do not need it).

.seealso VecScatterCreate(), VecScatterCreateToZero(), VecScatterBegin(), VecScatterEnd()

@*/
PetscErrorCode  VecScatterCreateToAll(Vec vin,VecScatter *ctx,Vec *vout)
{

  PetscErrorCode ierr;
  PetscInt       N;
  IS             is;
  Vec            tmp;
  Vec           *tmpv;
  PetscBool      tmpvout = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,1);
  PetscValidType(vin,1);
  PetscValidPointer(ctx,2);
  if (vout) {
    PetscValidPointer(vout,3);
    tmpv = vout;
  } else {
    tmpvout = PETSC_TRUE;
    tmpv = &tmp;
  }

  /* Create seq vec on each proc, with the same size of the original mpi vec */
  ierr = VecGetSize(vin,&N);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,tmpv);CHKERRQ(ierr);
  /* Create the VecScatter ctx with the communication info */
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(vin,is,*tmpv,is,ctx);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  if (tmpvout) {ierr = VecDestroy(tmpv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecScatterCreateToZero"
/*@
      VecScatterCreateToZero - Creates an output vector and a scatter context used to
              copy all vector values into the output vector on the zeroth processor

  Collective on Vec

  Input Parameter:
.  vin  - input MPIVEC

  Output Parameter:
+  ctx - scatter context
-  vout - output SEQVEC that is large enough to scatter into on processor 0 and
          of length zero on all other processors

  Level: intermediate

   Note: vout may be PETSC_NULL [PETSC_NULL_OBJECT from fortran] if you do not
   need to have it created

   Usage:
$        VecScatterCreateToZero(vin,&ctx,&vout);
$
$        // scatter as many times as you need
$        VecScatterBegin(ctx,vin,vout,INSERT_VALUES,SCATTER_FORWARD);
$        VecScatterEnd(ctx,vin,vout,INSERT_VALUES,SCATTER_FORWARD);
$
$        // destroy scatter context and local vector when no longer needed
$        VecScatterDestroy(ctx);
$        VecDestroy(vout);

.seealso VecScatterCreate(), VecScatterCreateToAll(), VecScatterBegin(), VecScatterEnd()

    Do NOT create a vector and then pass it in as the final argument vout! vout is created by this routine
  automatically (unless you pass PETSC_NULL in for that argument if you do not need it).

@*/
PetscErrorCode  VecScatterCreateToZero(Vec vin,VecScatter *ctx,Vec *vout)
{

  PetscErrorCode ierr;
  PetscInt       N;
  PetscMPIInt    rank;
  IS             is;
  Vec            tmp;
  Vec           *tmpv;
  PetscBool      tmpvout = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,1);
  PetscValidType(vin,1);
  PetscValidPointer(ctx,2);
  if (vout) {
    PetscValidPointer(vout,3);
    tmpv = vout;
  } else {
    tmpvout = PETSC_TRUE;
    tmpv = &tmp;
  }

  /* Create vec on each proc, with the same size of the original mpi vec (all on process 0)*/
  ierr = VecGetSize(vin,&N);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)vin)->comm,&rank);CHKERRQ(ierr);
  if (rank) N = 0;
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,tmpv);CHKERRQ(ierr);
  /* Create the VecScatter ctx with the communication info */
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(vin,is,*tmpv,is,ctx);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  if (tmpvout) {ierr = VecDestroy(tmpv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

