
/*
     Tools to help solve the coarse grid problem redundantly.
  Provides two scatter contexts that (1) map from the usual global vector
  to all processors the entire vector in NATURAL numbering and (2)
  from the entire vector on each processor in natural numbering extracts
  out this processors piece in GLOBAL numbering
*/

#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DMDAGlobalToNaturalAllCreate"
/*@
   DMDAGlobalToNaturalAllCreate - Creates a scatter context that maps from the 
     global vector the entire vector to each processor in natural numbering

   Collective on DMDA

   Input Parameter:
.  da - the distributed array context

   Output Parameter:
.  scatter - the scatter context

   Level: advanced

.keywords: distributed array, global to local, begin, coarse problem

.seealso: DMDAGlobalToNaturalEnd(), DMLocalToGlobalBegin(), DMDACreate2d(), 
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()
@*/
PetscErrorCode  DMDAGlobalToNaturalAllCreate(DM da,VecScatter *scatter)
{
  PetscErrorCode ierr;
  PetscInt       N;
  IS             from,to;
  Vec            tmplocal,global;
  AO             ao;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(scatter,2);
  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);

  /* create the scatter context */
  ierr = VecCreateMPIWithArray(((PetscObject)da)->comm,dd->w,dd->Nlocal,PETSC_DETERMINE,0,&global);CHKERRQ(ierr);
  ierr = VecGetSize(global,&N);CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)da)->comm,N,0,1,&to);CHKERRQ(ierr);
  ierr = AOPetscToApplicationIS(ao,to);CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)da)->comm,N,0,1,&from);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,dd->w,N,0,&tmplocal);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,tmplocal,to,scatter);CHKERRQ(ierr);
  ierr = VecDestroy(&tmplocal);CHKERRQ(ierr);  
  ierr = VecDestroy(&global);CHKERRQ(ierr);  
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDANaturalAllToGlobalCreate"
/*@
   DMDANaturalAllToGlobalCreate - Creates a scatter context that maps from a copy
     of the entire vector on each processor to its local part in the global vector.

   Collective on DMDA

   Input Parameter:
.  da - the distributed array context

   Output Parameter:
.  scatter - the scatter context

   Level: advanced

.keywords: distributed array, global to local, begin, coarse problem

.seealso: DMDAGlobalToNaturalEnd(), DMLocalToGlobalBegin(), DMDACreate2d(), 
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()
@*/
PetscErrorCode  DMDANaturalAllToGlobalCreate(DM da,VecScatter *scatter)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;
  PetscInt       M,m = dd->Nlocal,start;
  IS             from,to;
  Vec            tmplocal,global;
  AO             ao;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(scatter,2);
  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);

  /* create the scatter context */
  ierr = MPI_Allreduce(&m,&M,1,MPIU_INT,MPI_SUM,((PetscObject)da)->comm);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)da)->comm,dd->w,m,PETSC_DETERMINE,0,&global);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(global,&start,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)da)->comm,m,start,1,&from);CHKERRQ(ierr);
  ierr = AOPetscToApplicationIS(ao,from);CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)da)->comm,m,start,1,&to);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,dd->w,M,0,&tmplocal);CHKERRQ(ierr);
  ierr = VecScatterCreate(tmplocal,from,global,to,scatter);CHKERRQ(ierr);
  ierr = VecDestroy(&tmplocal);CHKERRQ(ierr);  
  ierr = VecDestroy(&global);CHKERRQ(ierr);  
  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

