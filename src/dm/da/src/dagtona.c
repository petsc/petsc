#define PETSCDM_DLL

/*
     Tools to help solve the coarse grid problem redundantly.
  Provides two scatter contexts that (1) map from the usual global vector
  to all processors the entire vector in NATURAL numbering and (2)
  from the entire vector on each processor in natural numbering extracts
  out this processors piece in GLOBAL numbering
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DAGlobalToNaturalAllCreate"
/*@
   DAGlobalToNaturalAllCreate - Creates a scatter context that maps from the 
     global vector the entire vector to each processor in natural numbering

   Collective on DA

   Input Parameter:
.  da - the distributed array context

   Output Parameter:
.  scatter - the scatter context

   Level: advanced

.keywords: distributed array, global to local, begin, coarse problem

.seealso: DAGlobalToNaturalEnd(), DALocalToGlobal(), DACreate2d(), 
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGlobalToNaturalAllCreate(DA da,VecScatter *scatter)
{
  PetscErrorCode ierr;
  PetscInt       N;
  IS             from,to;
  Vec            tmplocal,global;
  AO             ao;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(scatter,2);
  ierr = DAGetAO(da,&ao);CHKERRQ(ierr);

  /* create the scatter context */
  ierr = VecCreateMPIWithArray(((PetscObject)da)->comm,da->Nlocal,PETSC_DETERMINE,0,&global);CHKERRQ(ierr);
  ierr = VecGetSize(global,&N);CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)da)->comm,N,0,1,&to);CHKERRQ(ierr);
  ierr = AOPetscToApplicationIS(ao,to);CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)da)->comm,N,0,1,&from);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,N,0,&tmplocal);CHKERRQ(ierr);
  ierr = VecSetBlockSize(tmplocal,da->w);CHKERRQ(ierr);
  ierr = VecSetBlockSize(global,da->w);CHKERRQ(ierr);
  ierr = VecScatterCreate(global,from,tmplocal,to,scatter);CHKERRQ(ierr);
  ierr = VecDestroy(tmplocal);CHKERRQ(ierr);  
  ierr = VecDestroy(global);CHKERRQ(ierr);  
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DANaturalAllToGlobalCreate"
/*@
   DANaturalAllToGlobalCreate - Creates a scatter context that maps from a copy
     of the entire vector on each processor to its local part in the global vector.

   Collective on DA

   Input Parameter:
.  da - the distributed array context

   Output Parameter:
.  scatter - the scatter context

   Level: advanced

.keywords: distributed array, global to local, begin, coarse problem

.seealso: DAGlobalToNaturalEnd(), DALocalToGlobal(), DACreate2d(), 
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DACreateNaturalVector()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DANaturalAllToGlobalCreate(DA da,VecScatter *scatter)
{
  PetscErrorCode ierr;
  PetscInt       M,m = da->Nlocal,start;
  IS             from,to;
  Vec            tmplocal,global;
  AO             ao;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidPointer(scatter,2);
  ierr = DAGetAO(da,&ao);CHKERRQ(ierr);

  /* create the scatter context */
  ierr = MPI_Allreduce(&m,&M,1,MPIU_INT,MPI_SUM,((PetscObject)da)->comm);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)da)->comm,m,PETSC_DETERMINE,0,&global);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(global,&start,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)da)->comm,m,start,1,&from);CHKERRQ(ierr);
  ierr = AOPetscToApplicationIS(ao,from);CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)da)->comm,m,start,1,&to);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,M,0,&tmplocal);CHKERRQ(ierr);
  ierr = VecSetBlockSize(tmplocal,da->w);CHKERRQ(ierr);
  ierr = VecSetBlockSize(global,da->w);CHKERRQ(ierr);
  ierr = VecScatterCreate(tmplocal,from,global,to,scatter);CHKERRQ(ierr);
  ierr = VecDestroy(tmplocal);CHKERRQ(ierr);  
  ierr = VecDestroy(global);CHKERRQ(ierr);  
  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

