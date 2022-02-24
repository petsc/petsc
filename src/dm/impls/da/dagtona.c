
/*
     Tools to help solve the coarse grid problem redundantly.
  Provides two scatter contexts that (1) map from the usual global vector
  to all processors the entire vector in NATURAL numbering and (2)
  from the entire vector on each processor in natural numbering extracts
  out this processors piece in GLOBAL numbering
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

/*@
   DMDAGlobalToNaturalAllCreate - Creates a scatter context that maps from the
     global vector the entire vector to each processor in natural numbering

   Collective on da

   Input Parameter:
.  da - the distributed array context

   Output Parameter:
.  scatter - the scatter context

   Level: advanced

.seealso: DMDAGlobalToNaturalEnd(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()
@*/
PetscErrorCode  DMDAGlobalToNaturalAllCreate(DM da,VecScatter *scatter)
{
  PetscInt       N;
  IS             from,to;
  Vec            tmplocal,global;
  AO             ao;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidPointer(scatter,2);
  CHKERRQ(DMDAGetAO(da,&ao));

  /* create the scatter context */
  CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)da),dd->w,dd->Nlocal,PETSC_DETERMINE,NULL,&global));
  CHKERRQ(VecGetSize(global,&N));
  CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)da),N,0,1,&to));
  CHKERRQ(AOPetscToApplicationIS(ao,to));
  CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)da),N,0,1,&from));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,dd->w,N,NULL,&tmplocal));
  CHKERRQ(VecScatterCreate(global,from,tmplocal,to,scatter));
  CHKERRQ(VecDestroy(&tmplocal));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(ISDestroy(&from));
  CHKERRQ(ISDestroy(&to));
  PetscFunctionReturn(0);
}

/*@
   DMDANaturalAllToGlobalCreate - Creates a scatter context that maps from a copy
     of the entire vector on each processor to its local part in the global vector.

   Collective on da

   Input Parameter:
.  da - the distributed array context

   Output Parameter:
.  scatter - the scatter context

   Level: advanced

.seealso: DMDAGlobalToNaturalEnd(), DMLocalToGlobalBegin(), DMDACreate2d(),
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMDACreateNaturalVector()
@*/
PetscErrorCode  DMDANaturalAllToGlobalCreate(DM da,VecScatter *scatter)
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscInt       M,m = dd->Nlocal,start;
  IS             from,to;
  Vec            tmplocal,global;
  AO             ao;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidPointer(scatter,2);
  CHKERRQ(DMDAGetAO(da,&ao));

  /* create the scatter context */
  CHKERRMPI(MPIU_Allreduce(&m,&M,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)da)));
  CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)da),dd->w,m,PETSC_DETERMINE,NULL,&global));
  CHKERRQ(VecGetOwnershipRange(global,&start,NULL));
  CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)da),m,start,1,&from));
  CHKERRQ(AOPetscToApplicationIS(ao,from));
  CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)da),m,start,1,&to));
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,dd->w,M,NULL,&tmplocal));
  CHKERRQ(VecScatterCreate(tmplocal,from,global,to,scatter));
  CHKERRQ(VecDestroy(&tmplocal));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(ISDestroy(&from));
  CHKERRQ(ISDestroy(&to));
  PetscFunctionReturn(0);
}
