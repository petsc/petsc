
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
  PetscCall(DMDAGetAO(da,&ao));

  /* create the scatter context */
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)da),dd->w,dd->Nlocal,PETSC_DETERMINE,NULL,&global));
  PetscCall(VecGetSize(global,&N));
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)da),N,0,1,&to));
  PetscCall(AOPetscToApplicationIS(ao,to));
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)da),N,0,1,&from));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,dd->w,N,NULL,&tmplocal));
  PetscCall(VecScatterCreate(global,from,tmplocal,to,scatter));
  PetscCall(VecDestroy(&tmplocal));
  PetscCall(VecDestroy(&global));
  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
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
  PetscCall(DMDAGetAO(da,&ao));

  /* create the scatter context */
  PetscCall(MPIU_Allreduce(&m,&M,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)da)));
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)da),dd->w,m,PETSC_DETERMINE,NULL,&global));
  PetscCall(VecGetOwnershipRange(global,&start,NULL));
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)da),m,start,1,&from));
  PetscCall(AOPetscToApplicationIS(ao,from));
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)da),m,start,1,&to));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,dd->w,M,NULL,&tmplocal));
  PetscCall(VecScatterCreate(tmplocal,from,global,to,scatter));
  PetscCall(VecDestroy(&tmplocal));
  PetscCall(VecDestroy(&global));
  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
  PetscFunctionReturn(0);
}
