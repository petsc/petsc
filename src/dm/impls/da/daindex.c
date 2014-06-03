
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMDAGetNatural_Private"
/*
   Gets the natural number for each global number on the process.

   Used by DMDAGetAO() and DMDAGlobalToNatural_Create()
*/
PetscErrorCode DMDAGetNatural_Private(DM da,PetscInt *outNlocal,IS *isnatural)
{
  PetscErrorCode ierr;
  PetscInt       Nlocal,i,j,k,*lidx,lict = 0;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  Nlocal = (dd->xe-dd->xs);
  if (dd->dim > 1) Nlocal *= (dd->ye-dd->ys);
  if (dd->dim > 2) Nlocal *= (dd->ze-dd->zs);

  ierr = PetscMalloc1(Nlocal,&lidx);CHKERRQ(ierr);

  if (dd->dim == 1) {
    for (i=dd->xs; i<dd->xe; i++) {
      /*  global number in natural ordering */
      lidx[lict++] = i;
    }
  } else if (dd->dim == 2) {
    for (j=dd->ys; j<dd->ye; j++) {
      for (i=dd->xs; i<dd->xe; i++) {
        /*  global number in natural ordering */
        lidx[lict++] = i + j*dd->M*dd->w;
      }
    }
  } else if (dd->dim == 3) {
    for (k=dd->zs; k<dd->ze; k++) {
      for (j=dd->ys; j<dd->ye; j++) {
        for (i=dd->xs; i<dd->xe; i++) {
          lidx[lict++] = i + j*dd->M*dd->w + k*dd->M*dd->N*dd->w;
        }
      }
    }
  }
  *outNlocal = Nlocal;
  ierr       = ISCreateGeneral(PetscObjectComm((PetscObject)da),Nlocal,lidx,PETSC_OWN_POINTER,isnatural);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetAO"
/*@
   DMDAGetAO - Gets the application ordering context for a distributed array.

   Collective on DMDA

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  ao - the application ordering context for DMDAs

   Level: intermediate

   Notes:
   In this case, the AO maps to the natural grid ordering that would be used
   for the DMDA if only 1 processor were employed (ordering most rapidly in the
   x-direction, then y, then z).  Multiple degrees of freedom are numbered
   for each node (rather than 1 component for the whole grid, then the next
   component, etc.)

.keywords: distributed array, get, global, indices, local-to-global

.seealso: DMDACreate2d(), DMDAGetGhostCorners(), DMDAGetCorners(), DMDALocalToGlocal()
          DMGlobalToLocalBegin(), DMGlobalToLocalEnd(), DMLocalToLocalBegin(), DMLocalToLocalEnd(), DMDAGetOwnershipRanges(),
          AO, AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode  DMDAGetAO(DM da,AO *ao)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(ao,2);

  /*
     Build the natural ordering to PETSc ordering mappings.
  */
  if (!dd->ao) {
    IS             ispetsc,isnatural;
    PetscErrorCode ierr;
    PetscInt       Nlocal;

    ierr = DMDAGetNatural_Private(da,&Nlocal,&isnatural);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)da),Nlocal,dd->base,1,&ispetsc);CHKERRQ(ierr);
    ierr = AOCreateBasicIS(isnatural,ispetsc,&dd->ao);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)da,(PetscObject)dd->ao);CHKERRQ(ierr);
    ierr = ISDestroy(&ispetsc);CHKERRQ(ierr);
    ierr = ISDestroy(&isnatural);CHKERRQ(ierr);
  }
  *ao = dd->ao;
  PetscFunctionReturn(0);
}


