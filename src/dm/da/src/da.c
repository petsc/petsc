#define PETSCDM_DLL
#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DASetPeriodicity"
/*@
  DASetInterpolationType - Sets the type of periodicity

  Not collective

  Input Parameter:
+ da    - The DA
- ptype - One of DA_NONPERIODIC, DA_XPERIODIC, DA_YPERIODIC, DA_ZPERIODIC, DA_XYPERIODIC, DA_XZPERIODIC, DA_YZPERIODIC, or DA_XYZPERIODIC

  Level: intermediate

.keywords:  distributed array, periodicity
.seealso: DACreate(), DADestroy(), DA, DAPeriodicType
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetPeriodicity(DA da, DAPeriodicType ptype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->wrap = ptype;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetDof"
/*@
  DASetDof - Sets the number of degrees of freedom per vertex

  Not collective

  Input Parameter:
+ da  - The DA
- dof - Number of degrees of freedom

  Level: intermediate

.keywords:  distributed array, degrees of freedom
.seealso: DACreate(), DADestroy(), DA
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetDof(DA da, int dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->w = dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetStencilWidth"
/*@
  DASetStencilWidth - Sets the width of the communication stencil

  Not collective

  Input Parameter:
+ da    - The DA
- width - The stencil width

  Level: intermediate

.keywords:  distributed array, stencil
.seealso: DACreate(), DADestroy(), DA
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetStencilWidth(DA da, int width)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  da->s = width;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetVertexDivision"
/*@
  DASetVertexDivision - Sets the number of nodes in each direction on each process

  Not collective

  Input Parameter:
+ da - The DA
. lx - array containing number of nodes in the X direction on each process, or PETSC_NULL. If non-null, must be of length 'size'.
. ly - array containing number of nodes in the Y direction on each process, or PETSC_NULL. If non-null, must be of length 'size'.
- lz - array containing number of nodes in the Z direction on each process, or PETSC_NULL. If non-null, must be of length 'size'.

  Level: intermediate

.keywords:  distributed array
.seealso: DACreate(), DADestroy(), DA
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetVertexDivision(DA da, const PetscInt lx[], const PetscInt ly[], const PetscInt lz[])
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscObjectGetComm((PetscObject) da, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (lx) {
    ierr = PetscMemcpy(da->lx, lx, size*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (ly) {
    ierr = PetscMemcpy(da->ly, ly, size*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (lz) {
    ierr = PetscMemcpy(da->lz, lz, size*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

