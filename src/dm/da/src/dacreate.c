#define PETSCDM_DLL
#include "../src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DASetOptionsPrefix"
/*@C
   DASetOptionsPrefix - Sets the prefix used for searching for all 
   DA options in the database.

   Collective on DA

   Input Parameter:
+  da - the DA context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: DA, set, options, prefix, database

.seealso: DASetFromOptions()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetOptionsPrefix(DA da,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)da,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetSizes"
/*@
  DASetSizes - Sets the global sizes

  Collective on DA

  Input Parameters:
+ da - the DA
- M - the global X size (or PETSC_DECIDE)
- N - the global Y size (or PETSC_DECIDE)
- P - the global Z size (or PETSC_DECIDE)

  Level: intermediate

.seealso: DAGetSize(), PetscSplitOwnership()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetSizes(DA da, PetscInt M, PetscInt N, PetscInt P)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_COOKIE, 1);
  da->M = M;
  da->N = N;
  da->P = P;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DACreate"
/*@
  DACreate - Creates an empty DA object. The type can then be set with DASetType(),
  or DASetFromOptions().

   If you never  call DASetType() or DASetFromOptions() it will generate an 
   error when you try to use the DA.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DA object

  Output Parameter:
. da  - The DA object

  Level: beginner

.keywords: DA, create
.seealso: DASetType(), DASetSizes(), DADuplicate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DACreate(MPI_Comm comm, DA *da)
{
  DA             d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(da,2);
  *da = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(d, _p_DA, struct _DAOps, DM_COOKIE, 0, "DM", comm, DADestroy, DAView);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) d, "DA");CHKERRQ(ierr);
  ierr = PetscMemzero(d->ops, sizeof(struct _DAOps));CHKERRQ(ierr);

  d->dim        = -1;
  d->interptype = DA_Q1;
  d->refine_x   = 2;
  d->fieldname  = PETSC_NULL;
  d->nlocal     = -1;
  d->Nlocal     = -1;
  d->M          = -1;
  d->N          = -1;
  d->P          = -1;
  d->m          = -1;
  d->n          = -1;
  d->p          = -1;
  d->w          = -1;
  d->s          = -1;
  d->xs = -1; d->xe = -1; d->ys = -1; d->ye = -1; d->zs = -1; d->ze = -1;
  d->Xs = -1; d->Xe = -1; d->Ys = -1; d->Ye = -1; d->Zs = -1; d->Ze = -1;

  d->gtol         = PETSC_NULL;
  d->ltog         = PETSC_NULL;
  d->ltol         = PETSC_NULL;
  d->ltogmap      = PETSC_NULL;
  d->ltogmapb     = PETSC_NULL;
  d->ao           = PETSC_NULL;
  d->base         = -1;
  d->wrap         = DA_NONPERIODIC;
  d->stencil_type = DA_STENCIL_STAR;
  d->idx          = PETSC_NULL;
  d->Nl           = -1;

  *da = d; 
  PetscFunctionReturn(0);
}
