#include <petsc-private/kspimpl.h> /*I "petscksp.h" I*/
#include <petscdm.h>         /*I "petscdm.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "DMKSPDestroy"
static PetscErrorCode DMKSPDestroy(DMKSP *kdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
 if (!*kdm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*kdm),DMKSP_CLASSID,1);
  if (--((PetscObject)(*kdm))->refct > 0) {*kdm = 0; PetscFunctionReturn(0);}
  if ((*kdm)->ops->destroy) {ierr = ((*kdm)->ops->destroy)(kdm);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(kdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPCreate"
static PetscErrorCode DMKSPCreate(MPI_Comm comm,DMKSP *kdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = KSPInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscHeaderCreate(*kdm, _p_DMKSP, struct _DMKSPOps, DMKSP_CLASSID, -1, "DMSKP", "DMKSP", "DMKSP", comm, DMKSPDestroy, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMemzero((*kdm)->ops, sizeof(struct _DMKSPOps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHook_DMKSP"
/* Attaches the DMKSP to the coarse level.
 * Under what conditions should we copy versus duplicate?
 */
static PetscErrorCode DMCoarsenHook_DMKSP(DM dm,DM dmc,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCopyDMKSP(dm,dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefineHook_DMKSP"
/* Attaches the DMKSP to the coarse level.
 * Under what conditions should we copy versus duplicate?
 */
static PetscErrorCode DMRefineHook_DMKSP(DM dm,DM dmc,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCopyDMKSP(dm,dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetDMKSP"
/*@C
   DMGetDMKSP - get read-only private DMKSP context from a DM

   Not Collective

   Input Argument:
.  dm - DM to be used with KSP

   Output Argument:
.  snesdm - private DMKSP context

   Level: developer

   Notes:
   Use DMGetDMKSPWrite() if write access is needed. The DMKSPSetXXX API should be used wherever possible.

.seealso: DMGetDMKSPWrite()
@*/
PetscErrorCode DMGetDMKSP(DM dm,DMKSP *kspdm)
{
  PetscErrorCode ierr;
  DMKSP          tmpkspdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)dm,"DMKSP",(PetscObject*)kspdm);CHKERRQ(ierr);
  if (!*kspdm) {
    ierr = PetscInfo(dm,"Creating new DMKSP\n");CHKERRQ(ierr);
    ierr = DMKSPCreate(((PetscObject)dm)->comm,kspdm);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"DMKSP",(PetscObject)*kspdm);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_DMKSP,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = DMRefineHookAdd(dm,DMRefineHook_DMKSP,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    tmpkspdm = *kspdm;
    ierr = DMKSPDestroy(&tmpkspdm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetDMKSPWrite"
/*@C
   DMGetDMKSPWrite - get write access to private DMKSP context from a DM

   Not Collective

   Input Argument:
.  dm - DM to be used with KSP

   Output Argument:
.  kspdm - private DMKSP context

   Level: developer

.seealso: DMGetDMKSP()
@*/
PetscErrorCode DMGetDMKSPWrite(DM dm,DMKSP *kspdm)
{
  PetscErrorCode ierr;
  DMKSP          kdm,tkdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMKSP(dm,&kdm);CHKERRQ(ierr);
  if (!kdm->originaldm) kdm->originaldm = dm;
  if (kdm->originaldm != dm) {  /* Copy on write */
    DMKSP          oldkdm = kdm;
    ierr = PetscInfo(dm,"Copying DMKSP due to write\n");CHKERRQ(ierr);
    ierr = DMKSPCreate(((PetscObject)dm)->comm,&kdm);CHKERRQ(ierr);
    kdm->ops->computeoperators     = oldkdm->ops->computeoperators;
    kdm->ops->computerhs           = oldkdm->ops->computerhs;
    kdm->ops->computeinitialguess  = oldkdm->ops->computeinitialguess;
    kdm->ops->destroy              = oldkdm->ops->destroy;
    kdm->ops->duplicate            = oldkdm->ops->duplicate;

    kdm->operatorsctx    = oldkdm->operatorsctx;
    kdm->rhsctx          = oldkdm->rhsctx;
    kdm->initialguessctx = oldkdm->initialguessctx;
    kdm->data            = oldkdm->data;

    kdm->fortran_func_pointers[0] = oldkdm->fortran_func_pointers[0];
    kdm->fortran_func_pointers[1] = oldkdm->fortran_func_pointers[1];
    kdm->fortran_func_pointers[2] = oldkdm->fortran_func_pointers[2];

    ierr = PetscObjectCompose((PetscObject)dm,"DMKSP",(PetscObject)kdm);CHKERRQ(ierr);
    tkdm = kdm;
    ierr = DMKSPDestroy(&tkdm);CHKERRQ(ierr);
    /* implementation specific copy hooks */
    if (kdm->ops->duplicate) {ierr = (*kdm->ops->duplicate)(oldkdm,dm);CHKERRQ(ierr);}
  }
  *kspdm = kdm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCopyDMKSP"
/*@C
   DMCopyDMKSP - copies a DM context to a new DM

   Logically Collective

   Input Arguments:
+  dmsrc - DM to obtain context from
-  dmdest - DM to add context to

   Level: developer

   Note:
   The context is copied by reference. This function does not ensure that a context exists.

.seealso: DMGetDMKSP(), KSPSetDM()
@*/
PetscErrorCode DMCopyDMKSP(DM dmsrc,DM dmdest)
{
  PetscErrorCode ierr;
  DMKSP          kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmdest,DM_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)dmsrc,"DMKSP",(PetscObject*)&kdm);CHKERRQ(ierr);
  if (kdm) {
    ierr = PetscObjectCompose((PetscObject)dmdest,"DMKSP",(PetscObject)kdm);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(dmdest,DMCoarsenHook_DMKSP,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = DMRefineHookAdd(dmdest,DMRefineHook_DMKSP,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPSetComputeOperators"
/*@C
   DMKSPSetComputeOperators - set KSP matrix evaluation function

   Not Collective

   Input Argument:
+  dm - DM to be used with KSP
.  func - matrix evaluation function, see KSPSetComputeOperators() for calling sequence
-  ctx - context for matrix evaluation

   Level: advanced

   Note:
   KSPSetComputeOperators() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the matrix.

.seealso: DMKSPSetContext(), DMKSPGetComputeOperators(), KSPSetOperators()
@*/
PetscErrorCode DMKSPSetComputeOperators(DM dm,PetscErrorCode (*func)(KSP,Mat,Mat,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMKSP          kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMKSPWrite(dm,&kdm);CHKERRQ(ierr);
  if (func) kdm->ops->computeoperators = func;
  if (ctx)  kdm->operatorsctx          = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPGetComputeOperators"
/*@C
   DMKSPGetComputeOperators - get KSP matrix evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with KSP

   Output Arguments:
+  func - matrix evaluation function, see KSPSetComputeOperators() for calling sequence
-  ctx - context for matrix evaluation

   Level: advanced

.seealso: DMKSPSetContext(), KSPSetComputeOperators(), DMKSPSetComputeOperators()
@*/
PetscErrorCode DMKSPGetComputeOperators(DM dm,PetscErrorCode (**func)(KSP,Mat,Mat,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMKSP          kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMKSP(dm,&kdm);CHKERRQ(ierr);
  if (func) *func = kdm->ops->computeoperators;
  if (ctx)  *(void**)ctx = kdm->operatorsctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPSetComputeRHS"
/*@C
   DMKSPSetComputeRHS - set KSP right hand side evaluation function

   Not Collective

   Input Argument:
+  dm - DM to be used with KSP
.  func - right hand side evaluation function, see KSPSetComputeRHS() for calling sequence
-  ctx - context for right hand side evaluation

   Level: advanced

   Note:
   KSPSetComputeRHS() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the matrix.

.seealso: DMKSPSetContext(), DMKSPGetComputeRHS(), KSPSetRHS()
@*/
PetscErrorCode DMKSPSetComputeRHS(DM dm,PetscErrorCode (*func)(KSP,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMKSP          kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMKSPWrite(dm,&kdm);CHKERRQ(ierr);
  if (func) kdm->ops->computerhs = func;
  if (ctx)  kdm->rhsctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPSetComputeInitialGuess"
/*@C
   DMKSPSetComputeInitialGuess - set KSP initial guess evaluation function

   Not Collective

   Input Argument:
+  dm - DM to be used with KSP
.  func - initial guess evaluation function, see KSPSetComputeInitialGuess() for calling sequence
-  ctx - context for right hand side evaluation

   Level: advanced

   Note:
   KSPSetComputeInitialGuess() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  

.seealso: DMKSPSetContext(), DMKSPGetComputeRHS(), KSPSetRHS()
@*/
PetscErrorCode DMKSPSetComputeInitialGuess(DM dm,PetscErrorCode (*func)(KSP,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMKSP          kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMKSPWrite(dm,&kdm);CHKERRQ(ierr);
  if (func) kdm->ops->computeinitialguess = func;
  if (ctx)  kdm->initialguessctx      = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPGetComputeRHS"
/*@C
   DMKSPGetComputeRHS - get KSP right hand side evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with KSP

   Output Arguments:
+  func - right hand side evaluation function, see KSPSetComputeRHS() for calling sequence
-  ctx - context for right hand side evaluation

   Level: advanced

.seealso: DMKSPSetContext(), KSPSetComputeRHS(), DMKSPSetComputeRHS()
@*/
PetscErrorCode DMKSPGetComputeRHS(DM dm,PetscErrorCode (**func)(KSP,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMKSP          kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMKSP(dm,&kdm);CHKERRQ(ierr);
  if (func) *func = kdm->ops->computerhs;
  if (ctx)  *(void**)ctx = kdm->rhsctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMKSPGetComputeInitialGuess"
/*@C
   DMKSPGetComputeInitialGuess - get KSP initial guess evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with KSP

   Output Arguments:
+  func - initial guess evaluation function, see KSPSetComputeInitialGuess() for calling sequence
-  ctx - context for right hand side evaluation

   Level: advanced

.seealso: DMKSPSetContext(), KSPSetComputeRHS(), DMKSPSetComputeRHS()
@*/
PetscErrorCode DMKSPGetComputeInitialGuess(DM dm,PetscErrorCode (**func)(KSP,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMKSP          kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMKSP(dm,&kdm);CHKERRQ(ierr);
  if (func) *func = kdm->ops->computeinitialguess;
  if (ctx)  *(void**)ctx = kdm->initialguessctx;
  PetscFunctionReturn(0);
}
