#include <petsc/private/dmimpl.h>
#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/
#include <petscdm.h>

static PetscErrorCode DMKSPDestroy(DMKSP *kdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*kdm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*kdm),DMKSP_CLASSID,1);
  if (--((PetscObject)(*kdm))->refct > 0) {*kdm = NULL; PetscFunctionReturn(0);}
  if ((*kdm)->ops->destroy) {ierr = ((*kdm)->ops->destroy)(kdm);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(kdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMKSPCreate(MPI_Comm comm,DMKSP *kdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*kdm, DMKSP_CLASSID, "DMKSP", "DMKSP", "DMKSP", comm, DMKSPDestroy, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

/*@C
   DMKSPCopy - copies the information in a DMKSP to another DMKSP

   Not Collective

   Input Parameters:
+  kdm - Original DMKSP
-  nkdm - DMKSP to receive the data, should have been created with DMKSPCreate()

   Level: developer

.seealso: DMKSPCreate(), DMKSPDestroy()
@*/
PetscErrorCode DMKSPCopy(DMKSP kdm,DMKSP nkdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(kdm,DMKSP_CLASSID,1);
  PetscValidHeaderSpecific(nkdm,DMKSP_CLASSID,2);
  nkdm->ops->computeoperators    = kdm->ops->computeoperators;
  nkdm->ops->computerhs          = kdm->ops->computerhs;
  nkdm->ops->computeinitialguess = kdm->ops->computeinitialguess;
  nkdm->ops->destroy             = kdm->ops->destroy;
  nkdm->ops->duplicate           = kdm->ops->duplicate;

  nkdm->operatorsctx    = kdm->operatorsctx;
  nkdm->rhsctx          = kdm->rhsctx;
  nkdm->initialguessctx = kdm->initialguessctx;
  nkdm->data            = kdm->data;
  /* nkdm->originaldm   = kdm->originaldm; */ /* No need since nkdm->originaldm will be immediately updated in caller DMGetDMKSPWrite */

  nkdm->fortran_func_pointers[0] = kdm->fortran_func_pointers[0];
  nkdm->fortran_func_pointers[1] = kdm->fortran_func_pointers[1];
  nkdm->fortran_func_pointers[2] = kdm->fortran_func_pointers[2];

  /* implementation specific copy hooks */
  if (kdm->ops->duplicate) {ierr = (*kdm->ops->duplicate)(kdm,nkdm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
   DMGetDMKSP - get read-only private DMKSP context from a DM

   Logically Collective

   Input Parameter:
.  dm - DM to be used with KSP

   Output Parameter:
.  snesdm - private DMKSP context

   Level: developer

   Notes:
   Use DMGetDMKSPWrite() if write access is needed. The DMKSPSetXXX API should be used wherever possible.

.seealso: DMGetDMKSPWrite()
@*/
PetscErrorCode DMGetDMKSP(DM dm,DMKSP *kspdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *kspdm = (DMKSP) dm->dmksp;
  if (!*kspdm) {
    ierr                 = PetscInfo(dm,"Creating new DMKSP\n");CHKERRQ(ierr);
    ierr                 = DMKSPCreate(PetscObjectComm((PetscObject)dm),kspdm);CHKERRQ(ierr);
    dm->dmksp            = (PetscObject) *kspdm;
    (*kspdm)->originaldm = dm;
    ierr                 = DMCoarsenHookAdd(dm,DMCoarsenHook_DMKSP,NULL,NULL);CHKERRQ(ierr);
    ierr                 = DMRefineHookAdd(dm,DMRefineHook_DMKSP,NULL,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   DMGetDMKSPWrite - get write access to private DMKSP context from a DM

   Logically Collective

   Input Parameter:
.  dm - DM to be used with KSP

   Output Parameter:
.  kspdm - private DMKSP context

   Level: developer

.seealso: DMGetDMKSP()
@*/
PetscErrorCode DMGetDMKSPWrite(DM dm,DMKSP *kspdm)
{
  PetscErrorCode ierr;
  DMKSP          kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMKSP(dm,&kdm);CHKERRQ(ierr);
  if (!kdm->originaldm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"DMKSP has a NULL originaldm");
  if (kdm->originaldm != dm) {  /* Copy on write */
    DMKSP oldkdm = kdm;
    ierr      = PetscInfo(dm,"Copying DMKSP due to write\n");CHKERRQ(ierr);
    ierr      = DMKSPCreate(PetscObjectComm((PetscObject)dm),&kdm);CHKERRQ(ierr);
    ierr      = DMKSPCopy(oldkdm,kdm);CHKERRQ(ierr);
    ierr      = DMKSPDestroy((DMKSP*)&dm->dmksp);CHKERRQ(ierr);
    dm->dmksp = (PetscObject)kdm;
    kdm->originaldm = dm;
  }
  *kspdm = kdm;
  PetscFunctionReturn(0);
}

/*@C
   DMCopyDMKSP - copies a DM context to a new DM

   Logically Collective

   Input Parameters:
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmdest,DM_CLASSID,2);
  ierr          = DMKSPDestroy((DMKSP*)&dmdest->dmksp);CHKERRQ(ierr);
  dmdest->dmksp = dmsrc->dmksp;
  ierr          = PetscObjectReference(dmdest->dmksp);CHKERRQ(ierr);
  ierr          = DMCoarsenHookAdd(dmdest,DMCoarsenHook_DMKSP,NULL,NULL);CHKERRQ(ierr);
  ierr          = DMRefineHookAdd(dmdest,DMRefineHook_DMKSP,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMKSPSetComputeOperators - set KSP matrix evaluation function

   Not Collective

   Input Parameters:
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
PetscErrorCode DMKSPSetComputeOperators(DM dm,PetscErrorCode (*func)(KSP,Mat,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMKSP          kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMKSPWrite(dm,&kdm);CHKERRQ(ierr);
  if (func) kdm->ops->computeoperators = func;
  if (ctx) kdm->operatorsctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMKSPGetComputeOperators - get KSP matrix evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with KSP

   Output Parameters:
+  func - matrix evaluation function, see KSPSetComputeOperators() for calling sequence
-  ctx - context for matrix evaluation

   Level: advanced

.seealso: DMKSPSetContext(), KSPSetComputeOperators(), DMKSPSetComputeOperators()
@*/
PetscErrorCode DMKSPGetComputeOperators(DM dm,PetscErrorCode (**func)(KSP,Mat,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMKSP          kdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMKSP(dm,&kdm);CHKERRQ(ierr);
  if (func) *func = kdm->ops->computeoperators;
  if (ctx) *(void**)ctx = kdm->operatorsctx;
  PetscFunctionReturn(0);
}

/*@C
   DMKSPSetComputeRHS - set KSP right hand side evaluation function

   Not Collective

   Input Parameters:
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
  if (ctx) kdm->rhsctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMKSPSetComputeInitialGuess - set KSP initial guess evaluation function

   Not Collective

   Input Parameters:
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
  if (ctx) kdm->initialguessctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMKSPGetComputeRHS - get KSP right hand side evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with KSP

   Output Parameters:
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
  if (ctx) *(void**)ctx = kdm->rhsctx;
  PetscFunctionReturn(0);
}

/*@C
   DMKSPGetComputeInitialGuess - get KSP initial guess evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with KSP

   Output Parameters:
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
  if (ctx) *(void**)ctx = kdm->initialguessctx;
  PetscFunctionReturn(0);
}
