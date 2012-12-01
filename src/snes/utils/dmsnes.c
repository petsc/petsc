#include <petsc-private/snesimpl.h>   /*I "petscsnes.h" I*/
#include <petsc-private/dmimpl.h>     /*I "petscdm.h" I*/

#undef __FUNCT__
#define __FUNCT__ "DMSNESDestroy"
static PetscErrorCode DMSNESDestroy(DMSNES *kdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
 if (!*kdm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*kdm),DMSNES_CLASSID,1);
  if (--((PetscObject)(*kdm))->refct > 0) {*kdm = 0; PetscFunctionReturn(0);}
  if ((*kdm)->ops->destroy) {ierr = ((*kdm)->ops->destroy)(*kdm);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(kdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESCreate"
static PetscErrorCode DMSNESCreate(MPI_Comm comm,DMSNES *kdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = SNESInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscHeaderCreate(*kdm, _p_DMSNES, struct _DMSNESOps, DMSNES_CLASSID, -1, "DMSNES", "DMSNES", "DMSNES", comm, DMSNESDestroy, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMemzero((*kdm)->ops, sizeof(struct _DMSNESOps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHook_DMSNES"
/* Attaches the DMSNES to the coarse level.
 * Under what conditions should we copy versus duplicate?
 */
static PetscErrorCode DMCoarsenHook_DMSNES(DM dm,DM dmc,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCopyDMSNES(dm,dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestrictHook_DMSNES"
/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMRestrictHook_DMSNES(DM dm,Mat Restrict,Vec rscale,Mat Inject,DM dmc,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefineHook_DMSNES"
static PetscErrorCode DMRefineHook_DMSNES(DM dm,DM dmf,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCopyDMSNES(dm,dmf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMInterpolateHook_DMSNES"
/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMInterpolateHook_DMSNES(DM dm,Mat Interp,DM dmf,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESCopy"
/*@C
   DMSNESCopy - copies the information in a DMSNES to another DMSNES

   Not Collective

   Input Argument:
+  kdm - Original DMSNES
-  nkdm - DMSNES to receive the data, should have been created with DMSNESCreate()

   Level: developer

.seealso: DMSNESCreate(), DMSNESDestroy()
@*/
PetscErrorCode DMSNESCopy(DMSNES kdm,DMSNES nkdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(kdm,DMSNES_CLASSID,1);
  PetscValidHeaderSpecific(nkdm,DMSNES_CLASSID,2);
  nkdm->ops->computefunction       = kdm->ops->computefunction;
  nkdm->ops->computegs             = kdm->ops->computegs;
  nkdm->ops->computeobjective      = kdm->ops->computeobjective;
  nkdm->ops->computepjacobian      = kdm->ops->computepjacobian;
  nkdm->ops->computepfunction      = kdm->ops->computepfunction;
  nkdm->ops->computeblockfunction  = kdm->ops->computeblockfunction;
  nkdm->ops->computeblockjacobian  = kdm->ops->computeblockjacobian;
  nkdm->ops->destroy               = kdm->ops->destroy;
  nkdm->ops->duplicate             = kdm->ops->duplicate;

  nkdm->functionctx      = kdm->functionctx;
  nkdm->gsctx            = kdm->gsctx;
  nkdm->pctx             = kdm->pctx;
  nkdm->jacobianctx      = kdm->jacobianctx;
  nkdm->objectivectx     = kdm->objectivectx;
  nkdm->blockfunctionctx = kdm->blockfunctionctx;
  nkdm->blockjacobianctx = kdm->blockjacobianctx;
  nkdm->data             = kdm->data;

  /*
  nkdm->fortran_func_pointers[0] = kdm->fortran_func_pointers[0];
  nkdm->fortran_func_pointers[1] = kdm->fortran_func_pointers[1];
  nkdm->fortran_func_pointers[2] = kdm->fortran_func_pointers[2];
  */

  /* implementation specific copy hooks */
  if (kdm->ops->duplicate) {ierr = (*kdm->ops->duplicate)(kdm,nkdm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetDMSNES"
/*@C
   DMGetDMSNES - get read-only private DMSNES context from a DM

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Argument:
.  snesdm - private DMSNES context

   Level: developer

   Notes:
   Use DMGetDMSNESWrite() if write access is needed. The DMSNESSetXXX API should be used wherever possible.

.seealso: DMGetDMSNESWrite()
@*/
PetscErrorCode DMGetDMSNES(DM dm,DMSNES *snesdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *snesdm = (DMSNES) dm->dmsnes;
  if (!*snesdm) {
    ierr = PetscInfo(dm,"Creating new DMSNES\n");CHKERRQ(ierr);
    ierr = DMSNESCreate(((PetscObject)dm)->comm,snesdm);CHKERRQ(ierr);
    dm->dmsnes = (PetscObject) *snesdm;
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_DMSNES,DMRestrictHook_DMSNES,PETSC_NULL);CHKERRQ(ierr);
    ierr = DMRefineHookAdd(dm,DMRefineHook_DMSNES,DMInterpolateHook_DMSNES,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMGetDMSNESWrite"
/*@C
   DMGetDMSNESWrite - get write access to private DMSNES context from a DM

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Argument:
.  snesdm - private DMSNES context

   Level: developer

.seealso: DMGetDMSNES()
@*/
PetscErrorCode DMGetDMSNESWrite(DM dm,DMSNES *snesdm)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (!sdm->originaldm) sdm->originaldm = dm;
  if (sdm->originaldm != dm) {  /* Copy on write */
    DMSNES          oldsdm = sdm;
    ierr = PetscInfo(dm,"Copying DMSNES due to write\n");CHKERRQ(ierr);
    ierr = DMSNESCreate(((PetscObject)dm)->comm,&sdm);CHKERRQ(ierr);
    ierr = DMSNESCopy(oldsdm,sdm);CHKERRQ(ierr);
    ierr = DMSNESDestroy((DMSNES*)&dm->dmsnes);CHKERRQ(ierr);
    dm->dmsnes = (PetscObject)sdm;
  }
  *snesdm = sdm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCopyDMSNES"
/*@C
   DMCopyDMSNES - copies a DM context to a new DM
 
   Logically Collective

   Input Arguments:
+  dmsrc - DM to obtain context from
-  dmdest - DM to add context to

   Level: developer

   Note:
   The context is copied by reference. This function does not ensure that a context exists.

.seealso: DMGetDMSNES(), SNESSetDM()
@*/
PetscErrorCode DMCopyDMSNES(DM dmsrc,DM dmdest)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmdest,DM_CLASSID,2);
  ierr = DMSNESDestroy((DMSNES*)&dmdest->dmsnes);CHKERRQ(ierr);
  dmdest->dmsnes = dmsrc->dmsnes;
  ierr = PetscObjectReference(dmdest->dmsnes);CHKERRQ(ierr);
  ierr = DMCoarsenHookAdd(dmdest,DMCoarsenHook_DMSNES,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMRefineHookAdd(dmdest,DMRefineHook_DMSNES,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetFunction"
/*@C
   DMSNESSetFunction - set SNES residual evaluation function

   Not Collective

   Input Arguments:
+  dm - DM to be used with SNES
.  func - residual evaluation function, see SNESSetFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESSetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian()
@*/
PetscErrorCode DMSNESSetFunction(DM dm,PetscErrorCode (*func)(SNES,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (func || ctx) {
    ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  }
  if (func) sdm->ops->computefunction = func;
  if (ctx)  sdm->functionctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESGetFunction"
/*@C
   DMSNESGetFunction - get SNES residual evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Arguments:
+  func - residual evaluation function, see SNESSetFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMSNESSetContext(), DMSNESSetFunction(), SNESSetFunction()
@*/
PetscErrorCode DMSNESGetFunction(DM dm,PetscErrorCode (**func)(SNES,Vec,Vec,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (func) *func = sdm->ops->computefunction;
  if (ctx)  *ctx = sdm->functionctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetObjective"
/*@C
   DMSNESSetObjective - set SNES objective evaluation function

   Not Collective

   Input Arguments:
+  dm - DM to be used with SNES
.  func - residual evaluation function, see SNESSetObjective() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

.seealso: DMSNESSetContext(), SNESGetObjective(), DMSNESSetFunction()
@*/
PetscErrorCode DMSNESSetObjective(DM dm,SNESObjective func,void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (func || ctx) {
    ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  }
  if (func) sdm->ops->computeobjective = func;
  if (ctx)  sdm->objectivectx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESGetObjective"
/*@C
   DMSNESGetObjective - get SNES objective evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Arguments:
+  func - residual evaluation function, see SNESSetObjective() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMSNESSetContext(), DMSNESSetObjective(), SNESSetFunction()
@*/
PetscErrorCode DMSNESGetObjective(DM dm,SNESObjective *func,void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (func) *func = sdm->ops->computeobjective;
  if (ctx)  *ctx = sdm->objectivectx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetGS"
/*@C
   DMSNESSetGS - set SNES Gauss-Seidel relaxation function

   Not Collective

   Input Argument:
+  dm - DM to be used with SNES
.  func - relaxation function, see SNESSetGS() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESSetGS() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian(), DMSNESSetFunction()
@*/
PetscErrorCode DMSNESSetGS(DM dm,PetscErrorCode (*func)(SNES,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (func || ctx) {
    ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  }
  if (func) sdm->ops->computegs = func;
  if (ctx)  sdm->gsctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESGetGS"
/*@C
   DMSNESGetGS - get SNES Gauss-Seidel relaxation function

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Arguments:
+  func - relaxation function, see SNESSetGS() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetGS() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMSNESSetContext(), SNESGetGS(), DMSNESGetJacobian(), DMSNESGetFunction()
@*/
PetscErrorCode DMSNESGetGS(DM dm,PetscErrorCode (**func)(SNES,Vec,Vec,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (func) *func = sdm->ops->computegs;
  if (ctx)  *ctx = sdm->gsctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetJacobian"
/*@C
   DMSNESSetJacobian - set SNES Jacobian evaluation function

   Not Collective

   Input Argument:
+  dm - DM to be used with SNES
.  func - Jacobian evaluation function, see SNESSetJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESSetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESGetJacobian(), SNESSetJacobian()
@*/
PetscErrorCode DMSNESSetJacobian(DM dm,PetscErrorCode (*func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (func || ctx) {
    ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  }
  if (func) sdm->ops->computejacobian = func;
  if (ctx)  sdm->jacobianctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESGetJacobian"
/*@C
   DMSNESGetJacobian - get SNES Jacobian evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Arguments:
+  func - Jacobian evaluation function, see SNESSetJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian()
@*/
PetscErrorCode DMSNESGetJacobian(DM dm,PetscErrorCode (**func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (func) *func = sdm->ops->computejacobian;
  if (ctx)  *ctx = sdm->jacobianctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetPicard"
/*@C
   DMSNESSetPicard - set SNES Picard iteration matrix and RHS evaluation functions.

   Not Collective

   Input Argument:
+  dm - DM to be used with SNES
.  func - RHS evaluation function, see SNESSetFunction() for calling sequence
.  pjac - Picard matrix evaluation function, see SNESSetJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

.seealso: SNESSetPicard(), DMSNESSetFunction(), DMSNESSetJacobian()
@*/
PetscErrorCode DMSNESSetPicard(DM dm,PetscErrorCode (*pfunc)(SNES,Vec,Vec,void*),PetscErrorCode (*pjac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (pfunc) sdm->ops->computepfunction = pfunc;
  if (pjac)  sdm->ops->computepjacobian = pjac;
  if (ctx)   sdm->pctx             = ctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMSNESGetPicard"
/*@C
   DMSNESGetPicard - get SNES Picard iteration evaluation functions

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Arguments:
+  pfunc - Jacobian evaluation function, see SNESSetJacobian() for calling sequence
.  pjac  - RHS evaluation function, see SNESSetFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian()
@*/
PetscErrorCode DMSNESGetPicard(DM dm,PetscErrorCode (**pfunc)(SNES,Vec,Vec,void*),PetscErrorCode (**pjac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (pfunc) *pfunc = sdm->ops->computepfunction;
  if (pjac) *pjac   = sdm->ops->computepjacobian;
  if (ctx)  *ctx    = sdm->pctx;
  PetscFunctionReturn(0);
}

/* block functions */

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetBlockFunction"
/*@C
   DMSNESSetBlockFunction - set SNES residual evaluation function

   Not Collective

   Input Arguments:
+  dm - DM to be used with SNES
.  func - residual evaluation function, see SNESSetFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: developer

   Note:
   Mostly for use in DM implementations and transferred to a block function rather than being called from here.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian()
@*/
PetscErrorCode DMSNESSetBlockFunction(DM dm,PetscErrorCode (*func)(SNES,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (func || ctx) {
    ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  }
  if (func) sdm->ops->computeblockfunction = func;
  if (ctx)  sdm->blockfunctionctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESGetBlockFunction"
/*@C
   DMSNESGetBlockFunction - get SNES residual evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Arguments:
+  func - residual evaluation function, see SNESSetFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: developer

.seealso: DMSNESSetContext(), DMSNESSetFunction(), SNESSetFunction()
@*/
PetscErrorCode DMSNESGetBlockFunction(DM dm,PetscErrorCode (**func)(SNES,Vec,Vec,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (func) *func = sdm->ops->computeblockfunction;
  if (ctx)  *ctx = sdm->blockfunctionctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMSNESSetBlockJacobian"
/*@C
   DMSNESSetJacobian - set SNES Jacobian evaluation function

   Not Collective

   Input Argument:
+  dm - DM to be used with SNES
.  func - Jacobian evaluation function, see SNESSetJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   Mostly for use in DM implementations and transferred to a block function rather than being called from here.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESGetJacobian(), SNESSetJacobian()
@*/
PetscErrorCode DMSNESSetBlockJacobian(DM dm,PetscErrorCode (*func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (func || ctx) {
    ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  }
  if (func) sdm->ops->computeblockjacobian = func;
  if (ctx)  sdm->blockjacobianctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESGetBlockJacobian"
/*@C
   DMSNESGetBlockJacobian - get SNES Jacobian evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Arguments:
+  func - Jacobian evaluation function, see SNESSetJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian()
@*/
PetscErrorCode DMSNESGetBlockJacobian(DM dm,PetscErrorCode (**func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (func) *func = sdm->ops->computeblockjacobian;
  if (ctx)  *ctx = sdm->blockjacobianctx;
  PetscFunctionReturn(0);
}
