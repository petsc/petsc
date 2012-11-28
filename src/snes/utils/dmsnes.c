#include <petsc-private/snesimpl.h>   /*I "petscsnes.h" I*/
#include <petsc-private/dmimpl.h>     /*I "petscdm.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PetscContainerDestroy_DMSNES"
static PetscErrorCode PetscContainerDestroy_DMSNES(void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm = (DMSNES)ctx;

  PetscFunctionBegin;
  if (sdm->destroy) {ierr = (*sdm->destroy)(sdm);CHKERRQ(ierr);}
  ierr = PetscFree(sdm);CHKERRQ(ierr);
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
  PetscContainer container;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)dm,"DMSNES",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void**)snesdm);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(dm,"Creating new DMSNES\n");CHKERRQ(ierr);
    ierr = PetscContainerCreate(((PetscObject)dm)->comm,&container);CHKERRQ(ierr);
    ierr = PetscNewLog(dm,struct _n_DMSNES,&sdm);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,sdm);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_DMSNES);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"DMSNES",(PetscObject)container);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_DMSNES,DMRestrictHook_DMSNES,PETSC_NULL);CHKERRQ(ierr);
    ierr = DMRefineHookAdd(dm,DMRefineHook_DMSNES,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscContainerGetPointer(container,(void**)snesdm);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
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
    PetscContainer container;
    DMSNES         oldsdm = sdm;
    ierr = PetscInfo(dm,"Copying DMSNES due to write\n");CHKERRQ(ierr);
    ierr = PetscContainerCreate(((PetscObject)dm)->comm,&container);CHKERRQ(ierr);
    ierr = PetscNewLog(dm,struct _n_DMSNES,&sdm);CHKERRQ(ierr);
    ierr = PetscMemcpy(sdm,oldsdm,sizeof(*sdm));CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,sdm);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_DMSNES);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"DMSNES",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
    /* implementation specific copy hooks */
    if (sdm->duplicate) {ierr = (*sdm->duplicate)(oldsdm,dm);CHKERRQ(ierr);}
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
  PetscContainer container;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmdest,DM_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)dmsrc,"DMSNES",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscObjectCompose((PetscObject)dmdest,"DMSNES",(PetscObject)container);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(dmdest,DMCoarsenHook_DMSNES,DMRestrictHook_DMSNES,PETSC_NULL);CHKERRQ(ierr);
    ierr = DMRefineHookAdd(dmdest,DMRefineHook_DMSNES,DMInterpolateHook_DMSNES,PETSC_NULL);CHKERRQ(ierr);
  }
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
  if (func) sdm->computefunction = func;
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
  if (func) *func = sdm->computefunction;
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
  if (func) sdm->computeobjective = func;
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
  if (func) *func = sdm->computeobjective;
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
  if (func) sdm->computegs = func;
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
  if (func) *func = sdm->computegs;
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
  if (func) sdm->computejacobian = func;
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
  if (func) *func = sdm->computejacobian;
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
  if (pfunc) sdm->computepfunction = pfunc;
  if (pjac)  sdm->computepjacobian = pjac;
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
  if (pfunc) *pfunc = sdm->computepfunction;
  if (pjac) *pjac   = sdm->computepjacobian;
  if (ctx)  *ctx    = sdm->pctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetUpLegacy"
/* Sets up calling of legacy DM routines */
PetscErrorCode DMSNESSetUpLegacy(DM dm)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (!sdm->computefunction) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"Function never provided to SNES object");
  if (!sdm->computejacobian) {
    ierr = DMSNESSetJacobian(dm,SNESDefaultComputeJacobianColor,PETSC_NULL);CHKERRQ(ierr);
  }
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
  if (func) sdm->computeblockfunction = func;
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
  if (func) *func = sdm->computeblockfunction;
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
  if (func) sdm->computeblockjacobian = func;
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
  if (func) *func = sdm->computeblockjacobian;
  if (ctx)  *ctx = sdm->blockjacobianctx;
  PetscFunctionReturn(0);
}
