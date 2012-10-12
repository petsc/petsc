#include <petsc-private/tsimpl.h>     /*I "petscts.h" I*/
#include <petsc-private/dmimpl.h>     /*I "petscdm.h" I*/


#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHook_TSDM"
/* Attaches the SNESDM to the coarse level.
 * Under what conditions should we copy versus duplicate?
 */
static PetscErrorCode DMCoarsenHook_TSDM(DM dm,DM dmc,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMTSCopyContext(dm,dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestrictHook_TSDM"
/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMRestrictHook_TSDM(DM dm,Mat Restrict,Vec rscale,Mat Inject,DM dmc,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscContainerDestroy_TSDM"
static PetscErrorCode PetscContainerDestroy_TSDM(void *ctx)
{
  PetscErrorCode ierr;
  TSDM tsdm = (TSDM)ctx;

  PetscFunctionBegin;
  if (tsdm->destroy) {ierr = (*tsdm->destroy)(tsdm);CHKERRQ(ierr);}
  ierr = PetscFree(tsdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSGetContext"
/*@C
   DMTSGetContext - get read-only private TSDM context from a DM

   Not Collective

   Input Argument:
.  dm - DM to be used with TS

   Output Argument:
.  tsdm - private TSDM context

   Level: developer

   Notes:
   Use DMTSGetContextWrite() if write access is needed. The DMTSSetXXX API should be used wherever possible.

.seealso: DMTSGetContextWrite()
@*/
PetscErrorCode DMTSGetContext(DM dm,TSDM *tsdm)
{
  PetscErrorCode ierr;
  PetscContainer container;
  TSDM           tsdmnew;


  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)dm,"TSDM",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void**)tsdm);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(dm,"Creating new TSDM\n");CHKERRQ(ierr);
    ierr = PetscContainerCreate(((PetscObject)dm)->comm,&container);CHKERRQ(ierr);
    ierr = PetscNewLog(dm,struct _n_TSDM,&tsdmnew);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,tsdmnew);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_TSDM);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"TSDM",(PetscObject)container);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_TSDM,DMRestrictHook_TSDM,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscContainerGetPointer(container,(void**)tsdm);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSGetContextWrite"
/*@C
   DMTSGetContextWrite - get write access to private TSDM context from a DM

   Not Collective

   Input Argument:
.  dm - DM to be used with TS

   Output Argument:
.  tsdm - private TSDM context

   Level: developer

.seealso: DMTSGetContext()
@*/
PetscErrorCode DMTSGetContextWrite(DM dm,TSDM *tsdm)
{
  PetscErrorCode ierr;
  TSDM           sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContext(dm,&sdm);CHKERRQ(ierr);
  if (!sdm->originaldm) sdm->originaldm = dm;
  if (sdm->originaldm != dm) {  /* Copy on write */
    PetscContainer container;
    TSDM         oldsdm = sdm;
    ierr = PetscInfo(dm,"Copying TSDM due to write\n");CHKERRQ(ierr);
    ierr = PetscContainerCreate(((PetscObject)dm)->comm,&container);CHKERRQ(ierr);
    ierr = PetscNewLog(dm,struct _n_TSDM,&sdm);CHKERRQ(ierr);
    ierr = PetscMemcpy(sdm,oldsdm,sizeof(*sdm));CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,sdm);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_TSDM);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"TSDM",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }
  *tsdm = sdm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSCopyContext"
/*@C
   DMTSCopyContext - copies a DM context to a new DM

   Logically Collective

   Input Arguments:
+  dmsrc - DM to obtain context from
-  dmdest - DM to add context to

   Level: developer

   Note:
   The context is copied by reference. This function does not ensure that a context exists.

.seealso: DMTSGetContext(), TSSetDM()
@*/
PetscErrorCode DMTSCopyContext(DM dmsrc,DM dmdest)
{
  PetscErrorCode ierr;
  PetscContainer container;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmdest,DM_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)dmsrc,"TSDM",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscObjectCompose((PetscObject)dmdest,"TSDM",(PetscObject)container);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(dmdest,DMCoarsenHook_TSDM,DMRestrictHook_TSDM,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSSetIFunction"
/*@C
   DMTSSetIFunction - set TS implicit function evaluation function

   Not Collective

   Input Arguments:
+  dm - DM to be used with TS
.  func - function evaluation function, see TSSetIFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSSetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSSetIFunction(DM dm,TSIFunction func,void *ctx)
{
  PetscErrorCode ierr;
  TSDM tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContextWrite(dm,&tsdm);CHKERRQ(ierr);
  if (func) tsdm->ifunction = func;
  if (ctx)  tsdm->ifunctionctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSGetIFunction"
/*@C
   DMTSGetIFunction - get TS implicit residual evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with TS

   Output Arguments:
+  func - function evaluation function, see TSSetIFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSGetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMTSSetContext(), DMTSSetFunction(), TSSetFunction()
@*/
PetscErrorCode DMTSGetIFunction(DM dm,TSIFunction *func,void **ctx)
{
  PetscErrorCode ierr;
  TSDM tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContext(dm,&tsdm);CHKERRQ(ierr);
  if (func) *func = tsdm->ifunction;
  if (ctx)  *ctx = tsdm->ifunctionctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMTSSetRHSFunction"
/*@C
   DMTSSetRHSFunction - set TS explicit residual evaluation function

   Not Collective

   Input Arguments:
+  dm - DM to be used with TS
.  func - RHS function evaluation function, see TSSetRHSFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSSetRSHFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSSetRHSFunction(DM dm,TSRHSFunction func,void *ctx)
{
  PetscErrorCode ierr;
  TSDM           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContextWrite(dm,&tsdm);CHKERRQ(ierr);
  if (func) tsdm->rhsfunction = func;
  if (ctx)  tsdm->rhsfunctionctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSGetSolutionFunction"
/*@C
   DMTSGetSolutionFunction - gets the TS solution evaluation function

   Not Collective

   Input Arguments:
.  dm - DM to be used with TS

   Output Parameters:
+  func - solution function evaluation function, see TSSetSolution() for calling sequence
-  ctx - context for solution evaluation

   Level: advanced

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSGetSolutionFunction(DM dm,TSSolutionFunction *func,void **ctx)
{
  PetscErrorCode ierr;
  TSDM           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContext(dm,&tsdm);CHKERRQ(ierr);
  if (func) *func = tsdm->solution;
  if (ctx)  *ctx  = tsdm->solutionctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSSetSolutionFunction"
/*@C
   DMTSSetSolutionFunction - set TS solution evaluation function

   Not Collective

   Input Arguments:
+  dm - DM to be used with TS
.  func - solution function evaluation function, see TSSetSolution() for calling sequence
-  ctx - context for solution evaluation

   Level: advanced

   Note:
   TSSetSolutionFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSSetSolutionFunction(DM dm,TSSolutionFunction func,void *ctx)
{
  PetscErrorCode ierr;
  TSDM           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContextWrite(dm,&tsdm);CHKERRQ(ierr);
  if (func) tsdm->solution    = func;
  if (ctx)  tsdm->solutionctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSGetRHSFunction"
/*@C
   DMTSGetRHSFunction - get TS explicit residual evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with TS

   Output Arguments:
+  func - residual evaluation function, see TSSetRHSFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSGetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMTSSetContext(), DMTSSetFunction(), TSSetFunction()
@*/
PetscErrorCode DMTSGetRHSFunction(DM dm,TSRHSFunction *func,void **ctx)
{
  PetscErrorCode ierr;
  TSDM tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContext(dm,&tsdm);CHKERRQ(ierr);
  if (func) *func = tsdm->rhsfunction;
  if (ctx)  *ctx = tsdm->rhsfunctionctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSSetIJacobian"
/*@C
   DMTSSetIJacobian - set TS Jacobian evaluation function

   Not Collective

   Input Argument:
+  dm - DM to be used with TS
.  func - Jacobian evaluation function, see TSSetIJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSSetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSGetJacobian(), TSSetJacobian()
@*/
PetscErrorCode DMTSSetIJacobian(DM dm,TSIJacobian func,void *ctx)
{
  PetscErrorCode ierr;
  TSDM sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContextWrite(dm,&sdm);CHKERRQ(ierr);
  if (func) sdm->ijacobian = func;
  if (ctx)  sdm->ijacobianctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSGetIJacobian"
/*@C
   DMTSGetIJacobian - get TS Jacobian evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with TS

   Output Arguments:
+  func - Jacobian evaluation function, see TSSetIJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSGetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSGetIJacobian(DM dm,TSIJacobian *func,void **ctx)
{
  PetscErrorCode ierr;
  TSDM tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContext(dm,&tsdm);CHKERRQ(ierr);
  if (func) *func = tsdm->ijacobian;
  if (ctx)  *ctx = tsdm->ijacobianctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMTSSetRHSJacobian"
/*@C
   DMTSSetRHSJacobian - set TS Jacobian evaluation function

   Not Collective

   Input Argument:
+  dm - DM to be used with TS
.  func - Jacobian evaluation function, see TSSetRHSJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSSetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSGetJacobian(), TSSetJacobian()
@*/
PetscErrorCode DMTSSetRHSJacobian(DM dm,TSRHSJacobian func,void *ctx)
{
  PetscErrorCode ierr;
  TSDM tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContextWrite(dm,&tsdm);CHKERRQ(ierr);
  if (func) tsdm->rhsjacobian = func;
  if (ctx)  tsdm->rhsjacobianctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSGetRHSJacobian"
/*@C
   DMTSGetRHSJacobian - get TS Jacobian evaluation function

   Not Collective

   Input Argument:
.  dm - DM to be used with TS

   Output Arguments:
+  func - Jacobian evaluation function, see TSSetRHSJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSGetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSGetRHSJacobian(DM dm,TSRHSJacobian *func,void **ctx)
{
  PetscErrorCode ierr;
  TSDM tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContext(dm,&tsdm);CHKERRQ(ierr);
  if (func) *func = tsdm->rhsjacobian;
  if (ctx)  *ctx = tsdm->rhsjacobianctx;
  PetscFunctionReturn(0);
}
