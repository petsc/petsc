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
#define __FUNCT__ "DMSNESLoad"
PetscErrorCode DMSNESLoad(DMSNES kdm,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryRead(viewer,&kdm->ops->computefunction,1,PETSC_FUNCTION);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&kdm->ops->computejacobian,1,PETSC_FUNCTION);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESView"
PetscErrorCode DMSNESView(DMSNES kdm,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii,isbinary;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (isascii) {
#if defined(PETSC_SERIALIZE_FUNCTIONS)
    const char *fname;

    ierr = PetscFPTFind(kdm->ops->computefunction,&fname);CHKERRQ(ierr);
    if (fname) {
      ierr = PetscViewerASCIIPrintf(viewer,"Function used by SNES: %s\n",fname);CHKERRQ(ierr);
    }
    ierr = PetscFPTFind(kdm->ops->computejacobian,&fname);CHKERRQ(ierr);
    if (fname) {
      ierr = PetscViewerASCIIPrintf(viewer,"Jacobian function used by SNES: %s\n",fname);CHKERRQ(ierr);
    }
#endif
  } else if (isbinary) {
    struct {
      PetscErrorCode (*func)(SNES,Vec,Vec,void*);
    } funcstruct;
    struct {
      PetscErrorCode (*jac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
    } jacstruct;
    funcstruct.func = kdm->ops->computefunction;
    jacstruct.jac   = kdm->ops->computejacobian;
    ierr = PetscViewerBinaryWrite(viewer,&funcstruct,1,PETSC_FUNCTION,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&jacstruct,1,PETSC_FUNCTION,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESCreate"
static PetscErrorCode DMSNESCreate(MPI_Comm comm,DMSNES *kdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*kdm, _p_DMSNES, struct _DMSNESOps, DMSNES_CLASSID,  "DMSNES", "DMSNES", "DMSNES", comm, DMSNESDestroy, DMSNESView);CHKERRQ(ierr);
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
#define __FUNCT__ "DMSubDomainHook_DMSNES"
/* Attaches the DMSNES to the subdomain. */
static PetscErrorCode DMSubDomainHook_DMSNES(DM dm,DM subdm,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCopyDMSNES(dm,subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSubDomainRestrictHook_DMSNES"
/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMSubDomainRestrictHook_DMSNES(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
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
  nkdm->ops->computefunction  = kdm->ops->computefunction;
  nkdm->ops->computejacobian  = kdm->ops->computejacobian;
  nkdm->ops->computegs        = kdm->ops->computegs;
  nkdm->ops->computeobjective = kdm->ops->computeobjective;
  nkdm->ops->computepjacobian = kdm->ops->computepjacobian;
  nkdm->ops->computepfunction = kdm->ops->computepfunction;
  nkdm->ops->destroy          = kdm->ops->destroy;
  nkdm->ops->duplicate        = kdm->ops->duplicate;

  nkdm->functionctx  = kdm->functionctx;
  nkdm->gsctx        = kdm->gsctx;
  nkdm->pctx         = kdm->pctx;
  nkdm->jacobianctx  = kdm->jacobianctx;
  nkdm->objectivectx = kdm->objectivectx;
  nkdm->data         = kdm->data;

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
    ierr = DMSNESCreate(PetscObjectComm((PetscObject)dm),snesdm);CHKERRQ(ierr);

    dm->dmsnes = (PetscObject) *snesdm;

    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_DMSNES,DMRestrictHook_DMSNES,NULL);CHKERRQ(ierr);
    ierr = DMRefineHookAdd(dm,DMRefineHook_DMSNES,DMInterpolateHook_DMSNES,NULL);CHKERRQ(ierr);
    ierr = DMSubDomainHookAdd(dm,DMSubDomainHook_DMSNES,DMSubDomainRestrictHook_DMSNES,NULL);CHKERRQ(ierr);
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
    DMSNES oldsdm = sdm;
    ierr       = PetscInfo(dm,"Copying DMSNES due to write\n");CHKERRQ(ierr);
    ierr       = DMSNESCreate(PetscObjectComm((PetscObject)dm),&sdm);CHKERRQ(ierr);
    ierr       = DMSNESCopy(oldsdm,sdm);CHKERRQ(ierr);
    ierr       = DMSNESDestroy((DMSNES*)&dm->dmsnes);CHKERRQ(ierr);
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
  ierr = DMCoarsenHookAdd(dmdest,DMCoarsenHook_DMSNES,NULL,NULL);CHKERRQ(ierr);
  ierr = DMRefineHookAdd(dmdest,DMRefineHook_DMSNES,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSubDomainHookAdd(dmdest,DMSubDomainHook_DMSNES,DMSubDomainRestrictHook_DMSNES,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetFunction"
/*@C
   DMSNESSetFunction - set SNES residual evaluation function

   Not Collective

   Input Arguments:
+  dm - DM to be used with SNES
.  SNESFunction - residual evaluation function
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESSetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian(), SNESFunction
@*/
PetscErrorCode DMSNESSetFunction(DM dm,PetscErrorCode (*SNESFunction)(SNES,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (SNESFunction || ctx) {
    ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  }
  if (SNESFunction) sdm->ops->computefunction = SNESFunction;
  if (ctx) sdm->functionctx = ctx;
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
+  SNESFunction - residual evaluation function
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMSNESSetContext(), DMSNESSetFunction(), SNESSetFunction(), SNESFunction
@*/
PetscErrorCode DMSNESGetFunction(DM dm,PetscErrorCode (**SNESFunction)(SNES,Vec,Vec,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (SNESFunction) *SNESFunction = sdm->ops->computefunction;
  if (ctx) *ctx = sdm->functionctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetObjective"
/*@C
   DMSNESSetObjective - set SNES objective evaluation function

   Not Collective

   Input Arguments:
+  dm - DM to be used with SNES
.  SNESObjectiveFunction - residual evaluation function
-  ctx - context for residual evaluation

   Level: advanced

.seealso: DMSNESSetContext(), SNESGetObjective(), DMSNESSetFunction()
@*/
PetscErrorCode DMSNESSetObjective(DM dm,PetscErrorCode (*SNESObjectiveFunction)(SNES,Vec,PetscReal*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (SNESObjectiveFunction || ctx) {
    ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  }
  if (SNESObjectiveFunction) sdm->ops->computeobjective = SNESObjectiveFunction;
  if (ctx) sdm->objectivectx = ctx;
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
+  SNESObjectiveFunction- residual evaluation function
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMSNESSetContext(), DMSNESSetObjective(), SNESSetFunction()
@*/
PetscErrorCode DMSNESGetObjective(DM dm,PetscErrorCode (**SNESObjectiveFunction)(SNES,Vec,PetscReal*,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (SNESObjectiveFunction) *SNESObjectiveFunction = sdm->ops->computeobjective;
  if (ctx) *ctx = sdm->objectivectx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetGS"
/*@C
   DMSNESSetGS - set SNES Gauss-Seidel relaxation function

   Not Collective

   Input Argument:
+  dm - DM to be used with SNES
.  SNESGSFunction - relaxation function
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESSetGS() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian(), DMSNESSetFunction(), SNESGSFunction
@*/
PetscErrorCode DMSNESSetGS(DM dm,PetscErrorCode (*SNESGSFunction)(SNES,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (SNESGSFunction || ctx) {
    ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  }
  if (SNESGSFunction) sdm->ops->computegs = SNESGSFunction;
  if (ctx) sdm->gsctx = ctx;
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
+  SNESGSFunction - relaxation function which performs Gauss-Seidel sweeps
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetGS() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMSNESSetContext(), SNESGetGS(), DMSNESGetJacobian(), DMSNESGetFunction(), SNESGSFunction
@*/
PetscErrorCode DMSNESGetGS(DM dm,PetscErrorCode (**SNESGSFunction)(SNES,Vec,Vec,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (SNESGSFunction) *SNESGSFunction = sdm->ops->computegs;
  if (ctx) *ctx = sdm->gsctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetJacobian"
/*@C
   DMSNESSetJacobian - set SNES Jacobian evaluation function

   Not Collective

   Input Argument:
+  dm - DM to be used with SNES
.  SNESJacobianFunction - Jacobian evaluation function
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESSetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESGetJacobian(), SNESSetJacobian(), SNESJacobianFunction
@*/
PetscErrorCode DMSNESSetJacobian(DM dm,PetscErrorCode (*SNESJacobianFunction)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (SNESJacobianFunction || ctx) {
    ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  }
  if (SNESJacobianFunction) sdm->ops->computejacobian = SNESJacobianFunction;
  if (ctx) sdm->jacobianctx = ctx;
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
+  SNESJacobianFunction - Jacobian evaluation function
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian(), SNESJacobianFunction
@*/
PetscErrorCode DMSNESGetJacobian(DM dm,PetscErrorCode (**SNESJacobianFunction)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (SNESJacobianFunction) *SNESJacobianFunction = sdm->ops->computejacobian;
  if (ctx) *ctx = sdm->jacobianctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetPicard"
/*@C
   DMSNESSetPicard - set SNES Picard iteration matrix and RHS evaluation functions.

   Not Collective

   Input Argument:
+  dm - DM to be used with SNES
.  SNESFunction - RHS evaluation function
.  SNESJacobianFunction - Picard matrix evaluation function
-  ctx - context for residual evaluation

   Level: advanced

.seealso: SNESSetPicard(), DMSNESSetFunction(), DMSNESSetJacobian()
@*/
PetscErrorCode DMSNESSetPicard(DM dm,PetscErrorCode (*SNESFunction)(SNES,Vec,Vec,void*),PetscErrorCode (*SNESJacobianFunction)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (SNESFunction) sdm->ops->computepfunction = SNESFunction;
  if (SNESJacobianFunction) sdm->ops->computepjacobian = SNESJacobianFunction;
  if (ctx) sdm->pctx = ctx;
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
+  SNESFunction - Jacobian evaluation function;
.  SNESJacobianFunction  - RHS evaluation function;
-  ctx - context for residual evaluation

   Level: advanced

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian()
@*/
PetscErrorCode DMSNESGetPicard(DM dm,PetscErrorCode (**SNESFunction)(SNES,Vec,Vec,void*),PetscErrorCode (**SNESJacobianFunction)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void **ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm,&sdm);CHKERRQ(ierr);
  if (SNESFunction) *SNESFunction = sdm->ops->computepfunction;
  if (SNESJacobianFunction) *SNESJacobianFunction = sdm->ops->computepjacobian;
  if (ctx) *ctx = sdm->pctx;
  PetscFunctionReturn(0);
}
