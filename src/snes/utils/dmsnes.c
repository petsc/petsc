#include <petsc/private/snesimpl.h>   /*I "petscsnes.h" I*/
#include <petsc/private/dmimpl.h>     /*I "petscdm.h" I*/

static PetscErrorCode DMSNESDestroy(DMSNES *kdm)
{
  PetscFunctionBegin;
  if (!*kdm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*kdm),DMSNES_CLASSID,1);
  if (--((PetscObject)(*kdm))->refct > 0) {*kdm = NULL; PetscFunctionReturn(0);}
  if ((*kdm)->ops->destroy) PetscCall(((*kdm)->ops->destroy)(*kdm));
  PetscCall(PetscHeaderDestroy(kdm));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESLoad(DMSNES kdm,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryRead(viewer,&kdm->ops->computefunction,1,NULL,PETSC_FUNCTION));
  PetscCall(PetscViewerBinaryRead(viewer,&kdm->ops->computejacobian,1,NULL,PETSC_FUNCTION));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESView(DMSNES kdm,PetscViewer viewer)
{
  PetscBool      isascii,isbinary;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  if (isascii) {
#if defined(PETSC_SERIALIZE_FUNCTIONS) && defined(PETSC_SERIALIZE_FUNCTIONS_VIEW)
    const char *fname;

    PetscCall(PetscFPTFind(kdm->ops->computefunction,&fname));
    if (fname) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Function used by SNES: %s\n",fname));
    }
    PetscCall(PetscFPTFind(kdm->ops->computejacobian,&fname));
    if (fname) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Jacobian function used by SNES: %s\n",fname));
    }
#endif
  } else if (isbinary) {
    struct {
      PetscErrorCode (*func)(SNES,Vec,Vec,void*);
    } funcstruct;
    struct {
      PetscErrorCode (*jac)(SNES,Vec,Mat,Mat,void*);
    } jacstruct;
    funcstruct.func = kdm->ops->computefunction;
    jacstruct.jac   = kdm->ops->computejacobian;
    PetscCall(PetscViewerBinaryWrite(viewer,&funcstruct,1,PETSC_FUNCTION));
    PetscCall(PetscViewerBinaryWrite(viewer,&jacstruct,1,PETSC_FUNCTION));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSNESCreate(MPI_Comm comm,DMSNES *kdm)
{
  PetscFunctionBegin;
  PetscCall(SNESInitializePackage());
  PetscCall(PetscHeaderCreate(*kdm, DMSNES_CLASSID,  "DMSNES", "DMSNES", "DMSNES", comm, DMSNESDestroy, DMSNESView));
  PetscFunctionReturn(0);
}

/* Attaches the DMSNES to the coarse level.
 * Under what conditions should we copy versus duplicate?
 */
static PetscErrorCode DMCoarsenHook_DMSNES(DM dm,DM dmc,void *ctx)
{
  PetscFunctionBegin;
  PetscCall(DMCopyDMSNES(dm,dmc));
  PetscFunctionReturn(0);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMRestrictHook_DMSNES(DM dm,Mat Restrict,Vec rscale,Mat Inject,DM dmc,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* Attaches the DMSNES to the subdomain. */
static PetscErrorCode DMSubDomainHook_DMSNES(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscCall(DMCopyDMSNES(dm,subdm));
  PetscFunctionReturn(0);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMSubDomainRestrictHook_DMSNES(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRefineHook_DMSNES(DM dm,DM dmf,void *ctx)
{
  PetscFunctionBegin;
  PetscCall(DMCopyDMSNES(dm,dmf));
  PetscFunctionReturn(0);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMInterpolateHook_DMSNES(DM dm,Mat Interp,DM dmf,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESCopy - copies the information in a DMSNES to another DMSNES

   Not Collective

   Input Parameters:
+  kdm - Original DMSNES
-  nkdm - DMSNES to receive the data, should have been created with DMSNESCreate()

   Level: developer

.seealso: DMSNESCreate(), DMSNESDestroy()
@*/
PetscErrorCode DMSNESCopy(DMSNES kdm,DMSNES nkdm)
{
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
  nkdm->originaldm   = kdm->originaldm;

  /*
  nkdm->fortran_func_pointers[0] = kdm->fortran_func_pointers[0];
  nkdm->fortran_func_pointers[1] = kdm->fortran_func_pointers[1];
  nkdm->fortran_func_pointers[2] = kdm->fortran_func_pointers[2];
  */

  /* implementation specific copy hooks */
  if (kdm->ops->duplicate) PetscCall((*kdm->ops->duplicate)(kdm,nkdm));
  PetscFunctionReturn(0);
}

/*@C
   DMGetDMSNES - get read-only private DMSNES context from a DM

   Not Collective

   Input Parameter:
.  dm - DM to be used with SNES

   Output Parameter:
.  snesdm - private DMSNES context

   Level: developer

   Notes:
   Use DMGetDMSNESWrite() if write access is needed. The DMSNESSetXXX API should be used wherever possible.

.seealso: DMGetDMSNESWrite()
@*/
PetscErrorCode DMGetDMSNES(DM dm,DMSNES *snesdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *snesdm = (DMSNES) dm->dmsnes;
  if (!*snesdm) {
    PetscCall(PetscInfo(dm,"Creating new DMSNES\n"));
    PetscCall(DMSNESCreate(PetscObjectComm((PetscObject)dm),snesdm));

    dm->dmsnes            = (PetscObject) *snesdm;
    (*snesdm)->originaldm = dm;
    PetscCall(DMCoarsenHookAdd(dm,DMCoarsenHook_DMSNES,DMRestrictHook_DMSNES,NULL));
    PetscCall(DMRefineHookAdd(dm,DMRefineHook_DMSNES,DMInterpolateHook_DMSNES,NULL));
    PetscCall(DMSubDomainHookAdd(dm,DMSubDomainHook_DMSNES,DMSubDomainRestrictHook_DMSNES,NULL));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMGetDMSNESWrite - get write access to private DMSNES context from a DM

   Not Collective

   Input Parameter:
.  dm - DM to be used with SNES

   Output Parameter:
.  snesdm - private DMSNES context

   Level: developer

.seealso: DMGetDMSNES()
@*/
PetscErrorCode DMGetDMSNESWrite(DM dm,DMSNES *snesdm)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNES(dm,&sdm));
  PetscCheck(sdm->originaldm,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"DMSNES has a NULL originaldm");
  if (sdm->originaldm != dm) {  /* Copy on write */
    DMSNES oldsdm = sdm;
    PetscCall(PetscInfo(dm,"Copying DMSNES due to write\n"));
    PetscCall(DMSNESCreate(PetscObjectComm((PetscObject)dm),&sdm));
    PetscCall(DMSNESCopy(oldsdm,sdm));
    PetscCall(DMSNESDestroy((DMSNES*)&dm->dmsnes));
    dm->dmsnes = (PetscObject)sdm;
    sdm->originaldm = dm;
  }
  *snesdm = sdm;
  PetscFunctionReturn(0);
}

/*@C
   DMCopyDMSNES - copies a DM context to a new DM

   Logically Collective

   Input Parameters:
+  dmsrc - DM to obtain context from
-  dmdest - DM to add context to

   Level: developer

   Note:
   The context is copied by reference. This function does not ensure that a context exists.

.seealso: DMGetDMSNES(), SNESSetDM()
@*/
PetscErrorCode DMCopyDMSNES(DM dmsrc,DM dmdest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmdest,DM_CLASSID,2);
  if (!dmdest->dmsnes) PetscCall(DMSNESCreate(PetscObjectComm((PetscObject) dmdest), (DMSNES *) &dmdest->dmsnes));
  PetscCall(DMSNESCopy((DMSNES) dmsrc->dmsnes, (DMSNES) dmdest->dmsnes));
  PetscCall(DMCoarsenHookAdd(dmdest,DMCoarsenHook_DMSNES,NULL,NULL));
  PetscCall(DMRefineHookAdd(dmdest,DMRefineHook_DMSNES,NULL,NULL));
  PetscCall(DMSubDomainHookAdd(dmdest,DMSubDomainHook_DMSNES,DMSubDomainRestrictHook_DMSNES,NULL));
  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetFunction - set SNES residual evaluation function

   Not Collective

   Input Parameters:
+  dm - DM to be used with SNES
.  f - residual evaluation function; see SNESFunction for details
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESSetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian(), SNESFunction
@*/
PetscErrorCode DMSNESSetFunction(DM dm,PetscErrorCode (*f)(SNES,Vec,Vec,void*),void *ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (f || ctx) {
    PetscCall(DMGetDMSNESWrite(dm,&sdm));
  }
  if (f) sdm->ops->computefunction = f;
  if (ctx) sdm->functionctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetMFFunction - set SNES residual evaluation function used in applying the matrix-free Jacobian with -snes_mf_operator

   Logically Collective on dm

   Input Parameters:
+  dm - DM to be used with SNES
-  f - residual evaluation function; see SNESFunction for details

   Level: advanced

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian(), SNESFunction, DMSNESSetFunction()
@*/
PetscErrorCode DMSNESSetMFFunction(DM dm,PetscErrorCode (*f)(SNES,Vec,Vec,void*),void *ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (f || ctx) {
    PetscCall(DMGetDMSNESWrite(dm,&sdm));
  }
  if (f) sdm->ops->computemffunction = f;
  if (ctx) sdm->mffunctionctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESGetFunction - get SNES residual evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with SNES

   Output Parameters:
+  f - residual evaluation function; see SNESFunction for details
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMSNESSetContext(), DMSNESSetFunction(), SNESSetFunction(), SNESFunction
@*/
PetscErrorCode DMSNESGetFunction(DM dm,PetscErrorCode (**f)(SNES,Vec,Vec,void*),void **ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (f) *f = sdm->ops->computefunction;
  if (ctx) *ctx = sdm->functionctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetObjective - set SNES objective evaluation function

   Not Collective

   Input Parameters:
+  dm - DM to be used with SNES
.  obj - objective evaluation function; see SNESObjectiveFunction for details
-  ctx - context for residual evaluation

   Level: advanced

.seealso: DMSNESSetContext(), SNESGetObjective(), DMSNESSetFunction()
@*/
PetscErrorCode DMSNESSetObjective(DM dm,PetscErrorCode (*obj)(SNES,Vec,PetscReal*,void*),void *ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (obj || ctx) {
    PetscCall(DMGetDMSNESWrite(dm,&sdm));
  }
  if (obj) sdm->ops->computeobjective = obj;
  if (ctx) sdm->objectivectx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESGetObjective - get SNES objective evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with SNES

   Output Parameters:
+  obj- residual evaluation function; see SNESObjectiveFunction for details
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMSNESSetContext(), DMSNESSetObjective(), SNESSetFunction()
@*/
PetscErrorCode DMSNESGetObjective(DM dm,PetscErrorCode (**obj)(SNES,Vec,PetscReal*,void*),void **ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (obj) *obj = sdm->ops->computeobjective;
  if (ctx) *ctx = sdm->objectivectx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetNGS - set SNES Gauss-Seidel relaxation function

   Not Collective

   Input Parameters:
+  dm - DM to be used with SNES
.  f  - relaxation function, see SNESGSFunction
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESSetNGS() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian(), DMSNESSetFunction(), SNESGSFunction
@*/
PetscErrorCode DMSNESSetNGS(DM dm,PetscErrorCode (*f)(SNES,Vec,Vec,void*),void *ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (f || ctx) {
    PetscCall(DMGetDMSNESWrite(dm,&sdm));
  }
  if (f) sdm->ops->computegs = f;
  if (ctx) sdm->gsctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESGetNGS - get SNES Gauss-Seidel relaxation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with SNES

   Output Parameters:
+  f - relaxation function which performs Gauss-Seidel sweeps, see SNESGSFunction
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetNGS() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMSNESSetContext(), SNESGetNGS(), DMSNESGetJacobian(), DMSNESGetFunction(), SNESNGSFunction
@*/
PetscErrorCode DMSNESGetNGS(DM dm,PetscErrorCode (**f)(SNES,Vec,Vec,void*),void **ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (f) *f = sdm->ops->computegs;
  if (ctx) *ctx = sdm->gsctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetJacobian - set SNES Jacobian evaluation function

   Not Collective

   Input Parameters:
+  dm - DM to be used with SNES
.  J - Jacobian evaluation function
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESSetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESGetJacobian(), SNESSetJacobian(), SNESJacobianFunction
@*/
PetscErrorCode DMSNESSetJacobian(DM dm,PetscErrorCode (*J)(SNES,Vec,Mat,Mat,void*),void *ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (J || ctx) {
    PetscCall(DMGetDMSNESWrite(dm,&sdm));
  }
  if (J) sdm->ops->computejacobian = J;
  if (ctx) sdm->jacobianctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESGetJacobian - get SNES Jacobian evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with SNES

   Output Parameters:
+  J - Jacobian evaluation function; see SNESJacobianFunction for all calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   SNESGetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian(), SNESJacobianFunction
@*/
PetscErrorCode DMSNESGetJacobian(DM dm,PetscErrorCode (**J)(SNES,Vec,Mat,Mat,void*),void **ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (J) *J = sdm->ops->computejacobian;
  if (ctx) *ctx = sdm->jacobianctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetPicard - set SNES Picard iteration matrix and RHS evaluation functions.

   Not Collective

   Input Parameters:
+  dm - DM to be used with SNES
.  b - RHS evaluation function
.  J - Picard matrix evaluation function
-  ctx - context for residual evaluation

   Level: advanced

.seealso: SNESSetPicard(), DMSNESSetFunction(), DMSNESSetJacobian()
@*/
PetscErrorCode DMSNESSetPicard(DM dm,PetscErrorCode (*b)(SNES,Vec,Vec,void*),PetscErrorCode (*J)(SNES,Vec,Mat,Mat,void*),void *ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (b) sdm->ops->computepfunction = b;
  if (J) sdm->ops->computepjacobian = J;
  if (ctx) sdm->pctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESGetPicard - get SNES Picard iteration evaluation functions

   Not Collective

   Input Parameter:
.  dm - DM to be used with SNES

   Output Parameters:
+  b - RHS evaluation function; see SNESFunction for details
.  J  - RHS evaluation function; see SNESJacobianFunction for detailsa
-  ctx - context for residual evaluation

   Level: advanced

.seealso: DMSNESSetContext(), SNESSetFunction(), DMSNESSetJacobian()
@*/
PetscErrorCode DMSNESGetPicard(DM dm,PetscErrorCode (**b)(SNES,Vec,Vec,void*),PetscErrorCode (**J)(SNES,Vec,Mat,Mat,void*),void **ctx)
{
  DMSNES         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNES(dm,&sdm));
  if (b) *b = sdm->ops->computepfunction;
  if (J) *J = sdm->ops->computepjacobian;
  if (ctx) *ctx = sdm->pctx;
  PetscFunctionReturn(0);
}
