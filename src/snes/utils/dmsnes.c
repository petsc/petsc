#include <petsc-private/snesimpl.h>   /*I "petscsnes.h" I*/
#include <petscdm.h>            /*I "petscdm.h" I*/

#undef __FUNCT__
#define __FUNCT__ "SNESDMComputeFunction"
/*@C
  SNESDMComputeFunction - This is a universal function evaluation routine that
  may be used with SNESSetFunction() as long as the user context has a DM
  as its first record and the user has called DMSetLocalFunction().

  Collective on SNES

  Input Parameters:
+ snes - the SNES context
. X - input vector
. F - function vector
- ptr - pointer to a structure that must have a DM as its first entry.
        This ptr must have been passed into SNESDMComputeFunction() as the context.

  Level: intermediate

.seealso: DMSetLocalFunction(), DMSetLocalJacobian(), SNESSetFunction(), SNESSetJacobian()
@*/
PetscErrorCode SNESDMComputeFunction(SNES snes, Vec X, Vec F, void *ptr)
{
  DM               dm = *(DM*) ptr;
  PetscErrorCode (*lf)(DM, Vec, Vec, void *);
  Vec              localX, localF;
  PetscInt         N, n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  if (!dm) SETERRQ(((PetscObject)snes)->comm, PETSC_ERR_ARG_WRONGSTATE, "Looks like you called SNESSetFromFuntion(snes,SNESDMComputeFunction,) without the DM context");
  PetscValidHeaderSpecific(dm, DM_CLASSID, 4);

  /* determine whether X = localX */
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &localF);CHKERRQ(ierr);
  ierr = VecGetSize(X, &N);CHKERRQ(ierr);
  ierr = VecGetSize(localX, &n);CHKERRQ(ierr);

  if (n != N){ /* X != localX */
    /* Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
    */
    ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  } else {
    ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
    localX = X;
  }
  ierr = DMGetLocalFunction(dm, &lf);CHKERRQ(ierr);
  ierr = (*lf)(dm, localX, localF, ptr);CHKERRQ(ierr);
  if (n != N){
    ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  }
  ierr = DMLocalToGlobalBegin(dm, localF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, localF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDMComputeJacobian"
/*
  SNESDMComputeJacobian - This is a universal Jacobian evaluation routine that
  may be used with SNESSetJacobian() as long as the user context has a DM
  as its first record and the user has called DMSetLocalJacobian().

  Collective on SNES

  Input Parameters:
+ snes - the SNES context
. X - input vector
. J - Jacobian
. B - Jacobian used in preconditioner (usally same as J)
. flag - indicates if the matrix changed its structure
- ptr - pointer to a structure that must have a DM as its first entry.
        This ptr must have been passed into SNESDMComputeFunction() as the context.

  Level: intermediate

.seealso: DMSetLocalFunction(), DMSetLocalJacobian(), SNESSetFunction(), SNESSetJacobian()
*/
PetscErrorCode SNESDMComputeJacobian(SNES snes, Vec X, Mat *J, Mat *B, MatStructure *flag, void *ptr)
{
  DM               dm = *(DM*) ptr;
  PetscErrorCode (*lj)(DM, Vec, Mat, Mat, void *);
  Vec              localX;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm, &localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, localX);CHKERRQ(ierr);
  ierr = DMGetLocalJacobian(dm, &lj);CHKERRQ(ierr);
  ierr = (*lj)(dm, localX, *J, *B, ptr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &localX);CHKERRQ(ierr);
  /* Assemble true Jacobian; if it is different */
  if (*J != *B) {
    ierr  = MatAssemblyBegin(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(*J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr  = MatSetOption(*B, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscContainerDestroy_SNESDM"
static PetscErrorCode PetscContainerDestroy_SNESDM(void *ctx)
{
  PetscErrorCode ierr;
  SNESDM sdm = (SNESDM)ctx;

  PetscFunctionBegin;
  if (sdm->destroy) {ierr = (*sdm->destroy)(sdm);CHKERRQ(ierr);}
  ierr = PetscFree(sdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHook_SNESDM"
/* Attaches the SNESDM to the coarse level.
 * Under what conditions should we copy versus duplicate?
 */
static PetscErrorCode DMCoarsenHook_SNESDM(DM dm,DM dmc,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSNESCopyContext(dm,dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestrictHook_SNESDM"
/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMRestrictHook_SNESDM(DM dm,Mat Restrict,Vec rscale,Mat Inject,DM dmc,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESGetContext"
/*@C
   DMSNESGetContext - get read-only private SNESDM context from a DM

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Argument:
.  snesdm - private SNESDM context

   Level: developer

   Notes:
   Use DMSNESGetContextWrite() if write access is needed. The DMSNESSetXXX API should be used wherever possible.

.seealso: DMSNESGetContextWrite()
@*/
PetscErrorCode DMSNESGetContext(DM dm,SNESDM *snesdm)
{
  PetscErrorCode ierr;
  PetscContainer container;
  SNESDM         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)dm,"SNESDM",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void**)snesdm);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(dm,"Creating new SNESDM\n");CHKERRQ(ierr);
    ierr = PetscContainerCreate(((PetscObject)dm)->comm,&container);CHKERRQ(ierr);
    ierr = PetscNewLog(dm,struct _n_SNESDM,&sdm);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,sdm);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_SNESDM);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"SNESDM",(PetscObject)container);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_SNESDM,DMRestrictHook_SNESDM,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscContainerGetPointer(container,(void**)snesdm);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESGetContextWrite"
/*@C
   DMSNESGetContextWrite - get write access to private SNESDM context from a DM

   Not Collective

   Input Argument:
.  dm - DM to be used with SNES

   Output Argument:
.  snesdm - private SNESDM context

   Level: developer

.seealso: DMSNESGetContext()
@*/
PetscErrorCode DMSNESGetContextWrite(DM dm,SNESDM *snesdm)
{
  PetscErrorCode ierr;
  SNESDM         sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (!sdm->originaldm) sdm->originaldm = dm;
  if (sdm->originaldm != dm) {  /* Copy on write */
    PetscContainer container;
    SNESDM         oldsdm = sdm;
    ierr = PetscInfo(dm,"Copying SNESDM due to write\n");CHKERRQ(ierr);
    ierr = PetscContainerCreate(((PetscObject)dm)->comm,&container);CHKERRQ(ierr);
    ierr = PetscNewLog(dm,struct _n_SNESDM,&sdm);CHKERRQ(ierr);
    ierr = PetscMemcpy(sdm,oldsdm,sizeof *sdm);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,sdm);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_SNESDM);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"SNESDM",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }
  *snesdm = sdm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESCopyContext"
/*@C
   DMSNESCopyContext - copies a DM context to a new DM

   Logically Collective

   Input Arguments:
+  dmsrc - DM to obtain context from
-  dmdest - DM to add context to

   Level: developer

   Note:
   The context is copied by reference. This function does not ensure that a context exists.

.seealso: DMSNESGetContext(), SNESSetDM()
@*/
PetscErrorCode DMSNESCopyContext(DM dmsrc,DM dmdest)
{
  PetscErrorCode ierr;
  PetscContainer container;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmdest,DM_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)dmsrc,"SNESDM",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscObjectCompose((PetscObject)dmdest,"SNESDM",(PetscObject)container);CHKERRQ(ierr);
    ierr = DMCoarsenHookAdd(dmdest,DMCoarsenHook_SNESDM,DMRestrictHook_SNESDM,PETSC_NULL);CHKERRQ(ierr);
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
  SNESDM sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContextWrite(dm,&sdm);CHKERRQ(ierr);
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
  SNESDM sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (func) *func = sdm->computefunction;
  if (ctx)  *ctx = sdm->functionctx;
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
  SNESDM sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContextWrite(dm,&sdm);CHKERRQ(ierr);
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
  SNESDM sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
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
  SNESDM sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContextWrite(dm,&sdm);CHKERRQ(ierr);
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
  SNESDM sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
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
  SNESDM sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
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
  SNESDM sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (pfunc) *pfunc = sdm->computepfunction;
  if (pjac) *pjac   = sdm->computepjacobian;
  if (ctx)  *ctx    = sdm->pctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESDefaultComputeFunction_DMLegacy"
static PetscErrorCode SNESDefaultComputeFunction_DMLegacy(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMComputeFunction(dm,X,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDefaultComputeJacobian_DMLegacy"
static PetscErrorCode SNESDefaultComputeJacobian_DMLegacy(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *mstr,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMComputeJacobian(dm,X,*A,*B,mstr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSNESSetUpLegacy"
/* Sets up calling of legacy DM routines */
PetscErrorCode DMSNESSetUpLegacy(DM dm)
{
  PetscErrorCode ierr;
  SNESDM         sdm;

  PetscFunctionBegin;
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (!sdm->computefunction) {ierr = DMSNESSetFunction(dm,SNESDefaultComputeFunction_DMLegacy,PETSC_NULL);CHKERRQ(ierr);}
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  if (!sdm->computejacobian) {ierr = DMSNESSetJacobian(dm,SNESDefaultComputeJacobian_DMLegacy,PETSC_NULL);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
