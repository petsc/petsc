#include <petscdmda.h>          /*I "petscdmda.h" I*/
#include <petsc/private/dmimpl.h>
#include <petsc/private/snesimpl.h>   /*I "petscsnes.h" I*/

/* This structure holds the user-provided DMDA callbacks */
typedef struct {
  PetscErrorCode (*residuallocal)(DMDALocalInfo*,void*,void*,void*);
  PetscErrorCode (*jacobianlocal)(DMDALocalInfo*,void*,Mat,Mat,void*);
  PetscErrorCode (*objectivelocal)(DMDALocalInfo*,void*,PetscReal*,void*);
  void       *residuallocalctx;
  void       *jacobianlocalctx;
  void       *objectivelocalctx;
  InsertMode residuallocalimode;

  /*   For Picard iteration defined locally */
  PetscErrorCode (*rhsplocal)(DMDALocalInfo*,void*,void*,void*);
  PetscErrorCode (*jacobianplocal)(DMDALocalInfo*,void*,Mat,Mat,void*);
  void *picardlocalctx;
} DMSNES_DA;

static PetscErrorCode DMSNESDestroy_DMDA(DMSNES sdm)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(sdm->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSNESDuplicate_DMDA(DMSNES oldsdm,DMSNES sdm)
{
  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(sdm,(DMSNES_DA**)&sdm->data));
  if (oldsdm->data) {
    CHKERRQ(PetscMemcpy(sdm->data,oldsdm->data,sizeof(DMSNES_DA)));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDASNESGetContext(DM dm,DMSNES sdm,DMSNES_DA  **dmdasnes)
{
  PetscFunctionBegin;
  *dmdasnes = NULL;
  if (!sdm->data) {
    CHKERRQ(PetscNewLog(dm,(DMSNES_DA**)&sdm->data));
    sdm->ops->destroy   = DMSNESDestroy_DMDA;
    sdm->ops->duplicate = DMSNESDuplicate_DMDA;
  }
  *dmdasnes = (DMSNES_DA*)sdm->data;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputeFunction_DMDA(SNES snes,Vec X,Vec F,void *ctx)
{
  DM             dm;
  DMSNES_DA      *dmdasnes = (DMSNES_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  PetscCheck(dmdasnes->residuallocal,PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetLocalVector(dm,&Xloc));
  CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMDAGetLocalInfo(dm,&info));
  CHKERRQ(DMDAVecGetArray(dm,Xloc,&x));
  switch (dmdasnes->residuallocalimode) {
  case INSERT_VALUES: {
    CHKERRQ(DMDAVecGetArray(dm,F,&f));
    CHKERRQ(PetscLogEventBegin(SNES_FunctionEval,snes,X,F,0));
    CHKMEMQ;
    CHKERRQ((*dmdasnes->residuallocal)(&info,x,f,dmdasnes->residuallocalctx));
    CHKMEMQ;
    CHKERRQ(PetscLogEventEnd(SNES_FunctionEval,snes,X,F,0));
    CHKERRQ(DMDAVecRestoreArray(dm,F,&f));
  } break;
  case ADD_VALUES: {
    Vec Floc;
    CHKERRQ(DMGetLocalVector(dm,&Floc));
    CHKERRQ(VecZeroEntries(Floc));
    CHKERRQ(DMDAVecGetArray(dm,Floc,&f));
    CHKERRQ(PetscLogEventBegin(SNES_FunctionEval,snes,X,F,0));
    CHKMEMQ;
    CHKERRQ((*dmdasnes->residuallocal)(&info,x,f,dmdasnes->residuallocalctx));
    CHKMEMQ;
    CHKERRQ(PetscLogEventEnd(SNES_FunctionEval,snes,X,F,0));
    CHKERRQ(DMDAVecRestoreArray(dm,Floc,&f));
    CHKERRQ(VecZeroEntries(F));
    CHKERRQ(DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F));
    CHKERRQ(DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F));
    CHKERRQ(DMRestoreLocalVector(dm,&Floc));
  } break;
  default: SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdasnes->residuallocalimode);
  }
  CHKERRQ(DMDAVecRestoreArray(dm,Xloc,&x));
  CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  if (snes->domainerror) {
    CHKERRQ(VecSetInf(F));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputeObjective_DMDA(SNES snes,Vec X,PetscReal *ob,void *ctx)
{
  DM             dm;
  DMSNES_DA      *dmdasnes = (DMSNES_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidRealPointer(ob,3);
  PetscCheck(dmdasnes->objectivelocal,PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetLocalVector(dm,&Xloc));
  CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMDAGetLocalInfo(dm,&info));
  CHKERRQ(DMDAVecGetArray(dm,Xloc,&x));
  CHKMEMQ;
  CHKERRQ((*dmdasnes->objectivelocal)(&info,x,ob,dmdasnes->objectivelocalctx));
  CHKMEMQ;
  CHKERRQ(DMDAVecRestoreArray(dm,Xloc,&x));
  CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  PetscFunctionReturn(0);
}

/* Routine is called by example, hence must be labeled PETSC_EXTERN */
PETSC_EXTERN PetscErrorCode SNESComputeJacobian_DMDA(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  DM             dm;
  DMSNES_DA      *dmdasnes = (DMSNES_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x;

  PetscFunctionBegin;
  PetscCheck(dmdasnes->residuallocal,PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  CHKERRQ(SNESGetDM(snes,&dm));

  if (dmdasnes->jacobianlocal) {
    CHKERRQ(DMGetLocalVector(dm,&Xloc));
    CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
    CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
    CHKERRQ(DMDAGetLocalInfo(dm,&info));
    CHKERRQ(DMDAVecGetArray(dm,Xloc,&x));
    CHKMEMQ;
    CHKERRQ((*dmdasnes->jacobianlocal)(&info,x,A,B,dmdasnes->jacobianlocalctx));
    CHKMEMQ;
    CHKERRQ(DMDAVecRestoreArray(dm,Xloc,&x));
    CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  } else {
    MatFDColoring fdcoloring;
    CHKERRQ(PetscObjectQuery((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject*)&fdcoloring));
    if (!fdcoloring) {
      ISColoring coloring;

      CHKERRQ(DMCreateColoring(dm,dm->coloringtype,&coloring));
      CHKERRQ(MatFDColoringCreate(B,coloring,&fdcoloring));
      switch (dm->coloringtype) {
      case IS_COLORING_GLOBAL:
        CHKERRQ(MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))SNESComputeFunction_DMDA,dmdasnes));
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_SUP,"No support for coloring type '%s'",ISColoringTypes[dm->coloringtype]);
      }
      CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)fdcoloring,((PetscObject)dm)->prefix));
      CHKERRQ(MatFDColoringSetFromOptions(fdcoloring));
      CHKERRQ(MatFDColoringSetUp(B,coloring,fdcoloring));
      CHKERRQ(ISColoringDestroy(&coloring));
      CHKERRQ(PetscObjectCompose((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject)fdcoloring));
      CHKERRQ(PetscObjectDereference((PetscObject)fdcoloring));

      /* The following breaks an ugly reference counting loop that deserves a paragraph. MatFDColoringApply() will call
       * VecDuplicate() with the state Vec and store inside the MatFDColoring. This Vec will duplicate the Vec, but the
       * MatFDColoring is composed with the DM. We dereference the DM here so that the reference count will eventually
       * drop to 0. Note the code in DMDestroy() that exits early for a negative reference count. That code path will be
       * taken when the PetscObjectList for the Vec inside MatFDColoring is destroyed.
       */
      CHKERRQ(PetscObjectDereference((PetscObject)dm));
    }
    CHKERRQ(MatFDColoringApply(B,fdcoloring,X,snes));
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetFunctionLocal - set a local residual evaluation function

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  imode - INSERT_VALUES if local function computes owned part, ADD_VALUES if it contributes to ghosted part
.  func - local residual evaluation
-  ctx - optional context for local residual evaluation

   Calling sequence:
   For PetscErrorCode (*func)(DMDALocalInfo *info,void *x, void *f, void *ctx),
+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  x - dimensional pointer to state at which to evaluate residual (e.g. PetscScalar *x or **x or ***x)
.  f - dimensional pointer to residual, write the residual here (e.g. PetscScalar *f or **f or ***f)
-  ctx - optional context passed above

   Level: beginner

.seealso: DMDASNESSetJacobianLocal(), DMSNESSetFunction(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMDASNESSetFunctionLocal(DM dm,InsertMode imode,PetscErrorCode (*func)(DMDALocalInfo*,void*,void*,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMSNESWrite(dm,&sdm));
  CHKERRQ(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->residuallocalimode = imode;
  dmdasnes->residuallocal      = func;
  dmdasnes->residuallocalctx   = ctx;

  CHKERRQ(DMSNESSetFunction(dm,SNESComputeFunction_DMDA,dmdasnes));
  if (!sdm->ops->computejacobian) {  /* Call us for the Jacobian too, can be overridden by the user. */
    CHKERRQ(DMSNESSetJacobian(dm,SNESComputeJacobian_DMDA,dmdasnes));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetJacobianLocal - set a local Jacobian evaluation function

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  func - local Jacobian evaluation
-  ctx - optional context for local Jacobian evaluation

   Calling sequence:
   For PetscErrorCode (*func)(DMDALocalInfo *info,void *x,Mat J,Mat M,void *ctx),
+  info - DMDALocalInfo defining the subdomain to evaluate the Jacobian at
.  x - dimensional pointer to state at which to evaluate Jacobian (e.g. PetscScalar *x or **x or ***x)
.  J - Mat object for the Jacobian
.  M - Mat object for the Jacobian preconditioner matrix
-  ctx - optional context passed above

   Level: beginner

.seealso: DMDASNESSetFunctionLocal(), DMSNESSetJacobian(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMDASNESSetJacobianLocal(DM dm,PetscErrorCode (*func)(DMDALocalInfo*,void*,Mat,Mat,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMSNESWrite(dm,&sdm));
  CHKERRQ(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->jacobianlocal    = func;
  dmdasnes->jacobianlocalctx = ctx;

  CHKERRQ(DMSNESSetJacobian(dm,SNESComputeJacobian_DMDA,dmdasnes));
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetObjectiveLocal - set a local residual evaluation function

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  func - local objective evaluation
-  ctx - optional context for local residual evaluation

   Calling sequence for func:
+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  x - dimensional pointer to state at which to evaluate residual
.  ob - eventual objective value
-  ctx - optional context passed above

   Level: beginner

.seealso: DMSNESSetFunction(), DMDASNESSetJacobianLocal(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMDASNESSetObjectiveLocal(DM dm,DMDASNESObjective func,void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMSNESWrite(dm,&sdm));
  CHKERRQ(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->objectivelocal    = func;
  dmdasnes->objectivelocalctx = ctx;

  CHKERRQ(DMSNESSetObjective(dm,SNESComputeObjective_DMDA,dmdasnes));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputePicard_DMDA(SNES snes,Vec X,Vec F,void *ctx)
{
  DM             dm;
  DMSNES_DA      *dmdasnes = (DMSNES_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  PetscCheck(dmdasnes->rhsplocal,PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetLocalVector(dm,&Xloc));
  CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMDAGetLocalInfo(dm,&info));
  CHKERRQ(DMDAVecGetArray(dm,Xloc,&x));
  switch (dmdasnes->residuallocalimode) {
  case INSERT_VALUES: {
    CHKERRQ(DMDAVecGetArray(dm,F,&f));
    CHKMEMQ;
    CHKERRQ((*dmdasnes->rhsplocal)(&info,x,f,dmdasnes->picardlocalctx));
    CHKMEMQ;
    CHKERRQ(DMDAVecRestoreArray(dm,F,&f));
  } break;
  case ADD_VALUES: {
    Vec Floc;
    CHKERRQ(DMGetLocalVector(dm,&Floc));
    CHKERRQ(VecZeroEntries(Floc));
    CHKERRQ(DMDAVecGetArray(dm,Floc,&f));
    CHKMEMQ;
    CHKERRQ((*dmdasnes->rhsplocal)(&info,x,f,dmdasnes->picardlocalctx));
    CHKMEMQ;
    CHKERRQ(DMDAVecRestoreArray(dm,Floc,&f));
    CHKERRQ(VecZeroEntries(F));
    CHKERRQ(DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F));
    CHKERRQ(DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F));
    CHKERRQ(DMRestoreLocalVector(dm,&Floc));
  } break;
  default: SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdasnes->residuallocalimode);
  }
  CHKERRQ(DMDAVecRestoreArray(dm,Xloc,&x));
  CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputePicardJacobian_DMDA(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  DM             dm;
  DMSNES_DA      *dmdasnes = (DMSNES_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x;

  PetscFunctionBegin;
  PetscCheck(dmdasnes->jacobianplocal,PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  CHKERRQ(SNESGetDM(snes,&dm));

  CHKERRQ(DMGetLocalVector(dm,&Xloc));
  CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMDAGetLocalInfo(dm,&info));
  CHKERRQ(DMDAVecGetArray(dm,Xloc,&x));
  CHKMEMQ;
  CHKERRQ((*dmdasnes->jacobianplocal)(&info,x,A,B,dmdasnes->picardlocalctx));
  CHKMEMQ;
  CHKERRQ(DMDAVecRestoreArray(dm,Xloc,&x));
  CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetPicardLocal - set a local right hand side and matrix evaluation function for Picard iteration

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  imode - INSERT_VALUES if local function computes owned part, ADD_VALUES if it contributes to ghosted part
.  func - local residual evaluation
-  ctx - optional context for local residual evaluation

   Calling sequence for func:
+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  x - dimensional pointer to state at which to evaluate residual
.  f - dimensional pointer to residual, write the residual here
-  ctx - optional context passed above

   Notes:
    The user must use
    CHKERRQ(SNESSetFunction(snes,NULL,SNESPicardComputeFunction,&user));
    in their code before calling this routine.

   Level: beginner

.seealso: DMSNESSetFunction(), DMDASNESSetJacobian(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMDASNESSetPicardLocal(DM dm,InsertMode imode,PetscErrorCode (*func)(DMDALocalInfo*,void*,void*,void*),
                                      PetscErrorCode (*jac)(DMDALocalInfo*,void*,Mat,Mat,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMSNESWrite(dm,&sdm));
  CHKERRQ(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->residuallocalimode = imode;
  dmdasnes->rhsplocal          = func;
  dmdasnes->jacobianplocal     = jac;
  dmdasnes->picardlocalctx     = ctx;

  CHKERRQ(DMSNESSetPicard(dm,SNESComputePicard_DMDA,SNESComputePicardJacobian_DMDA,dmdasnes));
  CHKERRQ(DMSNESSetMFFunction(dm,SNESComputeFunction_DMDA,dmdasnes));
  PetscFunctionReturn(0);
}
