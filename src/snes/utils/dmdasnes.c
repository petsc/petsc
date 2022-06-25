#include <petscdmda.h>          /*I "petscdmda.h" I*/
#include <petsc/private/dmimpl.h>
#include <petsc/private/snesimpl.h>   /*I "petscsnes.h" I*/

/* This structure holds the user-provided DMDA callbacks */
typedef struct {
  PetscErrorCode (*residuallocal)(DMDALocalInfo*,void*,void*,void*);
  PetscErrorCode (*jacobianlocal)(DMDALocalInfo*,void*,Mat,Mat,void*);
  PetscErrorCode (*objectivelocal)(DMDALocalInfo*,void*,PetscReal*,void*);

  PetscErrorCode (*residuallocalvec)(DMDALocalInfo*,Vec,Vec,void*);
  PetscErrorCode (*jacobianlocalvec)(DMDALocalInfo*,Vec,Mat,Mat,void*);
  PetscErrorCode (*objectivelocalvec)(DMDALocalInfo*,Vec,PetscReal*,void*);
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
  PetscCall(PetscFree(sdm->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSNESDuplicate_DMDA(DMSNES oldsdm,DMSNES sdm)
{
  PetscFunctionBegin;
  PetscCall(PetscNewLog(sdm,(DMSNES_DA**)&sdm->data));
  if (oldsdm->data) PetscCall(PetscMemcpy(sdm->data,oldsdm->data,sizeof(DMSNES_DA)));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDASNESGetContext(DM dm,DMSNES sdm,DMSNES_DA  **dmdasnes)
{
  PetscFunctionBegin;
  *dmdasnes = NULL;
  if (!sdm->data) {
    PetscCall(PetscNewLog(dm,(DMSNES_DA**)&sdm->data));
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
  PetscCheck(dmdasnes->residuallocal || dmdasnes->residuallocalvec,PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&Xloc));
  PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMDAGetLocalInfo(dm,&info));
  switch (dmdasnes->residuallocalimode) {
  case INSERT_VALUES: {
    PetscCall(PetscLogEventBegin(SNES_FunctionEval,snes,X,F,0));
    if (dmdasnes->residuallocalvec) PetscCallUser("SNES DMDA local callback function",(*dmdasnes->residuallocalvec)(&info,Xloc,F,dmdasnes->residuallocalctx));
    else {
      PetscCall(DMDAVecGetArray(dm,Xloc,&x));
      PetscCall(DMDAVecGetArray(dm,F,&f));
      PetscCallUser("SNES DMDA local callback function",(*dmdasnes->residuallocal)(&info,x,f,dmdasnes->residuallocalctx));
      PetscCall(DMDAVecRestoreArray(dm,Xloc,&x));
      PetscCall(DMDAVecRestoreArray(dm,F,&f));
    }
    PetscCall(PetscLogEventEnd(SNES_FunctionEval,snes,X,F,0));
  } break;
  case ADD_VALUES: {
    Vec Floc;
    PetscCall(DMGetLocalVector(dm,&Floc));
    PetscCall(VecZeroEntries(Floc));
    PetscCall(PetscLogEventBegin(SNES_FunctionEval,snes,X,F,0));
    if (dmdasnes->residuallocalvec) PetscCallUser("SNES DMDA local callback function",(*dmdasnes->residuallocalvec)(&info,Xloc,Floc,dmdasnes->residuallocalctx));
    else {
      PetscCall(DMDAVecGetArray(dm,Xloc,&x));
      PetscCall(DMDAVecGetArray(dm,Floc,&f));
      PetscCallUser("SNES DMDA local callback function",(*dmdasnes->residuallocal)(&info,x,f,dmdasnes->residuallocalctx));
      PetscCall(DMDAVecRestoreArray(dm,Xloc,&x));
      PetscCall(DMDAVecRestoreArray(dm,Floc,&f));
    }
    PetscCall(PetscLogEventEnd(SNES_FunctionEval,snes,X,F,0));
    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F));
    PetscCall(DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F));
    PetscCall(DMRestoreLocalVector(dm,&Floc));
  } break;
  default: SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdasnes->residuallocalimode);
  }
  PetscCall(DMRestoreLocalVector(dm,&Xloc));
  if (snes->domainerror) PetscCall(VecSetInf(F));
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
  PetscCheck(dmdasnes->objectivelocal || dmdasnes->objectivelocalvec,PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&Xloc));
  PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMDAGetLocalInfo(dm,&info));
  if (dmdasnes->objectivelocalvec) PetscCallUser("SNES DMDA local callback objective",(*dmdasnes->objectivelocalvec)(&info,Xloc,ob,dmdasnes->objectivelocalctx));
  else {
    PetscCall(DMDAVecGetArray(dm,Xloc,&x));
    PetscCallUser("SNES DMDA local callback objective",(*dmdasnes->objectivelocal)(&info,x,ob,dmdasnes->objectivelocalctx));
    PetscCall(DMDAVecRestoreArray(dm,Xloc,&x));
  }
  PetscCall(DMRestoreLocalVector(dm,&Xloc));
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
  PetscCheck(dmdasnes->residuallocal || dmdasnes->residuallocalvec,PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  PetscCall(SNESGetDM(snes,&dm));

  if (dmdasnes->jacobianlocal || dmdasnes->jacobianlocalctx) {
    PetscCall(DMGetLocalVector(dm,&Xloc));
    PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
    PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
    PetscCall(DMDAGetLocalInfo(dm,&info));
    if (dmdasnes->jacobianlocalvec) PetscCallUser("SNES DMDA local callback Jacobian",(*dmdasnes->jacobianlocalvec)(&info,Xloc,A,B,dmdasnes->jacobianlocalctx));
    else {
      PetscCall(DMDAVecGetArray(dm,Xloc,&x));
      PetscCallUser("SNES DMDA local callback Jacobian",(*dmdasnes->jacobianlocal)(&info,x,A,B,dmdasnes->jacobianlocalctx));
      PetscCall(DMDAVecRestoreArray(dm,Xloc,&x));
    }
    PetscCall(DMRestoreLocalVector(dm,&Xloc));
  } else {
    MatFDColoring fdcoloring;
    PetscCall(PetscObjectQuery((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject*)&fdcoloring));
    if (!fdcoloring) {
      ISColoring coloring;

      PetscCall(DMCreateColoring(dm,dm->coloringtype,&coloring));
      PetscCall(MatFDColoringCreate(B,coloring,&fdcoloring));
      switch (dm->coloringtype) {
      case IS_COLORING_GLOBAL:
        PetscCall(MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))SNESComputeFunction_DMDA,dmdasnes));
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_SUP,"No support for coloring type '%s'",ISColoringTypes[dm->coloringtype]);
      }
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fdcoloring,((PetscObject)dm)->prefix));
      PetscCall(MatFDColoringSetFromOptions(fdcoloring));
      PetscCall(MatFDColoringSetUp(B,coloring,fdcoloring));
      PetscCall(ISColoringDestroy(&coloring));
      PetscCall(PetscObjectCompose((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject)fdcoloring));
      PetscCall(PetscObjectDereference((PetscObject)fdcoloring));

      /* The following breaks an ugly reference counting loop that deserves a paragraph. MatFDColoringApply() will call
       * VecDuplicate() with the state Vec and store inside the MatFDColoring. This Vec will duplicate the Vec, but the
       * MatFDColoring is composed with the DM. We dereference the DM here so that the reference count will eventually
       * drop to 0. Note the code in DMDestroy() that exits early for a negative reference count. That code path will be
       * taken when the PetscObjectList for the Vec inside MatFDColoring is destroyed.
       */
      PetscCall(PetscObjectDereference((PetscObject)dm));
    }
    PetscCall(MatFDColoringApply(B,fdcoloring,X,snes));
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
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

.seealso: `DMDASNESSetJacobianLocal()`, `DMSNESSetFunction()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetFunctionLocal(DM dm,InsertMode imode,PetscErrorCode (*func)(DMDALocalInfo*,void*,void*,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNESWrite(dm,&sdm));
  PetscCall(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->residuallocalimode = imode;
  dmdasnes->residuallocal      = func;
  dmdasnes->residuallocalctx   = ctx;

  PetscCall(DMSNESSetFunction(dm,SNESComputeFunction_DMDA,dmdasnes));
  if (!sdm->ops->computejacobian) {  /* Call us for the Jacobian too, can be overridden by the user. */
    PetscCall(DMSNESSetJacobian(dm,SNESComputeJacobian_DMDA,dmdasnes));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetFunctionLocalVec - set a local residual evaluation function that operates on a local vector

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  imode - INSERT_VALUES if local function computes owned part, ADD_VALUES if it contributes to ghosted part
.  func - local residual evaluation
-  ctx - optional context for local residual evaluation

   Calling sequence:
   For PetscErrorCode (*func)(DMDALocalInfo *info,Vec x, Vec f, void *ctx),
+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  x - state vector at which to evaluate residual
.  f - residual vector
-  ctx - optional context passed above

   Level: beginner

.seealso: `DMDASNESSetFunctionLocal()`, `DMDASNESSetJacobianLocalVec()`, `DMSNESSetFunction()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetFunctionLocalVec(DM dm,InsertMode imode,PetscErrorCode (*func)(DMDALocalInfo*,Vec,Vec,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNESWrite(dm,&sdm));
  PetscCall(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->residuallocalimode = imode;
  dmdasnes->residuallocalvec   = func;
  dmdasnes->residuallocalctx   = ctx;

  PetscCall(DMSNESSetFunction(dm,SNESComputeFunction_DMDA,dmdasnes));
  if (!sdm->ops->computejacobian) {  /* Call us for the Jacobian too, can be overridden by the user. */
    PetscCall(DMSNESSetJacobian(dm,SNESComputeJacobian_DMDA,dmdasnes));
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

.seealso: `DMDASNESSetFunctionLocal()`, `DMSNESSetJacobian()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetJacobianLocal(DM dm,PetscErrorCode (*func)(DMDALocalInfo*,void*,Mat,Mat,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNESWrite(dm,&sdm));
  PetscCall(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->jacobianlocal    = func;
  dmdasnes->jacobianlocalctx = ctx;

  PetscCall(DMSNESSetJacobian(dm,SNESComputeJacobian_DMDA,dmdasnes));
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetJacobianLocalVec - set a local Jacobian evaluation function that operates on a local vector

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  func - local Jacobian evaluation
-  ctx - optional context for local Jacobian evaluation

   Calling sequence:
   For PetscErrorCode (*func)(DMDALocalInfo *info,Vec x,Mat J,Mat M,void *ctx),
+  info - DMDALocalInfo defining the subdomain to evaluate the Jacobian at
.  x - state vector at which to evaluate Jacobian
.  J - Mat object for the Jacobian
.  M - Mat object for the Jacobian preconditioner matrix
-  ctx - optional context passed above

   Level: beginner

.seealso: `DMDASNESSetJacobianLocal()`, `DMDASNESSetFunctionLocalVec()`, `DMSNESSetJacobian()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetJacobianLocalVec(DM dm,PetscErrorCode (*func)(DMDALocalInfo*,Vec,Mat,Mat,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNESWrite(dm,&sdm));
  PetscCall(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->jacobianlocalvec = func;
  dmdasnes->jacobianlocalctx = ctx;

  PetscCall(DMSNESSetJacobian(dm,SNESComputeJacobian_DMDA,dmdasnes));
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

.seealso: `DMSNESSetFunction()`, `DMDASNESSetJacobianLocal()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetObjectiveLocal(DM dm,DMDASNESObjective func,void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNESWrite(dm,&sdm));
  PetscCall(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->objectivelocal    = func;
  dmdasnes->objectivelocalctx = ctx;

  PetscCall(DMSNESSetObjective(dm,SNESComputeObjective_DMDA,dmdasnes));
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetObjectiveLocal - set a local residual evaluation function that operates on a local vector

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  func - local objective evaluation
-  ctx - optional context for local residual evaluation

   Calling sequence
   For PetscErrorCode (*func)(DMDALocalInfo *info,Vec x,PetscReal *ob,void *ctx);
+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  x - state vector at which to evaluate residual
.  ob - eventual objective value
-  ctx - optional context passed above

   Level: beginner

.seealso: `DMDASNESSetObjectiveLocal()`, `DMSNESSetFunction()`, `DMDASNESSetJacobianLocalVec()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetObjectiveLocalVec(DM dm,DMDASNESObjectiveVec func,void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNESWrite(dm,&sdm));
  PetscCall(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->objectivelocalvec = func;
  dmdasnes->objectivelocalctx = ctx;

  PetscCall(DMSNESSetObjective(dm,SNESComputeObjective_DMDA,dmdasnes));
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
  PetscCall(SNESGetDM(snes,&dm));
  PetscCall(DMGetLocalVector(dm,&Xloc));
  PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMDAGetLocalInfo(dm,&info));
  PetscCall(DMDAVecGetArray(dm,Xloc,&x));
  switch (dmdasnes->residuallocalimode) {
  case INSERT_VALUES: {
    PetscCall(DMDAVecGetArray(dm,F,&f));
    PetscCallUser("SNES Picard DMDA local callback function",(*dmdasnes->rhsplocal)(&info,x,f,dmdasnes->picardlocalctx));
    PetscCall(DMDAVecRestoreArray(dm,F,&f));
  } break;
  case ADD_VALUES: {
    Vec Floc;
    PetscCall(DMGetLocalVector(dm,&Floc));
    PetscCall(VecZeroEntries(Floc));
    PetscCall(DMDAVecGetArray(dm,Floc,&f));
    PetscCallUser("SNES Picard DMDA local callback function",(*dmdasnes->rhsplocal)(&info,x,f,dmdasnes->picardlocalctx));
    PetscCall(DMDAVecRestoreArray(dm,Floc,&f));
    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F));
    PetscCall(DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F));
    PetscCall(DMRestoreLocalVector(dm,&Floc));
  } break;
  default: SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdasnes->residuallocalimode);
  }
  PetscCall(DMDAVecRestoreArray(dm,Xloc,&x));
  PetscCall(DMRestoreLocalVector(dm,&Xloc));
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
  PetscCall(SNESGetDM(snes,&dm));

  PetscCall(DMGetLocalVector(dm,&Xloc));
  PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMDAGetLocalInfo(dm,&info));
  PetscCall(DMDAVecGetArray(dm,Xloc,&x));
  PetscCallUser("SNES Picard DMDA local callback Jacobian",(*dmdasnes->jacobianplocal)(&info,x,A,B,dmdasnes->picardlocalctx));
  PetscCall(DMDAVecRestoreArray(dm,Xloc,&x));
  PetscCall(DMRestoreLocalVector(dm,&Xloc));
  if (A != B) {
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
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
    PetscCall(SNESSetFunction(snes,NULL,SNESPicardComputeFunction,&user));
    in their code before calling this routine.

   Level: beginner

.seealso: `DMSNESSetFunction()`, `DMDASNESSetJacobian()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetPicardLocal(DM dm,InsertMode imode,PetscErrorCode (*func)(DMDALocalInfo*,void*,void*,void*),
                                      PetscErrorCode (*jac)(DMDALocalInfo*,void*,Mat,Mat,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMSNESWrite(dm,&sdm));
  PetscCall(DMDASNESGetContext(dm,sdm,&dmdasnes));

  dmdasnes->residuallocalimode = imode;
  dmdasnes->rhsplocal          = func;
  dmdasnes->jacobianplocal     = jac;
  dmdasnes->picardlocalctx     = ctx;

  PetscCall(DMSNESSetPicard(dm,SNESComputePicard_DMDA,SNESComputePicardJacobian_DMDA,dmdasnes));
  PetscCall(DMSNESSetMFFunction(dm,SNESComputeFunction_DMDA,dmdasnes));
  PetscFunctionReturn(0);
}
