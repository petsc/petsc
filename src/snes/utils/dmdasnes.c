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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(sdm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSNESDuplicate_DMDA(DMSNES oldsdm,DMSNES sdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(sdm,(DMSNES_DA**)&sdm->data);CHKERRQ(ierr);
  if (oldsdm->data) {
    ierr = PetscMemcpy(sdm->data,oldsdm->data,sizeof(DMSNES_DA));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDASNESGetContext(DM dm,DMSNES sdm,DMSNES_DA  **dmdasnes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *dmdasnes = NULL;
  if (!sdm->data) {
    ierr = PetscNewLog(dm,(DMSNES_DA**)&sdm->data);CHKERRQ(ierr);
    sdm->ops->destroy   = DMSNESDestroy_DMDA;
    sdm->ops->duplicate = DMSNESDuplicate_DMDA;
  }
  *dmdasnes = (DMSNES_DA*)sdm->data;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputeFunction_DMDA(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMSNES_DA      *dmdasnes = (DMSNES_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  if (!dmdasnes->residuallocal) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
  switch (dmdasnes->residuallocalimode) {
  case INSERT_VALUES: {
    ierr = DMDAVecGetArray(dm,F,&f);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(SNES_FunctionEval,snes,X,F,0);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdasnes->residuallocal)(&info,x,f,dmdasnes->residuallocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = PetscLogEventEnd(SNES_FunctionEval,snes,X,F,0);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dm,F,&f);CHKERRQ(ierr);
  } break;
  case ADD_VALUES: {
    Vec Floc;
    ierr = DMGetLocalVector(dm,&Floc);CHKERRQ(ierr);
    ierr = VecZeroEntries(Floc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Floc,&f);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(SNES_FunctionEval,snes,X,F,0);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdasnes->residuallocal)(&info,x,f,dmdasnes->residuallocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = PetscLogEventEnd(SNES_FunctionEval,snes,X,F,0);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dm,Floc,&f);CHKERRQ(ierr);
    ierr = VecZeroEntries(F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Floc);CHKERRQ(ierr);
  } break;
  default: SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdasnes->residuallocalimode);
  }
  ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  if (snes->domainerror) {
    ierr = VecSetInf(F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputeObjective_DMDA(SNES snes,Vec X,PetscReal *ob,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMSNES_DA      *dmdasnes = (DMSNES_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidPointer(ob,3);
  if (!dmdasnes->objectivelocal) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = (*dmdasnes->objectivelocal)(&info,x,ob,dmdasnes->objectivelocalctx);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Routine is called by example, hence must be labeled PETSC_EXTERN */
PETSC_EXTERN PetscErrorCode SNESComputeJacobian_DMDA(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMSNES_DA      *dmdasnes = (DMSNES_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x;

  PetscFunctionBegin;
  if (!dmdasnes->residuallocal) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);

  if (dmdasnes->jacobianlocal) {
    ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdasnes->jacobianlocal)(&info,x,A,B,dmdasnes->jacobianlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  } else {
    MatFDColoring fdcoloring;
    ierr = PetscObjectQuery((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject*)&fdcoloring);CHKERRQ(ierr);
    if (!fdcoloring) {
      ISColoring coloring;

      ierr = DMCreateColoring(dm,dm->coloringtype,&coloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(B,coloring,&fdcoloring);CHKERRQ(ierr);
      switch (dm->coloringtype) {
      case IS_COLORING_GLOBAL:
        ierr = MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))SNESComputeFunction_DMDA,dmdasnes);CHKERRQ(ierr);
        break;
      default: SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_SUP,"No support for coloring type '%s'",ISColoringTypes[dm->coloringtype]);
      }
      ierr = PetscObjectSetOptionsPrefix((PetscObject)fdcoloring,((PetscObject)dm)->prefix);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetUp(B,coloring,fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&coloring);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject)fdcoloring);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)fdcoloring);CHKERRQ(ierr);

      /* The following breaks an ugly reference counting loop that deserves a paragraph. MatFDColoringApply() will call
       * VecDuplicate() with the state Vec and store inside the MatFDColoring. This Vec will duplicate the Vec, but the
       * MatFDColoring is composed with the DM. We dereference the DM here so that the reference count will eventually
       * drop to 0. Note the code in DMDestroy() that exits early for a negative reference count. That code path will be
       * taken when the PetscObjectList for the Vec inside MatFDColoring is destroyed.
       */
      ierr = PetscObjectDereference((PetscObject)dm);CHKERRQ(ierr);
    }
    ierr = MatFDColoringApply(B,fdcoloring,X,snes);CHKERRQ(ierr);
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetFunctionLocal - set a local residual evaluation function

   Logically Collective

   Input Arguments:
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
  PetscErrorCode ierr;
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDASNESGetContext(dm,sdm,&dmdasnes);CHKERRQ(ierr);

  dmdasnes->residuallocalimode = imode;
  dmdasnes->residuallocal      = func;
  dmdasnes->residuallocalctx   = ctx;

  ierr = DMSNESSetFunction(dm,SNESComputeFunction_DMDA,dmdasnes);CHKERRQ(ierr);
  if (!sdm->ops->computejacobian) {  /* Call us for the Jacobian too, can be overridden by the user. */
    ierr = DMSNESSetJacobian(dm,SNESComputeJacobian_DMDA,dmdasnes);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetJacobianLocal - set a local Jacobian evaluation function

   Logically Collective

   Input Arguments:
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
  PetscErrorCode ierr;
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDASNESGetContext(dm,sdm,&dmdasnes);CHKERRQ(ierr);

  dmdasnes->jacobianlocal    = func;
  dmdasnes->jacobianlocalctx = ctx;

  ierr = DMSNESSetJacobian(dm,SNESComputeJacobian_DMDA,dmdasnes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetObjectiveLocal - set a local residual evaluation function

   Logically Collective

   Input Arguments:
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
  PetscErrorCode ierr;
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDASNESGetContext(dm,sdm,&dmdasnes);CHKERRQ(ierr);

  dmdasnes->objectivelocal    = func;
  dmdasnes->objectivelocalctx = ctx;

  ierr = DMSNESSetObjective(dm,SNESComputeObjective_DMDA,dmdasnes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputePicard_DMDA(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMSNES_DA      *dmdasnes = (DMSNES_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  if (!dmdasnes->rhsplocal) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
  switch (dmdasnes->residuallocalimode) {
  case INSERT_VALUES: {
    ierr = DMDAVecGetArray(dm,F,&f);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdasnes->rhsplocal)(&info,x,f,dmdasnes->picardlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,F,&f);CHKERRQ(ierr);
  } break;
  case ADD_VALUES: {
    Vec Floc;
    ierr = DMGetLocalVector(dm,&Floc);CHKERRQ(ierr);
    ierr = VecZeroEntries(Floc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Floc,&f);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdasnes->rhsplocal)(&info,x,f,dmdasnes->picardlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,Floc,&f);CHKERRQ(ierr);
    ierr = VecZeroEntries(F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Floc);CHKERRQ(ierr);
  } break;
  default: SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdasnes->residuallocalimode);
  }
  ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputePicardJacobian_DMDA(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMSNES_DA      *dmdasnes = (DMSNES_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x;

  PetscFunctionBegin;
  if (!dmdasnes->jacobianplocal) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_PLIB,"Corrupt context");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = (*dmdasnes->jacobianplocal)(&info,x,A,B,dmdasnes->picardlocalctx);CHKERRQ(ierr);
  CHKMEMQ;
  ierr  = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr  = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   DMDASNESSetPicardLocal - set a local right hand side and matrix evaluation function for Picard iteration

   Logically Collective

   Input Arguments:
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
    ierr = SNESSetFunction(snes,NULL,SNESPicardComputeFunction,&user);CHKERRQ(ierr);
    in their code before calling this routine.

   Level: beginner

.seealso: DMSNESSetFunction(), DMDASNESSetJacobian(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMDASNESSetPicardLocal(DM dm,InsertMode imode,PetscErrorCode (*func)(DMDALocalInfo*,void*,void*,void*),
                                      PetscErrorCode (*jac)(DMDALocalInfo*,void*,Mat,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;
  DMSNES_DA      *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDASNESGetContext(dm,sdm,&dmdasnes);CHKERRQ(ierr);

  dmdasnes->residuallocalimode = imode;
  dmdasnes->rhsplocal          = func;
  dmdasnes->jacobianplocal     = jac;
  dmdasnes->picardlocalctx     = ctx;

  ierr = DMSNESSetPicard(dm,SNESComputePicard_DMDA,SNESComputePicardJacobian_DMDA,dmdasnes);CHKERRQ(ierr);
  ierr = DMSNESSetMFFunction(dm,SNESComputeFunction_DMDA,dmdasnes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
