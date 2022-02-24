#include <petsc/private/dmimpl.h>
#include <petsc/private/snesimpl.h>   /*I "petscsnes.h" I*/

typedef struct {
  PetscErrorCode (*residuallocal)(DM,Vec,Vec,void*);
  PetscErrorCode (*jacobianlocal)(DM,Vec,Mat,Mat,void*);
  PetscErrorCode (*boundarylocal)(DM,Vec,void*);
  void *residuallocalctx;
  void *jacobianlocalctx;
  void *boundarylocalctx;
} DMSNES_Local;

static PetscErrorCode DMSNESDestroy_DMLocal(DMSNES sdm)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(sdm->data));
  sdm->data = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSNESDuplicate_DMLocal(DMSNES oldsdm,DMSNES sdm)
{
  PetscFunctionBegin;
  if (sdm->data != oldsdm->data) {
    CHKERRQ(PetscFree(sdm->data));
    CHKERRQ(PetscNewLog(sdm,(DMSNES_Local**)&sdm->data));
    if (oldsdm->data) CHKERRQ(PetscMemcpy(sdm->data,oldsdm->data,sizeof(DMSNES_Local)));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalSNESGetContext(DM dm,DMSNES sdm,DMSNES_Local **dmlocalsnes)
{
  PetscFunctionBegin;
  *dmlocalsnes = NULL;
  if (!sdm->data) {
    CHKERRQ(PetscNewLog(dm,(DMSNES_Local**)&sdm->data));

    sdm->ops->destroy   = DMSNESDestroy_DMLocal;
    sdm->ops->duplicate = DMSNESDuplicate_DMLocal;
  }
  *dmlocalsnes = (DMSNES_Local*)sdm->data;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputeFunction_DMLocal(SNES snes,Vec X,Vec F,void *ctx)
{
  DMSNES_Local  *dmlocalsnes = (DMSNES_Local *) ctx;
  DM             dm;
  Vec            Xloc,Floc;
  PetscBool      transform;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetLocalVector(dm,&Xloc));
  CHKERRQ(DMGetLocalVector(dm,&Floc));
  CHKERRQ(VecZeroEntries(Xloc));
  CHKERRQ(VecZeroEntries(Floc));
  /* Non-conforming routines needs boundary values before G2L */
  if (dmlocalsnes->boundarylocal) CHKERRQ((*dmlocalsnes->boundarylocal)(dm,Xloc,dmlocalsnes->boundarylocalctx));
  CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  /* Need to reset boundary values if we transformed */
  CHKERRQ(DMHasBasisTransform(dm, &transform));
  if (transform && dmlocalsnes->boundarylocal) CHKERRQ((*dmlocalsnes->boundarylocal)(dm,Xloc,dmlocalsnes->boundarylocalctx));
  CHKMEMQ;
  CHKERRQ((*dmlocalsnes->residuallocal)(dm,Xloc,Floc,dmlocalsnes->residuallocalctx));
  CHKMEMQ;
  CHKERRQ(VecZeroEntries(F));
  CHKERRQ(DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F));
  CHKERRQ(DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F));
  CHKERRQ(DMRestoreLocalVector(dm,&Floc));
  CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  {
    char        name[PETSC_MAX_PATH_LEN];
    char        oldname[PETSC_MAX_PATH_LEN];
    const char *tmp;
    PetscInt    it;

    CHKERRQ(SNESGetIterationNumber(snes, &it));
    CHKERRQ(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "Solution, Iterate %d", (int) it));
    CHKERRQ(PetscObjectGetName((PetscObject) X, &tmp));
    CHKERRQ(PetscStrncpy(oldname, tmp, PETSC_MAX_PATH_LEN-1));
    CHKERRQ(PetscObjectSetName((PetscObject) X, name));
    CHKERRQ(VecViewFromOptions(X, (PetscObject) snes, "-dmsnes_solution_vec_view"));
    CHKERRQ(PetscObjectSetName((PetscObject) X, oldname));
    CHKERRQ(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "Residual, Iterate %d", (int) it));
    CHKERRQ(PetscObjectSetName((PetscObject) F, name));
    CHKERRQ(VecViewFromOptions(F, (PetscObject) snes, "-dmsnes_residual_vec_view"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputeJacobian_DMLocal(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  DMSNES_Local  *dmlocalsnes = (DMSNES_Local *) ctx;
  DM             dm;
  Vec            Xloc;
  PetscBool      transform;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  if (dmlocalsnes->jacobianlocal) {
    CHKERRQ(DMGetLocalVector(dm,&Xloc));
    CHKERRQ(VecZeroEntries(Xloc));
    /* Non-conforming routines needs boundary values before G2L */
    if (dmlocalsnes->boundarylocal) CHKERRQ((*dmlocalsnes->boundarylocal)(dm,Xloc,dmlocalsnes->boundarylocalctx));
    CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
    CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
    /* Need to reset boundary values if we transformed */
    CHKERRQ(DMHasBasisTransform(dm, &transform));
    if (transform && dmlocalsnes->boundarylocal) CHKERRQ((*dmlocalsnes->boundarylocal)(dm,Xloc,dmlocalsnes->boundarylocalctx));
    CHKMEMQ;
    CHKERRQ((*dmlocalsnes->jacobianlocal)(dm,Xloc,A,B,dmlocalsnes->jacobianlocalctx));
    CHKMEMQ;
    CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  } else {
    MatFDColoring fdcoloring;
    CHKERRQ(PetscObjectQuery((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject*)&fdcoloring));
    if (!fdcoloring) {
      ISColoring coloring;

      CHKERRQ(DMCreateColoring(dm,dm->coloringtype,&coloring));
      CHKERRQ(MatFDColoringCreate(B,coloring,&fdcoloring));
      CHKERRQ(ISColoringDestroy(&coloring));
      switch (dm->coloringtype) {
      case IS_COLORING_GLOBAL:
        CHKERRQ(MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))SNESComputeFunction_DMLocal,dmlocalsnes));
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_SUP,"No support for coloring type '%s'",ISColoringTypes[dm->coloringtype]);
      }
      CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)fdcoloring,((PetscObject)dm)->prefix));
      CHKERRQ(MatFDColoringSetFromOptions(fdcoloring));
      CHKERRQ(MatFDColoringSetUp(B,coloring,fdcoloring));
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
   DMSNESSetFunctionLocal - set a local residual evaluation function. This function is called with local vector
      containing the local vector information PLUS ghost point information. It should compute a result for all local
      elements and DMSNES will automatically accumulate the overlapping values.

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  func - local residual evaluation
-  ctx - optional context for local residual evaluation

   Level: beginner

.seealso: DMSNESSetFunction(), DMDASNESSetJacobianLocal(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMSNESSetFunctionLocal(DM dm,PetscErrorCode (*func)(DM,Vec,Vec,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_Local   *dmlocalsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMSNESWrite(dm,&sdm));
  CHKERRQ(DMLocalSNESGetContext(dm,sdm,&dmlocalsnes));

  dmlocalsnes->residuallocal    = func;
  dmlocalsnes->residuallocalctx = ctx;

  CHKERRQ(DMSNESSetFunction(dm,SNESComputeFunction_DMLocal,dmlocalsnes));
  if (!sdm->ops->computejacobian) {  /* Call us for the Jacobian too, can be overridden by the user. */
    CHKERRQ(DMSNESSetJacobian(dm,SNESComputeJacobian_DMLocal,dmlocalsnes));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetBoundaryLocal - set a local boundary value function. This function is called with local vector
      containing the local vector information PLUS ghost point information. It should insert values into the local
      vector that do not come from the global vector, such as essential boundary condition data.

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  func - local boundary value evaluation
-  ctx - optional context for local boundary value evaluation

   Level: intermediate

.seealso: DMSNESSetFunctionLocal(), DMDASNESSetJacobianLocal()
@*/
PetscErrorCode DMSNESSetBoundaryLocal(DM dm,PetscErrorCode (*func)(DM,Vec,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_Local   *dmlocalsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMSNESWrite(dm,&sdm));
  CHKERRQ(DMLocalSNESGetContext(dm,sdm,&dmlocalsnes));

  dmlocalsnes->boundarylocal    = func;
  dmlocalsnes->boundarylocalctx = ctx;

  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetJacobianLocal - set a local Jacobian evaluation function

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  func - local Jacobian evaluation
-  ctx - optional context for local Jacobian evaluation

   Level: beginner

.seealso: DMSNESSetJacobian(), DMDASNESSetJacobian(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMSNESSetJacobianLocal(DM dm,PetscErrorCode (*func)(DM,Vec,Mat,Mat,void*),void *ctx)
{
  DMSNES         sdm;
  DMSNES_Local   *dmlocalsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMSNESWrite(dm,&sdm));
  CHKERRQ(DMLocalSNESGetContext(dm,sdm,&dmlocalsnes));

  dmlocalsnes->jacobianlocal    = func;
  dmlocalsnes->jacobianlocalctx = ctx;

  CHKERRQ(DMSNESSetJacobian(dm,SNESComputeJacobian_DMLocal,dmlocalsnes));
  PetscFunctionReturn(0);
}

/*@C
   DMSNESGetFunctionLocal - get the local residual evaluation function information set with DMSNESSetFunctionLocal.

   Not Collective

   Input Parameter:
.  dm - DM with the associated callback

   Output Parameters:
+  func - local residual evaluation
-  ctx - context for local residual evaluation

   Level: beginner

.seealso: DMSNESSetFunction(), DMSNESSetFunctionLocal(), DMDASNESSetJacobianLocal(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMSNESGetFunctionLocal(DM dm,PetscErrorCode (**func)(DM,Vec,Vec,void*),void **ctx)
{
  DMSNES         sdm;
  DMSNES_Local   *dmlocalsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMSNES(dm,&sdm));
  CHKERRQ(DMLocalSNESGetContext(dm,sdm,&dmlocalsnes));
  if (func) *func = dmlocalsnes->residuallocal;
  if (ctx)  *ctx  = dmlocalsnes->residuallocalctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESGetBoundaryLocal - get the local boundary value function set with DMSNESSetBoundaryLocal.

   Not Collective

   Input Parameter:
.  dm - DM with the associated callback

   Output Parameters:
+  func - local boundary value evaluation
-  ctx - context for local boundary value evaluation

   Level: intermediate

.seealso: DMSNESSetFunctionLocal(), DMSNESSetBoundaryLocal(), DMDASNESSetJacobianLocal()
@*/
PetscErrorCode DMSNESGetBoundaryLocal(DM dm,PetscErrorCode (**func)(DM,Vec,void*),void **ctx)
{
  DMSNES         sdm;
  DMSNES_Local   *dmlocalsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMSNES(dm,&sdm));
  CHKERRQ(DMLocalSNESGetContext(dm,sdm,&dmlocalsnes));
  if (func) *func = dmlocalsnes->boundarylocal;
  if (ctx)  *ctx  = dmlocalsnes->boundarylocalctx;
  PetscFunctionReturn(0);
}

/*@C
   DMSNESGetJacobianLocal - the local Jacobian evaluation function set with DMSNESSetJacobianLocal.

   Logically Collective

   Input Parameter:
.  dm - DM with the associated callback

   Output Parameters:
+  func - local Jacobian evaluation
-  ctx - context for local Jacobian evaluation

   Level: beginner

.seealso: DMSNESSetJacobianLocal(), DMDASNESSetJacobian(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMSNESGetJacobianLocal(DM dm,PetscErrorCode (**func)(DM,Vec,Mat,Mat,void*),void **ctx)
{
  DMSNES         sdm;
  DMSNES_Local   *dmlocalsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMSNES(dm,&sdm));
  CHKERRQ(DMLocalSNESGetContext(dm,sdm,&dmlocalsnes));
  if (func) *func = dmlocalsnes->jacobianlocal;
  if (ctx)  *ctx  = dmlocalsnes->jacobianlocalctx;
  PetscFunctionReturn(0);
}
