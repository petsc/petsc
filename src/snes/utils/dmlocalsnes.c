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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(sdm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSNESDuplicate_DMLocal(DMSNES oldsdm,DMSNES sdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(sdm,(DMSNES_Local**)&sdm->data);CHKERRQ(ierr);
  if (oldsdm->data) {
    ierr = PetscMemcpy(sdm->data,oldsdm->data,sizeof(DMSNES_Local));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalSNESGetContext(DM dm,DMSNES sdm,DMSNES_Local **dmlocalsnes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *dmlocalsnes = NULL;
  if (!sdm->data) {
    ierr = PetscNewLog(dm,(DMSNES_Local**)&sdm->data);CHKERRQ(ierr);

    sdm->ops->destroy   = DMSNESDestroy_DMLocal;
    sdm->ops->duplicate = DMSNESDuplicate_DMLocal;
  }
  *dmlocalsnes = (DMSNES_Local*)sdm->data;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputeFunction_DMLocal(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMSNES_Local   *dmlocalsnes = (DMSNES_Local*)ctx;
  Vec            Xloc,Floc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Floc);CHKERRQ(ierr);
  ierr = VecZeroEntries(Xloc);CHKERRQ(ierr);
  if (dmlocalsnes->boundarylocal) {ierr = (*dmlocalsnes->boundarylocal)(dm,Xloc,dmlocalsnes->boundarylocalctx);CHKERRQ(ierr);}
  ierr = VecZeroEntries(Floc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = (*dmlocalsnes->residuallocal)(dm,Xloc,Floc,dmlocalsnes->residuallocalctx);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Floc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESComputeJacobian_DMLocal(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMSNES_Local   *dmlocalsnes = (DMSNES_Local*)ctx;
  Vec            Xloc;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  if (dmlocalsnes->jacobianlocal) {
    ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
    ierr = VecZeroEntries(Xloc);CHKERRQ(ierr);
    if (dmlocalsnes->boundarylocal) {ierr = (*dmlocalsnes->boundarylocal)(dm,Xloc,dmlocalsnes->boundarylocalctx);CHKERRQ(ierr);}
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmlocalsnes->jacobianlocal)(dm,Xloc,A,B,dmlocalsnes->jacobianlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  } else {
    MatFDColoring fdcoloring;
    ierr = PetscObjectQuery((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject*)&fdcoloring);CHKERRQ(ierr);
    if (!fdcoloring) {
      ISColoring coloring;

      ierr = DMCreateColoring(dm,dm->coloringtype,&coloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(B,coloring,&fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&coloring);CHKERRQ(ierr);
      switch (dm->coloringtype) {
      case IS_COLORING_GLOBAL:
        ierr = MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))SNESComputeFunction_DMLocal,dmlocalsnes);CHKERRQ(ierr);
        break;
      default: SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_SUP,"No support for coloring type '%s'",ISColoringTypes[dm->coloringtype]);
      }
      ierr = PetscObjectSetOptionsPrefix((PetscObject)fdcoloring,((PetscObject)dm)->prefix);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetUp(B,coloring,fdcoloring);CHKERRQ(ierr);
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
    ierr  = MatFDColoringApply(B,fdcoloring,X,snes);CHKERRQ(ierr);
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetFunctionLocal - set a local residual evaluation function. This function is called with local vector
      containing the local vector information PLUS ghost point information. It should compute a result for all local
      elements and DMSNES will automatically accumulate the overlapping values.

   Logically Collective

   Input Arguments:
+  dm - DM to associate callback with
.  func - local residual evaluation
-  ctx - optional context for local residual evaluation

   Level: beginner

.seealso: DMSNESSetFunction(), DMDASNESSetJacobianLocal(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMSNESSetFunctionLocal(DM dm,PetscErrorCode (*func)(DM,Vec,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;
  DMSNES_Local   *dmlocalsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMLocalSNESGetContext(dm,sdm,&dmlocalsnes);CHKERRQ(ierr);

  dmlocalsnes->residuallocal    = func;
  dmlocalsnes->residuallocalctx = ctx;

  ierr = DMSNESSetFunction(dm,SNESComputeFunction_DMLocal,dmlocalsnes);CHKERRQ(ierr);
  if (!sdm->ops->computejacobian) {  /* Call us for the Jacobian too, can be overridden by the user. */
    ierr = DMSNESSetJacobian(dm,SNESComputeJacobian_DMLocal,dmlocalsnes);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetBoundaryLocal - set a local boundary value function. This function is called with local vector
      containing the local vector information PLUS ghost point information. It should insert values into the local
      vector that do not come from the global vector, such as essential boundary condition data.

   Logically Collective

   Input Arguments:
+  dm - DM to associate callback with
.  func - local boundary value evaluation
-  ctx - optional context for local boundary value evaluation

   Level: intermediate

.seealso: DMSNESSetFunctionLocal(), DMDASNESSetJacobianLocal()
@*/
PetscErrorCode DMSNESSetBoundaryLocal(DM dm,PetscErrorCode (*func)(DM,Vec,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;
  DMSNES_Local   *dmlocalsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMLocalSNESGetContext(dm,sdm,&dmlocalsnes);CHKERRQ(ierr);

  dmlocalsnes->boundarylocal    = func;
  dmlocalsnes->boundarylocalctx = ctx;

  PetscFunctionReturn(0);
}

/*@C
   DMSNESSetJacobianLocal - set a local Jacobian evaluation function

   Logically Collective

   Input Arguments:
+  dm - DM to associate callback with
.  func - local Jacobian evaluation
-  ctx - optional context for local Jacobian evaluation

   Level: beginner

.seealso: DMSNESSetJacobian(), DMDASNESSetJacobian(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMSNESSetJacobianLocal(DM dm,PetscErrorCode (*func)(DM,Vec,Mat,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;
  DMSNES         sdm;
  DMSNES_Local   *dmlocalsnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNESWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMLocalSNESGetContext(dm,sdm,&dmlocalsnes);CHKERRQ(ierr);

  dmlocalsnes->jacobianlocal    = func;
  dmlocalsnes->jacobianlocalctx = ctx;

  ierr = DMSNESSetJacobian(dm,SNESComputeJacobian_DMLocal,dmlocalsnes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

