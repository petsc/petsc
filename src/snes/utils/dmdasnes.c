#include <petscdmda.h>          /*I "petscdmda.h" I*/
#include <petsc-private/dmimpl.h>
#include <petsc-private/snesimpl.h>   /*I "petscsnes.h" I*/

/* This structure holds the user-provided DMDA callbacks */
typedef struct {
  PetscErrorCode (*residuallocal)(DMDALocalInfo*,void*,void*,void*);
  PetscErrorCode (*jacobianlocal)(DMDALocalInfo*,void*,Mat,Mat,MatStructure*,void*);
  void *residuallocalctx;
  void *jacobianlocalctx;
  InsertMode residuallocalimode;
} DM_DA_SNES;

#undef __FUNCT__
#define __FUNCT__ "SNESDMDestroy_DMDA"
static PetscErrorCode SNESDMDestroy_DMDA(SNESDM sdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(sdm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASNESGetContext"
static PetscErrorCode DMDASNESGetContext(DM dm,SNESDM sdm,DM_DA_SNES **dmdasnes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *dmdasnes = PETSC_NULL;
  if (!sdm->data) {
    ierr = PetscNewLog(dm,DM_DA_SNES,&sdm->data);CHKERRQ(ierr);
    sdm->destroy = SNESDMDestroy_DMDA;
  }
  *dmdasnes = (DM_DA_SNES*)sdm->data;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESComputeFunction_DMDA"
/*
  This function should eventually replace:
    DMDAComputeFunction() and DMDAComputeFunction1()
 */
static PetscErrorCode SNESComputeFunction_DMDA(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DM_DA_SNES     *dmdasnes = (DM_DA_SNES*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  if (!dmdasnes->residuallocal) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"Corrupt context");
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
    ierr = (*dmdasnes->residuallocal)(&info,x,f,dmdasnes->residuallocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,F,&f);CHKERRQ(ierr);
  } break;
  case ADD_VALUES: {
    Vec Floc;
    ierr = DMGetLocalVector(dm,&Floc);CHKERRQ(ierr);
    ierr = VecZeroEntries(Floc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Floc,&f);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdasnes->residuallocal)(&info,x,f,dmdasnes->residuallocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,Floc,&f);CHKERRQ(ierr);
    ierr = VecZeroEntries(F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Floc);CHKERRQ(ierr);
  } break;
  default: SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdasnes->residuallocalimode);
  }
  ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESComputeJacobian_DMDA"
/*
  This function should eventually replace:
    DMComputeJacobian() and DMDAComputeJacobian1()
 */
static PetscErrorCode SNESComputeJacobian_DMDA(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *mstr,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DM_DA_SNES     *dmdasnes = (DM_DA_SNES*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x;

  PetscFunctionBegin;
  if (!dmdasnes->residuallocal) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"Corrupt context");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);

  if (dmdasnes->jacobianlocal) {
    ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdasnes->jacobianlocal)(&info,x,*A,*B,mstr,dmdasnes->jacobianlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  } else {
    MatFDColoring fdcoloring;
    ierr = PetscObjectQuery((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject*)&fdcoloring);CHKERRQ(ierr);
    if (!fdcoloring) {
      ISColoring     coloring;

      ierr = DMCreateColoring(dm,dm->coloringtype,MATAIJ,&coloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(*B,coloring,&fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&coloring);CHKERRQ(ierr);
      switch (dm->coloringtype) {
      case IS_COLORING_GLOBAL:
        ierr = MatFDColoringSetFunction(fdcoloring,(PetscErrorCode(*)(void))SNESComputeFunction_DMDA,dmdasnes);CHKERRQ(ierr);
        break;
      default: SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_SUP,"No support for coloring type '%s'",ISColoringTypes[dm->coloringtype]);
      }
      ierr = PetscObjectSetOptionsPrefix((PetscObject)fdcoloring,((PetscObject)dm)->prefix);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)dm,"DMDASNES_FDCOLORING",(PetscObject)fdcoloring);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)fdcoloring);CHKERRQ(ierr);

      /* The following breaks an ugly reference counting loop that deserves a paragraph. MatFDColoringApply() will call
       * VecDuplicate() with the state Vec and store inside the MatFDColoring. This Vec will duplicate the Vec, but the
       * MatFDColoring is composed with the DM. We dereference the DM here so that the reference count will eventually
       * drop to 0. Note the code in DMDestroy() that exits early for a negative reference count. That code path will be
       * taken when the PetscOList for the Vec inside MatFDColoring is destroyed.
       */
      ierr = PetscObjectDereference((PetscObject)dm);CHKERRQ(ierr);
    }
    *mstr = SAME_NONZERO_PATTERN;
    ierr = MatFDColoringApply(*B,fdcoloring,X,mstr,snes);CHKERRQ(ierr);
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (*A != *B) {
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASNESSetFunctionLocal"
/*@C
   DMDASNESSetFunctionLocal - set a local residual evaluation function

   Logically Collective

   Input Arguments:
+  dm - DM to associate callback with
.  func - local residual evaluation
-  ctx - optional context for local residual evaluation

   Calling sequence for func:
+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  imode - INSERT_VALUES if local function computes owned part, ADD_VALUES if it contributes to ghosted part
.  x - dimensional pointer to state at which to evaluate residual
.  f - dimensional pointer to residual, write the residual here
-  ctx - optional context passed above

   Level: beginner

.seealso: DMSNESSetFunction(), DMDASNESSetJacobian(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMDASNESSetFunctionLocal(DM dm,InsertMode imode,PetscErrorCode (*func)(DMDALocalInfo*,void*,void*,void*),void *ctx)
{
  PetscErrorCode ierr;
  SNESDM         sdm;
  DM_DA_SNES     *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDASNESGetContext(dm,sdm,&dmdasnes);CHKERRQ(ierr);
  dmdasnes->residuallocalimode = imode;
  dmdasnes->residuallocal = func;
  dmdasnes->residuallocalctx = ctx;
  ierr = DMSNESSetFunction(dm,SNESComputeFunction_DMDA,dmdasnes);CHKERRQ(ierr);
  if (!sdm->computejacobian) {  /* Call us for the Jacobian too, can be overridden by the user. */
    ierr = DMSNESSetJacobian(dm,SNESComputeJacobian_DMDA,dmdasnes);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDASNESSetJacobianLocal"
/*@C
   DMDASNESSetJacobianLocal - set a local residual evaluation function

   Logically Collective

   Input Arguments:
+  dm - DM to associate callback with
.  func - local residual evaluation
-  ctx - optional context for local residual evaluation

   Calling sequence for func:
+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  x - dimensional pointer to state at which to evaluate residual
.  f - dimensional pointer to residual, write the residual here
-  ctx - optional context passed above

   Level: beginner

.seealso: DMSNESSetJacobian(), DMDASNESSetJacobian(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMDASNESSetJacobianLocal(DM dm,PetscErrorCode (*func)(DMDALocalInfo*,void*,Mat,Mat,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;
  SNESDM         sdm;
  DM_DA_SNES     *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDASNESGetContext(dm,sdm,&dmdasnes);CHKERRQ(ierr);
  dmdasnes->jacobianlocal = func;
  dmdasnes->jacobianlocalctx = ctx;
  ierr = DMSNESSetJacobian(dm,SNESComputeJacobian_DMDA,dmdasnes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
