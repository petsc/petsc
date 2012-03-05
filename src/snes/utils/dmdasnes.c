#include <petscdmda.h>          /*I "petscdmda.h" I*/
#include <private/snesimpl.h>   /*I "petscsnes.h" I*/

/* This structure holds the user-provided DMDA callbacks */
typedef struct {
  PetscErrorCode (*residuallocal)(DMDALocalInfo*,void*,void*,void*);
  PetscErrorCode (*jacobianlocal)(DMDALocalInfo*,void*,Mat,Mat,MatStructure*,void*);
  void *residuallocalctx;
  void *jacobianlocalctx;
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
  if (!dmdasnes->residuallocal) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_PLIB,"Corrupt context");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,F,&f);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = (*dmdasnes->residuallocal)(&info,x,f,dmdasnes->residuallocalctx);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
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
.  x - dimensional pointer to state at which to evaluate residual
.  f - dimensional pointer to residual, write the residual here
-  ctx - optional context passed above

   Level: beginner

.seealso: DMSNESSetFunction(), DMDASNESSetJacobian(), DMDACreate1d(), DMDACreate2d(), DMDACreate3d()
@*/
PetscErrorCode DMDASNESSetFunctionLocal(DM dm,PetscErrorCode (*func)(DMDALocalInfo*,void*,void*,void*),void *ctx)
{
  PetscErrorCode ierr;
  SNESDM         sdm;
  DM_DA_SNES     *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESGetContext(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDASNESGetContext(dm,sdm,&dmdasnes);CHKERRQ(ierr);
  dmdasnes->residuallocal = func;
  dmdasnes->residuallocalctx = ctx;
  ierr = DMSNESSetFunction(dm,SNESComputeFunction_DMDA,dmdasnes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
