#include <petsc/private/dmimpl.h>
#include <petsc/private/tsimpl.h>   /*I "petscts.h" I*/

typedef struct {
  PetscErrorCode (*boundarylocal)(DM,PetscReal,Vec,Vec,void*);
  PetscErrorCode (*ifunctionlocal)(DM,PetscReal,Vec,Vec,Vec,void*);
  PetscErrorCode (*ijacobianlocal)(DM,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
  PetscErrorCode (*rhsfunctionlocal)(DM,PetscReal,Vec,Vec,void*);
  void *boundarylocalctx;
  void *ifunctionlocalctx;
  void *ijacobianlocalctx;
  void *rhsfunctionlocalctx;
} DMTS_Local;

static PetscErrorCode DMTSDestroy_DMLocal(DMTS tdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(tdm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSDuplicate_DMLocal(DMTS oldtdm, DMTS tdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tdm, (DMTS_Local **) &tdm->data);CHKERRQ(ierr);
  if (oldtdm->data) {ierr = PetscMemcpy(tdm->data, oldtdm->data, sizeof(DMTS_Local));CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalTSGetContext(DM dm, DMTS tdm, DMTS_Local **dmlocalts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *dmlocalts = NULL;
  if (!tdm->data) {
    ierr = PetscNewLog(dm, (DMTS_Local **) &tdm->data);CHKERRQ(ierr);

    tdm->ops->destroy   = DMTSDestroy_DMLocal;
    tdm->ops->duplicate = DMTSDuplicate_DMLocal;
  }
  *dmlocalts = (DMTS_Local *) tdm->data;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeIFunction_DMLocal(TS ts, PetscReal time, Vec X, Vec X_t, Vec F, void *ctx)
{
  DM             dm;
  Vec            locX, locX_t, locF;
  DMTS_Local    *dmlocalts = (DMTS_Local *) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(X_t,VEC_CLASSID,4);
  PetscValidHeaderSpecific(F,VEC_CLASSID,5);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locX_t);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locF);CHKERRQ(ierr);
  ierr = VecZeroEntries(locX);CHKERRQ(ierr);
  ierr = VecZeroEntries(locX_t);CHKERRQ(ierr);
  if (dmlocalts->boundarylocal) {ierr = (*dmlocalts->boundarylocal)(dm, time, locX, locX_t,dmlocalts->boundarylocalctx);CHKERRQ(ierr);}
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X_t, INSERT_VALUES, locX_t);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X_t, INSERT_VALUES, locX_t);CHKERRQ(ierr);
  ierr = VecZeroEntries(locF);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = (*dmlocalts->ifunctionlocal)(dm, time, locX, locX_t, locF, dmlocalts->ifunctionlocalctx);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, locF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, locF, ADD_VALUES, F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX_t);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeRHSFunction_DMLocal(TS ts, PetscReal time, Vec X, Vec F, void *ctx)
{
  DM             dm;
  Vec            locX;
  DMTS_Local    *dmlocalts = (DMTS_Local *) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(F,VEC_CLASSID,4);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = VecZeroEntries(locX);CHKERRQ(ierr);
  if (dmlocalts->boundarylocal) {ierr = (*dmlocalts->boundarylocal)(dm,time,locX,NULL,dmlocalts->boundarylocalctx);CHKERRQ(ierr);}
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = (*dmlocalts->rhsfunctionlocal)(dm, time, locX, F, dmlocalts->rhsfunctionlocalctx);CHKERRQ(ierr);
  CHKMEMQ;
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeIJacobian_DMLocal(TS ts, PetscReal time, Vec X, Vec X_t, PetscReal a, Mat A, Mat B, void *ctx)
{
  DM             dm;
  Vec            locX, locX_t;
  DMTS_Local    *dmlocalts = (DMTS_Local *) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  if (dmlocalts->ijacobianlocal) {
    ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm, &locX_t);CHKERRQ(ierr);
    ierr = VecZeroEntries(locX);CHKERRQ(ierr);
    ierr = VecZeroEntries(locX_t);CHKERRQ(ierr);
    if (dmlocalts->boundarylocal) {ierr = (*dmlocalts->boundarylocal)(dm,time,locX,locX_t,dmlocalts->boundarylocalctx);CHKERRQ(ierr);}
    ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, X_t, INSERT_VALUES, locX_t);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, X_t, INSERT_VALUES, locX_t);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmlocalts->ijacobianlocal)(dm, time, locX, locX_t, a, A, B, dmlocalts->ijacobianlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locX_t);CHKERRQ(ierr);
  } else {
    MatFDColoring fdcoloring;
    ierr = PetscObjectQuery((PetscObject) dm, "DMDASNES_FDCOLORING", (PetscObject *) &fdcoloring);CHKERRQ(ierr);
    if (!fdcoloring) {
      ISColoring coloring;

      ierr = DMCreateColoring(dm, dm->coloringtype, &coloring);CHKERRQ(ierr);
      ierr = MatFDColoringCreate(B, coloring, &fdcoloring);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&coloring);CHKERRQ(ierr);
      switch (dm->coloringtype) {
      case IS_COLORING_GLOBAL:
        ierr = MatFDColoringSetFunction(fdcoloring, (PetscErrorCode (*)(void)) TSComputeIFunction_DMLocal, dmlocalts);CHKERRQ(ierr);
        break;
      default: SETERRQ1(PetscObjectComm((PetscObject) ts), PETSC_ERR_SUP, "No support for coloring type '%s'", ISColoringTypes[dm->coloringtype]);
      }
      ierr = PetscObjectSetOptionsPrefix((PetscObject) fdcoloring, ((PetscObject) dm)->prefix);CHKERRQ(ierr);
      ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);
      ierr = MatFDColoringSetUp(B, coloring, fdcoloring);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "DMDASNES_FDCOLORING", (PetscObject) fdcoloring);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject) fdcoloring);CHKERRQ(ierr);

      /* The following breaks an ugly reference counting loop that deserves a paragraph. MatFDColoringApply() will call
       * VecDuplicate() with the state Vec and store inside the MatFDColoring. This Vec will duplicate the Vec, but the
       * MatFDColoring is composed with the DM. We dereference the DM here so that the reference count will eventually
       * drop to 0. Note the code in DMDestroy() that exits early for a negative reference count. That code path will be
       * taken when the PetscObjectList for the Vec inside MatFDColoring is destroyed.
       */
      ierr = PetscObjectDereference((PetscObject) dm);CHKERRQ(ierr);
    }
    ierr = MatFDColoringApply(B, fdcoloring, X, ts);CHKERRQ(ierr);
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMTSSetBoundaryLocal - set the function for essential boundary data for a local implicit function evaluation.
    It should set the essential boundary data for the local portion of the solution X, as well its time derivative X_t (if it is not NULL).
    Vectors are initialized to zero before this function, so it is only needed for non homogeneous data.

  Note that this function is somewhat optional: boundary data could potentially be inserted by a function passed to
  DMTSSetIFunctionLocal().  The use case for this function is for discretizations with constraints (see
  DMGetDefaultConstraints()): this function inserts boundary values before constraint interpolation.

  Logically Collective

  Input Arguments:
+ dm   - DM to associate callback with
. func - local function evaluation
- ctx  - context for function evaluation

  Level: intermediate

.seealso: DMTSSetIFunction(), DMTSSetIJacobianLocal()
@*/
PetscErrorCode DMTSSetBoundaryLocal(DM dm, PetscErrorCode (*func)(DM, PetscReal, Vec, Vec, void *), void *ctx)
{
  DMTS           tdm;
  DMTS_Local    *dmlocalts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm, &tdm);CHKERRQ(ierr);
  ierr = DMLocalTSGetContext(dm, tdm, &dmlocalts);CHKERRQ(ierr);

  dmlocalts->boundarylocal    = func;
  dmlocalts->boundarylocalctx = ctx;

  PetscFunctionReturn(0);
}

/*@C
  DMTSSetIFunctionLocal - set a local implicit function evaluation function. This function is called with local vector
      containing the local vector information PLUS ghost point information. It should compute a result for all local
      elements and DMTS will automatically accumulate the overlapping values.

  Logically Collective

  Input Arguments:
+ dm   - DM to associate callback with
. func - local function evaluation
- ctx  - context for function evaluation

  Level: beginner

.seealso: DMTSSetIFunction(), DMTSSetIJacobianLocal()
@*/
PetscErrorCode DMTSSetIFunctionLocal(DM dm, PetscErrorCode (*func)(DM, PetscReal, Vec, Vec, Vec, void *), void *ctx)
{
  DMTS           tdm;
  DMTS_Local    *dmlocalts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm, &tdm);CHKERRQ(ierr);
  ierr = DMLocalTSGetContext(dm, tdm, &dmlocalts);CHKERRQ(ierr);

  dmlocalts->ifunctionlocal    = func;
  dmlocalts->ifunctionlocalctx = ctx;

  ierr = DMTSSetIFunction(dm, TSComputeIFunction_DMLocal, dmlocalts);CHKERRQ(ierr);
  if (!tdm->ops->ijacobian) {  /* Call us for the Jacobian too, can be overridden by the user. */
    ierr = DMTSSetIJacobian(dm, TSComputeIJacobian_DMLocal, dmlocalts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMTSSetIJacobianLocal - set a local Jacobian evaluation function

  Logically Collective

  Input Arguments:
+ dm - DM to associate callback with
. func - local Jacobian evaluation
- ctx - optional context for local Jacobian evaluation

  Level: beginner

.seealso: DMTSSetIFunctionLocal(), DMTSSetIJacobian(), DMTSSetIFunction()
@*/
PetscErrorCode DMTSSetIJacobianLocal(DM dm, PetscErrorCode (*func)(DM, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *), void *ctx)
{
  DMTS           tdm;
  DMTS_Local    *dmlocalts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm, &tdm);CHKERRQ(ierr);
  ierr = DMLocalTSGetContext(dm, tdm, &dmlocalts);CHKERRQ(ierr);

  dmlocalts->ijacobianlocal    = func;
  dmlocalts->ijacobianlocalctx = ctx;

  ierr = DMTSSetIJacobian(dm, TSComputeIJacobian_DMLocal, dmlocalts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMTSSetRHSFunctionLocal - set a local rhs function evaluation function. This function is called with local vector
      containing the local vector information PLUS ghost point information. It should compute a result for all local
      elements and DMTS will automatically accumulate the overlapping values.

  Logically Collective

  Input Arguments:
+ dm   - DM to associate callback with
. func - local function evaluation
- ctx  - context for function evaluation

  Level: beginner

.seealso: DMTSSetRHSFunction(), DMTSSetIFunction(), DMTSSetIJacobianLocal()
@*/
PetscErrorCode DMTSSetRHSFunctionLocal(DM dm, PetscErrorCode (*func)(DM, PetscReal, Vec, Vec, void *), void *ctx)
{
  DMTS           tdm;
  DMTS_Local    *dmlocalts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm, &tdm);CHKERRQ(ierr);
  ierr = DMLocalTSGetContext(dm, tdm, &dmlocalts);CHKERRQ(ierr);

  dmlocalts->rhsfunctionlocal    = func;
  dmlocalts->rhsfunctionlocalctx = ctx;

  ierr = DMTSSetRHSFunction(dm, TSComputeRHSFunction_DMLocal, dmlocalts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
