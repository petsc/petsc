#include <petscdmda.h> /*I "petscdmda.h" I*/
#include <petsc/private/dmimpl.h>
#include <petsc/private/snesimpl.h> /*I "petscsnes.h" I*/

/* This structure holds the user-provided DMDA callbacks */
typedef struct {
  /* array versions for vector data */
  DMDASNESFunctionFn  *residuallocal;
  DMDASNESJacobianFn  *jacobianlocal;
  DMDASNESObjectiveFn *objectivelocal;

  /* Vec version for vector data */
  DMDASNESFunctionVecFn  *residuallocalvec;
  DMDASNESJacobianVecFn  *jacobianlocalvec;
  DMDASNESObjectiveVecFn *objectivelocalvec;

  /* user contexts */
  void      *residuallocalctx;
  void      *jacobianlocalctx;
  void      *objectivelocalctx;
  InsertMode residuallocalimode;

  /*   For Picard iteration defined locally */
  PetscErrorCode (*rhsplocal)(DMDALocalInfo *, void *, void *, void *);
  PetscErrorCode (*jacobianplocal)(DMDALocalInfo *, void *, Mat, Mat, void *);
  void *picardlocalctx;
} DMSNES_DA;

static PetscErrorCode DMSNESDestroy_DMDA(DMSNES sdm)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(sdm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSNESDuplicate_DMDA(DMSNES oldsdm, DMSNES sdm)
{
  PetscFunctionBegin;
  PetscCall(PetscNew((DMSNES_DA **)&sdm->data));
  if (oldsdm->data) PetscCall(PetscMemcpy(sdm->data, oldsdm->data, sizeof(DMSNES_DA)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMDASNESGetContext(DM dm, DMSNES sdm, DMSNES_DA **dmdasnes)
{
  PetscFunctionBegin;
  *dmdasnes = NULL;
  if (!sdm->data) {
    PetscCall(PetscNew((DMSNES_DA **)&sdm->data));
    sdm->ops->destroy   = DMSNESDestroy_DMDA;
    sdm->ops->duplicate = DMSNESDuplicate_DMDA;
  }
  *dmdasnes = (DMSNES_DA *)sdm->data;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESComputeFunction_DMDA(SNES snes, Vec X, Vec F, void *ctx)
{
  DM            dm;
  DMSNES_DA    *dmdasnes = (DMSNES_DA *)ctx;
  DMDALocalInfo info;
  Vec           Xloc;
  void         *x, *f, *rctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  PetscCheck(dmdasnes->residuallocal || dmdasnes->residuallocalvec, PetscObjectComm((PetscObject)snes), PETSC_ERR_PLIB, "Corrupt context");
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetLocalVector(dm, &Xloc));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, Xloc));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, Xloc));
  PetscCall(DMDAGetLocalInfo(dm, &info));
  rctx = dmdasnes->residuallocalctx ? dmdasnes->residuallocalctx : snes->ctx;
  switch (dmdasnes->residuallocalimode) {
  case INSERT_VALUES: {
    PetscCall(PetscLogEventBegin(SNES_FunctionEval, snes, X, F, 0));
    if (dmdasnes->residuallocalvec) PetscCallBack("SNES DMDA local callback function", (*dmdasnes->residuallocalvec)(&info, Xloc, F, rctx));
    else {
      PetscCall(DMDAVecGetArray(dm, Xloc, &x));
      PetscCall(DMDAVecGetArray(dm, F, &f));
      PetscCallBack("SNES DMDA local callback function", (*dmdasnes->residuallocal)(&info, x, f, rctx));
      PetscCall(DMDAVecRestoreArray(dm, Xloc, &x));
      PetscCall(DMDAVecRestoreArray(dm, F, &f));
    }
    PetscCall(PetscLogEventEnd(SNES_FunctionEval, snes, X, F, 0));
  } break;
  case ADD_VALUES: {
    Vec Floc;
    PetscCall(DMGetLocalVector(dm, &Floc));
    PetscCall(VecZeroEntries(Floc));
    PetscCall(PetscLogEventBegin(SNES_FunctionEval, snes, X, F, 0));
    if (dmdasnes->residuallocalvec) PetscCallBack("SNES DMDA local callback function", (*dmdasnes->residuallocalvec)(&info, Xloc, Floc, rctx));
    else {
      PetscCall(DMDAVecGetArray(dm, Xloc, &x));
      PetscCall(DMDAVecGetArray(dm, Floc, &f));
      PetscCallBack("SNES DMDA local callback function", (*dmdasnes->residuallocal)(&info, x, f, rctx));
      PetscCall(DMDAVecRestoreArray(dm, Xloc, &x));
      PetscCall(DMDAVecRestoreArray(dm, Floc, &f));
    }
    PetscCall(PetscLogEventEnd(SNES_FunctionEval, snes, X, F, 0));
    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobalBegin(dm, Floc, ADD_VALUES, F));
    PetscCall(DMLocalToGlobalEnd(dm, Floc, ADD_VALUES, F));
    PetscCall(DMRestoreLocalVector(dm, &Floc));
  } break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_INCOMP, "Cannot use imode=%d", (int)dmdasnes->residuallocalimode);
  }
  PetscCall(DMRestoreLocalVector(dm, &Xloc));
  PetscCall(VecFlag(F, snes->domainerror));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESComputeObjective_DMDA(SNES snes, Vec X, PetscReal *ob, void *ctx)
{
  DM            dm;
  DMSNES_DA    *dmdasnes = (DMSNES_DA *)ctx;
  DMDALocalInfo info;
  Vec           Xloc;
  void         *x, *octx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscAssertPointer(ob, 3);
  PetscCheck(dmdasnes->objectivelocal || dmdasnes->objectivelocalvec, PetscObjectComm((PetscObject)snes), PETSC_ERR_PLIB, "Corrupt context");
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetLocalVector(dm, &Xloc));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, Xloc));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, Xloc));
  PetscCall(DMDAGetLocalInfo(dm, &info));
  octx = dmdasnes->objectivelocalctx ? dmdasnes->objectivelocalctx : snes->ctx;
  if (dmdasnes->objectivelocalvec) PetscCallBack("SNES DMDA local callback objective", (*dmdasnes->objectivelocalvec)(&info, Xloc, ob, octx));
  else {
    PetscCall(DMDAVecGetArray(dm, Xloc, &x));
    PetscCallBack("SNES DMDA local callback objective", (*dmdasnes->objectivelocal)(&info, x, ob, octx));
    PetscCall(DMDAVecRestoreArray(dm, Xloc, &x));
  }
  PetscCall(DMRestoreLocalVector(dm, &Xloc));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, ob, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)snes)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Routine is called by example, hence must be labeled PETSC_EXTERN */
PETSC_EXTERN PetscErrorCode SNESComputeJacobian_DMDA(SNES snes, Vec X, Mat A, Mat B, void *ctx)
{
  DM            dm;
  DMSNES_DA    *dmdasnes = (DMSNES_DA *)ctx;
  DMDALocalInfo info;
  Vec           Xloc;
  void         *x, *jctx;

  PetscFunctionBegin;
  PetscCheck(dmdasnes->residuallocal || dmdasnes->residuallocalvec, PetscObjectComm((PetscObject)snes), PETSC_ERR_PLIB, "Corrupt context");
  PetscCall(SNESGetDM(snes, &dm));
  jctx = dmdasnes->jacobianlocalctx ? dmdasnes->jacobianlocalctx : snes->ctx;
  if (dmdasnes->jacobianlocal || dmdasnes->jacobianlocalvec) {
    PetscCall(DMGetLocalVector(dm, &Xloc));
    PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, Xloc));
    PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, Xloc));
    PetscCall(DMDAGetLocalInfo(dm, &info));
    if (dmdasnes->jacobianlocalvec) PetscCallBack("SNES DMDA local callback Jacobian", (*dmdasnes->jacobianlocalvec)(&info, Xloc, A, B, jctx));
    else {
      PetscCall(DMDAVecGetArray(dm, Xloc, &x));
      PetscCallBack("SNES DMDA local callback Jacobian", (*dmdasnes->jacobianlocal)(&info, x, A, B, jctx));
      PetscCall(DMDAVecRestoreArray(dm, Xloc, &x));
    }
    PetscCall(DMRestoreLocalVector(dm, &Xloc));
  } else {
    MatFDColoring fdcoloring;
    PetscCall(PetscObjectQuery((PetscObject)dm, "DMDASNES_FDCOLORING", (PetscObject *)&fdcoloring));
    if (!fdcoloring) {
      ISColoring coloring;

      PetscCall(DMCreateColoring(dm, dm->coloringtype, &coloring));
      PetscCall(MatFDColoringCreate(B, coloring, &fdcoloring));
      switch (dm->coloringtype) {
      case IS_COLORING_GLOBAL:
        PetscCall(MatFDColoringSetFunction(fdcoloring, (MatFDColoringFn *)SNESComputeFunction_DMDA, dmdasnes));
        break;
      default:
        SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "No support for coloring type '%s'", ISColoringTypes[dm->coloringtype]);
      }
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)fdcoloring, ((PetscObject)dm)->prefix));
      PetscCall(MatFDColoringSetFromOptions(fdcoloring));
      PetscCall(MatFDColoringSetUp(B, coloring, fdcoloring));
      PetscCall(ISColoringDestroy(&coloring));
      PetscCall(PetscObjectCompose((PetscObject)dm, "DMDASNES_FDCOLORING", (PetscObject)fdcoloring));
      PetscCall(PetscObjectDereference((PetscObject)fdcoloring));

      /* The following breaks an ugly reference counting loop that deserves a paragraph. MatFDColoringApply() will call
       * VecDuplicate() with the state Vec and store inside the MatFDColoring. This Vec will duplicate the Vec, but the
       * MatFDColoring is composed with the DM. We dereference the DM here so that the reference count will eventually
       * drop to 0. Note the code in DMDestroy() that exits early for a negative reference count. That code path will be
       * taken when the PetscObjectList for the Vec inside MatFDColoring is destroyed.
       */
      PetscCall(PetscObjectDereference((PetscObject)dm));
    }
    PetscCall(MatFDColoringApply(B, fdcoloring, X, snes));
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDASNESSetFunctionLocal - set a local residual evaluation function for use with `DMDA`

  Logically Collective

  Input Parameters:
+ dm    - `DM` to associate callback with
. imode - `INSERT_VALUES` if local function computes owned part, `ADD_VALUES` if it contributes to ghosted part
. func  - local residual evaluation
- ctx   - optional context for local residual evaluation

  Calling sequence of `func`:
+ info - `DMDALocalInfo` defining the subdomain to evaluate the residual on
. x    - dimensional pointer to state at which to evaluate residual (e.g. PetscScalar *x or **x or ***x)
. f    - dimensional pointer to residual, write the residual here (e.g. PetscScalar *f or **f or ***f)
- ctx  - optional context passed above

  Level: beginner

.seealso: [](ch_snes), `DMDA`, `DMDASNESSetJacobianLocal()`, `DMSNESSetFunction()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetFunctionLocal(DM dm, InsertMode imode, PetscErrorCode (*func)(DMDALocalInfo *info, void *x, void *f, void *ctx), void *ctx)
{
  DMSNES     sdm;
  DMSNES_DA *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMSNESWrite(dm, &sdm));
  PetscCall(DMDASNESGetContext(dm, sdm, &dmdasnes));

  dmdasnes->residuallocalimode = imode;
  dmdasnes->residuallocal      = func;
  dmdasnes->residuallocalctx   = ctx;

  PetscCall(DMSNESSetFunction(dm, SNESComputeFunction_DMDA, dmdasnes));
  if (!sdm->ops->computejacobian) { /* Call us for the Jacobian too, can be overridden by the user. */
    PetscCall(DMSNESSetJacobian(dm, SNESComputeJacobian_DMDA, dmdasnes));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDASNESSetFunctionLocalVec - set a local residual evaluation function that operates on a local vector for `DMDA`

  Logically Collective

  Input Parameters:
+ dm    - `DM` to associate callback with
. imode - `INSERT_VALUES` if local function computes owned part, `ADD_VALUES` if it contributes to ghosted part
. func  - local residual evaluation
- ctx   - optional context for local residual evaluation

  Calling sequence of `func`:
+ info - `DMDALocalInfo` defining the subdomain to evaluate the residual on
. x    - state vector at which to evaluate residual
. f    - residual vector
- ctx  - optional context passed above

  Level: beginner

.seealso: [](ch_snes), `DMDA`, `DMDASNESSetFunctionLocal()`, `DMDASNESSetJacobianLocalVec()`, `DMSNESSetFunction()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetFunctionLocalVec(DM dm, InsertMode imode, PetscErrorCode (*func)(DMDALocalInfo *info, Vec x, Vec f, void *ctx), void *ctx)
{
  DMSNES     sdm;
  DMSNES_DA *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMSNESWrite(dm, &sdm));
  PetscCall(DMDASNESGetContext(dm, sdm, &dmdasnes));

  dmdasnes->residuallocalimode = imode;
  dmdasnes->residuallocalvec   = func;
  dmdasnes->residuallocalctx   = ctx;

  PetscCall(DMSNESSetFunction(dm, SNESComputeFunction_DMDA, dmdasnes));
  if (!sdm->ops->computejacobian) { /* Call us for the Jacobian too, can be overridden by the user. */
    PetscCall(DMSNESSetJacobian(dm, SNESComputeJacobian_DMDA, dmdasnes));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDASNESSetJacobianLocal - set a local Jacobian evaluation function for use with `DMDA`

  Logically Collective

  Input Parameters:
+ dm   - `DM` to associate callback with
. func - local Jacobian evaluation function
- ctx  - optional context for local Jacobian evaluation

  Calling sequence of `func`:
+ info - `DMDALocalInfo` defining the subdomain to evaluate the Jacobian at
. x    - dimensional pointer to state at which to evaluate Jacobian (e.g. PetscScalar *x or **x or ***x)
. J    - `Mat` object for the Jacobian
. M    - `Mat` object used to compute the preconditioner often `J`
- ctx  - optional context passed above

  Level: beginner

  Note:
  The `J` and `M` matrices are created internally by `DMCreateMatrix()`

.seealso: [](ch_snes), `DMDA`, `DMDASNESSetFunctionLocal()`, `DMSNESSetJacobian()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetJacobianLocal(DM dm, PetscErrorCode (*func)(DMDALocalInfo *info, void *x, Mat J, Mat M, void *ctx), void *ctx)
{
  DMSNES     sdm;
  DMSNES_DA *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMSNESWrite(dm, &sdm));
  PetscCall(DMDASNESGetContext(dm, sdm, &dmdasnes));

  dmdasnes->jacobianlocal    = func;
  dmdasnes->jacobianlocalctx = ctx;

  PetscCall(DMSNESSetJacobian(dm, SNESComputeJacobian_DMDA, dmdasnes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDASNESSetJacobianLocalVec - set a local Jacobian evaluation function that operates on a local vector with `DMDA`

  Logically Collective

  Input Parameters:
+ dm   - `DM` to associate callback with
. func - local Jacobian evaluation
- ctx  - optional context for local Jacobian evaluation

  Calling sequence of `func`:
+ info - `DMDALocalInfo` defining the subdomain to evaluate the Jacobian at
. x    - state vector at which to evaluate Jacobian
. J    - the Jacobian
. M    - approximate Jacobian from which the preconditioner will be computed, often `J`
- ctx  - optional context passed above

  Level: beginner

.seealso: [](ch_snes), `DMDA`, `DMDASNESSetJacobianLocal()`, `DMDASNESSetFunctionLocalVec()`, `DMSNESSetJacobian()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetJacobianLocalVec(DM dm, PetscErrorCode (*func)(DMDALocalInfo *info, Vec x, Mat J, Mat M, void *), void *ctx)
{
  DMSNES     sdm;
  DMSNES_DA *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMSNESWrite(dm, &sdm));
  PetscCall(DMDASNESGetContext(dm, sdm, &dmdasnes));

  dmdasnes->jacobianlocalvec = func;
  dmdasnes->jacobianlocalctx = ctx;

  PetscCall(DMSNESSetJacobian(dm, SNESComputeJacobian_DMDA, dmdasnes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDASNESSetObjectiveLocal - set a local residual evaluation function to used with a `DMDA`

  Logically Collective

  Input Parameters:
+ dm   - `DM` to associate callback with
. func - local objective evaluation, see `DMDASNESSetObjectiveLocal` for the calling sequence
- ctx  - optional context for local residual evaluation

  Calling sequence of `func`:
+ info - `DMDALocalInfo` defining the subdomain to evaluate the Jacobian at
. x    - dimensional pointer to state at which to evaluate the objective (e.g. PetscScalar *x or **x or ***x)
. obj  - returned objective value for the local subdomain
- ctx  - optional context passed above

  Level: beginner

.seealso: [](ch_snes), `DMDA`, `DMSNESSetFunction()`, `DMDASNESSetJacobianLocal()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMDASNESObjectiveFn`
@*/
PetscErrorCode DMDASNESSetObjectiveLocal(DM dm, PetscErrorCode (*func)(DMDALocalInfo *info, void *x, PetscReal *obj, void *), void *ctx)
{
  DMSNES     sdm;
  DMSNES_DA *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMSNESWrite(dm, &sdm));
  PetscCall(DMDASNESGetContext(dm, sdm, &dmdasnes));

  dmdasnes->objectivelocal    = func;
  dmdasnes->objectivelocalctx = ctx;

  PetscCall(DMSNESSetObjective(dm, SNESComputeObjective_DMDA, dmdasnes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDASNESSetObjectiveLocalVec - set a local residual evaluation function that operates on a local vector with `DMDA`

  Logically Collective

  Input Parameters:
+ dm   - `DM` to associate callback with
. func - local objective evaluation, see `DMDASNESSetObjectiveLocalVec` for the calling sequence
- ctx  - optional context for local residual evaluation

  Calling sequence of `func`:
+ info - `DMDALocalInfo` defining the subdomain to evaluate the Jacobian at
. x    - state vector at which to evaluate the objective
. obj  - returned objective value for the local subdomain
- ctx  - optional context passed above

  Level: beginner

.seealso: [](ch_snes), `DMDA`, `DMDASNESSetObjectiveLocal()`, `DMSNESSetFunction()`, `DMDASNESSetJacobianLocalVec()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMDASNESObjectiveVecFn`
@*/
PetscErrorCode DMDASNESSetObjectiveLocalVec(DM dm, PetscErrorCode (*func)(DMDALocalInfo *info, Vec x, PetscReal *obj, void *), void *ctx)
{
  DMSNES     sdm;
  DMSNES_DA *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMSNESWrite(dm, &sdm));
  PetscCall(DMDASNESGetContext(dm, sdm, &dmdasnes));

  dmdasnes->objectivelocalvec = func;
  dmdasnes->objectivelocalctx = ctx;

  PetscCall(DMSNESSetObjective(dm, SNESComputeObjective_DMDA, dmdasnes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESComputePicard_DMDA(SNES snes, Vec X, Vec F, void *ctx)
{
  DM            dm;
  DMSNES_DA    *dmdasnes = (DMSNES_DA *)ctx;
  DMDALocalInfo info;
  Vec           Xloc;
  void         *x, *f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  PetscCheck(dmdasnes->rhsplocal, PetscObjectComm((PetscObject)snes), PETSC_ERR_PLIB, "Corrupt context");
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetLocalVector(dm, &Xloc));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, Xloc));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, Xloc));
  PetscCall(DMDAGetLocalInfo(dm, &info));
  PetscCall(DMDAVecGetArray(dm, Xloc, &x));
  switch (dmdasnes->residuallocalimode) {
  case INSERT_VALUES: {
    PetscCall(DMDAVecGetArray(dm, F, &f));
    PetscCallBack("SNES Picard DMDA local callback function", (*dmdasnes->rhsplocal)(&info, x, f, dmdasnes->picardlocalctx));
    PetscCall(DMDAVecRestoreArray(dm, F, &f));
  } break;
  case ADD_VALUES: {
    Vec Floc;
    PetscCall(DMGetLocalVector(dm, &Floc));
    PetscCall(VecZeroEntries(Floc));
    PetscCall(DMDAVecGetArray(dm, Floc, &f));
    PetscCallBack("SNES Picard DMDA local callback function", (*dmdasnes->rhsplocal)(&info, x, f, dmdasnes->picardlocalctx));
    PetscCall(DMDAVecRestoreArray(dm, Floc, &f));
    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobalBegin(dm, Floc, ADD_VALUES, F));
    PetscCall(DMLocalToGlobalEnd(dm, Floc, ADD_VALUES, F));
    PetscCall(DMRestoreLocalVector(dm, &Floc));
  } break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_INCOMP, "Cannot use imode=%d", (int)dmdasnes->residuallocalimode);
  }
  PetscCall(DMDAVecRestoreArray(dm, Xloc, &x));
  PetscCall(DMRestoreLocalVector(dm, &Xloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESComputePicardJacobian_DMDA(SNES snes, Vec X, Mat A, Mat B, void *ctx)
{
  DM            dm;
  DMSNES_DA    *dmdasnes = (DMSNES_DA *)ctx;
  DMDALocalInfo info;
  Vec           Xloc;
  void         *x;

  PetscFunctionBegin;
  PetscCheck(dmdasnes->jacobianplocal, PetscObjectComm((PetscObject)snes), PETSC_ERR_PLIB, "Corrupt context");
  PetscCall(SNESGetDM(snes, &dm));

  PetscCall(DMGetLocalVector(dm, &Xloc));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, Xloc));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, Xloc));
  PetscCall(DMDAGetLocalInfo(dm, &info));
  PetscCall(DMDAVecGetArray(dm, Xloc, &x));
  PetscCallBack("SNES Picard DMDA local callback Jacobian", (*dmdasnes->jacobianplocal)(&info, x, A, B, dmdasnes->picardlocalctx));
  PetscCall(DMDAVecRestoreArray(dm, Xloc, &x));
  PetscCall(DMRestoreLocalVector(dm, &Xloc));
  if (A != B) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMDASNESSetPicardLocal - set a local right-hand side and matrix evaluation function for Picard iteration with `DMDA`

  Logically Collective

  Input Parameters:
+ dm    - `DM` to associate callback with
. imode - `INSERT_VALUES` if local function computes owned part, `ADD_VALUES` if it contributes to ghosted part
. func  - local residual evaluation
. jac   - function to compute Jacobian
- ctx   - optional context for local residual evaluation

  Calling sequence of `func`:
+ info - defines the subdomain to evaluate the residual on
. x    - dimensional pointer to state at which to evaluate residual
. f    - dimensional pointer to residual, write the residual here
- ctx  - optional context passed above

  Calling sequence of `jac`:
+ info - defines the subdomain to evaluate the residual on
. x    - dimensional pointer to state at which to evaluate residual
. jac  - the Jacobian
. Jp   - approximation to the Jacobian used to compute the preconditioner, often `J`
- ctx  - optional context passed above

  Level: beginner

  Note:
  The user must use `SNESSetFunction`(`snes`,`NULL`,`SNESPicardComputeFunction`,&user));
  in their code before calling this routine.

.seealso: [](ch_snes), `SNES`, `DMDA`, `DMSNESSetFunction()`, `DMDASNESSetJacobian()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDASNESSetPicardLocal(DM dm, InsertMode imode, PetscErrorCode (*func)(DMDALocalInfo *info, void *x, void *f, void *ctx), PetscErrorCode (*jac)(DMDALocalInfo *info, void *x, Mat jac, Mat Jp, void *ctx), void *ctx)
{
  DMSNES     sdm;
  DMSNES_DA *dmdasnes;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMSNESWrite(dm, &sdm));
  PetscCall(DMDASNESGetContext(dm, sdm, &dmdasnes));

  dmdasnes->residuallocalimode = imode;
  dmdasnes->rhsplocal          = func;
  dmdasnes->jacobianplocal     = jac;
  dmdasnes->picardlocalctx     = ctx;

  PetscCall(DMSNESSetPicard(dm, SNESComputePicard_DMDA, SNESComputePicardJacobian_DMDA, dmdasnes));
  PetscCall(DMSNESSetMFFunction(dm, SNESComputeFunction_DMDA, dmdasnes));
  PetscFunctionReturn(PETSC_SUCCESS);
}
