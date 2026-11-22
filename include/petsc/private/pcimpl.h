#pragma once

#include <petscksp.h>
#include <petscpc.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      PCRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PCRegisterAll(void);

typedef struct _PCOps *PCOps;
struct _PCOps {
  PetscErrorCode (*setup)(PC);
  PetscErrorCode (*apply)(PC, Vec, Vec);
  PetscErrorCode (*matapply)(PC, Mat, Mat);
  PetscErrorCode (*applyrichardson)(PC, Vec, Vec, Vec, PetscReal, PetscReal, PetscReal, PetscInt, PetscBool, PetscInt *, PCRichardsonConvergedReason *);
  PetscErrorCode (*matapplyrichardson)(PC, Mat, Mat, Mat, PetscReal, PetscReal, PetscReal, PetscInt, PetscBool, PetscInt *, PCRichardsonConvergedReason *);
  PetscErrorCode (*applyBA)(PC, PCSide, Vec, Vec, Vec);
  PetscErrorCode (*applytranspose)(PC, Vec, Vec);
  PetscErrorCode (*matapplytranspose)(PC, Mat, Mat);
  PetscErrorCode (*applyBAtranspose)(PC, PetscInt, Vec, Vec, Vec);
  PetscErrorCode (*setfromoptions)(PC, PetscOptionItems);
  PetscErrorCode (*presolve)(PC, KSP, Vec, Vec);
  PetscErrorCode (*postsolve)(PC, KSP, Vec, Vec);
  PetscErrorCode (*getfactoredmatrix)(PC, Mat *);
  PetscErrorCode (*applysymmetricleft)(PC, Vec, Vec);
  PetscErrorCode (*applysymmetricright)(PC, Vec, Vec);
  PetscErrorCode (*setuponblocks)(PC);
  PetscErrorCode (*destroy)(PC);
  PetscErrorCode (*view)(PC, PetscViewer);
  PetscErrorCode (*reset)(PC);
  PetscErrorCode (*load)(PC, PetscViewer);
};

/*
   Preconditioner context
*/
struct _p_PC {
  PETSCHEADER(struct _PCOps);
  DM           dm;
  PetscBool    setupcalled;
  MatState     matstate; /* last known state of the pmat associated with this PC */
  PetscBool    reusepreconditioner;
  MatStructure flag; /* reset each PCSetUp() to indicate to PC implementations if nonzero structure has changed */

  PetscInt  setfromoptionscalled;
  PetscBool erroriffailure; /* generate an error if FPE detected (for example a zero pivot) instead of returning */
  Mat       mat, pmat;
  Mat      *matwork;  /* owned work matrices for block applications */
  PetscInt  nmatwork; /* number of allocated Mat slots */

  PetscObjectParameterDeclare(PetscBool, useAmat); /* used by several PC that including applying the operator inside the preconditioner */

  PetscErrorCode (*modifysubmatrices)(PC, PetscInt, const IS[], const IS[], Mat[], void *); /* user provided routine */
  void          *modifysubmatricesP;                                                        /* context for user routine */
  void          *data;
  void          *ctx;              /* optional user-defined context */
  PCFailedReason failedreason;     /* after VecNorm or VecDot contains maximum of all rank failed reasons */
  PCFailedReason failedreasonrank; /* failed reason on this rank */
  PetscInt       presolvedone;
  PetscErrorCode (*postsetup)(PC);
  PetscInt kspnestlevel; /* how many levels of nesting does the KSP have that contains the PC */
};

PETSC_EXTERN PetscLogEvent PC_SetUp;
PETSC_EXTERN PetscLogEvent PC_SetUpOnBlocks;
PETSC_EXTERN PetscLogEvent PC_Apply;
PETSC_EXTERN PetscLogEvent PC_MatApply;
PETSC_EXTERN PetscLogEvent PC_ApplyCoarse;
PETSC_EXTERN PetscLogEvent PC_ApplySymmetricLeft;
PETSC_EXTERN PetscLogEvent PC_ApplySymmetricRight;
PETSC_EXTERN PetscLogEvent PC_ModifySubMatrices;
PETSC_EXTERN PetscLogEvent PC_ApplyOnBlocks;
PETSC_EXTERN PetscLogEvent PC_ApplyTransposeOnBlocks;
PETSC_EXTERN PetscLogStage PCMPIStage;

PETSC_INTERN PetscErrorCode PCMatWorkCreate(Vec, PetscInt, PetscInt, Mat *);

// SLEPc builds PCHPDDM in a separate library and needs the three following workspace operations inline

/*
  PCMatWorkClear - Releases the work matrices and sets their slots to `NULL` while retaining the slot array.

  Collective on `pc`

  Input Parameter:
. pc - preconditioner that owns the workspace

  Level: developer

.seealso: `PCMatWorkSetUp()`, `PCMatWorkReset()`
*/
static inline PetscErrorCode PCMatWorkClear(PC pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  for (PetscInt i = 0; i < pc->nmatwork; ++i) PetscCall(MatDestroy(&pc->matwork[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PCMatWorkReset - Releases the work matrices, frees the slot array, and resets its capacity to zero.

  Collective on `pc`

  Input Parameter:
. pc - preconditioner that owns the workspace

  Level: developer

.seealso: `PCMatWorkClear()`, `PCReset()`
*/
static inline PetscErrorCode PCMatWorkReset(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCMatWorkClear(pc));
  PetscCall(PetscFree(pc->matwork));
  pc->nmatwork = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PCMatWorkSetUp - Reserves at least `n` work matrix slots in a preconditioner.

  Collective on `pc`

  Input Parameters:
+ pc   - preconditioner that owns the workspace
. n    - minimum number of slots
- cols - expected global column count of the first matrix, or `PETSC_DECIDE` to skip the check

  Level: developer

  Notes:
  If the first slot contains a matrix with a different column count, the routine clears all slots.
  Increasing the capacity can move the array. New slots are initialized to `NULL`.

.seealso: `PCMatWorkCreate()`, `PCMatWorkClear()`, `PCMatWorkReset()`
*/
static inline PetscErrorCode PCMatWorkSetUp(PC pc, PetscInt n, PetscInt cols)
{
  PetscInt oldcols = cols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(pc, n, 2);
  PetscValidLogicalCollectiveInt(pc, cols, 3);
  PetscCheck(n >= 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_OUTOFRANGE, "n must be non-negative, got %" PetscInt_FMT, n);
  if (pc->matwork != NULL && cols >= 0 && pc->matwork[0] != NULL) PetscCall(MatGetSize(pc->matwork[0], NULL, &oldcols));
  if (cols != oldcols) PetscCall(PCMatWorkClear(pc));
  if (n > pc->nmatwork) {
    PetscCall(PetscRealloc(n * sizeof(*pc->matwork), &pc->matwork));
    PetscCall(PetscArrayzero(pc->matwork + pc->nmatwork, n - pc->nmatwork));
    pc->nmatwork = n;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
