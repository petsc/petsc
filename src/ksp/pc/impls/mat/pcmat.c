#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

typedef enum {
  PCMATOP_UNSPECIFIED,
  PCMATOP_MULT,
  PCMATOP_MULT_TRANSPOSE,
  PCMATOP_MULT_HERMITIAN_TRANSPOSE,
  PCMATOP_SOLVE,
  PCMATOP_SOLVE_TRANSPOSE,
} PCMatOperation;

const char *const PCMatOpTypes[] = {"Unspecified", "Mult", "MultTranspose", "MultHermitianTranspose", "Solve", "SolveTranspose", NULL};

typedef struct _PCMAT {
  PCMatOperation apply;
} PC_Mat;

static PetscErrorCode PCApply_Mat(PC pc, Vec x, Vec y)
{
  PC_Mat *pcmat = (PC_Mat *)pc->data;

  PetscFunctionBegin;
  switch (pcmat->apply) {
  case PCMATOP_MULT:
    PetscCall(MatMult(pc->pmat, x, y));
    break;
  case PCMATOP_MULT_TRANSPOSE:
    PetscCall(MatMultTranspose(pc->pmat, x, y));
    break;
  case PCMATOP_SOLVE:
    PetscCall(MatSolve(pc->pmat, x, y));
    break;
  case PCMATOP_SOLVE_TRANSPOSE:
    PetscCall(MatSolveTranspose(pc->pmat, x, y));
    break;
  case PCMATOP_MULT_HERMITIAN_TRANSPOSE:
    PetscCall(MatMultHermitianTranspose(pc->pmat, x, y));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_INCOMP, "Unsupported %s case", PCMatOpTypes[pcmat->apply]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_Mat(PC pc)
{
  PC_Mat *pcmat = (PC_Mat *)pc->data;

  PetscFunctionBegin;
  if (pcmat->apply == PCMATOP_UNSPECIFIED) {
    PetscBool hassolve;
    PetscCall(MatHasOperation(pc->pmat, MATOP_SOLVE, &hassolve));
    if (hassolve) pcmat->apply = PCMATOP_SOLVE;
    else pcmat->apply = PCMATOP_MULT;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatApply_Mat(PC pc, Mat X, Mat Y)
{
  PC_Mat *pcmat = (PC_Mat *)pc->data;
  Mat     W;

  PetscFunctionBegin;
  switch (pcmat->apply) {
  case PCMATOP_MULT:
    PetscCall(MatMatMult(pc->pmat, X, MAT_REUSE_MATRIX, PETSC_CURRENT, &Y));
    break;
  case PCMATOP_MULT_TRANSPOSE:
    PetscCall(MatTransposeMatMult(pc->pmat, X, MAT_REUSE_MATRIX, PETSC_CURRENT, &Y));
    break;
  case PCMATOP_SOLVE:
    PetscCall(MatMatSolve(pc->pmat, X, Y));
    break;
  case PCMATOP_SOLVE_TRANSPOSE:
    PetscCall(MatMatSolveTranspose(pc->pmat, X, Y));
    break;
  case PCMATOP_MULT_HERMITIAN_TRANSPOSE:
    PetscCall(MatDuplicate(X, MAT_COPY_VALUES, &W));
    PetscCall(MatConjugate(W));
    PetscCall(MatTransposeMatMult(pc->pmat, W, MAT_REUSE_MATRIX, PETSC_CURRENT, &Y));
    PetscCall(MatConjugate(Y));
    PetscCall(MatDestroy(&W));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_INCOMP, "Unsupported %s case", PCMatOpTypes[pcmat->apply]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_Mat(PC pc, Vec x, Vec y)
{
  PC_Mat *pcmat = (PC_Mat *)pc->data;
  Vec     w;

  PetscFunctionBegin;
  switch (pcmat->apply) {
  case PCMATOP_MULT:
    PetscCall(MatMultTranspose(pc->pmat, x, y));
    break;
  case PCMATOP_MULT_TRANSPOSE:
    PetscCall(MatMult(pc->pmat, x, y));
    break;
  case PCMATOP_SOLVE:
    PetscCall(MatSolveTranspose(pc->pmat, x, y));
    break;
  case PCMATOP_SOLVE_TRANSPOSE:
    PetscCall(MatSolve(pc->pmat, x, y));
    break;
  case PCMATOP_MULT_HERMITIAN_TRANSPOSE:
    PetscCall(VecDuplicate(x, &w));
    PetscCall(VecCopy(x, w));
    PetscCall(VecConjugate(w));
    PetscCall(MatMult(pc->pmat, w, y));
    PetscCall(VecConjugate(y));
    PetscCall(VecDestroy(&w));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_INCOMP, "Unsupported %s case", PCMatOpTypes[pcmat->apply]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Mat(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatSetApplyOperation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatGetApplyOperation_C", NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCMatSetApplyOperation - Set which matrix operation of the matrix implements `PCApply()` for `PCMAT`.

  Logically collective

  Input Parameters:
+ pc    - An instance of `PCMAT`
- matop - The selected `MatOperation`

  Level: intermediate

  Note:
  If you have a matrix type that implements an exact inverse that isn't a factorization,
  you can use `PCMatSetApplyOperation(pc, MATOP_SOLVE)`.

.seealso: [](ch_ksp), `PCMAT`, `PCMatGetApplyOperation()`, `PCApply()`, `MatOperation`
@*/
PetscErrorCode PCMatSetApplyOperation(PC pc, MatOperation matop)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCMatSetApplyOperation_C", (PC, MatOperation), (pc, matop));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCMatGetApplyOperation - Get which matrix operation of the matrix implements `PCApply()` for `PCMAT`.

  Logically collective

  Input Parameter:
. pc - An instance of `PCMAT`

  Output Parameter:
. matop - The `MatOperation`

  Level: intermediate

.seealso: [](ch_ksp), `PCMAT`, `PCMatSetApplyOperation()`, `PCApply()`, `MatOperation`
@*/
PetscErrorCode PCMatGetApplyOperation(PC pc, MatOperation *matop)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscAssertPointer(matop, 2);
  PetscUseMethod(pc, "PCMatGetApplyOperation_C", (PC, MatOperation *), (pc, matop));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatSetApplyOperation_Mat(PC pc, MatOperation matop)
{
  PC_Mat        *pcmat = (PC_Mat *)pc->data;
  PCMatOperation pcmatop;

  PetscFunctionBegin;
  // clang-format off
#define MATOP_TO_PCMATOP_CASE(var, OP) case MATOP_##OP: (var) = PCMATOP_##OP; break
  switch (matop) {
  MATOP_TO_PCMATOP_CASE(pcmatop, MULT);
  MATOP_TO_PCMATOP_CASE(pcmatop, MULT_TRANSPOSE);
  MATOP_TO_PCMATOP_CASE(pcmatop, MULT_HERMITIAN_TRANSPOSE);
  MATOP_TO_PCMATOP_CASE(pcmatop, SOLVE);
  MATOP_TO_PCMATOP_CASE(pcmatop, SOLVE_TRANSPOSE);
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_INCOMP, "Unsupported MatOperation %d for PCMatSetApplyOperation()", (int)matop);
  }
#undef MATOP_TO_PCMATOP_CASE
  // clang-format on

  pcmat->apply = pcmatop;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatGetApplyOperation_Mat(PC pc, MatOperation *matop_p)
{
  PC_Mat      *pcmat = (PC_Mat *)pc->data;
  MatOperation matop = MATOP_MULT;

  PetscFunctionBegin;
  if (!pc->setupcalled) PetscCall(PCSetUp(pc));

  // clang-format off
#define PCMATOP_TO_MATOP_CASE(var, OP) case PCMATOP_##OP: (var) = MATOP_##OP; break
  switch (pcmat->apply) {
  PCMATOP_TO_MATOP_CASE(matop, MULT);
  PCMATOP_TO_MATOP_CASE(matop, MULT_TRANSPOSE);
  PCMATOP_TO_MATOP_CASE(matop, MULT_HERMITIAN_TRANSPOSE);
  PCMATOP_TO_MATOP_CASE(matop, SOLVE);
  PCMATOP_TO_MATOP_CASE(matop, SOLVE_TRANSPOSE);
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_INCOMP, "Unsupported %s case", PCMatOpTypes[pcmat->apply]);
  }
#undef PCMATOP_TO_MATOP_CASE
  // clang-format on

  *matop_p = matop;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_Mat(PC pc, PetscViewer viewer)
{
  PC_Mat   *pcmat = (PC_Mat *)pc->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscViewerASCIIPrintf(viewer, "PCApply() == Mat%s()\n", PCMatOpTypes[pcmat->apply]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCMAT - A preconditioner obtained by applying an operation of the `pmat` provided in
             in `PCSetOperators(pc,amat,pmat)` or `KSPSetOperators(ksp,amat,pmat)`.  By default, the operation is `MATOP_MULT`,
             meaning that the `pmat` provides an approximate inverse of `amat`.
             If some other operation of `pmat` implements the approximate inverse,
             use `PCMatSetApplyOperation()` to select that operation.

   Level: intermediate

.seealso: [](ch_ksp), `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCSHELL`, `MatOperation`, `PCMatSetApplyOperation()`, `PCMatGetApplyOperation()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Mat(PC pc)
{
  PC_Mat *data;

  PetscFunctionBegin;
  PetscCall(PetscNew(&data));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatSetApplyOperation_C", PCMatSetApplyOperation_Mat));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCMatGetApplyOperation_C", PCMatGetApplyOperation_Mat));
  pc->data                     = data;
  pc->ops->apply               = PCApply_Mat;
  pc->ops->matapply            = PCMatApply_Mat;
  pc->ops->applytranspose      = PCApplyTranspose_Mat;
  pc->ops->setup               = PCSetUp_Mat;
  pc->ops->destroy             = PCDestroy_Mat;
  pc->ops->setfromoptions      = NULL;
  pc->ops->view                = PCView_Mat;
  pc->ops->applyrichardson     = NULL;
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
