/*
     This file implements a Jacobi preconditioner in PETSc as part of PC.
     You can use this as a starting point for implementing your own
     preconditioner that is not provided with PETSc. (You might also consider
     just using PCSHELL)

     The following basic routines are required for each preconditioner.
          PCCreate_XXX()          - Creates a preconditioner context
          PCSetFromOptions_XXX()  - Sets runtime options
          PCApply_XXX()           - Applies the preconditioner
          PCDestroy_XXX()         - Destroys the preconditioner context
     where the suffix "_XXX" denotes a particular implementation, in
     this case we use _Jacobi (e.g., PCCreate_Jacobi, PCApply_Jacobi).
     These routines are actually called via the common user interface
     routines PCCreate(), PCSetFromOptions(), PCApply(), and PCDestroy(),
     so the application code interface remains identical for all
     preconditioners.

     Another key routine is:
          PCSetUp_XXX()           - Prepares for the use of a preconditioner
     by setting data structures and options.   The interface routine PCSetUp()
     is not usually called directly by the user, but instead is called by
     PCApply() if necessary.

     Additional basic routines are:
          PCView_XXX()            - Prints details of runtime options that
                                    have actually been used.
     These are called by application codes via the interface routines
     PCView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  One exception is
     that the analogues of PCApply() for these components are KSPSolve(),
     SNESSolve(), and TSSolve().

     Additional optional functionality unique to preconditioners is left and
     right symmetric preconditioner application via PCApplySymmetricLeft()
     and PCApplySymmetricRight().  The Jacobi implementation is
     PCApplySymmetricLeftOrRight_Jacobi().
*/

/*
   Include files needed for the Jacobi preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

const char *const PCJacobiTypes[] = {"DIAGONAL", "ROWL1", "ROWMAX", "ROWSUM", "PCJacobiType", "PC_JACOBI_", NULL};

/*
   Private context (data structure) for the Jacobi preconditioner.
*/
typedef struct {
  Vec          diag;     /* vector containing the reciprocals of the diagonal elements of the matrix used to construct the preconditioner */
  Vec          diagsqrt; /* vector containing the reciprocals of the square roots of
                                    the diagonal elements of the matrix used to compute the preconditioner (used
                                    only for symmetric preconditioner application) */
  PCJacobiType type;
  PetscBool    useabs;  /* use the absolute values of the diagonal entries */
  PetscBool    fixdiag; /* fix zero diagonal terms */
  PetscReal    scale;   /* for scaling rowl1 off-diagonals */
} PC_Jacobi;

static PetscErrorCode PCReset_Jacobi(PC);

static PetscErrorCode PCJacobiSetType_Jacobi(PC pc, PCJacobiType type)
{
  PC_Jacobi   *j = (PC_Jacobi *)pc->data;
  PCJacobiType old_type;

  PetscFunctionBegin;
  PetscCall(PCJacobiGetType(pc, &old_type));
  if (old_type == type) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PCReset_Jacobi(pc));
  j->type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCJacobiGetUseAbs_Jacobi(PC pc, PetscBool *flg)
{
  PC_Jacobi *j = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  *flg = j->useabs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCJacobiSetUseAbs_Jacobi(PC pc, PetscBool flg)
{
  PC_Jacobi *j = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  j->useabs = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCJacobiGetType_Jacobi(PC pc, PCJacobiType *type)
{
  PC_Jacobi *j = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  *type = j->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCJacobiSetRowl1Scale_Jacobi(PC pc, PetscReal flg)
{
  PC_Jacobi *j = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  j->scale = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCJacobiGetRowl1Scale_Jacobi(PC pc, PetscReal *flg)
{
  PC_Jacobi *j = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  *flg = j->scale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCJacobiSetFixDiagonal_Jacobi(PC pc, PetscBool flg)
{
  PC_Jacobi *j = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  j->fixdiag = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCJacobiGetFixDiagonal_Jacobi(PC pc, PetscBool *flg)
{
  PC_Jacobi *j = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  *flg = j->fixdiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCJacobiGetDiagonal_Jacobi(PC pc, Vec diag, Vec diagsqrt)
{
  PC_Jacobi *j    = (PC_Jacobi *)pc->data;
  MPI_Comm   comm = PetscObjectComm((PetscObject)pc);

  PetscFunctionBegin;
  PetscCheck(j->diag || j->diagsqrt, comm, PETSC_ERR_ARG_WRONGSTATE, "Jacobi diagonal has not been created yet. Use PCApply to force creation");
  PetscCheck(!diag || (diag && j->diag), comm, PETSC_ERR_ARG_WRONGSTATE, "Jacobi diagonal not available. Check if PC is non-symmetric");
  PetscCheck(!diagsqrt || (diagsqrt && j->diagsqrt), comm, PETSC_ERR_ARG_WRONGSTATE, "Jacobi diagonal squareroot not available. Check if PC is symmetric");

  if (diag) PetscCall(VecCopy(j->diag, diag));
  if (diagsqrt) PetscCall(VecCopy(j->diagsqrt, diagsqrt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCSetUp_Jacobi - Prepares for the use of the Jacobi preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Note:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
static PetscErrorCode PCSetUp_Jacobi(PC pc)
{
  PC_Jacobi *jac = (PC_Jacobi *)pc->data;
  Vec        diag, diagsqrt;
  PetscInt   n, i;
  PetscBool  zeroflag = PETSC_FALSE, negflag = PETSC_FALSE;

  PetscFunctionBegin;
  /*
       For most preconditioners the code would begin here something like

  if (!pc->setupcalled) { allocate space the first time this is ever called
    PetscCall(MatCreateVecs(pc->mat,&jac->diag));
  }

    But for this preconditioner we want to support use of both the matrix' diagonal
    elements (for left or right preconditioning) and square root of diagonal elements
    (for symmetric preconditioning).  Hence we do not allocate space here, since we
    don't know at this point which will be needed (diag and/or diagsqrt) until the user
    applies the preconditioner, and we don't want to allocate BOTH unless we need
    them both.  Thus, the diag and diagsqrt are allocated in PCSetUp_Jacobi_NonSymmetric()
    and PCSetUp_Jacobi_Symmetric(), respectively.
  */

  /*
    Here we set up the preconditioner; that is, we copy the diagonal values from
    the matrix and put them into a format to make them quick to apply as a preconditioner.
  */
  diag     = jac->diag;
  diagsqrt = jac->diagsqrt;

  if (diag) {
    PetscBool isset, isspd;

    PetscCall(VecLockReadPop(diag));
    switch (jac->type) {
    case PC_JACOBI_DIAGONAL:
      PetscCall(MatGetDiagonal(pc->pmat, diag));
      break;
    case PC_JACOBI_ROWMAX:
      PetscCall(MatGetRowMaxAbs(pc->pmat, diag, NULL));
      break;
    case PC_JACOBI_ROWL1:
      PetscCall(MatGetRowSumAbs(pc->pmat, diag));
      // fix negative rows (eg, negative definite) -- this could be done for all, not needed for userowmax
      PetscCall(MatIsSPDKnown(pc->pmat, &isset, &isspd));
      if (jac->fixdiag && (!isset || !isspd)) {
        PetscScalar       *x2;
        const PetscScalar *x;
        Vec                true_diag;
        PetscCall(VecDuplicate(diag, &true_diag));
        PetscCall(MatGetDiagonal(pc->pmat, true_diag));
        PetscCall(VecGetLocalSize(diag, &n));
        PetscCall(VecGetArrayWrite(diag, &x2));
        PetscCall(VecGetArrayRead(true_diag, &x)); // to make more general -todo
        for (i = 0; i < n; i++) {
          if (PetscRealPart(x[i]) < 0.0) {
            x2[i]   = -x2[i]; // flip sign to keep DA > 0
            negflag = PETSC_TRUE;
          }
        }
        PetscCall(VecRestoreArrayRead(true_diag, &x));
        PetscCall(VecRestoreArrayWrite(diag, &x2));
        PetscCheck(!jac->useabs || !negflag, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_INCOMP, "Jacobi use_abs and l1 not compatible with negative diagonal");
        PetscCall(VecDestroy(&true_diag));
      }
      if (jac->scale != 1.0) {
        Vec true_diag;
        PetscCall(VecDuplicate(diag, &true_diag));
        PetscCall(MatGetDiagonal(pc->pmat, true_diag));
        PetscCall(VecAXPY(diag, -1, true_diag)); // subtract off diag
        PetscCall(VecScale(diag, jac->scale));   // scale off-diag
        PetscCall(VecAXPY(diag, 1, true_diag));  // add diag back in
        PetscCall(VecDestroy(&true_diag));
      }
      break;
    case PC_JACOBI_ROWSUM:
      PetscCall(MatGetRowSum(pc->pmat, diag));
      break;
    }
    PetscCall(VecReciprocal(diag));
    if (jac->useabs) PetscCall(VecAbs(diag));
    PetscCall(MatIsSPDKnown(pc->pmat, &isset, &isspd));
    if (jac->fixdiag && (!isset || !isspd)) {
      PetscScalar *x;
      PetscCall(VecGetLocalSize(diag, &n));
      PetscCall(VecGetArray(diag, &x));
      for (i = 0; i < n; i++) {
        if (x[i] == 0.0) {
          x[i]     = 1.0;
          zeroflag = PETSC_TRUE;
        }
      }
      PetscCall(VecRestoreArray(diag, &x));
    }
    PetscCall(VecLockReadPush(diag));
  }
  if (diagsqrt) {
    PetscScalar *x;

    PetscCall(VecLockReadPop(diagsqrt));
    switch (jac->type) {
    case PC_JACOBI_DIAGONAL:
      PetscCall(MatGetDiagonal(pc->pmat, diagsqrt));
      break;
    case PC_JACOBI_ROWMAX:
      PetscCall(MatGetRowMaxAbs(pc->pmat, diagsqrt, NULL));
      break;
    case PC_JACOBI_ROWL1:
      PetscCall(MatGetRowSumAbs(pc->pmat, diagsqrt));
      break;
    case PC_JACOBI_ROWSUM:
      PetscCall(MatGetRowSum(pc->pmat, diagsqrt));
      break;
    }
    PetscCall(VecGetLocalSize(diagsqrt, &n));
    PetscCall(VecGetArray(diagsqrt, &x));
    for (i = 0; i < n; i++) {
      if (PetscRealPart(x[i]) < 0.0) x[i] = 1.0 / PetscSqrtReal(PetscAbsScalar(-x[i]));
      else if (PetscRealPart(x[i]) > 0.0) x[i] = 1.0 / PetscSqrtReal(PetscAbsScalar(x[i]));
      else {
        x[i]     = 1.0;
        zeroflag = PETSC_TRUE;
      }
    }
    PetscCall(VecRestoreArray(diagsqrt, &x));
    PetscCall(VecLockReadPush(diagsqrt));
  }
  if (zeroflag) PetscCall(PetscInfo(pc, "Zero detected in diagonal of matrix, using 1 at those locations\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCSetUp_Jacobi_Symmetric - Allocates the vector needed to store the
   inverse of the square root of the diagonal entries of the matrix.  This
   is used for symmetric application of the Jacobi preconditioner.

   Input Parameter:
.  pc - the preconditioner context
*/
static PetscErrorCode PCSetUp_Jacobi_Symmetric(PC pc)
{
  PC_Jacobi *jac = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(pc->pmat, &jac->diagsqrt, NULL));
  PetscCall(VecLockReadPush(jac->diagsqrt));
  PetscCall(PCSetUp_Jacobi(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCSetUp_Jacobi_NonSymmetric - Allocates the vector needed to store the
   inverse of the diagonal entries of the matrix.  This is used for left of
   right application of the Jacobi preconditioner.

   Input Parameter:
.  pc - the preconditioner context
*/
static PetscErrorCode PCSetUp_Jacobi_NonSymmetric(PC pc)
{
  PC_Jacobi *jac = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(pc->pmat, &jac->diag, NULL));
  PetscCall(VecLockReadPush(jac->diag));
  PetscCall(PCSetUp_Jacobi(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCApply_Jacobi - Applies the Jacobi preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
static PetscErrorCode PCApply_Jacobi(PC pc, Vec x, Vec y)
{
  PC_Jacobi *jac = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  if (!jac->diag) PetscCall(PCSetUp_Jacobi_NonSymmetric(pc));
  PetscCall(VecPointwiseMult(y, x, jac->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCApplySymmetricLeftOrRight_Jacobi - Applies the left or right part of a
   symmetric preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routines: PCApplySymmetricLeft(), PCApplySymmetricRight()
*/
static PetscErrorCode PCApplySymmetricLeftOrRight_Jacobi(PC pc, Vec x, Vec y)
{
  PC_Jacobi *jac = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  if (!jac->diagsqrt) PetscCall(PCSetUp_Jacobi_Symmetric(pc));
  PetscCall(VecPointwiseMult(y, x, jac->diagsqrt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_Jacobi(PC pc)
{
  PC_Jacobi *jac = (PC_Jacobi *)pc->data;

  PetscFunctionBegin;
  if (jac->diag) PetscCall(VecLockReadPop(jac->diag));
  if (jac->diagsqrt) PetscCall(VecLockReadPop(jac->diagsqrt));
  PetscCall(VecDestroy(&jac->diag));
  PetscCall(VecDestroy(&jac->diagsqrt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCDestroy_Jacobi - Destroys the private context for the Jacobi preconditioner
   that was created with PCCreate_Jacobi().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
static PetscErrorCode PCDestroy_Jacobi(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCReset_Jacobi(pc));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiGetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiSetUseAbs_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiGetUseAbs_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiSetRowl1Scale_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiGetRowl1Scale_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiSetFixDiagonal_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiGetFixDiagonal_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiGetDiagonal_C", NULL));

  /*
      Free the private data structure that was hanging off the PC
  */
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_Jacobi(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_Jacobi   *jac = (PC_Jacobi *)pc->data;
  PetscBool    flg;
  PCJacobiType deflt, type;

  PetscFunctionBegin;
  PetscCall(PCJacobiGetType(pc, &deflt));
  PetscOptionsHeadBegin(PetscOptionsObject, "Jacobi options");
  PetscCall(PetscOptionsEnum("-pc_jacobi_type", "How to construct diagonal matrix", "PCJacobiSetType", PCJacobiTypes, (PetscEnum)deflt, (PetscEnum *)&type, &flg));
  if (flg) PetscCall(PCJacobiSetType(pc, type));
  PetscCall(PetscOptionsBool("-pc_jacobi_abs", "Use absolute values of diagonal entries", "PCJacobiSetUseAbs", jac->useabs, &jac->useabs, NULL));
  PetscCall(PetscOptionsBool("-pc_jacobi_fixdiagonal", "Fix null terms on diagonal", "PCJacobiSetFixDiagonal", jac->fixdiag, &jac->fixdiag, NULL));
  PetscCall(PetscOptionsRangeReal("-pc_jacobi_rowl1_scale", "scaling of off-diagonal elements for rowl1", "PCJacobiSetRowl1Scale", jac->scale, &jac->scale, NULL, 0.0, 1.0));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_Jacobi(PC pc, PetscViewer viewer)
{
  PC_Jacobi *jac = (PC_Jacobi *)pc->data;
  PetscBool  isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PCJacobiType      type;
    PetscBool         useAbs, fixdiag;
    PetscViewerFormat format;
    PetscReal         scale;

    PetscCall(PCJacobiGetType(pc, &type));
    PetscCall(PCJacobiGetUseAbs(pc, &useAbs));
    PetscCall(PCJacobiGetFixDiagonal(pc, &fixdiag));
    PetscCall(PCJacobiGetRowl1Scale(pc, &scale));
    if (type == PC_JACOBI_ROWL1)
      PetscCall(PetscViewerASCIIPrintf(viewer, "  type %s%s%s (l1-norm off-diagonal scaling %e)\n", PCJacobiTypes[type], useAbs ? ", using absolute value of entries" : "", !fixdiag ? ", not checking null diagonal entries" : "", (double)scale));
    else PetscCall(PetscViewerASCIIPrintf(viewer, "  type %s%s%s\n", PCJacobiTypes[type], useAbs ? ", using absolute value of entries" : "", !fixdiag ? ", not checking null diagonal entries" : ""));
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL && jac->diag) PetscCall(VecView(jac->diag, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCCreate_Jacobi - Creates a Jacobi preconditioner context, PC_Jacobi,
   and sets this as the private data within the generic preconditioning
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/

/*MC
     PCJACOBI - Jacobi (i.e. diagonal scaling preconditioning)

   Options Database Keys:
+    -pc_jacobi_type <diagonal,rowl1,rowmax,rowsum> - approach for forming the preconditioner
.    -pc_jacobi_abs - use the absolute value of the diagonal entry
.    -pc_jacobi_rowl1_scale - scaling of off-diagonal terms
-    -pc_jacobi_fixdiag - fix for zero diagonal terms by placing 1.0 in those locations

   Level: beginner

  Notes:
    By using `KSPSetPCSide`(ksp,`PC_SYMMETRIC`) or -ksp_pc_side symmetric
    can scale each side of the matrix by the square root of the diagonal entries.

    Zero entries along the diagonal are replaced with the value 1.0

    See `PCPBJACOBI` for fixed-size point block, `PCVPBJACOBI` for variable-sized point block, and `PCBJACOBI` for large size blocks

.seealso:  `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
           `PCJacobiSetType()`, `PCJacobiSetUseAbs()`, `PCJacobiGetUseAbs()`, `PCASM`,
           `PCJacobiSetFixDiagonal()`, `PCJacobiGetFixDiagonal()`
           `PCJacobiSetType()`, `PCJacobiSetUseAbs()`, `PCJacobiGetUseAbs()`, `PCPBJACOBI`, `PCBJACOBI`, `PCVPBJACOBI`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Jacobi(PC pc)
{
  PC_Jacobi *jac;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  PetscCall(PetscNew(&jac));
  pc->data = (void *)jac;

  /*
     Initialize the pointers to vectors to ZERO; these will be used to store
     diagonal entries of the matrix for fast preconditioner application.
  */
  jac->diag     = NULL;
  jac->diagsqrt = NULL;
  jac->type     = PC_JACOBI_DIAGONAL;
  jac->useabs   = PETSC_FALSE;
  jac->fixdiag  = PETSC_TRUE;
  jac->scale    = 1.0;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_Jacobi;
  pc->ops->applytranspose      = PCApply_Jacobi;
  pc->ops->setup               = PCSetUp_Jacobi;
  pc->ops->reset               = PCReset_Jacobi;
  pc->ops->destroy             = PCDestroy_Jacobi;
  pc->ops->setfromoptions      = PCSetFromOptions_Jacobi;
  pc->ops->view                = PCView_Jacobi;
  pc->ops->applyrichardson     = NULL;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeftOrRight_Jacobi;
  pc->ops->applysymmetricright = PCApplySymmetricLeftOrRight_Jacobi;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiSetType_C", PCJacobiSetType_Jacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiGetType_C", PCJacobiGetType_Jacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiSetRowl1Scale_C", PCJacobiSetRowl1Scale_Jacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiGetRowl1Scale_C", PCJacobiGetRowl1Scale_Jacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiSetUseAbs_C", PCJacobiSetUseAbs_Jacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiGetUseAbs_C", PCJacobiGetUseAbs_Jacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiSetFixDiagonal_C", PCJacobiSetFixDiagonal_Jacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiGetFixDiagonal_C", PCJacobiGetFixDiagonal_Jacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCJacobiGetDiagonal_C", PCJacobiGetDiagonal_Jacobi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCJacobiSetUseAbs - Causes the Jacobi preconditioner `PCJACOBI` to use the
  absolute values of the diagonal divisors in the preconditioner

  Logically Collective

  Input Parameters:
+ pc  - the preconditioner context
- flg - whether to use absolute values or not

  Options Database Key:
. -pc_jacobi_abs <bool> - use absolute values

  Note:
  This takes affect at the next construction of the preconditioner

  Level: intermediate

.seealso: [](ch_ksp), `PCJACOBI`, `PCJacobiSetType()`, `PCJacobiGetUseAbs()`
@*/
PetscErrorCode PCJacobiSetUseAbs(PC pc, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCJacobiSetUseAbs_C", (PC, PetscBool), (pc, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCJacobiGetUseAbs - Determines if the Jacobi preconditioner `PCJACOBI` uses the
  absolute values of the diagonal divisors in the preconditioner

  Logically Collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. flg - whether to use absolute values or not

  Level: intermediate

.seealso: [](ch_ksp), `PCJACOBI`, `PCJacobiSetType()`, `PCJacobiSetUseAbs()`, `PCJacobiGetType()`
@*/
PetscErrorCode PCJacobiGetUseAbs(PC pc, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCJacobiGetUseAbs_C", (PC, PetscBool *), (pc, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCJacobiSetRowl1Scale - Set scaling of off-diagonal of operator when computing l1 row norms, eg,
  Remark 6.1 in "Multigrid Smoothers for Ultraparallel Computing", Baker et al, with 0.5 scaling

  Logically Collective

  Input Parameters:
+ pc    - the preconditioner context
- scale - scaling

  Options Database Key:
. -pc_jacobi_rowl1_scale <real> - use absolute values

  Level: intermediate

.seealso: [](ch_ksp), `PCJACOBI`, `PCJacobiSetType()`, `PCJacobiGetRowl1Scale()`
@*/
PetscErrorCode PCJacobiSetRowl1Scale(PC pc, PetscReal scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCJacobiSetRowl1Scale_C", (PC, PetscReal), (pc, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCJacobiGetRowl1Scale - Get scaling of off-diagonal elements summed into l1-norm diagonal

  Logically Collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. scale - scaling

  Level: intermediate

.seealso: [](ch_ksp), `PCJACOBI`, `PCJacobiSetType()`, `PCJacobiSetRowl1Scale()`, `PCJacobiGetType()`
@*/
PetscErrorCode PCJacobiGetRowl1Scale(PC pc, PetscReal *scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCJacobiGetRowl1Scale_C", (PC, PetscReal *), (pc, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCJacobiSetFixDiagonal - Check for zero values on the diagonal and replace them with 1.0

  Logically Collective

  Input Parameters:
+ pc  - the preconditioner context
- flg - the boolean flag

  Options Database Key:
. -pc_jacobi_fixdiagonal <bool> - check for zero values on the diagonal

  Note:
  This takes affect at the next construction of the preconditioner

  Level: intermediate

.seealso: [](ch_ksp), `PCJACOBI`, `PCJacobiSetType()`, `PCJacobiGetFixDiagonal()`, `PCJacobiSetUseAbs()`
@*/
PetscErrorCode PCJacobiSetFixDiagonal(PC pc, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCJacobiSetFixDiagonal_C", (PC, PetscBool), (pc, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCJacobiGetFixDiagonal - Determines if the Jacobi preconditioner `PCJACOBI` checks for zero diagonal terms

  Logically Collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. flg - the boolean flag

  Options Database Key:
. -pc_jacobi_fixdiagonal <bool> - Fix 0 terms on diagonal by using 1

  Level: intermediate

.seealso: [](ch_ksp), `PCJACOBI`, `PCJacobiSetType()`, `PCJacobiSetFixDiagonal()`
@*/
PetscErrorCode PCJacobiGetFixDiagonal(PC pc, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCJacobiGetFixDiagonal_C", (PC, PetscBool *), (pc, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCJacobiGetDiagonal - Returns copy of the diagonal and/or diagonal squareroot `Vec`

  Logically Collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameters:
+ diagonal      - Copy of `Vec` of the inverted diagonal
- diagonal_sqrt - Copy of `Vec` of the inverted square root diagonal

  Level: developer

.seealso: [](ch_ksp), `PCJACOBI`, `PCJacobiSetType()`
@*/
PetscErrorCode PCJacobiGetDiagonal(PC pc, Vec diagonal, Vec diagonal_sqrt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCJacobiGetDiagonal_C", (PC, Vec, Vec), (pc, diagonal, diagonal_sqrt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCJacobiSetType - Causes the Jacobi preconditioner to use either the diagonal, the maximum entry in each row,
  of the sum of rows entries for the diagonal preconditioner

  Logically Collective

  Input Parameters:
+ pc   - the preconditioner context
- type - `PC_JACOBI_DIAGONAL`, `PC_JACOBI_ROWL1`, `PC_JACOBI_ROWMAX`, `PC_JACOBI_ROWSUM`

  Options Database Key:
. -pc_jacobi_type <diagonal,rowl1,rowmax,rowsum> - the type of diagonal matrix to use for Jacobi

  Level: intermediate

  Developer Notes:
  Why is there a separate function for using the absolute value?

.seealso: [](ch_ksp), `PCJACOBI`, `PCJacobiSetUseAbs()`, `PCJacobiGetType()`
@*/
PetscErrorCode PCJacobiSetType(PC pc, PCJacobiType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCJacobiSetType_C", (PC, PCJacobiType), (pc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCJacobiGetType - Gets how the diagonal matrix is produced for the preconditioner

  Not Collective

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. type - `PC_JACOBI_DIAGONAL`, `PC_JACOBI_ROWL1`, `PC_JACOBI_ROWMAX`, `PC_JACOBI_ROWSUM`

  Level: intermediate

.seealso: [](ch_ksp), `PCJACOBI`, `PCJacobiSetUseAbs()`, `PCJacobiSetType()`
@*/
PetscErrorCode PCJacobiGetType(PC pc, PCJacobiType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscUseMethod(pc, "PCJacobiGetType_C", (PC, PCJacobiType *), (pc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}
