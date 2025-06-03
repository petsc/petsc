#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/

typedef struct {
  PetscBool allocated, commute, scalediag;
  KSP       kspL, kspMass;
  Vec       Avec0, Avec1, Svec0, scale;
  Mat       L;
} PC_LSC;

static PetscErrorCode PCLSCAllocate_Private(PC pc)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;
  Mat     A;

  PetscFunctionBegin;
  if (lsc->allocated) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &lsc->kspL));
  PetscCall(KSPSetNestLevel(lsc->kspL, pc->kspnestlevel));
  PetscCall(KSPSetErrorIfNotConverged(lsc->kspL, pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)lsc->kspL, (PetscObject)pc, 1));
  PetscCall(KSPSetType(lsc->kspL, KSPPREONLY));
  PetscCall(KSPSetOptionsPrefix(lsc->kspL, ((PetscObject)pc)->prefix));
  PetscCall(KSPAppendOptionsPrefix(lsc->kspL, "lsc_"));
  PetscCall(MatSchurComplementGetSubMatrices(pc->mat, &A, NULL, NULL, NULL, NULL));
  PetscCall(MatCreateVecs(A, &lsc->Avec0, &lsc->Avec1));
  PetscCall(MatCreateVecs(pc->pmat, &lsc->Svec0, NULL));
  if (lsc->scalediag) PetscCall(VecDuplicate(lsc->Avec0, &lsc->scale));

  if (lsc->commute) {
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &lsc->kspMass));
    PetscCall(KSPSetErrorIfNotConverged(lsc->kspMass, pc->erroriffailure));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)lsc->kspMass, (PetscObject)pc, 1));
    PetscCall(KSPSetType(lsc->kspMass, KSPPREONLY));
    PetscCall(KSPSetOptionsPrefix(lsc->kspMass, ((PetscObject)pc)->prefix));
    PetscCall(KSPAppendOptionsPrefix(lsc->kspMass, "lsc_mass_"));
  } else lsc->kspMass = NULL;

  lsc->allocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_LSC(PC pc)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;
  Mat     L, Lp, Qscale;

  PetscFunctionBegin;
  PetscCall(PCLSCAllocate_Private(pc));

  /* Query for L operator */
  PetscCall(PetscObjectQuery((PetscObject)pc->mat, "LSC_L", (PetscObject *)&L));
  if (!L) PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_L", (PetscObject *)&L));
  PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_Lp", (PetscObject *)&Lp));
  if (!Lp) PetscCall(PetscObjectQuery((PetscObject)pc->mat, "LSC_Lp", (PetscObject *)&Lp));

  /* Query for mass operator */
  PetscCall(PetscObjectQuery((PetscObject)pc->pmat, "LSC_Qscale", (PetscObject *)&Qscale));
  if (!Qscale) PetscCall(PetscObjectQuery((PetscObject)pc->mat, "LSC_Qscale", (PetscObject *)&Qscale));

  if (lsc->commute) {
    PetscCheck(L || Lp, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "The user must provide an L operator for LSC preconditioning when commuting");
    if (!L && Lp) L = Lp;
    else if (L && !Lp) Lp = L;

    PetscCheck(Qscale, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "The user must provide a Qscale matrix for LSC preconditioning when commuting");
  } else {
    if (lsc->scale) {
      if (!Qscale) PetscCall(MatSchurComplementGetSubMatrices(pc->mat, NULL, &Qscale, NULL, NULL, NULL));
      PetscCall(MatGetDiagonal(Qscale, lsc->scale));
      PetscCall(VecReciprocal(lsc->scale));
    }
    if (!L) {
      Mat B, C;
      PetscCall(MatSchurComplementGetSubMatrices(pc->mat, NULL, NULL, &B, &C, NULL));
      if (lsc->scale) {
        Mat CAdiaginv;
        PetscCall(MatDuplicate(C, MAT_COPY_VALUES, &CAdiaginv));
        PetscCall(MatDiagonalScale(CAdiaginv, NULL, lsc->scale));
        if (!lsc->L) PetscCall(MatMatMult(CAdiaginv, B, MAT_INITIAL_MATRIX, PETSC_CURRENT, &lsc->L));
        else PetscCall(MatMatMult(CAdiaginv, B, MAT_REUSE_MATRIX, PETSC_CURRENT, &lsc->L));
        PetscCall(MatDestroy(&CAdiaginv));
      } else {
        if (!lsc->L) {
          PetscCall(MatProductCreate(C, B, NULL, &lsc->L));
          PetscCall(MatProductSetType(lsc->L, MATPRODUCT_AB));
          PetscCall(MatProductSetFromOptions(lsc->L));
          PetscCall(MatProductSymbolic(lsc->L));
        }
        PetscCall(MatProductNumeric(lsc->L));
      }
      Lp = L = lsc->L;
    }
  }

  PetscCall(KSPSetOperators(lsc->kspL, L, Lp));
  PetscCall(KSPSetFromOptions(lsc->kspL));
  if (lsc->commute) {
    PetscCall(KSPSetOperators(lsc->kspMass, Qscale, Qscale));
    PetscCall(KSPSetFromOptions(lsc->kspMass));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_LSC(PC pc, Vec x, Vec y)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;
  Mat     A, B, C;

  PetscFunctionBegin;
  PetscCall(MatSchurComplementGetSubMatrices(pc->mat, &A, NULL, &B, &C, NULL));
  if (lsc->commute) {
    PetscCall(KSPSolve(lsc->kspMass, x, lsc->Svec0));
    PetscCall(KSPCheckSolve(lsc->kspMass, pc, lsc->Svec0));
    PetscCall(MatMult(B, lsc->Svec0, lsc->Avec0));
    PetscCall(KSPSolve(lsc->kspL, lsc->Avec0, lsc->Avec1));
    PetscCall(KSPCheckSolve(lsc->kspL, pc, lsc->Avec1));
    PetscCall(MatMult(A, lsc->Avec1, lsc->Avec0));
    PetscCall(KSPSolve(lsc->kspL, lsc->Avec0, lsc->Avec1));
    PetscCall(KSPCheckSolve(lsc->kspL, pc, lsc->Avec1));
    PetscCall(MatMult(C, lsc->Avec1, lsc->Svec0));
    PetscCall(KSPSolve(lsc->kspMass, lsc->Svec0, y));
    PetscCall(KSPCheckSolve(lsc->kspMass, pc, y));
  } else {
    PetscCall(KSPSolve(lsc->kspL, x, lsc->Svec0));
    PetscCall(KSPCheckSolve(lsc->kspL, pc, lsc->Svec0));
    PetscCall(MatMult(B, lsc->Svec0, lsc->Avec0));
    if (lsc->scale) PetscCall(VecPointwiseMult(lsc->Avec0, lsc->Avec0, lsc->scale));
    PetscCall(MatMult(A, lsc->Avec0, lsc->Avec1));
    if (lsc->scale) PetscCall(VecPointwiseMult(lsc->Avec1, lsc->Avec1, lsc->scale));
    PetscCall(MatMult(C, lsc->Avec1, lsc->Svec0));
    PetscCall(KSPSolve(lsc->kspL, lsc->Svec0, y));
    PetscCall(KSPCheckSolve(lsc->kspL, pc, y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_LSC(PC pc)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&lsc->Avec0));
  PetscCall(VecDestroy(&lsc->Avec1));
  PetscCall(VecDestroy(&lsc->Svec0));
  PetscCall(KSPDestroy(&lsc->kspL));
  if (lsc->commute) PetscCall(KSPDestroy(&lsc->kspMass));
  if (lsc->L) PetscCall(MatDestroy(&lsc->L));
  if (lsc->scale) PetscCall(VecDestroy(&lsc->scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_LSC(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCReset_LSC(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_LSC(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_LSC *lsc = (PC_LSC *)pc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "LSC options");
  {
    PetscCall(PetscOptionsBool("-pc_lsc_commute", "Whether to commute the LSC preconditioner in the style of Olshanskii", "None", lsc->commute, &lsc->commute, NULL));
    PetscCall(PetscOptionsBool("-pc_lsc_scale_diag", "Whether to scale BBt products. Will use the inverse of the diagonal of Qscale or A if the former is not provided.", "None", lsc->scalediag, &lsc->scalediag, NULL));
    PetscCheck(!lsc->scalediag || !lsc->commute, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "Diagonal-based scaling is not used when doing a commuted LSC. Either do not ask for diagonal-based scaling or use non-commuted LSC in the original style of Elman");
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_LSC(PC pc, PetscViewer viewer)
{
  PC_LSC   *jac = (PC_LSC *)pc->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (jac->kspL) {
      PetscCall(KSPView(jac->kspL, viewer));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "PCLSC KSP object not yet created, hence cannot display"));
    }
    if (jac->commute) {
      if (jac->kspMass) {
        PetscCall(KSPView(jac->kspMass, viewer));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "PCLSC Mass KSP object not yet created, hence cannot display"));
      }
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PCLSC - Preconditioning for Schur complements, based on Least Squares Commutators {cite}`elmanhowleshadidshuttleworthtuminaro2006` {cite}`silvester2001efficient`

   Options Database Key:
+    -pc_lsc_commute    - Whether to commute the LSC preconditioner in the style of Olshanskii
-    -pc_lsc_scale_diag - Whether to scale $BB^T$ products. Will use the inverse of the diagonal of $Qscale$ or $A$ if the former is not provided

   Level: intermediate

   Notes:
   This preconditioner will normally be used with `PCFIELDSPLIT` to precondition the Schur complement, but
   it can be used for any Schur complement system.  Consider the Schur complement

   $$
   S = A11 - A10 A00^{-1} A01
   $$

   `PCLSC` currently doesn't do anything with $A11$, so let's assume it is 0.  The idea is that a good approximation to
   $S^{-1}$ is given by

   $$
   (A10 A01)^{-1} A10 A00 A01 (A10 A01)^{-1}
   $$

   The product $A10 A01$ can be computed for you, but you can provide it (this is
   usually more efficient anyway).  In the case of incompressible flow, $A10 A01$ is a Laplacian; call it $L$.  The current
   interface is to compose $L$ and a matrix from which to construct a preconditioner $Lp$ on the matrix.

   If you had called `KSPSetOperators`(ksp,S,Sp), $S$ should have type `MATSCHURCOMPLEMENT` and $Sp$ can be any type you
   like (`PCLSC` doesn't use it directly) but should have matrices composed with it, under the names "LSC_L" and "LSC_Lp".
   For example, you might have setup code like this

.vb
   PetscObjectCompose((PetscObject)Sp,"LSC_L",(PetscObject)L);
   PetscObjectCompose((PetscObject)Sp,"LSC_Lp",(PetscObject)Lp);
.ve

   And then your Jacobian assembly would look like

.vb
   PetscObjectQuery((PetscObject)Sp,"LSC_L",(PetscObject*)&L);
   PetscObjectQuery((PetscObject)Sp,"LSC_Lp",(PetscObject*)&Lp);
   if (L) { assemble L }
   if (Lp) { assemble Lp }
.ve

   With this, you should be able to choose LSC preconditioning, using e.g. the `PCML` algebraic multigrid to solve with L
.vb
   -fieldsplit_1_pc_type lsc -fieldsplit_1_lsc_pc_type ml
.ve

   Since we do not use the values in Sp, you can still put an assembled matrix there to use normal preconditioners.

.seealso: [](ch_ksp), `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `Block_Preconditioners`, `PCFIELDSPLIT`,
          `PCFieldSplitGetSubKSP()`, `PCFieldSplitSetFields()`, `PCFieldSplitSetType()`, `PCFieldSplitSetIS()`, `PCFieldSplitSetSchurPre()`,
          `MatCreateSchurComplement()`, `MatCreateSchurComplement()`, `MatSchurComplementSetSubMatrices()`, `MatSchurComplementUpdateSubMatrices()`,
          `MatSchurComplementSetAinvType()`, `MatGetSchurComplement()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_LSC(PC pc)
{
  PC_LSC *lsc;

  PetscFunctionBegin;
  PetscCall(PetscNew(&lsc));
  pc->data = (void *)lsc;

  pc->ops->apply           = PCApply_LSC;
  pc->ops->applytranspose  = NULL;
  pc->ops->setup           = PCSetUp_LSC;
  pc->ops->reset           = PCReset_LSC;
  pc->ops->destroy         = PCDestroy_LSC;
  pc->ops->setfromoptions  = PCSetFromOptions_LSC;
  pc->ops->view            = PCView_LSC;
  pc->ops->applyrichardson = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
