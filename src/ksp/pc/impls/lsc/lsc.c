#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/

typedef struct {
  PetscBool allocated;
  PetscBool scalediag;
  KSP       kspL;
  Vec       scale;
  Vec       x0,y0,x1;
  Mat       L;             /* keep a copy to reuse when obtained with L = A10*A01 */
} PC_LSC;

static PetscErrorCode PCLSCAllocate_Private(PC pc)
{
  PC_LSC         *lsc = (PC_LSC*)pc->data;
  Mat            A;

  PetscFunctionBegin;
  if (lsc->allocated) PetscFunctionReturn(0);
  CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)pc),&lsc->kspL));
  CHKERRQ(KSPSetErrorIfNotConverged(lsc->kspL,pc->erroriffailure));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)lsc->kspL,(PetscObject)pc,1));
  CHKERRQ(KSPSetType(lsc->kspL,KSPPREONLY));
  CHKERRQ(KSPSetOptionsPrefix(lsc->kspL,((PetscObject)pc)->prefix));
  CHKERRQ(KSPAppendOptionsPrefix(lsc->kspL,"lsc_"));
  CHKERRQ(MatSchurComplementGetSubMatrices(pc->mat,&A,NULL,NULL,NULL,NULL));
  CHKERRQ(MatCreateVecs(A,&lsc->x0,&lsc->y0));
  CHKERRQ(MatCreateVecs(pc->pmat,&lsc->x1,NULL));
  if (lsc->scalediag) {
    CHKERRQ(VecDuplicate(lsc->x0,&lsc->scale));
  }
  lsc->allocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_LSC(PC pc)
{
  PC_LSC         *lsc = (PC_LSC*)pc->data;
  Mat            L,Lp,B,C;

  PetscFunctionBegin;
  CHKERRQ(PCLSCAllocate_Private(pc));
  CHKERRQ(PetscObjectQuery((PetscObject)pc->mat,"LSC_L",(PetscObject*)&L));
  if (!L) CHKERRQ(PetscObjectQuery((PetscObject)pc->pmat,"LSC_L",(PetscObject*)&L));
  CHKERRQ(PetscObjectQuery((PetscObject)pc->pmat,"LSC_Lp",(PetscObject*)&Lp));
  if (!Lp) CHKERRQ(PetscObjectQuery((PetscObject)pc->mat,"LSC_Lp",(PetscObject*)&Lp));
  if (!L) {
    CHKERRQ(MatSchurComplementGetSubMatrices(pc->mat,NULL,NULL,&B,&C,NULL));
    if (!lsc->L) {
      CHKERRQ(MatMatMult(C,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&lsc->L));
    } else {
      CHKERRQ(MatMatMult(C,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&lsc->L));
    }
    Lp = L = lsc->L;
  }
  if (lsc->scale) {
    Mat Ap;
    CHKERRQ(MatSchurComplementGetSubMatrices(pc->mat,NULL,&Ap,NULL,NULL,NULL));
    CHKERRQ(MatGetDiagonal(Ap,lsc->scale)); /* Should be the mass matrix, but we don't have plumbing for that yet */
    CHKERRQ(VecReciprocal(lsc->scale));
  }
  CHKERRQ(KSPSetOperators(lsc->kspL,L,Lp));
  CHKERRQ(KSPSetFromOptions(lsc->kspL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_LSC(PC pc,Vec x,Vec y)
{
  PC_LSC         *lsc = (PC_LSC*)pc->data;
  Mat            A,B,C;

  PetscFunctionBegin;
  CHKERRQ(MatSchurComplementGetSubMatrices(pc->mat,&A,NULL,&B,&C,NULL));
  CHKERRQ(KSPSolve(lsc->kspL,x,lsc->x1));
  CHKERRQ(KSPCheckSolve(lsc->kspL,pc,lsc->x1));
  CHKERRQ(MatMult(B,lsc->x1,lsc->x0));
  if (lsc->scale) {
    CHKERRQ(VecPointwiseMult(lsc->x0,lsc->x0,lsc->scale));
  }
  CHKERRQ(MatMult(A,lsc->x0,lsc->y0));
  if (lsc->scale) {
    CHKERRQ(VecPointwiseMult(lsc->y0,lsc->y0,lsc->scale));
  }
  CHKERRQ(MatMult(C,lsc->y0,lsc->x1));
  CHKERRQ(KSPSolve(lsc->kspL,lsc->x1,y));
  CHKERRQ(KSPCheckSolve(lsc->kspL,pc,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_LSC(PC pc)
{
  PC_LSC         *lsc = (PC_LSC*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&lsc->x0));
  CHKERRQ(VecDestroy(&lsc->y0));
  CHKERRQ(VecDestroy(&lsc->x1));
  CHKERRQ(VecDestroy(&lsc->scale));
  CHKERRQ(KSPDestroy(&lsc->kspL));
  CHKERRQ(MatDestroy(&lsc->L));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_LSC(PC pc)
{
  PetscFunctionBegin;
  CHKERRQ(PCReset_LSC(pc));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_LSC(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_LSC         *lsc = (PC_LSC*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"LSC options"));
  {
    CHKERRQ(PetscOptionsBool("-pc_lsc_scale_diag","Use diagonal of velocity block (A) for scaling","None",lsc->scalediag,&lsc->scalediag,NULL));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_LSC(PC pc,PetscViewer viewer)
{
  PC_LSC         *jac = (PC_LSC*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    if (jac->kspL) {
      CHKERRQ(KSPView(jac->kspL,viewer));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"PCLSC KSP object not yet created, hence cannot display"));
    }
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*MC
     PCLSC - Preconditioning for Schur complements, based on Least Squares Commutators

   Options Database Key:
.    -pc_lsc_scale_diag - Use the diagonal of A for scaling

   Level: intermediate

   Notes:
   This preconditioner will normally be used with PCFieldSplit to precondition the Schur complement, but
   it can be used for any Schur complement system.  Consider the Schur complement

.vb
   S = A11 - A10 inv(A00) A01
.ve

   PCLSC currently doesn't do anything with A11, so let's assume it is 0.  The idea is that a good approximation to
   inv(S) is given by

.vb
   inv(A10 A01) A10 A00 A01 inv(A10 A01)
.ve

   The product A10 A01 can be computed for you, but you can provide it (this is
   usually more efficient anyway).  In the case of incompressible flow, A10 A01 is a Laplacian; call it L.  The current
   interface is to hang L and a preconditioning matrix Lp on the preconditioning matrix.

   If you had called KSPSetOperators(ksp,S,Sp), S should have type MATSCHURCOMPLEMENT and Sp can be any type you
   like (PCLSC doesn't use it directly) but should have matrices composed with it, under the names "LSC_L" and "LSC_Lp".
   For example, you might have setup code like this

.vb
   PetscObjectCompose((PetscObject)Sp,"LSC_L",(PetscObject)L);
   PetscObjectCompose((PetscObject)Sp,"LSC_Lp",(PetscObject)Lp);
.ve

   And then your Jacobian assembly would look like

.vb
   PetscObjectQuery((PetscObject)Sp,"LSC_L",(PetscObject*)&L);
   PetscObjectQuery((PetscObject)Sp,"LSC_Lp",(PetscObject*)&Lp);
   if (L) { assembly L }
   if (Lp) { assemble Lp }
.ve

   With this, you should be able to choose LSC preconditioning, using e.g. ML's algebraic multigrid to solve with L

.vb
   -fieldsplit_1_pc_type lsc -fieldsplit_1_lsc_pc_type ml
.ve

   Since we do not use the values in Sp, you can still put an assembled matrix there to use normal preconditioners.

   References:
+  * - Elman, Howle, Shadid, Shuttleworth, and Tuminaro, Block preconditioners based on approximate commutators, 2006.
-  * - Silvester, Elman, Kay, Wathen, Efficient preconditioning of the linearized Navier Stokes equations for incompressible flow, 2001.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, Block_Preconditioners, PCFIELDSPLIT,
           PCFieldSplitGetSubKSP(), PCFieldSplitSetFields(), PCFieldSplitSetType(), PCFieldSplitSetIS(), PCFieldSplitSetSchurPre(),
           MatCreateSchurComplement()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_LSC(PC pc)
{
  PC_LSC         *lsc;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pc,&lsc));
  pc->data = (void*)lsc;

  pc->ops->apply           = PCApply_LSC;
  pc->ops->applytranspose  = NULL;
  pc->ops->setup           = PCSetUp_LSC;
  pc->ops->reset           = PCReset_LSC;
  pc->ops->destroy         = PCDestroy_LSC;
  pc->ops->setfromoptions  = PCSetFromOptions_LSC;
  pc->ops->view            = PCView_LSC;
  pc->ops->applyrichardson = NULL;
  PetscFunctionReturn(0);
}
