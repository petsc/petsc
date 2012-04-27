
#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/

typedef struct {
  PetscBool  allocated;
  PetscBool  scalediag;
  KSP        kspL;
  Vec        scale;
  Vec        x0,y0,x1;
  Mat        L;            /* keep a copy to reuse when obtained with L = A10*A01 */
} PC_LSC;

#undef __FUNCT__  
#define __FUNCT__ "PCLSCAllocate_Private"
static PetscErrorCode PCLSCAllocate_Private(PC pc)
{
  PC_LSC         *lsc = (PC_LSC*)pc->data;
  Mat             A;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (lsc->allocated) PetscFunctionReturn(0);
  ierr = KSPCreate(((PetscObject)pc)->comm,&lsc->kspL);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)lsc->kspL,(PetscObject)pc,1);CHKERRQ(ierr);
  ierr = KSPSetType(lsc->kspL,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(lsc->kspL,((PetscObject)pc)->prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(lsc->kspL,"lsc_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(lsc->kspL);CHKERRQ(ierr);
  ierr = MatSchurComplementGetSubmatrices(pc->mat,&A,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&lsc->x0,&lsc->y0);CHKERRQ(ierr);
  ierr = MatGetVecs(pc->pmat,&lsc->x1,PETSC_NULL);CHKERRQ(ierr);
  if (lsc->scalediag) {
    ierr = VecDuplicate(lsc->x0,&lsc->scale);CHKERRQ(ierr);
  }
  lsc->allocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_LSC"
static PetscErrorCode PCSetUp_LSC(PC pc)
{
  PC_LSC         *lsc = (PC_LSC*)pc->data;
  Mat             L,Lp,B,C;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PCLSCAllocate_Private(pc);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)pc->mat,"LSC_L",(PetscObject*)&L);CHKERRQ(ierr);
  if (!L) {ierr = PetscObjectQuery((PetscObject)pc->pmat,"LSC_L",(PetscObject*)&L);CHKERRQ(ierr);}
  ierr = PetscObjectQuery((PetscObject)pc->pmat,"LSC_Lp",(PetscObject*)&Lp);CHKERRQ(ierr);
  if (!Lp) {ierr = PetscObjectQuery((PetscObject)pc->mat,"LSC_Lp",(PetscObject*)&Lp);CHKERRQ(ierr);}
  if (!L) {
    ierr = MatSchurComplementGetSubmatrices(pc->mat,PETSC_NULL,PETSC_NULL,&B,&C,PETSC_NULL);CHKERRQ(ierr);
    if (!lsc->L) {
      ierr = MatMatMult(C,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&lsc->L);CHKERRQ(ierr);
    } else {
      ierr = MatMatMult(C,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&lsc->L);CHKERRQ(ierr);
    }
    Lp = L = lsc->L;
  }
  if (lsc->scale) {
    Mat Ap;
    ierr = MatSchurComplementGetSubmatrices(pc->mat,PETSC_NULL,&Ap,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatGetDiagonal(Ap,lsc->scale);CHKERRQ(ierr); /* Should be the mass matrix, but we don't have plumbing for that yet */
    ierr = VecReciprocal(lsc->scale);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(lsc->kspL,L,Lp,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_LSC"
static PetscErrorCode PCApply_LSC(PC pc,Vec x,Vec y)
{
  PC_LSC        *lsc = (PC_LSC*)pc->data;
  Mat            A,B,C;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSchurComplementGetSubmatrices(pc->mat,&A,PETSC_NULL,&B,&C,PETSC_NULL);CHKERRQ(ierr);
  ierr = KSPSolve(lsc->kspL,x,lsc->x1);CHKERRQ(ierr);
  ierr = MatMult(B,lsc->x1,lsc->x0);CHKERRQ(ierr);
  if (lsc->scale) {
    ierr = VecPointwiseMult(lsc->x0,lsc->x0,lsc->scale);CHKERRQ(ierr);
  }
  ierr = MatMult(A,lsc->x0,lsc->y0);CHKERRQ(ierr);
  if (lsc->scale) {
    ierr = VecPointwiseMult(lsc->y0,lsc->y0,lsc->scale);CHKERRQ(ierr);
  }
  ierr = MatMult(C,lsc->y0,lsc->x1);CHKERRQ(ierr);
  ierr = KSPSolve(lsc->kspL,lsc->x1,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCReset_LSC"
static PetscErrorCode PCReset_LSC(PC pc)
{
  PC_LSC         *lsc = (PC_LSC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&lsc->x0);CHKERRQ(ierr);
  ierr = VecDestroy(&lsc->y0);CHKERRQ(ierr);
  ierr = VecDestroy(&lsc->x1);CHKERRQ(ierr);
  ierr = VecDestroy(&lsc->scale);CHKERRQ(ierr);
  ierr = KSPDestroy(&lsc->kspL);CHKERRQ(ierr);
  ierr = MatDestroy(&lsc->L);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_LSC"
static PetscErrorCode PCDestroy_LSC(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_LSC(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_LSC"
static PetscErrorCode PCSetFromOptions_LSC(PC pc)
{
  PC_LSC         *lsc = (PC_LSC*)pc->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("LSC options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsBool("-pc_lsc_scale_diag","Use diagonal of velocity block (A) for scaling","None",lsc->scalediag,&lsc->scalediag,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_LSC"
static PetscErrorCode PCView_LSC(PC pc,PetscViewer viewer)
{
  PC_LSC           *jac = (PC_LSC*)pc->data;
  PetscErrorCode   ierr;
  PetscBool        iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(jac->kspL,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for LSC",((PetscObject)viewer)->type_name);
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
   usually more efficient anyway).  In the case of incompressible flow, A10 A10 is a Laplacian, call it L.  The current
   interface is to hang L and a preconditioning matrix Lp on the preconditioning matrix.

   If you had called KSPSetOperators(ksp,S,Sp,flg), S should have type MATSCHURCOMPLEMENT and Sp can be any type you
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
+  Elman, Howle, Shadid, Shuttleworth, and Tuminaro, Block preconditioners based on approximate commutators, 2006.
-  Silvester, Elman, Kay, Wathen, Efficient preconditioning of the linearized Navier-Stokes equations for incompressible flow, 2001.

   Concepts: physics based preconditioners, block preconditioners

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, Block_Preconditioners, PCFIELDSPLIT,
           PCFieldSplitGetSubKSP(), PCFieldSplitSetFields(), PCFieldSplitSetType(), PCFieldSplitSetIS(), PCFieldSplitSchurPrecondition(),
           MatCreateSchurComplement()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_LSC"
PetscErrorCode  PCCreate_LSC(PC pc)
{
  PC_LSC         *lsc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr      = PetscNewLog(pc,PC_LSC,&lsc);CHKERRQ(ierr);
  pc->data  = (void*)lsc;

  pc->ops->apply               = PCApply_LSC;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_LSC;
  pc->ops->reset               = PCReset_LSC;
  pc->ops->destroy             = PCDestroy_LSC;
  pc->ops->setfromoptions      = PCSetFromOptions_LSC;
  pc->ops->view                = PCView_LSC;
  pc->ops->applyrichardson     = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
