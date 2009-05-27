#include "private/pcimpl.h"   /*I "petscpc.h" I*/

typedef struct {
  Mat        K;        /* K, the (0,0) block, is [M x M] */
  Mat        G;        /* G, the (0,1) block, is [M x N] */
  Mat        M;        /* The mass matrix corresponding to K */
  Vec        inv_diag_M;
  Mat        GtG;
  Vec        s, t, X;  /* s is [M], t is [N], X is [M] */
  KSP        ksp;
  PetscTruth form_GtG; /* Don't allow anything else yet */
  PetscTruth scaled;   /* Use K for M */
} PC_BFBt;

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCBFBtSetOperators_BFBt"
/* K and G must not be PETSC_NULL, M can be PETSC_NULL */
PetscErrorCode PETSCKSP_DLLEXPORT PCBFBtSetOperators_BFBt(PC pc, Mat K, Mat G, Mat M)
{
  PC_BFBt *ctx = (PC_BFBt *) pc->data;

  PetscFunctionBegin;
  ctx->K = K;
  ctx->G = G;
  ctx->M = M;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PCBFBtSetOperators"
/*@C
  PCBFBtSetOperators - Set the matrix blocks

  Input Parameters:
+ pc - The PC
. K  - The (0,0) block
. G  - The (0,1) block
- M  - The mass matrix associated to K, may be PETSC_NULL

  Level: Intermediate

  Note: K and G must not be PETSC_NULL, M can be PETSC_NULL
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCBFBtSetOperators(PC pc, Mat K, Mat G, Mat M)
{
  PetscErrorCode ierr, (*f)(PC, Mat, Mat, Mat);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject) pc, "PCBFBtSetOperators_C", (void (**)(void)) &f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc, K, G, M);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBFBtGetKSP"
/*@C
  PCBFBtGetKSP - Get the solver for the G^T G system

  Input Parameter:
. pc - The PC

  Output Parameter:
. ksp - The solver

  Level: Intermediate
@*/
PetscErrorCode PCBFBtGetKSP(PC pc, KSP *ksp)
{
  PC_BFBt *ctx = (PC_BFBt *) pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (ksp) {*ksp = ctx->ksp;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBFBtCreateGtG"
PetscErrorCode PCBFBtCreateGtG(PC pc, Mat G, Vec inv_diag_M, Mat *GtG)
{
  MPI_Comm       comm;
  Mat            Ident;
  const MatType  mtype;
  MatInfo        info;
  PetscReal      fill;
  PetscInt       nnz_I, nnz_G;
  PetscInt       M, N, m, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) pc, &comm);CHKERRQ(ierr);
  ierr = MatGetSize(G, &M, &N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(G, &m, &n);CHKERRQ(ierr);
  ierr = MatGetType(G, &mtype);CHKERRQ(ierr);
  ierr = MatCreate(comm, &Ident);CHKERRQ(ierr);
  ierr = MatSetSizes(Ident, m, m, M, M);CHKERRQ(ierr);
  ierr = MatSetType(Ident, mtype);CHKERRQ(ierr);
  if (inv_diag_M == PETSC_NULL) {
    Vec diag;

    ierr = MatGetVecs(G, PETSC_NULL, &diag);CHKERRQ(ierr);
    ierr = VecSet(diag, 1.0);CHKERRQ(ierr);
    ierr = MatDiagonalSet(Ident, diag, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecDestroy(diag);CHKERRQ(ierr);
  } else {
    ierr = MatDiagonalSet(Ident, inv_diag_M, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr  = MatGetInfo(Ident, MAT_GLOBAL_SUM, &info);CHKERRQ(ierr);
  nnz_I = (PetscInt) info.nz_used;
  ierr  = MatGetInfo(G,     MAT_GLOBAL_SUM, &info);CHKERRQ(ierr);
  nnz_G = (PetscInt) info.nz_used;
  /* Not sure the best way to estimate the fill factor.
     GtG is a laplacian on the pressure space.
     This might tell us something useful... */
  fill = ((PetscReal) nnz_G)/nnz_I;
  ierr = MatPtAP(Ident, G, MAT_INITIAL_MATRIX, fill, GtG);CHKERRQ(ierr);
  ierr = MatDestroy(Ident);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
Performs y <- S^{-1} x 
S^{-1} = ( G^T G )^{-1} G^T K G ( G^T G )^{-1}
*/
#undef __FUNCT__
#define __FUNCT__ "PCApply_BFBt"
PetscErrorCode PCApply_BFBt(PC pc, Vec x, Vec y)
{
  PC_BFBt       *ctx = (PC_BFBt *) pc->data;
  KSP            ksp = ctx->ksp;
  Mat            K   = ctx->K;
  Mat            G   = ctx->G;
  Vec            s   = ctx->s;
  Vec            t   = ctx->t;
  Vec            X   = ctx->X;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* t <- GtG_inv x */
  ierr = KSPSolve(ksp, x, t);CHKERRQ(ierr);
  /* s <- G t */
  ierr = MatMult(G, t, s);CHKERRQ(ierr);
  /* X <- K s */
  ierr = MatMult(K, s, X);CHKERRQ(ierr);
  /* t <- Gt X */
  ierr = MatMultTranspose(G, X, t);CHKERRQ(ierr);
  /* y <- GtG_inv t */
  ierr = KSPSolve(ksp, t, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
S^{-1} = ( G^T G )^{-1} G^T K G ( G^T G )^{-1}
       = A C A
S^{-T} = A^T (A C)^T
       = A^T C^T A^T, but A = G^T G which is symmetric
       = A C^T A
       = A G^T ( G^T K )^T A
       = A G^T K^T G A
*/
#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_BFBt"
PetscErrorCode PCApplyTranspose_BFBt(PC pc, Vec x, Vec y)
{
  PC_BFBt       *ctx = (PC_BFBt *) pc->data;
  KSP            ksp = ctx->ksp;
  Mat            K   = ctx->K;
  Mat            G   = ctx->G;
  Vec            s   = ctx->s;
  Vec            t   = ctx->t;
  Vec            X   = ctx->X;
  PetscErrorCode ierr;
	
  PetscFunctionBegin;
  /* t <- GtG_inv x */
  ierr = KSPSolve(ksp, x, t);CHKERRQ(ierr);
  /* s <- G t */
  ierr = MatMult(G, t, s);CHKERRQ(ierr);
  /* X <- K^T s */
  ierr = MatMultTranspose(K, s, X);CHKERRQ(ierr);
  /* t <- Gt X */
  ierr = MatMultTranspose(G, X, t);CHKERRQ(ierr);
  /* y <- GtG_inv t */
  ierr = KSPSolve(ksp, t, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* 
Performs y <- S^{-1} x 
S^{-1} = ( G^T Di G )^{-1} G^T Di K Di G ( G^T Di G )^{-1}
where Di = diag(M)^{-1}
*/
#undef __FUNCT__
#define __FUNCT__ "PCApply_BFBt_diagonal_scaling"
PetscErrorCode PCApply_BFBt_diagonal_scaling(PC pc, Vec x, Vec y)
{
  PC_BFBt       *ctx = (PC_BFBt *) pc->data;
  KSP            ksp = ctx->ksp;
  Mat            K   = ctx->K;
  Mat            G   = ctx->G;
  Vec            s   = ctx->s;
  Vec            t   = ctx->t;
  Vec            X   = ctx->X;
  Vec            di  = ctx->inv_diag_M;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* t <- GtG_inv x */
  ierr = KSPSolve(ksp, x, t);CHKERRQ(ierr);
  /* s <- G t */
  ierr = MatMult(G, t, s);CHKERRQ(ierr);
  /* s <- s * di */
  ierr = VecPointwiseMult(s, s, di);CHKERRQ(ierr);
  /* X <- K s */
  ierr = MatMult(K, s, X);CHKERRQ(ierr);
  /* X <- X * di */
  ierr = VecPointwiseMult(X, X, di);CHKERRQ(ierr);
  /* t <- Gt X */
  ierr = MatMultTranspose(G, X, t);CHKERRQ(ierr);
  /* y <- GtG_inv t */
  ierr = KSPSolve(ksp, t, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_BFBt_diagonal_scaling"
PetscErrorCode PCApplyTranspose_BFBt_diagonal_scaling(PC pc, Vec x, Vec y)
{
  PC_BFBt       *ctx = (PC_BFBt *) pc->data;
  KSP            ksp = ctx->ksp;
  Mat            K   = ctx->K;
  Mat            G   = ctx->G;
  Vec            s   = ctx->s;
  Vec            t   = ctx->t;
  Vec            X   = ctx->X;
  Vec            di  = ctx->inv_diag_M;
  PetscErrorCode ierr;
	
  PetscFunctionBegin;
  /* t <- GtG_inv x */
  ierr = KSPSolve(ksp, x, t);CHKERRQ(ierr);
  /* s <- G t */
  ierr = MatMult(G, t, s);CHKERRQ(ierr);
  /* s <- s * di */
  ierr = VecPointwiseMult(s, s, di);CHKERRQ(ierr);
  /* X <- K^T s */
  ierr = MatMultTranspose(K, s, X);CHKERRQ(ierr);
  /* X <- X * di */
  ierr = VecPointwiseMult(X, X, di);CHKERRQ(ierr);
  /* t <- Gt X */
  ierr = MatMultTranspose(G, X, t);CHKERRQ(ierr);
  /* y <- GtG_inv t */
  ierr = KSPSolve(ksp, t, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_BFBt"
PetscErrorCode PCSetUp_BFBt(PC pc)
{
  PC_BFBt       *ctx = (PC_BFBt*) pc->data;
  MPI_Comm       comm;
  PetscTruth     hasPmat;
  PetscErrorCode ierr;

  ierr = PetscObjectGetComm((PetscObject) pc, &comm);CHKERRQ(ierr);
  ierr = PCGetOperatorsSet(pc, PETSC_NULL, &hasPmat);CHKERRQ(ierr);
  if (hasPmat) {
    Mat        pmat;
    PetscTruth isSchur;

    ierr = PCGetOperators(pc, PETSC_NULL, &pmat, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject) pmat, MATSCHURCOMPLEMENT, &isSchur);CHKERRQ(ierr);
    if (isSchur) {
      ierr = MatSchurComplementGetSubmatrices(pmat, &ctx->K, PETSC_NULL, &ctx->G, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
    }
  }
  if (ctx->K == PETSC_NULL) {SETERRQ(PETSC_ERR_SUP, "bfbt: K matrix not set");}
  if (ctx->G == PETSC_NULL) {SETERRQ(PETSC_ERR_SUP, "bfbt: G matrix not set");}

  /* Check for existence of objects and trash any which exist */
  if (ctx->form_GtG && ctx->GtG != PETSC_NULL) {
    ierr = MatDestroy(ctx->GtG);CHKERRQ(ierr);
    ctx->GtG = PETSC_NULL;
  }
  if (ctx->s != PETSC_NULL) {
    ierr = VecDestroy(ctx->s);CHKERRQ(ierr);
    ctx->s = PETSC_NULL;
  }
  if (ctx->X != PETSC_NULL) {
    ierr = VecDestroy(ctx->X);CHKERRQ(ierr);
    ctx->X = PETSC_NULL;
  }
  if (ctx->t != PETSC_NULL) {
    ierr = VecDestroy(ctx->t);CHKERRQ(ierr);
    ctx->t = PETSC_NULL;
  }
  if (ctx->inv_diag_M != PETSC_NULL) {
    ierr = VecDestroy(ctx->inv_diag_M);CHKERRQ(ierr);
    ctx->inv_diag_M = PETSC_NULL;
  }
  /* Create structures */
  ierr = MatGetVecs(ctx->K, &ctx->s, &ctx->X);CHKERRQ(ierr);
  ierr = MatGetVecs(ctx->G, &ctx->t, PETSC_NULL);CHKERRQ(ierr);
  if ((ctx->M != PETSC_NULL) || (ctx->scaled)) {
    ierr = MatGetVecs(ctx->K, &ctx->inv_diag_M, PETSC_NULL);CHKERRQ(ierr);
    if (ctx->M != PETSC_NULL) {
      ierr = MatGetDiagonal(ctx->M, ctx->inv_diag_M);CHKERRQ(ierr);
      ierr = VecReciprocal(ctx->inv_diag_M);CHKERRQ(ierr);
    } else {
      ierr = MatGetDiagonal(ctx->K, ctx->inv_diag_M);CHKERRQ(ierr);
      ierr = VecReciprocal(ctx->inv_diag_M);CHKERRQ(ierr);
    }
    /* change the pc_apply routines */
    pc->ops->apply          = PCApply_BFBt_diagonal_scaling;
    pc->ops->applytranspose = PCApplyTranspose_BFBt_diagonal_scaling;
  }
  ierr = PCBFBtCreateGtG(pc, ctx->G, ctx->inv_diag_M, &ctx->GtG);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->ksp, ctx->GtG, ctx->GtG, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_BFBt"
PetscErrorCode PCDestroy_BFBt(PC pc)
{
  PC_BFBt       *ctx = (PC_BFBt *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx == PETSC_NULL) {PetscFunctionReturn(0);}
  if (ctx->form_GtG && ctx->GtG != PETSC_NULL) {
    ierr = MatDestroy(ctx->GtG);CHKERRQ(ierr);
  }
  if (ctx->ksp        != PETSC_NULL) {ierr = KSPDestroy(ctx->ksp);CHKERRQ(ierr);}
  if (ctx->s          != PETSC_NULL) {ierr = VecDestroy(ctx->s);CHKERRQ(ierr);}
  if (ctx->X          != PETSC_NULL) {ierr = VecDestroy(ctx->X);CHKERRQ(ierr);}
  if (ctx->t          != PETSC_NULL) {ierr = VecDestroy(ctx->t);CHKERRQ(ierr);}
  if (ctx->inv_diag_M != PETSC_NULL) {ierr = VecDestroy(ctx->inv_diag_M);CHKERRQ(ierr);}
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_BFBt"
PetscErrorCode PCSetFromOptions_BFBt(PC pc)
{
  PC_BFBt       *ctx = (PC_BFBt *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("BFBt options");CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-pc_bfbt_scaled","Scale by the diagonal of K","PCBFBtSetScaled",ctx->scaled,&ctx->scaled,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ctx->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_BFBt"
PetscErrorCode PCView_BFBt(PC pc, PetscViewer viewer)
{
  PC_BFBt       *ctx = (PC_BFBt *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  if (ctx->M != PETSC_NULL) {
    ierr = PetscViewerASCIIPrintf(viewer, "bfbt: Scaled by diag(M)\n");CHKERRQ(ierr);
  } else if (ctx->scaled) {
    ierr = PetscViewerASCIIPrintf(viewer, "bfbt: Scaled by diag(K)\n");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "bfbt: Standard\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf( viewer, "bfbt-ksp\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = KSPView(ctx->ksp, viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_BFBt"
PetscErrorCode PCCreate_BFBt(PC pc)
{
  PC_BFBt       *ctx = (PC_BFBt *) pc->data;
  MPI_Comm       comm;
  const char    *prefix;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* create memory for ctx */
  ierr = PetscNew(PC_BFBt, &ctx);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc, sizeof(PC_BFBt));CHKERRQ(ierr);
  /* init ctx */
  ctx->K          = PETSC_NULL;
  ctx->G          = PETSC_NULL;
  ctx->M          = PETSC_NULL;
  ctx->inv_diag_M = PETSC_NULL;
  ctx->GtG        = PETSC_NULL;
  ctx->s          = PETSC_NULL;
  ctx->t          = PETSC_NULL;
  ctx->X          = PETSC_NULL;
  ctx->ksp        = PETSC_NULL;
  ctx->form_GtG   = PETSC_TRUE;
  /* create internals */
  ierr = PetscObjectGetComm((PetscObject) pc, &comm);CHKERRQ(ierr);
  ierr = KSPCreate(comm, &ctx->ksp);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc, &prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ctx->ksp, prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(ctx->ksp, "pc_bfbt_");CHKERRQ(ierr);
  /* set ctx onto pc */
  pc->data = (void *) ctx;
  /* define operations */
  pc->ops->setup          = PCSetUp_BFBt;
  pc->ops->view           = PCView_BFBt;
  pc->ops->destroy        = PCDestroy_BFBt;
  pc->ops->setfromoptions = PCSetFromOptions_BFBt;
  pc->ops->apply          = PCApply_BFBt;
  pc->ops->applytranspose = PCApplyTranspose_BFBt;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject) pc, "PCBFBtSetOperators_C", "PCBFBtSetOperators_BFBt", PCBFBtSetOperators_BFBt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
