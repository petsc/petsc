#include <petsc/private/pcimpl.h>
#include <petsc/private/matimpl.h>
#include <h2opusconf.h>

/* Use GPU only if H2OPUS is configured for GPU */
#if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
#define PETSC_H2OPUS_USE_GPU
#endif

typedef struct {
  Mat         A;
  Mat         M;
  PetscScalar s0;

  /* sampler for Newton-Schultz */
  Mat      S;
  PetscInt hyperorder;
  Vec      wns[4];
  Mat      wnsmat[4];

  /* convergence testing */
  Mat       T;
  Vec       w;
  PetscBool testMA;

  /* Support for PCSetCoordinates */
  PetscInt  sdim;
  PetscInt  nlocc;
  PetscReal *coords;

  /* Newton-Schultz customization */
  PetscInt  maxits;
  PetscReal rtol,atol;
  PetscBool monitor;
  PetscBool useapproximatenorms;
  NormType  normtype;

  /* Used when pmat != MATH2OPUS */
  PetscReal eta;
  PetscInt  leafsize;
  PetscInt  max_rank;
  PetscInt  bs;
  PetscReal mrtol;

  PetscBool boundtocpu;
} PC_H2OPUS;

PETSC_EXTERN PetscErrorCode MatNorm_H2OPUS(Mat,NormType,PetscReal*);

static PetscErrorCode PCReset_H2OPUS(PC pc)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pch2opus->sdim  = 0;
  pch2opus->nlocc = 0;
  ierr = PetscFree(pch2opus->coords);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->A);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->M);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->T);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->w);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->S);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[2]);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[3]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[2]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[3]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetCoordinates_H2OPUS(PC pc, PetscInt sdim, PetscInt nlocc, PetscReal *coords)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscBool      reset = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pch2opus->sdim && sdim == pch2opus->sdim && nlocc == pch2opus->nlocc) {
    ierr  = PetscArraycmp(pch2opus->coords,coords,sdim*nlocc,&reset);CHKERRQ(ierr);
    reset = (PetscBool)!reset;
  }
  ierr = MPIU_Allreduce(MPI_IN_PLACE,&reset,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc));CHKERRMPI(ierr);
  if (reset) {
    ierr = PCReset_H2OPUS(pc);CHKERRQ(ierr);
    ierr = PetscMalloc1(sdim*nlocc,&pch2opus->coords);CHKERRQ(ierr);
    ierr = PetscArraycpy(pch2opus->coords,coords,sdim*nlocc);CHKERRQ(ierr);
    pch2opus->sdim  = sdim;
    pch2opus->nlocc = nlocc;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_H2OPUS(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_H2OPUS(pc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_H2OPUS(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"H2OPUS options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_h2opus_maxits","Maximum number of iterations for Newton-Schultz",NULL,pch2opus->maxits,&pch2opus->maxits,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_h2opus_monitor","Monitor Newton-Schultz convergence",NULL,pch2opus->monitor,&pch2opus->monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_h2opus_atol","Absolute tolerance",NULL,pch2opus->atol,&pch2opus->atol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_h2opus_rtol","Relative tolerance",NULL,pch2opus->rtol,&pch2opus->rtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_h2opus_norm_type","Norm type for convergence monitoring",NULL,NormTypes,(PetscEnum)pch2opus->normtype,(PetscEnum*)&pch2opus->normtype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_h2opus_hyperorder","Hyper power order of sampling",NULL,pch2opus->hyperorder,&pch2opus->hyperorder,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_h2opus_leafsize","Leaf size when constructed from kernel",NULL,pch2opus->leafsize,&pch2opus->leafsize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_h2opus_eta","Admissibility condition tolerance",NULL,pch2opus->eta,&pch2opus->eta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_h2opus_maxrank","Maximum rank when constructed from matvecs",NULL,pch2opus->max_rank,&pch2opus->max_rank,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_h2opus_samples","Number of samples to be taken concurrently when constructing from matvecs",NULL,pch2opus->bs,&pch2opus->bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_h2opus_mrtol","Relative tolerance for construction from sampling",NULL,pch2opus->mrtol,&pch2opus->mrtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  Mat A;
  Mat M;
  Vec w;
} AAtCtx;

static PetscErrorCode MatMult_AAt(Mat A, Vec x, Vec y)
{
  AAtCtx         *aat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void*)&aat);CHKERRQ(ierr);
  /* ierr = MatMultTranspose(aat->M,x,aat->w);CHKERRQ(ierr); */
  ierr = MatMultTranspose(aat->A,x,aat->w);CHKERRQ(ierr);
  ierr = MatMult(aat->A,aat->w,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCH2OpusSetUpInit(PC pc)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  Mat            A = pc->useAmat ? pc->mat : pc->pmat, AAt;
  PetscInt       M,m;
  VecType        vtype;
  PetscReal      n;
  AAtCtx         aat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  aat.A = A;
  aat.M = pch2opus->M; /* unused so far */
  ierr = MatCreateVecs(A,NULL,&aat.w);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)A),m,m,M,M,&aat,&AAt);CHKERRQ(ierr);
  ierr = MatBindToCPU(AAt,pch2opus->boundtocpu);CHKERRQ(ierr);
  ierr = MatShellSetOperation(AAt,MATOP_MULT,(void (*)(void))MatMult_AAt);CHKERRQ(ierr);
  ierr = MatShellSetOperation(AAt,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMult_AAt);CHKERRQ(ierr);
  ierr = MatShellSetOperation(AAt,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS);CHKERRQ(ierr);
  ierr = MatGetVecType(A,&vtype);CHKERRQ(ierr);
  ierr = MatShellSetVecType(AAt,vtype);CHKERRQ(ierr);
  ierr = MatNorm(AAt,NORM_1,&n);CHKERRQ(ierr);
  pch2opus->s0 = 1./n;
  ierr = VecDestroy(&aat.w);CHKERRQ(ierr);
  ierr = MatDestroy(&AAt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyKernel_H2OPUS(PC pc, Vec x, Vec y, PetscBool t)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (t) {
    ierr = MatMultTranspose(pch2opus->M,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatMult(pch2opus->M,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyMatKernel_H2OPUS(PC pc, Mat X, Mat Y, PetscBool t)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (t) {
    ierr = MatTransposeMatMult(pch2opus->M,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
  } else {
    ierr = MatMatMult(pch2opus->M,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyMat_H2OPUS(PC pc, Mat X, Mat Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyMatKernel_H2OPUS(pc,X,Y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTransposeMat_H2OPUS(PC pc, Mat X, Mat Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyMatKernel_H2OPUS(pc,X,Y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_H2OPUS(PC pc, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyKernel_H2OPUS(pc,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_H2OPUS(PC pc, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyKernel_H2OPUS(pc,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* used to test the norm of (M^-1 A - I) */
static PetscErrorCode MatMultKernel_MAmI(Mat M, Vec x, Vec y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_H2OPUS      *pch2opus;
  PetscErrorCode ierr;
  PetscBool      sideleft = PETSC_TRUE;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  pch2opus = (PC_H2OPUS*)pc->data;
  if (!pch2opus->w) {
    ierr = MatCreateVecs(pch2opus->M,&pch2opus->w,NULL);CHKERRQ(ierr);
  }
  A = pch2opus->A;
  ierr = VecBindToCPU(pch2opus->w,pch2opus->boundtocpu);CHKERRQ(ierr);
  if (t) {
    if (sideleft) {
      ierr = PCApplyTranspose_H2OPUS(pc,x,pch2opus->w);CHKERRQ(ierr);
      ierr = MatMultTranspose(A,pch2opus->w,y);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(A,x,pch2opus->w);CHKERRQ(ierr);
      ierr = PCApplyTranspose_H2OPUS(pc,pch2opus->w,y);CHKERRQ(ierr);
    }
  } else {
    if (sideleft) {
      ierr = MatMult(A,x,pch2opus->w);CHKERRQ(ierr);
      ierr = PCApply_H2OPUS(pc,pch2opus->w,y);CHKERRQ(ierr);
    } else {
      ierr = PCApply_H2OPUS(pc,x,pch2opus->w);CHKERRQ(ierr);
      ierr = MatMult(A,pch2opus->w,y);CHKERRQ(ierr);
    }
  }
  if (!pch2opus->testMA) {
    ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_MAmI(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_MAmI(A,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_MAmI(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_MAmI(A,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* HyperPower kernel:
Y = R = x
for i = 1 . . . l - 1 do
  R = (I - A * Xk) * R
  Y = Y + R
Y = Xk * Y
*/
static PetscErrorCode MatMultKernel_Hyper(Mat M, Vec x, Vec y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_H2OPUS      *pch2opus;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  pch2opus = (PC_H2OPUS*)pc->data;
  A = pch2opus->A;
  ierr = MatCreateVecs(pch2opus->M,pch2opus->wns[0] ? NULL : &pch2opus->wns[0],pch2opus->wns[1] ? NULL : &pch2opus->wns[1]);CHKERRQ(ierr);
  ierr = MatCreateVecs(pch2opus->M,pch2opus->wns[2] ? NULL : &pch2opus->wns[2],pch2opus->wns[3] ? NULL : &pch2opus->wns[3]);CHKERRQ(ierr);
  ierr = VecBindToCPU(pch2opus->wns[0],pch2opus->boundtocpu);CHKERRQ(ierr);
  ierr = VecBindToCPU(pch2opus->wns[1],pch2opus->boundtocpu);CHKERRQ(ierr);
  ierr = VecBindToCPU(pch2opus->wns[2],pch2opus->boundtocpu);CHKERRQ(ierr);
  ierr = VecBindToCPU(pch2opus->wns[3],pch2opus->boundtocpu);CHKERRQ(ierr);
  ierr = VecCopy(x,pch2opus->wns[0]);CHKERRQ(ierr);
  ierr = VecCopy(x,pch2opus->wns[3]);CHKERRQ(ierr);
  if (t) {
    for (i=0;i<pch2opus->hyperorder-1;i++) {
      ierr = MatMultTranspose(A,pch2opus->wns[0],pch2opus->wns[1]);CHKERRQ(ierr);
      ierr = PCApplyTranspose_H2OPUS(pc,pch2opus->wns[1],pch2opus->wns[2]);CHKERRQ(ierr);
      ierr = VecAXPY(pch2opus->wns[0],-1.,pch2opus->wns[2]);CHKERRQ(ierr);
      ierr = VecAXPY(pch2opus->wns[3], 1.,pch2opus->wns[0]);CHKERRQ(ierr);
    }
    ierr = PCApplyTranspose_H2OPUS(pc,pch2opus->wns[3],y);CHKERRQ(ierr);
  } else {
    for (i=0;i<pch2opus->hyperorder-1;i++) {
      ierr = PCApply_H2OPUS(pc,pch2opus->wns[0],pch2opus->wns[1]);CHKERRQ(ierr);
      ierr = MatMult(A,pch2opus->wns[1],pch2opus->wns[2]);CHKERRQ(ierr);
      ierr = VecAXPY(pch2opus->wns[0],-1.,pch2opus->wns[2]);CHKERRQ(ierr);
      ierr = VecAXPY(pch2opus->wns[3], 1.,pch2opus->wns[0]);CHKERRQ(ierr);
    }
    ierr = PCApply_H2OPUS(pc,pch2opus->wns[3],y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Hyper(Mat M, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_Hyper(M,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Hyper(Mat M, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_Hyper(M,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Hyper power kernel, MatMat version */
static PetscErrorCode MatMatMultKernel_Hyper(Mat M, Mat X, Mat Y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_H2OPUS      *pch2opus;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  pch2opus = (PC_H2OPUS*)pc->data;
  A = pch2opus->A;
  if (pch2opus->wnsmat[0] && pch2opus->wnsmat[0]->cmap->N != X->cmap->N) {
    ierr = MatDestroy(&pch2opus->wnsmat[0]);CHKERRQ(ierr);
    ierr = MatDestroy(&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  }
  if (!pch2opus->wnsmat[0]) {
    ierr = MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[0]);CHKERRQ(ierr);
    ierr = MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  }
  if (pch2opus->wnsmat[2] && pch2opus->wnsmat[2]->cmap->N != X->cmap->N) {
    ierr = MatDestroy(&pch2opus->wnsmat[2]);CHKERRQ(ierr);
    ierr = MatDestroy(&pch2opus->wnsmat[3]);CHKERRQ(ierr);
  }
  if (!pch2opus->wnsmat[2]) {
    ierr = MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[2]);CHKERRQ(ierr);
    ierr = MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[3]);CHKERRQ(ierr);
  }
  ierr = MatCopy(X,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatCopy(X,pch2opus->wnsmat[3],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  if (t) {
    for (i=0;i<pch2opus->hyperorder-1;i++) {
      ierr = MatTransposeMatMult(A,pch2opus->wnsmat[0],MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[1]);CHKERRQ(ierr);
      ierr = PCApplyTransposeMat_H2OPUS(pc,pch2opus->wnsmat[1],pch2opus->wnsmat[2]);CHKERRQ(ierr);
      ierr = MatAXPY(pch2opus->wnsmat[0],-1.,pch2opus->wnsmat[2],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(pch2opus->wnsmat[3],1.,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    ierr = PCApplyTransposeMat_H2OPUS(pc,pch2opus->wnsmat[3],Y);CHKERRQ(ierr);
  } else {
    for (i=0;i<pch2opus->hyperorder-1;i++) {
      ierr = PCApplyMat_H2OPUS(pc,pch2opus->wnsmat[0],pch2opus->wnsmat[1]);CHKERRQ(ierr);
      ierr = MatMatMult(A,pch2opus->wnsmat[1],MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[2]);CHKERRQ(ierr);
      ierr = MatAXPY(pch2opus->wnsmat[0],-1.,pch2opus->wnsmat[2],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(pch2opus->wnsmat[3],1.,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    ierr = PCApplyMat_H2OPUS(pc,pch2opus->wnsmat[3],Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_Hyper(Mat M, Mat X, Mat Y,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultKernel_Hyper(M,X,Y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Basic Newton-Schultz sampler: (2 * I - M * A)*M */
static PetscErrorCode MatMultKernel_NS(Mat M, Vec x, Vec y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_H2OPUS      *pch2opus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  pch2opus = (PC_H2OPUS*)pc->data;
  A = pch2opus->A;
  ierr = MatCreateVecs(pch2opus->M,pch2opus->wns[0] ? NULL : &pch2opus->wns[0],pch2opus->wns[1] ? NULL : &pch2opus->wns[1]);CHKERRQ(ierr);
  ierr = VecBindToCPU(pch2opus->wns[0],pch2opus->boundtocpu);CHKERRQ(ierr);
  ierr = VecBindToCPU(pch2opus->wns[1],pch2opus->boundtocpu);CHKERRQ(ierr);
  if (t) {
    ierr = PCApplyTranspose_H2OPUS(pc,x,y);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,y,pch2opus->wns[1]);CHKERRQ(ierr);
    ierr = PCApplyTranspose_H2OPUS(pc,pch2opus->wns[1],pch2opus->wns[0]);CHKERRQ(ierr);
    ierr = VecAXPBY(y,-1.,2.,pch2opus->wns[0]);CHKERRQ(ierr);
  } else {
    ierr = PCApply_H2OPUS(pc,x,y);CHKERRQ(ierr);
    ierr = MatMult(A,y,pch2opus->wns[0]);CHKERRQ(ierr);
    ierr = PCApply_H2OPUS(pc,pch2opus->wns[0],pch2opus->wns[1]);CHKERRQ(ierr);
    ierr = VecAXPBY(y,-1.,2.,pch2opus->wns[1]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_NS(Mat M, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_NS(M,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_NS(Mat M, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_NS(M,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Basic Newton-Schultz sampler: (2 * I - M * A)*M, MatMat version */
static PetscErrorCode MatMatMultKernel_NS(Mat M, Mat X, Mat Y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_H2OPUS      *pch2opus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  pch2opus = (PC_H2OPUS*)pc->data;
  A = pch2opus->A;
  if (pch2opus->wnsmat[0] && pch2opus->wnsmat[0]->cmap->N != X->cmap->N) {
    ierr = MatDestroy(&pch2opus->wnsmat[0]);CHKERRQ(ierr);
    ierr = MatDestroy(&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  }
  if (!pch2opus->wnsmat[0]) {
    ierr = MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[0]);CHKERRQ(ierr);
    ierr = MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  }
  if (t) {
    ierr = PCApplyTransposeMat_H2OPUS(pc,X,Y);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(A,Y,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[1]);CHKERRQ(ierr);
    ierr = PCApplyTransposeMat_H2OPUS(pc,pch2opus->wnsmat[1],pch2opus->wnsmat[0]);CHKERRQ(ierr);
    ierr = MatScale(Y,2.);CHKERRQ(ierr);
    ierr = MatAXPY(Y,-1.,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  } else {
    ierr = PCApplyMat_H2OPUS(pc,X,Y);CHKERRQ(ierr);
    ierr = MatMatMult(A,Y,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[0]);CHKERRQ(ierr);
    ierr = PCApplyMat_H2OPUS(pc,pch2opus->wnsmat[0],pch2opus->wnsmat[1]);CHKERRQ(ierr);
    ierr = MatScale(Y,2.);CHKERRQ(ierr);
    ierr = MatAXPY(Y,-1.,pch2opus->wnsmat[1],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_NS(Mat M, Mat X, Mat Y, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultKernel_NS(M,X,Y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCH2OpusSetUpSampler_Private(PC pc)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  Mat            A = pc->useAmat ? pc->mat : pc->pmat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pch2opus->S) {
    PetscInt M,N,m,n;

    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)A),m,n,M,N,pc,&pch2opus->S);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(pch2opus->S,A,A);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
    ierr = MatShellSetVecType(pch2opus->S,VECCUDA);CHKERRQ(ierr);
#endif
  }
  if (pch2opus->hyperorder >= 2) {
    ierr = MatShellSetOperation(pch2opus->S,MATOP_MULT,(void (*)(void))MatMult_Hyper);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pch2opus->S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_Hyper);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_Hyper,NULL,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_Hyper,NULL,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
  } else {
    ierr = MatShellSetOperation(pch2opus->S,MATOP_MULT,(void (*)(void))MatMult_NS);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pch2opus->S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_NS);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_NS,NULL,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_NS,NULL,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
  }
  ierr = MatPropagateSymmetryOptions(A,pch2opus->S);CHKERRQ(ierr);
  ierr = MatBindToCPU(pch2opus->S,pch2opus->boundtocpu);CHKERRQ(ierr);
  /* XXX */
  ierr = MatSetOption(pch2opus->S,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_H2OPUS(PC pc)
{
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  Mat            A = pc->useAmat ? pc->mat : pc->pmat;
  NormType       norm = pch2opus->normtype;
  PetscReal      initerr = 0.0,err;
  PetscReal      initerrMA = 0.0,errMA;
  PetscBool      ish2opus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pch2opus->T) {
    PetscInt    M,N,m,n;
    const char *prefix;

    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)pc->pmat),m,n,M,N,pc,&pch2opus->T);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(pch2opus->T,A,A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pch2opus->T,MATOP_MULT,(void (*)(void))MatMult_MAmI);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pch2opus->T,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_MAmI);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pch2opus->T,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
    ierr = MatShellSetVecType(pch2opus->T,VECCUDA);CHKERRQ(ierr);
#endif
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)pch2opus->T);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(pch2opus->T,prefix);CHKERRQ(ierr);
    ierr = MatAppendOptionsPrefix(pch2opus->T,"pc_h2opus_est_");CHKERRQ(ierr);
  }
  ierr = MatDestroy(&pch2opus->A);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (ish2opus) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    pch2opus->A = A;
  } else {
    const char *prefix;
    ierr = MatCreateH2OpusFromMat(A,pch2opus->sdim,pch2opus->coords,PETSC_FALSE,pch2opus->eta,pch2opus->leafsize,pch2opus->max_rank,pch2opus->bs,pch2opus->mrtol,&pch2opus->A);CHKERRQ(ierr);
    /* XXX */
    ierr = MatSetOption(pch2opus->A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(pch2opus->A,prefix);CHKERRQ(ierr);
    ierr = MatAppendOptionsPrefix(pch2opus->A,"pc_h2opus_init_");CHKERRQ(ierr);
    ierr = MatSetFromOptions(pch2opus->A);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(pch2opus->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pch2opus->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /* XXX */
    ierr = MatSetOption(pch2opus->A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  }
#if defined(PETSC_H2OPUS_USE_GPU)
  pch2opus->boundtocpu = pch2opus->A->boundtocpu;
#endif
  ierr = MatBindToCPU(pch2opus->T,pch2opus->boundtocpu);CHKERRQ(ierr);
  if (pch2opus->M) { /* see if we can reuse M as initial guess */
    PetscReal reuse;

    ierr = MatBindToCPU(pch2opus->M,pch2opus->boundtocpu);CHKERRQ(ierr);
    ierr = MatNorm(pch2opus->T,norm,&reuse);CHKERRQ(ierr);
    if (reuse >= 1.0) {
      ierr = MatDestroy(&pch2opus->M);CHKERRQ(ierr);
    }
  }
  if (!pch2opus->M) {
    const char *prefix;
    ierr = MatDuplicate(pch2opus->A,MAT_COPY_VALUES,&pch2opus->M);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(pch2opus->M,prefix);CHKERRQ(ierr);
    ierr = MatAppendOptionsPrefix(pch2opus->M,"pc_h2opus_inv_");CHKERRQ(ierr);
    ierr = MatSetFromOptions(pch2opus->M);CHKERRQ(ierr);
    ierr = PCH2OpusSetUpInit(pc);CHKERRQ(ierr);
    ierr = MatScale(pch2opus->M,pch2opus->s0);CHKERRQ(ierr);
  }
  /* A and M have the same h2 matrix structure, save on reordering routines */
  ierr = MatH2OpusSetNativeMult(pch2opus->A,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatH2OpusSetNativeMult(pch2opus->M,PETSC_TRUE);CHKERRQ(ierr);
  if (norm == NORM_1 || norm == NORM_2 || norm == NORM_INFINITY) {
    ierr = MatNorm(pch2opus->T,norm,&initerr);CHKERRQ(ierr);
    pch2opus->testMA = PETSC_TRUE;
    ierr = MatNorm(pch2opus->T,norm,&initerrMA);CHKERRQ(ierr);
    pch2opus->testMA = PETSC_FALSE;
  }
  if (PetscIsInfOrNanReal(initerr)) pc->failedreason = PC_SETUP_ERROR;
  err   = initerr;
  errMA = initerrMA;
  if (pch2opus->monitor) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: ||M*A - I|| NORM%s abs %g rel %g\n",0,NormTypes[norm],(double)err,(double)(err/initerr));CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: ||M*A||     NORM%s abs %g rel %g\n",0,NormTypes[norm],(double)errMA,(double)(errMA/initerrMA));CHKERRQ(ierr);
  }
  if (initerr > pch2opus->atol && !pc->failedreason) {
    PetscInt i;

    ierr = PCH2OpusSetUpSampler_Private(pc);CHKERRQ(ierr);
    for (i = 0; i < pch2opus->maxits; i++) {
      Mat         M;
      const char* prefix;

      ierr = MatDuplicate(pch2opus->M,MAT_SHARE_NONZERO_PATTERN,&M);CHKERRQ(ierr);
      ierr = MatGetOptionsPrefix(pch2opus->M,&prefix);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(M,prefix);CHKERRQ(ierr);
      ierr = MatH2OpusSetSamplingMat(M,pch2opus->S,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = MatSetFromOptions(M);CHKERRQ(ierr);
      ierr = MatH2OpusSetNativeMult(M,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      ierr = MatDestroy(&pch2opus->M);CHKERRQ(ierr);
      pch2opus->M = M;
      if (norm == NORM_1 || norm == NORM_2 || norm == NORM_INFINITY) {
        ierr = MatNorm(pch2opus->T,norm,&err);CHKERRQ(ierr);
        pch2opus->testMA = PETSC_TRUE;
        ierr = MatNorm(pch2opus->T,norm,&errMA);CHKERRQ(ierr);
        pch2opus->testMA = PETSC_FALSE;
      }
      if (pch2opus->monitor) {
        ierr = PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: ||M*A - I|| NORM%s abs %g rel %g\n",i+1,NormTypes[norm],(double)err,(double)(err/initerr));CHKERRQ(ierr);
        ierr = PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: ||M*A||     NORM%s abs %g rel %g\n",i+1,NormTypes[norm],(double)errMA,(double)(errMA/initerrMA));CHKERRQ(ierr);
      }
      if (PetscIsInfOrNanReal(err)) pc->failedreason = PC_SETUP_ERROR;
      if (err < pch2opus->atol || err < pch2opus->rtol*initerr || pc->failedreason) break;
    }
  }
  /* cleanup setup workspace */
  ierr = MatH2OpusSetNativeMult(pch2opus->A,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatH2OpusSetNativeMult(pch2opus->M,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[2]);CHKERRQ(ierr);
  ierr = VecDestroy(&pch2opus->wns[3]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[2]);CHKERRQ(ierr);
  ierr = MatDestroy(&pch2opus->wnsmat[3]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_H2OPUS(PC pc, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PC_H2OPUS      *pch2opus = (PC_H2OPUS*)pc->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (pch2opus->A && pch2opus->A != pc->mat && pch2opus->A != pc->pmat) {
      ierr = PetscViewerASCIIPrintf(viewer,"Initial approximation matrix\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
      ierr = MatView(pch2opus->A,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (pch2opus->M) {
      ierr = PetscViewerASCIIPrintf(viewer,"Inner matrix constructed\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
      ierr = MatView(pch2opus->M,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"Initial scaling: %g\n",pch2opus->s0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PCCreate_H2OPUS(PC pc)
{
  PetscErrorCode ierr;
  PC_H2OPUS      *pch2opus;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&pch2opus);CHKERRQ(ierr);
  pc->data = (void*)pch2opus;

  pch2opus->atol       = 1.e-2;
  pch2opus->rtol       = 1.e-6;
  pch2opus->maxits     = 50;
  pch2opus->hyperorder = 1; /* defaults to basic NewtonSchultz */
  pch2opus->normtype   = NORM_2;

  /* these are needed when we are sampling the pmat */
  pch2opus->eta        = PETSC_DECIDE;
  pch2opus->leafsize   = PETSC_DECIDE;
  pch2opus->max_rank   = PETSC_DECIDE;
  pch2opus->bs         = PETSC_DECIDE;
  pch2opus->mrtol      = PETSC_DECIDE;
#if defined(PETSC_H2OPUS_USE_GPU)
  pch2opus->boundtocpu = PETSC_FALSE;
#else
  pch2opus->boundtocpu = PETSC_TRUE;
#endif
  pc->ops->destroy        = PCDestroy_H2OPUS;
  pc->ops->setup          = PCSetUp_H2OPUS;
  pc->ops->apply          = PCApply_H2OPUS;
  pc->ops->matapply       = PCApplyMat_H2OPUS;
  pc->ops->applytranspose = PCApplyTranspose_H2OPUS;
  pc->ops->reset          = PCReset_H2OPUS;
  pc->ops->setfromoptions = PCSetFromOptions_H2OPUS;
  pc->ops->view           = PCView_H2OPUS;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_H2OPUS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
