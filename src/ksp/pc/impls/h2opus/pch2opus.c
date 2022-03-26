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
  PC_H2OPUS *pch2opus = (PC_H2OPUS*)pc->data;

  PetscFunctionBegin;
  pch2opus->sdim  = 0;
  pch2opus->nlocc = 0;
  PetscCall(PetscFree(pch2opus->coords));
  PetscCall(MatDestroy(&pch2opus->A));
  PetscCall(MatDestroy(&pch2opus->M));
  PetscCall(MatDestroy(&pch2opus->T));
  PetscCall(VecDestroy(&pch2opus->w));
  PetscCall(MatDestroy(&pch2opus->S));
  PetscCall(VecDestroy(&pch2opus->wns[0]));
  PetscCall(VecDestroy(&pch2opus->wns[1]));
  PetscCall(VecDestroy(&pch2opus->wns[2]));
  PetscCall(VecDestroy(&pch2opus->wns[3]));
  PetscCall(MatDestroy(&pch2opus->wnsmat[0]));
  PetscCall(MatDestroy(&pch2opus->wnsmat[1]));
  PetscCall(MatDestroy(&pch2opus->wnsmat[2]));
  PetscCall(MatDestroy(&pch2opus->wnsmat[3]));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetCoordinates_H2OPUS(PC pc, PetscInt sdim, PetscInt nlocc, PetscReal *coords)
{
  PC_H2OPUS *pch2opus = (PC_H2OPUS*)pc->data;
  PetscBool  reset    = PETSC_TRUE;

  PetscFunctionBegin;
  if (pch2opus->sdim && sdim == pch2opus->sdim && nlocc == pch2opus->nlocc) {
    PetscCall(PetscArraycmp(pch2opus->coords,coords,sdim*nlocc,&reset));
    reset = (PetscBool)!reset;
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE,&reset,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
  if (reset) {
    PetscCall(PCReset_H2OPUS(pc));
    PetscCall(PetscMalloc1(sdim*nlocc,&pch2opus->coords));
    PetscCall(PetscArraycpy(pch2opus->coords,coords,sdim*nlocc));
    pch2opus->sdim  = sdim;
    pch2opus->nlocc = nlocc;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_H2OPUS(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCReset_H2OPUS(pc));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_H2OPUS(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_H2OPUS *pch2opus = (PC_H2OPUS*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"H2OPUS options"));
  PetscCall(PetscOptionsInt("-pc_h2opus_maxits","Maximum number of iterations for Newton-Schultz",NULL,pch2opus->maxits,&pch2opus->maxits,NULL));
  PetscCall(PetscOptionsBool("-pc_h2opus_monitor","Monitor Newton-Schultz convergence",NULL,pch2opus->monitor,&pch2opus->monitor,NULL));
  PetscCall(PetscOptionsReal("-pc_h2opus_atol","Absolute tolerance",NULL,pch2opus->atol,&pch2opus->atol,NULL));
  PetscCall(PetscOptionsReal("-pc_h2opus_rtol","Relative tolerance",NULL,pch2opus->rtol,&pch2opus->rtol,NULL));
  PetscCall(PetscOptionsEnum("-pc_h2opus_norm_type","Norm type for convergence monitoring",NULL,NormTypes,(PetscEnum)pch2opus->normtype,(PetscEnum*)&pch2opus->normtype,NULL));
  PetscCall(PetscOptionsInt("-pc_h2opus_hyperorder","Hyper power order of sampling",NULL,pch2opus->hyperorder,&pch2opus->hyperorder,NULL));
  PetscCall(PetscOptionsInt("-pc_h2opus_leafsize","Leaf size when constructed from kernel",NULL,pch2opus->leafsize,&pch2opus->leafsize,NULL));
  PetscCall(PetscOptionsReal("-pc_h2opus_eta","Admissibility condition tolerance",NULL,pch2opus->eta,&pch2opus->eta,NULL));
  PetscCall(PetscOptionsInt("-pc_h2opus_maxrank","Maximum rank when constructed from matvecs",NULL,pch2opus->max_rank,&pch2opus->max_rank,NULL));
  PetscCall(PetscOptionsInt("-pc_h2opus_samples","Number of samples to be taken concurrently when constructing from matvecs",NULL,pch2opus->bs,&pch2opus->bs,NULL));
  PetscCall(PetscOptionsReal("-pc_h2opus_mrtol","Relative tolerance for construction from sampling",NULL,pch2opus->mrtol,&pch2opus->mrtol,NULL));
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

typedef struct {
  Mat A;
  Mat M;
  Vec w;
} AAtCtx;

static PetscErrorCode MatMult_AAt(Mat A, Vec x, Vec y)
{
  AAtCtx *aat;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,(void*)&aat));
  /* PetscCall(MatMultTranspose(aat->M,x,aat->w)); */
  PetscCall(MatMultTranspose(aat->A,x,aat->w));
  PetscCall(MatMult(aat->A,aat->w,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCH2OpusSetUpInit(PC pc)
{
  PC_H2OPUS *pch2opus = (PC_H2OPUS*)pc->data;
  Mat        A        = pc->useAmat ? pc->mat : pc->pmat, AAt;
  PetscInt   M,m;
  VecType    vtype;
  PetscReal  n;
  AAtCtx     aat;

  PetscFunctionBegin;
  aat.A = A;
  aat.M = pch2opus->M; /* unused so far */
  PetscCall(MatCreateVecs(A,NULL,&aat.w));
  PetscCall(MatGetSize(A,&M,NULL));
  PetscCall(MatGetLocalSize(A,&m,NULL));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)A),m,m,M,M,&aat,&AAt));
  PetscCall(MatBindToCPU(AAt,pch2opus->boundtocpu));
  PetscCall(MatShellSetOperation(AAt,MATOP_MULT,(void (*)(void))MatMult_AAt));
  PetscCall(MatShellSetOperation(AAt,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMult_AAt));
  PetscCall(MatShellSetOperation(AAt,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS));
  PetscCall(MatGetVecType(A,&vtype));
  PetscCall(MatShellSetVecType(AAt,vtype));
  PetscCall(MatNorm(AAt,NORM_1,&n));
  pch2opus->s0 = 1./n;
  PetscCall(VecDestroy(&aat.w));
  PetscCall(MatDestroy(&AAt));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyKernel_H2OPUS(PC pc, Vec x, Vec y, PetscBool t)
{
  PC_H2OPUS *pch2opus = (PC_H2OPUS*)pc->data;

  PetscFunctionBegin;
  if (t) PetscCall(MatMultTranspose(pch2opus->M,x,y));
  else  PetscCall(MatMult(pch2opus->M,x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyMatKernel_H2OPUS(PC pc, Mat X, Mat Y, PetscBool t)
{
  PC_H2OPUS *pch2opus = (PC_H2OPUS*)pc->data;

  PetscFunctionBegin;
  if (t) PetscCall(MatTransposeMatMult(pch2opus->M,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y));
  else   PetscCall(MatMatMult(pch2opus->M,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyMat_H2OPUS(PC pc, Mat X, Mat Y)
{
  PetscFunctionBegin;
  PetscCall(PCApplyMatKernel_H2OPUS(pc,X,Y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTransposeMat_H2OPUS(PC pc, Mat X, Mat Y)
{
  PetscFunctionBegin;
  PetscCall(PCApplyMatKernel_H2OPUS(pc,X,Y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_H2OPUS(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(PCApplyKernel_H2OPUS(pc,x,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_H2OPUS(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(PCApplyKernel_H2OPUS(pc,x,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/* used to test the norm of (M^-1 A - I) */
static PetscErrorCode MatMultKernel_MAmI(Mat M, Vec x, Vec y, PetscBool t)
{
  PC         pc;
  Mat        A;
  PC_H2OPUS *pch2opus;
  PetscBool  sideleft = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(M,(void*)&pc));
  pch2opus = (PC_H2OPUS*)pc->data;
  if (!pch2opus->w) PetscCall(MatCreateVecs(pch2opus->M,&pch2opus->w,NULL));
  A = pch2opus->A;
  PetscCall(VecBindToCPU(pch2opus->w,pch2opus->boundtocpu));
  if (t) {
    if (sideleft) {
      PetscCall(PCApplyTranspose_H2OPUS(pc,x,pch2opus->w));
      PetscCall(MatMultTranspose(A,pch2opus->w,y));
    } else {
      PetscCall(MatMultTranspose(A,x,pch2opus->w));
      PetscCall(PCApplyTranspose_H2OPUS(pc,pch2opus->w,y));
    }
  } else {
    if (sideleft) {
      PetscCall(MatMult(A,x,pch2opus->w));
      PetscCall(PCApply_H2OPUS(pc,pch2opus->w,y));
    } else {
      PetscCall(PCApply_H2OPUS(pc,x,pch2opus->w));
      PetscCall(MatMult(A,pch2opus->w,y));
    }
  }
  if (!pch2opus->testMA) PetscCall(VecAXPY(y,-1.0,x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_MAmI(Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatMultKernel_MAmI(A,x,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_MAmI(Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatMultKernel_MAmI(A,x,y,PETSC_TRUE));
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
  PC         pc;
  Mat        A;
  PC_H2OPUS *pch2opus;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(M,(void*)&pc));
  pch2opus = (PC_H2OPUS*)pc->data;
  A = pch2opus->A;
  PetscCall(MatCreateVecs(pch2opus->M,pch2opus->wns[0] ? NULL : &pch2opus->wns[0],pch2opus->wns[1] ? NULL : &pch2opus->wns[1]));
  PetscCall(MatCreateVecs(pch2opus->M,pch2opus->wns[2] ? NULL : &pch2opus->wns[2],pch2opus->wns[3] ? NULL : &pch2opus->wns[3]));
  PetscCall(VecBindToCPU(pch2opus->wns[0],pch2opus->boundtocpu));
  PetscCall(VecBindToCPU(pch2opus->wns[1],pch2opus->boundtocpu));
  PetscCall(VecBindToCPU(pch2opus->wns[2],pch2opus->boundtocpu));
  PetscCall(VecBindToCPU(pch2opus->wns[3],pch2opus->boundtocpu));
  PetscCall(VecCopy(x,pch2opus->wns[0]));
  PetscCall(VecCopy(x,pch2opus->wns[3]));
  if (t) {
    for (PetscInt i=0;i<pch2opus->hyperorder-1;i++) {
      PetscCall(MatMultTranspose(A,pch2opus->wns[0],pch2opus->wns[1]));
      PetscCall(PCApplyTranspose_H2OPUS(pc,pch2opus->wns[1],pch2opus->wns[2]));
      PetscCall(VecAXPY(pch2opus->wns[0],-1.,pch2opus->wns[2]));
      PetscCall(VecAXPY(pch2opus->wns[3], 1.,pch2opus->wns[0]));
    }
    PetscCall(PCApplyTranspose_H2OPUS(pc,pch2opus->wns[3],y));
  } else {
    for (PetscInt i=0;i<pch2opus->hyperorder-1;i++) {
      PetscCall(PCApply_H2OPUS(pc,pch2opus->wns[0],pch2opus->wns[1]));
      PetscCall(MatMult(A,pch2opus->wns[1],pch2opus->wns[2]));
      PetscCall(VecAXPY(pch2opus->wns[0],-1.,pch2opus->wns[2]));
      PetscCall(VecAXPY(pch2opus->wns[3], 1.,pch2opus->wns[0]));
    }
    PetscCall(PCApply_H2OPUS(pc,pch2opus->wns[3],y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Hyper(Mat M, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatMultKernel_Hyper(M,x,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Hyper(Mat M, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatMultKernel_Hyper(M,x,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/* Hyper power kernel, MatMat version */
static PetscErrorCode MatMatMultKernel_Hyper(Mat M, Mat X, Mat Y, PetscBool t)
{
  PC         pc;
  Mat        A;
  PC_H2OPUS *pch2opus;
  PetscInt   i;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(M,(void*)&pc));
  pch2opus = (PC_H2OPUS*)pc->data;
  A = pch2opus->A;
  if (pch2opus->wnsmat[0] && pch2opus->wnsmat[0]->cmap->N != X->cmap->N) {
    PetscCall(MatDestroy(&pch2opus->wnsmat[0]));
    PetscCall(MatDestroy(&pch2opus->wnsmat[1]));
  }
  if (!pch2opus->wnsmat[0]) {
    PetscCall(MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[0]));
    PetscCall(MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[1]));
  }
  if (pch2opus->wnsmat[2] && pch2opus->wnsmat[2]->cmap->N != X->cmap->N) {
    PetscCall(MatDestroy(&pch2opus->wnsmat[2]));
    PetscCall(MatDestroy(&pch2opus->wnsmat[3]));
  }
  if (!pch2opus->wnsmat[2]) {
    PetscCall(MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[2]));
    PetscCall(MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[3]));
  }
  PetscCall(MatCopy(X,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN));
  PetscCall(MatCopy(X,pch2opus->wnsmat[3],SAME_NONZERO_PATTERN));
  if (t) {
    for (i=0;i<pch2opus->hyperorder-1;i++) {
      PetscCall(MatTransposeMatMult(A,pch2opus->wnsmat[0],MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[1]));
      PetscCall(PCApplyTransposeMat_H2OPUS(pc,pch2opus->wnsmat[1],pch2opus->wnsmat[2]));
      PetscCall(MatAXPY(pch2opus->wnsmat[0],-1.,pch2opus->wnsmat[2],SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(pch2opus->wnsmat[3],1.,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN));
    }
    PetscCall(PCApplyTransposeMat_H2OPUS(pc,pch2opus->wnsmat[3],Y));
  } else {
    for (i=0;i<pch2opus->hyperorder-1;i++) {
      PetscCall(PCApplyMat_H2OPUS(pc,pch2opus->wnsmat[0],pch2opus->wnsmat[1]));
      PetscCall(MatMatMult(A,pch2opus->wnsmat[1],MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[2]));
      PetscCall(MatAXPY(pch2opus->wnsmat[0],-1.,pch2opus->wnsmat[2],SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(pch2opus->wnsmat[3],1.,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN));
    }
    PetscCall(PCApplyMat_H2OPUS(pc,pch2opus->wnsmat[3],Y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_Hyper(Mat M, Mat X, Mat Y,void *ctx)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultKernel_Hyper(M,X,Y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/* Basic Newton-Schultz sampler: (2 * I - M * A)*M */
static PetscErrorCode MatMultKernel_NS(Mat M, Vec x, Vec y, PetscBool t)
{
  PC         pc;
  Mat        A;
  PC_H2OPUS *pch2opus;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(M,(void*)&pc));
  pch2opus = (PC_H2OPUS*)pc->data;
  A = pch2opus->A;
  PetscCall(MatCreateVecs(pch2opus->M,pch2opus->wns[0] ? NULL : &pch2opus->wns[0],pch2opus->wns[1] ? NULL : &pch2opus->wns[1]));
  PetscCall(VecBindToCPU(pch2opus->wns[0],pch2opus->boundtocpu));
  PetscCall(VecBindToCPU(pch2opus->wns[1],pch2opus->boundtocpu));
  if (t) {
    PetscCall(PCApplyTranspose_H2OPUS(pc,x,y));
    PetscCall(MatMultTranspose(A,y,pch2opus->wns[1]));
    PetscCall(PCApplyTranspose_H2OPUS(pc,pch2opus->wns[1],pch2opus->wns[0]));
    PetscCall(VecAXPBY(y,-1.,2.,pch2opus->wns[0]));
  } else {
    PetscCall(PCApply_H2OPUS(pc,x,y));
    PetscCall(MatMult(A,y,pch2opus->wns[0]));
    PetscCall(PCApply_H2OPUS(pc,pch2opus->wns[0],pch2opus->wns[1]));
    PetscCall(VecAXPBY(y,-1.,2.,pch2opus->wns[1]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_NS(Mat M, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatMultKernel_NS(M,x,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_NS(Mat M, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatMultKernel_NS(M,x,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/* Basic Newton-Schultz sampler: (2 * I - M * A)*M, MatMat version */
static PetscErrorCode MatMatMultKernel_NS(Mat M, Mat X, Mat Y, PetscBool t)
{
  PC         pc;
  Mat        A;
  PC_H2OPUS *pch2opus;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(M,(void*)&pc));
  pch2opus = (PC_H2OPUS*)pc->data;
  A = pch2opus->A;
  if (pch2opus->wnsmat[0] && pch2opus->wnsmat[0]->cmap->N != X->cmap->N) {
    PetscCall(MatDestroy(&pch2opus->wnsmat[0]));
    PetscCall(MatDestroy(&pch2opus->wnsmat[1]));
  }
  if (!pch2opus->wnsmat[0]) {
    PetscCall(MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[0]));
    PetscCall(MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pch2opus->wnsmat[1]));
  }
  if (t) {
    PetscCall(PCApplyTransposeMat_H2OPUS(pc,X,Y));
    PetscCall(MatTransposeMatMult(A,Y,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[1]));
    PetscCall(PCApplyTransposeMat_H2OPUS(pc,pch2opus->wnsmat[1],pch2opus->wnsmat[0]));
    PetscCall(MatScale(Y,2.));
    PetscCall(MatAXPY(Y,-1.,pch2opus->wnsmat[0],SAME_NONZERO_PATTERN));
  } else {
    PetscCall(PCApplyMat_H2OPUS(pc,X,Y));
    PetscCall(MatMatMult(A,Y,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pch2opus->wnsmat[0]));
    PetscCall(PCApplyMat_H2OPUS(pc,pch2opus->wnsmat[0],pch2opus->wnsmat[1]));
    PetscCall(MatScale(Y,2.));
    PetscCall(MatAXPY(Y,-1.,pch2opus->wnsmat[1],SAME_NONZERO_PATTERN));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_NS(Mat M, Mat X, Mat Y, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultKernel_NS(M,X,Y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCH2OpusSetUpSampler_Private(PC pc)
{
  PC_H2OPUS *pch2opus = (PC_H2OPUS*)pc->data;
  Mat        A        = pc->useAmat ? pc->mat : pc->pmat;

  PetscFunctionBegin;
  if (!pch2opus->S) {
    PetscInt M,N,m,n;

    PetscCall(MatGetSize(A,&M,&N));
    PetscCall(MatGetLocalSize(A,&m,&n));
    PetscCall(MatCreateShell(PetscObjectComm((PetscObject)A),m,n,M,N,pc,&pch2opus->S));
    PetscCall(MatSetBlockSizesFromMats(pch2opus->S,A,A));
#if defined(PETSC_H2OPUS_USE_GPU)
    PetscCall(MatShellSetVecType(pch2opus->S,VECCUDA));
#endif
  }
  if (pch2opus->hyperorder >= 2) {
    PetscCall(MatShellSetOperation(pch2opus->S,MATOP_MULT,(void (*)(void))MatMult_Hyper));
    PetscCall(MatShellSetOperation(pch2opus->S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_Hyper));
    PetscCall(MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_Hyper,NULL,MATDENSE,MATDENSE));
    PetscCall(MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_Hyper,NULL,MATDENSECUDA,MATDENSECUDA));
  } else {
    PetscCall(MatShellSetOperation(pch2opus->S,MATOP_MULT,(void (*)(void))MatMult_NS));
    PetscCall(MatShellSetOperation(pch2opus->S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_NS));
    PetscCall(MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_NS,NULL,MATDENSE,MATDENSE));
    PetscCall(MatShellSetMatProductOperation(pch2opus->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_NS,NULL,MATDENSECUDA,MATDENSECUDA));
  }
  PetscCall(MatPropagateSymmetryOptions(A,pch2opus->S));
  PetscCall(MatBindToCPU(pch2opus->S,pch2opus->boundtocpu));
  /* XXX */
  PetscCall(MatSetOption(pch2opus->S,MAT_SYMMETRIC,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_H2OPUS(PC pc)
{
  PC_H2OPUS *pch2opus  = (PC_H2OPUS*)pc->data;
  Mat        A         = pc->useAmat ? pc->mat : pc->pmat;
  NormType   norm      = pch2opus->normtype;
  PetscReal  initerr   = 0.0,err;
  PetscReal  initerrMA = 0.0,errMA;
  PetscBool  ish2opus;

  PetscFunctionBegin;
  if (!pch2opus->T) {
    PetscInt    M,N,m,n;
    const char *prefix;

    PetscCall(PCGetOptionsPrefix(pc,&prefix));
    PetscCall(MatGetSize(A,&M,&N));
    PetscCall(MatGetLocalSize(A,&m,&n));
    PetscCall(MatCreateShell(PetscObjectComm((PetscObject)pc->pmat),m,n,M,N,pc,&pch2opus->T));
    PetscCall(MatSetBlockSizesFromMats(pch2opus->T,A,A));
    PetscCall(MatShellSetOperation(pch2opus->T,MATOP_MULT,(void (*)(void))MatMult_MAmI));
    PetscCall(MatShellSetOperation(pch2opus->T,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_MAmI));
    PetscCall(MatShellSetOperation(pch2opus->T,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS));
#if defined(PETSC_H2OPUS_USE_GPU)
    PetscCall(MatShellSetVecType(pch2opus->T,VECCUDA));
#endif
    PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)pch2opus->T));
    PetscCall(MatSetOptionsPrefix(pch2opus->T,prefix));
    PetscCall(MatAppendOptionsPrefix(pch2opus->T,"pc_h2opus_est_"));
  }
  PetscCall(MatDestroy(&pch2opus->A));
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus));
  if (ish2opus) {
    PetscCall(PetscObjectReference((PetscObject)A));
    pch2opus->A = A;
  } else {
    const char *prefix;
    PetscCall(MatCreateH2OpusFromMat(A,pch2opus->sdim,pch2opus->coords,PETSC_FALSE,pch2opus->eta,pch2opus->leafsize,pch2opus->max_rank,pch2opus->bs,pch2opus->mrtol,&pch2opus->A));
    /* XXX */
    PetscCall(MatSetOption(pch2opus->A,MAT_SYMMETRIC,PETSC_TRUE));
    PetscCall(PCGetOptionsPrefix(pc,&prefix));
    PetscCall(MatSetOptionsPrefix(pch2opus->A,prefix));
    PetscCall(MatAppendOptionsPrefix(pch2opus->A,"pc_h2opus_init_"));
    PetscCall(MatSetFromOptions(pch2opus->A));
    PetscCall(MatAssemblyBegin(pch2opus->A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(pch2opus->A,MAT_FINAL_ASSEMBLY));
    /* XXX */
    PetscCall(MatSetOption(pch2opus->A,MAT_SYMMETRIC,PETSC_TRUE));
  }
#if defined(PETSC_H2OPUS_USE_GPU)
  pch2opus->boundtocpu = pch2opus->A->boundtocpu;
#endif
  PetscCall(MatBindToCPU(pch2opus->T,pch2opus->boundtocpu));
  if (pch2opus->M) { /* see if we can reuse M as initial guess */
    PetscReal reuse;

    PetscCall(MatBindToCPU(pch2opus->M,pch2opus->boundtocpu));
    PetscCall(MatNorm(pch2opus->T,norm,&reuse));
    if (reuse >= 1.0) PetscCall(MatDestroy(&pch2opus->M));
  }
  if (!pch2opus->M) {
    const char *prefix;
    PetscCall(MatDuplicate(pch2opus->A,MAT_COPY_VALUES,&pch2opus->M));
    PetscCall(PCGetOptionsPrefix(pc,&prefix));
    PetscCall(MatSetOptionsPrefix(pch2opus->M,prefix));
    PetscCall(MatAppendOptionsPrefix(pch2opus->M,"pc_h2opus_inv_"));
    PetscCall(MatSetFromOptions(pch2opus->M));
    PetscCall(PCH2OpusSetUpInit(pc));
    PetscCall(MatScale(pch2opus->M,pch2opus->s0));
  }
  /* A and M have the same h2 matrix structure, save on reordering routines */
  PetscCall(MatH2OpusSetNativeMult(pch2opus->A,PETSC_TRUE));
  PetscCall(MatH2OpusSetNativeMult(pch2opus->M,PETSC_TRUE));
  if (norm == NORM_1 || norm == NORM_2 || norm == NORM_INFINITY) {
    PetscCall(MatNorm(pch2opus->T,norm,&initerr));
    pch2opus->testMA = PETSC_TRUE;
    PetscCall(MatNorm(pch2opus->T,norm,&initerrMA));
    pch2opus->testMA = PETSC_FALSE;
  }
  if (PetscIsInfOrNanReal(initerr)) pc->failedreason = PC_SETUP_ERROR;
  err   = initerr;
  errMA = initerrMA;
  if (pch2opus->monitor) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: ||M*A - I|| NORM%s abs %g rel %g\n",0,NormTypes[norm],(double)err,(double)(err/initerr)));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: ||M*A||     NORM%s abs %g rel %g\n",0,NormTypes[norm],(double)errMA,(double)(errMA/initerrMA)));
  }
  if (initerr > pch2opus->atol && !pc->failedreason) {
    PetscInt i;

    PetscCall(PCH2OpusSetUpSampler_Private(pc));
    for (i = 0; i < pch2opus->maxits; i++) {
      Mat         M;
      const char* prefix;

      PetscCall(MatDuplicate(pch2opus->M,MAT_SHARE_NONZERO_PATTERN,&M));
      PetscCall(MatGetOptionsPrefix(pch2opus->M,&prefix));
      PetscCall(MatSetOptionsPrefix(M,prefix));
      PetscCall(MatH2OpusSetSamplingMat(M,pch2opus->S,PETSC_DECIDE,PETSC_DECIDE));
      PetscCall(MatSetFromOptions(M));
      PetscCall(MatH2OpusSetNativeMult(M,PETSC_TRUE));
      PetscCall(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));

      PetscCall(MatDestroy(&pch2opus->M));
      pch2opus->M = M;
      if (norm == NORM_1 || norm == NORM_2 || norm == NORM_INFINITY) {
        PetscCall(MatNorm(pch2opus->T,norm,&err));
        pch2opus->testMA = PETSC_TRUE;
        PetscCall(MatNorm(pch2opus->T,norm,&errMA));
        pch2opus->testMA = PETSC_FALSE;
      }
      if (pch2opus->monitor) {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: ||M*A - I|| NORM%s abs %g rel %g\n",i+1,NormTypes[norm],(double)err,(double)(err/initerr)));
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: ||M*A||     NORM%s abs %g rel %g\n",i+1,NormTypes[norm],(double)errMA,(double)(errMA/initerrMA)));
      }
      if (PetscIsInfOrNanReal(err)) pc->failedreason = PC_SETUP_ERROR;
      if (err < pch2opus->atol || err < pch2opus->rtol*initerr || pc->failedreason) break;
    }
  }
  /* cleanup setup workspace */
  PetscCall(MatH2OpusSetNativeMult(pch2opus->A,PETSC_FALSE));
  PetscCall(MatH2OpusSetNativeMult(pch2opus->M,PETSC_FALSE));
  PetscCall(VecDestroy(&pch2opus->wns[0]));
  PetscCall(VecDestroy(&pch2opus->wns[1]));
  PetscCall(VecDestroy(&pch2opus->wns[2]));
  PetscCall(VecDestroy(&pch2opus->wns[3]));
  PetscCall(MatDestroy(&pch2opus->wnsmat[0]));
  PetscCall(MatDestroy(&pch2opus->wnsmat[1]));
  PetscCall(MatDestroy(&pch2opus->wnsmat[2]));
  PetscCall(MatDestroy(&pch2opus->wnsmat[3]));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_H2OPUS(PC pc, PetscViewer viewer)
{
  PC_H2OPUS *pch2opus = (PC_H2OPUS*)pc->data;
  PetscBool  isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    if (pch2opus->A && pch2opus->A != pc->mat && pch2opus->A != pc->pmat) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Initial approximation matrix\n"));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
      PetscCall(MatView(pch2opus->A,viewer));
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    if (pch2opus->M) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Inner matrix constructed\n"));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
      PetscCall(MatView(pch2opus->M,viewer));
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"Initial scaling: %g\n",pch2opus->s0));
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PCCreate_H2OPUS(PC pc)
{
  PC_H2OPUS *pch2opus;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&pch2opus));
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

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_H2OPUS));
  PetscFunctionReturn(0);
}
