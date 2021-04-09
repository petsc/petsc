#include <petsc/private/pcimpl.h>
#include <petsc/private/matimpl.h>

typedef struct {
  Mat         M;
  PetscScalar s0;

  /* sampler for Newton-Schultz */
  Mat      S;
  PetscInt hyperorder;
  Vec      wns[4];
  Mat      wnsmat[4];

  /* convergence testing */
  Mat T;
  Vec w;

  /* Support for PCSetCoordinates */
  PetscInt  sdim;
  PetscInt  nlocc;
  PetscReal *coords;

  /* Newton-Schultz customization */
  PetscInt  maxits;
  PetscReal rtol,atol;
  PetscBool monitor;
  PetscBool useapproximatenorms;
} PC_HARA;

static PetscErrorCode PCReset_HARA(PC pc)
{
  PC_HARA        *pchara = (PC_HARA*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pchara->sdim  = 0;
  pchara->nlocc = 0;
  ierr = PetscFree(pchara->coords);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->M);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->T);CHKERRQ(ierr);
  ierr = VecDestroy(&pchara->w);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->S);CHKERRQ(ierr);
  ierr = VecDestroy(&pchara->wns[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&pchara->wns[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&pchara->wns[2]);CHKERRQ(ierr);
  ierr = VecDestroy(&pchara->wns[3]);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->wnsmat[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->wnsmat[2]);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->wnsmat[3]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetCoordinates_HARA(PC pc, PetscInt sdim, PetscInt nlocc, PetscReal *coords)
{
  PC_HARA        *pchara = (PC_HARA*)pc->data;
  PetscBool      reset = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pchara->sdim && sdim == pchara->sdim && nlocc == pchara->nlocc) {
    ierr  = PetscArraycmp(pchara->coords,coords,sdim*nlocc,&reset);CHKERRQ(ierr);
    reset = (PetscBool)!reset;
  }
  ierr = MPIU_Allreduce(MPI_IN_PLACE,&reset,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc));CHKERRMPI(ierr);
  if (reset) {
    ierr = PCReset_HARA(pc);CHKERRQ(ierr);
    ierr = PetscMalloc1(sdim*nlocc,&pchara->coords);CHKERRQ(ierr);
    ierr = PetscArraycpy(pchara->coords,coords,sdim*nlocc);CHKERRQ(ierr);
    pchara->sdim  = sdim;
    pchara->nlocc = nlocc;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_HARA(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_HARA(pc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_HARA(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_HARA       *pchara = (PC_HARA*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HARA options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hara_maxits","Maximum number of iterations for Newton-Schultz",NULL,pchara->maxits,&pchara->maxits,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_hara_monitor","Monitor Newton-Schultz convergence",NULL,pchara->monitor,&pchara->monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_hara_atol","Absolute tolerance",NULL,pchara->atol,&pchara->atol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_hara_rtol","Relative tolerance",NULL,pchara->rtol,&pchara->rtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hara_hyperorder","Hyper power order of sampling",NULL,pchara->hyperorder,&pchara->hyperorder,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyKernel_HARA(PC pc, Vec x, Vec y, PetscBool t)
{
  PC_HARA        *pchara = (PC_HARA*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MatAssembled(pchara->M,&flg);CHKERRQ(ierr);
  if (flg) {
    if (t) {
      ierr = MatMultTranspose(pchara->M,x,y);CHKERRQ(ierr);
    } else {
      ierr = MatMult(pchara->M,x,y);CHKERRQ(ierr);
    }
  } else { /* Not assembled, initial approximation */
    Mat A = pc->useAmat ? pc->mat : pc->pmat;

    if (pchara->s0 < 0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong scaling");
    /* X_0 = s0 * A^T */
    if (t) {
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(A,x,y);CHKERRQ(ierr);
    }
    ierr = VecScale(y,pchara->s0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyMatKernel_HARA(PC pc, Mat X, Mat Y, PetscBool t)
{
  PC_HARA        *pchara = (PC_HARA*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MatAssembled(pchara->M,&flg);CHKERRQ(ierr);
  if (flg) {
    if (t) {
      ierr = MatTransposeMatMult(pchara->M,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
    } else {
      ierr = MatMatMult(pchara->M,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
    }
  } else { /* Not assembled, initial approximation */
    Mat A = pc->useAmat ? pc->mat : pc->pmat;

    if (pchara->s0 < 0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong scaling");
    /* X_0 = s0 * A^T */
    if (t) {
      ierr = MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
    } else {
      ierr = MatTransposeMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRQ(ierr);
    }
    ierr = MatScale(Y,pchara->s0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyMat_HARA(PC pc, Mat X, Mat Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyMatKernel_HARA(pc,X,Y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTransposeMat_HARA(PC pc, Mat X, Mat Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyMatKernel_HARA(pc,X,Y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_HARA(PC pc, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyKernel_HARA(pc,x,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_HARA(PC pc, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApplyKernel_HARA(pc,x,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* used to test norm of (M^-1 A - I) */
static PetscErrorCode MatMultKernel_MAmI(Mat M, Vec x, Vec y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_HARA        *pchara;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  pchara = (PC_HARA*)pc->data;
  if (!pchara->w) {
    ierr = MatCreateVecs(pchara->M,&pchara->w,NULL);CHKERRQ(ierr);
  }
  A = pc->useAmat ? pc->mat : pc->pmat;
  if (t) {
    ierr = PCApplyTranspose_HARA(pc,x,pchara->w);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,pchara->w,y);CHKERRQ(ierr);
  } else {
    ierr = MatMult(A,x,pchara->w);CHKERRQ(ierr);
    ierr = PCApply_HARA(pc,pchara->w,y);CHKERRQ(ierr);
  }
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
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
  R = (I - AXk)R
  Y = Y + R
Y = XkY
*/
static PetscErrorCode MatMultKernel_Hyper(Mat M, Vec x, Vec y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_HARA        *pchara;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  A = pc->useAmat ? pc->mat : pc->pmat;
  pchara = (PC_HARA*)pc->data;
  ierr = MatCreateVecs(pchara->M,pchara->wns[0] ? NULL : &pchara->wns[0],pchara->wns[1] ? NULL : &pchara->wns[1]);CHKERRQ(ierr);
  ierr = MatCreateVecs(pchara->M,pchara->wns[2] ? NULL : &pchara->wns[2],pchara->wns[3] ? NULL : &pchara->wns[3]);CHKERRQ(ierr);
  ierr = VecCopy(x,pchara->wns[0]);CHKERRQ(ierr);
  ierr = VecCopy(x,pchara->wns[3]);CHKERRQ(ierr);
  if (t) {
    for (i=0;i<pchara->hyperorder-1;i++) {
      ierr = MatMultTranspose(A,pchara->wns[0],pchara->wns[1]);CHKERRQ(ierr);
      ierr = PCApplyTranspose_HARA(pc,pchara->wns[1],pchara->wns[2]);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(pchara->wns[3],-1.,1.,1.,pchara->wns[2],pchara->wns[0]);CHKERRQ(ierr);
    }
    ierr = PCApplyTranspose_HARA(pc,pchara->wns[3],y);CHKERRQ(ierr);
  } else {
    for (i=0;i<pchara->hyperorder-1;i++) {
      ierr = PCApply_HARA(pc,pchara->wns[0],pchara->wns[1]);CHKERRQ(ierr);
      ierr = MatMult(A,pchara->wns[1],pchara->wns[2]);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(pchara->wns[3],-1.,1.,1.,pchara->wns[2],pchara->wns[0]);CHKERRQ(ierr);
    }
    ierr = PCApply_HARA(pc,pchara->wns[3],y);CHKERRQ(ierr);
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
  PC_HARA        *pchara;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  A = pc->useAmat ? pc->mat : pc->pmat;
  pchara = (PC_HARA*)pc->data;
  ierr = MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pchara->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pchara->wnsmat[1]);CHKERRQ(ierr);
  ierr = MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pchara->wnsmat[2]);CHKERRQ(ierr);
  ierr = MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pchara->wnsmat[3]);CHKERRQ(ierr);
  ierr = MatCopy(X,pchara->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatCopy(X,pchara->wnsmat[3],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  if (t) {
    for (i=0;i<pchara->hyperorder-1;i++) {
      ierr = MatTransposeMatMult(A,pchara->wnsmat[0],MAT_REUSE_MATRIX,PETSC_DEFAULT,&pchara->wnsmat[1]);CHKERRQ(ierr);
      ierr = PCApplyTransposeMat_HARA(pc,pchara->wnsmat[1],pchara->wnsmat[2]);CHKERRQ(ierr);
      ierr = MatAXPY(pchara->wnsmat[0],-1.,pchara->wnsmat[2],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(pchara->wnsmat[3],1.,pchara->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    ierr = PCApplyTransposeMat_HARA(pc,pchara->wnsmat[3],Y);CHKERRQ(ierr);
  } else {
    for (i=0;i<pchara->hyperorder-1;i++) {
      ierr = PCApplyMat_HARA(pc,pchara->wnsmat[0],pchara->wnsmat[1]);CHKERRQ(ierr);
      ierr = MatMatMult(A,pchara->wnsmat[1],MAT_REUSE_MATRIX,PETSC_DEFAULT,&pchara->wnsmat[2]);CHKERRQ(ierr);
      ierr = MatAXPY(pchara->wnsmat[0],-1.,pchara->wnsmat[2],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(pchara->wnsmat[3],1.,pchara->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    ierr = PCApplyMat_HARA(pc,pchara->wnsmat[3],Y);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&pchara->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->wnsmat[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->wnsmat[2]);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->wnsmat[3]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_Hyper(Mat M, Mat X, Mat Y,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultKernel_Hyper(M,X,Y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Basic Newton-Schultz sampler: (2 * I - M * A) * M */
static PetscErrorCode MatMultKernel_NS(Mat M, Vec x, Vec y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_HARA        *pchara;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  A = pc->useAmat ? pc->mat : pc->pmat;
  pchara = (PC_HARA*)pc->data;
  ierr = MatCreateVecs(pchara->M,pchara->wns[0] ? NULL : &pchara->wns[0],pchara->wns[1] ? NULL : &pchara->wns[1]);CHKERRQ(ierr);
  if (t) {
    ierr = PCApplyTranspose_HARA(pc,x,y);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,y,pchara->wns[1]);CHKERRQ(ierr);
    ierr = PCApplyTranspose_HARA(pc,pchara->wns[1],pchara->wns[0]);CHKERRQ(ierr);
    ierr = VecAXPBY(y,-1.,2.,pchara->wns[0]);CHKERRQ(ierr);
  } else {
    ierr = PCApply_HARA(pc,x,y);CHKERRQ(ierr);
    ierr = MatMult(A,y,pchara->wns[0]);CHKERRQ(ierr);
    ierr = PCApply_HARA(pc,pchara->wns[0],pchara->wns[1]);CHKERRQ(ierr);
    ierr = VecAXPBY(y,-1.,2.,pchara->wns[1]);CHKERRQ(ierr);
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

/* (2 * I - M * A) * M, MatMat version */
static PetscErrorCode MatMatMultKernel_NS(Mat M, Mat X, Mat Y, PetscBool t)
{
  PC             pc;
  Mat            A;
  PC_HARA        *pchara;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void*)&pc);CHKERRQ(ierr);
  A = pc->useAmat ? pc->mat : pc->pmat;
  pchara = (PC_HARA*)pc->data;
  ierr = MatDuplicate(X,MAT_SHARE_NONZERO_PATTERN,&pchara->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDuplicate(Y,MAT_SHARE_NONZERO_PATTERN,&pchara->wnsmat[1]);CHKERRQ(ierr);
  if (t) {
    ierr = PCApplyTransposeMat_HARA(pc,X,Y);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(A,Y,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pchara->wnsmat[1]);CHKERRQ(ierr);
    ierr = PCApplyTransposeMat_HARA(pc,pchara->wnsmat[1],pchara->wnsmat[0]);CHKERRQ(ierr);
    ierr = MatScale(Y,2.);CHKERRQ(ierr);
    ierr = MatAXPY(Y,-1.,pchara->wnsmat[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  } else {
    ierr = PCApplyMat_HARA(pc,X,Y);CHKERRQ(ierr);
    ierr = MatMatMult(A,Y,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pchara->wnsmat[0]);CHKERRQ(ierr);
    ierr = PCApplyMat_HARA(pc,pchara->wnsmat[0],pchara->wnsmat[1]);CHKERRQ(ierr);
    ierr = MatScale(Y,2.);CHKERRQ(ierr);
    ierr = MatAXPY(Y,-1.,pchara->wnsmat[1],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&pchara->wnsmat[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&pchara->wnsmat[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatMultNumeric_NS(Mat M, Mat X, Mat Y, void *)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultKernel_NS(M,X,Y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatNorm_HARA(Mat,NormType,PetscReal*);

static PetscErrorCode PCHaraSetUpSampler_Private(PC pc)
{
  PC_HARA        *pchara = (PC_HARA*)pc->data;
  Mat            A = pc->useAmat ? pc->mat : pc->pmat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pchara->S) {
    PetscInt M,N,m,n;

    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)A),m,n,M,N,pc,&pchara->S);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(pchara->S,A,A);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    ierr = MatShellSetVecType(pchara->S,VECCUDA);CHKERRQ(ierr);
#endif
  }
  if (pchara->hyperorder >= 2) {
    ierr = MatShellSetOperation(pchara->S,MATOP_MULT,(void (*)(void))MatMult_Hyper);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pchara->S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_Hyper);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pchara->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_Hyper,NULL,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pchara->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_Hyper,NULL,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
  } else {
    ierr = MatShellSetOperation(pchara->S,MATOP_MULT,(void (*)(void))MatMult_NS);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pchara->S,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_NS);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pchara->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_NS,NULL,MATDENSE,MATDENSE);CHKERRQ(ierr);
    ierr = MatShellSetMatProductOperation(pchara->S,MATPRODUCT_AB,NULL,MatMatMultNumeric_NS,NULL,MATDENSECUDA,MATDENSECUDA);CHKERRQ(ierr);
  }
  ierr = MatPropagateSymmetryOptions(A,pchara->S);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* NS */
static PetscErrorCode PCSetUp_HARA(PC pc)
{
  PC_HARA        *pchara = (PC_HARA*)pc->data;
  Mat            A = pc->useAmat ? pc->mat : pc->pmat;
  PetscErrorCode ierr;
  NormType       norm = NORM_2;
  PetscReal      initerr,err;

  PetscFunctionBegin;
  if (!pchara->T) {
    PetscInt M,N,m,n;

    ierr = MatGetSize(pc->pmat,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(pc->pmat,&m,&n);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)pc->pmat),m,n,M,N,pc,&pchara->T);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(pchara->T,pc->pmat,pc->pmat);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pchara->T,MATOP_MULT,(void (*)(void))MatMult_MAmI);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pchara->T,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_MAmI);CHKERRQ(ierr);
    ierr = MatShellSetOperation(pchara->T,MATOP_NORM,(void (*)(void))MatNorm_HARA);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    ierr = MatShellSetVecType(pchara->T,VECCUDA);CHKERRQ(ierr);
#endif
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)pchara->T);CHKERRQ(ierr);
  }
  if (!pchara->M) {
    Mat       Ain = pc->pmat;
    PetscBool ishara,flg;
    PetscReal onenormA,infnormA;
    void      (*normfunc)(void);

    ierr = PetscObjectTypeCompare((PetscObject)Ain,MATHARA,&ishara);CHKERRQ(ierr);
    if (!ishara) {
      Ain  = pc->mat;
      ierr = PetscObjectTypeCompare((PetscObject)Ain,MATHARA,&ishara);CHKERRQ(ierr);
    }
    if (!ishara) {
      ierr = MatCreateHaraFromMat(A,pchara->sdim,pchara->coords,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&pchara->M);CHKERRQ(ierr);
    } else {
      ierr = MatDuplicate(Ain,MAT_SHARE_NONZERO_PATTERN,&pchara->M);CHKERRQ(ierr);
    }

    ierr = MatGetOperation(A,MATOP_NORM,&normfunc);CHKERRQ(ierr);
    if (!normfunc || pchara->useapproximatenorms) {
      ierr = MatSetOperation(A,MATOP_NORM,(void (*)(void))MatNorm_HARA);CHKERRQ(ierr);
    }
    ierr = MatNorm(A,NORM_1,&onenormA);CHKERRQ(ierr);
    ierr = MatGetOption(A,MAT_SYMMETRIC,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = MatNorm(A,NORM_INFINITY,&infnormA);CHKERRQ(ierr);
    } else infnormA = onenormA;
    ierr = MatSetOperation(A,MATOP_NORM,normfunc);CHKERRQ(ierr);
    pchara->s0 = 1./(infnormA*onenormA);
  }
  ierr = MatNorm(pchara->T,norm,&initerr);CHKERRQ(ierr);
  if (initerr > pchara->atol) {
    PetscInt i;

    ierr = PCHaraSetUpSampler_Private(pc);CHKERRQ(ierr);
    err  = initerr;
    if (pchara->monitor) { ierr = PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: %g %g\n",0,(double)err,(double)(err/initerr));CHKERRQ(ierr); }
    for (i = 0; i < pchara->maxits; i++) {
      Mat         M;
      const char* prefix;

      ierr = MatDuplicate(pchara->M,MAT_SHARE_NONZERO_PATTERN,&M);CHKERRQ(ierr);
      ierr = MatGetOptionsPrefix(M,&prefix);CHKERRQ(ierr);
      if (!prefix) {
        ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
        ierr = MatSetOptionsPrefix(M,prefix);CHKERRQ(ierr);
        ierr = MatAppendOptionsPrefix(M,"pc_hara_inv_");CHKERRQ(ierr);
      }
#if 0
  {
     Mat Sd1,Sd2,Id;
     PetscReal err;
     ierr = MatComputeOperator(pchara->S,MATDENSE,&Sd1);CHKERRQ(ierr);
     ierr = MatDuplicate(Sd1,MAT_COPY_VALUES,&Id);CHKERRQ(ierr);
     ierr = MatZeroEntries(Id);CHKERRQ(ierr);
     ierr = MatShift(Id,1.);CHKERRQ(ierr);
     ierr = MatMatMult(pchara->S,Id,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Sd2);CHKERRQ(ierr);
     ierr = MatAXPY(Sd2,-1.,Sd1,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
     ierr = MatNorm(Sd2,NORM_FROBENIUS,&err);CHKERRQ(ierr);
     ierr = PetscPrintf(PetscObjectComm((PetscObject)Sd2),"ERR %g\n",err);CHKERRQ(ierr);
     ierr = MatViewFromOptions(Sd2,NULL,"-Sd_view");CHKERRQ(ierr);
     ierr = MatDestroy(&Sd1);CHKERRQ(ierr);
     ierr = MatDestroy(&Sd2);CHKERRQ(ierr);
     ierr = MatDestroy(&Id);CHKERRQ(ierr);
  }
#endif
      ierr = MatHaraSetSamplingMat(M,pchara->S,1,PETSC_DECIDE);CHKERRQ(ierr);
      if (pc->setfromoptionscalled) {
        ierr = MatSetFromOptions(M);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if 0
      {
         Mat Md;
         ierr = MatComputeOperator(M,MATDENSE,&Md);CHKERRQ(ierr);
         ierr = MatViewFromOptions(Md,NULL,"-Md_view");CHKERRQ(ierr);
         ierr = MatDestroy(&Md);CHKERRQ(ierr);
         ierr = MatComputeOperator(pchara->S,MATDENSE,&Md);CHKERRQ(ierr);
         ierr = MatViewFromOptions(Md,NULL,"-Md_view");CHKERRQ(ierr);
         ierr = MatDestroy(&Md);CHKERRQ(ierr);
      }
#endif
      ierr = MatDestroy(&pchara->M);CHKERRQ(ierr);
      pchara->M = M;
      ierr = MatNorm(pchara->T,norm,&err);CHKERRQ(ierr);
      if (pchara->monitor) { ierr = PetscPrintf(PetscObjectComm((PetscObject)pc),"%D: %g %g\n",i+1,(double)err,(double)(err/initerr));CHKERRQ(ierr); }
      if (err < pchara->atol || err < pchara->rtol*initerr) break;
    }
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PCCreate_HARA(PC pc)
{
  PetscErrorCode ierr;
  PC_HARA        *pchara;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&pchara);CHKERRQ(ierr);
  pc->data = (void*)pchara;

  pchara->atol       = 1.e-2;
  pchara->rtol       = 1.e-6;
  pchara->maxits     = 50;
  pchara->hyperorder = 1; /* default to basic NewtonSchultz */

  pc->ops->destroy        = PCDestroy_HARA;
  pc->ops->setup          = PCSetUp_HARA;
  pc->ops->apply          = PCApply_HARA;
  pc->ops->matapply       = PCApplyMat_HARA;
  pc->ops->applytranspose = PCApplyTranspose_HARA;
  pc->ops->reset          = PCReset_HARA;
  pc->ops->setfromoptions = PCSetFromOptions_HARA;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_HARA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
