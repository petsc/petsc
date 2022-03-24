static char help[] = "Simple example to test separable objective optimizers.\n";

#include <petsc.h>
#include <petsctao.h>
#include <petscvec.h>
#include <petscmath.h>

#define NWORKLEFT 4
#define NWORKRIGHT 12

typedef struct _UserCtx
{
  PetscInt    m;                     /* The row dimension of F */
  PetscInt    n;                     /* The column dimension of F */
  PetscInt    matops;                /* Matrix format. 0 for stencil, 1 for random */
  PetscInt    iter;                  /* Numer of iterations for ADMM */
  PetscReal   hStart;                /* Starting point for Taylor test */
  PetscReal   hFactor;               /* Taylor test step factor */
  PetscReal   hMin;                  /* Taylor test end goal */
  PetscReal   alpha;                 /* regularization constant applied to || x ||_p */
  PetscReal   eps;                   /* small constant for approximating gradient of || x ||_1 */
  PetscReal   mu;                    /* the augmented Lagrangian term in ADMM */
  PetscReal   abstol;
  PetscReal   reltol;
  Mat         F;                     /* matrix in least squares component $(1/2) * || F x - d ||_2^2$ */
  Mat         W;                     /* Workspace matrix. ATA */
  Mat         Hm;                    /* Hessian Misfit*/
  Mat         Hr;                    /* Hessian Reg*/
  Vec         d;                     /* RHS in least squares component $(1/2) * || F x - d ||_2^2$ */
  Vec         workLeft[NWORKLEFT];   /* Workspace for temporary vec */
  Vec         workRight[NWORKRIGHT]; /* Workspace for temporary vec */
  NormType    p;
  PetscRandom rctx;
  PetscBool   taylor;                /* Flag to determine whether to run Taylor test or not */
  PetscBool   use_admm;              /* Flag to determine whether to run Taylor test or not */
}* UserCtx;

static PetscErrorCode CreateRHS(UserCtx ctx)
{
  PetscFunctionBegin;
  /* build the rhs d in ctx */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&(ctx->d)));
  CHKERRQ(VecSetSizes(ctx->d,PETSC_DECIDE,ctx->m));
  CHKERRQ(VecSetFromOptions(ctx->d));
  CHKERRQ(VecSetRandom(ctx->d,ctx->rctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMatrix(UserCtx ctx)
{
  PetscInt       Istart,Iend,i,j,Ii,gridN,I_n, I_s, I_e, I_w;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage;
#endif

  PetscFunctionBegin;
  /* build the matrix F in ctx */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &(ctx->F)));
  CHKERRQ(MatSetSizes(ctx->F,PETSC_DECIDE, PETSC_DECIDE, ctx->m, ctx->n));
  CHKERRQ(MatSetType(ctx->F,MATAIJ)); /* TODO: Decide specific SetType other than dummy*/
  CHKERRQ(MatMPIAIJSetPreallocation(ctx->F, 5, NULL, 5, NULL)); /*TODO: some number other than 5?*/
  CHKERRQ(MatSeqAIJSetPreallocation(ctx->F, 5, NULL));
  CHKERRQ(MatSetUp(ctx->F));
  CHKERRQ(MatGetOwnershipRange(ctx->F,&Istart,&Iend));
  CHKERRQ(PetscLogStageRegister("Assembly", &stage));
  CHKERRQ(PetscLogStagePush(stage));

  /* Set matrix elements in  2-D five point stencil format. */
  if (!(ctx->matops)) {
    PetscCheck(ctx->m == ctx->n,PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Stencil matrix must be square");
    gridN = (PetscInt) PetscSqrtReal((PetscReal) ctx->m);
    PetscCheck(gridN * gridN == ctx->m,PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of rows must be square");
    for (Ii=Istart; Ii<Iend; Ii++) {
      i   = Ii / gridN; j = Ii % gridN;
      I_n = i * gridN + j + 1;
      if (j + 1 >= gridN) I_n = -1;
      I_s = i * gridN + j - 1;
      if (j - 1 < 0) I_s = -1;
      I_e = (i + 1) * gridN + j;
      if (i + 1 >= gridN) I_e = -1;
      I_w = (i - 1) * gridN + j;
      if (i - 1 < 0) I_w = -1;
      CHKERRQ(MatSetValue(ctx->F, Ii, Ii, 4., INSERT_VALUES));
      CHKERRQ(MatSetValue(ctx->F, Ii, I_n, -1., INSERT_VALUES));
      CHKERRQ(MatSetValue(ctx->F, Ii, I_s, -1., INSERT_VALUES));
      CHKERRQ(MatSetValue(ctx->F, Ii, I_e, -1., INSERT_VALUES));
      CHKERRQ(MatSetValue(ctx->F, Ii, I_w, -1., INSERT_VALUES));
    }
  } else CHKERRQ(MatSetRandom(ctx->F, ctx->rctx));
  CHKERRQ(MatAssemblyBegin(ctx->F, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(ctx->F, MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogStagePop());
  /* Stencil matrix is symmetric. Setting symmetric flag for ICC/Cholesky preconditioner */
  if (!(ctx->matops)) {
    CHKERRQ(MatSetOption(ctx->F,MAT_SYMMETRIC,PETSC_TRUE));
  }
  CHKERRQ(MatTransposeMatMult(ctx->F,ctx->F, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->W)));
  /* Setup Hessian Workspace in same shape as W */
  CHKERRQ(MatDuplicate(ctx->W,MAT_DO_NOT_COPY_VALUES,&(ctx->Hm)));
  CHKERRQ(MatDuplicate(ctx->W,MAT_DO_NOT_COPY_VALUES,&(ctx->Hr)));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupWorkspace(UserCtx ctx)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(MatCreateVecs(ctx->F, &ctx->workLeft[0], &ctx->workRight[0]));
  for (i=1; i<NWORKLEFT; i++) {
    CHKERRQ(VecDuplicate(ctx->workLeft[0], &(ctx->workLeft[i])));
  }
  for (i=1; i<NWORKRIGHT; i++) {
    CHKERRQ(VecDuplicate(ctx->workRight[0], &(ctx->workRight[i])));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ConfigureContext(UserCtx ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ctx->m        = 16;
  ctx->n        = 16;
  ctx->eps      = 1.e-3;
  ctx->abstol   = 1.e-4;
  ctx->reltol   = 1.e-2;
  ctx->hStart   = 1.;
  ctx->hMin     = 1.e-3;
  ctx->hFactor  = 0.5;
  ctx->alpha    = 1.;
  ctx->mu       = 1.0;
  ctx->matops   = 0;
  ctx->iter     = 10;
  ctx->p        = NORM_2;
  ctx->taylor   = PETSC_TRUE;
  ctx->use_admm = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "ex4.c");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-m", "The row dimension of matrix F", "ex4.c", ctx->m, &(ctx->m), NULL));
  CHKERRQ(PetscOptionsInt("-n", "The column dimension of matrix F", "ex4.c", ctx->n, &(ctx->n), NULL));
  CHKERRQ(PetscOptionsInt("-matrix_format","Decide format of F matrix. 0 for stencil, 1 for random", "ex4.c", ctx->matops, &(ctx->matops), NULL));
  CHKERRQ(PetscOptionsInt("-iter","Iteration number ADMM", "ex4.c", ctx->iter, &(ctx->iter), NULL));
  CHKERRQ(PetscOptionsReal("-alpha", "The regularization multiplier. 1 default", "ex4.c", ctx->alpha, &(ctx->alpha), NULL));
  CHKERRQ(PetscOptionsReal("-epsilon", "The small constant added to |x_i| in the denominator to approximate the gradient of ||x||_1", "ex4.c", ctx->eps, &(ctx->eps), NULL));
  CHKERRQ(PetscOptionsReal("-mu", "The augmented lagrangian multiplier in ADMM", "ex4.c", ctx->mu, &(ctx->mu), NULL));
  CHKERRQ(PetscOptionsReal("-hStart", "Taylor test starting point. 1 default.", "ex4.c", ctx->hStart, &(ctx->hStart), NULL));
  CHKERRQ(PetscOptionsReal("-hFactor", "Taylor test multiplier factor. 0.5 default", "ex4.c", ctx->hFactor, &(ctx->hFactor), NULL));
  CHKERRQ(PetscOptionsReal("-hMin", "Taylor test ending condition. 1.e-3 default", "ex4.c", ctx->hMin, &(ctx->hMin), NULL));
  CHKERRQ(PetscOptionsReal("-abstol", "Absolute stopping criterion for ADMM", "ex4.c", ctx->abstol, &(ctx->abstol), NULL));
  CHKERRQ(PetscOptionsReal("-reltol", "Relative stopping criterion for ADMM", "ex4.c", ctx->reltol, &(ctx->reltol), NULL));
  CHKERRQ(PetscOptionsBool("-taylor","Flag for Taylor test. Default is true.", "ex4.c", ctx->taylor, &(ctx->taylor), NULL));
  CHKERRQ(PetscOptionsBool("-use_admm","Use the ADMM solver in this example.", "ex4.c", ctx->use_admm, &(ctx->use_admm), NULL));
  CHKERRQ(PetscOptionsEnum("-p","Norm type.", "ex4.c", NormTypes, (PetscEnum)ctx->p, (PetscEnum *) &(ctx->p), NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /* Creating random ctx */
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&(ctx->rctx)));
  CHKERRQ(PetscRandomSetFromOptions(ctx->rctx));
  CHKERRQ(CreateMatrix(ctx));
  CHKERRQ(CreateRHS(ctx));
  CHKERRQ(SetupWorkspace(ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyContext(UserCtx *ctx)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&((*ctx)->F)));
  CHKERRQ(MatDestroy(&((*ctx)->W)));
  CHKERRQ(MatDestroy(&((*ctx)->Hm)));
  CHKERRQ(MatDestroy(&((*ctx)->Hr)));
  CHKERRQ(VecDestroy(&((*ctx)->d)));
  for (i=0; i<NWORKLEFT; i++) {
    CHKERRQ(VecDestroy(&((*ctx)->workLeft[i])));
  }
  for (i=0; i<NWORKRIGHT; i++) {
    CHKERRQ(VecDestroy(&((*ctx)->workRight[i])));
  }
  CHKERRQ(PetscRandomDestroy(&((*ctx)->rctx)));
  CHKERRQ(PetscFree(*ctx));
  PetscFunctionReturn(0);
}

/* compute (1/2) * ||F x - d||^2 */
static PetscErrorCode ObjectiveMisfit(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;
  Vec            y;

  PetscFunctionBegin;
  y    = ctx->workLeft[0];
  CHKERRQ(MatMult(ctx->F, x, y));
  CHKERRQ(VecAXPY(y, -1., ctx->d));
  CHKERRQ(VecDot(y, y, J));
  *J  *= 0.5;
  PetscFunctionReturn(0);
}

/* compute V = FTFx - FTd */
static PetscErrorCode GradientMisfit(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;
  Vec            FTFx, FTd;

  PetscFunctionBegin;
  /* work1 is A^T Ax, work2 is Ab, W is A^T A*/
  FTFx = ctx->workRight[0];
  FTd  = ctx->workRight[1];
  CHKERRQ(MatMult(ctx->W,x,FTFx));
  CHKERRQ(MatMultTranspose(ctx->F, ctx->d, FTd));
  CHKERRQ(VecWAXPY(V, -1., FTd, FTFx));
  PetscFunctionReturn(0);
}

/* returns FTF */
static PetscErrorCode HessianMisfit(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;

  PetscFunctionBegin;
  if (H != ctx->W) CHKERRQ(MatCopy(ctx->W, H, DIFFERENT_NONZERO_PATTERN));
  if (Hpre != ctx->W) CHKERRQ(MatCopy(ctx->W, Hpre, DIFFERENT_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

/* computes augment Lagrangian objective (with scaled dual):
 * 0.5 * ||F x - d||^2  + 0.5 * mu ||x - z + u||^2 */
static PetscErrorCode ObjectiveMisfitADMM(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;
  PetscReal      mu, workNorm, misfit;
  Vec            z, u, temp;

  PetscFunctionBegin;
  mu   = ctx->mu;
  z    = ctx->workRight[5];
  u    = ctx->workRight[6];
  temp = ctx->workRight[10];
  /* misfit = f(x) */
  CHKERRQ(ObjectiveMisfit(tao, x, &misfit, _ctx));
  CHKERRQ(VecCopy(x,temp));
  /* temp = x - z + u */
  CHKERRQ(VecAXPBYPCZ(temp,-1.,1.,1.,z,u));
  /* workNorm = ||x - z + u||^2 */
  CHKERRQ(VecDot(temp, temp, &workNorm));
  /* augment Lagrangian objective (with scaled dual): f(x) + 0.5 * mu ||x -z + u||^2 */
  *J = misfit + 0.5 * mu * workNorm;
  PetscFunctionReturn(0);
}

/* computes FTFx - FTd  mu*(x - z + u) */
static PetscErrorCode GradientMisfitADMM(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;
  PetscReal      mu;
  Vec            z, u, temp;

  PetscFunctionBegin;
  mu   = ctx->mu;
  z    = ctx->workRight[5];
  u    = ctx->workRight[6];
  temp = ctx->workRight[10];
  CHKERRQ(GradientMisfit(tao, x, V, _ctx));
  CHKERRQ(VecCopy(x, temp));
  /* temp = x - z + u */
  CHKERRQ(VecAXPBYPCZ(temp,-1.,1.,1.,z,u));
  /* V =  FTFx - FTd  mu*(x - z + u) */
  CHKERRQ(VecAXPY(V, mu, temp));
  PetscFunctionReturn(0);
}

/* returns FTF + diag(mu) */
static PetscErrorCode HessianMisfitADMM(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;

  PetscFunctionBegin;
  CHKERRQ(MatCopy(ctx->W, H, DIFFERENT_NONZERO_PATTERN));
  CHKERRQ(MatShift(H, ctx->mu));
  if (Hpre != H) {
    CHKERRQ(MatCopy(H, Hpre, DIFFERENT_NONZERO_PATTERN));
  }
  PetscFunctionReturn(0);
}

/* computes || x ||_p (mult by 0.5 in case of NORM_2) */
static PetscErrorCode ObjectiveRegularization(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;
  PetscReal      norm;

  PetscFunctionBegin;
  *J = 0;
  CHKERRQ(VecNorm (x, ctx->p, &norm));
  if (ctx->p == NORM_2) norm = 0.5 * norm * norm;
  *J = ctx->alpha * norm;
  PetscFunctionReturn(0);
}

/* NORM_2 Case: return x
 * NORM_1 Case: x/(|x| + eps)
 * Else: TODO */
static PetscErrorCode GradientRegularization(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;
  PetscReal      eps = ctx->eps;

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    CHKERRQ(VecCopy(x, V));
  } else if (ctx->p == NORM_1) {
    CHKERRQ(VecCopy(x, ctx->workRight[1]));
    CHKERRQ(VecAbs(ctx->workRight[1]));
    CHKERRQ(VecShift(ctx->workRight[1], eps));
    CHKERRQ(VecPointwiseDivide(V, x, ctx->workRight[1]));
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Example only works for NORM_1 and NORM_2");
  PetscFunctionReturn(0);
}

/* NORM_2 Case: returns diag(mu)
 * NORM_1 Case: diag(mu* 1/sqrt(x_i^2 + eps) * (1 - x_i^2/ABS(x_i^2+eps)))  */
static PetscErrorCode HessianRegularization(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;
  PetscReal      eps = ctx->eps;
  Vec            copy1,copy2,copy3;

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    /* Identity matrix scaled by mu */
    CHKERRQ(MatZeroEntries(H));
    CHKERRQ(MatShift(H,ctx->mu));
    if (Hpre != H) {
      CHKERRQ(MatZeroEntries(Hpre));
      CHKERRQ(MatShift(Hpre,ctx->mu));
    }
  } else if (ctx->p == NORM_1) {
    /* 1/sqrt(x_i^2 + eps) * (1 - x_i^2/ABS(x_i^2+eps)) */
    copy1 = ctx->workRight[1];
    copy2 = ctx->workRight[2];
    copy3 = ctx->workRight[3];
    /* copy1 : 1/sqrt(x_i^2 + eps) */
    CHKERRQ(VecCopy(x, copy1));
    CHKERRQ(VecPow(copy1,2));
    CHKERRQ(VecShift(copy1, eps));
    CHKERRQ(VecSqrtAbs(copy1));
    CHKERRQ(VecReciprocal(copy1));
    /* copy2:  x_i^2.*/
    CHKERRQ(VecCopy(x,copy2));
    CHKERRQ(VecPow(copy2,2));
    /* copy3: abs(x_i^2 + eps) */
    CHKERRQ(VecCopy(x,copy3));
    CHKERRQ(VecPow(copy3,2));
    CHKERRQ(VecShift(copy3, eps));
    CHKERRQ(VecAbs(copy3));
    /* copy2: 1 - x_i^2/abs(x_i^2 + eps) */
    CHKERRQ(VecPointwiseDivide(copy2, copy2,copy3));
    CHKERRQ(VecScale(copy2, -1.));
    CHKERRQ(VecShift(copy2, 1.));
    CHKERRQ(VecAXPY(copy1,1.,copy2));
    CHKERRQ(VecScale(copy1, ctx->mu));
    CHKERRQ(MatZeroEntries(H));
    CHKERRQ(MatDiagonalSet(H, copy1,INSERT_VALUES));
    if (Hpre != H) {
      CHKERRQ(MatZeroEntries(Hpre));
      CHKERRQ(MatDiagonalSet(Hpre, copy1,INSERT_VALUES));
    }
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Example only works for NORM_1 and NORM_2");
  PetscFunctionReturn(0);
}

/* NORM_2 Case: 0.5 || x ||_2 + 0.5 * mu * ||x + u - z||^2
 * Else : || x ||_2 + 0.5 * mu * ||x + u - z||^2 */
static PetscErrorCode ObjectiveRegularizationADMM(Tao tao, Vec z, PetscReal *J, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;
  PetscReal      mu, workNorm, reg;
  Vec            x, u, temp;

  PetscFunctionBegin;
  mu   = ctx->mu;
  x    = ctx->workRight[4];
  u    = ctx->workRight[6];
  temp = ctx->workRight[10];
  CHKERRQ(ObjectiveRegularization(tao, z, &reg, _ctx));
  CHKERRQ(VecCopy(z,temp));
  /* temp = x + u -z */
  CHKERRQ(VecAXPBYPCZ(temp,1.,1.,-1.,x,u));
  /* workNorm = ||x + u - z ||^2 */
  CHKERRQ(VecDot(temp, temp, &workNorm));
  *J   = reg + 0.5 * mu * workNorm;
  PetscFunctionReturn(0);
}

/* NORM_2 Case: x - mu*(x + u - z)
 * NORM_1 Case: x/(|x| + eps) - mu*(x + u - z)
 * Else: TODO */
static PetscErrorCode GradientRegularizationADMM(Tao tao, Vec z, Vec V, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;
  PetscReal      mu;
  Vec            x, u, temp;

  PetscFunctionBegin;
  mu   = ctx->mu;
  x    = ctx->workRight[4];
  u    = ctx->workRight[6];
  temp = ctx->workRight[10];
  CHKERRQ(GradientRegularization(tao, z, V, _ctx));
  CHKERRQ(VecCopy(z, temp));
  /* temp = x + u -z */
  CHKERRQ(VecAXPBYPCZ(temp,1.,1.,-1.,x,u));
  CHKERRQ(VecAXPY(V, -mu, temp));
  PetscFunctionReturn(0);
}

/* NORM_2 Case: returns diag(mu)
 * NORM_1 Case: FTF + diag(mu) */
static PetscErrorCode HessianRegularizationADMM(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx        ctx = (UserCtx) _ctx;

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    /* Identity matrix scaled by mu */
    CHKERRQ(MatZeroEntries(H));
    CHKERRQ(MatShift(H,ctx->mu));
    if (Hpre != H) {
      CHKERRQ(MatZeroEntries(Hpre));
      CHKERRQ(MatShift(Hpre,ctx->mu));
    }
  } else if (ctx->p == NORM_1) {
    CHKERRQ(HessianMisfit(tao, x, H, Hpre, (void*) ctx));
    CHKERRQ(MatShift(H, ctx->mu));
    if (Hpre != H) CHKERRQ(MatShift(Hpre, ctx->mu));
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Example only works for NORM_1 and NORM_2");
  PetscFunctionReturn(0);
}

/* NORM_2 Case : (1/2) * ||F x - d||^2 + 0.5 * || x ||_p
*  NORM_1 Case : (1/2) * ||F x - d||^2 + || x ||_p */
static PetscErrorCode ObjectiveComplete(Tao tao, Vec x, PetscReal *J, void *ctx)
{
  PetscReal      Jm, Jr;

  PetscFunctionBegin;
  CHKERRQ(ObjectiveMisfit(tao, x, &Jm, ctx));
  CHKERRQ(ObjectiveRegularization(tao, x, &Jr, ctx));
  *J   = Jm + Jr;
  PetscFunctionReturn(0);
}

/* NORM_2 Case: FTFx - FTd + x
 * NORM_1 Case: FTFx - FTd + x/(|x| + eps) */
static PetscErrorCode GradientComplete(Tao tao, Vec x, Vec V, void *ctx)
{
  UserCtx        cntx = (UserCtx) ctx;

  PetscFunctionBegin;
  CHKERRQ(GradientMisfit(tao, x, cntx->workRight[2], ctx));
  CHKERRQ(GradientRegularization(tao, x, cntx->workRight[3], ctx));
  CHKERRQ(VecWAXPY(V,1,cntx->workRight[2],cntx->workRight[3]));
  PetscFunctionReturn(0);
}

/* NORM_2 Case: diag(mu) + FTF
 * NORM_1 Case: diag(mu* 1/sqrt(x_i^2 + eps) * (1 - x_i^2/ABS(x_i^2+eps))) + FTF  */
static PetscErrorCode HessianComplete(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  Mat            tempH;

  PetscFunctionBegin;
  CHKERRQ(MatDuplicate(H, MAT_SHARE_NONZERO_PATTERN, &tempH));
  CHKERRQ(HessianMisfit(tao, x, H, H, ctx));
  CHKERRQ(HessianRegularization(tao, x, tempH, tempH, ctx));
  CHKERRQ(MatAXPY(H, 1., tempH, DIFFERENT_NONZERO_PATTERN));
  if (Hpre != H) {
    CHKERRQ(MatCopy(H, Hpre, DIFFERENT_NONZERO_PATTERN));
  }
  CHKERRQ(MatDestroy(&tempH));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolveADMM(UserCtx ctx,  Vec x)
{
  PetscInt       i;
  PetscReal      u_norm, r_norm, s_norm, primal, dual, x_norm, z_norm;
  Tao            tao1,tao2;
  Vec            xk,z,u,diff,zold,zdiff,temp;
  PetscReal      mu;

  PetscFunctionBegin;
  xk    = ctx->workRight[4];
  z     = ctx->workRight[5];
  u     = ctx->workRight[6];
  diff  = ctx->workRight[7];
  zold  = ctx->workRight[8];
  zdiff = ctx->workRight[9];
  temp  = ctx->workRight[11];
  mu    = ctx->mu;
  CHKERRQ(VecSet(u, 0.));
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD, &tao1));
  CHKERRQ(TaoSetType(tao1,TAONLS));
  CHKERRQ(TaoSetObjective(tao1, ObjectiveMisfitADMM, (void*) ctx));
  CHKERRQ(TaoSetGradient(tao1, NULL, GradientMisfitADMM, (void*) ctx));
  CHKERRQ(TaoSetHessian(tao1, ctx->Hm, ctx->Hm, HessianMisfitADMM, (void*) ctx));
  CHKERRQ(VecSet(xk, 0.));
  CHKERRQ(TaoSetSolution(tao1, xk));
  CHKERRQ(TaoSetOptionsPrefix(tao1, "misfit_"));
  CHKERRQ(TaoSetFromOptions(tao1));
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD, &tao2));
  if (ctx->p == NORM_2) {
    CHKERRQ(TaoSetType(tao2,TAONLS));
    CHKERRQ(TaoSetObjective(tao2, ObjectiveRegularizationADMM, (void*) ctx));
    CHKERRQ(TaoSetGradient(tao2, NULL, GradientRegularizationADMM, (void*) ctx));
    CHKERRQ(TaoSetHessian(tao2, ctx->Hr, ctx->Hr, HessianRegularizationADMM, (void*) ctx));
  }
  CHKERRQ(VecSet(z, 0.));
  CHKERRQ(TaoSetSolution(tao2, z));
  CHKERRQ(TaoSetOptionsPrefix(tao2, "reg_"));
  CHKERRQ(TaoSetFromOptions(tao2));

  for (i=0; i<ctx->iter; i++) {
    CHKERRQ(VecCopy(z,zold));
    CHKERRQ(TaoSolve(tao1)); /* Updates xk */
    if (ctx->p == NORM_1) {
      CHKERRQ(VecWAXPY(temp,1.,xk,u));
      CHKERRQ(TaoSoftThreshold(temp,-ctx->alpha/mu,ctx->alpha/mu,z));
    } else {
      CHKERRQ(TaoSolve(tao2)); /* Update zk */
    }
    /* u = u + xk -z */
    CHKERRQ(VecAXPBYPCZ(u,1.,-1.,1.,xk,z));
    /* r_norm : norm(x-z) */
    CHKERRQ(VecWAXPY(diff,-1.,z,xk));
    CHKERRQ(VecNorm(diff,NORM_2,&r_norm));
    /* s_norm : norm(-mu(z-zold)) */
    CHKERRQ(VecWAXPY(zdiff, -1.,zold,z));
    CHKERRQ(VecNorm(zdiff,NORM_2,&s_norm));
    s_norm = s_norm * mu;
    /* primal : sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z))*/
    CHKERRQ(VecNorm(xk,NORM_2,&x_norm));
    CHKERRQ(VecNorm(z,NORM_2,&z_norm));
    primal = PetscSqrtReal(ctx->n)*ctx->abstol + ctx->reltol*PetscMax(x_norm,z_norm);
    /* Duality : sqrt(n)*ABSTOL + RELTOL*norm(mu*u)*/
    CHKERRQ(VecNorm(u,NORM_2,&u_norm));
    dual = PetscSqrtReal(ctx->n)*ctx->abstol + ctx->reltol*u_norm*mu;
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)tao1),"Iter %D : ||x-z||: %g, mu*||z-zold||: %g\n", i, (double) r_norm, (double) s_norm));
    if (r_norm < primal && s_norm < dual) break;
  }
  CHKERRQ(VecCopy(xk, x));
  CHKERRQ(TaoDestroy(&tao1));
  CHKERRQ(TaoDestroy(&tao2));
  PetscFunctionReturn(0);
}

/* Second order Taylor remainder convergence test */
static PetscErrorCode TaylorTest(UserCtx ctx, Tao tao, Vec x, PetscReal *C)
{
  PetscReal      h,J,temp;
  PetscInt       i,j;
  PetscInt       numValues;
  PetscReal      Jx,Jxhat_comp,Jxhat_pred;
  PetscReal      *Js, *hs;
  PetscReal      gdotdx;
  PetscReal      minrate = PETSC_MAX_REAL;
  MPI_Comm       comm = PetscObjectComm((PetscObject)x);
  Vec            g, dx, xhat;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(x, &g));
  CHKERRQ(VecDuplicate(x, &xhat));
  /* choose a perturbation direction */
  CHKERRQ(VecDuplicate(x, &dx));
  CHKERRQ(VecSetRandom(dx,ctx->rctx));
  /* evaluate objective at x: J(x) */
  CHKERRQ(TaoComputeObjective(tao, x, &Jx));
  /* evaluate gradient at x, save in vector g */
  CHKERRQ(TaoComputeGradient(tao, x, g));
  CHKERRQ(VecDot(g, dx, &gdotdx));

  for (numValues=0, h=ctx->hStart; h>=ctx->hMin; h*=ctx->hFactor) numValues++;
  CHKERRQ(PetscCalloc2(numValues, &Js, numValues, &hs));
  for (i=0, h=ctx->hStart; h>=ctx->hMin; h*=ctx->hFactor, i++) {
    CHKERRQ(VecWAXPY(xhat, h, dx, x));
    CHKERRQ(TaoComputeObjective(tao, xhat, &Jxhat_comp));
    /* J(\hat(x)) \approx J(x) + g^T (xhat - x) = J(x) + h * g^T dx */
    Jxhat_pred = Jx + h * gdotdx;
    /* Vector to dJdm scalar? Dot?*/
    J     = PetscAbsReal(Jxhat_comp - Jxhat_pred);
    CHKERRQ(PetscPrintf (comm, "J(xhat): %g, predicted: %g, diff %g\n", (double) Jxhat_comp,(double) Jxhat_pred, (double) J));
    Js[i] = J;
    hs[i] = h;
  }
  for (j=1; j<numValues; j++) {
    temp    = PetscLogReal(Js[j]/Js[j-1]) / PetscLogReal (hs[j]/hs[j-1]);
    CHKERRQ(PetscPrintf (comm, "Convergence rate step %D: %g\n", j-1, (double) temp));
    minrate = PetscMin(minrate, temp);
  }
  /* If O is not ~2, then the test is wrong */
  CHKERRQ(PetscFree2(Js, hs));
  *C   = minrate;
  CHKERRQ(VecDestroy(&dx));
  CHKERRQ(VecDestroy(&xhat));
  CHKERRQ(VecDestroy(&g));
  PetscFunctionReturn(0);
}

int main(int argc, char ** argv)
{
  UserCtx ctx;
  Tao     tao;
  Vec     x;
  Mat     H;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(PetscNew(&ctx));
  CHKERRQ(ConfigureContext(ctx));
  /* Define two functions that could pass as objectives to TaoSetObjective(): one
   * for the misfit component, and one for the regularization component */
  /* ObjectiveMisfit() and ObjectiveRegularization() */

  /* Define a single function that calls both components adds them together: the complete objective,
   * in the absence of a Tao implementation that handles separability */
  /* ObjectiveComplete() */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD, &tao));
  CHKERRQ(TaoSetType(tao,TAONM));
  CHKERRQ(TaoSetObjective(tao, ObjectiveComplete, (void*) ctx));
  CHKERRQ(TaoSetGradient(tao, NULL, GradientComplete, (void*) ctx));
  CHKERRQ(MatDuplicate(ctx->W, MAT_SHARE_NONZERO_PATTERN, &H));
  CHKERRQ(TaoSetHessian(tao, H, H, HessianComplete, (void*) ctx));
  CHKERRQ(MatCreateVecs(ctx->F, NULL, &x));
  CHKERRQ(VecSet(x, 0.));
  CHKERRQ(TaoSetSolution(tao, x));
  CHKERRQ(TaoSetFromOptions(tao));
  if (ctx->use_admm) {
    CHKERRQ(TaoSolveADMM(ctx,x));
  } else CHKERRQ(TaoSolve(tao));
  /* examine solution */
  CHKERRQ(VecViewFromOptions(x, NULL, "-view_sol"));
  if (ctx->taylor) {
    PetscReal rate;
    CHKERRQ(TaylorTest(ctx, tao, x, &rate));
  }
  CHKERRQ(MatDestroy(&H));
  CHKERRQ(TaoDestroy(&tao));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(DestroyContext(&ctx));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex

  test:
    suffix: 0
    args:

  test:
    suffix: l1_1
    args: -p 1 -tao_type lmvm -alpha 1. -epsilon 1.e-7 -m 64 -n 64 -view_sol -matrix_format 1

  test:
    suffix: hessian_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -tao_type nls

  test:
    suffix: hessian_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -tao_type nls

  test:
    suffix: nm_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -tao_type nm -tao_max_it 50

  test:
    suffix: nm_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -tao_type nm -tao_max_it 50

  test:
    suffix: lmvm_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -tao_type lmvm -tao_max_it 40

  test:
    suffix: lmvm_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -tao_type lmvm -tao_max_it 15

  test:
    suffix: soft_threshold_admm_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -use_admm

  test:
    suffix: hessian_admm_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -use_admm -reg_tao_type nls -misfit_tao_type nls

  test:
    suffix: hessian_admm_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -use_admm -reg_tao_type nls -misfit_tao_type nls

  test:
    suffix: nm_admm_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -use_admm -reg_tao_type nm -misfit_tao_type nm

  test:
    suffix: nm_admm_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -use_admm -reg_tao_type nm -misfit_tao_type nm -iter 7

  test:
    suffix: lmvm_admm_1
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 1 -use_admm -reg_tao_type lmvm -misfit_tao_type lmvm

  test:
    suffix: lmvm_admm_2
    args: -matrix_format 1 -m 100 -n 100 -tao_monitor -p 2 -use_admm -reg_tao_type lmvm -misfit_tao_type lmvm

TEST*/
