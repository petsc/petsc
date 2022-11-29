static char help[] = "Simple example to test separable objective optimizers.\n";

#include <petsc.h>
#include <petsctao.h>
#include <petscvec.h>
#include <petscmath.h>

#define NWORKLEFT  4
#define NWORKRIGHT 12

typedef struct _UserCtx {
  PetscInt    m;       /* The row dimension of F */
  PetscInt    n;       /* The column dimension of F */
  PetscInt    matops;  /* Matrix format. 0 for stencil, 1 for random */
  PetscInt    iter;    /* Numer of iterations for ADMM */
  PetscReal   hStart;  /* Starting point for Taylor test */
  PetscReal   hFactor; /* Taylor test step factor */
  PetscReal   hMin;    /* Taylor test end goal */
  PetscReal   alpha;   /* regularization constant applied to || x ||_p */
  PetscReal   eps;     /* small constant for approximating gradient of || x ||_1 */
  PetscReal   mu;      /* the augmented Lagrangian term in ADMM */
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
  PetscBool   taylor;   /* Flag to determine whether to run Taylor test or not */
  PetscBool   use_admm; /* Flag to determine whether to run Taylor test or not */
} *UserCtx;

static PetscErrorCode CreateRHS(UserCtx ctx)
{
  PetscFunctionBegin;
  /* build the rhs d in ctx */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &(ctx->d)));
  PetscCall(VecSetSizes(ctx->d, PETSC_DECIDE, ctx->m));
  PetscCall(VecSetFromOptions(ctx->d));
  PetscCall(VecSetRandom(ctx->d, ctx->rctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMatrix(UserCtx ctx)
{
  PetscInt Istart, Iend, i, j, Ii, gridN, I_n, I_s, I_e, I_w;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif

  PetscFunctionBegin;
  /* build the matrix F in ctx */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &(ctx->F)));
  PetscCall(MatSetSizes(ctx->F, PETSC_DECIDE, PETSC_DECIDE, ctx->m, ctx->n));
  PetscCall(MatSetType(ctx->F, MATAIJ));                          /* TODO: Decide specific SetType other than dummy*/
  PetscCall(MatMPIAIJSetPreallocation(ctx->F, 5, NULL, 5, NULL)); /*TODO: some number other than 5?*/
  PetscCall(MatSeqAIJSetPreallocation(ctx->F, 5, NULL));
  PetscCall(MatSetUp(ctx->F));
  PetscCall(MatGetOwnershipRange(ctx->F, &Istart, &Iend));
  PetscCall(PetscLogStageRegister("Assembly", &stage));
  PetscCall(PetscLogStagePush(stage));

  /* Set matrix elements in  2-D five point stencil format. */
  if (!(ctx->matops)) {
    PetscCheck(ctx->m == ctx->n, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Stencil matrix must be square");
    gridN = (PetscInt)PetscSqrtReal((PetscReal)ctx->m);
    PetscCheck(gridN * gridN == ctx->m, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Number of rows must be square");
    for (Ii = Istart; Ii < Iend; Ii++) {
      i   = Ii / gridN;
      j   = Ii % gridN;
      I_n = i * gridN + j + 1;
      if (j + 1 >= gridN) I_n = -1;
      I_s = i * gridN + j - 1;
      if (j - 1 < 0) I_s = -1;
      I_e = (i + 1) * gridN + j;
      if (i + 1 >= gridN) I_e = -1;
      I_w = (i - 1) * gridN + j;
      if (i - 1 < 0) I_w = -1;
      PetscCall(MatSetValue(ctx->F, Ii, Ii, 4., INSERT_VALUES));
      PetscCall(MatSetValue(ctx->F, Ii, I_n, -1., INSERT_VALUES));
      PetscCall(MatSetValue(ctx->F, Ii, I_s, -1., INSERT_VALUES));
      PetscCall(MatSetValue(ctx->F, Ii, I_e, -1., INSERT_VALUES));
      PetscCall(MatSetValue(ctx->F, Ii, I_w, -1., INSERT_VALUES));
    }
  } else PetscCall(MatSetRandom(ctx->F, ctx->rctx));
  PetscCall(MatAssemblyBegin(ctx->F, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->F, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogStagePop());
  /* Stencil matrix is symmetric. Setting symmetric flag for ICC/Cholesky preconditioner */
  if (!(ctx->matops)) PetscCall(MatSetOption(ctx->F, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatTransposeMatMult(ctx->F, ctx->F, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(ctx->W)));
  /* Setup Hessian Workspace in same shape as W */
  PetscCall(MatDuplicate(ctx->W, MAT_DO_NOT_COPY_VALUES, &(ctx->Hm)));
  PetscCall(MatDuplicate(ctx->W, MAT_DO_NOT_COPY_VALUES, &(ctx->Hr)));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupWorkspace(UserCtx ctx)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(ctx->F, &ctx->workLeft[0], &ctx->workRight[0]));
  for (i = 1; i < NWORKLEFT; i++) PetscCall(VecDuplicate(ctx->workLeft[0], &(ctx->workLeft[i])));
  for (i = 1; i < NWORKRIGHT; i++) PetscCall(VecDuplicate(ctx->workRight[0], &(ctx->workRight[i])));
  PetscFunctionReturn(0);
}

static PetscErrorCode ConfigureContext(UserCtx ctx)
{
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
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "ex4.c");
  PetscCall(PetscOptionsInt("-m", "The row dimension of matrix F", "ex4.c", ctx->m, &(ctx->m), NULL));
  PetscCall(PetscOptionsInt("-n", "The column dimension of matrix F", "ex4.c", ctx->n, &(ctx->n), NULL));
  PetscCall(PetscOptionsInt("-matrix_format", "Decide format of F matrix. 0 for stencil, 1 for random", "ex4.c", ctx->matops, &(ctx->matops), NULL));
  PetscCall(PetscOptionsInt("-iter", "Iteration number ADMM", "ex4.c", ctx->iter, &(ctx->iter), NULL));
  PetscCall(PetscOptionsReal("-alpha", "The regularization multiplier. 1 default", "ex4.c", ctx->alpha, &(ctx->alpha), NULL));
  PetscCall(PetscOptionsReal("-epsilon", "The small constant added to |x_i| in the denominator to approximate the gradient of ||x||_1", "ex4.c", ctx->eps, &(ctx->eps), NULL));
  PetscCall(PetscOptionsReal("-mu", "The augmented lagrangian multiplier in ADMM", "ex4.c", ctx->mu, &(ctx->mu), NULL));
  PetscCall(PetscOptionsReal("-hStart", "Taylor test starting point. 1 default.", "ex4.c", ctx->hStart, &(ctx->hStart), NULL));
  PetscCall(PetscOptionsReal("-hFactor", "Taylor test multiplier factor. 0.5 default", "ex4.c", ctx->hFactor, &(ctx->hFactor), NULL));
  PetscCall(PetscOptionsReal("-hMin", "Taylor test ending condition. 1.e-3 default", "ex4.c", ctx->hMin, &(ctx->hMin), NULL));
  PetscCall(PetscOptionsReal("-abstol", "Absolute stopping criterion for ADMM", "ex4.c", ctx->abstol, &(ctx->abstol), NULL));
  PetscCall(PetscOptionsReal("-reltol", "Relative stopping criterion for ADMM", "ex4.c", ctx->reltol, &(ctx->reltol), NULL));
  PetscCall(PetscOptionsBool("-taylor", "Flag for Taylor test. Default is true.", "ex4.c", ctx->taylor, &(ctx->taylor), NULL));
  PetscCall(PetscOptionsBool("-use_admm", "Use the ADMM solver in this example.", "ex4.c", ctx->use_admm, &(ctx->use_admm), NULL));
  PetscCall(PetscOptionsEnum("-p", "Norm type.", "ex4.c", NormTypes, (PetscEnum)ctx->p, (PetscEnum *)&(ctx->p), NULL));
  PetscOptionsEnd();
  /* Creating random ctx */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &(ctx->rctx)));
  PetscCall(PetscRandomSetFromOptions(ctx->rctx));
  PetscCall(CreateMatrix(ctx));
  PetscCall(CreateRHS(ctx));
  PetscCall(SetupWorkspace(ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyContext(UserCtx *ctx)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&((*ctx)->F)));
  PetscCall(MatDestroy(&((*ctx)->W)));
  PetscCall(MatDestroy(&((*ctx)->Hm)));
  PetscCall(MatDestroy(&((*ctx)->Hr)));
  PetscCall(VecDestroy(&((*ctx)->d)));
  for (i = 0; i < NWORKLEFT; i++) PetscCall(VecDestroy(&((*ctx)->workLeft[i])));
  for (i = 0; i < NWORKRIGHT; i++) PetscCall(VecDestroy(&((*ctx)->workRight[i])));
  PetscCall(PetscRandomDestroy(&((*ctx)->rctx)));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(0);
}

/* compute (1/2) * ||F x - d||^2 */
static PetscErrorCode ObjectiveMisfit(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx ctx = (UserCtx)_ctx;
  Vec     y;

  PetscFunctionBegin;
  y = ctx->workLeft[0];
  PetscCall(MatMult(ctx->F, x, y));
  PetscCall(VecAXPY(y, -1., ctx->d));
  PetscCall(VecDot(y, y, J));
  *J *= 0.5;
  PetscFunctionReturn(0);
}

/* compute V = FTFx - FTd */
static PetscErrorCode GradientMisfit(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx ctx = (UserCtx)_ctx;
  Vec     FTFx, FTd;

  PetscFunctionBegin;
  /* work1 is A^T Ax, work2 is Ab, W is A^T A*/
  FTFx = ctx->workRight[0];
  FTd  = ctx->workRight[1];
  PetscCall(MatMult(ctx->W, x, FTFx));
  PetscCall(MatMultTranspose(ctx->F, ctx->d, FTd));
  PetscCall(VecWAXPY(V, -1., FTd, FTFx));
  PetscFunctionReturn(0);
}

/* returns FTF */
static PetscErrorCode HessianMisfit(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx ctx = (UserCtx)_ctx;

  PetscFunctionBegin;
  if (H != ctx->W) PetscCall(MatCopy(ctx->W, H, DIFFERENT_NONZERO_PATTERN));
  if (Hpre != ctx->W) PetscCall(MatCopy(ctx->W, Hpre, DIFFERENT_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

/* computes augment Lagrangian objective (with scaled dual):
 * 0.5 * ||F x - d||^2  + 0.5 * mu ||x - z + u||^2 */
static PetscErrorCode ObjectiveMisfitADMM(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx   ctx = (UserCtx)_ctx;
  PetscReal mu, workNorm, misfit;
  Vec       z, u, temp;

  PetscFunctionBegin;
  mu   = ctx->mu;
  z    = ctx->workRight[5];
  u    = ctx->workRight[6];
  temp = ctx->workRight[10];
  /* misfit = f(x) */
  PetscCall(ObjectiveMisfit(tao, x, &misfit, _ctx));
  PetscCall(VecCopy(x, temp));
  /* temp = x - z + u */
  PetscCall(VecAXPBYPCZ(temp, -1., 1., 1., z, u));
  /* workNorm = ||x - z + u||^2 */
  PetscCall(VecDot(temp, temp, &workNorm));
  /* augment Lagrangian objective (with scaled dual): f(x) + 0.5 * mu ||x -z + u||^2 */
  *J = misfit + 0.5 * mu * workNorm;
  PetscFunctionReturn(0);
}

/* computes FTFx - FTd  mu*(x - z + u) */
static PetscErrorCode GradientMisfitADMM(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx   ctx = (UserCtx)_ctx;
  PetscReal mu;
  Vec       z, u, temp;

  PetscFunctionBegin;
  mu   = ctx->mu;
  z    = ctx->workRight[5];
  u    = ctx->workRight[6];
  temp = ctx->workRight[10];
  PetscCall(GradientMisfit(tao, x, V, _ctx));
  PetscCall(VecCopy(x, temp));
  /* temp = x - z + u */
  PetscCall(VecAXPBYPCZ(temp, -1., 1., 1., z, u));
  /* V =  FTFx - FTd  mu*(x - z + u) */
  PetscCall(VecAXPY(V, mu, temp));
  PetscFunctionReturn(0);
}

/* returns FTF + diag(mu) */
static PetscErrorCode HessianMisfitADMM(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx ctx = (UserCtx)_ctx;

  PetscFunctionBegin;
  PetscCall(MatCopy(ctx->W, H, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatShift(H, ctx->mu));
  if (Hpre != H) PetscCall(MatCopy(H, Hpre, DIFFERENT_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

/* computes || x ||_p (mult by 0.5 in case of NORM_2) */
static PetscErrorCode ObjectiveRegularization(Tao tao, Vec x, PetscReal *J, void *_ctx)
{
  UserCtx   ctx = (UserCtx)_ctx;
  PetscReal norm;

  PetscFunctionBegin;
  *J = 0;
  PetscCall(VecNorm(x, ctx->p, &norm));
  if (ctx->p == NORM_2) norm = 0.5 * norm * norm;
  *J = ctx->alpha * norm;
  PetscFunctionReturn(0);
}

/* NORM_2 Case: return x
 * NORM_1 Case: x/(|x| + eps)
 * Else: TODO */
static PetscErrorCode GradientRegularization(Tao tao, Vec x, Vec V, void *_ctx)
{
  UserCtx   ctx = (UserCtx)_ctx;
  PetscReal eps = ctx->eps;

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    PetscCall(VecCopy(x, V));
  } else if (ctx->p == NORM_1) {
    PetscCall(VecCopy(x, ctx->workRight[1]));
    PetscCall(VecAbs(ctx->workRight[1]));
    PetscCall(VecShift(ctx->workRight[1], eps));
    PetscCall(VecPointwiseDivide(V, x, ctx->workRight[1]));
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Example only works for NORM_1 and NORM_2");
  PetscFunctionReturn(0);
}

/* NORM_2 Case: returns diag(mu)
 * NORM_1 Case: diag(mu* 1/sqrt(x_i^2 + eps) * (1 - x_i^2/ABS(x_i^2+eps)))  */
static PetscErrorCode HessianRegularization(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx   ctx = (UserCtx)_ctx;
  PetscReal eps = ctx->eps;
  Vec       copy1, copy2, copy3;

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    /* Identity matrix scaled by mu */
    PetscCall(MatZeroEntries(H));
    PetscCall(MatShift(H, ctx->mu));
    if (Hpre != H) {
      PetscCall(MatZeroEntries(Hpre));
      PetscCall(MatShift(Hpre, ctx->mu));
    }
  } else if (ctx->p == NORM_1) {
    /* 1/sqrt(x_i^2 + eps) * (1 - x_i^2/ABS(x_i^2+eps)) */
    copy1 = ctx->workRight[1];
    copy2 = ctx->workRight[2];
    copy3 = ctx->workRight[3];
    /* copy1 : 1/sqrt(x_i^2 + eps) */
    PetscCall(VecCopy(x, copy1));
    PetscCall(VecPow(copy1, 2));
    PetscCall(VecShift(copy1, eps));
    PetscCall(VecSqrtAbs(copy1));
    PetscCall(VecReciprocal(copy1));
    /* copy2:  x_i^2.*/
    PetscCall(VecCopy(x, copy2));
    PetscCall(VecPow(copy2, 2));
    /* copy3: abs(x_i^2 + eps) */
    PetscCall(VecCopy(x, copy3));
    PetscCall(VecPow(copy3, 2));
    PetscCall(VecShift(copy3, eps));
    PetscCall(VecAbs(copy3));
    /* copy2: 1 - x_i^2/abs(x_i^2 + eps) */
    PetscCall(VecPointwiseDivide(copy2, copy2, copy3));
    PetscCall(VecScale(copy2, -1.));
    PetscCall(VecShift(copy2, 1.));
    PetscCall(VecAXPY(copy1, 1., copy2));
    PetscCall(VecScale(copy1, ctx->mu));
    PetscCall(MatZeroEntries(H));
    PetscCall(MatDiagonalSet(H, copy1, INSERT_VALUES));
    if (Hpre != H) {
      PetscCall(MatZeroEntries(Hpre));
      PetscCall(MatDiagonalSet(Hpre, copy1, INSERT_VALUES));
    }
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Example only works for NORM_1 and NORM_2");
  PetscFunctionReturn(0);
}

/* NORM_2 Case: 0.5 || x ||_2 + 0.5 * mu * ||x + u - z||^2
 * Else : || x ||_2 + 0.5 * mu * ||x + u - z||^2 */
static PetscErrorCode ObjectiveRegularizationADMM(Tao tao, Vec z, PetscReal *J, void *_ctx)
{
  UserCtx   ctx = (UserCtx)_ctx;
  PetscReal mu, workNorm, reg;
  Vec       x, u, temp;

  PetscFunctionBegin;
  mu   = ctx->mu;
  x    = ctx->workRight[4];
  u    = ctx->workRight[6];
  temp = ctx->workRight[10];
  PetscCall(ObjectiveRegularization(tao, z, &reg, _ctx));
  PetscCall(VecCopy(z, temp));
  /* temp = x + u -z */
  PetscCall(VecAXPBYPCZ(temp, 1., 1., -1., x, u));
  /* workNorm = ||x + u - z ||^2 */
  PetscCall(VecDot(temp, temp, &workNorm));
  *J = reg + 0.5 * mu * workNorm;
  PetscFunctionReturn(0);
}

/* NORM_2 Case: x - mu*(x + u - z)
 * NORM_1 Case: x/(|x| + eps) - mu*(x + u - z)
 * Else: TODO */
static PetscErrorCode GradientRegularizationADMM(Tao tao, Vec z, Vec V, void *_ctx)
{
  UserCtx   ctx = (UserCtx)_ctx;
  PetscReal mu;
  Vec       x, u, temp;

  PetscFunctionBegin;
  mu   = ctx->mu;
  x    = ctx->workRight[4];
  u    = ctx->workRight[6];
  temp = ctx->workRight[10];
  PetscCall(GradientRegularization(tao, z, V, _ctx));
  PetscCall(VecCopy(z, temp));
  /* temp = x + u -z */
  PetscCall(VecAXPBYPCZ(temp, 1., 1., -1., x, u));
  PetscCall(VecAXPY(V, -mu, temp));
  PetscFunctionReturn(0);
}

/* NORM_2 Case: returns diag(mu)
 * NORM_1 Case: FTF + diag(mu) */
static PetscErrorCode HessianRegularizationADMM(Tao tao, Vec x, Mat H, Mat Hpre, void *_ctx)
{
  UserCtx ctx = (UserCtx)_ctx;

  PetscFunctionBegin;
  if (ctx->p == NORM_2) {
    /* Identity matrix scaled by mu */
    PetscCall(MatZeroEntries(H));
    PetscCall(MatShift(H, ctx->mu));
    if (Hpre != H) {
      PetscCall(MatZeroEntries(Hpre));
      PetscCall(MatShift(Hpre, ctx->mu));
    }
  } else if (ctx->p == NORM_1) {
    PetscCall(HessianMisfit(tao, x, H, Hpre, (void *)ctx));
    PetscCall(MatShift(H, ctx->mu));
    if (Hpre != H) PetscCall(MatShift(Hpre, ctx->mu));
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_OUTOFRANGE, "Example only works for NORM_1 and NORM_2");
  PetscFunctionReturn(0);
}

/* NORM_2 Case : (1/2) * ||F x - d||^2 + 0.5 * || x ||_p
*  NORM_1 Case : (1/2) * ||F x - d||^2 + || x ||_p */
static PetscErrorCode ObjectiveComplete(Tao tao, Vec x, PetscReal *J, void *ctx)
{
  PetscReal Jm, Jr;

  PetscFunctionBegin;
  PetscCall(ObjectiveMisfit(tao, x, &Jm, ctx));
  PetscCall(ObjectiveRegularization(tao, x, &Jr, ctx));
  *J = Jm + Jr;
  PetscFunctionReturn(0);
}

/* NORM_2 Case: FTFx - FTd + x
 * NORM_1 Case: FTFx - FTd + x/(|x| + eps) */
static PetscErrorCode GradientComplete(Tao tao, Vec x, Vec V, void *ctx)
{
  UserCtx cntx = (UserCtx)ctx;

  PetscFunctionBegin;
  PetscCall(GradientMisfit(tao, x, cntx->workRight[2], ctx));
  PetscCall(GradientRegularization(tao, x, cntx->workRight[3], ctx));
  PetscCall(VecWAXPY(V, 1, cntx->workRight[2], cntx->workRight[3]));
  PetscFunctionReturn(0);
}

/* NORM_2 Case: diag(mu) + FTF
 * NORM_1 Case: diag(mu* 1/sqrt(x_i^2 + eps) * (1 - x_i^2/ABS(x_i^2+eps))) + FTF  */
static PetscErrorCode HessianComplete(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  Mat tempH;

  PetscFunctionBegin;
  PetscCall(MatDuplicate(H, MAT_SHARE_NONZERO_PATTERN, &tempH));
  PetscCall(HessianMisfit(tao, x, H, H, ctx));
  PetscCall(HessianRegularization(tao, x, tempH, tempH, ctx));
  PetscCall(MatAXPY(H, 1., tempH, DIFFERENT_NONZERO_PATTERN));
  if (Hpre != H) PetscCall(MatCopy(H, Hpre, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatDestroy(&tempH));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolveADMM(UserCtx ctx, Vec x)
{
  PetscInt  i;
  PetscReal u_norm, r_norm, s_norm, primal, dual, x_norm, z_norm;
  Tao       tao1, tao2;
  Vec       xk, z, u, diff, zold, zdiff, temp;
  PetscReal mu;

  PetscFunctionBegin;
  xk    = ctx->workRight[4];
  z     = ctx->workRight[5];
  u     = ctx->workRight[6];
  diff  = ctx->workRight[7];
  zold  = ctx->workRight[8];
  zdiff = ctx->workRight[9];
  temp  = ctx->workRight[11];
  mu    = ctx->mu;
  PetscCall(VecSet(u, 0.));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao1));
  PetscCall(TaoSetType(tao1, TAONLS));
  PetscCall(TaoSetObjective(tao1, ObjectiveMisfitADMM, (void *)ctx));
  PetscCall(TaoSetGradient(tao1, NULL, GradientMisfitADMM, (void *)ctx));
  PetscCall(TaoSetHessian(tao1, ctx->Hm, ctx->Hm, HessianMisfitADMM, (void *)ctx));
  PetscCall(VecSet(xk, 0.));
  PetscCall(TaoSetSolution(tao1, xk));
  PetscCall(TaoSetOptionsPrefix(tao1, "misfit_"));
  PetscCall(TaoSetFromOptions(tao1));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao2));
  if (ctx->p == NORM_2) {
    PetscCall(TaoSetType(tao2, TAONLS));
    PetscCall(TaoSetObjective(tao2, ObjectiveRegularizationADMM, (void *)ctx));
    PetscCall(TaoSetGradient(tao2, NULL, GradientRegularizationADMM, (void *)ctx));
    PetscCall(TaoSetHessian(tao2, ctx->Hr, ctx->Hr, HessianRegularizationADMM, (void *)ctx));
  }
  PetscCall(VecSet(z, 0.));
  PetscCall(TaoSetSolution(tao2, z));
  PetscCall(TaoSetOptionsPrefix(tao2, "reg_"));
  PetscCall(TaoSetFromOptions(tao2));

  for (i = 0; i < ctx->iter; i++) {
    PetscCall(VecCopy(z, zold));
    PetscCall(TaoSolve(tao1)); /* Updates xk */
    if (ctx->p == NORM_1) {
      PetscCall(VecWAXPY(temp, 1., xk, u));
      PetscCall(TaoSoftThreshold(temp, -ctx->alpha / mu, ctx->alpha / mu, z));
    } else {
      PetscCall(TaoSolve(tao2)); /* Update zk */
    }
    /* u = u + xk -z */
    PetscCall(VecAXPBYPCZ(u, 1., -1., 1., xk, z));
    /* r_norm : norm(x-z) */
    PetscCall(VecWAXPY(diff, -1., z, xk));
    PetscCall(VecNorm(diff, NORM_2, &r_norm));
    /* s_norm : norm(-mu(z-zold)) */
    PetscCall(VecWAXPY(zdiff, -1., zold, z));
    PetscCall(VecNorm(zdiff, NORM_2, &s_norm));
    s_norm = s_norm * mu;
    /* primal : sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z))*/
    PetscCall(VecNorm(xk, NORM_2, &x_norm));
    PetscCall(VecNorm(z, NORM_2, &z_norm));
    primal = PetscSqrtReal(ctx->n) * ctx->abstol + ctx->reltol * PetscMax(x_norm, z_norm);
    /* Duality : sqrt(n)*ABSTOL + RELTOL*norm(mu*u)*/
    PetscCall(VecNorm(u, NORM_2, &u_norm));
    dual = PetscSqrtReal(ctx->n) * ctx->abstol + ctx->reltol * u_norm * mu;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao1), "Iter %" PetscInt_FMT " : ||x-z||: %g, mu*||z-zold||: %g\n", i, (double)r_norm, (double)s_norm));
    if (r_norm < primal && s_norm < dual) break;
  }
  PetscCall(VecCopy(xk, x));
  PetscCall(TaoDestroy(&tao1));
  PetscCall(TaoDestroy(&tao2));
  PetscFunctionReturn(0);
}

/* Second order Taylor remainder convergence test */
static PetscErrorCode TaylorTest(UserCtx ctx, Tao tao, Vec x, PetscReal *C)
{
  PetscReal  h, J, temp;
  PetscInt   i, j;
  PetscInt   numValues;
  PetscReal  Jx, Jxhat_comp, Jxhat_pred;
  PetscReal *Js, *hs;
  PetscReal  gdotdx;
  PetscReal  minrate = PETSC_MAX_REAL;
  MPI_Comm   comm    = PetscObjectComm((PetscObject)x);
  Vec        g, dx, xhat;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(x, &g));
  PetscCall(VecDuplicate(x, &xhat));
  /* choose a perturbation direction */
  PetscCall(VecDuplicate(x, &dx));
  PetscCall(VecSetRandom(dx, ctx->rctx));
  /* evaluate objective at x: J(x) */
  PetscCall(TaoComputeObjective(tao, x, &Jx));
  /* evaluate gradient at x, save in vector g */
  PetscCall(TaoComputeGradient(tao, x, g));
  PetscCall(VecDot(g, dx, &gdotdx));

  for (numValues = 0, h = ctx->hStart; h >= ctx->hMin; h *= ctx->hFactor) numValues++;
  PetscCall(PetscCalloc2(numValues, &Js, numValues, &hs));
  for (i = 0, h = ctx->hStart; h >= ctx->hMin; h *= ctx->hFactor, i++) {
    PetscCall(VecWAXPY(xhat, h, dx, x));
    PetscCall(TaoComputeObjective(tao, xhat, &Jxhat_comp));
    /* J(\hat(x)) \approx J(x) + g^T (xhat - x) = J(x) + h * g^T dx */
    Jxhat_pred = Jx + h * gdotdx;
    /* Vector to dJdm scalar? Dot?*/
    J = PetscAbsReal(Jxhat_comp - Jxhat_pred);
    PetscCall(PetscPrintf(comm, "J(xhat): %g, predicted: %g, diff %g\n", (double)Jxhat_comp, (double)Jxhat_pred, (double)J));
    Js[i] = J;
    hs[i] = h;
  }
  for (j = 1; j < numValues; j++) {
    temp = PetscLogReal(Js[j] / Js[j - 1]) / PetscLogReal(hs[j] / hs[j - 1]);
    PetscCall(PetscPrintf(comm, "Convergence rate step %" PetscInt_FMT ": %g\n", j - 1, (double)temp));
    minrate = PetscMin(minrate, temp);
  }
  /* If O is not ~2, then the test is wrong */
  PetscCall(PetscFree2(Js, hs));
  *C = minrate;
  PetscCall(VecDestroy(&dx));
  PetscCall(VecDestroy(&xhat));
  PetscCall(VecDestroy(&g));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  UserCtx ctx;
  Tao     tao;
  Vec     x;
  Mat     H;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscNew(&ctx));
  PetscCall(ConfigureContext(ctx));
  /* Define two functions that could pass as objectives to TaoSetObjective(): one
   * for the misfit component, and one for the regularization component */
  /* ObjectiveMisfit() and ObjectiveRegularization() */

  /* Define a single function that calls both components adds them together: the complete objective,
   * in the absence of a Tao implementation that handles separability */
  /* ObjectiveComplete() */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAONM));
  PetscCall(TaoSetObjective(tao, ObjectiveComplete, (void *)ctx));
  PetscCall(TaoSetGradient(tao, NULL, GradientComplete, (void *)ctx));
  PetscCall(MatDuplicate(ctx->W, MAT_SHARE_NONZERO_PATTERN, &H));
  PetscCall(TaoSetHessian(tao, H, H, HessianComplete, (void *)ctx));
  PetscCall(MatCreateVecs(ctx->F, NULL, &x));
  PetscCall(VecSet(x, 0.));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetFromOptions(tao));
  if (ctx->use_admm) PetscCall(TaoSolveADMM(ctx, x));
  else PetscCall(TaoSolve(tao));
  /* examine solution */
  PetscCall(VecViewFromOptions(x, NULL, "-view_sol"));
  if (ctx->taylor) {
    PetscReal rate;
    PetscCall(TaylorTest(ctx, tao, x, &rate));
  }
  PetscCall(MatDestroy(&H));
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(DestroyContext(&ctx));
  PetscCall(PetscFinalize());
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
