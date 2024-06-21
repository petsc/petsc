const char help[] = "Test TAOLMVM on a least-squares problem";

#include <petsctao.h>
#include <petscdevice.h>

typedef struct _n_AppCtx {
  Mat A;
  Vec b;
  Vec r;
} AppCtx;

static PetscErrorCode LSObjAndGrad(Tao tao, Vec x, PetscReal *obj, Vec g, void *_ctx)
{
  PetscFunctionBegin;
  AppCtx *ctx = (AppCtx *)_ctx;
  PetscCall(VecAXPBY(ctx->r, -1.0, 0.0, ctx->b));
  PetscCall(MatMultAdd(ctx->A, x, ctx->r, ctx->r));
  PetscCall(VecDotRealPart(ctx->r, ctx->r, obj));
  *obj *= 0.5;
  PetscCall(MatMultTranspose(ctx->A, ctx->r, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MPI_Comm  comm = PETSC_COMM_WORLD;
  AppCtx    ctx;
  Vec       sol;
  PetscBool flg, cuda = PETSC_FALSE;

  PetscInt M = 10;
  PetscInt N = 10;
  PetscOptionsBegin(comm, "", help, "TAO");
  PetscCall(PetscOptionsInt("-m", "data size", NULL, M, &M, NULL));
  PetscCall(PetscOptionsInt("-n", "data size", NULL, N, &N, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-cuda", &cuda, &flg));
  PetscOptionsEnd();

  if (cuda) {
    VecType vec_type;
    PetscCall(VecCreateSeqCUDA(comm, N, &ctx.b));
    PetscCall(VecGetType(ctx.b, &vec_type));
    PetscCall(MatCreateDenseFromVecType(comm, vec_type, M, N, PETSC_DECIDE, PETSC_DECIDE, -1, NULL, &ctx.A));
    PetscCall(MatCreateVecs(ctx.A, &sol, NULL));
  } else {
    PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &ctx.A));
    PetscCall(MatCreateVecs(ctx.A, &sol, &ctx.b));
  }
  PetscCall(VecDuplicate(ctx.b, &ctx.r));
  PetscCall(VecZeroEntries(sol));

  PetscRandom rand;
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatSetRandom(ctx.A, rand));
  PetscCall(VecSetRandom(ctx.b, rand));
  PetscCall(PetscRandomDestroy(&rand));

  Tao tao;
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetSolution(tao, sol));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, LSObjAndGrad, &ctx));
  PetscCall(TaoSetType(tao, TAOLMVM));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoDestroy(&tao));

  PetscCall(VecDestroy(&ctx.r));
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&ctx.b));
  PetscCall(MatDestroy(&ctx.A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !__float128 !single !defined(PETSC_USE_64BIT_INDICES)

  test:
    suffix: 0
    args: -tao_monitor -tao_ls_gtol 1.e-6 -tao_view -tao_lmvm_mat_lmvm_hist_size 20 -tao_ls_type more-thuente -tao_lmvm_mat_lmvm_scale_type none -tao_lmvm_mat_type lmvmbfgs

  test:
    suffix: 1
    args: -tao_monitor -tao_ls_gtol 1.e-6 -tao_view -tao_lmvm_mat_lmvm_hist_size 20 -tao_ls_type more-thuente -tao_lmvm_mat_lmvm_scale_type none -tao_lmvm_mat_type lmvmdbfgs

  test:
    suffix: 2
    args: -tao_monitor -tao_ls_gtol 1.e-6 -tao_view -tao_lmvm_mat_lmvm_hist_size 20 -tao_ls_type more-thuente -tao_lmvm_mat_type lmvmdbfgs -tao_lmvm_mat_lmvm_scale_type none -tao_lmvm_mat_lbfgs_type {{inplace reorder}}

TEST*/
