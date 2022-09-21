
/*
 Partial differential equation

   d  (1 + e*sine(2*pi*k*x)) d u = 1, 0 < x < 1,
   --                        ---
   dx                        dx
with boundary conditions

   u = 0 for x = 0, x = 1

   This uses multigrid to solve the linear system

*/

static char help[] = "Solves 1D variable coefficient Laplacian using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

static PetscErrorCode ComputeMatrix(KSP, Mat, Mat, void *);
static PetscErrorCode ComputeRHS(KSP, Vec, void *);

typedef struct {
  PetscInt    k;
  PetscScalar e;
} AppCtx;

int main(int argc, char **argv)
{
  KSP       ksp;
  DM        da;
  AppCtx    user;
  Mat       A;
  Vec       b, b2;
  Vec       x;
  PetscReal nrm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  user.k = 1;
  user.e = .99;
  PetscCall(PetscOptionsGetInt(NULL, 0, "-k", &user.k, 0));
  PetscCall(PetscOptionsGetScalar(NULL, 0, "-e", &user.e, 0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 128, 1, 1, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(KSPSetDM(ksp, da));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, &user));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrix, &user));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, NULL, NULL));

  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(KSPGetSolution(ksp, &x));
  PetscCall(KSPGetRhs(ksp, &b));
  PetscCall(VecDuplicate(b, &b2));
  PetscCall(MatMult(A, x, b2));
  PetscCall(VecAXPY(b2, -1.0, b));
  PetscCall(VecNorm(b2, NORM_MAX, &nrm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Residual norm %g\n", (double)nrm));

  PetscCall(VecDestroy(&b2));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  PetscInt    mx, idx[2];
  PetscScalar h, v[2];
  DM          da;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &da));
  PetscCall(DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  h = 1.0 / ((mx - 1));
  PetscCall(VecSet(b, h));
  idx[0] = 0;
  idx[1] = mx - 1;
  v[0] = v[1] = 0.0;
  PetscCall(VecSetValues(b, 2, idx, v, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx)
{
  AppCtx     *user = (AppCtx *)ctx;
  PetscInt    i, mx, xm, xs;
  PetscScalar v[3], h, xlow, xhigh;
  MatStencil  row, col[3];
  DM          da;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &da));
  PetscCall(DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, 0, 0, &xm, 0, 0));
  h = 1.0 / (mx - 1);

  for (i = xs; i < xs + xm; i++) {
    row.i = i;
    if (i == 0 || i == mx - 1) {
      v[0] = 2.0 / h;
      PetscCall(MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES));
    } else {
      xlow     = h * (PetscReal)i - .5 * h;
      xhigh    = xlow + h;
      v[0]     = (-1.0 - user->e * PetscSinScalar(2.0 * PETSC_PI * user->k * xlow)) / h;
      col[0].i = i - 1;
      v[1]     = (2.0 + user->e * PetscSinScalar(2.0 * PETSC_PI * user->k * xlow) + user->e * PetscSinScalar(2.0 * PETSC_PI * user->k * xhigh)) / h;
      col[1].i = row.i;
      v[2]     = (-1.0 - user->e * PetscSinScalar(2.0 * PETSC_PI * user->k * xhigh)) / h;
      col[2].i = i + 1;
      PetscCall(MatSetValuesStencil(jac, 1, &row, 3, col, v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type unpreconditioned -ksp_view -pc_mg_type full
      requires: !single

   test:
      suffix: 2
      nsize: 2
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -ksp_monitor_short -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type unpreconditioned -ksp_view -pc_mg_type full
      requires: !single

TEST*/
