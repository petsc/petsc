
/*
Laplacian in 3D. Modeled by the partial differential equation

   - Laplacian u = 1,0 < x,y,z < 1,

with boundary conditions

   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.

   This uses multigrid to solve the linear system

   See src/snes/tutorials/ex50.c

   Can also be run with -pc_type exotic -ksp_type fgmres

*/

static char help[] = "Solves 3D Laplacian using multigrid.\n\n";

#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>

extern PetscErrorCode ComputeMatrix(KSP, Mat, Mat, void *);
extern PetscErrorCode ComputeRHS(KSP, Vec, void *);
extern PetscErrorCode ComputeInitialGuess(KSP, Vec, void *);

int main(int argc, char **argv)
{
  KSP       ksp;
  PetscReal norm;
  DM        da;
  Vec       x, b, r;
  Mat       A;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 7, 7, 7, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(KSPSetDM(ksp, da));
  PetscCall(KSPSetComputeInitialGuess(ksp, ComputeInitialGuess, NULL));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, NULL));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrix, NULL));
  PetscCall(DMDestroy(&da));

  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, NULL, NULL));
  PetscCall(KSPGetSolution(ksp, &x));
  PetscCall(KSPGetRhs(ksp, &b));
  PetscCall(VecDuplicate(b, &r));
  PetscCall(KSPGetOperators(ksp, &A, NULL));

  PetscCall(MatMult(A, x, r));
  PetscCall(VecAXPY(r, -1.0, b));
  PetscCall(VecNorm(r, NORM_2, &norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Residual norm %g\n", (double)norm));

  PetscCall(VecDestroy(&r));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  PetscInt       i, j, k, mx, my, mz, xm, ym, zm, xs, ys, zs;
  DM             dm;
  PetscScalar    Hx, Hy, Hz, HxHydHz, HyHzdHx, HxHzdHy;
  PetscScalar ***barray;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMDAGetInfo(dm, 0, &mx, &my, &mz, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  Hx      = 1.0 / (PetscReal)(mx - 1);
  Hy      = 1.0 / (PetscReal)(my - 1);
  Hz      = 1.0 / (PetscReal)(mz - 1);
  HxHydHz = Hx * Hy / Hz;
  HxHzdHy = Hx * Hz / Hy;
  HyHzdHx = Hy * Hz / Hx;
  PetscCall(DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCall(DMDAVecGetArray(dm, b, &barray));

  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        if (i == 0 || j == 0 || k == 0 || i == mx - 1 || j == my - 1 || k == mz - 1) {
          barray[k][j][i] = 2.0 * (HxHydHz + HxHzdHy + HyHzdHx);
        } else {
          barray[k][j][i] = Hx * Hy * Hz;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(dm, b, &barray));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeInitialGuess(KSP ksp, Vec b, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(b, 0));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp, Mat jac, Mat B, void *ctx)
{
  DM          da;
  PetscInt    i, j, k, mx, my, mz, xm, ym, zm, xs, ys, zs;
  PetscScalar v[7], Hx, Hy, Hz, HxHydHz, HyHzdHx, HxHzdHy;
  MatStencil  row, col[7];

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &da));
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, &mz, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  Hx      = 1.0 / (PetscReal)(mx - 1);
  Hy      = 1.0 / (PetscReal)(my - 1);
  Hz      = 1.0 / (PetscReal)(mz - 1);
  HxHydHz = Hx * Hy / Hz;
  HxHzdHy = Hx * Hz / Hy;
  HyHzdHx = Hy * Hz / Hx;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        row.i = i;
        row.j = j;
        row.k = k;
        if (i == 0 || j == 0 || k == 0 || i == mx - 1 || j == my - 1 || k == mz - 1) {
          v[0] = 2.0 * (HxHydHz + HxHzdHy + HyHzdHx);
          PetscCall(MatSetValuesStencil(B, 1, &row, 1, &row, v, INSERT_VALUES));
        } else {
          v[0]     = -HxHydHz;
          col[0].i = i;
          col[0].j = j;
          col[0].k = k - 1;
          v[1]     = -HxHzdHy;
          col[1].i = i;
          col[1].j = j - 1;
          col[1].k = k;
          v[2]     = -HyHzdHx;
          col[2].i = i - 1;
          col[2].j = j;
          col[2].k = k;
          v[3]     = 2.0 * (HxHydHz + HxHzdHy + HyHzdHx);
          col[3].i = row.i;
          col[3].j = row.j;
          col[3].k = row.k;
          v[4]     = -HyHzdHx;
          col[4].i = i + 1;
          col[4].j = j;
          col[4].k = k;
          v[5]     = -HxHzdHy;
          col[5].i = i;
          col[5].j = j + 1;
          col[5].k = k;
          v[6]     = -HxHydHz;
          col[6].i = i;
          col[6].j = j;
          col[6].k = k + 1;
          PetscCall(MatSetValuesStencil(B, 1, &row, 7, col, v, INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 4
      args: -pc_type exotic -ksp_monitor_short -ksp_type fgmres -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 1 -mg_levels_pc_type bjacobi
      output_file: output/ex45_1.out

   test:
      suffix: 2
      nsize: 4
      args: -ksp_monitor_short -da_grid_x 21 -da_grid_y 21 -da_grid_z 21 -pc_type mg -pc_mg_levels 3 -mg_levels_ksp_type richardson -mg_levels_ksp_max_it 1 -mg_levels_pc_type bjacobi

   test:
      suffix: telescope
      nsize: 4
      args: -ksp_type fgmres -ksp_monitor_short -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -pc_mg_levels 2 -da_grid_x 65 -da_grid_y 65 -da_grid_z 65 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_ignore_kspcomputeoperators -mg_coarse_pc_telescope_reduction_factor 4 -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_galerkin pmat -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_pc_type jacobi -mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_ksp_type richardson -ksp_rtol 1.0e-4

   test:
      suffix: telescope_2
      nsize: 4
      args: -ksp_type fgmres -ksp_monitor_short -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -pc_mg_levels 2 -da_grid_x 65 -da_grid_y 65 -da_grid_z 65 -mg_coarse_pc_type telescope -mg_coarse_pc_telescope_reduction_factor 2 -mg_coarse_telescope_pc_type mg -mg_coarse_telescope_pc_mg_galerkin pmat -mg_coarse_telescope_pc_mg_levels 3 -mg_coarse_telescope_mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_pc_type jacobi -mg_levels_ksp_type richardson -mg_coarse_telescope_mg_levels_ksp_type richardson -ksp_rtol 1.0e-4

TEST*/
