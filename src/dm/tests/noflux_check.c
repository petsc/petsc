static char help[] = "Check to see of DM_BOUNDARY_MIRROR works in 3D for DMDA with star stencil\n";

#include "petscdmda.h"

/* Contributed by Gourav Kumbhojkar */

PetscErrorCode globalKMat_3d(Mat K, DMDALocalInfo info)
{
  MatStencil  row, col[7];
  PetscScalar vals[7];
  PetscInt    ncols;

  PetscFunctionBeginUser;
  for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
    for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
      for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        ncols = 0;
        row.i = i;
        row.j = j;
        row.k = k;

        col[0].i      = i;
        col[0].j      = j;
        col[0].k      = k;
        vals[ncols++] = -6.; //ncols=1

        col[ncols].i  = i - 1;
        col[ncols].j  = j;
        col[ncols].k  = k;
        vals[ncols++] = 1.; //ncols=2

        col[ncols].i  = i + 1;
        col[ncols].j  = j;
        col[ncols].k  = k;
        vals[ncols++] = 1.; //ncols=3

        col[ncols].i  = i;
        col[ncols].j  = j - 1;
        col[ncols].k  = k;
        vals[ncols++] = 1.; //ncols=4

        col[ncols].i  = i;
        col[ncols].j  = j + 1;
        col[ncols].k  = k;
        vals[ncols++] = 1.; //ncols=5

        col[ncols].i  = i;
        col[ncols].j  = j;
        col[ncols].k  = k + 1;
        vals[ncols++] = 1.; //ncols=6

        col[ncols].i  = i;
        col[ncols].j  = j;
        col[ncols].k  = k - 1;
        vals[ncols++] = 1.; //ncols=7

        PetscCall(MatSetValuesStencil(K, 1, &row, ncols, col, vals, ADD_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode globalKMat_2d(Mat K, DMDALocalInfo info)
{
  MatStencil  row, col[5];
  PetscScalar vals[5];
  PetscInt    ncols;

  PetscFunctionBeginUser;
  for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
    for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
      ncols = 0;
      row.i = i;
      row.j = j;

      col[0].i      = i;
      col[0].j      = j;
      vals[ncols++] = -4.; //ncols=1

      col[ncols].i  = i - 1;
      col[ncols].j  = j;
      vals[ncols++] = 1.; //ncols=2

      col[ncols].i  = i;
      col[ncols].j  = j - 1;
      vals[ncols++] = 1.; //ncols=3

      col[ncols].i  = i + 1;
      col[ncols].j  = j;
      vals[ncols++] = 1.; //ncols=4

      col[ncols].i  = i;
      col[ncols].j  = j + 1;
      vals[ncols++] = 1.; //ncols=5

      PetscCall(MatSetValuesStencil(K, 1, &row, ncols, col, vals, ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                     da3d, da2d;
  DMDALocalInfo          info3d, info2d;
  Mat                    K3d, K2d;
  PetscInt               ne, num_pts;
  ISLocalToGlobalMapping ltgm3d, ltgm2d;
  Vec                    row2d, row3d;
  PetscReal              norm2d, norm3d;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  ne      = 8;
  num_pts = ne + 1;

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_MIRROR, DM_BOUNDARY_MIRROR, DMDA_STENCIL_STAR, num_pts, num_pts + 1, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da2d));
  PetscCall(DMSetUp(da2d));
  PetscCall(DMDAGetLocalInfo(da2d, &info2d));
  PetscCall(DMCreateMatrix(da2d, &K2d));
  PetscCall(DMGetLocalToGlobalMapping(da2d, &ltgm2d));
  PetscCall(ISLocalToGlobalMappingView(ltgm2d, PETSC_VIEWER_STDOUT_WORLD));
  //PetscFinalize();
  PetscCall(globalKMat_2d(K2d, info2d));
  PetscCall(MatView(K2d, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatCreateVecs(K2d, &row2d, NULL));

  PetscCall(MatGetRowSum(K2d, row2d));
  PetscCall(VecNorm(row2d, NORM_2, &norm2d));

  PetscCheck(norm2d == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "2D atrix row sum should be zero");
  PetscCall(VecDestroy(&row2d));
  PetscCall(MatDestroy(&K2d));
  PetscCall(DMDestroy(&da2d));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_MIRROR, DM_BOUNDARY_MIRROR, DM_BOUNDARY_MIRROR, DMDA_STENCIL_STAR, num_pts, num_pts + 1, num_pts + 2, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &da3d));
  PetscCall(DMSetUp(da3d));
  PetscCall(DMCreateMatrix(da3d, &K3d));
  PetscCall(DMDAGetLocalInfo(da3d, &info3d));
  PetscCall(DMGetLocalToGlobalMapping(da3d, &ltgm3d));
  PetscCall(ISLocalToGlobalMappingView(ltgm3d, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(globalKMat_3d(K3d, info3d));
  PetscCall(MatView(K3d, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatCreateVecs(K3d, &row3d, NULL));
  PetscCall(MatGetRowSum(K3d, row3d));
  PetscCall(VecNorm(row3d, NORM_2, &norm3d));
  PetscCheck(norm3d == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "3D atrix row sum should be zero");
  PetscCall(VecDestroy(&row3d));

  PetscCall(DMDestroy(&da3d));
  PetscCall(MatDestroy(&K3d));
  return PetscFinalize();
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2

   test:
      suffix: 4
      nsize: 4

   test:
      suffix: 8
      nsize: 8

TEST*/
