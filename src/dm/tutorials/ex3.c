
static char help[] = "Tests DMCreateInterpolation() for nonuniform DMDA coordinates.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode SetCoordinates1d(DM da)
{
  PetscInt     i, start, m;
  Vec          local, global;
  PetscScalar *coors, *coorslocal;
  DM           cda;

  PetscFunctionBeginUser;
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinates(da, &global));
  PetscCall(DMGetCoordinatesLocal(da, &local));
  PetscCall(DMDAVecGetArray(cda, global, &coors));
  PetscCall(DMDAVecGetArrayRead(cda, local, &coorslocal));
  PetscCall(DMDAGetCorners(cda, &start, 0, 0, &m, 0, 0));
  for (i = start; i < start + m; i++) {
    if (i % 2) coors[i] = coorslocal[i - 1] + .1 * (coorslocal[i + 1] - coorslocal[i - 1]);
  }
  PetscCall(DMDAVecRestoreArray(cda, global, &coors));
  PetscCall(DMDAVecRestoreArrayRead(cda, local, &coorslocal));
  PetscCall(DMGlobalToLocalBegin(cda, global, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(cda, global, INSERT_VALUES, local));
  PetscFunctionReturn(0);
}

PetscErrorCode SetCoordinates2d(DM da)
{
  PetscInt     i, j, mstart, m, nstart, n;
  Vec          local, global;
  DMDACoor2d **coors, **coorslocal;
  DM           cda;

  PetscFunctionBeginUser;
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinates(da, &global));
  PetscCall(DMGetCoordinatesLocal(da, &local));
  PetscCall(DMDAVecGetArray(cda, global, &coors));
  PetscCall(DMDAVecGetArrayRead(cda, local, &coorslocal));
  PetscCall(DMDAGetCorners(cda, &mstart, &nstart, 0, &m, &n, 0));
  for (i = mstart; i < mstart + m; i++) {
    for (j = nstart; j < nstart + n; j++) {
      if (i % 2) coors[j][i].x = coorslocal[j][i - 1].x + .1 * (coorslocal[j][i + 1].x - coorslocal[j][i - 1].x);
      if (j % 2) coors[j][i].y = coorslocal[j - 1][i].y + .3 * (coorslocal[j + 1][i].y - coorslocal[j - 1][i].y);
    }
  }
  PetscCall(DMDAVecRestoreArray(cda, global, &coors));
  PetscCall(DMDAVecRestoreArrayRead(cda, local, &coorslocal));

  PetscCall(DMGlobalToLocalBegin(cda, global, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(cda, global, INSERT_VALUES, local));
  PetscFunctionReturn(0);
}

PetscErrorCode SetCoordinates3d(DM da)
{
  PetscInt      i, j, mstart, m, nstart, n, pstart, p, k;
  Vec           local, global;
  DMDACoor3d ***coors, ***coorslocal;
  DM            cda;

  PetscFunctionBeginUser;
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinates(da, &global));
  PetscCall(DMGetCoordinatesLocal(da, &local));
  PetscCall(DMDAVecGetArray(cda, global, &coors));
  PetscCall(DMDAVecGetArrayRead(cda, local, &coorslocal));
  PetscCall(DMDAGetCorners(cda, &mstart, &nstart, &pstart, &m, &n, &p));
  for (i = mstart; i < mstart + m; i++) {
    for (j = nstart; j < nstart + n; j++) {
      for (k = pstart; k < pstart + p; k++) {
        if (i % 2) coors[k][j][i].x = coorslocal[k][j][i - 1].x + .1 * (coorslocal[k][j][i + 1].x - coorslocal[k][j][i - 1].x);
        if (j % 2) coors[k][j][i].y = coorslocal[k][j - 1][i].y + .3 * (coorslocal[k][j + 1][i].y - coorslocal[k][j - 1][i].y);
        if (k % 2) coors[k][j][i].z = coorslocal[k - 1][j][i].z + .4 * (coorslocal[k + 1][j][i].z - coorslocal[k - 1][j][i].z);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(cda, global, &coors));
  PetscCall(DMDAVecRestoreArrayRead(cda, local, &coorslocal));
  PetscCall(DMGlobalToLocalBegin(cda, global, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(cda, global, INSERT_VALUES, local));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt        M = 5, N = 4, P = 3, m = PETSC_DECIDE, n = PETSC_DECIDE, p = PETSC_DECIDE, dim = 1;
  DM              dac, daf;
  DMBoundaryType  bx = DM_BOUNDARY_NONE, by = DM_BOUNDARY_NONE, bz = DM_BOUNDARY_NONE;
  DMDAStencilType stype = DMDA_STENCIL_BOX;
  Mat             A;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-P", &P, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-p", &p, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));

  /* Create distributed array and get vectors */
  if (dim == 1) {
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD, bx, M, 1, 1, NULL, &dac));
  } else if (dim == 2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, bx, by, stype, M, N, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &dac));
  } else if (dim == 3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, stype, M, N, P, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &dac));
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "dim must be 1,2, or 3");
  PetscCall(DMSetFromOptions(dac));
  PetscCall(DMSetUp(dac));

  PetscCall(DMRefine(dac, PETSC_COMM_WORLD, &daf));

  PetscCall(DMDASetUniformCoordinates(dac, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  if (dim == 1) {
    PetscCall(SetCoordinates1d(daf));
  } else if (dim == 2) {
    PetscCall(SetCoordinates2d(daf));
  } else if (dim == 3) {
    PetscCall(SetCoordinates3d(daf));
  }
  PetscCall(DMCreateInterpolation(dac, daf, &A, 0));

  /* Free memory */
  PetscCall(DMDestroy(&dac));
  PetscCall(DMDestroy(&daf));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3
      args: -mat_view

   test:
      suffix: 2
      nsize: 3
      args: -mat_view -dim 2

   test:
      suffix: 3
      nsize: 3
      args: -mat_view -dim 3

TEST*/
