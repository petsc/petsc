static char help[] = "Test DMStag 2d star stencil\n\n";
#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc, char **argv)
{
  DM             dm;
  Vec            vec, vecLocal1, vecLocal2;
  PetscScalar   *a, ***a1, ***a2, expected, sum;
  PetscInt       startx, starty, nx, ny, i, j, d, is, js, dof0, dof1, dof2, dofTotal, stencilWidth, ngx, ngy;
  DMBoundaryType boundaryTypex, boundaryTypey;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  dof0         = 1;
  dof1         = 1;
  dof2         = 1;
  stencilWidth = 2;
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, 4, 4, PETSC_DECIDE, PETSC_DECIDE, dof0, dof1, dof2, DMSTAG_STENCIL_STAR, stencilWidth, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagGetDOF(dm, &dof0, &dof1, &dof2, NULL));
  dofTotal = dof0 + 2 * dof1 + dof2;
  PetscCall(DMStagGetStencilWidth(dm, &stencilWidth));

  PetscCall(DMCreateLocalVector(dm, &vecLocal1));
  PetscCall(VecDuplicate(vecLocal1, &vecLocal2));

  PetscCall(DMCreateGlobalVector(dm, &vec));
  PetscCall(VecSet(vec, 1.0));
  PetscCall(VecSet(vecLocal1, 0.0));
  PetscCall(DMGlobalToLocalBegin(dm, vec, INSERT_VALUES, vecLocal1));
  PetscCall(DMGlobalToLocalEnd(dm, vec, INSERT_VALUES, vecLocal1));

  PetscCall(DMStagGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL, NULL, NULL, NULL));
  PetscCall(DMStagVecGetArrayRead(dm, vecLocal1, &a1));
  PetscCall(DMStagVecGetArray(dm, vecLocal2, &a2));
  for (j = starty; j < starty + ny; ++j) {
    for (i = startx; i < startx + nx; ++i) {
      for (d = 0; d < dofTotal; ++d) {
        if (a1[j][i][d] != 1.0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Unexpected value %g (expecting %g)\n", rank, (double)PetscRealPart(a1[j][i][d]), 1.0));
        a2[j][i][d] = 0.0;
        for (js = -stencilWidth; js <= stencilWidth; ++js) a2[j][i][d] += a1[j + js][i][d];
        for (is = -stencilWidth; is <= stencilWidth; ++is) a2[j][i][d] += a1[j][i + is][d];
        a2[j][i][d] -= a1[j][i][d];
      }
    }
  }
  PetscCall(DMStagVecRestoreArrayRead(dm, vecLocal1, &a1));
  PetscCall(DMStagVecRestoreArray(dm, vecLocal2, &a2));

  PetscCall(DMLocalToGlobalBegin(dm, vecLocal2, INSERT_VALUES, vec));
  PetscCall(DMLocalToGlobalEnd(dm, vecLocal2, INSERT_VALUES, vec));

  /* For the all-periodic case, some additional checks */
  PetscCall(DMStagGetBoundaryTypes(dm, &boundaryTypex, &boundaryTypey, NULL));
  if (boundaryTypex == DM_BOUNDARY_PERIODIC && boundaryTypey == DM_BOUNDARY_PERIODIC) {
    PetscCall(DMStagGetGhostCorners(dm, NULL, NULL, NULL, &ngx, &ngy, NULL));
    expected = (ngx * ngy - 4 * stencilWidth * stencilWidth) * dofTotal;
    PetscCall(VecSum(vecLocal1, &sum));
    if (sum != expected) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Unexpected sum of local entries %g (expected %g)\n", rank, (double)PetscRealPart(sum), (double)PetscRealPart(expected)));

    PetscCall(VecGetArray(vec, &a));
    expected = 1 + 4 * stencilWidth;
    for (i = 0; i < ny * nx * dofTotal; ++i) {
      if (a[i] != expected) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] Unexpected value %g (expecting %g)\n", rank, (double)PetscRealPart(a[i]), (double)PetscRealPart(expected)));
    }
    PetscCall(VecRestoreArray(vec, &a));
  }

  PetscCall(VecDestroy(&vec));
  PetscCall(VecDestroy(&vecLocal1));
  PetscCall(VecDestroy(&vecLocal2));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 4
      args: -stag_ranks_x 2 -stag_ranks_y 2 -stag_stencil_width 1

   test:
      suffix: 2
      nsize: 6
      args: -stag_ranks_x 3 -stag_ranks_y 2 -stag_dof_0 2 -stag_grid_x 6

   test:
      suffix: 3
      nsize: 4
      args: -stag_dof_0 3 -stag_dof_1 2 -stag_dof_2 4 2 -stag_stencil_width 3 -stag_grid_x 6 -stag_grid_y 6

   test:
      suffix: 4
      nsize: 4
      args: -stag_stencil_width 1 -stag_grid_x 2 -stag_grid_y 2 -stag_boundary_type_x ghosted

   test:
      suffix: 5
      nsize: 4
      args: -stag_stencil_width 1 -stag_grid_x 2 -stag_grid_y 2 -stag_boundary_type_y ghosted

   test:
      suffix: 6
      nsize: 4
      args: -stag_stencil_width 1 -stag_grid_x 3 -stag_grid_y 2 -stag_boundary_type_x ghosted -stag_boundary_type_y ghosted

   test:
      suffix: 7
      nsize: 4
      args: -stag_stencil_width 1 -stag_grid_x 2 -stag_grid_y 2 -stag_boundary_type_y ghosted

   test:
      suffix: 8
      nsize: 6
      args: -stag_stencil_width 1 -stag_grid_y 2 -stag_grid_x 19 -stag_boundary_type_y ghosted -stag_ranks_x 6
TEST*/
