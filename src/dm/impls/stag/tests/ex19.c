static char help[] = "(Partially) test DMStag default interpolation, 2d faces-only.\n\n";

#include <petscdm.h>
#include <petscdmstag.h>
#include <petscksp.h>

PetscErrorCode CreateSystem(DM dm, Mat *A, Vec *b);

int main(int argc, char **argv)
{
  DM  dm, dmCoarse;
  Mat Ai;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 2, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMCoarsen(dm, MPI_COMM_NULL, &dmCoarse));
  PetscCall(DMCreateInterpolation(dmCoarse, dm, &Ai, NULL));

  /* See what happens to a constant value on each sub-grid */
  {
    Vec localCoarse, globalCoarse, globalFine, localFine;
    PetscCall(DMGetGlobalVector(dm, &globalFine));
    PetscCall(DMGetGlobalVector(dmCoarse, &globalCoarse));
    PetscCall(DMGetLocalVector(dmCoarse, &localCoarse));
    PetscCall(DMGetLocalVector(dm, &localFine));
    PetscCall(VecSet(localCoarse, -1.0));
    PetscCall(VecSet(localFine, -1.0));
    {
      PetscInt       i, j, startx, starty, nx, ny, extrax, extray;
      PetscInt       p, vx, vy;
      PetscScalar ***arr;
      PetscCall(DMStagGetCorners(dmCoarse, &startx, &starty, NULL, &nx, &ny, NULL, &extrax, &extray, NULL));
      PetscCall(DMStagVecGetArray(dmCoarse, localCoarse, &arr));
      PetscCall(DMStagGetLocationSlot(dmCoarse, DMSTAG_LEFT, 0, &vx));
      PetscCall(DMStagGetLocationSlot(dmCoarse, DMSTAG_DOWN, 0, &vy));
      PetscCall(DMStagGetLocationSlot(dmCoarse, DMSTAG_ELEMENT, 0, &p));
      for (j = starty; j < starty + ny + extray; ++j) {
        for (i = startx; i < startx + nx + extrax; ++i) {
          arr[j][i][vy] = (i < startx + nx) ? 10.0 : -1;
          arr[j][i][vx] = (j < starty + ny) ? 20.0 : -1;
          arr[j][i][p]  = (i < startx + nx) && (j < starty + ny) ? 30.0 : -1;
        }
      }
      PetscCall(DMStagVecRestoreArray(dmCoarse, localCoarse, &arr));
    }
    PetscCall(DMLocalToGlobal(dmCoarse, localCoarse, INSERT_VALUES, globalCoarse));
    PetscCall(MatInterpolate(Ai, globalCoarse, globalFine));
    PetscCall(DMGlobalToLocal(dm, globalFine, INSERT_VALUES, localFine));
    {
      PetscInt       i, j, startx, starty, nx, ny, extrax, extray;
      PetscInt       p, vx, vy;
      PetscScalar ***arr;
      PetscCall(DMStagGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL, &extrax, &extray, NULL));
      PetscCall(DMStagVecGetArrayRead(dm, localFine, &arr));
      PetscCall(DMStagGetLocationSlot(dm, DMSTAG_LEFT, 0, &vx));
      PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN, 0, &vy));
      PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &p));
      for (j = starty; j < starty + ny + extray; ++j) {
        for (i = startx; i < startx + nx + extrax; ++i) {
          const PetscScalar expected_vy = (i < startx + nx) ? 10.0 : -1;
          const PetscScalar expected_vx = (j < starty + ny) ? 20.0 : -1;
          const PetscScalar expected_p  = (i < startx + nx) && (j < starty + ny) ? 30.0 : -1;
          if (arr[j][i][vy] != expected_vy) PetscCall(PetscPrintf(PETSC_COMM_SELF, "wrong %" PetscInt_FMT " %" PetscInt_FMT "\n", i, j));
          if (arr[j][i][vx] != expected_vx) PetscCall(PetscPrintf(PETSC_COMM_SELF, "wrong %" PetscInt_FMT " %" PetscInt_FMT "\n", i, j));
          if (arr[j][i][p] != expected_p) PetscCall(PetscPrintf(PETSC_COMM_SELF, "wrong %" PetscInt_FMT " %" PetscInt_FMT "\n", i, j));
        }
      }
      PetscCall(DMStagVecRestoreArrayRead(dm, localFine, &arr));
    }
    PetscCall(DMRestoreLocalVector(dmCoarse, &localCoarse));
    PetscCall(DMRestoreLocalVector(dm, &localFine));
    PetscCall(DMRestoreGlobalVector(dmCoarse, &globalCoarse));
    PetscCall(DMRestoreGlobalVector(dm, &globalFine));
  }

  PetscCall(MatDestroy(&Ai));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dmCoarse));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args:

   test:
      suffix: 2
      nsize: 4
      args: -stag_grid_x 8 -stag_grid_y 4

TEST*/
