
static char help[] = "Tests VecView() contour plotting for 2d DMDAs.\n\n";

/*
  MATLAB must be installed to configure PETSc to have MATLAB engine.
Unless you have specific important reasons for using the MATLAB engine, we do not
recommend it. If you want to use MATLAB for visualization and maybe a little post processing
then you can use the socket viewer and send the data to MATLAB via that.

  VecView() on DMDA vectors first puts the Vec elements into global natural ordering before printing (or plotting)
them. In 2d 5 by 2 DMDA this means the numbering is

     5  6   7   8   9
     0  1   2   3   4

Now the default split across 2 processors with the DM  is (by rank)

    0  0   0  1   1
    0  0   0  1   1

So the global PETSc ordering is

    3  4  5   8  9
    0  1  2   6  7

Use the options
     -da_grid_x <nx> - number of grid points in x direction, if M < 0
     -da_grid_y <ny> - number of grid points in y direction, if N < 0
     -da_processors_x <MX> number of processors in x directio
     -da_processors_y <MY> number of processors in x direction
*/

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  PetscMPIInt     rank;
  PetscInt        M = 10, N = 8;
  PetscBool       flg = PETSC_FALSE;
  DM              da;
  PetscViewer     viewer;
  Vec             local, global;
  PetscScalar     value;
  DMBoundaryType  bx = DM_BOUNDARY_NONE, by = DM_BOUNDARY_NONE;
  DMDAStencilType stype = DMDA_STENCIL_BOX;
#if defined(PETSC_HAVE_MATLAB)
  PetscViewer mviewer;
  PetscMPIInt size;
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_WORLD, 0, "", 300, 0, 300, 300, &viewer));
#if defined(PETSC_HAVE_MATLAB)
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  if (size == 1) PetscCall(PetscViewerMatlabOpen(PETSC_COMM_WORLD, "tmp.mat", FILE_MODE_WRITE, &mviewer));
#endif

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-star_stencil", &flg, NULL));
  if (flg) stype = DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, bx, by, stype, M, N, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da, &global));
  PetscCall(DMCreateLocalVector(da, &local));

  value = -3.0;
  PetscCall(VecSet(global, value));
  PetscCall(DMGlobalToLocalBegin(da, global, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(da, global, INSERT_VALUES, local));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  value = rank + 1;
  PetscCall(VecScale(local, value));
  PetscCall(DMLocalToGlobalBegin(da, local, ADD_VALUES, global));
  PetscCall(DMLocalToGlobalEnd(da, local, ADD_VALUES, global));

  flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_global", &flg, NULL));
  if (flg) { /* view global vector in natural ordering */
    PetscCall(VecView(global, PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(DMView(da, viewer));
  PetscCall(VecView(global, viewer));
#if defined(PETSC_HAVE_MATLAB)
  if (size == 1) {
    PetscCall(DMView(da, mviewer));
    PetscCall(VecView(global, mviewer));
  }
#endif

  /* Free memory */
#if defined(PETSC_HAVE_MATLAB)
  if (size == 1) PetscCall(PetscViewerDestroy(&mviewer));
#endif
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: x
      nsize: 2
      args: -nox

TEST*/
