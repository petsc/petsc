
static char help[] = "Tests VecView()/VecLoad() for DMDA vectors (this tests DMDAGlobalToNatural()).\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  PetscMPIInt     size;
  PetscInt        N = 6, m = PETSC_DECIDE, n = PETSC_DECIDE, p = PETSC_DECIDE, M = 8, dof = 1, stencil_width = 1, P = 5, pt = 0, st = 0;
  PetscBool       flg2, flg3, native = PETSC_FALSE;
  DMBoundaryType  bx = DM_BOUNDARY_NONE, by = DM_BOUNDARY_NONE, bz = DM_BOUNDARY_NONE;
  DMDAStencilType stencil_type = DMDA_STENCIL_STAR;
  DM              da;
  Vec             global1, global2, global3, global4;
  PetscScalar     mone = -1.0;
  PetscReal       norm;
  PetscViewer     viewer;
  PetscRandom     rdm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-P", &P, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dof", &dof, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-stencil_width", &stencil_width, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-periodic", &pt, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-native", &native, NULL));
  if (pt == 1) bx = DM_BOUNDARY_PERIODIC;
  if (pt == 2) by = DM_BOUNDARY_PERIODIC;
  if (pt == 3) {
    bx = DM_BOUNDARY_PERIODIC;
    by = DM_BOUNDARY_PERIODIC;
  }
  if (pt == 4) bz = DM_BOUNDARY_PERIODIC;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-stencil_type", &st, NULL));
  stencil_type = (DMDAStencilType)st;

  PetscCall(PetscOptionsHasName(NULL, NULL, "-one", &flg2));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-two", &flg2));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-three", &flg3));
  if (flg2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, bx, by, stencil_type, M, N, m, n, dof, stencil_width, 0, 0, &da));
  } else if (flg3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, stencil_type, M, N, P, m, n, p, dof, stencil_width, 0, 0, 0, &da));
  } else {
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD, bx, M, dof, stencil_width, NULL, &da));
  }
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateGlobalVector(da, &global1));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(DMCreateGlobalVector(da, &global2));
  PetscCall(DMCreateGlobalVector(da, &global3));
  PetscCall(DMCreateGlobalVector(da, &global4));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "temp", FILE_MODE_WRITE, &viewer));
  if (native) PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  PetscCall(VecSetRandom(global1, rdm));
  PetscCall(VecView(global1, viewer));
  PetscCall(VecSetRandom(global3, rdm));
  PetscCall(VecView(global3, viewer));
  if (native) PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "temp", FILE_MODE_READ, &viewer));
  if (native) PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE));
  PetscCall(VecLoad(global2, viewer));
  PetscCall(VecLoad(global4, viewer));
  if (native) PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  if (native) {
    Vec       filenative;
    PetscBool same;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "temp", FILE_MODE_READ, &viewer));
    PetscCall(DMDACreateNaturalVector(da, &filenative));
    /* DMDA "natural" Vec does not commandeer VecLoad.  The following load will only work when run on the same process
     * layout, where as the standard VecView/VecLoad (using DMDA and not PETSC_VIEWER_NATIVE) can be read on a different
     * number of processors. */
    PetscCall(VecLoad(filenative, viewer));
    PetscCall(VecEqual(global2, filenative, &same));
    if (!same) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ex23: global vector does not match contents of file\n"));
      PetscCall(VecView(global2, 0));
      PetscCall(VecView(filenative, 0));
    }
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecDestroy(&filenative));
  }

  PetscCall(VecAXPY(global2, mone, global1));
  PetscCall(VecNorm(global2, NORM_MAX, &norm));
  if (norm != 0.0) {
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ex23: Norm of difference %g should be zero\n", (double)norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Number of processors %d\n", size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  M,N,P,dof %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n", M, N, P, dof));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  stencil_width %" PetscInt_FMT " stencil_type %d periodic %d\n", stencil_width, (int)stencil_type, (int)bx));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  dimension %d\n", 1 + (int)flg2 + (int)flg3));
  }
  PetscCall(VecAXPY(global4, mone, global3));
  PetscCall(VecNorm(global4, NORM_MAX, &norm));
  if (norm != 0.0) {
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ex23: Norm of difference %g should be zero\n", (double)norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Number of processors %d\n", size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  M,N,P,dof %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n", M, N, P, dof));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  stencil_width %" PetscInt_FMT " stencil_type %d periodic %d\n", stencil_width, (int)stencil_type, (int)bx));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  dimension %d\n", 1 + (int)flg2 + (int)flg3));
  }

  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(DMDestroy(&da));
  PetscCall(VecDestroy(&global1));
  PetscCall(VecDestroy(&global2));
  PetscCall(VecDestroy(&global3));
  PetscCall(VecDestroy(&global4));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: {{1  3}}
      args: -one -dof {{1 2 3}} -stencil_type {{0 1}}

   test:
      suffix: 3
      nsize: {{2 4}}
      args: -two -dof {{1 3}} -stencil_type {{0 1}}

   test:
      suffix: 4
      nsize: {{1 4}}
      args: -three -dof {{2 3}} -stencil_type {{0 1}}

   test:
      suffix: 2
      nsize: 2
      args: -two -native

TEST*/
