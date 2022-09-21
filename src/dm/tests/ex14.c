
static char help[] = "Tests saving DMDA vectors to files.\n\n";

/*
    ex13.c reads in the DMDA and vector written by this program.
*/

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  PetscMPIInt rank;
  PetscInt    M = 10, N = 8, m = PETSC_DECIDE, n = PETSC_DECIDE, dof = 1;
  DM          da;
  Vec         local, global, natural;
  PetscScalar value;
  PetscViewer bviewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dof", &dof, NULL));

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, M, N, m, n, dof, 1, NULL, NULL, &da));
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

  PetscCall(DMDACreateNaturalVector(da, &natural));
  PetscCall(DMDAGlobalToNaturalBegin(da, global, INSERT_VALUES, natural));
  PetscCall(DMDAGlobalToNaturalEnd(da, global, INSERT_VALUES, natural));

  PetscCall(DMDASetFieldName(da, 0, "First field"));
  /*  PetscCall(VecView(global,PETSC_VIEWER_DRAW_WORLD)); */

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "daoutput", FILE_MODE_WRITE, &bviewer));
  PetscCall(DMView(da, bviewer));
  PetscCall(VecView(global, bviewer));
  PetscCall(PetscViewerDestroy(&bviewer));

  /* Free memory */
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(VecDestroy(&natural));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
