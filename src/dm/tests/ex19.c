
static char help[] = "Tests DMDA with variable multiple degrees of freedom per node.\n\n";

/*
   This code only compiles with gcc, since it is not ANSI C
*/

#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode doit(DM da, Vec global)
{
  PetscInt i, j, k, M, N, dof;

  PetscCall(DMDAGetInfo(da, 0, &M, &N, 0, 0, 0, 0, &dof, 0, 0, 0, 0, 0));
  {
    struct {
      PetscScalar inside[dof];
    } **mystruct;
    PetscCall(DMDAVecGetArrayRead(da, global, (void *)&mystruct));
    for (i = 0; i < N; i++) {
      for (j = 0; j < M; j++) {
        for (k = 0; k < dof; k++) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT " %" PetscInt_FMT " %g\n", i, j, (double)mystruct[i][j].inside[0]));

          mystruct[i][j].inside[1] = 2.1;
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(da, global, (void *)&mystruct));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt dof = 2, M = 3, N = 3, m = PETSC_DECIDE, n = PETSC_DECIDE;
  DM       da;
  Vec      global, local;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-dof", &dof, 0));
  /* Create distributed array and get vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, M, N, m, n, dof, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da, &global));
  PetscCall(DMCreateLocalVector(da, &local));

  PetscCall(doit(da, global));

  PetscCall(VecView(global, 0));

  /* Free memory */
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
