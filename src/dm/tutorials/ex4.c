
static char help[] = "Demonstrates various vector routines for DMDA.\n\n";

/*
  Include "petscpf.h" so that we can use pf functions and "petscdmda.h" so
 we can use the PETSc distributed arrays
*/

#include <petscpf.h>
#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode myfunction(void *ctx, PetscInt n, const PetscScalar *xy, PetscScalar *u)
{
  PetscInt i;

  PetscFunctionBeginUser;
  for (i = 0; i < n; i++) {
    u[2 * i]     = xy[2 * i];
    u[2 * i + 1] = xy[2 * i + 1];
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  Vec      u, xy;
  DM       da;
  PetscInt m = 10, n = 10, dof = 2;
  PF       pf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, m, n, PETSC_DECIDE, PETSC_DECIDE, dof, 1, 0, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMCreateGlobalVector(da, &u));
  PetscCall(DMGetCoordinates(da, &xy));

  PetscCall(DMDACreatePF(da, &pf));
  PetscCall(PFSet(pf, myfunction, 0, 0, 0, 0));
  PetscCall(PFSetFromOptions(pf));

  PetscCall(PFApplyVec(pf, xy, u));

  PetscCall(VecView(u, PETSC_VIEWER_DRAW_WORLD));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&u));
  PetscCall(PFDestroy(&pf));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
