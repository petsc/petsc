
static char help[] = "VecView() with a DMDA1d vector and draw viewer.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscao.h>

PetscErrorCode apply(void *ctx, PetscInt n, const PetscScalar *x, PetscScalar *y)
{
  PetscInt i;

  for (i = 0; i < n; i++) {
    y[3 * i]     = x[i];
    y[3 * i + 1] = x[i] * x[i];
    y[3 * i + 2] = x[i] * x[i] * x[i];
  }
  return 0;
}

int main(int argc, char **argv)
{
  DM  da;
  Vec global;
  PF  pf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 10, 3, 1, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da, &global));
  PetscCall(PFCreate(PETSC_COMM_WORLD, 1, 3, &pf));
  PetscCall(PFSet(pf, apply, NULL, NULL, NULL, NULL));
  PetscCall(PFApplyVec(pf, NULL, global));
  PetscCall(PFDestroy(&pf));
  PetscCall(VecView(global, PETSC_VIEWER_DRAW_WORLD));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      requires: x

TEST*/
