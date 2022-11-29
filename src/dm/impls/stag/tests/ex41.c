static char help[] = "Test -dm_preallocate_only with DMStag\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc, char **argv)
{
  DM            dm;
  PetscInt      dim;
  Mat           A;
  DMStagStencil row, col;
  PetscScalar   value;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  dim = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));

  switch (dim) {
  case 1:
    PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 4, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, &dm));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, dim);
  }
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  PetscCall(DMCreateMatrix(dm, &A));

  row.c   = 0;
  row.i   = 0;
  row.loc = DMSTAG_ELEMENT;

  col.c   = 0;
  col.i   = 1;
  col.loc = DMSTAG_ELEMENT;

  value = 1.234;

  PetscCall(DMStagMatSetValuesStencil(dm, A, 1, &row, 1, &col, &value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -dm_preallocate_only

TEST*/
