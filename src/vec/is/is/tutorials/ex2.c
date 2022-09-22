
static char help[] = "Demonstrates creating a stride index set.\n\n";

/*
  Include petscis.h so we can use PETSc IS objects. Note that this automatically
  includes petscsys.h.
*/

#include <petscis.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscInt        i, n, first, step;
  IS              set;
  const PetscInt *indices;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  n     = 10;
  first = 3;
  step  = 2;

  /*
    Create stride index set, starting at 3 with a stride of 2
    Note each processor is generating its own index set
    (in this case they are all identical)
  */
  PetscCall(ISCreateStride(PETSC_COMM_SELF, n, first, step, &set));
  PetscCall(ISView(set, PETSC_VIEWER_STDOUT_SELF));

  /*
    Extract indices from set.
  */
  PetscCall(ISGetIndices(set, &indices));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Printing indices directly\n"));
  for (i = 0; i < n; i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT "\n", indices[i]));

  PetscCall(ISRestoreIndices(set, &indices));

  /*
      Determine information on stride
  */
  PetscCall(ISStrideGetInfo(set, &first, &step));
  PetscCheck(first == 3 && step == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stride info not correct!");
  PetscCall(ISDestroy(&set));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
