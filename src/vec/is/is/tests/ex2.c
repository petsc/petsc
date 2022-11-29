
/*
       Formatted test for ISStride routines.
*/

static char help[] = "Tests IS stride routines.\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscInt        i, n, start, stride;
  const PetscInt *ii;
  IS              is;
  PetscBool       flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /*
     Test IS of size 0
  */
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 2, &is));
  PetscCall(ISGetSize(is, &n));
  PetscCheck(n == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISCreateStride");
  PetscCall(ISStrideGetInfo(is, &start, &stride));
  PetscCheck(start == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISStrideGetInfo");
  PetscCheck(stride == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISStrideGetInfo");
  PetscCall(PetscObjectTypeCompare((PetscObject)is, ISSTRIDE, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISStride");
  PetscCall(ISGetIndices(is, &ii));
  PetscCall(ISRestoreIndices(is, &ii));
  PetscCall(ISDestroy(&is));

  /*
     Test ISGetIndices()
  */
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 10000, -8, 3, &is));
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(ISGetIndices(is, &ii));
  for (i = 0; i < 10000; i++) PetscCheck(ii[i] == -8 + 3 * i, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetIndices");
  PetscCall(ISRestoreIndices(is, &ii));
  PetscCall(ISDestroy(&is));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     output_file: output/ex1_1.out

TEST*/
