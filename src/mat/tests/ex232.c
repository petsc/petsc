/* tests MatSeqSBAIJSetPreallocationCSR() and MatMPISBAIJSetPreallocationCSR() */

#include <petsc.h>

int main(int argc, char **args)
{
  PetscInt    ia[3] = {0, 2, 4};
  PetscInt    ja[4] = {0, 1, 0, 1};
  PetscScalar c[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  Mat         ssbaij, msbaij;
  PetscBool   v2 = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, (char *)0));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-v2", &v2, NULL));

  PetscCall(MatCreate(PETSC_COMM_SELF, &ssbaij));
  PetscCall(MatCreate(PETSC_COMM_SELF, &msbaij));
  if (!v2) {
    PetscCall(MatSetType(ssbaij, MATSEQSBAIJ));
    PetscCall(MatSetType(msbaij, MATMPISBAIJ));
  }
  PetscCall(MatSetBlockSize(ssbaij, 2));
  PetscCall(MatSetSizes(ssbaij, 4, 4, 4, 4));
  PetscCall(MatSetBlockSize(msbaij, 2));
  PetscCall(MatSetSizes(msbaij, 4, 4, 4, 4));
  if (v2) {
    PetscCall(MatSetUp(ssbaij));
    PetscCall(MatSetUp(msbaij));
    PetscCall(MatSetType(ssbaij, MATSEQSBAIJ));
    PetscCall(MatSetType(msbaij, MATMPISBAIJ));
  }
  PetscCall(MatSeqSBAIJSetPreallocationCSR(ssbaij, 2, ia, ja, c));
  PetscCall(MatMPISBAIJSetPreallocationCSR(msbaij, 2, ia, ja, c));
  PetscCall(MatView(ssbaij, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));
  PetscCall(MatView(msbaij, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));
  PetscCall(MatDestroy(&ssbaij));
  PetscCall(MatDestroy(&msbaij));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     filter: sed "s?\.??g"

   test:
     suffix: 2
     filter: sed "s?\.??g"
     args : -v2
     output_file: output/ex232_1.out

TEST*/
