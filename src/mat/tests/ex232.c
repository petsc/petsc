
/* tests MatSeqSBAIJSetPreallocationCSR() and MatMPISBAIJSetPreallocationCSR() */

#include <petsc.h>

int main(int argc, char **args)
{
  PetscInt       ia[3] = { 0, 2, 4};
  PetscInt       ja[4] = { 0, 1, 0, 1};
  PetscScalar    c[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  Mat            ssbaij;
  Mat            msbaij;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,(char*)0));

  PetscCall(MatCreate(PETSC_COMM_SELF, &ssbaij));
  PetscCall(MatCreate(PETSC_COMM_SELF, &msbaij));
  PetscCall(MatSetType(ssbaij, MATSEQSBAIJ));
  PetscCall(MatSetType(msbaij, MATMPISBAIJ));
  PetscCall(MatSetBlockSize(ssbaij, 2));
  PetscCall(MatSetSizes(ssbaij, 4, 4, 4, 4));
  PetscCall(MatSetBlockSize(msbaij, 2));
  PetscCall(MatSetSizes(msbaij, 4, 4, 4, 4));
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

TEST*/
