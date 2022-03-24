
/* tests MatSeqSBAIJSetPreallocationCSR() and MatMPISBAIJSetPreallocationCSR() */

#include <petsc.h>

int main(int argc, char **args)
{
  PetscInt       ia[3] = { 0, 2, 4};
  PetscInt       ja[4] = { 0, 1, 0, 1};
  PetscScalar    c[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  Mat            ssbaij;
  Mat            msbaij;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,(char*)0));

  CHKERRQ(MatCreate(PETSC_COMM_SELF, &ssbaij));
  CHKERRQ(MatCreate(PETSC_COMM_SELF, &msbaij));
  CHKERRQ(MatSetType(ssbaij, MATSEQSBAIJ));
  CHKERRQ(MatSetType(msbaij, MATMPISBAIJ));
  CHKERRQ(MatSetBlockSize(ssbaij, 2));
  CHKERRQ(MatSetSizes(ssbaij, 4, 4, 4, 4));
  CHKERRQ(MatSetBlockSize(msbaij, 2));
  CHKERRQ(MatSetSizes(msbaij, 4, 4, 4, 4));
  CHKERRQ(MatSeqSBAIJSetPreallocationCSR(ssbaij, 2, ia, ja, c));
  CHKERRQ(MatMPISBAIJSetPreallocationCSR(msbaij, 2, ia, ja, c));
  CHKERRQ(MatView(ssbaij, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));
  CHKERRQ(MatView(msbaij, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));
  CHKERRQ(MatDestroy(&ssbaij));
  CHKERRQ(MatDestroy(&msbaij));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     filter: sed "s?\.??g"

TEST*/
