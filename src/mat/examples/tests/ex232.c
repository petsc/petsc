
/* tests MatSeqSBAIJSetPreallocationCSR() and MatMPISBAIJSetPreallocationCSR() */

#include <petsc.h>

int main(int argc, char **args)
{
  PetscInt       ia[3] = { 0, 2, 4};
  PetscInt       ja[4] = { 0, 1, 0, 1};
  PetscScalar    c[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  Mat            ssbaij;
  Mat            msbaij;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,(char*)0);if (ierr) return ierr;

  ierr = MatCreate(PETSC_COMM_SELF, &ssbaij);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF, &msbaij);CHKERRQ(ierr);
  ierr = MatSetType(ssbaij, MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSetType(msbaij, MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatSetBlockSize(ssbaij, 2);CHKERRQ(ierr);
  ierr = MatSetSizes(ssbaij, 4, 4, 4, 4);CHKERRQ(ierr);
  ierr = MatSetBlockSize(msbaij, 2);CHKERRQ(ierr);
  ierr = MatSetSizes(msbaij, 4, 4, 4, 4);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocationCSR(ssbaij, 2, ia, ja, c);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocationCSR(msbaij, 2, ia, ja, c);CHKERRQ(ierr);
  ierr = MatView(ssbaij, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));CHKERRQ(ierr);
  ierr = MatView(msbaij, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));CHKERRQ(ierr);
  ierr = MatDestroy(&ssbaij);CHKERRQ(ierr);
  ierr = MatDestroy(&msbaij);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     filter: sed "s?\.??g"

TEST*/
