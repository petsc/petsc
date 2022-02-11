
static char help[] ="Tests MatPtAP() \n";

#include <petscmat.h>

/*
 * This is an extremely simple example to test MatPtAP. It is very useful when developing and debugging the code.
 *
 * A =

   1   2   0   4
   0   1   2   0
   2   0   4   0
   0   1   2   1
 *
 *
 *
 * P =

   1.00000   0.00000
   0.30000   0.50000
   0.00000   0.80000
   0.90000   0.00000
 *
 *
 *AP =
 *
 * 5.20000   1.00000
   0.30000   2.10000
   2.00000   3.20000
   1.20000   2.10000
 *
 * PT =

   1.00000   0.30000   0.00000   0.90000
   0.00000   0.50000   0.80000   0.00000

 *
 * C =

   6.3700   3.5200
   1.7500   3.6100
 *
 * */

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Mat            A,P,PtAP;
  PetscInt       i1[] = {0, 3, 5}, i2[] = {0,2,5};
  PetscInt       j1[] = {0, 1, 3, 1, 2}, j2[] = {0, 2, 1, 2, 3};
  PetscScalar    a1[] = {1, 2, 4, 1, 2}, a2[] = {2, 4, 1, 2, 1};
  PetscInt       pi1[] = {0,1,3}, pi2[] = {0,1,2};
  PetscInt       pj1[] = {0, 0, 1}, pj2[] = {1,0};
  PetscScalar    pa1[] = {1, 0.3, 0.5}, pa2[] = {0.8, 0.9};
  MPI_Comm       comm;
  PetscMPIInt    rank,size;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size != 2,comm,PETSC_ERR_ARG_INCOMP,"You have to use two processor cores to run this example ");
  ierr = MatCreateMPIAIJWithArrays(comm,2,2,PETSC_DETERMINE,PETSC_DETERMINE,rank? i2:i1,rank? j2:j1,rank? a2:a1,&A);CHKERRQ(ierr);
  ierr = MatCreateMPIAIJWithArrays(comm,2,1,PETSC_DETERMINE,PETSC_DETERMINE,rank? pi2:pi1,rank? pj2:pj1,rank? pa2:pa1,&P);CHKERRQ(ierr);
  ierr = MatPtAP(A,P,MAT_INITIAL_MATRIX,1.1,&PtAP);CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);
  ierr = MatView(P,NULL);CHKERRQ(ierr);
  ierr = MatView(PtAP,NULL);CHKERRQ(ierr);
  ierr = MatPtAP(A,P,MAT_REUSE_MATRIX,1.1,&PtAP);CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);
  ierr = MatView(P,NULL);CHKERRQ(ierr);
  ierr = MatView(PtAP,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  ierr = MatDestroy(&PtAP);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   test:
     nsize: 2
     args:   -matptap_via allatonce
     output_file: output/ex90_1.out

   test:
     nsize: 2
     suffix: merged
     args:   -matptap_via allatonce_merged
     output_file: output/ex90_1.out

TEST*/
