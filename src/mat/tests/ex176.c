
static char help[] ="Tests MatCreateMPIAIJWithArrays() abd MatUpdateMPIAIJWithArrays()\n";

#include <petscmat.h>

/*
 * This is an extremely simple example to test MatUpdateMPIAIJWithArrays()
 *
 * A =

   1    2   0   3  0  0
   0    4   5   0  0  6
   7    0   8   0  9  0
   0   10  11  12  0  13
   0   14  15   0  0  16
  17    0   0   0  0  18
 *
 * */

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Mat            A;
  PetscInt       i[3][3] = {{0, 3, 6},{0, 3, 7},{0, 3, 5}};
  PetscInt       j[3][7] = {{0, 1, 3, 1, 2, 5, -1},{0, 2, 4, 1, 2, 3, 5},{1, 2, 5, 0, 5, -1, -1}};
  PetscScalar    a[3][7] = {{1, 2, 3, 4, 5, 6, -1}, {7, 8, 9, 10, 11, 12, 13},{14, 15, 16, 17, 18, -1, -1}};
  PetscScalar    anew[3][7] = {{10, 20, 30, 40, 50, 60, -1}, {70, 80, 90, 100, 110, 120, 130},{140, 150, 160, 170, 180, -1, -1}};
  MPI_Comm       comm;
  PetscMPIInt    rank,size;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size != 3,comm,PETSC_ERR_ARG_INCOMP,"You have to use three MPI processes to run this example ");
  ierr = MatCreateMPIAIJWithArrays(comm,2,2,PETSC_DETERMINE,PETSC_DETERMINE,i[rank],j[rank],a[rank],&A);CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);
  ierr = MatUpdateMPIAIJWithArrays(A,2,2,PETSC_DETERMINE,PETSC_DETERMINE,i[rank],j[rank],anew[rank]);CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);
  ierr = MatUpdateMPIAIJWithArrays(A,2,2,PETSC_DETERMINE,PETSC_DETERMINE,i[rank],j[rank],a[rank]);CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   test:
     nsize: 3

TEST*/
