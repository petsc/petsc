
static char help[] = "Tests MatCreateMPISBAIJWithArrays().\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  PetscErrorCode ierr;
  PetscInt       m = 4, bs = 1,ii[5],jj[7];
  PetscMPIInt    size,rank;
  PetscScalar    aa[7];

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (size != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for two processes");

  if (!rank) {
    ii[0] = 0; ii[1] = 2; ii[2] = 5; ii[3] = 7; ii[4] = 7;
    jj[0] = 0; jj[1] = 1; jj[2] = 1; jj[3] = 2; jj[4] = 6; jj[5] = 3; jj[6] = 7;
    aa[0] = 0; aa[1] = 1; aa[2] = 2; aa[3] = 3; aa[4] = 4; aa[5] = 5; aa[6] = 6;
    /*  0 1
          1  2       6
   	     3          7 */
  } else {
    ii[0] = 0; ii[1] = 2; ii[2] = 4; ii[3] = 6; ii[4] = 7;
    jj[0] = 4; jj[1] = 5; jj[2] = 5; jj[3] = 7; jj[4] = 6; jj[5] = 7; jj[6] = 7;
    aa[0] = 8; aa[1] = 9; aa[2] = 10; aa[3] = 11; aa[4] = 12; aa[5] = 13; aa[6] = 14;
    /*    4  5
             5   7
               6 7
	         7 */
  }
  ierr = MatCreateMPISBAIJWithArrays(PETSC_COMM_WORLD,bs,m,m,PETSC_DECIDE,PETSC_DECIDE,ii,jj,aa,&A);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
