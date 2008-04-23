
static char help[] = "Tests sequential and parallel MatMatMult() and MatAXPY(...,SUBSET_NONZERO_PATTERN) \n\
Input arguments are:\n\
  -f <input_file>  : file to load\n\n";
/* e.g., mpiexec -n 3 ./ex113 -f <file> */

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,A1,A2,Mtmp,dstMat;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscReal      fill=4.0;
  char           file[128]; 
  PetscTruth     flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  
  /*  Load the matrix A */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,127,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"Must indicate a file name for matrix A with the -f option.");

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(viewer,MATAIJ,&A);CHKERRQ(ierr); 
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);             

  ierr = MatDuplicate(A,MAT_COPY_VALUES,&A1);CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);

  /* dstMat = A*A1*A2 */
  ierr = MatMatMult(A1,A2,MAT_INITIAL_MATRIX,fill,&Mtmp);CHKERRQ(ierr);
  ierr = MatMatMult(A,Mtmp,MAT_INITIAL_MATRIX,fill,&dstMat);CHKERRQ(ierr); 
  ierr = MatDestroy(Mtmp);CHKERRQ(ierr);

  /* dstMat += A1*A2 */
  ierr = MatMatMult(A1,A2,MAT_INITIAL_MATRIX,fill,&Mtmp);CHKERRQ(ierr);
  ierr = MatAXPY(dstMat,1.0,Mtmp,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatDestroy(Mtmp);CHKERRQ(ierr); 

  /* dstMat += A*A1 */
  ierr = MatMatMult(A,A1,MAT_INITIAL_MATRIX,fill,&Mtmp);CHKERRQ(ierr);
  ierr = MatAXPY(dstMat, 1.0, Mtmp,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatDestroy(Mtmp);CHKERRQ(ierr);

  /* dstMat += A */
  ierr = MatAXPY(dstMat, 1.0, A,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr); 

  ierr = MatDestroy(A);CHKERRQ(ierr); 
  ierr = MatDestroy(A1);CHKERRQ(ierr);
  ierr = MatDestroy(A2);CHKERRQ(ierr);  
  ierr = MatDestroy(dstMat);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

