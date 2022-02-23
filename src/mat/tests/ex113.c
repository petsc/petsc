
static char help[] = "Tests sequential and parallel MatMatMult() and MatAXPY(...,SUBSET_NONZERO_PATTERN) \n\
Input arguments are:\n\
  -f <input_file>  : file to load\n\n";
/* e.g., mpiexec -n 3 ./ex113 -f <file> */

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,A1,A2,Mtmp,dstMat;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscReal      fill=4.0;
  char           file[128];
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /*  Load the matrix A */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for matrix A with the -f option.");

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatDuplicate(A,MAT_COPY_VALUES,&A1);CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);

  /* dstMat = A*A1*A2 */
  ierr = MatMatMult(A1,A2,MAT_INITIAL_MATRIX,fill,&Mtmp);CHKERRQ(ierr);
  ierr = MatMatMult(A,Mtmp,MAT_INITIAL_MATRIX,fill,&dstMat);CHKERRQ(ierr);
  ierr = MatDestroy(&Mtmp);CHKERRQ(ierr);

  /* dstMat += A1*A2 */
  ierr = MatMatMult(A1,A2,MAT_INITIAL_MATRIX,fill,&Mtmp);CHKERRQ(ierr);
  ierr = MatAXPY(dstMat,1.0,Mtmp,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&Mtmp);CHKERRQ(ierr);

  /* dstMat += A*A1 */
  ierr = MatMatMult(A,A1,MAT_INITIAL_MATRIX,fill,&Mtmp);CHKERRQ(ierr);
  ierr = MatAXPY(dstMat, 1.0, Mtmp,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDestroy(&Mtmp);CHKERRQ(ierr);

  /* dstMat += A */
  ierr = MatAXPY(dstMat, 1.0, A,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&A1);CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&dstMat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

