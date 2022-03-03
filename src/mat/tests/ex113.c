
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
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for matrix A with the -f option.");

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A1));
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));

  /* dstMat = A*A1*A2 */
  CHKERRQ(MatMatMult(A1,A2,MAT_INITIAL_MATRIX,fill,&Mtmp));
  CHKERRQ(MatMatMult(A,Mtmp,MAT_INITIAL_MATRIX,fill,&dstMat));
  CHKERRQ(MatDestroy(&Mtmp));

  /* dstMat += A1*A2 */
  CHKERRQ(MatMatMult(A1,A2,MAT_INITIAL_MATRIX,fill,&Mtmp));
  CHKERRQ(MatAXPY(dstMat,1.0,Mtmp,SUBSET_NONZERO_PATTERN));
  CHKERRQ(MatDestroy(&Mtmp));

  /* dstMat += A*A1 */
  CHKERRQ(MatMatMult(A,A1,MAT_INITIAL_MATRIX,fill,&Mtmp));
  CHKERRQ(MatAXPY(dstMat, 1.0, Mtmp,SUBSET_NONZERO_PATTERN));
  CHKERRQ(MatDestroy(&Mtmp));

  /* dstMat += A */
  CHKERRQ(MatAXPY(dstMat, 1.0, A,SUBSET_NONZERO_PATTERN));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&A1));
  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(MatDestroy(&dstMat));
  ierr = PetscFinalize();
  return ierr;
}
