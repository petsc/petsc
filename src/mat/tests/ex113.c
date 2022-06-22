
static char help[] = "Tests sequential and parallel MatMatMult() and MatAXPY(...,SUBSET_NONZERO_PATTERN) \n\
Input arguments are:\n\
  -f <input_file>  : file to load\n\n";
/* e.g., mpiexec -n 3 ./ex113 -f <file> */

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,A1,A2,Mtmp,dstMat;
  PetscViewer    viewer;
  PetscReal      fill=4.0;
  char           file[128];
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  /*  Load the matrix A */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate a file name for matrix A with the -f option.");

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatLoad(A,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A1));
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&A2));

  /* dstMat = A*A1*A2 */
  PetscCall(MatMatMult(A1,A2,MAT_INITIAL_MATRIX,fill,&Mtmp));
  PetscCall(MatMatMult(A,Mtmp,MAT_INITIAL_MATRIX,fill,&dstMat));
  PetscCall(MatDestroy(&Mtmp));

  /* dstMat += A1*A2 */
  PetscCall(MatMatMult(A1,A2,MAT_INITIAL_MATRIX,fill,&Mtmp));
  PetscCall(MatAXPY(dstMat,1.0,Mtmp,SUBSET_NONZERO_PATTERN));
  PetscCall(MatDestroy(&Mtmp));

  /* dstMat += A*A1 */
  PetscCall(MatMatMult(A,A1,MAT_INITIAL_MATRIX,fill,&Mtmp));
  PetscCall(MatAXPY(dstMat, 1.0, Mtmp,SUBSET_NONZERO_PATTERN));
  PetscCall(MatDestroy(&Mtmp));

  /* dstMat += A */
  PetscCall(MatAXPY(dstMat, 1.0, A,SUBSET_NONZERO_PATTERN));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&A1));
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&dstMat));
  PetscCall(PetscFinalize());
  return 0;
}
