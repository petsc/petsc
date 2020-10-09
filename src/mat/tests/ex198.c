static char help[] = "Test MatMatMatMult\n\
Reads PETSc matrix A B and C, then comput D=A*B*C \n\
Input parameters include\n\
  -fA <input_file> -fB <input_file> -fC <input_file> \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B,C,D,BC,ABC;
  PetscViewer    fd;
  char           file[3][PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  /* read matrices A, B and C */
  ierr = PetscOptionsGetString(NULL,NULL,"-fA",file[0],sizeof(file[0]),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must indicate binary file with the -fA options");

  ierr = PetscOptionsGetString(NULL,NULL,"-fB",file[1],sizeof(file[1]),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must indicate binary file with the -fB options");

  ierr = PetscOptionsGetString(NULL,NULL,"-fC",file[2],sizeof(file[2]),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must indicate binary file with the -fC options");

  /* Load matrices */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatLoad(B,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatLoad(C,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Test MatMatMult() */
  ierr = MatMatMult(B,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&BC);CHKERRQ(ierr);
  ierr = MatMatMult(A,BC,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ABC);CHKERRQ(ierr);

  ierr = MatMatMatMult(A,B,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D);CHKERRQ(ierr);
  ierr = MatMatMatMult(A,B,C,MAT_REUSE_MATRIX,PETSC_DEFAULT,&D);CHKERRQ(ierr);
  /* ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  ierr = MatEqual(ABC,D,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"ABC != D");

  ierr = MatDestroy(&ABC);CHKERRQ(ierr);
  ierr = MatDestroy(&BC);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/matmatmatmult/A.bin -fB ${DATAFILESPATH}/matrices/matmatmatmult/B.bin -fC ${DATAFILESPATH}/matrices/matmatmatmult/C.bin
      output_file: output/ex198.out

   test:
      suffix: 2
      nsize: 3
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/matmatmatmult/A.bin -fB ${DATAFILESPATH}/matrices/matmatmatmult/B.bin -fC ${DATAFILESPATH}/matrices/matmatmatmult/C.bin
      output_file: output/ex198.out

TEST*/
