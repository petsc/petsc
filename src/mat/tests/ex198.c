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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  /* read matrices A, B and C */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fA",file[0],sizeof(file[0]),&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_USER,"Must indicate binary file with the -fA options");

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fB",file[1],sizeof(file[1]),&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_USER,"Must indicate binary file with the -fB options");

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-fC",file[2],sizeof(file[2]),&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_USER,"Must indicate binary file with the -fC options");

  /* Load matrices */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatLoad(B,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[2],FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatLoad(C,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* Test MatMatMult() */
  CHKERRQ(MatMatMult(B,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&BC));
  CHKERRQ(MatMatMult(A,BC,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&ABC));

  CHKERRQ(MatMatMatMult(A,B,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D));
  CHKERRQ(MatMatMatMult(A,B,C,MAT_REUSE_MATRIX,PETSC_DEFAULT,&D));
  /* CHKERRQ(MatView(D,PETSC_VIEWER_STDOUT_WORLD)); */

  CHKERRQ(MatEqual(ABC,D,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"ABC != D");

  CHKERRQ(MatDestroy(&ABC));
  CHKERRQ(MatDestroy(&BC));
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/matmatmatmult/A.bin -fB ${DATAFILESPATH}/matrices/matmatmatmult/B.bin -fC ${DATAFILESPATH}/matrices/matmatmatmult/C.bin
      output_file: output/ex198.out

   test:
      suffix: 2
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/matmatmatmult/A.bin -fB ${DATAFILESPATH}/matrices/matmatmatmult/B.bin -fC ${DATAFILESPATH}/matrices/matmatmatmult/C.bin
      output_file: output/ex198.out

TEST*/
