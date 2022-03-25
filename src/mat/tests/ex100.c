
static char help[] = "Tests various routines in MatMAIJ format.\n";

#include <petscmat.h>
#define IMAX 15
int main(int argc,char **args)
{
  Mat            A,B,MA;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscInt       m,n,M,N,dof=1;
  PetscMPIInt    rank,size;
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Load aij matrix A */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* Get dof, then create maij matrix MA */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  PetscCall(MatCreateMAIJ(A,dof,&MA));
  PetscCall(MatGetLocalSize(MA,&m,&n));
  PetscCall(MatGetSize(MA,&M,&N));

  if (size == 1) {
    PetscCall(MatConvert(MA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B));
  } else {
    PetscCall(MatConvert(MA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B));
  }

  /* Test MatMult() */
  PetscCall(MatMultEqual(MA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMul() for MAIJ matrix");
  /* Test MatMultAdd() */
  PetscCall(MatMultAddEqual(MA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMulAdd() for MAIJ matrix");

  /* Test MatMultTranspose() */
  PetscCall(MatMultTransposeEqual(MA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMulAdd() for MAIJ matrix");

  /* Test MatMultTransposeAdd() */
  PetscCall(MatMultTransposeAddEqual(MA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMulTransposeAdd() for MAIJ matrix");

  PetscCall(MatDestroy(&MA));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: {{1 3}}
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/arco1 -dof {{1 2 3 4 5 6 8 9 16}} -viewer_binary_skip_info

TEST*/
