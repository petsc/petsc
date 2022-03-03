
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
  PetscErrorCode ierr;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Load aij matrix A */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* Get dof, then create maij matrix MA */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(MatCreateMAIJ(A,dof,&MA));
  CHKERRQ(MatGetLocalSize(MA,&m,&n));
  CHKERRQ(MatGetSize(MA,&M,&N));

  if (size == 1) {
    CHKERRQ(MatConvert(MA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B));
  } else {
    CHKERRQ(MatConvert(MA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B));
  }

  /* Test MatMult() */
  CHKERRQ(MatMultEqual(MA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMul() for MAIJ matrix");
  /* Test MatMultAdd() */
  CHKERRQ(MatMultAddEqual(MA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMulAdd() for MAIJ matrix");

  /* Test MatMultTranspose() */
  CHKERRQ(MatMultTransposeEqual(MA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMulAdd() for MAIJ matrix");

  /* Test MatMultTransposeAdd() */
  CHKERRQ(MatMultTransposeAddEqual(MA,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMulTransposeAdd() for MAIJ matrix");

  CHKERRQ(MatDestroy(&MA));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: {{1 3}}
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/arco1 -dof {{1 2 3 4 5 6 8 9 16}} -viewer_binary_skip_info

TEST*/
