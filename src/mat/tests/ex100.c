
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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* Load aij matrix A */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Get dof, then create maij matrix MA */
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(A,dof,&MA);CHKERRQ(ierr);
  ierr = MatGetLocalSize(MA,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(MA,&M,&N);CHKERRQ(ierr);

  if (size == 1) {
    ierr = MatConvert(MA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(MA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  }

  /* Test MatMult() */
  ierr = MatMultEqual(MA,B,10,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMul() for MAIJ matrix");
  /* Test MatMultAdd() */
  ierr = MatMultAddEqual(MA,B,10,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMulAdd() for MAIJ matrix");

  /* Test MatMultTranspose() */
  ierr = MatMultTransposeEqual(MA,B,10,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMulAdd() for MAIJ matrix");

  /* Test MatMultTransposeAdd() */
  ierr = MatMultTransposeAddEqual(MA,B,10,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMulTransposeAdd() for MAIJ matrix");

  ierr = MatDestroy(&MA);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
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
