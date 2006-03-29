
static char help[] = "Tests vatious routines in MatMAIJ format.\n";

#include "petscmat.h"
#define IMAX 15 
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat               A,B,MA;
  PetscViewer       fd;
  char              file[PETSC_MAX_PATH_LEN];
  PetscRandom       rand;
  Vec               xx,yy,s1,s2;
  PetscInt          m,n,M,N,dof=1;
  PetscMPIInt       rank,size;
  PetscErrorCode    ierr;
  PetscTruth        flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  SETERRQ(1,"This example does not work with complex numbers");
#else

  /* Load aij matrix A */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_USER,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatLoad(fd,MATAIJ,&A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  /* Get dof, then create maij matrix MA */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(A,dof,&MA);CHKERRQ(ierr);
  ierr = MatGetLocalSize(MA,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(MA,&M,&N);CHKERRQ(ierr);

  if (size == 1){
    ierr = MatConvert(MA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(MA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  }
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&xx);CHKERRQ(ierr);
  ierr = VecSetSizes(xx,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(xx);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&yy);CHKERRQ(ierr);

  /* Test MatMult() */ 
  ierr = MatMultEqual(MA,B,10,&flg);CHKERRQ(ierr);
  if (!flg){
    SETERRQ(PETSC_ERR_CONV_FAILED,"Error: MatMul() for MAIJ matrix");
  }
  /* Test MatMultAdd() */
  ierr = MatMultAddEqual(MA,B,10,&flg);CHKERRQ(ierr);
  if (!flg){
    SETERRQ(PETSC_ERR_CONV_FAILED,"Error: MatMulAdd() for MAIJ matrix");
  }

  /* Test MatMultTranspose() */
  ierr = MatMultTransposeEqual(MA,B,10,&flg);CHKERRQ(ierr);
  if (!flg){
    SETERRQ(PETSC_ERR_CONV_FAILED,"Error: MatMulAdd() for MAIJ matrix");
  }
  
  /* Test MatMultTransposeAdd() */
   ierr = MatMultTransposeAddEqual(MA,B,10,&flg);CHKERRQ(ierr);
  if (!flg){
    SETERRQ(PETSC_ERR_CONV_FAILED,"Error: MatMulTransposeAdd() for MAIJ matrix");
  }

  ierr = MatDestroy(MA);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr); 
  ierr = MatDestroy(B);CHKERRQ(ierr); 
  ierr = VecDestroy(xx);CHKERRQ(ierr);
  ierr = VecDestroy(yy);CHKERRQ(ierr);
  ierr = VecDestroy(s1);CHKERRQ(ierr);
  ierr = VecDestroy(s2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
#endif
  return 0;
}
