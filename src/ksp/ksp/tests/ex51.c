static char help[] = "Test PCFailedReason.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat                A;            /* linear system matrix */
  KSP                ksp;          /* linear solver context */
  PC                 pc;           /* preconditioner context */
  PetscInt           i,n = 10,col[3];
  PetscMPIInt        size;
  PetscScalar        value[3],alpha,beta,sx;
  PetscBool          reverse=PETSC_FALSE;
  KSPConvergedReason reason;
  PCFailedReason     pcreason;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-reverse",&reverse,NULL));

  sx = PetscSinReal(n*PETSC_PI/2/(n+1));
  alpha = 4.0*sx*sx;   /* alpha is the largest eigenvalue of the matrix */
  beta = 4.0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    CHKERRQ(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(MatShift(A,reverse?-alpha:-beta));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCLU));
  CHKERRQ(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Factorize first matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"First matrix\n"));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPGetConvergedReason(ksp,&reason));
  if (reason < 0) {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp)),PETSC_VIEWER_DEFAULT));
    CHKERRQ(KSPConvergedReasonView(ksp,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp))));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp))));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Success!\n"));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Factorize second matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatShift(A,reverse?alpha-beta:beta-alpha));
  CHKERRQ(KSPSetOperators(ksp,A,A));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Second matrix\n"));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPGetConvergedReason(ksp,&reason));
  if (reason < 0) {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp)),PETSC_VIEWER_DEFAULT));
    CHKERRQ(KSPConvergedReasonView(ksp,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp))));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp))));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Success!\n"));
    CHKERRQ(PCGetFailedReason(pc,&pcreason));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"PC failed reason is %s\n",PCFailedReasons[pcreason]));
  }

  /*
     Free work space.
  */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -reverse

   test:
      suffix: 2
      args: -reverse -pc_type cholesky

TEST*/
