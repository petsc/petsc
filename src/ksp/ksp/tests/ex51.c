static char help[] = "Test PCFailedReason.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat                A;            /* linear system matrix */
  KSP                ksp;          /* linear solver context */
  PC                 pc;           /* preconditioner context */
  PetscErrorCode     ierr;
  PetscInt           i,n = 10,col[3];
  PetscMPIInt        size;
  PetscScalar        value[3],alpha,beta,sx;
  PetscBool          reverse=PETSC_FALSE;
  KSPConvergedReason reason;
  PCFailedReason     pcreason;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-reverse",&reverse,NULL);CHKERRQ(ierr);

  sx = PetscSinReal(n*PETSC_PI/2/(n+1));
  alpha = 4.0*sx*sx;   /* alpha is the largest eigenvalue of the matrix */
  beta = 4.0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = MatShift(A,reverse?-alpha:-beta);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Factorize first matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"First matrix\n");CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
  if (reason < 0) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp)),PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
    ierr = KSPConvergedReasonView(ksp,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp)));CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp)));CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Success!\n");CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Factorize second matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatShift(A,reverse?alpha-beta:beta-alpha);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Second matrix\n");CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
  if (reason < 0) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp)),PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
    ierr = KSPConvergedReasonView(ksp,PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp)));CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp)));CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Success!\n");CHKERRQ(ierr);
    ierr = PCGetFailedReason(pc,&pcreason);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"PC failed reason is %s\n",PCFailedReasons[pcreason]);CHKERRQ(ierr);
  }

  /*
     Free work space.
  */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      args: -reverse

   test:
      suffix: 2
      args: -reverse -pc_type cholesky

TEST*/
