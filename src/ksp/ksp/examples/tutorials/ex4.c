
static char help[] = "Solves a linear system in parallel with KSP amd HMG.\n\
Input parameters include:\n\
  -view_exact_sol    : write exact solution vector to stdout\n\
  -m  <mesh_x>       : number of mesh points in x-direction\n\
  -n  <mesh_n>       : number of mesh points in y-direction\n\
  -bs                : number of variables on each mesh vertex \n\n";

/*
  Simple example is used to test PCHMG
*/
#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;    /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PetscReal      norm;     /* norm of solution error */
  PetscInt       i,j,Ii,J,Istart,Iend,m = 8,n = 7,its,bs=1,II,JJ,jj;
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscScalar    v;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n*bs,m*n*bs);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  for (Ii=Istart/bs; Ii<Iend/bs; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii - n;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        ierr = MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    if (i<m-1) {
      J = Ii + n;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        ierr = MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    if (j>0) {
      J = Ii - 1;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        ierr = MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    if (j<n-1) {
      J = Ii + 1;
      for (jj=0; jj<bs; jj++) {
        II = Ii*bs + jj;
        JJ = J*bs + jj;
        ierr = MatSetValues(A,1,&II,1,&JJ,&v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    v = 4.0;
    for (jj=0; jj<bs; jj++) {
      II = Ii*bs + jj;
      ierr = MatSetValues(A,1,&II,1,&II,&v,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&u,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  ierr = VecSet(u,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-view_exact_sol",&flg,NULL);CHKERRQ(ierr);
  if (flg) {ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-2/((m+1)*(n+1)),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex !single hypre

   test:
      suffix: hmg
      nsize: 2
      args: -ksp_monitor -pc_type hmg

   test:
      suffix: hmg_block
      nsize: 2
      args: -ksp_monitor -pc_type hmg -bs 2

TEST*/
