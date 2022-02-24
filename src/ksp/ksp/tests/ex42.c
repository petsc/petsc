
static char help[] = "Solves a linear system in parallel with MINRES. Modified from ../tutorials/ex2.c \n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b;      /* approx solution, RHS */
  Mat            A;        /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PetscInt       Ii,Istart,Iend,m = 11;
  PetscErrorCode ierr;
  PetscScalar    v;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  /* Create parallel diagonal matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatMPIAIJSetPreallocation(A,1,NULL,1,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(A,1,NULL));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));

  for (Ii=Istart; Ii<Iend; Ii++) {
    v = (PetscReal)Ii+1;
    CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }
  /* Make A sigular */
  Ii = m - 1; /* last diagonal entry */
  v  = 0.0;
  CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* A is symmetric. Set symmetric flag to enable KSP_type = minres */
  CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  CHKERRQ(VecSetSizes(b,PETSC_DECIDE,m));
  CHKERRQ(VecSetFromOptions(b));
  CHKERRQ(VecDuplicate(b,&x));
  CHKERRQ(VecSet(x,1.0));
  CHKERRQ(MatMult(A,x,b));
  CHKERRQ(VecSet(x,0.0));

  /* Create linear solver context */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,b,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Free work space. */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -ksp_type minres -pc_type none -ksp_converged_reason

   test:
      suffix: 2
      nsize: 3
      args: -ksp_type minres -pc_type none -ksp_converged_reason

TEST*/
