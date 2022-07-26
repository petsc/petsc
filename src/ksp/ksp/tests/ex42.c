
static char help[] = "Solves a linear system in parallel with MINRES. Modified from ../tutorials/ex2.c \n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b;      /* approx solution, RHS */
  Mat            A;        /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PetscInt       Ii,Istart,Iend,m = 11;
  PetscScalar    v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  /* Create parallel diagonal matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A,1,NULL,1,NULL));
  PetscCall(MatSeqAIJSetPreallocation(A,1,NULL));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));

  for (Ii=Istart; Ii<Iend; Ii++) {
    v = (PetscReal)Ii+1;
    PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }
  /* Make A sigular */
  Ii = m - 1; /* last diagonal entry */
  v  = 0.0;
  PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* A is symmetric. Set symmetric flag to enable KSP_type = minres */
  PetscCall(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecSetSizes(b,PETSC_DECIDE,m));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecSet(x,1.0));
  PetscCall(MatMult(A,x,b));
  PetscCall(VecSet(x,0.0));

  /* Create linear solver context */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Free work space. */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ksp_type minres -pc_type none -ksp_converged_reason

   test:
      suffix: 2
      nsize: 3
      args: -ksp_type minres -pc_type none -ksp_converged_reason

TEST*/
