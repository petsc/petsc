
static char help[] = "Solves an ill-conditioned tridiagonal linear system with KSP for testing GMRES breakdown tolerance.\n\n";

/*T
   Concepts: KSP^solving an ill-conditioned system of linear equations for testing GMRES breakdown tolerance
   Processors: 1
T*/

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x, b, u;      /* approx solution, RHS, exact solution */
  Mat            A;            /* linear system matrix */
  KSP            ksp;          /* linear solver context */
  PetscErrorCode ierr;
  PetscInt       i,n = 10,col[3];
  PetscMPIInt    size;
  PetscScalar    value[3];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(PetscObjectSetName((PetscObject) x,"Solution"));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(VecDuplicate(x,&u));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  /*
     Set big off-diag values to make the system ill-conditioned
  */
  value[0] = 10.0; value[1] = 2.0; value[2] = 1.0;
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

  CHKERRQ(VecSet(u,1.0));
  CHKERRQ(MatMult(A,u,b));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,b,x));

  CHKERRQ(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
  CHKERRQ(PetscOptionsInsertString(NULL,"-ksp_type preonly -ksp_initial_guess_nonzero false"));
  CHKERRQ(PetscOptionsClearValue(NULL,"-ksp_converged_reason"));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,b,x));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: double !complex
      args: -ksp_rtol  1e-18 -pc_type sor -ksp_converged_reason -ksp_gmres_breakdown_tolerance 1.e-9
      output_file: output/ex70.out

TEST*/
