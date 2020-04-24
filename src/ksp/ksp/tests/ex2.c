
static char help[] = "Tests repeated solving linear system on 2 by 2 matrix provided by MUMPS developer, Dec 17, 2012.\n\n";
/*
We have investigated the problem further, and we have
been able to reproduce it and obtain an erroneous
solution with an even smaller, 2x2, matrix:
    [1 2]
    [2 3]
and a right-hand side vector with all ones (1,1)
The correct solution is the vector (-1,1), in both solves.

mpiexec -n 2 ./ex2 -ksp_type preonly -pc_type lu  -pc_factor_mat_solver_type mumps  -mat_mumps_icntl_7 6 -mat_mumps_cntl_1 0.99

With this combination of options, I get off-diagonal pivots during the
factorization, which is the cause of the problem (different isol_loc
returned in the second solve, whereas, as I understand it, Petsc expects
isol_loc not to change between successive solves).
*/

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscErrorCode ierr;
  PetscInt       N = 2,rowidx,colidx;
  Vec            u,b,r;
  KSP            ksp;
  PetscReal      norm;
  PetscMPIInt    rank,size;
  PetscScalar    v;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* create stiffness matrix C = [1 2; 2 3] */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  if (!rank) {
    rowidx = 0; colidx = 0; v = 1.0;
    ierr   = MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES);CHKERRQ(ierr);
    rowidx = 0; colidx = 1; v = 2.0;
    ierr   = MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES);CHKERRQ(ierr);

    rowidx = 1; colidx = 0; v = 2.0;
    ierr   = MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES);CHKERRQ(ierr);
    rowidx = 1; colidx = 1; v = 3.0;
    ierr   = MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* create right hand side and solution */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&r);CHKERRQ(ierr);
  ierr = VecSet(u,0.0);CHKERRQ(ierr);
  ierr = VecSet(b,1.0);CHKERRQ(ierr);

  /* solve linear system C*u = b */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,C,C);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);

  /* check residual r = C*u - b */
  ierr = MatMult(C,u,r);CHKERRQ(ierr);
  ierr = VecAXPY(r,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"|| C*u - b|| = %g\n",(double)norm);CHKERRQ(ierr);

  /* solve C^T*u = b twice */
  ierr = KSPSolveTranspose(ksp,b,u);CHKERRQ(ierr);
  /* check residual r = C^T*u - b */
  ierr = MatMultTranspose(C,u,r);CHKERRQ(ierr);
  ierr = VecAXPY(r,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"|| C^T*u - b|| =  %g\n",(double)norm);CHKERRQ(ierr);

  ierr = KSPSolveTranspose(ksp,b,u);CHKERRQ(ierr);
  ierr = MatMultTranspose(C,u,r);CHKERRQ(ierr);
  ierr = VecAXPY(r,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"|| C^T*u - b|| =  %g\n",(double)norm);CHKERRQ(ierr);

  /* solve C*u = b again */
  ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);
  ierr = MatMult(C,u,r);CHKERRQ(ierr);
  ierr = VecAXPY(r,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"|| C*u - b|| = %g\n",(double)norm);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
