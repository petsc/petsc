
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
  PetscInt       N = 2,rowidx,colidx;
  Vec            u,b,r;
  KSP            ksp;
  PetscReal      norm;
  PetscMPIInt    rank,size;
  PetscScalar    v;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* create stiffness matrix C = [1 2; 2 3] */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  if (rank == 0) {
    rowidx = 0; colidx = 0; v = 1.0;
    CHKERRQ(MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES));
    rowidx = 0; colidx = 1; v = 2.0;
    CHKERRQ(MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES));

    rowidx = 1; colidx = 0; v = 2.0;
    CHKERRQ(MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES));
    rowidx = 1; colidx = 1; v = 3.0;
    CHKERRQ(MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* create right hand side and solution */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,N));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecDuplicate(u,&r));
  CHKERRQ(VecSet(u,0.0));
  CHKERRQ(VecSet(b,1.0));

  /* solve linear system C*u = b */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,C,C));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,b,u));

  /* check residual r = C*u - b */
  CHKERRQ(MatMult(C,u,r));
  CHKERRQ(VecAXPY(r,-1.0,b));
  CHKERRQ(VecNorm(r,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"|| C*u - b|| = %g\n",(double)norm));

  /* solve C^T*u = b twice */
  CHKERRQ(KSPSolveTranspose(ksp,b,u));
  /* check residual r = C^T*u - b */
  CHKERRQ(MatMultTranspose(C,u,r));
  CHKERRQ(VecAXPY(r,-1.0,b));
  CHKERRQ(VecNorm(r,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"|| C^T*u - b|| =  %g\n",(double)norm));

  CHKERRQ(KSPSolveTranspose(ksp,b,u));
  CHKERRQ(MatMultTranspose(C,u,r));
  CHKERRQ(VecAXPY(r,-1.0,b));
  CHKERRQ(VecNorm(r,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"|| C^T*u - b|| =  %g\n",(double)norm));

  /* solve C*u = b again */
  CHKERRQ(KSPSolve(ksp,b,u));
  CHKERRQ(MatMult(C,u,r));
  CHKERRQ(VecAXPY(r,-1.0,b));
  CHKERRQ(VecNorm(r,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"|| C*u - b|| = %g\n",(double)norm));

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
  return 0;
}
