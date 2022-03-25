
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* create stiffness matrix C = [1 2; 2 3] */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  if (rank == 0) {
    rowidx = 0; colidx = 0; v = 1.0;
    PetscCall(MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES));
    rowidx = 0; colidx = 1; v = 2.0;
    PetscCall(MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES));

    rowidx = 1; colidx = 0; v = 2.0;
    PetscCall(MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES));
    rowidx = 1; colidx = 1; v = 3.0;
    PetscCall(MatSetValues(C,1,&rowidx,1,&colidx,&v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* create right hand side and solution */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,N));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(VecDuplicate(u,&r));
  PetscCall(VecSet(u,0.0));
  PetscCall(VecSet(b,1.0));

  /* solve linear system C*u = b */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,C,C));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,u));

  /* check residual r = C*u - b */
  PetscCall(MatMult(C,u,r));
  PetscCall(VecAXPY(r,-1.0,b));
  PetscCall(VecNorm(r,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"|| C*u - b|| = %g\n",(double)norm));

  /* solve C^T*u = b twice */
  PetscCall(KSPSolveTranspose(ksp,b,u));
  /* check residual r = C^T*u - b */
  PetscCall(MatMultTranspose(C,u,r));
  PetscCall(VecAXPY(r,-1.0,b));
  PetscCall(VecNorm(r,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"|| C^T*u - b|| =  %g\n",(double)norm));

  PetscCall(KSPSolveTranspose(ksp,b,u));
  PetscCall(MatMultTranspose(C,u,r));
  PetscCall(VecAXPY(r,-1.0,b));
  PetscCall(VecNorm(r,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"|| C^T*u - b|| =  %g\n",(double)norm));

  /* solve C*u = b again */
  PetscCall(KSPSolve(ksp,b,u));
  PetscCall(MatMult(C,u,r));
  PetscCall(VecAXPY(r,-1.0,b));
  PetscCall(VecNorm(r,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"|| C*u - b|| = %g\n",(double)norm));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}
