#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include "del2mat.h"

#define DEL2MAT_MULT   ((void(*)(void))Del2Mat_mult)
#define DEL2MAT_DIAG   ((void(*)(void))Del2Mat_diag)

int main(int argc,char **argv)
{
  PetscInt n;
  PetscScalar h;
  Del2Mat shell;
  Mat A;
  Vec x,b;
  KSP ksp;
  PC  pc;
  PetscMPIInt size;
  /* PETSc initialization  */
  PetscInitialize(&argc, &argv, NULL, NULL);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) {
    PetscPrintf(PETSC_COMM_WORLD, "This a sequential example\n");
    PetscFinalize();
    return 1;
  }
  /* number of nodes in each direction
   * excluding those at the boundary */
  n = 32;
  h = 1.0/(n+1); /* grid spacing */
  /* setup linear system (shell) matrix */
  MatCreate(PETSC_COMM_SELF, &A);
  MatSetSizes(A, n*n*n, n*n*n, n*n*n, n*n*n);
  MatSetType(A, MATSHELL);
  shell.N = n;
  PetscMalloc((n+2)*(n+2)*(n+2)*sizeof(PetscScalar),&shell.F);
  PetscMemzero(shell.F, (n+2)*(n+2)*(n+2)*sizeof(PetscScalar));
  MatShellSetContext(A, (void**)&shell);
  MatShellSetOperation(A, MATOP_MULT,           DEL2MAT_MULT);
  MatShellSetOperation(A, MATOP_MULT_TRANSPOSE, DEL2MAT_MULT);
  MatShellSetOperation(A, MATOP_GET_DIAGONAL,   DEL2MAT_DIAG);
  MatSetUp(A);
  /* setup linear system vectors */
  MatCreateVecs(A, &x, &b);
  VecSet(x, 0);
  VecSet(b, 1);
  /* setup Krylov linear solver */
  KSPCreate(PETSC_COMM_SELF, &ksp);
  KSPGetPC(ksp, &pc);
  KSPSetType(ksp, KSPCG); /* use conjugate gradients */
  PCSetType(pc, PCNONE);  /* with no preconditioning */
  KSPSetFromOptions(ksp);
  /* iteratively solve linear system of equations A*x=b */
  KSPSetOperators(ksp,A,A);
  KSPSolve(ksp, b, x);
  /* scale solution vector to account for grid spacing */
  VecScale(x, h*h);
  /* free memory and destroy objects */
  PetscFree(shell.F);
  VecDestroy(&x);
  VecDestroy(&b);
  MatDestroy(&A);
  KSPDestroy(&ksp);
  /* finalize PETSc */
  PetscFinalize();
  return 0;
}
