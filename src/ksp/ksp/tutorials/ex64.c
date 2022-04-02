/*
 *
 *  Created on: Aug 10, 2015
 *      Author: Fande Kong  <fdkong.jd@gmail.com>
 */

static char help[] = "Illustrates use of the preconditioner GASM.\n \
   using hierarchical partitioning and MatIncreaseOverlapSplit \
        -pc_gasm_total_subdomains\n \
        -pc_gasm_print_subdomains\n \n";

/*
   Note:  This example focuses on setting the subdomains for the GASM
   preconditioner for a problem on a 2D rectangular grid.  See ex1.c
   and ex2.c for more detailed comments on the basic usage of KSP
   (including working with matrices and vectors).

   The GASM preconditioner is fully parallel.  The user-space routine
   CreateSubdomains2D that computes the domain decomposition is also parallel
   and attempts to generate both subdomains straddling processors and multiple
   domains per processor.

   This matrix in this linear system arises from the discretized Laplacian,
   and thus is not very interesting in terms of experimenting with variants
   of the GASM preconditioner.
*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>
#include <petscmat.h>

int main(int argc,char **args)
{
  Vec            x,b,u;                  /* approx solution, RHS, exact solution */
  Mat            A,perA;                      /* linear system matrix */
  KSP            ksp;                    /* linear solver context */
  PC             pc;                     /* PC context */
  PetscInt       overlap;                /* width of subdomain overlap */
  PetscInt       m,n;                    /* mesh dimensions in x- and y- directions */
  PetscInt       M,N;                    /* number of subdomains in x- and y- directions */
  PetscInt       i,j,Ii,J,Istart,Iend;
  PetscMPIInt    size;
  PetscBool      flg;
  PetscBool       user_set_subdomains;
  PetscScalar     v, one = 1.0;
  MatPartitioning part;
  IS              coarseparts,fineparts;
  IS              is,isn,isrows;
  MPI_Comm        comm;
  PetscReal       e;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex62","PC");
  m = 15;
  PetscCall(PetscOptionsInt("-M", "Number of mesh points in the x-direction","PCGASMCreateSubdomains2D",m,&m,NULL));
  n = 17;
  PetscCall(PetscOptionsInt("-N","Number of mesh points in the y-direction","PCGASMCreateSubdomains2D",n,&n,NULL));
  user_set_subdomains = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-user_set_subdomains","Use the user-specified 2D tiling of mesh by subdomains","PCGASMCreateSubdomains2D",user_set_subdomains,&user_set_subdomains,NULL));
  M = 2;
  PetscCall(PetscOptionsInt("-Mdomains","Number of subdomain tiles in the x-direction","PCGASMSetSubdomains2D",M,&M,NULL));
  N = 1;
  PetscCall(PetscOptionsInt("-Ndomains","Number of subdomain tiles in the y-direction","PCGASMSetSubdomains2D",N,&N,NULL));
  overlap = 1;
  PetscCall(PetscOptionsInt("-overlap","Size of tile overlap.","PCGASMSetSubdomains2D",overlap,&overlap,NULL));
  PetscOptionsEnd();

  /* -------------------------------------------------------------------
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     ------------------------------------------------------------------- */

  /*
     Assemble the matrix for the five point stencil, YET AGAIN
  */
  PetscCall(MatCreate(comm,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
    v = 4.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /*
    Partition the graph of the matrix
  */
  PetscCall(MatPartitioningCreate(comm,&part));
  PetscCall(MatPartitioningSetAdjacency(part,A));
  PetscCall(MatPartitioningSetType(part,MATPARTITIONINGHIERARCH));
  PetscCall(MatPartitioningHierarchicalSetNcoarseparts(part,2));
  PetscCall(MatPartitioningHierarchicalSetNfineparts(part,2));
  PetscCall(MatPartitioningSetFromOptions(part));
  /* get new processor owner number of each vertex */
  PetscCall(MatPartitioningApply(part,&is));
  /* get coarse parts according to which we rearrange the matrix */
  PetscCall(MatPartitioningHierarchicalGetCoarseparts(part,&coarseparts));
  /* fine parts are used with coarse parts */
  PetscCall(MatPartitioningHierarchicalGetFineparts(part,&fineparts));
  /* get new global number of each old global number */
  PetscCall(ISPartitioningToNumbering(is,&isn));
  PetscCall(ISBuildTwoSided(is,NULL,&isrows));
  PetscCall(MatCreateSubMatrix(A,isrows,isrows,MAT_INITIAL_MATRIX,&perA));
  PetscCall(MatCreateVecs(perA,&b,NULL));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecDuplicate(b,&u));
  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecSet(u,one));
  PetscCall(MatMult(perA,u,b));
  PetscCall(KSPCreate(comm,&ksp));
  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  PetscCall(KSPSetOperators(ksp,perA,perA));

  /*
     Set the default preconditioner for this program to be GASM
  */
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCGASM));
  PetscCall(KSPSetFromOptions(ksp));
  /* -------------------------------------------------------------------
                      Solve the linear system
     ------------------------------------------------------------------- */

  PetscCall(KSPSolve(ksp,b,x));
  /* -------------------------------------------------------------------
                      Compare result to the exact solution
     ------------------------------------------------------------------- */
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_INFINITY, &e));

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-print_error",&flg,NULL));
  if (flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Infinity norm of the error: %g\n", (double)e));
  }
  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&perA));
  PetscCall(ISDestroy(&is));
  PetscCall(ISDestroy(&coarseparts));
  PetscCall(ISDestroy(&fineparts));
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&isn));
  PetscCall(MatPartitioningDestroy(&part));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4
      requires: superlu_dist parmetis
      args: -ksp_monitor -pc_gasm_overlap 1 -sub_pc_type lu -sub_pc_factor_mat_solver_type superlu_dist -pc_gasm_total_subdomains 2

TEST*/
