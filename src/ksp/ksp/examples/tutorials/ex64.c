/*
 * ex64.c
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

/*T
   Concepts: KSP^Additive Schwarz Method (GASM) with user-defined subdomains
   Processors: n
T*/

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
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscBool      flg;
  PetscBool       user_set_subdomains;
  PetscScalar     v, one = 1.0;
  MatPartitioning part;
  IS              coarseparts,fineparts;
  IS              is,isn,isrows;
  MPI_Comm        comm;
  PetscReal       e;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex62","PC");CHKERRQ(ierr);
  m = 15;
  ierr = PetscOptionsInt("-M", "Number of mesh points in the x-direction","PCGASMCreateSubdomains2D",m,&m,NULL);CHKERRQ(ierr);
  n = 17;
  ierr = PetscOptionsInt("-N","Number of mesh points in the y-direction","PCGASMCreateSubdomains2D",n,&n,NULL);CHKERRQ(ierr);
  user_set_subdomains = PETSC_FALSE;
  ierr = PetscOptionsBool("-user_set_subdomains","Use the user-specified 2D tiling of mesh by subdomains","PCGASMCreateSubdomains2D",user_set_subdomains,&user_set_subdomains,NULL);CHKERRQ(ierr);
  M = 2;
  ierr = PetscOptionsInt("-Mdomains","Number of subdomain tiles in the x-direction","PCGASMSetSubdomains2D",M,&M,NULL);CHKERRQ(ierr);
  N = 1;
  ierr = PetscOptionsInt("-Ndomains","Number of subdomain tiles in the y-direction","PCGASMSetSubdomains2D",N,&N,NULL);CHKERRQ(ierr);
  overlap = 1;
  ierr = PetscOptionsInt("-overlap","Size of tile overlap.","PCGASMSetSubdomains2D",overlap,&overlap,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* -------------------------------------------------------------------
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     ------------------------------------------------------------------- */

  /*
     Assemble the matrix for the five point stencil, YET AGAIN
  */
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
    Partition the graph of the matrix
  */
  ierr = MatPartitioningCreate(comm,&part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part,A);CHKERRQ(ierr);
  ierr = MatPartitioningSetType(part,MATPARTITIONINGHIERARCH);CHKERRQ(ierr);
  ierr = MatPartitioningHierarchicalSetNcoarseparts(part,2);CHKERRQ(ierr);
  ierr = MatPartitioningHierarchicalSetNfineparts(part,2);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
  /* get new processor owner number of each vertex */
  ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
  /* get coarse parts according to which we rearrange the matrix */
  ierr = MatPartitioningHierarchicalGetCoarseparts(part,&coarseparts);CHKERRQ(ierr);
  /* fine parts are used with coarse parts */
  ierr = MatPartitioningHierarchicalGetFineparts(part,&fineparts);CHKERRQ(ierr);
  /* get new global number of each old global number */
  ierr = ISPartitioningToNumbering(is,&isn);CHKERRQ(ierr);
  ierr = ISBuildTwoSided(is,NULL,&isrows);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(A,isrows,isrows,MAT_INITIAL_MATRIX,&perA);CHKERRQ(ierr);
  ierr = MatCreateVecs(perA,&b,NULL);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecSet(u,one);CHKERRQ(ierr);
  ierr = MatMult(perA,u,b);CHKERRQ(ierr);
  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = KSPSetOperators(ksp,perA,perA);CHKERRQ(ierr);

  /*
     Set the default preconditioner for this program to be GASM
  */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCGASM);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  /* -------------------------------------------------------------------
                      Solve the linear system
     ------------------------------------------------------------------- */

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  /* -------------------------------------------------------------------
                      Compare result to the exact solution
     ------------------------------------------------------------------- */
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY, &e);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-print_error",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Infinity norm of the error: %g\n", (double)e);CHKERRQ(ierr);
  }
  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&perA);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&coarseparts);CHKERRQ(ierr);
  ierr = ISDestroy(&fineparts);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&isn);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 4
      requires: superlu_dist parmetis
      args: -ksp_monitor -pc_gasm_overlap 1 -sub_pc_type lu -sub_pc_factor_mat_solver_type superlu_dist -pc_gasm_total_subdomains 2

TEST*/
