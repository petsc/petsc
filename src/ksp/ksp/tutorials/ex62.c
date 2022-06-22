static char help[] = "Illustrates use of PCGASM.\n\
The Generalized Additive Schwarz Method for solving a linear system in parallel with KSP.  The\n\
code indicates the procedure for setting user-defined subdomains.\n\
See section 'ex62' below for command-line options.\n\
Without -user_set_subdomains, the general PCGASM options are meaningful:\n\
  -pc_gasm_total_subdomains\n\
  -pc_gasm_print_subdomains\n\
\n";

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

PetscErrorCode AssembleMatrix(Mat,PetscInt m,PetscInt n);

int main(int argc,char **args)
{
  Vec            x,b,u;                  /* approx solution, RHS, exact solution */
  Mat            A;                      /* linear system matrix */
  KSP            ksp;                    /* linear solver context */
  PC             pc;                     /* PC context */
  IS             *inneris,*outeris;      /* array of index sets that define the subdomains */
  PetscInt       overlap;                /* width of subdomain overlap */
  PetscInt       Nsub;                   /* number of subdomains */
  PetscInt       m,n;                    /* mesh dimensions in x- and y- directions */
  PetscInt       M,N;                    /* number of subdomains in x- and y- directions */
  PetscMPIInt    size;
  PetscBool      flg=PETSC_FALSE;
  PetscBool      user_set_subdomains=PETSC_FALSE;
  PetscReal      one,e;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex62","PCGASM");
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
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(AssembleMatrix(A,m,n));

  /*
     Create and set vectors
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecSetSizes(b,PETSC_DECIDE,m*n));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecDuplicate(b,&u));
  PetscCall(VecDuplicate(b,&x));
  one  = 1.0;
  PetscCall(VecSet(u,one));
  PetscCall(MatMult(A,u,b));

  /*
     Create linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  PetscCall(KSPSetOperators(ksp,A,A));

  /*
     Set the default preconditioner for this program to be GASM
  */
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCGASM));

  /* -------------------------------------------------------------------
                  Define the problem decomposition
     ------------------------------------------------------------------- */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Basic method, should be sufficient for the needs of many users.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Set the overlap, using the default PETSc decomposition via
         PCGASMSetOverlap(pc,overlap);
     Could instead use the option -pc_gasm_overlap <ovl>

     Set the total number of blocks via -pc_gasm_blocks <blks>
     Note:  The GASM default is to use 1 block per processor.  To
     experiment on a single processor with various overlaps, you
     must specify use of multiple blocks!
  */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       More advanced method, setting user-defined subdomains
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Firstly, create index sets that define the subdomains.  The utility
     routine PCGASMCreateSubdomains2D() is a simple example, which partitions
     the 2D grid into MxN subdomains with an optional overlap.
     More generally, the user should write a custom routine for a particular
     problem geometry.

     Then call PCGASMSetLocalSubdomains() with resulting index sets
     to set the subdomains for the GASM preconditioner.
  */

  if (user_set_subdomains) { /* user-control version */
    PetscCall(PCGASMCreateSubdomains2D(pc, m,n,M,N,1,overlap,&Nsub,&inneris,&outeris));
    PetscCall(PCGASMSetSubdomains(pc,Nsub,inneris,outeris));
    PetscCall(PCGASMDestroySubdomains(Nsub,&inneris,&outeris));
    flg  = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-subdomain_view",&flg,NULL));
    if (flg) {
      PetscInt i;
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Nmesh points: %" PetscInt_FMT " x %" PetscInt_FMT "; subdomain partition: %" PetscInt_FMT " x %" PetscInt_FMT "; overlap: %" PetscInt_FMT "; Nsub: %" PetscInt_FMT "\n",m,n,M,N,overlap,Nsub));
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Outer IS:\n"));
      for (i=0; i<Nsub; i++) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"  outer IS[%" PetscInt_FMT "]\n",i));
        PetscCall(ISView(outeris[i],PETSC_VIEWER_STDOUT_SELF));
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Inner IS:\n"));
      for (i=0; i<Nsub; i++) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"  inner IS[%" PetscInt_FMT "]\n",i));
        PetscCall(ISView(inneris[i],PETSC_VIEWER_STDOUT_SELF));
      }
    }
  } else { /* basic setup */
    PetscCall(KSPSetFromOptions(ksp));
  }

  /* -------------------------------------------------------------------
                Set the linear solvers for the subblocks
     ------------------------------------------------------------------- */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Basic method, should be sufficient for the needs of most users.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     By default, the GASM preconditioner uses the same solver on each
     block of the problem.  To set the same solver options on all blocks,
     use the prefix -sub before the usual PC and KSP options, e.g.,
          -sub_pc_type <pc> -sub_ksp_type <ksp> -sub_ksp_rtol 1.e-4

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Advanced method, setting different solvers for various blocks.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Note that each block's KSP context is completely independent of
     the others, and the full range of uniprocessor KSP options is
     available for each block.

     - Use PCGASMGetSubKSP() to extract the array of KSP contexts for
       the local blocks.
     - See ex7.c for a simple example of setting different linear solvers
       for the individual blocks for the block Jacobi method (which is
       equivalent to the GASM method with zero overlap).
  */

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-user_set_subdomain_solvers",&flg,NULL));
  if (flg) {
    KSP       *subksp;        /* array of KSP contexts for local subblocks */
    PetscInt  i,nlocal,first;   /* number of local subblocks, first local subblock */
    PC        subpc;          /* PC context for subblock */
    PetscBool isasm;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"User explicitly sets subdomain solvers.\n"));

    /*
       Set runtime options
    */
    PetscCall(KSPSetFromOptions(ksp));

    /*
       Flag an error if PCTYPE is changed from the runtime options
     */
    PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCGASM,&isasm));
    PetscCheck(isasm,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Cannot Change the PCTYPE when manually changing the subdomain solver settings");

    /*
       Call KSPSetUp() to set the block Jacobi data structures (including
       creation of an internal KSP context for each block).

       Note: KSPSetUp() MUST be called before PCGASMGetSubKSP().
    */
    PetscCall(KSPSetUp(ksp));

    /*
       Extract the array of KSP contexts for the local blocks
    */
    PetscCall(PCGASMGetSubKSP(pc,&nlocal,&first,&subksp));

    /*
       Loop over the local blocks, setting various KSP options
       for each block.
    */
    for (i=0; i<nlocal; i++) {
      PetscCall(KSPGetPC(subksp[i],&subpc));
      PetscCall(PCSetType(subpc,PCILU));
      PetscCall(KSPSetType(subksp[i],KSPGMRES));
      PetscCall(KSPSetTolerances(subksp[i],1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
    }
  } else {
    /*
       Set runtime options
    */
    PetscCall(KSPSetFromOptions(ksp));
  }

  /* -------------------------------------------------------------------
                      Solve the linear system
     ------------------------------------------------------------------- */

  PetscCall(KSPSolve(ksp,b,x));

  /* -------------------------------------------------------------------
        Assemble the matrix again to test repeated setup and solves.
     ------------------------------------------------------------------- */

  PetscCall(AssembleMatrix(A,m,n));
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
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode AssembleMatrix(Mat A,PetscInt m,PetscInt n)
{
  PetscInt       i,j,Ii,J,Istart,Iend;
  PetscScalar    v;

  PetscFunctionBegin;
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

  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 2D_1
      args: -M 7 -N 9 -user_set_subdomains -Mdomains 1 -Ndomains 3 -overlap 1 -print_error -pc_gasm_print_subdomains

   test:
      suffix: 2D_2
      nsize: 2
      args: -M 7 -N 9 -user_set_subdomains -Mdomains 1 -Ndomains 3 -overlap 1 -print_error -pc_gasm_print_subdomains

   test:
      suffix: 2D_3
      nsize: 3
      args: -M 7 -N 9 -user_set_subdomains -Mdomains 1 -Ndomains 3 -overlap 1 -print_error -pc_gasm_print_subdomains

   test:
      suffix: hp
      nsize: 4
      requires: superlu_dist
      args: -M 7 -N 9 -pc_gasm_overlap 1 -sub_pc_type lu -sub_pc_factor_mat_solver_type superlu_dist -ksp_monitor -print_error -pc_gasm_total_subdomains 2 -pc_gasm_use_hierachical_partitioning 1
      output_file: output/ex62.out
      TODO: bug, triggers New nonzero at (0,15) caused a malloc in MatCreateSubMatrices_MPIAIJ_SingleIS_Local

   test:
      suffix: superlu_dist_1
      requires: superlu_dist
      args: -M 7 -N 9 -print_error -pc_gasm_total_subdomains 1 -pc_gasm_print_subdomains -sub_pc_type lu -sub_pc_factor_mat_solver_type superlu_dist

   test:
      suffix: superlu_dist_2
      nsize: 2
      requires: superlu_dist
      args: -M 7 -N 9 -print_error -pc_gasm_total_subdomains 1 -pc_gasm_print_subdomains -sub_pc_type lu -sub_pc_factor_mat_solver_type superlu_dist

   test:
      suffix: superlu_dist_3
      nsize: 3
      requires: superlu_dist
      args: -M 7 -N 9 -print_error -pc_gasm_total_subdomains 2 -pc_gasm_print_subdomains -sub_pc_type lu -sub_pc_factor_mat_solver_type superlu_dist

   test:
      suffix: superlu_dist_4
      nsize: 4
      requires: superlu_dist
      args: -M 7 -N 9 -print_error -pc_gasm_total_subdomains 2 -pc_gasm_print_subdomains -sub_pc_type lu -sub_pc_factor_mat_solver_type superlu_dist

TEST*/
