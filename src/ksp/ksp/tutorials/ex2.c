
static char help[] = "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_y>       : number of mesh points in y-direction\n\n";

/*T
   Concepts: KSP^basic parallel example;
   Concepts: KSP^Laplacian, 2d
   Concepts: Laplacian, 2d
   Processors: n
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.
*/
#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;    /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PetscReal      norm;     /* norm of solution error */
  PetscInt       i,j,Ii,J,Istart,Iend,m = 8,n = 7,its;
  PetscBool      flg;
  PetscScalar    v;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partitioning of the matrix is
     determined by PETSc at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  PetscCall(MatSeqAIJSetPreallocation(A,5,NULL));
  PetscCall(MatSeqSBAIJSetPreallocation(A,1,5,NULL));
  PetscCall(MatMPISBAIJSetPreallocation(A,1,5,NULL,5,NULL));
  PetscCall(MatMPISELLSetPreallocation(A,5,NULL,5,NULL));
  PetscCall(MatSeqSELLSetPreallocation(A,5,NULL));

  /*
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned.
  */
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));

  /*
     Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global rows and columns of matrix entries.

     Note: this uses the less common natural ordering that orders first
     all the unknowns for x = h then for x = 2h etc; Hence you see J = Ii +- n
     instead of J = I +- m as you might expect. The more standard ordering
     would first do all variables for y = h, then y = 2h etc.

   */
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    v = 4.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner */
  PetscCall(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));

  /*
     Create parallel vectors.
      - We form 1 vector from scratch and then duplicate as needed.
      - When using VecCreate(), VecSetSizes and VecSetFromOptions()
        in this example, we specify only the
        vector's global dimension; the parallel partitioning is determined
        at runtime.
      - When solving a linear system, the vectors and matrices MUST
        be partitioned accordingly.  PETSc automatically generates
        appropriately partitioned matrices and vectors when MatCreate()
        and VecCreate() are used with the same communicator.
      - The user can alternatively specify the local vector and matrix
        dimensions when more sophisticated partitioning is needed
        (replacing the PETSC_DECIDE argument in the VecSetSizes() statement
        below).
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,m*n));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(VecDuplicate(b,&x));

  /*
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;
  */
  PetscCall(VecSet(u,1.0));
  PetscCall(MatMult(A,u,b));

  /*
     View the exact solution vector if desired
  */
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-view_exact_sol",&flg,NULL));
  if (flg) PetscCall(VecView(u,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  PetscCall(KSPSetOperators(ksp,A,A));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following two statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions().  All of these defaults can be
       overridden at runtime, as indicated below.
  */
  PetscCall(KSPSetTolerances(ksp,1.e-2/((m+1)*(n+1)),1.e-50,PETSC_DEFAULT,PETSC_DEFAULT));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(KSPSolve(ksp,b,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));

  /*
     Print convergence information.  PetscPrintf() produces a single
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));  PetscCall(MatDestroy(&A));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !single

   test:
      suffix: chebyest_1
      args: -m 80 -n 80 -ksp_pc_side right -pc_type ksp -ksp_ksp_type chebyshev -ksp_ksp_max_it 5 -ksp_ksp_chebyshev_esteig 0.9,0,0,1.1 -ksp_monitor_short

   test:
      suffix: chebyest_2
      args: -m 80 -n 80 -ksp_pc_side right -pc_type ksp -ksp_ksp_type chebyshev -ksp_ksp_max_it 5 -ksp_ksp_chebyshev_esteig 0.9,0,0,1.1 -ksp_esteig_ksp_type cg -ksp_monitor_short

   test:
      args: -ksp_monitor_short -m 5 -n 5 -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2
      nsize: 2
      args: -ksp_monitor_short -m 5 -n 5 -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 3
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 4
      args: -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 5
      nsize: 2
      args: -ksp_monitor_short -m 5 -n 5 -mat_view draw -ksp_gmres_cgs_refinement_type refine_always -nox
      output_file: output/ex2_2.out

   test:
      suffix: bjacobi
      nsize: 4
      args: -pc_type bjacobi -pc_bjacobi_blocks 1 -ksp_monitor_short -sub_pc_type jacobi -sub_ksp_type gmres

   test:
      suffix: bjacobi_2
      nsize: 4
      args: -pc_type bjacobi -pc_bjacobi_blocks 2 -ksp_monitor_short -sub_pc_type jacobi -sub_ksp_type gmres -ksp_view

   test:
      suffix: bjacobi_3
      nsize: 4
      args: -pc_type bjacobi -pc_bjacobi_blocks 4 -ksp_monitor_short -sub_pc_type jacobi -sub_ksp_type gmres

   test:
      suffix: qmrcgs
      args: -ksp_type qmrcgs -pc_type ilu
      output_file: output/ex2_fbcgs.out

   test:
      suffix: qmrcgs_2
      nsize: 3
      args: -ksp_type qmrcgs -pc_type bjacobi
      output_file: output/ex2_fbcgs_2.out

   test:
      suffix: fbcgs
      args: -ksp_type fbcgs -pc_type ilu

   test:
      suffix: fbcgs_2
      nsize: 3
      args: -ksp_type fbcgsr -pc_type bjacobi

   test:
      suffix: groppcg
      args: -ksp_monitor_short -ksp_type groppcg -m 9 -n 9

   test:
      suffix: mkl_pardiso_cholesky
      requires: mkl_pardiso
      args: -ksp_type preonly -pc_type cholesky -mat_type sbaij -pc_factor_mat_solver_type mkl_pardiso

   test:
      suffix: mkl_pardiso_lu
      requires: mkl_pardiso
      args: -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mkl_pardiso

   test:
      suffix: pipebcgs
      args: -ksp_monitor_short -ksp_type pipebcgs -m 9 -n 9

   test:
      suffix: pipecg
      args: -ksp_monitor_short -ksp_type pipecg -m 9 -n 9

   test:
      suffix: pipecgrr
      args: -ksp_monitor_short -ksp_type pipecgrr -m 9 -n 9

   test:
      suffix: pipecr
      args: -ksp_monitor_short -ksp_type pipecr -m 9 -n 9

   test:
      suffix: pipelcg
      args: -ksp_monitor_short -ksp_type pipelcg -m 9 -n 9 -pc_type none -ksp_pipelcg_pipel 2 -ksp_pipelcg_lmax 2
      filter: grep -v "sqrt breakdown in iteration"

   test:
      suffix: sell
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -m 9 -n 9 -mat_type sell

   test:
      requires: mumps
      suffix: sell_mumps
      args: -ksp_type preonly -m 9 -n 12 -mat_type sell -pc_type lu -pc_factor_mat_solver_type mumps -pc_factor_mat_ordering_type natural

   test:
      suffix: telescope
      nsize: 4
      args: -m 100 -n 100 -ksp_converged_reason -pc_type telescope -pc_telescope_reduction_factor 4 -telescope_pc_type bjacobi

   test:
      suffix: umfpack
      requires: suitesparse
      args: -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type umfpack

   test:
      suffix: spqr
      requires: suitesparse
      args: -ksp_type preonly -pc_type qr -pc_factor_mat_solver_type spqr

   test:
     suffix: pc_symmetric
     args: -m 10 -n 9 -ksp_converged_reason -ksp_type gmres -ksp_pc_side symmetric -pc_type cholesky

   test:
      suffix: pipeprcg
      args: -ksp_monitor_short -ksp_type pipeprcg -m 9 -n 9

   test:
      suffix: pipeprcg_rcw
      args: -ksp_monitor_short -ksp_type pipeprcg -recompute_w false -m 9 -n 9

   test:
      suffix: pipecg2
      args: -ksp_monitor_short -ksp_type pipecg2 -m 9 -n 9 -ksp_norm_type {{preconditioned unpreconditioned natural}}

   test:
      suffix: pipecg2_2
      nsize: 4
      args: -ksp_monitor_short -ksp_type pipecg2 -m 15 -n 9 -ksp_norm_type {{preconditioned unpreconditioned natural}}

   test:
      suffix: hpddm
      nsize: 4
      requires: hpddm
      filter: sed -e "s/ iterations 9/ iterations 8/g"
      args: -ksp_converged_reason -ksp_type hpddm -ksp_hpddm_precision {{single double}shared output} -ksp_pc_side {{left right}shared output}

 TEST*/
