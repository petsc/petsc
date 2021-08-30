
static char help[] = "Solves a linear system in parallel with KSP. Modified from ex2.c \n\
                      Illustrate how to use external packages MUMPS, SUPERLU and STRUMPACK \n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_y>       : number of mesh points in y-direction\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;    /* approx solution, RHS, exact solution */
  Mat            A,F;
  KSP            ksp;      /* linear solver context */
  PC             pc;
  PetscRandom    rctx;     /* random number generator context */
  PetscReal      norm;     /* norm of solution error */
  PetscInt       i,j,Ii,J,Istart,Iend,m = 8,n = 7,its;
  PetscErrorCode ierr;
  PetscBool      flg=PETSC_FALSE,flg_ilu=PETSC_FALSE,flg_ch=PETSC_FALSE;
#if defined(PETSC_HAVE_MUMPS)
  PetscBool      flg_mumps=PETSC_FALSE,flg_mumps_ch=PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_SUPERLU) || defined(PETSC_HAVE_SUPERLU_DIST)
  PetscBool      flg_superlu=PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_STRUMPACK)
  PetscBool      flg_strumpack=PETSC_FALSE;
#endif
  PetscScalar    v;
  PetscMPIInt    rank,size;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage;
#endif

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /*
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned.
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

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
  ierr = PetscLogStageRegister("Assembly", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner */
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

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
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  /*
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;  Alternatively, using the runtime option
     -random_sol forms a solution vector with random components.
  */
  ierr = PetscOptionsGetBool(NULL,NULL,"-random_exact_sol",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(u,rctx);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  } else {
    ierr = VecSet(u,1.0);CHKERRQ(ierr);
  }
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  /*
     View the exact solution vector if desired
  */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-view_exact_sol",&flg,NULL);CHKERRQ(ierr);
  if (flg) {ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  /*
    Example of how to use external package MUMPS
    Note: runtime options
          '-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -mat_mumps_icntl_7 3 -mat_mumps_icntl_1 0.0'
          are equivalent to these procedural calls
  */
#if defined(PETSC_HAVE_MUMPS)
  flg_mumps    = PETSC_FALSE;
  flg_mumps_ch = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_mumps_lu",&flg_mumps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_mumps_ch",&flg_mumps_ch,NULL);CHKERRQ(ierr);
  if (flg_mumps || flg_mumps_ch) {
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    PetscInt  ival,icntl;
    PetscReal val;
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    if (flg_mumps) {
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    } else if (flg_mumps_ch) {
      ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr); /* set MUMPS id%SYM=1 */
      ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
    }
    ierr = PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverType(pc);CHKERRQ(ierr); /* call MatGetFactor() to create F */
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);

    /* sequential ordering */
    icntl = 7; ival = 2;
    ierr = MatMumpsSetIcntl(F,icntl,ival);CHKERRQ(ierr);

    /* threshold for row pivot detection */
    ierr = MatMumpsSetIcntl(F,24,1);CHKERRQ(ierr);
    icntl = 3; val = 1.e-6;
    ierr = MatMumpsSetCntl(F,icntl,val);CHKERRQ(ierr);

    /* compute determinant of A */
    ierr = MatMumpsSetIcntl(F,33,1);CHKERRQ(ierr);
  }
#endif

  /*
    Example of how to use external package SuperLU
    Note: runtime options
          '-ksp_type preonly -pc_type ilu -pc_factor_mat_solver_type superlu -mat_superlu_ilu_droptol 1.e-8'
          are equivalent to these procedual calls
  */
#if defined(PETSC_HAVE_SUPERLU) || defined(PETSC_HAVE_SUPERLU_DIST)
  flg_ilu     = PETSC_FALSE;
  flg_superlu = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_superlu_lu",&flg_superlu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_superlu_ilu",&flg_ilu,NULL);CHKERRQ(ierr);
  if (flg_superlu || flg_ilu) {
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    if (flg_superlu) {
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    } else if (flg_ilu) {
      ierr = PCSetType(pc,PCILU);CHKERRQ(ierr);
    }
    if (size == 1) {
#if !defined(PETSC_HAVE_SUPERLU)
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This test requires SUPERLU");
#else
      ierr = PCFactorSetMatSolverType(pc,MATSOLVERSUPERLU);CHKERRQ(ierr);
#endif
    } else {
#if !defined(PETSC_HAVE_SUPERLU_DIST)
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This test requires SUPERLU_DIST");
#else
      ierr = PCFactorSetMatSolverType(pc,MATSOLVERSUPERLU_DIST);CHKERRQ(ierr);
#endif
    }
    ierr = PCFactorSetUpMatSolverType(pc);CHKERRQ(ierr); /* call MatGetFactor() to create F */
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUPERLU)
    if (size == 1) {
      ierr = MatSuperluSetILUDropTol(F,1.e-8);CHKERRQ(ierr);
    }
#endif
  }
#endif

  /*
    Example of how to use external package STRUMPACK
    Note: runtime options
          '-pc_type lu/ilu \
           -pc_factor_mat_solver_type strumpack \
           -mat_strumpack_reordering METIS \
           -mat_strumpack_colperm 0 \
           -mat_strumpack_hss_rel_tol 1.e-3 \
           -mat_strumpack_hss_min_sep_size 50 \
           -mat_strumpack_max_rank 100 \
           -mat_strumpack_leaf_size 4'
       are equivalent to these procedural calls

    We refer to the STRUMPACK-sparse manual, section 5, for more info on
    how to tune the preconditioner.
  */
#if defined(PETSC_HAVE_STRUMPACK)
  flg_ilu       = PETSC_FALSE;
  flg_strumpack = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_strumpack_lu",&flg_strumpack,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_strumpack_ilu",&flg_ilu,NULL);CHKERRQ(ierr);
  if (flg_strumpack || flg_ilu) {
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    if (flg_strumpack) {
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    } else if (flg_ilu) {
      ierr = PCSetType(pc,PCILU);CHKERRQ(ierr);
    }
#if !defined(PETSC_HAVE_STRUMPACK)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This test requires STRUMPACK");
#endif
    ierr = PCFactorSetMatSolverType(pc,MATSOLVERSTRUMPACK);CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverType(pc);CHKERRQ(ierr); /* call MatGetFactor() to create F */
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
#if defined(PETSC_HAVE_STRUMPACK)
    /* Set the fill-reducing reordering.                              */
    ierr = MatSTRUMPACKSetReordering(F,MAT_STRUMPACK_METIS);CHKERRQ(ierr);
    /* Since this is a simple discretization, the diagonal is always  */
    /* nonzero, and there is no need for the extra MC64 permutation.  */
    ierr = MatSTRUMPACKSetColPerm(F,PETSC_FALSE);CHKERRQ(ierr);
    /* The compression tolerance used when doing low-rank compression */
    /* in the preconditioner. This is problem specific!               */
    ierr = MatSTRUMPACKSetHSSRelTol(F,1.e-3);CHKERRQ(ierr);
    /* Set minimum matrix size for HSS compression to 15 in order to  */
    /* demonstrate preconditioner on small problems. For performance  */
    /* a value of say 500 is better.                                  */
    ierr = MatSTRUMPACKSetHSSMinSepSize(F,15);CHKERRQ(ierr);
    /* You can further limit the fill in the preconditioner by        */
    /* setting a maximum rank                                         */
    ierr = MatSTRUMPACKSetHSSMaxRank(F,100);CHKERRQ(ierr);
    /* Set the size of the diagonal blocks (the leafs) in the HSS     */
    /* approximation. The default value should be better for real     */
    /* problems. This is mostly for illustration on a small problem.  */
    ierr = MatSTRUMPACKSetHSSLeafSize(F,4);CHKERRQ(ierr);
#endif
  }
#endif

  /*
    Example of how to use procedural calls that are equivalent to
          '-ksp_type preonly -pc_type lu/ilu -pc_factor_mat_solver_type petsc'
  */
  flg     = PETSC_FALSE;
  flg_ilu = PETSC_FALSE;
  flg_ch  = PETSC_FALSE;
  ierr    = PetscOptionsGetBool(NULL,NULL,"-use_petsc_lu",&flg,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetBool(NULL,NULL,"-use_petsc_ilu",&flg_ilu,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetBool(NULL,NULL,"-use_petsc_ch",&flg_ch,NULL);CHKERRQ(ierr);
  if (flg || flg_ilu || flg_ch) {
    Vec diag;

    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    if (flg) {
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    } else if (flg_ilu) {
      ierr = PCSetType(pc,PCILU);CHKERRQ(ierr);
    } else if (flg_ch) {
      ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
    }
    ierr = PCFactorSetMatSolverType(pc,MATSOLVERPETSC);CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverType(pc);CHKERRQ(ierr); /* call MatGetFactor() to create F */
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);

    /* Test MatGetDiagonal() */
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&diag);CHKERRQ(ierr);
    ierr = MatGetDiagonal(F,diag);CHKERRQ(ierr);
    /* ierr = VecView(diag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    ierr = VecDestroy(&diag);CHKERRQ(ierr);
  }

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Get info from matrix factors */
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

#if defined(PETSC_HAVE_MUMPS)
  if (flg_mumps || flg_mumps_ch) {
    PetscInt  icntl,infog34;
    PetscReal cntl,rinfo12,rinfo13;
    icntl = 3;
    ierr = MatMumpsGetCntl(F,icntl,&cntl);CHKERRQ(ierr);

    /* compute determinant */
    if (!rank) {
      ierr = MatMumpsGetInfog(F,34,&infog34);CHKERRQ(ierr);
      ierr = MatMumpsGetRinfog(F,12,&rinfo12);CHKERRQ(ierr);
      ierr = MatMumpsGetRinfog(F,13,&rinfo13);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Mumps row pivot threshold = %g\n",cntl);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Mumps determinant = (%g, %g) * 2^%D \n",(double)rinfo12,(double)rinfo13,infog34);CHKERRQ(ierr);
    }
  }
#endif

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  /*
     Print convergence information.  PetscPrintf() produces a single
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  if (norm < 1.e-12) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error < 1.e-12 iterations %D\n",its);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its);CHKERRQ(ierr);
 }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -use_petsc_lu
      output_file: output/ex52_2.out

   test:
      suffix: mumps
      nsize: 3
      requires: mumps
      args: -use_mumps_lu
      output_file: output/ex52_1.out

   test:
      suffix: mumps_2
      nsize: 3
      requires: mumps
      args: -use_mumps_ch
      output_file: output/ex52_1.out

   test:
      suffix: mumps_3
      nsize: 3
      requires: mumps
      args: -use_mumps_ch -mat_type sbaij
      output_file: output/ex52_1.out

   test:
      suffix: mumps_omp_2
      nsize: 4
      requires: mumps hwloc openmp pthread defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
      args: -use_mumps_lu -mat_mumps_use_omp_threads 2
      output_file: output/ex52_1.out

   test:
      suffix: mumps_omp_3
      nsize: 4
      requires: mumps hwloc openmp pthread defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
      args: -use_mumps_ch -mat_mumps_use_omp_threads 3
      # Ignore the warning since we are intentionally testing the imbalanced case
      filter: grep -v "Warning: number of OpenMP threads"
      output_file: output/ex52_1.out

   test:
      suffix: mumps_omp_4
      nsize: 4
      requires: mumps hwloc openmp pthread defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
      # let petsc guess a proper number for threads
      args: -use_mumps_ch -mat_type sbaij -mat_mumps_use_omp_threads
      output_file: output/ex52_1.out

   test:
      suffix: strumpack
      requires: strumpack
      args: -use_strumpack_lu
      output_file: output/ex52_3.out

   test:
      suffix: strumpack_2
      nsize: 2
      requires: strumpack
      args: -use_strumpack_lu
      output_file: output/ex52_3.out

   test:
      suffix: strumpack_ilu
      requires: strumpack
      args: -use_strumpack_ilu
      output_file: output/ex52_3.out

   test:
      suffix: strumpack_ilu_2
      nsize: 2
      requires: strumpack
      args: -use_strumpack_ilu
      output_file: output/ex52_3.out

   test:
      suffix: superlu
      requires: superlu superlu_dist
      args: -use_superlu_lu
      output_file: output/ex52_2.out

   test:
      suffix: superlu_dist
      nsize: 2
      requires: superlu superlu_dist
      args: -use_superlu_lu
      output_file: output/ex52_2.out

   test:
      suffix: superlu_ilu
      requires: superlu superlu_dist
      args: -use_superlu_ilu
      output_file: output/ex52_2.out

TEST*/
