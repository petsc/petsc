static char help[] = "Tests MatSolve() and MatMatSolve() (interface to superlu_dist, mumps and mkl_pardiso).\n\
Example: mpiexec -n <np> ./ex125 -f <matrix binary file> -nrhs 4 \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,RHS,C,F,X;
  Vec            u,x,b;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       m,n,nfact,nsolve,nrhs,ipack=0;
  PetscReal      norm,tol=1.e-10;
  IS             perm,iperm;
  MatFactorInfo  info;
  PetscRandom    rand;
  PetscBool      flg,testMatSolve=PETSC_TRUE,testMatMatSolve=PETSC_TRUE;
  PetscBool      chol=PETSC_FALSE,view=PETSC_FALSE,matsolvexx = PETSC_FALSE;
#if defined(PETSC_HAVE_MUMPS)
  PetscBool      test_mumps_opts=PETSC_FALSE;
#endif
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Determine file from which we read the matrix A */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  if (flg) { /* Load matrix A */
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
    CHKERRQ(MatSetFromOptions(A));
    CHKERRQ(MatLoad(A,fd));
    CHKERRQ(PetscViewerDestroy(&fd));
  } else {
    n = 13;
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
    CHKERRQ(MatSetType(A,MATAIJ));
    CHKERRQ(MatSetFromOptions(A));
    CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
    CHKERRQ(MatSetUp(A));
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatShift(A,1.0));
  }
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  PetscCheckFalse(m != n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")", m, n);

  /* if A is symmetric, set its flag -- required by MatGetInertia() */
  CHKERRQ(MatIsSymmetric(A,0.0,&flg));

  CHKERRQ(MatViewFromOptions(A,NULL,"-A_view"));

  /* Create dense matrix C and X; C holds true solution with identical columns */
  nrhs = 2;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ex125: nrhs %" PetscInt_FMT "\n",nrhs));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetOptionsPrefix(C,"rhs_"));
  CHKERRQ(MatSetSizes(C,m,PETSC_DECIDE,PETSC_DECIDE,nrhs));
  CHKERRQ(MatSetType(C,MATDENSE));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-view_factor",&view,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_matmatsolve",&testMatMatSolve,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-cholesky",&chol,NULL));
#if defined(PETSC_HAVE_MUMPS)
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_mumps_opts",&test_mumps_opts,NULL));
#endif

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(MatSetRandom(C,rand));
  CHKERRQ(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&X));

  /* Create vectors */
  CHKERRQ(MatCreateVecs(A,&x,&b));
  CHKERRQ(VecDuplicate(x,&u)); /* save the true solution */

  /* Test Factorization */
  CHKERRQ(MatGetOrdering(A,MATORDERINGND,&perm,&iperm));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_solver_type",&ipack,NULL));
  switch (ipack) {
#if defined(PETSC_HAVE_SUPERLU)
  case 0:
    PetscCheck(!chol,PETSC_COMM_WORLD,PETSC_ERR_SUP,"SuperLU does not provide Cholesky!");
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," SUPERLU LU:\n"));
    CHKERRQ(MatGetFactor(A,MATSOLVERSUPERLU,MAT_FACTOR_LU,&F));
    matsolvexx = PETSC_TRUE;
    break;
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
  case 1:
    PetscCheck(!chol,PETSC_COMM_WORLD,PETSC_ERR_SUP,"SuperLU does not provide Cholesky!");
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," SUPERLU_DIST LU:\n"));
    CHKERRQ(MatGetFactor(A,MATSOLVERSUPERLU_DIST,MAT_FACTOR_LU,&F));
    matsolvexx = PETSC_TRUE;
    break;
#endif
#if defined(PETSC_HAVE_MUMPS)
  case 2:
    if (chol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," MUMPS CHOLESKY:\n"));
      CHKERRQ(MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&F));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," MUMPS LU:\n"));
      CHKERRQ(MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_LU,&F));
    }
    matsolvexx = PETSC_TRUE;
    if (test_mumps_opts) {
      /* test mumps options */
      PetscInt  icntl;
      PetscReal cntl;

      icntl = 2;        /* sequential matrix ordering */
      CHKERRQ(MatMumpsSetIcntl(F,7,icntl));

      cntl = 1.e-6; /* threshold for row pivot detection */
      CHKERRQ(MatMumpsSetIcntl(F,24,1));
      CHKERRQ(MatMumpsSetCntl(F,3,cntl));
    }
    break;
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
  case 3:
    if (chol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," MKL_PARDISO CHOLESKY:\n"));
      CHKERRQ(MatGetFactor(A,MATSOLVERMKL_PARDISO,MAT_FACTOR_CHOLESKY,&F));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," MKL_PARDISO LU:\n"));
      CHKERRQ(MatGetFactor(A,MATSOLVERMKL_PARDISO,MAT_FACTOR_LU,&F));
    }
    break;
#endif
#if defined(PETSC_HAVE_CUDA)
  case 4:
    if (chol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," CUSPARSE CHOLESKY:\n"));
      CHKERRQ(MatGetFactor(A,MATSOLVERCUSPARSE,MAT_FACTOR_CHOLESKY,&F));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," CUSPARSE LU:\n"));
      CHKERRQ(MatGetFactor(A,MATSOLVERCUSPARSE,MAT_FACTOR_LU,&F));
    }
    break;
#endif
  default:
    if (chol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," PETSC CHOLESKY:\n"));
      CHKERRQ(MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&F));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," PETSC LU:\n"));
      CHKERRQ(MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&F));
    }
    matsolvexx = PETSC_TRUE;
  }

  CHKERRQ(MatFactorInfoInitialize(&info));
  info.fill      = 5.0;
  info.shifttype = (PetscReal) MAT_SHIFT_NONE;
  if (chol) {
    CHKERRQ(MatCholeskyFactorSymbolic(F,A,perm,&info));
  } else {
    CHKERRQ(MatLUFactorSymbolic(F,A,perm,iperm,&info));
  }

  for (nfact = 0; nfact < 2; nfact++) {
    if (chol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %" PetscInt_FMT "-the CHOLESKY numfactorization \n",nfact));
      CHKERRQ(MatCholeskyFactorNumeric(F,A,&info));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %" PetscInt_FMT "-the LU numfactorization \n",nfact));
      CHKERRQ(MatLUFactorNumeric(F,A,&info));
    }
    if (view) {
      CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
      CHKERRQ(MatView(F,PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
      view = PETSC_FALSE;
    }

#if defined(PETSC_HAVE_SUPERLU_DIST)
    if (ipack == 1) { /* Test MatSuperluDistGetDiagU()
       -- input: matrix factor F; output: main diagonal of matrix U on all processes */
      PetscInt    M;
      PetscScalar *diag;
#if !defined(PETSC_USE_COMPLEX)
      PetscInt nneg,nzero,npos;
#endif

      CHKERRQ(MatGetSize(F,&M,NULL));
      CHKERRQ(PetscMalloc1(M,&diag));
      CHKERRQ(MatSuperluDistGetDiagU(F,diag));
      CHKERRQ(PetscFree(diag));

#if !defined(PETSC_USE_COMPLEX)
      /* Test MatGetInertia() */
      CHKERRQ(MatGetInertia(F,&nneg,&nzero,&npos));
      CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD," MatInertia: nneg: %" PetscInt_FMT ", nzero: %" PetscInt_FMT ", npos: %" PetscInt_FMT "\n",nneg,nzero,npos));
#endif
    }
#endif

#if defined(PETSC_HAVE_MUMPS)
    /* mumps interface allows repeated call of MatCholeskyFactorSymbolic(), while the succession calls do nothing */
    if (ipack == 2) {
      if (chol) {
        CHKERRQ(MatCholeskyFactorSymbolic(F,A,perm,&info));
        CHKERRQ(MatCholeskyFactorNumeric(F,A,&info));
      } else {
        CHKERRQ(MatLUFactorSymbolic(F,A,perm,iperm,&info));
        CHKERRQ(MatLUFactorNumeric(F,A,&info));
      }
    }
#endif

    /* Test MatMatSolve() */
    if (testMatMatSolve) {
      if (!nfact) {
        CHKERRQ(MatMatMult(A,C,MAT_INITIAL_MATRIX,2.0,&RHS));
      } else {
        CHKERRQ(MatMatMult(A,C,MAT_REUSE_MATRIX,2.0,&RHS));
      }
      for (nsolve = 0; nsolve < 2; nsolve++) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"   %" PetscInt_FMT "-the MatMatSolve \n",nsolve));
        CHKERRQ(MatMatSolve(F,RHS,X));

        /* Check the error */
        CHKERRQ(MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN));
        CHKERRQ(MatNorm(X,NORM_FROBENIUS,&norm));
        if (norm > tol) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT "-the MatMatSolve: Norm of error %g, nsolve %" PetscInt_FMT "\n",nsolve,(double)norm,nsolve));
        }
      }
      if (matsolvexx) {
        /* Test MatMatSolve(F,RHS,RHS), RHS is a dense matrix */
        CHKERRQ(MatCopy(RHS,X,SAME_NONZERO_PATTERN));
        CHKERRQ(MatMatSolve(F,X,X));
        /* Check the error */
        CHKERRQ(MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN));
        CHKERRQ(MatNorm(X,NORM_FROBENIUS,&norm));
        if (norm > tol) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatMatSolve(F,RHS,RHS): Norm of error %g\n",(double)norm));
        }
      }

      if (ipack == 2 && size == 1) {
        Mat spRHS,spRHST,RHST;

        CHKERRQ(MatTranspose(RHS,MAT_INITIAL_MATRIX,&RHST));
        CHKERRQ(MatConvert(RHST,MATAIJ,MAT_INITIAL_MATRIX,&spRHST));
        CHKERRQ(MatCreateTranspose(spRHST,&spRHS));
        for (nsolve = 0; nsolve < 2; nsolve++) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"   %" PetscInt_FMT "-the sparse MatMatSolve \n",nsolve));
          CHKERRQ(MatMatSolve(F,spRHS,X));

          /* Check the error */
          CHKERRQ(MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN));
          CHKERRQ(MatNorm(X,NORM_FROBENIUS,&norm));
          if (norm > tol) {
            CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT "-the sparse MatMatSolve: Norm of error %g, nsolve %" PetscInt_FMT "\n",nsolve,(double)norm,nsolve));
          }
        }
        CHKERRQ(MatDestroy(&spRHST));
        CHKERRQ(MatDestroy(&spRHS));
        CHKERRQ(MatDestroy(&RHST));
      }
    }

    /* Test MatSolve() */
    if (testMatSolve) {
      for (nsolve = 0; nsolve < 2; nsolve++) {
        CHKERRQ(VecSetRandom(x,rand));
        CHKERRQ(VecCopy(x,u));
        CHKERRQ(MatMult(A,x,b));

        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"   %" PetscInt_FMT "-the MatSolve \n",nsolve));
        CHKERRQ(MatSolve(F,b,x));

        /* Check the error */
        CHKERRQ(VecAXPY(u,-1.0,x));  /* u <- (-1.0)x + u */
        CHKERRQ(VecNorm(u,NORM_2,&norm));
        if (norm > tol) {
          PetscReal resi;
          CHKERRQ(MatMult(A,x,u)); /* u = A*x */
          CHKERRQ(VecAXPY(u,-1.0,b));  /* u <- (-1.0)b + u */
          CHKERRQ(VecNorm(u,NORM_2,&resi));
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"MatSolve: Norm of error %g, resi %g, numfact %" PetscInt_FMT "\n",(double)norm,(double)resi,nfact));
        }
      }
    }
  }

  /* Free data structures */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(MatDestroy(&X));
  if (testMatMatSolve) {
    CHKERRQ(MatDestroy(&RHS));
  }

  CHKERRQ(PetscRandomDestroy(&rand));
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(ISDestroy(&iperm));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_solver_type 10
      output_file: output/ex125.out

   test:
      suffix: 2
      args: -mat_solver_type 10
      output_file: output/ex125.out

   test:
      suffix: mkl_pardiso
      requires: mkl_pardiso datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_solver_type 3

   test:
      suffix: mkl_pardiso_2
      requires: mkl_pardiso
      args: -mat_solver_type 3
      output_file: output/ex125_mkl_pardiso.out

   test:
      suffix: mumps
      requires: mumps datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_solver_type 2
      output_file: output/ex125_mumps_seq.out

   test:
      suffix: mumps_2
      nsize: 3
      requires: mumps datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_solver_type 2
      output_file: output/ex125_mumps_par.out

   test:
      suffix: mumps_3
      requires: mumps
      args: -mat_solver_type 2
      output_file: output/ex125_mumps_seq.out

   test:
      suffix: mumps_4
      nsize: 3
      requires: mumps
      args: -mat_solver_type 2
      output_file: output/ex125_mumps_par.out

   test:
      suffix: mumps_5
      nsize: 3
      requires: mumps
      args: -mat_solver_type 2 -cholesky
      output_file: output/ex125_mumps_par_cholesky.out

   test:
      suffix: superlu_dist
      nsize: {{1 3}}
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES) superlu_dist
      args: -f ${DATAFILESPATH}/matrices/small -mat_solver_type 1 -mat_superlu_dist_rowperm NOROWPERM

   test:
      suffix: superlu_dist_2
      nsize: {{1 3}}
      requires: superlu_dist !complex
      args: -n 36 -mat_solver_type 1 -mat_superlu_dist_rowperm NOROWPERM
      output_file: output/ex125_superlu_dist.out

   test:
      suffix: superlu_dist_complex
      nsize: 3
      requires: datafilespath superlu_dist complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/farzad_B_rhs -mat_solver_type 1
      output_file: output/ex125_superlu_dist_complex.out

   test:
      suffix: superlu_dist_complex_2
      nsize: 3
      requires: superlu_dist complex
      args: -mat_solver_type 1
      output_file: output/ex125_superlu_dist_complex.out

   test:
      suffix: cusparse
      requires: cuda datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -mat_type aijcusparse -f ${DATAFILESPATH}/matrices/small -mat_solver_type 4 -cholesky {{0 1}separate output}

   test:
      suffix: cusparse_2
      requires: cuda
      args: -mat_type aijcusparse -mat_solver_type 4 -cholesky {{0 1}separate output}

TEST*/
