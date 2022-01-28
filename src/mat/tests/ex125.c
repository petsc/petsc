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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);

  /* Determine file from which we read the matrix A */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (flg) { /* Load matrix A */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatLoad(A,fd);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  } else {
    n = 13;
    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatShift(A,1.0);CHKERRQ(ierr);
  }
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  PetscAssertFalse(m != n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")", m, n);

  /* if A is symmetric, set its flag -- required by MatGetInertia() */
  ierr = MatIsSymmetric(A,0.0,&flg);CHKERRQ(ierr);

  ierr = MatViewFromOptions(A,NULL,"-A_view");CHKERRQ(ierr);

  /* Create dense matrix C and X; C holds true solution with identical columns */
  nrhs = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"ex125: nrhs %" PetscInt_FMT "\n",nrhs);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C,"rhs_");CHKERRQ(ierr);
  ierr = MatSetSizes(C,m,PETSC_DECIDE,PETSC_DECIDE,nrhs);CHKERRQ(ierr);
  ierr = MatSetType(C,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-view_factor",&view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_matmatsolve",&testMatMatSolve,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-cholesky",&chol,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_mumps_opts",&test_mumps_opts,NULL);CHKERRQ(ierr);
#endif

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatSetRandom(C,rand);CHKERRQ(ierr);
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&X);CHKERRQ(ierr);

  /* Create vectors */
  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr); /* save the true solution */

  /* Test Factorization */
  ierr = MatGetOrdering(A,MATORDERINGND,&perm,&iperm);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-mat_solver_type",&ipack,NULL);CHKERRQ(ierr);
  switch (ipack) {
#if defined(PETSC_HAVE_SUPERLU)
  case 0:
    PetscAssertFalse(chol,PETSC_COMM_WORLD,PETSC_ERR_SUP,"SuperLU does not provide Cholesky!");
    ierr = PetscPrintf(PETSC_COMM_WORLD," SUPERLU LU:\n");CHKERRQ(ierr);
    ierr = MatGetFactor(A,MATSOLVERSUPERLU,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    matsolvexx = PETSC_TRUE;
    break;
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
  case 1:
    PetscAssertFalse(chol,PETSC_COMM_WORLD,PETSC_ERR_SUP,"SuperLU does not provide Cholesky!");
    ierr = PetscPrintf(PETSC_COMM_WORLD," SUPERLU_DIST LU:\n");CHKERRQ(ierr);
    ierr = MatGetFactor(A,MATSOLVERSUPERLU_DIST,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    matsolvexx = PETSC_TRUE;
    break;
#endif
#if defined(PETSC_HAVE_MUMPS)
  case 2:
    if (chol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD," MUMPS CHOLESKY:\n");CHKERRQ(ierr);
      ierr = MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD," MUMPS LU:\n");CHKERRQ(ierr);
      ierr = MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    }
    matsolvexx = PETSC_TRUE;
    if (test_mumps_opts) {
      /* test mumps options */
      PetscInt  icntl;
      PetscReal cntl;

      icntl = 2;        /* sequential matrix ordering */
      ierr  = MatMumpsSetIcntl(F,7,icntl);CHKERRQ(ierr);

      cntl = 1.e-6; /* threshold for row pivot detection */
      ierr = MatMumpsSetIcntl(F,24,1);CHKERRQ(ierr);
      ierr = MatMumpsSetCntl(F,3,cntl);CHKERRQ(ierr);
    }
    break;
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
  case 3:
    if (chol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD," MKL_PARDISO CHOLESKY:\n");CHKERRQ(ierr);
      ierr = MatGetFactor(A,MATSOLVERMKL_PARDISO,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD," MKL_PARDISO LU:\n");CHKERRQ(ierr);
      ierr = MatGetFactor(A,MATSOLVERMKL_PARDISO,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    }
    break;
#endif
#if defined(PETSC_HAVE_CUDA)
  case 4:
    if (chol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD," CUSPARSE CHOLESKY:\n");CHKERRQ(ierr);
      ierr = MatGetFactor(A,MATSOLVERCUSPARSE,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD," CUSPARSE LU:\n");CHKERRQ(ierr);
      ierr = MatGetFactor(A,MATSOLVERCUSPARSE,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    }
    break;
#endif
  default:
    if (chol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD," PETSC CHOLESKY:\n");CHKERRQ(ierr);
      ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD," PETSC LU:\n");CHKERRQ(ierr);
      ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    }
    matsolvexx = PETSC_TRUE;
  }

  ierr           = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
  info.fill      = 5.0;
  info.shifttype = (PetscReal) MAT_SHIFT_NONE;
  if (chol) {
    ierr = MatCholeskyFactorSymbolic(F,A,perm,&info);CHKERRQ(ierr);
  } else {
    ierr = MatLUFactorSymbolic(F,A,perm,iperm,&info);CHKERRQ(ierr);
  }

  for (nfact = 0; nfact < 2; nfact++) {
    if (chol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD," %" PetscInt_FMT "-the CHOLESKY numfactorization \n",nfact);CHKERRQ(ierr);
      ierr = MatCholeskyFactorNumeric(F,A,&info);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD," %" PetscInt_FMT "-the LU numfactorization \n",nfact);CHKERRQ(ierr);
      ierr = MatLUFactorNumeric(F,A,&info);CHKERRQ(ierr);
    }
    if (view) {
      ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = MatView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
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

      ierr = MatGetSize(F,&M,NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1(M,&diag);CHKERRQ(ierr);
      ierr = MatSuperluDistGetDiagU(F,diag);CHKERRQ(ierr);
      ierr = PetscFree(diag);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
      /* Test MatGetInertia() */
      ierr = MatGetInertia(F,&nneg,&nzero,&npos);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD," MatInertia: nneg: %" PetscInt_FMT ", nzero: %" PetscInt_FMT ", npos: %" PetscInt_FMT "\n",nneg,nzero,npos);CHKERRQ(ierr);
#endif
    }
#endif

#if defined(PETSC_HAVE_MUMPS)
    /* mumps interface allows repeated call of MatCholeskyFactorSymbolic(), while the succession calls do nothing */
    if (ipack == 2) {
      if (chol) {
        ierr = MatCholeskyFactorSymbolic(F,A,perm,&info);CHKERRQ(ierr);
        ierr = MatCholeskyFactorNumeric(F,A,&info);CHKERRQ(ierr);
      } else {
        ierr = MatLUFactorSymbolic(F,A,perm,iperm,&info);CHKERRQ(ierr);
        ierr = MatLUFactorNumeric(F,A,&info);CHKERRQ(ierr);
      }
    }
#endif

    /* Test MatMatSolve() */
    if (testMatMatSolve) {
      if (!nfact) {
        ierr = MatMatMult(A,C,MAT_INITIAL_MATRIX,2.0,&RHS);CHKERRQ(ierr);
      } else {
        ierr = MatMatMult(A,C,MAT_REUSE_MATRIX,2.0,&RHS);CHKERRQ(ierr);
      }
      for (nsolve = 0; nsolve < 2; nsolve++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"   %" PetscInt_FMT "-the MatMatSolve \n",nsolve);CHKERRQ(ierr);
        ierr = MatMatSolve(F,RHS,X);CHKERRQ(ierr);

        /* Check the error */
        ierr = MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatNorm(X,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
        if (norm > tol) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT "-the MatMatSolve: Norm of error %g, nsolve %" PetscInt_FMT "\n",nsolve,(double)norm,nsolve);CHKERRQ(ierr);
        }
      }
      if (matsolvexx) {
        /* Test MatMatSolve(F,RHS,RHS), RHS is a dense matrix */
        ierr = MatCopy(RHS,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatMatSolve(F,X,X);CHKERRQ(ierr);
        /* Check the error */
        ierr = MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatNorm(X,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
        if (norm > tol) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMatSolve(F,RHS,RHS): Norm of error %g\n",(double)norm);CHKERRQ(ierr);
        }
      }

      if (ipack == 2 && size == 1) {
        Mat spRHS,spRHST,RHST;

        ierr = MatTranspose(RHS,MAT_INITIAL_MATRIX,&RHST);CHKERRQ(ierr);
        ierr = MatConvert(RHST,MATAIJ,MAT_INITIAL_MATRIX,&spRHST);CHKERRQ(ierr);
        ierr = MatCreateTranspose(spRHST,&spRHS);CHKERRQ(ierr);
        for (nsolve = 0; nsolve < 2; nsolve++) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,"   %" PetscInt_FMT "-the sparse MatMatSolve \n",nsolve);CHKERRQ(ierr);
          ierr = MatMatSolve(F,spRHS,X);CHKERRQ(ierr);

          /* Check the error */
          ierr = MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
          ierr = MatNorm(X,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
          if (norm > tol) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT "-the sparse MatMatSolve: Norm of error %g, nsolve %" PetscInt_FMT "\n",nsolve,(double)norm,nsolve);CHKERRQ(ierr);
          }
        }
        ierr = MatDestroy(&spRHST);CHKERRQ(ierr);
        ierr = MatDestroy(&spRHS);CHKERRQ(ierr);
        ierr = MatDestroy(&RHST);CHKERRQ(ierr);
      }
    }

    /* Test MatSolve() */
    if (testMatSolve) {
      for (nsolve = 0; nsolve < 2; nsolve++) {
        ierr = VecSetRandom(x,rand);CHKERRQ(ierr);
        ierr = VecCopy(x,u);CHKERRQ(ierr);
        ierr = MatMult(A,x,b);CHKERRQ(ierr);

        ierr = PetscPrintf(PETSC_COMM_WORLD,"   %" PetscInt_FMT "-the MatSolve \n",nsolve);CHKERRQ(ierr);
        ierr = MatSolve(F,b,x);CHKERRQ(ierr);

        /* Check the error */
        ierr = VecAXPY(u,-1.0,x);CHKERRQ(ierr);  /* u <- (-1.0)x + u */
        ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
        if (norm > tol) {
          PetscReal resi;
          ierr = MatMult(A,x,u);CHKERRQ(ierr); /* u = A*x */
          ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);  /* u <- (-1.0)b + u */
          ierr = VecNorm(u,NORM_2,&resi);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_WORLD,"MatSolve: Norm of error %g, resi %g, numfact %" PetscInt_FMT "\n",(double)norm,(double)resi,nfact);CHKERRQ(ierr);
        }
      }
    }
  }

  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  if (testMatMatSolve) {
    ierr = MatDestroy(&RHS);CHKERRQ(ierr);
  }

  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
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
