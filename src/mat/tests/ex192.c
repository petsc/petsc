static char help[] = "Tests MatSolve() and MatMatSolve() with MUMPS or MKL_PARDISO sequential solvers in Schur complement mode.\n\
Example: mpiexec -n 1 ./ex192 -f <matrix binary file> -nrhs 4 -symmetric_solve -hermitian_solve -schur_ratio 0.3\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,RHS,C,F,X,S;
  Vec            u,x,b;
  Vec            xschur,bschur,uschur;
  IS             is_schur;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       isolver=0,size_schur,m,n,nfact,nsolve,nrhs;
  PetscReal      norm,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscRandom    rand;
  PetscBool      data_provided,herm,symm,use_lu,cuda = PETSC_FALSE;
  PetscReal      sratio = 5.1/12.;
  PetscViewer    fd;              /* viewer */
  char           solver[256];
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor test");
  /* Determine which type of solver we want to test for */
  herm = PETSC_FALSE;
  symm = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-symmetric_solve",&symm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-hermitian_solve",&herm,NULL);CHKERRQ(ierr);
  if (herm) symm = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-cuda_solve",&cuda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-tol",&tol,NULL);CHKERRQ(ierr);

  /* Determine file from which we read the matrix A */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&data_provided);CHKERRQ(ierr);
  if (!data_provided) { /* get matrices from PETSc distribution */
    ierr = PetscStrncpy(file,"${PETSC_DIR}/share/petsc/datafiles/matrices/",sizeof(file));CHKERRQ(ierr);
    if (symm) {
#if defined (PETSC_USE_COMPLEX)
      ierr = PetscStrlcat(file,"hpd-complex-",sizeof(file));CHKERRQ(ierr);
#else
      ierr = PetscStrlcat(file,"spd-real-",sizeof(file));CHKERRQ(ierr);
#endif
    } else {
#if defined (PETSC_USE_COMPLEX)
      ierr = PetscStrlcat(file,"nh-complex-",sizeof(file));CHKERRQ(ierr);
#else
      ierr = PetscStrlcat(file,"ns-real-",sizeof(file));CHKERRQ(ierr);
#endif
    }
#if defined(PETSC_USE_64BIT_INDICES)
    ierr = PetscStrlcat(file,"int64-",sizeof(file));CHKERRQ(ierr);
#else
    ierr = PetscStrlcat(file,"int32-",sizeof(file));CHKERRQ(ierr);
#endif
#if defined (PETSC_USE_REAL_SINGLE)
    ierr = PetscStrlcat(file,"float32",sizeof(file));CHKERRQ(ierr);
#else
    ierr = PetscStrlcat(file,"float64",sizeof(file));CHKERRQ(ierr);
#endif
  }
  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);

  /* Create dense matrix C and X; C holds true solution with identical colums */
  nrhs = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m,PETSC_DECIDE,PETSC_DECIDE,nrhs);CHKERRQ(ierr);
  ierr = MatSetType(C,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatSetRandom(C,rand);CHKERRQ(ierr);
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&X);CHKERRQ(ierr);

  /* Create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr); /* save the true solution */

  ierr = PetscOptionsGetInt(NULL,NULL,"-solver",&isolver,NULL);CHKERRQ(ierr);
  switch (isolver) {
#if defined(PETSC_HAVE_MUMPS)
    case 0:
      ierr = PetscStrcpy(solver,MATSOLVERMUMPS);CHKERRQ(ierr);
      break;
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
    case 1:
      ierr = PetscStrcpy(solver,MATSOLVERMKL_PARDISO);CHKERRQ(ierr);
      break;
#endif
    default:
      ierr = PetscStrcpy(solver,MATSOLVERPETSC);CHKERRQ(ierr);
      break;
  }

#if defined (PETSC_USE_COMPLEX)
  if (isolver == 0 && symm && !data_provided) { /* MUMPS (5.0.0) does not have support for hermitian matrices, so make them symmetric */
    PetscScalar im = PetscSqrtScalar((PetscScalar)-1.);
    PetscScalar val = -1.0;
    val = val + im;
    ierr = MatSetValue(A,1,0,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
#endif

  ierr = PetscOptionsGetReal(NULL,NULL,"-schur_ratio",&sratio,NULL);CHKERRQ(ierr);
  if (sratio < 0. || sratio > 1.) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid ratio for schur degrees of freedom %f", sratio);
  }
  size_schur = (PetscInt)(sratio*m);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Solving with %s: nrhs %D, sym %d, herm %d, size schur %D, size mat %D\n",solver,nrhs,symm,herm,size_schur,m);CHKERRQ(ierr);

  /* Test LU/Cholesky Factorization */
  use_lu = PETSC_FALSE;
  if (!symm) use_lu = PETSC_TRUE;
#if defined (PETSC_USE_COMPLEX)
  if (isolver == 1) use_lu = PETSC_TRUE;
#endif
  if (cuda && symm && !herm) use_lu = PETSC_TRUE;

  if (herm && !use_lu) { /* test also conversion routines inside the solver packages */
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatConvert(A,MATSEQSBAIJ,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  }

  if (use_lu) {
    ierr = MatGetFactor(A,solver,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  } else {
    if (herm) {
      ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_SPD,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = MatGetFactor(A,solver,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
  }
  ierr = ISCreateStride(PETSC_COMM_SELF,size_schur,m-size_schur,1,&is_schur);CHKERRQ(ierr);
  ierr = MatFactorSetSchurIS(F,is_schur);CHKERRQ(ierr);

  ierr = ISDestroy(&is_schur);CHKERRQ(ierr);
  if (use_lu) {
    ierr = MatLUFactorSymbolic(F,A,NULL,NULL,NULL);CHKERRQ(ierr);
  } else {
    ierr = MatCholeskyFactorSymbolic(F,A,NULL,NULL);CHKERRQ(ierr);
  }

  for (nfact = 0; nfact < 3; nfact++) {
    Mat AD;

    if (!nfact) {
      ierr = VecSetRandom(x,rand);CHKERRQ(ierr);
      if (symm && herm) {
        ierr = VecAbs(x);CHKERRQ(ierr);
      }
      ierr = MatDiagonalSet(A,x,ADD_VALUES);CHKERRQ(ierr);
    }
    if (use_lu) {
      ierr = MatLUFactorNumeric(F,A,NULL);CHKERRQ(ierr);
    } else {
      ierr = MatCholeskyFactorNumeric(F,A,NULL);CHKERRQ(ierr);
    }
    if (cuda) {
      ierr = MatFactorGetSchurComplement(F,&S,NULL);CHKERRQ(ierr);
      ierr = MatSetType(S,MATSEQDENSECUDA);CHKERRQ(ierr);
      ierr = MatCreateVecs(S,&xschur,&bschur);CHKERRQ(ierr);
      ierr = MatFactorRestoreSchurComplement(F,&S,MAT_FACTOR_SCHUR_UNFACTORED);CHKERRQ(ierr);
    }
    ierr = MatFactorCreateSchurComplement(F,&S,NULL);CHKERRQ(ierr);
    if (!cuda) {
      ierr = MatCreateVecs(S,&xschur,&bschur);CHKERRQ(ierr);
    }
    ierr = VecDuplicate(xschur,&uschur);CHKERRQ(ierr);
    if (nfact == 1 && (!cuda || (herm && symm))) {
      ierr = MatFactorInvertSchurComplement(F);CHKERRQ(ierr);
    }
    for (nsolve = 0; nsolve < 2; nsolve++) {
      ierr = VecSetRandom(x,rand);CHKERRQ(ierr);
      ierr = VecCopy(x,u);CHKERRQ(ierr);

      if (nsolve) {
        ierr = MatMult(A,x,b);CHKERRQ(ierr);
        ierr = MatSolve(F,b,x);CHKERRQ(ierr);
      } else {
        ierr = MatMultTranspose(A,x,b);CHKERRQ(ierr);
        ierr = MatSolveTranspose(F,b,x);CHKERRQ(ierr);
      }
      /* Check the error */
      ierr = VecAXPY(u,-1.0,x);CHKERRQ(ierr);  /* u <- (-1.0)x + u */
      ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
      if (norm > tol) {
        PetscReal resi;
        if (nsolve) {
          ierr = MatMult(A,x,u);CHKERRQ(ierr); /* u = A*x */
        } else {
          ierr = MatMultTranspose(A,x,u);CHKERRQ(ierr); /* u = A*x */
        }
        ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);  /* u <- (-1.0)b + u */
        ierr = VecNorm(u,NORM_2,&resi);CHKERRQ(ierr);
        if (nsolve) {
          ierr = PetscPrintf(PETSC_COMM_SELF,"(f %D, s %D) MatSolve error: Norm of error %g, residual %f\n",nfact,nsolve,norm,resi);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF,"(f %D, s %D) MatSolveTranspose error: Norm of error %g, residual %f\n",nfact,nsolve,norm,resi);CHKERRQ(ierr);
        }
      }
      ierr = VecSetRandom(xschur,rand);CHKERRQ(ierr);
      ierr = VecCopy(xschur,uschur);CHKERRQ(ierr);
      if (nsolve) {
        ierr = MatMult(S,xschur,bschur);CHKERRQ(ierr);
        ierr = MatFactorSolveSchurComplement(F,bschur,xschur);CHKERRQ(ierr);
      } else {
        ierr = MatMultTranspose(S,xschur,bschur);CHKERRQ(ierr);
        ierr = MatFactorSolveSchurComplementTranspose(F,bschur,xschur);CHKERRQ(ierr);
      }
      /* Check the error */
      ierr = VecAXPY(uschur,-1.0,xschur);CHKERRQ(ierr);  /* u <- (-1.0)x + u */
      ierr = VecNorm(uschur,NORM_2,&norm);CHKERRQ(ierr);
      if (norm > tol) {
        PetscReal resi;
        if (nsolve) {
          ierr = MatMult(S,xschur,uschur);CHKERRQ(ierr); /* u = A*x */
        } else {
          ierr = MatMultTranspose(S,xschur,uschur);CHKERRQ(ierr); /* u = A*x */
        }
        ierr = VecAXPY(uschur,-1.0,bschur);CHKERRQ(ierr);  /* u <- (-1.0)b + u */
        ierr = VecNorm(uschur,NORM_2,&resi);CHKERRQ(ierr);
        if (nsolve) {
          ierr = PetscPrintf(PETSC_COMM_SELF,"(f %D, s %D) MatFactorSolveSchurComplement error: Norm of error %g, residual %f\n",nfact,nsolve,norm,resi);CHKERRQ(ierr);
        } else {
          ierr = PetscPrintf(PETSC_COMM_SELF,"(f %D, s %D) MatFactorSolveSchurComplementTranspose error: Norm of error %g, residual %f\n",nfact,nsolve,norm,resi);CHKERRQ(ierr);
        }
      }
    }
    ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&AD);CHKERRQ(ierr);
    if (!nfact) {
      ierr = MatMatMult(AD,C,MAT_INITIAL_MATRIX,2.0,&RHS);CHKERRQ(ierr);
    } else {
      ierr = MatMatMult(AD,C,MAT_REUSE_MATRIX,2.0,&RHS);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&AD);CHKERRQ(ierr);
    for (nsolve = 0; nsolve < 2; nsolve++) {
      ierr = MatMatSolve(F,RHS,X);CHKERRQ(ierr);

      /* Check the error */
      ierr = MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(X,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
      if (norm > tol) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"(f %D, s %D) MatMatSolve: Norm of error %g\n",nfact,nsolve,norm);CHKERRQ(ierr);
      }
    }
    if (isolver == 0) {
      Mat spRHS,spRHST,RHST;

      ierr = MatTranspose(RHS,MAT_INITIAL_MATRIX,&RHST);CHKERRQ(ierr);
      ierr = MatConvert(RHST,MATSEQAIJ,MAT_INITIAL_MATRIX,&spRHST);CHKERRQ(ierr);
      ierr = MatCreateTranspose(spRHST,&spRHS);CHKERRQ(ierr);
      for (nsolve = 0; nsolve < 2; nsolve++) {
        ierr = MatMatSolve(F,spRHS,X);CHKERRQ(ierr);

        /* Check the error */
        ierr = MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatNorm(X,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
        if (norm > tol) {
          ierr = PetscPrintf(PETSC_COMM_SELF,"(f %D, s %D) sparse MatMatSolve: Norm of error %g\n",nfact,nsolve,norm);CHKERRQ(ierr);
        }
      }
      ierr = MatDestroy(&spRHST);CHKERRQ(ierr);
      ierr = MatDestroy(&spRHS);CHKERRQ(ierr);
      ierr = MatDestroy(&RHST);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&S);CHKERRQ(ierr);
    ierr = VecDestroy(&xschur);CHKERRQ(ierr);
    ierr = VecDestroy(&bschur);CHKERRQ(ierr);
    ierr = VecDestroy(&uschur);CHKERRQ(ierr);
  }
  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&RHS);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
     requires: mkl_pardiso double !complex
     args: -solver 1

     test:
       suffix: mkl_pardiso
     test:
       requires: cuda
       suffix: mkl_pardiso_cuda
       args: -cuda_solve
       output_file: output/ex192_mkl_pardiso.out
     test:
       suffix: mkl_pardiso_1
       args: -symmetric_solve
       output_file: output/ex192_mkl_pardiso_1.out
     test:
       requires: cuda
       suffix: mkl_pardiso_cuda_1
       args: -symmetric_solve -cuda_solve
       output_file: output/ex192_mkl_pardiso_1.out
     test:
       suffix: mkl_pardiso_3
       args: -symmetric_solve -hermitian_solve
       output_file: output/ex192_mkl_pardiso_3.out
     test:
       requires: cuda defined(PETSC_HAVE_CUSOLVERDNDPOTRI)
       suffix: mkl_pardiso_cuda_3
       args: -symmetric_solve -hermitian_solve -cuda_solve
       output_file: output/ex192_mkl_pardiso_3.out

   testset:
     requires: mumps double !complex
     args: -solver 0

     test:
       suffix: mumps
     test:
       requires: cuda
       suffix: mumps_cuda
       args: -cuda_solve
       output_file: output/ex192_mumps.out
     test:
       suffix: mumps_2
       args: -symmetric_solve
       output_file: output/ex192_mumps_2.out
     test:
       requires: cuda
       suffix: mumps_cuda_2
       args: -symmetric_solve -cuda_solve
       output_file: output/ex192_mumps_2.out
     test:
       suffix: mumps_3
       args: -symmetric_solve -hermitian_solve
       output_file: output/ex192_mumps_3.out
     test:
       requires: cuda defined(PETSC_HAVE_CUSOLVERDNDPOTRI)
       suffix: mumps_cuda_3
       args: -symmetric_solve -hermitian_solve -cuda_solve
       output_file: output/ex192_mumps_3.out

TEST*/
