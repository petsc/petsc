static char help[] = "Tests MatSolve() and MatMatSolve() with MUMPS or MKL_PARDISO sequential solvers in Schur complement mode.\n\
Example: mpiexec -n 1 ./ex192 -f <matrix binary file> -nrhs 4 -symmetric_solve -hermitian_solve -schur_ratio 0.3\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,RHS,C,F,X,S;
  Vec            u,x,b;
  Vec            xschur,bschur,uschur;
  IS             is_schur;
  PetscMPIInt    size;
  PetscInt       isolver=0,size_schur,m,n,nfact,nsolve,nrhs;
  PetscReal      norm,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscRandom    rand;
  PetscBool      data_provided,herm,symm,use_lu,cuda = PETSC_FALSE;
  PetscReal      sratio = 5.1/12.;
  PetscViewer    fd;              /* viewer */
  char           solver[256];
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor test");
  /* Determine which type of solver we want to test for */
  herm = PETSC_FALSE;
  symm = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-symmetric_solve",&symm,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-hermitian_solve",&herm,NULL));
  if (herm) symm = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-cuda_solve",&cuda,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tol",&tol,NULL));

  /* Determine file from which we read the matrix A */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&data_provided));
  if (!data_provided) { /* get matrices from PETSc distribution */
    CHKERRQ(PetscStrncpy(file,"${PETSC_DIR}/share/petsc/datafiles/matrices/",sizeof(file)));
    if (symm) {
#if defined (PETSC_USE_COMPLEX)
      CHKERRQ(PetscStrlcat(file,"hpd-complex-",sizeof(file)));
#else
      CHKERRQ(PetscStrlcat(file,"spd-real-",sizeof(file)));
#endif
    } else {
#if defined (PETSC_USE_COMPLEX)
      CHKERRQ(PetscStrlcat(file,"nh-complex-",sizeof(file)));
#else
      CHKERRQ(PetscStrlcat(file,"ns-real-",sizeof(file)));
#endif
    }
#if defined(PETSC_USE_64BIT_INDICES)
    CHKERRQ(PetscStrlcat(file,"int64-",sizeof(file)));
#else
    CHKERRQ(PetscStrlcat(file,"int32-",sizeof(file)));
#endif
#if defined (PETSC_USE_REAL_SINGLE)
    CHKERRQ(PetscStrlcat(file,"float32",sizeof(file)));
#else
    CHKERRQ(PetscStrlcat(file,"float64",sizeof(file)));
#endif
  }
  /* Load matrix A */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  CHKERRQ(MatGetSize(A,&m,&n));
  PetscCheckFalse(m != n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")", m, n);

  /* Create dense matrix C and X; C holds true solution with identical columns */
  nrhs = 2;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,m,PETSC_DECIDE,PETSC_DECIDE,nrhs));
  CHKERRQ(MatSetType(C,MATDENSE));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(MatSetRandom(C,rand));
  CHKERRQ(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&X));

  /* Create vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(VecDuplicate(x,&u)); /* save the true solution */

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-solver",&isolver,NULL));
  switch (isolver) {
#if defined(PETSC_HAVE_MUMPS)
    case 0:
      CHKERRQ(PetscStrcpy(solver,MATSOLVERMUMPS));
      break;
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
    case 1:
      CHKERRQ(PetscStrcpy(solver,MATSOLVERMKL_PARDISO));
      break;
#endif
    default:
      CHKERRQ(PetscStrcpy(solver,MATSOLVERPETSC));
      break;
  }

#if defined (PETSC_USE_COMPLEX)
  if (isolver == 0 && symm && !data_provided) { /* MUMPS (5.0.0) does not have support for hermitian matrices, so make them symmetric */
    PetscScalar im = PetscSqrtScalar((PetscScalar)-1.);
    PetscScalar val = -1.0;
    val = val + im;
    CHKERRQ(MatSetValue(A,1,0,val,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }
#endif

  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-schur_ratio",&sratio,NULL));
  PetscCheckFalse(sratio < 0. || sratio > 1.,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid ratio for schur degrees of freedom %g", (double)sratio);
  size_schur = (PetscInt)(sratio*m);

  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Solving with %s: nrhs %" PetscInt_FMT ", sym %d, herm %d, size schur %" PetscInt_FMT ", size mat %" PetscInt_FMT "\n",solver,nrhs,symm,herm,size_schur,m));

  /* Test LU/Cholesky Factorization */
  use_lu = PETSC_FALSE;
  if (!symm) use_lu = PETSC_TRUE;
#if defined (PETSC_USE_COMPLEX)
  if (isolver == 1) use_lu = PETSC_TRUE;
#endif
  if (cuda && symm && !herm) use_lu = PETSC_TRUE;

  if (herm && !use_lu) { /* test also conversion routines inside the solver packages */
    CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
    CHKERRQ(MatConvert(A,MATSEQSBAIJ,MAT_INPLACE_MATRIX,&A));
  }

  if (use_lu) {
    CHKERRQ(MatGetFactor(A,solver,MAT_FACTOR_LU,&F));
  } else {
    if (herm) {
      CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
      CHKERRQ(MatSetOption(A,MAT_SPD,PETSC_TRUE));
    } else {
      CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
      CHKERRQ(MatSetOption(A,MAT_SPD,PETSC_FALSE));
    }
    CHKERRQ(MatGetFactor(A,solver,MAT_FACTOR_CHOLESKY,&F));
  }
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,size_schur,m-size_schur,1,&is_schur));
  CHKERRQ(MatFactorSetSchurIS(F,is_schur));

  CHKERRQ(ISDestroy(&is_schur));
  if (use_lu) {
    CHKERRQ(MatLUFactorSymbolic(F,A,NULL,NULL,NULL));
  } else {
    CHKERRQ(MatCholeskyFactorSymbolic(F,A,NULL,NULL));
  }

  for (nfact = 0; nfact < 3; nfact++) {
    Mat AD;

    if (!nfact) {
      CHKERRQ(VecSetRandom(x,rand));
      if (symm && herm) {
        CHKERRQ(VecAbs(x));
      }
      CHKERRQ(MatDiagonalSet(A,x,ADD_VALUES));
    }
    if (use_lu) {
      CHKERRQ(MatLUFactorNumeric(F,A,NULL));
    } else {
      CHKERRQ(MatCholeskyFactorNumeric(F,A,NULL));
    }
    if (cuda) {
      CHKERRQ(MatFactorGetSchurComplement(F,&S,NULL));
      CHKERRQ(MatSetType(S,MATSEQDENSECUDA));
      CHKERRQ(MatCreateVecs(S,&xschur,&bschur));
      CHKERRQ(MatFactorRestoreSchurComplement(F,&S,MAT_FACTOR_SCHUR_UNFACTORED));
    }
    CHKERRQ(MatFactorCreateSchurComplement(F,&S,NULL));
    if (!cuda) {
      CHKERRQ(MatCreateVecs(S,&xschur,&bschur));
    }
    CHKERRQ(VecDuplicate(xschur,&uschur));
    if (nfact == 1 && (!cuda || (herm && symm))) {
      CHKERRQ(MatFactorInvertSchurComplement(F));
    }
    for (nsolve = 0; nsolve < 2; nsolve++) {
      CHKERRQ(VecSetRandom(x,rand));
      CHKERRQ(VecCopy(x,u));

      if (nsolve) {
        CHKERRQ(MatMult(A,x,b));
        CHKERRQ(MatSolve(F,b,x));
      } else {
        CHKERRQ(MatMultTranspose(A,x,b));
        CHKERRQ(MatSolveTranspose(F,b,x));
      }
      /* Check the error */
      CHKERRQ(VecAXPY(u,-1.0,x));  /* u <- (-1.0)x + u */
      CHKERRQ(VecNorm(u,NORM_2,&norm));
      if (norm > tol) {
        PetscReal resi;
        if (nsolve) {
          CHKERRQ(MatMult(A,x,u)); /* u = A*x */
        } else {
          CHKERRQ(MatMultTranspose(A,x,u)); /* u = A*x */
        }
        CHKERRQ(VecAXPY(u,-1.0,b));  /* u <- (-1.0)b + u */
        CHKERRQ(VecNorm(u,NORM_2,&resi));
        if (nsolve) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"(f %" PetscInt_FMT ", s %" PetscInt_FMT ") MatSolve error: Norm of error %g, residual %g\n",nfact,nsolve,(double)norm,(double)resi));
        } else {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"(f %" PetscInt_FMT ", s %" PetscInt_FMT ") MatSolveTranspose error: Norm of error %g, residual %f\n",nfact,nsolve,(double)norm,(double)resi));
        }
      }
      CHKERRQ(VecSetRandom(xschur,rand));
      CHKERRQ(VecCopy(xschur,uschur));
      if (nsolve) {
        CHKERRQ(MatMult(S,xschur,bschur));
        CHKERRQ(MatFactorSolveSchurComplement(F,bschur,xschur));
      } else {
        CHKERRQ(MatMultTranspose(S,xschur,bschur));
        CHKERRQ(MatFactorSolveSchurComplementTranspose(F,bschur,xschur));
      }
      /* Check the error */
      CHKERRQ(VecAXPY(uschur,-1.0,xschur));  /* u <- (-1.0)x + u */
      CHKERRQ(VecNorm(uschur,NORM_2,&norm));
      if (norm > tol) {
        PetscReal resi;
        if (nsolve) {
          CHKERRQ(MatMult(S,xschur,uschur)); /* u = A*x */
        } else {
          CHKERRQ(MatMultTranspose(S,xschur,uschur)); /* u = A*x */
        }
        CHKERRQ(VecAXPY(uschur,-1.0,bschur));  /* u <- (-1.0)b + u */
        CHKERRQ(VecNorm(uschur,NORM_2,&resi));
        if (nsolve) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"(f %" PetscInt_FMT ", s %" PetscInt_FMT ") MatFactorSolveSchurComplement error: Norm of error %g, residual %g\n",nfact,nsolve,(double)norm,(double)resi));
        } else {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"(f %" PetscInt_FMT ", s %" PetscInt_FMT ") MatFactorSolveSchurComplementTranspose error: Norm of error %g, residual %f\n",nfact,nsolve,(double)norm,(double)resi));
        }
      }
    }
    CHKERRQ(MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&AD));
    if (!nfact) {
      CHKERRQ(MatMatMult(AD,C,MAT_INITIAL_MATRIX,2.0,&RHS));
    } else {
      CHKERRQ(MatMatMult(AD,C,MAT_REUSE_MATRIX,2.0,&RHS));
    }
    CHKERRQ(MatDestroy(&AD));
    for (nsolve = 0; nsolve < 2; nsolve++) {
      CHKERRQ(MatMatSolve(F,RHS,X));

      /* Check the error */
      CHKERRQ(MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN));
      CHKERRQ(MatNorm(X,NORM_FROBENIUS,&norm));
      if (norm > tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"(f %" PetscInt_FMT ", s %" PetscInt_FMT ") MatMatSolve: Norm of error %g\n",nfact,nsolve,(double)norm));
      }
    }
    if (isolver == 0) {
      Mat spRHS,spRHST,RHST;

      CHKERRQ(MatTranspose(RHS,MAT_INITIAL_MATRIX,&RHST));
      CHKERRQ(MatConvert(RHST,MATSEQAIJ,MAT_INITIAL_MATRIX,&spRHST));
      CHKERRQ(MatCreateTranspose(spRHST,&spRHS));
      for (nsolve = 0; nsolve < 2; nsolve++) {
        CHKERRQ(MatMatSolve(F,spRHS,X));

        /* Check the error */
        CHKERRQ(MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN));
        CHKERRQ(MatNorm(X,NORM_FROBENIUS,&norm));
        if (norm > tol) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"(f %" PetscInt_FMT ", s %" PetscInt_FMT ") sparse MatMatSolve: Norm of error %g\n",nfact,nsolve,(double)norm));
        }
      }
      CHKERRQ(MatDestroy(&spRHST));
      CHKERRQ(MatDestroy(&spRHS));
      CHKERRQ(MatDestroy(&RHST));
    }
    CHKERRQ(MatDestroy(&S));
    CHKERRQ(VecDestroy(&xschur));
    CHKERRQ(VecDestroy(&bschur));
    CHKERRQ(VecDestroy(&uschur));
  }
  /* Free data structures */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(MatDestroy(&RHS));
  CHKERRQ(PetscRandomDestroy(&rand));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(PetscFinalize());
  return 0;
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
