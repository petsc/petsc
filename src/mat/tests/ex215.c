static char help[] = "Tests MatSolve(), MatSolveTranspose() and MatMatSolve() with SEQDENSE\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,RHS,C,F,X;
  Vec            u,x,b;
  PetscMPIInt    size;
  PetscInt       m,n,nsolve,nrhs;
  PetscReal      norm,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscRandom    rand;
  PetscBool      data_provided,herm,symm,hpd;
  MatFactorType  ftyp;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor test");
  /* Determine which type of solver we want to test for */
  herm = PETSC_FALSE;
  symm = PETSC_FALSE;
  hpd  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-symmetric_solve",&symm,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-hermitian_solve",&herm,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-hpd_solve",&hpd,NULL));

  /* Determine file from which we read the matrix A */
  ftyp = MAT_FACTOR_LU;
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&data_provided));
  if (!data_provided) { /* get matrices from PETSc distribution */
    PetscCall(PetscStrcpy(file,"${PETSC_DIR}/share/petsc/datafiles/matrices/"));
    if (hpd) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscStrcat(file,"hpd-complex-"));
#else
      PetscCall(PetscStrcat(file,"spd-real-"));
#endif
      ftyp = MAT_FACTOR_CHOLESKY;
    } else {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscStrcat(file,"nh-complex-"));
#else
      PetscCall(PetscStrcat(file,"ns-real-"));
#endif
    }
#if defined(PETSC_USE_64BIT_INDICES)
    PetscCall(PetscStrcat(file,"int64-"));
#else
    PetscCall(PetscStrcat(file,"int32-"));
#endif
#if defined(PETSC_USE_REAL_SINGLE)
    PetscCall(PetscStrcat(file,"float32"));
#else
    PetscCall(PetscStrcat(file,"float64"));
#endif
  }

  /* Load matrix A */
#if defined(PETSC_USE_REAL___FLOAT128)
  PetscCall(PetscOptionsInsertString(NULL,"-binary_read_double"));
#endif
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatLoad(A,fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatConvert(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A));
  PetscCall(MatGetSize(A,&m,&n));
  PetscCheck(m == n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")", m, n);

  /* Create dense matrix C and X; C holds true solution with identical columns */
  nrhs = 2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,m,PETSC_DECIDE,PETSC_DECIDE,nrhs));
  PetscCall(MatSetType(C,MATDENSE));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatSetRandom(C,rand));
  PetscCall(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&X));
  PetscCall(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&RHS));

  /* Create vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,n,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&b));
  PetscCall(VecDuplicate(x,&u)); /* save the true solution */

  /* make a symmetric matrix */
  if (symm) {
    Mat AT;

    PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX,&AT));
    PetscCall(MatAXPY(A,1.0,AT,SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&AT));
    ftyp = MAT_FACTOR_CHOLESKY;
  }
  /* make an hermitian matrix */
  if (herm) {
    Mat AH;

    PetscCall(MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&AH));
    PetscCall(MatAXPY(A,1.0,AH,SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&AH));
    ftyp = MAT_FACTOR_CHOLESKY;
  }
  PetscCall(PetscObjectSetName((PetscObject)A,"A"));
  PetscCall(MatViewFromOptions(A,NULL,"-amat_view"));

  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&F));
  PetscCall(MatSetOption(F,MAT_SYMMETRIC,symm));
  /* it seems that the SPD concept in PETSc extends naturally to Hermitian Positive definitess */
  PetscCall(MatSetOption(F,MAT_HERMITIAN,(PetscBool)(hpd || herm)));
  PetscCall(MatSetOption(F,MAT_SPD,hpd));
  {
    PetscInt iftyp = ftyp;
    PetscCall(PetscOptionsGetEList(NULL,NULL,"-ftype",MatFactorTypes,MAT_FACTOR_NUM_TYPES,&iftyp,NULL));
    ftyp = (MatFactorType) iftyp;
  }
  if (ftyp == MAT_FACTOR_LU) {
    PetscCall(MatLUFactor(F,NULL,NULL,NULL));
  } else if (ftyp == MAT_FACTOR_CHOLESKY) {
    PetscCall(MatCholeskyFactor(F,NULL,NULL));
  } else if (ftyp == MAT_FACTOR_QR) {
    PetscCall(MatQRFactor(F,NULL,NULL));
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Factorization %s not supported in this example", MatFactorTypes[ftyp]);

  for (nsolve = 0; nsolve < 2; nsolve++) {
    PetscCall(VecSetRandom(x,rand));
    PetscCall(VecCopy(x,u));
    if (nsolve) {
      PetscCall(MatMult(A,x,b));
      PetscCall(MatSolve(F,b,x));
    } else {
      PetscCall(MatMultTranspose(A,x,b));
      PetscCall(MatSolveTranspose(F,b,x));
    }
    /* Check the error */
    PetscCall(VecAXPY(u,-1.0,x));  /* u <- (-1.0)x + u */
    PetscCall(VecNorm(u,NORM_2,&norm));
    if (norm > tol) {
      PetscReal resi;
      if (nsolve) {
        PetscCall(MatMult(A,x,u)); /* u = A*x */
      } else {
        PetscCall(MatMultTranspose(A,x,u)); /* u = A*x */
      }
      PetscCall(VecAXPY(u,-1.0,b));  /* u <- (-1.0)b + u */
      PetscCall(VecNorm(u,NORM_2,&resi));
      if (nsolve) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"MatSolve error: Norm of error %g, residual %g\n",(double)norm,(double)resi));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"MatSolveTranspose error: Norm of error %g, residual %g\n",(double)norm,(double)resi));
      }
    }
  }
  PetscCall(MatMatMult(A,C,MAT_REUSE_MATRIX,2.0,&RHS));
  PetscCall(MatMatSolve(F,RHS,X));

  /* Check the error */
  PetscCall(MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(X,NORM_FROBENIUS,&norm));
  if (norm > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"MatMatSolve: Norm of error %g\n",(double)norm));
  }

  /* Free data structures */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&RHS));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/ex215.out
    test:
      suffix: ns
    test:
      suffix: sym
      args: -symmetric_solve
    test:
      suffix: herm
      args: -hermitian_solve
    test:
      suffix: hpd
      args: -hpd_solve
    test:
      suffix: qr
      args: -ftype qr

TEST*/
