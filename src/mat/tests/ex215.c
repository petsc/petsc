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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor test");
  /* Determine which type of solver we want to test for */
  herm = PETSC_FALSE;
  symm = PETSC_FALSE;
  hpd  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-symmetric_solve",&symm,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-hermitian_solve",&herm,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-hpd_solve",&hpd,NULL));

  /* Determine file from which we read the matrix A */
  ftyp = MAT_FACTOR_LU;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&data_provided));
  if (!data_provided) { /* get matrices from PETSc distribution */
    CHKERRQ(PetscStrcpy(file,"${PETSC_DIR}/share/petsc/datafiles/matrices/"));
    if (hpd) {
#if defined(PETSC_USE_COMPLEX)
      CHKERRQ(PetscStrcat(file,"hpd-complex-"));
#else
      CHKERRQ(PetscStrcat(file,"spd-real-"));
#endif
      ftyp = MAT_FACTOR_CHOLESKY;
    } else {
#if defined(PETSC_USE_COMPLEX)
      CHKERRQ(PetscStrcat(file,"nh-complex-"));
#else
      CHKERRQ(PetscStrcat(file,"ns-real-"));
#endif
    }
#if defined(PETSC_USE_64BIT_INDICES)
    CHKERRQ(PetscStrcat(file,"int64-"));
#else
    CHKERRQ(PetscStrcat(file,"int32-"));
#endif
#if defined(PETSC_USE_REAL_SINGLE)
    CHKERRQ(PetscStrcat(file,"float32"));
#else
    CHKERRQ(PetscStrcat(file,"float64"));
#endif
  }

  /* Load matrix A */
#if defined(PETSC_USE_REAL___FLOAT128)
  CHKERRQ(PetscOptionsInsertString(NULL,"-binary_read_double"));
#endif
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  CHKERRQ(MatConvert(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A));
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
  CHKERRQ(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&RHS));

  /* Create vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&b));
  CHKERRQ(VecDuplicate(x,&u)); /* save the true solution */

  /* make a symmetric matrix */
  if (symm) {
    Mat AT;

    CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&AT));
    CHKERRQ(MatAXPY(A,1.0,AT,SAME_NONZERO_PATTERN));
    CHKERRQ(MatDestroy(&AT));
    ftyp = MAT_FACTOR_CHOLESKY;
  }
  /* make an hermitian matrix */
  if (herm) {
    Mat AH;

    CHKERRQ(MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&AH));
    CHKERRQ(MatAXPY(A,1.0,AH,SAME_NONZERO_PATTERN));
    CHKERRQ(MatDestroy(&AH));
    ftyp = MAT_FACTOR_CHOLESKY;
  }
  CHKERRQ(PetscObjectSetName((PetscObject)A,"A"));
  CHKERRQ(MatViewFromOptions(A,NULL,"-amat_view"));

  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&F));
  CHKERRQ(MatSetOption(F,MAT_SYMMETRIC,symm));
  /* it seems that the SPD concept in PETSc extends naturally to Hermitian Positive definitess */
  CHKERRQ(MatSetOption(F,MAT_HERMITIAN,(PetscBool)(hpd || herm)));
  CHKERRQ(MatSetOption(F,MAT_SPD,hpd));
  {
    PetscInt iftyp = ftyp;
    CHKERRQ(PetscOptionsGetEList(NULL,NULL,"-ftype",MatFactorTypes,MAT_FACTOR_NUM_TYPES,&iftyp,NULL));
    ftyp = (MatFactorType) iftyp;
  }
  if (ftyp == MAT_FACTOR_LU) {
    CHKERRQ(MatLUFactor(F,NULL,NULL,NULL));
  } else if (ftyp == MAT_FACTOR_CHOLESKY) {
    CHKERRQ(MatCholeskyFactor(F,NULL,NULL));
  } else if (ftyp == MAT_FACTOR_QR) {
    CHKERRQ(MatQRFactor(F,NULL,NULL));
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Factorization %s not supported in this example", MatFactorTypes[ftyp]);

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
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"MatSolve error: Norm of error %g, residual %g\n",(double)norm,(double)resi));
      } else {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"MatSolveTranspose error: Norm of error %g, residual %g\n",(double)norm,(double)resi));
      }
    }
  }
  CHKERRQ(MatMatMult(A,C,MAT_REUSE_MATRIX,2.0,&RHS));
  CHKERRQ(MatMatSolve(F,RHS,X));

  /* Check the error */
  CHKERRQ(MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(X,NORM_FROBENIUS,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"MatMatSolve: Norm of error %g\n",(double)norm));
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
