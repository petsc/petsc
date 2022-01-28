static char help[] = "Tests MatSolve(), MatSolveTranspose() and MatMatSolve() with SEQDENSE\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,RHS,C,F,X;
  Vec            u,x,b;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       m,n,nsolve,nrhs;
  PetscReal      norm,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscRandom    rand;
  PetscBool      data_provided,herm,symm,hpd;
  MatFactorType  ftyp;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor test");
  /* Determine which type of solver we want to test for */
  herm = PETSC_FALSE;
  symm = PETSC_FALSE;
  hpd  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-symmetric_solve",&symm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-hermitian_solve",&herm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-hpd_solve",&hpd,NULL);CHKERRQ(ierr);

  /* Determine file from which we read the matrix A */
  ftyp = MAT_FACTOR_LU;
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&data_provided);CHKERRQ(ierr);
  if (!data_provided) { /* get matrices from PETSc distribution */
    ierr = PetscStrcpy(file,"${PETSC_DIR}/share/petsc/datafiles/matrices/");CHKERRQ(ierr);
    if (hpd) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscStrcat(file,"hpd-complex-");CHKERRQ(ierr);
#else
      ierr = PetscStrcat(file,"spd-real-");CHKERRQ(ierr);
#endif
      ftyp = MAT_FACTOR_CHOLESKY;
    } else {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscStrcat(file,"nh-complex-");CHKERRQ(ierr);
#else
      ierr = PetscStrcat(file,"ns-real-");CHKERRQ(ierr);
#endif
    }
#if defined(PETSC_USE_64BIT_INDICES)
    ierr = PetscStrcat(file,"int64-");CHKERRQ(ierr);
#else
    ierr = PetscStrcat(file,"int32-");CHKERRQ(ierr);
#endif
#if defined(PETSC_USE_REAL_SINGLE)
    ierr = PetscStrcat(file,"float32");CHKERRQ(ierr);
#else
    ierr = PetscStrcat(file,"float64");CHKERRQ(ierr);
#endif
  }

  /* Load matrix A */
#if defined(PETSC_USE_REAL___FLOAT128)
  ierr = PetscOptionsInsertString(NULL,"-binary_read_double");CHKERRQ(ierr);
#endif
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatConvert(A,MATSEQDENSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m != n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")", m, n);

  /* Create dense matrix C and X; C holds true solution with identical columns */
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
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&RHS);CHKERRQ(ierr);

  /* Create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr); /* save the true solution */

  /* make a symmetric matrix */
  if (symm) {
    Mat AT;

    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&AT);CHKERRQ(ierr);
    ierr = MatAXPY(A,1.0,AT,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroy(&AT);CHKERRQ(ierr);
    ftyp = MAT_FACTOR_CHOLESKY;
  }
  /* make an hermitian matrix */
  if (herm) {
    Mat AH;

    ierr = MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&AH);CHKERRQ(ierr);
    ierr = MatAXPY(A,1.0,AH,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroy(&AH);CHKERRQ(ierr);
    ftyp = MAT_FACTOR_CHOLESKY;
  }
  ierr = PetscObjectSetName((PetscObject)A,"A");CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-amat_view");CHKERRQ(ierr);

  ierr = MatDuplicate(A,MAT_COPY_VALUES,&F);CHKERRQ(ierr);
  ierr = MatSetOption(F,MAT_SYMMETRIC,symm);CHKERRQ(ierr);
  /* it seems that the SPD concept in PETSc extends naturally to Hermitian Positive definitess */
  ierr = MatSetOption(F,MAT_HERMITIAN,(PetscBool)(hpd || herm));CHKERRQ(ierr);
  ierr = MatSetOption(F,MAT_SPD,hpd);CHKERRQ(ierr);
  {
    PetscInt iftyp = ftyp;
    ierr = PetscOptionsGetEList(NULL,NULL,"-ftype",MatFactorTypes,MAT_FACTOR_NUM_TYPES,&iftyp,NULL);CHKERRQ(ierr);
    ftyp = (MatFactorType) iftyp;
  }
  if (ftyp == MAT_FACTOR_LU) {
    ierr = MatLUFactor(F,NULL,NULL,NULL);CHKERRQ(ierr);
  } else if (ftyp == MAT_FACTOR_CHOLESKY) {
    ierr = MatCholeskyFactor(F,NULL,NULL);CHKERRQ(ierr);
  } else if (ftyp == MAT_FACTOR_QR) {
    ierr = MatQRFactor(F,NULL,NULL);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Factorization %s not supported in this example", MatFactorTypes[ftyp]);

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
        ierr = PetscPrintf(PETSC_COMM_SELF,"MatSolve error: Norm of error %g, residual %g\n",(double)norm,(double)resi);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_SELF,"MatSolveTranspose error: Norm of error %g, residual %g\n",(double)norm,(double)resi);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatMatMult(A,C,MAT_REUSE_MATRIX,2.0,&RHS);CHKERRQ(ierr);
  ierr = MatMatSolve(F,RHS,X);CHKERRQ(ierr);

  /* Check the error */
  ierr = MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(X,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"MatMatSolve: Norm of error %g\n",(double)norm);CHKERRQ(ierr);
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
