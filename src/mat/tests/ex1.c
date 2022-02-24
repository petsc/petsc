
static char help[] = "Tests LU, Cholesky, and QR factorization and MatMatSolve() for a sequential dense matrix. \n\
                      For MATSEQDENSE matrix, the factorization is just a thin wrapper to LAPACK.       \n\
                      For MATSEQDENSECUDA, it uses cusolverDn routines \n\n";

#include <petscmat.h>

static PetscErrorCode createMatsAndVecs(PetscInt m, PetscInt n, PetscInt nrhs, PetscBool full, Mat *_mat, Mat *_RHS, Mat *_SOLU, Vec *_x, Vec *_y, Vec *_b)
{
  PetscRandom    rand;
  Mat            mat,RHS,SOLU;
  PetscInt       rstart, rend;
  PetscInt       cstart, cend;
  PetscScalar    value = 1.0;
  Vec            x, y, b;

  PetscFunctionBegin;
  /* create multiple vectors RHS and SOLU */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&RHS));
  CHKERRQ(MatSetSizes(RHS,PETSC_DECIDE,PETSC_DECIDE,m,nrhs));
  CHKERRQ(MatSetType(RHS,MATDENSE));
  CHKERRQ(MatSetOptionsPrefix(RHS,"rhs_"));
  CHKERRQ(MatSetFromOptions(RHS));
  CHKERRQ(MatSeqDenseSetPreallocation(RHS,NULL));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(MatSetRandom(RHS,rand));

  if (m == n) {
    CHKERRQ(MatDuplicate(RHS,MAT_DO_NOT_COPY_VALUES,&SOLU));
  } else {
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&SOLU));
    CHKERRQ(MatSetSizes(SOLU,PETSC_DECIDE,PETSC_DECIDE,n,nrhs));
    CHKERRQ(MatSetType(SOLU,MATDENSE));
    CHKERRQ(MatSeqDenseSetPreallocation(SOLU,NULL));
  }
  CHKERRQ(MatSetRandom(SOLU,rand));

  /* create matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&mat));
  CHKERRQ(MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetType(mat,MATDENSE));
  CHKERRQ(MatSetFromOptions(mat));
  CHKERRQ(MatSetUp(mat));
  CHKERRQ(MatGetOwnershipRange(mat,&rstart,&rend));
  CHKERRQ(MatGetOwnershipRangeColumn(mat,&cstart,&cend));
  if (!full) {
    for (PetscInt i=rstart; i<rend; i++) {
      if (m == n) {
        value = (PetscReal)i+1;
        CHKERRQ(MatSetValues(mat,1,&i,1,&i,&value,INSERT_VALUES));
      } else {
        for (PetscInt j = cstart; j < cend; j++) {
          value = ((PetscScalar)i+1.)/(PetscSqr(i - j) + 1.);
          CHKERRQ(MatSetValues(mat,1,&i,1,&j,&value,INSERT_VALUES));
        }
      }
    }
    CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  } else {
    CHKERRQ(MatSetRandom(mat,rand));
    if (m == n) {
      Mat T;

      CHKERRQ(MatMatTransposeMult(mat,mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T));
      CHKERRQ(MatDestroy(&mat));
      mat  = T;
    }
  }

  /* create single vectors */
  CHKERRQ(MatCreateVecs(mat,&x,&b));
  CHKERRQ(VecDuplicate(x,&y));
  CHKERRQ(VecSet(x,value));
  CHKERRQ(PetscRandomDestroy(&rand));
  *_mat  = mat;
  *_RHS  = RHS;
  *_SOLU = SOLU;
  *_x    = x;
  *_y    = y;
  *_b    = b;
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            mat,F,RHS,SOLU;
  MatInfo        info;
  PetscErrorCode ierr;
  PetscInt       m = 15, n = 10,i,j,nrhs=2;
  Vec            x,y,b,ytmp;
  IS             perm;
  PetscReal      norm,tol=PETSC_SMALL;
  PetscMPIInt    size;
  char           solver[64];
  PetscBool      inplace,full = PETSC_FALSE, ldl = PETSC_TRUE, qr = PETSC_TRUE;

  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");
  CHKERRQ(PetscStrcpy(solver,"petsc"));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-ldl",&ldl,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-qr",&qr,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-full",&full,NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-solver_type",solver,sizeof(solver),NULL));

  CHKERRQ(createMatsAndVecs(n, n, nrhs, full, &mat, &RHS, &SOLU, &x, &y, &b));
  CHKERRQ(VecDuplicate(y,&ytmp));

  /* Only SeqDense* support in-place factorizations and NULL permutations */
  CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQDENSE,&inplace));
  CHKERRQ(MatGetLocalSize(mat,&i,NULL));
  CHKERRQ(MatGetOwnershipRange(mat,&j,NULL));
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,i,j,1,&perm));

  CHKERRQ(MatGetInfo(mat,MAT_LOCAL,&info));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix nonzeros = %" PetscInt_FMT ", allocated nonzeros = %" PetscInt_FMT "\n",
                     (PetscInt)info.nz_used,(PetscInt)info.nz_allocated);CHKERRQ(ierr);
  CHKERRQ(MatMult(mat,x,b));

  /* Cholesky factorization - perm and factinfo are ignored by LAPACK */
  /* in-place Cholesky */
  if (inplace) {
    Mat RHS2;

    CHKERRQ(MatDuplicate(mat,MAT_COPY_VALUES,&F));
    if (!ldl) CHKERRQ(MatSetOption(F,MAT_SPD,PETSC_TRUE));
    CHKERRQ(MatCholeskyFactor(F,perm,0));
    CHKERRQ(MatSolve(F,b,y));
    CHKERRQ(VecAXPY(y,-1.0,x));
    CHKERRQ(VecNorm(y,NORM_2,&norm));
    if (norm > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for in-place Cholesky %g\n",(double)norm));
    }

    CHKERRQ(MatMatSolve(F,RHS,SOLU));
    CHKERRQ(MatMatMult(mat,SOLU,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&RHS2));
    CHKERRQ(MatAXPY(RHS,-1.0,RHS2,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(RHS,NORM_FROBENIUS,&norm));
    if (norm > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error: Norm of residual for in-place Cholesky (MatMatSolve) %g\n",(double)norm));
    }
    CHKERRQ(MatDestroy(&F));
    CHKERRQ(MatDestroy(&RHS2));
  }

  /* out-of-place Cholesky */
  CHKERRQ(MatGetFactor(mat,solver,MAT_FACTOR_CHOLESKY,&F));
  if (!ldl) CHKERRQ(MatSetOption(F,MAT_SPD,PETSC_TRUE));
  CHKERRQ(MatCholeskyFactorSymbolic(F,mat,perm,0));
  CHKERRQ(MatCholeskyFactorNumeric(F,mat,0));
  CHKERRQ(MatSolve(F,b,y));
  CHKERRQ(VecAXPY(y,-1.0,x));
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for out-of-place Cholesky %g\n",(double)norm));
  }
  CHKERRQ(MatDestroy(&F));

  /* LU factorization - perms and factinfo are ignored by LAPACK */
  i    = n-1;
  CHKERRQ(MatZeroRows(mat,1,&i,-1.0,NULL,NULL));
  CHKERRQ(MatMult(mat,x,b));

  /* in-place LU */
  if (inplace) {
    Mat RHS2;

    CHKERRQ(MatDuplicate(mat,MAT_COPY_VALUES,&F));
    CHKERRQ(MatLUFactor(F,perm,perm,0));
    CHKERRQ(MatSolve(F,b,y));
    CHKERRQ(VecAXPY(y,-1.0,x));
    CHKERRQ(VecNorm(y,NORM_2,&norm));
    if (norm > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for in-place LU %g\n",(double)norm));
    }
    CHKERRQ(MatMatSolve(F,RHS,SOLU));
    CHKERRQ(MatMatMult(mat,SOLU,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&RHS2));
    CHKERRQ(MatAXPY(RHS,-1.0,RHS2,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(RHS,NORM_FROBENIUS,&norm));
    if (norm > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error: Norm of residual for in-place LU (MatMatSolve) %g\n",(double)norm));
    }
    CHKERRQ(MatDestroy(&F));
    CHKERRQ(MatDestroy(&RHS2));
  }

  /* out-of-place LU */
  CHKERRQ(MatGetFactor(mat,solver,MAT_FACTOR_LU,&F));
  CHKERRQ(MatLUFactorSymbolic(F,mat,perm,perm,0));
  CHKERRQ(MatLUFactorNumeric(F,mat,0));
  CHKERRQ(MatSolve(F,b,y));
  CHKERRQ(VecAXPY(y,-1.0,x));
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for out-of-place LU %g\n",(double)norm));
  }

  /* free space */
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(MatDestroy(&mat));
  CHKERRQ(MatDestroy(&RHS));
  CHKERRQ(MatDestroy(&SOLU));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&ytmp));

  if (qr) {
    /* setup rectanglar */
    CHKERRQ(createMatsAndVecs(m, n, nrhs, full, &mat, &RHS, &SOLU, &x, &y, &b));
    CHKERRQ(VecDuplicate(y,&ytmp));

    /* QR factorization - perms and factinfo are ignored by LAPACK */
    CHKERRQ(MatMult(mat,x,b));

    /* in-place QR */
    if (inplace) {
      Mat SOLU2;

      CHKERRQ(MatDuplicate(mat,MAT_COPY_VALUES,&F));
      CHKERRQ(MatQRFactor(F,NULL,0));
      CHKERRQ(MatSolve(F,b,y));
      CHKERRQ(VecAXPY(y,-1.0,x));
      CHKERRQ(VecNorm(y,NORM_2,&norm));
      if (norm > tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for in-place QR %g\n",(double)norm));
      }
      CHKERRQ(MatMatMult(mat,SOLU,MAT_REUSE_MATRIX,PETSC_DEFAULT,&RHS));
      CHKERRQ(MatDuplicate(SOLU, MAT_DO_NOT_COPY_VALUES, &SOLU2));
      CHKERRQ(MatMatSolve(F,RHS,SOLU2));
      CHKERRQ(MatAXPY(SOLU2,-1.0,SOLU,SAME_NONZERO_PATTERN));
      CHKERRQ(MatNorm(SOLU2,NORM_FROBENIUS,&norm));
      if (norm > tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error: Norm of error for in-place QR (MatMatSolve) %g\n",(double)norm));
      }
      CHKERRQ(MatDestroy(&F));
      CHKERRQ(MatDestroy(&SOLU2));
    }

    /* out-of-place QR */
    CHKERRQ(MatGetFactor(mat,solver,MAT_FACTOR_QR,&F));
    CHKERRQ(MatQRFactorSymbolic(F,mat,NULL,NULL));
    CHKERRQ(MatQRFactorNumeric(F,mat,NULL));
    CHKERRQ(MatSolve(F,b,y));
    CHKERRQ(VecAXPY(y,-1.0,x));
    CHKERRQ(VecNorm(y,NORM_2,&norm));
    if (norm > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for out-of-place QR %g\n",(double)norm));
    }

    if (m == n) {
      /* out-of-place MatSolveTranspose */
      CHKERRQ(MatMultTranspose(mat,x,b));
      CHKERRQ(MatSolveTranspose(F,b,y));
      CHKERRQ(VecAXPY(y,-1.0,x));
      CHKERRQ(VecNorm(y,NORM_2,&norm));
      if (norm > tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for out-of-place QR %g\n",(double)norm));
      }
    }

    /* free space */
    CHKERRQ(MatDestroy(&F));
    CHKERRQ(MatDestroy(&mat));
    CHKERRQ(MatDestroy(&RHS));
    CHKERRQ(MatDestroy(&SOLU));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecDestroy(&y));
    CHKERRQ(VecDestroy(&ytmp));
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
     requires: cuda
     suffix: seqdensecuda
     args: -mat_type seqdensecuda -rhs_mat_type seqdensecuda -ldl 0 -solver_type {{petsc cuda}}
     output_file: output/ex1_1.out

   test:
     requires: cuda
     suffix: seqdensecuda_2
     args: -ldl 0 -solver_type cuda
     output_file: output/ex1_1.out

   test:
     requires: cuda
     suffix: seqdensecuda_seqaijcusparse
     args: -mat_type seqaijcusparse -rhs_mat_type seqdensecuda -qr 0
     output_file: output/ex1_2.out

   test:
     requires: cuda viennacl
     suffix: seqdensecuda_seqaijviennacl
     args: -mat_type seqaijviennacl -rhs_mat_type seqdensecuda -qr 0
     output_file: output/ex1_2.out

   test:
     suffix: 4
     args: -m 10 -n 10
     output_file: output/ex1_1.out

TEST*/
