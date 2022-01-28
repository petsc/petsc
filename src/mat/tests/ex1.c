
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* create multiple vectors RHS and SOLU */
  ierr = MatCreate(PETSC_COMM_WORLD,&RHS);CHKERRQ(ierr);
  ierr = MatSetSizes(RHS,PETSC_DECIDE,PETSC_DECIDE,m,nrhs);CHKERRQ(ierr);
  ierr = MatSetType(RHS,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(RHS,"rhs_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(RHS);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(RHS,NULL);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatSetRandom(RHS,rand);CHKERRQ(ierr);

  if (m == n) {
    ierr = MatDuplicate(RHS,MAT_DO_NOT_COPY_VALUES,&SOLU);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PETSC_COMM_WORLD,&SOLU);CHKERRQ(ierr);
    ierr = MatSetSizes(SOLU,PETSC_DECIDE,PETSC_DECIDE,n,nrhs);CHKERRQ(ierr);
    ierr = MatSetType(SOLU,MATDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(SOLU,NULL);CHKERRQ(ierr);
  }
  ierr = MatSetRandom(SOLU,rand);CHKERRQ(ierr);

  /* create matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(mat,&cstart,&cend);CHKERRQ(ierr);
  if (!full) {
    for (PetscInt i=rstart; i<rend; i++) {
      if (m == n) {
        value = (PetscReal)i+1;
        ierr  = MatSetValues(mat,1,&i,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        for (PetscInt j = cstart; j < cend; j++) {
          value = ((PetscScalar)i+1.)/(PetscSqr(i - j) + 1.);
          ierr  = MatSetValues(mat,1,&i,1,&j,&value,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    ierr = MatSetRandom(mat,rand);CHKERRQ(ierr);
    if (m == n) {
      Mat T;

      ierr = MatMatTransposeMult(mat,mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&T);CHKERRQ(ierr);
      ierr = MatDestroy(&mat);CHKERRQ(ierr);
      mat  = T;
    }
  }

  /* create single vectors */
  ierr  = MatCreateVecs(mat,&x,&b);CHKERRQ(ierr);
  ierr  = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr  = VecSet(x,value);CHKERRQ(ierr);
  ierr  = PetscRandomDestroy(&rand);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");
  ierr = PetscStrcpy(solver,"petsc");CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-ldl",&ldl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-qr",&qr,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-full",&full,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-solver_type",solver,sizeof(solver),NULL);CHKERRQ(ierr);

  ierr = createMatsAndVecs(n, n, nrhs, full, &mat, &RHS, &SOLU, &x, &y, &b);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&ytmp);CHKERRQ(ierr);

  /* Only SeqDense* support in-place factorizations and NULL permutations */
  ierr = PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQDENSE,&inplace);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&i,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&j,NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,i,j,1,&perm);CHKERRQ(ierr);

  ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix nonzeros = %" PetscInt_FMT ", allocated nonzeros = %" PetscInt_FMT "\n",
                     (PetscInt)info.nz_used,(PetscInt)info.nz_allocated);CHKERRQ(ierr);
  ierr = MatMult(mat,x,b);CHKERRQ(ierr);

  /* Cholesky factorization - perm and factinfo are ignored by LAPACK */
  /* in-place Cholesky */
  if (inplace) {
    Mat RHS2;

    ierr = MatDuplicate(mat,MAT_COPY_VALUES,&F);CHKERRQ(ierr);
    if (!ldl) { ierr = MatSetOption(F,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr); }
    ierr = MatCholeskyFactor(F,perm,0);CHKERRQ(ierr);
    ierr = MatSolve(F,b,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for in-place Cholesky %g\n",(double)norm);CHKERRQ(ierr);
    }

    ierr = MatMatSolve(F,RHS,SOLU);CHKERRQ(ierr);
    ierr = MatMatMult(mat,SOLU,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&RHS2);CHKERRQ(ierr);
    ierr = MatAXPY(RHS,-1.0,RHS2,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(RHS,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: Norm of residual for in-place Cholesky (MatMatSolve) %g\n",(double)norm);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = MatDestroy(&RHS2);CHKERRQ(ierr);
  }

  /* out-of-place Cholesky */
  ierr = MatGetFactor(mat,solver,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
  if (!ldl) { ierr = MatSetOption(F,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr); }
  ierr = MatCholeskyFactorSymbolic(F,mat,perm,0);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(F,mat,0);CHKERRQ(ierr);
  ierr = MatSolve(F,b,y);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for out-of-place Cholesky %g\n",(double)norm);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  /* LU factorization - perms and factinfo are ignored by LAPACK */
  i    = n-1;
  ierr = MatZeroRows(mat,1,&i,-1.0,NULL,NULL);CHKERRQ(ierr);
  ierr = MatMult(mat,x,b);CHKERRQ(ierr);

  /* in-place LU */
  if (inplace) {
    Mat RHS2;

    ierr = MatDuplicate(mat,MAT_COPY_VALUES,&F);CHKERRQ(ierr);
    ierr = MatLUFactor(F,perm,perm,0);CHKERRQ(ierr);
    ierr = MatSolve(F,b,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for in-place LU %g\n",(double)norm);CHKERRQ(ierr);
    }
    ierr = MatMatSolve(F,RHS,SOLU);CHKERRQ(ierr);
    ierr = MatMatMult(mat,SOLU,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&RHS2);CHKERRQ(ierr);
    ierr = MatAXPY(RHS,-1.0,RHS2,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(RHS,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: Norm of residual for in-place LU (MatMatSolve) %g\n",(double)norm);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = MatDestroy(&RHS2);CHKERRQ(ierr);
  }

  /* out-of-place LU */
  ierr = MatGetFactor(mat,solver,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(F,mat,perm,perm,0);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(F,mat,0);CHKERRQ(ierr);
  ierr = MatSolve(F,b,y);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for out-of-place LU %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* free space */
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = MatDestroy(&RHS);CHKERRQ(ierr);
  ierr = MatDestroy(&SOLU);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&ytmp);CHKERRQ(ierr);

  if (qr) {
    /* setup rectanglar */
    ierr = createMatsAndVecs(m, n, nrhs, full, &mat, &RHS, &SOLU, &x, &y, &b);CHKERRQ(ierr);
    ierr = VecDuplicate(y,&ytmp);CHKERRQ(ierr);

    /* QR factorization - perms and factinfo are ignored by LAPACK */
    ierr = MatMult(mat,x,b);CHKERRQ(ierr);

    /* in-place QR */
    if (inplace) {
      Mat SOLU2;

      ierr = MatDuplicate(mat,MAT_COPY_VALUES,&F);CHKERRQ(ierr);
      ierr = MatQRFactor(F,NULL,0);CHKERRQ(ierr);
      ierr = MatSolve(F,b,y);CHKERRQ(ierr);
      ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
      ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
      if (norm > tol) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for in-place QR %g\n",(double)norm);CHKERRQ(ierr);
      }
      ierr = MatMatMult(mat,SOLU,MAT_REUSE_MATRIX,PETSC_DEFAULT,&RHS);CHKERRQ(ierr);
      ierr = MatDuplicate(SOLU, MAT_DO_NOT_COPY_VALUES, &SOLU2);CHKERRQ(ierr);
      ierr = MatMatSolve(F,RHS,SOLU2);CHKERRQ(ierr);
      ierr = MatAXPY(SOLU2,-1.0,SOLU,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(SOLU2,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
      if (norm > tol) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: Norm of error for in-place QR (MatMatSolve) %g\n",(double)norm);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&F);CHKERRQ(ierr);
      ierr = MatDestroy(&SOLU2);CHKERRQ(ierr);
    }

    /* out-of-place QR */
    ierr = MatGetFactor(mat,solver,MAT_FACTOR_QR,&F);CHKERRQ(ierr);
    ierr = MatQRFactorSymbolic(F,mat,NULL,NULL);CHKERRQ(ierr);
    ierr = MatQRFactorNumeric(F,mat,NULL);CHKERRQ(ierr);
    ierr = MatSolve(F,b,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for out-of-place QR %g\n",(double)norm);CHKERRQ(ierr);
    }

    if (m == n) {
      /* out-of-place MatSolveTranspose */
      ierr = MatMultTranspose(mat,x,b);CHKERRQ(ierr);
      ierr = MatSolveTranspose(F,b,y);CHKERRQ(ierr);
      ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
      ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
      if (norm > tol) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for out-of-place QR %g\n",(double)norm);CHKERRQ(ierr);
      }
    }

    /* free space */
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = MatDestroy(&mat);CHKERRQ(ierr);
    ierr = MatDestroy(&RHS);CHKERRQ(ierr);
    ierr = MatDestroy(&SOLU);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecDestroy(&ytmp);CHKERRQ(ierr);
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
