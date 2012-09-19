
static char help[] = "Tests LU, Cholesky factorization and MatMatSolve() for a sequential dense matrix. \n\
                      For MATSEQDENSE matrix, the factorization is just a thin wrapper to LAPACK \n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            mat,F,RHS,SOLU;
  MatInfo        info;
  PetscErrorCode ierr;
  PetscInt       m = 10,n = 10,i,j,rstart,rend,nrhs=2;
  PetscScalar    value = 1.0;
  Vec            x,y,b,ytmp;
  PetscReal      norm,tol=1.e-15;
  PetscMPIInt    size;
  PetscScalar    *rhs_array,*solu_array;
  PetscRandom    rand;
  PetscScalar    *array,rval;

  PetscInitialize(&argc,&argv,(char*) 0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* create single vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&ytmp);CHKERRQ(ierr);
  ierr = VecSet(x,value);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);

  /* create multiple vectors RHS and SOLU */
  ierr = MatCreate(PETSC_COMM_WORLD,&RHS);CHKERRQ(ierr);
  ierr = MatSetSizes(RHS,PETSC_DECIDE,PETSC_DECIDE,n,nrhs);CHKERRQ(ierr);
  ierr = MatSetType(RHS,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(RHS);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(RHS,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatDenseGetArray(RHS,&array);CHKERRQ(ierr);
  for (j=0; j<nrhs; j++){
    for (i=0; i<n; i++){
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      array[n*j+i] = rval;
    }
  }
  ierr = MatDenseRestoreArray(RHS,&array);CHKERRQ(ierr);

  ierr = MatDuplicate(RHS,MAT_DO_NOT_COPY_VALUES,&SOLU);CHKERRQ(ierr);

  /* create matrix */
  ierr = MatCreateSeqDense(PETSC_COMM_WORLD,m,n,PETSC_NULL,&mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    value = (PetscReal)i+1;
    ierr = MatSetValues(mat,1,&i,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"matrix nonzeros = %D, allocated nonzeros = %D\n",
    (PetscInt)info.nz_used,(PetscInt)info.nz_allocated);CHKERRQ(ierr);

  /* Cholesky factorization - perm and factinfo are ignored by LAPACK */
  /* in-place Cholesky */
  ierr = MatMult(mat,x,b);CHKERRQ(ierr);
  ierr = MatDuplicate(mat,MAT_COPY_VALUES,&F);CHKERRQ(ierr);
  ierr = MatCholeskyFactor(F,0,0);CHKERRQ(ierr);
  ierr = MatSolve(F,b,y);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  value = -1.0; ierr = VecAXPY(y,value,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for Cholesky %G\n",norm);CHKERRQ(ierr);
  }

  /* out-place Cholesky */
  ierr = MatGetFactor(mat,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
  ierr = MatCholeskyFactorSymbolic(F,mat,0,0);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(F,mat,0);CHKERRQ(ierr);
  ierr = MatSolve(F,b,y);CHKERRQ(ierr);
  value = -1.0; ierr = VecAXPY(y,value,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for Cholesky %G\n",norm);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  /* LU factorization - perms and factinfo are ignored by LAPACK */
  i = m-1; value = 1.0;
  ierr = MatSetValues(mat,1,&i,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatMult(mat,x,b);CHKERRQ(ierr);
  ierr = MatDuplicate(mat,MAT_COPY_VALUES,&F);CHKERRQ(ierr);

  /* in-place LU */
  ierr = MatLUFactor(F,0,0,0);CHKERRQ(ierr);
  ierr = MatSolve(F,b,y);CHKERRQ(ierr);
  value = -1.0; ierr = VecAXPY(y,value,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for LU %G\n",norm);CHKERRQ(ierr);
  }
  ierr = MatMatSolve(F,RHS,SOLU);CHKERRQ(ierr);
  ierr = MatDenseGetArray(SOLU,&solu_array);CHKERRQ(ierr);
  ierr = MatDenseGetArray(RHS,&rhs_array);CHKERRQ(ierr);
  for (j=0; j<nrhs; j++){
    ierr = VecPlaceArray(y,solu_array+j*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(b,rhs_array+j*m);CHKERRQ(ierr);

    ierr = MatMult(mat,y,ytmp);CHKERRQ(ierr);
    ierr = VecAXPY(ytmp,-1.0,b);CHKERRQ(ierr); /* ytmp = mat*SOLU[:,j] - RHS[:,j] */
    ierr = VecNorm(ytmp,NORM_2,&norm);CHKERRQ(ierr);
    if (norm > tol){
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: Norm of residual for LU %G\n",norm);CHKERRQ(ierr);
    }

    ierr = VecResetArray(b);CHKERRQ(ierr);
    ierr = VecResetArray(y);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(RHS,&rhs_array);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(SOLU,&solu_array);CHKERRQ(ierr);

  ierr = MatDestroy(&F);CHKERRQ(ierr);

  /* out-place LU */
  ierr = MatGetFactor(mat,MATSOLVERPETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(F,mat,0,0,0);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(F,mat,0);CHKERRQ(ierr);
  ierr = MatSolve(F,b,y);CHKERRQ(ierr);
  value = -1.0; ierr = VecAXPY(y,value,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > tol){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for LU %G\n",norm);CHKERRQ(ierr);
  }

  /* free space */
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = MatDestroy(&RHS);CHKERRQ(ierr);
  ierr = MatDestroy(&SOLU);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&ytmp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

