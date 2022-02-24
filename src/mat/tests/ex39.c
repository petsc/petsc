
static char help[] = "Tests Elemental interface.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            Cdense,B,C,Ct;
  Vec            d;
  PetscInt       i,j,m = 5,n,nrows,ncols;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscErrorCode ierr;
  PetscScalar    *v;
  PetscMPIInt    rank,size;
  PetscReal      Cnorm;
  PetscBool      flg,mats_view=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n    = m;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,m,n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(C,MATELEMENTAL));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(MatGetOwnershipIS(C,&isrows,&iscols));
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  CHKERRQ(PetscMalloc1(nrows*ncols,&v));
#if defined(PETSC_USE_COMPLEX)
  PetscRandom rand;
  PetscScalar rval;
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      CHKERRQ(PetscRandomGetValue(rand,&rval));
      v[i*ncols+j] = rval;
    }
  }
  CHKERRQ(PetscRandomDestroy(&rand));
#else
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      v[i*ncols+j] = (PetscReal)(10000*rank+100*rows[i]+cols[j]);
    }
  }
#endif
  CHKERRQ(MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES));
  CHKERRQ(ISRestoreIndices(isrows,&rows));
  CHKERRQ(ISRestoreIndices(iscols,&cols));
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(ISDestroy(&isrows));
  CHKERRQ(ISDestroy(&iscols));

  /* Test MatView(), MatDuplicate() and out-of-place MatConvert() */
  CHKERRQ(MatDuplicate(C,MAT_COPY_VALUES,&B));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Duplicated C:\n"));
    CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatConvert(C,MATMPIDENSE,MAT_INITIAL_MATRIX,&Cdense));
  CHKERRQ(MatMultEqual(C,Cdense,5,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Cdense != C. MatConvert() fails");

  /* Test MatNorm() */
  CHKERRQ(MatNorm(C,NORM_1,&Cnorm));

  /* Test MatTranspose(), MatZeroEntries() and MatGetDiagonal() */
  CHKERRQ(MatTranspose(C,MAT_INITIAL_MATRIX,&Ct));
  CHKERRQ(MatConjugate(Ct));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"C's Transpose Conjugate:\n"));
    CHKERRQ(MatView(Ct,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(MatZeroEntries(Ct));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&d));
  CHKERRQ(VecSetSizes(d,m>n ? n : m,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(d));
  CHKERRQ(MatGetDiagonal(C,d));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Diagonal of C:\n"));
    CHKERRQ(VecView(d,PETSC_VIEWER_STDOUT_WORLD));
  }
  if (m>n) {
    CHKERRQ(MatDiagonalScale(C,NULL,d));
  } else {
    CHKERRQ(MatDiagonalScale(C,d,NULL));
  }
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Diagonal Scaled C:\n"));
    CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Test MatAXPY(), MatAYPX() and in-place MatConvert() */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,m,n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(B,MATELEMENTAL));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatGetOwnershipIS(B,&isrows,&iscols));
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      v[i*ncols+j] = (PetscReal)(1000*rows[i]+cols[j]);
    }
  }
  CHKERRQ(MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES));
  CHKERRQ(PetscFree(v));
  CHKERRQ(ISRestoreIndices(isrows,&rows));
  CHKERRQ(ISRestoreIndices(iscols,&cols));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAXPY(B,2.5,C,SAME_NONZERO_PATTERN));
  CHKERRQ(MatAYPX(B,3.75,C,SAME_NONZERO_PATTERN));
  CHKERRQ(MatConvert(B,MATDENSE,MAT_INPLACE_MATRIX,&B));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"B after MatAXPY and MatAYPX:\n"));
    CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(ISDestroy(&isrows));
  CHKERRQ(ISDestroy(&iscols));
  CHKERRQ(MatDestroy(&B));

  /* Test MatMatTransposeMult(): B = C*C^T */
  CHKERRQ(MatMatTransposeMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B));
  CHKERRQ(MatScale(C,2.0));
  CHKERRQ(MatMatTransposeMult(C,C,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
  CHKERRQ(MatMatTransposeMultEqual(C,C,B,10,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatMatTransposeMult: B != C*B^T");

  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"C MatMatTransposeMult C:\n"));
    CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }

  CHKERRQ(MatDestroy(&Cdense));
  CHKERRQ(PetscFree(v));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&Ct));
  CHKERRQ(VecDestroy(&d));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      args: -m 3 -n 2
      requires: elemental

   test:
      suffix: 2
      nsize: 6
      args: -m 2 -n 3
      requires: elemental

   test:
      suffix: 3
      nsize: 1
      args: -m 2 -n 3
      requires: elemental
      output_file: output/ex39_1.out

TEST*/
