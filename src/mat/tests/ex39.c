
static char help[] = "Tests Elemental interface.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            Cdense,B,C,Ct;
  Vec            d;
  PetscInt       i,j,m = 5,n,nrows,ncols;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscScalar    *v;
  PetscMPIInt    rank,size;
  PetscReal      Cnorm;
  PetscBool      flg,mats_view=PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n    = m;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,m,n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(C,MATELEMENTAL));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(MatGetOwnershipIS(C,&isrows,&iscols));
  PetscCall(ISGetLocalSize(isrows,&nrows));
  PetscCall(ISGetIndices(isrows,&rows));
  PetscCall(ISGetLocalSize(iscols,&ncols));
  PetscCall(ISGetIndices(iscols,&cols));
  PetscCall(PetscMalloc1(nrows*ncols,&v));
#if defined(PETSC_USE_COMPLEX)
  PetscRandom rand;
  PetscScalar rval;
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      PetscCall(PetscRandomGetValue(rand,&rval));
      v[i*ncols+j] = rval;
    }
  }
  PetscCall(PetscRandomDestroy(&rand));
#else
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      v[i*ncols+j] = (PetscReal)(10000*rank+100*rows[i]+cols[j]);
    }
  }
#endif
  PetscCall(MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES));
  PetscCall(ISRestoreIndices(isrows,&rows));
  PetscCall(ISRestoreIndices(iscols,&cols));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&iscols));

  /* Test MatView(), MatDuplicate() and out-of-place MatConvert() */
  PetscCall(MatDuplicate(C,MAT_COPY_VALUES,&B));
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Duplicated C:\n"));
    PetscCall(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(MatDestroy(&B));
  PetscCall(MatConvert(C,MATMPIDENSE,MAT_INITIAL_MATRIX,&Cdense));
  PetscCall(MatMultEqual(C,Cdense,5,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Cdense != C. MatConvert() fails");

  /* Test MatNorm() */
  PetscCall(MatNorm(C,NORM_1,&Cnorm));

  /* Test MatTranspose(), MatZeroEntries() and MatGetDiagonal() */
  PetscCall(MatTranspose(C,MAT_INITIAL_MATRIX,&Ct));
  PetscCall(MatConjugate(Ct));
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"C's Transpose Conjugate:\n"));
    PetscCall(MatView(Ct,PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(MatZeroEntries(Ct));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&d));
  PetscCall(VecSetSizes(d,m>n ? n : m,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(d));
  PetscCall(MatGetDiagonal(C,d));
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Diagonal of C:\n"));
    PetscCall(VecView(d,PETSC_VIEWER_STDOUT_WORLD));
  }
  if (m>n) {
    PetscCall(MatDiagonalScale(C,NULL,d));
  } else {
    PetscCall(MatDiagonalScale(C,d,NULL));
  }
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Diagonal Scaled C:\n"));
    PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Test MatAXPY(), MatAYPX() and in-place MatConvert() */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,m,n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(B,MATELEMENTAL));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  PetscCall(MatGetOwnershipIS(B,&isrows,&iscols));
  PetscCall(ISGetLocalSize(isrows,&nrows));
  PetscCall(ISGetIndices(isrows,&rows));
  PetscCall(ISGetLocalSize(iscols,&ncols));
  PetscCall(ISGetIndices(iscols,&cols));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      v[i*ncols+j] = (PetscReal)(1000*rows[i]+cols[j]);
    }
  }
  PetscCall(MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES));
  PetscCall(PetscFree(v));
  PetscCall(ISRestoreIndices(isrows,&rows));
  PetscCall(ISRestoreIndices(iscols,&cols));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAXPY(B,2.5,C,SAME_NONZERO_PATTERN));
  PetscCall(MatAYPX(B,3.75,C,SAME_NONZERO_PATTERN));
  PetscCall(MatConvert(B,MATDENSE,MAT_INPLACE_MATRIX,&B));
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"B after MatAXPY and MatAYPX:\n"));
    PetscCall(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&iscols));
  PetscCall(MatDestroy(&B));

  /* Test MatMatTransposeMult(): B = C*C^T */
  PetscCall(MatMatTransposeMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B));
  PetscCall(MatScale(C,2.0));
  PetscCall(MatMatTransposeMult(C,C,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B));
  PetscCall(MatMatTransposeMultEqual(C,C,B,10,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"MatMatTransposeMult: B != C*B^T");

  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"C MatMatTransposeMult C:\n"));
    PetscCall(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(MatDestroy(&Cdense));
  PetscCall(PetscFree(v));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&Ct));
  PetscCall(VecDestroy(&d));
  PetscCall(PetscFinalize());
  return 0;
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
