
static char help[] = "Tests ScaLAPACK interface.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            Cdense,Caij,B,C,Ct,Asub;
  Vec            d;
  PetscInt       i,j,M = 5,N,mb = 2,nb,nrows,ncols,mloc,nloc;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscScalar    *v;
  PetscMPIInt    rank,color;
  PetscReal      Cnorm;
  PetscBool      flg,mats_view=PETSC_FALSE;
  MPI_Comm       subcomm;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  N    = M;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mb",&mb,NULL));
  nb   = mb;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nb",&nb,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetType(C,MATSCALAPACK));
  mloc = PETSC_DECIDE;
  PetscCall(PetscSplitOwnershipEqual(PETSC_COMM_WORLD,&mloc,&M));
  nloc = PETSC_DECIDE;
  PetscCall(PetscSplitOwnershipEqual(PETSC_COMM_WORLD,&nloc,&N));
  PetscCall(MatSetSizes(C,mloc,nloc,M,N));
  PetscCall(MatScaLAPACKSetBlockSizes(C,mb,nb));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  /*PetscCall(MatCreateScaLAPACK(PETSC_COMM_WORLD,mb,nb,M,N,0,0,&C)); */

  PetscCall(MatGetOwnershipIS(C,&isrows,&iscols));
  PetscCall(ISGetLocalSize(isrows,&nrows));
  PetscCall(ISGetIndices(isrows,&rows));
  PetscCall(ISGetLocalSize(iscols,&ncols));
  PetscCall(ISGetIndices(iscols,&cols));
  PetscCall(PetscMalloc1(nrows*ncols,&v));
  for (i=0;i<nrows;i++) {
    for (j=0;j<ncols;j++) v[i*ncols+j] = (PetscReal)(rows[i]+1+(cols[j]+1)*0.01);
  }
  PetscCall(MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES));
  PetscCall(PetscFree(v));
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
  PetscCall(MatConvert(C,MATDENSE,MAT_INITIAL_MATRIX,&Cdense));
  PetscCall(MatMultEqual(C,Cdense,5,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Check fails: Cdense != C");

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
  if (M>N) PetscCall(MatCreateVecs(C,&d,NULL));
  else PetscCall(MatCreateVecs(C,NULL,&d));
  PetscCall(MatGetDiagonal(C,d));
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Diagonal of C:\n"));
    PetscCall(VecView(d,PETSC_VIEWER_STDOUT_WORLD));
  }
  if (M>N) {
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
  PetscCall(MatSetType(B,MATSCALAPACK));
  PetscCall(MatSetSizes(B,mloc,nloc,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatScaLAPACKSetBlockSizes(B,mb,nb));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  /* PetscCall(MatCreateScaLAPACK(PETSC_COMM_WORLD,mb,nb,M,N,0,0,&B)); */
  PetscCall(MatGetOwnershipIS(B,&isrows,&iscols));
  PetscCall(ISGetLocalSize(isrows,&nrows));
  PetscCall(ISGetIndices(isrows,&rows));
  PetscCall(ISGetLocalSize(iscols,&ncols));
  PetscCall(ISGetIndices(iscols,&cols));
  PetscCall(PetscMalloc1(nrows*ncols,&v));
  for (i=0;i<nrows;i++) {
    for (j=0;j<ncols;j++) v[i*ncols+j] = (PetscReal)(1000*rows[i]+cols[j]);
  }
  PetscCall(MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES));
  PetscCall(PetscFree(v));
  PetscCall(ISRestoreIndices(isrows,&rows));
  PetscCall(ISRestoreIndices(iscols,&cols));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"B:\n"));
    PetscCall(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }
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
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Check fails: B != C*C^T");

  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"C MatMatTransposeMult C:\n"));
    PetscCall(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Test MatMult() */
  PetscCall(MatComputeOperator(C,MATAIJ,&Caij));
  PetscCall(MatMultEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultEqual() fails");
  PetscCall(MatMultTransposeEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeEqual() fails");

  /* Test MatMultAdd() and MatMultTransposeAddEqual() */
  PetscCall(MatMultAddEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultAddEqual() fails");
  PetscCall(MatMultTransposeAddEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeAddEqual() fails");

  /* Test MatMatMult() */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-test_matmatmult",&flg));
  if (flg) {
    Mat CC,CCaij;
    PetscCall(MatMatMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CC));
    PetscCall(MatMatMult(Caij,Caij,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CCaij));
    PetscCall(MatMultEqual(CC,CCaij,5,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"CC != CCaij. MatMatMult() fails");
    PetscCall(MatDestroy(&CCaij));
    PetscCall(MatDestroy(&CC));
  }

  /* Test MatCreate() on subcomm */
  color = rank%2;
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD,color,0,&subcomm));
  if (color==0) {
    PetscCall(MatCreate(subcomm,&Asub));
    PetscCall(MatSetType(Asub,MATSCALAPACK));
    mloc = PETSC_DECIDE;
    PetscCall(PetscSplitOwnershipEqual(subcomm,&mloc,&M));
    nloc = PETSC_DECIDE;
    PetscCall(PetscSplitOwnershipEqual(subcomm,&nloc,&N));
    PetscCall(MatSetSizes(Asub,mloc,nloc,M,N));
    PetscCall(MatScaLAPACKSetBlockSizes(Asub,mb,nb));
    PetscCall(MatSetFromOptions(Asub));
    PetscCall(MatSetUp(Asub));
    PetscCall(MatDestroy(&Asub));
  }

  PetscCall(MatDestroy(&Cdense));
  PetscCall(MatDestroy(&Caij));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&Ct));
  PetscCall(VecDestroy(&d));
  PetscCallMPI(MPI_Comm_free(&subcomm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: scalapack

   test:
      nsize: 2
      args: -mb 5 -nb 5 -M 12 -N 10
      requires: scalapack

   test:
      suffix: 2
      nsize: 6
      args: -mb 8 -nb 6 -M 20 -N 50
      requires: scalapack
      output_file: output/ex242_1.out

   test:
      suffix: 3
      nsize: 3
      args: -mb 2 -nb 2 -M 20 -N 20 -test_matmatmult
      requires: scalapack
      output_file: output/ex242_1.out

TEST*/
