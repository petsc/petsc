
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  N    = M;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mb",&mb,NULL));
  nb   = mb;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nb",&nb,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetType(C,MATSCALAPACK));
  mloc = PETSC_DECIDE;
  CHKERRQ(PetscSplitOwnershipEqual(PETSC_COMM_WORLD,&mloc,&M));
  nloc = PETSC_DECIDE;
  CHKERRQ(PetscSplitOwnershipEqual(PETSC_COMM_WORLD,&nloc,&N));
  CHKERRQ(MatSetSizes(C,mloc,nloc,M,N));
  CHKERRQ(MatScaLAPACKSetBlockSizes(C,mb,nb));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  /*CHKERRQ(MatCreateScaLAPACK(PETSC_COMM_WORLD,mb,nb,M,N,0,0,&C)); */

  CHKERRQ(MatGetOwnershipIS(C,&isrows,&iscols));
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  CHKERRQ(PetscMalloc1(nrows*ncols,&v));
  for (i=0;i<nrows;i++) {
    for (j=0;j<ncols;j++) v[i*ncols+j] = (PetscReal)(rows[i]+1+(cols[j]+1)*0.01);
  }
  CHKERRQ(MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES));
  CHKERRQ(PetscFree(v));
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
  CHKERRQ(MatConvert(C,MATDENSE,MAT_INITIAL_MATRIX,&Cdense));
  CHKERRQ(MatMultEqual(C,Cdense,5,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Check fails: Cdense != C");

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
  if (M>N) CHKERRQ(MatCreateVecs(C,&d,NULL));
  else CHKERRQ(MatCreateVecs(C,NULL,&d));
  CHKERRQ(MatGetDiagonal(C,d));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Diagonal of C:\n"));
    CHKERRQ(VecView(d,PETSC_VIEWER_STDOUT_WORLD));
  }
  if (M>N) {
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
  CHKERRQ(MatSetType(B,MATSCALAPACK));
  CHKERRQ(MatSetSizes(B,mloc,nloc,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatScaLAPACKSetBlockSizes(B,mb,nb));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  /* CHKERRQ(MatCreateScaLAPACK(PETSC_COMM_WORLD,mb,nb,M,N,0,0,&B)); */
  CHKERRQ(MatGetOwnershipIS(B,&isrows,&iscols));
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  CHKERRQ(PetscMalloc1(nrows*ncols,&v));
  for (i=0;i<nrows;i++) {
    for (j=0;j<ncols;j++) v[i*ncols+j] = (PetscReal)(1000*rows[i]+cols[j]);
  }
  CHKERRQ(MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES));
  CHKERRQ(PetscFree(v));
  CHKERRQ(ISRestoreIndices(isrows,&rows));
  CHKERRQ(ISRestoreIndices(iscols,&cols));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"B:\n"));
    CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }
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
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Check fails: B != C*C^T");

  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"C MatMatTransposeMult C:\n"));
    CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Test MatMult() */
  CHKERRQ(MatComputeOperator(C,MATAIJ,&Caij));
  CHKERRQ(MatMultEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultEqual() fails");
  CHKERRQ(MatMultTransposeEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeEqual() fails");

  /* Test MatMultAdd() and MatMultTransposeAddEqual() */
  CHKERRQ(MatMultAddEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultAddEqual() fails");
  CHKERRQ(MatMultTransposeAddEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeAddEqual() fails");

  /* Test MatMatMult() */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-test_matmatmult",&flg));
  if (flg) {
    Mat CC,CCaij;
    CHKERRQ(MatMatMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CC));
    CHKERRQ(MatMatMult(Caij,Caij,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CCaij));
    CHKERRQ(MatMultEqual(CC,CCaij,5,&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"CC != CCaij. MatMatMult() fails");
    CHKERRQ(MatDestroy(&CCaij));
    CHKERRQ(MatDestroy(&CC));
  }

  /* Test MatCreate() on subcomm */
  color = rank%2;
  CHKERRMPI(MPI_Comm_split(PETSC_COMM_WORLD,color,0,&subcomm));
  if (color==0) {
    CHKERRQ(MatCreate(subcomm,&Asub));
    CHKERRQ(MatSetType(Asub,MATSCALAPACK));
    mloc = PETSC_DECIDE;
    CHKERRQ(PetscSplitOwnershipEqual(subcomm,&mloc,&M));
    nloc = PETSC_DECIDE;
    CHKERRQ(PetscSplitOwnershipEqual(subcomm,&nloc,&N));
    CHKERRQ(MatSetSizes(Asub,mloc,nloc,M,N));
    CHKERRQ(MatScaLAPACKSetBlockSizes(Asub,mb,nb));
    CHKERRQ(MatSetFromOptions(Asub));
    CHKERRQ(MatSetUp(Asub));
    CHKERRQ(MatDestroy(&Asub));
  }

  CHKERRQ(MatDestroy(&Cdense));
  CHKERRQ(MatDestroy(&Caij));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&Ct));
  CHKERRQ(VecDestroy(&d));
  CHKERRMPI(MPI_Comm_free(&subcomm));
  CHKERRQ(PetscFinalize());
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
