
static char help[] = "Tests ScaLAPACK interface.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            Cdense,Caij,B,C,Ct,Asub;
  Vec            d;
  PetscInt       i,j,M = 5,N,mb = 2,nb,nrows,ncols,mloc,nloc;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscErrorCode ierr;
  PetscScalar    *v;
  PetscMPIInt    rank,color;
  PetscReal      Cnorm;
  PetscBool      flg,mats_view=PETSC_FALSE;
  MPI_Comm       subcomm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  N    = M;
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mb",&mb,NULL);CHKERRQ(ierr);
  nb   = mb;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nb",&nb,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSCALAPACK);CHKERRQ(ierr);
  mloc = PETSC_DECIDE;
  ierr = PetscSplitOwnershipEqual(PETSC_COMM_WORLD,&mloc,&M);CHKERRQ(ierr);
  nloc = PETSC_DECIDE;
  ierr = PetscSplitOwnershipEqual(PETSC_COMM_WORLD,&nloc,&N);CHKERRQ(ierr);
  ierr = MatSetSizes(C,mloc,nloc,M,N);CHKERRQ(ierr);
  ierr = MatScaLAPACKSetBlockSizes(C,mb,nb);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  /*ierr = MatCreateScaLAPACK(PETSC_COMM_WORLD,mb,nb,M,N,0,0,&C);CHKERRQ(ierr); */

  ierr = MatGetOwnershipIS(C,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows*ncols,&v);CHKERRQ(ierr);
  for (i=0;i<nrows;i++) {
    for (j=0;j<ncols;j++) v[i*ncols+j] = (PetscReal)(rows[i]+1+(cols[j]+1)*0.01);
  }
  ierr = MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);

  /* Test MatView(), MatDuplicate() and out-of-place MatConvert() */
  ierr = MatDuplicate(C,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Duplicated C:\n");CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatConvert(C,MATDENSE,MAT_INITIAL_MATRIX,&Cdense);CHKERRQ(ierr);
  ierr = MatMultEqual(C,Cdense,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Check fails: Cdense != C");

  /* Test MatNorm() */
  ierr = MatNorm(C,NORM_1,&Cnorm);CHKERRQ(ierr);

  /* Test MatTranspose(), MatZeroEntries() and MatGetDiagonal() */
  ierr = MatTranspose(C,MAT_INITIAL_MATRIX,&Ct);CHKERRQ(ierr);
  ierr = MatConjugate(Ct);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"C's Transpose Conjugate:\n");CHKERRQ(ierr);
    ierr = MatView(Ct,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatZeroEntries(Ct);CHKERRQ(ierr);
  if (M>N) { ierr = MatCreateVecs(C,&d,NULL);CHKERRQ(ierr); }
  else { ierr = MatCreateVecs(C,NULL,&d);CHKERRQ(ierr); }
  ierr = MatGetDiagonal(C,d);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Diagonal of C:\n");CHKERRQ(ierr);
    ierr = VecView(d,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  if (M>N) {
    ierr = MatDiagonalScale(C,NULL,d);CHKERRQ(ierr);
  } else {
    ierr = MatDiagonalScale(C,d,NULL);CHKERRQ(ierr);
  }
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Diagonal Scaled C:\n");CHKERRQ(ierr);
    ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Test MatAXPY(), MatAYPX() and in-place MatConvert() */
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSCALAPACK);CHKERRQ(ierr);
  ierr = MatSetSizes(B,mloc,nloc,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatScaLAPACKSetBlockSizes(B,mb,nb);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  /* ierr = MatCreateScaLAPACK(PETSC_COMM_WORLD,mb,nb,M,N,0,0,&B);CHKERRQ(ierr); */
  ierr = MatGetOwnershipIS(B,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows*ncols,&v);CHKERRQ(ierr);
  for (i=0;i<nrows;i++) {
    for (j=0;j<ncols;j++) v[i*ncols+j] = (PetscReal)(1000*rows[i]+cols[j]);
  }
  ierr = MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"B:\n");CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatAXPY(B,2.5,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAYPX(B,3.75,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatConvert(B,MATDENSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"B after MatAXPY and MatAYPX:\n");CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  /* Test MatMatTransposeMult(): B = C*C^T */
  ierr = MatMatTransposeMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
  ierr = MatScale(C,2.0);CHKERRQ(ierr);
  ierr = MatMatTransposeMult(C,C,MAT_REUSE_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
  ierr = MatMatTransposeMultEqual(C,C,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Check fails: B != C*C^T");

  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"C MatMatTransposeMult C:\n");CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Test MatMult() */
  ierr = MatComputeOperator(C,MATAIJ,&Caij);CHKERRQ(ierr);
  ierr = MatMultEqual(C,Caij,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultEqual() fails");
  ierr = MatMultTransposeEqual(C,Caij,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeEqual() fails");

  /* Test MatMultAdd() and MatMultTransposeAddEqual() */
  ierr = MatMultAddEqual(C,Caij,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultAddEqual() fails");
  ierr = MatMultTransposeAddEqual(C,Caij,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeAddEqual() fails");

  /* Test MatMatMult() */
  ierr = PetscOptionsHasName(NULL,NULL,"-test_matmatmult",&flg);CHKERRQ(ierr);
  if (flg) {
    Mat CC,CCaij;
    ierr = MatMatMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CC);CHKERRQ(ierr);
    ierr = MatMatMult(Caij,Caij,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CCaij);CHKERRQ(ierr);
    ierr = MatMultEqual(CC,CCaij,5,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"CC != CCaij. MatMatMult() fails");
    ierr = MatDestroy(&CCaij);CHKERRQ(ierr);
    ierr = MatDestroy(&CC);CHKERRQ(ierr);
  }

  /* Test MatCreate() on subcomm */
  color = rank%2;
  ierr = MPI_Comm_split(PETSC_COMM_WORLD,color,0,&subcomm);CHKERRMPI(ierr);
  if (color==0) {
    ierr = MatCreate(subcomm,&Asub);CHKERRQ(ierr);
    ierr = MatSetType(Asub,MATSCALAPACK);CHKERRQ(ierr);
    mloc = PETSC_DECIDE;
    ierr = PetscSplitOwnershipEqual(subcomm,&mloc,&M);CHKERRQ(ierr);
    nloc = PETSC_DECIDE;
    ierr = PetscSplitOwnershipEqual(subcomm,&nloc,&N);CHKERRQ(ierr);
    ierr = MatSetSizes(Asub,mloc,nloc,M,N);CHKERRQ(ierr);
    ierr = MatScaLAPACKSetBlockSizes(Asub,mb,nb);CHKERRQ(ierr);
    ierr = MatSetFromOptions(Asub);CHKERRQ(ierr);
    ierr = MatSetUp(Asub);CHKERRQ(ierr);
    ierr = MatDestroy(&Asub);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&Cdense);CHKERRQ(ierr);
  ierr = MatDestroy(&Caij);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&Ct);CHKERRQ(ierr);
  ierr = VecDestroy(&d);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&subcomm);CHKERRMPI(ierr);
  ierr = PetscFinalize();
  return ierr;
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
