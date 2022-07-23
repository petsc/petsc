
static char help[] = "Test interface of Elemental. \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,Caij;
  PetscInt       i,j,m = 5,n,nrows,ncols;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscBool      flg,Test_MatMatMult=PETSC_FALSE,mats_view=PETSC_FALSE;
  PetscScalar    *v;
  PetscMPIInt    rank,size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view));

  /* Get local block or element size*/
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n    = m;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,m,n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(C,MATELEMENTAL));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-row_oriented",&flg));
  if (flg) PetscCall(MatSetOption(C,MAT_ROW_ORIENTED,PETSC_TRUE));
  PetscCall(MatGetOwnershipIS(C,&isrows,&iscols));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-Cexp_view_ownership",&flg));
  if (flg) { /* View ownership of explicit C */
    IS tmp;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Ownership of explicit C:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Row index set:\n"));
    PetscCall(ISOnComm(isrows,PETSC_COMM_WORLD,PETSC_USE_POINTER,&tmp));
    PetscCall(ISView(tmp,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(ISDestroy(&tmp));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Column index set:\n"));
    PetscCall(ISOnComm(iscols,PETSC_COMM_WORLD,PETSC_USE_POINTER,&tmp));
    PetscCall(ISView(tmp,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(ISDestroy(&tmp));
  }

  /* Set local matrix entries */
  PetscCall(ISGetLocalSize(isrows,&nrows));
  PetscCall(ISGetIndices(isrows,&rows));
  PetscCall(ISGetLocalSize(iscols,&ncols));
  PetscCall(ISGetIndices(iscols,&cols));
  PetscCall(PetscMalloc1(nrows*ncols,&v));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      /*v[i*ncols+j] = (PetscReal)(rank);*/
      v[i*ncols+j] = (PetscReal)(rank*10000+100*rows[i]+cols[j]);
    }
  }
  PetscCall(MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES));
  PetscCall(ISRestoreIndices(isrows,&rows));
  PetscCall(ISRestoreIndices(iscols,&cols));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Test MatView() */
  if (mats_view) PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  /* Test MatMissingDiagonal() */
  PetscCall(MatMissingDiagonal(C,&flg,NULL));
  PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"MatMissingDiagonal() did not return false!");

  /* Set unowned matrix entries - add subdiagonals and diagonals from proc[0] */
  if (rank == 0) {
    PetscInt M,N,cols[2];
    PetscCall(MatGetSize(C,&M,&N));
    for (i=0; i<M; i++) {
      cols[0] = i;   v[0] = i + 0.5;
      cols[1] = i-1; v[1] = 0.5;
      if (i) {
        PetscCall(MatSetValues(C,1,&i,2,cols,v,ADD_VALUES));
      } else {
        PetscCall(MatSetValues(C,1,&i,1,&i,v,ADD_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Test MatMult() */
  PetscCall(MatComputeOperator(C,MATAIJ,&Caij));
  PetscCall(MatMultEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultEqual() fails");
  PetscCall(MatMultTransposeEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeEqual() fails");

  /* Test MatMultAdd() and MatMultTransposeAddEqual() */
  PetscCall(MatMultAddEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultAddEqual() fails");
  PetscCall(MatMultTransposeAddEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeAddEqual() fails");

  /* Test MatMatMult() */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-test_matmatmult",&Test_MatMatMult));
  if (Test_MatMatMult) {
    Mat CCelem,CCaij;
    PetscCall(MatMatMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CCelem));
    PetscCall(MatMatMult(Caij,Caij,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CCaij));
    PetscCall(MatMultEqual(CCelem,CCaij,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"CCelem != CCaij. MatMatMult() fails");
    PetscCall(MatDestroy(&CCaij));
    PetscCall(MatDestroy(&CCelem));
  }

  PetscCall(PetscFree(v));
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&iscols));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&Caij));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      requires: elemental
      args: -mat_type elemental -m 2 -n 3

   test:
      suffix: 2
      nsize: 6
      requires: elemental
      args: -mat_type elemental -m 2 -n 2

   test:
      suffix: 3
      nsize: 6
      requires: elemental
      args: -mat_type elemental -m 2 -n 2 -test_matmatmult
      output_file: output/ex38_2.out

TEST*/
