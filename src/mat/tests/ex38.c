
static char help[] = "Test interface of Elemental. \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,Caij;
  PetscInt       i,j,m = 5,n,nrows,ncols;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscErrorCode ierr;
  PetscBool      flg,Test_MatMatMult=PETSC_FALSE,mats_view=PETSC_FALSE;
  PetscScalar    *v;
  PetscMPIInt    rank,size;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view));

  /* Get local block or element size*/
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n    = m;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,m,n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(C,MATELEMENTAL));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-row_oriented",&flg));
  if (flg) CHKERRQ(MatSetOption(C,MAT_ROW_ORIENTED,PETSC_TRUE));
  CHKERRQ(MatGetOwnershipIS(C,&isrows,&iscols));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-Cexp_view_ownership",&flg));
  if (flg) { /* View ownership of explicit C */
    IS tmp;
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Ownership of explicit C:\n"));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Row index set:\n"));
    CHKERRQ(ISOnComm(isrows,PETSC_COMM_WORLD,PETSC_USE_POINTER,&tmp));
    CHKERRQ(ISView(tmp,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(ISDestroy(&tmp));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Column index set:\n"));
    CHKERRQ(ISOnComm(iscols,PETSC_COMM_WORLD,PETSC_USE_POINTER,&tmp));
    CHKERRQ(ISView(tmp,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(ISDestroy(&tmp));
  }

  /* Set local matrix entries */
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  CHKERRQ(PetscMalloc1(nrows*ncols,&v));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      /*v[i*ncols+j] = (PetscReal)(rank);*/
      v[i*ncols+j] = (PetscReal)(rank*10000+100*rows[i]+cols[j]);
    }
  }
  CHKERRQ(MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES));
  CHKERRQ(ISRestoreIndices(isrows,&rows));
  CHKERRQ(ISRestoreIndices(iscols,&cols));
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Test MatView() */
  if (mats_view) {
    CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Test MatMissingDiagonal() */
  CHKERRQ(MatMissingDiagonal(C,&flg,NULL));
  PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"MatMissingDiagonal() did not return false!");

  /* Set unowned matrix entries - add subdiagonals and diagonals from proc[0] */
  if (rank == 0) {
    PetscInt M,N,cols[2];
    CHKERRQ(MatGetSize(C,&M,&N));
    for (i=0; i<M; i++) {
      cols[0] = i;   v[0] = i + 0.5;
      cols[1] = i-1; v[1] = 0.5;
      if (i) {
        CHKERRQ(MatSetValues(C,1,&i,2,cols,v,ADD_VALUES));
      } else {
        CHKERRQ(MatSetValues(C,1,&i,1,&i,v,ADD_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Test MatMult() */
  CHKERRQ(MatComputeOperator(C,MATAIJ,&Caij));
  CHKERRQ(MatMultEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultEqual() fails");
  CHKERRQ(MatMultTransposeEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeEqual() fails");

  /* Test MatMultAdd() and MatMultTransposeAddEqual() */
  CHKERRQ(MatMultAddEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultAddEqual() fails");
  CHKERRQ(MatMultTransposeAddEqual(C,Caij,5,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeAddEqual() fails");

  /* Test MatMatMult() */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-test_matmatmult",&Test_MatMatMult));
  if (Test_MatMatMult) {
    Mat CCelem,CCaij;
    CHKERRQ(MatMatMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CCelem));
    CHKERRQ(MatMatMult(Caij,Caij,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CCaij));
    CHKERRQ(MatMultEqual(CCelem,CCaij,5,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"CCelem != CCaij. MatMatMult() fails");
    CHKERRQ(MatDestroy(&CCaij));
    CHKERRQ(MatDestroy(&CCelem));
  }

  CHKERRQ(PetscFree(v));
  CHKERRQ(ISDestroy(&isrows));
  CHKERRQ(ISDestroy(&iscols));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&Caij));
  ierr = PetscFinalize();
  return ierr;
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
