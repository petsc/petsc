
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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view);CHKERRQ(ierr);

  /* Get local block or element size*/
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  n    = m;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(C,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-row_oriented",&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatSetOption(C,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = MatGetOwnershipIS(C,&isrows,&iscols);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-Cexp_view_ownership",&flg);CHKERRQ(ierr);
  if (flg) { /* View ownership of explicit C */
    IS tmp;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Ownership of explicit C:\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row index set:\n");CHKERRQ(ierr);
    ierr = ISOnComm(isrows,PETSC_COMM_WORLD,PETSC_USE_POINTER,&tmp);CHKERRQ(ierr);
    ierr = ISView(tmp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Column index set:\n");CHKERRQ(ierr);
    ierr = ISOnComm(iscols,PETSC_COMM_WORLD,PETSC_USE_POINTER,&tmp);CHKERRQ(ierr);
    ierr = ISView(tmp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
  }

  /* Set local matrix entries */
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows*ncols,&v);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      /*v[i*ncols+j] = (PetscReal)(rank);*/
      v[i*ncols+j] = (PetscReal)(rank*10000+100*rows[i]+cols[j]);
    }
  }
  ierr = MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test MatView() */
  if (mats_view) {
    ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Test MatMissingDiagonal() */
  ierr = MatMissingDiagonal(C,&flg,NULL);CHKERRQ(ierr);
  if (flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"MatMissingDiagonal() did not return false!\n");

  /* Set unowned matrix entries - add subdiagonals and diagonals from proc[0] */
  if (!rank) {
    PetscInt M,N,cols[2];
    ierr = MatGetSize(C,&M,&N);CHKERRQ(ierr);
    for (i=0; i<M; i++) {
      cols[0] = i;   v[0] = i + 0.5;
      cols[1] = i-1; v[1] = 0.5;
      if (i) {
        ierr = MatSetValues(C,1,&i,2,cols,v,ADD_VALUES);CHKERRQ(ierr);
      } else {
        ierr = MatSetValues(C,1,&i,1,&i,v,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test MatMult() */
  ierr = MatComputeOperator(C,MATAIJ,&Caij);CHKERRQ(ierr);
  ierr = MatMultEqual(C,Caij,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultEqual() fails");
  ierr = MatMultTransposeEqual(C,Caij,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeEqual() fails");

  /* Test MatMultAdd() and MatMultTransposeAddEqual() */
  ierr = MatMultAddEqual(C,Caij,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultAddEqual() fails");
  ierr = MatMultTransposeAddEqual(C,Caij,5,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"C != Caij. MatMultTransposeAddEqual() fails");

  /* Test MatMatMult() */
  ierr = PetscOptionsHasName(NULL,NULL,"-test_matmatmult",&Test_MatMatMult);CHKERRQ(ierr);
  if (Test_MatMatMult) {
    Mat CCelem,CCaij;
    ierr = MatMatMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CCelem);CHKERRQ(ierr);
    ierr = MatMatMult(Caij,Caij,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&CCaij);CHKERRQ(ierr);
    ierr = MatMultEqual(CCelem,CCaij,5,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"CCelem != CCaij. MatMatMult() fails");
    ierr = MatDestroy(&CCaij);CHKERRQ(ierr);
    ierr = MatDestroy(&CCelem);CHKERRQ(ierr);
  }

  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&Caij);CHKERRQ(ierr);
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
