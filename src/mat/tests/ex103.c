static char help[] = "Test MatSetValues() by converting MATDENSE to MATELEMENTAL. \n\
Modified from the code contributed by Yaning Liu @lbl.gov \n\n";
/*
 Example:
   mpiexec -n <np> ./ex103
   mpiexec -n <np> ./ex103 -mat_type elemental -mat_view
   mpiexec -n <np> ./ex103 -mat_type aij
*/

#include <petscmat.h>

int main(int argc, char** argv)
{
  Mat            A,A_elemental;
  PetscInt       i,j,M=10,N=5,nrows,ncols;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  IS             isrows,iscols;
  const PetscInt *rows,*cols;
  PetscScalar    *v;
  MatType        type;
  PetscBool      isDense,isAIJ,flg;

  ierr = PetscInitialize(&argc, &argv, (char*)0, help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Creat a matrix */
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /* Set local matrix entries */
  ierr = MatGetOwnershipIS(A,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows*ncols,&v);CHKERRQ(ierr);

  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      if (size == 1) {
        v[i*ncols+j] = (PetscScalar)(i+j);
      } else {
        v[i*ncols+j] = (PetscScalar)rank+j*0.1;
      }
    }
  }
  ierr = MatSetValues(A,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%" PetscInt_FMT "] local nrows %" PetscInt_FMT ", ncols %" PetscInt_FMT "\n",rank,nrows,ncols);CHKERRQ(ierr);
  //ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  /* Test MatSetValues() by converting A to A_elemental */
  ierr = MatGetType(A,&type);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isDense);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isAIJ);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&isDense);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&isAIJ);CHKERRQ(ierr);
  }

  if (isDense || isAIJ) {
    Mat Aexplicit;
    ierr = MatConvert(A, MATELEMENTAL, MAT_INITIAL_MATRIX, &A_elemental);CHKERRQ(ierr);
    ierr = MatComputeOperator(A_elemental,isAIJ ? MATAIJ : MATDENSE,&Aexplicit);CHKERRQ(ierr);
    ierr = MatMultEqual(Aexplicit,A_elemental,5,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Aexplicit != A_elemental.");
    ierr = MatDestroy(&Aexplicit);CHKERRQ(ierr);

    /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
    ierr = MatConvert(A, MATELEMENTAL, MAT_INPLACE_MATRIX, &A);CHKERRQ(ierr);
    ierr = MatMultEqual(A_elemental,A,5,&flg);CHKERRQ(ierr);
    PetscAssertFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"A_elemental != A.");
    ierr = MatDestroy(&A_elemental);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: elemental

   test:
      nsize: 6

   test:
      suffix: 2
      nsize: 6
      args: -mat_type aij
      output_file: output/ex103_1.out

   test:
      suffix: 3
      nsize: 6
      args: -mat_type elemental
      output_file: output/ex103_1.out

TEST*/
