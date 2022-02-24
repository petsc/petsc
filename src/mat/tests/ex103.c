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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Creat a matrix */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  CHKERRQ(MatSetType(A,MATDENSE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  /* Set local matrix entries */
  CHKERRQ(MatGetOwnershipIS(A,&isrows,&iscols));
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  CHKERRQ(PetscMalloc1(nrows*ncols,&v));

  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      if (size == 1) {
        v[i*ncols+j] = (PetscScalar)(i+j);
      } else {
        v[i*ncols+j] = (PetscScalar)rank+j*0.1;
      }
    }
  }
  CHKERRQ(MatSetValues(A,nrows,rows,ncols,cols,v,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  //CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%" PetscInt_FMT "] local nrows %" PetscInt_FMT ", ncols %" PetscInt_FMT "\n",rank,nrows,ncols));
  //CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  /* Test MatSetValues() by converting A to A_elemental */
  CHKERRQ(MatGetType(A,&type));
  if (size == 1) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isDense));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isAIJ));
  } else {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATMPIDENSE,&isDense));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&isAIJ));
  }

  if (isDense || isAIJ) {
    Mat Aexplicit;
    CHKERRQ(MatConvert(A, MATELEMENTAL, MAT_INITIAL_MATRIX, &A_elemental));
    CHKERRQ(MatComputeOperator(A_elemental,isAIJ ? MATAIJ : MATDENSE,&Aexplicit));
    CHKERRQ(MatMultEqual(Aexplicit,A_elemental,5,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Aexplicit != A_elemental.");
    CHKERRQ(MatDestroy(&Aexplicit));

    /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
    CHKERRQ(MatConvert(A, MATELEMENTAL, MAT_INPLACE_MATRIX, &A));
    CHKERRQ(MatMultEqual(A_elemental,A,5,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"A_elemental != A.");
    CHKERRQ(MatDestroy(&A_elemental));
  }

  CHKERRQ(ISRestoreIndices(isrows,&rows));
  CHKERRQ(ISRestoreIndices(iscols,&cols));
  CHKERRQ(ISDestroy(&isrows));
  CHKERRQ(ISDestroy(&iscols));
  CHKERRQ(PetscFree(v));
  CHKERRQ(MatDestroy(&A));
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
