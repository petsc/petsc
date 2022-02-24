static char help[] = "Test conversion of ScaLAPACK matrices.\n\n";

#include <petscmat.h>

int main(int argc, char** argv)
{
  Mat            A,A_scalapack;
  PetscInt       i,j,M=10,N=5,nloc,mloc,nrows,ncols;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  IS             isrows,iscols;
  const PetscInt *rows,*cols;
  PetscScalar    *v;
  MatType        type;
  PetscBool      isDense,isAIJ,flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));

  /* Create a matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  mloc = PETSC_DECIDE;
  CHKERRQ(PetscSplitOwnershipEqual(PETSC_COMM_WORLD,&mloc,&M));
  nloc = PETSC_DECIDE;
  CHKERRQ(PetscSplitOwnershipEqual(PETSC_COMM_WORLD,&nloc,&N));
  CHKERRQ(MatSetSizes(A,mloc,nloc,M,N));
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
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Test MatSetValues() by converting A to A_scalapack */
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
    CHKERRQ(MatConvert(A,MATSCALAPACK,MAT_INITIAL_MATRIX,&A_scalapack));
    CHKERRQ(MatComputeOperator(A_scalapack,isAIJ?MATAIJ:MATDENSE,&Aexplicit));
    CHKERRQ(MatMultEqual(Aexplicit,A_scalapack,5,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Aexplicit != A_scalapack.");
    CHKERRQ(MatDestroy(&Aexplicit));

    /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
    CHKERRQ(MatConvert(A,MATSCALAPACK,MAT_INPLACE_MATRIX,&A));
    CHKERRQ(MatMultEqual(A_scalapack,A,5,&flg));
    PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"A_scalapack != A.");
    CHKERRQ(MatDestroy(&A_scalapack));
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
      requires: scalapack

   test:
      nsize: 6

   test:
      suffix: 2
      nsize: 6
      args: -mat_type aij
      output_file: output/ex243_1.out

   test:
      suffix: 3
      nsize: 6
      args: -mat_type scalapack
      output_file: output/ex243_1.out

TEST*/
