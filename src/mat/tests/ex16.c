static char help[] = "Tests MatDenseGetArray() and MatView()/MatLoad() with binary viewers.\n\n";

#include <petscmat.h>
#include <petscviewer.h>

static PetscErrorCode CheckValues(Mat A,PetscBool one)
{
  const PetscScalar *array;
  PetscInt          M,N,rstart,rend,lda,i,j;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArrayRead(A,&array));
  CHKERRQ(MatDenseGetLDA(A,&lda));
  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<N; j++) {
      PetscInt ii = i - rstart, jj = j;
      PetscReal v = (PetscReal)(one ? 1 : (1 + i + j*M));
      PetscReal w = PetscRealPart(array[ii + jj*lda]);
      PetscCheckFalse(PetscAbsReal(v-w) > 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix entry (%" PetscInt_FMT ",%" PetscInt_FMT ") should be %g, got %g",i,j,(double)v,(double)w);
    }
  }
  CHKERRQ(MatDenseRestoreArrayRead(A,&array));
  PetscFunctionReturn(0);
}

#define CheckValuesIJ(A)  CheckValues(A,PETSC_FALSE)
#define CheckValuesOne(A) CheckValues(A,PETSC_TRUE)

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       i,j,M = 4,N = 3,rstart,rend;
  PetscErrorCode ierr;
  PetscScalar    *array;
  char           mattype[256];
  PetscViewer    view;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscStrcpy(mattype,MATMPIDENSE));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-mat_type",mattype,sizeof(mattype),NULL));
  /*
      Create a parallel dense matrix shared by all processors
  */
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,NULL,&A));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatConvert(A,mattype,MAT_INPLACE_MATRIX,&A));
  /*
     Set values into the matrix
  */
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      PetscScalar v = (PetscReal)(1 + i + j*M);
      CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatScale(A,2.0));
  CHKERRQ(MatScale(A,1.0/2.0));

  /*
      Store the binary matrix to a file
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrix.dat", FILE_MODE_WRITE, &view));
  for (i=0; i<2; i++) {
    CHKERRQ(MatView(A,view));
    CHKERRQ(PetscViewerPushFormat(view,PETSC_VIEWER_NATIVE));
    CHKERRQ(MatView(A,view));
    CHKERRQ(PetscViewerPopFormat(view));
  }
  CHKERRQ(PetscViewerDestroy(&view));
  CHKERRQ(MatDestroy(&A));

  /*
      Now reload the matrix and check its values
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,mattype));
  for (i=0; i<4; i++) {
    if (i > 0) CHKERRQ(MatZeroEntries(A));
    CHKERRQ(MatLoad(A,view));
    CHKERRQ(CheckValuesIJ(A));
  }
  CHKERRQ(PetscViewerDestroy(&view));

  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  CHKERRQ(PetscMalloc1((rend-rstart)*N,&array));
  for (i=0; i<(rend-rstart)*N; i++) array[i] = (PetscReal)1;
  CHKERRQ(MatDensePlaceArray(A,array));
  CHKERRQ(MatScale(A,2.0));
  CHKERRQ(MatScale(A,1.0/2.0));
  CHKERRQ(CheckValuesOne(A));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_WRITE,&view));
  CHKERRQ(MatView(A,view));
  CHKERRQ(MatDenseResetArray(A));
  CHKERRQ(PetscFree(array));
  CHKERRQ(CheckValuesIJ(A));
  CHKERRQ(PetscViewerBinarySetSkipHeader(view,PETSC_TRUE));
  CHKERRQ(MatView(A,view));
  CHKERRQ(PetscViewerBinarySetSkipHeader(view,PETSC_FALSE));
  CHKERRQ(PetscViewerDestroy(&view));
  CHKERRQ(MatDestroy(&A));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,mattype));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view));
  CHKERRQ(MatLoad(A,view));
  CHKERRQ(CheckValuesOne(A));
  CHKERRQ(MatZeroEntries(A));
  CHKERRQ(PetscViewerBinarySetSkipHeader(view,PETSC_TRUE));
  CHKERRQ(MatLoad(A,view));
  CHKERRQ(PetscViewerBinarySetSkipHeader(view,PETSC_FALSE));
  CHKERRQ(CheckValuesIJ(A));
  CHKERRQ(PetscViewerDestroy(&view));
  CHKERRQ(MatDestroy(&A));

  {
    PetscInt m = PETSC_DECIDE, n = PETSC_DECIDE;
    CHKERRQ(PetscSplitOwnership(PETSC_COMM_WORLD,&m,&M));
    CHKERRQ(PetscSplitOwnership(PETSC_COMM_WORLD,&n,&N));
    /* TODO: MatCreateDense requires data!=NULL at all processes! */
    CHKERRQ(PetscMalloc1(m*N+1,&array));

    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view));
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,m,n,M,N,array,&A));
    CHKERRQ(MatLoad(A,view));
    CHKERRQ(CheckValuesOne(A));
    CHKERRQ(PetscViewerBinarySetSkipHeader(view,PETSC_TRUE));
    CHKERRQ(MatLoad(A,view));
    CHKERRQ(PetscViewerBinarySetSkipHeader(view,PETSC_FALSE));
    CHKERRQ(CheckValuesIJ(A));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(PetscViewerDestroy(&view));

    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,m,n,M,N,array,&A));
    CHKERRQ(CheckValuesIJ(A));
    CHKERRQ(MatDestroy(&A));

    CHKERRQ(PetscFree(array));
  }

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -viewer_binary_mpiio 0
      output_file: output/ex16.out
      test:
        suffix: stdio_1
        nsize: 1
        args: -mat_type seqdense
      test:
        suffix: stdio_2
        nsize: 2
      test:
        suffix: stdio_3
        nsize: 3
      test:
        suffix: stdio_4
        nsize: 4
      test:
        suffix: stdio_5
        nsize: 5
      test:
        requires: cuda
        args: -mat_type seqdensecuda
        suffix: stdio_cuda_1
        nsize: 1
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: stdio_cuda_2
        nsize: 2
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: stdio_cuda_3
        nsize: 3
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: stdio_cuda_4
        nsize: 4
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: stdio_cuda_5
        nsize: 5

   testset:
      requires: mpiio
      args: -viewer_binary_mpiio 1
      output_file: output/ex16.out
      test:
        suffix: mpiio_1
        nsize: 1
      test:
        suffix: mpiio_2
        nsize: 2
      test:
        suffix: mpiio_3
        nsize: 3
      test:
        suffix: mpiio_4
        nsize: 4
      test:
        suffix: mpiio_5
        nsize: 5
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: mpiio_cuda_1
        nsize: 1
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: mpiio_cuda_2
        nsize: 2
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: mpiio_cuda_3
        nsize: 3
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: mpiio_cuda_4
        nsize: 4
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: mpiio_cuda_5
        nsize: 5

TEST*/
