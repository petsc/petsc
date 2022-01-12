static char help[] = "Tests MatDenseGetArray() and MatView()/MatLoad() with binary viewers.\n\n";

#include <petscmat.h>
#include <petscviewer.h>

static PetscErrorCode CheckValues(Mat A,PetscBool one)
{
  const PetscScalar *array;
  PetscInt          M,N,rstart,rend,lda,i,j;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatDenseGetArrayRead(A,&array);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(A,&lda);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<N; j++) {
      PetscInt ii = i - rstart, jj = j;
      PetscReal v = (PetscReal)(one ? 1 : (1 + i + j*M));
      PetscReal w = PetscRealPart(array[ii + jj*lda]);
      if (PetscAbsReal(v-w) > 0) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix entry (%" PetscInt_FMT ",%" PetscInt_FMT ") should be %g, got %g",i,j,(double)v,(double)w);
    }
  }
  ierr = MatDenseRestoreArrayRead(A,&array);CHKERRQ(ierr);
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
  ierr = PetscStrcpy(mattype,MATMPIDENSE);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-mat_type",mattype,sizeof(mattype),NULL);CHKERRQ(ierr);
  /*
      Create a parallel dense matrix shared by all processors
  */
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,NULL,&A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatConvert(A,mattype,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  /*
     Set values into the matrix
  */
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      PetscScalar v = (PetscReal)(1 + i + j*M);
      ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatScale(A,2.0);CHKERRQ(ierr);
  ierr = MatScale(A,1.0/2.0);CHKERRQ(ierr);

  /*
      Store the binary matrix to a file
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrix.dat", FILE_MODE_WRITE, &view);CHKERRQ(ierr);
  for (i=0; i<2; i++) {
    ierr = MatView(A,view);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(view,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
    ierr = MatView(A,view);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(view);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /*
      Now reload the matrix and check its values
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,mattype);CHKERRQ(ierr);
  for (i=0; i<4; i++) {
    if (i > 0) {ierr = MatZeroEntries(A);CHKERRQ(ierr);}
    ierr = MatLoad(A,view);CHKERRQ(ierr);
    ierr = CheckValuesIJ(A);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscMalloc1((rend-rstart)*N,&array);CHKERRQ(ierr);
  for (i=0; i<(rend-rstart)*N; i++) array[i] = (PetscReal)1;
  ierr = MatDensePlaceArray(A,array);CHKERRQ(ierr);
  ierr = MatScale(A,2.0);CHKERRQ(ierr);
  ierr = MatScale(A,1.0/2.0);CHKERRQ(ierr);
  ierr = CheckValuesOne(A);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_WRITE,&view);CHKERRQ(ierr);
  ierr = MatView(A,view);CHKERRQ(ierr);
  ierr = MatDenseResetArray(A);CHKERRQ(ierr);
  ierr = PetscFree(array);CHKERRQ(ierr);
  ierr = CheckValuesIJ(A);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipHeader(view,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatView(A,view);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipHeader(view,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,mattype);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatLoad(A,view);CHKERRQ(ierr);
  ierr = CheckValuesOne(A);CHKERRQ(ierr);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipHeader(view,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatLoad(A,view);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetSkipHeader(view,PETSC_FALSE);CHKERRQ(ierr);
  ierr = CheckValuesIJ(A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  {
    PetscInt m = PETSC_DECIDE, n = PETSC_DECIDE;
    ierr = PetscSplitOwnership(PETSC_COMM_WORLD,&m,&M);CHKERRQ(ierr);
    ierr = PetscSplitOwnership(PETSC_COMM_WORLD,&n,&N);CHKERRQ(ierr);
    /* TODO: MatCreateDense requires data!=NULL at all processes! */
    ierr = PetscMalloc1(m*N+1,&array);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_WORLD,m,n,M,N,array,&A);CHKERRQ(ierr);
    ierr = MatLoad(A,view);CHKERRQ(ierr);
    ierr = CheckValuesOne(A);CHKERRQ(ierr);
    ierr = PetscViewerBinarySetSkipHeader(view,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatLoad(A,view);CHKERRQ(ierr);
    ierr = PetscViewerBinarySetSkipHeader(view,PETSC_FALSE);CHKERRQ(ierr);
    ierr = CheckValuesIJ(A);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);

    ierr = MatCreateDense(PETSC_COMM_WORLD,m,n,M,N,array,&A);CHKERRQ(ierr);
    ierr = CheckValuesIJ(A);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);

    ierr = PetscFree(array);CHKERRQ(ierr);
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
