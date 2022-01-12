static char help[] = "Tests MatView()/MatLoad() with binary viewers for BAIJ matrices.\n\n";

#include <petscmat.h>
#include <petscviewer.h>

#include <petsc/private/hashtable.h>
static PetscReal MakeValue(PetscInt i,PetscInt j,PetscInt M)
{
  PetscHash_t h = PetscHashCombine(PetscHashInt(i),PetscHashInt(j));
  return (PetscReal) ((h % 5 == 0) ? (1 + i + j*M) : 0);
}

static PetscErrorCode CheckValuesAIJ(Mat A)
{
  PetscInt        M,N,rstart,rend,i,j;
  PetscReal       v,w;
  PetscScalar     val;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<N; j++) {
      ierr = MatGetValue(A,i,j,&val);CHKERRQ(ierr);
      v = MakeValue(i,j,M); w = PetscRealPart(val);
      if (PetscAbsReal(v-w) > 0) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix entry (%" PetscInt_FMT ",%" PetscInt_FMT ") should be %g, got %g",i,j,(double)v,(double)w);
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       M = 24,N = 48,bs = 2;
  PetscInt       rstart,rend,i,j;
  PetscErrorCode ierr;
  PetscViewer    view;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  /*
      Create a parallel BAIJ matrix shared by all processors
  */
  ierr = MatCreateBAIJ(PETSC_COMM_WORLD,
                       bs,
                       PETSC_DECIDE,PETSC_DECIDE,
                       M,N,
                       PETSC_DECIDE,NULL,
                       PETSC_DECIDE,NULL,
                       &A);CHKERRQ(ierr);

  /*
      Set values into the matrix
  */
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<N; j++) {
      PetscReal v = MakeValue(i,j,M);
      if (PetscAbsReal(v) > 0) {
        ierr = MatSetValue(A,i,j,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-mat_base_view");CHKERRQ(ierr);

  /*
      Store the binary matrix to a file
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrix.dat", FILE_MODE_WRITE, &view);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    ierr = MatView(A,view);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /*
      Now reload the matrix and check its values
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATBAIJ);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    if (i > 0) {ierr = MatZeroEntries(A);CHKERRQ(ierr);}
    ierr = MatLoad(A,view);CHKERRQ(ierr);
    ierr = CheckValuesAIJ(A);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-mat_load_view");CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /*
      Reload in SEQBAIJ matrix and check its values
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"matrix.dat",FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQBAIJ);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    if (i > 0) {ierr = MatZeroEntries(A);CHKERRQ(ierr);}
    ierr = MatLoad(A,view);CHKERRQ(ierr);
    ierr = CheckValuesAIJ(A);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /*
     Reload in MPIBAIJ matrix and check its values
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIBAIJ);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    if (i > 0) {ierr = MatZeroEntries(A);CHKERRQ(ierr);}
    ierr = MatLoad(A,view);CHKERRQ(ierr);
    ierr = CheckValuesAIJ(A);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -viewer_binary_mpiio 0
      output_file: output/ex45.out
      test:
        suffix: stdio_1
        nsize: 1
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
        nsize: 4

   testset:
      requires: mpiio
      args: -viewer_binary_mpiio 1
      output_file: output/ex45.out
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

TEST*/
