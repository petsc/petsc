static char help[] = "Tests MatView()/MatLoad() with binary viewers for AIJ matrices.\n\n";

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

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<N; j++) {
      CHKERRQ(MatGetValue(A,i,j,&val));
      v = MakeValue(i,j,M); w = PetscRealPart(val);
      PetscCheckFalse(PetscAbsReal(v-w) > 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix entry (%" PetscInt_FMT ",%" PetscInt_FMT ") should be %g, got %g",i,j,(double)v,(double)w);
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       M = 11,N = 13;
  PetscInt       rstart,rend,i,j;
  PetscErrorCode ierr;
  PetscViewer    view;

  CHKERRQ(PetscInitialize(&argc,&args,NULL,help));
  /*
      Create a parallel AIJ matrix shared by all processors
  */
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,
                      PETSC_DECIDE,PETSC_DECIDE,
                      M,N,
                      PETSC_DECIDE,NULL,
                      PETSC_DECIDE,NULL,
                      &A);CHKERRQ(ierr);

  /*
      Set values into the matrix
  */
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<N; j++) {
      PetscReal v = MakeValue(i,j,M);
      if (PetscAbsReal(v) > 0) {
        CHKERRQ(MatSetValue(A,i,j,v,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatViewFromOptions(A,NULL,"-mat_base_view"));

  /*
      Store the binary matrix to a file
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrix.dat", FILE_MODE_WRITE, &view));
  for (i=0; i<3; i++) {
    CHKERRQ(MatView(A,view));
  }
  CHKERRQ(PetscViewerDestroy(&view));
  CHKERRQ(MatDestroy(&A));

  /*
      Now reload the matrix and check its values
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATAIJ));
  for (i=0; i<3; i++) {
    if (i > 0) CHKERRQ(MatZeroEntries(A));
    CHKERRQ(MatLoad(A,view));
    CHKERRQ(CheckValuesAIJ(A));
  }
  CHKERRQ(PetscViewerDestroy(&view));
  CHKERRQ(MatViewFromOptions(A,NULL,"-mat_load_view"));
  CHKERRQ(MatDestroy(&A));

  /*
      Reload in SEQAIJ matrix and check its values
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,"matrix.dat",FILE_MODE_READ,&view));
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetType(A,MATSEQAIJ));
  for (i=0; i<3; i++) {
    if (i > 0) CHKERRQ(MatZeroEntries(A));
    CHKERRQ(MatLoad(A,view));
    CHKERRQ(CheckValuesAIJ(A));
  }
  CHKERRQ(PetscViewerDestroy(&view));
  CHKERRQ(MatDestroy(&A));

  /*
     Reload in MPIAIJ matrix and check its values
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",FILE_MODE_READ,&view));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATMPIAIJ));
  for (i=0; i<3; i++) {
    if (i > 0) CHKERRQ(MatZeroEntries(A));
    CHKERRQ(MatLoad(A,view));
    CHKERRQ(CheckValuesAIJ(A));
  }
  CHKERRQ(PetscViewerDestroy(&view));
  CHKERRQ(MatDestroy(&A));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      args: -viewer_binary_mpiio 0
      output_file: output/ex44.out
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
        suffix: stdio_15
        nsize: 15

   testset:
      requires: mpiio
      args: -viewer_binary_mpiio 1
      output_file: output/ex44.out
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
        suffix: mpiio_15
        nsize: 15

TEST*/
