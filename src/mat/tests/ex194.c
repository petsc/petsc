
static char help[] = "Tests MatCreateSubmatrix() with certain entire rows of matrix, modified from ex181.c.";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i,j,m = 3,n = 2,rstart,rend,cstart,cend;
  PetscMPIInt    size,rank;
  PetscScalar    v;
  IS             isrow,iscol;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  /*
        This is JUST to generate a nice test matrix, all processors fill up
    the entire matrix. This is not something one would ever do in practice.
  */
  PetscCall(MatGetOwnershipRange(C,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    for (j=0; j<m*n; j++) {
      v    = i + j + 1;
      PetscCall(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON));
  PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));

  /*
     Generate a new matrix consisting every column of the original matrix
  */
  PetscCall(MatGetOwnershipRange(C,&rstart,&rend));
  PetscCall(MatGetOwnershipRangeColumn(C,&cstart,&cend));

  PetscCall(ISCreateStride(PETSC_COMM_WORLD,rend-rstart > 0 ? rend-rstart-1 : 0,rstart,1,&isrow));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,cend-cstart,cstart,1,&iscol));
  PetscCall(MatCreateSubMatrix(C,isrow,NULL,MAT_INITIAL_MATRIX,&A));

  /* Change C to test the case MAT_REUSE_MATRIX */
  if (rank == 0) {
    i = 0; j = 0; v = 100;
    PetscCall(MatSetValues(C,1,&i,1,&j,&v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateSubMatrix(C,isrow,NULL,MAT_REUSE_MATRIX,&A));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
