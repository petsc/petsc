
static char help[] = "Tests converting a parallel AIJ formatted matrix to the parallel Row format.\n\
 This also tests MatGetRow() and MatRestoreRow() for the parallel case.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat               C,A;
  PetscInt          i,j,m = 3,n = 2,Ii,J,rstart,rend,nz;
  PetscMPIInt       rank,size;
  const PetscInt    *idx;
  PetscScalar       v;
  const PetscScalar *values;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  /* create the matrix for the five point stencil, YET AGAIN*/
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatMPIAIJSetPreallocation(C,5,NULL,5,NULL));
  PetscCall(MatSeqAIJSetPreallocation(C,5,NULL));
  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatGetOwnershipRange(C,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    PetscCall(MatGetRow(C,i,&nz,&idx,&values));
    PetscCall(PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"[%d] get row %" PetscInt_FMT ": ",rank,i));
    for (j=0; j<nz; j++) {
      PetscCall(PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"%" PetscInt_FMT " %g  ",idx[j],(double)PetscRealPart(values[j])));
    }
    PetscCall(PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"\n"));
    PetscCall(MatRestoreRow(C,i,&nz,&idx,&values));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));

  PetscCall(MatConvert(C,MATSAME,MAT_INITIAL_MATRIX,&A));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -mat_type seqaij

TEST*/
