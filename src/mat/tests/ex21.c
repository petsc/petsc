
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  /* create the matrix for the five point stencil, YET AGAIN*/
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatMPIAIJSetPreallocation(C,5,NULL,5,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(C,5,NULL));
  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatGetOwnershipRange(C,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    CHKERRQ(MatGetRow(C,i,&nz,&idx,&values));
    CHKERRQ(PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"[%d] get row %" PetscInt_FMT ": ",rank,i));
    for (j=0; j<nz; j++) {
      CHKERRQ(PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"%" PetscInt_FMT " %g  ",idx[j],(double)PetscRealPart(values[j])));
    }
    CHKERRQ(PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"\n"));
    CHKERRQ(MatRestoreRow(C,i,&nz,&idx,&values));
  }
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));

  CHKERRQ(MatConvert(C,MATSAME,MAT_INITIAL_MATRIX,&A));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -mat_type seqaij

TEST*/
