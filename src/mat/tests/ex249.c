static char help[] = "Test MatCreateSubMatrices\n\n";

#include <petscis.h>
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat             A,*submats,*submats2;
  IS              *irow,*icol;
  PetscInt        i,n;
  PetscMPIInt     rank;
  PetscViewer     matfd,rowfd,colfd;
  PetscBool       same;
  char            matfile[PETSC_MAX_PATH_LEN],rowfile[PETSC_MAX_PATH_LEN],colfile[PETSC_MAX_PATH_LEN];
  char            rankstr[16]={0};

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscOptionsGetString(NULL,NULL,"-A",matfile,sizeof(matfile),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-row",rowfile,sizeof(rowfile),NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-col",colfile,sizeof(colfile),NULL));

  /* Each rank has its own files for row/col ISes */
  PetscCall(PetscSNPrintf(rankstr,16,"-%d",rank));
  PetscCall(PetscStrlcat(rowfile,rankstr,PETSC_MAX_PATH_LEN));
  PetscCall(PetscStrlcat(colfile,rankstr,PETSC_MAX_PATH_LEN));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,matfile,FILE_MODE_READ,&matfd));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF,rowfile,FILE_MODE_READ,&rowfd));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF,colfile,FILE_MODE_READ,&colfd));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,matfd));

  /* We stored the number of ISes at the beginning of rowfd */
  PetscCall(PetscViewerBinaryRead(rowfd,&n,1,NULL,PETSC_INT));
  PetscCall(PetscMalloc2(n,&irow,n,&icol));
  for (i=0; i<n; i++) {
    PetscCall(ISCreate(PETSC_COMM_SELF,&irow[i]));
    PetscCall(ISCreate(PETSC_COMM_SELF,&icol[i]));
    PetscCall(ISLoad(irow[i],rowfd));
    PetscCall(ISLoad(icol[i],colfd));
  }

  PetscCall(PetscViewerDestroy(&matfd));
  PetscCall(PetscViewerDestroy(&rowfd));
  PetscCall(PetscViewerDestroy(&colfd));

  /* Create submats for the first time */
  PetscCall(MatCreateSubMatrices(A,n,irow,icol,MAT_INITIAL_MATRIX,&submats));

  /* Dup submats to submats2 for later comparison */
  PetscCall(PetscMalloc1(n,&submats2));
  for (i=0; i<n; i++) {
    PetscCall(MatDuplicate(submats[i],MAT_COPY_VALUES,&submats2[i]));
  }

  /* Create submats again */
  PetscCall(MatCreateSubMatrices(A,n,irow,icol,MAT_REUSE_MATRIX,&submats));

  /* Compare submats and submats2 */
  for (i=0; i<n; i++) {
    PetscCall(MatEqual(submats[i],submats2[i],&same));
    PetscCheck(same,PETSC_COMM_SELF,PETSC_ERR_PLIB,"submatrix %" PetscInt_FMT " is not same",i);
  }

  PetscCall(MatDestroy(&A));
  for (i=0; i<n; i++) {
    PetscCall(ISDestroy(&irow[i]));
    PetscCall(ISDestroy(&icol[i]));
  }
  PetscCall(MatDestroySubMatrices(n,&submats));
  PetscCall(MatDestroyMatrices(n,&submats2));
  PetscCall(PetscFree2(irow,icol));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 1
     nsize: 2
     requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
     args: -mat_type {{aij baij}} -A ${DATAFILESPATH}/matrices/CreateSubMatrices/A -row ${DATAFILESPATH}/matrices/CreateSubMatrices/row -col ${DATAFILESPATH}/matrices/CreateSubMatrices/col

TEST*/
