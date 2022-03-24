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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-A",matfile,sizeof(matfile),NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-row",rowfile,sizeof(rowfile),NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-col",colfile,sizeof(colfile),NULL));

  /* Each rank has its own files for row/col ISes */
  CHKERRQ(PetscSNPrintf(rankstr,16,"-%d",rank));
  CHKERRQ(PetscStrlcat(rowfile,rankstr,PETSC_MAX_PATH_LEN));
  CHKERRQ(PetscStrlcat(colfile,rankstr,PETSC_MAX_PATH_LEN));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,matfile,FILE_MODE_READ,&matfd));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,rowfile,FILE_MODE_READ,&rowfd));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,colfile,FILE_MODE_READ,&colfd));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,matfd));

  /* We stored the number of ISes at the beginning of rowfd */
  CHKERRQ(PetscViewerBinaryRead(rowfd,&n,1,NULL,PETSC_INT));
  CHKERRQ(PetscMalloc2(n,&irow,n,&icol));
  for (i=0; i<n; i++) {
    CHKERRQ(ISCreate(PETSC_COMM_SELF,&irow[i]));
    CHKERRQ(ISCreate(PETSC_COMM_SELF,&icol[i]));
    CHKERRQ(ISLoad(irow[i],rowfd));
    CHKERRQ(ISLoad(icol[i],colfd));
  }

  CHKERRQ(PetscViewerDestroy(&matfd));
  CHKERRQ(PetscViewerDestroy(&rowfd));
  CHKERRQ(PetscViewerDestroy(&colfd));

  /* Create submats for the first time */
  CHKERRQ(MatCreateSubMatrices(A,n,irow,icol,MAT_INITIAL_MATRIX,&submats));

  /* Dup submats to submats2 for later comparison */
  CHKERRQ(PetscMalloc1(n,&submats2));
  for (i=0; i<n; i++) {
    CHKERRQ(MatDuplicate(submats[i],MAT_COPY_VALUES,&submats2[i]));
  }

  /* Create submats again */
  CHKERRQ(MatCreateSubMatrices(A,n,irow,icol,MAT_REUSE_MATRIX,&submats));

  /* Compare submats and submats2 */
  for (i=0; i<n; i++) {
    CHKERRQ(MatEqual(submats[i],submats2[i],&same));
    PetscCheck(same,PETSC_COMM_SELF,PETSC_ERR_PLIB,"submatrix %" PetscInt_FMT " is not same",i);
  }

  CHKERRQ(MatDestroy(&A));
  for (i=0; i<n; i++) {
    CHKERRQ(ISDestroy(&irow[i]));
    CHKERRQ(ISDestroy(&icol[i]));
  }
  CHKERRQ(MatDestroySubMatrices(n,&submats));
  CHKERRQ(MatDestroyMatrices(n,&submats2));
  CHKERRQ(PetscFree2(irow,icol));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 1
     nsize: 2
     requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
     args: -mat_type {{aij baij}} -A ${DATAFILESPATH}/matrices/CreateSubMatrices/A -row ${DATAFILESPATH}/matrices/CreateSubMatrices/row -col ${DATAFILESPATH}/matrices/CreateSubMatrices/col

TEST*/
