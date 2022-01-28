static char help[] = "Test MatCreateSubMatrices\n\n";

#include <petscis.h>
#include <petscmat.h>

int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  Mat             A,*submats,*submats2;
  IS              *irow,*icol;
  PetscInt        i,n;
  PetscMPIInt     rank;
  PetscViewer     matfd,rowfd,colfd;
  PetscBool       same;
  char            matfile[PETSC_MAX_PATH_LEN],rowfile[PETSC_MAX_PATH_LEN],colfile[PETSC_MAX_PATH_LEN];
  char            rankstr[16]={0};

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-A",matfile,sizeof(matfile),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-row",rowfile,sizeof(rowfile),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-col",colfile,sizeof(colfile),NULL);CHKERRQ(ierr);

  /* Each rank has its own files for row/col ISes */
  ierr = PetscSNPrintf(rankstr,16,"-%d",rank);CHKERRQ(ierr);
  ierr = PetscStrlcat(rowfile,rankstr,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscStrlcat(colfile,rankstr,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,matfile,FILE_MODE_READ,&matfd);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,rowfile,FILE_MODE_READ,&rowfd);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,colfile,FILE_MODE_READ,&colfd);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,matfd);CHKERRQ(ierr);

  /* We stored the number of ISes at the beginning of rowfd */
  ierr = PetscViewerBinaryRead(rowfd,&n,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscMalloc2(n,&irow,n,&icol);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = ISCreate(PETSC_COMM_SELF,&irow[i]);CHKERRQ(ierr);
    ierr = ISCreate(PETSC_COMM_SELF,&icol[i]);CHKERRQ(ierr);
    ierr = ISLoad(irow[i],rowfd);CHKERRQ(ierr);
    ierr = ISLoad(icol[i],colfd);CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&matfd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&rowfd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&colfd);CHKERRQ(ierr);

  /* Create submats for the first time */
  ierr = MatCreateSubMatrices(A,n,irow,icol,MAT_INITIAL_MATRIX,&submats);CHKERRQ(ierr);

  /* Dup submats to submats2 for later comparison */
  ierr = PetscMalloc1(n,&submats2);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = MatDuplicate(submats[i],MAT_COPY_VALUES,&submats2[i]);CHKERRQ(ierr);
  }

  /* Create submats again */
  ierr = MatCreateSubMatrices(A,n,irow,icol,MAT_REUSE_MATRIX,&submats);CHKERRQ(ierr);

  /* Compare submats and submats2 */
  for (i=0; i<n; i++) {
    ierr = MatEqual(submats[i],submats2[i],&same);CHKERRQ(ierr);
    PetscAssertFalse(!same,PETSC_COMM_SELF,PETSC_ERR_PLIB,"submatrix %" PetscInt_FMT " is not same",i);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = ISDestroy(&irow[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&icol[i]);CHKERRQ(ierr);
  }
  ierr = MatDestroySubMatrices(n,&submats);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(n,&submats2);CHKERRQ(ierr);
  ierr = PetscFree2(irow,icol);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     suffix: 1
     nsize: 2
     requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
     args: -mat_type {{aij baij}} -A ${DATAFILESPATH}/matrices/CreateSubMatrices/A -row ${DATAFILESPATH}/matrices/CreateSubMatrices/row -col ${DATAFILESPATH}/matrices/CreateSubMatrices/col

TEST*/
