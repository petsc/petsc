
static char help[] = "Tests MatIncreaseOverlap() - the parallel case. This example\n\
is similar to ex40.c; here the index sets used are random. Input arguments are:\n\
  -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\
  -nd <size>      : > 0  no of domains per processor \n\
  -ov <overlap>   : >=0  amount of overlap between domains\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscInt       nd = 2,ov=1,i,j,m,n,*idx,lsize;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscBool      flg;
  Mat            A,B;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  IS             *is1,*is2;
  PetscRandom    r;
  PetscScalar    rand;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL);CHKERRQ(ierr);

  /* Read matrix and RHS */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Read the matrix again as a seq matrix */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad(B,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Create the Random no generator */
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);

  /* Create the IS corresponding to subdomains */
  ierr = PetscMalloc1(nd,&is1);CHKERRQ(ierr);
  ierr = PetscMalloc1(nd,&is2);CHKERRQ(ierr);
  ierr = PetscMalloc1(m ,&idx);CHKERRQ(ierr);

  /* Create the random Index Sets */
  for (i=0; i<nd; i++) {
    for (j=0; j<rank; j++) {
      ierr = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
    }
    ierr  = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
    lsize = (PetscInt)(rand*m);
    for (j=0; j<lsize; j++) {
      ierr   = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
      idx[j] = (PetscInt)(rand*m);
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,lsize,idx,PETSC_COPY_VALUES,is1+i);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,lsize,idx,PETSC_COPY_VALUES,is2+i);CHKERRQ(ierr);
  }

  ierr = MatIncreaseOverlap(A,nd,is1,ov);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(B,nd,is2,ov);CHKERRQ(ierr);

  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) {
    PetscInt sz1,sz2;
    ierr = ISEqual(is1[i],is2[i],&flg);CHKERRQ(ierr);
    ierr = ISGetSize(is1[i],&sz1);CHKERRQ(ierr);
    ierr = ISGetSize(is2[i],&sz2);CHKERRQ(ierr);
    if (!flg) SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_PLIB,"proc:[%d], i=%" PetscInt_FMT ", flg =%d  sz1 = %" PetscInt_FMT " sz2 = %" PetscInt_FMT,rank,i,(int)flg,sz1,sz2);
  }

  /* Free Allocated Memory */
  for (i=0; i<nd; ++i) {
    ierr = ISDestroy(&is1[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&is2[i]);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = PetscFree(is1);CHKERRQ(ierr);
  ierr = PetscFree(is2);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: 3
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 3 -ov 1

TEST*/
