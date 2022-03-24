
static char help[] = "Tests MatIncreaseOverlap() - the parallel case. This example\n\
is similar to ex40.c; here the index sets used are random. Input arguments are:\n\
  -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\
  -nd <size>      : > 0  no of domains per processor \n\
  -ov <overlap>   : >=0  amount of overlap between domains\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscInt       nd = 2,ov=1,i,j,m,n,*idx,lsize;
  PetscMPIInt    rank;
  PetscBool      flg;
  Mat            A,B;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  IS             *is1,*is2;
  PetscRandom    r;
  PetscScalar    rand;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));

  /* Read matrix and RHS */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATMPIAIJ));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* Read the matrix again as a seq matrix */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&B));
  CHKERRQ(MatSetType(B,MATSEQAIJ));
  CHKERRQ(MatLoad(B,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* Create the Random no generator */
  CHKERRQ(MatGetSize(A,&m,&n));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&r));
  CHKERRQ(PetscRandomSetFromOptions(r));

  /* Create the IS corresponding to subdomains */
  CHKERRQ(PetscMalloc1(nd,&is1));
  CHKERRQ(PetscMalloc1(nd,&is2));
  CHKERRQ(PetscMalloc1(m ,&idx));

  /* Create the random Index Sets */
  for (i=0; i<nd; i++) {
    for (j=0; j<rank; j++) {
      CHKERRQ(PetscRandomGetValue(r,&rand));
    }
    CHKERRQ(PetscRandomGetValue(r,&rand));
    lsize = (PetscInt)(rand*m);
    for (j=0; j<lsize; j++) {
      CHKERRQ(PetscRandomGetValue(r,&rand));
      idx[j] = (PetscInt)(rand*m);
    }
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,lsize,idx,PETSC_COPY_VALUES,is1+i));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,lsize,idx,PETSC_COPY_VALUES,is2+i));
  }

  CHKERRQ(MatIncreaseOverlap(A,nd,is1,ov));
  CHKERRQ(MatIncreaseOverlap(B,nd,is2,ov));

  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) {
    PetscInt sz1,sz2;
    CHKERRQ(ISEqual(is1[i],is2[i],&flg));
    CHKERRQ(ISGetSize(is1[i],&sz1));
    CHKERRQ(ISGetSize(is2[i],&sz2));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"proc:[%d], i=%" PetscInt_FMT ", flg =%d  sz1 = %" PetscInt_FMT " sz2 = %" PetscInt_FMT,rank,i,(int)flg,sz1,sz2);
  }

  /* Free Allocated Memory */
  for (i=0; i<nd; ++i) {
    CHKERRQ(ISDestroy(&is1[i]));
    CHKERRQ(ISDestroy(&is2[i]));
  }
  CHKERRQ(PetscRandomDestroy(&r));
  CHKERRQ(PetscFree(is1));
  CHKERRQ(PetscFree(is2));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFree(idx));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: 3
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 3 -ov 1

TEST*/
