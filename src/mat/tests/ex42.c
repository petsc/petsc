
static char help[] = "Tests MatIncreaseOverlap() and MatCreateSubmatrices() for the parallel case.\n\
This example is similar to ex40.c; here the index sets used are random.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\
  -nd <size>      : > 0  no of domains per processor \n\
  -ov <overlap>   : >=0  amount of overlap between domains\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       nd = 2,ov=1,i,j,lsize,m,n,*idx,bs;
  PetscMPIInt    rank, size;
  PetscBool      flg;
  Mat            A,B,*submatA,*submatB;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  IS             *is1,*is2;
  PetscRandom    r;
  PetscBool      test_unsorted = PETSC_FALSE;
  PetscScalar    rand;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_unsorted",&test_unsorted,NULL));

  /* Read matrix A and RHS */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* Read the same matrix as a seq matrix B */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&B));
  CHKERRQ(MatSetType(B,MATSEQAIJ));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatLoad(B,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  CHKERRQ(MatGetBlockSize(A,&bs));

  /* Create the Random no generator */
  CHKERRQ(MatGetSize(A,&m,&n));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&r));
  CHKERRQ(PetscRandomSetFromOptions(r));

  /* Create the IS corresponding to subdomains */
  CHKERRQ(PetscMalloc1(nd,&is1));
  CHKERRQ(PetscMalloc1(nd,&is2));
  CHKERRQ(PetscMalloc1(m ,&idx));
  for (i = 0; i < m; i++) {idx[i] = i;}

  /* Create the random Index Sets */
  for (i=0; i<nd; i++) {
    /* Skip a few,so that the IS on different procs are diffeent*/
    for (j=0; j<rank; j++) {
      CHKERRQ(PetscRandomGetValue(r,&rand));
    }
    CHKERRQ(PetscRandomGetValue(r,&rand));
    lsize = (PetscInt)(rand*(m/bs));
    /* shuffle */
    for (j=0; j<lsize; j++) {
      PetscInt k, swap, l;

      CHKERRQ(PetscRandomGetValue(r,&rand));
      k      = j + (PetscInt)(rand*((m/bs)-j));
      for (l = 0; l < bs; l++) {
        swap        = idx[bs*j+l];
        idx[bs*j+l] = idx[bs*k+l];
        idx[bs*k+l] = swap;
      }
    }
    if (!test_unsorted) CHKERRQ(PetscSortInt(lsize*bs,idx));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,lsize*bs,idx,PETSC_COPY_VALUES,is1+i));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,lsize*bs,idx,PETSC_COPY_VALUES,is2+i));
    CHKERRQ(ISSetBlockSize(is1[i],bs));
    CHKERRQ(ISSetBlockSize(is2[i],bs));
  }

  if (!test_unsorted) {
    CHKERRQ(MatIncreaseOverlap(A,nd,is1,ov));
    CHKERRQ(MatIncreaseOverlap(B,nd,is2,ov));

    for (i=0; i<nd; ++i) {
      CHKERRQ(ISSort(is1[i]));
      CHKERRQ(ISSort(is2[i]));
    }
  }

  CHKERRQ(MatCreateSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA));
  CHKERRQ(MatCreateSubMatrices(B,nd,is2,is2,MAT_INITIAL_MATRIX,&submatB));

  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) {
    CHKERRQ(MatEqual(submatA[i],submatB[i],&flg));
    PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"%" PetscInt_FMT "-th paralle submatA != seq submatB",i);
  }

  /* Free Allocated Memory */
  for (i=0; i<nd; ++i) {
    CHKERRQ(ISDestroy(&is1[i]));
    CHKERRQ(ISDestroy(&is2[i]));
  }
  CHKERRQ(MatDestroySubMatrices(nd,&submatA));
  CHKERRQ(MatDestroySubMatrices(nd,&submatB));

  CHKERRQ(PetscRandomDestroy(&r));
  CHKERRQ(PetscFree(is1));
  CHKERRQ(PetscFree(is2));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFree(idx));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: 3
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 5 -ov 2

   test:
      suffix: 2
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 8 -ov 2
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex

   test:
      suffix: unsorted_baij_mpi
      nsize: 3
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/cfd.1.10 -nd 8 -mat_type baij -test_unsorted

   test:
      suffix: unsorted_baij_seq
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/cfd.1.10 -nd 8 -mat_type baij -test_unsorted

   test:
      suffix: unsorted_mpi
      nsize: 3
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 8 -test_unsorted

   test:
      suffix: unsorted_seq
      requires: datafilespath double !defined(PETSC_USE_64BIT_INDICES) !complex
      args: -f ${DATAFILESPATH}/matrices/arco1 -nd 8 -test_unsorted

TEST*/
