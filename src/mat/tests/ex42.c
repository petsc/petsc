
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_unsorted",&test_unsorted,NULL);CHKERRQ(ierr);

  /* Read matrix A and RHS */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Read the same matrix as a seq matrix B */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatLoad(B,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);

  /* Create the Random no generator */
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);

  /* Create the IS corresponding to subdomains */
  ierr = PetscMalloc1(nd,&is1);CHKERRQ(ierr);
  ierr = PetscMalloc1(nd,&is2);CHKERRQ(ierr);
  ierr = PetscMalloc1(m ,&idx);CHKERRQ(ierr);
  for (i = 0; i < m; i++) {idx[i] = i;CHKERRQ(ierr);}

  /* Create the random Index Sets */
  for (i=0; i<nd; i++) {
    /* Skip a few,so that the IS on different procs are diffeent*/
    for (j=0; j<rank; j++) {
      ierr = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
    }
    ierr  = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
    lsize = (PetscInt)(rand*(m/bs));
    /* shuffle */
    for (j=0; j<lsize; j++) {
      PetscInt k, swap, l;

      ierr   = PetscRandomGetValue(r,&rand);CHKERRQ(ierr);
      k      = j + (PetscInt)(rand*((m/bs)-j));
      for (l = 0; l < bs; l++) {
        swap        = idx[bs*j+l];
        idx[bs*j+l] = idx[bs*k+l];
        idx[bs*k+l] = swap;
      }
    }
    if (!test_unsorted) {ierr = PetscSortInt(lsize*bs,idx);CHKERRQ(ierr);}
    ierr = ISCreateGeneral(PETSC_COMM_SELF,lsize*bs,idx,PETSC_COPY_VALUES,is1+i);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,lsize*bs,idx,PETSC_COPY_VALUES,is2+i);CHKERRQ(ierr);
    ierr = ISSetBlockSize(is1[i],bs);CHKERRQ(ierr);
    ierr = ISSetBlockSize(is2[i],bs);CHKERRQ(ierr);
  }

  if (!test_unsorted) {
    ierr = MatIncreaseOverlap(A,nd,is1,ov);CHKERRQ(ierr);
    ierr = MatIncreaseOverlap(B,nd,is2,ov);CHKERRQ(ierr);

    for (i=0; i<nd; ++i) {
      ierr = ISSort(is1[i]);CHKERRQ(ierr);
      ierr = ISSort(is2[i]);CHKERRQ(ierr);
    }
  }

  ierr = MatCreateSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(B,nd,is2,is2,MAT_INITIAL_MATRIX,&submatB);CHKERRQ(ierr);

  /* Now see if the serial and parallel case have the same answers */
  for (i=0; i<nd; ++i) {
    ierr = MatEqual(submatA[i],submatB[i],&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"%D-th paralle submatA != seq submatB",i);
  }

  /* Free Allocated Memory */
  for (i=0; i<nd; ++i) {
    ierr = ISDestroy(&is1[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&is2[i]);CHKERRQ(ierr);
  }
  ierr = MatDestroySubMatrices(nd,&submatA);CHKERRQ(ierr);
  ierr = MatDestroySubMatrices(nd,&submatB);CHKERRQ(ierr);

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
