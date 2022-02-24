
static char help[] = "Tests various routines in MatMPIBAIJ format.\n";

#include <petscmat.h>
#define IMAX 15
int main(int argc,char **args)
{
  Mat               A,B,C,At,Bt;
  PetscViewer       fd;
  char              file[PETSC_MAX_PATH_LEN];
  PetscRandom       rand;
  Vec               xx,yy,s1,s2;
  PetscReal         s1norm,s2norm,rnorm,tol=1.e-10;
  PetscInt          rstart,rend,rows[2],cols[2],m,n,i,j,M,N,ct,row,ncols1,ncols2,bs;
  PetscMPIInt       rank,size;
  PetscErrorCode    ierr = 0;
  const PetscInt    *cols1,*cols2;
  PetscScalar       vals1[4],vals2[4],v;
  const PetscScalar *v1,*v2;
  PetscBool         flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Check out if MatLoad() works */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Input file not specified");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATBAIJ));
  CHKERRQ(MatLoad(A,fd));

  CHKERRQ(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B));
  CHKERRQ(PetscViewerDestroy(&fd));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&xx));
  CHKERRQ(VecSetSizes(xx,m,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(xx));
  CHKERRQ(VecDuplicate(xx,&s1));
  CHKERRQ(VecDuplicate(xx,&s2));
  CHKERRQ(VecDuplicate(xx,&yy));
  CHKERRQ(MatGetBlockSize(A,&bs));

  /* Test MatNorm() */
  CHKERRQ(MatNorm(A,NORM_FROBENIUS,&s1norm));
  CHKERRQ(MatNorm(B,NORM_FROBENIUS,&s2norm));
  rnorm = PetscAbsScalar(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatNorm_FROBENIUS()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs));
  }
  CHKERRQ(MatNorm(A,NORM_INFINITY,&s1norm));
  CHKERRQ(MatNorm(B,NORM_INFINITY,&s2norm));
  rnorm = PetscAbsScalar(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatNorm_INFINITY()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs));
  }
  CHKERRQ(MatNorm(A,NORM_1,&s1norm));
  CHKERRQ(MatNorm(B,NORM_1,&s2norm));
  rnorm = PetscAbsScalar(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatNorm_NORM_1()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs));
  }

  /* Test MatMult() */
  for (i=0; i<IMAX; i++) {
    CHKERRQ(VecSetRandom(xx,rand));
    CHKERRQ(MatMult(A,xx,s1));
    CHKERRQ(MatMult(B,xx,s2));
    CHKERRQ(VecAXPY(s2,-1.0,s1));
    CHKERRQ(VecNorm(s2,NORM_2,&rnorm));
    if (rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatMult - Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)rnorm,bs));
    }
  }

  /* test MatMultAdd() */
  for (i=0; i<IMAX; i++) {
    CHKERRQ(VecSetRandom(xx,rand));
    CHKERRQ(VecSetRandom(yy,rand));
    CHKERRQ(MatMultAdd(A,xx,yy,s1));
    CHKERRQ(MatMultAdd(B,xx,yy,s2));
    CHKERRQ(VecAXPY(s2,-1.0,s1));
    CHKERRQ(VecNorm(s2,NORM_2,&rnorm));
    if (rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatMultAdd - Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)rnorm,bs));
    }
  }

  /* Test MatMultTranspose() */
  for (i=0; i<IMAX; i++) {
    CHKERRQ(VecSetRandom(xx,rand));
    CHKERRQ(MatMultTranspose(A,xx,s1));
    CHKERRQ(MatMultTranspose(B,xx,s2));
    CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
    CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatMultTranspose - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs));
    }
  }
  /* Test MatMultTransposeAdd() */
  for (i=0; i<IMAX; i++) {
    CHKERRQ(VecSetRandom(xx,rand));
    CHKERRQ(VecSetRandom(yy,rand));
    CHKERRQ(MatMultTransposeAdd(A,xx,yy,s1));
    CHKERRQ(MatMultTransposeAdd(B,xx,yy,s2));
    CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
    CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatMultTransposeAdd - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs));
    }
  }

  /* Check MatGetValues() */
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  CHKERRQ(MatGetSize(A,&M,&N));

  for (i=0; i<IMAX; i++) {
    /* Create random row numbers ad col numbers */
    CHKERRQ(PetscRandomGetValue(rand,&v));
    cols[0] = (int)(PetscRealPart(v)*N);
    CHKERRQ(PetscRandomGetValue(rand,&v));
    cols[1] = (int)(PetscRealPart(v)*N);
    CHKERRQ(PetscRandomGetValue(rand,&v));
    rows[0] = rstart + (int)(PetscRealPart(v)*m);
    CHKERRQ(PetscRandomGetValue(rand,&v));
    rows[1] = rstart + (int)(PetscRealPart(v)*m);

    CHKERRQ(MatGetValues(A,2,rows,2,cols,vals1));
    CHKERRQ(MatGetValues(B,2,rows,2,cols,vals2));

    for (j=0; j<4; j++) {
      if (vals1[j] != vals2[j]) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d]: Error: MatGetValues rstart = %2" PetscInt_FMT "  row = %2" PetscInt_FMT " col = %2" PetscInt_FMT " val1 = %e val2 = %e bs = %" PetscInt_FMT "\n",rank,rstart,rows[j/2],cols[j%2],(double)PetscRealPart(vals1[j]),(double)PetscRealPart(vals2[j]),bs));
      }
    }
  }

  /* Test MatGetRow()/ MatRestoreRow() */
  for (ct=0; ct<100; ct++) {
    CHKERRQ(PetscRandomGetValue(rand,&v));
    row  = rstart + (PetscInt)(PetscRealPart(v)*m);
    CHKERRQ(MatGetRow(A,row,&ncols1,&cols1,&v1));
    CHKERRQ(MatGetRow(B,row,&ncols2,&cols2,&v2));

    for (i=0,j=0; i<ncols1 && j<ncols2; j++) {
      while (cols2[j] != cols1[i]) i++;
      PetscCheckFalse(v1[i] != v2[j],PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetRow() failed - vals incorrect.");
    }
    PetscCheckFalse(j<ncols2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetRow() failed - cols incorrect");

    CHKERRQ(MatRestoreRow(A,row,&ncols1,&cols1,&v1));
    CHKERRQ(MatRestoreRow(B,row,&ncols2,&cols2,&v2));
  }

  /* Test MatConvert() */
  CHKERRQ(MatConvert(A,MATSAME,MAT_INITIAL_MATRIX,&C));

  /* See if MatMult Says both are same */
  for (i=0; i<IMAX; i++) {
    CHKERRQ(VecSetRandom(xx,rand));
    CHKERRQ(MatMult(A,xx,s1));
    CHKERRQ(MatMult(C,xx,s2));
    CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
    CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Error in MatConvert: MatMult - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs));
    }
  }
  CHKERRQ(MatDestroy(&C));

  /* Test MatTranspose() */
  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&At));
  CHKERRQ(MatTranspose(B,MAT_INITIAL_MATRIX,&Bt));
  for (i=0; i<IMAX; i++) {
    CHKERRQ(VecSetRandom(xx,rand));
    CHKERRQ(MatMult(At,xx,s1));
    CHKERRQ(MatMult(Bt,xx,s2));
    CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
    CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Error in MatConvert:MatMult - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs));
    }
  }
  CHKERRQ(MatDestroy(&At));
  CHKERRQ(MatDestroy(&Bt));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(VecDestroy(&xx));
  CHKERRQ(VecDestroy(&yy));
  CHKERRQ(VecDestroy(&s1));
  CHKERRQ(VecDestroy(&s2));
  CHKERRQ(PetscRandomDestroy(&rand));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 1 -f ${DATAFILESPATH}/matrices/small

   test:
      suffix: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 2 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      suffix: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 4 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 4
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 5 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      suffix: 5
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 6 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 6
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 7 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 7
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 3
      args: -matload_block_size 8 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

   test:
      suffix: 8
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      nsize: 4
      args: -matload_block_size 3 -f ${DATAFILESPATH}/matrices/small
      output_file: output/ex53_1.out

TEST*/
