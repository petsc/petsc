
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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* Check out if MatLoad() works */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Input file not specified");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATBAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);

  ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&xx);CHKERRQ(ierr);
  ierr = VecSetSizes(xx,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(xx);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&yy);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);

  /* Test MatNorm() */
  ierr  = MatNorm(A,NORM_FROBENIUS,&s1norm);CHKERRQ(ierr);
  ierr  = MatNorm(B,NORM_FROBENIUS,&s2norm);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatNorm_FROBENIUS()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs);CHKERRQ(ierr);
  }
  ierr  = MatNorm(A,NORM_INFINITY,&s1norm);CHKERRQ(ierr);
  ierr  = MatNorm(B,NORM_INFINITY,&s2norm);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatNorm_INFINITY()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs);CHKERRQ(ierr);
  }
  ierr  = MatNorm(A,NORM_1,&s1norm);CHKERRQ(ierr);
  ierr  = MatNorm(B,NORM_1,&s2norm);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatNorm_NORM_1()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs);CHKERRQ(ierr);
  }

  /* Test MatMult() */
  for (i=0; i<IMAX; i++) {
    ierr = VecSetRandom(xx,rand);CHKERRQ(ierr);
    ierr = MatMult(A,xx,s1);CHKERRQ(ierr);
    ierr = MatMult(B,xx,s2);CHKERRQ(ierr);
    ierr = VecAXPY(s2,-1.0,s1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_2,&rnorm);CHKERRQ(ierr);
    if (rnorm>tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatMult - Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)rnorm,bs);CHKERRQ(ierr);
    }
  }

  /* test MatMultAdd() */
  for (i=0; i<IMAX; i++) {
    ierr = VecSetRandom(xx,rand);CHKERRQ(ierr);
    ierr = VecSetRandom(yy,rand);CHKERRQ(ierr);
    ierr = MatMultAdd(A,xx,yy,s1);CHKERRQ(ierr);
    ierr = MatMultAdd(B,xx,yy,s2);CHKERRQ(ierr);
    ierr = VecAXPY(s2,-1.0,s1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_2,&rnorm);CHKERRQ(ierr);
    if (rnorm>tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatMultAdd - Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)rnorm,bs);CHKERRQ(ierr);
    }
  }

  /* Test MatMultTranspose() */
  for (i=0; i<IMAX; i++) {
    ierr  = VecSetRandom(xx,rand);CHKERRQ(ierr);
    ierr  = MatMultTranspose(A,xx,s1);CHKERRQ(ierr);
    ierr  = MatMultTranspose(B,xx,s2);CHKERRQ(ierr);
    ierr  = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
    ierr  = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatMultTranspose - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs);CHKERRQ(ierr);
    }
  }
  /* Test MatMultTransposeAdd() */
  for (i=0; i<IMAX; i++) {
    ierr  = VecSetRandom(xx,rand);CHKERRQ(ierr);
    ierr  = VecSetRandom(yy,rand);CHKERRQ(ierr);
    ierr  = MatMultTransposeAdd(A,xx,yy,s1);CHKERRQ(ierr);
    ierr  = MatMultTransposeAdd(B,xx,yy,s2);CHKERRQ(ierr);
    ierr  = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
    ierr  = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Error: MatMultTransposeAdd - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs);CHKERRQ(ierr);
    }
  }

  /* Check MatGetValues() */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

  for (i=0; i<IMAX; i++) {
    /* Create random row numbers ad col numbers */
    ierr    = PetscRandomGetValue(rand,&v);CHKERRQ(ierr);
    cols[0] = (int)(PetscRealPart(v)*N);
    ierr    = PetscRandomGetValue(rand,&v);CHKERRQ(ierr);
    cols[1] = (int)(PetscRealPart(v)*N);
    ierr    = PetscRandomGetValue(rand,&v);CHKERRQ(ierr);
    rows[0] = rstart + (int)(PetscRealPart(v)*m);
    ierr    = PetscRandomGetValue(rand,&v);CHKERRQ(ierr);
    rows[1] = rstart + (int)(PetscRealPart(v)*m);

    ierr = MatGetValues(A,2,rows,2,cols,vals1);CHKERRQ(ierr);
    ierr = MatGetValues(B,2,rows,2,cols,vals2);CHKERRQ(ierr);

    for (j=0; j<4; j++) {
      if (vals1[j] != vals2[j]) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"[%d]: Error: MatGetValues rstart = %2" PetscInt_FMT "  row = %2" PetscInt_FMT " col = %2" PetscInt_FMT " val1 = %e val2 = %e bs = %" PetscInt_FMT "\n",rank,rstart,rows[j/2],cols[j%2],(double)PetscRealPart(vals1[j]),(double)PetscRealPart(vals2[j]),bs);CHKERRQ(ierr);
      }
    }
  }

  /* Test MatGetRow()/ MatRestoreRow() */
  for (ct=0; ct<100; ct++) {
    ierr = PetscRandomGetValue(rand,&v);CHKERRQ(ierr);
    row  = rstart + (PetscInt)(PetscRealPart(v)*m);
    ierr = MatGetRow(A,row,&ncols1,&cols1,&v1);CHKERRQ(ierr);
    ierr = MatGetRow(B,row,&ncols2,&cols2,&v2);CHKERRQ(ierr);

    for (i=0,j=0; i<ncols1 && j<ncols2; j++) {
      while (cols2[j] != cols1[i]) i++;
      if (v1[i] != v2[j]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetRow() failed - vals incorrect.");
    }
    if (j<ncols2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetRow() failed - cols incorrect");

    ierr = MatRestoreRow(A,row,&ncols1,&cols1,&v1);CHKERRQ(ierr);
    ierr = MatRestoreRow(B,row,&ncols2,&cols2,&v2);CHKERRQ(ierr);
  }

  /* Test MatConvert() */
  ierr = MatConvert(A,MATSAME,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);

  /* See if MatMult Says both are same */
  for (i=0; i<IMAX; i++) {
    ierr  = VecSetRandom(xx,rand);CHKERRQ(ierr);
    ierr  = MatMult(A,xx,s1);CHKERRQ(ierr);
    ierr  = MatMult(C,xx,s2);CHKERRQ(ierr);
    ierr  = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
    ierr  = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Error in MatConvert: MatMult - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  /* Test MatTranspose() */
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&At);CHKERRQ(ierr);
  ierr = MatTranspose(B,MAT_INITIAL_MATRIX,&Bt);CHKERRQ(ierr);
  for (i=0; i<IMAX; i++) {
    ierr  = VecSetRandom(xx,rand);CHKERRQ(ierr);
    ierr  = MatMult(At,xx,s1);CHKERRQ(ierr);
    ierr  = MatMult(Bt,xx,s2);CHKERRQ(ierr);
    ierr  = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
    ierr  = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Error in MatConvert:MatMult - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",rank,(double)s1norm,(double)s2norm,bs);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&At);CHKERRQ(ierr);
  ierr = MatDestroy(&Bt);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&xx);CHKERRQ(ierr);
  ierr = VecDestroy(&yy);CHKERRQ(ierr);
  ierr = VecDestroy(&s1);CHKERRQ(ierr);
  ierr = VecDestroy(&s2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
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
