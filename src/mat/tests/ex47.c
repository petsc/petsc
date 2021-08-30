
static char help[] = "Tests the various routines in MatBAIJ format.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat               A,B,C;
  PetscViewer       va,vb,vc;
  Vec               x,y;
  PetscErrorCode    ierr;
  PetscInt          i,j,row,m,n,ncols1,ncols2,ct,m2,n2;
  const PetscInt    *cols1,*cols2;
  char              file[PETSC_MAX_PATH_LEN];
  PetscBool         tflg;
  PetscScalar       rval;
  const PetscScalar *vals1,*vals2;
  PetscReal         norm1,norm2,rnorm;
  PetscRandom       r;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL);CHKERRQ(ierr);

  /* Load the matrix as AIJ format */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&va);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,va);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&va);CHKERRQ(ierr);

  /* Load the matrix as BAIJ format */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&vb);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatLoad(B,vb);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vb);CHKERRQ(ierr);

  /* Load the matrix as BAIJ format */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&vc);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatLoad(C,vc);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vc);CHKERRQ(ierr);

  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(B,&m2,&n2);CHKERRQ(ierr);
  if (m!=m2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrices are of different size. Cannot run this example");

  /* Test MatEqual() */
  ierr = MatEqual(B,C,&tflg);CHKERRQ(ierr);
  if (!tflg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatEqual() failed");

  /* Test MatGetDiagonal() */
  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&y);CHKERRQ(ierr);

  ierr = MatGetDiagonal(A,x);CHKERRQ(ierr);
  ierr = MatGetDiagonal(B,y);CHKERRQ(ierr);

  ierr = VecEqual(x,y,&tflg);CHKERRQ(ierr);
  if (!tflg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetDiagonal() failed");

  /* Test MatDiagonalScale() */
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
  ierr = VecSetRandom(x,r);CHKERRQ(ierr);
  ierr = VecSetRandom(y,r);CHKERRQ(ierr);

  ierr  = MatDiagonalScale(A,x,y);CHKERRQ(ierr);
  ierr  = MatDiagonalScale(B,x,y);CHKERRQ(ierr);
  ierr  = MatMult(A,x,y);CHKERRQ(ierr);
  ierr  = VecNorm(y,NORM_2,&norm1);CHKERRQ(ierr);
  ierr  = MatMult(B,x,y);CHKERRQ(ierr);
  ierr  = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
  rnorm = ((norm1-norm2)*100)/norm1;
  if (rnorm<-0.1 || rnorm>0.01) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDiagonalScale() failed Norm1 %g Norm2 %g",(double)norm1,(double)norm2);

  /* Test MatGetRow()/ MatRestoreRow() */
  for (ct=0; ct<100; ct++) {
    ierr = PetscRandomGetValue(r,&rval);CHKERRQ(ierr);
    row  = (int)(rval*m);
    ierr = MatGetRow(A,row,&ncols1,&cols1,&vals1);CHKERRQ(ierr);
    ierr = MatGetRow(B,row,&ncols2,&cols2,&vals2);CHKERRQ(ierr);

    for (i=0,j=0; i<ncols1 && j<ncols2; i++) {
      while (cols2[j] != cols1[i]) j++;
      if (vals1[i] != vals2[j]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetRow() failed - vals incorrect.");
    }
    if (i<ncols1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetRow() failed - cols incorrect");

    ierr = MatRestoreRow(A,row,&ncols1,&cols1,&vals1);CHKERRQ(ierr);
    ierr = MatRestoreRow(B,row,&ncols2,&cols2,&vals2);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      args: -f ${DATAFILESPATH}/matrices/cfd.1.10 -mat_block_size 5
      requires: !complex double datafilespath !defined(PETSC_USE_64BIT_INDICES)

TEST*/
