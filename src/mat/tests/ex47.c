
static char help[] = "Tests the various routines in MatBAIJ format.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat               A,B,C;
  PetscViewer       va,vb,vc;
  Vec               x,y;
  PetscInt          i,j,row,m,n,ncols1,ncols2,ct,m2,n2;
  const PetscInt    *cols1,*cols2;
  char              file[PETSC_MAX_PATH_LEN];
  PetscBool         tflg;
  PetscScalar       rval;
  const PetscScalar *vals1,*vals2;
  PetscReal         norm1,norm2,rnorm;
  PetscRandom       r;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));

  /* Load the matrix as AIJ format */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&va));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATSEQAIJ));
  CHKERRQ(MatLoad(A,va));
  CHKERRQ(PetscViewerDestroy(&va));

  /* Load the matrix as BAIJ format */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&vb));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetType(B,MATSEQBAIJ));
  CHKERRQ(MatLoad(B,vb));
  CHKERRQ(PetscViewerDestroy(&vb));

  /* Load the matrix as BAIJ format */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&vc));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetType(C,MATSEQBAIJ));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatLoad(C,vc));
  CHKERRQ(PetscViewerDestroy(&vc));

  CHKERRQ(MatGetSize(A,&m,&n));
  CHKERRQ(MatGetSize(B,&m2,&n2));
  PetscCheckFalse(m!=m2,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrices are of different size. Cannot run this example");

  /* Test MatEqual() */
  CHKERRQ(MatEqual(B,C,&tflg));
  PetscCheck(tflg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatEqual() failed");

  /* Test MatGetDiagonal() */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,m,&x));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,m,&y));

  CHKERRQ(MatGetDiagonal(A,x));
  CHKERRQ(MatGetDiagonal(B,y));

  CHKERRQ(VecEqual(x,y,&tflg));
  PetscCheck(tflg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetDiagonal() failed");

  /* Test MatDiagonalScale() */
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&r));
  CHKERRQ(PetscRandomSetFromOptions(r));
  CHKERRQ(VecSetRandom(x,r));
  CHKERRQ(VecSetRandom(y,r));

  CHKERRQ(MatDiagonalScale(A,x,y));
  CHKERRQ(MatDiagonalScale(B,x,y));
  CHKERRQ(MatMult(A,x,y));
  CHKERRQ(VecNorm(y,NORM_2,&norm1));
  CHKERRQ(MatMult(B,x,y));
  CHKERRQ(VecNorm(y,NORM_2,&norm2));
  rnorm = ((norm1-norm2)*100)/norm1;
  PetscCheckFalse(rnorm<-0.1 || rnorm>0.01,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDiagonalScale() failed Norm1 %g Norm2 %g",(double)norm1,(double)norm2);

  /* Test MatGetRow()/ MatRestoreRow() */
  for (ct=0; ct<100; ct++) {
    CHKERRQ(PetscRandomGetValue(r,&rval));
    row  = (int)(rval*m);
    CHKERRQ(MatGetRow(A,row,&ncols1,&cols1,&vals1));
    CHKERRQ(MatGetRow(B,row,&ncols2,&cols2,&vals2));

    for (i=0,j=0; i<ncols1 && j<ncols2; i++) {
      while (cols2[j] != cols1[i]) j++;
      PetscCheckFalse(vals1[i] != vals2[j],PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetRow() failed - vals incorrect.");
    }
    PetscCheckFalse(i<ncols1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetRow() failed - cols incorrect");

    CHKERRQ(MatRestoreRow(A,row,&ncols1,&cols1,&vals1));
    CHKERRQ(MatRestoreRow(B,row,&ncols2,&cols2,&vals2));
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(PetscRandomDestroy(&r));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      args: -f ${DATAFILESPATH}/matrices/cfd.1.10 -mat_block_size 5
      requires: !complex double datafilespath !defined(PETSC_USE_64BIT_INDICES)

TEST*/
