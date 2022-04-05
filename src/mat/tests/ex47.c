
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),NULL));

  /* Load the matrix as AIJ format */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&va));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetType(A,MATSEQAIJ));
  PetscCall(MatLoad(A,va));
  PetscCall(PetscViewerDestroy(&va));

  /* Load the matrix as BAIJ format */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&vb));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetType(B,MATSEQBAIJ));
  PetscCall(MatLoad(B,vb));
  PetscCall(PetscViewerDestroy(&vb));

  /* Load the matrix as BAIJ format */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&vc));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetType(C,MATSEQBAIJ));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatLoad(C,vc));
  PetscCall(PetscViewerDestroy(&vc));

  PetscCall(MatGetSize(A,&m,&n));
  PetscCall(MatGetSize(B,&m2,&n2));
  PetscCheck(m==m2,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrices are of different size. Cannot run this example");

  /* Test MatEqual() */
  PetscCall(MatEqual(B,C,&tflg));
  PetscCheck(tflg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatEqual() failed");

  /* Test MatGetDiagonal() */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,m,&x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,m,&y));

  PetscCall(MatGetDiagonal(A,x));
  PetscCall(MatGetDiagonal(B,y));

  PetscCall(VecEqual(x,y,&tflg));
  PetscCheck(tflg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetDiagonal() failed");

  /* Test MatDiagonalScale() */
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&r));
  PetscCall(PetscRandomSetFromOptions(r));
  PetscCall(VecSetRandom(x,r));
  PetscCall(VecSetRandom(y,r));

  PetscCall(MatDiagonalScale(A,x,y));
  PetscCall(MatDiagonalScale(B,x,y));
  PetscCall(MatMult(A,x,y));
  PetscCall(VecNorm(y,NORM_2,&norm1));
  PetscCall(MatMult(B,x,y));
  PetscCall(VecNorm(y,NORM_2,&norm2));
  rnorm = ((norm1-norm2)*100)/norm1;
  PetscCheckFalse(rnorm<-0.1 || rnorm>0.01,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatDiagonalScale() failed Norm1 %g Norm2 %g",(double)norm1,(double)norm2);

  /* Test MatGetRow()/ MatRestoreRow() */
  for (ct=0; ct<100; ct++) {
    PetscCall(PetscRandomGetValue(r,&rval));
    row  = (int)(rval*m);
    PetscCall(MatGetRow(A,row,&ncols1,&cols1,&vals1));
    PetscCall(MatGetRow(B,row,&ncols2,&cols2,&vals2));

    for (i=0,j=0; i<ncols1 && j<ncols2; i++) {
      while (cols2[j] != cols1[i]) j++;
      PetscCheck(vals1[i] == vals2[j],PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetRow() failed - vals incorrect.");
    }
    PetscCheck(i>=ncols1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MatGetRow() failed - cols incorrect");

    PetscCall(MatRestoreRow(A,row,&ncols1,&cols1,&vals1));
    PetscCall(MatRestoreRow(B,row,&ncols2,&cols2,&vals2));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscRandomDestroy(&r));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      args: -f ${DATAFILESPATH}/matrices/cfd.1.10 -mat_block_size 5
      requires: !complex double datafilespath !defined(PETSC_USE_64BIT_INDICES)

TEST*/
