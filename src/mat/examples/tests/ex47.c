/*$Id: ex47.c,v 1.15 2000/01/11 21:01:03 bsmith Exp balay $*/

static char help[] = 
"Tests the various routines in MatBAIJ format.\n\
Input arguments are:\n\
  -f <input_file> : file to load.  For a 5X5 example of the 5-pt. stencil,\n\
                    use the file petsc/src/mat/examples/matbinary.ex\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         A,B,C;
  Viewer      va,vb,vc;
  Vec         x,y;
  int         ierr,i,j,row,m,n,ncols1,ncols2,*cols1,*cols2,ct,m2,n2;
  char        file[128];
  PetscTruth  tflg;
  Scalar      rval,*vals1,*vals2;
  double      norm1,norm2,rnorm;
  PetscRandom r;


  PetscInitialize(&argc,&args,(char *)0,help);
#if defined(PETSC_USE_COMPLEX)
  SETERRA(1,0,"This example does not work with complex numbers");
#else
  
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,PETSC_NULL);CHKERRA(ierr);

  /* Load the matrix as AIJ format */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&va);CHKERRA(ierr);
  ierr = MatLoad(va,MATSEQAIJ,&A);CHKERRA(ierr);
  ierr = ViewerDestroy(va);CHKERRA(ierr);

  /* Load the matrix as BAIJ format */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&vb);CHKERRA(ierr);
  ierr = MatLoad(vb,MATSEQBAIJ,&B);CHKERRA(ierr);
  ierr = ViewerDestroy(vb);CHKERRA(ierr);

  /* Load the matrix as BAIJ format */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&vc);CHKERRA(ierr);
  ierr = MatLoad(vc,MATSEQBAIJ,&C);CHKERRA(ierr);
  ierr = ViewerDestroy(vc);CHKERRA(ierr);

  ierr = MatGetSize(A,&m,&n);CHKERRA(ierr);
  ierr = MatGetSize(B,&m2,&n2);CHKERRA(ierr);
  if (m!=m2) SETERRA(1,0,"Matrices are of different sixe. Cannot run this example");
 
  /* Test MatEqual() */
  ierr = MatEqual(B,C,&tflg);CHKERRQ(ierr);
  if (!tflg) SETERRA(1,0,"MatEqual() failed");

  /* Test MatGetDiagonal() */
   ierr = VecCreateSeq(PETSC_COMM_SELF,m,&x);CHKERRA(ierr);
   ierr = VecCreateSeq(PETSC_COMM_SELF,m,&y);CHKERRA(ierr);

  ierr = MatGetDiagonal(A,x);CHKERRA(ierr);
  ierr = MatGetDiagonal(B,y);CHKERRA(ierr);
  
  ierr = VecEqual(x,y,&tflg);CHKERRA(ierr);
  if (!tflg)  SETERRA(1,0,"MatGetDiagonal() failed");

  /* Test MatDiagonalScale() */
  
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r);CHKERRA(ierr);
  ierr = VecSetRandom(r,x);CHKERRA(ierr);
  ierr = VecSetRandom(r,y);CHKERRA(ierr);

  ierr = MatDiagonalScale(A,x,y);CHKERRA(ierr);
  ierr = MatDiagonalScale(B,x,y);CHKERRA(ierr);
  ierr = MatMult(A,x,y);CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm1);CHKERRA(ierr);
  ierr = MatMult(B,x,y);CHKERRA(ierr);
  ierr = VecNorm(y,NORM_2,&norm2);CHKERRA(ierr);
  rnorm = ((norm1-norm2)*100)/norm1;
  if (rnorm<-0.1 || rnorm>0.01) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Norm1=%e Norm2=%e\n",norm1,norm2);CHKERRA(ierr);
    SETERRA(1,0,"MatDiagonalScale() failed");
  }

  /* Test MatGetRow()/ MatRestoreRow() */
  for (ct=0; ct<100; ct++) {
    ierr = PetscRandomGetValue(r,&rval);
    row  = (int)(rval*m);
    ierr = MatGetRow(A,row,&ncols1,&cols1,&vals1);CHKERRA(ierr);
    ierr = MatGetRow(B,row,&ncols2,&cols2,&vals2);CHKERRA(ierr);
    
    for (i=0,j=0; i<ncols1 && j<ncols2; i++) {
      while (cols2[j] != cols1[i]) j++;
      if (vals1[i] != vals2[j]) SETERRA(1,0,"MatGetRow() failed - vals incorrect.");
    }
    if (i<ncols1) SETERRA(1,0,"MatGetRow() failed - cols incorrect");
    
    ierr = MatRestoreRow(A,row,&ncols1,&cols1,&vals1);CHKERRA(ierr);
    ierr = MatRestoreRow(B,row,&ncols2,&cols2,&vals2);CHKERRA(ierr);
  }
    
  MatDestroy(A);
  MatDestroy(B);
  MatDestroy(C);
  VecDestroy(x);
  VecDestroy(y);
  PetscRandomDestroy(r);
  PetscFinalize();
#endif
  return 0;
}

