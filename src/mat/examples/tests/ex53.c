/*$Id: ex53.c,v 1.13 1999/11/05 14:45:44 bsmith Exp bsmith $*/

static char help[] = "Tests the vatious routines in MatMPIBAIJ format.\n";


#include "mat.h"
#define IMAX 15
#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         A,B,C,At,Bt;
  Viewer      fd;
  char        file[128];
  PetscRandom rand;
  Vec         xx,yy,s1,s2;
  double      s1norm,s2norm,rnorm,tol = 1.e-10;
  int         rstart,rend,rows[2],cols[2],m,n,i,j,ierr,M,N,rank,ct,row,ncols1;
  int         *cols1,ncols2,*cols2,bs;
  Scalar      vals1[4],vals2[4],v,*v1,*v2;
  PetscTruth  flg;


  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

#if defined(PETSC_USE_COMPLEX)
  SETERRA(1,0,"This example does not work with complex numbers");
#else

 /* Check out if MatLoad() works */
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,&flg);CHKERRA(ierr);
  if (!flg) SETERRA(1,0,"Input file not specified");
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatLoad(fd,MATMPIBAIJ,&A);CHKERRA(ierr);
  ierr = ViewerDestroy(fd);CHKERRA(ierr);

  ierr = MatConvert(A,MATMPIAIJ,&B);CHKERRA(ierr);
 
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rand);CHKERRA(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRA(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,m,PETSC_DECIDE,&xx);CHKERRA(ierr);
  ierr = VecDuplicate(xx,&s1);CHKERRA(ierr);
  ierr = VecDuplicate(xx,&s2);CHKERRA(ierr);
  ierr = VecDuplicate(xx,&yy);CHKERRA(ierr);

  ierr = MatGetBlockSize(A,&bs);CHKERRA(ierr);
  /* Test MatMult() */ 
  for ( i=0; i<IMAX; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = MatMult(A,xx,s1);CHKERRA(ierr);
    ierr = MatMult(B,xx,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e bs = %d\n",
s1norm,s2norm,bs);CHKERRA(ierr);  
    }
  } 
  /* test MatMultAdd() */
  for ( i=0; i<IMAX; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = VecSetRandom(rand,yy);CHKERRA(ierr);
    ierr = MatMultAdd(A,xx,yy,s1);CHKERRA(ierr);
    ierr = MatMultAdd(B,xx,yy,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMultAdd - Norm1=%16.14e Norm2=%16.14e bs = %d\n",s1norm,s2norm,bs);CHKERRA(ierr);
    } 
  }
    /* Test MatMultTranspose() */
  for ( i=0; i<IMAX; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = MatMultTranspose(A,xx,s1);CHKERRA(ierr);
    ierr = MatMultTranspose(B,xx,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMultTranspose - Norm1=%16.14e Norm2=%16.14e bs = %d\n",s1norm,s2norm,bs);CHKERRA(ierr);  
    } 
  }
  /* Test MatMultTransposeAdd() */
  for ( i=0; i<IMAX; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = VecSetRandom(rand,yy);CHKERRA(ierr);
    ierr = MatMultTransposeAdd(A,xx,yy,s1);CHKERRA(ierr);
    ierr = MatMultTransposeAdd(B,xx,yy,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMultTransposeAdd - Norm1=%16.14e Norm2=%16.14e bs = %d\n",s1norm,s2norm,bs);CHKERRA(ierr);
    } 
  }

  /* Check MatGetValues() */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRA(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRA(ierr);


  for ( i=0; i<IMAX; i++ ) {
    /* Create random row numbers ad col numbers */
    ierr = PetscRandomGetValue(rand,&v);CHKERRA(ierr);
    cols[0] = (int)(PetscReal(v)*N);
    ierr = PetscRandomGetValue(rand,&v);CHKERRA(ierr);
    cols[1] = (int)(PetscReal(v)*N);
    ierr = PetscRandomGetValue(rand,&v);CHKERRA(ierr);
    rows[0] = rstart + (int)(PetscReal(v)*m);
    ierr = PetscRandomGetValue(rand,&v);CHKERRA(ierr);
    rows[1] = rstart + (int)(PetscReal(v)*m);
    
    ierr = MatGetValues(A,2,rows,2,cols,vals1);CHKERRA(ierr);
    ierr = MatGetValues(B,2,rows,2,cols,vals2);CHKERRA(ierr);


    for ( j=0; j<4; j++ ) {
      if( vals1[j] != vals2[j] )
        ierr = PetscPrintf(PETSC_COMM_SELF,"[%d]: Error:MatGetValues rstart = %2d  row = %2d col = %2d val1 = %e val2 = %e bs = %d\n",rank,rstart,rows[j/2],cols[j%2],PetscReal(vals1[j]),PetscReal(vals2[j]),bs);CHKERRA(ierr);
    }
  }

  /* Test MatGetRow()/ MatRestoreRow() */
  for ( ct=0; ct<100; ct++ ) {
    ierr = PetscRandomGetValue(rand,&v);
    row  = rstart + (int)(PetscReal(v)*m);
    ierr = MatGetRow(A,row,&ncols1,&cols1,&v1);CHKERRA(ierr);
    ierr = MatGetRow(B,row,&ncols2,&cols2,&v2);CHKERRA(ierr);
    
    for ( i=0,j=0; i<ncols1 && j<ncols2; j++ ) {
      while (cols2[j] != cols1[i]) i++;
      if (v1[i] != v2[j]) SETERRA(1,0, "MatGetRow() failed - vals incorrect.");
    }
    if (j<ncols2) SETERRA(1,0, "MatGetRow() failed - cols incorrect");
    
    ierr = MatRestoreRow(A,row,&ncols1,&cols1,&v1);CHKERRA(ierr);
    ierr = MatRestoreRow(B,row,&ncols2,&cols2,&v2);CHKERRA(ierr);
  }
  
  /* Test MatConvert() */
  ierr = MatConvert(A,MATSAME,&C);CHKERRQ(ierr);
  
  /* See if MatMult Says both are same */ 
  for ( i=0; i<IMAX; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = MatMult(A,xx,s1);CHKERRA(ierr);
    ierr = MatMult(C,xx,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error in MatConvert:MatMult - Norm1=%16.14e Norm2=%16.14e bs = %d\n",
s1norm,s2norm,bs);CHKERRA(ierr);  
    }
  }
  ierr = MatDestroy(C);CHKERRA(ierr);

  /* Test MatTranspose() */
  ierr = MatTranspose(A,&At);CHKERRA(ierr);
  ierr = MatTranspose(B,&Bt);CHKERRA(ierr);
  for ( i=0; i<IMAX; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = MatMult(At,xx,s1);CHKERRA(ierr);
    ierr = MatMult(Bt,xx,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error in MatConvert:MatMult - Norm1=%16.14e Norm2=%16.14e bs = %d\n",
                  s1norm,s2norm,bs);CHKERRA(ierr);
    }
  }
  ierr = MatDestroy(At);CHKERRA(ierr);
  ierr = MatDestroy(Bt);CHKERRA(ierr);

  ierr = MatDestroy(A);CHKERRA(ierr); 
  ierr = MatDestroy(B);CHKERRA(ierr);
  ierr = VecDestroy(xx);CHKERRA(ierr);
  ierr = VecDestroy(yy);CHKERRA(ierr);
  ierr = VecDestroy(s1);CHKERRA(ierr);
  ierr = VecDestroy(s2);CHKERRA(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRA(ierr);
  PetscFinalize();
#endif
  return 0;
}
