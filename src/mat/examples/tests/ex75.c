/*$Id: ex75.c,v 1.22 2000/10/30 18:23:51 hzhang Exp bsmith $*/

/* Program usage:  mpirun -np <procs> ex75 [-help] [all PETSc options] */ 

static char help[] = "Tests the vatious routines in MatMPISBAIJ format.\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int         bs=1, mbs=16, d_nz=3, o_nz=3, prob=2;
  Vec         x,y,u,s1,s2;    
  Mat         A,sA;     
  PetscRandom rctx;         
  double      r1,r2,tol=1.e-10;
  int         i,j,i2,j2,I,J,ierr;
  Scalar      one=1.0, neg_one=-1.0, value[3], four=4.0,alpha=0.1,*vr;
  int         n,rank,size,col[3],n1,block,row;
  int         ncols,*cols,rstart,rend;
  IS          isrow;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mbs",&mbs,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-bs",&bs,PETSC_NULL);CHKERRA(ierr);
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  
  n = mbs*bs;
  
  /* Assemble MPISBAIJ matrix sA */
  ierr = MatCreateMPISBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,n,n,d_nz,PETSC_NULL,o_nz,PETSC_NULL,&sA);CHKERRA(ierr);

  if (bs == 1){
    if (prob == 1){ /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        /* PetscTrValid(0,0,0,0); */
        ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 =  (int) sqrt(n); 
      if (n1*n1 != n){
        SETERRQ(PETSC_ERR_ARG_SIZ,"n must be a perfect square of n1");
      }
        
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          I = j + n1*i;
          if (i>0)   {J = I - n1; ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (i<n1-1) {J = I + n1; ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (j>0)   {J = I - 1; ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (j<n1-1) {J = I + 1; ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          ierr = MatSetValues(sA,1,&I,1,&I,&four,INSERT_VALUES);CHKERRA(ierr);
        }
      }                   
    }
  } /* end of if (bs == 1) */
  else {  /* bs > 1 */
  for (block=0; block<n/bs; block++){
    /* diagonal blocks */
    value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
    for (i=1+block*bs; i<bs-1+block*bs; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);    
    }
    i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
    value[0]=-1.0; value[1]=4.0;  
    ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES);CHKERRA(ierr); 

    i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
    value[0]=4.0; value[1] = -1.0; 
    ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES);CHKERRA(ierr);  
  }
  /* off-diagonal blocks */
  value[0]=-1.0;
  for (i=0; i<(n/bs-1)*bs; i++){
    col[0]=i+bs;
    ierr = MatSetValues(sA,1,&i,1,col,value,INSERT_VALUES);CHKERRA(ierr);
    col[0]=i; row=i+bs;
    ierr = MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES);CHKERRA(ierr);
  }
  }
  ierr = MatAssemblyBegin(sA,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(sA,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* Test MatView() */  
  /*
  ierr = MatView(sA, PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr); 
  ierr = MatView(sA, PETSC_VIEWER_DRAW_WORLD);CHKERRA(ierr);
  */
  /* Assemble MPIBAIJ matrix A */
  ierr = MatCreateMPIBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,n,n,d_nz,PETSC_NULL,o_nz,PETSC_NULL,&A);CHKERRA(ierr);

  if (bs == 1){
    if (prob == 1){ /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        /* PetscTrValid(0,0,0,0); */
        ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 = (int) sqrt(n); 
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          I = j + n1*i;
          if (i>0)   {J = I - n1; ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (i<n1-1) {J = I + n1; ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (j>0)   {J = I - 1; ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (j<n1-1) {J = I + 1; ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          ierr = MatSetValues(A,1,&I,1,&I,&four,INSERT_VALUES);CHKERRA(ierr);
        }
      }                   
    }
  } /* end of if (bs == 1) */
  else {  /* bs > 1 */
  for (block=0; block<n/bs; block++){
    /* diagonal blocks */
    value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
    for (i=1+block*bs; i<bs-1+block*bs; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRA(ierr);    
    }
    i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
    value[0]=-1.0; value[1]=4.0;  
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRA(ierr); 

    i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
    value[0]=4.0; value[1] = -1.0; 
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRA(ierr);  
  }
  /* off-diagonal blocks */
  value[0]=-1.0;
  for (i=0; i<(n/bs-1)*bs; i++){
    col[0]=i+bs;
    ierr = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES);CHKERRA(ierr);
    col[0]=i; row=i+bs;
    ierr = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES);CHKERRA(ierr);
  }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* Test MatGetSize(), MatGetLocalSize() */
  ierr = MatGetSize(sA, &i,&j); ierr = MatGetSize(A, &i2,&j2);
  i -= i2; j -= j2;
  if (i || j) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatGetSize()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }
    
  ierr = MatGetLocalSize(sA, &i,&j); ierr = MatGetLocalSize(A, &i2,&j2);
  i2 -= i; j2 -= j;
  if (i2 || j2) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatGetLocalSize()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }

  /* vectors */
  /*--------------------*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD,i,PETSC_DECIDE,&x); /* i is obtained from MatGetLocalSize() */
 CHKERRA(ierr);
  ierr = VecDuplicate(x,&y);CHKERRA(ierr); 
  ierr = VecDuplicate(x,&u);CHKERRA(ierr);  
  ierr = VecDuplicate(x,&s1);CHKERRA(ierr);
  ierr = VecDuplicate(x,&s2);CHKERRA(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rctx);CHKERRA(ierr);
  ierr = VecSetRandom(rctx,x);CHKERRA(ierr);
  ierr = VecSet(&one,u);CHKERRA(ierr);

  /* Test MatNorm() */
  ierr = MatNorm(A,NORM_FROBENIUS,&r1);CHKERRA(ierr); 
  ierr = MatNorm(sA,NORM_FROBENIUS,&r2);CHKERRA(ierr);
  r1 -= r2;
  if (r1<-tol || r1>tol){    
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatNorm(), A_fnorm - sA_fnorm = %16.14e\n",rank,r1);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }

  /* Test MatGetOwnershipRange() */ 
  ierr = MatGetOwnershipRange(sA,&rstart,&rend);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(A,&i2,&j2);CHKERRA(ierr);
  i2 -= rstart; j2 -= rend;
  if (i2 || j2) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MaGetOwnershipRange()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }  

  /* Test MatGetRow(): can only obtain rows associated with the given processor */
  for (i=rstart; i<rstart+1; i++) {
    ierr = MatGetRow(sA,i,&ncols,&cols,&vr);CHKERRA(ierr);
    /*
    ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"[%d] get row %d: ",rank,i);CHKERRA(ierr);
    for (j=0; j<ncols; j++) {
      ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"%d %g  ",cols[j],vr[j]);CHKERRA(ierr);
    }
    ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"\n");CHKERRA(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);CHKERRA(ierr);
    */
    ierr = MatRestoreRow(sA,i,&ncols,&cols,&vr);CHKERRA(ierr);
  } 

  /* Test MatDiagonalScale(), MatGetDiagonal(), MatScale() */
  ierr = MatDiagonalScale(A,x,x);CHKERRA(ierr);
  ierr = MatDiagonalScale(sA,x,x);CHKERRA(ierr);
  ierr = MatGetDiagonal(A,s1);CHKERRA(ierr);  
  ierr = MatGetDiagonal(sA,s2);CHKERRA(ierr);
  ierr = VecNorm(s1,NORM_1,&r1);CHKERRA(ierr);
  ierr = VecNorm(s2,NORM_1,&r2);CHKERRA(ierr);
  r1 -= r2;
  if (r1<-tol || r1>tol) { 
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDiagonalScale() or MatGetDiagonal(), r1=%g \n",rank,r1);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }
  
  ierr = MatScale(&alpha,A);CHKERRA(ierr);
  ierr = MatScale(&alpha,sA);CHKERRA(ierr);

  /* Test MatGetRowMax() */
  ierr = MatGetRowMax(A,s1);CHKERRA(ierr);  
  ierr = MatGetRowMax(sA,s2);CHKERRA(ierr);

  ierr = VecNorm(s1,NORM_1,&r1);CHKERRA(ierr);
  ierr = VecNorm(s2,NORM_1,&r2);CHKERRA(ierr);
  /* MatView(A,PETSC_VIEWER_STDOUT_WORLD);
  PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], r1: %g, r2: %g\n",rank,r1,r2);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      */
  r1 -= r2;
  if (r1<-tol || r1>tol) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetRowMax() \n");CHKERRA(ierr);
  } 

  /* Test MatMult(), MatMultAdd() */
  for (i=0; i<10; i++) {
    ierr = VecSetRandom(rctx,x);CHKERRA(ierr);
    ierr = MatMult(A,x,s1);CHKERRA(ierr);
    ierr = MatMult(sA,x,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_1,&r1);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_1,&r2);CHKERRA(ierr);
    r1 -= r2;
    if (r1<-tol || r1>tol) {
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMult() or MatScale(), err=%g\n",rank,r1);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
    }
  }

  for (i=0; i<10; i++) {
    ierr = VecSetRandom(rctx,x);CHKERRA(ierr);
    ierr = VecSetRandom(rctx,y);CHKERRA(ierr);
    ierr = MatMultAdd(A,x,y,s1);CHKERRA(ierr);
    ierr = MatMultAdd(sA,x,y,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_1,&r1);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_1,&r2);CHKERRA(ierr);
    r1 -= r2;
    if (r1<-tol || r1>tol) {
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMultAdd(), err=%g \n",rank,r1);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);      
    }
  }                

  /* Test MatZeroRows() */
  ierr = ISCreateStride(PETSC_COMM_SELF,2,rstart,2,&isrow);CHKERRA(ierr);   
  /* ISView(isrow, PETSC_VIEWER_STDOUT_SELF);CHKERRA(ierr); */
  ierr = MatZeroRows(sA,isrow,PETSC_NULL);CHKERRA(ierr); 
  ierr = ISDestroy(isrow);CHKERRA(ierr);
  /* ierr = MatView(sA, PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);  */
  /* ierr = MatView(sA, PETSC_VIEWER_DRAW_WORLD);CHKERRA(ierr);  */
  
  ierr = VecDestroy(u);CHKERRA(ierr);  
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr); 
  ierr = VecDestroy(s1);CHKERRA(ierr);
  ierr = VecDestroy(s2);CHKERRA(ierr);
  ierr = MatDestroy(sA);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = PetscRandomDestroy(rctx);CHKERRA(ierr);
 
  PetscFinalize();
  return 0;
}
