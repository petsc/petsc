
/* Program usage:  mpiexec -n <procs> ex75 [-help] [all PETSc options] */ 

static char help[] = "Tests the vatious routines in MatMPISBAIJ format.\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec               x,y,u,s1,s2;    
  Mat               A,sA,sB;     
  PetscRandom       rctx;         
  PetscReal         r1,r2,rnorm,tol=1.e-10;
  PetscScalar       one=1.0, neg_one=-1.0, value[3], four=4.0,alpha=0.1;
  PetscInt          n,col[3],n1,block,row,i,j,i2,j2,Ii,J,rstart,rend,bs=1,mbs=16,d_nz=3,o_nz=3,prob=2;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscTruth        flg;
  const MatType     type;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mbs",&mbs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-bs",&bs,PETSC_NULL);CHKERRQ(ierr);
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  
  n = mbs*bs;
  
  /* Assemble MPISBAIJ matrix sA */
  ierr = MatCreate(PETSC_COMM_WORLD,&sA);CHKERRQ(ierr);
  ierr = MatSetSizes(sA,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(sA,MATSBAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(sA);CHKERRQ(ierr);
  ierr = MatGetType(sA,&type);CHKERRQ(ierr);
  /* printf(" mattype: %s\n",type); */
  ierr = MatMPISBAIJSetPreallocation(sA,bs,d_nz,PETSC_NULL,o_nz,PETSC_NULL);CHKERRQ(ierr);

  if (bs == 1){
    if (prob == 1){ /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 =  (int) sqrt((double)n); 
      if (n1*n1 != n){
        SETERRQ(PETSC_ERR_ARG_SIZ,"n must be a perfect square of n1");
      }
        
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          Ii = j + n1*i;
          if (i>0)    {J = Ii - n1; ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          if (i<n1-1) {J = Ii + n1; ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          if (j>0)    {J = Ii - 1;  ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          if (j<n1-1) {J = Ii + 1;  ierr = MatSetValues(sA,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          ierr = MatSetValues(sA,1,&Ii,1,&Ii,&four,INSERT_VALUES);CHKERRQ(ierr);
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
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);    
    }
    i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
    value[0]=-1.0; value[1]=4.0;  
    ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr); 

    i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
    value[0]=4.0; value[1] = -1.0; 
    ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);  
  }
  /* off-diagonal blocks */
  value[0]=-1.0;
  for (i=0; i<(n/bs-1)*bs; i++){
    col[0]=i+bs;
    ierr = MatSetValues(sA,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    col[0]=i; row=i+bs;
    ierr = MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  }
  ierr = MatAssemblyBegin(sA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test MatView() */  
  /*
  ierr = MatView(sA, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  ierr = MatView(sA, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  */
  /* Assemble MPIBAIJ matrix A */
  ierr = MatCreateMPIBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,n,n,d_nz,PETSC_NULL,o_nz,PETSC_NULL,&A);CHKERRQ(ierr);

  if (bs == 1){
    if (prob == 1){ /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 = (int) sqrt((double)n); 
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          Ii = j + n1*i;
          if (i>0)    {J = Ii - n1; ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          if (i<n1-1) {J = Ii + n1; ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          if (j>0)    {J = Ii - 1;  ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          if (j<n1-1) {J = Ii + 1;  ierr = MatSetValues(A,1,&Ii,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);}
          ierr = MatSetValues(A,1,&Ii,1,&Ii,&four,INSERT_VALUES);CHKERRQ(ierr);
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
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);    
    }
    i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
    value[0]=-1.0; value[1]=4.0;  
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr); 

    i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
    value[0]=4.0; value[1] = -1.0; 
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);  
  }
  /* off-diagonal blocks */
  value[0]=-1.0;
  for (i=0; i<(n/bs-1)*bs; i++){
    col[0]=i+bs;
    ierr = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    col[0]=i; row=i+bs;
    ierr = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

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
  /* i is obtained from MatGetLocalSize() */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,i,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr); 
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);  
  ierr = VecDuplicate(x,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&s2);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
  ierr = VecSet(u,one);CHKERRQ(ierr);

  /* Test MatNorm() */
  ierr = MatNorm(A,NORM_FROBENIUS,&r1);CHKERRQ(ierr); 
  ierr = MatNorm(sA,NORM_FROBENIUS,&r2);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(r1-r2)/r2;
  if (rnorm > tol && !rank){    
    PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_FROBENIUS(), Anorm=%16.14e, sAnorm=%16.14e bs=%D\n",r1,r2,bs);
  }
  ierr = MatNorm(A,NORM_INFINITY,&r1);CHKERRQ(ierr); 
  ierr = MatNorm(sA,NORM_INFINITY,&r2);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(r1-r2)/r2;
  if (rnorm > tol && !rank){    
    PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm_INFINITY(), Anorm=%16.14e, sAnorm=%16.14e bs=%D\n",r1,r2,bs);
  }
  ierr = MatNorm(A,NORM_1,&r1);CHKERRQ(ierr); 
  ierr = MatNorm(sA,NORM_1,&r2);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(r1-r2)/r2;
  if (rnorm > tol && !rank){    
    PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm_1(), Anorm=%16.14e, sAnorm=%16.14e bs=%D\n",r1,r2,bs);
  }

  /* Test MatGetOwnershipRange() */ 
  ierr = MatGetOwnershipRange(sA,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&i2,&j2);CHKERRQ(ierr);
  i2 -= rstart; j2 -= rend;
  if (i2 || j2) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MaGetOwnershipRange()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }  

  /* Test MatDiagonalScale() */
  ierr = MatDiagonalScale(A,x,x);CHKERRQ(ierr);
  ierr = MatDiagonalScale(sA,x,x);CHKERRQ(ierr);
  ierr = MatMultEqual(A,sA,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_NOTSAMETYPE,"Error in MatDiagonalScale");
  
  /* Test MatGetDiagonal(), MatScale() */
  ierr = MatGetDiagonal(A,s1);CHKERRQ(ierr);  
  ierr = MatGetDiagonal(sA,s2);CHKERRQ(ierr);
  ierr = VecNorm(s1,NORM_1,&r1);CHKERRQ(ierr);
  ierr = VecNorm(s2,NORM_1,&r2);CHKERRQ(ierr);
  r1 -= r2;
  if (r1<-tol || r1>tol) { 
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDiagonalScale() or MatGetDiagonal(), r1=%G \n",rank,r1);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }
  
  ierr = MatScale(A,alpha);CHKERRQ(ierr);
  ierr = MatScale(sA,alpha);CHKERRQ(ierr);

  /* Test MatGetRowMaxAbs() */
  ierr = MatGetRowMaxAbs(A,s1,PETSC_NULL);CHKERRQ(ierr);  
  ierr = MatGetRowMaxAbs(sA,s2,PETSC_NULL);CHKERRQ(ierr);

  ierr = VecNorm(s1,NORM_1,&r1);CHKERRQ(ierr);
  ierr = VecNorm(s2,NORM_1,&r2);CHKERRQ(ierr);
  r1 -= r2;
  if (r1<-tol || r1>tol) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetRowMaxAbs() \n");CHKERRQ(ierr);
  } 

  /* Test MatMult(), MatMultAdd() */
  ierr = MatMultEqual(A,sA,10,&flg);CHKERRQ(ierr);
  if (!flg){
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMult() or MatScale()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }

  ierr = MatMultAddEqual(A,sA,10,&flg);CHKERRQ(ierr);
  if (!flg){
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMultAdd()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }

  /* Test MatMultTranspose(), MatMultTransposeAdd() */
  for (i=0; i<10; i++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,x,s1);CHKERRQ(ierr);
    ierr = MatMultTranspose(sA,x,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_1,&r1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_1,&r2);CHKERRQ(ierr);
    r1 -= r2;
    if (r1<-tol || r1>tol) {
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMult() or MatScale(), err=%G\n",rank,r1);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
    }
  }
  for (i=0; i<10; i++) {
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(y,rctx);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(A,x,y,s1);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(sA,x,y,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_1,&r1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_1,&r2);CHKERRQ(ierr);
    r1 -= r2;
    if (r1<-tol || r1>tol) {
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatMultAdd(), err=%G \n",rank,r1);
      PetscSynchronizedFlush(PETSC_COMM_WORLD);      
    }
  }
  /* ierr = MatView(sA, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);  */
  /* ierr = MatView(sA, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);  */

  /* Test MatDuplicate() */
  ierr = MatDuplicate(sA,MAT_COPY_VALUES,&sB);CHKERRQ(ierr);
  ierr = MatEqual(sA,sB,&flg);CHKERRQ(ierr);
  if (!flg){
    PetscPrintf(PETSC_COMM_WORLD," Error in MatDuplicate(), sA != sB \n");CHKERRQ(ierr);
  } 
  ierr = MatMultEqual(sA,sB,5,&flg);CHKERRQ(ierr);
  if (!flg){
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDuplicate() or MatMult()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }
  ierr = MatMultAddEqual(sA,sB,5,&flg);CHKERRQ(ierr);
  if (!flg){
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatDuplicate() or MatMultAdd(()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }
  ierr = MatDestroy(sB);CHKERRQ(ierr); 
  
  ierr = VecDestroy(u);CHKERRQ(ierr);  
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr); 
  ierr = VecDestroy(s1);CHKERRQ(ierr);
  ierr = VecDestroy(s2);CHKERRQ(ierr);
  ierr = MatDestroy(sA);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rctx);CHKERRQ(ierr);
 
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
