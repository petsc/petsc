/*$Id: ex74.c,v 1.47 2001/08/07 21:30:08 bsmith Exp $*/

static char help[] = "Tests the various sequential routines in MatSBAIJ format.\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec          x,y,b,s1,s2;      
  Mat          A;             /* linear system matrix */ 
  Mat          sA,sC;         /* symmetric part of the matrices */ 
  int          n,mbs=16,bs=1,nz=3,prob=1;
  int          ierr,i,j,col[3],size,block, row,I,J,n1,*ip_ptr,inc; 
  int          lf;           /* level of fill for icc */
  int          *cols1,*cols2;
  PetscReal    norm1,norm2,tol=1.e-10,fill;
  PetscScalar  neg_one = -1.0,four=4.0,value[3],alpha=0.1;  
  PetscScalar  *vr1,*vr2,*vr1_wk,*vr2_wk;
  IS           perm, isrow, iscol;
  PetscRandom  rand;
  PetscTruth   getrow=PETSC_FALSE;
  MatInfo      minfo1,minfo2;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(1,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(PETSC_NULL,"-bs",&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mbs",&mbs,PETSC_NULL);CHKERRQ(ierr);

  n = mbs*bs;
  ierr=MatCreateSeqBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,PETSC_NULL, &A);CHKERRQ(ierr);
  ierr=MatCreateSeqSBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,PETSC_NULL, &sA);CHKERRQ(ierr);

  /* Test MatGetOwnershipRange() */
  ierr = MatGetOwnershipRange(A,&I,&J);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(sA,&i,&j);CHKERRQ(ierr);
  if (i-I || j-J){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetOwnershipRange() in MatSBAIJ format\n");CHKERRQ(ierr);
  }

  /* Assemble matrix */
  if (bs == 1){
    ierr = PetscOptionsGetInt(PETSC_NULL,"-test_problem",&prob,PETSC_NULL);CHKERRQ(ierr);
    if (prob == 1){ /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i = 0; 
      col[0] = n-1;  col[1] = 1; col[2]=0; 
      value[0] = 0.1; value[1] = -1.0; value[2]=2;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 = (int) (sqrt((PetscReal)n) + 0.001); 
      if (n1*n1 - n) SETERRQ(PETSC_ERR_ARG_WRONG,"sqrt(n) must be a positive interger!"); 
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          I = j + n1*i;
          if (i>0)   {
            J = I - n1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr); 
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (i<n1-1) {
            J = I + n1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (j>0)   {
            J = I - 1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          if (j<n1-1) {
            J = I + 1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRQ(ierr);
          }
          ierr = MatSetValues(A,1,&I,1,&I,&four,INSERT_VALUES);CHKERRQ(ierr);
          ierr = MatSetValues(sA,1,&I,1,&I,&four,INSERT_VALUES);CHKERRQ(ierr);
        }
      }                   
    }
  } 
  else { /* bs > 1 */
    for (block=0; block<n/bs; block++){
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);    
      }
      i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
      value[0]=-1.0; value[1]=4.0;  
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr); 

      i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
      value[0]=4.0; value[1] = -1.0; 
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);  
    }
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(n/bs-1)*bs; i++){
      col[0]=i+bs;
      ierr = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      col[0]=i; row=i+bs;
      ierr = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* PetscPrintf(PETSC_COMM_SELF,"\n The Matrix: \n");
  MatView(A, PETSC_VIEWER_DRAW_WORLD);
  MatView(A, PETSC_VIEWER_STDOUT_WORLD); */ 

  ierr = MatAssemblyBegin(sA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  /* PetscPrintf(PETSC_COMM_SELF,"\n Symmetric Part of Matrix: \n");
  MatView(sA, PETSC_VIEWER_DRAW_WORLD); 
  MatView(sA, PETSC_VIEWER_STDOUT_WORLD); 
  */

  /* Test MatNorm(), MatDuplicate() */
  ierr = MatNorm(A,NORM_FROBENIUS,&norm1);CHKERRQ(ierr); 
  ierr = MatDuplicate(sA,MAT_COPY_VALUES,&sC);CHKERRQ(ierr);
  ierr = MatNorm(sC,NORM_FROBENIUS,&norm2);CHKERRQ(ierr);
  ierr = MatDestroy(sC);CHKERRQ(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol){ 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm(), fnorm1-fnorm2=%16.14e\n",norm1);CHKERRQ(ierr);
  }
  ierr = MatNorm(A,NORM_INFINITY,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(sA,NORM_INFINITY,&norm2);CHKERRQ(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol){ 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm(), inf_norm1-inf_norm2=%16.14e\n",norm1);CHKERRQ(ierr);
  }
  
  /* Test MatGetInfo(), MatGetSize(), MatGetBlockSize() */
  ierr = MatGetInfo(A,MAT_LOCAL,&minfo1);CHKERRQ(ierr);
  ierr = MatGetInfo(sA,MAT_LOCAL,&minfo2);CHKERRQ(ierr);
  /*
  printf("matrix nonzeros (BAIJ format) = %d, allocated nonzeros= %d\n", (int)minfo1.nz_used,(int)minfo1.nz_allocated); 
  printf("matrix nonzeros(SBAIJ format) = %d, allocated nonzeros= %d\n", (int)minfo2.nz_used,(int)minfo2.nz_allocated); 
  */
  i = (int) (minfo1.nz_used - minfo2.nz_used); 
  j = (int) (minfo2.nz_allocated - minfo2.nz_used);
  if (i<0 || j<0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetInfo()\n");CHKERRQ(ierr);
  }

  ierr = MatGetSize(A,&I,&J);CHKERRQ(ierr);
  ierr = MatGetSize(sA,&i,&j);CHKERRQ(ierr); 
  if (i-I || j-J) {
    PetscPrintf(PETSC_COMM_SELF,"Error: MatGetSize()\n");CHKERRQ(ierr);
  }
 
  ierr = MatGetBlockSize(A, &I);CHKERRQ(ierr);
  ierr = MatGetBlockSize(sA, &i);CHKERRQ(ierr);
  if (i-I){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetBlockSize()\n");CHKERRQ(ierr);
  }

  /* Test MatGetRow() */
  if (getrow){
    row = n/2; 
    ierr = PetscMalloc(n*sizeof(PetscScalar),&vr1);CHKERRQ(ierr); 
    vr1_wk = vr1;  
    ierr = PetscMalloc(n*sizeof(PetscScalar),&vr2);CHKERRQ(ierr); 
    vr2_wk = vr2;
    ierr = MatGetRow(A,row,&J,&cols1,&vr1);CHKERRQ(ierr); 
    vr1_wk += J-1;
    ierr = MatGetRow(sA,row,&j,&cols2,&vr2);CHKERRQ(ierr); 
    vr2_wk += j-1;
    ierr = VecCreateSeq(PETSC_COMM_SELF,j,&x);CHKERRQ(ierr);
 
    for (i=j-1; i>-1; i--){
      ierr = VecSetValue(x,i,*vr2_wk - *vr1_wk,INSERT_VALUES);CHKERRQ(ierr);
      vr2_wk--; vr1_wk--;
    }  
    ierr = VecNorm(x,NORM_1,&norm2);CHKERRQ(ierr);
    if (norm2<-tol || norm2>tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetRow()\n");CHKERRQ(ierr);
    } 
    ierr = VecDestroy(x);CHKERRQ(ierr);  
    ierr = MatRestoreRow(A,row,&J,&cols1,&vr1);CHKERRQ(ierr);
    ierr = MatRestoreRow(sA,row,&j,&cols2,&vr2);CHKERRQ(ierr);
    ierr = PetscFree(vr1);CHKERRQ(ierr); 
    ierr = PetscFree(vr2);CHKERRQ(ierr);

    /* Test GetSubMatrix() */
    /* get a submatrix consisting of every next block row and column of the original matrix */
    /* for symm. matrix, iscol=isrow. */
    ierr = PetscMalloc(n*sizeof(IS),&isrow);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(int),&ip_ptr);CHKERRQ(ierr);
    j = 0;
    for (n1=0; n1<mbs; n1 += 2){ /* n1: block row */
      for (i=0; i<bs; i++) ip_ptr[j++] = n1*bs + i;  
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF, j, ip_ptr, &isrow);CHKERRQ(ierr);
    /* ISView(isrow, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr); */
    
    ierr = MatGetSubMatrix(sA,isrow,isrow,PETSC_DECIDE,MAT_INITIAL_MATRIX,&sC);CHKERRQ(ierr);
    ierr = ISDestroy(isrow);CHKERRQ(ierr);
    ierr = PetscFree(ip_ptr);CHKERRQ(ierr);
    printf("sA =\n");
    ierr = MatView(sA,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    printf("submatrix of sA =\n");
    ierr = MatView(sC,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(sC);CHKERRQ(ierr);
  }  

  /* Test MatDiagonalScale(), MatGetDiagonal(), MatScale() */
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&rand);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);     
  ierr = VecDuplicate(x,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecSetRandom(rand,x);CHKERRQ(ierr);

  ierr = MatDiagonalScale(A,x,x);CHKERRQ(ierr);
  ierr = MatDiagonalScale(sA,x,x);CHKERRQ(ierr);

  ierr = MatGetDiagonal(A,s1);CHKERRQ(ierr);  
  ierr = MatGetDiagonal(sA,s2);CHKERRQ(ierr);
  ierr = VecNorm(s1,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = VecNorm(s2,NORM_1,&norm2);CHKERRQ(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatGetDiagonal() \n");CHKERRQ(ierr);
  } 

  ierr = MatScale(&alpha,A);CHKERRQ(ierr);
  ierr = MatScale(&alpha,sA);CHKERRQ(ierr);

  /* Test MatGetRowMax() */
  ierr = MatGetRowMax(A,s1);CHKERRQ(ierr);
  ierr = MatGetRowMax(sA,s2);CHKERRQ(ierr); 
  ierr = VecNorm(s1,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = VecNorm(s2,NORM_1,&norm2);CHKERRQ(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatGetRowMax() \n");CHKERRQ(ierr);
  } 

  /* Test MatMult(), MatMultAdd() */
  for (i=0; i<40; i++) { 
    ierr = VecSetRandom(rand,x);CHKERRQ(ierr);
    ierr = MatMult(A,x,s1);CHKERRQ(ierr);
    ierr = MatMult(sA,x,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_1,&norm1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_1,&norm2);CHKERRQ(ierr);
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMult(), MatDiagonalScale() or MatScale()\n");CHKERRQ(ierr);
    }
  }  

  for (i=0; i<40; i++) {
    ierr = VecSetRandom(rand,x);CHKERRQ(ierr);
    ierr = VecSetRandom(rand,y);CHKERRQ(ierr);
    ierr = MatMultAdd(A,x,y,s1);CHKERRQ(ierr);
    ierr = MatMultAdd(sA,x,y,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_1,&norm1);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_1,&norm2);CHKERRQ(ierr);
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMultAdd(), MatDiagonalScale() or MatScale() \n");CHKERRQ(ierr);
    } 
  }

  /* Test MatCholeskyFactor(), MatICCFactor() with natural ordering */
  ierr = MatGetOrdering(A,MATORDERING_NATURAL,&perm,&iscol);CHKERRQ(ierr); 
  ierr = ISDestroy(iscol);CHKERRQ(ierr);
  norm1 = tol;  
  inc   = bs;
  for (lf=-1; lf<10; lf += inc){   
    if (lf==-1) {  /* Cholesky factor */
      fill = 5.0;   
      ierr = MatCholeskyFactorSymbolic(sA,perm,fill,&sC);CHKERRQ(ierr); 
    } else {       /* incomplete Cholesky factor */
      fill          = 5.0;
      ierr = MatICCFactorSymbolic(sA,perm,fill,lf,&sC);CHKERRQ(ierr);
    }
    ierr = MatCholeskyFactorNumeric(sA,&sC);CHKERRQ(ierr);
    /* MatView(sC, PETSC_VIEWER_DRAW_WORLD); */
      
    ierr = MatMult(sA,x,b);CHKERRQ(ierr);
    ierr = MatSolve(sC,b,y);CHKERRQ(ierr);
    ierr = MatDestroy(sC);CHKERRQ(ierr);
      
    /* Check the error */
    ierr = VecAXPY(&neg_one,x,y);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
    /* printf("lf: %d, error: %g\n", lf,norm2); */
    if (10*norm1 < norm2 && lf-inc != -1){
      ierr = PetscPrintf(PETSC_COMM_SELF,"lf=%d, %d, Norm of error=%g, %g\n",lf-inc,lf,norm1,norm2);CHKERRQ(ierr); 
    } 
    norm1 = norm2;
    if (norm2 < tol && lf != -1) break;
  } 

  ierr = ISDestroy(perm);CHKERRQ(ierr);

  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(sA);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = VecDestroy(s1);CHKERRQ(ierr);
  ierr = VecDestroy(s2);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
