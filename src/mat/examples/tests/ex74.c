/*$Id: ex74.c,v 1.12 2000/07/14 14:45:36 hzhang Exp hzhang $*/

static char help[] = "Tests the vatious sequential routines in MatSBAIJ format.\n";

#include "petscmat.h"

/* extern int MatReorderingSeqSBAIJ(Mat,IS); */

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec     x,y,b,s1,s2;      
  Mat     A;           /* linear system matrix */ 
  Mat     sA,sC;         /* symmetric part of the matrices */ 

  int     n = 16,bs=1,nz=3,prob=1;
  Scalar  neg_one = -1.0,four=4.0,value[3],alpha=0.1;
  int     ierr,i,j,col[3],size,block, row,I,J,n1,*ip_ptr;
  IS      ip, isrow, iscol;
  PetscRandom rand;

  PetscTruth       reorder=PETSC_FALSE,getrow=PETSC_FALSE;
  MatInfo          minfo1,minfo2;
  MatILUInfo       info;
  
  int      lf; /* level of fill for ilu */
  Scalar   *vr1,*vr2,*vr1_wk,*vr2_wk;
  int      *cols1,*cols2,mbs;
  double   norm1,norm2,tol=1.e-10;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  if (size != 1) SETERRA(1,0,"This is a uniprocessor example only!");
  ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mat_size",&n,PETSC_NULL);CHKERRA(ierr);

  mbs = n/bs;
  if (mbs*bs != n) SETERRA(1,0,"Number rows/cols must be divisible by bs");
  ierr=MatCreateSeqBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,PETSC_NULL, &A); CHKERRA(ierr);
  ierr=MatCreateSeqSBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,PETSC_NULL, &sA); CHKERRA(ierr);

  /* Test MatGetOwnershipRange() */
  ierr = MatGetOwnershipRange(A,&I,&J);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(sA,&i,&j);CHKERRA(ierr);
  if (i-I || j-J){
    PetscPrintf(PETSC_COMM_SELF,"Error: MatGetOwnershipRange() in MatSBAIJ format\n");
  }

  /* Assemble matrix */
  if (bs == 1){
    ierr = OptionsGetInt(PETSC_NULL,"-test_problem",&prob,PETSC_NULL);CHKERRA(ierr);
    if (prob == 1){ /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
        ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 = sqrt(n); 
      if (n1*n1 - n) SETERRQ(PETSC_ERR_ARG_WRONG,0,"sqrt(n) must be a positive interger!"); 
      printf("n1 = %d\n",n1);
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          I = j + n1*i;
          if (i>0)   {
            J = I - n1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr); 
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);
          }
          if (i<n1-1) {
            J = I + n1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);
          }
          if (j>0)   {
            J = I - 1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);
          }
          if (j<n1-1) {
            J = I + 1; 
            ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);
            ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);
          }
          ierr = MatSetValues(A,1,&I,1,&I,&four,INSERT_VALUES);CHKERRA(ierr);
          ierr = MatSetValues(sA,1,&I,1,&I,&four,INSERT_VALUES);CHKERRA(ierr);
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
        ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
        ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);    
      }
      i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
      value[0]=-1.0; value[1]=4.0;  
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr); 

      i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
      value[0]=4.0; value[1] = -1.0; 
      ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);  
    }
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(n/bs-1)*bs; i++){
      col[0]=i+bs;
      ierr = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(sA,1,&i,1,col,value,INSERT_VALUES); CHKERRA(ierr);
      col[0]=i; row=i+bs;
      ierr = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES); CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  /* PetscPrintf(PETSC_COMM_SELF,"\n The Matrix: \n");
  MatView(A, VIEWER_DRAW_WORLD);
  MatView(A, VIEWER_STDOUT_WORLD); */ 

  ierr = MatAssemblyBegin(sA,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(sA,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);  
  /* PetscPrintf(PETSC_COMM_SELF,"\n Symmetric Part of Matrix: \n");
  MatView(sA, VIEWER_DRAW_WORLD); 
  MatView(sA, VIEWER_STDOUT_WORLD); 
  */

  /* Test MatNorm() */
  ierr = MatNorm(A,NORM_FROBENIUS,&norm1);CHKERRA(ierr); 
  ierr = MatNorm(sA,NORM_FROBENIUS,&norm2);CHKERRA(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol){ 
    PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm(), fnorm1-fnorm2=%16.14e\n,norm1");
  }
  ierr = MatNorm(A,NORM_INFINITY,&norm1); CHKERRA(ierr);
  ierr = MatNorm(sA,NORM_INFINITY,&norm2); CHKERRA(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol){ 
    PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm(), inf_norm1-inf_norm2=%16.14e\n,norm1");
  }
  
  /* Test MatGetInfo(), MatGetSize(), MatGetBlockSize() */
  ierr = MatGetInfo(A,MAT_LOCAL,&minfo1);CHKERRA(ierr);
  ierr = MatGetInfo(sA,MAT_LOCAL,&minfo2);CHKERRA(ierr);
  /*
  printf("matrix nonzeros (BAIJ format) = %d, allocated nonzeros= %d\n", (int)minfo1.nz_used,(int)minfo1.nz_allocated); 
  printf("matrix nonzeros(SBAIJ format) = %d, allocated nonzeros= %d\n", (int)minfo2.nz_used,(int)minfo2.nz_allocated); 
  */
  i = minfo1.nz_used - minfo2.nz_used; 
  j = minfo1.nz_allocated - minfo2.nz_allocated;
  if (i<0 || j<0) {
    PetscPrintf(PETSC_COMM_SELF,"Error: MatGetInfo()\n");
  }

  ierr = MatGetSize(A,&I,&J);CHKERRA(ierr);
  ierr = MatGetSize(sA,&i,&j);CHKERRA(ierr); 
  if (i-I || j-J) {
    PetscPrintf(PETSC_COMM_SELF,"Error: MatGetSize()\n");
  }
 
  ierr = MatGetBlockSize(A, &I); CHKERRA(ierr);
  ierr = MatGetBlockSize(sA, &i); CHKERRA(ierr);
  if (i-I){
    PetscPrintf(PETSC_COMM_SELF,"Error: MatGetBlockSize()\n");
  }

  /* Test MatGetRow() */
  if (getrow){
    row = n/2; 
    vr1 =  (Scalar*)PetscMalloc(n*sizeof(Scalar));CHKPTRQ(vr1); 
    vr1_wk = vr1;  
    vr2 =  (Scalar*)PetscMalloc(n*sizeof(Scalar));CHKPTRQ(vr2); 
    vr2_wk = vr2;
    ierr = MatGetRow(A,row,&J,&cols1,&vr1); CHKERRA(ierr); 
    vr1_wk += J-1;
    ierr = MatGetRow(sA,row,&j,&cols2,&vr2); CHKERRA(ierr); 
    vr2_wk += j-1;
    ierr = VecCreateSeq(PETSC_COMM_SELF,j,&x);CHKERRA(ierr);
 
    for (i=j-1; i>-1; i--){
      VecSetValue(x,i,*vr2_wk - *vr1_wk, INSERT_VALUES);
      vr2_wk--; vr1_wk--;
    }  
    ierr = VecNorm(x,NORM_1,&norm2); CHKERRA(ierr);
    if (norm2<-tol || norm2>tol) {
      PetscPrintf(PETSC_COMM_SELF,"Error: MatGetRow()\n");
    } 
    ierr = VecDestroy(x);CHKERRA(ierr);  
    ierr = MatRestoreRow( A,row,&J,&cols1,&vr1); CHKERRA(ierr);
    ierr = MatRestoreRow(sA,row,&j,&cols2,&vr2); CHKERRA(ierr);
    ierr = PetscFree(vr1); CHKERRA(ierr); 
    ierr = PetscFree(vr2); CHKERRA(ierr);

    /* Test GetSubMatrix() */
    /* get a submatrix consisting of every next block row and column of the original matrix */
    /* for symm. matrix, iscol=isrow. */
    ip_ptr = (int*)PetscMalloc(n*sizeof(int)); CHKERRA(ierr);
    j = 0;
    for (n1=0; n1<mbs; n1 += 2){ /* n1: block row */
      for (i=0; i<bs; i++) ip_ptr[j++] = n1*bs + i;  
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF, j, ip_ptr, &isrow); CHKERRA(ierr);
    ierr = PetscFree(ip_ptr); CHKERRA(ierr); 
    /* ISView(isrow, VIEWER_STDOUT_SELF); CHKERRA(ierr); */
    
    ierr = MatGetSubMatrix(sA,isrow,isrow,PETSC_DECIDE,MAT_INITIAL_MATRIX,&sC);
    CHKERRA(ierr);
    ierr = ISDestroy(isrow);CHKERRA(ierr);
    printf("sA =\n");
    ierr = MatView(sA,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
    printf("submatrix of sA =\n");
    ierr = MatView(sC,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  }  

  /* Test MatDiagonalScale(), MatGetDiagonal(), MatScale() */
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&rand);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRA(ierr);     
  ierr = VecDuplicate(x,&s1);CHKERRA(ierr);
  ierr = VecDuplicate(x,&s2);CHKERRA(ierr);
  ierr = VecDuplicate(x,&y);CHKERRA(ierr);
  ierr = VecDuplicate(x,&b);CHKERRA(ierr);

  ierr = VecSetRandom(rand,x);CHKERRA(ierr);
  ierr = MatDiagonalScale(A,x,x);CHKERRA(ierr);
  ierr = MatDiagonalScale(sA,x,x);CHKERRA(ierr);

  ierr = MatGetDiagonal(A,s1); CHKERRA(ierr);  
  ierr = MatGetDiagonal(sA,s2); CHKERRA(ierr);
  ierr = VecNorm(s1,NORM_1,&norm1);CHKERRA(ierr);
  ierr = VecNorm(s2,NORM_1,&norm2);CHKERRA(ierr);
  norm1 -= norm2;
  if (norm1<-tol || norm1>tol) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatGetDiagonal() \n");CHKERRA(ierr);
  } 

  ierr = MatScale(&alpha,A);CHKERRA(ierr);
  ierr = MatScale(&alpha,sA);CHKERRA(ierr);

  /* Test MatMult(), MatMultAdd() */
  for (i=0; i<40; i++) { 
    ierr = VecSetRandom(rand,x);CHKERRA(ierr);
    ierr = MatMult(A,x,s1);CHKERRA(ierr);
    ierr = MatMult(sA,x,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_1,&norm1);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_1,&norm2);CHKERRA(ierr);
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMult(), MatDiagonalScale() or MatScale()\n");CHKERRA(ierr);
    }
  }  

  for (i=0; i<40; i++) {
    ierr = VecSetRandom(rand,x);CHKERRA(ierr);
    ierr = VecSetRandom(rand,y);CHKERRA(ierr);
    ierr = MatMultAdd(A,x,y,s1);CHKERRA(ierr);
    ierr = MatMultAdd(sA,x,y,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_1,&norm1);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_1,&norm2);CHKERRA(ierr);
    norm1 -= norm2;
    if (norm1<-tol || norm1>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMultAdd(), MatDiagonalScale() or MatScale() \n");CHKERRA(ierr);
    } 
  }

  /* Test MatReordering(), MatLUFactor(), MatILUFactor() */
  ierr = MatGetOrdering(A,MATORDERING_NATURAL,&isrow,&iscol);CHKERRA(ierr); 
  ip = isrow;

  if(reorder){
    ISGetIndices(ip,&ip_ptr);      
    i = ip_ptr[1]; ip_ptr[1] = ip_ptr[n-2]; ip_ptr[n-2] = i; 
    i = ip_ptr[0]; ip_ptr[0] = ip_ptr[n-1]; ip_ptr[n-1] = i; 
    ierr= ISRestoreIndices(ip,&ip_ptr); CHKERRA(ierr);

    ierr = MatReorderingSeqSBAIJ(sA, ip); CHKERRA(ierr);  
    /* ierr = ISView(ip, VIEWER_STDOUT_SELF); CHKERRA(ierr); 
       ierr = MatView(sA,VIEWER_DRAW_SELF); CHKERRA(ierr); */
  }

  if (bs>1) SETERRQ(PETSC_ERR_ARG_WRONG,0,"bs>1 is not supported for Cholesky Factor in SBAIJ matrix format");
    
  for (lf=-1; lf<10; lf++){   
    if (lf==-1) {  /* LU */
      ierr = MatLUFactorSymbolic(sA,ip,ip,PETSC_NULL,&sC);CHKERRA(ierr);
    } else {       /* ILU */
      info.levels        = lf;
      info.fill          = 2.0;
      info.diagonal_fill = 0;
      /* ierr = OptionsHasName(PETSC_NULL,"-lf",&flg);CHKERRA(ierr);*/
      ierr = MatILUFactorSymbolic(sA,ip,ip,&info,&sC);CHKERRA(ierr);
    }
    ierr = MatLUFactorNumeric(sA,&sC);CHKERRA(ierr);
    /* MatView(sC, VIEWER_DRAW_WORLD); */

    ierr = MatMult(sA,x,b);CHKERRA(ierr);
    ierr = MatSolve(sC,b,y);CHKERRA(ierr);
  
    /* Check the error */
    ierr = VecAXPY(&neg_one,x,y);CHKERRA(ierr);
    ierr = VecNorm(y,NORM_2,&norm1);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"lf=%d, Norm of error=%g\n",lf,norm1);CHKERRA(ierr);
  } 
  
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = MatDestroy(sA);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);
  ierr = VecDestroy(s1);CHKERRA(ierr);
  ierr = VecDestroy(s2);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
