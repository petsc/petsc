/*$Id: ex48.c,v 1.9 1999/10/13 20:37:41 bsmith Exp bsmith $*/

static char help[] = 
"Tests the vatious routines in MatBAIJ format.\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         A,B;
  Vec         xx,s1,s2,yy;
  int         m=45,ierr,flg,rows[2],cols[2],bs=1,i,row,col,*idx,M;
  Scalar      rval,vals1[4],vals2[4],zero=0.0;
  PetscRandom rand;
  IS          is1,is2;
  double      s1norm,s2norm,rnorm,tol = 1.e-10;
  PetscTruth  flag;
  
  PetscInitialize(&argc,&args,(char *)0,help);
  
  /* Test MatSetValues() and MatGetValues() */
  ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mat_size",&m,&flg);CHKERRA(ierr);
  M    = m*bs;
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,1,PETSC_NULL,&A);CHKERRA(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,M,M,15,PETSC_NULL, &B);CHKERRA(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&rand);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,M,&xx);CHKERRA(ierr);
  ierr = VecDuplicate(xx,&s1);CHKERRA(ierr);
  ierr = VecDuplicate(xx,&s2);CHKERRA(ierr);
  ierr = VecDuplicate(xx,&yy);CHKERRA(ierr);
  
  /* For each row add atleast 15 elements */
  for (row=0; row<M; row++ ) {
    for ( i=0; i<25*bs; i++) {
      ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
      col  = (int)(PetscReal(rval)*M);
      ierr = MatSetValues(A,1,&row,1,&col,&rval,ADD_VALUES);CHKERRA(ierr);
      ierr = MatSetValues(B,1,&row,1,&col,&rval,ADD_VALUES);CHKERRA(ierr);
    }
  }
  
  /* Now set blocks of values */
  for ( i=0; i<20*bs; i++ ) {
    ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
    cols[0] = (int)(PetscReal(rval)*M);
    vals1[0] = rval;
    ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
    cols[1] = (int)(PetscReal(rval)*M);
    vals1[1] = rval;
    ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
    rows[0] = (int)(PetscReal(rval)*M);
    vals1[2] = rval;
    ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
    rows[1] = (int)(PetscReal(rval)*M);
    vals1[3] = rval;
    ierr = MatSetValues(A,2,rows,2,cols,vals1,ADD_VALUES);CHKERRA(ierr);
    ierr = MatSetValues(B,2,rows,2,cols,vals1,ADD_VALUES);CHKERRA(ierr);
  }
  
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  
  /* Test MatNorm() */
  ierr = MatNorm(A,NORM_FROBENIUS,&s1norm);CHKERRA(ierr);
  ierr = MatNorm(B,NORM_FROBENIUS,&s2norm);CHKERRA(ierr);
  rnorm = s2norm-s1norm;
  if (rnorm<-tol || rnorm>tol) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm()- Norm1=%16.14e Norm2=%16.14e \n",s1norm,s2norm);CHKERRA(ierr);
  }
  /* MatScale() */
  rval = 10*s1norm;
  ierr = MatShift(&rval,A);CHKERRA(ierr);
  ierr = MatShift(&rval,B);CHKERRA(ierr);
  
  /* Test MatTranspose() */
  ierr = MatTranspose(A,PETSC_NULL);CHKERRA(ierr);
  ierr = MatTranspose(B,PETSC_NULL);CHKERRA(ierr);
  
  
  /* Now do MatGetValues()  */
  for ( i=0; i<30; i++ ) {
    ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
    cols[0] = (int)(PetscReal(rval)*M);
    ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
    cols[1] = (int)(PetscReal(rval)*M);
    ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
    rows[0] = (int)(PetscReal(rval)*M);
    ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
    rows[1] = (int)(PetscReal(rval)*M);
    ierr = MatGetValues(A,2,rows,2,cols,vals1);CHKERRA(ierr);
    ierr = MatGetValues(B,2,rows,2,cols,vals2);CHKERRA(ierr);
    ierr = PetscMemcmp(vals1,vals2,4*sizeof(Scalar),&flag);CHKERRA(ierr);
    if (!flag) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetValues\n");CHKERRA(ierr);
    }
  }
  
  /* Test MatMult(), MatMultAdd() */
  for ( i=0; i<40; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = VecSet(&zero,s2);CHKERRA(ierr);
    ierr = MatMult(A,xx,s1);CHKERRA(ierr);
    ierr = MatMultAdd(A,xx,s2,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"MatMult notequalto MatMultAdd Norm1=%e Norm2=%e \n",s1norm,s2norm);CHKERRA(ierr);
    }
  }

  /* Test MatMult() */
  for ( i=0; i<40; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = MatMult(A,xx,s1);CHKERRA(ierr);
    ierr = MatMult(B,xx,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",s1norm,s2norm);CHKERRA(ierr);
    }
  } 
  
  /* Test MatMultAdd() */
  for ( i=0; i<40; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = VecSetRandom(rand,yy);CHKERRA(ierr);
    ierr = MatMultAdd(A,xx,yy,s1);CHKERRA(ierr);
    ierr = MatMultAdd(B,xx,yy,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMultAdd - Norm1=%16.14e Norm2=%16.14e\n",s1norm,s2norm);CHKERRA(ierr);
    } 
  }
  
  /* Test MatMultTrans() */
  for ( i=0; i<40; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = MatMultTrans(A,xx,s1);CHKERRA(ierr);
    ierr = MatMultTrans(B,xx,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMultTrans - Norm1=%16.14e Norm2=%16.14e\n",s1norm,s2norm);CHKERRA(ierr);
    } 
  }
  /* Test MatMultTransAdd() */
  for ( i=0; i<40; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = VecSetRandom(rand,yy);CHKERRA(ierr);
    ierr = MatMultTransAdd(A,xx,yy,s1);CHKERRA(ierr);
    ierr = MatMultTransAdd(B,xx,yy,s2);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMultTransAdd - Norm1=%16.14e Norm2=%16.14e\n",s1norm,s2norm);CHKERRA(ierr);
    } 
  }
  
  
  /* Do LUFactor() on both the matrices */
  idx  = (int *)PetscMalloc(M*sizeof(int));CHKPTRA(idx);
  for ( i=0; i<M; i++ ) idx[i] = i;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,M,idx,&is1);CHKERRA(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,M,idx,&is2);CHKERRA(ierr);
  ierr = PetscFree(idx);CHKERRA(ierr);
  ierr = ISSetPermutation(is1);CHKERRA(ierr);
  ierr = ISSetPermutation(is2);CHKERRA(ierr);
  ierr = MatLUFactor(B,is1,is2,3);CHKERRA(ierr);
  ierr = MatLUFactor(A,is1,is2,3);CHKERRA(ierr);
  
  
  /* Test MatSolveAdd() */
  for ( i=0; i<40; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = VecSetRandom(rand,yy);CHKERRA(ierr);
    ierr = MatSolveAdd(B,xx,yy,s2);CHKERRA(ierr);
    ierr = MatSolveAdd(A,xx,yy,s1);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveAdd - Norm1=%16.14e Norm2=%16.14e\n",s1norm,s2norm);CHKERRA(ierr);
    } 
  }
  
  /* Test MatSolveAdd() when x = A'b +x */
  for ( i=0; i<40; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = VecSetRandom(rand,s1);CHKERRA(ierr);
    ierr = VecCopy(s2,s1);CHKERRA(ierr);
    ierr = MatSolveAdd(B,xx,s2,s2);CHKERRA(ierr);
    ierr = MatSolveAdd(A,xx,s1,s1);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveAdd(same) - Norm1=%16.14e Norm2=%16.14e\n",s1norm,s2norm);CHKERRA(ierr);
    } 
  }
  
  /* Test MatSolve() */
  for ( i=0; i<40; i++) {
    ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
    ierr = MatSolve(B,xx,s2);CHKERRA(ierr);
    ierr = MatSolve(A,xx,s1);CHKERRA(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatSolve - Norm1=%16.14e Norm2=%16.14e\n",s1norm,s2norm);CHKERRA(ierr);
    } 
  }
  
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);
  ierr = VecDestroy(xx);CHKERRA(ierr);
  ierr = VecDestroy(s1);CHKERRA(ierr);
  ierr = VecDestroy(s2);CHKERRA(ierr);
  ierr = VecDestroy(yy);CHKERRA(ierr);
  ierr = ISDestroy(is1); CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
