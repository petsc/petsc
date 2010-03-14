
static char help[] = "Tests the vatious routines in MatSeqBAIJ format.\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat             A,B;
  Vec             xx,s1,s2,yy;
  PetscErrorCode ierr;
  PetscInt        m=45,rows[2],cols[2],bs=1,i,row,col,*idx,M; 
  PetscScalar     rval,vals1[4],vals2[4];
  PetscRandom     rdm;
  IS              is1,is2;
  PetscReal       s1norm,s2norm,rnorm,tol = 1.e-4;
  PetscTruth      flg;
  MatFactorInfo   info;
  
  PetscInitialize(&argc,&args,(char *)0,help);
  
  /* Test MatSetValues() and MatGetValues() */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_size",&m,PETSC_NULL);CHKERRQ(ierr);
  M    = m*bs;
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,1,PETSC_NULL,&A);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,M,M,15,PETSC_NULL,&B);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,M,&xx);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&s1);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&s2);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&yy);CHKERRQ(ierr);
  
  /* For each row add atleast 15 elements */
  for (row=0; row<M; row++) {
    for (i=0; i<25*bs; i++) {
      ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
      col  = (PetscInt)(PetscRealPart(rval)*M);
      ierr = MatSetValues(A,1,&row,1,&col,&rval,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(B,1,&row,1,&col,&rval,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  
  /* Now set blocks of values */
  for (i=0; i<20*bs; i++) {
    ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    cols[0] = (PetscInt)(PetscRealPart(rval)*M);
    vals1[0] = rval;
    ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    cols[1] = (PetscInt)(PetscRealPart(rval)*M);
    vals1[1] = rval;
    ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    rows[0] = (PetscInt)(PetscRealPart(rval)*M);
    vals1[2] = rval;
    ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    rows[1] = (PetscInt)(PetscRealPart(rval)*M);
    vals1[3] = rval;
    ierr = MatSetValues(A,2,rows,2,cols,vals1,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(B,2,rows,2,cols,vals1,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  /* Test MatNorm() */
  ierr = MatNorm(A,NORM_FROBENIUS,&s1norm);CHKERRQ(ierr);
  ierr = MatNorm(B,NORM_FROBENIUS,&s2norm);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(s2norm-s1norm)/s2norm;
  if ( rnorm>tol ) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_FROBENIUS()- NormA=%16.14e NormB=%16.14e bs = %D\n",s1norm,s2norm,bs);CHKERRQ(ierr);
  }
  ierr = MatNorm(A,NORM_INFINITY,&s1norm);CHKERRQ(ierr);
  ierr = MatNorm(B,NORM_INFINITY,&s2norm);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(s2norm-s1norm)/s2norm;
  if ( rnorm>tol ) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_INFINITY()- NormA=%16.14e NormB=%16.14e bs = %D\n",s1norm,s2norm,bs);CHKERRQ(ierr);
  }
  ierr = MatNorm(A,NORM_1,&s1norm);CHKERRQ(ierr);
  ierr = MatNorm(B,NORM_1,&s2norm);CHKERRQ(ierr);
  rnorm = PetscAbsScalar(s2norm-s1norm)/s2norm;
  if ( rnorm>tol ) { 
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_NORM_1()- NormA=%16.14e NormB=%16.14e bs = %D\n",s1norm,s2norm,bs);CHKERRQ(ierr);
  }

  /* MatShift() */
  rval = 10*s1norm;
  ierr = MatShift(A,rval);CHKERRQ(ierr);
  ierr = MatShift(B,rval);CHKERRQ(ierr);

  /* Test MatTranspose() */
  ierr = MatTranspose(A,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatTranspose(B,MAT_REUSE_MATRIX,&B);CHKERRQ(ierr);

  /* Now do MatGetValues()  */
  for (i=0; i<30; i++) {
    ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    cols[0] = (PetscInt)(PetscRealPart(rval)*M);
    ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    cols[1] = (PetscInt)(PetscRealPart(rval)*M);
    ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    rows[0] = (PetscInt)(PetscRealPart(rval)*M);
    ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    rows[1] = (PetscInt)(PetscRealPart(rval)*M);
    ierr = MatGetValues(A,2,rows,2,cols,vals1);CHKERRQ(ierr);
    ierr = MatGetValues(B,2,rows,2,cols,vals2);CHKERRQ(ierr);
    ierr = PetscMemcmp(vals1,vals2,4*sizeof(PetscScalar),&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatGetValues bs = %D\n",bs);CHKERRQ(ierr);
    }
  }
  
  /* Test MatMult(), MatMultAdd() */
  for (i=0; i<40; i++) {
    ierr = VecSetRandom(xx,rdm);CHKERRQ(ierr);
    ierr = VecSet(s2,0.0);CHKERRQ(ierr);
    ierr = MatMult(A,xx,s1);CHKERRQ(ierr);
    ierr = MatMultAdd(A,xx,s2,s2);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"MatMult not equal to MatMultAdd Norm1=%e Norm2=%e bs = %D\n",s1norm,s2norm,bs);CHKERRQ(ierr);
    }
  }

  /* Test MatMult() */
  ierr = MatMultEqual(A,B,10,&flg);CHKERRQ(ierr);
  if (!flg){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMult()\n");CHKERRQ(ierr);
  }
  
  /* Test MatMultAdd() */
  ierr = MatMultAddEqual(A,B,10,&flg);CHKERRQ(ierr);
  if (!flg){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMultAdd()\n");CHKERRQ(ierr);
  }
  
  /* Test MatMultTranspose() */
  ierr = MatMultTransposeEqual(A,B,10,&flg);CHKERRQ(ierr);
  if (!flg){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMultTranspose()\n");CHKERRQ(ierr);
  }

  /* Test MatMultTransposeAdd() */
  ierr = MatMultTransposeAddEqual(A,B,10,&flg);CHKERRQ(ierr);
  if (!flg){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: MatMultTransposeAdd()\n");CHKERRQ(ierr);
  }

  /* Do LUFactor() on both the matrices */
  ierr = PetscMalloc(M*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  for (i=0; i<M; i++) idx[i] = i;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,M,idx,&is1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,M,idx,&is2);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = ISSetPermutation(is1);CHKERRQ(ierr);
  ierr = ISSetPermutation(is2);CHKERRQ(ierr);

  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);   
  info.fill      = 2.0;
  info.dtcol     = 0.0; 
  info.zeropivot = 1.e-14; 
  info.pivotinblocks = 1.0; 
  ierr = MatLUFactor(B,is1,is2,&info);CHKERRQ(ierr);
  ierr = MatLUFactor(A,is1,is2,&info);CHKERRQ(ierr);
  
  /* Test MatSolveAdd() */
  for (i=0; i<10; i++) {
    ierr = VecSetRandom(xx,rdm);CHKERRQ(ierr);
    ierr = VecSetRandom(yy,rdm);CHKERRQ(ierr);
    ierr = MatSolveAdd(B,xx,yy,s2);CHKERRQ(ierr);
    ierr = MatSolveAdd(A,xx,yy,s1);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveAdd - Norm1=%16.14e Norm2=%16.14e bs = %D\n",s1norm,s2norm,bs);CHKERRQ(ierr);
    } 
  }
  
  /* Test MatSolveAdd() when x = A'b +x */
  for (i=0; i<10; i++) {
    ierr = VecSetRandom(xx,rdm);CHKERRQ(ierr);
    ierr = VecSetRandom(s1,rdm);CHKERRQ(ierr);
    ierr = VecCopy(s2,s1);CHKERRQ(ierr);
    ierr = MatSolveAdd(B,xx,s2,s2);CHKERRQ(ierr);
    ierr = MatSolveAdd(A,xx,s1,s1);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveAdd(same) - Norm1=%16.14e Norm2=%16.14e bs = %D\n",s1norm,s2norm,bs);CHKERRQ(ierr);
    } 
  }
  
  /* Test MatSolve() */
  for (i=0; i<10; i++) {
    ierr = VecSetRandom(xx,rdm);CHKERRQ(ierr);
    ierr = MatSolve(B,xx,s2);CHKERRQ(ierr);
    ierr = MatSolve(A,xx,s1);CHKERRQ(ierr);
    ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
    ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) { 
      ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatSolve - Norm1=%16.14e Norm2=%16.14e bs = %D\n",s1norm,s2norm,bs);CHKERRQ(ierr);
    } 
  }
  
  /* Test MatSolveTranspose() */
  if (bs < 8) {
    for (i=0; i<10; i++) {
      ierr = VecSetRandom(xx,rdm);CHKERRQ(ierr);
      ierr = MatSolveTranspose(B,xx,s2);CHKERRQ(ierr);
      ierr = MatSolveTranspose(A,xx,s1);CHKERRQ(ierr);
      ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) { 
        ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveTranspose - Norm1=%16.14e Norm2=%16.14e bs = %D\n",s1norm,s2norm,bs);CHKERRQ(ierr);
      }
    } 
  }

  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = VecDestroy(xx);CHKERRQ(ierr);
  ierr = VecDestroy(s1);CHKERRQ(ierr);
  ierr = VecDestroy(s2);CHKERRQ(ierr);
  ierr = VecDestroy(yy);CHKERRQ(ierr);
  ierr = ISDestroy(is1);CHKERRQ(ierr);
  ierr = ISDestroy(is2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rdm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
