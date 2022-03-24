
static char help[] = "Tests various routines in MatSeqBAIJ format.\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B,C,D,Fact;
  Vec            xx,s1,s2,yy;
  PetscInt       m=45,rows[2],cols[2],bs=1,i,row,col,*idx,M;
  PetscScalar    rval,vals1[4],vals2[4];
  PetscRandom    rdm;
  IS             is1,is2;
  PetscReal      s1norm,s2norm,rnorm,tol = 1.e-4;
  PetscBool      flg;
  MatFactorInfo  info;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  /* Test MatSetValues() and MatGetValues() */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));
  M    = m*bs;
  CHKERRQ(MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,1,NULL,&A));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,M,M,15,NULL,&B));
  CHKERRQ(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,M,&xx));
  CHKERRQ(VecDuplicate(xx,&s1));
  CHKERRQ(VecDuplicate(xx,&s2));
  CHKERRQ(VecDuplicate(xx,&yy));

  /* For each row add atleast 15 elements */
  for (row=0; row<M; row++) {
    for (i=0; i<25*bs; i++) {
      CHKERRQ(PetscRandomGetValue(rdm,&rval));
      col  = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
      CHKERRQ(MatSetValues(A,1,&row,1,&col,&rval,INSERT_VALUES));
      CHKERRQ(MatSetValues(B,1,&row,1,&col,&rval,INSERT_VALUES));
    }
  }

  /* Now set blocks of values */
  for (i=0; i<20*bs; i++) {
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    cols[0]  = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    vals1[0] = rval;
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    cols[1]  = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    vals1[1] = rval;
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    rows[0]  = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    vals1[2] = rval;
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    rows[1]  = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    vals1[3] = rval;
    CHKERRQ(MatSetValues(A,2,rows,2,cols,vals1,INSERT_VALUES));
    CHKERRQ(MatSetValues(B,2,rows,2,cols,vals1,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Test MatNorm() */
  CHKERRQ(MatNorm(A,NORM_FROBENIUS,&s1norm));
  CHKERRQ(MatNorm(B,NORM_FROBENIUS,&s2norm));
  rnorm = PetscAbsReal(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_FROBENIUS()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
  }
  CHKERRQ(MatNorm(A,NORM_INFINITY,&s1norm));
  CHKERRQ(MatNorm(B,NORM_INFINITY,&s2norm));
  rnorm = PetscAbsReal(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_INFINITY()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
  }
  CHKERRQ(MatNorm(A,NORM_1,&s1norm));
  CHKERRQ(MatNorm(B,NORM_1,&s2norm));
  rnorm = PetscAbsReal(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_NORM_1()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
  }

  /* MatShift() */
  rval = 10*s1norm;
  CHKERRQ(MatShift(A,rval));
  CHKERRQ(MatShift(B,rval));

  /* Test MatTranspose() */
  CHKERRQ(MatTranspose(A,MAT_INPLACE_MATRIX,&A));
  CHKERRQ(MatTranspose(B,MAT_INPLACE_MATRIX,&B));

  /* Now do MatGetValues()  */
  for (i=0; i<30; i++) {
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    cols[0] = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    cols[1] = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    rows[0] = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    rows[1] = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    CHKERRQ(MatGetValues(A,2,rows,2,cols,vals1));
    CHKERRQ(MatGetValues(B,2,rows,2,cols,vals2));
    CHKERRQ(PetscArraycmp(vals1,vals2,4,&flg));
    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetValues bs = %" PetscInt_FMT "\n",bs));
    }
  }

  /* Test MatMult(), MatMultAdd() */
  for (i=0; i<40; i++) {
    CHKERRQ(VecSetRandom(xx,rdm));
    CHKERRQ(VecSet(s2,0.0));
    CHKERRQ(MatMult(A,xx,s1));
    CHKERRQ(MatMultAdd(A,xx,s2,s2));
    CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
    CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"MatMult not equal to MatMultAdd Norm1=%e Norm2=%e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
    }
  }

  /* Test MatMult() */
  CHKERRQ(MatMultEqual(A,B,10,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatMult()\n"));
  }

  /* Test MatMultAdd() */
  CHKERRQ(MatMultAddEqual(A,B,10,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatMultAdd()\n"));
  }

  /* Test MatMultTranspose() */
  CHKERRQ(MatMultTransposeEqual(A,B,10,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatMultTranspose()\n"));
  }

  /* Test MatMultTransposeAdd() */
  CHKERRQ(MatMultTransposeAddEqual(A,B,10,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatMultTransposeAdd()\n"));
  }

  /* Test MatMatMult() */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,M,40,NULL,&C));
  CHKERRQ(MatSetRandom(C,rdm));
  CHKERRQ(MatMatMult(A,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D));
  CHKERRQ(MatMatMultEqual(A,C,D,40,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatMatMult()\n"));
  }
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,M,40,NULL,&D));
  CHKERRQ(MatMatMult(A,C,MAT_REUSE_MATRIX,PETSC_DEFAULT,&D));
  CHKERRQ(MatMatMultEqual(A,C,D,40,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error: MatMatMult()\n"));
  }

  /* Do LUFactor() on both the matrices */
  CHKERRQ(PetscMalloc1(M,&idx));
  for (i=0; i<M; i++) idx[i] = i;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,M,idx,PETSC_COPY_VALUES,&is1));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,M,idx,PETSC_COPY_VALUES,&is2));
  CHKERRQ(PetscFree(idx));
  CHKERRQ(ISSetPermutation(is1));
  CHKERRQ(ISSetPermutation(is2));

  CHKERRQ(MatFactorInfoInitialize(&info));

  info.fill          = 2.0;
  info.dtcol         = 0.0;
  info.zeropivot     = 1.e-14;
  info.pivotinblocks = 1.0;

  if (bs < 4) {
    CHKERRQ(MatGetFactor(A,"petsc",MAT_FACTOR_LU,&Fact));
    CHKERRQ(MatLUFactorSymbolic(Fact,A,is1,is2,&info));
    CHKERRQ(MatLUFactorNumeric(Fact,A,&info));
    CHKERRQ(VecSetRandom(yy,rdm));
    CHKERRQ(MatForwardSolve(Fact,yy,xx));
    CHKERRQ(MatBackwardSolve(Fact,xx,s1));
    CHKERRQ(MatDestroy(&Fact));
    CHKERRQ(VecScale(s1,-1.0));
    CHKERRQ(MatMultAdd(A,s1,yy,yy));
    CHKERRQ(VecNorm(yy,NORM_2,&rnorm));
    if (rnorm<-tol || rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatForwardSolve/MatBackwardSolve - Norm1=%16.14e bs = %" PetscInt_FMT "\n",(double)rnorm,bs));
    }
  }

  CHKERRQ(MatLUFactor(B,is1,is2,&info));
  CHKERRQ(MatLUFactor(A,is1,is2,&info));

  /* Test MatSolveAdd() */
  for (i=0; i<10; i++) {
    CHKERRQ(VecSetRandom(xx,rdm));
    CHKERRQ(VecSetRandom(yy,rdm));
    CHKERRQ(MatSolveAdd(B,xx,yy,s2));
    CHKERRQ(MatSolveAdd(A,xx,yy,s1));
    CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
    CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveAdd - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
    }
  }

  /* Test MatSolveAdd() when x = A'b +x */
  for (i=0; i<10; i++) {
    CHKERRQ(VecSetRandom(xx,rdm));
    CHKERRQ(VecSetRandom(s1,rdm));
    CHKERRQ(VecCopy(s2,s1));
    CHKERRQ(MatSolveAdd(B,xx,s2,s2));
    CHKERRQ(MatSolveAdd(A,xx,s1,s1));
    CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
    CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveAdd(same) - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
    }
  }

  /* Test MatSolve() */
  for (i=0; i<10; i++) {
    CHKERRQ(VecSetRandom(xx,rdm));
    CHKERRQ(MatSolve(B,xx,s2));
    CHKERRQ(MatSolve(A,xx,s1));
    CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
    CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatSolve - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
    }
  }

  /* Test MatSolveTranspose() */
  if (bs < 8) {
    for (i=0; i<10; i++) {
      CHKERRQ(VecSetRandom(xx,rdm));
      CHKERRQ(MatSolveTranspose(B,xx,s2));
      CHKERRQ(MatSolveTranspose(A,xx,s1));
      CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
      CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveTranspose - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
      }
    }
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&D));
  CHKERRQ(VecDestroy(&xx));
  CHKERRQ(VecDestroy(&s1));
  CHKERRQ(VecDestroy(&s2));
  CHKERRQ(VecDestroy(&yy));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -mat_block_size {{1 2 3 4 5 6 7 8}}

TEST*/
