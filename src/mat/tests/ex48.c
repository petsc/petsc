
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  /* Test MatSetValues() and MatGetValues() */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));
  M    = m*bs;
  PetscCall(MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,1,NULL,&A));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,M,M,15,NULL,&B));
  PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,M,&xx));
  PetscCall(VecDuplicate(xx,&s1));
  PetscCall(VecDuplicate(xx,&s2));
  PetscCall(VecDuplicate(xx,&yy));

  /* For each row add atleast 15 elements */
  for (row=0; row<M; row++) {
    for (i=0; i<25*bs; i++) {
      PetscCall(PetscRandomGetValue(rdm,&rval));
      col  = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
      PetscCall(MatSetValues(A,1,&row,1,&col,&rval,INSERT_VALUES));
      PetscCall(MatSetValues(B,1,&row,1,&col,&rval,INSERT_VALUES));
    }
  }

  /* Now set blocks of values */
  for (i=0; i<20*bs; i++) {
    PetscCall(PetscRandomGetValue(rdm,&rval));
    cols[0]  = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    vals1[0] = rval;
    PetscCall(PetscRandomGetValue(rdm,&rval));
    cols[1]  = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    vals1[1] = rval;
    PetscCall(PetscRandomGetValue(rdm,&rval));
    rows[0]  = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    vals1[2] = rval;
    PetscCall(PetscRandomGetValue(rdm,&rval));
    rows[1]  = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    vals1[3] = rval;
    PetscCall(MatSetValues(A,2,rows,2,cols,vals1,INSERT_VALUES));
    PetscCall(MatSetValues(B,2,rows,2,cols,vals1,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Test MatNorm() */
  PetscCall(MatNorm(A,NORM_FROBENIUS,&s1norm));
  PetscCall(MatNorm(B,NORM_FROBENIUS,&s2norm));
  rnorm = PetscAbsReal(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_FROBENIUS()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
  }
  PetscCall(MatNorm(A,NORM_INFINITY,&s1norm));
  PetscCall(MatNorm(B,NORM_INFINITY,&s2norm));
  rnorm = PetscAbsReal(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_INFINITY()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
  }
  PetscCall(MatNorm(A,NORM_1,&s1norm));
  PetscCall(MatNorm(B,NORM_1,&s2norm));
  rnorm = PetscAbsReal(s2norm-s1norm)/s2norm;
  if (rnorm>tol) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatNorm_NORM_1()- NormA=%16.14e NormB=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
  }

  /* MatShift() */
  rval = 10*s1norm;
  PetscCall(MatShift(A,rval));
  PetscCall(MatShift(B,rval));

  /* Test MatTranspose() */
  PetscCall(MatTranspose(A,MAT_INPLACE_MATRIX,&A));
  PetscCall(MatTranspose(B,MAT_INPLACE_MATRIX,&B));

  /* Now do MatGetValues()  */
  for (i=0; i<30; i++) {
    PetscCall(PetscRandomGetValue(rdm,&rval));
    cols[0] = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    PetscCall(PetscRandomGetValue(rdm,&rval));
    cols[1] = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    PetscCall(PetscRandomGetValue(rdm,&rval));
    rows[0] = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    PetscCall(PetscRandomGetValue(rdm,&rval));
    rows[1] = PetscMin(M-1,(PetscInt)(PetscRealPart(rval)*M));
    PetscCall(MatGetValues(A,2,rows,2,cols,vals1));
    PetscCall(MatGetValues(B,2,rows,2,cols,vals2));
    PetscCall(PetscArraycmp(vals1,vals2,4,&flg));
    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatGetValues bs = %" PetscInt_FMT "\n",bs));
    }
  }

  /* Test MatMult(), MatMultAdd() */
  for (i=0; i<40; i++) {
    PetscCall(VecSetRandom(xx,rdm));
    PetscCall(VecSet(s2,0.0));
    PetscCall(MatMult(A,xx,s1));
    PetscCall(MatMultAdd(A,xx,s2,s2));
    PetscCall(VecNorm(s1,NORM_2,&s1norm));
    PetscCall(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"MatMult not equal to MatMultAdd Norm1=%e Norm2=%e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
    }
  }

  /* Test MatMult() */
  PetscCall(MatMultEqual(A,B,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatMult()\n"));
  }

  /* Test MatMultAdd() */
  PetscCall(MatMultAddEqual(A,B,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatMultAdd()\n"));
  }

  /* Test MatMultTranspose() */
  PetscCall(MatMultTransposeEqual(A,B,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatMultTranspose()\n"));
  }

  /* Test MatMultTransposeAdd() */
  PetscCall(MatMultTransposeAddEqual(A,B,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatMultTransposeAdd()\n"));
  }

  /* Test MatMatMult() */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,M,40,NULL,&C));
  PetscCall(MatSetRandom(C,rdm));
  PetscCall(MatMatMult(A,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&D));
  PetscCall(MatMatMultEqual(A,C,D,40,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatMatMult()\n"));
  }
  PetscCall(MatDestroy(&D));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,M,40,NULL,&D));
  PetscCall(MatMatMult(A,C,MAT_REUSE_MATRIX,PETSC_DEFAULT,&D));
  PetscCall(MatMatMultEqual(A,C,D,40,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error: MatMatMult()\n"));
  }

  /* Do LUFactor() on both the matrices */
  PetscCall(PetscMalloc1(M,&idx));
  for (i=0; i<M; i++) idx[i] = i;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,M,idx,PETSC_COPY_VALUES,&is1));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,M,idx,PETSC_COPY_VALUES,&is2));
  PetscCall(PetscFree(idx));
  PetscCall(ISSetPermutation(is1));
  PetscCall(ISSetPermutation(is2));

  PetscCall(MatFactorInfoInitialize(&info));

  info.fill          = 2.0;
  info.dtcol         = 0.0;
  info.zeropivot     = 1.e-14;
  info.pivotinblocks = 1.0;

  if (bs < 4) {
    PetscCall(MatGetFactor(A,"petsc",MAT_FACTOR_LU,&Fact));
    PetscCall(MatLUFactorSymbolic(Fact,A,is1,is2,&info));
    PetscCall(MatLUFactorNumeric(Fact,A,&info));
    PetscCall(VecSetRandom(yy,rdm));
    PetscCall(MatForwardSolve(Fact,yy,xx));
    PetscCall(MatBackwardSolve(Fact,xx,s1));
    PetscCall(MatDestroy(&Fact));
    PetscCall(VecScale(s1,-1.0));
    PetscCall(MatMultAdd(A,s1,yy,yy));
    PetscCall(VecNorm(yy,NORM_2,&rnorm));
    if (rnorm<-tol || rnorm>tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error:MatForwardSolve/MatBackwardSolve - Norm1=%16.14e bs = %" PetscInt_FMT "\n",(double)rnorm,bs));
    }
  }

  PetscCall(MatLUFactor(B,is1,is2,&info));
  PetscCall(MatLUFactor(A,is1,is2,&info));

  /* Test MatSolveAdd() */
  for (i=0; i<10; i++) {
    PetscCall(VecSetRandom(xx,rdm));
    PetscCall(VecSetRandom(yy,rdm));
    PetscCall(MatSolveAdd(B,xx,yy,s2));
    PetscCall(MatSolveAdd(A,xx,yy,s1));
    PetscCall(VecNorm(s1,NORM_2,&s1norm));
    PetscCall(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveAdd - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
    }
  }

  /* Test MatSolveAdd() when x = A'b +x */
  for (i=0; i<10; i++) {
    PetscCall(VecSetRandom(xx,rdm));
    PetscCall(VecSetRandom(s1,rdm));
    PetscCall(VecCopy(s2,s1));
    PetscCall(MatSolveAdd(B,xx,s2,s2));
    PetscCall(MatSolveAdd(A,xx,s1,s1));
    PetscCall(VecNorm(s1,NORM_2,&s1norm));
    PetscCall(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveAdd(same) - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
    }
  }

  /* Test MatSolve() */
  for (i=0; i<10; i++) {
    PetscCall(VecSetRandom(xx,rdm));
    PetscCall(MatSolve(B,xx,s2));
    PetscCall(MatSolve(A,xx,s1));
    PetscCall(VecNorm(s1,NORM_2,&s1norm));
    PetscCall(VecNorm(s2,NORM_2,&s2norm));
    rnorm = s2norm-s1norm;
    if (rnorm<-tol || rnorm>tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error:MatSolve - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
    }
  }

  /* Test MatSolveTranspose() */
  if (bs < 8) {
    for (i=0; i<10; i++) {
      PetscCall(VecSetRandom(xx,rdm));
      PetscCall(MatSolveTranspose(B,xx,s2));
      PetscCall(MatSolveTranspose(A,xx,s1));
      PetscCall(VecNorm(s1,NORM_2,&s1norm));
      PetscCall(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error:MatSolveTranspose - Norm1=%16.14e Norm2=%16.14e bs = %" PetscInt_FMT "\n",(double)s1norm,(double)s2norm,bs));
      }
    }
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&xx));
  PetscCall(VecDestroy(&s1));
  PetscCall(VecDestroy(&s2));
  PetscCall(VecDestroy(&yy));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -mat_block_size {{1 2 3 4 5 6 7 8}}

TEST*/
