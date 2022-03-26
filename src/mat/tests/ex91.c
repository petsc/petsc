
static char help[] = "Tests MatIncreaseOverlap(), MatCreateSubMatrices() for sequential MatSBAIJ format. Derived from ex51.c\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,Atrans,sA,*submatA,*submatsA;
  PetscInt       bs=1,m=43,ov=1,i,j,k,*rows,*cols,M,nd=5,*idx,mm,nn;
  PetscMPIInt    size;
  PetscScalar    *vals,rval,one=1.0;
  IS             *is1,*is2;
  PetscRandom    rand;
  Vec            xx,s1,s2;
  PetscReal      s1norm,s2norm,rnorm,tol = 10*PETSC_SMALL;
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));

  /* create a SeqBAIJ matrix A */
  M    = m*bs;
  PetscCall(MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,1,NULL,&A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rand));
  PetscCall(PetscRandomSetFromOptions(rand));

  PetscCall(PetscMalloc1(bs,&rows));
  PetscCall(PetscMalloc1(bs,&cols));
  PetscCall(PetscMalloc1(bs*bs,&vals));
  PetscCall(PetscMalloc1(M,&idx));

  /* Now set blocks of random values */
  /* first, set diagonal blocks as zero */
  for (j=0; j<bs*bs; j++) vals[j] = 0.0;
  for (i=0; i<m; i++) {
    cols[0] = i*bs; rows[0] = i*bs;
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }
    PetscCall(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
  }
  /* second, add random blocks */
  for (i=0; i<20*bs; i++) {
    PetscCall(PetscRandomGetValue(rand,&rval));
    cols[0] = bs*(int)(PetscRealPart(rval)*m);
    PetscCall(PetscRandomGetValue(rand,&rval));
    rows[0] = bs*(int)(PetscRealPart(rval)*m);
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }

    for (j=0; j<bs*bs; j++) {
      PetscCall(PetscRandomGetValue(rand,&rval));
      vals[j] = rval;
    }
    PetscCall(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* make A a symmetric matrix: A <- A^T + A */
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX, &Atrans));
  PetscCall(MatAXPY(A,one,Atrans,DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatDestroy(&Atrans));
  PetscCall(MatTranspose(A,MAT_INITIAL_MATRIX, &Atrans));
  PetscCall(MatEqual(A, Atrans, &flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"A+A^T is non-symmetric");
  PetscCall(MatDestroy(&Atrans));

  /* create a SeqSBAIJ matrix sA (= A) */
  PetscCall(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
  PetscCall(MatConvert(A,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&sA));

  /* Test sA==A through MatMult() */
  for (i=0; i<nd; i++) {
    PetscCall(MatGetSize(A,&mm,&nn));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    PetscCall(VecDuplicate(xx,&s1));
    PetscCall(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      PetscCall(VecSetRandom(xx,rand));
      PetscCall(MatMult(A,xx,s1));
      PetscCall(MatMult(sA,xx,s2));
      PetscCall(VecNorm(s1,NORM_2,&s1norm));
      PetscCall(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",(double)s1norm,(double)s2norm));
      }
    }
    PetscCall(VecDestroy(&xx));
    PetscCall(VecDestroy(&s1));
    PetscCall(VecDestroy(&s2));
  }

  /* Test MatIncreaseOverlap() */
  PetscCall(PetscMalloc1(nd,&is1));
  PetscCall(PetscMalloc1(nd,&is2));

  for (i=0; i<nd; i++) {
    PetscCall(PetscRandomGetValue(rand,&rval));
    size = (int)(PetscRealPart(rval)*m);
    for (j=0; j<size; j++) {
      PetscCall(PetscRandomGetValue(rand,&rval));
      idx[j*bs] = bs*(int)(PetscRealPart(rval)*m);
      for (k=1; k<bs; k++) idx[j*bs+k] = idx[j*bs]+k;
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,size*bs,idx,PETSC_COPY_VALUES,is1+i));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,size*bs,idx,PETSC_COPY_VALUES,is2+i));
  }
  /* for debugging */
  /*
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(MatView(sA,PETSC_VIEWER_STDOUT_SELF));
  */

  PetscCall(MatIncreaseOverlap(A,nd,is1,ov));
  PetscCall(MatIncreaseOverlap(sA,nd,is2,ov));

  for (i=0; i<nd; ++i) {
    PetscCall(ISSort(is1[i]));
    PetscCall(ISSort(is2[i]));
  }

  for (i=0; i<nd; ++i) {
    PetscCall(ISEqual(is1[i],is2[i],&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"i=%" PetscInt_FMT ", is1 != is2",i);
  }

  PetscCall(MatCreateSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA));
  PetscCall(MatCreateSubMatrices(sA,nd,is2,is2,MAT_INITIAL_MATRIX,&submatsA));

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    PetscCall(MatGetSize(submatA[i],&mm,&nn));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    PetscCall(VecDuplicate(xx,&s1));
    PetscCall(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      PetscCall(VecSetRandom(xx,rand));
      PetscCall(MatMult(submatA[i],xx,s1));
      PetscCall(MatMult(submatsA[i],xx,s2));
      PetscCall(VecNorm(s1,NORM_2,&s1norm));
      PetscCall(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",(double)s1norm,(double)s2norm));
      }
    }
    PetscCall(VecDestroy(&xx));
    PetscCall(VecDestroy(&s1));
    PetscCall(VecDestroy(&s2));
  }

  /* Now test MatCreateSubmatrices with MAT_REUSE_MATRIX option */
  PetscCall(MatCreateSubMatrices(A,nd,is1,is1,MAT_REUSE_MATRIX,&submatA));
  PetscCall(MatCreateSubMatrices(sA,nd,is2,is2,MAT_REUSE_MATRIX,&submatsA));

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    PetscCall(MatGetSize(submatA[i],&mm,&nn));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    PetscCall(VecDuplicate(xx,&s1));
    PetscCall(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      PetscCall(VecSetRandom(xx,rand));
      PetscCall(MatMult(submatA[i],xx,s1));
      PetscCall(MatMult(submatsA[i],xx,s2));
      PetscCall(VecNorm(s1,NORM_2,&s1norm));
      PetscCall(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",(double)s1norm,(double)s2norm));
      }
    }
    PetscCall(VecDestroy(&xx));
    PetscCall(VecDestroy(&s1));
    PetscCall(VecDestroy(&s2));
  }

  /* Free allocated memory */
  for (i=0; i<nd; ++i) {
    PetscCall(ISDestroy(&is1[i]));
    PetscCall(ISDestroy(&is2[i]));
  }
  PetscCall(MatDestroySubMatrices(nd,&submatA));
  PetscCall(MatDestroySubMatrices(nd,&submatsA));

  PetscCall(PetscFree(is1));
  PetscCall(PetscFree(is2));
  PetscCall(PetscFree(idx));
  PetscCall(PetscFree(rows));
  PetscCall(PetscFree(cols));
  PetscCall(PetscFree(vals));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&sA));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ov 2

TEST*/
