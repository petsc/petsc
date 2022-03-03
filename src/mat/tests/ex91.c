
static char help[] = "Tests MatIncreaseOverlap(), MatCreateSubMatrices() for sequential MatSBAIJ format. Derived from ex51.c\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,Atrans,sA,*submatA,*submatsA;
  PetscInt       bs=1,m=43,ov=1,i,j,k,*rows,*cols,M,nd=5,*idx,mm,nn;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscScalar    *vals,rval,one=1.0;
  IS             *is1,*is2;
  PetscRandom    rand;
  Vec            xx,s1,s2;
  PetscReal      s1norm,s2norm,rnorm,tol = 10*PETSC_SMALL;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));

  /* create a SeqBAIJ matrix A */
  M    = m*bs;
  CHKERRQ(MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,1,NULL,&A));
  CHKERRQ(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));

  CHKERRQ(PetscMalloc1(bs,&rows));
  CHKERRQ(PetscMalloc1(bs,&cols));
  CHKERRQ(PetscMalloc1(bs*bs,&vals));
  CHKERRQ(PetscMalloc1(M,&idx));

  /* Now set blocks of random values */
  /* first, set diagonal blocks as zero */
  for (j=0; j<bs*bs; j++) vals[j] = 0.0;
  for (i=0; i<m; i++) {
    cols[0] = i*bs; rows[0] = i*bs;
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }
    CHKERRQ(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
  }
  /* second, add random blocks */
  for (i=0; i<20*bs; i++) {
    CHKERRQ(PetscRandomGetValue(rand,&rval));
    cols[0] = bs*(int)(PetscRealPart(rval)*m);
    CHKERRQ(PetscRandomGetValue(rand,&rval));
    rows[0] = bs*(int)(PetscRealPart(rval)*m);
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }

    for (j=0; j<bs*bs; j++) {
      CHKERRQ(PetscRandomGetValue(rand,&rval));
      vals[j] = rval;
    }
    CHKERRQ(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* make A a symmetric matrix: A <- A^T + A */
  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX, &Atrans));
  CHKERRQ(MatAXPY(A,one,Atrans,DIFFERENT_NONZERO_PATTERN));
  CHKERRQ(MatDestroy(&Atrans));
  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX, &Atrans));
  CHKERRQ(MatEqual(A, Atrans, &flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"A+A^T is non-symmetric");
  CHKERRQ(MatDestroy(&Atrans));

  /* create a SeqSBAIJ matrix sA (= A) */
  CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(MatConvert(A,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&sA));

  /* Test sA==A through MatMult() */
  for (i=0; i<nd; i++) {
    CHKERRQ(MatGetSize(A,&mm,&nn));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    CHKERRQ(VecDuplicate(xx,&s1));
    CHKERRQ(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      CHKERRQ(VecSetRandom(xx,rand));
      CHKERRQ(MatMult(A,xx,s1));
      CHKERRQ(MatMult(sA,xx,s2));
      CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
      CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",(double)s1norm,(double)s2norm));
      }
    }
    CHKERRQ(VecDestroy(&xx));
    CHKERRQ(VecDestroy(&s1));
    CHKERRQ(VecDestroy(&s2));
  }

  /* Test MatIncreaseOverlap() */
  CHKERRQ(PetscMalloc1(nd,&is1));
  CHKERRQ(PetscMalloc1(nd,&is2));

  for (i=0; i<nd; i++) {
    CHKERRQ(PetscRandomGetValue(rand,&rval));
    size = (int)(PetscRealPart(rval)*m);
    for (j=0; j<size; j++) {
      CHKERRQ(PetscRandomGetValue(rand,&rval));
      idx[j*bs] = bs*(int)(PetscRealPart(rval)*m);
      for (k=1; k<bs; k++) idx[j*bs+k] = idx[j*bs]+k;
    }
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,size*bs,idx,PETSC_COPY_VALUES,is1+i));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,size*bs,idx,PETSC_COPY_VALUES,is2+i));
  }
  /* for debugging */
  /*
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(MatView(sA,PETSC_VIEWER_STDOUT_SELF));
  */

  CHKERRQ(MatIncreaseOverlap(A,nd,is1,ov));
  CHKERRQ(MatIncreaseOverlap(sA,nd,is2,ov));

  for (i=0; i<nd; ++i) {
    CHKERRQ(ISSort(is1[i]));
    CHKERRQ(ISSort(is2[i]));
  }

  for (i=0; i<nd; ++i) {
    CHKERRQ(ISEqual(is1[i],is2[i],&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"i=%" PetscInt_FMT ", is1 != is2",i);
  }

  CHKERRQ(MatCreateSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA));
  CHKERRQ(MatCreateSubMatrices(sA,nd,is2,is2,MAT_INITIAL_MATRIX,&submatsA));

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    CHKERRQ(MatGetSize(submatA[i],&mm,&nn));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    CHKERRQ(VecDuplicate(xx,&s1));
    CHKERRQ(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      CHKERRQ(VecSetRandom(xx,rand));
      CHKERRQ(MatMult(submatA[i],xx,s1));
      CHKERRQ(MatMult(submatsA[i],xx,s2));
      CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
      CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",(double)s1norm,(double)s2norm));
      }
    }
    CHKERRQ(VecDestroy(&xx));
    CHKERRQ(VecDestroy(&s1));
    CHKERRQ(VecDestroy(&s2));
  }

  /* Now test MatCreateSubmatrices with MAT_REUSE_MATRIX option */
  CHKERRQ(MatCreateSubMatrices(A,nd,is1,is1,MAT_REUSE_MATRIX,&submatA));
  CHKERRQ(MatCreateSubMatrices(sA,nd,is2,is2,MAT_REUSE_MATRIX,&submatsA));

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    CHKERRQ(MatGetSize(submatA[i],&mm,&nn));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    CHKERRQ(VecDuplicate(xx,&s1));
    CHKERRQ(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      CHKERRQ(VecSetRandom(xx,rand));
      CHKERRQ(MatMult(submatA[i],xx,s1));
      CHKERRQ(MatMult(submatsA[i],xx,s2));
      CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
      CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",(double)s1norm,(double)s2norm));
      }
    }
    CHKERRQ(VecDestroy(&xx));
    CHKERRQ(VecDestroy(&s1));
    CHKERRQ(VecDestroy(&s2));
  }

  /* Free allocated memory */
  for (i=0; i<nd; ++i) {
    CHKERRQ(ISDestroy(&is1[i]));
    CHKERRQ(ISDestroy(&is2[i]));
  }
  CHKERRQ(MatDestroySubMatrices(nd,&submatA));
  CHKERRQ(MatDestroySubMatrices(nd,&submatsA));

  CHKERRQ(PetscFree(is1));
  CHKERRQ(PetscFree(is2));
  CHKERRQ(PetscFree(idx));
  CHKERRQ(PetscFree(rows));
  CHKERRQ(PetscFree(cols));
  CHKERRQ(PetscFree(vals));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&sA));
  CHKERRQ(PetscRandomDestroy(&rand));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -ov 2

TEST*/
