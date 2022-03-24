
static char help[] = "Tests MatIncreaseOverlap(), MatCreateSubMatrices() for parallel AIJ and BAIJ formats.\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B,*submatA,*submatB;
  PetscInt       bs=1,m=11,ov=1,i,j,k,*rows,*cols,nd=5,*idx,rstart,rend,sz,mm,nn,M,N,Mbs;
  PetscMPIInt    size,rank;
  PetscScalar    *vals,rval;
  IS             *is1,*is2;
  PetscRandom    rdm;
  Vec            xx,s1,s2;
  PetscReal      s1norm,s2norm,rnorm,tol = 100*PETSC_SMALL;
  PetscBool      flg,test_nd0=PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_nd0",&test_nd0,NULL));

  /* Create a AIJ matrix A */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,m*bs,m*bs,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(A,PETSC_DEFAULT,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(A,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

  /* Create a BAIJ matrix B */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,m*bs,m*bs,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(B,MATBAIJ));
  CHKERRQ(MatSeqBAIJSetPreallocation(B,bs,PETSC_DEFAULT,NULL));
  CHKERRQ(MatMPIBAIJSetPreallocation(B,bs,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));

  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  CHKERRQ(MatGetSize(A,&M,&N));
  Mbs  = M/bs;

  CHKERRQ(PetscMalloc1(bs,&rows));
  CHKERRQ(PetscMalloc1(bs,&cols));
  CHKERRQ(PetscMalloc1(bs*bs,&vals));
  CHKERRQ(PetscMalloc1(M,&idx));

  /* Now set blocks of values */
  for (i=0; i<40*bs; i++) {
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    cols[0] = bs*(int)(PetscRealPart(rval)*Mbs);
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    rows[0] = rstart + bs*(int)(PetscRealPart(rval)*m);
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }

    for (j=0; j<bs*bs; j++) {
      CHKERRQ(PetscRandomGetValue(rdm,&rval));
      vals[j] = rval;
    }
    CHKERRQ(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
    CHKERRQ(MatSetValues(B,bs,rows,bs,cols,vals,ADD_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Test MatIncreaseOverlap() */
  CHKERRQ(PetscMalloc1(nd,&is1));
  CHKERRQ(PetscMalloc1(nd,&is2));

  if (rank == 0 && test_nd0) nd = 0; /* test case */

  for (i=0; i<nd; i++) {
    CHKERRQ(PetscRandomGetValue(rdm,&rval));
    sz   = (int)(PetscRealPart(rval)*m);
    for (j=0; j<sz; j++) {
      CHKERRQ(PetscRandomGetValue(rdm,&rval));
      idx[j*bs] = bs*(int)(PetscRealPart(rval)*Mbs);
      for (k=1; k<bs; k++) idx[j*bs+k] = idx[j*bs]+k;
    }
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,PETSC_COPY_VALUES,is1+i));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,PETSC_COPY_VALUES,is2+i));
  }
  CHKERRQ(MatIncreaseOverlap(A,nd,is1,ov));
  CHKERRQ(MatIncreaseOverlap(B,nd,is2,ov));

  for (i=0; i<nd; ++i) {
    CHKERRQ(ISEqual(is1[i],is2[i],&flg));

    if (!flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"i=%" PetscInt_FMT ", flg=%d :bs=%" PetscInt_FMT " m=%" PetscInt_FMT " ov=%" PetscInt_FMT " nd=%" PetscInt_FMT " np=%d\n",i,flg,bs,m,ov,nd,size));
    }
  }

  for (i=0; i<nd; ++i) {
    CHKERRQ(ISSort(is1[i]));
    CHKERRQ(ISSort(is2[i]));
  }

  CHKERRQ(MatCreateSubMatrices(B,nd,is2,is2,MAT_INITIAL_MATRIX,&submatB));
  CHKERRQ(MatCreateSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA));

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    CHKERRQ(MatGetSize(submatA[i],&mm,&nn));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    CHKERRQ(VecDuplicate(xx,&s1));
    CHKERRQ(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      CHKERRQ(VecSetRandom(xx,rdm));
      CHKERRQ(MatMult(submatA[i],xx,s1));
      CHKERRQ(MatMult(submatB[i],xx,s2));
      CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
      CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d]Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",rank,(double)s1norm,(double)s2norm));
      }
    }
    CHKERRQ(VecDestroy(&xx));
    CHKERRQ(VecDestroy(&s1));
    CHKERRQ(VecDestroy(&s2));
  }

  /* Now test MatCreateSubmatrices with MAT_REUSE_MATRIX option */
  CHKERRQ(MatCreateSubMatrices(A,nd,is1,is1,MAT_REUSE_MATRIX,&submatA));
  CHKERRQ(MatCreateSubMatrices(B,nd,is2,is2,MAT_REUSE_MATRIX,&submatB));

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    CHKERRQ(MatGetSize(submatA[i],&mm,&nn));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    CHKERRQ(VecDuplicate(xx,&s1));
    CHKERRQ(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      CHKERRQ(VecSetRandom(xx,rdm));
      CHKERRQ(MatMult(submatA[i],xx,s1));
      CHKERRQ(MatMult(submatB[i],xx,s2));
      CHKERRQ(VecNorm(s1,NORM_2,&s1norm));
      CHKERRQ(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d]Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",rank,(double)s1norm,(double)s2norm));
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
  CHKERRQ(MatDestroySubMatrices(nd,&submatB));

  CHKERRQ(PetscFree(is1));
  CHKERRQ(PetscFree(is2));
  CHKERRQ(PetscFree(idx));
  CHKERRQ(PetscFree(rows));
  CHKERRQ(PetscFree(cols));
  CHKERRQ(PetscFree(vals));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: {{1 3}}
      args: -mat_block_size {{1 3 4 6 8}} -ov {{1 3}} -mat_size {{11 13}} -nd {{7}} ; done
      output_file: output/ex54.out

   test:
      suffix: 2
      args: -nd 2 -test_nd0
      output_file: output/ex54.out

   test:
      suffix: 3
      nsize: 3
      args: -nd 2 -test_nd0
      output_file: output/ex54.out

TEST*/
