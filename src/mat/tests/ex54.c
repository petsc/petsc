
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-test_nd0",&test_nd0,NULL));

  /* Create a AIJ matrix A */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,m*bs,m*bs,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(A,MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A,PETSC_DEFAULT,NULL));
  PetscCall(MatMPIAIJSetPreallocation(A,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

  /* Create a BAIJ matrix B */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,m*bs,m*bs,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(B,MATBAIJ));
  PetscCall(MatSeqBAIJSetPreallocation(B,bs,PETSC_DEFAULT,NULL));
  PetscCall(MatMPIBAIJSetPreallocation(B,bs,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));

  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  PetscCall(MatGetSize(A,&M,&N));
  Mbs  = M/bs;

  PetscCall(PetscMalloc1(bs,&rows));
  PetscCall(PetscMalloc1(bs,&cols));
  PetscCall(PetscMalloc1(bs*bs,&vals));
  PetscCall(PetscMalloc1(M,&idx));

  /* Now set blocks of values */
  for (i=0; i<40*bs; i++) {
    PetscCall(PetscRandomGetValue(rdm,&rval));
    cols[0] = bs*(int)(PetscRealPart(rval)*Mbs);
    PetscCall(PetscRandomGetValue(rdm,&rval));
    rows[0] = rstart + bs*(int)(PetscRealPart(rval)*m);
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }

    for (j=0; j<bs*bs; j++) {
      PetscCall(PetscRandomGetValue(rdm,&rval));
      vals[j] = rval;
    }
    PetscCall(MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES));
    PetscCall(MatSetValues(B,bs,rows,bs,cols,vals,ADD_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* Test MatIncreaseOverlap() */
  PetscCall(PetscMalloc1(nd,&is1));
  PetscCall(PetscMalloc1(nd,&is2));

  if (rank == 0 && test_nd0) nd = 0; /* test case */

  for (i=0; i<nd; i++) {
    PetscCall(PetscRandomGetValue(rdm,&rval));
    sz   = (int)(PetscRealPart(rval)*m);
    for (j=0; j<sz; j++) {
      PetscCall(PetscRandomGetValue(rdm,&rval));
      idx[j*bs] = bs*(int)(PetscRealPart(rval)*Mbs);
      for (k=1; k<bs; k++) idx[j*bs+k] = idx[j*bs]+k;
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,PETSC_COPY_VALUES,is1+i));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,PETSC_COPY_VALUES,is2+i));
  }
  PetscCall(MatIncreaseOverlap(A,nd,is1,ov));
  PetscCall(MatIncreaseOverlap(B,nd,is2,ov));

  for (i=0; i<nd; ++i) {
    PetscCall(ISEqual(is1[i],is2[i],&flg));

    if (!flg) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"i=%" PetscInt_FMT ", flg=%d :bs=%" PetscInt_FMT " m=%" PetscInt_FMT " ov=%" PetscInt_FMT " nd=%" PetscInt_FMT " np=%d\n",i,flg,bs,m,ov,nd,size));
    }
  }

  for (i=0; i<nd; ++i) {
    PetscCall(ISSort(is1[i]));
    PetscCall(ISSort(is2[i]));
  }

  PetscCall(MatCreateSubMatrices(B,nd,is2,is2,MAT_INITIAL_MATRIX,&submatB));
  PetscCall(MatCreateSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA));

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    PetscCall(MatGetSize(submatA[i],&mm,&nn));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    PetscCall(VecDuplicate(xx,&s1));
    PetscCall(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      PetscCall(VecSetRandom(xx,rdm));
      PetscCall(MatMult(submatA[i],xx,s1));
      PetscCall(MatMult(submatB[i],xx,s2));
      PetscCall(VecNorm(s1,NORM_2,&s1norm));
      PetscCall(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d]Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",rank,(double)s1norm,(double)s2norm));
      }
    }
    PetscCall(VecDestroy(&xx));
    PetscCall(VecDestroy(&s1));
    PetscCall(VecDestroy(&s2));
  }

  /* Now test MatCreateSubmatrices with MAT_REUSE_MATRIX option */
  PetscCall(MatCreateSubMatrices(A,nd,is1,is1,MAT_REUSE_MATRIX,&submatA));
  PetscCall(MatCreateSubMatrices(B,nd,is2,is2,MAT_REUSE_MATRIX,&submatB));

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    PetscCall(MatGetSize(submatA[i],&mm,&nn));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,mm,&xx));
    PetscCall(VecDuplicate(xx,&s1));
    PetscCall(VecDuplicate(xx,&s2));
    for (j=0; j<3; j++) {
      PetscCall(VecSetRandom(xx,rdm));
      PetscCall(MatMult(submatA[i],xx,s1));
      PetscCall(MatMult(submatB[i],xx,s2));
      PetscCall(VecNorm(s1,NORM_2,&s1norm));
      PetscCall(VecNorm(s2,NORM_2,&s2norm));
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d]Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",rank,(double)s1norm,(double)s2norm));
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
  PetscCall(MatDestroySubMatrices(nd,&submatB));

  PetscCall(PetscFree(is1));
  PetscCall(PetscFree(is2));
  PetscCall(PetscFree(idx));
  PetscCall(PetscFree(rows));
  PetscCall(PetscFree(cols));
  PetscCall(PetscFree(vals));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFinalize());
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
