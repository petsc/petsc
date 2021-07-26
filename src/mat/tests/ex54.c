
static char help[] = "Tests MatIncreaseOverlap(), MatCreateSubMatrices() for parallel AIJ and BAIJ formats.\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B,*submatA,*submatB;
  PetscInt       bs=1,m=11,ov=1,i,j,k,*rows,*cols,nd=5,*idx,rstart,rend,sz,mm,nn,M,N,Mbs;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscScalar    *vals,rval;
  IS             *is1,*is2;
  PetscRandom    rdm;
  Vec            xx,s1,s2;
  PetscReal      s1norm,s2norm,rnorm,tol = 100*PETSC_SMALL;
  PetscBool      flg,test_nd0=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ov",&ov,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nd",&nd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_nd0",&test_nd0,NULL);CHKERRQ(ierr);

  /* Create a AIJ matrix A */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m*bs,m*bs,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,PETSC_DEFAULT,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

  /* Create a BAIJ matrix B */
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m*bs,m*bs,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATBAIJ);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(B,bs,PETSC_DEFAULT,NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(B,bs,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  Mbs  = M/bs;

  ierr = PetscMalloc1(bs,&rows);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs*bs,&vals);CHKERRQ(ierr);
  ierr = PetscMalloc1(M,&idx);CHKERRQ(ierr);

  /* Now set blocks of values */
  for (i=0; i<40*bs; i++) {
    ierr    = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    cols[0] = bs*(int)(PetscRealPart(rval)*Mbs);
    ierr    = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    rows[0] = rstart + bs*(int)(PetscRealPart(rval)*m);
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }

    for (j=0; j<bs*bs; j++) {
      ierr    = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
      vals[j] = rval;
    }
    ierr = MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(B,bs,rows,bs,cols,vals,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Test MatIncreaseOverlap() */
  ierr = PetscMalloc1(nd,&is1);CHKERRQ(ierr);
  ierr = PetscMalloc1(nd,&is2);CHKERRQ(ierr);

  if (rank == 0 && test_nd0) nd = 0; /* test case */

  for (i=0; i<nd; i++) {
    ierr = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
    sz   = (int)(PetscRealPart(rval)*m);
    for (j=0; j<sz; j++) {
      ierr      = PetscRandomGetValue(rdm,&rval);CHKERRQ(ierr);
      idx[j*bs] = bs*(int)(PetscRealPart(rval)*Mbs);
      for (k=1; k<bs; k++) idx[j*bs+k] = idx[j*bs]+k;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,PETSC_COPY_VALUES,is1+i);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,PETSC_COPY_VALUES,is2+i);CHKERRQ(ierr);
  }
  ierr = MatIncreaseOverlap(A,nd,is1,ov);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(B,nd,is2,ov);CHKERRQ(ierr);

  for (i=0; i<nd; ++i) {
    ierr = ISEqual(is1[i],is2[i],&flg);CHKERRQ(ierr);

    if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"i=%D, flg=%d :bs=%D m=%D ov=%D nd=%D np=%D\n",i,flg,bs,m,ov,nd,size);CHKERRQ(ierr);
    }
  }

  for (i=0; i<nd; ++i) {
    ierr = ISSort(is1[i]);CHKERRQ(ierr);
    ierr = ISSort(is2[i]);CHKERRQ(ierr);
  }

  ierr = MatCreateSubMatrices(B,nd,is2,is2,MAT_INITIAL_MATRIX,&submatB);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA);CHKERRQ(ierr);

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    ierr = MatGetSize(submatA[i],&mm,&nn);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,mm,&xx);CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&s1);CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&s2);CHKERRQ(ierr);
    for (j=0; j<3; j++) {
      ierr  = VecSetRandom(xx,rdm);CHKERRQ(ierr);
      ierr  = MatMult(submatA[i],xx,s1);CHKERRQ(ierr);
      ierr  = MatMult(submatB[i],xx,s2);CHKERRQ(ierr);
      ierr  = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
      ierr  = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"[%d]Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",rank,s1norm,s2norm);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(&xx);CHKERRQ(ierr);
    ierr = VecDestroy(&s1);CHKERRQ(ierr);
    ierr = VecDestroy(&s2);CHKERRQ(ierr);
  }

  /* Now test MatCreateSubmatrices with MAT_REUSE_MATRIX option */
  ierr = MatCreateSubMatrices(A,nd,is1,is1,MAT_REUSE_MATRIX,&submatA);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(B,nd,is2,is2,MAT_REUSE_MATRIX,&submatB);CHKERRQ(ierr);

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    ierr = MatGetSize(submatA[i],&mm,&nn);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,mm,&xx);CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&s1);CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&s2);CHKERRQ(ierr);
    for (j=0; j<3; j++) {
      ierr  = VecSetRandom(xx,rdm);CHKERRQ(ierr);
      ierr  = MatMult(submatA[i],xx,s1);CHKERRQ(ierr);
      ierr  = MatMult(submatB[i],xx,s2);CHKERRQ(ierr);
      ierr  = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
      ierr  = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"[%d]Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",rank,s1norm,s2norm);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(&xx);CHKERRQ(ierr);
    ierr = VecDestroy(&s1);CHKERRQ(ierr);
    ierr = VecDestroy(&s2);CHKERRQ(ierr);
  }

  /* Free allocated memory */
  for (i=0; i<nd; ++i) {
    ierr = ISDestroy(&is1[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&is2[i]);CHKERRQ(ierr);
  }
  ierr = MatDestroySubMatrices(nd,&submatA);CHKERRQ(ierr);
  ierr = MatDestroySubMatrices(nd,&submatB);CHKERRQ(ierr);

  ierr = PetscFree(is1);CHKERRQ(ierr);
  ierr = PetscFree(is2);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
