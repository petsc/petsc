
static char help[] = "Tests MatIncreaseOverlap(), MatGetSubMatrices() for parallel MatSBAIJ format.\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,Atrans,sA,*submatA,*submatsA;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       bs=1,mbs=11,ov=1,i,j,k,*rows,*cols,nd=5,*idx,rstart,rend,sz,mm,M,N,Mbs;
  PetscScalar    *vals,rval,one=1.0;
  IS             *is1,*is2;
  PetscRandom    rand;
  Vec            xx,s1,s2;
  PetscReal      s1norm,s2norm,rnorm,tol = 1.e-10;
  PetscTruth     flg;
  PetscLogStage  stages[2];

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mat_size",&mbs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ov",&ov,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nd",&nd,PETSC_NULL);CHKERRQ(ierr);

  ierr = MatCreateMPIBAIJ(PETSC_COMM_WORLD,bs,mbs*bs,mbs*bs,PETSC_DECIDE,PETSC_DECIDE,
                          PETSC_DEFAULT,PETSC_NULL,PETSC_DEFAULT,PETSC_NULL,&A);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);
  Mbs  = M/bs;

  ierr = PetscMalloc(bs*sizeof(PetscInt),&rows);CHKERRQ(ierr);
  ierr = PetscMalloc(bs*sizeof(PetscInt),&cols);CHKERRQ(ierr);
  ierr = PetscMalloc(bs*bs*sizeof(PetscScalar),&vals);CHKERRQ(ierr);
  ierr = PetscMalloc(M*sizeof(PetscScalar),&idx);CHKERRQ(ierr);

  /* Now set blocks of values */
  for (j=0; j<bs*bs; j++) vals[j] = 0.0;
  for (i=0; i<Mbs; i++){
    cols[0] = i*bs; rows[0] = i*bs; 
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }
    ierr = MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES);CHKERRQ(ierr);
  }
  /* second, add random blocks */
  for (i=0; i<20*bs; i++) {
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      cols[0] = bs*(PetscInt)(PetscRealPart(rval)*Mbs);
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      rows[0] = rstart + bs*(PetscInt)(PetscRealPart(rval)*mbs);
      for (j=1; j<bs; j++) {
        rows[j] = rows[j-1]+1;
        cols[j] = cols[j-1]+1;
      }

      for (j=0; j<bs*bs; j++) {
        ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
        vals[j] = rval;
      }
      ierr = MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  /* make A a symmetric matrix: A <- A^T + A */
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX, &Atrans);CHKERRQ(ierr);
  ierr = MatAXPY(A,one,Atrans,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); 
  ierr = MatDestroy(Atrans);CHKERRQ(ierr);
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX, &Atrans);
  ierr = MatEqual(A, Atrans, &flg);
  if (!flg) {
    SETERRQ(1,"A+A^T is non-symmetric");
  }
  ierr = MatDestroy(Atrans);CHKERRQ(ierr);
  /* ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* create a SeqSBAIJ matrix sA (= A) */
  ierr = MatConvert(A,MATMPISBAIJ,MAT_INITIAL_MATRIX,&sA);CHKERRQ(ierr); 
  /* ierr = MatView(sA,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Test sA==A through MatMult() */
  for (i=0; i<nd; i++) {
    ierr = MatGetLocalSize(A, &mm,PETSC_NULL);
    ierr = VecCreate(PETSC_COMM_WORLD,&xx);CHKERRQ(ierr);
    ierr = VecSetSizes(xx,mm,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(xx);CHKERRQ(ierr); 
    ierr = VecDuplicate(xx,&s1);CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&s2);CHKERRQ(ierr);
    for (j=0; j<3; j++) {
      ierr = VecSetRandom(xx,rand);CHKERRQ(ierr);
      ierr = MatMult(A,xx,s1);CHKERRQ(ierr);
      ierr = MatMult(sA,xx,s2);CHKERRQ(ierr);
      ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) { 
        ierr = PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",s1norm,s2norm);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(xx);CHKERRQ(ierr);
    ierr = VecDestroy(s1);CHKERRQ(ierr);
    ierr = VecDestroy(s2);CHKERRQ(ierr);
  } 
  
  /* Test MatIncreaseOverlap() */
  ierr = PetscMalloc(nd*sizeof(IS **),&is1);CHKERRQ(ierr);
  ierr = PetscMalloc(nd*sizeof(IS **),&is2);CHKERRQ(ierr);

  for (i=0; i<nd; i++) {
    ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
    sz = (PetscInt)((0.5 + 0.2*PetscRealPart(rval))*mbs); /* 0.5*mbs < sz < 0.7*mbs */
    sz /= (size*nd*10);
    /*
    if (!rank){
      ierr = PetscPrintf(PETSC_COMM_SELF," [%d] IS sz[%d]: %d\n",rank,i,sz);CHKERRQ(ierr);
    } 
    */
    for (j=0; j<sz; j++) {
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      idx[j*bs] = bs*(PetscInt)(PetscRealPart(rval)*Mbs);
      for (k=1; k<bs; k++) idx[j*bs+k] = idx[j*bs]+k;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,is1+i);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,sz*bs,idx,is2+i);CHKERRQ(ierr);
  }

  ierr = PetscLogStageRegister("MatOv_SBAIJ",&stages[0]);
  ierr = PetscLogStageRegister("MatOv_ BAIJ",&stages[1]);

  ierr = PetscLogStagePush(stages[0]);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(sA,nd,is2,ov);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogStagePush(stages[1]);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(A,nd,is1,ov);CHKERRQ(ierr); 
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  for (i=0; i<nd; ++i) { 
    ierr = ISEqual(is1[i],is2[i],&flg);CHKERRQ(ierr);
    if (!flg ){
      if (rank == size){
        ierr = ISSort(is1[i]);CHKERRQ(ierr);
        ISView(is1[i],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        ierr = ISSort(is2[i]);CHKERRQ(ierr); 
        ISView(is2[i],PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      } 
      SETERRQ1(1,"i=%D, is1 != is2",i);
    }
  }
 
  for (i=0; i<nd; ++i) { 
    ierr = ISSort(is1[i]);CHKERRQ(ierr);
    ierr = ISSort(is2[i]);CHKERRQ(ierr);
  }
  ierr = MatGetSubMatrices(sA,nd,is2,is2,MAT_INITIAL_MATRIX,&submatsA);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA);CHKERRQ(ierr);

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    ierr = MatGetSize(submatA[i],&mm,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,mm,&xx);CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&s1);CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&s2);CHKERRQ(ierr);
    for (j=0; j<3; j++) {
      ierr = VecSetRandom(xx,rand);CHKERRQ(ierr);
      ierr = MatMult(submatA[i],xx,s1);CHKERRQ(ierr);
      ierr = MatMult(submatsA[i],xx,s2);CHKERRQ(ierr);
      ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) { 
        ierr = PetscPrintf(PETSC_COMM_SELF,"[%d]Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",rank,s1norm,s2norm);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(xx);CHKERRQ(ierr);
    ierr = VecDestroy(s1);CHKERRQ(ierr);
    ierr = VecDestroy(s2);CHKERRQ(ierr);
  } 

  /* Now test MatGetSubmatrices with MAT_REUSE_MATRIX option */
  ierr = MatGetSubMatrices(A,nd,is1,is1,MAT_REUSE_MATRIX,&submatA);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(sA,nd,is2,is2,MAT_REUSE_MATRIX,&submatsA);CHKERRQ(ierr);

  /* Test MatMult() */
  for (i=0; i<nd; i++) {
    ierr = MatGetSize(submatA[i],&mm,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,mm,&xx);CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&s1);CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&s2);CHKERRQ(ierr);
    for (j=0; j<3; j++) {
      ierr = VecSetRandom(xx,rand);CHKERRQ(ierr);
      ierr = MatMult(submatA[i],xx,s1);CHKERRQ(ierr);
      ierr = MatMult(submatsA[i],xx,s2);CHKERRQ(ierr);
      ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRQ(ierr);
      ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRQ(ierr);
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) { 
        ierr = PetscPrintf(PETSC_COMM_SELF,"[%d]Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",rank,s1norm,s2norm);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(xx);CHKERRQ(ierr);
    ierr = VecDestroy(s1);CHKERRQ(ierr);
    ierr = VecDestroy(s2);CHKERRQ(ierr);
  } 

  /* Free allocated memory */
  for (i=0; i<nd; ++i) { 
    ierr = ISDestroy(is1[i]);CHKERRQ(ierr);
    ierr = ISDestroy(is2[i]);CHKERRQ(ierr);
    ierr = MatDestroy(submatA[i]);CHKERRQ(ierr);
    ierr = MatDestroy(submatsA[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(submatA);CHKERRQ(ierr);
  ierr = PetscFree(submatsA);CHKERRQ(ierr);
  ierr = PetscFree(is1);CHKERRQ(ierr);
  ierr = PetscFree(is2);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(sA);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
