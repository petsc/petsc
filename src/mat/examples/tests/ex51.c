#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex51.c,v 1.6 1999/03/19 21:19:59 bsmith Exp balay $";
#endif

static char help[] = 
"Tests MatIncreaseOverlap(), MatGetSubMatrices() for MatBAIJ format.\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         A,B,*submatA, *submatB;
  int         bs=1,m=43,ov=1,i,j,k,*rows,*cols,ierr,flg,M,nd=5,*idx,size,mm,nn;
  Scalar      *vals,rval;
  IS          *is1,*is2;
  PetscRandom rand;
  Vec         xx,s1,s2;
  double      s1norm,s2norm,rnorm,tol = 1.e-10;

  PetscInitialize(&argc,&args,(char *)0,help);
 

  ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mat_size",&m,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-ov",&ov,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-nd",&nd,&flg);CHKERRA(ierr);
  M    = m*bs;

  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,1,PETSC_NULL,&A);CHKERRA(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,M,M,15,PETSC_NULL, &B);CHKERRA(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&rand);CHKERRA(ierr);

  rows  = (int *) PetscMalloc(bs*sizeof(int));CHKPTRA(rows);
  cols  = (int *) PetscMalloc(bs*sizeof(int));CHKPTRA(cols);
  vals  = (Scalar *) PetscMalloc(bs*bs*sizeof(Scalar));CHKPTRA(vals);
  idx   = (int *) PetscMalloc(M*sizeof(Scalar));CHKPTRA(idx);

  /* Now set blocks of values */
  for ( i=0; i<20*bs; i++ ) {
      ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
      cols[0] = bs*(int)(PetscReal(rval)*m);
      ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
      rows[0] = bs*(int)(PetscReal(rval)*m);
      for ( j=1; j<bs; j++ ) {
        rows[j] = rows[j-1]+1;
        cols[j] = cols[j-1]+1;
      }

      for ( j=0; j<bs*bs; j++) {
        ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
        vals[j] = rval;
      }
      ierr = MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES);CHKERRA(ierr);
      ierr = MatSetValues(B,bs,rows,bs,cols,vals,ADD_VALUES);CHKERRA(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

    /* Test MatIncreaseOverlap() */
  is1 = (IS *) PetscMalloc( nd*sizeof(IS **) );CHKPTRA(is1);
  is2 = (IS *) PetscMalloc( nd*sizeof(IS **) );CHKPTRA(is2);

  
  for ( i=0; i<nd; i++) {
    ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
    size = (int)(PetscReal(rval)*m);
    for (j=0; j<size; j++ ) {
      ierr = PetscRandomGetValue(rand,&rval);CHKERRA(ierr);
      idx[j*bs] = bs*(int)(PetscReal(rval)*m);
      for ( k=1; k<bs; k++) idx[j*bs+k] = idx[j*bs]+k;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size*bs,idx,is1+i);CHKERRA(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size*bs,idx,is2+i);CHKERRA(ierr);
  }
  ierr = MatIncreaseOverlap(A,nd,is1,ov);CHKERRA(ierr);
  ierr = MatIncreaseOverlap(B,nd,is2,ov);CHKERRA(ierr);

  for (i=0; i<nd; ++i) { 
    ierr = ISEqual(is1[i],is2[i],(PetscTruth*)&flg);CHKERRA(ierr);
    PetscPrintf(PETSC_COMM_SELF,"i=%d, flg =%d\n",i,flg);
  }

  for (i=0; i<nd; ++i) { 
    ierr = ISSort(is1[i]);CHKERRQ(ierr);
    ierr = ISSort(is2[i]);CHKERRQ(ierr);
  }
  
  ierr = MatGetSubMatrices(A,nd,is1,is1,MAT_INITIAL_MATRIX,&submatA);CHKERRA(ierr);
  ierr = MatGetSubMatrices(B,nd,is2,is2,MAT_INITIAL_MATRIX,&submatB);CHKERRA(ierr);

  /* Test MatMult() */
  for ( i=0; i<nd; i++) {
    ierr = MatGetSize(submatA[i],&mm,&nn);CHKERRA(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,mm,&xx);CHKERRA(ierr);
    ierr = VecDuplicate(xx,&s1);CHKERRA(ierr);
    ierr = VecDuplicate(xx,&s2);CHKERRA(ierr);
    for ( j=0; j<3; j++ ) {
      ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
      ierr = MatMult(submatA[i],xx,s1);CHKERRA(ierr);
      ierr = MatMult(submatB[i],xx,s2);CHKERRA(ierr);
      ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
      ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) { 
        PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",s1norm,s2norm);  
      }
    }
    VecDestroy(xx);
    VecDestroy(s1);
    VecDestroy(s2);
  } 
  /* Now test MatGetSubmatrices with MAT_REUSE_MATRIX option */
  ierr = MatGetSubMatrices(A,nd,is1,is1,MAT_REUSE_MATRIX,&submatA);CHKERRA(ierr);
  ierr = MatGetSubMatrices(B,nd,is2,is2,MAT_REUSE_MATRIX,&submatB);CHKERRA(ierr);
  
  /* Test MatMult() */
  for ( i=0; i<nd; i++) {
    ierr = MatGetSize(submatA[i],&mm,&nn);CHKERRA(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,mm,&xx);CHKERRA(ierr);
    ierr = VecDuplicate(xx,&s1);CHKERRA(ierr);
    ierr = VecDuplicate(xx,&s2);CHKERRA(ierr);
    for ( j=0; j<3; j++ ) {
      ierr = VecSetRandom(rand,xx);CHKERRA(ierr);
      ierr = MatMult(submatA[i],xx,s1);CHKERRA(ierr);
      ierr = MatMult(submatB[i],xx,s2);CHKERRA(ierr);
      ierr = VecNorm(s1,NORM_2,&s1norm);CHKERRA(ierr);
      ierr = VecNorm(s2,NORM_2,&s2norm);CHKERRA(ierr);
      rnorm = s2norm-s1norm;
      if (rnorm<-tol || rnorm>tol) { 
        PetscPrintf(PETSC_COMM_SELF,"Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n",s1norm,s2norm);  
      }
    }
    VecDestroy(xx);
    VecDestroy(s1);
    VecDestroy(s2);
  } 
     
  /* Free allocated memory */
  for (i=0; i<nd; ++i) { 
    ierr = ISDestroy(is1[i]);CHKERRA(ierr);
    ierr = ISDestroy(is2[i]);CHKERRA(ierr);
    ierr = MatDestroy(submatA[i]);CHKERRA(ierr);
    ierr = MatDestroy(submatB[i]);CHKERRA(ierr);
 }
  PetscFree(is1);
  PetscFree(is2);
  PetscFree(idx);
  PetscFree(rows);
  PetscFree(cols);
  PetscFree(vals);
  MatDestroy(A);
  MatDestroy(B);
  PetscFree(submatA); 
  PetscFree(submatB);
  PetscRandomDestroy(rand);
  PetscFinalize();
  return 0;
}
