#define PETSCMAT_DLL

/*
    Factorization code for BAIJ format. 
*/
#include "src/mat/impls/baij/seq/baij.h"
#include "src/inline/ilu.h"

/* ----------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_N"
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_N(Mat A,MatFactorInfo *info,Mat *B)
{
  Mat            C = *B;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ *)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  PetscInt       *r,*ic,i,j,n = a->mbs,*bi = b->i,*bj = b->j;
  PetscInt       *ajtmpold,*ajtmp,nz,row,bslog,*ai=a->i,*aj=a->j,k,flg;
  PetscInt       *diag_offset=b->diag,diag,bs=A->bs,bs2 = a->bs2,*pj,*v_pivots;
  MatScalar      *ba = b->a,*aa = a->a,*pv,*v,*rtmp,*multiplier,*v_work,*pc,*w;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc(bs2*(n+1)*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,bs2*(n+1)*sizeof(MatScalar));CHKERRQ(ierr);
  /* generate work space needed by dense LU factorization */
  ierr       = PetscMalloc(bs*sizeof(PetscInt) + (bs+bs2)*sizeof(MatScalar),&v_work);CHKERRQ(ierr);
  multiplier = v_work + bs;
  v_pivots   = (PetscInt*)(multiplier + bs2);

  /* flops in while loop */
  bslog = 2*bs*bs2;

  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      ierr = PetscMemzero(rtmp+bs2*ajtmp[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }
    /* load in initial (unfactored row) */
    nz       = ai[r[i]+1] - ai[r[i]];
    ajtmpold = aj + ai[r[i]];
    v        = aa + bs2*ai[r[i]];
    for (j=0; j<nz; j++) {
      ierr = PetscMemcpy(rtmp+bs2*ic[ajtmpold[j]],v+bs2*j,bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + bs2*row;
/*      if (*pc) { */
      for (flg=0,k=0; k<bs2; k++) { if (pc[k]!=0.0) { flg = 1; break; }}
      if (flg) {
        pv = ba + bs2*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
        Kernel_A_gets_A_times_B(bs,pc,pv,multiplier); 
        nz = bi[row+1] - diag_offset[row] - 1;
        pv += bs2;
        for (j=0; j<nz; j++) {
          Kernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j);
        }
        ierr = PetscLogFlops(bslog*(nz+1)-bs);CHKERRQ(ierr);
      } 
        row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + bs2*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      ierr = PetscMemcpy(pv+bs2*j,rtmp+bs2*pj[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }
    diag = diag_offset[i] - bi[i];
    /* invert diagonal block */
    w = pv + bs2*diag; 
    ierr = Kernel_A_gets_inverse_A(bs,w,v_pivots,v_work);CHKERRQ(ierr);
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = PetscFree(v_work);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  C->factor = FACTOR_LU;
  C->assembled = PETSC_TRUE;
  ierr = PetscLogFlops(1.3333*bs*bs2*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
