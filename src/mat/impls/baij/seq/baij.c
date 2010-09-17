#define PETSCMAT_DLL

/*
    Defines the basic matrix operations for the BAIJ (compressed row)
  matrix storage format.
*/
#include "../src/mat/impls/baij/seq/baij.h"  /*I   "petscmat.h"  I*/
#include "petscblaslapack.h"
#include "../src/mat/blockinvert.h"

#undef __FUNCT__
#define __FUNCT__ "MatSeqBAIJInvertBlockDiagonal"
/*@
  MatSeqBAIJInvertBlockDiagonal - Inverts the block diagonal entries.

  Collective on Mat

  Input Parameters:
. mat - the matrix

  Level: advanced
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBAIJInvertBlockDiagonal(Mat mat)
{
  PetscErrorCode ierr,(*f)(Mat);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSeqBAIJInvertBlockDiagonal_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"Currently only implemented for SeqBAIJ.");
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatInvertBlockDiagonal_SeqBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatInvertBlockDiagonal_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*) A->data;
  PetscErrorCode ierr;
  PetscInt       *diag_offset,i,bs = A->rmap->bs,mbs = a->mbs,ipvt[5];
  MatScalar      *v = a->a,*odiag,*diag,*mdiag,work[25];
  PetscReal      shift = 0.0;

  PetscFunctionBegin;
  if (a->idiagvalid) PetscFunctionReturn(0);
  ierr = MatMarkDiagonal_SeqBAIJ(A);CHKERRQ(ierr);
  diag_offset = a->diag;
  if (!a->idiag) {
    ierr = PetscMalloc(2*bs*bs*mbs*sizeof(PetscScalar),&a->idiag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,2*bs*bs*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  diag  = a->idiag;
  mdiag = a->idiag+bs*bs*mbs; 
  /* factor and invert each block */
  switch (bs){
    case 1:
      for (i=0; i<mbs; i++) {
        odiag = v + 1*diag_offset[i];
        diag[0]  = odiag[0];
        mdiag[0] = odiag[0];
        diag[0]  = 1.0 / (diag[0] + shift);
        diag    += 1;
        mdiag   += 1;
      }
      break;
    case 2:
      for (i=0; i<mbs; i++) {
        odiag   = v + 4*diag_offset[i];
        diag[0]  = odiag[0]; diag[1] = odiag[1]; diag[2] = odiag[2]; diag[3] = odiag[3];
	mdiag[0] = odiag[0]; mdiag[1] = odiag[1]; mdiag[2] = odiag[2]; mdiag[3] = odiag[3];
	ierr     = Kernel_A_gets_inverse_A_2(diag,shift);CHKERRQ(ierr);
	diag    += 4;
	mdiag   += 4;
      }
      break;
    case 3:
      for (i=0; i<mbs; i++) {
        odiag    = v + 9*diag_offset[i];
        diag[0]  = odiag[0]; diag[1] = odiag[1]; diag[2] = odiag[2]; diag[3] = odiag[3];
        diag[4]  = odiag[4]; diag[5] = odiag[5]; diag[6] = odiag[6]; diag[7] = odiag[7];
        diag[8]  = odiag[8]; 
        mdiag[0] = odiag[0]; mdiag[1] = odiag[1]; mdiag[2] = odiag[2]; mdiag[3] = odiag[3];
        mdiag[4] = odiag[4]; mdiag[5] = odiag[5]; mdiag[6] = odiag[6]; mdiag[7] = odiag[7];
        mdiag[8] = odiag[8]; 
	ierr     = Kernel_A_gets_inverse_A_3(diag,shift);CHKERRQ(ierr);
	diag    += 9;
	mdiag   += 9;
      }
      break;
    case 4:
      for (i=0; i<mbs; i++) {
        odiag  = v + 16*diag_offset[i];
        ierr   = PetscMemcpy(diag,odiag,16*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr   = PetscMemcpy(mdiag,odiag,16*sizeof(PetscScalar));CHKERRQ(ierr);
	ierr   = Kernel_A_gets_inverse_A_4(diag,shift);CHKERRQ(ierr);
	diag  += 16;
	mdiag += 16;
      }
      break;
    case 5:
      for (i=0; i<mbs; i++) {
        odiag = v + 25*diag_offset[i];
        ierr   = PetscMemcpy(diag,odiag,25*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr   = PetscMemcpy(mdiag,odiag,25*sizeof(PetscScalar));CHKERRQ(ierr);
	ierr   = Kernel_A_gets_inverse_A_5(diag,ipvt,work,shift);CHKERRQ(ierr);
	diag  += 25;
	mdiag += 25;
      }
      break;
    case 6:
      for (i=0; i<mbs; i++) {
        odiag = v + 36*diag_offset[i];
        ierr   = PetscMemcpy(diag,odiag,36*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr   = PetscMemcpy(mdiag,odiag,36*sizeof(PetscScalar));CHKERRQ(ierr);
	ierr   = Kernel_A_gets_inverse_A_6(diag,shift);CHKERRQ(ierr);
	diag  += 36;
	mdiag += 36;
      }
      break;
    default: 
      SETERRQ1(PETSC_ERR_SUP,"not supported for block size %D",bs);
  }
  a->idiagvalid = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSOR_SeqBAIJ_1"
PetscErrorCode MatSOR_SeqBAIJ_1(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar        *x,x1,s1;
  const PetscScalar  *b;
  const MatScalar    *aa = a->a, *idiag,*mdiag,*v;
  PetscErrorCode     ierr;
  PetscInt           m = a->mbs,i,i2,nz,j;
  const PetscInt     *diag,*ai = a->i,*aj = a->j,*vi;

  PetscFunctionBegin;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_ERR_SUP,"No support yet for Eisenstat");
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for diagonal shift");
  if (omega != 1.0) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for non-trivial relaxation factor");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for applying upper or lower triangular parts");
  if (its > 1) SETERRQ(PETSC_ERR_SUP,"Sorry, no support yet for multiple point block SOR iterations");

  if (!a->idiagvalid){ierr = MatInvertBlockDiagonal_SeqBAIJ(A);CHKERRQ(ierr);}

  diag  = a->diag;
  idiag = a->idiag;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      x[0] = b[0]*idiag[0];
      i2     = 1;
      idiag += 1;
      for (i=1; i<m; i++) {
        v     = aa + ai[i];
        vi    = aj + ai[i];
        nz    = diag[i] - ai[i];
        s1    = b[i2];
        for (j=0; j<nz; j++) {
          s1 -= v[j]*x[vi[j]];
        }
        x[i2]   = idiag[0]*s1;
        idiag   += 1;
        i2      += 1;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
    }
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      i2    = 0;
      mdiag = a->idiag+a->mbs;
      for (i=0; i<m; i++) {
        x1      = x[i2];
        x[i2]   = mdiag[0]*x1;
        mdiag  += 1;
        i2     += 1;
      }
      ierr = PetscLogFlops(m);CHKERRQ(ierr);
    } else if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      ierr = PetscMemcpy(x,b,A->rmap->N*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      idiag   = a->idiag+a->mbs - 1;
      i2      = m - 1;
      x1      = x[i2];
      x[i2]   = idiag[0]*x1;
      idiag -= 1;
      i2    -= 1;
      for (i=m-2; i>=0; i--) {
        v     = aa + (diag[i]+1);
        vi    = aj + diag[i] + 1;
        nz    = ai[i+1] - diag[i] - 1;
        s1    = x[i2];
        for (j=0; j<nz; j++) {
          s1 -= v[j]*x[vi[j]];
        }
        x[i2]   = idiag[0]*s1;
        idiag   -= 1;
        i2      -= 1;
      }
      ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"Only supports point block SOR with zero initial guess");
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatSOR_SeqBAIJ_2"
PetscErrorCode MatSOR_SeqBAIJ_2(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar        *x,x1,x2,s1,s2;
  const PetscScalar  *b;
  const MatScalar    *v,*aa = a->a, *idiag,*mdiag;
  PetscErrorCode     ierr;
  PetscInt           m = a->mbs,i,i2,nz,idx,j,it;
  const PetscInt     *diag,*ai = a->i,*aj = a->j,*vi;

  PetscFunctionBegin;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_ERR_SUP,"No support yet for Eisenstat");
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for diagonal shift");
  if (omega != 1.0) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for non-trivial relaxation factor");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for applying upper or lower triangular parts");
  if (its > 1) SETERRQ(PETSC_ERR_SUP,"Sorry, no support yet for multiple point block SOR iterations");

  if (!a->idiagvalid){ierr = MatInvertBlockDiagonal_SeqBAIJ(A);CHKERRQ(ierr);}

  diag  = a->diag;
  idiag = a->idiag;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      x[0] = b[0]*idiag[0] + b[1]*idiag[2];
      x[1] = b[0]*idiag[1] + b[1]*idiag[3];
      i2     = 2;
      idiag += 4;
      for (i=1; i<m; i++) {
	v     = aa + 4*ai[i];
	vi    = aj + ai[i];
	nz    = diag[i] - ai[i];
	s1    = b[i2]; s2 = b[i2+1];
        for (j=0; j<nz; j++) {
	  idx  = 2*vi[j];
          it   = 4*j;
	  x1   = x[idx]; x2 = x[1+idx];
	  s1  -= v[it]*x1 + v[it+2]*x2;
	  s2  -= v[it+1]*x1 + v[it+3]*x2;
	}
	x[i2]   = idiag[0]*s1 + idiag[2]*s2;
	x[i2+1] = idiag[1]*s1 + idiag[3]*s2;
        idiag   += 4;
        i2      += 2;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(4.0*(a->nz));CHKERRQ(ierr);
    }
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      i2    = 0;
      mdiag = a->idiag+4*a->mbs;
      for (i=0; i<m; i++) {
        x1      = x[i2]; x2 = x[i2+1];
        x[i2]   = mdiag[0]*x1 + mdiag[2]*x2;
        x[i2+1] = mdiag[1]*x1 + mdiag[3]*x2;
        mdiag  += 4;
        i2     += 2;
      }
      ierr = PetscLogFlops(6.0*m);CHKERRQ(ierr);
    } else if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      ierr = PetscMemcpy(x,b,A->rmap->N*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      idiag   = a->idiag+4*a->mbs - 4;
      i2      = 2*m - 2;
      x1      = x[i2]; x2 = x[i2+1];
      x[i2]   = idiag[0]*x1 + idiag[2]*x2;
      x[i2+1] = idiag[1]*x1 + idiag[3]*x2;
      idiag -= 4;
      i2    -= 2;
      for (i=m-2; i>=0; i--) {
	v     = aa + 4*(diag[i]+1);
	vi    = aj + diag[i] + 1;
	nz    = ai[i+1] - diag[i] - 1;
	s1    = x[i2]; s2 = x[i2+1];
        for (j=0; j<nz; j++) {
 	  idx  = 2*vi[j];
          it   = 4*j;
	  x1   = x[idx]; x2 = x[1+idx];
	  s1  -= v[it]*x1 + v[it+2]*x2;
	  s2  -= v[it+1]*x1 + v[it+3]*x2;
	}
	x[i2]   = idiag[0]*s1 + idiag[2]*s2;
	x[i2+1] = idiag[1]*s1 + idiag[3]*s2;
        idiag   -= 4;
        i2      -= 2;
      }
      ierr = PetscLogFlops(4.0*(a->nz));CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"Only supports point block SOR with zero initial guess");
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatSOR_SeqBAIJ_3"
PetscErrorCode MatSOR_SeqBAIJ_3(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar        *x,x1,x2,x3,s1,s2,s3;
  const MatScalar    *v,*aa = a->a, *idiag,*mdiag;
  const PetscScalar  *b;
  PetscErrorCode     ierr;
  PetscInt           m = a->mbs,i,i2,nz,idx;
  const PetscInt     *diag,*ai = a->i,*aj = a->j,*vi;

  PetscFunctionBegin;
  its = its*lits;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_ERR_SUP,"No support yet for Eisenstat");
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for diagonal shift");
  if (omega != 1.0) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for non-trivial relaxation factor");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for applying upper or lower triangular parts");
  if (its > 1) SETERRQ(PETSC_ERR_SUP,"Sorry, no support yet for multiple point block SOR iterations");

  if (!a->idiagvalid){ierr = MatInvertBlockDiagonal_SeqBAIJ(A);CHKERRQ(ierr);}

  diag  = a->diag;
  idiag = a->idiag;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      x[0] = b[0]*idiag[0] + b[1]*idiag[3] + b[2]*idiag[6];
      x[1] = b[0]*idiag[1] + b[1]*idiag[4] + b[2]*idiag[7];
      x[2] = b[0]*idiag[2] + b[1]*idiag[5] + b[2]*idiag[8];
      i2     = 3;
      idiag += 9;
      for (i=1; i<m; i++) {
        v     = aa + 9*ai[i];
        vi    = aj + ai[i];
        nz    = diag[i] - ai[i];
        s1    = b[i2]; s2 = b[i2+1]; s3 = b[i2+2];
        while (nz--) {
          idx  = 3*(*vi++);
          x1   = x[idx]; x2 = x[1+idx];x3 = x[2+idx];
          s1  -= v[0]*x1 + v[3]*x2 + v[6]*x3;
          s2  -= v[1]*x1 + v[4]*x2 + v[7]*x3;
          s3  -= v[2]*x1 + v[5]*x2 + v[8]*x3;
          v   += 9;
        }
        x[i2]   = idiag[0]*s1 + idiag[3]*s2 + idiag[6]*s3;
        x[i2+1] = idiag[1]*s1 + idiag[4]*s2 + idiag[7]*s3;
        x[i2+2] = idiag[2]*s1 + idiag[5]*s2 + idiag[8]*s3;
        idiag   += 9;
        i2      += 3;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(9.0*(a->nz));CHKERRQ(ierr);
    }
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      i2    = 0;
      mdiag = a->idiag+9*a->mbs;
      for (i=0; i<m; i++) {
        x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2];
        x[i2]   = mdiag[0]*x1 + mdiag[3]*x2 + mdiag[6]*x3;
        x[i2+1] = mdiag[1]*x1 + mdiag[4]*x2 + mdiag[7]*x3;
        x[i2+2] = mdiag[2]*x1 + mdiag[5]*x2 + mdiag[8]*x3;
        mdiag  += 9;
        i2     += 3;
      }
      ierr = PetscLogFlops(15.0*m);CHKERRQ(ierr);
    } else if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      ierr = PetscMemcpy(x,b,A->rmap->N*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      idiag   = a->idiag+9*a->mbs - 9;
      i2      = 3*m - 3;
      x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2];
      x[i2]   = idiag[0]*x1 + idiag[3]*x2 + idiag[6]*x3;
      x[i2+1] = idiag[1]*x1 + idiag[4]*x2 + idiag[7]*x3;
      x[i2+2] = idiag[2]*x1 + idiag[5]*x2 + idiag[8]*x3;
      idiag -= 9;
      i2    -= 3;
      for (i=m-2; i>=0; i--) {
        v     = aa + 9*(diag[i]+1);
        vi    = aj + diag[i] + 1;
        nz    = ai[i+1] - diag[i] - 1;
        s1    = x[i2]; s2 = x[i2+1]; s3 = x[i2+2];
        while (nz--) {
          idx  = 3*(*vi++);
          x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx];
          s1  -= v[0]*x1 + v[3]*x2 + v[6]*x3;
          s2  -= v[1]*x1 + v[4]*x2 + v[7]*x3;
          s3  -= v[2]*x1 + v[5]*x2 + v[8]*x3;
          v   += 9;
        }
        x[i2]   = idiag[0]*s1 + idiag[3]*s2 + idiag[6]*s3;
        x[i2+1] = idiag[1]*s1 + idiag[4]*s2 + idiag[7]*s3;
        x[i2+2] = idiag[2]*s1 + idiag[5]*s2 + idiag[8]*s3;
        idiag   -= 9;
        i2      -= 3;
      }
      ierr = PetscLogFlops(9.0*(a->nz));CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"Only supports point block SOR with zero initial guess");
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatSOR_SeqBAIJ_4"
PetscErrorCode MatSOR_SeqBAIJ_4(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar        *x,x1,x2,x3,x4,s1,s2,s3,s4;
  const MatScalar    *v,*aa = a->a, *idiag,*mdiag;
  const PetscScalar  *b;
  PetscErrorCode     ierr;
  PetscInt           m = a->mbs,i,i2,nz,idx;
  const PetscInt     *diag,*ai = a->i,*aj = a->j,*vi;

  PetscFunctionBegin;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_ERR_SUP,"No support yet for Eisenstat");
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for diagonal shift");
  if (omega != 1.0) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for non-trivial relaxation factor");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for applying upper or lower triangular parts");
  if (its > 1) SETERRQ(PETSC_ERR_SUP,"Sorry, no support yet for multiple point block SOR iterations");

  if (!a->idiagvalid){ierr = MatInvertBlockDiagonal_SeqBAIJ(A);CHKERRQ(ierr);}

  diag  = a->diag;
  idiag = a->idiag;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      x[0] = b[0]*idiag[0] + b[1]*idiag[4] + b[2]*idiag[8]  + b[3]*idiag[12];
      x[1] = b[0]*idiag[1] + b[1]*idiag[5] + b[2]*idiag[9]  + b[3]*idiag[13];
      x[2] = b[0]*idiag[2] + b[1]*idiag[6] + b[2]*idiag[10] + b[3]*idiag[14];
      x[3] = b[0]*idiag[3] + b[1]*idiag[7] + b[2]*idiag[11] + b[3]*idiag[15];
      i2     = 4;
      idiag += 16;
      for (i=1; i<m; i++) {
	v     = aa + 16*ai[i];
	vi    = aj + ai[i];
	nz    = diag[i] - ai[i];
	s1    = b[i2]; s2 = b[i2+1]; s3 = b[i2+2]; s4 = b[i2+3];
	while (nz--) {
	  idx  = 4*(*vi++);
	  x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx];
	  s1  -= v[0]*x1 + v[4]*x2 + v[8]*x3  + v[12]*x4;
	  s2  -= v[1]*x1 + v[5]*x2 + v[9]*x3  + v[13]*x4;
	  s3  -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
	  s4  -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;
	  v   += 16;
	}
	x[i2]   = idiag[0]*s1 + idiag[4]*s2 + idiag[8]*s3  + idiag[12]*s4;
	x[i2+1] = idiag[1]*s1 + idiag[5]*s2 + idiag[9]*s3  + idiag[13]*s4;
	x[i2+2] = idiag[2]*s1 + idiag[6]*s2 + idiag[10]*s3 + idiag[14]*s4;
	x[i2+3] = idiag[3]*s1 + idiag[7]*s2 + idiag[11]*s3 + idiag[15]*s4;
        idiag   += 16;
        i2      += 4;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(16.0*(a->nz));CHKERRQ(ierr);
    }
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      i2    = 0;
      mdiag = a->idiag+16*a->mbs;
      for (i=0; i<m; i++) {
        x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3];
        x[i2]   = mdiag[0]*x1 + mdiag[4]*x2 + mdiag[8]*x3  + mdiag[12]*x4;
        x[i2+1] = mdiag[1]*x1 + mdiag[5]*x2 + mdiag[9]*x3  + mdiag[13]*x4;
        x[i2+2] = mdiag[2]*x1 + mdiag[6]*x2 + mdiag[10]*x3 + mdiag[14]*x4;
        x[i2+3] = mdiag[3]*x1 + mdiag[7]*x2 + mdiag[11]*x3 + mdiag[15]*x4;
        mdiag  += 16;
        i2     += 4;
      }
      ierr = PetscLogFlops(28.0*m);CHKERRQ(ierr);
    } else if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      ierr = PetscMemcpy(x,b,A->rmap->N*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      idiag   = a->idiag+16*a->mbs - 16;
      i2      = 4*m - 4;
      x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3];
      x[i2]   = idiag[0]*x1 + idiag[4]*x2 + idiag[8]*x3  + idiag[12]*x4;
      x[i2+1] = idiag[1]*x1 + idiag[5]*x2 + idiag[9]*x3  + idiag[13]*x4;
      x[i2+2] = idiag[2]*x1 + idiag[6]*x2 + idiag[10]*x3 + idiag[14]*x4;
      x[i2+3] = idiag[3]*x1 + idiag[7]*x2 + idiag[11]*x3 + idiag[15]*x4;
      idiag -= 16;
      i2    -= 4;
      for (i=m-2; i>=0; i--) {
	v     = aa + 16*(diag[i]+1);
	vi    = aj + diag[i] + 1;
	nz    = ai[i+1] - diag[i] - 1;
	s1    = x[i2]; s2 = x[i2+1]; s3 = x[i2+2]; s4 = x[i2+3];
	while (nz--) {
	  idx  = 4*(*vi++);
	  x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx];
	  s1  -= v[0]*x1 + v[4]*x2 + v[8]*x3  + v[12]*x4;
	  s2  -= v[1]*x1 + v[5]*x2 + v[9]*x3  + v[13]*x4;
	  s3  -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
	  s4  -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;
	  v   += 16;
	}
	x[i2]   = idiag[0]*s1 + idiag[4]*s2 + idiag[8]*s3  + idiag[12]*s4;
	x[i2+1] = idiag[1]*s1 + idiag[5]*s2 + idiag[9]*s3  + idiag[13]*s4;
	x[i2+2] = idiag[2]*s1 + idiag[6]*s2 + idiag[10]*s3 + idiag[14]*s4;
	x[i2+3] = idiag[3]*s1 + idiag[7]*s2 + idiag[11]*s3 + idiag[15]*s4;
        idiag   -= 16;
        i2      -= 4;
      }
      ierr = PetscLogFlops(16.0*(a->nz));CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"Only supports point block SOR with zero initial guess");
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatSOR_SeqBAIJ_5"
PetscErrorCode MatSOR_SeqBAIJ_5(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar        *x,x1,x2,x3,x4,x5,s1,s2,s3,s4,s5;
  const MatScalar    *v,*aa = a->a, *idiag,*mdiag;
  const PetscScalar  *b;
  PetscErrorCode     ierr;
  PetscInt           m = a->mbs,i,i2,nz,idx;
  const PetscInt     *diag,*ai = a->i,*aj = a->j,*vi;

  PetscFunctionBegin;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_ERR_SUP,"No support yet for Eisenstat");
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for diagonal shift");
  if (omega != 1.0) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for non-trivial relaxation factor");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for applying upper or lower triangular parts");
  if (its > 1) SETERRQ(PETSC_ERR_SUP,"Sorry, no support yet for multiple point block SOR iterations");

  if (!a->idiagvalid){ierr = MatInvertBlockDiagonal_SeqBAIJ(A);CHKERRQ(ierr);}

  diag  = a->diag;
  idiag = a->idiag;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      x[0] = b[0]*idiag[0] + b[1]*idiag[5] + b[2]*idiag[10] + b[3]*idiag[15] + b[4]*idiag[20];
      x[1] = b[0]*idiag[1] + b[1]*idiag[6] + b[2]*idiag[11] + b[3]*idiag[16] + b[4]*idiag[21];
      x[2] = b[0]*idiag[2] + b[1]*idiag[7] + b[2]*idiag[12] + b[3]*idiag[17] + b[4]*idiag[22];
      x[3] = b[0]*idiag[3] + b[1]*idiag[8] + b[2]*idiag[13] + b[3]*idiag[18] + b[4]*idiag[23];
      x[4] = b[0]*idiag[4] + b[1]*idiag[9] + b[2]*idiag[14] + b[3]*idiag[19] + b[4]*idiag[24];
      i2     = 5;
      idiag += 25;
      for (i=1; i<m; i++) {
	v     = aa + 25*ai[i];
	vi    = aj + ai[i];
	nz    = diag[i] - ai[i];
	s1    = b[i2]; s2 = b[i2+1]; s3 = b[i2+2]; s4 = b[i2+3]; s5 = b[i2+4];
	while (nz--) {
	  idx  = 5*(*vi++);
	  x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx];
	  s1  -= v[0]*x1 + v[5]*x2 + v[10]*x3 + v[15]*x4 + v[20]*x5;
	  s2  -= v[1]*x1 + v[6]*x2 + v[11]*x3 + v[16]*x4 + v[21]*x5;
	  s3  -= v[2]*x1 + v[7]*x2 + v[12]*x3 + v[17]*x4 + v[22]*x5;
	  s4  -= v[3]*x1 + v[8]*x2 + v[13]*x3 + v[18]*x4 + v[23]*x5;
	  s5  -= v[4]*x1 + v[9]*x2 + v[14]*x3 + v[19]*x4 + v[24]*x5;
	  v   += 25;
	}
	x[i2]   = idiag[0]*s1 + idiag[5]*s2 + idiag[10]*s3 + idiag[15]*s4 + idiag[20]*s5;
	x[i2+1] = idiag[1]*s1 + idiag[6]*s2 + idiag[11]*s3 + idiag[16]*s4 + idiag[21]*s5;
	x[i2+2] = idiag[2]*s1 + idiag[7]*s2 + idiag[12]*s3 + idiag[17]*s4 + idiag[22]*s5;
	x[i2+3] = idiag[3]*s1 + idiag[8]*s2 + idiag[13]*s3 + idiag[18]*s4 + idiag[23]*s5;
	x[i2+4] = idiag[4]*s1 + idiag[9]*s2 + idiag[14]*s3 + idiag[19]*s4 + idiag[24]*s5;
        idiag   += 25;
        i2      += 5;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(25.0*(a->nz));CHKERRQ(ierr);
    }
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      i2    = 0;
      mdiag = a->idiag+25*a->mbs;
      for (i=0; i<m; i++) {
        x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3]; x5 = x[i2+4];
        x[i2]   = mdiag[0]*x1 + mdiag[5]*x2 + mdiag[10]*x3 + mdiag[15]*x4 + mdiag[20]*x5;
        x[i2+1] = mdiag[1]*x1 + mdiag[6]*x2 + mdiag[11]*x3 + mdiag[16]*x4 + mdiag[21]*x5;
        x[i2+2] = mdiag[2]*x1 + mdiag[7]*x2 + mdiag[12]*x3 + mdiag[17]*x4 + mdiag[22]*x5;
        x[i2+3] = mdiag[3]*x1 + mdiag[8]*x2 + mdiag[13]*x3 + mdiag[18]*x4 + mdiag[23]*x5;
        x[i2+4] = mdiag[4]*x1 + mdiag[9]*x2 + mdiag[14]*x3 + mdiag[19]*x4 + mdiag[24]*x5;
        mdiag  += 25;
        i2     += 5;
      }
      ierr = PetscLogFlops(45.0*m);CHKERRQ(ierr);
    } else if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      ierr = PetscMemcpy(x,b,A->rmap->N*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      idiag   = a->idiag+25*a->mbs - 25;
      i2      = 5*m - 5;
      x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3]; x5 = x[i2+4];
      x[i2]   = idiag[0]*x1 + idiag[5]*x2 + idiag[10]*x3 + idiag[15]*x4 + idiag[20]*x5;
      x[i2+1] = idiag[1]*x1 + idiag[6]*x2 + idiag[11]*x3 + idiag[16]*x4 + idiag[21]*x5;
      x[i2+2] = idiag[2]*x1 + idiag[7]*x2 + idiag[12]*x3 + idiag[17]*x4 + idiag[22]*x5;
      x[i2+3] = idiag[3]*x1 + idiag[8]*x2 + idiag[13]*x3 + idiag[18]*x4 + idiag[23]*x5;
      x[i2+4] = idiag[4]*x1 + idiag[9]*x2 + idiag[14]*x3 + idiag[19]*x4 + idiag[24]*x5;
      idiag -= 25;
      i2    -= 5;
      for (i=m-2; i>=0; i--) {
	v     = aa + 25*(diag[i]+1);
	vi    = aj + diag[i] + 1;
	nz    = ai[i+1] - diag[i] - 1;
	s1    = x[i2]; s2 = x[i2+1]; s3 = x[i2+2]; s4 = x[i2+3]; s5 = x[i2+4];
	while (nz--) {
	  idx  = 5*(*vi++);
	  x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx];
	  s1  -= v[0]*x1 + v[5]*x2 + v[10]*x3 + v[15]*x4 + v[20]*x5;
	  s2  -= v[1]*x1 + v[6]*x2 + v[11]*x3 + v[16]*x4 + v[21]*x5;
	  s3  -= v[2]*x1 + v[7]*x2 + v[12]*x3 + v[17]*x4 + v[22]*x5;
	  s4  -= v[3]*x1 + v[8]*x2 + v[13]*x3 + v[18]*x4 + v[23]*x5;
	  s5  -= v[4]*x1 + v[9]*x2 + v[14]*x3 + v[19]*x4 + v[24]*x5;
	  v   += 25;
	}
	x[i2]   = idiag[0]*s1 + idiag[5]*s2 + idiag[10]*s3 + idiag[15]*s4 + idiag[20]*s5;
	x[i2+1] = idiag[1]*s1 + idiag[6]*s2 + idiag[11]*s3 + idiag[16]*s4 + idiag[21]*s5;
	x[i2+2] = idiag[2]*s1 + idiag[7]*s2 + idiag[12]*s3 + idiag[17]*s4 + idiag[22]*s5;
	x[i2+3] = idiag[3]*s1 + idiag[8]*s2 + idiag[13]*s3 + idiag[18]*s4 + idiag[23]*s5;
	x[i2+4] = idiag[4]*s1 + idiag[9]*s2 + idiag[14]*s3 + idiag[19]*s4 + idiag[24]*s5;
        idiag   -= 25;
        i2      -= 5;
      }
      ierr = PetscLogFlops(25.0*(a->nz));CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"Only supports point block SOR with zero initial guess");
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatSOR_SeqBAIJ_6"
PetscErrorCode MatSOR_SeqBAIJ_6(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar        *x,x1,x2,x3,x4,x5,x6,s1,s2,s3,s4,s5,s6;
  const MatScalar    *v,*aa = a->a, *idiag,*mdiag;
  const PetscScalar  *b;
  PetscErrorCode     ierr;
  PetscInt           m = a->mbs,i,i2,nz,idx;
  const PetscInt     *diag,*ai = a->i,*aj = a->j,*vi;

  PetscFunctionBegin;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_ERR_SUP,"No support yet for Eisenstat");
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for diagonal shift");
  if (omega != 1.0) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for non-trivial relaxation factor");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for applying upper or lower triangular parts");
  if (its > 1) SETERRQ(PETSC_ERR_SUP,"Sorry, no support yet for multiple point block SOR iterations");

  if (!a->idiagvalid){ierr = MatInvertBlockDiagonal_SeqBAIJ(A);CHKERRQ(ierr);}

  diag  = a->diag;
  idiag = a->idiag;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      x[0] = b[0]*idiag[0] + b[1]*idiag[6]  + b[2]*idiag[12] + b[3]*idiag[18] + b[4]*idiag[24] + b[5]*idiag[30];
      x[1] = b[0]*idiag[1] + b[1]*idiag[7]  + b[2]*idiag[13] + b[3]*idiag[19] + b[4]*idiag[25] + b[5]*idiag[31];
      x[2] = b[0]*idiag[2] + b[1]*idiag[8]  + b[2]*idiag[14] + b[3]*idiag[20] + b[4]*idiag[26] + b[5]*idiag[32];
      x[3] = b[0]*idiag[3] + b[1]*idiag[9]  + b[2]*idiag[15] + b[3]*idiag[21] + b[4]*idiag[27] + b[5]*idiag[33];
      x[4] = b[0]*idiag[4] + b[1]*idiag[10] + b[2]*idiag[16] + b[3]*idiag[22] + b[4]*idiag[28] + b[5]*idiag[34];
      x[5] = b[0]*idiag[5] + b[1]*idiag[11] + b[2]*idiag[17] + b[3]*idiag[23] + b[4]*idiag[29] + b[5]*idiag[35];
      i2     = 6;
      idiag += 36;
      for (i=1; i<m; i++) {
        v     = aa + 36*ai[i];
        vi    = aj + ai[i];
        nz    = diag[i] - ai[i];
        s1    = b[i2]; s2 = b[i2+1]; s3 = b[i2+2]; s4 = b[i2+3]; s5 = b[i2+4]; s6 = b[i2+5];
        while (nz--) {
          idx  = 6*(*vi++);
          x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx]; x6 = x[5+idx];
          s1  -= v[0]*x1 + v[6]*x2  + v[12]*x3 + v[18]*x4 + v[24]*x5 + v[30]*x6;
          s2  -= v[1]*x1 + v[7]*x2  + v[13]*x3 + v[19]*x4 + v[25]*x5 + v[31]*x6;
          s3  -= v[2]*x1 + v[8]*x2  + v[14]*x3 + v[20]*x4 + v[26]*x5 + v[32]*x6;
          s4  -= v[3]*x1 + v[9]*x2  + v[15]*x3 + v[21]*x4 + v[27]*x5 + v[33]*x6;
          s5  -= v[4]*x1 + v[10]*x2 + v[16]*x3 + v[22]*x4 + v[28]*x5 + v[34]*x6;
          s6  -= v[5]*x1 + v[11]*x2 + v[17]*x3 + v[23]*x4 + v[29]*x5 + v[35]*x6;
          v   += 36;
        }
        x[i2]   = idiag[0]*s1 + idiag[6]*s2  + idiag[12]*s3 + idiag[18]*s4 + idiag[24]*s5 + idiag[30]*s6;
        x[i2+1] = idiag[1]*s1 + idiag[7]*s2  + idiag[13]*s3 + idiag[19]*s4 + idiag[25]*s5 + idiag[31]*s6;
        x[i2+2] = idiag[2]*s1 + idiag[8]*s2  + idiag[14]*s3 + idiag[20]*s4 + idiag[26]*s5 + idiag[32]*s6;
        x[i2+3] = idiag[3]*s1 + idiag[9]*s2  + idiag[15]*s3 + idiag[21]*s4 + idiag[27]*s5 + idiag[33]*s6;
        x[i2+4] = idiag[4]*s1 + idiag[10]*s2 + idiag[16]*s3 + idiag[22]*s4 + idiag[28]*s5 + idiag[34]*s6;
        x[i2+5] = idiag[5]*s1 + idiag[11]*s2 + idiag[17]*s3 + idiag[23]*s4 + idiag[29]*s5 + idiag[35]*s6;
        idiag   += 36;
        i2      += 6;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(36.0*(a->nz));CHKERRQ(ierr);
    }
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      i2    = 0;
      mdiag = a->idiag+36*a->mbs;
      for (i=0; i<m; i++) {
        x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3]; x5 = x[i2+4]; x6 = x[i2+5];
        x[i2]   = mdiag[0]*x1 + mdiag[6]*x2  + mdiag[12]*x3 + mdiag[18]*x4 + mdiag[24]*x5 + mdiag[30]*x6;
        x[i2+1] = mdiag[1]*x1 + mdiag[7]*x2  + mdiag[13]*x3 + mdiag[19]*x4 + mdiag[25]*x5 + mdiag[31]*x6;
        x[i2+2] = mdiag[2]*x1 + mdiag[8]*x2  + mdiag[14]*x3 + mdiag[20]*x4 + mdiag[26]*x5 + mdiag[32]*x6;
        x[i2+3] = mdiag[3]*x1 + mdiag[9]*x2  + mdiag[15]*x3 + mdiag[21]*x4 + mdiag[27]*x5 + mdiag[33]*x6;
        x[i2+4] = mdiag[4]*x1 + mdiag[10]*x2 + mdiag[16]*x3 + mdiag[22]*x4 + mdiag[28]*x5 + mdiag[34]*x6;
        x[i2+5] = mdiag[5]*x1 + mdiag[11]*x2 + mdiag[17]*x3 + mdiag[23]*x4 + mdiag[29]*x5 + mdiag[35]*x6;
        mdiag  += 36;
        i2     += 6;
      }
      ierr = PetscLogFlops(60.0*m);CHKERRQ(ierr);
    } else if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      ierr = PetscMemcpy(x,b,A->rmap->N*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      idiag   = a->idiag+36*a->mbs - 36;
      i2      = 6*m - 6;
      x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3]; x5 = x[i2+4]; x6 = x[i2+5];
      x[i2]   = idiag[0]*x1 + idiag[6]*x2  + idiag[12]*x3 + idiag[18]*x4 + idiag[24]*x5 + idiag[30]*x6;
      x[i2+1] = idiag[1]*x1 + idiag[7]*x2  + idiag[13]*x3 + idiag[19]*x4 + idiag[25]*x5 + idiag[31]*x6;
      x[i2+2] = idiag[2]*x1 + idiag[8]*x2  + idiag[14]*x3 + idiag[20]*x4 + idiag[26]*x5 + idiag[32]*x6;
      x[i2+3] = idiag[3]*x1 + idiag[9]*x2  + idiag[15]*x3 + idiag[21]*x4 + idiag[27]*x5 + idiag[33]*x6;
      x[i2+4] = idiag[4]*x1 + idiag[10]*x2 + idiag[16]*x3 + idiag[22]*x4 + idiag[28]*x5 + idiag[34]*x6;
      x[i2+5] = idiag[5]*x1 + idiag[11]*x2 + idiag[17]*x3 + idiag[23]*x4 + idiag[29]*x5 + idiag[35]*x6;
      idiag -= 36;
      i2    -= 6;
      for (i=m-2; i>=0; i--) {
        v     = aa + 36*(diag[i]+1);
        vi    = aj + diag[i] + 1;
        nz    = ai[i+1] - diag[i] - 1;
        s1    = x[i2]; s2 = x[i2+1]; s3 = x[i2+2]; s4 = x[i2+3]; s5 = x[i2+4]; s6 = x[i2+5];
        while (nz--) {
          idx  = 6*(*vi++);
          x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx]; x6 = x[5+idx];
          s1  -= v[0]*x1 + v[6]*x2  + v[12]*x3 + v[18]*x4 + v[24]*x5 + v[30]*x6;
          s2  -= v[1]*x1 + v[7]*x2  + v[13]*x3 + v[19]*x4 + v[25]*x5 + v[31]*x6;
          s3  -= v[2]*x1 + v[8]*x2  + v[14]*x3 + v[20]*x4 + v[26]*x5 + v[32]*x6;
          s4  -= v[3]*x1 + v[9]*x2  + v[15]*x3 + v[21]*x4 + v[27]*x5 + v[33]*x6;
          s5  -= v[4]*x1 + v[10]*x2 + v[16]*x3 + v[22]*x4 + v[28]*x5 + v[34]*x6;
          s6  -= v[5]*x1 + v[11]*x2 + v[17]*x3 + v[23]*x4 + v[29]*x5 + v[35]*x6;
          v   += 36;
        }
        x[i2]   = idiag[0]*s1 + idiag[6]*s2  + idiag[12]*s3 + idiag[18]*s4 + idiag[24]*s5 + idiag[30]*s6;
        x[i2+1] = idiag[1]*s1 + idiag[7]*s2  + idiag[13]*s3 + idiag[19]*s4 + idiag[25]*s5 + idiag[31]*s6;
        x[i2+2] = idiag[2]*s1 + idiag[8]*s2  + idiag[14]*s3 + idiag[20]*s4 + idiag[26]*s5 + idiag[32]*s6;
        x[i2+3] = idiag[3]*s1 + idiag[9]*s2  + idiag[15]*s3 + idiag[21]*s4 + idiag[27]*s5 + idiag[33]*s6;
        x[i2+4] = idiag[4]*s1 + idiag[10]*s2 + idiag[16]*s3 + idiag[22]*s4 + idiag[28]*s5 + idiag[34]*s6;
        x[i2+5] = idiag[5]*s1 + idiag[11]*s2 + idiag[17]*s3 + idiag[23]*s4 + idiag[29]*s5 + idiag[35]*s6;
        idiag   -= 36;
        i2      -= 6;
      }
      ierr = PetscLogFlops(36.0*(a->nz));CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"Only supports point block SOR with zero initial guess");
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatSOR_SeqBAIJ_7"
PetscErrorCode MatSOR_SeqBAIJ_7(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar        *x,x1,x2,x3,x4,x5,x6,x7,s1,s2,s3,s4,s5,s6,s7;
  const MatScalar    *v,*aa = a->a, *idiag,*mdiag;
  const PetscScalar  *b;
  PetscErrorCode     ierr;
  PetscInt           m = a->mbs,i,i2,nz,idx;
  const PetscInt     *diag,*ai = a->i,*aj = a->j,*vi;

  PetscFunctionBegin;
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_ERR_SUP,"No support yet for Eisenstat");
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for diagonal shift");
  if (omega != 1.0) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for non-trivial relaxation factor");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) SETERRQ(PETSC_ERR_SUP,"Sorry, no support for applying upper or lower triangular parts");
  if (its > 1) SETERRQ(PETSC_ERR_SUP,"Sorry, no support yet for multiple point block SOR iterations");

  if (!a->idiagvalid){ierr = MatInvertBlockDiagonal_SeqBAIJ(A);CHKERRQ(ierr);}

  diag  = a->diag;
  idiag = a->idiag;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      x[0] = b[0]*idiag[0] + b[1]*idiag[7]  + b[2]*idiag[14] + b[3]*idiag[21] + b[4]*idiag[28] + b[5]*idiag[35] + b[6]*idiag[42];
      x[1] = b[0]*idiag[1] + b[1]*idiag[8]  + b[2]*idiag[15] + b[3]*idiag[22] + b[4]*idiag[29] + b[5]*idiag[36] + b[6]*idiag[43];
      x[2] = b[0]*idiag[2] + b[1]*idiag[9]  + b[2]*idiag[16] + b[3]*idiag[23] + b[4]*idiag[30] + b[5]*idiag[37] + b[6]*idiag[44];
      x[3] = b[0]*idiag[3] + b[1]*idiag[10] + b[2]*idiag[17] + b[3]*idiag[24] + b[4]*idiag[31] + b[5]*idiag[38] + b[6]*idiag[45];
      x[4] = b[0]*idiag[4] + b[1]*idiag[11] + b[2]*idiag[18] + b[3]*idiag[25] + b[4]*idiag[32] + b[5]*idiag[39] + b[6]*idiag[46];
      x[5] = b[0]*idiag[5] + b[1]*idiag[12] + b[2]*idiag[19] + b[3]*idiag[26] + b[4]*idiag[33] + b[5]*idiag[40] + b[6]*idiag[47];
      x[6] = b[0]*idiag[6] + b[1]*idiag[13] + b[2]*idiag[20] + b[3]*idiag[27] + b[4]*idiag[34] + b[5]*idiag[41] + b[6]*idiag[48];
      i2     = 7;
      idiag += 49;
      for (i=1; i<m; i++) {
        v     = aa + 49*ai[i];
        vi    = aj + ai[i];
        nz    = diag[i] - ai[i];
        s1    = b[i2]; s2 = b[i2+1]; s3 = b[i2+2]; s4 = b[i2+3]; s5 = b[i2+4]; s6 = b[i2+5]; s7 = b[i2+6];
        while (nz--) {
          idx  = 7*(*vi++);
          x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx]; x6 = x[5+idx]; x7 = x[6+idx];
          s1  -= v[0]*x1 + v[7]*x2  + v[14]*x3 + v[21]*x4 + v[28]*x5 + v[35]*x6 + v[42]*x7;
          s2  -= v[1]*x1 + v[8]*x2  + v[15]*x3 + v[22]*x4 + v[29]*x5 + v[36]*x6 + v[43]*x7;
          s3  -= v[2]*x1 + v[9]*x2  + v[16]*x3 + v[23]*x4 + v[30]*x5 + v[37]*x6 + v[44]*x7;
          s4  -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4 + v[31]*x5 + v[38]*x6 + v[45]*x7;
          s5  -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4 + v[32]*x5 + v[39]*x6 + v[46]*x7;
          s6  -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4 + v[33]*x5 + v[40]*x6 + v[47]*x7;
          s7  -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4 + v[34]*x5 + v[41]*x6 + v[48]*x7;
          v   += 49;
        }
        x[i2]   = idiag[0]*s1 + idiag[7]*s2  + idiag[14]*s3 + idiag[21]*s4 + idiag[28]*s5 + idiag[35]*s6 + idiag[42]*s7;
        x[i2+1] = idiag[1]*s1 + idiag[8]*s2  + idiag[15]*s3 + idiag[22]*s4 + idiag[29]*s5 + idiag[36]*s6 + idiag[43]*s7;
        x[i2+2] = idiag[2]*s1 + idiag[9]*s2  + idiag[16]*s3 + idiag[23]*s4 + idiag[30]*s5 + idiag[37]*s6 + idiag[44]*s7;
        x[i2+3] = idiag[3]*s1 + idiag[10]*s2 + idiag[17]*s3 + idiag[24]*s4 + idiag[31]*s5 + idiag[38]*s6 + idiag[45]*s7;
        x[i2+4] = idiag[4]*s1 + idiag[11]*s2 + idiag[18]*s3 + idiag[25]*s4 + idiag[32]*s5 + idiag[39]*s6 + idiag[46]*s7;
        x[i2+5] = idiag[5]*s1 + idiag[12]*s2 + idiag[19]*s3 + idiag[26]*s4 + idiag[33]*s5 + idiag[40]*s6 + idiag[47]*s7;
        x[i2+6] = idiag[6]*s1 + idiag[13]*s2 + idiag[20]*s3 + idiag[27]*s4 + idiag[34]*s5 + idiag[41]*s6 + idiag[48]*s7;
        idiag   += 49;
        i2      += 7;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(49.0*(a->nz));CHKERRQ(ierr);
    }
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      i2    = 0;
      mdiag = a->idiag+49*a->mbs;
      for (i=0; i<m; i++) {
        x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3]; x5 = x[i2+4]; x6 = x[i2+5]; x7 = x[i2+6];
        x[i2]   = mdiag[0]*x1 + mdiag[7]*x2  + mdiag[14]*x3 + mdiag[21]*x4 + mdiag[28]*x5 + mdiag[35]*x6 + mdiag[35]*x7;
        x[i2+1] = mdiag[1]*x1 + mdiag[8]*x2  + mdiag[15]*x3 + mdiag[22]*x4 + mdiag[29]*x5 + mdiag[36]*x6 + mdiag[36]*x7;
        x[i2+2] = mdiag[2]*x1 + mdiag[9]*x2  + mdiag[16]*x3 + mdiag[23]*x4 + mdiag[30]*x5 + mdiag[37]*x6 + mdiag[37]*x7;
        x[i2+3] = mdiag[3]*x1 + mdiag[10]*x2 + mdiag[17]*x3 + mdiag[24]*x4 + mdiag[31]*x5 + mdiag[38]*x6 + mdiag[38]*x7;
        x[i2+4] = mdiag[4]*x1 + mdiag[11]*x2 + mdiag[18]*x3 + mdiag[25]*x4 + mdiag[32]*x5 + mdiag[39]*x6 + mdiag[39]*x7;
        x[i2+5] = mdiag[5]*x1 + mdiag[12]*x2 + mdiag[19]*x3 + mdiag[26]*x4 + mdiag[33]*x5 + mdiag[40]*x6 + mdiag[40]*x7;
        x[i2+6] = mdiag[6]*x1 + mdiag[13]*x2 + mdiag[20]*x3 + mdiag[27]*x4 + mdiag[34]*x5 + mdiag[41]*x6 + mdiag[41]*x7;
        mdiag  += 36;
        i2     += 6;
      }
      ierr = PetscLogFlops(93.0*m);CHKERRQ(ierr);
    } else if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      ierr = PetscMemcpy(x,b,A->rmap->N*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      idiag   = a->idiag+49*a->mbs - 49;
      i2      = 7*m - 7;
      x1      = x[i2]; x2 = x[i2+1]; x3 = x[i2+2]; x4 = x[i2+3]; x5 = x[i2+4]; x6 = x[i2+5]; x7 = x[i2+6];
      x[i2]   = idiag[0]*x1 + idiag[7]*x2  + idiag[14]*x3 + idiag[21]*x4 + idiag[28]*x5 + idiag[35]*x6 + idiag[42]*x7;
      x[i2+1] = idiag[1]*x1 + idiag[8]*x2  + idiag[15]*x3 + idiag[22]*x4 + idiag[29]*x5 + idiag[36]*x6 + idiag[43]*x7;
      x[i2+2] = idiag[2]*x1 + idiag[9]*x2  + idiag[16]*x3 + idiag[23]*x4 + idiag[30]*x5 + idiag[37]*x6 + idiag[44]*x7;
      x[i2+3] = idiag[3]*x1 + idiag[10]*x2 + idiag[17]*x3 + idiag[24]*x4 + idiag[31]*x5 + idiag[38]*x6 + idiag[45]*x7;
      x[i2+4] = idiag[4]*x1 + idiag[11]*x2 + idiag[18]*x3 + idiag[25]*x4 + idiag[32]*x5 + idiag[39]*x6 + idiag[46]*x7;
      x[i2+5] = idiag[5]*x1 + idiag[12]*x2 + idiag[19]*x3 + idiag[26]*x4 + idiag[33]*x5 + idiag[40]*x6 + idiag[47]*x7;
      x[i2+6] = idiag[6]*x1 + idiag[13]*x2 + idiag[20]*x3 + idiag[27]*x4 + idiag[34]*x5 + idiag[41]*x6 + idiag[48]*x7;
      idiag -= 49;
      i2    -= 7;
      for (i=m-2; i>=0; i--) {
        v     = aa + 49*(diag[i]+1);
        vi    = aj + diag[i] + 1;
        nz    = ai[i+1] - diag[i] - 1;
        s1    = x[i2]; s2 = x[i2+1]; s3 = x[i2+2]; s4 = x[i2+3]; s5 = x[i2+4]; s6 = x[i2+5]; s7 = x[i2+6];
        while (nz--) {
          idx  = 7*(*vi++);
          x1   = x[idx]; x2 = x[1+idx]; x3 = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx]; x6 = x[5+idx]; x7 = x[6+idx];
          s1  -= v[0]*x1 + v[7]*x2  + v[14]*x3 + v[21]*x4 + v[28]*x5 + v[35]*x6 + v[42]*x7;
          s2  -= v[1]*x1 + v[8]*x2  + v[15]*x3 + v[22]*x4 + v[29]*x5 + v[36]*x6 + v[43]*x7;
          s3  -= v[2]*x1 + v[9]*x2  + v[16]*x3 + v[23]*x4 + v[30]*x5 + v[37]*x6 + v[44]*x7;
          s4  -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4 + v[31]*x5 + v[38]*x6 + v[45]*x7;
          s5  -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4 + v[32]*x5 + v[39]*x6 + v[46]*x7;
          s6  -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4 + v[33]*x5 + v[40]*x6 + v[47]*x7;
          s7  -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4 + v[34]*x5 + v[41]*x6 + v[48]*x7;
          v   += 49;
        }
        x[i2]   = idiag[0]*s1 + idiag[7]*s2  + idiag[14]*s3 + idiag[21]*s4 + idiag[28]*s5 + idiag[35]*s6 + idiag[42]*s7;
        x[i2+1] = idiag[1]*s1 + idiag[8]*s2  + idiag[15]*s3 + idiag[22]*s4 + idiag[29]*s5 + idiag[36]*s6 + idiag[43]*s7;
        x[i2+2] = idiag[2]*s1 + idiag[9]*s2  + idiag[16]*s3 + idiag[23]*s4 + idiag[30]*s5 + idiag[37]*s6 + idiag[44]*s7;
        x[i2+3] = idiag[3]*s1 + idiag[10]*s2 + idiag[17]*s3 + idiag[24]*s4 + idiag[31]*s5 + idiag[38]*s6 + idiag[45]*s7;
        x[i2+4] = idiag[4]*s1 + idiag[11]*s2 + idiag[18]*s3 + idiag[25]*s4 + idiag[32]*s5 + idiag[39]*s6 + idiag[46]*s7;
        x[i2+5] = idiag[5]*s1 + idiag[12]*s2 + idiag[19]*s3 + idiag[26]*s4 + idiag[33]*s5 + idiag[40]*s6 + idiag[47]*s7;
        x[i2+6] = idiag[6]*s1 + idiag[13]*s2 + idiag[20]*s3 + idiag[27]*s4 + idiag[34]*s5 + idiag[41]*s6 + idiag[48]*s7;
        idiag   -= 49;
        i2      -= 7;
      }
      ierr = PetscLogFlops(49.0*(a->nz));CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"Only supports point block SOR with zero initial guess");
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

/*
    Special version for direct calls from Fortran (Used in PETSc-fun3d)
*/
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsetvaluesblocked4_ MATSETVALUESBLOCKED4
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsetvaluesblocked4_ matsetvaluesblocked4
#endif

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "matsetvaluesblocked4_"
void PETSCMAT_DLLEXPORT matsetvaluesblocked4_(Mat *AA,PetscInt *mm,const PetscInt im[],PetscInt *nn,const PetscInt in[],const PetscScalar v[])
{
  Mat               A = *AA;
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscInt          *rp,k,low,high,t,ii,jj,row,nrow,i,col,l,N,m = *mm,n = *nn;
  PetscInt          *ai=a->i,*ailen=a->ilen;
  PetscInt          *aj=a->j,stepval,lastcol = -1;
  const PetscScalar *value = v;
  MatScalar         *ap,*aa = a->a,*bap;

  PetscFunctionBegin;
  if (A->rmap->bs != 4) SETERRABORT(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Can only be called with a block size of 4");
  stepval = (n-1)*4;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; 
    rp   = aj + ai[row]; 
    ap   = aa + 16*ai[row];
    nrow = ailen[row]; 
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      col = in[l]; 
      if (col <= lastcol) low = 0; else high = nrow;
      lastcol = col;
      value = v + k*(stepval+4 + l)*4;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          bap  = ap +  16*i;
          for (ii=0; ii<4; ii++,value+=stepval) {
            for (jj=ii; jj<16; jj+=4) {
              bap[jj] += *value++; 
            }
          }
          goto noinsert2;
        }
      } 
      N = nrow++ - 1; 
      high++; /* added new column index thus must search to one higher than before */
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        PetscMemcpy(ap+16*(ii+1),ap+16*(ii),16*sizeof(MatScalar));
      }
      if (N >= i) {
        PetscMemzero(ap+16*i,16*sizeof(MatScalar));
      }
      rp[i] = col; 
      bap   = ap +  16*i;
      for (ii=0; ii<4; ii++,value+=stepval) {
        for (jj=ii; jj<16; jj+=4) {
          bap[jj] = *value++; 
        }
      }
      noinsert2:;
      low = i;
    }
    ailen[row] = nrow;
  }
  PetscFunctionReturnVoid();
} 
EXTERN_C_END

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsetvalues4_ MATSETVALUES4
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsetvalues4_ matsetvalues4
#endif

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSetValues4_"
void PETSCMAT_DLLEXPORT matsetvalues4_(Mat *AA,PetscInt *mm,PetscInt *im,PetscInt *nn,PetscInt *in,PetscScalar *v)
{
  Mat         A = *AA;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  PetscInt    *rp,k,low,high,t,ii,row,nrow,i,col,l,N,n = *nn,m = *mm;
  PetscInt    *ai=a->i,*ailen=a->ilen;
  PetscInt    *aj=a->j,brow,bcol;
  PetscInt    ridx,cidx,lastcol = -1;
  MatScalar   *ap,value,*aa=a->a,*bap;
  
  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; brow = row/4;  
    rp   = aj + ai[brow]; 
    ap   = aa + 16*ai[brow];
    nrow = ailen[brow]; 
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      col = in[l]; bcol = col/4;
      ridx = row % 4; cidx = col % 4;
      value = v[l + k*n]; 
      if (col <= lastcol) low = 0; else high = nrow;
      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else              low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          bap  = ap +  16*i + 4*cidx + ridx;
          *bap += value;  
          goto noinsert1;
        }
      } 
      N = nrow++ - 1; 
      high++; /* added new column thus must search to one higher than before */
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        PetscMemcpy(ap+16*(ii+1),ap+16*(ii),16*sizeof(MatScalar));
      }
      if (N>=i) {
        PetscMemzero(ap+16*i,16*sizeof(MatScalar));
      }
      rp[i]                    = bcol; 
      ap[16*i + 4*cidx + ridx] = value; 
      noinsert1:;
      low = i;
    }
    ailen[brow] = nrow;
  }
  PetscFunctionReturnVoid();
} 
EXTERN_C_END

/*
     Checks for missing diagonals
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMissingDiagonal_SeqBAIJ"
PetscErrorCode MatMissingDiagonal_SeqBAIJ(Mat A,PetscTruth *missing,PetscInt *d)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data; 
  PetscErrorCode ierr;
  PetscInt       *diag,*jj = a->j,i;

  PetscFunctionBegin;
  ierr = MatMarkDiagonal_SeqBAIJ(A);CHKERRQ(ierr);
  *missing = PETSC_FALSE;
  if (A->rmap->n > 0 && !jj) {
    *missing  = PETSC_TRUE;
    if (d) *d = 0;
    PetscInfo(A,"Matrix has no entries therefor is missing diagonal");
  } else {
    diag     = a->diag;
    for (i=0; i<a->mbs; i++) {
      if (jj[diag[i]] != i) {
        *missing  = PETSC_TRUE;
        if (d) *d = i;
        PetscInfo1(A,"Matrix is missing block diagonal number %D",i);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMarkDiagonal_SeqBAIJ"
PetscErrorCode MatMarkDiagonal_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data; 
  PetscErrorCode ierr;
  PetscInt       i,j,m = a->mbs;

  PetscFunctionBegin;
  if (!a->diag) {
    ierr = PetscMalloc(m*sizeof(PetscInt),&a->diag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,m*sizeof(PetscInt));CHKERRQ(ierr);
    a->free_diag = PETSC_TRUE;
  }
  for (i=0; i<m; i++) {
    a->diag[i] = a->i[i+1];
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      if (a->j[j] == i) {
        a->diag[i] = j;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}


EXTERN PetscErrorCode MatToSymmetricIJ_SeqAIJ(PetscInt,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt**,PetscInt**);

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowIJ_SeqBAIJ"
static PetscErrorCode MatGetRowIJ_SeqBAIJ(Mat A,PetscInt oshift,PetscTruth symmetric,PetscTruth blockcompressed,PetscInt *nn,PetscInt *ia[],PetscInt *ja[],PetscTruth *done)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,n = a->mbs,nz = a->i[n],bs = A->rmap->bs,nbs = 1,k,l,cnt;
  PetscInt       *tia, *tja;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(n,a->i,a->j,0,0,&tia,&tja);CHKERRQ(ierr);
    nz   = tia[n];
  } else {
    tia = a->i; tja = a->j;
  }
    
  if (!blockcompressed && bs > 1) {
    (*nn) *= bs;
    nbs    = bs;
    /* malloc & create the natural set of indices */
    ierr = PetscMalloc((n+1)*bs*sizeof(PetscInt),ia);CHKERRQ(ierr);
    if (n) {
      (*ia)[0] = 0;
      for (j=1; j<bs; j++) {
        (*ia)[j] = (tia[1]-tia[0])*bs+(*ia)[j-1];
      }
    }

    for (i=1; i<n; i++) {
      (*ia)[i*bs] = (tia[i]-tia[i-1])*bs + (*ia)[i*bs-1];
      for (j=1; j<bs; j++) {
        (*ia)[i*bs+j] = (tia[i+1]-tia[i])*bs + (*ia)[i*bs+j-1];
      }
    }
    if (n) {
      (*ia)[n*bs] = (tia[n]-tia[n-1])*bs + (*ia)[n*bs-1];
    }

    if (ja) {
      ierr = PetscMalloc(nz*bs*bs*sizeof(PetscInt),ja);CHKERRQ(ierr);
      cnt = 0;
      for (i=0; i<n; i++) {
        for (j=0; j<bs; j++) {
          for (k=tia[i]; k<tia[i+1]; k++) {
            for (l=0; l<bs; l++) {
              (*ja)[cnt++] = bs*tja[k] + l;
	    }
          }
        }
      }
    }

    n     *= bs;
    nz *= bs*bs;
    if (symmetric) { /* deallocate memory allocated in MatToSymmetricIJ_SeqAIJ() */
      ierr = PetscFree(tia);CHKERRQ(ierr);
      ierr = PetscFree(tja);CHKERRQ(ierr);
    }
  } else if (oshift == 1) {
    if (symmetric) {
      PetscInt nz = tia[A->rmap->n/bs]; 
      /*  add 1 to i and j indices */
      for (i=0; i<A->rmap->n/bs+1; i++) tia[i] = tia[i] + 1;
      *ia = tia;
      if (ja) {
	for (i=0; i<nz; i++) tja[i] = tja[i] + 1;
        *ja = tja;
      }
    } else {
      PetscInt nz = a->i[A->rmap->n/bs]; 
      /* malloc space and  add 1 to i and j indices */
      ierr = PetscMalloc((A->rmap->n/bs+1)*sizeof(PetscInt),ia);CHKERRQ(ierr);
      for (i=0; i<A->rmap->n/bs+1; i++) (*ia)[i] = a->i[i] + 1;
      if (ja) {
	ierr = PetscMalloc(nz*sizeof(PetscInt),ja);CHKERRQ(ierr);
	for (i=0; i<nz; i++) (*ja)[i] = a->j[i] + 1;
      }
    }
  } else {
    *ia = tia;
    if (ja) *ja = tja;
  }
  
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowIJ_SeqBAIJ" 
static PetscErrorCode MatRestoreRowIJ_SeqBAIJ(Mat A,PetscInt oshift,PetscTruth symmetric,PetscTruth blockcompressed,PetscInt *nn,PetscInt *ia[],PetscInt *ja[],PetscTruth *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);
  if ((!blockcompressed && A->rmap->bs > 1) || (symmetric || oshift == 1)) {
    ierr = PetscFree(*ia);CHKERRQ(ierr);
    if (ja) {ierr = PetscFree(*ja);CHKERRQ(ierr);}
  } 
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqBAIJ"
PetscErrorCode MatDestroy_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%D, Cols=%D, NZ=%D",A->rmap->N,A->cmap->n,a->nz);
#endif
  ierr = MatSeqXAIJFreeAIJ(A,&a->a,&a->j,&a->i);CHKERRQ(ierr);
  if (a->row) {
    ierr = ISDestroy(a->row);CHKERRQ(ierr);
  }
  if (a->col) {
    ierr = ISDestroy(a->col);CHKERRQ(ierr);
  }
  if (a->free_diag) {ierr = PetscFree(a->diag);CHKERRQ(ierr);}
  ierr = PetscFree(a->idiag);CHKERRQ(ierr);
  if (a->free_imax_ilen) {ierr = PetscFree2(a->imax,a->ilen);CHKERRQ(ierr);}
  ierr = PetscFree(a->solve_work);CHKERRQ(ierr);
  ierr = PetscFree(a->mult_work);CHKERRQ(ierr);
  if (a->icol) {ierr = ISDestroy(a->icol);CHKERRQ(ierr);}
  ierr = PetscFree(a->saved_values);CHKERRQ(ierr);
  ierr = PetscFree(a->xtoy);CHKERRQ(ierr);
  if (a->compressedrow.use){ierr = PetscFree2(a->compressedrow.i,a->compressedrow.rindex);} 

  if (a->sbaijMat) {ierr = MatDestroy(a->sbaijMat);CHKERRQ(ierr);}
  if (a->parent) {ierr = MatDestroy(a->parent);CHKERRQ(ierr);}
  ierr = PetscFree(a);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJInvertBlockDiagonal_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatStoreValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatRetrieveValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJSetColumnIndices_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_seqaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_seqbaij_seqsbaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqBAIJSetPreallocationCSR_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_SeqBAIJ"
PetscErrorCode MatSetOption_SeqBAIJ(Mat A,MatOption op,PetscTruth flg)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    a->roworiented    = flg;
    break;
  case MAT_KEEP_NONZERO_PATTERN:
    a->keepnonzeropattern = flg;
    break;
  case MAT_NEW_NONZERO_LOCATIONS:
    a->nonew          = (flg ? 0 : 1);
    break;
  case MAT_NEW_NONZERO_LOCATION_ERR:
    a->nonew          = (flg ? -1 : 0);
    break;
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    a->nonew          = (flg ? -2 : 0);
    break;
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
    a->nounused       = (flg ? -1 : 0);
    break;
  case MAT_NEW_DIAGONALS:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_SeqBAIJ"
PetscErrorCode MatGetRow_SeqBAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       itmp,i,j,k,M,*ai,*aj,bs,bn,bp,*idx_i,bs2;
  MatScalar      *aa,*aa_i;
  PetscScalar    *v_i;

  PetscFunctionBegin;
  bs  = A->rmap->bs;
  ai  = a->i;
  aj  = a->j;
  aa  = a->a;
  bs2 = a->bs2;
  
  if (row < 0 || row >= A->rmap->N) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Row %D out of range", row);
  
  bn  = row/bs;   /* Block number */
  bp  = row % bs; /* Block Position */
  M   = ai[bn+1] - ai[bn];
  *nz = bs*M;
  
  if (v) {
    *v = 0;
    if (*nz) {
      ierr = PetscMalloc((*nz)*sizeof(PetscScalar),v);CHKERRQ(ierr);
      for (i=0; i<M; i++) { /* for each block in the block row */
        v_i  = *v + i*bs;
        aa_i = aa + bs2*(ai[bn] + i);
        for (j=bp,k=0; j<bs2; j+=bs,k++) {v_i[k] = aa_i[j];}
      }
    }
  }

  if (idx) {
    *idx = 0;
    if (*nz) {
      ierr = PetscMalloc((*nz)*sizeof(PetscInt),idx);CHKERRQ(ierr);
      for (i=0; i<M; i++) { /* for each block in the block row */
        idx_i = *idx + i*bs;
        itmp  = bs*aj[ai[bn] + i];
        for (j=0; j<bs; j++) {idx_i[j] = itmp++;}
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_SeqBAIJ"
PetscErrorCode MatRestoreRow_SeqBAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (idx) {ierr = PetscFree(*idx);CHKERRQ(ierr);}
  if (v)   {ierr = PetscFree(*v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatSetValues_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_SeqBAIJ"
PetscErrorCode MatTranspose_SeqBAIJ(Mat A,MatReuse reuse,Mat *B)
{ 
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ *)A->data;
  Mat            C;
  PetscErrorCode ierr;
  PetscInt       i,j,k,*aj=a->j,*ai=a->i,bs=A->rmap->bs,mbs=a->mbs,nbs=a->nbs,len,*col;
  PetscInt       *rows,*cols,bs2=a->bs2;
  MatScalar      *array;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX && A == *B && mbs != nbs) SETERRQ(PETSC_ERR_ARG_SIZ,"Square matrix only for in-place");
  if (reuse == MAT_INITIAL_MATRIX || A == *B) {
    ierr = PetscMalloc((1+nbs)*sizeof(PetscInt),&col);CHKERRQ(ierr);
    ierr = PetscMemzero(col,(1+nbs)*sizeof(PetscInt));CHKERRQ(ierr);

    for (i=0; i<ai[mbs]; i++) col[aj[i]] += 1;
    ierr = MatCreate(((PetscObject)A)->comm,&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,A->cmap->n,A->rmap->N,A->cmap->n,A->rmap->N);CHKERRQ(ierr);
    ierr = MatSetType(C,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation_SeqBAIJ(C,bs,PETSC_NULL,col);CHKERRQ(ierr);
    ierr = PetscFree(col);CHKERRQ(ierr);
  } else {
    C = *B;
  }

  array = a->a;
  ierr = PetscMalloc2(bs,PetscInt,&rows,bs,PetscInt,&cols);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    cols[0] = i*bs;
    for (k=1; k<bs; k++) cols[k] = cols[k-1] + 1;
    len = ai[i+1] - ai[i];
    for (j=0; j<len; j++) {
      rows[0] = (*aj++)*bs;
      for (k=1; k<bs; k++) rows[k] = rows[k-1] + 1;
      ierr = MatSetValues_SeqBAIJ(C,bs,rows,bs,cols,array,INSERT_VALUES);CHKERRQ(ierr);
      array += bs2;
    }
  }
  ierr = PetscFree2(rows,cols);CHKERRQ(ierr);
  
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  if (reuse == MAT_INITIAL_MATRIX || *B != A) {
    *B = C;
  } else {
    ierr = MatHeaderCopy(A,C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBAIJ_Binary"
static PetscErrorCode MatView_SeqBAIJ_Binary(Mat A,PetscViewer viewer)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,*col_lens,bs = A->rmap->bs,count,*jj,j,k,l,bs2=a->bs2;
  int            fd;
  PetscScalar    *aa;
  FILE           *file;

  PetscFunctionBegin;
  ierr        = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr        = PetscMalloc((4+A->rmap->N)*sizeof(PetscInt),&col_lens);CHKERRQ(ierr);
  col_lens[0] = MAT_FILE_COOKIE;

  col_lens[1] = A->rmap->N;
  col_lens[2] = A->cmap->n;
  col_lens[3] = a->nz*bs2;

  /* store lengths of each row and write (including header) to file */
  count = 0;
  for (i=0; i<a->mbs; i++) {
    for (j=0; j<bs; j++) {
      col_lens[4+count++] = bs*(a->i[i+1] - a->i[i]);
    }
  }
  ierr = PetscBinaryWrite(fd,col_lens,4+A->rmap->N,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFree(col_lens);CHKERRQ(ierr);

  /* store column indices (zero start index) */
  ierr  = PetscMalloc((a->nz+1)*bs2*sizeof(PetscInt),&jj);CHKERRQ(ierr);
  count = 0;
  for (i=0; i<a->mbs; i++) {
    for (j=0; j<bs; j++) {
      for (k=a->i[i]; k<a->i[i+1]; k++) {
        for (l=0; l<bs; l++) {
          jj[count++] = bs*a->j[k] + l;
        }
      }
    }
  }
  ierr = PetscBinaryWrite(fd,jj,bs2*a->nz,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscFree(jj);CHKERRQ(ierr);

  /* store nonzero values */
  ierr  = PetscMalloc((a->nz+1)*bs2*sizeof(PetscScalar),&aa);CHKERRQ(ierr);
  count = 0;
  for (i=0; i<a->mbs; i++) {
    for (j=0; j<bs; j++) {
      for (k=a->i[i]; k<a->i[i+1]; k++) {
        for (l=0; l<bs; l++) {
          aa[count++] = a->a[bs2*k + l*bs + j];
        }
      }
    }
  }
  ierr = PetscBinaryWrite(fd,aa,bs2*a->nz,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscFree(aa);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
  if (file) {
    fprintf(file,"-matload_block_size %d\n",(int)A->rmap->bs);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBAIJ_ASCII"
static PetscErrorCode MatView_SeqBAIJ_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,bs = A->rmap->bs,k,l,bs2=a->bs2;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer,"  block size is %D\n",bs);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    Mat aij;
    ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&aij);CHKERRQ(ierr);
    ierr = MatView(aij,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(aij);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
     PetscFunctionReturn(0); 
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<a->mbs; i++) {
      for (j=0; j<bs; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i*bs+j);CHKERRQ(ierr);
        for (k=a->i[i]; k<a->i[i+1]; k++) {
          for (l=0; l<bs; l++) {
#if defined(PETSC_USE_COMPLEX)
            if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) > 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G + %Gi) ",bs*a->j[k]+l,
                      PetscRealPart(a->a[bs2*k + l*bs + j]),PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G - %Gi) ",bs*a->j[k]+l,
                      PetscRealPart(a->a[bs2*k + l*bs + j]),-PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",bs*a->j[k]+l,PetscRealPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            }
#else
            if (a->a[bs2*k + l*bs + j] != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",bs*a->j[k]+l,a->a[bs2*k + l*bs + j]);CHKERRQ(ierr);
            }
#endif
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    } 
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<a->mbs; i++) {
      for (j=0; j<bs; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i*bs+j);CHKERRQ(ierr);
        for (k=a->i[i]; k<a->i[i+1]; k++) {
          for (l=0; l<bs; l++) {
#if defined(PETSC_USE_COMPLEX)
            if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) > 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G + %G i) ",bs*a->j[k]+l,
                PetscRealPart(a->a[bs2*k + l*bs + j]),PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G - %G i) ",bs*a->j[k]+l,
                PetscRealPart(a->a[bs2*k + l*bs + j]),-PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",bs*a->j[k]+l,PetscRealPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            }
#else
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",bs*a->j[k]+l,a->a[bs2*k + l*bs + j]);CHKERRQ(ierr);
#endif
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    } 
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBAIJ_Draw_Zoom"
static PetscErrorCode MatView_SeqBAIJ_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat            A = (Mat) Aa;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       row,i,j,k,l,mbs=a->mbs,color,bs=A->rmap->bs,bs2=a->bs2;
  PetscReal      xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  MatScalar      *aa;
  PetscViewer    viewer;

  PetscFunctionBegin; 

  /* still need to add support for contour plot of nonzeros; see MatView_SeqAIJ_Draw_Zoom()*/
  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr); 

  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);

  /* loop over matrix elements drawing boxes */
  color = PETSC_DRAW_BLUE;
  for (i=0,row=0; i<mbs; i++,row+=bs) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          if (PetscRealPart(*aa++) >=  0.) continue;
          ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
        }
      }
    } 
  }
  color = PETSC_DRAW_CYAN;
  for (i=0,row=0; i<mbs; i++,row+=bs) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          if (PetscRealPart(*aa++) != 0.) continue;
          ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
        }
      }
    } 
  }

  color = PETSC_DRAW_RED;
  for (i=0,row=0; i<mbs; i++,row+=bs) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      y_l = A->rmap->N - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          if (PetscRealPart(*aa++) <= 0.) continue;
          ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
        }
      }
    } 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBAIJ_Draw"
static PetscErrorCode MatView_SeqBAIJ_Draw(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal      xl,yl,xr,yr,w,h;
  PetscDraw      draw;
  PetscTruth     isnull;

  PetscFunctionBegin; 

  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  xr  = A->cmap->n; yr = A->rmap->N; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqBAIJ_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBAIJ"
PetscErrorCode MatView_SeqBAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     iascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (iascii){
    ierr = MatView_SeqBAIJ_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqBAIJ_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqBAIJ_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    Mat B;
    ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatView(B,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_SeqBAIJ"
PetscErrorCode MatGetValues_SeqBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],PetscScalar v[])
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  PetscInt    *rp,k,low,high,t,row,nrow,i,col,l,*aj = a->j;
  PetscInt    *ai = a->i,*ailen = a->ilen;
  PetscInt    brow,bcol,ridx,cidx,bs=A->rmap->bs,bs2=a->bs2;
  MatScalar   *ap,*aa = a->a;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over rows */
    row  = im[k]; brow = row/bs;  
    if (row < 0) {v += n; continue;} /* SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative row"); */
    if (row >= A->rmap->N) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Row %D too large", row);
    rp   = aj + ai[brow] ; ap = aa + bs2*ai[brow] ;
    nrow = ailen[brow]; 
    for (l=0; l<n; l++) { /* loop over columns */
      if (in[l] < 0) {v++; continue;} /* SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative column"); */
      if (in[l] >= A->cmap->n) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Column %D too large", in[l]);
      col  = in[l] ; 
      bcol = col/bs;
      cidx = col%bs; 
      ridx = row%bs;
      high = nrow; 
      low  = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          *v++ = ap[bs2*i+bs*cidx+ridx];
          goto finished;
        }
      } 
      *v++ = 0.0;
      finished:;
    }
  }
  PetscFunctionReturn(0);
} 

#define CHUNKSIZE 10
#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlocked_SeqBAIJ"
PetscErrorCode MatSetValuesBlocked_SeqBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscInt          *rp,k,low,high,t,ii,jj,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt          *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscErrorCode    ierr;
  PetscInt          *aj=a->j,nonew=a->nonew,bs2=a->bs2,bs=A->rmap->bs,stepval;
  PetscTruth        roworiented=a->roworiented; 
  const PetscScalar *value = v;
  MatScalar         *ap,*aa = a->a,*bap;

  PetscFunctionBegin;
  if (roworiented) { 
    stepval = (n-1)*bs;
  } else {
    stepval = (m-1)*bs;
  }
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; 
    if (row < 0) continue;
#if defined(PETSC_USE_DEBUG)  
    if (row >= a->mbs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,a->mbs-1);
#endif
    rp   = aj + ai[row]; 
    ap   = aa + bs2*ai[row];
    rmax = imax[row]; 
    nrow = ailen[row]; 
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
#if defined(PETSC_USE_DEBUG)  
      if (in[l] >= a->nbs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l],a->nbs-1);
#endif
      col = in[l]; 
      if (roworiented) { 
        value = v + k*(stepval+bs)*bs + l*bs;
      } else {
        value = v + l*(stepval+bs)*bs + k*bs;
      }
      if (col <= lastcol) low = 0; else high = nrow;
      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          bap  = ap +  bs2*i;
          if (roworiented) { 
            if (is == ADD_VALUES) {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=ii; jj<bs2; jj+=bs) {
                  bap[jj] += *value++; 
                }
              }
            } else {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=ii; jj<bs2; jj+=bs) {
                  bap[jj] = *value++; 
                }
              }
            }
          } else {
            if (is == ADD_VALUES) {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=0; jj<bs; jj++) {
                  *bap++ += *value++; 
                }
              }
            } else {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=0; jj<bs; jj++) {
                  *bap++  = *value++; 
                }
              }
            }
          }
          goto noinsert2;
        }
      } 
      if (nonew == 1) goto noinsert2;
      if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) in the matrix", row, col);
      MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        ierr = PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
      if (N >= i) {
        ierr = PetscMemzero(ap+bs2*i,bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
      rp[i] = col; 
      bap   = ap +  bs2*i;
      if (roworiented) { 
        for (ii=0; ii<bs; ii++,value+=stepval) {
          for (jj=ii; jj<bs2; jj+=bs) {
            bap[jj] = *value++; 
          }
        }
      } else {
        for (ii=0; ii<bs; ii++,value+=stepval) {
          for (jj=0; jj<bs; jj++) {
            *bap++  = *value++; 
          }
        }
      }
      noinsert2:;
      low = i;
    }
    ailen[row] = nrow;
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqBAIJ"
PetscErrorCode MatAssemblyEnd_SeqBAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       fshift = 0,i,j,*ai = a->i,*aj = a->j,*imax = a->imax;
  PetscInt       m = A->rmap->N,*ip,N,*ailen = a->ilen;
  PetscErrorCode ierr;
  PetscInt       mbs = a->mbs,bs2 = a->bs2,rmax = 0;
  MatScalar      *aa = a->a,*ap;
  PetscReal      ratio=0.6; 

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0];
  for (i=1; i<mbs; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax   = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i]; ap = aa + bs2*ai[i];
      N = ailen[i];
      for (j=0; j<N; j++) {
        ip[j-fshift] = ip[j];
        ierr = PetscMemcpy(ap+(j-fshift)*bs2,ap+j*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
    } 
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (mbs) {
    fshift += imax[mbs-1] - ailen[mbs-1];
    ai[mbs] = ai[mbs-1] + ailen[mbs-1];
  }
  /* reset ilen and imax for each row */
  for (i=0; i<mbs; i++) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
  }
  a->nz = ai[mbs]; 

  /* diagonals may have moved, so kill the diagonal pointers */
  a->idiagvalid = PETSC_FALSE;
  if (fshift && a->diag) {
    ierr = PetscFree(a->diag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,-(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
    a->diag = 0;
  }
  if (fshift && a->nounused == -1) {
    SETERRQ4(PETSC_ERR_PLIB, "Unused space detected in matrix: %D X %D block size %D, %D unneeded", m, A->cmap->n, A->rmap->bs, fshift*bs2);
  }
  ierr = PetscInfo5(A,"Matrix size: %D X %D, block size %D; storage space: %D unneeded, %D used\n",m,A->cmap->n,A->rmap->bs,fshift*bs2,a->nz*bs2);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Number of mallocs during MatSetValues is %D\n",a->reallocs);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Most nonzeros blocks in any row is %D\n",rmax);CHKERRQ(ierr);
  a->reallocs          = 0;
  A->info.nz_unneeded  = (PetscReal)fshift*bs2;

  /* check for zero rows. If found a large number of zero rows, use CompressedRow functions */
  if (a->compressedrow.use){ 
    ierr = Mat_CheckCompressedRow(A,&a->compressedrow,a->i,mbs,ratio);CHKERRQ(ierr);
  } 

  A->same_nonzero = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* 
   This function returns an array of flags which indicate the locations of contiguous
   blocks that should be zeroed. for eg: if bs = 3  and is = [0,1,2,3,5,6,7,8,9]
   then the resulting sizes = [3,1,1,3,1] correspondig to sets [(0,1,2),(3),(5),(6,7,8),(9)]
   Assume: sizes should be long enough to hold all the values.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_SeqBAIJ_Check_Blocks"
static PetscErrorCode MatZeroRows_SeqBAIJ_Check_Blocks(PetscInt idx[],PetscInt n,PetscInt bs,PetscInt sizes[], PetscInt *bs_max)
{
  PetscInt   i,j,k,row;
  PetscTruth flg;

  PetscFunctionBegin;
  for (i=0,j=0; i<n; j++) {
    row = idx[i];
    if (row%bs!=0) { /* Not the begining of a block */
      sizes[j] = 1;
      i++; 
    } else if (i+bs > n) { /* complete block doesn't exist (at idx end) */
      sizes[j] = 1;         /* Also makes sure atleast 'bs' values exist for next else */
      i++; 
    } else { /* Begining of the block, so check if the complete block exists */
      flg = PETSC_TRUE;
      for (k=1; k<bs; k++) {
        if (row+k != idx[i+k]) { /* break in the block */
          flg = PETSC_FALSE;
          break;
        }
      }
      if (flg) { /* No break in the bs */
        sizes[j] = bs;
        i+= bs;
      } else {
        sizes[j] = 1;
        i++;
      }
    }
  }
  *bs_max = j;
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_SeqBAIJ"
PetscErrorCode MatZeroRows_SeqBAIJ(Mat A,PetscInt is_n,const PetscInt is_idx[],PetscScalar diag)
{
  Mat_SeqBAIJ    *baij=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,k,count,*rows;
  PetscInt       bs=A->rmap->bs,bs2=baij->bs2,*sizes,row,bs_max;
  PetscScalar    zero = 0.0;
  MatScalar      *aa;

  PetscFunctionBegin;
  /* Make a copy of the IS and  sort it */
  /* allocate memory for rows,sizes */
  ierr  = PetscMalloc2(is_n,PetscInt,&rows,2*is_n,PetscInt,&sizes);CHKERRQ(ierr);

  /* copy IS values to rows, and sort them */
  for (i=0; i<is_n; i++) { rows[i] = is_idx[i]; }
  ierr = PetscSortInt(is_n,rows);CHKERRQ(ierr);
  if (baij->keepnonzeropattern) {
    for (i=0; i<is_n; i++) { sizes[i] = 1; }
    bs_max = is_n;
    A->same_nonzero = PETSC_TRUE;
  } else {
    ierr = MatZeroRows_SeqBAIJ_Check_Blocks(rows,is_n,bs,sizes,&bs_max);CHKERRQ(ierr);
    A->same_nonzero = PETSC_FALSE;
  }

  for (i=0,j=0; i<bs_max; j+=sizes[i],i++) {
    row   = rows[j];
    if (row < 0 || row > A->rmap->N) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"row %D out of range",row);
    count = (baij->i[row/bs +1] - baij->i[row/bs])*bs;
    aa    = ((MatScalar*)(baij->a)) + baij->i[row/bs]*bs2 + (row%bs);
    if (sizes[i] == bs && !baij->keepnonzeropattern) {
      if (diag != 0.0) {
        if (baij->ilen[row/bs] > 0) {
          baij->ilen[row/bs]       = 1;
          baij->j[baij->i[row/bs]] = row/bs;
          ierr = PetscMemzero(aa,count*bs*sizeof(MatScalar));CHKERRQ(ierr);
        } 
        /* Now insert all the diagonal values for this bs */
        for (k=0; k<bs; k++) {
          ierr = (*A->ops->setvalues)(A,1,rows+j+k,1,rows+j+k,&diag,INSERT_VALUES);CHKERRQ(ierr);
        } 
      } else { /* (diag == 0.0) */
        baij->ilen[row/bs] = 0;
      } /* end (diag == 0.0) */
    } else { /* (sizes[i] != bs) */
#if defined (PETSC_USE_DEBUG)
      if (sizes[i] != 1) SETERRQ(PETSC_ERR_PLIB,"Internal Error. Value should be 1");
#endif
      for (k=0; k<count; k++) { 
        aa[0] =  zero; 
        aa    += bs;
      }
      if (diag != 0.0) {
        ierr = (*A->ops->setvalues)(A,1,rows+j,1,rows+j,&diag,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = PetscFree2(rows,sizes);CHKERRQ(ierr);
  ierr = MatAssemblyEnd_SeqBAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_SeqBAIJ"
PetscErrorCode MatSetValues_SeqBAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscInt       *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt       *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt       *aj=a->j,nonew=a->nonew,bs=A->rmap->bs,brow,bcol;
  PetscErrorCode ierr;
  PetscInt       ridx,cidx,bs2=a->bs2;
  PetscTruth     roworiented=a->roworiented;
  MatScalar      *ap,value,*aa=a->a,*bap;

  PetscFunctionBegin;
  if (v) PetscValidScalarPointer(v,6);
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k];
    brow = row/bs;  
    if (row < 0) continue;
#if defined(PETSC_USE_DEBUG)  
    if (row >= A->rmap->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->N-1);
#endif
    rp   = aj + ai[brow]; 
    ap   = aa + bs2*ai[brow];
    rmax = imax[brow]; 
    nrow = ailen[brow]; 
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
#if defined(PETSC_USE_DEBUG)  
      if (in[l] >= A->cmap->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l],A->cmap->n-1);
#endif
      col = in[l]; bcol = col/bs;
      ridx = row % bs; cidx = col % bs;
      if (roworiented) {
        value = v[l + k*n]; 
      } else {
        value = v[k + l*m];
      }
      if (col <= lastcol) low = 0; else high = nrow;
      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else              low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          bap  = ap +  bs2*i + bs*cidx + ridx;
          if (is == ADD_VALUES) *bap += value;  
          else                  *bap  = value; 
          goto noinsert1;
        }
      } 
      if (nonew == 1) goto noinsert1;
      if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) in the matrix", row, col);
      MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        ierr     = PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
      if (N>=i) {
        ierr = PetscMemzero(ap+bs2*i,bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
      rp[i]                      = bcol; 
      ap[bs2*i + bs*cidx + ridx] = value; 
      a->nz++;
      noinsert1:;
      low = i;
    }
    ailen[brow] = nrow;
  }
  A->same_nonzero = PETSC_FALSE;
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactor_SeqBAIJ"
PetscErrorCode MatILUFactor_SeqBAIJ(Mat inA,IS row,IS col,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)inA->data;
  Mat            outA;
  PetscErrorCode ierr;
  PetscTruth     row_identity,col_identity;

  PetscFunctionBegin;
  if (info->levels != 0) SETERRQ(PETSC_ERR_SUP,"Only levels = 0 supported for in-place ILU");
  ierr = ISIdentity(row,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(col,&col_identity);CHKERRQ(ierr);
  if (!row_identity || !col_identity) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Row and column permutations must be identity for in-place ILU");
  }

  outA          = inA; 
  inA->factor   = MAT_FACTOR_LU;

  ierr = MatMarkDiagonal_SeqBAIJ(inA);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)row);CHKERRQ(ierr);
  if (a->row) { ierr = ISDestroy(a->row);CHKERRQ(ierr); }
  a->row = row;
  ierr = PetscObjectReference((PetscObject)col);CHKERRQ(ierr);
  if (a->col) { ierr = ISDestroy(a->col);CHKERRQ(ierr); }
  a->col = col;
  
  /* Create the invert permutation so that it can be used in MatLUFactorNumeric() */
  if (a->icol) {
    ierr = ISDestroy(a->icol);CHKERRQ(ierr);
  }
  ierr = ISInvertPermutation(col,PETSC_DECIDE,&a->icol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(inA,a->icol);CHKERRQ(ierr);
  
  ierr = MatSeqBAIJSetNumericFactorization_inplace(inA,(PetscTruth)(row_identity && col_identity));CHKERRQ(ierr);
  if (!a->solve_work) {
    ierr = PetscMalloc((inA->rmap->N+inA->rmap->bs)*sizeof(PetscScalar),&a->solve_work);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(inA,(inA->rmap->N+inA->rmap->bs)*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric(outA,inA,info);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqBAIJSetColumnIndices_SeqBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBAIJSetColumnIndices_SeqBAIJ(Mat mat,PetscInt *indices)
{
  Mat_SeqBAIJ *baij = (Mat_SeqBAIJ *)mat->data;
  PetscInt    i,nz,mbs;

  PetscFunctionBegin;
  nz  = baij->maxnz;
  mbs = baij->mbs;
  for (i=0; i<nz; i++) {
    baij->j[i] = indices[i];
  }
  baij->nz = nz;
  for (i=0; i<mbs; i++) {
    baij->ilen[i] = baij->imax[i];
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSeqBAIJSetColumnIndices"
/*@
    MatSeqBAIJSetColumnIndices - Set the column indices for all the rows
       in the matrix.

  Input Parameters:
+  mat - the SeqBAIJ matrix
-  indices - the column indices

  Level: advanced

  Notes:
    This can be called if you have precomputed the nonzero structure of the 
  matrix and want to provide it to the matrix object to improve the performance
  of the MatSetValues() operation.

    You MUST have set the correct numbers of nonzeros per row in the call to 
  MatCreateSeqBAIJ(), and the columns indices MUST be sorted.

    MUST be called before any calls to MatSetValues();

@*/ 
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBAIJSetColumnIndices(Mat mat,PetscInt *indices)
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(indices,2);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSeqBAIJSetColumnIndices_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,indices);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Wrong type of matrix to set column indices");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMaxAbs_SeqBAIJ"
PetscErrorCode MatGetRowMaxAbs_SeqBAIJ(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,n,row,bs,*ai,*aj,mbs;
  PetscReal      atmp;
  PetscScalar    *x,zero = 0.0;
  MatScalar      *aa;
  PetscInt       ncols,brow,krow,kcol; 

  PetscFunctionBegin;
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");  
  bs   = A->rmap->bs;
  aa   = a->a;
  ai   = a->i;
  aj   = a->j;
  mbs  = a->mbs;

  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->rmap->N) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<mbs; i++) {
    ncols = ai[1] - ai[0]; ai++;
    brow  = bs*i;
    for (j=0; j<ncols; j++){
      for (kcol=0; kcol<bs; kcol++){
        for (krow=0; krow<bs; krow++){         
          atmp = PetscAbsScalar(*aa);aa++;
          row = brow + krow;    /* row index */
          /* printf("val[%d,%d]: %G\n",row,bcol+kcol,atmp); */
          if (PetscAbsScalar(x[row]) < atmp) {x[row] = atmp; if (idx) idx[row] = bs*(*aj) + kcol;}
        }
      }
      aj++; 
    }   
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy_SeqBAIJ"
PetscErrorCode MatCopy_SeqBAIJ(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* If the two matrices have the same copy implementation, use fast copy. */
  if (str == SAME_NONZERO_PATTERN && (A->ops->copy == B->ops->copy)) {
    Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data; 
    Mat_SeqBAIJ *b = (Mat_SeqBAIJ*)B->data; 

    if (a->i[A->rmap->N] != b->i[B->rmap->N]) {
      SETERRQ(PETSC_ERR_ARG_INCOMP,"Number of nonzeros in two matrices are different");
    }
    ierr = PetscMemcpy(b->a,a->a,(a->i[A->rmap->N])*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_SeqBAIJ"
PetscErrorCode MatSetUpPreallocation_SeqBAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatSeqBAIJSetPreallocation_SeqBAIJ(A,-PetscMax(A->rmap->bs,1),PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetArray_SeqBAIJ"
PetscErrorCode MatGetArray_SeqBAIJ(Mat A,PetscScalar *array[])
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data; 
  PetscFunctionBegin;
  *array = a->a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreArray_SeqBAIJ"
PetscErrorCode MatRestoreArray_SeqBAIJ(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_SeqBAIJ"
PetscErrorCode MatAXPY_SeqBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_SeqBAIJ    *x  = (Mat_SeqBAIJ *)X->data,*y = (Mat_SeqBAIJ *)Y->data;
  PetscErrorCode ierr;
  PetscInt       i,bs=Y->rmap->bs,j,bs2;
  PetscBLASInt   one=1,bnz = PetscBLASIntCast(x->nz);

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {   
    PetscScalar alpha = a;
    BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one);
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    if (y->xtoy && y->XtoY != X) {
      ierr = PetscFree(y->xtoy);CHKERRQ(ierr);
      ierr = MatDestroy(y->XtoY);CHKERRQ(ierr);
    }
    if (!y->xtoy) { /* get xtoy */
      ierr = MatAXPYGetxtoy_Private(x->mbs,x->i,x->j,PETSC_NULL, y->i,y->j,PETSC_NULL, &y->xtoy);CHKERRQ(ierr);
      y->XtoY = X;
      ierr = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
    }
    bs2 = bs*bs;
    for (i=0; i<x->nz; i++) {
      j = 0;
      while (j < bs2){
        y->a[bs2*y->xtoy[i]+j] += a*(x->a[bs2*i+j]); 
        j++; 
      }
    }
    ierr = PetscInfo3(Y,"ratio of nnz(X)/nnz(Y): %D/%D = %G\n",bs2*x->nz,bs2*y->nz,(PetscReal)(bs2*x->nz)/(bs2*y->nz));CHKERRQ(ierr);
  } else {
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetBlockSize_SeqBAIJ"
PetscErrorCode MatSetBlockSize_SeqBAIJ(Mat A,PetscInt bs)
{
  PetscInt rbs,cbs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(A->rmap,&rbs);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(A->cmap,&cbs);CHKERRQ(ierr);
  if (rbs != bs) SETERRQ2(PETSC_ERR_ARG_SIZ,"Attempt to set block size %d with BAIJ %d",bs,rbs);
  if (cbs != bs) SETERRQ2(PETSC_ERR_ARG_SIZ,"Attempt to set block size %d with BAIJ %d",bs,cbs);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRealPart_SeqBAIJ"
PetscErrorCode MatRealPart_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data; 
  PetscInt       i,nz = a->bs2*a->i[a->mbs];
  MatScalar      *aa = a->a;

  PetscFunctionBegin;  
  for (i=0; i<nz; i++) aa[i] = PetscRealPart(aa[i]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatImaginaryPart_SeqBAIJ"
PetscErrorCode MatImaginaryPart_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data; 
  PetscInt       i,nz = a->bs2*a->i[a->mbs];
  MatScalar      *aa = a->a;

  PetscFunctionBegin;  
  for (i=0; i<nz; i++) aa[i] = PetscImaginaryPart(aa[i]);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatFDColoringCreate_SeqAIJ(Mat,ISColoring,MatFDColoring);

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnIJ_SeqBAIJ"
/*
    Code almost idential to MatGetColumnIJ_SeqAIJ() should share common code
*/
PetscErrorCode MatGetColumnIJ_SeqBAIJ(Mat A,PetscInt oshift,PetscTruth symmetric,PetscTruth inodecompressed,PetscInt *nn,PetscInt *ia[],PetscInt *ja[],PetscTruth *done)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       bs = A->rmap->bs,i,*collengths,*cia,*cja,n = A->cmap->n/bs,m = A->rmap->n/bs;
  PetscInt       nz = a->i[m],row,*jj,mr,col;

  PetscFunctionBegin;  
  *nn = n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    SETERRQ(PETSC_ERR_SUP,"Not for BAIJ matrices");
  } else {
    ierr = PetscMalloc((n+1)*sizeof(PetscInt),&collengths);CHKERRQ(ierr);
    ierr = PetscMemzero(collengths,n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMalloc((n+1)*sizeof(PetscInt),&cia);CHKERRQ(ierr);
    ierr = PetscMalloc((nz+1)*sizeof(PetscInt),&cja);CHKERRQ(ierr);
    jj = a->j;
    for (i=0; i<nz; i++) {
      collengths[jj[i]]++;
    }
    cia[0] = oshift;
    for (i=0; i<n; i++) {
      cia[i+1] = cia[i] + collengths[i];
    }
    ierr = PetscMemzero(collengths,n*sizeof(PetscInt));CHKERRQ(ierr);
    jj   = a->j;
    for (row=0; row<m; row++) {
      mr = a->i[row+1] - a->i[row];
      for (i=0; i<mr; i++) {
        col = *jj++;
        cja[cia[col] + collengths[col]++ - oshift] = row + oshift;  
      }
    }
    ierr = PetscFree(collengths);CHKERRQ(ierr);
    *ia = cia; *ja = cja;
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreColumnIJ_SeqBAIJ"
PetscErrorCode MatRestoreColumnIJ_SeqBAIJ(Mat A,PetscInt oshift,PetscTruth symmetric,PetscTruth inodecompressed,PetscInt *n,PetscInt *ia[],PetscInt *ja[],PetscTruth *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  if (!ia) PetscFunctionReturn(0);
  ierr = PetscFree(*ia);CHKERRQ(ierr);
  ierr = PetscFree(*ja);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringApply_BAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatFDColoringApply_BAIJ(Mat J,MatFDColoring coloring,Vec x1,MatStructure *flag,void *sctx)
{
  PetscErrorCode (*f)(void*,Vec,Vec,void*) = (PetscErrorCode (*)(void*,Vec,Vec,void *))coloring->f;
  PetscErrorCode ierr;
  PetscInt       bs = J->rmap->bs,i,j,k,start,end,l,row,col,*srows,**vscaleforrow,m1,m2;
  PetscScalar    dx,*y,*xx,*w3_array;
  PetscScalar    *vscale_array;
  PetscReal      epsilon = coloring->error_rel,umin = coloring->umin,unorm; 
  Vec            w1=coloring->w1,w2=coloring->w2,w3;
  void           *fctx = coloring->fctx;
  PetscTruth     flg = PETSC_FALSE;
  PetscInt       ctype=coloring->ctype,N,col_start=0,col_end=0;
  Vec            x1_tmp;

  PetscFunctionBegin;    
  PetscValidHeaderSpecific(J,MAT_COOKIE,1);
  PetscValidHeaderSpecific(coloring,MAT_FDCOLORING_COOKIE,2);
  PetscValidHeaderSpecific(x1,VEC_COOKIE,3);
  if (!f) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call MatFDColoringSetFunction()");

  ierr = PetscLogEventBegin(MAT_FDColoringApply,coloring,J,x1,0);CHKERRQ(ierr);
  ierr = MatSetUnfactored(J);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-mat_fd_coloring_dont_rezero",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscInfo(coloring,"Not calling MatZeroEntries()\n");CHKERRQ(ierr);
  } else {
    PetscTruth assembled;
    ierr = MatAssembled(J,&assembled);CHKERRQ(ierr);
    if (assembled) {
      ierr = MatZeroEntries(J);CHKERRQ(ierr);
    }
  }

  x1_tmp = x1; 
  if (!coloring->vscale){ 
    ierr = VecDuplicate(x1_tmp,&coloring->vscale);CHKERRQ(ierr);
  }
    
  /*
    This is a horrible, horrible, hack. See DMMGComputeJacobian_Multigrid() it inproperly sets
    coloring->F for the coarser grids from the finest
  */
  if (coloring->F) {
    ierr = VecGetLocalSize(coloring->F,&m1);CHKERRQ(ierr);
    ierr = VecGetLocalSize(w1,&m2);CHKERRQ(ierr);
    if (m1 != m2) {  
      coloring->F = 0; 
      }    
    }   

  if (coloring->htype[0] == 'w') { /* tacky test; need to make systematic if we add other approaches to computing h*/
    ierr = VecNorm(x1_tmp,NORM_2,&unorm);CHKERRQ(ierr);
  }
  ierr = VecGetOwnershipRange(w1,&start,&end);CHKERRQ(ierr); /* OwnershipRange is used by ghosted x! */
      
  /* Set w1 = F(x1) */
  if (coloring->F) {
    w1          = coloring->F; /* use already computed value of function */
    coloring->F = 0; 
  } else {
    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,x1_tmp,w1,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
  }
      
  if (!coloring->w3) {
    ierr = VecDuplicate(x1_tmp,&coloring->w3);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(coloring,coloring->w3);CHKERRQ(ierr);
  }
  w3 = coloring->w3;

    CHKMEMQ;
    /* Compute all the local scale factors, including ghost points */
  ierr = VecGetLocalSize(x1_tmp,&N);CHKERRQ(ierr);
  ierr = VecGetArray(x1_tmp,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  if (ctype == IS_COLORING_GHOSTED){
    col_start = 0; col_end = N;
  } else if (ctype == IS_COLORING_GLOBAL){
    xx = xx - start;
    vscale_array = vscale_array - start;
    col_start = start; col_end = N + start;
  }    CHKMEMQ;
  for (col=col_start; col<col_end; col++){ 
    /* Loop over each local column, vscale[col] = 1./(epsilon*dx[col]) */      
    if (coloring->htype[0] == 'w') {
      dx = 1.0 + unorm;
    } else {
      dx  = xx[col];
    }
    if (dx == 0.0) dx = 1.0;
#if !defined(PETSC_USE_COMPLEX)
    if (dx < umin && dx >= 0.0)      dx = umin;
    else if (dx < 0.0 && dx > -umin) dx = -umin;
#else
    if (PetscAbsScalar(dx) < umin && PetscRealPart(dx) >= 0.0)     dx = umin;
    else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < umin) dx = -umin;
#endif
    dx               *= epsilon;
    vscale_array[col] = 1.0/dx;
  }     CHKMEMQ;
  if (ctype == IS_COLORING_GLOBAL)  vscale_array = vscale_array + start;      
  ierr = VecRestoreArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  if (ctype == IS_COLORING_GLOBAL){
    ierr = VecGhostUpdateBegin(coloring->vscale,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(coloring->vscale,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
        CHKMEMQ;
  if (coloring->vscaleforrow) {
    vscaleforrow = coloring->vscaleforrow;
  } else {                     
    SETERRQ(PETSC_ERR_ARG_NULL,"Null Object: coloring->vscaleforrow");
  }


  ierr = PetscMalloc(bs*sizeof(PetscInt),&srows);CHKERRQ(ierr);
  /*
    Loop over each color
  */
  ierr = VecGetArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  for (k=0; k<coloring->ncolors; k++) { 
    coloring->currentcolor = k;
    for (i=0; i<bs; i++) {
      ierr = VecCopy(x1_tmp,w3);CHKERRQ(ierr);
      ierr = VecGetArray(w3,&w3_array);CHKERRQ(ierr);
      if (ctype == IS_COLORING_GLOBAL) w3_array = w3_array - start;
      /*
	Loop over each column associated with color 
	adding the perturbation to the vector w3.
      */
      for (l=0; l<coloring->ncolumns[k]; l++) {
	col = i + bs*coloring->columns[k][l];    /* local column of the matrix we are probing for */
	if (coloring->htype[0] == 'w') {
	  dx = 1.0 + unorm;
	} else {
	  dx  = xx[col];
	}
	if (dx == 0.0) dx = 1.0;
#if !defined(PETSC_USE_COMPLEX)
	if (dx < umin && dx >= 0.0)      dx = umin;
	else if (dx < 0.0 && dx > -umin) dx = -umin;
#else
	if (PetscAbsScalar(dx) < umin && PetscRealPart(dx) >= 0.0)     dx = umin;
	else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < umin) dx = -umin;
#endif
	dx            *= epsilon;
	if (!PetscAbsScalar(dx)) SETERRQ(PETSC_ERR_PLIB,"Computed 0 differencing parameter");
	w3_array[col] += dx;
      } 
      if (ctype == IS_COLORING_GLOBAL) w3_array = w3_array + start;
      ierr = VecRestoreArray(w3,&w3_array);CHKERRQ(ierr);

      /*
	Evaluate function at w3 = x1 + dx (here dx is a vector of perturbations)
	w2 = F(x1 + dx) - F(x1)
      */
      ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
      ierr = (*f)(sctx,w3,w2,fctx);CHKERRQ(ierr);        
      ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
      ierr = VecAXPY(w2,-1.0,w1);CHKERRQ(ierr); 
        
      /*
	Loop over rows of vector, putting results into Jacobian matrix
      */
      ierr = VecGetArray(w2,&y);CHKERRQ(ierr);
      for (l=0; l<coloring->nrows[k]; l++) {
	row    = bs*coloring->rows[k][l];             /* local row index */
	col    = i + bs*coloring->columnsforrow[k][l];    /* global column index */
        for (j=0; j<bs; j++) {
  	  y[row+j] *= vscale_array[j+bs*vscaleforrow[k][l]];
          srows[j]  = row + start + j;
        }
	ierr   = MatSetValues(J,bs,srows,1,&col,y+row,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(w2,&y);CHKERRQ(ierr);
    }
  } /* endof for each color */
  if (ctype == IS_COLORING_GLOBAL) xx = xx + start; 
  ierr = VecRestoreArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x1_tmp,&xx);CHKERRQ(ierr);
  ierr = PetscFree(srows);CHKERRQ(ierr);
   
  coloring->currentcolor = -1;
  ierr  = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_FDColoringApply,coloring,J,x1,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqBAIJ,
       MatGetRow_SeqBAIJ,
       MatRestoreRow_SeqBAIJ,
       MatMult_SeqBAIJ_N,
/* 4*/ MatMultAdd_SeqBAIJ_N,
       MatMultTranspose_SeqBAIJ,
       MatMultTransposeAdd_SeqBAIJ,
       0,
       0,
       0,
/*10*/ 0,
       MatLUFactor_SeqBAIJ,
       0,
       0,
       MatTranspose_SeqBAIJ,
/*15*/ MatGetInfo_SeqBAIJ,
       MatEqual_SeqBAIJ,
       MatGetDiagonal_SeqBAIJ,
       MatDiagonalScale_SeqBAIJ,
       MatNorm_SeqBAIJ,
/*20*/ 0,
       MatAssemblyEnd_SeqBAIJ,
       MatSetOption_SeqBAIJ,
       MatZeroEntries_SeqBAIJ,
/*24*/ MatZeroRows_SeqBAIJ,
       0,
       0,
       0,
       0,
/*29*/ MatSetUpPreallocation_SeqBAIJ,
       0,
       0,
       MatGetArray_SeqBAIJ,
       MatRestoreArray_SeqBAIJ,
/*34*/ MatDuplicate_SeqBAIJ,
       0,
       0,
       MatILUFactor_SeqBAIJ,
       0,
/*39*/ MatAXPY_SeqBAIJ,
       MatGetSubMatrices_SeqBAIJ,
       MatIncreaseOverlap_SeqBAIJ,
       MatGetValues_SeqBAIJ,
       MatCopy_SeqBAIJ,
/*44*/ 0,
       MatScale_SeqBAIJ,
       0,
       0,
       0,
/*49*/ MatSetBlockSize_SeqBAIJ,
       MatGetRowIJ_SeqBAIJ,
       MatRestoreRowIJ_SeqBAIJ,
       MatGetColumnIJ_SeqBAIJ,
       MatRestoreColumnIJ_SeqBAIJ,
/*54*/ MatFDColoringCreate_SeqAIJ,
       0,
       0,
       0,
       MatSetValuesBlocked_SeqBAIJ,
/*59*/ MatGetSubMatrix_SeqBAIJ,
       MatDestroy_SeqBAIJ,
       MatView_SeqBAIJ,
       0,       
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ MatGetRowMaxAbs_SeqBAIJ,
       0,
       MatConvert_Basic,
       0,
       0,
/*74*/ 0,
       MatFDColoringApply_BAIJ,
       0,
       0,
       0,
/*79*/ 0,
       0,
       0,
       0,
       MatLoad_SeqBAIJ,
/*84*/ 0,
       0,
       0,
       0,
       0,
/*89*/ 0,
       0,
       0,
       0,
       0,
/*94*/ 0,
       0,
       0,
       0,
       0,
/*99*/0,
       0,
       0,
       0,
       0,
/*104*/0,
       MatRealPart_SeqBAIJ,
       MatImaginaryPart_SeqBAIJ,
       0,
       0,
/*109*/0,
       0,
       0,
       0,
       MatMissingDiagonal_SeqBAIJ,
/*114*/0,
       0,
       0,
       0,
       0,
/*119*/0,
       0,
       MatMultHermitianTranspose_SeqBAIJ,
       MatMultHermitianTransposeAdd_SeqBAIJ
};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues_SeqBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatStoreValues_SeqBAIJ(Mat mat)
{
  Mat_SeqBAIJ    *aij = (Mat_SeqBAIJ *)mat->data;
  PetscInt       nz = aij->i[mat->rmap->N]*mat->rmap->bs*aij->bs2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (aij->nonew != 1) {
    SETERRQ(PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  }

  /* allocate space for values if not already there */
  if (!aij->saved_values) {
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&aij->saved_values);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(mat,(nz+1)*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  /* copy values over */
  ierr = PetscMemcpy(aij->saved_values,aij->a,nz*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatRetrieveValues_SeqBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatRetrieveValues_SeqBAIJ(Mat mat)
{
  Mat_SeqBAIJ    *aij = (Mat_SeqBAIJ *)mat->data;
  PetscErrorCode ierr;
  PetscInt       nz = aij->i[mat->rmap->N]*mat->rmap->bs*aij->bs2;

  PetscFunctionBegin;
  if (aij->nonew != 1) {
    SETERRQ(PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  }
  if (!aij->saved_values) {
    SETERRQ(PETSC_ERR_ORDER,"Must call MatStoreValues(A);first");
  }

  /* copy values over */
  ierr = PetscMemcpy(aij->a,aij->saved_values,nz*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqBAIJ_SeqAIJ(Mat, MatType,MatReuse,Mat*);
extern PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqBAIJ_SeqSBAIJ(Mat, MatType,MatReuse,Mat*);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqBAIJSetPreallocation_SeqBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBAIJSetPreallocation_SeqBAIJ(Mat B,PetscInt bs,PetscInt nz,PetscInt *nnz)
{
  Mat_SeqBAIJ    *b;
  PetscErrorCode ierr;
  PetscInt       i,mbs,nbs,bs2,newbs = PetscAbs(bs);
  PetscTruth     flg,skipallocation = PETSC_FALSE;

  PetscFunctionBegin;

  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }

  if (bs < 0) {
    ierr = PetscOptionsBegin(((PetscObject)B)->comm,((PetscObject)B)->prefix,"Block options for SEQBAIJ matrix 1","Mat");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-mat_block_size","Set the blocksize used to store the matrix","MatSeqBAIJSetPreallocation",newbs,&newbs,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    bs   = PetscAbs(bs);
  }
  if (nnz && newbs != bs) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot change blocksize from command line if setting nnz");
  }
  bs   = newbs;

  ierr = PetscLayoutSetBlockSize(B->rmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);

  B->preallocated = PETSC_TRUE;

  mbs  = B->rmap->n/bs;
  nbs  = B->cmap->n/bs;
  bs2  = bs*bs;

  if (mbs*bs!=B->rmap->n || nbs*bs!=B->cmap->n) {
    SETERRQ3(PETSC_ERR_ARG_SIZ,"Number rows %D, cols %D must be divisible by blocksize %D",B->rmap->N,B->cmap->n,bs);
  }

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  if (nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %D",nz);
  if (nnz) {
    for (i=0; i<mbs; i++) {
      if (nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local row %D value %D",i,nnz[i]);
      if (nnz[i] > nbs) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than block row length: local row %D value %D rowlength %D",i,nnz[i],nbs);
    }
  }

  b       = (Mat_SeqBAIJ*)B->data;
  ierr = PetscOptionsBegin(((PetscObject)B)->comm,PETSC_NULL,"Optimize options for SEQBAIJ matrix 2 ","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-mat_no_unroll","Do not optimize for block size (slow)",PETSC_NULL,PETSC_FALSE,&flg,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!flg) {
    switch (bs) {
    case 1:
      B->ops->mult            = MatMult_SeqBAIJ_1;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_1;
      B->ops->sor             = MatSOR_SeqBAIJ_1;
      break;
    case 2:
      B->ops->mult            = MatMult_SeqBAIJ_2;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_2;
      B->ops->sor             = MatSOR_SeqBAIJ_2;
      break;
    case 3:
      B->ops->mult            = MatMult_SeqBAIJ_3;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_3;
      B->ops->sor             = MatSOR_SeqBAIJ_3;
      break;
    case 4:
      B->ops->mult            = MatMult_SeqBAIJ_4;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_4;
      B->ops->sor             = MatSOR_SeqBAIJ_4;
      break;
    case 5:
      B->ops->mult            = MatMult_SeqBAIJ_5;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_5;
      B->ops->sor             = MatSOR_SeqBAIJ_5;
      break;
    case 6:
      B->ops->mult            = MatMult_SeqBAIJ_6;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_6;
      B->ops->sor             = MatSOR_SeqBAIJ_6;
      break;
    case 7:
      B->ops->mult            = MatMult_SeqBAIJ_7; 
      B->ops->multadd         = MatMultAdd_SeqBAIJ_7;
      B->ops->sor             = MatSOR_SeqBAIJ_7;
      break;
    case 15:
      B->ops->mult = MatMult_SeqBAIJ_15_ver1;
      break;
    default:
      B->ops->mult            = MatMult_SeqBAIJ_N; 
      B->ops->multadd         = MatMultAdd_SeqBAIJ_N;
      break;
    }
  }
  B->rmap->bs      = bs;
  b->mbs     = mbs;
  b->nbs     = nbs;
  if (!skipallocation) {
    if (!b->imax) {
      ierr = PetscMalloc2(mbs,PetscInt,&b->imax,mbs,PetscInt,&b->ilen);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(B,2*mbs*sizeof(PetscInt));
      b->free_imax_ilen = PETSC_TRUE;
    }
    /* b->ilen will count nonzeros in each block row so far. */
    for (i=0; i<mbs; i++) { b->ilen[i] = 0;}
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
      else if (nz <= 0)        nz = 1;
      for (i=0; i<mbs; i++) b->imax[i] = nz;
      nz = nz*mbs;
    } else {
      nz = 0;
      for (i=0; i<mbs; i++) {b->imax[i] = nnz[i]; nz += nnz[i];}
    }

    /* allocate the matrix space */
    ierr = MatSeqXAIJFreeAIJ(B,&b->a,&b->j,&b->i);CHKERRQ(ierr);
    ierr = PetscMalloc3(bs2*nz,PetscScalar,&b->a,nz,PetscInt,&b->j,B->rmap->N+1,PetscInt,&b->i);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(B,(B->rmap->N+1)*sizeof(PetscInt)+nz*(bs2*sizeof(PetscScalar)+sizeof(PetscInt)));CHKERRQ(ierr);
    ierr  = PetscMemzero(b->a,nz*bs2*sizeof(MatScalar));CHKERRQ(ierr);
    ierr  = PetscMemzero(b->j,nz*sizeof(PetscInt));CHKERRQ(ierr);
    b->singlemalloc = PETSC_TRUE;
    b->i[0] = 0;
    for (i=1; i<mbs+1; i++) {
      b->i[i] = b->i[i-1] + b->imax[i-1];
    }
    b->free_a     = PETSC_TRUE;
    b->free_ij    = PETSC_TRUE;
  } else {
    b->free_a     = PETSC_FALSE;
    b->free_ij    = PETSC_FALSE;
  }

  B->rmap->bs          = bs;
  b->bs2              = bs2;
  b->mbs              = mbs;
  b->nz               = 0;
  b->maxnz            = nz;
  B->info.nz_unneeded = (PetscReal)b->maxnz*bs2;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqBAIJSetPreallocationCSR_SeqBAIJ"
PetscErrorCode MatSeqBAIJSetPreallocationCSR_SeqBAIJ(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[],const PetscScalar V[])
{
  PetscInt       i,m,nz,nz_max=0,*nnz;
  PetscScalar    *values=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (bs < 1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %D",bs);

  ierr = PetscLayoutSetBlockSize(B->rmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  m = B->rmap->n/bs;

  if (ii[0] != 0) { SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "ii[0] must be 0 but it is %D",ii[0]); }
  ierr = PetscMalloc((m+1) * sizeof(PetscInt), &nnz);CHKERRQ(ierr);
  for(i=0; i<m; i++) {
    nz = ii[i+1]- ii[i];
    if (nz < 0) { SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Local row %D has a negative number of columns %D",i,nz); }
    nz_max = PetscMax(nz_max, nz);
    nnz[i] = nz; 
  }
  ierr = MatSeqBAIJSetPreallocation(B,bs,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);

  values = (PetscScalar*)V;
  if (!values) {
    ierr = PetscMalloc(bs*bs*(nz_max+1)*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,bs*bs*nz_max*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (i=0; i<m; i++) {
    PetscInt          ncols  = ii[i+1] - ii[i];
    const PetscInt    *icols = jj + ii[i];
    const PetscScalar *svals = values + (V ? (bs*bs*ii[i]) : 0);
    ierr = MatSetValuesBlocked_SeqBAIJ(B,1,&i,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (!V) { ierr = PetscFree(values);CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


EXTERN_C_BEGIN
extern PetscErrorCode PETSCMAT_DLLEXPORT MatGetFactor_seqbaij_petsc(Mat,MatFactorType,Mat*);
extern PetscErrorCode PETSCMAT_DLLEXPORT MatGetFactorAvailable_seqbaij_petsc(Mat,MatFactorType,Mat*);
EXTERN_C_END

/*MC
   MATSEQBAIJ - MATSEQBAIJ = "seqbaij" - A matrix type to be used for sequential block sparse matrices, based on 
   block sparse compressed row format.

   Options Database Keys:
. -mat_type seqbaij - sets the matrix type to "seqbaij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSeqBAIJ()
M*/


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqBAIJ(Mat B)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat_SeqBAIJ    *b;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)B)->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  ierr    = PetscNewLog(B,Mat_SeqBAIJ,&b);CHKERRQ(ierr);
  B->data = (void*)b;
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->mapping               = 0;
  b->row                   = 0;
  b->col                   = 0;
  b->icol                  = 0;
  b->reallocs              = 0;
  b->saved_values          = 0;

  b->roworiented           = PETSC_TRUE;
  b->nonew                 = 0;
  b->diag                  = 0;
  b->solve_work            = 0;
  b->mult_work             = 0;
  B->spptr                 = 0;
  B->info.nz_unneeded      = (PetscReal)b->maxnz*b->bs2;
  b->keepnonzeropattern    = PETSC_FALSE;
  b->xtoy                  = 0;
  b->XtoY                  = 0;
  b->compressedrow.use     = PETSC_FALSE;
  b->compressedrow.nrows   = 0;
  b->compressedrow.i       = PETSC_NULL;
  b->compressedrow.rindex  = PETSC_NULL;
  b->compressedrow.checked = PETSC_FALSE;
  B->same_nonzero          = PETSC_FALSE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactorAvailable_petsc_C",
                                     "MatGetFactorAvailable_seqbaij_petsc",
                                     MatGetFactorAvailable_seqbaij_petsc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_petsc_C",
                                     "MatGetFactor_seqbaij_petsc",
                                     MatGetFactor_seqbaij_petsc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqBAIJInvertBlockDiagonal_C",
                                     "MatInvertBlockDiagonal_SeqBAIJ",
                                      MatInvertBlockDiagonal_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatStoreValues_C",
                                     "MatStoreValues_SeqBAIJ",
                                      MatStoreValues_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatRetrieveValues_C",
                                     "MatRetrieveValues_SeqBAIJ",
                                      MatRetrieveValues_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqBAIJSetColumnIndices_C",
                                     "MatSeqBAIJSetColumnIndices_SeqBAIJ",
                                      MatSeqBAIJSetColumnIndices_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqbaij_seqaij_C",
                                     "MatConvert_SeqBAIJ_SeqAIJ",
                                      MatConvert_SeqBAIJ_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqbaij_seqsbaij_C",
                                     "MatConvert_SeqBAIJ_SeqSBAIJ",
                                      MatConvert_SeqBAIJ_SeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqBAIJSetPreallocation_C",
                                     "MatSeqBAIJSetPreallocation_SeqBAIJ",
                                      MatSeqBAIJSetPreallocation_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqBAIJSetPreallocationCSR_C",
                                     "MatSeqBAIJSetPreallocationCSR_SeqBAIJ",
                                      MatSeqBAIJSetPreallocationCSR_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQBAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicateNoCreate_SeqBAIJ"
PetscErrorCode MatDuplicateNoCreate_SeqBAIJ(Mat C,Mat A,MatDuplicateOption cpvalues,PetscTruth mallocmatspace)
{
  Mat_SeqBAIJ    *c = (Mat_SeqBAIJ*)C->data,*a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,mbs = a->mbs,nz = a->nz,bs2 = a->bs2;

  PetscFunctionBegin;
  if (a->i[mbs] != nz) SETERRQ(PETSC_ERR_PLIB,"Corrupt matrix");

  if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
    c->imax = a->imax;
    c->ilen = a->ilen;
    c->free_imax_ilen = PETSC_FALSE;
  } else {
    ierr = PetscMalloc2(mbs,PetscInt,&c->imax,mbs,PetscInt,&c->ilen);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(C,2*mbs*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      c->imax[i] = a->imax[i];
      c->ilen[i] = a->ilen[i]; 
    }
    c->free_imax_ilen = PETSC_TRUE;
  }

  /* allocate the matrix space */
  if (mallocmatspace){
    if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
      ierr = PetscMalloc(bs2*nz*sizeof(PetscScalar),&c->a);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(C,a->i[mbs]*bs2*sizeof(PetscScalar));CHKERRQ(ierr);
      c->singlemalloc = PETSC_FALSE;
      c->free_ij      = PETSC_FALSE;
      c->i            = a->i;
      c->j            = a->j;
      c->parent       = A;
      ierr            = PetscObjectReference((PetscObject)A);CHKERRQ(ierr); 
      ierr            = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
      ierr            = MatSetOption(C,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc3(bs2*nz,PetscScalar,&c->a,nz,PetscInt,&c->j,mbs+1,PetscInt,&c->i);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(C,a->i[mbs]*(bs2*sizeof(PetscScalar)+sizeof(PetscInt))+(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
      c->singlemalloc = PETSC_TRUE;
      c->free_ij      = PETSC_TRUE;
      ierr = PetscMemcpy(c->i,a->i,(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
      if (mbs > 0) {
	ierr = PetscMemcpy(c->j,a->j,nz*sizeof(PetscInt));CHKERRQ(ierr);
	if (cpvalues == MAT_COPY_VALUES) {
	  ierr = PetscMemcpy(c->a,a->a,bs2*nz*sizeof(MatScalar));CHKERRQ(ierr);
	} else {
	  ierr = PetscMemzero(c->a,bs2*nz*sizeof(MatScalar));CHKERRQ(ierr);
	}
      }
    }
  }

  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;
  ierr = PetscLayoutCopy(A->rmap,&C->rmap);CHKERRQ(ierr);  
  ierr = PetscLayoutCopy(A->cmap,&C->cmap);CHKERRQ(ierr);  
  c->bs2         = a->bs2;
  c->mbs         = a->mbs;
  c->nbs         = a->nbs;

  if (a->diag) {
    if (cpvalues == MAT_SHARE_NONZERO_PATTERN) {
      c->diag      = a->diag;
      c->free_diag = PETSC_FALSE;
    } else {
      ierr = PetscMalloc((mbs+1)*sizeof(PetscInt),&c->diag);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(C,(mbs+1)*sizeof(PetscInt));CHKERRQ(ierr);
      for (i=0; i<mbs; i++) {
        c->diag[i] = a->diag[i];
      }
      c->free_diag = PETSC_TRUE;
    }
  } else c->diag        = 0;
  c->nz                 = a->nz;
  c->maxnz              = a->maxnz;
  c->solve_work         = 0;
  c->mult_work          = 0;
  c->free_a             = PETSC_TRUE;
  c->free_ij            = PETSC_TRUE;
  C->preallocated       = PETSC_TRUE;
  C->assembled          = PETSC_TRUE;

  c->compressedrow.use     = a->compressedrow.use;
  c->compressedrow.nrows   = a->compressedrow.nrows;
  c->compressedrow.checked = a->compressedrow.checked;
  if (a->compressedrow.checked && a->compressedrow.use){
    i = a->compressedrow.nrows;
    ierr = PetscMalloc2(i+1,PetscInt,&c->compressedrow.i,i+1,PetscInt,&c->compressedrow.rindex);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(C,(2*i+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(c->compressedrow.i,a->compressedrow.i,(i+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(c->compressedrow.rindex,a->compressedrow.rindex,i*sizeof(PetscInt));CHKERRQ(ierr); 
  } else {
    c->compressedrow.use    = PETSC_FALSE;
    c->compressedrow.i      = PETSC_NULL;
    c->compressedrow.rindex = PETSC_NULL;
  }
  C->same_nonzero = A->same_nonzero;
  ierr = PetscFListDuplicate(((PetscObject)A)->qlist,&((PetscObject)C)->qlist);CHKERRQ(ierr);
  ierr = PetscMemcpy(C->ops,A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_SeqBAIJ"
PetscErrorCode MatDuplicate_SeqBAIJ(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
    PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(((PetscObject)A)->comm,B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,A->rmap->N,A->cmap->n,A->rmap->N,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*B,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatDuplicateNoCreate_SeqBAIJ(*B,A,cpvalues,PETSC_TRUE);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_SeqBAIJ"
PetscErrorCode MatLoad_SeqBAIJ(PetscViewer viewer, const MatType type,Mat *A)
{
  Mat_SeqBAIJ    *a;
  Mat            B;
  PetscErrorCode ierr;
  PetscInt       i,nz,header[4],*rowlengths=0,M,N,bs=1;
  PetscInt       *mask,mbs,*jj,j,rowcount,nzcount,k,*browlengths,maskcount;
  PetscInt       kmax,jcount,block,idx,point,nzcountb,extra_rows;
  PetscInt       *masked,nmask,tmp,bs2,ishift;
  PetscMPIInt    size;
  int            fd;
  PetscScalar    *aa;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(comm,PETSC_NULL,"Options for loading SEQBAIJ matrix","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  bs2  = bs*bs;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,"view must have one processor");
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT);CHKERRQ(ierr);
  if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"not Mat object");
  M = header[1]; N = header[2]; nz = header[3];

  if (header[3] < 0) {
    SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format, cannot load as SeqBAIJ");
  }

  if (M != N) SETERRQ(PETSC_ERR_SUP,"Can only do square matrices");

  /* 
     This code adds extra rows to make sure the number of rows is 
    divisible by the blocksize
  */
  mbs        = M/bs;
  extra_rows = bs - M + bs*(mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  mbs++;
  if (extra_rows) {
    ierr = PetscInfo(viewer,"Padding loaded matrix to match blocksize\n");CHKERRQ(ierr);
  }

  /* read in row lengths */
  ierr = PetscMalloc((M+extra_rows)*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
  for (i=0; i<extra_rows; i++) rowlengths[M+i] = 1;

  /* read in column indices */
  ierr = PetscMalloc((nz+extra_rows)*sizeof(PetscInt),&jj);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,jj,nz,PETSC_INT);CHKERRQ(ierr);
  for (i=0; i<extra_rows; i++) jj[nz+i] = M+i;

  /* loop over row lengths determining block row lengths */
  ierr     = PetscMalloc(mbs*sizeof(PetscInt),&browlengths);CHKERRQ(ierr);
  ierr     = PetscMemzero(browlengths,mbs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr     = PetscMalloc2(mbs,PetscInt,&mask,mbs,PetscInt,&masked);CHKERRQ(ierr);
  ierr     = PetscMemzero(mask,mbs*sizeof(PetscInt));CHKERRQ(ierr);
  rowcount = 0;
  nzcount = 0;
  for (i=0; i<mbs; i++) {
    nmask = 0;
    for (j=0; j<bs; j++) {
      kmax = rowlengths[rowcount];
      for (k=0; k<kmax; k++) {
        tmp = jj[nzcount++]/bs;
        if (!mask[tmp]) {masked[nmask++] = tmp; mask[tmp] = 1;}
      }
      rowcount++;
    }
    browlengths[i] += nmask;
    /* zero out the mask elements we set */
    for (j=0; j<nmask; j++) mask[masked[j]] = 0;
  }

  /* create our matrix */
  ierr = MatCreate(comm,&B);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,M+extra_rows,N+extra_rows);
  ierr = MatSetType(B,type);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation_SeqBAIJ(B,bs,0,browlengths);CHKERRQ(ierr);
  a = (Mat_SeqBAIJ*)B->data;

  /* set matrix "i" values */
  a->i[0] = 0;
  for (i=1; i<= mbs; i++) {
    a->i[i]      = a->i[i-1] + browlengths[i-1];
    a->ilen[i-1] = browlengths[i-1];
  }
  a->nz         = 0;
  for (i=0; i<mbs; i++) a->nz += browlengths[i];

  /* read in nonzero values */
  ierr = PetscMalloc((nz+extra_rows)*sizeof(PetscScalar),&aa);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,aa,nz,PETSC_SCALAR);CHKERRQ(ierr);
  for (i=0; i<extra_rows; i++) aa[nz+i] = 1.0;

  /* set "a" and "j" values into matrix */
  nzcount = 0; jcount = 0;
  for (i=0; i<mbs; i++) {
    nzcountb = nzcount;
    nmask    = 0;
    for (j=0; j<bs; j++) {
      kmax = rowlengths[i*bs+j];
      for (k=0; k<kmax; k++) {
        tmp = jj[nzcount++]/bs;
	if (!mask[tmp]) { masked[nmask++] = tmp; mask[tmp] = 1;}
      }
    }
    /* sort the masked values */
    ierr = PetscSortInt(nmask,masked);CHKERRQ(ierr);

    /* set "j" values into matrix */
    maskcount = 1;
    for (j=0; j<nmask; j++) {
      a->j[jcount++]  = masked[j];
      mask[masked[j]] = maskcount++; 
    }
    /* set "a" values into matrix */
    ishift = bs2*a->i[i];
    for (j=0; j<bs; j++) {
      kmax = rowlengths[i*bs+j];
      for (k=0; k<kmax; k++) {
        tmp       = jj[nzcountb]/bs ;
        block     = mask[tmp] - 1;
        point     = jj[nzcountb] - bs*tmp;
        idx       = ishift + bs2*block + j + bs*point;
        a->a[idx] = (MatScalar)aa[nzcountb++];
      }
    }
    /* zero out the mask elements we set */
    for (j=0; j<nmask; j++) mask[masked[j]] = 0;
  }
  if (jcount != a->nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Bad binary matrix");

  ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  ierr = PetscFree(browlengths);CHKERRQ(ierr);
  ierr = PetscFree(aa);CHKERRQ(ierr);
  ierr = PetscFree(jj);CHKERRQ(ierr);
  ierr = PetscFree2(mask,masked);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView_Private(B);CHKERRQ(ierr);

  *A = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqBAIJ"
/*@C
   MatCreateSeqBAIJ - Creates a sparse matrix in block AIJ (block
   compressed row) format.  For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  bs - size of block
.  m - number of rows
.  n - number of columns
.  nz - number of nonzero blocks  per block row (same for all rows)
-  nnz - array containing the number of nonzero blocks in the various block rows 
         (possibly different for each block row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Options Database Keys:
.   -mat_no_unroll - uses code that does not unroll the loops in the 
                     block calculations (much slower)
.    -mat_block_size - size of the blocks to use

   Level: intermediate

   Notes:
   The number of rows and columns must be divisible by blocksize.

   If the nnz parameter is given then the nz parameter is ignored

   A nonzero block is any block that as 1 or more nonzeros in it

   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For additional details, see the users manual chapter on
   matrices.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateMPIBAIJ()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqBAIJ(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;
 
  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation_SeqBAIJ(*A,bs,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSeqBAIJSetPreallocation"
/*@C
   MatSeqBAIJSetPreallocation - Sets the block size and expected nonzeros
   per row in the matrix. For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  A - the matrix
.  bs - size of block
.  nz - number of block nonzeros per block row (same for all rows)
-  nnz - array containing the number of block nonzeros in the various block rows 
         (possibly different for each block row) or PETSC_NULL

   Options Database Keys:
.   -mat_no_unroll - uses code that does not unroll the loops in the 
                     block calculations (much slower)
.    -mat_block_size - size of the blocks to use

   Level: intermediate

   Notes:
   If the nnz parameter is given then the nz parameter is ignored

   You can call MatGetInfo() to get information on how effective the preallocation was;
   for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
   You can also run with the option -info and look for messages with the string 
   malloc in them to see if additional memory allocation was needed.

   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For additional details, see the users manual chapter on
   matrices.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateMPIBAIJ(), MatGetInfo()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBAIJSetPreallocation(Mat B,PetscInt bs,PetscInt nz,const PetscInt nnz[])
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt,PetscInt,const PetscInt[]);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatSeqBAIJSetPreallocation_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,bs,nz,nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSeqBAIJSetPreallocationCSR"
/*@C
   MatSeqBAIJSetPreallocationCSR - Allocates memory for a sparse sequential matrix in AIJ format
   (the default sequential PETSc format).  

   Collective on MPI_Comm

   Input Parameters:
+  A - the matrix 
.  i - the indices into j for the start of each local row (starts with zero)
.  j - the column indices for each local row (starts with zero) these must be sorted for each row
-  v - optional values in the matrix

   Level: developer

.keywords: matrix, aij, compressed row, sparse

.seealso: MatCreate(), MatCreateSeqBAIJ(), MatSetValues(), MatSeqBAIJSetPreallocation(), MATSEQBAIJ
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqBAIJSetPreallocationCSR(Mat B,PetscInt bs,const PetscInt i[],const PetscInt j[], const PetscScalar v[])
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatSeqBAIJSetPreallocationCSR_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,bs,i,j,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqBAIJWithArrays"
/*@
     MatCreateSeqBAIJWithArrays - Creates an sequential BAIJ matrix using matrix elements 
              (upper triangular entries in CSR format) provided by the user.

     Collective on MPI_Comm

   Input Parameters:
+  comm - must be an MPI communicator of size 1
.  bs - size of block
.  m - number of rows
.  n - number of columns
.  i - row indices
.  j - column indices
-  a - matrix values

   Output Parameter:
.  mat - the matrix

   Level: intermediate

   Notes:
       The i, j, and a arrays are not copied by this routine, the user must free these arrays
    once the matrix is destroyed

       You cannot set new nonzero locations into this matrix, that will generate an error.

       The i and j indices are 0 based

.seealso: MatCreate(), MatCreateMPIBAIJ(), MatCreateSeqBAIJ()

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqBAIJWithArrays(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt* i,PetscInt*j,PetscScalar *a,Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       ii;
  Mat_SeqBAIJ    *baij;

  PetscFunctionBegin;
  if (bs != 1) SETERRQ1(PETSC_ERR_SUP,"block size %D > 1 is not supported yet",bs);
  if (i[0]) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation_SeqBAIJ(*mat,bs,MAT_SKIP_ALLOCATION,0);CHKERRQ(ierr);
  baij = (Mat_SeqBAIJ*)(*mat)->data;
  ierr = PetscMalloc2(m,PetscInt,&baij->imax,m,PetscInt,&baij->ilen);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(*mat,2*m*sizeof(PetscInt));CHKERRQ(ierr);

  baij->i = i;
  baij->j = j;
  baij->a = a;
  baij->singlemalloc = PETSC_FALSE;
  baij->nonew        = -1;             /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  baij->free_a       = PETSC_FALSE; 
  baij->free_ij       = PETSC_FALSE; 

  for (ii=0; ii<m; ii++) {
    baij->ilen[ii] = baij->imax[ii] = i[ii+1] - i[ii];
#if defined(PETSC_USE_DEBUG)
    if (i[ii+1] - i[ii] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Negative row length in i (row indices) row = %d length = %d",ii,i[ii+1] - i[ii]);
#endif    
  }
#if defined(PETSC_USE_DEBUG)
  for (ii=0; ii<baij->i[m]; ii++) {
    if (j[ii] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Negative column index at location = %d index = %d",ii,j[ii]);
    if (j[ii] > n - 1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column index to large at location = %d index = %d",ii,j[ii]);
  }
#endif    

  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
