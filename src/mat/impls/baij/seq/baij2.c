#define PETSCMAT_DLL

#include "src/mat/impls/baij/seq/baij.h"
#include "src/inline/spops.h"
#include "src/inline/ilu.h"
#include "petscbt.h"

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_SeqBAIJ"
PetscErrorCode MatIncreaseOverlap_SeqBAIJ(Mat A,PetscInt is_max,IS is[],PetscInt ov)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       row,i,j,k,l,m,n,*idx,*nidx,isz,val,ival;
  PetscInt       start,end,*ai,*aj,bs,*nidx2;
  PetscBT        table;

  PetscFunctionBegin;
  m     = a->mbs;
  ai    = a->i;
  aj    = a->j;
  bs    = A->bs;

  if (ov < 0)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");

  ierr = PetscBTCreate(m,table);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&nidx);CHKERRQ(ierr); 
  ierr = PetscMalloc((A->m+1)*sizeof(PetscInt),&nidx2);CHKERRQ(ierr);

  for (i=0; i<is_max; i++) {
    /* Initialise the two local arrays */
    isz  = 0;
    ierr = PetscBTMemzero(m,table);CHKERRQ(ierr);
                 
    /* Extract the indices, assume there can be duplicate entries */
    ierr = ISGetIndices(is[i],&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is[i],&n);CHKERRQ(ierr);

    /* Enter these into the temp arrays i.e mark table[row], enter row into new index */
    for (j=0; j<n ; ++j){
      ival = idx[j]/bs; /* convert the indices into block indices */
      if (ival>=m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"index greater than mat-dim");
      if(!PetscBTLookupSet(table,ival)) { nidx[isz++] = ival;}
    }
    ierr = ISRestoreIndices(is[i],&idx);CHKERRQ(ierr);
    ierr = ISDestroy(is[i]);CHKERRQ(ierr);
    
    k = 0;
    for (j=0; j<ov; j++){ /* for each overlap*/
      n = isz;
      for (; k<n ; k++){ /* do only those rows in nidx[k], which are not done yet */
        row   = nidx[k];
        start = ai[row];
        end   = ai[row+1];
        for (l = start; l<end ; l++){
          val = aj[l];
          if (!PetscBTLookupSet(table,val)) {nidx[isz++] = val;}
        }
      }
    }
    /* expand the Index Set */
    for (j=0; j<isz; j++) {
      for (k=0; k<bs; k++)
        nidx2[j*bs+k] = nidx[j]*bs+k;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,isz*bs,nidx2,is+i);CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(table);CHKERRQ(ierr);
  ierr = PetscFree(nidx);CHKERRQ(ierr);
  ierr = PetscFree(nidx2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_SeqBAIJ_Private"
PetscErrorCode MatGetSubMatrix_SeqBAIJ_Private(Mat A,IS isrow,IS iscol,PetscInt cs,MatReuse scall,Mat *B)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*c;
  PetscErrorCode ierr;
  PetscInt       *smap,i,k,kstart,kend,oldcols = a->nbs,*lens;
  PetscInt       row,mat_i,*mat_j,tcol,*mat_ilen;
  PetscInt       *irow,*icol,nrows,ncols,*ssmap,bs=A->bs,bs2=a->bs2;
  PetscInt       *aj = a->j,*ai = a->i;
  MatScalar      *mat_a;
  Mat            C;
  PetscTruth     flag;

  PetscFunctionBegin;
  ierr = ISSorted(iscol,(PetscTruth*)&i);CHKERRQ(ierr);
  if (!i) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"IS is not sorted");

  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);

  ierr = PetscMalloc((1+oldcols)*sizeof(PetscInt),&smap);CHKERRQ(ierr);
  ssmap = smap;
  ierr = PetscMalloc((1+nrows)*sizeof(PetscInt),&lens);CHKERRQ(ierr);
  ierr  = PetscMemzero(smap,oldcols*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<ncols; i++) smap[icol[i]] = i+1;
  /* determine lens of each row */
  for (i=0; i<nrows; i++) {
    kstart  = ai[irow[i]]; 
    kend    = kstart + a->ilen[irow[i]];
    lens[i] = 0;
      for (k=kstart; k<kend; k++) {
        if (ssmap[aj[k]]) {
          lens[i]++;
        }
      }
    }
  /* Create and fill new matrix */
  if (scall == MAT_REUSE_MATRIX) {
    c = (Mat_SeqBAIJ *)((*B)->data);

    if (c->mbs!=nrows || c->nbs!=ncols || (*B)->bs!=bs) SETERRQ(PETSC_ERR_ARG_SIZ,"Submatrix wrong size");
    ierr = PetscMemcmp(c->ilen,lens,c->mbs *sizeof(PetscInt),&flag);CHKERRQ(ierr);
    if (!flag) {
      SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong no of nonzeros");
    }
    ierr = PetscMemzero(c->ilen,c->mbs*sizeof(PetscInt));CHKERRQ(ierr);
    C = *B;
  } else {  
    ierr = MatCreate(A->comm,&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,nrows*bs,ncols*bs,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetType(C,A->type_name);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation_SeqBAIJ(C,bs,0,lens);CHKERRQ(ierr);
  }
  c = (Mat_SeqBAIJ *)(C->data);
  for (i=0; i<nrows; i++) {
    row    = irow[i];
    kstart = ai[row]; 
    kend   = kstart + a->ilen[row];
    mat_i  = c->i[i];
    mat_j  = c->j + mat_i; 
    mat_a  = c->a + mat_i*bs2;
    mat_ilen = c->ilen + i;
    for (k=kstart; k<kend; k++) {
      if ((tcol=ssmap[a->j[k]])) {
        *mat_j++ = tcol - 1;
        ierr     = PetscMemcpy(mat_a,a->a+k*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
        mat_a   += bs2;
        (*mat_ilen)++;
      }
    }
  }
    
  /* Free work space */
  ierr = ISRestoreIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = PetscFree(smap);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  *B = C;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_SeqBAIJ"
PetscErrorCode MatGetSubMatrix_SeqBAIJ(Mat A,IS isrow,IS iscol,PetscInt cs,MatReuse scall,Mat *B)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  IS             is1,is2;
  PetscErrorCode ierr;
  PetscInt       *vary,*iary,*irow,*icol,nrows,ncols,i,bs=A->bs,count;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);
  
  /* Verify if the indices corespond to each element in a block 
   and form the IS with compressed IS */
  ierr = PetscMalloc(2*(a->mbs+1)*sizeof(PetscInt),&vary);CHKERRQ(ierr);
  iary = vary + a->mbs;
  ierr = PetscMemzero(vary,(a->mbs)*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<nrows; i++) vary[irow[i]/bs]++;
  count = 0;
  for (i=0; i<a->mbs; i++) {
    if (vary[i]!=0 && vary[i]!=bs) SETERRQ(PETSC_ERR_ARG_SIZ,"Index set does not match blocks");
    if (vary[i]==bs) iary[count++] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,count,iary,&is1);CHKERRQ(ierr);
  
  ierr = PetscMemzero(vary,(a->mbs)*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<ncols; i++) vary[icol[i]/bs]++;
  count = 0;
  for (i=0; i<a->mbs; i++) {
    if (vary[i]!=0 && vary[i]!=bs) SETERRQ(PETSC_ERR_PLIB,"Internal error in PETSc");
    if (vary[i]==bs) iary[count++] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,count,iary,&is2);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = PetscFree(vary);CHKERRQ(ierr);

  ierr = MatGetSubMatrix_SeqBAIJ_Private(A,is1,is2,cs,scall,B);CHKERRQ(ierr);
  ISDestroy(is1);
  ISDestroy(is2);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_SeqBAIJ"
PetscErrorCode MatGetSubMatrices_SeqBAIJ(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscMalloc((n+1)*sizeof(Mat),B);CHKERRQ(ierr);
  }

  for (i=0; i<n; i++) {
    ierr = MatGetSubMatrix_SeqBAIJ(A,irow[i],icol[i],PETSC_DECIDE,scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------*/
/* Should check that shapes of vectors and matrices match */
/* -------------------------------------------------------*/
#include "petscblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_1"
PetscErrorCode MatMult_SeqBAIJ_1(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*z,sum;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs,i,*idx,*ii,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  idx = a->j;
  v   = a->a;
  if (usecprow){
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
  }

  for (i=0; i<mbs; i++) {
    n    = ii[1] - ii[0]; ii++;
    sum  = 0.0;
    while (n--) sum += *v++ * x[*idx++];
    if (usecprow){
      z[ridx[i]] = sum;
    } else {
      z[i] = sum;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*a->nz - mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_2"
PetscErrorCode MatMult_SeqBAIJ_2(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*z = 0,*xb,sum1,sum2,*zarray;
  PetscScalar    x1,x2;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

  idx = a->j;
  v   = a->a;
  if (usecprow){
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0;
    for (j=0; j<n; j++) {
      xb = x + 2*(*idx++); x1 = xb[0]; x2 = xb[1];
      sum1 += v[0]*x1 + v[2]*x2;
      sum2 += v[1]*x1 + v[3]*x2;
      v += 4;
    }
    if (usecprow) z = zarray + 2*ridx[i];
    z[0] = sum1; z[1] = sum2;
    if (!usecprow) z += 2;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(8*a->nz - 2*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_3"
PetscErrorCode MatMult_SeqBAIJ_3(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*z = 0,*xb,sum1,sum2,sum3,x1,x2,x3,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;
  

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*v,*z,*xb)
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

  idx = a->j;
  v   = a->a;
  if (usecprow){
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
    for (j=0; j<n; j++) {
      xb = x + 3*(*idx++); x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
      sum1 += v[0]*x1 + v[3]*x2 + v[6]*x3;
      sum2 += v[1]*x1 + v[4]*x2 + v[7]*x3;
      sum3 += v[2]*x1 + v[5]*x2 + v[8]*x3;
      v += 9;
    }
    if (usecprow) z = zarray + 3*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3;
    if (!usecprow) z += 3;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(18*a->nz - 3*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_4"
PetscErrorCode MatMult_SeqBAIJ_4(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*z = 0,*xb,sum1,sum2,sum3,sum4,x1,x2,x3,x4,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

  idx = a->j;
  v   = a->a;
  if (usecprow){
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0;
    for (j=0; j<n; j++) {
      xb = x + 4*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      sum1 += v[0]*x1 + v[4]*x2 + v[8]*x3   + v[12]*x4;
      sum2 += v[1]*x1 + v[5]*x2 + v[9]*x3   + v[13]*x4;
      sum3 += v[2]*x1 + v[6]*x2 + v[10]*x3  + v[14]*x4;
      sum4 += v[3]*x1 + v[7]*x2 + v[11]*x3  + v[15]*x4;
      v += 16;
    }
    if (usecprow) z = zarray + 4*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4;
    if (!usecprow) z += 4;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(32*a->nz - 4*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_5"
PetscErrorCode MatMult_SeqBAIJ_5(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    sum1,sum2,sum3,sum4,sum5,x1,x2,x3,x4,x5,*xb,*z = 0,*x,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

  idx = a->j;
  v   = a->a;
  if (usecprow){
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0; sum5 = 0.0;
    for (j=0; j<n; j++) {
      xb = x + 5*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4];
      sum1 += v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      sum2 += v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      sum3 += v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      sum4 += v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      sum5 += v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v += 25;
    }
    if (usecprow) z = zarray + 5*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5;
    if (!usecprow) z += 5;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(50*a->nz - 5*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_6"
PetscErrorCode MatMult_SeqBAIJ_6(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*z = 0,*xb,sum1,sum2,sum3,sum4,sum5,sum6;
  PetscScalar    x1,x2,x3,x4,x5,x6,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

  idx = a->j;
  v   = a->a;
  if (usecprow){
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0; sum5 = 0.0; sum6 = 0.0;
    for (j=0; j<n; j++) {
      xb = x + 6*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4]; x6 = xb[5];
      sum1 += v[0]*x1 + v[6]*x2  + v[12]*x3  + v[18]*x4 + v[24]*x5 + v[30]*x6;
      sum2 += v[1]*x1 + v[7]*x2  + v[13]*x3  + v[19]*x4 + v[25]*x5 + v[31]*x6;
      sum3 += v[2]*x1 + v[8]*x2  + v[14]*x3  + v[20]*x4 + v[26]*x5 + v[32]*x6;
      sum4 += v[3]*x1 + v[9]*x2  + v[15]*x3  + v[21]*x4 + v[27]*x5 + v[33]*x6;
      sum5 += v[4]*x1 + v[10]*x2 + v[16]*x3  + v[22]*x4 + v[28]*x5 + v[34]*x6;
      sum6 += v[5]*x1 + v[11]*x2 + v[17]*x3  + v[23]*x4 + v[29]*x5 + v[35]*x6;
      v += 36;
    }
    if (usecprow) z = zarray + 6*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5; z[5] = sum6;
    if (!usecprow) z += 6;
  }

  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(72*a->nz - 6*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_7"
PetscErrorCode MatMult_SeqBAIJ_7(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*z = 0,*xb,sum1,sum2,sum3,sum4,sum5,sum6,sum7;
  PetscScalar    x1,x2,x3,x4,x5,x6,x7,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

  idx = a->j;
  v   = a->a;
  if (usecprow){
    mbs    = a->compressedrow.nrows;
    ii     = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0; sum5 = 0.0; sum6 = 0.0; sum7 = 0.0;
    for (j=0; j<n; j++) {
      xb = x + 7*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4]; x6 = xb[5]; x7 = xb[6];
      sum1 += v[0]*x1 + v[7]*x2  + v[14]*x3  + v[21]*x4 + v[28]*x5 + v[35]*x6 + v[42]*x7;
      sum2 += v[1]*x1 + v[8]*x2  + v[15]*x3  + v[22]*x4 + v[29]*x5 + v[36]*x6 + v[43]*x7;
      sum3 += v[2]*x1 + v[9]*x2  + v[16]*x3  + v[23]*x4 + v[30]*x5 + v[37]*x6 + v[44]*x7;
      sum4 += v[3]*x1 + v[10]*x2 + v[17]*x3  + v[24]*x4 + v[31]*x5 + v[38]*x6 + v[45]*x7;
      sum5 += v[4]*x1 + v[11]*x2 + v[18]*x3  + v[25]*x4 + v[32]*x5 + v[39]*x6 + v[46]*x7;
      sum6 += v[5]*x1 + v[12]*x2 + v[19]*x3  + v[26]*x4 + v[33]*x5 + v[40]*x6 + v[47]*x7;
      sum7 += v[6]*x1 + v[13]*x2 + v[20]*x3  + v[27]*x4 + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v += 49;
    }
    if (usecprow) z = zarray + 7*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5; z[5] = sum6; z[6] = sum7;
    if (!usecprow) z += 7;
  }

  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(98*a->nz - 7*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    This will not work with MatScalar == float because it calls the BLAS
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_N"
PetscErrorCode MatMult_SeqBAIJ_N(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*z = 0,*xb,*work,*workt,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,bs=A->bs,j,n,bs2=a->bs2;
  PetscInt       ncols,k,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

  idx = a->j;
  v   = a->a;
  if (usecprow){
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
    z   = zarray;
  }

  if (!a->mult_work) {
    k    = PetscMax(A->m,A->n);
    ierr = PetscMalloc((k+1)*sizeof(PetscScalar),&a->mult_work);CHKERRQ(ierr);
  }
  work = a->mult_work;
  for (i=0; i<mbs; i++) {
    n     = ii[1] - ii[0]; ii++;
    ncols = n*bs;
    workt = work;
    for (j=0; j<n; j++) {
      xb = x + bs*(*idx++);
      for (k=0; k<bs; k++) workt[k] = xb[k];
      workt += bs;
    }
    if (usecprow) z = zarray + bs*ridx[i];
    Kernel_w_gets_Ar_times_v(bs,ncols,work,v,z);
    /* BLASgemv_("N",&bs,&ncols,&_DOne,v,&bs,work,&_One,&_DZero,z,&_One); */
    v += n*bs2;
    if (!usecprow) z += bs;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*a->nz*bs2 - bs*mbs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_1"
PetscErrorCode MatMultAdd_SeqBAIJ_1(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*y = 0,*z = 0,sum,*yarray,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);
  } else {
    zarray = yarray;
  }

  idx = a->j;
  v   = a->a;
  if (usecprow){
    if (zz != yy){
      ierr = PetscMemcpy(zarray,yarray,mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    ii  = a->i;
    y   = yarray; 
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n    = ii[1] - ii[0]; ii++;
    if (usecprow){
      z = zarray + ridx[i];
      y = yarray + ridx[i];
    } 
    sum = y[0];
    while (n--) sum += *v++ * x[*idx++];
    z[0] = sum;
    if (!usecprow){
      z++; y++;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(2*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_2"
PetscErrorCode MatMultAdd_SeqBAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*y = 0,*z = 0,*xb,sum1,sum2;
  PetscScalar    x1,x2,*yarray,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);
  } else {
    zarray = yarray;
  }

  idx = a->j;
  v   = a->a;
  if (usecprow){
    if (zz != yy){
      ierr = PetscMemcpy(zarray,yarray,2*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    if (zz != yy){
      ierr = PetscMemcpy(zarray,yarray,a->mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  } else {
    ii  = a->i;
    y   = yarray; 
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    if (usecprow){
      z = zarray + 2*ridx[i];
      y = yarray + 2*ridx[i];
    }
    sum1 = y[0]; sum2 = y[1];
    for (j=0; j<n; j++) {
      xb = x + 2*(*idx++); x1 = xb[0]; x2 = xb[1];
      sum1 += v[0]*x1 + v[2]*x2;
      sum2 += v[1]*x1 + v[3]*x2;
      v += 4;
    }
    z[0] = sum1; z[1] = sum2;
    if (!usecprow){
      z += 2; y += 2;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(4*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_3"
PetscErrorCode MatMultAdd_SeqBAIJ_3(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*y = 0,*z = 0,*xb,sum1,sum2,sum3,x1,x2,x3,*yarray,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);
  } else {
    zarray = yarray;
  }

  idx = a->j;
  v   = a->a;
  if (usecprow){
    if (zz != yy){
      ierr = PetscMemcpy(zarray,yarray,3*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    ii  = a->i;
    y   = yarray; 
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    if (usecprow){
      z = zarray + 3*ridx[i];
      y = yarray + 3*ridx[i];
    }
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2];
    for (j=0; j<n; j++) {
      xb = x + 3*(*idx++); x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
      sum1 += v[0]*x1 + v[3]*x2 + v[6]*x3;
      sum2 += v[1]*x1 + v[4]*x2 + v[7]*x3;
      sum3 += v[2]*x1 + v[5]*x2 + v[8]*x3;
      v += 9;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3;
    if (!usecprow){
      z += 3; y += 3;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(18*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_4"
PetscErrorCode MatMultAdd_SeqBAIJ_4(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*y = 0,*z = 0,*xb,sum1,sum2,sum3,sum4,x1,x2,x3,x4,*yarray,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);
  } else {
    zarray = yarray;
  }

  idx   = a->j;
  v     = a->a;
  if (usecprow){
    if (zz != yy){
      ierr = PetscMemcpy(zarray,yarray,4*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    ii  = a->i;
    y   = yarray; 
    z   = zarray;
  }  

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    if (usecprow){
      z = zarray + 4*ridx[i];
      y = yarray + 4*ridx[i];
    }
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2]; sum4 = y[3];
    for (j=0; j<n; j++) {
      xb = x + 4*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      sum1 += v[0]*x1 + v[4]*x2 + v[8]*x3   + v[12]*x4;
      sum2 += v[1]*x1 + v[5]*x2 + v[9]*x3   + v[13]*x4;
      sum3 += v[2]*x1 + v[6]*x2 + v[10]*x3  + v[14]*x4;
      sum4 += v[3]*x1 + v[7]*x2 + v[11]*x3  + v[15]*x4;
      v += 16;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4;
    if (!usecprow){
      z += 4; y += 4;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(32*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_5"
PetscErrorCode MatMultAdd_SeqBAIJ_5(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*y = 0,*z = 0,*xb,sum1,sum2,sum3,sum4,sum5,x1,x2,x3,x4,x5;
  PetscScalar    *yarray,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);
  } else {
    zarray = yarray;
  }

  idx = a->j;
  v   = a->a;
  if (usecprow){
    if (zz != yy){
      ierr = PetscMemcpy(zarray,yarray,5*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    ii  = a->i;
    y   = yarray; 
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    if (usecprow){
      z = zarray + 5*ridx[i];
      y = yarray + 5*ridx[i];
    }
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2]; sum4 = y[3]; sum5 = y[4];
    for (j=0; j<n; j++) {
      xb = x + 5*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4];
      sum1 += v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      sum2 += v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      sum3 += v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      sum4 += v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      sum5 += v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v += 25;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5;
    if (!usecprow){
      z += 5; y += 5;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(50*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_6"
PetscErrorCode MatMultAdd_SeqBAIJ_6(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*y = 0,*z = 0,*xb,sum1,sum2,sum3,sum4,sum5,sum6;
  PetscScalar    x1,x2,x3,x4,x5,x6,*yarray,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);
  } else {
    zarray = yarray;
  }

  idx = a->j;
  v   = a->a;
  if (usecprow){
    if (zz != yy){
      ierr = PetscMemcpy(zarray,yarray,6*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    ii  = a->i;
    y   = yarray; 
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    if (usecprow){
      z = zarray + 6*ridx[i];
      y = yarray + 6*ridx[i];
    }
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2]; sum4 = y[3]; sum5 = y[4]; sum6 = y[5];
    for (j=0; j<n; j++) {
      xb = x + 6*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4]; x6 = xb[5];
      sum1 += v[0]*x1 + v[6]*x2  + v[12]*x3  + v[18]*x4 + v[24]*x5 + v[30]*x6;
      sum2 += v[1]*x1 + v[7]*x2  + v[13]*x3  + v[19]*x4 + v[25]*x5 + v[31]*x6;
      sum3 += v[2]*x1 + v[8]*x2  + v[14]*x3  + v[20]*x4 + v[26]*x5 + v[32]*x6;
      sum4 += v[3]*x1 + v[9]*x2  + v[15]*x3  + v[21]*x4 + v[27]*x5 + v[33]*x6;
      sum5 += v[4]*x1 + v[10]*x2 + v[16]*x3  + v[22]*x4 + v[28]*x5 + v[34]*x6;
      sum6 += v[5]*x1 + v[11]*x2 + v[17]*x3  + v[23]*x4 + v[29]*x5 + v[35]*x6;
      v += 36;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5; z[5] = sum6;
    if (!usecprow){
      z += 6; y += 6;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(72*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_7"
PetscErrorCode MatMultAdd_SeqBAIJ_7(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*y = 0,*z = 0,*xb,sum1,sum2,sum3,sum4,sum5,sum6,sum7;
  PetscScalar    x1,x2,x3,x4,x5,x6,x7,*yarray,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);
  } else {
    zarray = yarray;
  }

  idx = a->j;
  v   = a->a;
  if (usecprow){
    if (zz != yy){
      ierr = PetscMemcpy(zarray,yarray,7*mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    ii  = a->i;
    y   = yarray; 
    z   = zarray;
  }

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    if (usecprow){
      z = zarray + 7*ridx[i];
      y = yarray + 7*ridx[i];
    }
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2]; sum4 = y[3]; sum5 = y[4]; sum6 = y[5]; sum7 = y[6];
    for (j=0; j<n; j++) {
      xb = x + 7*(*idx++);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4]; x6 = xb[5]; x7 = xb[6];
      sum1 += v[0]*x1 + v[7]*x2  + v[14]*x3  + v[21]*x4 + v[28]*x5 + v[35]*x6 + v[42]*x7;
      sum2 += v[1]*x1 + v[8]*x2  + v[15]*x3  + v[22]*x4 + v[29]*x5 + v[36]*x6 + v[43]*x7;
      sum3 += v[2]*x1 + v[9]*x2  + v[16]*x3  + v[23]*x4 + v[30]*x5 + v[37]*x6 + v[44]*x7;
      sum4 += v[3]*x1 + v[10]*x2 + v[17]*x3  + v[24]*x4 + v[31]*x5 + v[38]*x6 + v[45]*x7;
      sum5 += v[4]*x1 + v[11]*x2 + v[18]*x3  + v[25]*x4 + v[32]*x5 + v[39]*x6 + v[46]*x7;
      sum6 += v[5]*x1 + v[12]*x2 + v[19]*x3  + v[26]*x4 + v[33]*x5 + v[40]*x6 + v[47]*x7;
      sum7 += v[6]*x1 + v[13]*x2 + v[20]*x3  + v[27]*x4 + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v += 49;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5; z[5] = sum6; z[6] = sum7;
    if (!usecprow){
      z += 7; y += 7;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&yarray);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(98*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_N"
PetscErrorCode MatMultAdd_SeqBAIJ_N(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*z = 0,*xb,*work,*workt,*y,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs,i,*idx,*ii,bs=A->bs,j,n,bs2=a->bs2;
  PetscInt       ncols,k,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);
  if (zz != yy) { 
    ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
    ierr = PetscMemcpy(zarray,y,yy->n*sizeof(PetscScalar));CHKERRQ(ierr); 
    ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  }

  idx = a->j;
  v   = a->a;
  if (usecprow){
    mbs    = a->compressedrow.nrows;
    ii     = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    mbs = a->mbs;
    ii  = a->i;
    z   = zarray;
  }

  if (!a->mult_work) {
    k    = PetscMax(A->m,A->n);
    ierr = PetscMalloc((k+1)*sizeof(PetscScalar),&a->mult_work);CHKERRQ(ierr);
  }
  work = a->mult_work;
  for (i=0; i<mbs; i++) {
    n     = ii[1] - ii[0]; ii++;
    ncols = n*bs;
    workt = work;
    for (j=0; j<n; j++) {
      xb = x + bs*(*idx++);
      for (k=0; k<bs; k++) workt[k] = xb[k];
      workt += bs;
    }
    if (usecprow) z = zarray + bs*ridx[i];
    Kernel_w_gets_w_plus_Ar_times_v(bs,ncols,work,v,z);
    /* BLASgemv_("N",&bs,&ncols,&_DOne,v,&bs,work,&_One,&_DOne,z,&_One); */
    v += n*bs2;
    if (!usecprow){
      z += bs;
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*a->nz*bs2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqBAIJ"
PetscErrorCode MatMultTranspose_SeqBAIJ(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    zero = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(zz,zero);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd_SeqBAIJ(A,xx,zz,zz);CHKERRQ(ierr);

  ierr = PetscLogFlops(2*a->nz*a->bs2 - A->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqBAIJ"
PetscErrorCode MatMultTransposeAdd_SeqBAIJ(Mat A,Vec xx,Vec yy,Vec zz)

{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *zb,*x,*z,*xb = 0,x1,x2,x3,x4,x5;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs,i,*idx,*ii,rval,bs=A->bs,j,n,bs2=a->bs2,*ib,*ridx=PETSC_NULL;
  Mat_CompressedRow cprow = a->compressedrow;
  PetscTruth        usecprow=cprow.use;

  PetscFunctionBegin;
  if (yy != zz) { ierr = VecCopy(yy,zz);CHKERRQ(ierr); }
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr); 

  idx = a->j;
  v   = a->a;
  if (usecprow){
    mbs  = cprow.nrows;
    ii   = cprow.i;
    ridx = cprow.rindex;
  } else {
    mbs=a->mbs;
    ii = a->i;
    xb = x;
  }

  switch (bs) {
  case 1:
    for (i=0; i<mbs; i++) {
      if (usecprow) xb = x + ridx[i];
      x1 = xb[0];
      ib = idx + ii[0];
      n  = ii[1] - ii[0]; ii++; 
      for (j=0; j<n; j++) {
        rval    = ib[j];
        z[rval] += *v * x1;
        v++;
      }
      if (!usecprow) xb++;
    }
    break;
  case 2:
    for (i=0; i<mbs; i++) {
      if (usecprow) xb = x + 2*ridx[i];
      x1 = xb[0]; x2 = xb[1];
      ib = idx + ii[0];
      n  = ii[1] - ii[0]; ii++;
      for (j=0; j<n; j++) {
        rval      = ib[j]*2;
        z[rval++] += v[0]*x1 + v[1]*x2;
        z[rval++] += v[2]*x1 + v[3]*x2;
        v  += 4;
      }
      if (!usecprow) xb += 2;
    }
    break;
  case 3:
    for (i=0; i<mbs; i++) {
      if (usecprow) xb = x + 3*ridx[i];
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
      ib = idx + ii[0];
      n  = ii[1] - ii[0]; ii++; 
      for (j=0; j<n; j++) {
        rval      = ib[j]*3;
        z[rval++] += v[0]*x1 + v[1]*x2 + v[2]*x3;
        z[rval++] += v[3]*x1 + v[4]*x2 + v[5]*x3;
        z[rval++] += v[6]*x1 + v[7]*x2 + v[8]*x3;
        v  += 9;
      }
      if (!usecprow) xb += 3;
    }
    break;
  case 4:
    for (i=0; i<mbs; i++) {
      if (usecprow) xb = x + 4*ridx[i];
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      ib = idx + ii[0];
      n  = ii[1] - ii[0]; ii++; 
      for (j=0; j<n; j++) {
        rval      = ib[j]*4;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4;
        z[rval++] +=  v[4]*x1 +  v[5]*x2 +  v[6]*x3 +  v[7]*x4;
        z[rval++] +=  v[8]*x1 +  v[9]*x2 + v[10]*x3 + v[11]*x4;
        z[rval++] += v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4;
        v  += 16;
      }
      if (!usecprow) xb += 4;
    }
    break;
  case 5:
    for (i=0; i<mbs; i++) {
      if (usecprow) xb = x + 5*ridx[i];
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; 
      x4 = xb[3]; x5 = xb[4];
      ib = idx + ii[0];
      n  = ii[1] - ii[0]; ii++; 
      for (j=0; j<n; j++) {
        rval      = ib[j]*5;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5;
        z[rval++] +=  v[5]*x1 +  v[6]*x2 +  v[7]*x3 +  v[8]*x4 +  v[9]*x5;
        z[rval++] += v[10]*x1 + v[11]*x2 + v[12]*x3 + v[13]*x4 + v[14]*x5;
        z[rval++] += v[15]*x1 + v[16]*x2 + v[17]*x3 + v[18]*x4 + v[19]*x5;
        z[rval++] += v[20]*x1 + v[21]*x2 + v[22]*x3 + v[23]*x4 + v[24]*x5;
        v  += 25;
      }
      if (!usecprow) xb += 5;
    }
    break;
  default: {      /* block sizes larger then 5 by 5 are handled by BLAS */
      PetscInt     ncols,k; 
      PetscScalar  *work,*workt,*xtmp;

      if (!a->mult_work) {
        k = PetscMax(A->m,A->n);
        ierr = PetscMalloc((k+1)*sizeof(PetscScalar),&a->mult_work);CHKERRQ(ierr);
      }
      work = a->mult_work;
      xtmp = x;
      for (i=0; i<mbs; i++) {
        n     = ii[1] - ii[0]; ii++;
        ncols = n*bs;
        ierr  = PetscMemzero(work,ncols*sizeof(PetscScalar));CHKERRQ(ierr);
        if (usecprow) {
          xtmp = x + bs*ridx[i];
        } 
        Kernel_w_gets_w_plus_trans_Ar_times_v(bs,ncols,xtmp,v,work);
        /* BLASgemv_("T",&bs,&ncols,&_DOne,v,&bs,xtmp,&_One,&_DOne,work,&_One); */
        v += n*bs2;
        if (!usecprow) xtmp += bs;
        workt = work;
        for (j=0; j<n; j++) {
          zb = z + bs*(*idx++);
          for (k=0; k<bs; k++) zb[k] += workt[k] ;
          workt += bs;
        }
      }
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*a->nz*a->bs2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_SeqBAIJ"
PetscErrorCode MatScale_SeqBAIJ(Mat inA,PetscScalar alpha)
{
  Mat_SeqBAIJ  *a = (Mat_SeqBAIJ*)inA->data;
  PetscInt     totalnz = a->bs2*a->nz;
  PetscScalar  oalpha = alpha;
  PetscErrorCode ierr;
#if defined(PETSC_USE_MAT_SINGLE)
  PetscInt     i;
#else
  PetscBLASInt tnz = (PetscBLASInt) totalnz,one = 1;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_MAT_SINGLE)
  for (i=0; i<totalnz; i++) a->a[i] *= oalpha;
#else
  BLASscal_(&tnz,&oalpha,a->a,&one);
#endif
  ierr = PetscLogFlops(totalnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_SeqBAIJ"
PetscErrorCode MatNorm_SeqBAIJ(Mat A,NormType type,PetscReal *norm)
{
  PetscErrorCode ierr;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  MatScalar      *v = a->a;
  PetscReal      sum = 0.0;
  PetscInt       i,j,k,bs=A->bs,nz=a->nz,bs2=a->bs2,k1;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for (i=0; i< bs2*nz; i++) {
#if defined(PETSC_USE_COMPLEX)
      sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
      sum += (*v)*(*v); v++;
#endif
    }
    *norm = sqrt(sum);
  } else if (type == NORM_1) { /* maximum column sum */
    PetscReal *tmp;
    PetscInt  *bcol = a->j;
    ierr = PetscMalloc((A->n+1)*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
    ierr = PetscMemzero(tmp,A->n*sizeof(PetscReal));CHKERRQ(ierr);
    for (i=0; i<nz; i++){
      for (j=0; j<bs; j++){
        k1 = bs*(*bcol) + j; /* column index */
        for (k=0; k<bs; k++){
          tmp[k1] += PetscAbsScalar(*v); v++;
        }
      }
      bcol++;
    }
    *norm = 0.0;
    for (j=0; j<A->n; j++) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) { /* maximum row sum */
    *norm = 0.0;
    for (k=0; k<bs; k++) {
      for (j=0; j<a->mbs; j++) {
        v = a->a + bs2*a->i[j] + k;
        sum = 0.0;
        for (i=0; i<a->i[j+1]-a->i[j]; i++) {
          for (k1=0; k1<bs; k1++){ 
            sum += PetscAbsScalar(*v); 
            v   += bs;
          }
        }
        if (sum > *norm) *norm = sum;
      }
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"No support for this norm yet");
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatEqual_SeqBAIJ"
PetscErrorCode MatEqual_SeqBAIJ(Mat A,Mat B,PetscTruth* flg)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ *)A->data,*b = (Mat_SeqBAIJ *)B->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* If the  matrix/block dimensions are not equal, or no of nonzeros or shift */
  if ((A->m != B->m) || (A->n != B->n) || (A->bs != B->bs)|| (a->nz != b->nz)) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0); 
  }
  
  /* if the a->i are the same */
  ierr = PetscMemcmp(a->i,b->i,(a->mbs+1)*sizeof(PetscInt),flg);CHKERRQ(ierr);
  if (!*flg) {
    PetscFunctionReturn(0);
  }
  
  /* if a->j are the same */
  ierr = PetscMemcmp(a->j,b->j,(a->nz)*sizeof(PetscInt),flg);CHKERRQ(ierr);
  if (!*flg) {
    PetscFunctionReturn(0);
  }  
  /* if a->a are the same */
  ierr = PetscMemcmp(a->a,b->a,(a->nz)*(A->bs)*(B->bs)*sizeof(PetscScalar),flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_SeqBAIJ"
PetscErrorCode MatGetDiagonal_SeqBAIJ(Mat A,Vec v)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,k,n,row,bs,*ai,*aj,ambs,bs2;
  PetscScalar    *x,zero = 0.0;
  MatScalar      *aa,*aa_j;

  PetscFunctionBegin;
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");  
  bs   = A->bs;
  aa   = a->a;
  ai   = a->i;
  aj   = a->j;
  ambs = a->mbs;
  bs2  = a->bs2;

  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<ambs; i++) {
    for (j=ai[i]; j<ai[i+1]; j++) {
      if (aj[j] == i) {
        row  = i*bs;
        aa_j = aa+j*bs2;
        for (k=0; k<bs2; k+=(bs+1),row++) x[row] = aa_j[k];
        break;
      }
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_SeqBAIJ"
PetscErrorCode MatDiagonalScale_SeqBAIJ(Mat A,Vec ll,Vec rr)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *l,*r,x,*li,*ri;
  MatScalar      *aa,*v;
  PetscErrorCode ierr;
  PetscInt       i,j,k,lm,rn,M,m,n,*ai,*aj,mbs,tmp,bs,bs2;

  PetscFunctionBegin;
  ai  = a->i;
  aj  = a->j;
  aa  = a->a;
  m   = A->m;
  n   = A->n;
  bs  = A->bs;
  mbs = a->mbs;
  bs2 = a->bs2;
  if (ll) {
    ierr = VecGetArray(ll,&l);CHKERRQ(ierr);
    ierr = VecGetLocalSize(ll,&lm);CHKERRQ(ierr);
    if (lm != m) SETERRQ(PETSC_ERR_ARG_SIZ,"Left scaling vector wrong length");
    for (i=0; i<mbs; i++) { /* for each block row */
      M  = ai[i+1] - ai[i];
      li = l + i*bs;
      v  = aa + bs2*ai[i];
      for (j=0; j<M; j++) { /* for each block */
        for (k=0; k<bs2; k++) {
          (*v++) *= li[k%bs];
        } 
      }  
    }
    ierr = VecRestoreArray(ll,&l);CHKERRQ(ierr);
    ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
  }
  
  if (rr) {
    ierr = VecGetArray(rr,&r);CHKERRQ(ierr);
    ierr = VecGetLocalSize(rr,&rn);CHKERRQ(ierr);
    if (rn != n) SETERRQ(PETSC_ERR_ARG_SIZ,"Right scaling vector wrong length");
    for (i=0; i<mbs; i++) { /* for each block row */
      M  = ai[i+1] - ai[i];
      v  = aa + bs2*ai[i];
      for (j=0; j<M; j++) { /* for each block */
        ri = r + bs*aj[ai[i]+j];
        for (k=0; k<bs; k++) {
          x = ri[k];
          for (tmp=0; tmp<bs; tmp++) (*v++) *= x;
        } 
      }  
    }
    ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr);
    ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_SeqBAIJ"
PetscErrorCode MatGetInfo_SeqBAIJ(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;

  PetscFunctionBegin;
  info->rows_global    = (double)A->m;
  info->columns_global = (double)A->n;
  info->rows_local     = (double)A->m;
  info->columns_local  = (double)A->n;
  info->block_size     = a->bs2;
  info->nz_allocated   = a->maxnz;
  info->nz_used        = a->bs2*a->nz;
  info->nz_unneeded    = (double)(info->nz_allocated - info->nz_used);
  info->assemblies   = A->num_ass;
  info->mallocs      = a->reallocs;
  info->memory       = A->mem;
  if (A->factor) {
    info->fill_ratio_given  = A->info.fill_ratio_given;
    info->fill_ratio_needed = A->info.fill_ratio_needed;
    info->factor_mallocs    = A->info.factor_mallocs;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_SeqBAIJ"
PetscErrorCode MatZeroEntries_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(a->a,a->bs2*a->i[a->mbs]*sizeof(MatScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
