#define PETSCMAT_DLL

#include "../src/mat/impls/baij/seq/baij.h"
#include "../src/mat/blockinvert.h"
#include "petscbt.h"
#include "petscblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_SeqBAIJ"
PetscErrorCode MatIncreaseOverlap_SeqBAIJ(Mat A,PetscInt is_max,IS is[],PetscInt ov)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       row,i,j,k,l,m,n,*nidx,isz,val,ival;
  const PetscInt *idx;
  PetscInt       start,end,*ai,*aj,bs,*nidx2;
  PetscBT        table;

  PetscFunctionBegin;
  m     = a->mbs;
  ai    = a->i;
  aj    = a->j;
  bs    = A->rmap->bs;

  if (ov < 0)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");

  ierr = PetscBTCreate(m,table);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&nidx);CHKERRQ(ierr); 
  ierr = PetscMalloc((A->rmap->N+1)*sizeof(PetscInt),&nidx2);CHKERRQ(ierr);

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
PetscErrorCode MatGetSubMatrix_SeqBAIJ_Private(Mat A,IS isrow,IS iscol,MatReuse scall,Mat *B)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*c;
  PetscErrorCode ierr;
  PetscInt       *smap,i,k,kstart,kend,oldcols = a->nbs,*lens;
  PetscInt       row,mat_i,*mat_j,tcol,*mat_ilen;
  const PetscInt *irow,*icol;
  PetscInt       nrows,ncols,*ssmap,bs=A->rmap->bs,bs2=a->bs2;
  PetscInt       *aj = a->j,*ai = a->i;
  MatScalar      *mat_a;
  Mat            C;
  PetscTruth     flag,sorted;

  PetscFunctionBegin;
  ierr = ISSorted(iscol,&sorted);CHKERRQ(ierr);
  if (!sorted) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"IS is not sorted");

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

    if (c->mbs!=nrows || c->nbs!=ncols || (*B)->rmap->bs!=bs) SETERRQ(PETSC_ERR_ARG_SIZ,"Submatrix wrong size");
    ierr = PetscMemcmp(c->ilen,lens,c->mbs *sizeof(PetscInt),&flag);CHKERRQ(ierr);
    if (!flag) {
      SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong no of nonzeros");
    }
    ierr = PetscMemzero(c->ilen,c->mbs*sizeof(PetscInt));CHKERRQ(ierr);
    C = *B;
  } else {  
    ierr = MatCreate(((PetscObject)A)->comm,&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,nrows*bs,ncols*bs,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetType(C,((PetscObject)A)->type_name);CHKERRQ(ierr);
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
PetscErrorCode MatGetSubMatrix_SeqBAIJ(Mat A,IS isrow,IS iscol,MatReuse scall,Mat *B)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  IS             is1,is2;
  PetscErrorCode ierr;
  PetscInt       *vary,*iary,nrows,ncols,i,bs=A->rmap->bs,count;
  const PetscInt *irow,*icol;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);
  
  /* Verify if the indices corespond to each element in a block 
   and form the IS with compressed IS */
  ierr = PetscMalloc2(a->mbs,PetscInt,&vary,a->mbs,PetscInt,&iary);CHKERRQ(ierr);
  ierr = PetscMemzero(vary,a->mbs*sizeof(PetscInt));CHKERRQ(ierr);
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
  ierr = PetscFree2(vary,iary);CHKERRQ(ierr);

  ierr = MatGetSubMatrix_SeqBAIJ_Private(A,is1,is2,scall,B);CHKERRQ(ierr);
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
    ierr = MatGetSubMatrix_SeqBAIJ(A,irow[i],icol[i],scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------*/
/* Should check that shapes of vectors and matrices match */
/* -------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_1"
PetscErrorCode MatMult_SeqBAIJ_1(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *z,sum;
  const PetscScalar *x;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  PetscInt          mbs,i,n,nonzerorow=0;
  const PetscInt    *idx,*ii,*ridx=PETSC_NULL;
  PetscTruth        usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  if (usecprow){
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    ierr = PetscMemzero(z,mbs*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    mbs = a->mbs;
    ii  = a->i;
  }

  for (i=0; i<mbs; i++) {
    n    = ii[1] - ii[0]; 
    v    = a->a + ii[0];
    idx  = a->j + ii[0]; 
    ii++;
    sum  = 0.0;
    PetscSparseDensePlusDot(sum,x,v,idx,n);
    if (usecprow){
      z[ridx[i]] = sum;
    } else {
      nonzerorow += (n>0);
      z[i] = sum;
    }
  }
  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_2"
PetscErrorCode MatMult_SeqBAIJ_2(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *z = 0,sum1,sum2,*zarray;
  const PetscScalar *x,*xb;
  PetscScalar       x1,x2;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  PetscInt          mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscTruth        usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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
  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(8.0*a->nz - 2.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_3"
PetscErrorCode MatMult_SeqBAIJ_3(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *z = 0,sum1,sum2,sum3,x1,x2,x3,*zarray;
  const PetscScalar *x,*xb;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  PetscInt          mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscTruth        usecprow=a->compressedrow.use;
  

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*v,*z,*xb)
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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
  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(18.0*a->nz - 3.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_4"
PetscErrorCode MatMult_SeqBAIJ_4(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *z = 0,sum1,sum2,sum3,sum4,x1,x2,x3,x4,*zarray;
  const PetscScalar *x,*xb;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  PetscInt          mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscTruth        usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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
  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(32.0*a->nz - 4.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_5"
PetscErrorCode MatMult_SeqBAIJ_5(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       sum1,sum2,sum3,sum4,sum5,x1,x2,x3,x4,x5,*z = 0,*zarray;
  const PetscScalar *xb,*x;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  const PetscInt    *idx,*ii,*ridx=PETSC_NULL;
  PetscInt          mbs,i,j,n,nonzerorow=0;
  PetscTruth        usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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
  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(50.0*a->nz - 5.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_6"
PetscErrorCode MatMult_SeqBAIJ_6(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *z = 0,sum1,sum2,sum3,sum4,sum5,sum6;
  const PetscScalar *x,*xb;
  PetscScalar       x1,x2,x3,x4,x5,x6,*zarray;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  PetscInt          mbs=a->mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscTruth        usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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

  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(72.0*a->nz - 6.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_7"
PetscErrorCode MatMult_SeqBAIJ_7(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *z = 0,sum1,sum2,sum3,sum4,sum5,sum6,sum7;
  const PetscScalar *x,*xb;
  PetscScalar       x1,x2,x3,x4,x5,x6,x7,*zarray;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  PetscInt          mbs,i,*idx,*ii,j,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscTruth        usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
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
    nonzerorow += (n>0);
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

  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(98.0*a->nz - 7.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* MatMult_SeqBAIJ_15 version 1: Columns in the block are accessed one at a time */
/* Default MatMult for block size 15 */

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_15_ver1"
PetscErrorCode MatMult_SeqBAIJ_15_ver1(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *z = 0,sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12,sum13,sum14,sum15;
  const PetscScalar *x,*xb;
  PetscScalar        x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,*zarray;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  const PetscInt    *ii,*ij=a->j,*idx;
  PetscInt          mbs,i,j,k,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscTruth        usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

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
    n  = ii[i+1] - ii[i];
    idx = ij + ii[i];
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0; sum5 = 0.0; sum6 = 0.0; sum7 = 0.0;
    sum8 = 0.0; sum9 = 0.0; sum10 = 0.0; sum11 = 0.0; sum12 = 0.0; sum13 = 0.0; sum14 = 0.0;sum15 = 0.0;

    nonzerorow += (n>0);
    for (j=0; j<n; j++) {
      xb = x + 15*(idx[j]);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4]; x6 = xb[5]; x7 = xb[6];
      x8 = xb[7]; x9 = xb[8]; x10 = xb[9]; x11 = xb[10]; x12 = xb[11]; x13 = xb[12]; x14 = xb[13];x15 = xb[14];

      for(k=0;k<15;k++){
	sum1  += v[0]*xb[k];
        sum2  += v[1]*xb[k];
	sum3  += v[2]*xb[k];
	sum4  += v[3]*xb[k];	
	sum5  += v[4]*xb[k];
        sum6  += v[5]*xb[k];
	sum7  += v[6]*xb[k];
	sum8  += v[7]*xb[k];	
        sum9  += v[8]*xb[k];
        sum10 += v[9]*xb[k];
	sum11 += v[10]*xb[k];
	sum12 += v[11]*xb[k];	
	sum13 += v[12]*xb[k];
        sum14 += v[13]*xb[k];
	sum15 += v[14]*xb[k];
	v += 15;
      }
    } 
    if (usecprow) z = zarray + 15*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5; z[5] = sum6; z[6] = sum7;
    z[7] = sum8; z[8] = sum9; z[9] = sum10; z[10] = sum11; z[11] = sum12; z[12] = sum13; z[13] = sum14;z[14] = sum15;

    if (!usecprow) z += 15;
  }

  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(450.0*a->nz - 15.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* MatMult_SeqBAIJ_15_ver2 : Columns in the block are accessed in sets of 4,4,4,3 */
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_15_ver2"
PetscErrorCode MatMult_SeqBAIJ_15_ver2(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *z = 0,sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12,sum13,sum14,sum15;
  const PetscScalar *x,*xb;
  PetscScalar        x1,x2,x3,x4,*zarray;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  const PetscInt    *ii,*ij=a->j,*idx;
  PetscInt          mbs,i,j,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscTruth        usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

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
    n  = ii[i+1] - ii[i];
    idx = ij + ii[i];
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0; sum5 = 0.0; sum6 = 0.0; sum7 = 0.0;
    sum8 = 0.0; sum9 = 0.0; sum10 = 0.0; sum11 = 0.0; sum12 = 0.0; sum13 = 0.0; sum14 = 0.0;sum15 = 0.0;

    nonzerorow += (n>0);
    for (j=0; j<n; j++) {
      xb = x + 15*(idx[j]);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];

      sum1  += v[0]*x1 + v[15]*x2 + v[30]*x3   + v[45]*x4;
      sum2  += v[1]*x1 + v[16]*x2 + v[31]*x3   + v[46]*x4;
      sum3  += v[2]*x1 + v[17]*x2 + v[32]*x3  + v[47]*x4;
      sum4  += v[3]*x1 + v[18]*x2 + v[33]*x3  + v[48]*x4;
      sum5  += v[4]*x1 + v[19]*x2 + v[34]*x3   + v[49]*x4;
      sum6  += v[5]*x1 + v[20]*x2 + v[35]*x3   + v[50]*x4;
      sum7  += v[6]*x1 + v[21]*x2 + v[36]*x3  + v[51]*x4;
      sum8  += v[7]*x1 + v[22]*x2 + v[37]*x3  + v[52]*x4;
      sum9  += v[8]*x1 + v[23]*x2 + v[38]*x3   + v[53]*x4;
      sum10 += v[9]*x1 + v[24]*x2 + v[39]*x3   + v[54]*x4;
      sum11 += v[10]*x1 + v[25]*x2 + v[40]*x3  + v[55]*x4;
      sum12 += v[11]*x1 + v[26]*x2 + v[41]*x3  + v[56]*x4;
      sum13 += v[12]*x1 + v[27]*x2 + v[42]*x3   + v[57]*x4;
      sum14 += v[13]*x1 + v[28]*x2 + v[43]*x3   + v[58]*x4;
      sum15 += v[14]*x1 + v[29]*x2 + v[44]*x3  + v[59]*x4;

      v += 60;

      x1 = xb[4]; x2 = xb[5]; x3 = xb[6]; x4 = xb[7];

      sum1  += v[0]*x1 + v[15]*x2 + v[30]*x3   + v[45]*x4;
      sum2  += v[1]*x1 + v[16]*x2 + v[31]*x3   + v[46]*x4;
      sum3  += v[2]*x1 + v[17]*x2 + v[32]*x3  + v[47]*x4;
      sum4  += v[3]*x1 + v[18]*x2 + v[33]*x3  + v[48]*x4;
      sum5  += v[4]*x1 + v[19]*x2 + v[34]*x3   + v[49]*x4;
      sum6  += v[5]*x1 + v[20]*x2 + v[35]*x3   + v[50]*x4;
      sum7  += v[6]*x1 + v[21]*x2 + v[36]*x3  + v[51]*x4;
      sum8  += v[7]*x1 + v[22]*x2 + v[37]*x3  + v[52]*x4;
      sum9  += v[8]*x1 + v[23]*x2 + v[38]*x3   + v[53]*x4;
      sum10 += v[9]*x1 + v[24]*x2 + v[39]*x3   + v[54]*x4;
      sum11 += v[10]*x1 + v[25]*x2 + v[40]*x3  + v[55]*x4;
      sum12 += v[11]*x1 + v[26]*x2 + v[41]*x3  + v[56]*x4;
      sum13 += v[12]*x1 + v[27]*x2 + v[42]*x3   + v[57]*x4;
      sum14 += v[13]*x1 + v[28]*x2 + v[43]*x3   + v[58]*x4;
      sum15 += v[14]*x1 + v[29]*x2 + v[44]*x3  + v[59]*x4;
      v += 60;

      x1 = xb[8]; x2 = xb[9]; x3 = xb[10]; x4 = xb[11];
      sum1  += v[0]*x1 + v[15]*x2 + v[30]*x3   + v[45]*x4;
      sum2  += v[1]*x1 + v[16]*x2 + v[31]*x3   + v[46]*x4;
      sum3  += v[2]*x1 + v[17]*x2 + v[32]*x3  + v[47]*x4;
      sum4  += v[3]*x1 + v[18]*x2 + v[33]*x3  + v[48]*x4;
      sum5  += v[4]*x1 + v[19]*x2 + v[34]*x3   + v[49]*x4;
      sum6  += v[5]*x1 + v[20]*x2 + v[35]*x3   + v[50]*x4;
      sum7  += v[6]*x1 + v[21]*x2 + v[36]*x3  + v[51]*x4;
      sum8  += v[7]*x1 + v[22]*x2 + v[37]*x3  + v[52]*x4;
      sum9  += v[8]*x1 + v[23]*x2 + v[38]*x3   + v[53]*x4;
      sum10 += v[9]*x1 + v[24]*x2 + v[39]*x3   + v[54]*x4;
      sum11 += v[10]*x1 + v[25]*x2 + v[40]*x3  + v[55]*x4;
      sum12 += v[11]*x1 + v[26]*x2 + v[41]*x3  + v[56]*x4;
      sum13 += v[12]*x1 + v[27]*x2 + v[42]*x3   + v[57]*x4;
      sum14 += v[13]*x1 + v[28]*x2 + v[43]*x3   + v[58]*x4;
      sum15 += v[14]*x1 + v[29]*x2 + v[44]*x3  + v[59]*x4;
      v  += 60;

      x1 = xb[12]; x2 = xb[13]; x3 = xb[14];
      sum1  += v[0]*x1 + v[15]*x2 + v[30]*x3;
      sum2  += v[1]*x1 + v[16]*x2 + v[31]*x3;
      sum3  += v[2]*x1 + v[17]*x2 + v[32]*x3;
      sum4  += v[3]*x1 + v[18]*x2 + v[33]*x3;
      sum5  += v[4]*x1 + v[19]*x2 + v[34]*x3;
      sum6  += v[5]*x1 + v[20]*x2 + v[35]*x3;
      sum7  += v[6]*x1 + v[21]*x2 + v[36]*x3;
      sum8  += v[7]*x1 + v[22]*x2 + v[37]*x3;
      sum9  += v[8]*x1 + v[23]*x2 + v[38]*x3;
      sum10 += v[9]*x1 + v[24]*x2 + v[39]*x3;
      sum11 += v[10]*x1 + v[25]*x2 + v[40]*x3;
      sum12 += v[11]*x1 + v[26]*x2 + v[41]*x3;
      sum13 += v[12]*x1 + v[27]*x2 + v[42]*x3;
      sum14 += v[13]*x1 + v[28]*x2 + v[43]*x3;
      sum15 += v[14]*x1 + v[29]*x2 + v[44]*x3;
      v += 45;
    } 
    if (usecprow) z = zarray + 15*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5; z[5] = sum6; z[6] = sum7;
    z[7] = sum8; z[8] = sum9; z[9] = sum10; z[10] = sum11; z[11] = sum12; z[12] = sum13; z[13] = sum14;z[14] = sum15;

    if (!usecprow) z += 15;
  }

  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(450.0*a->nz - 15.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* MatMult_SeqBAIJ_15_ver3 : Columns in the block are accessed in sets of 8,7 */
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_15_ver3"
PetscErrorCode MatMult_SeqBAIJ_15_ver3(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *z = 0,sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12,sum13,sum14,sum15;
  const PetscScalar *x,*xb;
  PetscScalar        x1,x2,x3,x4,x5,x6,x7,x8,*zarray;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  const PetscInt    *ii,*ij=a->j,*idx;
  PetscInt          mbs,i,j,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscTruth        usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

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
    n  = ii[i+1] - ii[i];
    idx = ij + ii[i];
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0; sum5 = 0.0; sum6 = 0.0; sum7 = 0.0;
    sum8 = 0.0; sum9 = 0.0; sum10 = 0.0; sum11 = 0.0; sum12 = 0.0; sum13 = 0.0; sum14 = 0.0;sum15 = 0.0;

    nonzerorow += (n>0);
    for (j=0; j<n; j++) {
      xb = x + 15*(idx[j]);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4]; x6 = xb[5]; x7 = xb[6];
      x8 = xb[7];

      sum1  += v[0]*x1 + v[15]*x2  + v[30]*x3  + v[45]*x4 + v[60]*x5 + v[75]*x6 + v[90]*x7 + v[105]*x8;
      sum2  += v[1]*x1 + v[16]*x2  + v[31]*x3  + v[46]*x4 + v[61]*x5 + v[76]*x6 + v[91]*x7 + v[106]*x8;
      sum3  += v[2]*x1 + v[17]*x2  + v[32]*x3  + v[47]*x4 + v[62]*x5 + v[77]*x6 + v[92]*x7 + v[107]*x8;
      sum4  += v[3]*x1 + v[18]*x2 + v[33]*x3  + v[48]*x4 + v[63]*x5 + v[78]*x6 + v[93]*x7 + v[108]*x8;
      sum5  += v[4]*x1 + v[19]*x2 + v[34]*x3  + v[49]*x4 + v[64]*x5 + v[79]*x6 + v[94]*x7 + v[109]*x8;
      sum6  += v[5]*x1 + v[20]*x2 + v[35]*x3  + v[50]*x4 + v[65]*x5 + v[80]*x6 + v[95]*x7 + v[110]*x8;
      sum7  += v[6]*x1 + v[21]*x2 + v[36]*x3  + v[51]*x4 + v[66]*x5 + v[81]*x6 + v[96]*x7 + v[111]*x8;
      sum8  += v[7]*x1 + v[22]*x2  + v[37]*x3  + v[52]*x4 + v[67]*x5 + v[82]*x6 + v[97]*x7 + v[112]*x8;
      sum9  += v[8]*x1 + v[23]*x2  + v[38]*x3  + v[53]*x4 + v[68]*x5 + v[83]*x6 + v[98]*x7 + v[113]*x8;
      sum10 += v[9]*x1 + v[24]*x2  + v[39]*x3  + v[54]*x4 + v[69]*x5 + v[84]*x6 + v[99]*x7 + v[114]*x8;
      sum11 += v[10]*x1 + v[25]*x2 + v[40]*x3  + v[55]*x4 + v[70]*x5 + v[85]*x6 + v[100]*x7 + v[115]*x8;
      sum12 += v[11]*x1 + v[26]*x2 + v[41]*x3  + v[56]*x4 + v[71]*x5 + v[86]*x6 + v[101]*x7 + v[116]*x8;
      sum13 += v[12]*x1 + v[27]*x2 + v[42]*x3  + v[57]*x4 + v[72]*x5 + v[87]*x6 + v[102]*x7 + v[117]*x8;
      sum14 += v[13]*x1 + v[28]*x2 + v[43]*x3  + v[58]*x4 + v[73]*x5 + v[88]*x6 + v[103]*x7 + v[118]*x8;
      sum15 += v[14]*x1 + v[29]*x2 + v[44]*x3  + v[59]*x4 + v[74]*x5 + v[89]*x6 + v[104]*x7 + v[119]*x8;
      v += 120;

      x1 = xb[8]; x2 = xb[9]; x3 = xb[10]; x4 = xb[11]; x5 = xb[12]; x6 = xb[13]; x7 = xb[14];

      sum1  += v[0]*x1 + v[15]*x2  + v[30]*x3  + v[45]*x4 + v[60]*x5 + v[75]*x6 + v[90]*x7;
      sum2  += v[1]*x1 + v[16]*x2  + v[31]*x3  + v[46]*x4 + v[61]*x5 + v[76]*x6 + v[91]*x7;
      sum3  += v[2]*x1 + v[17]*x2  + v[32]*x3  + v[47]*x4 + v[62]*x5 + v[77]*x6 + v[92]*x7;
      sum4  += v[3]*x1 + v[18]*x2 + v[33]*x3  + v[48]*x4 + v[63]*x5 + v[78]*x6 + v[93]*x7;
      sum5  += v[4]*x1 + v[19]*x2 + v[34]*x3  + v[49]*x4 + v[64]*x5 + v[79]*x6 + v[94]*x7;
      sum6  += v[5]*x1 + v[20]*x2 + v[35]*x3  + v[50]*x4 + v[65]*x5 + v[80]*x6 + v[95]*x7;
      sum7  += v[6]*x1 + v[21]*x2 + v[36]*x3  + v[51]*x4 + v[66]*x5 + v[81]*x6 + v[96]*x7;
      sum8  += v[7]*x1 + v[22]*x2  + v[37]*x3  + v[52]*x4 + v[67]*x5 + v[82]*x6 + v[97]*x7;
      sum9  += v[8]*x1 + v[23]*x2  + v[38]*x3  + v[53]*x4 + v[68]*x5 + v[83]*x6 + v[98]*x7;
      sum10 += v[9]*x1 + v[24]*x2  + v[39]*x3  + v[54]*x4 + v[69]*x5 + v[84]*x6 + v[99]*x7;
      sum11 += v[10]*x1 + v[25]*x2 + v[40]*x3  + v[55]*x4 + v[70]*x5 + v[85]*x6 + v[100]*x7;
      sum12 += v[11]*x1 + v[26]*x2 + v[41]*x3  + v[56]*x4 + v[71]*x5 + v[86]*x6 + v[101]*x7;
      sum13 += v[12]*x1 + v[27]*x2 + v[42]*x3  + v[57]*x4 + v[72]*x5 + v[87]*x6 + v[102]*x7;
      sum14 += v[13]*x1 + v[28]*x2 + v[43]*x3  + v[58]*x4 + v[73]*x5 + v[88]*x6 + v[103]*x7;
      sum15 += v[14]*x1 + v[29]*x2 + v[44]*x3  + v[59]*x4 + v[74]*x5 + v[89]*x6 + v[104]*x7;
      v += 105;
    } 
    if (usecprow) z = zarray + 15*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5; z[5] = sum6; z[6] = sum7;
    z[7] = sum8; z[8] = sum9; z[9] = sum10; z[10] = sum11; z[11] = sum12; z[12] = sum13; z[13] = sum14;z[14] = sum15;

    if (!usecprow) z += 15;
  }

  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(450.0*a->nz - 15.0*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* MatMult_SeqBAIJ_15_ver4 : All columns in the block are accessed at once */

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_15_ver4"
PetscErrorCode MatMult_SeqBAIJ_15_ver4(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *z = 0,sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12,sum13,sum14,sum15;
  const PetscScalar *x,*xb;
  PetscScalar        x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,*zarray;
  const MatScalar   *v;
  PetscErrorCode    ierr;
  const PetscInt    *ii,*ij=a->j,*idx;
  PetscInt          mbs,i,j,n,*ridx=PETSC_NULL,nonzerorow=0;
  PetscTruth        usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&zarray);CHKERRQ(ierr);

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
    n  = ii[i+1] - ii[i];
    idx = ij + ii[i];
    sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0; sum5 = 0.0; sum6 = 0.0; sum7 = 0.0;
    sum8 = 0.0; sum9 = 0.0; sum10 = 0.0; sum11 = 0.0; sum12 = 0.0; sum13 = 0.0; sum14 = 0.0;sum15 = 0.0;

    nonzerorow += (n>0);
    for (j=0; j<n; j++) {
      xb = x + 15*(idx[j]);
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4]; x6 = xb[5]; x7 = xb[6];
      x8 = xb[7]; x9 = xb[8]; x10 = xb[9]; x11 = xb[10]; x12 = xb[11]; x13 = xb[12]; x14 = xb[13];x15 = xb[14];

      sum1 +=  v[0]*x1  + v[15]*x2 + v[30]*x3 + v[45]*x4 + v[60]*x5 + v[75]*x6 + v[90]*x7  + v[105]*x8 + v[120]*x9 + v[135]*x10 + v[150]*x11 + v[165]*x12 + v[180]*x13 + v[195]*x14 + v[210]*x15;
      sum2 +=  v[1]*x1  + v[16]*x2 + v[31]*x3 + v[46]*x4 + v[61]*x5 + v[76]*x6 + v[91]*x7  + v[106]*x8 + v[121]*x9 + v[136]*x10 + v[151]*x11 + v[166]*x12 + v[181]*x13 + v[196]*x14 + v[211]*x15;
      sum3 +=  v[2]*x1  + v[17]*x2 + v[32]*x3 + v[47]*x4 + v[62]*x5 + v[77]*x6 + v[92]*x7  + v[107]*x8 + v[122]*x9 + v[137]*x10 + v[152]*x11 + v[167]*x12 + v[182]*x13 + v[197]*x14 + v[212]*x15;
      sum4 +=  v[3]*x1  + v[18]*x2 + v[33]*x3 + v[48]*x4 + v[63]*x5 + v[78]*x6 + v[93]*x7  + v[108]*x8 + v[123]*x9 + v[138]*x10 + v[153]*x11 + v[168]*x12 + v[183]*x13 + v[198]*x14 + v[213]*x15;
      sum5  += v[4]*x1  + v[19]*x2 + v[34]*x3 + v[49]*x4 + v[64]*x5 + v[79]*x6 + v[94]*x7  + v[109]*x8 + v[124]*x9 + v[139]*x10 + v[154]*x11 + v[169]*x12 + v[184]*x13 + v[199]*x14 + v[214]*x15;
      sum6  += v[5]*x1  + v[20]*x2 + v[35]*x3 + v[50]*x4 + v[65]*x5 + v[80]*x6 + v[95]*x7  + v[110]*x8 + v[125]*x9 + v[140]*x10 + v[155]*x11 + v[170]*x12 + v[185]*x13 + v[200]*x14 + v[215]*x15;
      sum7  += v[6]*x1  + v[21]*x2 + v[36]*x3 + v[51]*x4 + v[66]*x5 + v[81]*x6 + v[96]*x7  + v[111]*x8 + v[126]*x9 + v[141]*x10 + v[156]*x11 + v[171]*x12 + v[186]*x13 + v[201]*x14 + v[216]*x15;
      sum8  += v[7]*x1  + v[22]*x2 + v[37]*x3 + v[52]*x4 + v[67]*x5 + v[82]*x6 + v[97]*x7  + v[112]*x8 + v[127]*x9 + v[142]*x10 + v[157]*x11 + v[172]*x12 + v[187]*x13 + v[202]*x14 + v[217]*x15;
      sum9  += v[8]*x1  + v[23]*x2 + v[38]*x3 + v[53]*x4 + v[68]*x5 + v[83]*x6 + v[98]*x7  + v[113]*x8 + v[128]*x9 + v[143]*x10 + v[158]*x11 + v[173]*x12 + v[188]*x13 + v[203]*x14 + v[218]*x15;
      sum10 += v[9]*x1  + v[24]*x2 + v[39]*x3 + v[54]*x4 + v[69]*x5 + v[84]*x6 + v[99]*x7  + v[114]*x8 + v[129]*x9 + v[144]*x10 + v[159]*x11 + v[174]*x12 + v[189]*x13 + v[204]*x14 + v[219]*x15;
      sum11 += v[10]*x1 + v[25]*x2 + v[40]*x3 + v[55]*x4 + v[70]*x5 + v[85]*x6 + v[100]*x7 + v[115]*x8 + v[130]*x9 + v[145]*x10 + v[160]*x11 + v[175]*x12 + v[190]*x13 + v[205]*x14 + v[220]*x15;
      sum12 += v[11]*x1 + v[26]*x2 + v[41]*x3 + v[56]*x4 + v[71]*x5 + v[86]*x6 + v[101]*x7 + v[116]*x8 + v[131]*x9 + v[146]*x10 + v[161]*x11 + v[176]*x12 + v[191]*x13 + v[206]*x14 + v[221]*x15;
      sum13 += v[12]*x1 + v[27]*x2 + v[42]*x3 + v[57]*x4 + v[72]*x5 + v[87]*x6 + v[102]*x7 + v[117]*x8 + v[132]*x9 + v[147]*x10 + v[162]*x11 + v[177]*x12 + v[192]*x13 + v[207]*x14 + v[222]*x15;
      sum14 += v[13]*x1 + v[28]*x2 + v[43]*x3 + v[58]*x4 + v[73]*x5 + v[88]*x6 + v[103]*x7 + v[118]*x8 + v[133]*x9 + v[148]*x10 + v[163]*x11 + v[178]*x12 + v[193]*x13 + v[208]*x14 + v[223]*x15;
      sum15 += v[14]*x1 + v[29]*x2 + v[44]*x3 + v[59]*x4 + v[74]*x5 + v[89]*x6 + v[104]*x7 + v[119]*x8 + v[134]*x9 + v[149]*x10 + v[164]*x11 + v[179]*x12 + v[194]*x13 + v[209]*x14 + v[224]*x15;
      v += 225;
    } 
    if (usecprow) z = zarray + 15*ridx[i];
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5; z[5] = sum6; z[6] = sum7;
    z[7] = sum8; z[8] = sum9; z[9] = sum10; z[10] = sum11; z[11] = sum12; z[12] = sum13; z[13] = sum14;z[14] = sum15;

    if (!usecprow) z += 15;
  }

  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zarray);CHKERRQ(ierr);
  ierr = PetscLogFlops(450.0*a->nz - 15.0*nonzerorow);CHKERRQ(ierr);
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
  PetscInt       mbs=a->mbs,i,*idx,*ii,bs=A->rmap->bs,j,n,bs2=a->bs2;
  PetscInt       ncols,k,*ridx=PETSC_NULL,nonzerorow=0;
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
    k    = PetscMax(A->rmap->n,A->cmap->n);
    ierr = PetscMalloc((k+1)*sizeof(PetscScalar),&a->mult_work);CHKERRQ(ierr);
  }
  work = a->mult_work;
  for (i=0; i<mbs; i++) {
    n     = ii[1] - ii[0]; ii++;
    ncols = n*bs;
    workt = work;
    nonzerorow += (n>0);
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
  ierr = PetscLogFlops(2.0*a->nz*bs2 - bs*nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode VecCopy_Seq(Vec,Vec);
#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_1"
PetscErrorCode MatMultAdd_SeqBAIJ_1(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  const PetscScalar  *x;
  PetscScalar        *y,*z,sum;
  const MatScalar    *v;
  PetscErrorCode     ierr;
  PetscInt           mbs=a->mbs,i,n,*ridx=PETSC_NULL,nonzerorow=0;
  const PetscInt     *idx,*ii;
  PetscTruth         usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }

  idx = a->j;
  v   = a->a;
  if (usecprow){
    if (zz != yy){
      ierr = PetscMemcpy(z,y,mbs*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    mbs  = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
  } else {
    ii  = a->i;
  }

  for (i=0; i<mbs; i++) {
    n    = ii[1] - ii[0]; 
    ii++;
    if (!usecprow){
      nonzerorow += (n>0);
      sum = y[i];
    } else {
      sum = y[ridx[i]];
    } 
    PetscSparseDensePlusDot(sum,x,v,idx,n);
    v += n;
    idx += n;
    if (usecprow){
      z[ridx[i]] = sum;
    } else {
      z[i] = sum;
    }
  }
  ierr = VecRestoreArray(xx,(PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(4.0*a->nz);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(18.0*a->nz);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(32.0*a->nz);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(50.0*a->nz);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(72.0*a->nz);CHKERRQ(ierr);
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
  ierr = PetscLogFlops(98.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_N"
PetscErrorCode MatMultAdd_SeqBAIJ_N(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*z = 0,*xb,*work,*workt,*zarray;
  MatScalar      *v;
  PetscErrorCode ierr;
  PetscInt       mbs,i,*idx,*ii,bs=A->rmap->bs,j,n,bs2=a->bs2;
  PetscInt       ncols,k,*ridx=PETSC_NULL;
  PetscTruth     usecprow=a->compressedrow.use;

  PetscFunctionBegin;
  ierr = VecCopy_Seq(yy,zz);CHKERRQ(ierr);
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

  if (!a->mult_work) {
    k    = PetscMax(A->rmap->n,A->cmap->n);
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
  ierr = PetscLogFlops(2.0*a->nz*bs2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultHermitianTranspose_SeqBAIJ"
PetscErrorCode MatMultHermitianTranspose_SeqBAIJ(Mat A,Vec xx,Vec zz)
{
  PetscScalar    zero = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(zz,zero);CHKERRQ(ierr);
  ierr = MatMultHermitianTransposeAdd_SeqBAIJ(A,xx,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqBAIJ"
PetscErrorCode MatMultTranspose_SeqBAIJ(Mat A,Vec xx,Vec zz)
{
  PetscScalar    zero = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(zz,zero);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd_SeqBAIJ(A,xx,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultHermitianTransposeAdd_SeqBAIJ"
PetscErrorCode MatMultHermitianTransposeAdd_SeqBAIJ(Mat A,Vec xx,Vec yy,Vec zz)

{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *zb,*x,*z,*xb = 0,x1,x2,x3,x4,x5;
  MatScalar         *v;
  PetscErrorCode    ierr;
  PetscInt          mbs,i,*idx,*ii,rval,bs=A->rmap->bs,j,n,bs2=a->bs2,*ib,*ridx=PETSC_NULL;
  Mat_CompressedRow cprow = a->compressedrow;
  PetscTruth        usecprow=cprow.use;

  PetscFunctionBegin;
  if (yy != zz) { ierr = VecCopy_Seq(yy,zz);CHKERRQ(ierr); }
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
        z[rval] += PetscConj(*v) * x1;
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
        z[rval++] += PetscConj(v[0])*x1 + PetscConj(v[1])*x2;
        z[rval++] += PetscConj(v[2])*x1 + PetscConj(v[3])*x2;
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
        z[rval++] += PetscConj(v[0])*x1 + PetscConj(v[1])*x2 + PetscConj(v[2])*x3;
        z[rval++] += PetscConj(v[3])*x1 + PetscConj(v[4])*x2 + PetscConj(v[5])*x3;
        z[rval++] += PetscConj(v[6])*x1 + PetscConj(v[7])*x2 + PetscConj(v[8])*x3;
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
        z[rval++] +=  PetscConj(v[0])*x1 + PetscConj(v[1])*x2  + PetscConj(v[2])*x3  + PetscConj(v[3])*x4;
        z[rval++] +=  PetscConj(v[4])*x1 + PetscConj(v[5])*x2  + PetscConj(v[6])*x3  + PetscConj(v[7])*x4;
        z[rval++] +=  PetscConj(v[8])*x1 + PetscConj(v[9])*x2  + PetscConj(v[10])*x3 + PetscConj(v[11])*x4;
        z[rval++] += PetscConj(v[12])*x1 + PetscConj(v[13])*x2 + PetscConj(v[14])*x3 + PetscConj(v[15])*x4;
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
        z[rval++] +=  PetscConj(v[0])*x1 +  PetscConj(v[1])*x2 +  PetscConj(v[2])*x3 +  PetscConj(v[3])*x4 +  PetscConj(v[4])*x5;
        z[rval++] +=  PetscConj(v[5])*x1 +  PetscConj(v[6])*x2 +  PetscConj(v[7])*x3 +  PetscConj(v[8])*x4 +  PetscConj(v[9])*x5;
        z[rval++] += PetscConj(v[10])*x1 + PetscConj(v[11])*x2 + PetscConj(v[12])*x3 + PetscConj(v[13])*x4 + PetscConj(v[14])*x5;
        z[rval++] += PetscConj(v[15])*x1 + PetscConj(v[16])*x2 + PetscConj(v[17])*x3 + PetscConj(v[18])*x4 + PetscConj(v[19])*x5;
        z[rval++] += PetscConj(v[20])*x1 + PetscConj(v[21])*x2 + PetscConj(v[22])*x3 + PetscConj(v[23])*x4 + PetscConj(v[24])*x5;
        v  += 25;
      }
      if (!usecprow) xb += 5;
    }
    break;
  default: {      /* block sizes larger than 5 by 5 are handled by BLAS */
      PetscInt     ncols,k; 
      PetscScalar  *work,*workt,*xtmp;

      SETERRQ(PETSC_ERR_SUP,"block size larger than 5 is not supported yet");
      if (!a->mult_work) {
        k = PetscMax(A->rmap->n,A->cmap->n);
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
  ierr = PetscLogFlops(2.0*a->nz*a->bs2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqBAIJ"
PetscErrorCode MatMultTransposeAdd_SeqBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar       *zb,*x,*z,*xb = 0,x1,x2,x3,x4,x5;
  MatScalar         *v;
  PetscErrorCode    ierr;
  PetscInt          mbs,i,*idx,*ii,rval,bs=A->rmap->bs,j,n,bs2=a->bs2,*ib,*ridx=PETSC_NULL;
  Mat_CompressedRow cprow = a->compressedrow;
  PetscTruth        usecprow=cprow.use;

  PetscFunctionBegin;
  if (yy != zz) { ierr = VecCopy_Seq(yy,zz);CHKERRQ(ierr); }
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
        k = PetscMax(A->rmap->n,A->cmap->n);
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
  ierr = PetscLogFlops(2.0*a->nz*a->bs2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_SeqBAIJ"
PetscErrorCode MatScale_SeqBAIJ(Mat inA,PetscScalar alpha)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)inA->data;
  PetscInt       totalnz = a->bs2*a->nz;
  PetscScalar    oalpha = alpha;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,tnz = PetscBLASIntCast(totalnz);

  PetscFunctionBegin;
  BLASscal_(&tnz,&oalpha,a->a,&one);
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
  PetscInt       i,j,k,bs=A->rmap->bs,nz=a->nz,bs2=a->bs2,k1;

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
    ierr = PetscMalloc((A->cmap->n+1)*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
    ierr = PetscMemzero(tmp,A->cmap->n*sizeof(PetscReal));CHKERRQ(ierr);
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
    for (j=0; j<A->cmap->n; j++) {
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
  if ((A->rmap->N != B->rmap->N) || (A->cmap->n != B->cmap->n) || (A->rmap->bs != B->rmap->bs)|| (a->nz != b->nz)) {
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
  ierr = PetscMemcmp(a->a,b->a,(a->nz)*(A->rmap->bs)*(B->rmap->bs)*sizeof(PetscScalar),flg);CHKERRQ(ierr);
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
  bs   = A->rmap->bs;
  aa   = a->a;
  ai   = a->i;
  aj   = a->j;
  ambs = a->mbs;
  bs2  = a->bs2;

  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->rmap->N) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
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
  m   = A->rmap->n;
  n   = A->cmap->n;
  bs  = A->rmap->bs;
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
  info->block_size     = a->bs2;
  info->nz_allocated   = a->maxnz;
  info->nz_used        = a->bs2*a->nz;
  info->nz_unneeded    = (double)(info->nz_allocated - info->nz_used);
  info->assemblies   = A->num_ass;
  info->mallocs      = a->reallocs;
  info->memory       = ((PetscObject)A)->mem;
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
