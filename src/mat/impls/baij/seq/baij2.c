
#include "src/mat/impls/baij/seq/baij.h"
#include "src/inline/spops.h"
#include "src/inline/ilu.h"
#include "petscbt.h"

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_SeqBAIJ"
int MatIncreaseOverlap_SeqBAIJ(Mat A,int is_max,IS is[],int ov)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int         row,i,j,k,l,m,n,*idx,ierr,*nidx,isz,val,ival;
  int         start,end,*ai,*aj,bs,*nidx2;
  PetscBT     table;

  PetscFunctionBegin;
  m     = a->mbs;
  ai    = a->i;
  aj    = a->j;
  bs    = a->bs;

  if (ov < 0)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");

  ierr = PetscBTCreate(m,table);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(int),&nidx);CHKERRQ(ierr); 
  ierr = PetscMalloc((A->m+1)*sizeof(int),&nidx2);CHKERRQ(ierr);

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
int MatGetSubMatrix_SeqBAIJ_Private(Mat A,IS isrow,IS iscol,int cs,MatReuse scall,Mat *B)
{
  Mat_SeqBAIJ  *a = (Mat_SeqBAIJ*)A->data,*c;
  int          *smap,i,k,kstart,kend,ierr,oldcols = a->nbs,*lens;
  int          row,mat_i,*mat_j,tcol,*mat_ilen;
  int          *irow,*icol,nrows,ncols,*ssmap,bs=a->bs,bs2=a->bs2;
  int          *aj = a->j,*ai = a->i;
  MatScalar    *mat_a;
  Mat          C;
  PetscTruth   flag;

  PetscFunctionBegin;
  ierr = ISSorted(iscol,(PetscTruth*)&i);CHKERRQ(ierr);
  if (!i) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"IS is not sorted");

  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);

  ierr = PetscMalloc((1+oldcols)*sizeof(int),&smap);CHKERRQ(ierr);
  ssmap = smap;
  ierr = PetscMalloc((1+nrows)*sizeof(int),&lens);CHKERRQ(ierr);
  ierr  = PetscMemzero(smap,oldcols*sizeof(int));CHKERRQ(ierr);
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

    if (c->mbs!=nrows || c->nbs!=ncols || c->bs!=bs) SETERRQ(PETSC_ERR_ARG_SIZ,"Submatrix wrong size");
    ierr = PetscMemcmp(c->ilen,lens,c->mbs *sizeof(int),&flag);CHKERRQ(ierr);
    if (flag == PETSC_FALSE) {
      SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong no of nonzeros");
    }
    ierr = PetscMemzero(c->ilen,c->mbs*sizeof(int));CHKERRQ(ierr);
    C = *B;
  } else {  
    ierr = MatCreate(A->comm,nrows*bs,ncols*bs,PETSC_DETERMINE,PETSC_DETERMINE,&C);CHKERRQ(ierr);
    ierr = MatSetType(C,A->type_name);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation(C,bs,0,lens);CHKERRQ(ierr);
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
int MatGetSubMatrix_SeqBAIJ(Mat A,IS isrow,IS iscol,int cs,MatReuse scall,Mat *B)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  IS          is1,is2;
  int         *vary,*iary,*irow,*icol,nrows,ncols,i,ierr,bs=a->bs,count;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);
  
  /* Verify if the indices corespond to each element in a block 
   and form the IS with compressed IS */
  ierr = PetscMalloc(2*(a->mbs+1)*sizeof(int),&vary);CHKERRQ(ierr);
  iary = vary + a->mbs;
  ierr = PetscMemzero(vary,(a->mbs)*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<nrows; i++) vary[irow[i]/bs]++;
  count = 0;
  for (i=0; i<a->mbs; i++) {
    if (vary[i]!=0 && vary[i]!=bs) SETERRQ(1,"Index set does not match blocks");
    if (vary[i]==bs) iary[count++] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,count,iary,&is1);CHKERRQ(ierr);
  
  ierr = PetscMemzero(vary,(a->mbs)*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<ncols; i++) vary[icol[i]/bs]++;
  count = 0;
  for (i=0; i<a->mbs; i++) {
    if (vary[i]!=0 && vary[i]!=bs) SETERRQ(1,"Internal error in PETSc");
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
int MatGetSubMatrices_SeqBAIJ(Mat A,int n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  int ierr,i;

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
int MatMult_SeqBAIJ_1(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*z,sum;
  MatScalar       *v;
  int             mbs=a->mbs,i,*idx,*ii,n,ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for (i=0; i<mbs; i++) {
    n    = ii[1] - ii[0]; ii++;
    sum  = 0.0;
    while (n--) sum += *v++ * x[*idx++];
    z[i] = sum;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz - A->m);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_2"
int MatMult_SeqBAIJ_2(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*z,*xb,sum1,sum2;
  PetscScalar     x1,x2;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = 0.0; sum2 = 0.0;
    for (j=0; j<n; j++) {
      xb = x + 2*(*idx++); x1 = xb[0]; x2 = xb[1];
      sum1 += v[0]*x1 + v[2]*x2;
      sum2 += v[1]*x1 + v[3]*x2;
      v += 4;
    }
    z[0] = sum1; z[1] = sum2;
    z += 2;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(8*a->nz - A->m);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_3"
int MatMult_SeqBAIJ_3(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ  *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar  *x,*z,*xb,sum1,sum2,sum3,x1,x2,x3;
  MatScalar    *v;
  int          ierr,mbs=a->mbs,i,*idx,*ii,j,n;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*v,*z,*xb)
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

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
    z[0] = sum1; z[1] = sum2; z[2] = sum3;
    z += 3;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(18*a->nz - A->m);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_4"
int MatMult_SeqBAIJ_4(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*z,*xb,sum1,sum2,sum3,sum4,x1,x2,x3,x4;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

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
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4;
    z += 4;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(32*a->nz - A->m);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_5"
int MatMult_SeqBAIJ_5(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     sum1,sum2,sum3,sum4,sum5,x1,x2,x3,x4,x5,*xb,*z,*x;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

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
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5;
    z += 5;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(50*a->nz - A->m);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_6"
int MatMult_SeqBAIJ_6(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*z,*xb,sum1,sum2,sum3,sum4,sum5,sum6;
  PetscScalar     x1,x2,x3,x4,x5,x6;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

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
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5; z[5] = sum6;
    z += 6;
  }

  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(72*a->nz - A->m);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_7"
int MatMult_SeqBAIJ_7(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*z,*xb,sum1,sum2,sum3,sum4,sum5,sum6,sum7;
  PetscScalar     x1,x2,x3,x4,x5,x6,x7;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

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
    z[0] = sum1; z[1] = sum2; z[2] = sum3; z[3] = sum4; z[4] = sum5; z[5] = sum6; z[6] = sum7;
    z += 7;
  }

  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(98*a->nz - A->m);
  PetscFunctionReturn(0);
}

/*
    This will not work with MatScalar == float because it calls the BLAS
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBAIJ_N"
int MatMult_SeqBAIJ_N(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*z,*xb,*work,*workt;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,bs=a->bs,j,n,bs2=a->bs2;
  int             ncols,k;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);

  idx   = a->j;
  v     = a->a;
  ii    = a->i;


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
    Kernel_w_gets_Ar_times_v(bs,ncols,work,v,z);
    /* LAgemv_("N",&bs,&ncols,&_DOne,v,&bs,work,&_One,&_DZero,z,&_One); */
    v += n*bs2;
    z += bs;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz*bs2 - A->m);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_1"
int MatMultAdd_SeqBAIJ_1(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*y,*z,sum;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for (i=0; i<mbs; i++) {
    n    = ii[1] - ii[0]; ii++;
    sum  = y[i];
    while (n--) sum += *v++ * x[*idx++];
    z[i] = sum;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }
  PetscLogFlops(2*a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_2"
int MatMultAdd_SeqBAIJ_2(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,sum1,sum2;
  PetscScalar     x1,x2;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = y[0]; sum2 = y[1];
    for (j=0; j<n; j++) {
      xb = x + 2*(*idx++); x1 = xb[0]; x2 = xb[1];
      sum1 += v[0]*x1 + v[2]*x2;
      sum2 += v[1]*x1 + v[3]*x2;
      v += 4;
    }
    z[0] = sum1; z[1] = sum2;
    z += 2; y += 2;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }
  PetscLogFlops(4*a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_3"
int MatMultAdd_SeqBAIJ_3(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,sum1,sum2,sum3,x1,x2,x3;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
    sum1 = y[0]; sum2 = y[1]; sum3 = y[2];
    for (j=0; j<n; j++) {
      xb = x + 3*(*idx++); x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
      sum1 += v[0]*x1 + v[3]*x2 + v[6]*x3;
      sum2 += v[1]*x1 + v[4]*x2 + v[7]*x3;
      sum3 += v[2]*x1 + v[5]*x2 + v[8]*x3;
      v += 9;
    }
    z[0] = sum1; z[1] = sum2; z[2] = sum3;
    z += 3; y += 3;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }
  PetscLogFlops(18*a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_4"
int MatMultAdd_SeqBAIJ_4(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,sum1,sum2,sum3,sum4,x1,x2,x3,x4;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii;
  int             j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
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
    z += 4; y += 4;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }
  PetscLogFlops(32*a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_5"
int MatMultAdd_SeqBAIJ_5(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,sum1,sum2,sum3,sum4,sum5,x1,x2,x3,x4,x5;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
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
    z += 5; y += 5;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }
  PetscLogFlops(50*a->nz);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_6"
int MatMultAdd_SeqBAIJ_6(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,sum1,sum2,sum3,sum4,sum5,sum6;
  PetscScalar     x1,x2,x3,x4,x5,x6;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
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
    z += 6; y += 6;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }
  PetscLogFlops(72*a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_7"
int MatMultAdd_SeqBAIJ_7(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *x,*y,*z,*xb,sum1,sum2,sum3,sum4,sum5,sum6,sum7;
  PetscScalar     x1,x2,x3,x4,x5,x6,x7;
  MatScalar       *v;
  int             ierr,mbs=a->mbs,i,*idx,*ii,j,n;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }

  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  for (i=0; i<mbs; i++) {
    n  = ii[1] - ii[0]; ii++; 
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
    z += 7; y += 7;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }
  PetscLogFlops(98*a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBAIJ_N"
int MatMultAdd_SeqBAIJ_N(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar    *x,*z,*xb,*work,*workt,*y;
  MatScalar      *v;
  int            mbs=a->mbs,i,*idx,*ii,bs=a->bs,j,n,bs2=a->bs2,ierr;
  int            ncols,k;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  if (zz != yy) { 
    ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
    ierr = PetscMemcpy(z,y,yy->n*sizeof(PetscScalar));CHKERRQ(ierr); 
    ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  }

  idx   = a->j;
  v     = a->a;
  ii    = a->i;


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
    Kernel_w_gets_w_plus_Ar_times_v(bs,ncols,work,v,z);
    /* LAgemv_("N",&bs,&ncols,&_DOne,v,&bs,work,&_One,&_DOne,z,&_One); */
    v += n*bs2;
    z += bs;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz*bs2);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqBAIJ"
int MatMultTranspose_SeqBAIJ(Mat A,Vec xx,Vec zz)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *xg,*zg,*zb,zero = 0.0;
  PetscScalar     *x,*z,*xb,x1,x2,x3,x4,x5,x6,x7;
  MatScalar       *v;
  int             mbs=a->mbs,i,*idx,*ii,*ai=a->i,rval;
  int             bs=a->bs,j,n,bs2=a->bs2,*ib,ierr;

  PetscFunctionBegin;
  ierr = VecSet(&zero,zz);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&xg);CHKERRQ(ierr); x = xg;
  ierr = VecGetArray(zz,&zg);CHKERRQ(ierr); z = zg;

  idx   = a->j;
  v     = a->a;
  ii    = a->i;
  xb    = x;
  switch (bs) {
  case 1:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++;
      x1 = xb[0];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval    = ib[j];
        z[rval] += *v * x1;
        v++;
      }
      xb++;
    }
    break;
  case 2:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++; 
      x1 = xb[0]; x2 = xb[1];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval      = ib[j]*2;
        z[rval++] += v[0]*x1 + v[1]*x2;
        z[rval]   += v[2]*x1 + v[3]*x2;
        v  += 4;
      }
      xb += 2;
    }
    break;
  case 3:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++; 
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval      = ib[j]*3;
        z[rval++] += v[0]*x1 + v[1]*x2 + v[2]*x3;
        z[rval++] += v[3]*x1 + v[4]*x2 + v[5]*x3;
        z[rval]   += v[6]*x1 + v[7]*x2 + v[8]*x3;
        v  += 9;
      }
      xb += 3;
    }
    break;
  case 4:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++; 
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval      = ib[j]*4;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4;
        z[rval++] +=  v[4]*x1 +  v[5]*x2 +  v[6]*x3 +  v[7]*x4;
        z[rval++] +=  v[8]*x1 +  v[9]*x2 + v[10]*x3 + v[11]*x4;
        z[rval]   += v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4;
        v  += 16;
      }
      xb += 4;
    }
    break;
  case 5:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++; 
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; 
      x4 = xb[3]; x5 = xb[4];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval      = ib[j]*5;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5;
        z[rval++] +=  v[5]*x1 +  v[6]*x2 +  v[7]*x3 +  v[8]*x4 +  v[9]*x5;
        z[rval++] += v[10]*x1 + v[11]*x2 + v[12]*x3 + v[13]*x4 + v[14]*x5;
        z[rval++] += v[15]*x1 + v[16]*x2 + v[17]*x3 + v[18]*x4 + v[19]*x5;
        z[rval]   += v[20]*x1 + v[21]*x2 + v[22]*x3 + v[23]*x4 + v[24]*x5;
        v  += 25;
      }
      xb += 5;
    }
    break;
  case 6:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++; 
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; 
      x4 = xb[3]; x5 = xb[4]; x6 = xb[5];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval      = ib[j]*6;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 + v[4]*x5 + v[5]*x6;
        z[rval++] +=  v[6]*x1 +  v[7]*x2 +  v[8]*x3 +  v[9]*x4 + v[10]*x5 + v[11]*x6;
        z[rval++] += v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4 + v[16]*x5 + v[17]*x6;
        z[rval++] += v[18]*x1 + v[19]*x2 + v[20]*x3 + v[21]*x4 + v[22]*x5 + v[23]*x6;
        z[rval++] += v[24]*x1 + v[25]*x2 + v[26]*x3 + v[27]*x4 + v[28]*x5 + v[29]*x6;
        z[rval]   += v[30]*x1 + v[31]*x2 + v[32]*x3 + v[33]*x4 + v[34]*x5 + v[35]*x6;
        v  += 36;
      }
      xb += 6;
    }
    break;
  case 7:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++; 
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; 
      x4 = xb[3]; x5 = xb[4]; x6 = xb[5]; x7 = xb[6];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval      = ib[j]*7;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 + v[4]*x5 + v[5]*x6 + v[6]*x7;
        z[rval++] +=  v[7]*x1 +  v[8]*x2 +  v[9]*x3 + v[10]*x4 + v[11]*x5 + v[12]*x6 + v[13]*x7;
        z[rval++] += v[14]*x1 + v[15]*x2 + v[16]*x3 + v[17]*x4 + v[18]*x5 + v[19]*x6 + v[20]*x7;
        z[rval++] += v[21]*x1 + v[22]*x2 + v[23]*x3 + v[24]*x4 + v[25]*x5 + v[26]*x6 + v[27]*x7;
        z[rval++] += v[28]*x1 + v[29]*x2 + v[30]*x3 + v[31]*x4 + v[32]*x5 + v[33]*x6 + v[34]*x7;
        z[rval++] += v[35]*x1 + v[36]*x2 + v[37]*x3 + v[38]*x4 + v[39]*x5 + v[40]*x6 + v[41]*x7;
        z[rval]   += v[42]*x1 + v[43]*x2 + v[44]*x3 + v[45]*x4 + v[46]*x5 + v[47]*x6 + v[48]*x7;
        v  += 49;
      }
      xb += 7;
    }
    break;
  default: {       /* block sizes larger then 7 by 7 are handled by BLAS */
      int          ncols,k;
      PetscScalar  *work,*workt;

      if (!a->mult_work) {
        k = PetscMax(A->m,A->n);
        ierr = PetscMalloc((k+1)*sizeof(PetscScalar),&a->mult_work);CHKERRQ(ierr);
      }
      work = a->mult_work;
      for (i=0; i<mbs; i++) {
        n     = ii[1] - ii[0]; ii++;
        ncols = n*bs;
        ierr  = PetscMemzero(work,ncols*sizeof(PetscScalar));CHKERRQ(ierr);
        Kernel_w_gets_w_plus_trans_Ar_times_v(bs,ncols,x,v,work);
        /* LAgemv_("T",&bs,&ncols,&_DOne,v,&bs,x,&_One,&_DOne,work,&_One); */
        v += n*bs2;
        x += bs;
        workt = work;
        for (j=0; j<n; j++) {
          zb = z + bs*(*idx++);
          for (k=0; k<bs; k++) zb[k] += workt[k] ;
          workt += bs;
        }
      }
    }
  }
  ierr = VecRestoreArray(xx,&xg);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zg);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz*a->bs2 - A->n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqBAIJ"
int MatMultTransposeAdd_SeqBAIJ(Mat A,Vec xx,Vec yy,Vec zz)

{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar     *xg,*zg,*zb,*x,*z,*xb,x1,x2,x3,x4,x5;
  MatScalar       *v;
  int             mbs=a->mbs,i,*idx,*ii,*ai=a->i,rval,bs=a->bs,j,n,bs2=a->bs2,*ib,ierr;

  PetscFunctionBegin;
  if (yy != zz) { ierr = VecCopy(yy,zz);CHKERRQ(ierr); }
  ierr = VecGetArray(xx,&xg);CHKERRQ(ierr); x = xg;
  ierr = VecGetArray(zz,&zg);CHKERRQ(ierr); z = zg;


  idx   = a->j;
  v     = a->a;
  ii    = a->i;
  xb    = x;

  switch (bs) {
  case 1:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++;
      x1 = xb[0];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval    = ib[j];
        z[rval] += *v * x1;
        v++;
      }
      xb++;
    }
    break;
  case 2:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++; 
      x1 = xb[0]; x2 = xb[1];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval      = ib[j]*2;
        z[rval++] += v[0]*x1 + v[1]*x2;
        z[rval++] += v[2]*x1 + v[3]*x2;
        v  += 4;
      }
      xb += 2;
    }
    break;
  case 3:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++; 
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval      = ib[j]*3;
        z[rval++] += v[0]*x1 + v[1]*x2 + v[2]*x3;
        z[rval++] += v[3]*x1 + v[4]*x2 + v[5]*x3;
        z[rval++] += v[6]*x1 + v[7]*x2 + v[8]*x3;
        v  += 9;
      }
      xb += 3;
    }
    break;
  case 4:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++; 
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval      = ib[j]*4;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4;
        z[rval++] +=  v[4]*x1 +  v[5]*x2 +  v[6]*x3 +  v[7]*x4;
        z[rval++] +=  v[8]*x1 +  v[9]*x2 + v[10]*x3 + v[11]*x4;
        z[rval++] += v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4;
        v  += 16;
      }
      xb += 4;
    }
    break;
  case 5:
    for (i=0; i<mbs; i++) {
      n  = ii[1] - ii[0]; ii++; 
      x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; 
      x4 = xb[3]; x5 = xb[4];
      ib = idx + ai[i];
      for (j=0; j<n; j++) {
        rval      = ib[j]*5;
        z[rval++] +=  v[0]*x1 +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5;
        z[rval++] +=  v[5]*x1 +  v[6]*x2 +  v[7]*x3 +  v[8]*x4 +  v[9]*x5;
        z[rval++] += v[10]*x1 + v[11]*x2 + v[12]*x3 + v[13]*x4 + v[14]*x5;
        z[rval++] += v[15]*x1 + v[16]*x2 + v[17]*x3 + v[18]*x4 + v[19]*x5;
        z[rval++] += v[20]*x1 + v[21]*x2 + v[22]*x3 + v[23]*x4 + v[24]*x5;
        v  += 25;
      }
      xb += 5;
    }
    break;
  default: {      /* block sizes larger then 5 by 5 are handled by BLAS */
      int          ncols,k; 
      PetscScalar  *work,*workt;

      if (!a->mult_work) {
        k = PetscMax(A->m,A->n);
        ierr = PetscMalloc((k+1)*sizeof(PetscScalar),&a->mult_work);CHKERRQ(ierr);
      }
      work = a->mult_work;
      for (i=0; i<mbs; i++) {
        n     = ii[1] - ii[0]; ii++;
        ncols = n*bs;
        ierr  = PetscMemzero(work,ncols*sizeof(PetscScalar));CHKERRQ(ierr);
        Kernel_w_gets_w_plus_trans_Ar_times_v(bs,ncols,x,v,work);
        /* LAgemv_("T",&bs,&ncols,&_DOne,v,&bs,x,&_One,&_DOne,work,&_One); */
        v += n*bs2;
        x += bs;
        workt = work;
        for (j=0; j<n; j++) {
          zb = z + bs*(*idx++);
          for (k=0; k<bs; k++) zb[k] += workt[k] ;
          workt += bs;
        }
      }
    }
  }
  ierr = VecRestoreArray(xx,&xg);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&zg);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz*a->bs2);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_SeqBAIJ"
int MatScale_SeqBAIJ(const PetscScalar *alpha,Mat inA)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)inA->data;
  int         totalnz = a->bs2*a->nz;
#if defined(PETSC_USE_MAT_SINGLE)
  int         i;
#else
  int         one = 1;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_MAT_SINGLE)
  for (i=0; i<totalnz; i++) a->a[i] *= *alpha;
#else
  BLscal_(&totalnz,(PetscScalar*)alpha,a->a,&one);
#endif
  PetscLogFlops(totalnz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_SeqBAIJ"
int MatNorm_SeqBAIJ(Mat A,NormType type,PetscReal *norm)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  MatScalar   *v = a->a;
  PetscReal   sum = 0.0;
  int         i,j,k,bs = a->bs,nz=a->nz,bs2=a->bs2,k1;

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
  }  else if (type == NORM_INFINITY) { /* maximum row sum */
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
int MatEqual_SeqBAIJ(Mat A,Mat B,PetscTruth* flg)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *)A->data,*b = (Mat_SeqBAIJ *)B->data;
  int         ierr;

  PetscFunctionBegin;
  /* If the  matrix/block dimensions are not equal, or no of nonzeros or shift */
  if ((A->m != B->m) || (A->n != B->n) || (a->bs != b->bs)|| (a->nz != b->nz)) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0); 
  }
  
  /* if the a->i are the same */
  ierr = PetscMemcmp(a->i,b->i,(a->mbs+1)*sizeof(int),flg);CHKERRQ(ierr);
  if (*flg == PETSC_FALSE) {
    PetscFunctionReturn(0);
  }
  
  /* if a->j are the same */
  ierr = PetscMemcmp(a->j,b->j,(a->nz)*sizeof(int),flg);CHKERRQ(ierr);
  if (*flg == PETSC_FALSE) {
    PetscFunctionReturn(0);
  }  
  /* if a->a are the same */
  ierr = PetscMemcmp(a->a,b->a,(a->nz)*(a->bs)*(a->bs)*sizeof(PetscScalar),flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
  
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_SeqBAIJ"
int MatGetDiagonal_SeqBAIJ(Mat A,Vec v)
{
  Mat_SeqBAIJ  *a = (Mat_SeqBAIJ*)A->data;
  int          ierr,i,j,k,n,row,bs,*ai,*aj,ambs,bs2;
  PetscScalar  *x,zero = 0.0;
  MatScalar    *aa,*aa_j;

  PetscFunctionBegin;
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");  
  bs   = a->bs;
  aa   = a->a;
  ai   = a->i;
  aj   = a->j;
  ambs = a->mbs;
  bs2  = a->bs2;

  ierr = VecSet(&zero,v);CHKERRQ(ierr);
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
int MatDiagonalScale_SeqBAIJ(Mat A,Vec ll,Vec rr)
{
  Mat_SeqBAIJ  *a = (Mat_SeqBAIJ*)A->data;
  PetscScalar  *l,*r,x,*li,*ri;
  MatScalar    *aa,*v;
  int          ierr,i,j,k,lm,rn,M,m,n,*ai,*aj,mbs,tmp,bs,bs2;

  PetscFunctionBegin;
  ai  = a->i;
  aj  = a->j;
  aa  = a->a;
  m   = A->m;
  n   = A->n;
  bs  = a->bs;
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
    PetscLogFlops(a->nz);
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
    PetscLogFlops(a->nz);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_SeqBAIJ"
int MatGetInfo_SeqBAIJ(Mat A,MatInfoType flag,MatInfo *info)
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
int MatZeroEntries_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data; 
  int         ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(a->a,a->bs2*a->i[a->mbs]*sizeof(MatScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
