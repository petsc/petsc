#include <../src/mat/impls/baij/mpi/mpibaij.h>   /*I  "petscmat.h"  I*/

#include <petsc/private/hashseti.h>
#include <petscblaslapack.h>
#include <petscsf.h>

#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatConvert_AIJ_HYPRE(Mat,MatType,MatReuse,Mat*);
#endif

PetscErrorCode MatGetRowMaxAbs_MPIBAIJ(Mat A,Vec v,PetscInt idx[])
{
  Mat_MPIBAIJ       *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,*idxb = NULL,m = A->rmap->n,bs = A->cmap->bs;
  PetscScalar       *va,*vv;
  Vec               vB,vA;
  const PetscScalar *vb;

  PetscFunctionBegin;
  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&vA);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(a->A,vA,idx);CHKERRQ(ierr);

  ierr = VecGetArrayWrite(vA,&va);CHKERRQ(ierr);
  if (idx) {
    for (i=0; i<m; i++) {
      if (PetscAbsScalar(va[i])) idx[i] += A->cmap->rstart;
    }
  }

  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&vB);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&idxb);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(a->B,vB,idxb);CHKERRQ(ierr);

  ierr = VecGetArrayWrite(v,&vv);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vB,&vb);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    if (PetscAbsScalar(va[i]) < PetscAbsScalar(vb[i])) {
      vv[i] = vb[i];
      if (idx) idx[i] = bs*a->garray[idxb[i]/bs] + (idxb[i] % bs);
    } else {
      vv[i] = va[i];
      if (idx && PetscAbsScalar(va[i]) == PetscAbsScalar(vb[i]) && idxb[i] != -1 && idx[i] > bs*a->garray[idxb[i]/bs] + (idxb[i] % bs))
        idx[i] = bs*a->garray[idxb[i]/bs] + (idxb[i] % bs);
    }
  }
  ierr = VecRestoreArrayWrite(vA,&vv);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(vA,&va);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vB,&vb);CHKERRQ(ierr);
  ierr = PetscFree(idxb);CHKERRQ(ierr);
  ierr = VecDestroy(&vA);CHKERRQ(ierr);
  ierr = VecDestroy(&vB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatStoreValues_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ    *aij = (Mat_MPIBAIJ*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatStoreValues(aij->A);CHKERRQ(ierr);
  ierr = MatStoreValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatRetrieveValues_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ    *aij = (Mat_MPIBAIJ*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRetrieveValues(aij->A);CHKERRQ(ierr);
  ierr = MatRetrieveValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Local utility routine that creates a mapping from the global column
   number to the local number in the off-diagonal part of the local
   storage of the matrix.  This is done in a non scalable way since the
   length of colmap equals the global matrix length.
*/
PetscErrorCode MatCreateColmap_MPIBAIJ_Private(Mat mat)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ    *B    = (Mat_SeqBAIJ*)baij->B->data;
  PetscErrorCode ierr;
  PetscInt       nbs = B->nbs,i,bs=mat->rmap->bs;

  PetscFunctionBegin;
#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableCreate(baij->nbs,baij->Nbs+1,&baij->colmap);CHKERRQ(ierr);
  for (i=0; i<nbs; i++) {
    ierr = PetscTableAdd(baij->colmap,baij->garray[i]+1,i*bs+1,INSERT_VALUES);CHKERRQ(ierr);
  }
#else
  ierr = PetscCalloc1(baij->Nbs+1,&baij->colmap);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)mat,baij->Nbs*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<nbs; i++) baij->colmap[baij->garray[i]] = i*bs+1;
#endif
  PetscFunctionReturn(0);
}

#define  MatSetValues_SeqBAIJ_A_Private(row,col,value,addv,orow,ocol)       \
  { \
    brow = row/bs;  \
    rp   = aj + ai[brow]; ap = aa + bs2*ai[brow]; \
    rmax = aimax[brow]; nrow = ailen[brow]; \
    bcol = col/bs; \
    ridx = row % bs; cidx = col % bs; \
    low  = 0; high = nrow; \
    while (high-low > 3) { \
      t = (low+high)/2; \
      if (rp[t] > bcol) high = t; \
      else              low  = t; \
    } \
    for (_i=low; _i<high; _i++) { \
      if (rp[_i] > bcol) break; \
      if (rp[_i] == bcol) { \
        bap = ap +  bs2*_i + bs*cidx + ridx; \
        if (addv == ADD_VALUES) *bap += value;  \
        else                    *bap  = value;  \
        goto a_noinsert; \
      } \
    } \
    if (a->nonew == 1) goto a_noinsert; \
    PetscCheckFalse(a->nonew == -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,aimax,a->nonew,MatScalar); \
    N = nrow++ - 1;  \
    /* shift up all the later entries in this row */ \
    ierr = PetscArraymove(rp+_i+1,rp+_i,N-_i+1);CHKERRQ(ierr);\
    ierr = PetscArraymove(ap+bs2*(_i+1),ap+bs2*_i,bs2*(N-_i+1));CHKERRQ(ierr); \
    ierr = PetscArrayzero(ap+bs2*_i,bs2);CHKERRQ(ierr);  \
    rp[_i]                      = bcol;  \
    ap[bs2*_i + bs*cidx + ridx] = value;  \
a_noinsert:; \
    ailen[brow] = nrow; \
  }

#define  MatSetValues_SeqBAIJ_B_Private(row,col,value,addv,orow,ocol)       \
  { \
    brow = row/bs;  \
    rp   = bj + bi[brow]; ap = ba + bs2*bi[brow]; \
    rmax = bimax[brow]; nrow = bilen[brow]; \
    bcol = col/bs; \
    ridx = row % bs; cidx = col % bs; \
    low  = 0; high = nrow; \
    while (high-low > 3) { \
      t = (low+high)/2; \
      if (rp[t] > bcol) high = t; \
      else              low  = t; \
    } \
    for (_i=low; _i<high; _i++) { \
      if (rp[_i] > bcol) break; \
      if (rp[_i] == bcol) { \
        bap = ap +  bs2*_i + bs*cidx + ridx; \
        if (addv == ADD_VALUES) *bap += value;  \
        else                    *bap  = value;  \
        goto b_noinsert; \
      } \
    } \
    if (b->nonew == 1) goto b_noinsert; \
    PetscCheckFalse(b->nonew == -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column  (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(B,b->mbs,bs2,nrow,brow,bcol,rmax,ba,bi,bj,rp,ap,bimax,b->nonew,MatScalar); \
    N = nrow++ - 1;  \
    /* shift up all the later entries in this row */ \
    ierr = PetscArraymove(rp+_i+1,rp+_i,N-_i+1);CHKERRQ(ierr);\
    ierr = PetscArraymove(ap+bs2*(_i+1),ap+bs2*_i,bs2*(N-_i+1));CHKERRQ(ierr);\
    ierr = PetscArrayzero(ap+bs2*_i,bs2);CHKERRQ(ierr);  \
    rp[_i]                      = bcol;  \
    ap[bs2*_i + bs*cidx + ridx] = value;  \
b_noinsert:; \
    bilen[brow] = nrow; \
  }

PetscErrorCode MatSetValues_MPIBAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  MatScalar      value;
  PetscBool      roworiented = baij->roworiented;
  PetscErrorCode ierr;
  PetscInt       i,j,row,col;
  PetscInt       rstart_orig=mat->rmap->rstart;
  PetscInt       rend_orig  =mat->rmap->rend,cstart_orig=mat->cmap->rstart;
  PetscInt       cend_orig  =mat->cmap->rend,bs=mat->rmap->bs;

  /* Some Variables required in the macro */
  Mat         A     = baij->A;
  Mat_SeqBAIJ *a    = (Mat_SeqBAIJ*)(A)->data;
  PetscInt    *aimax=a->imax,*ai=a->i,*ailen=a->ilen,*aj=a->j;
  MatScalar   *aa   =a->a;

  Mat         B     = baij->B;
  Mat_SeqBAIJ *b    = (Mat_SeqBAIJ*)(B)->data;
  PetscInt    *bimax=b->imax,*bi=b->i,*bilen=b->ilen,*bj=b->j;
  MatScalar   *ba   =b->a;

  PetscInt  *rp,ii,nrow,_i,rmax,N,brow,bcol;
  PetscInt  low,high,t,ridx,cidx,bs2=a->bs2;
  MatScalar *ap,*bap;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
    PetscCheckFalse(im[i] >= mat->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,im[i],mat->rmap->N-1);
    if (im[i] >= rstart_orig && im[i] < rend_orig) {
      row = im[i] - rstart_orig;
      for (j=0; j<n; j++) {
        if (in[j] >= cstart_orig && in[j] < cend_orig) {
          col = in[j] - cstart_orig;
          if (roworiented) value = v[i*n+j];
          else             value = v[i+j*m];
          MatSetValues_SeqBAIJ_A_Private(row,col,value,addv,im[i],in[j]);
        } else if (in[j] < 0) continue;
        else PetscCheckFalse(in[j] >= mat->cmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,in[j],mat->cmap->N-1);
        else {
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = MatCreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
            }
#if defined(PETSC_USE_CTABLE)
            ierr = PetscTableFind(baij->colmap,in[j]/bs + 1,&col);CHKERRQ(ierr);
            col  = col - 1;
#else
            col = baij->colmap[in[j]/bs] - 1;
#endif
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->B->data))->nonew) {
              ierr = MatDisAssemble_MPIBAIJ(mat);CHKERRQ(ierr);
              col  =  in[j];
              /* Reinitialize the variables required by MatSetValues_SeqBAIJ_B_Private() */
              B    = baij->B;
              b    = (Mat_SeqBAIJ*)(B)->data;
              bimax=b->imax;bi=b->i;bilen=b->ilen;bj=b->j;
              ba   =b->a;
            } else PetscCheckFalse(col < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", im[i], in[j]);
            else col += in[j]%bs;
          } else col = in[j];
          if (roworiented) value = v[i*n+j];
          else             value = v[i+j*m];
          MatSetValues_SeqBAIJ_B_Private(row,col,value,addv,im[i],in[j]);
          /* ierr = MatSetValues_SeqBAIJ(baij->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
      }
    } else {
      PetscCheckFalse(mat->nooffprocentries,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
      if (!baij->donotstash) {
        mat->assembled = PETSC_FALSE;
        if (roworiented) {
          ierr = MatStashValuesRow_Private(&mat->stash,im[i],n,in,v+i*n,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m,PETSC_FALSE);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatSetValuesBlocked_SeqBAIJ_Inlined(Mat A,PetscInt row,PetscInt col,const PetscScalar v[],InsertMode is,PetscInt orow,PetscInt ocol)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscInt          *rp,low,high,t,ii,jj,nrow,i,rmax,N;
  PetscInt          *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscErrorCode    ierr;
  PetscInt          *aj        =a->j,nonew=a->nonew,bs2=a->bs2,bs=A->rmap->bs;
  PetscBool         roworiented=a->roworiented;
  const PetscScalar *value     = v;
  MatScalar         *ap,*aa = a->a,*bap;

  PetscFunctionBegin;
  rp   = aj + ai[row];
  ap   = aa + bs2*ai[row];
  rmax = imax[row];
  nrow = ailen[row];
  value = v;
  low = 0;
  high = nrow;
  while (high-low > 7) {
    t = (low+high)/2;
    if (rp[t] > col) high = t;
    else             low  = t;
  }
  for (i=low; i<high; i++) {
    if (rp[i] > col) break;
    if (rp[i] == col) {
      bap = ap +  bs2*i;
      if (roworiented) {
        if (is == ADD_VALUES) {
          for (ii=0; ii<bs; ii++) {
            for (jj=ii; jj<bs2; jj+=bs) {
              bap[jj] += *value++;
            }
          }
        } else {
          for (ii=0; ii<bs; ii++) {
            for (jj=ii; jj<bs2; jj+=bs) {
              bap[jj] = *value++;
            }
          }
        }
      } else {
        if (is == ADD_VALUES) {
          for (ii=0; ii<bs; ii++,value+=bs) {
            for (jj=0; jj<bs; jj++) {
              bap[jj] += value[jj];
            }
            bap += bs;
          }
        } else {
          for (ii=0; ii<bs; ii++,value+=bs) {
            for (jj=0; jj<bs; jj++) {
              bap[jj]  = value[jj];
            }
            bap += bs;
          }
        }
      }
      goto noinsert2;
    }
  }
  if (nonew == 1) goto noinsert2;
  PetscCheckFalse(nonew == -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new global block indexed nonzero block (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", orow, ocol);
  MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
  N = nrow++ - 1; high++;
  /* shift up all the later entries in this row */
  ierr = PetscArraymove(rp+i+1,rp+i,N-i+1);CHKERRQ(ierr);
  ierr = PetscArraymove(ap+bs2*(i+1),ap+bs2*i,bs2*(N-i+1));CHKERRQ(ierr);
  rp[i] = col;
  bap   = ap +  bs2*i;
  if (roworiented) {
    for (ii=0; ii<bs; ii++) {
      for (jj=ii; jj<bs2; jj+=bs) {
        bap[jj] = *value++;
      }
    }
  } else {
    for (ii=0; ii<bs; ii++) {
      for (jj=0; jj<bs; jj++) {
        *bap++ = *value++;
      }
    }
  }
  noinsert2:;
  ailen[row] = nrow;
  PetscFunctionReturn(0);
}

/*
    This routine should be optimized so that the block copy at ** Here a copy is required ** below is not needed
    by passing additional stride information into the MatSetValuesBlocked_SeqBAIJ_Inlined() routine
*/
PetscErrorCode MatSetValuesBlocked_MPIBAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIBAIJ       *baij = (Mat_MPIBAIJ*)mat->data;
  const PetscScalar *value;
  MatScalar         *barray     = baij->barray;
  PetscBool         roworiented = baij->roworiented;
  PetscErrorCode    ierr;
  PetscInt          i,j,ii,jj,row,col,rstart=baij->rstartbs;
  PetscInt          rend=baij->rendbs,cstart=baij->cstartbs,stepval;
  PetscInt          cend=baij->cendbs,bs=mat->rmap->bs,bs2=baij->bs2;

  PetscFunctionBegin;
  if (!barray) {
    ierr         = PetscMalloc1(bs2,&barray);CHKERRQ(ierr);
    baij->barray = barray;
  }

  if (roworiented) stepval = (n-1)*bs;
  else stepval = (m-1)*bs;

  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
    PetscCheck(im[i] < baij->Mbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block indexed row too large %" PetscInt_FMT " max %" PetscInt_FMT,im[i],baij->Mbs-1);
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for (j=0; j<n; j++) {
        /* If NumCol = 1 then a copy is not required */
        if ((roworiented) && (n == 1)) {
          barray = (MatScalar*)v + i*bs2;
        } else if ((!roworiented) && (m == 1)) {
          barray = (MatScalar*)v + j*bs2;
        } else { /* Here a copy is required */
          if (roworiented) {
            value = v + (i*(stepval+bs) + j)*bs;
          } else {
            value = v + (j*(stepval+bs) + i)*bs;
          }
          for (ii=0; ii<bs; ii++,value+=bs+stepval) {
            for (jj=0; jj<bs; jj++) barray[jj] = value[jj];
            barray += bs;
          }
          barray -= bs2;
        }

        if (in[j] >= cstart && in[j] < cend) {
          col  = in[j] - cstart;
          ierr = MatSetValuesBlocked_SeqBAIJ_Inlined(baij->A,row,col,barray,addv,im[i],in[j]);CHKERRQ(ierr);
        } else if (in[j] < 0) continue;
        else PetscCheck(in[j] < baij->Nbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block indexed column too large %" PetscInt_FMT " max %" PetscInt_FMT,in[j],baij->Nbs-1);
        else {
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = MatCreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
            }

#if defined(PETSC_USE_DEBUG)
#if defined(PETSC_USE_CTABLE)
            { PetscInt data;
              ierr = PetscTableFind(baij->colmap,in[j]+1,&data);CHKERRQ(ierr);
              PetscCheckFalse((data - 1) % bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect colmap");
            }
#else
            PetscCheckFalse((baij->colmap[in[j]] - 1) % bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect colmap");
#endif
#endif
#if defined(PETSC_USE_CTABLE)
            ierr = PetscTableFind(baij->colmap,in[j]+1,&col);CHKERRQ(ierr);
            col  = (col - 1)/bs;
#else
            col = (baij->colmap[in[j]] - 1)/bs;
#endif
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->B->data))->nonew) {
              ierr = MatDisAssemble_MPIBAIJ(mat);CHKERRQ(ierr);
              col  =  in[j];
            } else PetscCheckFalse(col < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new blocked indexed nonzero block (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix",im[i],in[j]);
          } else col = in[j];
          ierr = MatSetValuesBlocked_SeqBAIJ_Inlined(baij->B,row,col,barray,addv,im[i],in[j]);CHKERRQ(ierr);
        }
      }
    } else {
      PetscCheckFalse(mat->nooffprocentries,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process block indexed row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
      if (!baij->donotstash) {
        if (roworiented) {
          ierr = MatStashValuesRowBlocked_Private(&mat->bstash,im[i],n,in,v,m,n,i);CHKERRQ(ierr);
        } else {
          ierr = MatStashValuesColBlocked_Private(&mat->bstash,im[i],n,in,v,m,n,i);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#define HASH_KEY 0.6180339887
#define HASH(size,key,tmp) (tmp = (key)*HASH_KEY,(PetscInt)((size)*(tmp-(PetscInt)tmp)))
/* #define HASH(size,key) ((PetscInt)((size)*fmod(((key)*HASH_KEY),1))) */
/* #define HASH(size,key,tmp) ((PetscInt)((size)*fmod(((key)*HASH_KEY),1))) */
PetscErrorCode MatSetValues_MPIBAIJ_HT(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIBAIJ    *baij       = (Mat_MPIBAIJ*)mat->data;
  PetscBool      roworiented = baij->roworiented;
  PetscErrorCode ierr;
  PetscInt       i,j,row,col;
  PetscInt       rstart_orig=mat->rmap->rstart;
  PetscInt       rend_orig  =mat->rmap->rend,Nbs=baij->Nbs;
  PetscInt       h1,key,size=baij->ht_size,bs=mat->rmap->bs,*HT=baij->ht,idx;
  PetscReal      tmp;
  MatScalar      **HD = baij->hd,value;
  PetscInt       total_ct=baij->ht_total_ct,insert_ct=baij->ht_insert_ct;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (PetscDefined(USE_DEBUG)) {
      PetscCheckFalse(im[i] < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row");
      PetscCheckFalse(im[i] >= mat->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,im[i],mat->rmap->N-1);
    }
    row = im[i];
    if (row >= rstart_orig && row < rend_orig) {
      for (j=0; j<n; j++) {
        col = in[j];
        if (roworiented) value = v[i*n+j];
        else             value = v[i+j*m];
        /* Look up PetscInto the Hash Table */
        key = (row/bs)*Nbs+(col/bs)+1;
        h1  = HASH(size,key,tmp);

        idx = h1;
        if (PetscDefined(USE_DEBUG)) {
          insert_ct++;
          total_ct++;
          if (HT[idx] != key) {
            for (idx=h1; (idx<size) && (HT[idx]!=key); idx++,total_ct++) ;
            if (idx == size) {
              for (idx=0; (idx<h1) && (HT[idx]!=key); idx++,total_ct++) ;
              PetscCheckFalse(idx == h1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"(%" PetscInt_FMT ",%" PetscInt_FMT ") has no entry in the hash table", row, col);
            }
          }
        } else if (HT[idx] != key) {
          for (idx=h1; (idx<size) && (HT[idx]!=key); idx++) ;
          if (idx == size) {
            for (idx=0; (idx<h1) && (HT[idx]!=key); idx++) ;
            PetscCheckFalse(idx == h1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"(%" PetscInt_FMT ",%" PetscInt_FMT ") has no entry in the hash table", row, col);
          }
        }
        /* A HASH table entry is found, so insert the values at the correct address */
        if (addv == ADD_VALUES) *(HD[idx]+ (col % bs)*bs + (row % bs)) += value;
        else                    *(HD[idx]+ (col % bs)*bs + (row % bs))  = value;
      }
    } else if (!baij->donotstash) {
      if (roworiented) {
        ierr = MatStashValuesRow_Private(&mat->stash,im[i],n,in,v+i*n,PETSC_FALSE);CHKERRQ(ierr);
      } else {
        ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
  }
  if (PetscDefined(USE_DEBUG)) {
    baij->ht_total_ct  += total_ct;
    baij->ht_insert_ct += insert_ct;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValuesBlocked_MPIBAIJ_HT(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIBAIJ       *baij       = (Mat_MPIBAIJ*)mat->data;
  PetscBool         roworiented = baij->roworiented;
  PetscErrorCode    ierr;
  PetscInt          i,j,ii,jj,row,col;
  PetscInt          rstart=baij->rstartbs;
  PetscInt          rend  =mat->rmap->rend,stepval,bs=mat->rmap->bs,bs2=baij->bs2,nbs2=n*bs2;
  PetscInt          h1,key,size=baij->ht_size,idx,*HT=baij->ht,Nbs=baij->Nbs;
  PetscReal         tmp;
  MatScalar         **HD = baij->hd,*baij_a;
  const PetscScalar *v_t,*value;
  PetscInt          total_ct=baij->ht_total_ct,insert_ct=baij->ht_insert_ct;

  PetscFunctionBegin;
  if (roworiented) stepval = (n-1)*bs;
  else stepval = (m-1)*bs;

  for (i=0; i<m; i++) {
    if (PetscDefined(USE_DEBUG)) {
      PetscCheckFalse(im[i] < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %" PetscInt_FMT,im[i]);
      PetscCheckFalse(im[i] >= baij->Mbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,im[i],baij->Mbs-1);
    }
    row = im[i];
    v_t = v + i*nbs2;
    if (row >= rstart && row < rend) {
      for (j=0; j<n; j++) {
        col = in[j];

        /* Look up into the Hash Table */
        key = row*Nbs+col+1;
        h1  = HASH(size,key,tmp);

        idx = h1;
        if (PetscDefined(USE_DEBUG)) {
          total_ct++;
          insert_ct++;
          if (HT[idx] != key) {
            for (idx=h1; (idx<size) && (HT[idx]!=key); idx++,total_ct++) ;
            if (idx == size) {
              for (idx=0; (idx<h1) && (HT[idx]!=key); idx++,total_ct++) ;
              PetscCheckFalse(idx == h1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"(%" PetscInt_FMT ",%" PetscInt_FMT ") has no entry in the hash table", row, col);
            }
          }
        } else if (HT[idx] != key) {
          for (idx=h1; (idx<size) && (HT[idx]!=key); idx++) ;
          if (idx == size) {
            for (idx=0; (idx<h1) && (HT[idx]!=key); idx++) ;
            PetscCheckFalse(idx == h1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"(%" PetscInt_FMT ",%" PetscInt_FMT ") has no entry in the hash table", row, col);
          }
        }
        baij_a = HD[idx];
        if (roworiented) {
          /*value = v + i*(stepval+bs)*bs + j*bs;*/
          /* value = v + (i*(stepval+bs)+j)*bs; */
          value = v_t;
          v_t  += bs;
          if (addv == ADD_VALUES) {
            for (ii=0; ii<bs; ii++,value+=stepval) {
              for (jj=ii; jj<bs2; jj+=bs) {
                baij_a[jj] += *value++;
              }
            }
          } else {
            for (ii=0; ii<bs; ii++,value+=stepval) {
              for (jj=ii; jj<bs2; jj+=bs) {
                baij_a[jj] = *value++;
              }
            }
          }
        } else {
          value = v + j*(stepval+bs)*bs + i*bs;
          if (addv == ADD_VALUES) {
            for (ii=0; ii<bs; ii++,value+=stepval,baij_a+=bs) {
              for (jj=0; jj<bs; jj++) {
                baij_a[jj] += *value++;
              }
            }
          } else {
            for (ii=0; ii<bs; ii++,value+=stepval,baij_a+=bs) {
              for (jj=0; jj<bs; jj++) {
                baij_a[jj] = *value++;
              }
            }
          }
        }
      }
    } else {
      if (!baij->donotstash) {
        if (roworiented) {
          ierr = MatStashValuesRowBlocked_Private(&mat->bstash,im[i],n,in,v,m,n,i);CHKERRQ(ierr);
        } else {
          ierr = MatStashValuesColBlocked_Private(&mat->bstash,im[i],n,in,v,m,n,i);CHKERRQ(ierr);
        }
      }
    }
  }
  if (PetscDefined(USE_DEBUG)) {
    baij->ht_total_ct  += total_ct;
    baij->ht_insert_ct += insert_ct;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetValues_MPIBAIJ(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       bs       = mat->rmap->bs,i,j,bsrstart = mat->rmap->rstart,bsrend = mat->rmap->rend;
  PetscInt       bscstart = mat->cmap->rstart,bscend = mat->cmap->rend,row,col,data;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue; /* negative row */
    PetscCheck(idxm[i] < mat->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,idxm[i],mat->rmap->N-1);
    if (idxm[i] >= bsrstart && idxm[i] < bsrend) {
      row = idxm[i] - bsrstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) continue; /* negative column */
        PetscCheck(idxn[j] < mat->cmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,idxn[j],mat->cmap->N-1);
        if (idxn[j] >= bscstart && idxn[j] < bscend) {
          col  = idxn[j] - bscstart;
          ierr = MatGetValues_SeqBAIJ(baij->A,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
        } else {
          if (!baij->colmap) {
            ierr = MatCreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
          }
#if defined(PETSC_USE_CTABLE)
          ierr = PetscTableFind(baij->colmap,idxn[j]/bs+1,&data);CHKERRQ(ierr);
          data--;
#else
          data = baij->colmap[idxn[j]/bs]-1;
#endif
          if ((data < 0) || (baij->garray[data/bs] != idxn[j]/bs)) *(v+i*n+j) = 0.0;
          else {
            col  = data + idxn[j]%bs;
            ierr = MatGetValues_SeqBAIJ(baij->B,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
          }
        }
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local values currently supported");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatNorm_MPIBAIJ(Mat mat,NormType type,PetscReal *nrm)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ    *amat = (Mat_SeqBAIJ*)baij->A->data,*bmat = (Mat_SeqBAIJ*)baij->B->data;
  PetscErrorCode ierr;
  PetscInt       i,j,bs2=baij->bs2,bs=baij->A->rmap->bs,nz,row,col;
  PetscReal      sum = 0.0;
  MatScalar      *v;

  PetscFunctionBegin;
  if (baij->size == 1) {
    ierr =  MatNorm(baij->A,type,nrm);CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      v  = amat->a;
      nz = amat->nz*bs2;
      for (i=0; i<nz; i++) {
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
      }
      v  = bmat->a;
      nz = bmat->nz*bs2;
      for (i=0; i<nz; i++) {
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
      }
      ierr = MPIU_Allreduce(&sum,nrm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
      *nrm = PetscSqrtReal(*nrm);
    } else if (type == NORM_1) { /* max column sum */
      PetscReal *tmp,*tmp2;
      PetscInt  *jj,*garray=baij->garray,cstart=baij->rstartbs;
      ierr = PetscCalloc1(mat->cmap->N,&tmp);CHKERRQ(ierr);
      ierr = PetscMalloc1(mat->cmap->N,&tmp2);CHKERRQ(ierr);
      v    = amat->a; jj = amat->j;
      for (i=0; i<amat->nz; i++) {
        for (j=0; j<bs; j++) {
          col = bs*(cstart + *jj) + j; /* column index */
          for (row=0; row<bs; row++) {
            tmp[col] += PetscAbsScalar(*v);  v++;
          }
        }
        jj++;
      }
      v = bmat->a; jj = bmat->j;
      for (i=0; i<bmat->nz; i++) {
        for (j=0; j<bs; j++) {
          col = bs*garray[*jj] + j;
          for (row=0; row<bs; row++) {
            tmp[col] += PetscAbsScalar(*v); v++;
          }
        }
        jj++;
      }
      ierr = MPIU_Allreduce(tmp,tmp2,mat->cmap->N,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
      *nrm = 0.0;
      for (j=0; j<mat->cmap->N; j++) {
        if (tmp2[j] > *nrm) *nrm = tmp2[j];
      }
      ierr = PetscFree(tmp);CHKERRQ(ierr);
      ierr = PetscFree(tmp2);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) { /* max row sum */
      PetscReal *sums;
      ierr = PetscMalloc1(bs,&sums);CHKERRQ(ierr);
      sum  = 0.0;
      for (j=0; j<amat->mbs; j++) {
        for (row=0; row<bs; row++) sums[row] = 0.0;
        v  = amat->a + bs2*amat->i[j];
        nz = amat->i[j+1]-amat->i[j];
        for (i=0; i<nz; i++) {
          for (col=0; col<bs; col++) {
            for (row=0; row<bs; row++) {
              sums[row] += PetscAbsScalar(*v); v++;
            }
          }
        }
        v  = bmat->a + bs2*bmat->i[j];
        nz = bmat->i[j+1]-bmat->i[j];
        for (i=0; i<nz; i++) {
          for (col=0; col<bs; col++) {
            for (row=0; row<bs; row++) {
              sums[row] += PetscAbsScalar(*v); v++;
            }
          }
        }
        for (row=0; row<bs; row++) {
          if (sums[row] > sum) sum = sums[row];
        }
      }
      ierr = MPIU_Allreduce(&sum,nrm,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
      ierr = PetscFree(sums);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"No support for this norm yet");
  }
  PetscFunctionReturn(0);
}

/*
  Creates the hash table, and sets the table
  This table is created only once.
  If new entried need to be added to the matrix
  then the hash table has to be destroyed and
  recreated.
*/
PetscErrorCode MatCreateHashTable_MPIBAIJ_Private(Mat mat,PetscReal factor)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  Mat            A     = baij->A,B=baij->B;
  Mat_SeqBAIJ    *a    = (Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ*)B->data;
  PetscInt       i,j,k,nz=a->nz+b->nz,h1,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  PetscErrorCode ierr;
  PetscInt       ht_size,bs2=baij->bs2,rstart=baij->rstartbs;
  PetscInt       cstart=baij->cstartbs,*garray=baij->garray,row,col,Nbs=baij->Nbs;
  PetscInt       *HT,key;
  MatScalar      **HD;
  PetscReal      tmp;
#if defined(PETSC_USE_INFO)
  PetscInt ct=0,max=0;
#endif

  PetscFunctionBegin;
  if (baij->ht) PetscFunctionReturn(0);

  baij->ht_size = (PetscInt)(factor*nz);
  ht_size       = baij->ht_size;

  /* Allocate Memory for Hash Table */
  ierr = PetscCalloc2(ht_size,&baij->hd,ht_size,&baij->ht);CHKERRQ(ierr);
  HD   = baij->hd;
  HT   = baij->ht;

  /* Loop Over A */
  for (i=0; i<a->mbs; i++) {
    for (j=ai[i]; j<ai[i+1]; j++) {
      row = i+rstart;
      col = aj[j]+cstart;

      key = row*Nbs + col + 1;
      h1  = HASH(ht_size,key,tmp);
      for (k=0; k<ht_size; k++) {
        if (!HT[(h1+k)%ht_size]) {
          HT[(h1+k)%ht_size] = key;
          HD[(h1+k)%ht_size] = a->a + j*bs2;
          break;
#if defined(PETSC_USE_INFO)
        } else {
          ct++;
#endif
        }
      }
#if defined(PETSC_USE_INFO)
      if (k> max) max = k;
#endif
    }
  }
  /* Loop Over B */
  for (i=0; i<b->mbs; i++) {
    for (j=bi[i]; j<bi[i+1]; j++) {
      row = i+rstart;
      col = garray[bj[j]];
      key = row*Nbs + col + 1;
      h1  = HASH(ht_size,key,tmp);
      for (k=0; k<ht_size; k++) {
        if (!HT[(h1+k)%ht_size]) {
          HT[(h1+k)%ht_size] = key;
          HD[(h1+k)%ht_size] = b->a + j*bs2;
          break;
#if defined(PETSC_USE_INFO)
        } else {
          ct++;
#endif
        }
      }
#if defined(PETSC_USE_INFO)
      if (k> max) max = k;
#endif
    }
  }

  /* Print Summary */
#if defined(PETSC_USE_INFO)
  for (i=0,j=0; i<ht_size; i++) {
    if (HT[i]) j++;
  }
  ierr = PetscInfo(mat,"Average Search = %5.2g,max search = %" PetscInt_FMT "\n",(!j) ? (double)0.0:(double)(((PetscReal)(ct+j))/(double)j),max);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyBegin_MPIBAIJ(Mat mat,MatAssemblyType mode)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nstash,reallocs;

  PetscFunctionBegin;
  if (baij->donotstash || mat->nooffprocentries) PetscFunctionReturn(0);

  ierr = MatStashScatterBegin_Private(mat,&mat->stash,mat->rmap->range);CHKERRQ(ierr);
  ierr = MatStashScatterBegin_Private(mat,&mat->bstash,baij->rangebs);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo(mat,"Stash has %" PetscInt_FMT " entries,uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->bstash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo(mat,"Block-Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIBAIJ(Mat mat,MatAssemblyType mode)
{
  Mat_MPIBAIJ    *baij=(Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ    *a   =(Mat_SeqBAIJ*)baij->A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart,ncols,flg,bs2=baij->bs2;
  PetscInt       *row,*col;
  PetscBool      r1,r2,r3,other_disassembled;
  MatScalar      *val;
  PetscMPIInt    n;

  PetscFunctionBegin;
  /* do not use 'b=(Mat_SeqBAIJ*)baij->B->data' as B can be reset in disassembly */
  if (!baij->donotstash && !mat->nooffprocentries) {
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) {
          if (row[j] != rstart) break;
        }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        /* Now assemble all these values with a single function call */
        ierr = MatSetValues_MPIBAIJ(mat,1,row+i,ncols,col+i,val+i,mat->insertmode);CHKERRQ(ierr);
        i    = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);
    /* Now process the block-stash. Since the values are stashed column-oriented,
       set the roworiented flag to column oriented, and after MatSetValues()
       restore the original flags */
    r1 = baij->roworiented;
    r2 = a->roworiented;
    r3 = ((Mat_SeqBAIJ*)baij->B->data)->roworiented;

    baij->roworiented = PETSC_FALSE;
    a->roworiented    = PETSC_FALSE;

    (((Mat_SeqBAIJ*)baij->B->data))->roworiented = PETSC_FALSE; /* b->roworiented */
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->bstash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) {
          if (row[j] != rstart) break;
        }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        ierr = MatSetValuesBlocked_MPIBAIJ(mat,1,row+i,ncols,col+i,val+i*bs2,mat->insertmode);CHKERRQ(ierr);
        i    = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->bstash);CHKERRQ(ierr);

    baij->roworiented = r1;
    a->roworiented    = r2;

    ((Mat_SeqBAIJ*)baij->B->data)->roworiented = r3; /* b->roworiented */
  }

  ierr = MatAssemblyBegin(baij->A,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->A,mode);CHKERRQ(ierr);

  /* determine if any processor has disassembled, if so we must
     also disassemble ourselfs, in order that we may reassemble. */
  /*
     if nonzero structure of submatrix B cannot change then we know that
     no processor disassembled thus we can skip this stuff
  */
  if (!((Mat_SeqBAIJ*)baij->B->data)->nonew) {
    ierr = MPIU_Allreduce(&mat->was_assembled,&other_disassembled,1,MPIU_BOOL,MPI_PROD,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
    if (mat->was_assembled && !other_disassembled) {
      ierr = MatDisAssemble_MPIBAIJ(mat);CHKERRQ(ierr);
    }
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIBAIJ(mat);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(baij->B,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->B,mode);CHKERRQ(ierr);

#if defined(PETSC_USE_INFO)
  if (baij->ht && mode== MAT_FINAL_ASSEMBLY) {
    ierr = PetscInfo(mat,"Average Hash Table Search in MatSetValues = %5.2f\n",(double)((PetscReal)baij->ht_total_ct)/baij->ht_insert_ct);CHKERRQ(ierr);

    baij->ht_total_ct  = 0;
    baij->ht_insert_ct = 0;
  }
#endif
  if (baij->ht_flag && !baij->ht && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatCreateHashTable_MPIBAIJ_Private(mat,baij->ht_fact);CHKERRQ(ierr);

    mat->ops->setvalues        = MatSetValues_MPIBAIJ_HT;
    mat->ops->setvaluesblocked = MatSetValuesBlocked_MPIBAIJ_HT;
  }

  ierr = PetscFree2(baij->rowvalues,baij->rowindices);CHKERRQ(ierr);

  baij->rowvalues = NULL;

  /* if no new nonzero locations are allowed in matrix then only set the matrix state the first time through */
  if ((!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) || !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
    PetscObjectState state = baij->A->nonzerostate + baij->B->nonzerostate;
    ierr = MPIU_Allreduce(&state,&mat->nonzerostate,1,MPIU_INT64,MPI_SUM,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatView_SeqBAIJ(Mat,PetscViewer);
#include <petscdraw.h>
static PetscErrorCode MatView_MPIBAIJ_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPIBAIJ       *baij = (Mat_MPIBAIJ*)mat->data;
  PetscErrorCode    ierr;
  PetscMPIInt       rank = baij->rank;
  PetscInt          bs   = mat->rmap->bs;
  PetscBool         iascii,isdraw;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo info;
      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&rank);CHKERRMPI(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %" PetscInt_FMT " nz %" PetscInt_FMT " nz alloced %" PetscInt_FMT " bs %" PetscInt_FMT " mem %g\n",
                                                rank,mat->rmap->n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,mat->rmap->bs,(double)info.memory);CHKERRQ(ierr);
      ierr = MatGetInfo(baij->A,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] on-diagonal part: nz %" PetscInt_FMT " \n",rank,(PetscInt)info.nz_used);CHKERRQ(ierr);
      ierr = MatGetInfo(baij->B,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] off-diagonal part: nz %" PetscInt_FMT " \n",rank,(PetscInt)info.nz_used);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Information on VecScatter used in matrix-vector product: \n");CHKERRQ(ierr);
      ierr = VecScatterView(baij->Mvctx,viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"  block size is %" PetscInt_FMT "\n",bs);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      PetscFunctionReturn(0);
    }
  }

  if (isdraw) {
    PetscDraw draw;
    PetscBool isnull;
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
    if (isnull) PetscFunctionReturn(0);
  }

  {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    Mat_SeqBAIJ *Aloc;
    PetscInt    M = mat->rmap->N,N = mat->cmap->N,*ai,*aj,col,i,j,k,*rvals,mbs = baij->mbs;
    MatScalar   *a;
    const char  *matname;

    /* Here we are creating a temporary matrix, so will assume MPIBAIJ is acceptable */
    /* Perhaps this should be the type of mat? */
    ierr = MatCreate(PetscObjectComm((PetscObject)mat),&A);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = MatSetSizes(A,M,N,M,N);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,M,N);CHKERRQ(ierr);
    }
    ierr = MatSetType(A,MATMPIBAIJ);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(A,mat->rmap->bs,0,NULL,0,NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)A);CHKERRQ(ierr);

    /* copy over the A part */
    Aloc = (Mat_SeqBAIJ*)baij->A->data;
    ai   = Aloc->i; aj = Aloc->j; a = Aloc->a;
    ierr = PetscMalloc1(bs,&rvals);CHKERRQ(ierr);

    for (i=0; i<mbs; i++) {
      rvals[0] = bs*(baij->rstartbs + i);
      for (j=1; j<bs; j++) rvals[j] = rvals[j-1] + 1;
      for (j=ai[i]; j<ai[i+1]; j++) {
        col = (baij->cstartbs+aj[j])*bs;
        for (k=0; k<bs; k++) {
          ierr      = MatSetValues_MPIBAIJ(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
          col++; a += bs;
        }
      }
    }
    /* copy over the B part */
    Aloc = (Mat_SeqBAIJ*)baij->B->data;
    ai   = Aloc->i; aj = Aloc->j; a = Aloc->a;
    for (i=0; i<mbs; i++) {
      rvals[0] = bs*(baij->rstartbs + i);
      for (j=1; j<bs; j++) rvals[j] = rvals[j-1] + 1;
      for (j=ai[i]; j<ai[i+1]; j++) {
        col = baij->garray[aj[j]]*bs;
        for (k=0; k<bs; k++) {
          ierr      = MatSetValues_MPIBAIJ(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
          col++; a += bs;
        }
      }
    }
    ierr = PetscFree(rvals);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /*
       Everyone has to call to draw the matrix since the graphics waits are
       synchronized across all processors that share the PetscDraw object
    */
    ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject)mat,&matname);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = PetscObjectSetName((PetscObject)((Mat_MPIBAIJ*)(A->data))->A,matname);CHKERRQ(ierr);
      ierr = MatView_SeqBAIJ(((Mat_MPIBAIJ*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Used for both MPIBAIJ and MPISBAIJ matrices */
PetscErrorCode MatView_MPIBAIJ_Binary(Mat mat,PetscViewer viewer)
{
  Mat_MPIBAIJ    *aij = (Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ    *A   = (Mat_SeqBAIJ*)aij->A->data;
  Mat_SeqBAIJ    *B   = (Mat_SeqBAIJ*)aij->B->data;
  const PetscInt *garray = aij->garray;
  PetscInt       header[4],M,N,m,rs,cs,bs,nz,cnt,i,j,ja,jb,k,l;
  PetscInt       *rowlens,*colidxs;
  PetscScalar    *matvals;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);

  M  = mat->rmap->N;
  N  = mat->cmap->N;
  m  = mat->rmap->n;
  rs = mat->rmap->rstart;
  cs = mat->cmap->rstart;
  bs = mat->rmap->bs;
  nz = bs*bs*(A->nz + B->nz);

  /* write matrix header */
  header[0] = MAT_FILE_CLASSID;
  header[1] = M; header[2] = N; header[3] = nz;
  ierr = MPI_Reduce(&nz,&header[3],1,MPIU_INT,MPI_SUM,0,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
  ierr = PetscViewerBinaryWrite(viewer,header,4,PETSC_INT);CHKERRQ(ierr);

  /* fill in and store row lengths */
  ierr = PetscMalloc1(m,&rowlens);CHKERRQ(ierr);
  for (cnt=0, i=0; i<A->mbs; i++)
    for (j=0; j<bs; j++)
      rowlens[cnt++] = bs*(A->i[i+1] - A->i[i] + B->i[i+1] - B->i[i]);
  ierr = PetscViewerBinaryWriteAll(viewer,rowlens,m,rs,M,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscFree(rowlens);CHKERRQ(ierr);

  /* fill in and store column indices */
  ierr = PetscMalloc1(nz,&colidxs);CHKERRQ(ierr);
  for (cnt=0, i=0; i<A->mbs; i++) {
    for (k=0; k<bs; k++) {
      for (jb=B->i[i]; jb<B->i[i+1]; jb++) {
        if (garray[B->j[jb]] > cs/bs) break;
        for (l=0; l<bs; l++)
          colidxs[cnt++] = bs*garray[B->j[jb]] + l;
      }
      for (ja=A->i[i]; ja<A->i[i+1]; ja++)
        for (l=0; l<bs; l++)
          colidxs[cnt++] = bs*A->j[ja] + l + cs;
      for (; jb<B->i[i+1]; jb++)
        for (l=0; l<bs; l++)
          colidxs[cnt++] = bs*garray[B->j[jb]] + l;
    }
  }
  PetscCheckFalse(cnt != nz,PETSC_COMM_SELF,PETSC_ERR_LIB,"Internal PETSc error: cnt = %" PetscInt_FMT " nz = %" PetscInt_FMT,cnt,nz);
  ierr = PetscViewerBinaryWriteAll(viewer,colidxs,nz,PETSC_DECIDE,PETSC_DECIDE,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscFree(colidxs);CHKERRQ(ierr);

  /* fill in and store nonzero values */
  ierr = PetscMalloc1(nz,&matvals);CHKERRQ(ierr);
  for (cnt=0, i=0; i<A->mbs; i++) {
    for (k=0; k<bs; k++) {
      for (jb=B->i[i]; jb<B->i[i+1]; jb++) {
        if (garray[B->j[jb]] > cs/bs) break;
        for (l=0; l<bs; l++)
          matvals[cnt++] = B->a[bs*(bs*jb + l) + k];
      }
      for (ja=A->i[i]; ja<A->i[i+1]; ja++)
        for (l=0; l<bs; l++)
          matvals[cnt++] = A->a[bs*(bs*ja + l) + k];
      for (; jb<B->i[i+1]; jb++)
        for (l=0; l<bs; l++)
          matvals[cnt++] = B->a[bs*(bs*jb + l) + k];
    }
  }
  ierr = PetscViewerBinaryWriteAll(viewer,matvals,nz,PETSC_DECIDE,PETSC_DECIDE,PETSC_SCALAR);CHKERRQ(ierr);
  ierr = PetscFree(matvals);CHKERRQ(ierr);

  /* write block size option to the viewer's .info file */
  ierr = MatView_Binary_BlockSizes(mat,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_MPIBAIJ(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isdraw,issocket,isbinary;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii || isdraw || issocket) {
    ierr = MatView_MPIBAIJ_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_MPIBAIJ_Binary(mat,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%" PetscInt_FMT ",Cols=%" PetscInt_FMT,mat->rmap->N,mat->cmap->N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = MatStashDestroy_Private(&mat->bstash);CHKERRQ(ierr);
  ierr = MatDestroy(&baij->A);CHKERRQ(ierr);
  ierr = MatDestroy(&baij->B);CHKERRQ(ierr);
#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableDestroy(&baij->colmap);CHKERRQ(ierr);
#else
  ierr = PetscFree(baij->colmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(baij->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&baij->Mvctx);CHKERRQ(ierr);
  ierr = PetscFree2(baij->rowvalues,baij->rowindices);CHKERRQ(ierr);
  ierr = PetscFree(baij->barray);CHKERRQ(ierr);
  ierr = PetscFree2(baij->hd,baij->ht);CHKERRQ(ierr);
  ierr = PetscFree(baij->rangebs);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)mat,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatStoreValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatRetrieveValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIBAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIBAIJSetPreallocationCSR_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDiagonalScaleLocal_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatSetHashTableFactor_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpibaij_mpisbaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpibaij_mpibstrm_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpibaij_hypre_C",NULL);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpibaij_is_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  PetscCheckFalse(nt != A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A and xx");
  ierr = VecGetLocalSize(yy,&nt);CHKERRQ(ierr);
  PetscCheckFalse(nt != A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible parition of A and yy");
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
  /* add partial results together */
  ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_MPIBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtransposeadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  /* add partial results together */
  ierr = VecScatterBegin(a->Mvctx,a->lvec,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This only works correctly for square matrices where the subblock A->A is the
   diagonal block
*/
PetscErrorCode MatGetDiagonal_MPIBAIJ(Mat A,Vec v)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(A->rmap->N != A->cmap->N,PETSC_COMM_SELF,PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block");
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_MPIBAIJ(Mat A,PetscScalar aa)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(a->A,aa);CHKERRQ(ierr);
  ierr = MatScale(a->B,aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_MPIBAIJ(Mat matin,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIBAIJ    *mat = (Mat_MPIBAIJ*)matin->data;
  PetscScalar    *vworkA,*vworkB,**pvA,**pvB,*v_p;
  PetscErrorCode ierr;
  PetscInt       bs = matin->rmap->bs,bs2 = mat->bs2,i,*cworkA,*cworkB,**pcA,**pcB;
  PetscInt       nztot,nzA,nzB,lrow,brstart = matin->rmap->rstart,brend = matin->rmap->rend;
  PetscInt       *cmap,*idx_p,cstart = mat->cstartbs;

  PetscFunctionBegin;
  PetscCheckFalse(row < brstart || row >= brend,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local rows");
  PetscCheckFalse(mat->getrowactive,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqBAIJ *Aa = (Mat_SeqBAIJ*)mat->A->data,*Ba = (Mat_SeqBAIJ*)mat->B->data;
    PetscInt    max = 1,mbs = mat->mbs,tmp;
    for (i=0; i<mbs; i++) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i];
      if (max < tmp) max = tmp;
    }
    ierr = PetscMalloc2(max*bs2,&mat->rowvalues,max*bs2,&mat->rowindices);CHKERRQ(ierr);
  }
  lrow = row - brstart;

  pvA = &vworkA; pcA = &cworkA; pvB = &vworkB; pcB = &cworkB;
  if (!v)   {pvA = NULL; pvB = NULL;}
  if (!idx) {pcA = NULL; if (!v) pcB = NULL;}
  ierr  = (*mat->A->ops->getrow)(mat->A,lrow,&nzA,pcA,pvA);CHKERRQ(ierr);
  ierr  = (*mat->B->ops->getrow)(mat->B,lrow,&nzB,pcB,pvB);CHKERRQ(ierr);
  nztot = nzA + nzB;

  cmap = mat->garray;
  if (v  || idx) {
    if (nztot) {
      /* Sort by increasing column numbers, assuming A and B already sorted */
      PetscInt imark = -1;
      if (v) {
        *v = v_p = mat->rowvalues;
        for (i=0; i<nzB; i++) {
          if (cmap[cworkB[i]/bs] < cstart) v_p[i] = vworkB[i];
          else break;
        }
        imark = i;
        for (i=0; i<nzA; i++)     v_p[imark+i] = vworkA[i];
        for (i=imark; i<nzB; i++) v_p[nzA+i]   = vworkB[i];
      }
      if (idx) {
        *idx = idx_p = mat->rowindices;
        if (imark > -1) {
          for (i=0; i<imark; i++) {
            idx_p[i] = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs;
          }
        } else {
          for (i=0; i<nzB; i++) {
            if (cmap[cworkB[i]/bs] < cstart) idx_p[i] = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs;
            else break;
          }
          imark = i;
        }
        for (i=0; i<nzA; i++)     idx_p[imark+i] = cstart*bs + cworkA[i];
        for (i=imark; i<nzB; i++) idx_p[nzA+i]   = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs ;
      }
    } else {
      if (idx) *idx = NULL;
      if (v)   *v   = NULL;
    }
  }
  *nz  = nztot;
  ierr = (*mat->A->ops->restorerow)(mat->A,lrow,&nzA,pcA,pvA);CHKERRQ(ierr);
  ierr = (*mat->B->ops->restorerow)(mat->B,lrow,&nzB,pcB,pvB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_MPIBAIJ(Mat mat,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ*)mat->data;

  PetscFunctionBegin;
  PetscCheckFalse(!baij->getrowactive,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"MatGetRow not called");
  baij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ    *l = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_MPIBAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)matin->data;
  Mat            A  = a->A,B = a->B;
  PetscErrorCode ierr;
  PetscLogDouble isend[5],irecv[5];

  PetscFunctionBegin;
  info->block_size = (PetscReal)matin->rmap->bs;

  ierr = MatGetInfo(A,MAT_LOCAL,info);CHKERRQ(ierr);

  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;

  ierr = MatGetInfo(B,MAT_LOCAL,info);CHKERRQ(ierr);

  isend[0] += info->nz_used; isend[1] += info->nz_allocated; isend[2] += info->nz_unneeded;
  isend[3] += info->memory;  isend[4] += info->mallocs;

  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    ierr = MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_MAX,PetscObjectComm((PetscObject)matin));CHKERRMPI(ierr);

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_SUM,PetscObjectComm((PetscObject)matin));CHKERRMPI(ierr);

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else SETERRQ(PetscObjectComm((PetscObject)matin),PETSC_ERR_ARG_WRONG,"Unknown MatInfoType argument %d",(int)flag);
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_MPIBAIJ(Mat A,MatOption op,PetscBool flg)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_NEW_NONZERO_LOCATION_ERR:
    MatCheckPreallocated(A,1);
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op,flg);CHKERRQ(ierr);
    break;
  case MAT_ROW_ORIENTED:
    MatCheckPreallocated(A,1);
    a->roworiented = flg;

    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op,flg);CHKERRQ(ierr);
    break;
  case MAT_FORCE_DIAGONAL_ENTRIES:
  case MAT_SORTED_FULL:
    ierr = PetscInfo(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = flg;
    break;
  case MAT_USE_HASH_TABLE:
    a->ht_flag = flg;
    a->ht_fact = 1.39;
    break;
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SUBMAT_SINGLEIS:
  case MAT_SYMMETRY_ETERNAL:
    MatCheckPreallocated(A,1);
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatTranspose_MPIBAIJ(Mat A,MatReuse reuse,Mat *matout)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)A->data;
  Mat_SeqBAIJ    *Aloc;
  Mat            B;
  PetscErrorCode ierr;
  PetscInt       M =A->rmap->N,N=A->cmap->N,*ai,*aj,i,*rvals,j,k,col;
  PetscInt       bs=A->rmap->bs,mbs=baij->mbs;
  MatScalar      *a;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->cmap->n,A->rmap->n,N,M);CHKERRQ(ierr);
    ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
    /* Do not know preallocation information, but must set block size */
    ierr = MatMPIBAIJSetPreallocation(B,A->rmap->bs,PETSC_DECIDE,NULL,PETSC_DECIDE,NULL);CHKERRQ(ierr);
  } else {
    B = *matout;
  }

  /* copy over the A part */
  Aloc = (Mat_SeqBAIJ*)baij->A->data;
  ai   = Aloc->i; aj = Aloc->j; a = Aloc->a;
  ierr = PetscMalloc1(bs,&rvals);CHKERRQ(ierr);

  for (i=0; i<mbs; i++) {
    rvals[0] = bs*(baij->rstartbs + i);
    for (j=1; j<bs; j++) rvals[j] = rvals[j-1] + 1;
    for (j=ai[i]; j<ai[i+1]; j++) {
      col = (baij->cstartbs+aj[j])*bs;
      for (k=0; k<bs; k++) {
        ierr = MatSetValues_MPIBAIJ(B,1,&col,bs,rvals,a,INSERT_VALUES);CHKERRQ(ierr);

        col++; a += bs;
      }
    }
  }
  /* copy over the B part */
  Aloc = (Mat_SeqBAIJ*)baij->B->data;
  ai   = Aloc->i; aj = Aloc->j; a = Aloc->a;
  for (i=0; i<mbs; i++) {
    rvals[0] = bs*(baij->rstartbs + i);
    for (j=1; j<bs; j++) rvals[j] = rvals[j-1] + 1;
    for (j=ai[i]; j<ai[i+1]; j++) {
      col = baij->garray[aj[j]]*bs;
      for (k=0; k<bs; k++) {
        ierr = MatSetValues_MPIBAIJ(B,1,&col,bs,rvals,a,INSERT_VALUES);CHKERRQ(ierr);
        col++;
        a += bs;
      }
    }
  }
  ierr = PetscFree(rvals);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) *matout = B;
  else {
    ierr = MatHeaderMerge(A,&B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_MPIBAIJ(Mat mat,Vec ll,Vec rr)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  Mat            a     = baij->A,b = baij->B;
  PetscErrorCode ierr;
  PetscInt       s1,s2,s3;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(mat,&s2,&s3);CHKERRQ(ierr);
  if (rr) {
    ierr = VecGetLocalSize(rr,&s1);CHKERRQ(ierr);
    PetscCheckFalse(s1!=s3,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"right vector non-conforming local size");
    /* Overlap communication with computation. */
    ierr = VecScatterBegin(baij->Mvctx,rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (ll) {
    ierr = VecGetLocalSize(ll,&s1);CHKERRQ(ierr);
    PetscCheckFalse(s1!=s2,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"left vector non-conforming local size");
    ierr = (*b->ops->diagonalscale)(b,ll,NULL);CHKERRQ(ierr);
  }
  /* scale  the diagonal block */
  ierr = (*a->ops->diagonalscale)(a,ll,rr);CHKERRQ(ierr);

  if (rr) {
    /* Do a scatter end and then right scale the off-diagonal block */
    ierr = VecScatterEnd(baij->Mvctx,rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = (*b->ops->diagonalscale)(b,NULL,baij->lvec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRows_MPIBAIJ(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_MPIBAIJ   *l      = (Mat_MPIBAIJ *) A->data;
  PetscInt      *lrows;
  PetscInt       r, len;
  PetscBool      cong;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get locally owned rows */
  ierr = MatZeroRowsMapLocal_Private(A,N,rows,&len,&lrows);CHKERRQ(ierr);
  /* fix right hand side if needed */
  if (x && b) {
    const PetscScalar *xx;
    PetscScalar       *bb;

    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    for (r = 0; r < len; ++r) bb[lrows[r]] = diag*xx[lrows[r]];
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }

  /* actually zap the local rows */
  /*
        Zero the required rows. If the "diagonal block" of the matrix
     is square and the user wishes to set the diagonal we use separate
     code so that MatSetValues() is not called for each diagonal allocating
     new memory, thus calling lots of mallocs and slowing things down.

  */
  /* must zero l->B before l->A because the (diag) case below may put values into l->B*/
  ierr = MatZeroRows_SeqBAIJ(l->B,len,lrows,0.0,NULL,NULL);CHKERRQ(ierr);
  ierr = MatHasCongruentLayouts(A,&cong);CHKERRQ(ierr);
  if ((diag != 0.0) && cong) {
    ierr = MatZeroRows_SeqBAIJ(l->A,len,lrows,diag,NULL,NULL);CHKERRQ(ierr);
  } else if (diag != 0.0) {
    ierr = MatZeroRows_SeqBAIJ(l->A,len,lrows,0.0,NULL,NULL);CHKERRQ(ierr);
    PetscCheckFalse(((Mat_SeqBAIJ*)l->A->data)->nonew,PETSC_COMM_SELF,PETSC_ERR_SUP,"MatZeroRows() on rectangular matrices cannot be used with the Mat options \n\
       MAT_NEW_NONZERO_LOCATIONS,MAT_NEW_NONZERO_LOCATION_ERR,MAT_NEW_NONZERO_ALLOCATION_ERR");
    for (r = 0; r < len; ++r) {
      const PetscInt row = lrows[r] + A->rmap->rstart;
      ierr = MatSetValues(A,1,&row,1,&row,&diag,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    ierr = MatZeroRows_SeqBAIJ(l->A,len,lrows,0.0,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = PetscFree(lrows);CHKERRQ(ierr);

  /* only change matrix nonzero state if pattern was allowed to be changed */
  if (!((Mat_SeqBAIJ*)(l->A->data))->keepnonzeropattern) {
    PetscObjectState state = l->A->nonzerostate + l->B->nonzerostate;
    ierr = MPIU_Allreduce(&state,&A->nonzerostate,1,MPIU_INT64,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRowsColumns_MPIBAIJ(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_MPIBAIJ       *l = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscMPIInt       n = A->rmap->n,p = 0;
  PetscInt          i,j,k,r,len = 0,row,col,count;
  PetscInt          *lrows,*owners = A->rmap->range;
  PetscSFNode       *rrows;
  PetscSF           sf;
  const PetscScalar *xx;
  PetscScalar       *bb,*mask;
  Vec               xmask,lmask;
  Mat_SeqBAIJ       *baij = (Mat_SeqBAIJ*)l->B->data;
  PetscInt           bs = A->rmap->bs, bs2 = baij->bs2;
  PetscScalar       *aa;

  PetscFunctionBegin;
  /* Create SF where leaves are input rows and roots are owned rows */
  ierr = PetscMalloc1(n, &lrows);CHKERRQ(ierr);
  for (r = 0; r < n; ++r) lrows[r] = -1;
  ierr = PetscMalloc1(N, &rrows);CHKERRQ(ierr);
  for (r = 0; r < N; ++r) {
    const PetscInt idx   = rows[r];
    PetscCheckFalse(idx < 0 || A->rmap->N <= idx,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " out of range [0,%" PetscInt_FMT ")",idx,A->rmap->N);
    if (idx < owners[p] || owners[p+1] <= idx) { /* short-circuit the search if the last p owns this row too */
      ierr = PetscLayoutFindOwner(A->rmap,idx,&p);CHKERRQ(ierr);
    }
    rrows[r].rank  = p;
    rrows[r].index = rows[r] - owners[p];
  }
  ierr = PetscSFCreate(PetscObjectComm((PetscObject) A), &sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf, n, N, NULL, PETSC_OWN_POINTER, rrows, PETSC_OWN_POINTER);CHKERRQ(ierr);
  /* Collect flags for rows to be zeroed */
  ierr = PetscSFReduceBegin(sf, MPIU_INT, (PetscInt *) rows, lrows, MPI_LOR);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf, MPIU_INT, (PetscInt *) rows, lrows, MPI_LOR);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  /* Compress and put in row numbers */
  for (r = 0; r < n; ++r) if (lrows[r] >= 0) lrows[len++] = r;
  /* zero diagonal part of matrix */
  ierr = MatZeroRowsColumns(l->A,len,lrows,diag,x,b);CHKERRQ(ierr);
  /* handle off diagonal part of matrix */
  ierr = MatCreateVecs(A,&xmask,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(l->lvec,&lmask);CHKERRQ(ierr);
  ierr = VecGetArray(xmask,&bb);CHKERRQ(ierr);
  for (i=0; i<len; i++) bb[lrows[i]] = 1;
  ierr = VecRestoreArray(xmask,&bb);CHKERRQ(ierr);
  ierr = VecScatterBegin(l->Mvctx,xmask,lmask,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(l->Mvctx,xmask,lmask,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecDestroy(&xmask);CHKERRQ(ierr);
  if (x) {
    ierr = VecScatterBegin(l->Mvctx,x,l->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(l->Mvctx,x,l->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArrayRead(l->lvec,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
  }
  ierr = VecGetArray(lmask,&mask);CHKERRQ(ierr);
  /* remove zeroed rows of off diagonal matrix */
  for (i = 0; i < len; ++i) {
    row   = lrows[i];
    count = (baij->i[row/bs +1] - baij->i[row/bs])*bs;
    aa    = ((MatScalar*)(baij->a)) + baij->i[row/bs]*bs2 + (row%bs);
    for (k = 0; k < count; ++k) {
      aa[0] = 0.0;
      aa   += bs;
    }
  }
  /* loop over all elements of off process part of matrix zeroing removed columns*/
  for (i = 0; i < l->B->rmap->N; ++i) {
    row = i/bs;
    for (j = baij->i[row]; j < baij->i[row+1]; ++j) {
      for (k = 0; k < bs; ++k) {
        col = bs*baij->j[j] + k;
        if (PetscAbsScalar(mask[col])) {
          aa = ((MatScalar*)(baij->a)) + j*bs2 + (i%bs) + bs*k;
          if (x) bb[i] -= aa[0]*xx[col];
          aa[0] = 0.0;
        }
      }
    }
  }
  if (x) {
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(l->lvec,&xx);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(lmask,&mask);CHKERRQ(ierr);
  ierr = VecDestroy(&lmask);CHKERRQ(ierr);
  ierr = PetscFree(lrows);CHKERRQ(ierr);

  /* only change matrix nonzero state if pattern was allowed to be changed */
  if (!((Mat_SeqBAIJ*)(l->A->data))->keepnonzeropattern) {
    PetscObjectState state = l->A->nonzerostate + l->B->nonzerostate;
    ierr = MPIU_Allreduce(&state,&A->nonzerostate,1,MPIU_INT64,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUnfactored_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPIBAIJ(Mat,MatDuplicateOption,Mat*);

PetscErrorCode MatEqual_MPIBAIJ(Mat A,Mat B,PetscBool  *flag)
{
  Mat_MPIBAIJ    *matB = (Mat_MPIBAIJ*)B->data,*matA = (Mat_MPIBAIJ*)A->data;
  Mat            a,b,c,d;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  a = matA->A; b = matA->B;
  c = matB->A; d = matB->B;

  ierr = MatEqual(a,c,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatEqual(b,d,&flg);CHKERRQ(ierr);
  }
  ierr = MPIU_Allreduce(&flg,flag,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_MPIBAIJ(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  Mat_MPIBAIJ    *b = (Mat_MPIBAIJ*)B->data;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if ((str != SAME_NONZERO_PATTERN) || (A->ops->copy != B->ops->copy)) {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(a->A,b->A,str);CHKERRQ(ierr);
    ierr = MatCopy(a->B,b->B,str);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_MPIBAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMPIBAIJSetPreallocation(A,A->rmap->bs,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPYGetPreallocation_MPIBAIJ(Mat Y,const PetscInt *yltog,Mat X,const PetscInt *xltog,PetscInt *nnz)
{
  PetscErrorCode ierr;
  PetscInt       bs = Y->rmap->bs,m = Y->rmap->N/bs;
  Mat_SeqBAIJ    *x = (Mat_SeqBAIJ*)X->data;
  Mat_SeqBAIJ    *y = (Mat_SeqBAIJ*)Y->data;

  PetscFunctionBegin;
  ierr = MatAXPYGetPreallocation_MPIX_private(m,x->i,x->j,xltog,y->i,y->j,yltog,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_MPIBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPIBAIJ    *xx=(Mat_MPIBAIJ*)X->data,*yy=(Mat_MPIBAIJ*)Y->data;
  PetscBLASInt   bnz,one=1;
  Mat_SeqBAIJ    *x,*y;
  PetscInt       bs2 = Y->rmap->bs*Y->rmap->bs;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {
    PetscScalar alpha = a;
    x    = (Mat_SeqBAIJ*)xx->A->data;
    y    = (Mat_SeqBAIJ*)yy->A->data;
    ierr = PetscBLASIntCast(x->nz*bs2,&bnz);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one));
    x    = (Mat_SeqBAIJ*)xx->B->data;
    y    = (Mat_SeqBAIJ*)yy->B->data;
    ierr = PetscBLASIntCast(x->nz*bs2,&bnz);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one));
    ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
  } else {
    Mat      B;
    PetscInt *nnz_d,*nnz_o,bs=Y->rmap->bs;
    ierr = PetscMalloc1(yy->A->rmap->N,&nnz_d);CHKERRQ(ierr);
    ierr = PetscMalloc1(yy->B->rmap->N,&nnz_o);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)Y),&B);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)B,((PetscObject)Y)->name);CHKERRQ(ierr);
    ierr = MatSetSizes(B,Y->rmap->n,Y->cmap->n,Y->rmap->N,Y->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(B,Y,Y);CHKERRQ(ierr);
    ierr = MatSetType(B,MATMPIBAIJ);CHKERRQ(ierr);
    ierr = MatAXPYGetPreallocation_SeqBAIJ(yy->A,xx->A,nnz_d);CHKERRQ(ierr);
    ierr = MatAXPYGetPreallocation_MPIBAIJ(yy->B,yy->garray,xx->B,xx->garray,nnz_o);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(B,bs,0,nnz_d,0,nnz_o);CHKERRQ(ierr);
    /* MatAXPY_BasicWithPreallocation() for BAIJ matrix is much slower than AIJ, even for bs=1 ! */
    ierr = MatAXPY_BasicWithPreallocation(B,Y,a,X,str);CHKERRQ(ierr);
    ierr = MatHeaderMerge(Y,&B);CHKERRQ(ierr);
    ierr = PetscFree(nnz_d);CHKERRQ(ierr);
    ierr = PetscFree(nnz_o);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConjugate_MPIBAIJ(Mat mat)
{
#if defined(PETSC_USE_COMPLEX)
  PetscErrorCode ierr;
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)mat->data;

  PetscFunctionBegin;
  ierr = MatConjugate_SeqBAIJ(a->A);CHKERRQ(ierr);
  ierr = MatConjugate_SeqBAIJ(a->B);CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatRealPart_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRealPart(a->A);CHKERRQ(ierr);
  ierr = MatRealPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatImaginaryPart(a->A);CHKERRQ(ierr);
  ierr = MatImaginaryPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrix_MPIBAIJ(Mat mat,IS isrow,IS iscol,MatReuse call,Mat *newmat)
{
  PetscErrorCode ierr;
  IS             iscol_local;
  PetscInt       csize;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(iscol,&csize);CHKERRQ(ierr);
  if (call == MAT_REUSE_MATRIX) {
    ierr = PetscObjectQuery((PetscObject)*newmat,"ISAllGather",(PetscObject*)&iscol_local);CHKERRQ(ierr);
    PetscCheckFalse(!iscol_local,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Submatrix passed in was not used before, cannot reuse");
  } else {
    ierr = ISAllGather(iscol,&iscol_local);CHKERRQ(ierr);
  }
  ierr = MatCreateSubMatrix_MPIBAIJ_Private(mat,isrow,iscol_local,csize,call,newmat);CHKERRQ(ierr);
  if (call == MAT_INITIAL_MATRIX) {
    ierr = PetscObjectCompose((PetscObject)*newmat,"ISAllGather",(PetscObject)iscol_local);CHKERRQ(ierr);
    ierr = ISDestroy(&iscol_local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Not great since it makes two copies of the submatrix, first an SeqBAIJ
  in local and then by concatenating the local matrices the end result.
  Writing it directly would be much like MatCreateSubMatrices_MPIBAIJ().
  This routine is used for BAIJ and SBAIJ matrices (unfortunate dependency).
*/
PetscErrorCode MatCreateSubMatrix_MPIBAIJ_Private(Mat mat,IS isrow,IS iscol,PetscInt csize,MatReuse call,Mat *newmat)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       i,m,n,rstart,row,rend,nz,*cwork,j,bs;
  PetscInt       *ii,*jj,nlocal,*dlens,*olens,dlen,olen,jend,mglobal;
  Mat            M,Mreuse;
  MatScalar      *vwork,*aa;
  MPI_Comm       comm;
  IS             isrow_new, iscol_new;
  Mat_SeqBAIJ    *aij;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  /* The compression and expansion should be avoided. Doesn't point
     out errors, might change the indices, hence buggey */
  ierr = ISCompressIndicesGeneral(mat->rmap->N,mat->rmap->n,mat->rmap->bs,1,&isrow,&isrow_new);CHKERRQ(ierr);
  ierr = ISCompressIndicesGeneral(mat->cmap->N,mat->cmap->n,mat->cmap->bs,1,&iscol,&iscol_new);CHKERRQ(ierr);

  if (call ==  MAT_REUSE_MATRIX) {
    ierr = PetscObjectQuery((PetscObject)*newmat,"SubMatrix",(PetscObject*)&Mreuse);CHKERRQ(ierr);
    PetscCheckFalse(!Mreuse,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Submatrix passed in was not used before, cannot reuse");
    ierr = MatCreateSubMatrices_MPIBAIJ_local(mat,1,&isrow_new,&iscol_new,MAT_REUSE_MATRIX,&Mreuse);CHKERRQ(ierr);
  } else {
    ierr = MatCreateSubMatrices_MPIBAIJ_local(mat,1,&isrow_new,&iscol_new,MAT_INITIAL_MATRIX,&Mreuse);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&isrow_new);CHKERRQ(ierr);
  ierr = ISDestroy(&iscol_new);CHKERRQ(ierr);
  /*
      m - number of local rows
      n - number of columns (same on all processors)
      rstart - first row in new global matrix generated
  */
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = MatGetSize(Mreuse,&m,&n);CHKERRQ(ierr);
  m    = m/bs;
  n    = n/bs;

  if (call == MAT_INITIAL_MATRIX) {
    aij = (Mat_SeqBAIJ*)(Mreuse)->data;
    ii  = aij->i;
    jj  = aij->j;

    /*
        Determine the number of non-zeros in the diagonal and off-diagonal
        portions of the matrix in order to do correct preallocation
    */

    /* first get start and end of "diagonal" columns */
    if (csize == PETSC_DECIDE) {
      ierr = ISGetSize(isrow,&mglobal);CHKERRQ(ierr);
      if (mglobal == n*bs) { /* square matrix */
        nlocal = m;
      } else {
        nlocal = n/size + ((n % size) > rank);
      }
    } else {
      nlocal = csize/bs;
    }
    ierr   = MPI_Scan(&nlocal,&rend,1,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
    rstart = rend - nlocal;
    PetscCheckFalse(rank == size - 1 && rend != n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local column sizes %" PetscInt_FMT " do not add up to total number of columns %" PetscInt_FMT,rend,n);

    /* next, compute all the lengths */
    ierr  = PetscMalloc2(m+1,&dlens,m+1,&olens);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      jend = ii[i+1] - ii[i];
      olen = 0;
      dlen = 0;
      for (j=0; j<jend; j++) {
        if (*jj < rstart || *jj >= rend) olen++;
        else dlen++;
        jj++;
      }
      olens[i] = olen;
      dlens[i] = dlen;
    }
    ierr = MatCreate(comm,&M);CHKERRQ(ierr);
    ierr = MatSetSizes(M,bs*m,bs*nlocal,PETSC_DECIDE,bs*n);CHKERRQ(ierr);
    ierr = MatSetType(M,((PetscObject)mat)->type_name);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(M,bs,0,dlens,0,olens);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(M,bs,0,dlens,0,olens);CHKERRQ(ierr);
    ierr = PetscFree2(dlens,olens);CHKERRQ(ierr);
  } else {
    PetscInt ml,nl;

    M    = *newmat;
    ierr = MatGetLocalSize(M,&ml,&nl);CHKERRQ(ierr);
    PetscCheckFalse(ml != m,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Previous matrix must be same size/layout as request");
    ierr = MatZeroEntries(M);CHKERRQ(ierr);
    /*
         The next two lines are needed so we may call MatSetValues_MPIAIJ() below directly,
       rather than the slower MatSetValues().
    */
    M->was_assembled = PETSC_TRUE;
    M->assembled     = PETSC_FALSE;
  }
  ierr = MatSetOption(M,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(M,&rstart,&rend);CHKERRQ(ierr);
  aij  = (Mat_SeqBAIJ*)(Mreuse)->data;
  ii   = aij->i;
  jj   = aij->j;
  aa   = aij->a;
  for (i=0; i<m; i++) {
    row   = rstart/bs + i;
    nz    = ii[i+1] - ii[i];
    cwork = jj;     jj += nz;
    vwork = aa;     aa += nz*bs*bs;
    ierr  = MatSetValuesBlocked_MPIBAIJ(M,1,&row,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr    = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr    = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *newmat = M;

  /* save submatrix used in processor for next request */
  if (call ==  MAT_INITIAL_MATRIX) {
    ierr = PetscObjectCompose((PetscObject)M,"SubMatrix",(PetscObject)Mreuse);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)Mreuse);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatPermute_MPIBAIJ(Mat A,IS rowp,IS colp,Mat *B)
{
  MPI_Comm       comm,pcomm;
  PetscInt       clocal_size,nrows;
  const PetscInt *rows;
  PetscMPIInt    size;
  IS             crowp,lcolp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  /* make a collective version of 'rowp' */
  ierr = PetscObjectGetComm((PetscObject)rowp,&pcomm);CHKERRQ(ierr);
  if (pcomm==comm) {
    crowp = rowp;
  } else {
    ierr = ISGetSize(rowp,&nrows);CHKERRQ(ierr);
    ierr = ISGetIndices(rowp,&rows);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,nrows,rows,PETSC_COPY_VALUES,&crowp);CHKERRQ(ierr);
    ierr = ISRestoreIndices(rowp,&rows);CHKERRQ(ierr);
  }
  ierr = ISSetPermutation(crowp);CHKERRQ(ierr);
  /* make a local version of 'colp' */
  ierr = PetscObjectGetComm((PetscObject)colp,&pcomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(pcomm,&size);CHKERRMPI(ierr);
  if (size==1) {
    lcolp = colp;
  } else {
    ierr = ISAllGather(colp,&lcolp);CHKERRQ(ierr);
  }
  ierr = ISSetPermutation(lcolp);CHKERRQ(ierr);
  /* now we just get the submatrix */
  ierr = MatGetLocalSize(A,NULL,&clocal_size);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix_MPIBAIJ_Private(A,crowp,lcolp,clocal_size,MAT_INITIAL_MATRIX,B);CHKERRQ(ierr);
  /* clean up */
  if (pcomm!=comm) {
    ierr = ISDestroy(&crowp);CHKERRQ(ierr);
  }
  if (size>1) {
    ierr = ISDestroy(&lcolp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatGetGhosts_MPIBAIJ(Mat mat,PetscInt *nghosts,const PetscInt *ghosts[])
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ*) mat->data;
  Mat_SeqBAIJ *B    = (Mat_SeqBAIJ*)baij->B->data;

  PetscFunctionBegin;
  if (nghosts) *nghosts = B->nbs;
  if (ghosts) *ghosts = baij->garray;
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetSeqNonzeroStructure_MPIBAIJ(Mat A,Mat *newmat)
{
  Mat            B;
  Mat_MPIBAIJ    *a  = (Mat_MPIBAIJ*)A->data;
  Mat_SeqBAIJ    *ad = (Mat_SeqBAIJ*)a->A->data,*bd = (Mat_SeqBAIJ*)a->B->data;
  Mat_SeqAIJ     *b;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,*recvcounts = NULL,*displs = NULL;
  PetscInt       sendcount,i,*rstarts = A->rmap->range,n,cnt,j,bs = A->rmap->bs;
  PetscInt       m,*garray = a->garray,*lens,*jsendbuf,*a_jsendbuf,*b_jsendbuf;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank);CHKERRMPI(ierr);

  /* ----------------------------------------------------------------
     Tell every processor the number of nonzeros per row
  */
  ierr = PetscMalloc1(A->rmap->N/bs,&lens);CHKERRQ(ierr);
  for (i=A->rmap->rstart/bs; i<A->rmap->rend/bs; i++) {
    lens[i] = ad->i[i-A->rmap->rstart/bs+1] - ad->i[i-A->rmap->rstart/bs] + bd->i[i-A->rmap->rstart/bs+1] - bd->i[i-A->rmap->rstart/bs];
  }
  ierr      = PetscMalloc1(2*size,&recvcounts);CHKERRQ(ierr);
  displs    = recvcounts + size;
  for (i=0; i<size; i++) {
    recvcounts[i] = A->rmap->range[i+1]/bs - A->rmap->range[i]/bs;
    displs[i]     = A->rmap->range[i]/bs;
  }
  ierr = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,lens,recvcounts,displs,MPIU_INT,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  /* ---------------------------------------------------------------
     Create the sequential matrix of the same type as the local block diagonal
  */
  ierr = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->N/bs,A->cmap->N/bs,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,lens);CHKERRQ(ierr);
  b    = (Mat_SeqAIJ*)B->data;

  /*--------------------------------------------------------------------
    Copy my part of matrix column indices over
  */
  sendcount  = ad->nz + bd->nz;
  jsendbuf   = b->j + b->i[rstarts[rank]/bs];
  a_jsendbuf = ad->j;
  b_jsendbuf = bd->j;
  n          = A->rmap->rend/bs - A->rmap->rstart/bs;
  cnt        = 0;
  for (i=0; i<n; i++) {

    /* put in lower diagonal portion */
    m = bd->i[i+1] - bd->i[i];
    while (m > 0) {
      /* is it above diagonal (in bd (compressed) numbering) */
      if (garray[*b_jsendbuf] > A->rmap->rstart/bs + i) break;
      jsendbuf[cnt++] = garray[*b_jsendbuf++];
      m--;
    }

    /* put in diagonal portion */
    for (j=ad->i[i]; j<ad->i[i+1]; j++) {
      jsendbuf[cnt++] = A->rmap->rstart/bs + *a_jsendbuf++;
    }

    /* put in upper diagonal portion */
    while (m-- > 0) {
      jsendbuf[cnt++] = garray[*b_jsendbuf++];
    }
  }
  PetscCheckFalse(cnt != sendcount,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupted PETSc matrix: nz given %" PetscInt_FMT " actual nz %" PetscInt_FMT,sendcount,cnt);

  /*--------------------------------------------------------------------
    Gather all column indices to all processors
  */
  for (i=0; i<size; i++) {
    recvcounts[i] = 0;
    for (j=A->rmap->range[i]/bs; j<A->rmap->range[i+1]/bs; j++) {
      recvcounts[i] += lens[j];
    }
  }
  displs[0] = 0;
  for (i=1; i<size; i++) {
    displs[i] = displs[i-1] + recvcounts[i-1];
  }
  ierr = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,b->j,recvcounts,displs,MPIU_INT,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  /*--------------------------------------------------------------------
    Assemble the matrix into useable form (note numerical values not yet set)
  */
  /* set the b->ilen (length of each row) values */
  ierr = PetscArraycpy(b->ilen,lens,A->rmap->N/bs);CHKERRQ(ierr);
  /* set the b->i indices */
  b->i[0] = 0;
  for (i=1; i<=A->rmap->N/bs; i++) {
    b->i[i] = b->i[i-1] + lens[i-1];
  }
  ierr = PetscFree(lens);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(recvcounts);CHKERRQ(ierr);

  if (A->symmetric) {
    ierr = MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  } else if (A->hermitian) {
    ierr = MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  } else if (A->structurally_symmetric) {
    ierr = MatSetOption(B,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  }
  *newmat = B;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_MPIBAIJ(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_MPIBAIJ    *mat = (Mat_MPIBAIJ*)matin->data;
  PetscErrorCode ierr;
  Vec            bb1 = NULL;

  PetscFunctionBegin;
  if (flag == SOR_APPLY_UPPER) {
    ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (its > 1 || ~flag & SOR_ZERO_INITIAL_GUESS) {
    ierr = VecDuplicate(bb,&bb1);CHKERRQ(ierr);
  }

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
      its--;
    }

    while (its--) {
      ierr = VecScatterBegin(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* update rhs: bb1 = bb - B*x */
      ierr = VecScale(mat->lvec,-1.0);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb,bb1);CHKERRQ(ierr);

      /* local sweep */
      ierr = (*mat->A->ops->sor)(mat->A,bb1,omega,SOR_SYMMETRIC_SWEEP,fshift,lits,1,xx);CHKERRQ(ierr);
    }
  } else if (flag & SOR_LOCAL_FORWARD_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
      its--;
    }
    while (its--) {
      ierr = VecScatterBegin(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* update rhs: bb1 = bb - B*x */
      ierr = VecScale(mat->lvec,-1.0);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb,bb1);CHKERRQ(ierr);

      /* local sweep */
      ierr = (*mat->A->ops->sor)(mat->A,bb1,omega,SOR_FORWARD_SWEEP,fshift,lits,1,xx);CHKERRQ(ierr);
    }
  } else if (flag & SOR_LOCAL_BACKWARD_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
      its--;
    }
    while (its--) {
      ierr = VecScatterBegin(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* update rhs: bb1 = bb - B*x */
      ierr = VecScale(mat->lvec,-1.0);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb,bb1);CHKERRQ(ierr);

      /* local sweep */
      ierr = (*mat->A->ops->sor)(mat->A,bb1,omega,SOR_BACKWARD_SWEEP,fshift,lits,1,xx);CHKERRQ(ierr);
    }
  } else SETERRQ(PetscObjectComm((PetscObject)matin),PETSC_ERR_SUP,"Parallel version of SOR requested not supported");

  ierr = VecDestroy(&bb1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetColumnReductions_MPIBAIJ(Mat A,PetscInt type,PetscReal *reductions)
{
  PetscErrorCode ierr;
  Mat_MPIBAIJ    *aij = (Mat_MPIBAIJ*)A->data;
  PetscInt       m,N,i,*garray = aij->garray;
  PetscInt       ib,jb,bs = A->rmap->bs;
  Mat_SeqBAIJ    *a_aij = (Mat_SeqBAIJ*) aij->A->data;
  MatScalar      *a_val = a_aij->a;
  Mat_SeqBAIJ    *b_aij = (Mat_SeqBAIJ*) aij->B->data;
  MatScalar      *b_val = b_aij->a;
  PetscReal      *work;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&N);CHKERRQ(ierr);
  ierr = PetscCalloc1(N,&work);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=a_aij->i[0]; i<a_aij->i[aij->A->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          work[A->cmap->rstart + a_aij->j[i] * bs + jb] += PetscAbsScalar(*a_val * *a_val);
          a_val++;
        }
      }
    }
    for (i=b_aij->i[0]; i<b_aij->i[aij->B->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          work[garray[b_aij->j[i]] * bs + jb] += PetscAbsScalar(*b_val * *b_val);
          b_val++;
        }
      }
    }
  } else if (type == NORM_1) {
    for (i=a_aij->i[0]; i<a_aij->i[aij->A->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          work[A->cmap->rstart + a_aij->j[i] * bs + jb] += PetscAbsScalar(*a_val);
          a_val++;
        }
      }
    }
    for (i=b_aij->i[0]; i<b_aij->i[aij->B->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
       for (ib=0; ib<bs; ib++) {
          work[garray[b_aij->j[i]] * bs + jb] += PetscAbsScalar(*b_val);
          b_val++;
        }
      }
    }
  } else if (type == NORM_INFINITY) {
    for (i=a_aij->i[0]; i<a_aij->i[aij->A->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          int col = A->cmap->rstart + a_aij->j[i] * bs + jb;
          work[col] = PetscMax(PetscAbsScalar(*a_val), work[col]);
          a_val++;
        }
      }
    }
    for (i=b_aij->i[0]; i<b_aij->i[aij->B->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          int col = garray[b_aij->j[i]] * bs + jb;
          work[col] = PetscMax(PetscAbsScalar(*b_val), work[col]);
          b_val++;
        }
      }
    }
  } else if (type == REDUCTION_SUM_REALPART || type == REDUCTION_MEAN_REALPART) {
    for (i=a_aij->i[0]; i<a_aij->i[aij->A->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          work[A->cmap->rstart + a_aij->j[i] * bs + jb] += PetscRealPart(*a_val);
          a_val++;
        }
      }
    }
    for (i=b_aij->i[0]; i<b_aij->i[aij->B->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
       for (ib=0; ib<bs; ib++) {
          work[garray[b_aij->j[i]] * bs + jb] += PetscRealPart(*b_val);
          b_val++;
        }
      }
    }
  } else if (type == REDUCTION_SUM_IMAGINARYPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i=a_aij->i[0]; i<a_aij->i[aij->A->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
        for (ib=0; ib<bs; ib++) {
          work[A->cmap->rstart + a_aij->j[i] * bs + jb] += PetscImaginaryPart(*a_val);
          a_val++;
        }
      }
    }
    for (i=b_aij->i[0]; i<b_aij->i[aij->B->rmap->n/bs]; i++) {
      for (jb=0; jb<bs; jb++) {
       for (ib=0; ib<bs; ib++) {
          work[garray[b_aij->j[i]] * bs + jb] += PetscImaginaryPart(*b_val);
          b_val++;
        }
      }
    }
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Unknown reduction type");
  if (type == NORM_INFINITY) {
    ierr = MPIU_Allreduce(work,reductions,N,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  } else {
    ierr = MPIU_Allreduce(work,reductions,N,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<N; i++) reductions[i] = PetscSqrtReal(reductions[i]);
  } else if (type == REDUCTION_MEAN_REALPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i=0; i<N; i++) reductions[i] /= m;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatInvertBlockDiagonal_MPIBAIJ(Mat A,const PetscScalar **values)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatInvertBlockDiagonal(a->A,values);CHKERRQ(ierr);
  A->factorerrortype             = a->A->factorerrortype;
  A->factorerror_zeropivot_value = a->A->factorerror_zeropivot_value;
  A->factorerror_zeropivot_row   = a->A->factorerror_zeropivot_row;
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_MPIBAIJ(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_MPIBAIJ    *maij = (Mat_MPIBAIJ*)Y->data;
  Mat_SeqBAIJ    *aij = (Mat_SeqBAIJ*)maij->A->data;

  PetscFunctionBegin;
  if (!Y->preallocated) {
    ierr = MatMPIBAIJSetPreallocation(Y,Y->rmap->bs,1,NULL,0,NULL);CHKERRQ(ierr);
  } else if (!aij->nz) {
    PetscInt nonew = aij->nonew;
    ierr = MatSeqBAIJSetPreallocation(maij->A,Y->rmap->bs,1,NULL);CHKERRQ(ierr);
    aij->nonew = nonew;
  }
  ierr = MatShift_Basic(Y,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMissingDiagonal_MPIBAIJ(Mat A,PetscBool  *missing,PetscInt *d)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(A->rmap->n != A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only works for square matrices");
  ierr = MatMissingDiagonal(a->A,missing,d);CHKERRQ(ierr);
  if (d) {
    PetscInt rstart;
    ierr = MatGetOwnershipRange(A,&rstart,NULL);CHKERRQ(ierr);
    *d += rstart/A->rmap->bs;

  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatGetDiagonalBlock_MPIBAIJ(Mat A,Mat *a)
{
  PetscFunctionBegin;
  *a = ((Mat_MPIBAIJ*)A->data)->A;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_MPIBAIJ,
                                       MatGetRow_MPIBAIJ,
                                       MatRestoreRow_MPIBAIJ,
                                       MatMult_MPIBAIJ,
                                /* 4*/ MatMultAdd_MPIBAIJ,
                                       MatMultTranspose_MPIBAIJ,
                                       MatMultTransposeAdd_MPIBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*10*/ NULL,
                                       NULL,
                                       NULL,
                                       MatSOR_MPIBAIJ,
                                       MatTranspose_MPIBAIJ,
                                /*15*/ MatGetInfo_MPIBAIJ,
                                       MatEqual_MPIBAIJ,
                                       MatGetDiagonal_MPIBAIJ,
                                       MatDiagonalScale_MPIBAIJ,
                                       MatNorm_MPIBAIJ,
                                /*20*/ MatAssemblyBegin_MPIBAIJ,
                                       MatAssemblyEnd_MPIBAIJ,
                                       MatSetOption_MPIBAIJ,
                                       MatZeroEntries_MPIBAIJ,
                                /*24*/ MatZeroRows_MPIBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*29*/ MatSetUp_MPIBAIJ,
                                       NULL,
                                       NULL,
                                       MatGetDiagonalBlock_MPIBAIJ,
                                       NULL,
                                /*34*/ MatDuplicate_MPIBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*39*/ MatAXPY_MPIBAIJ,
                                       MatCreateSubMatrices_MPIBAIJ,
                                       MatIncreaseOverlap_MPIBAIJ,
                                       MatGetValues_MPIBAIJ,
                                       MatCopy_MPIBAIJ,
                                /*44*/ NULL,
                                       MatScale_MPIBAIJ,
                                       MatShift_MPIBAIJ,
                                       NULL,
                                       MatZeroRowsColumns_MPIBAIJ,
                                /*49*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*54*/ MatFDColoringCreate_MPIXAIJ,
                                       NULL,
                                       MatSetUnfactored_MPIBAIJ,
                                       MatPermute_MPIBAIJ,
                                       MatSetValuesBlocked_MPIBAIJ,
                                /*59*/ MatCreateSubMatrix_MPIBAIJ,
                                       MatDestroy_MPIBAIJ,
                                       MatView_MPIBAIJ,
                                       NULL,
                                       NULL,
                                /*64*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*69*/ MatGetRowMaxAbs_MPIBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*74*/ NULL,
                                       MatFDColoringApply_BAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*79*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatLoad_MPIBAIJ,
                                /*84*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*89*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*94*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                /*99*/ NULL,
                                       NULL,
                                       NULL,
                                       MatConjugate_MPIBAIJ,
                                       NULL,
                                /*104*/NULL,
                                       MatRealPart_MPIBAIJ,
                                       MatImaginaryPart_MPIBAIJ,
                                       NULL,
                                       NULL,
                                /*109*/NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatMissingDiagonal_MPIBAIJ,
                                /*114*/MatGetSeqNonzeroStructure_MPIBAIJ,
                                       NULL,
                                       MatGetGhosts_MPIBAIJ,
                                       NULL,
                                       NULL,
                                /*119*/NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatGetMultiProcBlock_MPIBAIJ,
                                /*124*/NULL,
                                       MatGetColumnReductions_MPIBAIJ,
                                       MatInvertBlockDiagonal_MPIBAIJ,
                                       NULL,
                                       NULL,
                               /*129*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*134*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*139*/ MatSetBlockSizes_Default,
                                       NULL,
                                       NULL,
                                       MatFDColoringSetUp_MPIXAIJ,
                                       NULL,
                                /*144*/MatCreateMPIMatConcatenateSeqMat_MPIBAIJ,
                                       NULL,
                                       NULL,
                                       NULL
};

PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJ_MPISBAIJ(Mat,MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatConvert_XAIJ_IS(Mat,MatType,MatReuse,Mat*);

PetscErrorCode MatMPIBAIJSetPreallocationCSR_MPIBAIJ(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[],const PetscScalar V[])
{
  PetscInt       m,rstart,cstart,cend;
  PetscInt       i,j,dlen,olen,nz,nz_max=0,*d_nnz=NULL,*o_nnz=NULL;
  const PetscInt *JJ    =NULL;
  PetscScalar    *values=NULL;
  PetscBool      roworiented = ((Mat_MPIBAIJ*)B->data)->roworiented;
  PetscErrorCode ierr;
  PetscBool      nooffprocentries;

  PetscFunctionBegin;
  ierr   = PetscLayoutSetBlockSize(B->rmap,bs);CHKERRQ(ierr);
  ierr   = PetscLayoutSetBlockSize(B->cmap,bs);CHKERRQ(ierr);
  ierr   = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr   = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr   = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);
  m      = B->rmap->n/bs;
  rstart = B->rmap->rstart/bs;
  cstart = B->cmap->rstart/bs;
  cend   = B->cmap->rend/bs;

  PetscCheckFalse(ii[0],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ii[0] must be 0 but it is %" PetscInt_FMT,ii[0]);
  ierr = PetscMalloc2(m,&d_nnz,m,&o_nnz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nz = ii[i+1] - ii[i];
    PetscCheckFalse(nz < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local row %" PetscInt_FMT " has a negative number of columns %" PetscInt_FMT,i,nz);
    nz_max = PetscMax(nz_max,nz);
    dlen   = 0;
    olen   = 0;
    JJ     = jj + ii[i];
    for (j=0; j<nz; j++) {
      if (*JJ < cstart || *JJ >= cend) olen++;
      else dlen++;
      JJ++;
    }
    d_nnz[i] = dlen;
    o_nnz[i] = olen;
  }
  ierr = MatMPIBAIJSetPreallocation(B,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
  ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);

  values = (PetscScalar*)V;
  if (!values) {
    ierr = PetscCalloc1(bs*bs*nz_max,&values);CHKERRQ(ierr);
  }
  for (i=0; i<m; i++) {
    PetscInt          row    = i + rstart;
    PetscInt          ncols  = ii[i+1] - ii[i];
    const PetscInt    *icols = jj + ii[i];
    if (bs == 1 || !roworiented) {         /* block ordering matches the non-nested layout of MatSetValues so we can insert entire rows */
      const PetscScalar *svals = values + (V ? (bs*bs*ii[i]) : 0);
      ierr = MatSetValuesBlocked_MPIBAIJ(B,1,&row,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
    } else {                    /* block ordering does not match so we can only insert one block at a time. */
      PetscInt j;
      for (j=0; j<ncols; j++) {
        const PetscScalar *svals = values + (V ? (bs*bs*(ii[i]+j)) : 0);
        ierr = MatSetValuesBlocked_MPIBAIJ(B,1,&row,1,&icols[j],svals,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  if (!V) { ierr = PetscFree(values);CHKERRQ(ierr); }
  nooffprocentries    = B->nooffprocentries;
  B->nooffprocentries = PETSC_TRUE;
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  B->nooffprocentries = nooffprocentries;

  ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatMPIBAIJSetPreallocationCSR - Creates a sparse parallel matrix in BAIJ format using the given nonzero structure and (optional) numerical values

   Collective

   Input Parameters:
+  B - the matrix
.  bs - the block size
.  i - the indices into j for the start of each local row (starts with zero)
.  j - the column indices for each local row (starts with zero) these must be sorted for each row
-  v - optional values in the matrix

   Level: advanced

   Notes:
    The order of the entries in values is specified by the MatOption MAT_ROW_ORIENTED.  For example, C programs
   may want to use the default MAT_ROW_ORIENTED=PETSC_TRUE and use an array v[nnz][bs][bs] where the second index is
   over rows within a block and the last index is over columns within a block row.  Fortran programs will likely set
   MAT_ROW_ORIENTED=PETSC_FALSE and use a Fortran array v(bs,bs,nnz) in which the first index is over rows within a
   block column and the second index is over columns within a block.

   Though this routine has Preallocation() in the name it also sets the exact nonzero locations of the matrix entries and usually the numerical values as well

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIBAIJSetPreallocation(), MatCreateAIJ(), MPIAIJ, MatCreateMPIBAIJWithArrays(), MPIBAIJ
@*/
PetscErrorCode  MatMPIBAIJSetPreallocationCSR(Mat B,PetscInt bs,const PetscInt i[],const PetscInt j[], const PetscScalar v[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscTryMethod(B,"MatMPIBAIJSetPreallocationCSR_C",(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]),(B,bs,i,j,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatMPIBAIJSetPreallocation_MPIBAIJ(Mat B,PetscInt bs,PetscInt d_nz,const PetscInt *d_nnz,PetscInt o_nz,const PetscInt *o_nnz)
{
  Mat_MPIBAIJ    *b;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatSetBlockSize(B,PetscAbs(bs));CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);

  if (d_nnz) {
    for (i=0; i<B->rmap->n/bs; i++) {
      PetscCheckFalse(d_nnz[i] < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than -1: local row %" PetscInt_FMT " value %" PetscInt_FMT,i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->rmap->n/bs; i++) {
      PetscCheckFalse(o_nnz[i] < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than -1: local row %" PetscInt_FMT " value %" PetscInt_FMT,i,o_nnz[i]);
    }
  }

  b      = (Mat_MPIBAIJ*)B->data;
  b->bs2 = bs*bs;
  b->mbs = B->rmap->n/bs;
  b->nbs = B->cmap->n/bs;
  b->Mbs = B->rmap->N/bs;
  b->Nbs = B->cmap->N/bs;

  for (i=0; i<=b->size; i++) {
    b->rangebs[i] = B->rmap->range[i]/bs;
  }
  b->rstartbs = B->rmap->rstart/bs;
  b->rendbs   = B->rmap->rend/bs;
  b->cstartbs = B->cmap->rstart/bs;
  b->cendbs   = B->cmap->rend/bs;

#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableDestroy(&b->colmap);CHKERRQ(ierr);
#else
  ierr = PetscFree(b->colmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(b->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&b->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->Mvctx);CHKERRQ(ierr);

  /* Because the B will have been resized we simply destroy it and create a new one each time */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRMPI(ierr);
  ierr = MatDestroy(&b->B);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
  ierr = MatSetSizes(b->B,B->rmap->n,size > 1 ? B->cmap->N : 0,B->rmap->n,size > 1 ? B->cmap->N : 0);CHKERRQ(ierr);
  ierr = MatSetType(b->B,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);

  if (!B->preallocated) {
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQBAIJ);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
    ierr = MatStashCreate_Private(PetscObjectComm((PetscObject)B),bs,&B->bstash);CHKERRQ(ierr);
  }

  ierr = MatSeqBAIJSetPreallocation(b->A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(b->B,bs,o_nz,o_nnz);CHKERRQ(ierr);
  B->preallocated  = PETSC_TRUE;
  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  PetscFunctionReturn(0);
}

extern PetscErrorCode  MatDiagonalScaleLocal_MPIBAIJ(Mat,Vec);
extern PetscErrorCode  MatSetHashTableFactor_MPIBAIJ(Mat,PetscReal);

PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJ_MPIAdj(Mat B, MatType newtype,MatReuse reuse,Mat *adj)
{
  Mat_MPIBAIJ    *b = (Mat_MPIBAIJ*)B->data;
  PetscErrorCode ierr;
  Mat_SeqBAIJ    *d  = (Mat_SeqBAIJ*) b->A->data,*o = (Mat_SeqBAIJ*) b->B->data;
  PetscInt       M   = B->rmap->n/B->rmap->bs,i,*ii,*jj,cnt,j,k,rstart = B->rmap->rstart/B->rmap->bs;
  const PetscInt *id = d->i, *jd = d->j, *io = o->i, *jo = o->j, *garray = b->garray;

  PetscFunctionBegin;
  ierr  = PetscMalloc1(M+1,&ii);CHKERRQ(ierr);
  ii[0] = 0;
  for (i=0; i<M; i++) {
    PetscCheckFalse((id[i+1] - id[i]) < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Indices wrong %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT,i,id[i],id[i+1]);
    PetscCheckFalse((io[i+1] - io[i]) < 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Indices wrong %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT,i,io[i],io[i+1]);
    ii[i+1] = ii[i] + id[i+1] - id[i] + io[i+1] - io[i];
    /* remove one from count of matrix has diagonal */
    for (j=id[i]; j<id[i+1]; j++) {
      if (jd[j] == i) {ii[i+1]--;break;}
    }
  }
  ierr = PetscMalloc1(ii[M],&jj);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<M; i++) {
    for (j=io[i]; j<io[i+1]; j++) {
      if (garray[jo[j]] > rstart) break;
      jj[cnt++] = garray[jo[j]];
    }
    for (k=id[i]; k<id[i+1]; k++) {
      if (jd[k] != i) {
        jj[cnt++] = rstart + jd[k];
      }
    }
    for (; j<io[i+1]; j++) {
      jj[cnt++] = garray[jo[j]];
    }
  }
  ierr = MatCreateMPIAdj(PetscObjectComm((PetscObject)B),M,B->cmap->N/B->rmap->bs,ii,jj,NULL,adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/mpi/mpiaij.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqAIJ(Mat,MatType,MatReuse,Mat*);

PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJ_MPIAIJ(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  Mat            B;
  Mat_MPIAIJ     *b;

  PetscFunctionBegin;
  PetscCheckFalse(!A->assembled,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Matrix must be assembled");

  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
  } else {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetType(B,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(B,A->rmap->bs,A->cmap->bs);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(B,0,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(B,0,NULL,0,NULL);CHKERRQ(ierr);
  }
  b = (Mat_MPIAIJ*) B->data;

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatConvert_SeqBAIJ_SeqAIJ(a->A, MATSEQAIJ, MAT_REUSE_MATRIX, &b->A);CHKERRQ(ierr);
    ierr = MatConvert_SeqBAIJ_SeqAIJ(a->B, MATSEQAIJ, MAT_REUSE_MATRIX, &b->B);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy(&b->A);CHKERRQ(ierr);
    ierr = MatDestroy(&b->B);CHKERRQ(ierr);
    ierr = MatDisAssemble_MPIBAIJ(A);CHKERRQ(ierr);
    ierr = MatConvert_SeqBAIJ_SeqAIJ(a->A, MATSEQAIJ, MAT_INITIAL_MATRIX, &b->A);CHKERRQ(ierr);
    ierr = MatConvert_SeqBAIJ_SeqAIJ(a->B, MATSEQAIJ, MAT_INITIAL_MATRIX, &b->B);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else {
   *newmat = B;
  }
  PetscFunctionReturn(0);
}

/*MC
   MATMPIBAIJ - MATMPIBAIJ = "mpibaij" - A matrix type to be used for distributed block sparse matrices.

   Options Database Keys:
+ -mat_type mpibaij - sets the matrix type to "mpibaij" during a call to MatSetFromOptions()
. -mat_block_size <bs> - set the blocksize used to store the matrix
. -mat_baij_mult_version version - indicate the version of the matrix-vector product to use  (0 often indicates using BLAS)
- -mat_use_hash_table <fact>

   Level: beginner

   Notes:
    MatSetOptions(,MAT_STRUCTURE_ONLY,PETSC_TRUE) may be called for this matrix type. In this no
    space is allocated for the nonzero entries and any entries passed with MatSetValues() are ignored

.seealso: MatCreateBAIJ
M*/

PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJ_MPIBSTRM(Mat,MatType,MatReuse,Mat*);

PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJ(Mat B)
{
  Mat_MPIBAIJ    *b;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr    = PetscNewLog(B,&b);CHKERRQ(ierr);
  B->data = (void*)b;

  ierr         = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->assembled = PETSC_FALSE;

  B->insertmode = NOT_SET_VALUES;
  ierr          = MPI_Comm_rank(PetscObjectComm((PetscObject)B),&b->rank);CHKERRMPI(ierr);
  ierr          = MPI_Comm_size(PetscObjectComm((PetscObject)B),&b->size);CHKERRMPI(ierr);

  /* build local table of row and column ownerships */
  ierr = PetscMalloc1(b->size+1,&b->rangebs);CHKERRQ(ierr);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(PetscObjectComm((PetscObject)B),1,&B->stash);CHKERRQ(ierr);

  b->donotstash  = PETSC_FALSE;
  b->colmap      = NULL;
  b->garray      = NULL;
  b->roworiented = PETSC_TRUE;

  /* stuff used in block assembly */
  b->barray = NULL;

  /* stuff used for matrix vector multiply */
  b->lvec  = NULL;
  b->Mvctx = NULL;

  /* stuff for MatGetRow() */
  b->rowindices   = NULL;
  b->rowvalues    = NULL;
  b->getrowactive = PETSC_FALSE;

  /* hash table stuff */
  b->ht           = NULL;
  b->hd           = NULL;
  b->ht_size      = 0;
  b->ht_flag      = PETSC_FALSE;
  b->ht_fact      = 0;
  b->ht_total_ct  = 0;
  b->ht_insert_ct = 0;

  /* stuff for MatCreateSubMatrices_MPIBAIJ_local() */
  b->ijonly = PETSC_FALSE;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpibaij_mpiadj_C",MatConvert_MPIBAIJ_MPIAdj);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpibaij_mpiaij_C",MatConvert_MPIBAIJ_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpibaij_mpisbaij_C",MatConvert_MPIBAIJ_MPISBAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpibaij_hypre_C",MatConvert_AIJ_HYPRE);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPIBAIJSetPreallocation_C",MatMPIBAIJSetPreallocation_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPIBAIJSetPreallocationCSR_C",MatMPIBAIJSetPreallocationCSR_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDiagonalScaleLocal_C",MatDiagonalScaleLocal_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSetHashTableFactor_C",MatSetHashTableFactor_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpibaij_is_C",MatConvert_XAIJ_IS);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIBAIJ);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)B),NULL,"Options for loading MPIBAIJ matrix 1","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsName("-mat_use_hash_table","Use hash table to save time in constructing matrix","MatSetOption",&flg);CHKERRQ(ierr);
  if (flg) {
    PetscReal fact = 1.39;
    ierr = MatSetOption(B,MAT_USE_HASH_TABLE,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_use_hash_table","Use hash table factor","MatMPIBAIJSetHashTableFactor",fact,&fact,NULL);CHKERRQ(ierr);
    if (fact <= 1.0) fact = 1.39;
    ierr = MatMPIBAIJSetHashTableFactor(B,fact);CHKERRQ(ierr);
    ierr = PetscInfo(B,"Hash table Factor used %5.2g\n",(double)fact);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATBAIJ - MATBAIJ = "baij" - A matrix type to be used for block sparse matrices.

   This matrix type is identical to MATSEQBAIJ when constructed with a single process communicator,
   and MATMPIBAIJ otherwise.

   Options Database Keys:
. -mat_type baij - sets the matrix type to "baij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateBAIJ(),MATSEQBAIJ,MATMPIBAIJ, MatMPIBAIJSetPreallocation(), MatMPIBAIJSetPreallocationCSR()
M*/

/*@C
   MatMPIBAIJSetPreallocation - Allocates memory for a sparse parallel matrix in block AIJ format
   (block compressed row).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

   Collective on Mat

   Input Parameters:
+  B - the matrix
.  bs   - size of block, the blocks are ALWAYS square. One can use MatSetBlockSizes() to set a different row and column blocksize but the row
          blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with MatCreateVecs()
.  d_nz  - number of block nonzeros per block row in diagonal portion of local
           submatrix  (same for all local rows)
.  d_nnz - array containing the number of block nonzeros in the various block rows
           of the in diagonal portion of the local (possibly different for each block
           row) or NULL.  If you plan to factor the matrix you must leave room for the diagonal entry and
           set it even if it is zero.
.  o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or NULL.

   If the *_nnz parameter is given then the *_nz parameter is ignored

   Options Database Keys:
+   -mat_block_size - size of the blocks to use
-   -mat_use_hash_table <fact>

   Notes:
   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one processor
   than it must be used on all processors that share the object for that argument.

   Storage Information:
   For a square global matrix we define each processor's diagonal portion
   to be its local rows and the corresponding columns (a square submatrix);
   each processor's off-diagonal portion encompasses the remainder of the
   local matrix (a rectangular submatrix).

   The user can specify preallocated storage for the diagonal part of
   the local submatrix with either d_nz or d_nnz (not both).  Set
   d_nz=PETSC_DEFAULT and d_nnz=NULL for PETSc to control dynamic
   memory allocation.  Likewise, specify preallocated storage for the
   off-diagonal part of the local submatrix with o_nz or o_nnz (not both).

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

.vb
           0 1 2 3 4 5 6 7 8 9 10 11
          --------------------------
   row 3  |o o o d d d o o o o  o  o
   row 4  |o o o d d d o o o o  o  o
   row 5  |o o o d d d o o o o  o  o
          --------------------------
.ve

   Thus, any entries in the d locations are stored in the d (diagonal)
   submatrix, and any entries in the o locations are stored in the
   o (off-diagonal) submatrix.  Note that the d and the o submatrices are
   stored simply in the MATSEQBAIJ format for compressed row storage.

   Now d_nz should indicate the number of block nonzeros per row in the d matrix,
   and o_nz should indicate the number of block nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices.

   You can call MatGetInfo() to get information on how effective the preallocation was;
   for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
   You can also run with the option -info and look for messages with the string
   malloc in them to see if additional memory allocation was needed.

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqBAIJ(), MatSetValues(), MatCreateBAIJ(), MatMPIBAIJSetPreallocationCSR(), PetscSplitOwnership()
@*/
PetscErrorCode  MatMPIBAIJSetPreallocation(Mat B,PetscInt bs,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscTryMethod(B,"MatMPIBAIJSetPreallocation_C",(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]),(B,bs,d_nz,d_nnz,o_nz,o_nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatCreateBAIJ - Creates a sparse parallel matrix in block AIJ format
   (block compressed row).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  bs   - size of block, the blocks are ALWAYS square. One can use MatSetBlockSizes() to set a different row and column blocksize but the row
          blocksize always defines the size of the blocks. The column blocksize sets the blocksize of the vectors obtained with MatCreateVecs()
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
.  n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
           This value should be the same as the local size used in creating the
           x vector for the matrix-vector product y = Ax.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.  d_nz  - number of nonzero blocks per block row in diagonal portion of local
           submatrix  (same for all local rows)
.  d_nnz - array containing the number of nonzero blocks in the various block rows
           of the in diagonal portion of the local (possibly different for each block
           row) or NULL.  If you plan to factor the matrix you must leave room for the diagonal entry
           and set it even if it is zero.
.  o_nz  - number of nonzero blocks per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzero blocks in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or NULL.

   Output Parameter:
.  A - the matrix

   Options Database Keys:
+   -mat_block_size - size of the blocks to use
-   -mat_use_hash_table <fact>

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If the *_nnz parameter is given then the *_nz parameter is ignored

   A nonzero block is any block that as 1 or more nonzeros in it

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one processor
   than it must be used on all processors that share the object for that argument.

   Storage Information:
   For a square global matrix we define each processor's diagonal portion
   to be its local rows and the corresponding columns (a square submatrix);
   each processor's off-diagonal portion encompasses the remainder of the
   local matrix (a rectangular submatrix).

   The user can specify preallocated storage for the diagonal part of
   the local submatrix with either d_nz or d_nnz (not both).  Set
   d_nz=PETSC_DEFAULT and d_nnz=NULL for PETSc to control dynamic
   memory allocation.  Likewise, specify preallocated storage for the
   off-diagonal part of the local submatrix with o_nz or o_nnz (not both).

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

.vb
           0 1 2 3 4 5 6 7 8 9 10 11
          --------------------------
   row 3  |o o o d d d o o o o  o  o
   row 4  |o o o d d d o o o o  o  o
   row 5  |o o o d d d o o o o  o  o
          --------------------------
.ve

   Thus, any entries in the d locations are stored in the d (diagonal)
   submatrix, and any entries in the o locations are stored in the
   o (off-diagonal) submatrix.  Note that the d and the o submatrices are
   stored simply in the MATSEQBAIJ format for compressed row storage.

   Now d_nz should indicate the number of block nonzeros per row in the d matrix,
   and o_nz should indicate the number of block nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices.

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqBAIJ(), MatSetValues(), MatCreateBAIJ(), MatMPIBAIJSetPreallocation(), MatMPIBAIJSetPreallocationCSR()
@*/
PetscErrorCode  MatCreateBAIJ(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIBAIJ);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(*A,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQBAIJ);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation(*A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPIBAIJ(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPIBAIJ    *a,*oldmat = (Mat_MPIBAIJ*)matin->data;
  PetscErrorCode ierr;
  PetscInt       len=0;

  PetscFunctionBegin;
  *newmat = NULL;
  ierr    = MatCreate(PetscObjectComm((PetscObject)matin),&mat);CHKERRQ(ierr);
  ierr    = MatSetSizes(mat,matin->rmap->n,matin->cmap->n,matin->rmap->N,matin->cmap->N);CHKERRQ(ierr);
  ierr    = MatSetType(mat,((PetscObject)matin)->type_name);CHKERRQ(ierr);

  mat->factortype   = matin->factortype;
  mat->preallocated = PETSC_TRUE;
  mat->assembled    = PETSC_TRUE;
  mat->insertmode   = NOT_SET_VALUES;

  a             = (Mat_MPIBAIJ*)mat->data;
  mat->rmap->bs = matin->rmap->bs;
  a->bs2        = oldmat->bs2;
  a->mbs        = oldmat->mbs;
  a->nbs        = oldmat->nbs;
  a->Mbs        = oldmat->Mbs;
  a->Nbs        = oldmat->Nbs;

  ierr = PetscLayoutReference(matin->rmap,&mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(matin->cmap,&mat->cmap);CHKERRQ(ierr);

  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  a->donotstash   = oldmat->donotstash;
  a->roworiented  = oldmat->roworiented;
  a->rowindices   = NULL;
  a->rowvalues    = NULL;
  a->getrowactive = PETSC_FALSE;
  a->barray       = NULL;
  a->rstartbs     = oldmat->rstartbs;
  a->rendbs       = oldmat->rendbs;
  a->cstartbs     = oldmat->cstartbs;
  a->cendbs       = oldmat->cendbs;

  /* hash table stuff */
  a->ht           = NULL;
  a->hd           = NULL;
  a->ht_size      = 0;
  a->ht_flag      = oldmat->ht_flag;
  a->ht_fact      = oldmat->ht_fact;
  a->ht_total_ct  = 0;
  a->ht_insert_ct = 0;

  ierr = PetscArraycpy(a->rangebs,oldmat->rangebs,a->size+1);CHKERRQ(ierr);
  if (oldmat->colmap) {
#if defined(PETSC_USE_CTABLE)
    ierr = PetscTableCreateCopy(oldmat->colmap,&a->colmap);CHKERRQ(ierr);
#else
    ierr = PetscMalloc1(a->Nbs,&a->colmap);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)mat,(a->Nbs)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscArraycpy(a->colmap,oldmat->colmap,a->Nbs);CHKERRQ(ierr);
#endif
  } else a->colmap = NULL;

  if (oldmat->garray && (len = ((Mat_SeqBAIJ*)(oldmat->B->data))->nbs)) {
    ierr = PetscMalloc1(len,&a->garray);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)mat,len*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscArraycpy(a->garray,oldmat->garray,len);CHKERRQ(ierr);
  } else a->garray = NULL;

  ierr = MatStashCreate_Private(PetscObjectComm((PetscObject)matin),matin->rmap->bs,&mat->bstash);CHKERRQ(ierr);
  ierr = VecDuplicate(oldmat->lvec,&a->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->lvec);CHKERRQ(ierr);
  ierr = VecScatterCopy(oldmat->Mvctx,&a->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->Mvctx);CHKERRQ(ierr);

  ierr    = MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  ierr    = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->A);CHKERRQ(ierr);
  ierr    = MatDuplicate(oldmat->B,cpvalues,&a->B);CHKERRQ(ierr);
  ierr    = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->B);CHKERRQ(ierr);
  ierr    = PetscFunctionListDuplicate(((PetscObject)matin)->qlist,&((PetscObject)mat)->qlist);CHKERRQ(ierr);
  *newmat = mat;
  PetscFunctionReturn(0);
}

/* Used for both MPIBAIJ and MPISBAIJ matrices */
PetscErrorCode MatLoad_MPIBAIJ_Binary(Mat mat,PetscViewer viewer)
{
  PetscInt       header[4],M,N,nz,bs,m,n,mbs,nbs,rows,cols,sum,i,j,k;
  PetscInt       *rowidxs,*colidxs,rs,cs,ce;
  PetscScalar    *matvals;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);

  /* read in matrix header */
  ierr = PetscViewerBinaryRead(viewer,header,4,NULL,PETSC_INT);CHKERRQ(ierr);
  PetscCheckFalse(header[0] != MAT_FILE_CLASSID,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Not a matrix object in file");
  M  = header[1]; N = header[2]; nz = header[3];
  PetscCheckFalse(M < 0,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Matrix row size (%" PetscInt_FMT ") in file is negative",M);
  PetscCheckFalse(N < 0,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Matrix column size (%" PetscInt_FMT ") in file is negative",N);
  PetscCheckFalse(nz < 0,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format on disk, cannot load as MPIBAIJ");

  /* set block sizes from the viewer's .info file */
  ierr = MatLoad_Binary_BlockSizes(mat,viewer);CHKERRQ(ierr);
  /* set local sizes if not set already */
  if (mat->rmap->n < 0 && M == N) mat->rmap->n = mat->cmap->n;
  if (mat->cmap->n < 0 && M == N) mat->cmap->n = mat->rmap->n;
  /* set global sizes if not set already */
  if (mat->rmap->N < 0) mat->rmap->N = M;
  if (mat->cmap->N < 0) mat->cmap->N = N;
  ierr = PetscLayoutSetUp(mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(mat->cmap);CHKERRQ(ierr);

  /* check if the matrix sizes are correct */
  ierr = MatGetSize(mat,&rows,&cols);CHKERRQ(ierr);
  PetscCheckFalse(M != rows || N != cols,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Matrix in file of different sizes (%" PetscInt_FMT ", %" PetscInt_FMT ") than the input matrix (%" PetscInt_FMT ", %" PetscInt_FMT ")",M,N,rows,cols);
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(mat->rmap,&rs,NULL);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(mat->cmap,&cs,&ce);CHKERRQ(ierr);
  mbs = m/bs; nbs = n/bs;

  /* read in row lengths and build row indices */
  ierr = PetscMalloc1(m+1,&rowidxs);CHKERRQ(ierr);
  ierr = PetscViewerBinaryReadAll(viewer,rowidxs+1,m,PETSC_DECIDE,M,PETSC_INT);CHKERRQ(ierr);
  rowidxs[0] = 0; for (i=0; i<m; i++) rowidxs[i+1] += rowidxs[i];
  ierr = MPIU_Allreduce(&rowidxs[m],&sum,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)viewer));CHKERRMPI(ierr);
  PetscCheckFalse(sum != nz,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Inconsistent matrix data in file: nonzeros = %" PetscInt_FMT ", sum-row-lengths = %" PetscInt_FMT,nz,sum);

  /* read in column indices and matrix values */
  ierr = PetscMalloc2(rowidxs[m],&colidxs,rowidxs[m],&matvals);CHKERRQ(ierr);
  ierr = PetscViewerBinaryReadAll(viewer,colidxs,rowidxs[m],PETSC_DETERMINE,PETSC_DETERMINE,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryReadAll(viewer,matvals,rowidxs[m],PETSC_DETERMINE,PETSC_DETERMINE,PETSC_SCALAR);CHKERRQ(ierr);

  { /* preallocate matrix storage */
    PetscBT    bt; /* helper bit set to count diagonal nonzeros */
    PetscHSetI ht; /* helper hash set to count off-diagonal nonzeros */
    PetscBool  sbaij,done;
    PetscInt   *d_nnz,*o_nnz;

    ierr = PetscBTCreate(nbs,&bt);CHKERRQ(ierr);
    ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
    ierr = PetscCalloc2(mbs,&d_nnz,mbs,&o_nnz);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPISBAIJ,&sbaij);CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      ierr = PetscBTMemzero(nbs,bt);CHKERRQ(ierr);
      ierr = PetscHSetIClear(ht);CHKERRQ(ierr);
      for (k=0; k<bs; k++) {
        PetscInt row = bs*i + k;
        for (j=rowidxs[row]; j<rowidxs[row+1]; j++) {
          PetscInt col = colidxs[j];
          if (!sbaij || col >= row) {
            if (col >= cs && col < ce) {
              if (!PetscBTLookupSet(bt,(col-cs)/bs)) d_nnz[i]++;
            } else {
              ierr = PetscHSetIQueryAdd(ht,col/bs,&done);CHKERRQ(ierr);
              if (done) o_nnz[i]++;
            }
          }
        }
      }
    }
    ierr = PetscBTDestroy(&bt);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&ht);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(mat,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(mat,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
    ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);
  }

  /* store matrix values */
  for (i=0; i<m; i++) {
    PetscInt row = rs + i, s = rowidxs[i], e = rowidxs[i+1];
    ierr = (*mat->ops->setvalues)(mat,1,&row,e-s,colidxs+s,matvals+s,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = PetscFree(rowidxs);CHKERRQ(ierr);
  ierr = PetscFree2(colidxs,matvals);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_MPIBAIJ(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  PetscCheckFalse(!isbinary,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer type %s not yet supported for reading %s matrices",((PetscObject)viewer)->type_name,((PetscObject)mat)->type_name);
  ierr = MatLoad_MPIBAIJ_Binary(mat,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatMPIBAIJSetHashTableFactor - Sets the factor required to compute the size of the HashTable.

   Input Parameters:
+  mat  - the matrix
-  fact - factor

   Not Collective, each process can use a different factor

   Level: advanced

  Notes:
   This can also be set by the command line option: -mat_use_hash_table <fact>

.seealso: MatSetOption()
@*/
PetscErrorCode  MatMPIBAIJSetHashTableFactor(Mat mat,PetscReal fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(mat,"MatSetHashTableFactor_C",(Mat,PetscReal),(mat,fact));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatSetHashTableFactor_MPIBAIJ(Mat mat,PetscReal fact)
{
  Mat_MPIBAIJ *baij;

  PetscFunctionBegin;
  baij          = (Mat_MPIBAIJ*)mat->data;
  baij->ht_fact = fact;
  PetscFunctionReturn(0);
}

PetscErrorCode  MatMPIBAIJGetSeqBAIJ(Mat A,Mat *Ad,Mat *Ao,const PetscInt *colmap[])
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIBAIJ,&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"This function requires a MATMPIBAIJ matrix as input");
  if (Ad)     *Ad     = a->A;
  if (Ao)     *Ao     = a->B;
  if (colmap) *colmap = a->garray;
  PetscFunctionReturn(0);
}

/*
    Special version for direct calls from Fortran (to eliminate two function call overheads
*/
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matmpibaijsetvaluesblocked_ MATMPIBAIJSETVALUESBLOCKED
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matmpibaijsetvaluesblocked_ matmpibaijsetvaluesblocked
#endif

/*@C
  MatMPIBAIJSetValuesBlocked - Direct Fortran call to replace call to MatSetValuesBlocked()

  Collective on Mat

  Input Parameters:
+ mat - the matrix
. min - number of input rows
. im - input rows
. nin - number of input columns
. in - input columns
. v - numerical values input
- addvin - INSERT_VALUES or ADD_VALUES

  Notes:
    This has a complete copy of MatSetValuesBlocked_MPIBAIJ() which is terrible code un-reuse.

  Level: advanced

.seealso:   MatSetValuesBlocked()
@*/
PetscErrorCode matmpibaijsetvaluesblocked_(Mat *matin,PetscInt *min,const PetscInt im[],PetscInt *nin,const PetscInt in[],const MatScalar v[],InsertMode *addvin)
{
  /* convert input arguments to C version */
  Mat        mat  = *matin;
  PetscInt   m    = *min, n = *nin;
  InsertMode addv = *addvin;

  Mat_MPIBAIJ     *baij = (Mat_MPIBAIJ*)mat->data;
  const MatScalar *value;
  MatScalar       *barray     = baij->barray;
  PetscBool       roworiented = baij->roworiented;
  PetscErrorCode  ierr;
  PetscInt        i,j,ii,jj,row,col,rstart=baij->rstartbs;
  PetscInt        rend=baij->rendbs,cstart=baij->cstartbs,stepval;
  PetscInt        cend=baij->cendbs,bs=mat->rmap->bs,bs2=baij->bs2;

  PetscFunctionBegin;
  /* tasks normally handled by MatSetValuesBlocked() */
  if (mat->insertmode == NOT_SET_VALUES) mat->insertmode = addv;
  else PetscCheckFalse(mat->insertmode != addv,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  PetscCheckFalse(mat->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE;
    mat->assembled     = PETSC_FALSE;
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);

  if (!barray) {
    ierr         = PetscMalloc1(bs2,&barray);CHKERRQ(ierr);
    baij->barray = barray;
  }

  if (roworiented) stepval = (n-1)*bs;
  else stepval = (m-1)*bs;

  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
    PetscCheck(im[i] < baij->Mbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large, row %" PetscInt_FMT " max %" PetscInt_FMT,im[i],baij->Mbs-1);
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for (j=0; j<n; j++) {
        /* If NumCol = 1 then a copy is not required */
        if ((roworiented) && (n == 1)) {
          barray = (MatScalar*)v + i*bs2;
        } else if ((!roworiented) && (m == 1)) {
          barray = (MatScalar*)v + j*bs2;
        } else { /* Here a copy is required */
          if (roworiented) {
            value = v + i*(stepval+bs)*bs + j*bs;
          } else {
            value = v + j*(stepval+bs)*bs + i*bs;
          }
          for (ii=0; ii<bs; ii++,value+=stepval) {
            for (jj=0; jj<bs; jj++) {
              *barray++ = *value++;
            }
          }
          barray -=bs2;
        }

        if (in[j] >= cstart && in[j] < cend) {
          col  = in[j] - cstart;
          ierr = MatSetValuesBlocked_SeqBAIJ_Inlined(baij->A,row,col,barray,addv,im[i],in[j]);CHKERRQ(ierr);
        } else if (in[j] < 0) continue;
        else PetscCheck(in[j] < baij->Nbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large, col %" PetscInt_FMT " max %" PetscInt_FMT,in[j],baij->Nbs-1);
        else {
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = MatCreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
            }

#if defined(PETSC_USE_DEBUG)
#if defined(PETSC_USE_CTABLE)
            { PetscInt data;
              ierr = PetscTableFind(baij->colmap,in[j]+1,&data);CHKERRQ(ierr);
              PetscCheckFalse((data - 1) % bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect colmap");
            }
#else
            PetscCheckFalse((baij->colmap[in[j]] - 1) % bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect colmap");
#endif
#endif
#if defined(PETSC_USE_CTABLE)
            ierr = PetscTableFind(baij->colmap,in[j]+1,&col);CHKERRQ(ierr);
            col  = (col - 1)/bs;
#else
            col = (baij->colmap[in[j]] - 1)/bs;
#endif
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
              ierr = MatDisAssemble_MPIBAIJ(mat);CHKERRQ(ierr);
              col  =  in[j];
            }
          } else col = in[j];
          ierr = MatSetValuesBlocked_SeqBAIJ_Inlined(baij->B,row,col,barray,addv,im[i],in[j]);CHKERRQ(ierr);
        }
      }
    } else {
      if (!baij->donotstash) {
        if (roworiented) {
          ierr = MatStashValuesRowBlocked_Private(&mat->bstash,im[i],n,in,v,m,n,i);CHKERRQ(ierr);
        } else {
          ierr = MatStashValuesColBlocked_Private(&mat->bstash,im[i],n,in,v,m,n,i);CHKERRQ(ierr);
        }
      }
    }
  }

  /* task normally handled by MatSetValuesBlocked() */
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
     MatCreateMPIBAIJWithArrays - creates a MPI BAIJ matrix using arrays that contain in standard block
         CSR format the local rows.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  bs - the block size, only a block size of 1 is supported
.  m - number of local rows (Cannot be PETSC_DECIDE)
.  n - This value should be the same as the local size used in creating the
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if N is given) For square matrices n is almost always m.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.   i - row indices; that is i[0] = 0, i[row] = i[row-1] + number of block elements in that rowth block row of the matrix
.   j - column indices
-   a - matrix values

   Output Parameter:
.   mat - the matrix

   Level: intermediate

   Notes:
       The i, j, and a arrays ARE copied by this routine into the internal format used by PETSc;
     thus you CANNOT change the matrix entries by changing the values of a[] after you have
     called this routine. Use MatCreateMPIAIJWithSplitArrays() to avoid needing to copy the arrays.

     The order of the entries in values is the same as the block compressed sparse row storage format; that is, it is
     the same as a three dimensional array in Fortran values(bs,bs,nnz) that contains the first column of the first
     block, followed by the second column of the first block etc etc.  That is, the blocks are contiguous in memory
     with column-major ordering within blocks.

       The i and j indices are 0 based, and i indices are indices corresponding to the local j array.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIAIJSetPreallocation(), MatMPIAIJSetPreallocationCSR(),
          MPIAIJ, MatCreateAIJ(), MatCreateMPIAIJWithSplitArrays()
@*/
PetscErrorCode  MatCreateMPIBAIJWithArrays(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt i[],const PetscInt j[],const PetscScalar a[],Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(i[0],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  PetscCheckFalse(m < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"local number of rows (m) cannot be PETSC_DECIDE, or negative");
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATMPIBAIJ);CHKERRQ(ierr);
  ierr = MatSetBlockSize(*mat,bs);CHKERRQ(ierr);
  ierr = MatSetUp(*mat);CHKERRQ(ierr);
  ierr = MatSetOption(*mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocationCSR(*mat,bs,i,j,a);CHKERRQ(ierr);
  ierr = MatSetOption(*mat,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_MPIBAIJ(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  PetscErrorCode ierr;
  PetscInt       m,N,i,rstart,nnz,Ii,bs,cbs;
  PetscInt       *indx;
  PetscScalar    *values;

  PetscFunctionBegin;
  ierr = MatGetSize(inmat,&m,&N);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX) { /* symbolic phase */
    Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)inmat->data;
    PetscInt       *dnz,*onz,mbs,Nbs,nbs;
    PetscInt       *bindx,rmax=a->rmax,j;
    PetscMPIInt    rank,size;

    ierr = MatGetBlockSizes(inmat,&bs,&cbs);CHKERRQ(ierr);
    mbs = m/bs; Nbs = N/cbs;
    if (n == PETSC_DECIDE) {
      ierr = PetscSplitOwnershipBlock(comm,cbs,&n,&N);CHKERRQ(ierr);
    }
    nbs = n/cbs;

    ierr = PetscMalloc1(rmax,&bindx);CHKERRQ(ierr);
    ierr = MatPreallocateInitialize(comm,mbs,nbs,dnz,onz);CHKERRQ(ierr); /* inline function, output __end and __rstart are used below */

    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
    ierr = MPI_Comm_rank(comm,&size);CHKERRMPI(ierr);
    if (rank == size-1) {
      /* Check sum(nbs) = Nbs */
      PetscCheckFalse(__end != Nbs,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sum of local block columns %" PetscInt_FMT " != global block columns %" PetscInt_FMT,__end,Nbs);
    }

    rstart = __rstart; /* block rstart of *outmat; see inline function MatPreallocateInitialize */
    for (i=0; i<mbs; i++) {
      ierr = MatGetRow_SeqBAIJ(inmat,i*bs,&nnz,&indx,NULL);CHKERRQ(ierr); /* non-blocked nnz and indx */
      nnz = nnz/bs;
      for (j=0; j<nnz; j++) bindx[j] = indx[j*bs]/bs;
      ierr = MatPreallocateSet(i+rstart,nnz,bindx,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow_SeqBAIJ(inmat,i*bs,&nnz,&indx,NULL);CHKERRQ(ierr);
    }
    ierr = PetscFree(bindx);CHKERRQ(ierr);

    ierr = MatCreate(comm,outmat);CHKERRQ(ierr);
    ierr = MatSetSizes(*outmat,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(*outmat,bs,cbs);CHKERRQ(ierr);
    ierr = MatSetType(*outmat,MATBAIJ);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation(*outmat,bs,0,dnz);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(*outmat,bs,0,dnz,0,onz);CHKERRQ(ierr);
    ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
    ierr = MatSetOption(*outmat,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* numeric phase */
  ierr = MatGetBlockSizes(inmat,&bs,&cbs);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*outmat,&rstart,NULL);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    ierr = MatGetRow_SeqBAIJ(inmat,i,&nnz,&indx,&values);CHKERRQ(ierr);
    Ii   = i + rstart;
    ierr = MatSetValues(*outmat,1,&Ii,nnz,indx,values,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow_SeqBAIJ(inmat,i,&nnz,&indx,&values);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*outmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*outmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
