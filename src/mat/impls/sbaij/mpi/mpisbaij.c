
#include <../src/mat/impls/baij/mpi/mpibaij.h>    /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <petscblaslapack.h>

#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatConvert_MPISBAIJ_Elemental(Mat,MatType,MatReuse,Mat*);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
PETSC_INTERN PetscErrorCode MatConvert_SBAIJ_ScaLAPACK(Mat,MatType,MatReuse,Mat*);
#endif

/* This could be moved to matimpl.h */
static PetscErrorCode MatPreallocateWithMats_Private(Mat B, PetscInt nm, Mat X[], PetscBool symm[], PetscBool fill)
{
  Mat            preallocator;
  PetscInt       r,rstart,rend;
  PetscInt       bs,i,m,n,M,N;
  PetscBool      cong = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(B,nm,2);
  for (i = 0; i < nm; i++) {
    PetscValidHeaderSpecific(X[i],MAT_CLASSID,3);
    PetscCall(PetscLayoutCompare(B->rmap,X[i]->rmap,&cong));
    PetscCheck(cong,PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"Not for different layouts");
  }
  PetscValidLogicalCollectiveBool(B,fill,5);
  PetscCall(MatGetBlockSize(B,&bs));
  PetscCall(MatGetSize(B,&M,&N));
  PetscCall(MatGetLocalSize(B,&m,&n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)B),&preallocator));
  PetscCall(MatSetType(preallocator,MATPREALLOCATOR));
  PetscCall(MatSetBlockSize(preallocator,bs));
  PetscCall(MatSetSizes(preallocator,m,n,M,N));
  PetscCall(MatSetUp(preallocator));
  PetscCall(MatGetOwnershipRange(preallocator,&rstart,&rend));
  for (r = rstart; r < rend; ++r) {
    PetscInt          ncols;
    const PetscInt    *row;
    const PetscScalar *vals;

    for (i = 0; i < nm; i++) {
      PetscCall(MatGetRow(X[i],r,&ncols,&row,&vals));
      PetscCall(MatSetValues(preallocator,1,&r,ncols,row,vals,INSERT_VALUES));
      if (symm && symm[i]) {
        PetscCall(MatSetValues(preallocator,ncols,row,1,&r,vals,INSERT_VALUES));
      }
      PetscCall(MatRestoreRow(X[i],r,&ncols,&row,&vals));
    }
  }
  PetscCall(MatAssemblyBegin(preallocator,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(preallocator,MAT_FINAL_ASSEMBLY));
  PetscCall(MatPreallocatorPreallocate(preallocator,fill,B));
  PetscCall(MatDestroy(&preallocator));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPISBAIJ_Basic(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat            B;
  PetscInt       r;

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    PetscBool symm = PETSC_TRUE,isdense;
    PetscInt  bs;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
    PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
    PetscCall(MatSetType(B,newtype));
    PetscCall(MatGetBlockSize(A,&bs));
    PetscCall(MatSetBlockSize(B,bs));
    PetscCall(PetscLayoutSetUp(B->rmap));
    PetscCall(PetscLayoutSetUp(B->cmap));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)B,&isdense,MATSEQDENSE,MATMPIDENSE,MATSEQDENSECUDA,""));
    if (!isdense) {
      PetscCall(MatGetRowUpperTriangular(A));
      PetscCall(MatPreallocateWithMats_Private(B,1,&A,&symm,PETSC_TRUE));
      PetscCall(MatRestoreRowUpperTriangular(A));
    } else {
      PetscCall(MatSetUp(B));
    }
  } else {
    B    = *newmat;
    PetscCall(MatZeroEntries(B));
  }

  PetscCall(MatGetRowUpperTriangular(A));
  for (r = A->rmap->rstart; r < A->rmap->rend; r++) {
    PetscInt          ncols;
    const PetscInt    *row;
    const PetscScalar *vals;

    PetscCall(MatGetRow(A,r,&ncols,&row,&vals));
    PetscCall(MatSetValues(B,1,&r,ncols,row,vals,INSERT_VALUES));
#if defined(PETSC_USE_COMPLEX)
    if (A->hermitian) {
      PetscInt i;
      for (i = 0; i < ncols; i++) {
        PetscCall(MatSetValue(B,row[i],r,PetscConj(vals[i]),INSERT_VALUES));
      }
    } else {
      PetscCall(MatSetValues(B,ncols,row,1,&r,vals,INSERT_VALUES));
    }
#else
    PetscCall(MatSetValues(B,ncols,row,1,&r,vals,INSERT_VALUES));
#endif
    PetscCall(MatRestoreRow(A,r,&ncols,&row,&vals));
  }
  PetscCall(MatRestoreRowUpperTriangular(A));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&B));
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatStoreValues_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *aij = (Mat_MPISBAIJ*)mat->data;

  PetscFunctionBegin;
  PetscCall(MatStoreValues(aij->A));
  PetscCall(MatStoreValues(aij->B));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatRetrieveValues_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *aij = (Mat_MPISBAIJ*)mat->data;

  PetscFunctionBegin;
  PetscCall(MatRetrieveValues(aij->A));
  PetscCall(MatRetrieveValues(aij->B));
  PetscFunctionReturn(0);
}

#define  MatSetValues_SeqSBAIJ_A_Private(row,col,value,addv,orow,ocol)      \
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
        bap = ap + bs2*_i + bs*cidx + ridx; \
        if (addv == ADD_VALUES) *bap += value;  \
        else                    *bap  = value;  \
        goto a_noinsert; \
      } \
    } \
    if (a->nonew == 1) goto a_noinsert; \
    PetscCheck(a->nonew != -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,aimax,a->nonew,MatScalar); \
    N = nrow++ - 1;  \
    /* shift up all the later entries in this row */ \
    PetscCall(PetscArraymove(rp+_i+1,rp+_i,N-_i+1)); \
    PetscCall(PetscArraymove(ap+bs2*(_i+1),ap+bs2*_i,bs2*(N-_i+1))); \
    PetscCall(PetscArrayzero(ap+bs2*_i,bs2));  \
    rp[_i]                      = bcol;  \
    ap[bs2*_i + bs*cidx + ridx] = value;  \
    A->nonzerostate++;\
a_noinsert:; \
    ailen[brow] = nrow; \
  }

#define  MatSetValues_SeqSBAIJ_B_Private(row,col,value,addv,orow,ocol) \
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
        bap = ap + bs2*_i + bs*cidx + ridx; \
        if (addv == ADD_VALUES) *bap += value;  \
        else                    *bap  = value;  \
        goto b_noinsert; \
      } \
    } \
    if (b->nonew == 1) goto b_noinsert; \
    PetscCheck(b->nonew != -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(B,b->mbs,bs2,nrow,brow,bcol,rmax,ba,bi,bj,rp,ap,bimax,b->nonew,MatScalar); \
    N = nrow++ - 1;  \
    /* shift up all the later entries in this row */ \
    PetscCall(PetscArraymove(rp+_i+1,rp+_i,N-_i+1)); \
    PetscCall(PetscArraymove(ap+bs2*(_i+1),ap+bs2*_i,bs2*(N-_i+1))); \
    PetscCall(PetscArrayzero(ap+bs2*_i,bs2)); \
    rp[_i]                      = bcol;  \
    ap[bs2*_i + bs*cidx + ridx] = value;  \
    B->nonzerostate++;\
b_noinsert:; \
    bilen[brow] = nrow; \
  }

/* Only add/insert a(i,j) with i<=j (blocks).
   Any a(i,j) with i>j input by user is ingored or generates an error
*/
PetscErrorCode MatSetValues_MPISBAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  MatScalar      value;
  PetscBool      roworiented = baij->roworiented;
  PetscInt       i,j,row,col;
  PetscInt       rstart_orig=mat->rmap->rstart;
  PetscInt       rend_orig  =mat->rmap->rend,cstart_orig=mat->cmap->rstart;
  PetscInt       cend_orig  =mat->cmap->rend,bs=mat->rmap->bs;

  /* Some Variables required in the macro */
  Mat          A     = baij->A;
  Mat_SeqSBAIJ *a    = (Mat_SeqSBAIJ*)(A)->data;
  PetscInt     *aimax=a->imax,*ai=a->i,*ailen=a->ilen,*aj=a->j;
  MatScalar    *aa   =a->a;

  Mat         B     = baij->B;
  Mat_SeqBAIJ *b    = (Mat_SeqBAIJ*)(B)->data;
  PetscInt    *bimax=b->imax,*bi=b->i,*bilen=b->ilen,*bj=b->j;
  MatScalar   *ba   =b->a;

  PetscInt  *rp,ii,nrow,_i,rmax,N,brow,bcol;
  PetscInt  low,high,t,ridx,cidx,bs2=a->bs2;
  MatScalar *ap,*bap;

  /* for stash */
  PetscInt  n_loc, *in_loc = NULL;
  MatScalar *v_loc = NULL;

  PetscFunctionBegin;
  if (!baij->donotstash) {
    if (n > baij->n_loc) {
      PetscCall(PetscFree(baij->in_loc));
      PetscCall(PetscFree(baij->v_loc));
      PetscCall(PetscMalloc1(n,&baij->in_loc));
      PetscCall(PetscMalloc1(n,&baij->v_loc));

      baij->n_loc = n;
    }
    in_loc = baij->in_loc;
    v_loc  = baij->v_loc;
  }

  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
    PetscCheck(im[i] < mat->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,im[i],mat->rmap->N-1);
    if (im[i] >= rstart_orig && im[i] < rend_orig) { /* this processor entry */
      row = im[i] - rstart_orig;              /* local row index */
      for (j=0; j<n; j++) {
        if (im[i]/bs > in[j]/bs) {
          if (a->ignore_ltriangular) {
            continue;    /* ignore lower triangular blocks */
          } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Lower triangular value cannot be set for sbaij format. Ignoring these values, run with -mat_ignore_lower_triangular or call MatSetOption(mat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE)");
        }
        if (in[j] >= cstart_orig && in[j] < cend_orig) {  /* diag entry (A) */
          col  = in[j] - cstart_orig;         /* local col index */
          brow = row/bs; bcol = col/bs;
          if (brow > bcol) continue;  /* ignore lower triangular blocks of A */
          if (roworiented) value = v[i*n+j];
          else             value = v[i+j*m];
          MatSetValues_SeqSBAIJ_A_Private(row,col,value,addv,im[i],in[j]);
          /* PetscCall(MatSetValues_SeqBAIJ(baij->A,1,&row,1,&col,&value,addv)); */
        } else if (in[j] < 0) continue;
        else PetscCheck(in[j] < mat->cmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,in[j],mat->cmap->N-1);
        else {  /* off-diag entry (B) */
          if (mat->was_assembled) {
            if (!baij->colmap) {
              PetscCall(MatCreateColmap_MPIBAIJ_Private(mat));
            }
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscTableFind(baij->colmap,in[j]/bs + 1,&col));
            col  = col - 1;
#else
            col = baij->colmap[in[j]/bs] - 1;
#endif
            if (col < 0 && !((Mat_SeqSBAIJ*)(baij->A->data))->nonew) {
              PetscCall(MatDisAssemble_MPISBAIJ(mat));
              col  =  in[j];
              /* Reinitialize the variables required by MatSetValues_SeqBAIJ_B_Private() */
              B    = baij->B;
              b    = (Mat_SeqBAIJ*)(B)->data;
              bimax= b->imax;bi=b->i;bilen=b->ilen;bj=b->j;
              ba   = b->a;
            } else col += in[j]%bs;
          } else col = in[j];
          if (roworiented) value = v[i*n+j];
          else             value = v[i+j*m];
          MatSetValues_SeqSBAIJ_B_Private(row,col,value,addv,im[i],in[j]);
          /* PetscCall(MatSetValues_SeqBAIJ(baij->B,1,&row,1,&col,&value,addv)); */
        }
      }
    } else {  /* off processor entry */
      PetscCheck(!mat->nooffprocentries,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
      if (!baij->donotstash) {
        mat->assembled = PETSC_FALSE;
        n_loc          = 0;
        for (j=0; j<n; j++) {
          if (im[i]/bs > in[j]/bs) continue; /* ignore lower triangular blocks */
          in_loc[n_loc] = in[j];
          if (roworiented) {
            v_loc[n_loc] = v[i*n+j];
          } else {
            v_loc[n_loc] = v[j*m+i];
          }
          n_loc++;
        }
        PetscCall(MatStashValuesRow_Private(&mat->stash,im[i],n_loc,in_loc,v_loc,PETSC_FALSE));
      }
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatSetValuesBlocked_SeqSBAIJ_Inlined(Mat A,PetscInt row,PetscInt col,const PetscScalar v[],InsertMode is,PetscInt orow,PetscInt ocol)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  PetscInt          *rp,low,high,t,ii,jj,nrow,i,rmax,N;
  PetscInt          *imax      =a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt          *aj        =a->j,nonew=a->nonew,bs2=a->bs2,bs=A->rmap->bs;
  PetscBool         roworiented=a->roworiented;
  const PetscScalar *value     = v;
  MatScalar         *ap,*aa = a->a,*bap;

  PetscFunctionBegin;
  if (col < row) {
    if (a->ignore_ltriangular) PetscFunctionReturn(0); /* ignore lower triangular block */
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Lower triangular value cannot be set for sbaij format. Ignoring these values, run with -mat_ignore_lower_triangular or call MatSetOption(mat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE)");
  }
  rp   = aj + ai[row];
  ap   = aa + bs2*ai[row];
  rmax = imax[row];
  nrow = ailen[row];
  value = v;
  low   = 0;
  high  = nrow;

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
          for (ii=0; ii<bs; ii++) {
            for (jj=0; jj<bs; jj++) {
              *bap++ += *value++;
            }
          }
        } else {
          for (ii=0; ii<bs; ii++) {
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
  PetscCheck(nonew != -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new block index nonzero block (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", orow, ocol);
  MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
  N = nrow++ - 1; high++;
  /* shift up all the later entries in this row */
  PetscCall(PetscArraymove(rp+i+1,rp+i,N-i+1));
  PetscCall(PetscArraymove(ap+bs2*(i+1),ap+bs2*i,bs2*(N-i+1)));
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
   This routine is exactly duplicated in mpibaij.c
*/
static inline PetscErrorCode MatSetValuesBlocked_SeqBAIJ_Inlined(Mat A,PetscInt row,PetscInt col,const PetscScalar v[],InsertMode is,PetscInt orow,PetscInt ocol)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscInt          *rp,low,high,t,ii,jj,nrow,i,rmax,N;
  PetscInt          *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt          *aj        =a->j,nonew=a->nonew,bs2=a->bs2,bs=A->rmap->bs;
  PetscBool         roworiented=a->roworiented;
  const PetscScalar *value     = v;
  MatScalar         *ap,*aa = a->a,*bap;

  PetscFunctionBegin;
  rp   = aj + ai[row];
  ap   = aa + bs2*ai[row];
  rmax = imax[row];
  nrow = ailen[row];
  low  = 0;
  high = nrow;
  value = v;
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
  PetscCheck(nonew != -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new global block indexed nonzero block (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", orow, ocol);
  MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
  N = nrow++ - 1; high++;
  /* shift up all the later entries in this row */
  PetscCall(PetscArraymove(rp+i+1,rp+i,N-i+1));
  PetscCall(PetscArraymove(ap+bs2*(i+1),ap+bs2*i,bs2*(N-i+1)));
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
    This routine could be optimized by removing the need for the block copy below and passing stride information
  to the above inline routines; similarly in MatSetValuesBlocked_MPIBAIJ()
*/
PetscErrorCode MatSetValuesBlocked_MPISBAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const MatScalar v[],InsertMode addv)
{
  Mat_MPISBAIJ    *baij = (Mat_MPISBAIJ*)mat->data;
  const MatScalar *value;
  MatScalar       *barray     =baij->barray;
  PetscBool       roworiented = baij->roworiented,ignore_ltriangular = ((Mat_SeqSBAIJ*)baij->A->data)->ignore_ltriangular;
  PetscInt        i,j,ii,jj,row,col,rstart=baij->rstartbs;
  PetscInt        rend=baij->rendbs,cstart=baij->cstartbs,stepval;
  PetscInt        cend=baij->cendbs,bs=mat->rmap->bs,bs2=baij->bs2;

  PetscFunctionBegin;
  if (!barray) {
    PetscCall(PetscMalloc1(bs2,&barray));
    baij->barray = barray;
  }

  if (roworiented) {
    stepval = (n-1)*bs;
  } else {
    stepval = (m-1)*bs;
  }
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
    PetscCheck(im[i] < baij->Mbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block indexed row too large %" PetscInt_FMT " max %" PetscInt_FMT,im[i],baij->Mbs-1);
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for (j=0; j<n; j++) {
        if (im[i] > in[j]) {
          if (ignore_ltriangular) continue; /* ignore lower triangular blocks */
          else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Lower triangular value cannot be set for sbaij format. Ignoring these values, run with -mat_ignore_lower_triangular or call MatSetOption(mat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE)");
        }
        /* If NumCol = 1 then a copy is not required */
        if ((roworiented) && (n == 1)) {
          barray = (MatScalar*) v + i*bs2;
        } else if ((!roworiented) && (m == 1)) {
          barray = (MatScalar*) v + j*bs2;
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
          PetscCall(MatSetValuesBlocked_SeqSBAIJ_Inlined(baij->A,row,col,barray,addv,im[i],in[j]));
        } else if (in[j] < 0) continue;
        else PetscCheck(in[j] < baij->Nbs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block indexed column too large %" PetscInt_FMT " max %" PetscInt_FMT,in[j],baij->Nbs-1);
        else {
          if (mat->was_assembled) {
            if (!baij->colmap) {
              PetscCall(MatCreateColmap_MPIBAIJ_Private(mat));
            }

#if defined(PETSC_USE_DEBUG)
#if defined(PETSC_USE_CTABLE)
            { PetscInt data;
              PetscCall(PetscTableFind(baij->colmap,in[j]+1,&data));
              PetscCheckFalse((data - 1) % bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect colmap");
            }
#else
            PetscCheckFalse((baij->colmap[in[j]] - 1) % bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect colmap");
#endif
#endif
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscTableFind(baij->colmap,in[j]+1,&col));
            col  = (col - 1)/bs;
#else
            col = (baij->colmap[in[j]] - 1)/bs;
#endif
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
              PetscCall(MatDisAssemble_MPISBAIJ(mat));
              col  = in[j];
            }
          } else col = in[j];
          PetscCall(MatSetValuesBlocked_SeqBAIJ_Inlined(baij->B,row,col,barray,addv,im[i],in[j]));
        }
      }
    } else {
      PetscCheck(!mat->nooffprocentries,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process block indexed row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
      if (!baij->donotstash) {
        if (roworiented) {
          PetscCall(MatStashValuesRowBlocked_Private(&mat->bstash,im[i],n,in,v,m,n,i));
        } else {
          PetscCall(MatStashValuesColBlocked_Private(&mat->bstash,im[i],n,in,v,m,n,i));
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetValues_MPISBAIJ(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
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
          PetscCall(MatGetValues_SeqSBAIJ(baij->A,1,&row,1,&col,v+i*n+j));
        } else {
          if (!baij->colmap) {
            PetscCall(MatCreateColmap_MPIBAIJ_Private(mat));
          }
#if defined(PETSC_USE_CTABLE)
          PetscCall(PetscTableFind(baij->colmap,idxn[j]/bs+1,&data));
          data--;
#else
          data = baij->colmap[idxn[j]/bs]-1;
#endif
          if ((data < 0) || (baij->garray[data/bs] != idxn[j]/bs)) *(v+i*n+j) = 0.0;
          else {
            col  = data + idxn[j]%bs;
            PetscCall(MatGetValues_SeqBAIJ(baij->B,1,&row,1,&col,v+i*n+j));
          }
        }
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local values currently supported");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatNorm_MPISBAIJ(Mat mat,NormType type,PetscReal *norm)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  PetscReal      sum[2],*lnorm2;

  PetscFunctionBegin;
  if (baij->size == 1) {
    PetscCall(MatNorm(baij->A,type,norm));
  } else {
    if (type == NORM_FROBENIUS) {
      PetscCall(PetscMalloc1(2,&lnorm2));
      PetscCall(MatNorm(baij->A,type,lnorm2));
      *lnorm2 = (*lnorm2)*(*lnorm2); lnorm2++;            /* squar power of norm(A) */
      PetscCall(MatNorm(baij->B,type,lnorm2));
      *lnorm2 = (*lnorm2)*(*lnorm2); lnorm2--;             /* squar power of norm(B) */
      PetscCall(MPIU_Allreduce(lnorm2,sum,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)mat)));
      *norm   = PetscSqrtReal(sum[0] + 2*sum[1]);
      PetscCall(PetscFree(lnorm2));
    } else if (type == NORM_INFINITY || type == NORM_1) { /* max row/column sum */
      Mat_SeqSBAIJ *amat=(Mat_SeqSBAIJ*)baij->A->data;
      Mat_SeqBAIJ  *bmat=(Mat_SeqBAIJ*)baij->B->data;
      PetscReal    *rsum,*rsum2,vabs;
      PetscInt     *jj,*garray=baij->garray,rstart=baij->rstartbs,nz;
      PetscInt     brow,bcol,col,bs=baij->A->rmap->bs,row,grow,gcol,mbs=amat->mbs;
      MatScalar    *v;

      PetscCall(PetscMalloc2(mat->cmap->N,&rsum,mat->cmap->N,&rsum2));
      PetscCall(PetscArrayzero(rsum,mat->cmap->N));
      /* Amat */
      v = amat->a; jj = amat->j;
      for (brow=0; brow<mbs; brow++) {
        grow = bs*(rstart + brow);
        nz   = amat->i[brow+1] - amat->i[brow];
        for (bcol=0; bcol<nz; bcol++) {
          gcol = bs*(rstart + *jj); jj++;
          for (col=0; col<bs; col++) {
            for (row=0; row<bs; row++) {
              vabs            = PetscAbsScalar(*v); v++;
              rsum[gcol+col] += vabs;
              /* non-diagonal block */
              if (bcol > 0 && vabs > 0.0) rsum[grow+row] += vabs;
            }
          }
        }
        PetscCall(PetscLogFlops(nz*bs*bs));
      }
      /* Bmat */
      v = bmat->a; jj = bmat->j;
      for (brow=0; brow<mbs; brow++) {
        grow = bs*(rstart + brow);
        nz = bmat->i[brow+1] - bmat->i[brow];
        for (bcol=0; bcol<nz; bcol++) {
          gcol = bs*garray[*jj]; jj++;
          for (col=0; col<bs; col++) {
            for (row=0; row<bs; row++) {
              vabs            = PetscAbsScalar(*v); v++;
              rsum[gcol+col] += vabs;
              rsum[grow+row] += vabs;
            }
          }
        }
        PetscCall(PetscLogFlops(nz*bs*bs));
      }
      PetscCall(MPIU_Allreduce(rsum,rsum2,mat->cmap->N,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)mat)));
      *norm = 0.0;
      for (col=0; col<mat->cmap->N; col++) {
        if (rsum2[col] > *norm) *norm = rsum2[col];
      }
      PetscCall(PetscFree2(rsum,rsum2));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for this norm yet");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyBegin_MPISBAIJ(Mat mat,MatAssemblyType mode)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  PetscInt       nstash,reallocs;

  PetscFunctionBegin;
  if (baij->donotstash || mat->nooffprocentries) PetscFunctionReturn(0);

  PetscCall(MatStashScatterBegin_Private(mat,&mat->stash,mat->rmap->range));
  PetscCall(MatStashScatterBegin_Private(mat,&mat->bstash,baij->rangebs));
  PetscCall(MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs));
  PetscCall(PetscInfo(mat,"Stash has %" PetscInt_FMT " entries,uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs));
  PetscCall(MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs));
  PetscCall(PetscInfo(mat,"Block-Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPISBAIJ(Mat mat,MatAssemblyType mode)
{
  Mat_MPISBAIJ   *baij=(Mat_MPISBAIJ*)mat->data;
  Mat_SeqSBAIJ   *a   =(Mat_SeqSBAIJ*)baij->A->data;
  PetscInt       i,j,rstart,ncols,flg,bs2=baij->bs2;
  PetscInt       *row,*col;
  PetscBool      other_disassembled;
  PetscMPIInt    n;
  PetscBool      r1,r2,r3;
  MatScalar      *val;

  /* do not use 'b=(Mat_SeqBAIJ*)baij->B->data' as B can be reset in disassembly */
  PetscFunctionBegin;
  if (!baij->donotstash &&  !mat->nooffprocentries) {
    while (1) {
      PetscCall(MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg));
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) {
          if (row[j] != rstart) break;
        }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        /* Now assemble all these values with a single function call */
        PetscCall(MatSetValues_MPISBAIJ(mat,1,row+i,ncols,col+i,val+i,mat->insertmode));
        i    = j;
      }
    }
    PetscCall(MatStashScatterEnd_Private(&mat->stash));
    /* Now process the block-stash. Since the values are stashed column-oriented,
       set the roworiented flag to column oriented, and after MatSetValues()
       restore the original flags */
    r1 = baij->roworiented;
    r2 = a->roworiented;
    r3 = ((Mat_SeqBAIJ*)baij->B->data)->roworiented;

    baij->roworiented = PETSC_FALSE;
    a->roworiented    = PETSC_FALSE;

    ((Mat_SeqBAIJ*)baij->B->data)->roworiented = PETSC_FALSE; /* b->roworinted */
    while (1) {
      PetscCall(MatStashScatterGetMesg_Private(&mat->bstash,&n,&row,&col,&val,&flg));
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) {
          if (row[j] != rstart) break;
        }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        PetscCall(MatSetValuesBlocked_MPISBAIJ(mat,1,row+i,ncols,col+i,val+i*bs2,mat->insertmode));
        i    = j;
      }
    }
    PetscCall(MatStashScatterEnd_Private(&mat->bstash));

    baij->roworiented = r1;
    a->roworiented    = r2;

    ((Mat_SeqBAIJ*)baij->B->data)->roworiented = r3; /* b->roworinted */
  }

  PetscCall(MatAssemblyBegin(baij->A,mode));
  PetscCall(MatAssemblyEnd(baij->A,mode));

  /* determine if any processor has disassembled, if so we must
     also disassemble ourselfs, in order that we may reassemble. */
  /*
     if nonzero structure of submatrix B cannot change then we know that
     no processor disassembled thus we can skip this stuff
  */
  if (!((Mat_SeqBAIJ*)baij->B->data)->nonew) {
    PetscCall(MPIU_Allreduce(&mat->was_assembled,&other_disassembled,1,MPIU_BOOL,MPI_PROD,PetscObjectComm((PetscObject)mat)));
    if (mat->was_assembled && !other_disassembled) {
      PetscCall(MatDisAssemble_MPISBAIJ(mat));
    }
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    PetscCall(MatSetUpMultiply_MPISBAIJ(mat)); /* setup Mvctx and sMvctx */
  }
  PetscCall(MatAssemblyBegin(baij->B,mode));
  PetscCall(MatAssemblyEnd(baij->B,mode));

  PetscCall(PetscFree2(baij->rowvalues,baij->rowindices));

  baij->rowvalues = NULL;

  /* if no new nonzero locations are allowed in matrix then only set the matrix state the first time through */
  if ((!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) || !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
    PetscObjectState state = baij->A->nonzerostate + baij->B->nonzerostate;
    PetscCall(MPIU_Allreduce(&state,&mat->nonzerostate,1,MPIU_INT64,MPI_SUM,PetscObjectComm((PetscObject)mat)));
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatSetValues_MPIBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
#include <petscdraw.h>
static PetscErrorCode MatView_MPISBAIJ_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPISBAIJ      *baij = (Mat_MPISBAIJ*)mat->data;
  PetscInt          bs   = mat->rmap->bs;
  PetscMPIInt       rank = baij->rank;
  PetscBool         iascii,isdraw;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo info;
      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&rank));
      PetscCall(MatGetInfo(mat,MAT_LOCAL,&info));
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %" PetscInt_FMT " nz %" PetscInt_FMT " nz alloced %" PetscInt_FMT " bs %" PetscInt_FMT " mem %g\n",rank,mat->rmap->n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,mat->rmap->bs,(double)info.memory));
      PetscCall(MatGetInfo(baij->A,MAT_LOCAL,&info));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] on-diagonal part: nz %" PetscInt_FMT " \n",rank,(PetscInt)info.nz_used));
      PetscCall(MatGetInfo(baij->B,MAT_LOCAL,&info));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] off-diagonal part: nz %" PetscInt_FMT " \n",rank,(PetscInt)info.nz_used));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer,"Information on VecScatter used in matrix-vector product: \n"));
      PetscCall(VecScatterView(baij->Mvctx,viewer));
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  block size is %" PetscInt_FMT "\n",bs));
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      PetscFunctionReturn(0);
    }
  }

  if (isdraw) {
    PetscDraw draw;
    PetscBool isnull;
    PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
    PetscCall(PetscDrawIsNull(draw,&isnull));
    if (isnull) PetscFunctionReturn(0);
  }

  {
    /* assemble the entire matrix onto first processor. */
    Mat          A;
    Mat_SeqSBAIJ *Aloc;
    Mat_SeqBAIJ  *Bloc;
    PetscInt     M = mat->rmap->N,N = mat->cmap->N,*ai,*aj,col,i,j,k,*rvals,mbs = baij->mbs;
    MatScalar    *a;
    const char   *matname;

    /* Should this be the same type as mat? */
    PetscCall(MatCreate(PetscObjectComm((PetscObject)mat),&A));
    if (rank == 0) {
      PetscCall(MatSetSizes(A,M,N,M,N));
    } else {
      PetscCall(MatSetSizes(A,0,0,M,N));
    }
    PetscCall(MatSetType(A,MATMPISBAIJ));
    PetscCall(MatMPISBAIJSetPreallocation(A,mat->rmap->bs,0,NULL,0,NULL));
    PetscCall(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
    PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)A));

    /* copy over the A part */
    Aloc = (Mat_SeqSBAIJ*)baij->A->data;
    ai   = Aloc->i; aj = Aloc->j; a = Aloc->a;
    PetscCall(PetscMalloc1(bs,&rvals));

    for (i=0; i<mbs; i++) {
      rvals[0] = bs*(baij->rstartbs + i);
      for (j=1; j<bs; j++) rvals[j] = rvals[j-1] + 1;
      for (j=ai[i]; j<ai[i+1]; j++) {
        col = (baij->cstartbs+aj[j])*bs;
        for (k=0; k<bs; k++) {
          PetscCall(MatSetValues_MPISBAIJ(A,bs,rvals,1,&col,a,INSERT_VALUES));
          col++;
          a += bs;
        }
      }
    }
    /* copy over the B part */
    Bloc = (Mat_SeqBAIJ*)baij->B->data;
    ai   = Bloc->i; aj = Bloc->j; a = Bloc->a;
    for (i=0; i<mbs; i++) {

      rvals[0] = bs*(baij->rstartbs + i);
      for (j=1; j<bs; j++) rvals[j] = rvals[j-1] + 1;
      for (j=ai[i]; j<ai[i+1]; j++) {
        col = baij->garray[aj[j]]*bs;
        for (k=0; k<bs; k++) {
          PetscCall(MatSetValues_MPIBAIJ(A,bs,rvals,1,&col,a,INSERT_VALUES));
          col++;
          a += bs;
        }
      }
    }
    PetscCall(PetscFree(rvals));
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    /*
       Everyone has to call to draw the matrix since the graphics waits are
       synchronized across all processors that share the PetscDraw object
    */
    PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    PetscCall(PetscObjectGetName((PetscObject)mat,&matname));
    if (rank == 0) {
      PetscCall(PetscObjectSetName((PetscObject)((Mat_MPISBAIJ*)(A->data))->A,matname));
      PetscCall(MatView_SeqSBAIJ(((Mat_MPISBAIJ*)(A->data))->A,sviewer));
    }
    PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(MatDestroy(&A));
  }
  PetscFunctionReturn(0);
}

/* Used for both MPIBAIJ and MPISBAIJ matrices */
#define MatView_MPISBAIJ_Binary MatView_MPIBAIJ_Binary

PetscErrorCode MatView_MPISBAIJ(Mat mat,PetscViewer viewer)
{
  PetscBool      iascii,isdraw,issocket,isbinary;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  if (iascii || isdraw || issocket) {
    PetscCall(MatView_MPISBAIJ_ASCIIorDraworSocket(mat,viewer));
  } else if (isbinary) {
    PetscCall(MatView_MPISBAIJ_Binary(mat,viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%" PetscInt_FMT ",Cols=%" PetscInt_FMT,mat->rmap->N,mat->cmap->N);
#endif
  PetscCall(MatStashDestroy_Private(&mat->stash));
  PetscCall(MatStashDestroy_Private(&mat->bstash));
  PetscCall(MatDestroy(&baij->A));
  PetscCall(MatDestroy(&baij->B));
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscTableDestroy(&baij->colmap));
#else
  PetscCall(PetscFree(baij->colmap));
#endif
  PetscCall(PetscFree(baij->garray));
  PetscCall(VecDestroy(&baij->lvec));
  PetscCall(VecScatterDestroy(&baij->Mvctx));
  PetscCall(VecDestroy(&baij->slvec0));
  PetscCall(VecDestroy(&baij->slvec0b));
  PetscCall(VecDestroy(&baij->slvec1));
  PetscCall(VecDestroy(&baij->slvec1a));
  PetscCall(VecDestroy(&baij->slvec1b));
  PetscCall(VecScatterDestroy(&baij->sMvctx));
  PetscCall(PetscFree2(baij->rowvalues,baij->rowindices));
  PetscCall(PetscFree(baij->barray));
  PetscCall(PetscFree(baij->hd));
  PetscCall(VecDestroy(&baij->diag));
  PetscCall(VecDestroy(&baij->bb1));
  PetscCall(VecDestroy(&baij->xx1));
#if defined(PETSC_USE_REAL_MAT_SINGLE)
  PetscCall(PetscFree(baij->setvaluescopy));
#endif
  PetscCall(PetscFree(baij->in_loc));
  PetscCall(PetscFree(baij->v_loc));
  PetscCall(PetscFree(baij->rangebs));
  PetscCall(PetscFree(mat->data));

  PetscCall(PetscObjectChangeTypeName((PetscObject)mat,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatStoreValues_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatRetrieveValues_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMPISBAIJSetPreallocation_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatMPISBAIJSetPreallocationCSR_C",NULL));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpisbaij_elemental_C",NULL));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpisbaij_scalapack_C",NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpisbaij_mpiaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpisbaij_mpibaij_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPISBAIJ_Hermitian(Mat A,Vec xx,Vec yy)
{
  Mat_MPISBAIJ      *a = (Mat_MPISBAIJ*)A->data;
  PetscInt          mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar       *from;
  const PetscScalar *x;

  PetscFunctionBegin;
  /* diagonal part */
  PetscCall((*a->A->ops->mult)(a->A,xx,a->slvec1a));
  PetscCall(VecSet(a->slvec1b,0.0));

  /* subdiagonal part */
  PetscCheck(a->B->ops->multhermitiantranspose,PetscObjectComm((PetscObject)a->B),PETSC_ERR_SUP,"Not for type %s",((PetscObject)a->B)->type_name);
  PetscCall((*a->B->ops->multhermitiantranspose)(a->B,xx,a->slvec0b));

  /* copy x into the vec slvec0 */
  PetscCall(VecGetArray(a->slvec0,&from));
  PetscCall(VecGetArrayRead(xx,&x));

  PetscCall(PetscArraycpy(from,x,bs*mbs));
  PetscCall(VecRestoreArray(a->slvec0,&from));
  PetscCall(VecRestoreArrayRead(xx,&x));

  PetscCall(VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD));
  /* supperdiagonal part */
  PetscCall((*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPISBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPISBAIJ      *a = (Mat_MPISBAIJ*)A->data;
  PetscInt          mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar       *from;
  const PetscScalar *x;

  PetscFunctionBegin;
  /* diagonal part */
  PetscCall((*a->A->ops->mult)(a->A,xx,a->slvec1a));
  PetscCall(VecSet(a->slvec1b,0.0));

  /* subdiagonal part */
  PetscCall((*a->B->ops->multtranspose)(a->B,xx,a->slvec0b));

  /* copy x into the vec slvec0 */
  PetscCall(VecGetArray(a->slvec0,&from));
  PetscCall(VecGetArrayRead(xx,&x));

  PetscCall(PetscArraycpy(from,x,bs*mbs));
  PetscCall(VecRestoreArray(a->slvec0,&from));
  PetscCall(VecRestoreArrayRead(xx,&x));

  PetscCall(VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD));
  /* supperdiagonal part */
  PetscCall((*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPISBAIJ_Hermitian(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPISBAIJ      *a = (Mat_MPISBAIJ*)A->data;
  PetscInt          mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar       *from,zero=0.0;
  const PetscScalar *x;

  PetscFunctionBegin;
  /* diagonal part */
  PetscCall((*a->A->ops->multadd)(a->A,xx,yy,a->slvec1a));
  PetscCall(VecSet(a->slvec1b,zero));

  /* subdiagonal part */
  PetscCheck(a->B->ops->multhermitiantranspose,PetscObjectComm((PetscObject)a->B),PETSC_ERR_SUP,"Not for type %s",((PetscObject)a->B)->type_name);
  PetscCall((*a->B->ops->multhermitiantranspose)(a->B,xx,a->slvec0b));

  /* copy x into the vec slvec0 */
  PetscCall(VecGetArray(a->slvec0,&from));
  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(PetscArraycpy(from,x,bs*mbs));
  PetscCall(VecRestoreArray(a->slvec0,&from));

  PetscCall(VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD));

  /* supperdiagonal part */
  PetscCall((*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,zz));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPISBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPISBAIJ      *a = (Mat_MPISBAIJ*)A->data;
  PetscInt          mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar       *from,zero=0.0;
  const PetscScalar *x;

  PetscFunctionBegin;
  /* diagonal part */
  PetscCall((*a->A->ops->multadd)(a->A,xx,yy,a->slvec1a));
  PetscCall(VecSet(a->slvec1b,zero));

  /* subdiagonal part */
  PetscCall((*a->B->ops->multtranspose)(a->B,xx,a->slvec0b));

  /* copy x into the vec slvec0 */
  PetscCall(VecGetArray(a->slvec0,&from));
  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(PetscArraycpy(from,x,bs*mbs));
  PetscCall(VecRestoreArray(a->slvec0,&from));

  PetscCall(VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD));

  /* supperdiagonal part */
  PetscCall((*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,zz));
  PetscFunctionReturn(0);
}

/*
  This only works correctly for square matrices where the subblock A->A is the
   diagonal block
*/
PetscErrorCode MatGetDiagonal_MPISBAIJ(Mat A,Vec v)
{
  Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)A->data;

  PetscFunctionBegin;
  /* PetscCheckFalse(a->rmap->N != a->cmap->N,PETSC_COMM_SELF,PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block"); */
  PetscCall(MatGetDiagonal(a->A,v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_MPISBAIJ(Mat A,PetscScalar aa)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(MatScale(a->A,aa));
  PetscCall(MatScale(a->B,aa));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_MPISBAIJ(Mat matin,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPISBAIJ *mat = (Mat_MPISBAIJ*)matin->data;
  PetscScalar  *vworkA,*vworkB,**pvA,**pvB,*v_p;
  PetscInt     bs = matin->rmap->bs,bs2 = mat->bs2,i,*cworkA,*cworkB,**pcA,**pcB;
  PetscInt     nztot,nzA,nzB,lrow,brstart = matin->rmap->rstart,brend = matin->rmap->rend;
  PetscInt    *cmap,*idx_p,cstart = mat->rstartbs;

  PetscFunctionBegin;
  PetscCheck(!mat->getrowactive,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqSBAIJ *Aa = (Mat_SeqSBAIJ*)mat->A->data;
    Mat_SeqBAIJ  *Ba = (Mat_SeqBAIJ*)mat->B->data;
    PetscInt     max = 1,mbs = mat->mbs,tmp;
    for (i=0; i<mbs; i++) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i]; /* row length */
      if (max < tmp) max = tmp;
    }
    PetscCall(PetscMalloc2(max*bs2,&mat->rowvalues,max*bs2,&mat->rowindices));
  }

  PetscCheck(row >= brstart && row < brend,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local rows");
  lrow = row - brstart;  /* local row index */

  pvA = &vworkA; pcA = &cworkA; pvB = &vworkB; pcB = &cworkB;
  if (!v)   {pvA = NULL; pvB = NULL;}
  if (!idx) {pcA = NULL; if (!v) pcB = NULL;}
  PetscCall((*mat->A->ops->getrow)(mat->A,lrow,&nzA,pcA,pvA));
  PetscCall((*mat->B->ops->getrow)(mat->B,lrow,&nzB,pcB,pvB));
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
  PetscCall((*mat->A->ops->restorerow)(mat->A,lrow,&nzA,pcA,pvA));
  PetscCall((*mat->B->ops->restorerow)(mat->B,lrow,&nzB,pcB,pvB));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_MPISBAIJ(Mat mat,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;

  PetscFunctionBegin;
  PetscCheck(baij->getrowactive,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"MatGetRow() must be called first");
  baij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRowUpperTriangular_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ *a  = (Mat_MPISBAIJ*)A->data;
  Mat_SeqSBAIJ *aA = (Mat_SeqSBAIJ*)a->A->data;

  PetscFunctionBegin;
  aA->getrow_utriangular = PETSC_TRUE;
  PetscFunctionReturn(0);
}
PetscErrorCode MatRestoreRowUpperTriangular_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ *a  = (Mat_MPISBAIJ*)A->data;
  Mat_SeqSBAIJ *aA = (Mat_SeqSBAIJ*)a->A->data;

  PetscFunctionBegin;
  aA->getrow_utriangular = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConjugate_MPISBAIJ(Mat mat)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)mat->data;

    PetscCall(MatConjugate(a->A));
    PetscCall(MatConjugate(a->B));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatRealPart_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(MatRealPart(a->A));
  PetscCall(MatRealPart(a->B));
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(MatImaginaryPart(a->A));
  PetscCall(MatImaginaryPart(a->B));
  PetscFunctionReturn(0);
}

/* Check if isrow is a subset of iscol_local, called by MatCreateSubMatrix_MPISBAIJ()
   Input: isrow       - distributed(parallel),
          iscol_local - locally owned (seq)
*/
PetscErrorCode ISEqual_private(IS isrow,IS iscol_local,PetscBool  *flg)
{
  PetscInt       sz1,sz2,*a1,*a2,i,j,k,nmatch;
  const PetscInt *ptr1,*ptr2;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(isrow,&sz1));
  PetscCall(ISGetLocalSize(iscol_local,&sz2));
  if (sz1 > sz2) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  PetscCall(ISGetIndices(isrow,&ptr1));
  PetscCall(ISGetIndices(iscol_local,&ptr2));

  PetscCall(PetscMalloc1(sz1,&a1));
  PetscCall(PetscMalloc1(sz2,&a2));
  PetscCall(PetscArraycpy(a1,ptr1,sz1));
  PetscCall(PetscArraycpy(a2,ptr2,sz2));
  PetscCall(PetscSortInt(sz1,a1));
  PetscCall(PetscSortInt(sz2,a2));

  nmatch=0;
  k     = 0;
  for (i=0; i<sz1; i++) {
    for (j=k; j<sz2; j++) {
      if (a1[i] == a2[j]) {
        k = j; nmatch++;
        break;
      }
    }
  }
  PetscCall(ISRestoreIndices(isrow,&ptr1));
  PetscCall(ISRestoreIndices(iscol_local,&ptr2));
  PetscCall(PetscFree(a1));
  PetscCall(PetscFree(a2));
  if (nmatch < sz1) {
    *flg = PETSC_FALSE;
  } else {
    *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrix_MPISBAIJ(Mat mat,IS isrow,IS iscol,MatReuse call,Mat *newmat)
{
  IS             iscol_local;
  PetscInt       csize;
  PetscBool      isequal;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(iscol,&csize));
  if (call == MAT_REUSE_MATRIX) {
    PetscCall(PetscObjectQuery((PetscObject)*newmat,"ISAllGather",(PetscObject*)&iscol_local));
    PetscCheck(iscol_local,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Submatrix passed in was not used before, cannot reuse");
  } else {
    PetscBool issorted;

    PetscCall(ISAllGather(iscol,&iscol_local));
    PetscCall(ISEqual_private(isrow,iscol_local,&isequal));
    PetscCall(ISSorted(iscol_local, &issorted));
    PetscCheck(isequal && issorted,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"For symmetric format, iscol must equal isrow and be sorted");
  }

  /* now call MatCreateSubMatrix_MPIBAIJ() */
  PetscCall(MatCreateSubMatrix_MPIBAIJ_Private(mat,isrow,iscol_local,csize,call,newmat));
  if (call == MAT_INITIAL_MATRIX) {
    PetscCall(PetscObjectCompose((PetscObject)*newmat,"ISAllGather",(PetscObject)iscol_local));
    PetscCall(ISDestroy(&iscol_local));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *l = (Mat_MPISBAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(l->A));
  PetscCall(MatZeroEntries(l->B));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_MPISBAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)matin->data;
  Mat            A  = a->A,B = a->B;
  PetscLogDouble isend[5],irecv[5];

  PetscFunctionBegin;
  info->block_size = (PetscReal)matin->rmap->bs;

  PetscCall(MatGetInfo(A,MAT_LOCAL,info));

  isend[0] = info->nz_used; isend[1] = info->nz_allocated; isend[2] = info->nz_unneeded;
  isend[3] = info->memory;  isend[4] = info->mallocs;

  PetscCall(MatGetInfo(B,MAT_LOCAL,info));

  isend[0] += info->nz_used; isend[1] += info->nz_allocated; isend[2] += info->nz_unneeded;
  isend[3] += info->memory;  isend[4] += info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    PetscCall(MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_MAX,PetscObjectComm((PetscObject)matin)));

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    PetscCall(MPIU_Allreduce(isend,irecv,5,MPIU_PETSCLOGDOUBLE,MPI_SUM,PetscObjectComm((PetscObject)matin)));

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown MatInfoType argument %d",(int)flag);
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_MPISBAIJ(Mat A,MatOption op,PetscBool flg)
{
  Mat_MPISBAIJ   *a  = (Mat_MPISBAIJ*)A->data;
  Mat_SeqSBAIJ   *aA = (Mat_SeqSBAIJ*)a->A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_SUBMAT_SINGLEIS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
    MatCheckPreallocated(A,1);
    PetscCall(MatSetOption(a->A,op,flg));
    PetscCall(MatSetOption(a->B,op,flg));
    break;
  case MAT_ROW_ORIENTED:
    MatCheckPreallocated(A,1);
    a->roworiented = flg;

    PetscCall(MatSetOption(a->A,op,flg));
    PetscCall(MatSetOption(a->B,op,flg));
    break;
  case MAT_FORCE_DIAGONAL_ENTRIES:
  case MAT_SORTED_FULL:
    PetscCall(PetscInfo(A,"Option %s ignored\n",MatOptions[op]));
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = flg;
    break;
  case MAT_USE_HASH_TABLE:
    a->ht_flag = flg;
    break;
  case MAT_HERMITIAN:
    MatCheckPreallocated(A,1);
    PetscCall(MatSetOption(a->A,op,flg));
#if defined(PETSC_USE_COMPLEX)
    if (flg) { /* need different mat-vec ops */
      A->ops->mult             = MatMult_MPISBAIJ_Hermitian;
      A->ops->multadd          = MatMultAdd_MPISBAIJ_Hermitian;
      A->ops->multtranspose    = NULL;
      A->ops->multtransposeadd = NULL;
      A->symmetric = PETSC_FALSE;
    }
#endif
    break;
  case MAT_SPD:
  case MAT_SYMMETRIC:
    MatCheckPreallocated(A,1);
    PetscCall(MatSetOption(a->A,op,flg));
#if defined(PETSC_USE_COMPLEX)
    if (flg) { /* restore to use default mat-vec ops */
      A->ops->mult             = MatMult_MPISBAIJ;
      A->ops->multadd          = MatMultAdd_MPISBAIJ;
      A->ops->multtranspose    = MatMult_MPISBAIJ;
      A->ops->multtransposeadd = MatMultAdd_MPISBAIJ;
    }
#endif
    break;
  case MAT_STRUCTURALLY_SYMMETRIC:
    MatCheckPreallocated(A,1);
    PetscCall(MatSetOption(a->A,op,flg));
    break;
  case MAT_SYMMETRY_ETERNAL:
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix must be symmetric");
    PetscCall(PetscInfo(A,"Option %s ignored\n",MatOptions[op]));
    break;
  case MAT_IGNORE_LOWER_TRIANGULAR:
    aA->ignore_ltriangular = flg;
    break;
  case MAT_ERROR_LOWER_TRIANGULAR:
    aA->ignore_ltriangular = flg;
    break;
  case MAT_GETROW_UPPERTRIANGULAR:
    aA->getrow_utriangular = flg;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatTranspose_MPISBAIJ(Mat A,MatReuse reuse,Mat *B)
{
  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A,MAT_COPY_VALUES,B));
  }  else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(A,*B,SAME_NONZERO_PATTERN));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_MPISBAIJ(Mat mat,Vec ll,Vec rr)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  Mat            a     = baij->A, b=baij->B;
  PetscInt       nv,m,n;
  PetscBool      flg;

  PetscFunctionBegin;
  if (ll != rr) {
    PetscCall(VecEqual(ll,rr,&flg));
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"For symmetric format, left and right scaling vectors must be same");
  }
  if (!ll) PetscFunctionReturn(0);

  PetscCall(MatGetLocalSize(mat,&m,&n));
  PetscCheck(m == n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"For symmetric format, local size %" PetscInt_FMT " %" PetscInt_FMT " must be same",m,n);

  PetscCall(VecGetLocalSize(rr,&nv));
  PetscCheck(nv==n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left and right vector non-conforming local size");

  PetscCall(VecScatterBegin(baij->Mvctx,rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD));

  /* left diagonalscale the off-diagonal part */
  PetscCall((*b->ops->diagonalscale)(b,ll,NULL));

  /* scale the diagonal part */
  PetscCall((*a->ops->diagonalscale)(a,ll,rr));

  /* right diagonalscale the off-diagonal part */
  PetscCall(VecScatterEnd(baij->Mvctx,rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall((*b->ops->diagonalscale)(b,NULL,baij->lvec));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUnfactored_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(MatSetUnfactored(a->A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPISBAIJ(Mat,MatDuplicateOption,Mat*);

PetscErrorCode MatEqual_MPISBAIJ(Mat A,Mat B,PetscBool  *flag)
{
  Mat_MPISBAIJ   *matB = (Mat_MPISBAIJ*)B->data,*matA = (Mat_MPISBAIJ*)A->data;
  Mat            a,b,c,d;
  PetscBool      flg;

  PetscFunctionBegin;
  a = matA->A; b = matA->B;
  c = matB->A; d = matB->B;

  PetscCall(MatEqual(a,c,&flg));
  if (flg) {
    PetscCall(MatEqual(b,d,&flg));
  }
  PetscCall(MPIU_Allreduce(&flg,flag,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_MPISBAIJ(Mat A,Mat B,MatStructure str)
{
  PetscBool      isbaij;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B,&isbaij,MATSEQSBAIJ,MATMPISBAIJ,""));
  PetscCheck(isbaij,PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)B)->type_name);
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if ((str != SAME_NONZERO_PATTERN) || (A->ops->copy != B->ops->copy)) {
    PetscCall(MatGetRowUpperTriangular(A));
    PetscCall(MatCopy_Basic(A,B,str));
    PetscCall(MatRestoreRowUpperTriangular(A));
  } else {
    Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)A->data;
    Mat_MPISBAIJ *b = (Mat_MPISBAIJ*)B->data;

    PetscCall(MatCopy(a->A,b->A,str));
    PetscCall(MatCopy(a->B,b->B,str));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)B));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_MPISBAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatMPISBAIJSetPreallocation(A,A->rmap->bs,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_MPISBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_MPISBAIJ   *xx=(Mat_MPISBAIJ*)X->data,*yy=(Mat_MPISBAIJ*)Y->data;
  PetscBLASInt   bnz,one=1;
  Mat_SeqSBAIJ   *xa,*ya;
  Mat_SeqBAIJ    *xb,*yb;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {
    PetscScalar alpha = a;
    xa   = (Mat_SeqSBAIJ*)xx->A->data;
    ya   = (Mat_SeqSBAIJ*)yy->A->data;
    PetscCall(PetscBLASIntCast(xa->nz,&bnz));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&alpha,xa->a,&one,ya->a,&one));
    xb   = (Mat_SeqBAIJ*)xx->B->data;
    yb   = (Mat_SeqBAIJ*)yy->B->data;
    PetscCall(PetscBLASIntCast(xb->nz,&bnz));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&alpha,xb->a,&one,yb->a,&one));
    PetscCall(PetscObjectStateIncrease((PetscObject)Y));
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    PetscCall(MatSetOption(X,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE));
    PetscCall(MatAXPY_Basic(Y,a,X,str));
    PetscCall(MatSetOption(X,MAT_GETROW_UPPERTRIANGULAR,PETSC_FALSE));
  } else {
    Mat      B;
    PetscInt *nnz_d,*nnz_o,bs=Y->rmap->bs;
    PetscCheck(bs == X->rmap->bs,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrices must have same block size");
    PetscCall(MatGetRowUpperTriangular(X));
    PetscCall(MatGetRowUpperTriangular(Y));
    PetscCall(PetscMalloc1(yy->A->rmap->N,&nnz_d));
    PetscCall(PetscMalloc1(yy->B->rmap->N,&nnz_o));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)Y),&B));
    PetscCall(PetscObjectSetName((PetscObject)B,((PetscObject)Y)->name));
    PetscCall(MatSetSizes(B,Y->rmap->n,Y->cmap->n,Y->rmap->N,Y->cmap->N));
    PetscCall(MatSetBlockSizesFromMats(B,Y,Y));
    PetscCall(MatSetType(B,MATMPISBAIJ));
    PetscCall(MatAXPYGetPreallocation_SeqSBAIJ(yy->A,xx->A,nnz_d));
    PetscCall(MatAXPYGetPreallocation_MPIBAIJ(yy->B,yy->garray,xx->B,xx->garray,nnz_o));
    PetscCall(MatMPISBAIJSetPreallocation(B,bs,0,nnz_d,0,nnz_o));
    PetscCall(MatAXPY_BasicWithPreallocation(B,Y,a,X,str));
    PetscCall(MatHeaderMerge(Y,&B));
    PetscCall(PetscFree(nnz_d));
    PetscCall(PetscFree(nnz_o));
    PetscCall(MatRestoreRowUpperTriangular(X));
    PetscCall(MatRestoreRowUpperTriangular(Y));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_MPISBAIJ(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscInt       i;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(MatCreateSubMatrices_MPIBAIJ(A,n,irow,icol,scall,B)); /* B[] are sbaij matrices */
  for (i=0; i<n; i++) {
    PetscCall(ISEqual(irow[i],icol[i],&flg));
    if (!flg) {
      PetscCall(MatSeqSBAIJZeroOps_Private(*B[i]));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_MPISBAIJ(Mat Y,PetscScalar a)
{
  Mat_MPISBAIJ    *maij = (Mat_MPISBAIJ*)Y->data;
  Mat_SeqSBAIJ    *aij = (Mat_SeqSBAIJ*)maij->A->data;

  PetscFunctionBegin;
  if (!Y->preallocated) {
    PetscCall(MatMPISBAIJSetPreallocation(Y,Y->rmap->bs,1,NULL,0,NULL));
  } else if (!aij->nz) {
    PetscInt nonew = aij->nonew;
    PetscCall(MatSeqSBAIJSetPreallocation(maij->A,Y->rmap->bs,1,NULL));
    aij->nonew = nonew;
  }
  PetscCall(MatShift_Basic(Y,a));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMissingDiagonal_MPISBAIJ(Mat A,PetscBool  *missing,PetscInt *d)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;

  PetscFunctionBegin;
  PetscCheck(A->rmap->n == A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only works for square matrices");
  PetscCall(MatMissingDiagonal(a->A,missing,d));
  if (d) {
    PetscInt rstart;
    PetscCall(MatGetOwnershipRange(A,&rstart,NULL));
    *d += rstart/A->rmap->bs;

  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatGetDiagonalBlock_MPISBAIJ(Mat A,Mat *a)
{
  PetscFunctionBegin;
  *a = ((Mat_MPISBAIJ*)A->data)->A;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_MPISBAIJ,
                                       MatGetRow_MPISBAIJ,
                                       MatRestoreRow_MPISBAIJ,
                                       MatMult_MPISBAIJ,
                               /*  4*/ MatMultAdd_MPISBAIJ,
                                       MatMult_MPISBAIJ,       /* transpose versions are same as non-transpose */
                                       MatMultAdd_MPISBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 10*/ NULL,
                                       NULL,
                                       NULL,
                                       MatSOR_MPISBAIJ,
                                       MatTranspose_MPISBAIJ,
                               /* 15*/ MatGetInfo_MPISBAIJ,
                                       MatEqual_MPISBAIJ,
                                       MatGetDiagonal_MPISBAIJ,
                                       MatDiagonalScale_MPISBAIJ,
                                       MatNorm_MPISBAIJ,
                               /* 20*/ MatAssemblyBegin_MPISBAIJ,
                                       MatAssemblyEnd_MPISBAIJ,
                                       MatSetOption_MPISBAIJ,
                                       MatZeroEntries_MPISBAIJ,
                               /* 24*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 29*/ MatSetUp_MPISBAIJ,
                                       NULL,
                                       NULL,
                                       MatGetDiagonalBlock_MPISBAIJ,
                                       NULL,
                               /* 34*/ MatDuplicate_MPISBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 39*/ MatAXPY_MPISBAIJ,
                                       MatCreateSubMatrices_MPISBAIJ,
                                       MatIncreaseOverlap_MPISBAIJ,
                                       MatGetValues_MPISBAIJ,
                                       MatCopy_MPISBAIJ,
                               /* 44*/ NULL,
                                       MatScale_MPISBAIJ,
                                       MatShift_MPISBAIJ,
                                       NULL,
                                       NULL,
                               /* 49*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 54*/ NULL,
                                       NULL,
                                       MatSetUnfactored_MPISBAIJ,
                                       NULL,
                                       MatSetValuesBlocked_MPISBAIJ,
                               /* 59*/ MatCreateSubMatrix_MPISBAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 64*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 69*/ MatGetRowMaxAbs_MPISBAIJ,
                                       NULL,
                                       MatConvert_MPISBAIJ_Basic,
                                       NULL,
                                       NULL,
                               /* 74*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 79*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatLoad_MPISBAIJ,
                               /* 84*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 89*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 94*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 99*/ NULL,
                                       NULL,
                                       NULL,
                                       MatConjugate_MPISBAIJ,
                                       NULL,
                               /*104*/ NULL,
                                       MatRealPart_MPISBAIJ,
                                       MatImaginaryPart_MPISBAIJ,
                                       MatGetRowUpperTriangular_MPISBAIJ,
                                       MatRestoreRowUpperTriangular_MPISBAIJ,
                               /*109*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatMissingDiagonal_MPISBAIJ,
                               /*114*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*119*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*124*/ NULL,
                                       NULL,
                                       NULL,
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
                                       NULL,
                                       NULL,
                                /*144*/MatCreateMPIMatConcatenateSeqMat_MPISBAIJ,
                                       NULL,
                                       NULL,
                                       NULL
};

PetscErrorCode  MatMPISBAIJSetPreallocation_MPISBAIJ(Mat B,PetscInt bs,PetscInt d_nz,const PetscInt *d_nnz,PetscInt o_nz,const PetscInt *o_nnz)
{
  Mat_MPISBAIJ   *b = (Mat_MPISBAIJ*)B->data;
  PetscInt       i,mbs,Mbs;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCall(MatSetBlockSize(B,PetscAbs(bs)));
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  PetscCall(PetscLayoutGetBlockSize(B->rmap,&bs));
  PetscCheck(B->rmap->N <= B->cmap->N,PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"MPISBAIJ matrix cannot have more rows %" PetscInt_FMT " than columns %" PetscInt_FMT,B->rmap->N,B->cmap->N);
  PetscCheck(B->rmap->n <= B->cmap->n,PETSC_COMM_SELF,PETSC_ERR_SUP,"MPISBAIJ matrix cannot have more local rows %" PetscInt_FMT " than columns %" PetscInt_FMT,B->rmap->n,B->cmap->n);

  mbs = B->rmap->n/bs;
  Mbs = B->rmap->N/bs;
  PetscCheck(mbs*bs == B->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"No of local rows %" PetscInt_FMT " must be divisible by blocksize %" PetscInt_FMT,B->rmap->N,bs);

  B->rmap->bs = bs;
  b->bs2      = bs*bs;
  b->mbs      = mbs;
  b->Mbs      = Mbs;
  b->nbs      = B->cmap->n/bs;
  b->Nbs      = B->cmap->N/bs;

  for (i=0; i<=b->size; i++) {
    b->rangebs[i] = B->rmap->range[i]/bs;
  }
  b->rstartbs = B->rmap->rstart/bs;
  b->rendbs   = B->rmap->rend/bs;

  b->cstartbs = B->cmap->rstart/bs;
  b->cendbs   = B->cmap->rend/bs;

#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscTableDestroy(&b->colmap));
#else
  PetscCall(PetscFree(b->colmap));
#endif
  PetscCall(PetscFree(b->garray));
  PetscCall(VecDestroy(&b->lvec));
  PetscCall(VecScatterDestroy(&b->Mvctx));
  PetscCall(VecDestroy(&b->slvec0));
  PetscCall(VecDestroy(&b->slvec0b));
  PetscCall(VecDestroy(&b->slvec1));
  PetscCall(VecDestroy(&b->slvec1a));
  PetscCall(VecDestroy(&b->slvec1b));
  PetscCall(VecScatterDestroy(&b->sMvctx));

  /* Because the B will have been resized we simply destroy it and create a new one each time */
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B),&size));
  PetscCall(MatDestroy(&b->B));
  PetscCall(MatCreate(PETSC_COMM_SELF,&b->B));
  PetscCall(MatSetSizes(b->B,B->rmap->n,size > 1 ? B->cmap->N : 0,B->rmap->n,size > 1 ? B->cmap->N : 0));
  PetscCall(MatSetType(b->B,MATSEQBAIJ));
  PetscCall(PetscLogObjectParent((PetscObject)B,(PetscObject)b->B));

  if (!B->preallocated) {
    PetscCall(MatCreate(PETSC_COMM_SELF,&b->A));
    PetscCall(MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n));
    PetscCall(MatSetType(b->A,MATSEQSBAIJ));
    PetscCall(PetscLogObjectParent((PetscObject)B,(PetscObject)b->A));
    PetscCall(MatStashCreate_Private(PetscObjectComm((PetscObject)B),bs,&B->bstash));
  }

  PetscCall(MatSeqSBAIJSetPreallocation(b->A,bs,d_nz,d_nnz));
  PetscCall(MatSeqBAIJSetPreallocation(b->B,bs,o_nz,o_nnz));

  B->preallocated  = PETSC_TRUE;
  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPISBAIJSetPreallocationCSR_MPISBAIJ(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[],const PetscScalar V[])
{
  PetscInt       m,rstart,cend;
  PetscInt       i,j,d,nz,bd, nz_max=0,*d_nnz=NULL,*o_nnz=NULL;
  const PetscInt *JJ    =NULL;
  PetscScalar    *values=NULL;
  PetscBool      roworiented = ((Mat_MPISBAIJ*)B->data)->roworiented;
  PetscBool      nooffprocentries;

  PetscFunctionBegin;
  PetscCheck(bs >= 1,PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %" PetscInt_FMT,bs);
  PetscCall(PetscLayoutSetBlockSize(B->rmap,bs));
  PetscCall(PetscLayoutSetBlockSize(B->cmap,bs));
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  PetscCall(PetscLayoutGetBlockSize(B->rmap,&bs));
  m      = B->rmap->n/bs;
  rstart = B->rmap->rstart/bs;
  cend   = B->cmap->rend/bs;

  PetscCheck(!ii[0],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ii[0] must be 0 but it is %" PetscInt_FMT,ii[0]);
  PetscCall(PetscMalloc2(m,&d_nnz,m,&o_nnz));
  for (i=0; i<m; i++) {
    nz = ii[i+1] - ii[i];
    PetscCheck(nz >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local row %" PetscInt_FMT " has a negative number of columns %" PetscInt_FMT,i,nz);
    /* count the ones on the diagonal and above, split into diagonal and off diagonal portions. */
    JJ     = jj + ii[i];
    bd     = 0;
    for (j=0; j<nz; j++) {
      if (*JJ >= i + rstart) break;
      JJ++;
      bd++;
    }
    d  = 0;
    for (; j<nz; j++) {
      if (*JJ++ >= cend) break;
      d++;
    }
    d_nnz[i] = d;
    o_nnz[i] = nz - d - bd;
    nz       = nz - bd;
    nz_max = PetscMax(nz_max,nz);
  }
  PetscCall(MatMPISBAIJSetPreallocation(B,bs,0,d_nnz,0,o_nnz));
  PetscCall(MatSetOption(B,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE));
  PetscCall(PetscFree2(d_nnz,o_nnz));

  values = (PetscScalar*)V;
  if (!values) {
    PetscCall(PetscCalloc1(bs*bs*nz_max,&values));
  }
  for (i=0; i<m; i++) {
    PetscInt          row    = i + rstart;
    PetscInt          ncols  = ii[i+1] - ii[i];
    const PetscInt    *icols = jj + ii[i];
    if (bs == 1 || !roworiented) {         /* block ordering matches the non-nested layout of MatSetValues so we can insert entire rows */
      const PetscScalar *svals = values + (V ? (bs*bs*ii[i]) : 0);
      PetscCall(MatSetValuesBlocked_MPISBAIJ(B,1,&row,ncols,icols,svals,INSERT_VALUES));
    } else {                    /* block ordering does not match so we can only insert one block at a time. */
      PetscInt j;
      for (j=0; j<ncols; j++) {
        const PetscScalar *svals = values + (V ? (bs*bs*(ii[i]+j)) : 0);
        PetscCall(MatSetValuesBlocked_MPISBAIJ(B,1,&row,1,&icols[j],svals,INSERT_VALUES));
      }
    }
  }

  if (!V) PetscCall(PetscFree(values));
  nooffprocentries    = B->nooffprocentries;
  B->nooffprocentries = PETSC_TRUE;
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  B->nooffprocentries = nooffprocentries;

  PetscCall(MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*MC
   MATMPISBAIJ - MATMPISBAIJ = "mpisbaij" - A matrix type to be used for distributed symmetric sparse block matrices,
   based on block compressed sparse row format.  Only the upper triangular portion of the "diagonal" portion of
   the matrix is stored.

   For complex numbers by default this matrix is symmetric, NOT Hermitian symmetric. To make it Hermitian symmetric you
   can call MatSetOption(Mat, MAT_HERMITIAN);

   Options Database Keys:
. -mat_type mpisbaij - sets the matrix type to "mpisbaij" during a call to MatSetFromOptions()

   Notes:
     The number of rows in the matrix must be less than or equal to the number of columns. Similarly the number of rows in the
     diagonal portion of the matrix of each process has to less than or equal the number of columns.

   Level: beginner

.seealso: MatCreateBAIJ(), MATSEQSBAIJ, MatType
M*/

PETSC_EXTERN PetscErrorCode MatCreate_MPISBAIJ(Mat B)
{
  Mat_MPISBAIJ   *b;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(B,&b));
  B->data = (void*)b;
  PetscCall(PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps)));

  B->ops->destroy = MatDestroy_MPISBAIJ;
  B->ops->view    = MatView_MPISBAIJ;
  B->assembled    = PETSC_FALSE;
  B->insertmode   = NOT_SET_VALUES;

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)B),&b->rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B),&b->size));

  /* build local table of row and column ownerships */
  PetscCall(PetscMalloc1(b->size+2,&b->rangebs));

  /* build cache for off array entries formed */
  PetscCall(MatStashCreate_Private(PetscObjectComm((PetscObject)B),1,&B->stash));

  b->donotstash  = PETSC_FALSE;
  b->colmap      = NULL;
  b->garray      = NULL;
  b->roworiented = PETSC_TRUE;

  /* stuff used in block assembly */
  b->barray = NULL;

  /* stuff used for matrix vector multiply */
  b->lvec    = NULL;
  b->Mvctx   = NULL;
  b->slvec0  = NULL;
  b->slvec0b = NULL;
  b->slvec1  = NULL;
  b->slvec1a = NULL;
  b->slvec1b = NULL;
  b->sMvctx  = NULL;

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

  b->in_loc = NULL;
  b->v_loc  = NULL;
  b->n_loc  = 0;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_MPISBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_MPISBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMPISBAIJSetPreallocation_C",MatMPISBAIJSetPreallocation_MPISBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatMPISBAIJSetPreallocationCSR_C",MatMPISBAIJSetPreallocationCSR_MPISBAIJ));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaij_elemental_C",MatConvert_MPISBAIJ_Elemental));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaij_scalapack_C",MatConvert_SBAIJ_ScaLAPACK));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaij_mpiaij_C",MatConvert_MPISBAIJ_Basic));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaij_mpibaij_C",MatConvert_MPISBAIJ_Basic));

  B->symmetric                  = PETSC_TRUE;
  B->structurally_symmetric     = PETSC_TRUE;
  B->symmetric_set              = PETSC_TRUE;
  B->structurally_symmetric_set = PETSC_TRUE;
  B->symmetric_eternal          = PETSC_TRUE;
#if defined(PETSC_USE_COMPLEX)
  B->hermitian                  = PETSC_FALSE;
  B->hermitian_set              = PETSC_FALSE;
#else
  B->hermitian                  = PETSC_TRUE;
  B->hermitian_set              = PETSC_TRUE;
#endif

  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATMPISBAIJ));
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)B),NULL,"Options for loading MPISBAIJ matrix 1","Mat");PetscCall(ierr);
  PetscCall(PetscOptionsBool("-mat_use_hash_table","Use hash table to save memory in constructing matrix","MatSetOption",flg,&flg,NULL));
  if (flg) {
    PetscReal fact = 1.39;
    PetscCall(MatSetOption(B,MAT_USE_HASH_TABLE,PETSC_TRUE));
    PetscCall(PetscOptionsReal("-mat_use_hash_table","Use hash table factor","MatMPIBAIJSetHashTableFactor",fact,&fact,NULL));
    if (fact <= 1.0) fact = 1.39;
    PetscCall(MatMPIBAIJSetHashTableFactor(B,fact));
    PetscCall(PetscInfo(B,"Hash table Factor used %5.2g\n",(double)fact));
  }
  ierr = PetscOptionsEnd();PetscCall(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATSBAIJ - MATSBAIJ = "sbaij" - A matrix type to be used for symmetric block sparse matrices.

   This matrix type is identical to MATSEQSBAIJ when constructed with a single process communicator,
   and MATMPISBAIJ otherwise.

   Options Database Keys:
. -mat_type sbaij - sets the matrix type to "sbaij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSBAIJ, MATSEQSBAIJ, MATMPISBAIJ
M*/

/*@C
   MatMPISBAIJSetPreallocation - For good matrix assembly performance
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
           in the upper triangular and diagonal part of the in diagonal portion of the local
           (possibly different for each block row) or NULL.  If you plan to factor the matrix you must leave room
           for the diagonal entry and set a value even if it is zero.
.  o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix that is right of the diagonal
           (possibly different for each block row) or NULL.

   Options Database Keys:
+   -mat_no_unroll - uses code that does not unroll the loops in the
                     block calculations (much slower)
-   -mat_block_size - size of the blocks to use

   Notes:

   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one processor
   than it must be used on all processors that share the object for that argument.

   If the *_nnz parameter is given then the *_nz parameter is ignored

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

   You can call MatGetInfo() to get information on how effective the preallocation was;
   for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
   You can also run with the option -info and look for messages with the string
   malloc in them to see if additional memory allocation was needed.

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

.vb
           0 1 2 3 4 5 6 7 8 9 10 11
          --------------------------
   row 3  |. . . d d d o o o o  o  o
   row 4  |. . . d d d o o o o  o  o
   row 5  |. . . d d d o o o o  o  o
          --------------------------
.ve

   Thus, any entries in the d locations are stored in the d (diagonal)
   submatrix, and any entries in the o locations are stored in the
   o (off-diagonal) submatrix.  Note that the d matrix is stored in
   MatSeqSBAIJ format and the o submatrix in MATSEQBAIJ format.

   Now d_nz should indicate the number of block nonzeros per row in the upper triangular
   plus the diagonal part of the d matrix,
   and o_nz should indicate the number of block nonzeros per row in the o matrix

   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices.

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqSBAIJ(), MatSetValues(), MatCreateBAIJ(), PetscSplitOwnership()
@*/
PetscErrorCode  MatMPISBAIJSetPreallocation(Mat B,PetscInt bs,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  PetscTryMethod(B,"MatMPISBAIJSetPreallocation_C",(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]),(B,bs,d_nz,d_nnz,o_nz,o_nnz));
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSBAIJ - Creates a sparse parallel matrix in symmetric block AIJ format
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
.  d_nz  - number of block nonzeros per block row in diagonal portion of local
           submatrix  (same for all local rows)
.  d_nnz - array containing the number of block nonzeros in the various block rows
           in the upper triangular portion of the in diagonal portion of the local
           (possibly different for each block block row) or NULL.
           If you plan to factor the matrix you must leave room for the diagonal entry and
           set its value even if it is zero.
.  o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or NULL.

   Output Parameter:
.  A - the matrix

   Options Database Keys:
+   -mat_no_unroll - uses code that does not unroll the loops in the
                     block calculations (much slower)
.   -mat_block_size - size of the blocks to use
-   -mat_mpi - use the parallel matrix data structures even on one processor
               (defaults to using SeqBAIJ format on one processor)

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   The number of rows and columns must be divisible by blocksize.
   This matrix type does not support complex Hermitian operation.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one processor
   than it must be used on all processors that share the object for that argument.

   If the *_nnz parameter is given then the *_nz parameter is ignored

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
   row 3  |. . . d d d o o o o  o  o
   row 4  |. . . d d d o o o o  o  o
   row 5  |. . . d d d o o o o  o  o
          --------------------------
.ve

   Thus, any entries in the d locations are stored in the d (diagonal)
   submatrix, and any entries in the o locations are stored in the
   o (off-diagonal) submatrix.  Note that the d matrix is stored in
   MatSeqSBAIJ format and the o submatrix in MATSEQBAIJ format.

   Now d_nz should indicate the number of block nonzeros per row in the upper triangular
   plus the diagonal part of the d matrix,
   and o_nz should indicate the number of block nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices.

   Level: intermediate

.seealso: MatCreate(), MatCreateSeqSBAIJ(), MatSetValues(), MatCreateBAIJ()
@*/

PetscErrorCode  MatCreateSBAIJ(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,M,N));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size > 1) {
    PetscCall(MatSetType(*A,MATMPISBAIJ));
    PetscCall(MatMPISBAIJSetPreallocation(*A,bs,d_nz,d_nnz,o_nz,o_nnz));
  } else {
    PetscCall(MatSetType(*A,MATSEQSBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(*A,bs,d_nz,d_nnz));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPISBAIJ(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPISBAIJ   *a,*oldmat = (Mat_MPISBAIJ*)matin->data;
  PetscInt       len=0,nt,bs=matin->rmap->bs,mbs=oldmat->mbs;
  PetscScalar    *array;

  PetscFunctionBegin;
  *newmat = NULL;

  PetscCall(MatCreate(PetscObjectComm((PetscObject)matin),&mat));
  PetscCall(MatSetSizes(mat,matin->rmap->n,matin->cmap->n,matin->rmap->N,matin->cmap->N));
  PetscCall(MatSetType(mat,((PetscObject)matin)->type_name));
  PetscCall(PetscLayoutReference(matin->rmap,&mat->rmap));
  PetscCall(PetscLayoutReference(matin->cmap,&mat->cmap));

  mat->factortype   = matin->factortype;
  mat->preallocated = PETSC_TRUE;
  mat->assembled    = PETSC_TRUE;
  mat->insertmode   = NOT_SET_VALUES;

  a      = (Mat_MPISBAIJ*)mat->data;
  a->bs2 = oldmat->bs2;
  a->mbs = oldmat->mbs;
  a->nbs = oldmat->nbs;
  a->Mbs = oldmat->Mbs;
  a->Nbs = oldmat->Nbs;

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

  PetscCall(PetscArraycpy(a->rangebs,oldmat->rangebs,a->size+2));
  if (oldmat->colmap) {
#if defined(PETSC_USE_CTABLE)
    PetscCall(PetscTableCreateCopy(oldmat->colmap,&a->colmap));
#else
    PetscCall(PetscMalloc1(a->Nbs,&a->colmap));
    PetscCall(PetscLogObjectMemory((PetscObject)mat,(a->Nbs)*sizeof(PetscInt)));
    PetscCall(PetscArraycpy(a->colmap,oldmat->colmap,a->Nbs));
#endif
  } else a->colmap = NULL;

  if (oldmat->garray && (len = ((Mat_SeqBAIJ*)(oldmat->B->data))->nbs)) {
    PetscCall(PetscMalloc1(len,&a->garray));
    PetscCall(PetscLogObjectMemory((PetscObject)mat,len*sizeof(PetscInt)));
    PetscCall(PetscArraycpy(a->garray,oldmat->garray,len));
  } else a->garray = NULL;

  PetscCall(MatStashCreate_Private(PetscObjectComm((PetscObject)matin),matin->rmap->bs,&mat->bstash));
  PetscCall(VecDuplicate(oldmat->lvec,&a->lvec));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->lvec));
  PetscCall(VecScatterCopy(oldmat->Mvctx,&a->Mvctx));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->Mvctx));

  PetscCall(VecDuplicate(oldmat->slvec0,&a->slvec0));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec0));
  PetscCall(VecDuplicate(oldmat->slvec1,&a->slvec1));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec1));

  PetscCall(VecGetLocalSize(a->slvec1,&nt));
  PetscCall(VecGetArray(a->slvec1,&array));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,bs*mbs,array,&a->slvec1a));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,nt-bs*mbs,array+bs*mbs,&a->slvec1b));
  PetscCall(VecRestoreArray(a->slvec1,&array));
  PetscCall(VecGetArray(a->slvec0,&array));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,nt-bs*mbs,array+bs*mbs,&a->slvec0b));
  PetscCall(VecRestoreArray(a->slvec0,&array));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec0));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec1));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec0b));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec1a));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec1b));

  /* ierr =  VecScatterCopy(oldmat->sMvctx,&a->sMvctx); - not written yet, replaced by the lazy trick: */
  PetscCall(PetscObjectReference((PetscObject)oldmat->sMvctx));
  a->sMvctx = oldmat->sMvctx;
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->sMvctx));

  PetscCall(MatDuplicate(oldmat->A,cpvalues,&a->A));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->A));
  PetscCall(MatDuplicate(oldmat->B,cpvalues,&a->B));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)a->B));
  PetscCall(PetscFunctionListDuplicate(((PetscObject)matin)->qlist,&((PetscObject)mat)->qlist));
  *newmat = mat;
  PetscFunctionReturn(0);
}

/* Used for both MPIBAIJ and MPISBAIJ matrices */
#define MatLoad_MPISBAIJ_Binary MatLoad_MPIBAIJ_Binary

PetscErrorCode MatLoad_MPISBAIJ(Mat mat,PetscViewer viewer)
{
  PetscBool      isbinary;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCheck(isbinary,PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer type %s not yet supported for reading %s matrices",((PetscObject)viewer)->type_name,((PetscObject)mat)->type_name);
  PetscCall(MatLoad_MPISBAIJ_Binary(mat,viewer));
  PetscFunctionReturn(0);
}

/*XXXXX@
   MatMPISBAIJSetHashTableFactor - Sets the factor required to compute the size of the HashTable.

   Input Parameters:
.  mat  - the matrix
.  fact - factor

   Not Collective on Mat, each process can have a different hash factor

   Level: advanced

  Notes:
   This can also be set by the command line option: -mat_use_hash_table fact

.seealso: MatSetOption()
@XXXXX*/

PetscErrorCode MatGetRowMaxAbs_MPISBAIJ(Mat A,Vec v,PetscInt idx[])
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  Mat_SeqBAIJ    *b = (Mat_SeqBAIJ*)(a->B)->data;
  PetscReal      atmp;
  PetscReal      *work,*svalues,*rvalues;
  PetscInt       i,bs,mbs,*bi,*bj,brow,j,ncols,krow,kcol,col,row,Mbs,bcol;
  PetscMPIInt    rank,size;
  PetscInt       *rowners_bs,dest,count,source;
  PetscScalar    *va;
  MatScalar      *ba;
  MPI_Status     stat;

  PetscFunctionBegin;
  PetscCheck(!idx,PETSC_COMM_SELF,PETSC_ERR_SUP,"Send email to petsc-maint@mcs.anl.gov");
  PetscCall(MatGetRowMaxAbs(a->A,v,NULL));
  PetscCall(VecGetArray(v,&va));

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank));

  bs  = A->rmap->bs;
  mbs = a->mbs;
  Mbs = a->Mbs;
  ba  = b->a;
  bi  = b->i;
  bj  = b->j;

  /* find ownerships */
  rowners_bs = A->rmap->range;

  /* each proc creates an array to be distributed */
  PetscCall(PetscCalloc1(bs*Mbs,&work));

  /* row_max for B */
  if (rank != size-1) {
    for (i=0; i<mbs; i++) {
      ncols = bi[1] - bi[0]; bi++;
      brow  = bs*i;
      for (j=0; j<ncols; j++) {
        bcol = bs*(*bj);
        for (kcol=0; kcol<bs; kcol++) {
          col  = bcol + kcol;                /* local col index */
          col += rowners_bs[rank+1];      /* global col index */
          for (krow=0; krow<bs; krow++) {
            atmp = PetscAbsScalar(*ba); ba++;
            row  = brow + krow;   /* local row index */
            if (PetscRealPart(va[row]) < atmp) va[row] = atmp;
            if (work[col] < atmp) work[col] = atmp;
          }
        }
        bj++;
      }
    }

    /* send values to its owners */
    for (dest=rank+1; dest<size; dest++) {
      svalues = work + rowners_bs[dest];
      count   = rowners_bs[dest+1]-rowners_bs[dest];
      PetscCallMPI(MPI_Send(svalues,count,MPIU_REAL,dest,rank,PetscObjectComm((PetscObject)A)));
    }
  }

  /* receive values */
  if (rank) {
    rvalues = work;
    count   = rowners_bs[rank+1]-rowners_bs[rank];
    for (source=0; source<rank; source++) {
      PetscCallMPI(MPI_Recv(rvalues,count,MPIU_REAL,MPI_ANY_SOURCE,MPI_ANY_TAG,PetscObjectComm((PetscObject)A),&stat));
      /* process values */
      for (i=0; i<count; i++) {
        if (PetscRealPart(va[i]) < rvalues[i]) va[i] = rvalues[i];
      }
    }
  }

  PetscCall(VecRestoreArray(v,&va));
  PetscCall(PetscFree(work));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_MPISBAIJ(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_MPISBAIJ      *mat = (Mat_MPISBAIJ*)matin->data;
  PetscInt          mbs=mat->mbs,bs=matin->rmap->bs;
  PetscScalar       *x,*ptr,*from;
  Vec               bb1;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCheck(its > 0 && lits > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %" PetscInt_FMT " and local its %" PetscInt_FMT " both positive",its,lits);
  PetscCheck(bs <= 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"SSOR for block size > 1 is not yet implemented");

  if (flag == SOR_APPLY_UPPER) {
    PetscCall((*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx));
    PetscFunctionReturn(0);
  }

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      PetscCall((*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,lits,xx));
      its--;
    }

    PetscCall(VecDuplicate(bb,&bb1));
    while (its--) {

      /* lower triangular part: slvec0b = - B^T*xx */
      PetscCall((*mat->B->ops->multtranspose)(mat->B,xx,mat->slvec0b));

      /* copy xx into slvec0a */
      PetscCall(VecGetArray(mat->slvec0,&ptr));
      PetscCall(VecGetArray(xx,&x));
      PetscCall(PetscArraycpy(ptr,x,bs*mbs));
      PetscCall(VecRestoreArray(mat->slvec0,&ptr));

      PetscCall(VecScale(mat->slvec0,-1.0));

      /* copy bb into slvec1a */
      PetscCall(VecGetArray(mat->slvec1,&ptr));
      PetscCall(VecGetArrayRead(bb,&b));
      PetscCall(PetscArraycpy(ptr,b,bs*mbs));
      PetscCall(VecRestoreArray(mat->slvec1,&ptr));

      /* set slvec1b = 0 */
      PetscCall(VecSet(mat->slvec1b,0.0));

      PetscCall(VecScatterBegin(mat->sMvctx,mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD));
      PetscCall(VecRestoreArray(xx,&x));
      PetscCall(VecRestoreArrayRead(bb,&b));
      PetscCall(VecScatterEnd(mat->sMvctx,mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD));

      /* upper triangular part: bb1 = bb1 - B*x */
      PetscCall((*mat->B->ops->multadd)(mat->B,mat->slvec1b,mat->slvec1a,bb1));

      /* local diagonal sweep */
      PetscCall((*mat->A->ops->sor)(mat->A,bb1,omega,SOR_SYMMETRIC_SWEEP,fshift,lits,lits,xx));
    }
    PetscCall(VecDestroy(&bb1));
  } else if ((flag & SOR_LOCAL_FORWARD_SWEEP) && (its == 1) && (flag & SOR_ZERO_INITIAL_GUESS)) {
    PetscCall((*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx));
  } else if ((flag & SOR_LOCAL_BACKWARD_SWEEP) && (its == 1) && (flag & SOR_ZERO_INITIAL_GUESS)) {
    PetscCall((*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx));
  } else if (flag & SOR_EISENSTAT) {
    Vec               xx1;
    PetscBool         hasop;
    const PetscScalar *diag;
    PetscScalar       *sl,scale = (omega - 2.0)/omega;
    PetscInt          i,n;

    if (!mat->xx1) {
      PetscCall(VecDuplicate(bb,&mat->xx1));
      PetscCall(VecDuplicate(bb,&mat->bb1));
    }
    xx1 = mat->xx1;
    bb1 = mat->bb1;

    PetscCall((*mat->A->ops->sor)(mat->A,bb,omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_BACKWARD_SWEEP),fshift,lits,1,xx));

    if (!mat->diag) {
      /* this is wrong for same matrix with new nonzero values */
      PetscCall(MatCreateVecs(matin,&mat->diag,NULL));
      PetscCall(MatGetDiagonal(matin,mat->diag));
    }
    PetscCall(MatHasOperation(matin,MATOP_MULT_DIAGONAL_BLOCK,&hasop));

    if (hasop) {
      PetscCall(MatMultDiagonalBlock(matin,xx,bb1));
      PetscCall(VecAYPX(mat->slvec1a,scale,bb));
    } else {
      /*
          These two lines are replaced by code that may be a bit faster for a good compiler
      PetscCall(VecPointwiseMult(mat->slvec1a,mat->diag,xx));
      PetscCall(VecAYPX(mat->slvec1a,scale,bb));
      */
      PetscCall(VecGetArray(mat->slvec1a,&sl));
      PetscCall(VecGetArrayRead(mat->diag,&diag));
      PetscCall(VecGetArrayRead(bb,&b));
      PetscCall(VecGetArray(xx,&x));
      PetscCall(VecGetLocalSize(xx,&n));
      if (omega == 1.0) {
        for (i=0; i<n; i++) sl[i] = b[i] - diag[i]*x[i];
        PetscCall(PetscLogFlops(2.0*n));
      } else {
        for (i=0; i<n; i++) sl[i] = b[i] + scale*diag[i]*x[i];
        PetscCall(PetscLogFlops(3.0*n));
      }
      PetscCall(VecRestoreArray(mat->slvec1a,&sl));
      PetscCall(VecRestoreArrayRead(mat->diag,&diag));
      PetscCall(VecRestoreArrayRead(bb,&b));
      PetscCall(VecRestoreArray(xx,&x));
    }

    /* multiply off-diagonal portion of matrix */
    PetscCall(VecSet(mat->slvec1b,0.0));
    PetscCall((*mat->B->ops->multtranspose)(mat->B,xx,mat->slvec0b));
    PetscCall(VecGetArray(mat->slvec0,&from));
    PetscCall(VecGetArray(xx,&x));
    PetscCall(PetscArraycpy(from,x,bs*mbs));
    PetscCall(VecRestoreArray(mat->slvec0,&from));
    PetscCall(VecRestoreArray(xx,&x));
    PetscCall(VecScatterBegin(mat->sMvctx,mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mat->sMvctx,mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD));
    PetscCall((*mat->B->ops->multadd)(mat->B,mat->slvec1b,mat->slvec1a,mat->slvec1a));

    /* local sweep */
    PetscCall((*mat->A->ops->sor)(mat->A,mat->slvec1a,omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_FORWARD_SWEEP),fshift,lits,1,xx1));
    PetscCall(VecAXPY(xx,1.0,xx1));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatSORType is not supported for SBAIJ matrix format");
  PetscFunctionReturn(0);
}

/*@
     MatCreateMPISBAIJWithArrays - creates a MPI SBAIJ matrix using arrays that contain in standard
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
.   i - row indices; that is i[0] = 0, i[row] = i[row-1] + number of block elements in that row block row of the matrix
.   j - column indices
-   a - matrix values

   Output Parameter:
.   mat - the matrix

   Level: intermediate

   Notes:
       The i, j, and a arrays ARE copied by this routine into the internal format used by PETSc;
     thus you CANNOT change the matrix entries by changing the values of a[] after you have
     called this routine. Use MatCreateMPIAIJWithSplitArrays() to avoid needing to copy the arrays.

       The i and j indices are 0 based, and i indices are indices corresponding to the local j array.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIAIJSetPreallocation(), MatMPIAIJSetPreallocationCSR(),
          MPIAIJ, MatCreateAIJ(), MatCreateMPIAIJWithSplitArrays()
@*/
PetscErrorCode  MatCreateMPISBAIJWithArrays(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt i[],const PetscInt j[],const PetscScalar a[],Mat *mat)
{
  PetscFunctionBegin;
  PetscCheck(!i[0],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  PetscCheck(m >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"local number of rows (m) cannot be PETSC_DECIDE, or negative");
  PetscCall(MatCreate(comm,mat));
  PetscCall(MatSetSizes(*mat,m,n,M,N));
  PetscCall(MatSetType(*mat,MATMPISBAIJ));
  PetscCall(MatMPISBAIJSetPreallocationCSR(*mat,bs,i,j,a));
  PetscFunctionReturn(0);
}

/*@C
   MatMPISBAIJSetPreallocationCSR - Creates a sparse parallel matrix in SBAIJ format using the given nonzero structure and (optional) numerical values

   Collective

   Input Parameters:
+  B - the matrix
.  bs - the block size
.  i - the indices into j for the start of each local row (starts with zero)
.  j - the column indices for each local row (starts with zero) these must be sorted for each row
-  v - optional values in the matrix

   Level: advanced

   Notes:
   Though this routine has Preallocation() in the name it also sets the exact nonzero locations of the matrix entries
   and usually the numerical values as well

   Any entries below the diagonal are ignored

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIBAIJSetPreallocation(), MatCreateAIJ(), MPIAIJ
@*/
PetscErrorCode  MatMPISBAIJSetPreallocationCSR(Mat B,PetscInt bs,const PetscInt i[],const PetscInt j[], const PetscScalar v[])
{
  PetscFunctionBegin;
  PetscTryMethod(B,"MatMPISBAIJSetPreallocationCSR_C",(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]),(B,bs,i,j,v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_MPISBAIJ(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  PetscErrorCode ierr;
  PetscInt       m,N,i,rstart,nnz,Ii,bs,cbs;
  PetscInt       *indx;
  PetscScalar    *values;

  PetscFunctionBegin;
  PetscCall(MatGetSize(inmat,&m,&N));
  if (scall == MAT_INITIAL_MATRIX) { /* symbolic phase */
    Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)inmat->data;
    PetscInt       *dnz,*onz,mbs,Nbs,nbs;
    PetscInt       *bindx,rmax=a->rmax,j;
    PetscMPIInt    rank,size;

    PetscCall(MatGetBlockSizes(inmat,&bs,&cbs));
    mbs = m/bs; Nbs = N/cbs;
    if (n == PETSC_DECIDE) {
      PetscCall(PetscSplitOwnershipBlock(comm,cbs,&n,&N));
    }
    nbs = n/cbs;

    PetscCall(PetscMalloc1(rmax,&bindx));
    ierr = MatPreallocateInitialize(comm,mbs,nbs,dnz,onz);PetscCall(ierr); /* inline function, output __end and __rstart are used below */

    PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCallMPI(MPI_Comm_rank(comm,&size));
    if (rank == size-1) {
      /* Check sum(nbs) = Nbs */
      PetscCheck(__end == Nbs,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sum of local block columns %" PetscInt_FMT " != global block columns %" PetscInt_FMT,__end,Nbs);
    }

    rstart = __rstart; /* block rstart of *outmat; see inline function MatPreallocateInitialize */
    PetscCall(MatSetOption(inmat,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE));
    for (i=0; i<mbs; i++) {
      PetscCall(MatGetRow_SeqSBAIJ(inmat,i*bs,&nnz,&indx,NULL)); /* non-blocked nnz and indx */
      nnz  = nnz/bs;
      for (j=0; j<nnz; j++) bindx[j] = indx[j*bs]/bs;
      PetscCall(MatPreallocateSet(i+rstart,nnz,bindx,dnz,onz));
      PetscCall(MatRestoreRow_SeqSBAIJ(inmat,i*bs,&nnz,&indx,NULL));
    }
    PetscCall(MatSetOption(inmat,MAT_GETROW_UPPERTRIANGULAR,PETSC_FALSE));
    PetscCall(PetscFree(bindx));

    PetscCall(MatCreate(comm,outmat));
    PetscCall(MatSetSizes(*outmat,m,n,PETSC_DETERMINE,PETSC_DETERMINE));
    PetscCall(MatSetBlockSizes(*outmat,bs,cbs));
    PetscCall(MatSetType(*outmat,MATSBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(*outmat,bs,0,dnz));
    PetscCall(MatMPISBAIJSetPreallocation(*outmat,bs,0,dnz,0,onz));
    ierr = MatPreallocateFinalize(dnz,onz);PetscCall(ierr);
  }

  /* numeric phase */
  PetscCall(MatGetBlockSizes(inmat,&bs,&cbs));
  PetscCall(MatGetOwnershipRange(*outmat,&rstart,NULL));

  PetscCall(MatSetOption(inmat,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE));
  for (i=0; i<m; i++) {
    PetscCall(MatGetRow_SeqSBAIJ(inmat,i,&nnz,&indx,&values));
    Ii   = i + rstart;
    PetscCall(MatSetValues(*outmat,1,&Ii,nnz,indx,values,INSERT_VALUES));
    PetscCall(MatRestoreRow_SeqSBAIJ(inmat,i,&nnz,&indx,&values));
  }
  PetscCall(MatSetOption(inmat,MAT_GETROW_UPPERTRIANGULAR,PETSC_FALSE));
  PetscCall(MatAssemblyBegin(*outmat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*outmat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}
