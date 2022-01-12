
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(B,nm,2);
  for (i = 0; i < nm; i++) {
    PetscValidHeaderSpecific(X[i],MAT_CLASSID,3);
    ierr = PetscLayoutCompare(B->rmap,X[i]->rmap,&cong);CHKERRQ(ierr);
    if (!cong) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"Not for different layouts");
  }
  PetscValidLogicalCollectiveBool(B,fill,5);
  ierr = MatGetBlockSize(B,&bs);CHKERRQ(ierr);
  ierr = MatGetSize(B,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)B),&preallocator);CHKERRQ(ierr);
  ierr = MatSetType(preallocator,MATPREALLOCATOR);CHKERRQ(ierr);
  ierr = MatSetBlockSize(preallocator,bs);CHKERRQ(ierr);
  ierr = MatSetSizes(preallocator,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetUp(preallocator);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(preallocator,&rstart,&rend);CHKERRQ(ierr);
  for (r = rstart; r < rend; ++r) {
    PetscInt          ncols;
    const PetscInt    *row;
    const PetscScalar *vals;

    for (i = 0; i < nm; i++) {
      ierr = MatGetRow(X[i],r,&ncols,&row,&vals);CHKERRQ(ierr);
      ierr = MatSetValues(preallocator,1,&r,ncols,row,vals,INSERT_VALUES);CHKERRQ(ierr);
      if (symm && symm[i]) {
        ierr = MatSetValues(preallocator,ncols,row,1,&r,vals,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(X[i],r,&ncols,&row,&vals);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(preallocator,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(preallocator,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatPreallocatorPreallocate(preallocator,fill,B);CHKERRQ(ierr);
  ierr = MatDestroy(&preallocator);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPISBAIJ_Basic(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat            B;
  PetscErrorCode ierr;
  PetscInt       r;

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    PetscBool symm = PETSC_TRUE,isdense;
    PetscInt  bs;

    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(B,newtype);CHKERRQ(ierr);
    ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
    ierr = MatSetBlockSize(B,bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompareAny((PetscObject)B,&isdense,MATSEQDENSE,MATMPIDENSE,MATSEQDENSECUDA,"");CHKERRQ(ierr);
    if (!isdense) {
      ierr = MatGetRowUpperTriangular(A);CHKERRQ(ierr);
      ierr = MatPreallocateWithMats_Private(B,1,&A,&symm,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatRestoreRowUpperTriangular(A);CHKERRQ(ierr);
    } else {
      ierr = MatSetUp(B);CHKERRQ(ierr);
    }
  } else {
    B    = *newmat;
    ierr = MatZeroEntries(B);CHKERRQ(ierr);
  }

  ierr = MatGetRowUpperTriangular(A);CHKERRQ(ierr);
  for (r = A->rmap->rstart; r < A->rmap->rend; r++) {
    PetscInt          ncols;
    const PetscInt    *row;
    const PetscScalar *vals;

    ierr = MatGetRow(A,r,&ncols,&row,&vals);CHKERRQ(ierr);
    ierr = MatSetValues(B,1,&r,ncols,row,vals,INSERT_VALUES);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    if (A->hermitian) {
      PetscInt i;
      for (i = 0; i < ncols; i++) {
        ierr = MatSetValue(B,row[i],r,PetscConj(vals[i]),INSERT_VALUES);CHKERRQ(ierr);
      }
    } else {
      ierr = MatSetValues(B,ncols,row,1,&r,vals,INSERT_VALUES);CHKERRQ(ierr);
    }
#else
    ierr = MatSetValues(B,ncols,row,1,&r,vals,INSERT_VALUES);CHKERRQ(ierr);
#endif
    ierr = MatRestoreRow(A,r,&ncols,&row,&vals);CHKERRQ(ierr);
  }
  ierr = MatRestoreRowUpperTriangular(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatStoreValues_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *aij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatStoreValues(aij->A);CHKERRQ(ierr);
  ierr = MatStoreValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatRetrieveValues_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *aij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRetrieveValues(aij->A);CHKERRQ(ierr);
  ierr = MatRetrieveValues(aij->B);CHKERRQ(ierr);
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
    if (a->nonew == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,aimax,a->nonew,MatScalar); \
    N = nrow++ - 1;  \
    /* shift up all the later entries in this row */ \
    ierr  = PetscArraymove(rp+_i+1,rp+_i,N-_i+1);CHKERRQ(ierr); \
    ierr  = PetscArraymove(ap+bs2*(_i+1),ap+bs2*_i,bs2*(N-_i+1));CHKERRQ(ierr); \
    ierr = PetscArrayzero(ap+bs2*_i,bs2);CHKERRQ(ierr);  \
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
    if (b->nonew == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(B,b->mbs,bs2,nrow,brow,bcol,rmax,ba,bi,bj,rp,ap,bimax,b->nonew,MatScalar); \
    N = nrow++ - 1;  \
    /* shift up all the later entries in this row */ \
    ierr  = PetscArraymove(rp+_i+1,rp+_i,N-_i+1);CHKERRQ(ierr); \
    ierr  = PetscArraymove(ap+bs2*(_i+1),ap+bs2*_i,bs2*(N-_i+1));CHKERRQ(ierr); \
    ierr = PetscArrayzero(ap+bs2*_i,bs2);CHKERRQ(ierr); \
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
  PetscErrorCode ierr;
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
      ierr = PetscFree(baij->in_loc);CHKERRQ(ierr);
      ierr = PetscFree(baij->v_loc);CHKERRQ(ierr);
      ierr = PetscMalloc1(n,&baij->in_loc);CHKERRQ(ierr);
      ierr = PetscMalloc1(n,&baij->v_loc);CHKERRQ(ierr);

      baij->n_loc = n;
    }
    in_loc = baij->in_loc;
    v_loc  = baij->v_loc;
  }

  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
    if (PetscUnlikely(im[i] >= mat->rmap->N)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,im[i],mat->rmap->N-1);
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
          /* ierr = MatSetValues_SeqBAIJ(baij->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        } else if (in[j] < 0) continue;
        else if (in[j] >= mat->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,in[j],mat->cmap->N-1);
        else {  /* off-diag entry (B) */
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
            if (col < 0 && !((Mat_SeqSBAIJ*)(baij->A->data))->nonew) {
              ierr = MatDisAssemble_MPISBAIJ(mat);CHKERRQ(ierr);
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
          /* ierr = MatSetValues_SeqBAIJ(baij->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
      }
    } else {  /* off processor entry */
      if (mat->nooffprocentries) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
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
        ierr = MatStashValuesRow_Private(&mat->stash,im[i],n_loc,in_loc,v_loc,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode MatSetValuesBlocked_SeqSBAIJ_Inlined(Mat A,PetscInt row,PetscInt col,const PetscScalar v[],InsertMode is,PetscInt orow,PetscInt ocol)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode    ierr;
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
  if (nonew == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new block index nonzero block (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", orow, ocol);
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
   This routine is exactly duplicated in mpibaij.c
*/
PETSC_STATIC_INLINE PetscErrorCode MatSetValuesBlocked_SeqBAIJ_Inlined(Mat A,PetscInt row,PetscInt col,const PetscScalar v[],InsertMode is,PetscInt orow,PetscInt ocol)
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
  if (nonew == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new global block indexed nonzero block (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", orow, ocol);
  MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
  N = nrow++ - 1; high++;
  /* shift up all the later entries in this row */
  ierr  = PetscArraymove(rp+i+1,rp+i,N-i+1);CHKERRQ(ierr);
  ierr  = PetscArraymove(ap+bs2*(i+1),ap+bs2*i,bs2*(N-i+1));CHKERRQ(ierr);
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
  PetscErrorCode  ierr;
  PetscInt        i,j,ii,jj,row,col,rstart=baij->rstartbs;
  PetscInt        rend=baij->rendbs,cstart=baij->cstartbs,stepval;
  PetscInt        cend=baij->cendbs,bs=mat->rmap->bs,bs2=baij->bs2;

  PetscFunctionBegin;
  if (!barray) {
    ierr         = PetscMalloc1(bs2,&barray);CHKERRQ(ierr);
    baij->barray = barray;
  }

  if (roworiented) {
    stepval = (n-1)*bs;
  } else {
    stepval = (m-1)*bs;
  }
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
    if (PetscUnlikelyDebug(im[i] >= baij->Mbs)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block indexed row too large %" PetscInt_FMT " max %" PetscInt_FMT,im[i],baij->Mbs-1);
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
          ierr = MatSetValuesBlocked_SeqSBAIJ_Inlined(baij->A,row,col,barray,addv,im[i],in[j]);CHKERRQ(ierr);
        } else if (in[j] < 0) continue;
        else if (PetscUnlikelyDebug(in[j] >= baij->Nbs)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Block indexed column too large %" PetscInt_FMT " max %" PetscInt_FMT,in[j],baij->Nbs-1);
        else {
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = MatCreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
            }

#if defined(PETSC_USE_DEBUG)
#if defined(PETSC_USE_CTABLE)
            { PetscInt data;
              ierr = PetscTableFind(baij->colmap,in[j]+1,&data);CHKERRQ(ierr);
              if ((data - 1) % bs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect colmap");
            }
#else
            if ((baij->colmap[in[j]] - 1) % bs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect colmap");
#endif
#endif
#if defined(PETSC_USE_CTABLE)
            ierr = PetscTableFind(baij->colmap,in[j]+1,&col);CHKERRQ(ierr);
            col  = (col - 1)/bs;
#else
            col = (baij->colmap[in[j]] - 1)/bs;
#endif
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
              ierr = MatDisAssemble_MPISBAIJ(mat);CHKERRQ(ierr);
              col  = in[j];
            }
          } else col = in[j];
          ierr = MatSetValuesBlocked_SeqBAIJ_Inlined(baij->B,row,col,barray,addv,im[i],in[j]);CHKERRQ(ierr);
        }
      }
    } else {
      if (mat->nooffprocentries) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process block indexed row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
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

PetscErrorCode MatGetValues_MPISBAIJ(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       bs       = mat->rmap->bs,i,j,bsrstart = mat->rmap->rstart,bsrend = mat->rmap->rend;
  PetscInt       bscstart = mat->cmap->rstart,bscend = mat->cmap->rend,row,col,data;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue; /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %" PetscInt_FMT,idxm[i]); */
    if (idxm[i] >= mat->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,idxm[i],mat->rmap->N-1);
    if (idxm[i] >= bsrstart && idxm[i] < bsrend) {
      row = idxm[i] - bsrstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) continue; /* SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column %" PetscInt_FMT,idxn[j]); */
        if (idxn[j] >= mat->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,idxn[j],mat->cmap->N-1);
        if (idxn[j] >= bscstart && idxn[j] < bscend) {
          col  = idxn[j] - bscstart;
          ierr = MatGetValues_SeqSBAIJ(baij->A,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
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

PetscErrorCode MatNorm_MPISBAIJ(Mat mat,NormType type,PetscReal *norm)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscReal      sum[2],*lnorm2;

  PetscFunctionBegin;
  if (baij->size == 1) {
    ierr =  MatNorm(baij->A,type,norm);CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      ierr    = PetscMalloc1(2,&lnorm2);CHKERRQ(ierr);
      ierr    =  MatNorm(baij->A,type,lnorm2);CHKERRQ(ierr);
      *lnorm2 = (*lnorm2)*(*lnorm2); lnorm2++;            /* squar power of norm(A) */
      ierr    =  MatNorm(baij->B,type,lnorm2);CHKERRQ(ierr);
      *lnorm2 = (*lnorm2)*(*lnorm2); lnorm2--;             /* squar power of norm(B) */
      ierr    = MPIU_Allreduce(lnorm2,sum,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
      *norm   = PetscSqrtReal(sum[0] + 2*sum[1]);
      ierr    = PetscFree(lnorm2);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY || type == NORM_1) { /* max row/column sum */
      Mat_SeqSBAIJ *amat=(Mat_SeqSBAIJ*)baij->A->data;
      Mat_SeqBAIJ  *bmat=(Mat_SeqBAIJ*)baij->B->data;
      PetscReal    *rsum,*rsum2,vabs;
      PetscInt     *jj,*garray=baij->garray,rstart=baij->rstartbs,nz;
      PetscInt     brow,bcol,col,bs=baij->A->rmap->bs,row,grow,gcol,mbs=amat->mbs;
      MatScalar    *v;

      ierr = PetscMalloc2(mat->cmap->N,&rsum,mat->cmap->N,&rsum2);CHKERRQ(ierr);
      ierr = PetscArrayzero(rsum,mat->cmap->N);CHKERRQ(ierr);
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
        ierr = PetscLogFlops(nz*bs*bs);CHKERRQ(ierr);
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
        ierr = PetscLogFlops(nz*bs*bs);CHKERRQ(ierr);
      }
      ierr  = MPIU_Allreduce(rsum,rsum2,mat->cmap->N,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
      *norm = 0.0;
      for (col=0; col<mat->cmap->N; col++) {
        if (rsum2[col] > *norm) *norm = rsum2[col];
      }
      ierr = PetscFree2(rsum,rsum2);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for this norm yet");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyBegin_MPISBAIJ(Mat mat,MatAssemblyType mode)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nstash,reallocs;

  PetscFunctionBegin;
  if (baij->donotstash || mat->nooffprocentries) PetscFunctionReturn(0);

  ierr = MatStashScatterBegin_Private(mat,&mat->stash,mat->rmap->range);CHKERRQ(ierr);
  ierr = MatStashScatterBegin_Private(mat,&mat->bstash,baij->rangebs);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo(mat,"Stash has %" PetscInt_FMT " entries,uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo(mat,"Block-Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPISBAIJ(Mat mat,MatAssemblyType mode)
{
  Mat_MPISBAIJ   *baij=(Mat_MPISBAIJ*)mat->data;
  Mat_SeqSBAIJ   *a   =(Mat_SeqSBAIJ*)baij->A->data;
  PetscErrorCode ierr;
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
        ierr = MatSetValues_MPISBAIJ(mat,1,row+i,ncols,col+i,val+i,mat->insertmode);CHKERRQ(ierr);
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

    ((Mat_SeqBAIJ*)baij->B->data)->roworiented = PETSC_FALSE; /* b->roworinted */
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
        ierr = MatSetValuesBlocked_MPISBAIJ(mat,1,row+i,ncols,col+i,val+i*bs2,mat->insertmode);CHKERRQ(ierr);
        i    = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->bstash);CHKERRQ(ierr);

    baij->roworiented = r1;
    a->roworiented    = r2;

    ((Mat_SeqBAIJ*)baij->B->data)->roworiented = r3; /* b->roworinted */
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
      ierr = MatDisAssemble_MPISBAIJ(mat);CHKERRQ(ierr);
    }
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPISBAIJ(mat);CHKERRQ(ierr); /* setup Mvctx and sMvctx */
  }
  ierr = MatAssemblyBegin(baij->B,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->B,mode);CHKERRQ(ierr);

  ierr = PetscFree2(baij->rowvalues,baij->rowindices);CHKERRQ(ierr);

  baij->rowvalues = NULL;

  /* if no new nonzero locations are allowed in matrix then only set the matrix state the first time through */
  if ((!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) || !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
    PetscObjectState state = baij->A->nonzerostate + baij->B->nonzerostate;
    ierr = MPIU_Allreduce(&state,&mat->nonzerostate,1,MPIU_INT64,MPI_SUM,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatSetValues_MPIBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
#include <petscdraw.h>
static PetscErrorCode MatView_MPISBAIJ_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPISBAIJ      *baij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode    ierr;
  PetscInt          bs   = mat->rmap->bs;
  PetscMPIInt       rank = baij->rank;
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
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %" PetscInt_FMT " nz %" PetscInt_FMT " nz alloced %" PetscInt_FMT " bs %" PetscInt_FMT " mem %g\n",rank,mat->rmap->n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,mat->rmap->bs,(double)info.memory);CHKERRQ(ierr);
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
    Mat          A;
    Mat_SeqSBAIJ *Aloc;
    Mat_SeqBAIJ  *Bloc;
    PetscInt     M = mat->rmap->N,N = mat->cmap->N,*ai,*aj,col,i,j,k,*rvals,mbs = baij->mbs;
    MatScalar    *a;
    const char   *matname;

    /* Should this be the same type as mat? */
    ierr = MatCreate(PetscObjectComm((PetscObject)mat),&A);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = MatSetSizes(A,M,N,M,N);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,M,N);CHKERRQ(ierr);
    }
    ierr = MatSetType(A,MATMPISBAIJ);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(A,mat->rmap->bs,0,NULL,0,NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)A);CHKERRQ(ierr);

    /* copy over the A part */
    Aloc = (Mat_SeqSBAIJ*)baij->A->data;
    ai   = Aloc->i; aj = Aloc->j; a = Aloc->a;
    ierr = PetscMalloc1(bs,&rvals);CHKERRQ(ierr);

    for (i=0; i<mbs; i++) {
      rvals[0] = bs*(baij->rstartbs + i);
      for (j=1; j<bs; j++) rvals[j] = rvals[j-1] + 1;
      for (j=ai[i]; j<ai[i+1]; j++) {
        col = (baij->cstartbs+aj[j])*bs;
        for (k=0; k<bs; k++) {
          ierr = MatSetValues_MPISBAIJ(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
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
          ierr = MatSetValues_MPIBAIJ(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
          col++;
          a += bs;
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
      ierr = PetscObjectSetName((PetscObject)((Mat_MPISBAIJ*)(A->data))->A,matname);CHKERRQ(ierr);
      ierr = MatView_SeqSBAIJ(((Mat_MPISBAIJ*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Used for both MPIBAIJ and MPISBAIJ matrices */
#define MatView_MPISBAIJ_Binary MatView_MPIBAIJ_Binary

PetscErrorCode MatView_MPISBAIJ(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isdraw,issocket,isbinary;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii || isdraw || issocket) {
    ierr = MatView_MPISBAIJ_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_MPISBAIJ_Binary(mat,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
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
  ierr = VecDestroy(&baij->slvec0);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->slvec0b);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->slvec1);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->slvec1a);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->slvec1b);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&baij->sMvctx);CHKERRQ(ierr);
  ierr = PetscFree2(baij->rowvalues,baij->rowindices);CHKERRQ(ierr);
  ierr = PetscFree(baij->barray);CHKERRQ(ierr);
  ierr = PetscFree(baij->hd);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->diag);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->bb1);CHKERRQ(ierr);
  ierr = VecDestroy(&baij->xx1);CHKERRQ(ierr);
#if defined(PETSC_USE_REAL_MAT_SINGLE)
  ierr = PetscFree(baij->setvaluescopy);CHKERRQ(ierr);
#endif
  ierr = PetscFree(baij->in_loc);CHKERRQ(ierr);
  ierr = PetscFree(baij->v_loc);CHKERRQ(ierr);
  ierr = PetscFree(baij->rangebs);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)mat,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatStoreValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatRetrieveValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPISBAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPISBAIJSetPreallocationCSR_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpisbaij_elemental_C",NULL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpisbaij_scalapack_C",NULL);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpisbaij_mpiaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpisbaij_mpibaij_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPISBAIJ_Hermitian(Mat A,Vec xx,Vec yy)
{
  Mat_MPISBAIJ      *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar       *from;
  const PetscScalar *x;

  PetscFunctionBegin;
  /* diagonal part */
  ierr = (*a->A->ops->mult)(a->A,xx,a->slvec1a);CHKERRQ(ierr);
  ierr = VecSet(a->slvec1b,0.0);CHKERRQ(ierr);

  /* subdiagonal part */
  if (!a->B->ops->multhermitiantranspose) SETERRQ(PetscObjectComm((PetscObject)a->B),PETSC_ERR_SUP,"Not for type %s",((PetscObject)a->B)->type_name);
  ierr = (*a->B->ops->multhermitiantranspose)(a->B,xx,a->slvec0b);CHKERRQ(ierr);

  /* copy x into the vec slvec0 */
  ierr = VecGetArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);

  ierr = PetscArraycpy(from,x,bs*mbs);CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);

  ierr = VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* supperdiagonal part */
  ierr = (*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPISBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPISBAIJ      *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar       *from;
  const PetscScalar *x;

  PetscFunctionBegin;
  /* diagonal part */
  ierr = (*a->A->ops->mult)(a->A,xx,a->slvec1a);CHKERRQ(ierr);
  ierr = VecSet(a->slvec1b,0.0);CHKERRQ(ierr);

  /* subdiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->slvec0b);CHKERRQ(ierr);

  /* copy x into the vec slvec0 */
  ierr = VecGetArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);

  ierr = PetscArraycpy(from,x,bs*mbs);CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);

  ierr = VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* supperdiagonal part */
  ierr = (*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPISBAIJ_Hermitian(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPISBAIJ      *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar       *from,zero=0.0;
  const PetscScalar *x;

  PetscFunctionBegin;
  /* diagonal part */
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,a->slvec1a);CHKERRQ(ierr);
  ierr = VecSet(a->slvec1b,zero);CHKERRQ(ierr);

  /* subdiagonal part */
  if (!a->B->ops->multhermitiantranspose) SETERRQ(PetscObjectComm((PetscObject)a->B),PETSC_ERR_SUP,"Not for type %s",((PetscObject)a->B)->type_name);
  ierr = (*a->B->ops->multhermitiantranspose)(a->B,xx,a->slvec0b);CHKERRQ(ierr);

  /* copy x into the vec slvec0 */
  ierr = VecGetArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = PetscArraycpy(from,x,bs*mbs);CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&from);CHKERRQ(ierr);

  ierr = VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* supperdiagonal part */
  ierr = (*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPISBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPISBAIJ      *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar       *from,zero=0.0;
  const PetscScalar *x;

  PetscFunctionBegin;
  /* diagonal part */
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,a->slvec1a);CHKERRQ(ierr);
  ierr = VecSet(a->slvec1b,zero);CHKERRQ(ierr);

  /* subdiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->slvec0b);CHKERRQ(ierr);

  /* copy x into the vec slvec0 */
  ierr = VecGetArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = PetscArraycpy(from,x,bs*mbs);CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&from);CHKERRQ(ierr);

  ierr = VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* supperdiagonal part */
  ierr = (*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This only works correctly for square matrices where the subblock A->A is the
   diagonal block
*/
PetscErrorCode MatGetDiagonal_MPISBAIJ(Mat A,Vec v)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* if (a->rmap->N != a->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block"); */
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_MPISBAIJ(Mat A,PetscScalar aa)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(a->A,aa);CHKERRQ(ierr);
  ierr = MatScale(a->B,aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_MPISBAIJ(Mat matin,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPISBAIJ   *mat = (Mat_MPISBAIJ*)matin->data;
  PetscScalar    *vworkA,*vworkB,**pvA,**pvB,*v_p;
  PetscErrorCode ierr;
  PetscInt       bs = matin->rmap->bs,bs2 = mat->bs2,i,*cworkA,*cworkB,**pcA,**pcB;
  PetscInt       nztot,nzA,nzB,lrow,brstart = matin->rmap->rstart,brend = matin->rmap->rend;
  PetscInt       *cmap,*idx_p,cstart = mat->rstartbs;

  PetscFunctionBegin;
  if (mat->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
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
    ierr = PetscMalloc2(max*bs2,&mat->rowvalues,max*bs2,&mat->rowindices);CHKERRQ(ierr);
  }

  if (row < brstart || row >= brend) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local rows");
  lrow = row - brstart;  /* local row index */

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

PetscErrorCode MatRestoreRow_MPISBAIJ(Mat mat,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;

  PetscFunctionBegin;
  if (!baij->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"MatGetRow() must be called first");
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
#if defined(PETSC_USE_COMPLEX)
  PetscErrorCode ierr;
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)mat->data;

  PetscFunctionBegin;
  ierr = MatConjugate(a->A);CHKERRQ(ierr);
  ierr = MatConjugate(a->B);CHKERRQ(ierr);
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatRealPart_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRealPart(a->A);CHKERRQ(ierr);
  ierr = MatRealPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatImaginaryPart(a->A);CHKERRQ(ierr);
  ierr = MatImaginaryPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Check if isrow is a subset of iscol_local, called by MatCreateSubMatrix_MPISBAIJ()
   Input: isrow       - distributed(parallel),
          iscol_local - locally owned (seq)
*/
PetscErrorCode ISEqual_private(IS isrow,IS iscol_local,PetscBool  *flg)
{
  PetscErrorCode ierr;
  PetscInt       sz1,sz2,*a1,*a2,i,j,k,nmatch;
  const PetscInt *ptr1,*ptr2;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(isrow,&sz1);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol_local,&sz2);CHKERRQ(ierr);
  if (sz1 > sz2) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  ierr = ISGetIndices(isrow,&ptr1);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol_local,&ptr2);CHKERRQ(ierr);

  ierr = PetscMalloc1(sz1,&a1);CHKERRQ(ierr);
  ierr = PetscMalloc1(sz2,&a2);CHKERRQ(ierr);
  ierr = PetscArraycpy(a1,ptr1,sz1);CHKERRQ(ierr);
  ierr = PetscArraycpy(a2,ptr2,sz2);CHKERRQ(ierr);
  ierr = PetscSortInt(sz1,a1);CHKERRQ(ierr);
  ierr = PetscSortInt(sz2,a2);CHKERRQ(ierr);

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
  ierr = ISRestoreIndices(isrow,&ptr1);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol_local,&ptr2);CHKERRQ(ierr);
  ierr = PetscFree(a1);CHKERRQ(ierr);
  ierr = PetscFree(a2);CHKERRQ(ierr);
  if (nmatch < sz1) {
    *flg = PETSC_FALSE;
  } else {
    *flg = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrix_MPISBAIJ(Mat mat,IS isrow,IS iscol,MatReuse call,Mat *newmat)
{
  PetscErrorCode ierr;
  IS             iscol_local;
  PetscInt       csize;
  PetscBool      isequal;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(iscol,&csize);CHKERRQ(ierr);
  if (call == MAT_REUSE_MATRIX) {
    ierr = PetscObjectQuery((PetscObject)*newmat,"ISAllGather",(PetscObject*)&iscol_local);CHKERRQ(ierr);
    if (!iscol_local) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Submatrix passed in was not used before, cannot reuse");
  } else {
    PetscBool issorted;

    ierr = ISAllGather(iscol,&iscol_local);CHKERRQ(ierr);
    ierr = ISEqual_private(isrow,iscol_local,&isequal);CHKERRQ(ierr);
    ierr = ISSorted(iscol_local, &issorted);CHKERRQ(ierr);
    if (!isequal || !issorted) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"For symmetric format, iscol must equal isrow and be sorted");
  }

  /* now call MatCreateSubMatrix_MPIBAIJ() */
  ierr = MatCreateSubMatrix_MPIBAIJ_Private(mat,isrow,iscol_local,csize,call,newmat);CHKERRQ(ierr);
  if (call == MAT_INITIAL_MATRIX) {
    ierr = PetscObjectCompose((PetscObject)*newmat,"ISAllGather",(PetscObject)iscol_local);CHKERRQ(ierr);
    ierr = ISDestroy(&iscol_local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *l = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_MPISBAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)matin->data;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_SUBMAT_SINGLEIS:
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
    break;
  case MAT_HERMITIAN:
    MatCheckPreallocated(A,1);
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
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
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
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
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  case MAT_SYMMETRY_ETERNAL:
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix must be symmetric");
    ierr = PetscInfo(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,B);CHKERRQ(ierr);
  }  else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(A,*B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_MPISBAIJ(Mat mat,Vec ll,Vec rr)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  Mat            a     = baij->A, b=baij->B;
  PetscErrorCode ierr;
  PetscInt       nv,m,n;
  PetscBool      flg;

  PetscFunctionBegin;
  if (ll != rr) {
    ierr = VecEqual(ll,rr,&flg);CHKERRQ(ierr);
    if (PetscUnlikely(!flg)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"For symmetric format, left and right scaling vectors must be same");
  }
  if (!ll) PetscFunctionReturn(0);

  ierr = MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
  if (PetscUnlikely(m != n)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"For symmetric format, local size %" PetscInt_FMT " %" PetscInt_FMT " must be same",m,n);

  ierr = VecGetLocalSize(rr,&nv);CHKERRQ(ierr);
  if (PetscUnlikely(nv!=n)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left and right vector non-conforming local size");

  ierr = VecScatterBegin(baij->Mvctx,rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* left diagonalscale the off-diagonal part */
  ierr = (*b->ops->diagonalscale)(b,ll,NULL);CHKERRQ(ierr);

  /* scale the diagonal part */
  ierr = (*a->ops->diagonalscale)(a,ll,rr);CHKERRQ(ierr);

  /* right diagonalscale the off-diagonal part */
  ierr = VecScatterEnd(baij->Mvctx,rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->ops->diagonalscale)(b,NULL,baij->lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUnfactored_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPISBAIJ(Mat,MatDuplicateOption,Mat*);

PetscErrorCode MatEqual_MPISBAIJ(Mat A,Mat B,PetscBool  *flag)
{
  Mat_MPISBAIJ   *matB = (Mat_MPISBAIJ*)B->data,*matA = (Mat_MPISBAIJ*)A->data;
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

PetscErrorCode MatCopy_MPISBAIJ(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;
  PetscBool      isbaij;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)B,&isbaij,MATSEQSBAIJ,MATMPISBAIJ,"");CHKERRQ(ierr);
  if (!isbaij) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"Not for matrix type %s",((PetscObject)B)->type_name);
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if ((str != SAME_NONZERO_PATTERN) || (A->ops->copy != B->ops->copy)) {
    ierr = MatGetRowUpperTriangular(A);CHKERRQ(ierr);
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(A);CHKERRQ(ierr);
  } else {
    Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)A->data;
    Mat_MPISBAIJ *b = (Mat_MPISBAIJ*)B->data;

    ierr = MatCopy(a->A,b->A,str);CHKERRQ(ierr);
    ierr = MatCopy(a->B,b->B,str);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_MPISBAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMPISBAIJSetPreallocation(A,A->rmap->bs,PETSC_DEFAULT,NULL,PETSC_DEFAULT,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_MPISBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPISBAIJ   *xx=(Mat_MPISBAIJ*)X->data,*yy=(Mat_MPISBAIJ*)Y->data;
  PetscBLASInt   bnz,one=1;
  Mat_SeqSBAIJ   *xa,*ya;
  Mat_SeqBAIJ    *xb,*yb;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {
    PetscScalar alpha = a;
    xa   = (Mat_SeqSBAIJ*)xx->A->data;
    ya   = (Mat_SeqSBAIJ*)yy->A->data;
    ierr = PetscBLASIntCast(xa->nz,&bnz);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&alpha,xa->a,&one,ya->a,&one));
    xb   = (Mat_SeqBAIJ*)xx->B->data;
    yb   = (Mat_SeqBAIJ*)yy->B->data;
    ierr = PetscBLASIntCast(xb->nz,&bnz);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bnz,&alpha,xb->a,&one,yb->a,&one));
    ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    ierr = MatSetOption(X,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
    ierr = MatSetOption(X,MAT_GETROW_UPPERTRIANGULAR,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    Mat      B;
    PetscInt *nnz_d,*nnz_o,bs=Y->rmap->bs;
    if (bs != X->rmap->bs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Matrices must have same block size");
    ierr = MatGetRowUpperTriangular(X);CHKERRQ(ierr);
    ierr = MatGetRowUpperTriangular(Y);CHKERRQ(ierr);
    ierr = PetscMalloc1(yy->A->rmap->N,&nnz_d);CHKERRQ(ierr);
    ierr = PetscMalloc1(yy->B->rmap->N,&nnz_o);CHKERRQ(ierr);
    ierr = MatCreate(PetscObjectComm((PetscObject)Y),&B);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)B,((PetscObject)Y)->name);CHKERRQ(ierr);
    ierr = MatSetSizes(B,Y->rmap->n,Y->cmap->n,Y->rmap->N,Y->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(B,Y,Y);CHKERRQ(ierr);
    ierr = MatSetType(B,MATMPISBAIJ);CHKERRQ(ierr);
    ierr = MatAXPYGetPreallocation_SeqSBAIJ(yy->A,xx->A,nnz_d);CHKERRQ(ierr);
    ierr = MatAXPYGetPreallocation_MPIBAIJ(yy->B,yy->garray,xx->B,xx->garray,nnz_o);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(B,bs,0,nnz_d,0,nnz_o);CHKERRQ(ierr);
    ierr = MatAXPY_BasicWithPreallocation(B,Y,a,X,str);CHKERRQ(ierr);
    ierr = MatHeaderMerge(Y,&B);CHKERRQ(ierr);
    ierr = PetscFree(nnz_d);CHKERRQ(ierr);
    ierr = PetscFree(nnz_o);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(X);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_MPISBAIJ(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = MatCreateSubMatrices_MPIBAIJ(A,n,irow,icol,scall,B);CHKERRQ(ierr); /* B[] are sbaij matrices */
  for (i=0; i<n; i++) {
    ierr = ISEqual(irow[i],icol[i],&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = MatSeqSBAIJZeroOps_Private(*B[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_MPISBAIJ(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_MPISBAIJ    *maij = (Mat_MPISBAIJ*)Y->data;
  Mat_SeqSBAIJ    *aij = (Mat_SeqSBAIJ*)maij->A->data;

  PetscFunctionBegin;
  if (!Y->preallocated) {
    ierr = MatMPISBAIJSetPreallocation(Y,Y->rmap->bs,1,NULL,0,NULL);CHKERRQ(ierr);
  } else if (!aij->nz) {
    PetscInt nonew = aij->nonew;
    ierr = MatSeqSBAIJSetPreallocation(maij->A,Y->rmap->bs,1,NULL);CHKERRQ(ierr);
    aij->nonew = nonew;
  }
  ierr = MatShift_Basic(Y,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMissingDiagonal_MPISBAIJ(Mat A,PetscBool  *missing,PetscInt *d)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only works for square matrices");
  ierr = MatMissingDiagonal(a->A,missing,d);CHKERRQ(ierr);
  if (d) {
    PetscInt rstart;
    ierr = MatGetOwnershipRange(A,&rstart,NULL);CHKERRQ(ierr);
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
                                /*144*/MatCreateMPIMatConcatenateSeqMat_MPISBAIJ
};

PetscErrorCode  MatMPISBAIJSetPreallocation_MPISBAIJ(Mat B,PetscInt bs,PetscInt d_nz,const PetscInt *d_nnz,PetscInt o_nz,const PetscInt *o_nnz)
{
  Mat_MPISBAIJ   *b = (Mat_MPISBAIJ*)B->data;
  PetscErrorCode ierr;
  PetscInt       i,mbs,Mbs;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatSetBlockSize(B,PetscAbs(bs));CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);
  if (B->rmap->N > B->cmap->N) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"MPISBAIJ matrix cannot have more rows %" PetscInt_FMT " than columns %" PetscInt_FMT,B->rmap->N,B->cmap->N);
  if (B->rmap->n > B->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MPISBAIJ matrix cannot have more local rows %" PetscInt_FMT " than columns %" PetscInt_FMT,B->rmap->n,B->cmap->n);

  mbs = B->rmap->n/bs;
  Mbs = B->rmap->N/bs;
  if (mbs*bs != B->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"No of local rows %" PetscInt_FMT " must be divisible by blocksize %" PetscInt_FMT,B->rmap->N,bs);

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
  ierr = PetscTableDestroy(&b->colmap);CHKERRQ(ierr);
#else
  ierr = PetscFree(b->colmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(b->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&b->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->Mvctx);CHKERRQ(ierr);
  ierr = VecDestroy(&b->slvec0);CHKERRQ(ierr);
  ierr = VecDestroy(&b->slvec0b);CHKERRQ(ierr);
  ierr = VecDestroy(&b->slvec1);CHKERRQ(ierr);
  ierr = VecDestroy(&b->slvec1a);CHKERRQ(ierr);
  ierr = VecDestroy(&b->slvec1b);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->sMvctx);CHKERRQ(ierr);

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
    ierr = MatSetType(b->A,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
    ierr = MatStashCreate_Private(PetscObjectComm((PetscObject)B),bs,&B->bstash);CHKERRQ(ierr);
  }

  ierr = MatSeqSBAIJSetPreallocation(b->A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(b->B,bs,o_nz,o_nnz);CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  PetscBool      nooffprocentries;

  PetscFunctionBegin;
  if (PetscUnlikely(bs < 1)) SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %" PetscInt_FMT,bs);
  ierr   = PetscLayoutSetBlockSize(B->rmap,bs);CHKERRQ(ierr);
  ierr   = PetscLayoutSetBlockSize(B->cmap,bs);CHKERRQ(ierr);
  ierr   = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr   = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr   = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);
  m      = B->rmap->n/bs;
  rstart = B->rmap->rstart/bs;
  cend   = B->cmap->rend/bs;

  if (PetscUnlikely(ii[0])) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ii[0] must be 0 but it is %" PetscInt_FMT,ii[0]);
  ierr = PetscMalloc2(m,&d_nnz,m,&o_nnz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nz = ii[i+1] - ii[i];
    if (PetscUnlikely(nz < 0)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local row %" PetscInt_FMT " has a negative number of columns %" PetscInt_FMT,i,nz);
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
  ierr = MatMPISBAIJSetPreallocation(B,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
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
      ierr = MatSetValuesBlocked_MPISBAIJ(B,1,&row,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
    } else {                    /* block ordering does not match so we can only insert one block at a time. */
      PetscInt j;
      for (j=0; j<ncols; j++) {
        const PetscScalar *svals = values + (V ? (bs*bs*(ii[i]+j)) : 0);
        ierr = MatSetValuesBlocked_MPISBAIJ(B,1,&row,1,&icols[j],svals,INSERT_VALUES);CHKERRQ(ierr);
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
  ierr    = PetscNewLog(B,&b);CHKERRQ(ierr);
  B->data = (void*)b;
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  B->ops->destroy = MatDestroy_MPISBAIJ;
  B->ops->view    = MatView_MPISBAIJ;
  B->assembled    = PETSC_FALSE;
  B->insertmode   = NOT_SET_VALUES;

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)B),&b->rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&b->size);CHKERRMPI(ierr);

  /* build local table of row and column ownerships */
  ierr = PetscMalloc1(b->size+2,&b->rangebs);CHKERRQ(ierr);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(PetscObjectComm((PetscObject)B),1,&B->stash);CHKERRQ(ierr);

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

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_MPISBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_MPISBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPISBAIJSetPreallocation_C",MatMPISBAIJSetPreallocation_MPISBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPISBAIJSetPreallocationCSR_C",MatMPISBAIJSetPreallocationCSR_MPISBAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaij_elemental_C",MatConvert_MPISBAIJ_Elemental);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaij_scalapack_C",MatConvert_SBAIJ_ScaLAPACK);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaij_mpiaij_C",MatConvert_MPISBAIJ_Basic);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpisbaij_mpibaij_C",MatConvert_MPISBAIJ_Basic);CHKERRQ(ierr);

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

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)B),NULL,"Options for loading MPISBAIJ matrix 1","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_use_hash_table","Use hash table to save memory in constructing matrix","MatSetOption",flg,&flg,NULL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscTryMethod(B,"MatMPISBAIJSetPreallocation_C",(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]),(B,bs,d_nz,d_nnz,o_nz,o_nnz));CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPISBAIJ);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(*A,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(*A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPISBAIJ(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPISBAIJ   *a,*oldmat = (Mat_MPISBAIJ*)matin->data;
  PetscErrorCode ierr;
  PetscInt       len=0,nt,bs=matin->rmap->bs,mbs=oldmat->mbs;
  PetscScalar    *array;

  PetscFunctionBegin;
  *newmat = NULL;

  ierr = MatCreate(PetscObjectComm((PetscObject)matin),&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,matin->rmap->n,matin->cmap->n,matin->rmap->N,matin->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(mat,((PetscObject)matin)->type_name);CHKERRQ(ierr);
  ierr = PetscLayoutReference(matin->rmap,&mat->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(matin->cmap,&mat->cmap);CHKERRQ(ierr);

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

  ierr = PetscArraycpy(a->rangebs,oldmat->rangebs,a->size+2);CHKERRQ(ierr);
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

  ierr = VecDuplicate(oldmat->slvec0,&a->slvec0);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec0);CHKERRQ(ierr);
  ierr = VecDuplicate(oldmat->slvec1,&a->slvec1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec1);CHKERRQ(ierr);

  ierr = VecGetLocalSize(a->slvec1,&nt);CHKERRQ(ierr);
  ierr = VecGetArray(a->slvec1,&array);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,bs*mbs,array,&a->slvec1a);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,nt-bs*mbs,array+bs*mbs,&a->slvec1b);CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec1,&array);CHKERRQ(ierr);
  ierr = VecGetArray(a->slvec0,&array);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,nt-bs*mbs,array+bs*mbs,&a->slvec0b);CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&array);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec0);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec0b);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec1a);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->slvec1b);CHKERRQ(ierr);

  /* ierr =  VecScatterCopy(oldmat->sMvctx,&a->sMvctx); - not written yet, replaced by the lazy trick: */
  ierr      = PetscObjectReference((PetscObject)oldmat->sMvctx);CHKERRQ(ierr);
  a->sMvctx = oldmat->sMvctx;
  ierr      = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->sMvctx);CHKERRQ(ierr);

  ierr    = MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  ierr    = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->A);CHKERRQ(ierr);
  ierr    = MatDuplicate(oldmat->B,cpvalues,&a->B);CHKERRQ(ierr);
  ierr    = PetscLogObjectParent((PetscObject)mat,(PetscObject)a->B);CHKERRQ(ierr);
  ierr    = PetscFunctionListDuplicate(((PetscObject)matin)->qlist,&((PetscObject)mat)->qlist);CHKERRQ(ierr);
  *newmat = mat;
  PetscFunctionReturn(0);
}

/* Used for both MPIBAIJ and MPISBAIJ matrices */
#define MatLoad_MPISBAIJ_Binary MatLoad_MPIBAIJ_Binary

PetscErrorCode MatLoad_MPISBAIJ(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Viewer type %s not yet supported for reading %s matrices",((PetscObject)viewer)->type_name,((PetscObject)mat)->type_name);
  ierr = MatLoad_MPISBAIJ_Binary(mat,viewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,bs,mbs,*bi,*bj,brow,j,ncols,krow,kcol,col,row,Mbs,bcol;
  PetscMPIInt    rank,size;
  PetscInt       *rowners_bs,dest,count,source;
  PetscScalar    *va;
  MatScalar      *ba;
  MPI_Status     stat;

  PetscFunctionBegin;
  if (idx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Send email to petsc-maint@mcs.anl.gov");
  ierr = MatGetRowMaxAbs(a->A,v,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(v,&va);CHKERRQ(ierr);

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank);CHKERRMPI(ierr);

  bs  = A->rmap->bs;
  mbs = a->mbs;
  Mbs = a->Mbs;
  ba  = b->a;
  bi  = b->i;
  bj  = b->j;

  /* find ownerships */
  rowners_bs = A->rmap->range;

  /* each proc creates an array to be distributed */
  ierr = PetscCalloc1(bs*Mbs,&work);CHKERRQ(ierr);

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
      ierr    = MPI_Send(svalues,count,MPIU_REAL,dest,rank,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
    }
  }

  /* receive values */
  if (rank) {
    rvalues = work;
    count   = rowners_bs[rank+1]-rowners_bs[rank];
    for (source=0; source<rank; source++) {
      ierr = MPI_Recv(rvalues,count,MPIU_REAL,MPI_ANY_SOURCE,MPI_ANY_TAG,PetscObjectComm((PetscObject)A),&stat);CHKERRMPI(ierr);
      /* process values */
      for (i=0; i<count; i++) {
        if (PetscRealPart(va[i]) < rvalues[i]) va[i] = rvalues[i];
      }
    }
  }

  ierr = VecRestoreArray(v,&va);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_MPISBAIJ(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_MPISBAIJ      *mat = (Mat_MPISBAIJ*)matin->data;
  PetscErrorCode    ierr;
  PetscInt          mbs=mat->mbs,bs=matin->rmap->bs;
  PetscScalar       *x,*ptr,*from;
  Vec               bb1;
  const PetscScalar *b;

  PetscFunctionBegin;
  if (its <= 0 || lits <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %" PetscInt_FMT " and local its %" PetscInt_FMT " both positive",its,lits);
  if (bs > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SSOR for block size > 1 is not yet implemented");

  if (flag == SOR_APPLY_UPPER) {
    ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,lits,xx);CHKERRQ(ierr);
      its--;
    }

    ierr = VecDuplicate(bb,&bb1);CHKERRQ(ierr);
    while (its--) {

      /* lower triangular part: slvec0b = - B^T*xx */
      ierr = (*mat->B->ops->multtranspose)(mat->B,xx,mat->slvec0b);CHKERRQ(ierr);

      /* copy xx into slvec0a */
      ierr = VecGetArray(mat->slvec0,&ptr);CHKERRQ(ierr);
      ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
      ierr = PetscArraycpy(ptr,x,bs*mbs);CHKERRQ(ierr);
      ierr = VecRestoreArray(mat->slvec0,&ptr);CHKERRQ(ierr);

      ierr = VecScale(mat->slvec0,-1.0);CHKERRQ(ierr);

      /* copy bb into slvec1a */
      ierr = VecGetArray(mat->slvec1,&ptr);CHKERRQ(ierr);
      ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
      ierr = PetscArraycpy(ptr,b,bs*mbs);CHKERRQ(ierr);
      ierr = VecRestoreArray(mat->slvec1,&ptr);CHKERRQ(ierr);

      /* set slvec1b = 0 */
      ierr = VecSet(mat->slvec1b,0.0);CHKERRQ(ierr);

      ierr = VecScatterBegin(mat->sMvctx,mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
      ierr = VecScatterEnd(mat->sMvctx,mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      /* upper triangular part: bb1 = bb1 - B*x */
      ierr = (*mat->B->ops->multadd)(mat->B,mat->slvec1b,mat->slvec1a,bb1);CHKERRQ(ierr);

      /* local diagonal sweep */
      ierr = (*mat->A->ops->sor)(mat->A,bb1,omega,SOR_SYMMETRIC_SWEEP,fshift,lits,lits,xx);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&bb1);CHKERRQ(ierr);
  } else if ((flag & SOR_LOCAL_FORWARD_SWEEP) && (its == 1) && (flag & SOR_ZERO_INITIAL_GUESS)) {
    ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
  } else if ((flag & SOR_LOCAL_BACKWARD_SWEEP) && (its == 1) && (flag & SOR_ZERO_INITIAL_GUESS)) {
    ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
  } else if (flag & SOR_EISENSTAT) {
    Vec               xx1;
    PetscBool         hasop;
    const PetscScalar *diag;
    PetscScalar       *sl,scale = (omega - 2.0)/omega;
    PetscInt          i,n;

    if (!mat->xx1) {
      ierr = VecDuplicate(bb,&mat->xx1);CHKERRQ(ierr);
      ierr = VecDuplicate(bb,&mat->bb1);CHKERRQ(ierr);
    }
    xx1 = mat->xx1;
    bb1 = mat->bb1;

    ierr = (*mat->A->ops->sor)(mat->A,bb,omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_BACKWARD_SWEEP),fshift,lits,1,xx);CHKERRQ(ierr);

    if (!mat->diag) {
      /* this is wrong for same matrix with new nonzero values */
      ierr = MatCreateVecs(matin,&mat->diag,NULL);CHKERRQ(ierr);
      ierr = MatGetDiagonal(matin,mat->diag);CHKERRQ(ierr);
    }
    ierr = MatHasOperation(matin,MATOP_MULT_DIAGONAL_BLOCK,&hasop);CHKERRQ(ierr);

    if (hasop) {
      ierr = MatMultDiagonalBlock(matin,xx,bb1);CHKERRQ(ierr);
      ierr = VecAYPX(mat->slvec1a,scale,bb);CHKERRQ(ierr);
    } else {
      /*
          These two lines are replaced by code that may be a bit faster for a good compiler
      ierr = VecPointwiseMult(mat->slvec1a,mat->diag,xx);CHKERRQ(ierr);
      ierr = VecAYPX(mat->slvec1a,scale,bb);CHKERRQ(ierr);
      */
      ierr = VecGetArray(mat->slvec1a,&sl);CHKERRQ(ierr);
      ierr = VecGetArrayRead(mat->diag,&diag);CHKERRQ(ierr);
      ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
      ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
      ierr = VecGetLocalSize(xx,&n);CHKERRQ(ierr);
      if (omega == 1.0) {
        for (i=0; i<n; i++) sl[i] = b[i] - diag[i]*x[i];
        ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) sl[i] = b[i] + scale*diag[i]*x[i];
        ierr = PetscLogFlops(3.0*n);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(mat->slvec1a,&sl);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(mat->diag,&diag);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
      ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    }

    /* multiply off-diagonal portion of matrix */
    ierr = VecSet(mat->slvec1b,0.0);CHKERRQ(ierr);
    ierr = (*mat->B->ops->multtranspose)(mat->B,xx,mat->slvec0b);CHKERRQ(ierr);
    ierr = VecGetArray(mat->slvec0,&from);CHKERRQ(ierr);
    ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
    ierr = PetscArraycpy(from,x,bs*mbs);CHKERRQ(ierr);
    ierr = VecRestoreArray(mat->slvec0,&from);CHKERRQ(ierr);
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    ierr = VecScatterBegin(mat->sMvctx,mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mat->sMvctx,mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = (*mat->B->ops->multadd)(mat->B,mat->slvec1b,mat->slvec1a,mat->slvec1a);CHKERRQ(ierr);

    /* local sweep */
    ierr = (*mat->A->ops->sor)(mat->A,mat->slvec1a,omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_FORWARD_SWEEP),fshift,lits,1,xx1);CHKERRQ(ierr);
    ierr = VecAXPY(xx,1.0,xx1);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (i[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  if (m < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"local number of rows (m) cannot be PETSC_DECIDE, or negative");
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocationCSR(*mat,bs,i,j,a);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(B,"MatMPISBAIJSetPreallocationCSR_C",(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]),(B,bs,i,j,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_MPISBAIJ(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  PetscErrorCode ierr;
  PetscInt       m,N,i,rstart,nnz,Ii,bs,cbs;
  PetscInt       *indx;
  PetscScalar    *values;

  PetscFunctionBegin;
  ierr = MatGetSize(inmat,&m,&N);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX) { /* symbolic phase */
    Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)inmat->data;
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
      if (__end != Nbs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sum of local block columns %" PetscInt_FMT " != global block columns %" PetscInt_FMT,__end,Nbs);
    }

    rstart = __rstart; /* block rstart of *outmat; see inline function MatPreallocateInitialize */
    ierr = MatSetOption(inmat,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      ierr = MatGetRow_SeqSBAIJ(inmat,i*bs,&nnz,&indx,NULL);CHKERRQ(ierr); /* non-blocked nnz and indx */
      nnz  = nnz/bs;
      for (j=0; j<nnz; j++) bindx[j] = indx[j*bs]/bs;
      ierr = MatPreallocateSet(i+rstart,nnz,bindx,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow_SeqSBAIJ(inmat,i*bs,&nnz,&indx,NULL);CHKERRQ(ierr);
    }
    ierr = MatSetOption(inmat,MAT_GETROW_UPPERTRIANGULAR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscFree(bindx);CHKERRQ(ierr);

    ierr = MatCreate(comm,outmat);CHKERRQ(ierr);
    ierr = MatSetSizes(*outmat,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(*outmat,bs,cbs);CHKERRQ(ierr);
    ierr = MatSetType(*outmat,MATSBAIJ);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(*outmat,bs,0,dnz);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(*outmat,bs,0,dnz,0,onz);CHKERRQ(ierr);
    ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  }

  /* numeric phase */
  ierr = MatGetBlockSizes(inmat,&bs,&cbs);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*outmat,&rstart,NULL);CHKERRQ(ierr);

  ierr = MatSetOption(inmat,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = MatGetRow_SeqSBAIJ(inmat,i,&nnz,&indx,&values);CHKERRQ(ierr);
    Ii   = i + rstart;
    ierr = MatSetValues(*outmat,1,&Ii,nnz,indx,values,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow_SeqSBAIJ(inmat,i,&nnz,&indx,&values);CHKERRQ(ierr);
  }
  ierr = MatSetOption(inmat,MAT_GETROW_UPPERTRIANGULAR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*outmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*outmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
