
#include <../src/mat/impls/baij/mpi/mpibaij.h>    /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <petscblaslapack.h>

extern PetscErrorCode MatSetUpMultiply_MPISBAIJ(Mat); 
extern PetscErrorCode MatSetUpMultiply_MPISBAIJ_2comm(Mat); 
extern PetscErrorCode MatDisAssemble_MPISBAIJ(Mat);
extern PetscErrorCode MatIncreaseOverlap_MPISBAIJ(Mat,PetscInt,IS[],PetscInt);
extern PetscErrorCode MatGetValues_SeqSBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar []);
extern PetscErrorCode MatGetValues_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar []);
extern PetscErrorCode MatSetValues_SeqSBAIJ(Mat,PetscInt,const PetscInt [],PetscInt,const PetscInt [],const PetscScalar [],InsertMode);
extern PetscErrorCode MatSetValuesBlocked_SeqSBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
extern PetscErrorCode MatSetValuesBlocked_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
extern PetscErrorCode MatGetRow_SeqSBAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
extern PetscErrorCode MatRestoreRow_SeqSBAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
extern PetscErrorCode MatZeroRows_SeqSBAIJ(Mat,IS,PetscScalar*,Vec,Vec);
extern PetscErrorCode MatZeroRows_SeqBAIJ(Mat,IS,PetscScalar *,Vec,Vec);
extern PetscErrorCode MatGetRowMaxAbs_MPISBAIJ(Mat,Vec,PetscInt[]);
extern PetscErrorCode MatSOR_MPISBAIJ(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues_MPISBAIJ"
PetscErrorCode  MatStoreValues_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *aij = (Mat_MPISBAIJ *)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatStoreValues(aij->A);CHKERRQ(ierr);
  ierr = MatStoreValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatRetrieveValues_MPISBAIJ"
PetscErrorCode  MatRetrieveValues_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *aij = (Mat_MPISBAIJ *)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRetrieveValues(aij->A);CHKERRQ(ierr);
  ierr = MatRetrieveValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#define  MatSetValues_SeqSBAIJ_A_Private(row,col,value,addv) \
{ \
 \
    brow = row/bs;  \
    rp   = aj + ai[brow]; ap = aa + bs2*ai[brow]; \
    rmax = aimax[brow]; nrow = ailen[brow]; \
      bcol = col/bs; \
      ridx = row % bs; cidx = col % bs; \
      low = 0; high = nrow; \
      while (high-low > 3) { \
        t = (low+high)/2; \
        if (rp[t] > bcol) high = t; \
        else              low  = t; \
      } \
      for (_i=low; _i<high; _i++) { \
        if (rp[_i] > bcol) break; \
        if (rp[_i] == bcol) { \
          bap  = ap +  bs2*_i + bs*cidx + ridx; \
          if (addv == ADD_VALUES) *bap += value;  \
          else                    *bap  = value;  \
          goto a_noinsert; \
        } \
      } \
      if (a->nonew == 1) goto a_noinsert; \
      if (a->nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) into matrix", row, col); \
      MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,aimax,a->nonew,MatScalar); \
      N = nrow++ - 1;  \
      /* shift up all the later entries in this row */ \
      for (ii=N; ii>=_i; ii--) { \
        rp[ii+1] = rp[ii]; \
        ierr = PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(MatScalar));CHKERRQ(ierr); \
      } \
      if (N>=_i) { ierr = PetscMemzero(ap+bs2*_i,bs2*sizeof(MatScalar));CHKERRQ(ierr); }  \
      rp[_i]                      = bcol;  \
      ap[bs2*_i + bs*cidx + ridx] = value;  \
      a_noinsert:; \
    ailen[brow] = nrow; \
} 

#define  MatSetValues_SeqSBAIJ_B_Private(row,col,value,addv) \
{ \
    brow = row/bs;  \
    rp   = bj + bi[brow]; ap = ba + bs2*bi[brow]; \
    rmax = bimax[brow]; nrow = bilen[brow]; \
      bcol = col/bs; \
      ridx = row % bs; cidx = col % bs; \
      low = 0; high = nrow; \
      while (high-low > 3) { \
        t = (low+high)/2; \
        if (rp[t] > bcol) high = t; \
        else              low  = t; \
      } \
      for (_i=low; _i<high; _i++) { \
        if (rp[_i] > bcol) break; \
        if (rp[_i] == bcol) { \
          bap  = ap +  bs2*_i + bs*cidx + ridx; \
          if (addv == ADD_VALUES) *bap += value;  \
          else                    *bap  = value;  \
          goto b_noinsert; \
        } \
      } \
      if (b->nonew == 1) goto b_noinsert; \
      if (b->nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) into matrix", row, col); \
      MatSeqXAIJReallocateAIJ(B,b->mbs,bs2,nrow,brow,bcol,rmax,ba,bi,bj,rp,ap,bimax,b->nonew,MatScalar); \
      N = nrow++ - 1;  \
      /* shift up all the later entries in this row */ \
      for (ii=N; ii>=_i; ii--) { \
        rp[ii+1] = rp[ii]; \
        ierr = PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(MatScalar));CHKERRQ(ierr); \
      } \
      if (N>=_i) { ierr = PetscMemzero(ap+bs2*_i,bs2*sizeof(MatScalar));CHKERRQ(ierr);}  \
      rp[_i]                      = bcol;  \
      ap[bs2*_i + bs*cidx + ridx] = value;  \
      b_noinsert:; \
    bilen[brow] = nrow; \
} 

/* Only add/insert a(i,j) with i<=j (blocks). 
   Any a(i,j) with i>j input by user is ingored. 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPISBAIJ"
PetscErrorCode MatSetValues_MPISBAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  MatScalar      value;
  PetscBool      roworiented = baij->roworiented;
  PetscErrorCode ierr;
  PetscInt       i,j,row,col;
  PetscInt       rstart_orig=mat->rmap->rstart;
  PetscInt       rend_orig=mat->rmap->rend,cstart_orig=mat->cmap->rstart;
  PetscInt       cend_orig=mat->cmap->rend,bs=mat->rmap->bs;

  /* Some Variables required in the macro */
  Mat            A = baij->A;
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)(A)->data; 
  PetscInt       *aimax=a->imax,*ai=a->i,*ailen=a->ilen,*aj=a->j; 
  MatScalar      *aa=a->a;

  Mat            B = baij->B;
  Mat_SeqBAIJ   *b = (Mat_SeqBAIJ*)(B)->data; 
  PetscInt      *bimax=b->imax,*bi=b->i,*bilen=b->ilen,*bj=b->j; 
  MatScalar     *ba=b->a;

  PetscInt      *rp,ii,nrow,_i,rmax,N,brow,bcol; 
  PetscInt      low,high,t,ridx,cidx,bs2=a->bs2; 
  MatScalar     *ap,*bap;

  /* for stash */
  PetscInt      n_loc, *in_loc = PETSC_NULL;
  MatScalar     *v_loc = PETSC_NULL;

  PetscFunctionBegin;
  if (v) PetscValidScalarPointer(v,6);
  if (!baij->donotstash){
    if (n > baij->n_loc) {
      ierr = PetscFree(baij->in_loc);CHKERRQ(ierr);
      ierr = PetscFree(baij->v_loc);CHKERRQ(ierr);
      ierr = PetscMalloc(n*sizeof(PetscInt),&baij->in_loc);CHKERRQ(ierr);
      ierr = PetscMalloc(n*sizeof(MatScalar),&baij->v_loc);CHKERRQ(ierr);
      baij->n_loc = n;
    }
    in_loc = baij->in_loc;
    v_loc  = baij->v_loc;
  }

  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (im[i] >= mat->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i],mat->rmap->N-1);
#endif
    if (im[i] >= rstart_orig && im[i] < rend_orig) { /* this processor entry */
      row = im[i] - rstart_orig;              /* local row index */
      for (j=0; j<n; j++) {
        if (im[i]/bs > in[j]/bs){
          if (a->ignore_ltriangular){
            continue;    /* ignore lower triangular blocks */
          } else {
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Lower triangular value cannot be set for sbaij format. Ignoring these values, run with -mat_ignore_lower_triangular or call MatSetOption(mat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE)");
          }
        }
        if (in[j] >= cstart_orig && in[j] < cend_orig){  /* diag entry (A) */
          col = in[j] - cstart_orig;          /* local col index */
          brow = row/bs; bcol = col/bs;
          if (brow > bcol) continue;  /* ignore lower triangular blocks of A */           
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqSBAIJ_A_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqBAIJ(baij->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        } else if (in[j] < 0) continue;
#if defined(PETSC_USE_DEBUG)
        else if (in[j] >= mat->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[j],mat->cmap->N-1);
#endif
        else {  /* off-diag entry (B) */
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = MatCreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
            }
#if defined (PETSC_USE_CTABLE)
            ierr = PetscTableFind(baij->colmap,in[j]/bs + 1,&col);CHKERRQ(ierr);
            col  = col - 1;
#else
            col = baij->colmap[in[j]/bs] - 1;
#endif
            if (col < 0 && !((Mat_SeqSBAIJ*)(baij->A->data))->nonew) {
              ierr = MatDisAssemble_MPISBAIJ(mat);CHKERRQ(ierr); 
              col =  in[j]; 
              /* Reinitialize the variables required by MatSetValues_SeqBAIJ_B_Private() */
              B = baij->B;
              b = (Mat_SeqBAIJ*)(B)->data; 
              bimax=b->imax;bi=b->i;bilen=b->ilen;bj=b->j; 
              ba=b->a;
            } else col += in[j]%bs;
          } else col = in[j];
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqSBAIJ_B_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqBAIJ(baij->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
      }
    } else {  /* off processor entry */
      if (mat->nooffprocentries) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %D even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
      if (!baij->donotstash) {
        mat->assembled = PETSC_FALSE;
        n_loc = 0;
        for (j=0; j<n; j++){
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

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlocked_MPISBAIJ"
PetscErrorCode MatSetValuesBlocked_MPISBAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const MatScalar v[],InsertMode addv)
{
  Mat_MPISBAIJ    *baij = (Mat_MPISBAIJ*)mat->data;
  const MatScalar *value;
  MatScalar       *barray=baij->barray;
  PetscBool       roworiented = baij->roworiented,ignore_ltriangular = ((Mat_SeqSBAIJ*)baij->A->data)->ignore_ltriangular;
  PetscErrorCode  ierr;
  PetscInt        i,j,ii,jj,row,col,rstart=baij->rstartbs;
  PetscInt        rend=baij->rendbs,cstart=baij->rstartbs,stepval;
  PetscInt        cend=baij->rendbs,bs=mat->rmap->bs,bs2=baij->bs2;

  PetscFunctionBegin;  
  if(!barray) {
    ierr         = PetscMalloc(bs2*sizeof(MatScalar),&barray);CHKERRQ(ierr);
    baij->barray = barray;
  }

  if (roworiented) { 
    stepval = (n-1)*bs;
  } else {
    stepval = (m-1)*bs;
  }
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (im[i] >= baij->Mbs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large, row %D max %D",im[i],baij->Mbs-1);
#endif
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
        } else if((!roworiented) && (m == 1)) {
          barray = (MatScalar*) v + j*bs2;
        } else { /* Here a copy is required */
          if (roworiented) { 
            value = v + i*(stepval+bs)*bs + j*bs;
          } else {
            value = v + j*(stepval+bs)*bs + i*bs;
          }
          for (ii=0; ii<bs; ii++,value+=stepval) {
            for (jj=0; jj<bs; jj++) {
              *barray++  = *value++; 
            }
          }
          barray -=bs2;
        }
          
        if (in[j] >= cstart && in[j] < cend){
          col  = in[j] - cstart;
          ierr = MatSetValuesBlocked_SeqSBAIJ(baij->A,1,&row,1,&col,barray,addv);CHKERRQ(ierr);
        }
        else if (in[j] < 0) continue;
#if defined(PETSC_USE_DEBUG)
        else if (in[j] >= baij->Nbs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large, col %D max %D",in[j],baij->Nbs-1);
#endif
        else {
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = MatCreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
            }

#if defined(PETSC_USE_DEBUG)
#if defined (PETSC_USE_CTABLE)
            { PetscInt data;
              ierr = PetscTableFind(baij->colmap,in[j]+1,&data);CHKERRQ(ierr);
              if ((data - 1) % bs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect colmap");
            }
#else
            if ((baij->colmap[in[j]] - 1) % bs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect colmap");
#endif
#endif
#if defined (PETSC_USE_CTABLE)
	    ierr = PetscTableFind(baij->colmap,in[j]+1,&col);CHKERRQ(ierr);
            col  = (col - 1)/bs;
#else
            col = (baij->colmap[in[j]] - 1)/bs;
#endif
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
              ierr = MatDisAssemble_MPISBAIJ(mat);CHKERRQ(ierr); 
              col =  in[j];              
            }
          }
          else col = in[j];
          ierr = MatSetValuesBlocked_SeqBAIJ(baij->B,1,&row,1,&col,barray,addv);CHKERRQ(ierr);
        }
      }
    } else {
      if (mat->nooffprocentries) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %D even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
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

#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_MPISBAIJ"
PetscErrorCode MatGetValues_MPISBAIJ(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       bs=mat->rmap->bs,i,j,bsrstart = mat->rmap->rstart,bsrend = mat->rmap->rend;
  PetscInt       bscstart = mat->cmap->rstart,bscend = mat->cmap->rend,row,col,data;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue; /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %D",idxm[i]); */
    if (idxm[i] >= mat->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",idxm[i],mat->rmap->N-1);
    if (idxm[i] >= bsrstart && idxm[i] < bsrend) {
      row = idxm[i] - bsrstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) continue; /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column %D",idxn[j]); */
        if (idxn[j] >= mat->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",idxn[j],mat->cmap->N-1);
        if (idxn[j] >= bscstart && idxn[j] < bscend){
          col = idxn[j] - bscstart;
          ierr = MatGetValues_SeqSBAIJ(baij->A,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
        } else {
          if (!baij->colmap) {
            ierr = MatCreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
          } 
#if defined (PETSC_USE_CTABLE)
          ierr = PetscTableFind(baij->colmap,idxn[j]/bs+1,&data);CHKERRQ(ierr);
          data --;
#else
          data = baij->colmap[idxn[j]/bs]-1;
#endif
          if((data < 0) || (baij->garray[data/bs] != idxn[j]/bs)) *(v+i*n+j) = 0.0;
          else {
            col  = data + idxn[j]%bs;
            ierr = MatGetValues_SeqBAIJ(baij->B,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
          } 
        }
      }
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local values currently supported");
    }
  }
 PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_MPISBAIJ"
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
      ierr = PetscMalloc(2*sizeof(PetscReal),&lnorm2);CHKERRQ(ierr); 
      ierr =  MatNorm(baij->A,type,lnorm2);CHKERRQ(ierr);
      *lnorm2 = (*lnorm2)*(*lnorm2); lnorm2++;            /* squar power of norm(A) */
      ierr =  MatNorm(baij->B,type,lnorm2);CHKERRQ(ierr);
      *lnorm2 = (*lnorm2)*(*lnorm2); lnorm2--;             /* squar power of norm(B) */
      ierr = MPI_Allreduce(lnorm2,&sum,2,MPIU_REAL,MPIU_SUM,((PetscObject)mat)->comm);CHKERRQ(ierr);
      *norm = PetscSqrtReal(sum[0] + 2*sum[1]);
      ierr = PetscFree(lnorm2);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY || type == NORM_1) { /* max row/column sum */
      Mat_SeqSBAIJ *amat=(Mat_SeqSBAIJ*)baij->A->data;
      Mat_SeqBAIJ  *bmat=(Mat_SeqBAIJ*)baij->B->data;
      PetscReal    *rsum,*rsum2,vabs; 
      PetscInt     *jj,*garray=baij->garray,rstart=baij->rstartbs,nz;
      PetscInt     brow,bcol,col,bs=baij->A->rmap->bs,row,grow,gcol,mbs=amat->mbs;
      MatScalar    *v;

      ierr  = PetscMalloc2(mat->cmap->N,PetscReal,&rsum,mat->cmap->N,PetscReal,&rsum2);CHKERRQ(ierr);
      ierr  = PetscMemzero(rsum,mat->cmap->N*sizeof(PetscReal));CHKERRQ(ierr);
      /* Amat */
      v = amat->a; jj = amat->j;
      for (brow=0; brow<mbs; brow++) {
        grow = bs*(rstart + brow);
        nz = amat->i[brow+1] - amat->i[brow];
        for (bcol=0; bcol<nz; bcol++){
          gcol = bs*(rstart + *jj); jj++;
          for (col=0; col<bs; col++){
            for (row=0; row<bs; row++){
              vabs = PetscAbsScalar(*v); v++;
              rsum[gcol+col] += vabs;  
              /* non-diagonal block */
              if (bcol > 0 && vabs > 0.0) rsum[grow+row] += vabs; 
            }
          }
        }
      }
      /* Bmat */
      v = bmat->a; jj = bmat->j;
      for (brow=0; brow<mbs; brow++) {
        grow = bs*(rstart + brow);
        nz = bmat->i[brow+1] - bmat->i[brow];
        for (bcol=0; bcol<nz; bcol++){
          gcol = bs*garray[*jj]; jj++;
          for (col=0; col<bs; col++){
            for (row=0; row<bs; row++){
              vabs = PetscAbsScalar(*v); v++;
              rsum[gcol+col] += vabs;
              rsum[grow+row] += vabs; 
            }
          }
        }
      }
      ierr = MPI_Allreduce(rsum,rsum2,mat->cmap->N,MPIU_REAL,MPIU_SUM,((PetscObject)mat)->comm);CHKERRQ(ierr);
      *norm = 0.0;
      for (col=0; col<mat->cmap->N; col++) {
        if (rsum2[col] > *norm) *norm = rsum2[col];
      }
      ierr = PetscFree2(rsum,rsum2);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for this norm yet");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_MPISBAIJ"
PetscErrorCode MatAssemblyBegin_MPISBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nstash,reallocs;
  InsertMode     addv;

  PetscFunctionBegin;
  if (baij->donotstash || mat->nooffprocentries) {
    PetscFunctionReturn(0);
  }

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,((PetscObject)mat)->comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_ARG_WRONGSTATE,"Some processors inserted others added");
  mat->insertmode = addv; /* in case this processor had no cache */

  ierr = MatStashScatterBegin_Private(mat,&mat->stash,mat->rmap->range);CHKERRQ(ierr);
  ierr = MatStashScatterBegin_Private(mat,&mat->bstash,baij->rangebs);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(mat,"Stash has %D entries,uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(mat,"Block-Stash has %D entries, uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPISBAIJ"
PetscErrorCode MatAssemblyEnd_MPISBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPISBAIJ   *baij=(Mat_MPISBAIJ*)mat->data;
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)baij->A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart,ncols,flg,bs2=baij->bs2;
  PetscInt       *row,*col;
  PetscBool      other_disassembled;
  PetscMPIInt    n;
  PetscBool      r1,r2,r3;
  MatScalar      *val;
  InsertMode     addv = mat->insertmode;

  /* do not use 'b=(Mat_SeqBAIJ*)baij->B->data' as B can be reset in disassembly */
  PetscFunctionBegin;

  if (!baij->donotstash &&  !mat->nooffprocentries) { 
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        /* Now assemble all these values with a single function call */
        ierr = MatSetValues_MPISBAIJ(mat,1,row+i,ncols,col+i,val+i,addv);CHKERRQ(ierr);
        i = j;
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
    ((Mat_SeqBAIJ*)baij->B->data)->roworiented    = PETSC_FALSE; /* b->roworinted */
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->bstash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;
      
      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        ierr = MatSetValuesBlocked_MPISBAIJ(mat,1,row+i,ncols,col+i,val+i*bs2,addv);CHKERRQ(ierr);
        i = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->bstash);CHKERRQ(ierr);
    baij->roworiented = r1;
    a->roworiented    = r2;
    ((Mat_SeqBAIJ*)baij->B->data)->roworiented    = r3; /* b->roworinted */
  }

  ierr = MatAssemblyBegin(baij->A,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->A,mode);CHKERRQ(ierr);

  /* determine if any processor has disassembled, if so we must 
     also disassemble ourselfs, in order that we may reassemble. */
  /*
     if nonzero structure of submatrix B cannot change then we know that
     no processor disassembled thus we can skip this stuff
  */
  if (!((Mat_SeqBAIJ*)baij->B->data)->nonew)  {
    ierr = MPI_Allreduce(&mat->was_assembled,&other_disassembled,1,MPI_INT,MPI_PROD,((PetscObject)mat)->comm);CHKERRQ(ierr);
    if (mat->was_assembled && !other_disassembled) {
      ierr = MatDisAssemble_MPISBAIJ(mat);CHKERRQ(ierr);
    }
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPISBAIJ(mat);CHKERRQ(ierr); /* setup Mvctx and sMvctx */
  }
  ierr = MatSetOption(baij->B,MAT_CHECK_COMPRESSED_ROW,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(baij->B,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->B,mode);CHKERRQ(ierr);
  
  ierr = PetscFree2(baij->rowvalues,baij->rowindices);CHKERRQ(ierr);
  baij->rowvalues = 0;

  PetscFunctionReturn(0);
}

extern PetscErrorCode MatSetValues_MPIBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
#undef __FUNCT__  
#define __FUNCT__ "MatView_MPISBAIJ_ASCIIorDraworSocket"
static PetscErrorCode MatView_MPISBAIJ_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPISBAIJ      *baij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode    ierr;
  PetscInt          bs = mat->rmap->bs;
  PetscMPIInt       size = baij->size,rank = baij->rank;
  PetscBool         iascii,isdraw;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) { 
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo info;
      ierr = MPI_Comm_rank(((PetscObject)mat)->comm,&rank);CHKERRQ(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %D nz %D nz alloced %D bs %D mem %D\n",rank,mat->rmap->n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,mat->rmap->bs,(PetscInt)info.memory);CHKERRQ(ierr);
      ierr = MatGetInfo(baij->A,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] on-diagonal part: nz %D \n",rank,(PetscInt)info.nz_used);CHKERRQ(ierr);
      ierr = MatGetInfo(baij->B,MAT_LOCAL,&info);CHKERRQ(ierr); 
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] off-diagonal part: nz %D \n",rank,(PetscInt)info.nz_used);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Information on VecScatter used in matrix-vector product: \n");CHKERRQ(ierr);
      ierr = VecScatterView(baij->Mvctx,viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0); 
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"  block size is %D\n",bs);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      PetscFunctionReturn(0);
    }
  }

  if (isdraw) {
    PetscDraw  draw;
    PetscBool  isnull;
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  }

  if (size == 1) {
    ierr = PetscObjectSetName((PetscObject)baij->A,((PetscObject)mat)->name);CHKERRQ(ierr);
    ierr = MatView(baij->A,viewer);CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat          A;
    Mat_SeqSBAIJ *Aloc;
    Mat_SeqBAIJ  *Bloc;
    PetscInt     M = mat->rmap->N,N = mat->cmap->N,*ai,*aj,col,i,j,k,*rvals,mbs = baij->mbs;
    MatScalar    *a;

    /* Should this be the same type as mat? */
    ierr = MatCreate(((PetscObject)mat)->comm,&A);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatSetSizes(A,M,N,M,N);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,M,N);CHKERRQ(ierr);
    }
    ierr = MatSetType(A,MATMPISBAIJ);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(A,mat->rmap->bs,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(mat,A);CHKERRQ(ierr);

    /* copy over the A part */
    Aloc  = (Mat_SeqSBAIJ*)baij->A->data;
    ai    = Aloc->i; aj = Aloc->j; a = Aloc->a;
    ierr  = PetscMalloc(bs*sizeof(PetscInt),&rvals);CHKERRQ(ierr);

    for (i=0; i<mbs; i++) {
      rvals[0] = bs*(baij->rstartbs + i);
      for (j=1; j<bs; j++) { rvals[j] = rvals[j-1] + 1; }
      for (j=ai[i]; j<ai[i+1]; j++) {
        col = (baij->cstartbs+aj[j])*bs;
        for (k=0; k<bs; k++) {
          ierr = MatSetValues_MPISBAIJ(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
          col++; a += bs;
        }
      }
    } 
    /* copy over the B part */
    Bloc = (Mat_SeqBAIJ*)baij->B->data;
    ai = Bloc->i; aj = Bloc->j; a = Bloc->a;
    for (i=0; i<mbs; i++) {
      
      rvals[0] = bs*(baij->rstartbs + i);
      for (j=1; j<bs; j++) { rvals[j] = rvals[j-1] + 1; }
      for (j=ai[i]; j<ai[i+1]; j++) {
        col = baij->garray[aj[j]]*bs;
        for (k=0; k<bs; k++) {
          ierr = MatSetValues_MPIBAIJ(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
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
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscObjectSetName((PetscObject)((Mat_MPISBAIJ*)(A->data))->A,((PetscObject)mat)->name);CHKERRQ(ierr);
          /* Set the type name to MATMPISBAIJ so that the correct type can be printed out by PetscObjectPrintClassNamePrefixType() in MatView_SeqSBAIJ_ASCII()*/
      PetscStrcpy(((PetscObject)((Mat_MPISBAIJ*)(A->data))->A)->type_name,MATMPISBAIJ);
      ierr = MatView(((Mat_MPISBAIJ*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPISBAIJ"
PetscErrorCode MatView_MPISBAIJ(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isdraw,issocket,isbinary;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii || isdraw || issocket || isbinary) { 
    ierr = MatView_MPISBAIJ_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by MPISBAIJ matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPISBAIJ"
PetscErrorCode MatDestroy_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%D,Cols=%D",mat->rmap->N,mat->cmap->N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = MatStashDestroy_Private(&mat->bstash);CHKERRQ(ierr);
  ierr = MatDestroy(&baij->A);CHKERRQ(ierr);
  ierr = MatDestroy(&baij->B);CHKERRQ(ierr);
#if defined (PETSC_USE_CTABLE)
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

  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatStoreValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatRetrieveValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatGetDiagonalBlock_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPISBAIJSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpisbaij_mpisbstrm_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_MPISBAIJ_Hermitian"
PetscErrorCode MatMult_MPISBAIJ_Hermitian(Mat A,Vec xx,Vec yy)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt,mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar    *x,*from;
 
  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A and xx");

  /* diagonal part */
  ierr = (*a->A->ops->mult)(a->A,xx,a->slvec1a);CHKERRQ(ierr); 
  ierr = VecSet(a->slvec1b,0.0);CHKERRQ(ierr); 

  /* subdiagonal part */  
  ierr = (*a->B->ops->multhermitiantranspose)(a->B,xx,a->slvec0b);CHKERRQ(ierr);

  /* copy x into the vec slvec0 */
  ierr = VecGetArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  ierr = PetscMemcpy(from,x,bs*mbs*sizeof(MatScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&from);CHKERRQ(ierr);  
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  
  ierr = VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
  ierr = VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
  /* supperdiagonal part */
  ierr = (*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,yy);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_MPISBAIJ"
PetscErrorCode MatMult_MPISBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt,mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar    *x,*from;
 
  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A and xx");

  /* diagonal part */
  ierr = (*a->A->ops->mult)(a->A,xx,a->slvec1a);CHKERRQ(ierr); 
  ierr = VecSet(a->slvec1b,0.0);CHKERRQ(ierr); 

  /* subdiagonal part */  
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->slvec0b);CHKERRQ(ierr);

  /* copy x into the vec slvec0 */
  ierr = VecGetArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  ierr = PetscMemcpy(from,x,bs*mbs*sizeof(MatScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&from);CHKERRQ(ierr);  
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  
  ierr = VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
  ierr = VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
  /* supperdiagonal part */
  ierr = (*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,yy);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_MPISBAIJ_2comm"
PetscErrorCode MatMult_MPISBAIJ_2comm(Mat A,Vec xx,Vec yy)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A and xx");

  ierr = VecGetLocalSize(yy,&nt);CHKERRQ(ierr);
  if (nt != A->rmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible parition of A and yy");

  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
  /* do diagonal part */
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  /* do supperdiagonal part */
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  /* do subdiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_MPISBAIJ"
PetscErrorCode MatMultAdd_MPISBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,bs=A->rmap->bs;
  PetscScalar    *x,*from,zero=0.0;
 
  PetscFunctionBegin;
  /*
  PetscSynchronizedPrintf(((PetscObject)A)->comm," MatMultAdd is called ...\n");
  PetscSynchronizedFlush(((PetscObject)A)->comm);
  */
  /* diagonal part */
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,a->slvec1a);CHKERRQ(ierr); 
  ierr = VecSet(a->slvec1b,zero);CHKERRQ(ierr); 

  /* subdiagonal part */  
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->slvec0b);CHKERRQ(ierr);

  /* copy x into the vec slvec0 */
  ierr = VecGetArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(from,x,bs*mbs*sizeof(MatScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&from);CHKERRQ(ierr);  
  
  ierr = VecScatterBegin(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecScatterEnd(a->sMvctx,a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
  
  /* supperdiagonal part */
  ierr = (*a->B->ops->multadd)(a->B,a->slvec1b,a->slvec1a,zz);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_MPISBAIJ_2comm"
PetscErrorCode MatMultAdd_MPISBAIJ_2comm(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
  /* do diagonal part */
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  /* do supperdiagonal part */
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);

  /* do subdiagonal part */    
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,a->lvec,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_MPISBAIJ"
PetscErrorCode MatGetDiagonal_MPISBAIJ(Mat A,Vec v)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* if (a->rmap->N != a->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block"); */
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_MPISBAIJ"
PetscErrorCode MatScale_MPISBAIJ(Mat A,PetscScalar aa)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(a->A,aa);CHKERRQ(ierr);
  ierr = MatScale(a->B,aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_MPISBAIJ"
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
      if (max < tmp) { max = tmp; }
    }
    ierr = PetscMalloc2(max*bs2,PetscScalar,&mat->rowvalues,max*bs2,PetscInt,&mat->rowindices);CHKERRQ(ierr);
  }
       
  if (row < brstart || row >= brend) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local rows");
  lrow = row - brstart;  /* local row index */

  pvA = &vworkA; pcA = &cworkA; pvB = &vworkB; pcB = &cworkB;
  if (!v)   {pvA = 0; pvB = 0;}
  if (!idx) {pcA = 0; if (!v) pcB = 0;}
  ierr = (*mat->A->ops->getrow)(mat->A,lrow,&nzA,pcA,pvA);CHKERRQ(ierr);
  ierr = (*mat->B->ops->getrow)(mat->B,lrow,&nzB,pcB,pvB);CHKERRQ(ierr);
  nztot = nzA + nzB;

  cmap  = mat->garray;
  if (v  || idx) {
    if (nztot) {
      /* Sort by increasing column numbers, assuming A and B already sorted */
      PetscInt imark = -1;
      if (v) {
        *v = v_p = mat->rowvalues;
        for (i=0; i<nzB; i++) {
          if (cmap[cworkB[i]/bs] < cstart)   v_p[i] = vworkB[i];
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
            if (cmap[cworkB[i]/bs] < cstart)   
              idx_p[i] = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs ;
            else break;
          }
          imark = i;
        }
        for (i=0; i<nzA; i++)     idx_p[imark+i] = cstart*bs + cworkA[i];
        for (i=imark; i<nzB; i++) idx_p[nzA+i]   = cmap[cworkB[i]/bs]*bs + cworkB[i]%bs ;
      } 
    } else {
      if (idx) *idx = 0;
      if (v)   *v   = 0;
    }
  }
  *nz = nztot;
  ierr = (*mat->A->ops->restorerow)(mat->A,lrow,&nzA,pcA,pvA);CHKERRQ(ierr);
  ierr = (*mat->B->ops->restorerow)(mat->B,lrow,&nzB,pcB,pvB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_MPISBAIJ"
PetscErrorCode MatRestoreRow_MPISBAIJ(Mat mat,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;

  PetscFunctionBegin;
  if (!baij->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"MatGetRow() must be called first");
  baij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowUpperTriangular_MPISBAIJ"
PetscErrorCode MatGetRowUpperTriangular_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  Mat_SeqSBAIJ   *aA = (Mat_SeqSBAIJ*)a->A->data;

  PetscFunctionBegin;
  aA->getrow_utriangular = PETSC_TRUE;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowUpperTriangular_MPISBAIJ"
PetscErrorCode MatRestoreRowUpperTriangular_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  Mat_SeqSBAIJ   *aA = (Mat_SeqSBAIJ*)a->A->data;

  PetscFunctionBegin;
  aA->getrow_utriangular = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRealPart_MPISBAIJ"
PetscErrorCode MatRealPart_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatRealPart(a->A);CHKERRQ(ierr);
  ierr = MatRealPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatImaginaryPart_MPISBAIJ"
PetscErrorCode MatImaginaryPart_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatImaginaryPart(a->A);CHKERRQ(ierr);
  ierr = MatImaginaryPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_MPISBAIJ"
PetscErrorCode MatZeroEntries_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *l = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_MPISBAIJ"
PetscErrorCode MatGetInfo_MPISBAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)matin->data;
  Mat            A = a->A,B = a->B;
  PetscErrorCode ierr;
  PetscReal      isend[5],irecv[5];

  PetscFunctionBegin;
  info->block_size     = (PetscReal)matin->rmap->bs;
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
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_MAX,((PetscObject)matin)->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_SUM,((PetscObject)matin)->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown MatInfoType argument %d",(int)flag);
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPISBAIJ"
PetscErrorCode MatSetOption_MPISBAIJ(Mat A,MatOption op,PetscBool  flg)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  Mat_SeqSBAIJ   *aA = (Mat_SeqSBAIJ*)a->A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_NEW_NONZERO_LOCATION_ERR:
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op,flg);CHKERRQ(ierr);
    break;
  case MAT_ROW_ORIENTED:
    a->roworiented = flg;
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op,flg);CHKERRQ(ierr);
    break;
  case MAT_NEW_DIAGONALS:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = flg;
    break;
  case MAT_USE_HASH_TABLE:
    a->ht_flag = flg;
    break;
  case MAT_HERMITIAN:
    if (!A->assembled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call MatAssemblyEnd() first");
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    A->ops->mult = MatMult_MPISBAIJ_Hermitian;
    break;
  case MAT_SPD:
    A->spd_set                         = PETSC_TRUE;
    A->spd                             = flg;
    if (flg) {
      A->symmetric                     = PETSC_TRUE;
      A->structurally_symmetric        = PETSC_TRUE;
      A->symmetric_set                 = PETSC_TRUE;
      A->structurally_symmetric_set    = PETSC_TRUE;
    }
    break;
  case MAT_SYMMETRIC:
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  case MAT_STRUCTURALLY_SYMMETRIC:
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  case MAT_SYMMETRY_ETERNAL:
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix must be symmetric");
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
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
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_MPISBAIJ"
PetscErrorCode MatTranspose_MPISBAIJ(Mat A,MatReuse reuse,Mat *B)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (MAT_INITIAL_MATRIX || *B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_MPISBAIJ"
PetscErrorCode MatDiagonalScale_MPISBAIJ(Mat mat,Vec ll,Vec rr)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  Mat            a=baij->A, b=baij->B;
  PetscErrorCode ierr;
  PetscInt       nv,m,n;
  PetscBool      flg;

  PetscFunctionBegin;
  if (ll != rr){
    ierr = VecEqual(ll,rr,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"For symmetric format, left and right scaling vectors must be same\n");
  }
  if (!ll) PetscFunctionReturn(0);

  ierr = MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
  if (m != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"For symmetric format, local size %d %d must be same",m,n);
  
  ierr = VecGetLocalSize(rr,&nv);CHKERRQ(ierr);
  if (nv!=n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left and right vector non-conforming local size");

  ierr = VecScatterBegin(baij->Mvctx,rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
   
  /* left diagonalscale the off-diagonal part */
  ierr = (*b->ops->diagonalscale)(b,ll,PETSC_NULL);CHKERRQ(ierr);
    
  /* scale the diagonal part */
  ierr = (*a->ops->diagonalscale)(a,ll,rr);CHKERRQ(ierr);

  /* right diagonalscale the off-diagonal part */
  ierr = VecScatterEnd(baij->Mvctx,rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->ops->diagonalscale)(b,PETSC_NULL,baij->lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUnfactored_MPISBAIJ"
PetscErrorCode MatSetUnfactored_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPISBAIJ(Mat,MatDuplicateOption,Mat *);

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_MPISBAIJ"
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
  ierr = MPI_Allreduce(&flg,flag,1,MPI_INT,MPI_LAND,((PetscObject)A)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy_MPISBAIJ"
PetscErrorCode MatCopy_MPISBAIJ(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ *)A->data;
  Mat_MPISBAIJ   *b = (Mat_MPISBAIJ *)B->data;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if ((str != SAME_NONZERO_PATTERN) || (A->ops->copy != B->ops->copy)) {
    ierr = MatGetRowUpperTriangular(A);CHKERRQ(ierr);
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(A);CHKERRQ(ierr); 
  } else {
    ierr = MatCopy(a->A,b->A,str);CHKERRQ(ierr);
    ierr = MatCopy(a->B,b->B,str);CHKERRQ(ierr);  
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUp_MPISBAIJ"
PetscErrorCode MatSetUp_MPISBAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMPISBAIJSetPreallocation(A,A->rmap->bs,PETSC_DEFAULT,0,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_MPISBAIJ"
PetscErrorCode MatAXPY_MPISBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPISBAIJ   *xx=(Mat_MPISBAIJ *)X->data,*yy=(Mat_MPISBAIJ *)Y->data;
  PetscBLASInt   bnz,one=1;
  Mat_SeqSBAIJ   *xa,*ya;
  Mat_SeqBAIJ    *xb,*yb;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {  
    PetscScalar alpha = a;
    xa = (Mat_SeqSBAIJ *)xx->A->data;
    ya = (Mat_SeqSBAIJ *)yy->A->data;
    bnz = PetscBLASIntCast(xa->nz);
    BLASaxpy_(&bnz,&alpha,xa->a,&one,ya->a,&one);    
    xb = (Mat_SeqBAIJ *)xx->B->data;
    yb = (Mat_SeqBAIJ *)yy->B->data;
    bnz = PetscBLASIntCast(xb->nz);
    BLASaxpy_(&bnz,&alpha,xb->a,&one,yb->a,&one);
  } else {
    ierr = MatGetRowUpperTriangular(X);CHKERRQ(ierr); 
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
    ierr = MatRestoreRowUpperTriangular(X);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_MPISBAIJ"
PetscErrorCode MatGetSubMatrices_MPISBAIJ(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      flg;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    ierr = ISEqual(irow[i],icol[i],&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only get symmetric submatrix for MPISBAIJ matrices");
  }
  ierr = MatGetSubMatrices_MPIBAIJ(A,n,irow,icol,scall,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {
       MatSetValues_MPISBAIJ,
       MatGetRow_MPISBAIJ,
       MatRestoreRow_MPISBAIJ,
       MatMult_MPISBAIJ,
/* 4*/ MatMultAdd_MPISBAIJ,
       MatMult_MPISBAIJ,       /* transpose versions are same as non-transpose */
       MatMultAdd_MPISBAIJ,
       0,
       0,
       0,
/*10*/ 0,
       0,
       0,
       MatSOR_MPISBAIJ,
       MatTranspose_MPISBAIJ,
/*15*/ MatGetInfo_MPISBAIJ,
       MatEqual_MPISBAIJ,
       MatGetDiagonal_MPISBAIJ,
       MatDiagonalScale_MPISBAIJ,
       MatNorm_MPISBAIJ,
/*20*/ MatAssemblyBegin_MPISBAIJ,
       MatAssemblyEnd_MPISBAIJ,
       MatSetOption_MPISBAIJ,
       MatZeroEntries_MPISBAIJ,
/*24*/ 0,
       0,
       0,
       0,
       0,
/*29*/ MatSetUp_MPISBAIJ,
       0,
       0,
       0,
       0,
/*34*/ MatDuplicate_MPISBAIJ,
       0,
       0,
       0,
       0,
/*39*/ MatAXPY_MPISBAIJ,
       MatGetSubMatrices_MPISBAIJ,
       MatIncreaseOverlap_MPISBAIJ,
       MatGetValues_MPISBAIJ,
       MatCopy_MPISBAIJ,
/*44*/ 0,
       MatScale_MPISBAIJ,
       0,
       0,
       0,
/*49*/ 0,
       0,
       0,
       0,
       0,
/*54*/ 0,
       0,
       MatSetUnfactored_MPISBAIJ,
       0,
       MatSetValuesBlocked_MPISBAIJ,
/*59*/ 0,
       0,
       0,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ MatGetRowMaxAbs_MPISBAIJ,
       0,
       0,
       0,
       0,
/*74*/ 0,
       0,
       0,
       0,
       0,
/*79*/ 0,
       0,
       0,
       0,
       MatLoad_MPISBAIJ,
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
/*99*/ 0,
       0,
       0,
       0,
       0,
/*104*/0,
       MatRealPart_MPISBAIJ,
       MatImaginaryPart_MPISBAIJ,
       MatGetRowUpperTriangular_MPISBAIJ,
       MatRestoreRowUpperTriangular_MPISBAIJ,
/*109*/0,
       0,
       0,
       0,
       0,
/*114*/0,
       0,
       0,
       0,
       0,
/*119*/0,
       0,
       0,
       0
};


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonalBlock_MPISBAIJ"
PetscErrorCode  MatGetDiagonalBlock_MPISBAIJ(Mat A,Mat *a)
{
  PetscFunctionBegin;
  *a = ((Mat_MPISBAIJ *)A->data)->A;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatMPISBAIJSetPreallocation_MPISBAIJ"
PetscErrorCode  MatMPISBAIJSetPreallocation_MPISBAIJ(Mat B,PetscInt bs,PetscInt d_nz,PetscInt *d_nnz,PetscInt o_nz,PetscInt *o_nnz)
{
  Mat_MPISBAIJ   *b;
  PetscErrorCode ierr;
  PetscInt       i,mbs,Mbs;

  PetscFunctionBegin;
  if (d_nz == PETSC_DECIDE || d_nz == PETSC_DEFAULT) d_nz = 3;
  if (o_nz == PETSC_DECIDE || o_nz == PETSC_DEFAULT) o_nz = 1;
  if (d_nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nz cannot be less than 0: value %D",d_nz);
  if (o_nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nz cannot be less than 0: value %D",o_nz);

  ierr = PetscLayoutSetBlockSize(B->rmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);

  if (d_nnz) {
    for (i=0; i<B->rmap->n/bs; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than -1: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->rmap->n/bs; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than -1: local row %D value %D",i,o_nnz[i]);
    }
  }

  b   = (Mat_MPISBAIJ*)B->data;
  mbs = B->rmap->n/bs;
  Mbs = B->rmap->N/bs;
  if (mbs*bs != B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"No of local rows %D must be divisible by blocksize %D",B->rmap->N,bs);

  B->rmap->bs  = bs;
  b->bs2 = bs*bs;
  b->mbs = mbs;
  b->nbs = mbs; 
  b->Mbs = Mbs;
  b->Nbs = Mbs; 

  for (i=0; i<=b->size; i++) {
    b->rangebs[i] = B->rmap->range[i]/bs;
  }
  b->rstartbs = B->rmap->rstart/bs;
  b->rendbs   = B->rmap->rend/bs;
  
  b->cstartbs = B->cmap->rstart/bs;
  b->cendbs   = B->cmap->rend/bs;

  if (!B->preallocated) {    
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(B,b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQBAIJ);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(B,b->B);CHKERRQ(ierr);
    ierr = MatStashCreate_Private(((PetscObject)B)->comm,bs,&B->bstash);CHKERRQ(ierr);
  }

  ierr = MatSeqSBAIJSetPreallocation(b->A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(b->B,bs,o_nz,o_nnz);CHKERRQ(ierr);
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMPISBAIJSetPreallocationCSR_MPISBAIJ"
PetscErrorCode MatMPISBAIJSetPreallocationCSR_MPISBAIJ(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[],const PetscScalar V[])
{
  PetscInt       m,rstart,cstart,cend;
  PetscInt       i,j,d,nz,nz_max=0,*d_nnz=0,*o_nnz=0;
  const PetscInt *JJ=0;
  PetscScalar    *values=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (bs < 1) SETERRQ1(((PetscObject)B)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive but it is %D",bs);
  ierr = PetscLayoutSetBlockSize(B->rmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(B->rmap,&bs);CHKERRQ(ierr);
  m      = B->rmap->n/bs;
  rstart = B->rmap->rstart/bs;
  cstart = B->cmap->rstart/bs;
  cend   = B->cmap->rend/bs;

  if (ii[0]) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ii[0] must be 0 but it is %D",ii[0]);
  ierr  = PetscMalloc2(m,PetscInt,&d_nnz,m,PetscInt,&o_nnz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nz = ii[i+1] - ii[i];
    if (nz < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local row %D has a negative number of columns %D",i,nz);
    nz_max = PetscMax(nz_max,nz);
    JJ  = jj + ii[i];
    for (j=0; j<nz; j++) {
      if (*JJ >= cstart) break;
      JJ++;
    }
    d = 0;
    for (; j<nz; j++) {
      if (*JJ++ >= cend) break;
      d++;
    }
    d_nnz[i] = d; 
    o_nnz[i] = nz - d;
  }
  ierr = MatMPISBAIJSetPreallocation(B,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
  ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);

  values = (PetscScalar*)V;
  if (!values) {
    ierr = PetscMalloc(bs*bs*nz_max*sizeof(PetscScalar),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,bs*bs*nz_max*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (i=0; i<m; i++) {
    PetscInt          row    = i + rstart;
    PetscInt          ncols  = ii[i+1] - ii[i];
    const PetscInt    *icols = jj + ii[i];
    const PetscScalar *svals = values + (V ? (bs*bs*ii[i]) : 0);
    ierr = MatSetValuesBlocked_MPISBAIJ(B,1,&row,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
  }

  if (!V) { ierr = PetscFree(values);CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_MUMPS)
extern PetscErrorCode  MatGetFactor_sbaij_mumps(Mat,MatFactorType,Mat*);
#endif
#if defined(PETSC_HAVE_SPOOLES)
extern PetscErrorCode  MatGetFactor_mpisbaij_spooles(Mat,MatFactorType,Mat*);
#endif
#if defined(PETSC_HAVE_PASTIX)
extern PetscErrorCode MatGetFactor_mpisbaij_pastix(Mat,MatFactorType,Mat*);
#endif
EXTERN_C_END 

/*MC
   MATMPISBAIJ - MATMPISBAIJ = "mpisbaij" - A matrix type to be used for distributed symmetric sparse block matrices, 
   based on block compressed sparse row format.  Only the upper triangular portion of the "diagonal" portion of 
   the matrix is stored.

  For complex numbers by default this matrix is symmetric, NOT Hermitian symmetric. To make it Hermitian symmetric you
  can call MatSetOption(Mat, MAT_HERMITIAN); 

   Options Database Keys:
. -mat_type mpisbaij - sets the matrix type to "mpisbaij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPISBAIJ
M*/

EXTERN_C_BEGIN
extern PetscErrorCode MatConvert_MPISBAIJ_MPISBSTRM(Mat,const MatType,MatReuse,Mat*);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPISBAIJ"
PetscErrorCode  MatCreate_MPISBAIJ(Mat B)
{
  Mat_MPISBAIJ   *b;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;

  ierr    = PetscNewLog(B,Mat_MPISBAIJ,&b);CHKERRQ(ierr);
  B->data = (void*)b;
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  B->ops->destroy    = MatDestroy_MPISBAIJ;
  B->ops->view       = MatView_MPISBAIJ;
  B->assembled       = PETSC_FALSE;

  B->insertmode = NOT_SET_VALUES;
  ierr = MPI_Comm_rank(((PetscObject)B)->comm,&b->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)B)->comm,&b->size);CHKERRQ(ierr);

  /* build local table of row and column ownerships */
  ierr  = PetscMalloc((b->size+2)*sizeof(PetscInt),&b->rangebs);CHKERRQ(ierr);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(((PetscObject)B)->comm,1,&B->stash);CHKERRQ(ierr);
  b->donotstash  = PETSC_FALSE;
  b->colmap      = PETSC_NULL;
  b->garray      = PETSC_NULL;
  b->roworiented = PETSC_TRUE;

  /* stuff used in block assembly */
  b->barray       = 0;

  /* stuff used for matrix vector multiply */
  b->lvec         = 0;
  b->Mvctx        = 0;
  b->slvec0       = 0;
  b->slvec0b      = 0;
  b->slvec1       = 0;
  b->slvec1a      = 0;
  b->slvec1b      = 0;
  b->sMvctx       = 0;

  /* stuff for MatGetRow() */
  b->rowindices   = 0;
  b->rowvalues    = 0;
  b->getrowactive = PETSC_FALSE;

  /* hash table stuff */
  b->ht           = 0;
  b->hd           = 0;
  b->ht_size      = 0;
  b->ht_flag      = PETSC_FALSE;
  b->ht_fact      = 0;
  b->ht_total_ct  = 0;
  b->ht_insert_ct = 0;

  /* stuff for MatGetSubMatrices_MPIBAIJ_local() */
  b->ijonly       = PETSC_FALSE;

  b->in_loc       = 0;
  b->v_loc        = 0;
  b->n_loc        = 0;
  ierr = PetscOptionsBegin(((PetscObject)B)->comm,PETSC_NULL,"Options for loading MPISBAIJ matrix 1","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-mat_use_hash_table","Use hash table to save memory in constructing matrix","MatSetOption",PETSC_FALSE,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) { 
      PetscReal fact = 1.39;
      ierr = MatSetOption(B,MAT_USE_HASH_TABLE,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscOptionsReal("-mat_use_hash_table","Use hash table factor","MatMPIBAIJSetHashTableFactor",fact,&fact,PETSC_NULL);CHKERRQ(ierr);
      if (fact <= 1.0) fact = 1.39;
      ierr = MatMPIBAIJSetHashTableFactor(B,fact);CHKERRQ(ierr);
      ierr = PetscInfo1(B,"Hash table Factor used %5.2f\n",fact);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

#if defined(PETSC_HAVE_PASTIX)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_pastix_C",
					   "MatGetFactor_mpisbaij_pastix",
					   MatGetFactor_mpisbaij_pastix);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MUMPS)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_mumps_C",
                                     "MatGetFactor_sbaij_mumps",
                                     MatGetFactor_sbaij_mumps);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SPOOLES)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_spooles_C",
                                     "MatGetFactor_mpisbaij_spooles",
                                     MatGetFactor_mpisbaij_spooles);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatStoreValues_C",
                                     "MatStoreValues_MPISBAIJ",
                                     MatStoreValues_MPISBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatRetrieveValues_C",
                                     "MatRetrieveValues_MPISBAIJ",
                                     MatRetrieveValues_MPISBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetDiagonalBlock_C",
                                     "MatGetDiagonalBlock_MPISBAIJ",
                                     MatGetDiagonalBlock_MPISBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPISBAIJSetPreallocation_C",
                                     "MatMPISBAIJSetPreallocation_MPISBAIJ",
                                     MatMPISBAIJSetPreallocation_MPISBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPISBAIJSetPreallocationCSR_C",
                                     "MatMPISBAIJSetPreallocationCSR_MPISBAIJ",
                                     MatMPISBAIJSetPreallocationCSR_MPISBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpisbaij_mpisbstrm_C",
                                     "MatConvert_MPISBAIJ_MPISBSTRM",
                                      MatConvert_MPISBAIJ_MPISBSTRM);CHKERRQ(ierr);

  B->symmetric                  = PETSC_TRUE;
  B->structurally_symmetric     = PETSC_TRUE;
  B->symmetric_set              = PETSC_TRUE;
  B->structurally_symmetric_set = PETSC_TRUE;
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPISBAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATSBAIJ - MATSBAIJ = "sbaij" - A matrix type to be used for symmetric block sparse matrices.

   This matrix type is identical to MATSEQSBAIJ when constructed with a single process communicator,
   and MATMPISBAIJ otherwise.

   Options Database Keys:
. -mat_type sbaij - sets the matrix type to "sbaij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPISBAIJ,MATSEQSBAIJ,MATMPISBAIJ
M*/

#undef __FUNCT__  
#define __FUNCT__ "MatMPISBAIJSetPreallocation"
/*@C
   MatMPISBAIJSetPreallocation - For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters 
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

   Collective on Mat

   Input Parameters:
+  A - the matrix 
.  bs   - size of blockk
.  d_nz  - number of block nonzeros per block row in diagonal portion of local 
           submatrix  (same for all local rows)
.  d_nnz - array containing the number of block nonzeros in the various block rows 
           in the upper triangular and diagonal part of the in diagonal portion of the local
           (possibly different for each block row) or PETSC_NULL.  If you plan to factor the matrix you must leave room 
           for the diagonal entry and set a value even if it is zero.
.  o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix that is right of the diagonal 
           (possibly different for each block row) or PETSC_NULL.


   Options Database Keys:
.   -mat_no_unroll - uses code that does not unroll the loops in the 
                     block calculations (much slower)
.   -mat_block_size - size of the blocks to use

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
   d_nz=PETSC_DEFAULT and d_nnz=PETSC_NULL for PETSc to control dynamic
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
          -------------------
   row 3  |  . . . d d d o o o o o o
   row 4  |  . . . d d d o o o o o o
   row 5  |  . . . d d d o o o o o o
          -------------------
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

.keywords: matrix, block, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqSBAIJ(), MatSetValues(), MatCreateBAIJ()
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

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSBAIJ"
/*@C
   MatCreateSBAIJ - Creates a sparse parallel matrix in symmetric block AIJ format
   (block compressed row).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameters 
   d_nz (or d_nnz) and o_nz (or o_nnz).  By setting these parameters accurately,
   performance can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  bs   - size of blockk
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
           (possibly different for each block block row) or PETSC_NULL.  
           If you plan to factor the matrix you must leave room for the diagonal entry and 
           set its value even if it is zero.
.  o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or PETSC_NULL.

   Output Parameter:
.  A - the matrix 

   Options Database Keys:
.   -mat_no_unroll - uses code that does not unroll the loops in the 
                     block calculations (much slower)
.   -mat_block_size - size of the blocks to use
.   -mat_mpi - use the parallel matrix data structures even on one processor 
               (defaults to using SeqBAIJ format on one processor)

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly. 
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
   d_nz=PETSC_DEFAULT and d_nnz=PETSC_NULL for PETSc to control dynamic
   memory allocation.  Likewise, specify preallocated storage for the
   off-diagonal part of the local submatrix with o_nz or o_nnz (not both).

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

.vb
           0 1 2 3 4 5 6 7 8 9 10 11
          -------------------
   row 3  |  . . . d d d o o o o o o
   row 4  |  . . . d d d o o o o o o
   row 5  |  . . . d d d o o o o o o
          -------------------
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

.keywords: matrix, block, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqSBAIJ(), MatSetValues(), MatCreateBAIJ()
@*/

PetscErrorCode  MatCreateSBAIJ(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPISBAIJ);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(*A,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(*A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_MPISBAIJ"
static PetscErrorCode MatDuplicate_MPISBAIJ(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPISBAIJ   *a,*oldmat = (Mat_MPISBAIJ*)matin->data;
  PetscErrorCode ierr;
  PetscInt       len=0,nt,bs=matin->rmap->bs,mbs=oldmat->mbs;
  PetscScalar    *array;

  PetscFunctionBegin;
  *newmat       = 0;
  ierr = MatCreate(((PetscObject)matin)->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,matin->rmap->n,matin->cmap->n,matin->rmap->N,matin->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(mat,((PetscObject)matin)->type_name);CHKERRQ(ierr);
  ierr = PetscMemcpy(mat->ops,matin->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  ierr = PetscLayoutReference(matin->rmap,&mat->rmap);CHKERRQ(ierr);  
  ierr = PetscLayoutReference(matin->cmap,&mat->cmap);CHKERRQ(ierr);  
  
  mat->factortype   = matin->factortype; 
  mat->preallocated = PETSC_TRUE;
  mat->assembled    = PETSC_TRUE;
  mat->insertmode   = NOT_SET_VALUES;

  a = (Mat_MPISBAIJ*)mat->data;
  a->bs2   = oldmat->bs2;
  a->mbs   = oldmat->mbs;
  a->nbs   = oldmat->nbs;
  a->Mbs   = oldmat->Mbs;
  a->Nbs   = oldmat->Nbs;


  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  a->donotstash   = oldmat->donotstash;
  a->roworiented  = oldmat->roworiented;
  a->rowindices   = 0;
  a->rowvalues    = 0;
  a->getrowactive = PETSC_FALSE;
  a->barray       = 0;
  a->rstartbs    = oldmat->rstartbs;
  a->rendbs      = oldmat->rendbs;
  a->cstartbs    = oldmat->cstartbs;
  a->cendbs      = oldmat->cendbs;

  /* hash table stuff */
  a->ht           = 0;
  a->hd           = 0;
  a->ht_size      = 0;
  a->ht_flag      = oldmat->ht_flag;
  a->ht_fact      = oldmat->ht_fact;
  a->ht_total_ct  = 0;
  a->ht_insert_ct = 0;
  
  ierr = PetscMemcpy(a->rangebs,oldmat->rangebs,(a->size+2)*sizeof(PetscInt));CHKERRQ(ierr);
  if (oldmat->colmap) {
#if defined (PETSC_USE_CTABLE)
    ierr = PetscTableCreateCopy(oldmat->colmap,&a->colmap);CHKERRQ(ierr); 
#else
    ierr = PetscMalloc((a->Nbs)*sizeof(PetscInt),&a->colmap);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(mat,(a->Nbs)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(a->colmap,oldmat->colmap,(a->Nbs)*sizeof(PetscInt));CHKERRQ(ierr);
#endif
  } else a->colmap = 0;

  if (oldmat->garray && (len = ((Mat_SeqBAIJ*)(oldmat->B->data))->nbs)) {
    ierr = PetscMalloc(len*sizeof(PetscInt),&a->garray);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(mat,len*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(a->garray,oldmat->garray,len*sizeof(PetscInt));CHKERRQ(ierr);
  } else a->garray = 0;
  
  ierr = MatStashCreate_Private(((PetscObject)matin)->comm,matin->rmap->bs,&mat->bstash);CHKERRQ(ierr);
  ierr = VecDuplicate(oldmat->lvec,&a->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterCopy(oldmat->Mvctx,&a->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->Mvctx);CHKERRQ(ierr);

  ierr =  VecDuplicate(oldmat->slvec0,&a->slvec0);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->slvec0);CHKERRQ(ierr);
  ierr =  VecDuplicate(oldmat->slvec1,&a->slvec1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->slvec1);CHKERRQ(ierr);

  ierr = VecGetLocalSize(a->slvec1,&nt);CHKERRQ(ierr);
  ierr = VecGetArray(a->slvec1,&array);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,bs*mbs,array,&a->slvec1a);CHKERRQ(ierr); 
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,nt-bs*mbs,array+bs*mbs,&a->slvec1b);CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec1,&array);CHKERRQ(ierr);
  ierr = VecGetArray(a->slvec0,&array);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,nt-bs*mbs,array+bs*mbs,&a->slvec0b);CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&array);CHKERRQ(ierr); 
  ierr = PetscLogObjectParent(mat,a->slvec0);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->slvec1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->slvec0b);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->slvec1a);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->slvec1b);CHKERRQ(ierr);

  /* ierr =  VecScatterCopy(oldmat->sMvctx,&a->sMvctx); - not written yet, replaced by the lazy trick: */
  ierr = PetscObjectReference((PetscObject)oldmat->sMvctx);CHKERRQ(ierr);
  a->sMvctx = oldmat->sMvctx;
  ierr = PetscLogObjectParent(mat,a->sMvctx);CHKERRQ(ierr);

  ierr =  MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->A);CHKERRQ(ierr);
  ierr =  MatDuplicate(oldmat->B,cpvalues,&a->B);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->B);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)matin)->qlist,&((PetscObject)mat)->qlist);CHKERRQ(ierr);
  *newmat = mat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPISBAIJ"
PetscErrorCode MatLoad_MPISBAIJ(Mat newmat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       i,nz,j,rstart,rend;
  PetscScalar    *vals,*buf;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;
  MPI_Status     status;
  PetscMPIInt    rank,size,tag = ((PetscObject)viewer)->tag,*sndcounts = 0,*browners,maxnz,*rowners,*locrowlens,mmbs;
  PetscInt       header[4],*rowlengths = 0,M,N,m,*cols;
  PetscInt       *procsnz = 0,jj,*mycols,*ibuf;
  PetscInt       bs=1,Mbs,mbs,extra_rows;
  PetscInt       *dlens,*odlens,*mask,*masked1,*masked2,rowcount,odcount;
  PetscInt       dcount,kmax,k,nzcount,tmp,sizesset=1,grows,gcols;
  int            fd;
 
  PetscFunctionBegin;
  ierr = PetscOptionsBegin(comm,PETSC_NULL,"Options for loading MPISBAIJ matrix 2","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
    if (header[3] < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format, cannot load as MPISBAIJ");
  }

  if (newmat->rmap->n < 0 && newmat->rmap->N < 0 && newmat->cmap->n < 0 && newmat->cmap->N < 0) sizesset = 0;

  ierr = MPI_Bcast(header+1,3,MPIU_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2];

  /* If global rows/cols are set to PETSC_DECIDE, set it to the sizes given in the file */
  if (sizesset && newmat->rmap->N < 0) newmat->rmap->N = M;
  if (sizesset && newmat->cmap->N < 0) newmat->cmap->N = N;
  
  /* If global sizes are set, check if they are consistent with that given in the file */
  if (sizesset) {
    ierr = MatGetSize(newmat,&grows,&gcols);CHKERRQ(ierr);
  } 
  if (sizesset && newmat->rmap->N != grows) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Inconsistent # of rows:Matrix in file has (%d) and input matrix has (%d)",M,grows);
  if (sizesset && newmat->cmap->N != gcols) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Inconsistent # of cols:Matrix in file has (%d) and input matrix has (%d)",N,gcols);

  if (M != N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only do square matrices");

  /* 
     This code adds extra rows to make sure the number of rows is 
     divisible by the blocksize
  */
  Mbs        = M/bs;
  extra_rows = bs - M + bs*(Mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  Mbs++;
  if (extra_rows &&!rank) {
    ierr = PetscInfo(viewer,"Padding loaded matrix to match blocksize\n");CHKERRQ(ierr);
  }

  /* determine ownership of all rows */
  if (newmat->rmap->n < 0) { /* PETSC_DECIDE */
    mbs        = Mbs/size + ((Mbs % size) > rank);
    m          = mbs*bs;
  } else { /* User Set */
    m          = newmat->rmap->n;
    mbs        = m/bs;
  }    
  ierr       = PetscMalloc2(size+1,PetscMPIInt,&rowners,size+1,PetscMPIInt,&browners);CHKERRQ(ierr);
  mmbs       = PetscMPIIntCast(mbs);
  ierr       = MPI_Allgather(&mmbs,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) rowners[i] += rowners[i-1];
  for (i=0; i<=size;  i++) browners[i] = rowners[i]*bs;
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 
  
  /* distribute row lengths to all processors */
  ierr = PetscMalloc((rend-rstart)*bs*sizeof(PetscMPIInt),&locrowlens);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscMalloc((M+extra_rows)*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
    for (i=0; i<extra_rows; i++) rowlengths[M+i] = 1;
    ierr = PetscMalloc(size*sizeof(PetscMPIInt),&sndcounts);CHKERRQ(ierr);
    for (i=0; i<size; i++) sndcounts[i] = browners[i+1] - browners[i];
    ierr = MPI_Scatterv(rowlengths,sndcounts,browners,MPIU_INT,locrowlens,(rend-rstart)*bs,MPIU_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscFree(sndcounts);CHKERRQ(ierr);
  } else {
    ierr = MPI_Scatterv(0,0,0,MPIU_INT,locrowlens,(rend-rstart)*bs,MPIU_INT,0,comm);CHKERRQ(ierr);
  }
  
  if (!rank) {   /* procs[0] */
    /* calculate the number of nonzeros on each processor */
    ierr = PetscMalloc(size*sizeof(PetscInt),&procsnz);CHKERRQ(ierr);
    ierr = PetscMemzero(procsnz,size*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<size; i++) {
      for (j=rowners[i]*bs; j< rowners[i+1]*bs; j++) {
        procsnz[i] += rowlengths[j];
      }
    }
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);
    
    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for (i=0; i<size; i++) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    ierr = PetscMalloc(maxnz*sizeof(PetscInt),&cols);CHKERRQ(ierr);

    /* read in my part of the matrix column indices  */
    nz     = procsnz[0];
    ierr   = PetscMalloc(nz*sizeof(PetscInt),&ibuf);CHKERRQ(ierr);
    mycols = ibuf;
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,mycols,nz,PETSC_INT);CHKERRQ(ierr);
    if (size == 1)  for (i=0; i< extra_rows; i++) { mycols[nz+i] = M+i; }

    /* read in every ones (except the last) and ship off */
    for (i=1; i<size-1; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPIU_INT,i,tag,comm);CHKERRQ(ierr);
    }
    /* read in the stuff for the last proc */
    if (size != 1) {
      nz   = procsnz[size-1] - extra_rows;  /* the extra rows are not on the disk */
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      for (i=0; i<extra_rows; i++) cols[nz+i] = M+i;
      ierr = MPI_Send(cols,nz+extra_rows,MPIU_INT,size-1,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
  } else {  /* procs[i], i>0 */
    /* determine buffer space needed for message */
    nz = 0;
    for (i=0; i<m; i++) {
      nz += locrowlens[i];
    }
    ierr   = PetscMalloc(nz*sizeof(PetscInt),&ibuf);CHKERRQ(ierr);
    mycols = ibuf;
    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPIU_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");
  }

  /* loop over local rows, determining number of off diagonal entries */
  ierr     = PetscMalloc2(rend-rstart,PetscInt,&dlens,rend-rstart,PetscInt,&odlens);CHKERRQ(ierr);
  ierr     = PetscMalloc3(Mbs,PetscInt,&mask,Mbs,PetscInt,&masked1,Mbs,PetscInt,&masked2);CHKERRQ(ierr);
  ierr     = PetscMemzero(mask,Mbs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr     = PetscMemzero(masked1,Mbs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr     = PetscMemzero(masked2,Mbs*sizeof(PetscInt));CHKERRQ(ierr);
  rowcount = 0; 
  nzcount  = 0;
  for (i=0; i<mbs; i++) {
    dcount  = 0;
    odcount = 0;
    for (j=0; j<bs; j++) {
      kmax = locrowlens[rowcount];
      for (k=0; k<kmax; k++) {
        tmp = mycols[nzcount++]/bs; /* block col. index */
        if (!mask[tmp]) {
          mask[tmp] = 1;
          if (tmp < rstart || tmp >= rend) masked2[odcount++] = tmp; /* entry in off-diag portion */
          else masked1[dcount++] = tmp; /* entry in diag portion */
        }
      }
      rowcount++;
    }
  
    dlens[i]  = dcount;  /* d_nzz[i] */
    odlens[i] = odcount; /* o_nzz[i] */

    /* zero out the mask elements we set */
    for (j=0; j<dcount; j++) mask[masked1[j]] = 0;
    for (j=0; j<odcount; j++) mask[masked2[j]] = 0; 
  }
    if (!sizesset) {
    ierr = MatSetSizes(newmat,m,m,M+extra_rows,N+extra_rows);CHKERRQ(ierr);
  }
  ierr = MatMPISBAIJSetPreallocation(newmat,bs,0,dlens,0,odlens);CHKERRQ(ierr);
  ierr = MatSetOption(newmat,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
  
  if (!rank) {
    ierr = PetscMalloc(maxnz*sizeof(PetscScalar),&buf);CHKERRQ(ierr);
    /* read in my part of the matrix numerical values  */
    nz = procsnz[0];
    vals = buf;
    mycols = ibuf;
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
    if (size == 1)  for (i=0; i< extra_rows; i++) { vals[nz+i] = 1.0; }

    /* insert into matrix */
    jj      = rstart*bs;
    for (i=0; i<m; i++) {
      ierr = MatSetValues(newmat,1,&jj,locrowlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr); 
      mycols += locrowlens[i];
      vals   += locrowlens[i];
      jj++;
    }

    /* read in other processors (except the last one) and ship out */
    for (i=1; i<size-1; i++) {
      nz   = procsnz[i];
      vals = buf;
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,((PetscObject)newmat)->tag,comm);CHKERRQ(ierr);
    }
    /* the last proc */
    if (size != 1){
      nz   = procsnz[i] - extra_rows;
      vals = buf;
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      for (i=0; i<extra_rows; i++) vals[nz+i] = 1.0;
      ierr = MPI_Send(vals,nz+extra_rows,MPIU_SCALAR,size-1,((PetscObject)newmat)->tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(procsnz);CHKERRQ(ierr);

  } else {
    /* receive numeric values */
    ierr = PetscMalloc(nz*sizeof(PetscScalar),&buf);CHKERRQ(ierr);

    /* receive message of values*/
    vals   = buf;
    mycols = ibuf;
    ierr   = MPI_Recv(vals,nz,MPIU_SCALAR,0,((PetscObject)newmat)->tag,comm,&status);CHKERRQ(ierr);
    ierr   = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart*bs;
    for (i=0; i<m; i++) {
      ierr    = MatSetValues_MPISBAIJ(newmat,1,&jj,locrowlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr);
      mycols += locrowlens[i];
      vals   += locrowlens[i];
      jj++;
    }
  }

  ierr = PetscFree(locrowlens);CHKERRQ(ierr);
  ierr = PetscFree(buf);CHKERRQ(ierr);
  ierr = PetscFree(ibuf);CHKERRQ(ierr);
  ierr = PetscFree2(rowners,browners);CHKERRQ(ierr);
  ierr = PetscFree2(dlens,odlens);CHKERRQ(ierr);
  ierr = PetscFree3(mask,masked1,masked2);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMPISBAIJSetHashTableFactor"
/*XXXXX@
   MatMPISBAIJSetHashTableFactor - Sets the factor required to compute the size of the HashTable.

   Input Parameters:
.  mat  - the matrix
.  fact - factor

   Not Collective on Mat, each process can have a different hash factor

   Level: advanced

  Notes:
   This can also be set by the command line option: -mat_use_hash_table fact

.keywords: matrix, hashtable, factor, HT

.seealso: MatSetOption()
@XXXXX*/


#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMaxAbs_MPISBAIJ"
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
  ierr = MatGetRowMaxAbs(a->A,v,PETSC_NULL);CHKERRQ(ierr); 
  ierr = VecGetArray(v,&va);CHKERRQ(ierr);  

  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)A)->comm,&rank);CHKERRQ(ierr);

  bs   = A->rmap->bs;
  mbs  = a->mbs;
  Mbs  = a->Mbs;
  ba   = b->a;
  bi   = b->i;
  bj   = b->j;

  /* find ownerships */
  rowners_bs = A->rmap->range;

  /* each proc creates an array to be distributed */
  ierr = PetscMalloc(bs*Mbs*sizeof(PetscReal),&work);CHKERRQ(ierr); 
  ierr = PetscMemzero(work,bs*Mbs*sizeof(PetscReal));CHKERRQ(ierr);

  /* row_max for B */
  if (rank != size-1){
    for (i=0; i<mbs; i++) {
      ncols = bi[1] - bi[0]; bi++;
      brow  = bs*i;
      for (j=0; j<ncols; j++){
        bcol = bs*(*bj); 
        for (kcol=0; kcol<bs; kcol++){
          col = bcol + kcol;                 /* local col index */
          col += rowners_bs[rank+1];      /* global col index */
          for (krow=0; krow<bs; krow++){         
            atmp = PetscAbsScalar(*ba); ba++;         
            row = brow + krow;    /* local row index */
            if (PetscRealPart(va[row]) < atmp) va[row] = atmp;
            if (work[col] < atmp) work[col] = atmp;
          }
        }
        bj++;
      }   
    }

    /* send values to its owners */
    for (dest=rank+1; dest<size; dest++){
      svalues = work + rowners_bs[dest];
      count   = rowners_bs[dest+1]-rowners_bs[dest];
      ierr    = MPI_Send(svalues,count,MPIU_REAL,dest,rank,((PetscObject)A)->comm);CHKERRQ(ierr);
    }
  }
  
  /* receive values */
  if (rank){
    rvalues = work;
    count   = rowners_bs[rank+1]-rowners_bs[rank];
    for (source=0; source<rank; source++){     
      ierr = MPI_Recv(rvalues,count,MPIU_REAL,MPI_ANY_SOURCE,MPI_ANY_TAG,((PetscObject)A)->comm,&stat);CHKERRQ(ierr);
      /* process values */     
      for (i=0; i<count; i++){
        if (PetscRealPart(va[i]) < rvalues[i]) va[i] = rvalues[i];
      }   
    } 
  }

  ierr = VecRestoreArray(v,&va);CHKERRQ(ierr); 
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSOR_MPISBAIJ"
PetscErrorCode MatSOR_MPISBAIJ(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_MPISBAIJ      *mat = (Mat_MPISBAIJ*)matin->data;
  PetscErrorCode    ierr;
  PetscInt          mbs=mat->mbs,bs=matin->rmap->bs;
  PetscScalar       *x,*ptr,*from;
  Vec               bb1;
  const PetscScalar *b;
 
  PetscFunctionBegin;
  if (its <= 0 || lits <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (bs > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SSOR for block size > 1 is not yet implemented");

  if (flag == SOR_APPLY_UPPER) {
    ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP){
    if ( flag & SOR_ZERO_INITIAL_GUESS ) {
      ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,lits,xx);CHKERRQ(ierr);
      its--; 
    }

    ierr = VecDuplicate(bb,&bb1);CHKERRQ(ierr);
    while (its--){
 
      /* lower triangular part: slvec0b = - B^T*xx */
      ierr = (*mat->B->ops->multtranspose)(mat->B,xx,mat->slvec0b);CHKERRQ(ierr);
      
      /* copy xx into slvec0a */
      ierr = VecGetArray(mat->slvec0,&ptr);CHKERRQ(ierr);
      ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 
      ierr = PetscMemcpy(ptr,x,bs*mbs*sizeof(MatScalar));CHKERRQ(ierr);
      ierr = VecRestoreArray(mat->slvec0,&ptr);CHKERRQ(ierr);  

      ierr = VecScale(mat->slvec0,-1.0);CHKERRQ(ierr);

      /* copy bb into slvec1a */
      ierr = VecGetArray(mat->slvec1,&ptr);CHKERRQ(ierr);
      ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
      ierr = PetscMemcpy(ptr,b,bs*mbs*sizeof(MatScalar));CHKERRQ(ierr);
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
  } else if ((flag & SOR_LOCAL_FORWARD_SWEEP) && (its == 1) && (flag & SOR_ZERO_INITIAL_GUESS)){
    ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
  } else if ((flag & SOR_LOCAL_BACKWARD_SWEEP) && (its == 1) && (flag & SOR_ZERO_INITIAL_GUESS)){
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
      ierr = MatGetVecs(matin,&mat->diag,PETSC_NULL);CHKERRQ(ierr);
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
	for (i=0; i<n; i++) {
	  sl[i] = b[i] - diag[i]*x[i];
	}
        ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
      } else {
	for (i=0; i<n; i++) {
	  sl[i] = b[i] + scale*diag[i]*x[i];
	}
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
    ierr = PetscMemcpy(from,x,bs*mbs*sizeof(MatScalar));CHKERRQ(ierr);
    ierr = VecRestoreArray(mat->slvec0,&from);CHKERRQ(ierr);  
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
    ierr = VecScatterBegin(mat->sMvctx,mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
    ierr = VecScatterEnd(mat->sMvctx,mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
    ierr = (*mat->B->ops->multadd)(mat->B,mat->slvec1b,mat->slvec1a,mat->slvec1a);CHKERRQ(ierr); 

    /* local sweep */
    ierr = (*mat->A->ops->sor)(mat->A,mat->slvec1a,omega,(MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_FORWARD_SWEEP),fshift,lits,1,xx1);CHKERRQ(ierr);
    ierr = VecAXPY(xx,1.0,xx1);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatSORType is not supported for SBAIJ matrix format");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSOR_MPISBAIJ_2comm"
PetscErrorCode MatSOR_MPISBAIJ_2comm(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_MPISBAIJ   *mat = (Mat_MPISBAIJ*)matin->data;
  PetscErrorCode ierr;
  Vec            lvec1,bb1;
 
  PetscFunctionBegin;
  if (its <= 0 || lits <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (matin->rmap->bs > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SSOR for block size > 1 is not yet implemented");

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP){
    if ( flag & SOR_ZERO_INITIAL_GUESS ) {
      ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,lits,xx);CHKERRQ(ierr);
      its--; 
    }

    ierr = VecDuplicate(mat->lvec,&lvec1);CHKERRQ(ierr);
    ierr = VecDuplicate(bb,&bb1);CHKERRQ(ierr);
    while (its--){
      ierr = VecScatterBegin(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
   
      /* lower diagonal part: bb1 = bb - B^T*xx */
      ierr = (*mat->B->ops->multtranspose)(mat->B,xx,lvec1);CHKERRQ(ierr);
      ierr = VecScale(lvec1,-1.0);CHKERRQ(ierr); 

      ierr = VecScatterEnd(mat->Mvctx,xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr); 
      ierr = VecCopy(bb,bb1);CHKERRQ(ierr);
      ierr = VecScatterBegin(mat->Mvctx,lvec1,bb1,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

      /* upper diagonal part: bb1 = bb1 - B*x */ 
      ierr = VecScale(mat->lvec,-1.0);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb1,bb1);CHKERRQ(ierr);

      ierr = VecScatterEnd(mat->Mvctx,lvec1,bb1,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr); 
  
      /* diagonal sweep */
      ierr = (*mat->A->ops->sor)(mat->A,bb1,omega,SOR_SYMMETRIC_SWEEP,fshift,lits,lits,xx);CHKERRQ(ierr); 
    }
    ierr = VecDestroy(&lvec1);CHKERRQ(ierr);
    ierr = VecDestroy(&bb1);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatSORType is not supported for SBAIJ matrix format");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPISBAIJWithArrays"
/*@
     MatCreateMPISBAIJWithArrays - creates a MPI SBAIJ matrix using arrays that contain in standard
         CSR format the local rows. 

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  bs - the block size, only a block size of 1 is supported
.  m - number of local rows (Cannot be PETSC_DECIDE)
.  n - This value should be the same as the local size used in creating the 
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if N is given) For square matrices n is almost always m.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.   i - row indices
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

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIAIJSetPreallocation(), MatMPIAIJSetPreallocationCSR(),
          MPIAIJ, MatCreateAIJ(), MatCreateMPIAIJWithSplitArrays()
@*/
PetscErrorCode  MatCreateMPISBAIJWithArrays(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt i[],const PetscInt j[],const PetscScalar a[],Mat *mat)
{
  PetscErrorCode ierr;


 PetscFunctionBegin;
  if (i[0]) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  }
  if (m < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"local number of rows (m) cannot be PETSC_DECIDE, or negative");
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATMPISBAIJ);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocationCSR(*mat,bs,i,j,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMPISBAIJSetPreallocationCSR"
/*@C
   MatMPISBAIJSetPreallocationCSR - Allocates memory for a sparse parallel matrix in BAIJ format
   (the default parallel PETSc format).  

   Collective on MPI_Comm

   Input Parameters:
+  A - the matrix 
.  bs - the block size
.  i - the indices into j for the start of each local row (starts with zero)
.  j - the column indices for each local row (starts with zero) these must be sorted for each row
-  v - optional values in the matrix

   Level: developer

.keywords: matrix, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIBAIJSetPreallocation(), MatCreateAIJ(), MPIAIJ
@*/
PetscErrorCode  MatMPISBAIJSetPreallocationCSR(Mat B,PetscInt bs,const PetscInt i[],const PetscInt j[], const PetscScalar v[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(B,"MatMPISBAIJSetPreallocationCSR_C",(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]),(B,bs,i,j,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


