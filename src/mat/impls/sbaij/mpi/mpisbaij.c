#define PETSCMAT_DLL

#include "src/mat/impls/baij/mpi/mpibaij.h"    /*I "petscmat.h" I*/
#include "mpisbaij.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"

EXTERN PetscErrorCode MatSetUpMultiply_MPISBAIJ(Mat); 
EXTERN PetscErrorCode MatSetUpMultiply_MPISBAIJ_2comm(Mat); 
EXTERN PetscErrorCode DisAssemble_MPISBAIJ(Mat);
EXTERN PetscErrorCode MatIncreaseOverlap_MPISBAIJ(Mat,PetscInt,IS[],PetscInt);
EXTERN PetscErrorCode MatGetValues_SeqSBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar []);
EXTERN PetscErrorCode MatGetValues_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar []);
EXTERN PetscErrorCode MatSetValues_SeqSBAIJ(Mat,PetscInt,const PetscInt [],PetscInt,const PetscInt [],const PetscScalar [],InsertMode);
EXTERN PetscErrorCode MatSetValuesBlocked_SeqSBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode MatSetValuesBlocked_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode MatGetRow_SeqSBAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
EXTERN PetscErrorCode MatRestoreRow_SeqSBAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
EXTERN PetscErrorCode MatPrintHelp_SeqSBAIJ(Mat);
EXTERN PetscErrorCode MatZeroRows_SeqSBAIJ(Mat,IS,PetscScalar*);
EXTERN PetscErrorCode MatZeroRows_SeqBAIJ(Mat,IS,PetscScalar *);
EXTERN PetscErrorCode MatGetRowMax_MPISBAIJ(Mat,Vec);
EXTERN PetscErrorCode MatRelax_MPISBAIJ(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);

/*  UGLY, ugly, ugly
   When MatScalar == PetscScalar the function MatSetValuesBlocked_MPIBAIJ_MatScalar() does 
   not exist. Otherwise ..._MatScalar() takes matrix elements in single precision and 
   inserts them into the single precision data structure. The function MatSetValuesBlocked_MPIBAIJ()
   converts the entries into single precision and then calls ..._MatScalar() to put them
   into the single precision data structures.
*/
#if defined(PETSC_USE_MAT_SINGLE)
EXTERN PetscErrorCode MatSetValuesBlocked_SeqSBAIJ_MatScalar(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const MatScalar[],InsertMode);
EXTERN PetscErrorCode MatSetValues_MPISBAIJ_MatScalar(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const MatScalar[],InsertMode);
EXTERN PetscErrorCode MatSetValuesBlocked_MPISBAIJ_MatScalar(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const MatScalar[],InsertMode);
EXTERN PetscErrorCode MatSetValues_MPISBAIJ_HT_MatScalar(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const MatScalar[],InsertMode);
EXTERN PetscErrorCode MatSetValuesBlocked_MPISBAIJ_HT_MatScalar(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const MatScalar[],InsertMode);
#else
#define MatSetValuesBlocked_SeqSBAIJ_MatScalar      MatSetValuesBlocked_SeqSBAIJ
#define MatSetValues_MPISBAIJ_MatScalar             MatSetValues_MPISBAIJ
#define MatSetValuesBlocked_MPISBAIJ_MatScalar      MatSetValuesBlocked_MPISBAIJ
#define MatSetValues_MPISBAIJ_HT_MatScalar          MatSetValues_MPISBAIJ_HT
#define MatSetValuesBlocked_MPISBAIJ_HT_MatScalar   MatSetValuesBlocked_MPISBAIJ_HT
#endif

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues_MPISBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatStoreValues_MPISBAIJ(Mat mat)
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
PetscErrorCode PETSCMAT_DLLEXPORT MatRetrieveValues_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ   *aij = (Mat_MPISBAIJ *)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRetrieveValues(aij->A);CHKERRQ(ierr);
  ierr = MatRetrieveValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#define CHUNKSIZE  10

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
      else if (a->nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) into matrix", row, col); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        PetscInt       new_nz = ai[a->mbs] + CHUNKSIZE,len,*new_i,*new_j; \
        MatScalar *new_a; \
 \
        if (a->nonew == -2) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) in the matrix", row, col); \
 \
        /* malloc new storage space */ \
        ierr = PetscMalloc3(bs2*new_nz,PetscScalar,&new_a,new_nz,PetscInt,&new_j,a->mbs+1,PetscInt,&new_i);CHKERRQ(ierr);\
 \
        /* copy over old data into new slots */ \
        for (ii=0; ii<brow+1; ii++) {new_i[ii] = ai[ii];} \
        for (ii=brow+1; ii<a->mbs+1; ii++) {new_i[ii] = ai[ii]+CHUNKSIZE;} \
        ierr = PetscMemcpy(new_j,aj,(ai[brow]+nrow)*sizeof(PetscInt));CHKERRQ(ierr); \
        len = (new_nz - CHUNKSIZE - ai[brow] - nrow); \
        ierr = PetscMemcpy(new_j+ai[brow]+nrow+CHUNKSIZE,aj+ai[brow]+nrow,len*sizeof(PetscInt));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a,aa,(ai[brow]+nrow)*bs2*sizeof(MatScalar));CHKERRQ(ierr); \
        ierr = PetscMemzero(new_a+bs2*(ai[brow]+nrow),bs2*CHUNKSIZE*sizeof(PetscScalar));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a+bs2*(ai[brow]+nrow+CHUNKSIZE), aa+bs2*(ai[brow]+nrow),bs2*len*sizeof(MatScalar));CHKERRQ(ierr);  \
        /* free up old matrix storage */ \
        ierr = MatSeqXAIJFreeAIJ(a->singlemalloc,a->a,a->j,a->i);CHKERRQ(ierr);\
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j;  \
        a->singlemalloc = PETSC_TRUE; \
 \
        rp   = aj + ai[brow]; ap = aa + bs2*ai[brow]; \
        rmax = aimax[brow] = aimax[brow] + CHUNKSIZE; \
        ierr = PetscLogObjectMemory(A,CHUNKSIZE*(sizeof(PetscInt) + bs2*sizeof(MatScalar)));CHKERRQ(ierr); \
        a->maxnz += bs2*CHUNKSIZE; \
        a->reallocs++; \
        a->nz++; \
      } \
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
#ifndef MatSetValues_SeqBAIJ_B_Private
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
      else if (b->nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) into matrix", row, col); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        PetscInt  new_nz = bi[b->mbs] + CHUNKSIZE,len,*new_i,*new_j; \
        MatScalar *new_a; \
 \
        if (b->nonew == -2) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) in the matrix", row, col); \
 \
        /* malloc new storage space */ \
        ierr = PetscMalloc3(bs2*new_nz,PetscScalar,&new_a,new_nz,PetscInt,&new_j,b->mbs+1,PetscInt,&new_i);CHKERRQ(ierr);\
 \
        /* copy over old data into new slots */ \
        for (ii=0; ii<brow+1; ii++) {new_i[ii] = bi[ii];} \
        for (ii=brow+1; ii<b->mbs+1; ii++) {new_i[ii] = bi[ii]+CHUNKSIZE;} \
        ierr = PetscMemcpy(new_j,bj,(bi[brow]+nrow)*sizeof(PetscInt));CHKERRQ(ierr); \
        len  = (new_nz - CHUNKSIZE - bi[brow] - nrow); \
        ierr = PetscMemcpy(new_j+bi[brow]+nrow+CHUNKSIZE,bj+bi[brow]+nrow,len*sizeof(PetscInt));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a,ba,(bi[brow]+nrow)*bs2*sizeof(MatScalar));CHKERRQ(ierr); \
        ierr = PetscMemzero(new_a+bs2*(bi[brow]+nrow),bs2*CHUNKSIZE*sizeof(MatScalar));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a+bs2*(bi[brow]+nrow+CHUNKSIZE),ba+bs2*(bi[brow]+nrow),bs2*len*sizeof(MatScalar));CHKERRQ(ierr);  \
        /* free up old matrix storage */ \
        ierr = MatSeqXAIJFreeAIJ(b->singlemalloc,b->a,b->j,b->i);CHKERRQ(ierr);\
        ba = b->a = new_a; bi = b->i = new_i; bj = b->j = new_j;  \
        b->singlemalloc = PETSC_TRUE; \
 \
        rp   = bj + bi[brow]; ap = ba + bs2*bi[brow]; \
        rmax = bimax[brow] = bimax[brow] + CHUNKSIZE; \
        ierr = PetscLogObjectMemory(B,CHUNKSIZE*(sizeof(PetscInt) + bs2*sizeof(MatScalar)));CHKERRQ(ierr); \
        b->maxnz += bs2*CHUNKSIZE; \
        b->reallocs++; \
        b->nz++; \
      } \
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
#endif

#if defined(PETSC_USE_MAT_SINGLE)
#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPISBAIJ"
PetscErrorCode MatSetValues_MPISBAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPISBAIJ   *b = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,N = m*n;
  MatScalar      *vsingle;

  PetscFunctionBegin;  
  if (N > b->setvalueslen) {
    if (b->setvaluescopy) {ierr = PetscFree(b->setvaluescopy);CHKERRQ(ierr);}
    ierr = PetscMalloc(N*sizeof(MatScalar),&b->setvaluescopy);CHKERRQ(ierr);
    b->setvalueslen  = N;
  }
  vsingle = b->setvaluescopy;

  for (i=0; i<N; i++) {
    vsingle[i] = v[i];
  }
  ierr = MatSetValues_MPISBAIJ_MatScalar(mat,m,im,n,in,vsingle,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlocked_MPISBAIJ"
PetscErrorCode MatSetValuesBlocked_MPISBAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIBAIJ    *b = (Mat_MPIBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       i,N = m*n*b->bs2;
  MatScalar      *vsingle;

  PetscFunctionBegin;  
  if (N > b->setvalueslen) {
    if (b->setvaluescopy) {ierr = PetscFree(b->setvaluescopy);CHKERRQ(ierr);}
    ierr = PetscMalloc(N*sizeof(MatScalar),&b->setvaluescopy);CHKERRQ(ierr);
    b->setvalueslen  = N;
  }
  vsingle = b->setvaluescopy;
  for (i=0; i<N; i++) {
    vsingle[i] = v[i];
  }
  ierr = MatSetValuesBlocked_MPISBAIJ_MatScalar(mat,m,im,n,in,vsingle,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
#endif

/* Only add/insert a(i,j) with i<=j (blocks). 
   Any a(i,j) with i>j input by user is ingored. 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPIBAIJ_MatScalar"
PetscErrorCode MatSetValues_MPISBAIJ_MatScalar(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const MatScalar v[],InsertMode addv)
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  MatScalar      value;
  PetscTruth     roworiented = baij->roworiented;
  PetscErrorCode ierr;
  PetscInt       i,j,row,col;
  PetscInt       rstart_orig=baij->rstart_bs;
  PetscInt       rend_orig=baij->rend_bs,cstart_orig=baij->cstart_bs;
  PetscInt       cend_orig=baij->cend_bs,bs=mat->bs;

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
  PetscInt      n_loc, *in_loc=0;
  MatScalar     *v_loc=0;

  PetscFunctionBegin;

  if(!baij->donotstash){
    ierr = PetscMalloc(n*sizeof(PetscInt),&in_loc);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(MatScalar),&v_loc);CHKERRQ(ierr);
  }

  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (im[i] >= mat->M) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i],mat->M-1);
#endif
    if (im[i] >= rstart_orig && im[i] < rend_orig) { /* this processor entry */
      row = im[i] - rstart_orig;              /* local row index */
      for (j=0; j<n; j++) {
        if (im[i]/bs > in[j]/bs) continue;    /* ignore lower triangular blocks */
        if (in[j] >= cstart_orig && in[j] < cend_orig){  /* diag entry (A) */
          col = in[j] - cstart_orig;          /* local col index */
          brow = row/bs; bcol = col/bs;
          if (brow > bcol) continue;  /* ignore lower triangular blocks of A */           
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqSBAIJ_A_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqBAIJ(baij->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        } else if (in[j] < 0) continue;
#if defined(PETSC_USE_DEBUG)
        else if (in[j] >= mat->N) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[j],mat->N-1);}
#endif
        else {  /* off-diag entry (B) */
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = CreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
            }
#if defined (PETSC_USE_CTABLE)
            ierr = PetscTableFind(baij->colmap,in[j]/bs + 1,&col);CHKERRQ(ierr);
            col  = col - 1;
#else
            col = baij->colmap[in[j]/bs] - 1;
#endif
            if (col < 0 && !((Mat_SeqSBAIJ*)(baij->A->data))->nonew) {
              ierr = DisAssemble_MPISBAIJ(mat);CHKERRQ(ierr); 
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
      if (!baij->donotstash) {
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
        ierr = MatStashValuesRow_Private(&mat->stash,im[i],n_loc,in_loc,v_loc);CHKERRQ(ierr);        
      }
    }
  }

  if(!baij->donotstash){
    ierr = PetscFree(in_loc);CHKERRQ(ierr);
    ierr = PetscFree(v_loc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlocked_MPISBAIJ_MatScalar"
PetscErrorCode MatSetValuesBlocked_MPISBAIJ_MatScalar(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const MatScalar v[],InsertMode addv)
{
  Mat_MPISBAIJ    *baij = (Mat_MPISBAIJ*)mat->data;
  const MatScalar *value;
  MatScalar       *barray=baij->barray;
  PetscTruth      roworiented = baij->roworiented;
  PetscErrorCode  ierr;
  PetscInt        i,j,ii,jj,row,col,rstart=baij->rstart;
  PetscInt        rend=baij->rend,cstart=baij->cstart,stepval;
  PetscInt        cend=baij->cend,bs=mat->bs,bs2=baij->bs2;

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
    if (im[i] >= baij->Mbs) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large, row %D max %D",im[i],baij->Mbs-1);
#endif
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for (j=0; j<n; j++) {
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
        else if (in[j] >= baij->Nbs) {SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large, col %D max %D",in[j],baij->Nbs-1);}
#endif
        else {
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = CreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
            }

#if defined(PETSC_USE_DEBUG)
#if defined (PETSC_USE_CTABLE)
            { PetscInt data;
              ierr = PetscTableFind(baij->colmap,in[j]+1,&data);CHKERRQ(ierr);
              if ((data - 1) % bs) SETERRQ(PETSC_ERR_PLIB,"Incorrect colmap");
            }
#else
            if ((baij->colmap[in[j]] - 1) % bs) SETERRQ(PETSC_ERR_PLIB,"Incorrect colmap");
#endif
#endif
#if defined (PETSC_USE_CTABLE)
	    ierr = PetscTableFind(baij->colmap,in[j]+1,&col);CHKERRQ(ierr);
            col  = (col - 1)/bs;
#else
            col = (baij->colmap[in[j]] - 1)/bs;
#endif
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
              ierr = DisAssemble_MPISBAIJ(mat);CHKERRQ(ierr); 
              col =  in[j];              
            }
          }
          else col = in[j];
          ierr = MatSetValuesBlocked_SeqBAIJ(baij->B,1,&row,1,&col,barray,addv);CHKERRQ(ierr);
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
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_MPISBAIJ"
PetscErrorCode MatGetValues_MPISBAIJ(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPISBAIJ   *baij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       bs=mat->bs,i,j,bsrstart = baij->rstart*bs,bsrend = baij->rend*bs;
  PetscInt       bscstart = baij->cstart*bs,bscend = baij->cend*bs,row,col,data;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %D",idxm[i]);
    if (idxm[i] >= mat->M) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",idxm[i],mat->M-1);
    if (idxm[i] >= bsrstart && idxm[i] < bsrend) {
      row = idxm[i] - bsrstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative column %D",idxn[j]);
        if (idxn[j] >= mat->N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",idxn[j],mat->N-1);
        if (idxn[j] >= bscstart && idxn[j] < bscend){
          col = idxn[j] - bscstart;
          ierr = MatGetValues_SeqSBAIJ(baij->A,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
        } else {
          if (!baij->colmap) {
            ierr = CreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
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
      SETERRQ(PETSC_ERR_SUP,"Only local values currently supported");
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
      ierr = MPI_Allreduce(lnorm2,&sum,2,MPIU_REAL,MPI_SUM,mat->comm);CHKERRQ(ierr);
      *norm = sqrt(sum[0] + 2*sum[1]);
      ierr = PetscFree(lnorm2);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY || type == NORM_1) { /* max row/column sum */
      Mat_SeqSBAIJ *amat=(Mat_SeqSBAIJ*)baij->A->data;
      Mat_SeqBAIJ  *bmat=(Mat_SeqBAIJ*)baij->B->data;
      PetscReal    *rsum,*rsum2,vabs; 
      PetscInt     *jj,*garray=baij->garray,rstart=baij->rstart,nz;
      PetscInt     brow,bcol,col,bs=baij->A->bs,row,grow,gcol,mbs=amat->mbs;
      MatScalar    *v;

      ierr  = PetscMalloc((2*mat->N+1)*sizeof(PetscReal),&rsum);CHKERRQ(ierr);
      rsum2 = rsum + mat->N;
      ierr  = PetscMemzero(rsum,mat->N*sizeof(PetscReal));CHKERRQ(ierr);
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
      ierr = MPI_Allreduce(rsum,rsum2,mat->N,MPIU_REAL,MPI_SUM,mat->comm);CHKERRQ(ierr);
      *norm = 0.0;
      for (col=0; col<mat->N; col++) {
        if (rsum2[col] > *norm) *norm = rsum2[col];
      }
      ierr = PetscFree(rsum);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"No support for this norm yet");
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
  if (baij->donotstash) {
    PetscFunctionReturn(0);
  }

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,mat->comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Some processors inserted others added");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  ierr = MatStashScatterBegin_Private(&mat->stash,baij->rowners_bs);CHKERRQ(ierr);
  ierr = MatStashScatterBegin_Private(&mat->bstash,baij->rowners);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscLogInfo((0,"MatAssemblyBegin_MPISBAIJ:Stash has %D entries,uses %D mallocs.\n",nstash,reallocs));CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscLogInfo((0,"MatAssemblyBegin_MPISBAIJ:Block-Stash has %D entries, uses %D mallocs.\n",nstash,reallocs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPISBAIJ"
PetscErrorCode MatAssemblyEnd_MPISBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPISBAIJ   *baij=(Mat_MPISBAIJ*)mat->data;
  Mat_SeqSBAIJ   *a=(Mat_SeqSBAIJ*)baij->A->data;
  Mat_SeqBAIJ    *b=(Mat_SeqBAIJ*)baij->B->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart,ncols,flg,bs2=baij->bs2;
  PetscInt       *row,*col,other_disassembled;
  PetscMPIInt    n;
  PetscTruth     r1,r2,r3;
  MatScalar      *val;
  InsertMode     addv = mat->insertmode;

  PetscFunctionBegin;

  if (!baij->donotstash) { 
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        /* Now assemble all these values with a single function call */
        ierr = MatSetValues_MPISBAIJ_MatScalar(mat,1,row+i,ncols,col+i,val+i,addv);CHKERRQ(ierr);
        i = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);
    /* Now process the block-stash. Since the values are stashed column-oriented,
       set the roworiented flag to column oriented, and after MatSetValues() 
       restore the original flags */
    r1 = baij->roworiented;
    r2 = a->roworiented;
    r3 = b->roworiented;
    baij->roworiented = PETSC_FALSE;
    a->roworiented    = PETSC_FALSE;
    b->roworiented    = PETSC_FALSE;
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->bstash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;
      
      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        ierr = MatSetValuesBlocked_MPISBAIJ_MatScalar(mat,1,row+i,ncols,col+i,val+i*bs2,addv);CHKERRQ(ierr);
        i = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->bstash);CHKERRQ(ierr);
    baij->roworiented = r1;
    a->roworiented    = r2;
    b->roworiented    = r3;
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
    ierr = MPI_Allreduce(&mat->was_assembled,&other_disassembled,1,MPI_INT,MPI_PROD,mat->comm);CHKERRQ(ierr);
    if (mat->was_assembled && !other_disassembled) {
      ierr = DisAssemble_MPISBAIJ(mat);CHKERRQ(ierr);
    }
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPISBAIJ(mat);CHKERRQ(ierr); /* setup Mvctx and sMvctx */
  }
  b->compressedrow.use = PETSC_TRUE; 
  ierr = MatAssemblyBegin(baij->B,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->B,mode);CHKERRQ(ierr);
  
  if (baij->rowvalues) {
    ierr = PetscFree(baij->rowvalues);CHKERRQ(ierr);
    baij->rowvalues = 0;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPISBAIJ_ASCIIorDraworSocket"
static PetscErrorCode MatView_MPISBAIJ_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPISBAIJ      *baij = (Mat_MPISBAIJ*)mat->data;
  PetscErrorCode    ierr;
  PetscInt          bs = mat->bs;
  PetscMPIInt       size = baij->size,rank = baij->rank;
  PetscTruth        iascii,isdraw;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) { 
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo info;
      ierr = MPI_Comm_rank(mat->comm,&rank);CHKERRQ(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %D nz %D nz alloced %D bs %D mem %D\n",
              rank,mat->m,(PetscInt)info.nz_used*bs,(PetscInt)info.nz_allocated*bs,
              mat->bs,(PetscInt)info.memory);CHKERRQ(ierr);      
      ierr = MatGetInfo(baij->A,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] on-diagonal part: nz %D \n",rank,(PetscInt)info.nz_used*bs);CHKERRQ(ierr);
      ierr = MatGetInfo(baij->B,MAT_LOCAL,&info);CHKERRQ(ierr); 
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] off-diagonal part: nz %D \n",rank,(PetscInt)info.nz_used*bs);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = VecScatterView(baij->Mvctx,viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0); 
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"  block size is %D\n",bs);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }

  if (isdraw) {
    PetscDraw       draw;
    PetscTruth isnull;
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  }

  if (size == 1) {
    ierr = PetscObjectSetName((PetscObject)baij->A,mat->name);CHKERRQ(ierr);
    ierr = MatView(baij->A,viewer);CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    Mat_SeqSBAIJ *Aloc;
    Mat_SeqBAIJ *Bloc;
    PetscInt         M = mat->M,N = mat->N,*ai,*aj,col,i,j,k,*rvals,mbs = baij->mbs;
    MatScalar   *a;

    /* Should this be the same type as mat? */
    if (!rank) {
      ierr = MatCreate(mat->comm,M,N,M,N,&A);CHKERRQ(ierr);
    } else {
      ierr = MatCreate(mat->comm,0,0,M,N,&A);CHKERRQ(ierr);
    }
    ierr = MatSetType(A,MATMPISBAIJ);CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(A,mat->bs,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(mat,A);CHKERRQ(ierr);

    /* copy over the A part */
    Aloc  = (Mat_SeqSBAIJ*)baij->A->data;
    ai    = Aloc->i; aj = Aloc->j; a = Aloc->a;
    ierr  = PetscMalloc(bs*sizeof(PetscInt),&rvals);CHKERRQ(ierr);

    for (i=0; i<mbs; i++) {
      rvals[0] = bs*(baij->rstart + i);
      for (j=1; j<bs; j++) { rvals[j] = rvals[j-1] + 1; }
      for (j=ai[i]; j<ai[i+1]; j++) {
        col = (baij->cstart+aj[j])*bs;
        for (k=0; k<bs; k++) {
          ierr = MatSetValues_MPISBAIJ_MatScalar(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
          col++; a += bs;
        }
      }
    } 
    /* copy over the B part */
    Bloc = (Mat_SeqBAIJ*)baij->B->data;
    ai = Bloc->i; aj = Bloc->j; a = Bloc->a;
    for (i=0; i<mbs; i++) {
      rvals[0] = bs*(baij->rstart + i);
      for (j=1; j<bs; j++) { rvals[j] = rvals[j-1] + 1; }
      for (j=ai[i]; j<ai[i+1]; j++) {
        col = baij->garray[aj[j]]*bs;
        for (k=0; k<bs; k++) {
          ierr = MatSetValues_MPISBAIJ_MatScalar(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
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
      ierr = PetscObjectSetName((PetscObject)((Mat_MPISBAIJ*)(A->data))->A,mat->name);CHKERRQ(ierr);
      ierr = MatView(((Mat_MPISBAIJ*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPISBAIJ"
PetscErrorCode MatView_MPISBAIJ(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscTruth     iascii,isdraw,issocket,isbinary;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (iascii || isdraw || issocket || isbinary) { 
    ierr = MatView_MPISBAIJ_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by MPISBAIJ matrices",((PetscObject)viewer)->type_name);
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
  PetscLogObjectState((PetscObject)mat,"Rows=%D,Cols=%D",mat->M,mat->N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = MatStashDestroy_Private(&mat->bstash);CHKERRQ(ierr);
  ierr = PetscFree(baij->rowners);CHKERRQ(ierr);
  ierr = MatDestroy(baij->A);CHKERRQ(ierr);
  ierr = MatDestroy(baij->B);CHKERRQ(ierr);
#if defined (PETSC_USE_CTABLE)
  if (baij->colmap) {ierr = PetscTableDelete(baij->colmap);CHKERRQ(ierr);}
#else
  if (baij->colmap) {ierr = PetscFree(baij->colmap);CHKERRQ(ierr);}
#endif
  if (baij->garray) {ierr = PetscFree(baij->garray);CHKERRQ(ierr);}
  if (baij->lvec)   {ierr = VecDestroy(baij->lvec);CHKERRQ(ierr);}
  if (baij->Mvctx)  {ierr = VecScatterDestroy(baij->Mvctx);CHKERRQ(ierr);}
  if (baij->slvec0) {
    ierr = VecDestroy(baij->slvec0);CHKERRQ(ierr);
    ierr = VecDestroy(baij->slvec0b);CHKERRQ(ierr); 
  }
  if (baij->slvec1) {
    ierr = VecDestroy(baij->slvec1);CHKERRQ(ierr);
    ierr = VecDestroy(baij->slvec1a);CHKERRQ(ierr);
    ierr = VecDestroy(baij->slvec1b);CHKERRQ(ierr); 
  }
  if (baij->sMvctx)  {ierr = VecScatterDestroy(baij->sMvctx);CHKERRQ(ierr);} 
  if (baij->rowvalues) {ierr = PetscFree(baij->rowvalues);CHKERRQ(ierr);}
  if (baij->barray) {ierr = PetscFree(baij->barray);CHKERRQ(ierr);}
  if (baij->hd) {ierr = PetscFree(baij->hd);CHKERRQ(ierr);}
#if defined(PETSC_USE_MAT_SINGLE)
  if (baij->setvaluescopy) {ierr = PetscFree(baij->setvaluescopy);CHKERRQ(ierr);}
#endif
  ierr = PetscFree(baij);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatStoreValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatRetrieveValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatGetDiagonalBlock_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPISBAIJSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_MPISBAIJ"
PetscErrorCode MatMult_MPISBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt,mbs=a->mbs,bs=A->bs;
  PetscScalar    *x,*from,zero=0.0;
 
  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->n) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"Incompatible partition of A and xx");
  }
  ierr = VecGetLocalSize(yy,&nt);CHKERRQ(ierr);
  if (nt != A->m) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"Incompatible parition of A and yy");
  }

  /* diagonal part */
  ierr = (*a->A->ops->mult)(a->A,xx,a->slvec1a);CHKERRQ(ierr); 
  ierr = VecSet(&zero,a->slvec1b);CHKERRQ(ierr); 

  /* subdiagonal part */  
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->slvec0b);CHKERRQ(ierr);

  /* copy x into the vec slvec0 */
  ierr = VecGetArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(from,x,bs*mbs*sizeof(MatScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&from);CHKERRQ(ierr);  
  
  ierr = VecScatterBegin(a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD,a->sMvctx);CHKERRQ(ierr); 
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecScatterEnd(a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD,a->sMvctx);CHKERRQ(ierr); 
  
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
  if (nt != A->n) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"Incompatible partition of A and xx");
  }
  ierr = VecGetLocalSize(yy,&nt);CHKERRQ(ierr);
  if (nt != A->m) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"Incompatible parition of A and yy");
  }

  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr); 
  /* do diagonal part */
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  /* do supperdiagonal part */
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  /* do subdiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,yy,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_MPISBAIJ"
PetscErrorCode MatMultAdd_MPISBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       mbs=a->mbs,bs=A->bs;
  PetscScalar    *x,*from,zero=0.0;
 
  PetscFunctionBegin;
  /*
  PetscSynchronizedPrintf(A->comm," MatMultAdd is called ...\n");
  PetscSynchronizedFlush(A->comm);
  */
  /* diagonal part */
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,a->slvec1a);CHKERRQ(ierr); 
  ierr = VecSet(&zero,a->slvec1b);CHKERRQ(ierr); 

  /* subdiagonal part */  
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->slvec0b);CHKERRQ(ierr);

  /* copy x into the vec slvec0 */
  ierr = VecGetArray(a->slvec0,&from);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(from,x,bs*mbs*sizeof(MatScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec0,&from);CHKERRQ(ierr);  
  
  ierr = VecScatterBegin(a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD,a->sMvctx);CHKERRQ(ierr); 
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecScatterEnd(a->slvec0,a->slvec1,ADD_VALUES,SCATTER_FORWARD,a->sMvctx);CHKERRQ(ierr); 
  
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
  ierr = VecScatterBegin(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr); 
  /* do diagonal part */
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  /* do supperdiagonal part */
  ierr = VecScatterEnd(xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD,a->Mvctx);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);

  /* do subdiagonal part */    
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->lvec,zz,ADD_VALUES,SCATTER_REVERSE,a->Mvctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_MPISBAIJ"
PetscErrorCode MatMultTranspose_MPISBAIJ(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(A,xx,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_MPISBAIJ"
PetscErrorCode MatMultTransposeAdd_MPISBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd(A,xx,yy,zz);CHKERRQ(ierr);
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
  /* if (a->M != a->N) SETERRQ(PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block"); */
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_MPISBAIJ"
PetscErrorCode MatScale_MPISBAIJ(const PetscScalar *aa,Mat A)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(aa,a->A);CHKERRQ(ierr);
  ierr = MatScale(aa,a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_MPISBAIJ"
PetscErrorCode MatGetRow_MPISBAIJ(Mat matin,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"MatGetRow is not supported for SBAIJ matrix format");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_MPISBAIJ"
PetscErrorCode MatRestoreRow_MPISBAIJ(Mat mat,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;

  PetscFunctionBegin;
  if (!baij->getrowactive) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"MatGetRow() must be called first");
  }
  baij->getrowactive = PETSC_FALSE;
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
  info->block_size     = (PetscReal)matin->bs;
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
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPI_MAX,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPI_SUM,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG,"Unknown MatInfoType argument %d",(int)flag);
  }
  info->rows_global       = (PetscReal)A->M;
  info->columns_global    = (PetscReal)A->N;
  info->rows_local        = (PetscReal)A->m;
  info->columns_local     = (PetscReal)A->N;
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPISBAIJ"
PetscErrorCode MatSetOption_MPISBAIJ(Mat A,MatOption op)
{
  Mat_MPISBAIJ   *a = (Mat_MPISBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NO_NEW_NONZERO_LOCATIONS:
  case MAT_YES_NEW_NONZERO_LOCATIONS:
  case MAT_COLUMNS_UNSORTED:
  case MAT_COLUMNS_SORTED:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_KEEP_ZEROED_ROWS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
    break;
  case MAT_ROW_ORIENTED:
    a->roworiented = PETSC_TRUE;
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
    break;
  case MAT_ROWS_SORTED: 
  case MAT_ROWS_UNSORTED:
  case MAT_YES_NEW_DIAGONALS:
    ierr = PetscLogInfo((A,"Info:MatSetOption_MPIBAIJ:Option ignored\n"));CHKERRQ(ierr);
    break;
  case MAT_COLUMN_ORIENTED:
    a->roworiented = PETSC_FALSE;
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = PETSC_TRUE;
    break;
  case MAT_NO_NEW_DIAGONALS:
    SETERRQ(PETSC_ERR_SUP,"MAT_NO_NEW_DIAGONALS");
  case MAT_USE_HASH_TABLE:
    a->ht_flag = PETSC_TRUE;
    break;
  case MAT_NOT_SYMMETRIC:
  case MAT_NOT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
    SETERRQ(PETSC_ERR_SUP,"Matrix must be symmetric");
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_NOT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
  case MAT_NOT_SYMMETRY_ETERNAL:
    break;
  default:
    SETERRQ(PETSC_ERR_SUP,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_MPISBAIJ"
PetscErrorCode MatTranspose_MPISBAIJ(Mat A,Mat *B)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatDuplicate(A,MAT_COPY_VALUES,B);CHKERRQ(ierr);
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
  PetscTruth     flg;

  PetscFunctionBegin;
  if (ll != rr){
    ierr = VecEqual(ll,rr,&flg);CHKERRQ(ierr);
    if (!flg) 
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"For symmetric format, left and right scaling vectors must be same\n");
  }
  if (!ll) PetscFunctionReturn(0);

  ierr = MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
  if (m != n) SETERRQ2(PETSC_ERR_ARG_SIZ,"For symmetric format, local size %d %d must be same",m,n);
  
  ierr = VecGetLocalSize(rr,&nv);CHKERRQ(ierr);
  if (nv!=n) SETERRQ(PETSC_ERR_ARG_SIZ,"Left and right vector non-conforming local size");

  ierr = VecScatterBegin(rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD,baij->Mvctx);CHKERRQ(ierr);
   
  /* left diagonalscale the off-diagonal part */
  ierr = (*b->ops->diagonalscale)(b,ll,PETSC_NULL);CHKERRQ(ierr);
    
  /* scale the diagonal part */
  ierr = (*a->ops->diagonalscale)(a,ll,rr);CHKERRQ(ierr);

  /* right diagonalscale the off-diagonal part */
  ierr = VecScatterEnd(rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD,baij->Mvctx);CHKERRQ(ierr);
  ierr = (*b->ops->diagonalscale)(b,PETSC_NULL,baij->lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPrintHelp_MPISBAIJ"
PetscErrorCode MatPrintHelp_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ      *a = (Mat_MPISBAIJ*)A->data;
  MPI_Comm          comm = A->comm;
  static PetscTruth called = PETSC_FALSE; 
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!a->rank) {
    ierr = MatPrintHelp_SeqSBAIJ(a->A);CHKERRQ(ierr);
  }
  if (called) {PetscFunctionReturn(0);} else called = PETSC_TRUE;
  ierr = (*PetscHelpPrintf)(comm," Options for MATMPISBAIJ matrix format (the defaults):\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_use_hash_table <factor>: Use hashtable for efficient matrix assembly\n");CHKERRQ(ierr);
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
PetscErrorCode MatEqual_MPISBAIJ(Mat A,Mat B,PetscTruth *flag)
{
  Mat_MPISBAIJ   *matB = (Mat_MPISBAIJ*)B->data,*matA = (Mat_MPISBAIJ*)A->data;
  Mat            a,b,c,d;
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  a = matA->A; b = matA->B;
  c = matB->A; d = matB->B;

  ierr = MatEqual(a,c,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatEqual(b,d,&flg);CHKERRQ(ierr);
  }
  ierr = MPI_Allreduce(&flg,flag,1,MPI_INT,MPI_LAND,A->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_MPISBAIJ"
PetscErrorCode MatSetUpPreallocation_MPISBAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMPISBAIJSetPreallocation(A,1,PETSC_DEFAULT,0,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_MPISBAIJ"
PetscErrorCode MatGetSubMatrices_MPISBAIJ(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscTruth     flg;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    ierr = ISEqual(irow[i],icol[i],&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(PETSC_ERR_SUP,"Can only get symmetric submatrix for MPISBAIJ matrices");
    }
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
       MatMultTranspose_MPISBAIJ,
       MatMultTransposeAdd_MPISBAIJ,
       0,
       0,
       0,
/*10*/ 0,
       0,
       0,
       MatRelax_MPISBAIJ,
       MatTranspose_MPISBAIJ,
/*15*/ MatGetInfo_MPISBAIJ,
       MatEqual_MPISBAIJ,
       MatGetDiagonal_MPISBAIJ,
       MatDiagonalScale_MPISBAIJ,
       MatNorm_MPISBAIJ,
/*20*/ MatAssemblyBegin_MPISBAIJ,
       MatAssemblyEnd_MPISBAIJ,
       0,
       MatSetOption_MPISBAIJ,
       MatZeroEntries_MPISBAIJ,
/*25*/ 0,
       0,
       0,
       0,
       0,
/*30*/ MatSetUpPreallocation_MPISBAIJ,
       0,
       0,
       0,
       0,
/*35*/ MatDuplicate_MPISBAIJ,
       0,
       0,
       0,
       0,
/*40*/ 0,
       MatGetSubMatrices_MPISBAIJ,
       MatIncreaseOverlap_MPISBAIJ,
       MatGetValues_MPISBAIJ,
       0,
/*45*/ MatPrintHelp_MPISBAIJ,
       MatScale_MPISBAIJ,
       0,
       0,
       0,
/*50*/ 0,
       0,
       0,
       0,
       0,
/*55*/ 0,
       0,
       MatSetUnfactored_MPISBAIJ,
       0,
       MatSetValuesBlocked_MPISBAIJ,
/*60*/ 0,
       0,
       0,
       MatGetPetscMaps_Petsc,
       0,
/*65*/ 0,
       0,
       0,
       0,
       0,
/*70*/ MatGetRowMax_MPISBAIJ,
       0,
       0,
       0,
       0,
/*75*/ 0,
       0,
       0,
       0,
       0,
/*80*/ 0,
       0,
       0,
       0,
       MatLoad_MPISBAIJ,
/*85*/ 0,
       0,
       0,
       0,
       0,
/*90*/ 0,
       0,
       0,
       0,
       0,
/*95*/ 0,
       0,
       0,
       0};


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonalBlock_MPISBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatGetDiagonalBlock_MPISBAIJ(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
{
  PetscFunctionBegin;
  *a      = ((Mat_MPISBAIJ *)A->data)->A;
  *iscopy = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatMPISBAIJSetPreallocation_MPISBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatMPISBAIJSetPreallocation_MPISBAIJ(Mat B,PetscInt bs,PetscInt d_nz,PetscInt *d_nnz,PetscInt o_nz,PetscInt *o_nnz)
{
  Mat_MPISBAIJ   *b;
  PetscErrorCode ierr;
  PetscInt       i,mbs,Mbs;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(B->prefix,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);

  if (bs < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive");
  if (d_nz == PETSC_DECIDE || d_nz == PETSC_DEFAULT) d_nz = 3;
  if (o_nz == PETSC_DECIDE || o_nz == PETSC_DEFAULT) o_nz = 1;
  if (d_nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"d_nz cannot be less than 0: value %D",d_nz);
  if (o_nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"o_nz cannot be less than 0: value %D",o_nz);
  if (d_nnz) {
    for (i=0; i<B->m/bs; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than -1: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->m/bs; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than -1: local row %D value %D",i,o_nnz[i]);
    }
  }
  B->preallocated = PETSC_TRUE;
  ierr = PetscSplitOwnershipBlock(B->comm,bs,&B->m,&B->M);CHKERRQ(ierr);
  ierr = PetscSplitOwnershipBlock(B->comm,bs,&B->n,&B->N);CHKERRQ(ierr);
  ierr = PetscMapCreateMPI(B->comm,B->m,B->M,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapCreateMPI(B->comm,B->m,B->M,&B->cmap);CHKERRQ(ierr);

  b   = (Mat_MPISBAIJ*)B->data;
  mbs = B->m/bs;
  Mbs = B->M/bs;
  if (mbs*bs != B->m) {
    SETERRQ2(PETSC_ERR_ARG_SIZ,"No of local rows %D must be divisible by blocksize %D",B->m,bs);
  }

  B->bs  = bs;
  b->bs2 = bs*bs;
  b->mbs = mbs;
  b->nbs = mbs; 
  b->Mbs = Mbs;
  b->Nbs = Mbs; 

  ierr = MPI_Allgather(&b->mbs,1,MPIU_INT,b->rowners+1,1,MPIU_INT,B->comm);CHKERRQ(ierr);
  b->rowners[0]    = 0;
  for (i=2; i<=b->size; i++) {
    b->rowners[i] += b->rowners[i-1];
  }
  b->rstart    = b->rowners[b->rank]; 
  b->rend      = b->rowners[b->rank+1]; 
  b->cstart    = b->rstart; 
  b->cend      = b->rend;   
  for (i=0; i<=b->size; i++) {
    b->rowners_bs[i] = b->rowners[i]*bs;
  }
  b->rstart_bs = b-> rstart*bs;
  b->rend_bs   = b->rend*bs;
  
  b->cstart_bs = b->cstart*bs;
  b->cend_bs   = b->cend*bs;
  
  ierr = MatCreate(PETSC_COMM_SELF,B->m,B->m,B->m,B->m,&b->A);CHKERRQ(ierr);
  ierr = MatSetType(b->A,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(b->A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(B,b->A);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,B->m,B->M,B->m,B->M,&b->B);CHKERRQ(ierr);
  ierr = MatSetType(b->B,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(b->B,bs,o_nz,o_nnz);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(B,b->B);CHKERRQ(ierr);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(B->comm,bs,&B->bstash);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATMPISBAIJ - MATMPISBAIJ = "mpisbaij" - A matrix type to be used for distributed symmetric sparse block matrices, 
   based on block compressed sparse row format.  Only the upper triangular portion of the matrix is stored.

   Options Database Keys:
. -mat_type mpisbaij - sets the matrix type to "mpisbaij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPISBAIJ
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPISBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_MPISBAIJ(Mat B)
{
  Mat_MPISBAIJ   *b;
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;

  ierr    = PetscNew(Mat_MPISBAIJ,&b);CHKERRQ(ierr);
  B->data = (void*)b;
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  B->ops->destroy    = MatDestroy_MPISBAIJ;
  B->ops->view       = MatView_MPISBAIJ;
  B->mapping    = 0;
  B->factor     = 0;
  B->assembled  = PETSC_FALSE;

  B->insertmode = NOT_SET_VALUES;
  ierr = MPI_Comm_rank(B->comm,&b->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(B->comm,&b->size);CHKERRQ(ierr);

  /* build local table of row and column ownerships */
  ierr          = PetscMalloc(3*(b->size+2)*sizeof(PetscInt),&b->rowners);CHKERRQ(ierr);
  b->cowners    = b->rowners + b->size + 2;
  b->rowners_bs = b->cowners + b->size + 2;
  ierr = PetscLogObjectMemory(B,3*(b->size+2)*sizeof(PetscInt)+sizeof(struct _p_Mat)+sizeof(Mat_MPISBAIJ));CHKERRQ(ierr);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(B->comm,1,&B->stash);CHKERRQ(ierr);
  b->donotstash  = PETSC_FALSE;
  b->colmap      = PETSC_NULL;
  b->garray      = PETSC_NULL;
  b->roworiented = PETSC_TRUE;

#if defined(PETSC_USE_MAT_SINGLE)
  /* stuff for MatSetValues_XXX in single precision */
  b->setvalueslen     = 0;
  b->setvaluescopy    = PETSC_NULL;
#endif

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

  ierr = PetscOptionsHasName(B->prefix,"-mat_use_hash_table",&flg);CHKERRQ(ierr);
  if (flg) { 
    PetscReal fact = 1.39;
    ierr = MatSetOption(B,MAT_USE_HASH_TABLE);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(B->prefix,"-mat_use_hash_table",&fact,PETSC_NULL);CHKERRQ(ierr);
    if (fact <= 1.0) fact = 1.39;
    ierr = MatMPIBAIJSetHashTableFactor(B,fact);CHKERRQ(ierr);
    ierr = PetscLogInfo((0,"MatCreateMPISBAIJ:Hash table Factor used %5.2f\n",fact));CHKERRQ(ierr);
  }
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
  B->symmetric                  = PETSC_TRUE;
  B->structurally_symmetric     = PETSC_TRUE;
  B->symmetric_set              = PETSC_TRUE;
  B->structurally_symmetric_set = PETSC_TRUE;
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

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SBAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SBAIJ(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSBAIJ);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQSBAIJ);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPISBAIJ);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
           (possibly different for each block row) or PETSC_NULL.  You must leave room 
           for the diagonal entry even if it is zero.
.  o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or PETSC_NULL.


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

   Consider a processor that owns rows 3, 4 and 5 of a parallel matrix. In
   the figure below we depict these three local rows and all columns (0-11).

.vb
           0 1 2 3 4 5 6 7 8 9 10 11
          -------------------
   row 3  |  o o o d d d o o o o o o
   row 4  |  o o o d d d o o o o o o
   row 5  |  o o o d d d o o o o o o
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

.seealso: MatCreate(), MatCreateSeqSBAIJ(), MatSetValues(), MatCreateMPIBAIJ()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatMPISBAIJSetPreallocation(Mat B,PetscInt bs,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatMPISBAIJSetPreallocation_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPISBAIJ"
/*@C
   MatCreateMPISBAIJ - Creates a sparse parallel matrix in symmetric block AIJ format
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
           You must leave room for the diagonal entry even if it is zero.
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

   Notes:
   The number of rows and columns must be divisible by blocksize.

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
   row 3  |  o o o d d d o o o o o o
   row 4  |  o o o d d d o o o o o o
   row 5  |  o o o d d d o o o o o o
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

.seealso: MatCreate(), MatCreateSeqSBAIJ(), MatSetValues(), MatCreateMPIBAIJ()
@*/

PetscErrorCode PETSCMAT_DLLEXPORT MatCreateMPISBAIJ(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,m,n,M,N,A);CHKERRQ(ierr);
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
  PetscInt       len=0,nt,bs=matin->bs,mbs=oldmat->mbs;
  PetscScalar    *array;

  PetscFunctionBegin;
  *newmat       = 0;
  ierr = MatCreate(matin->comm,matin->m,matin->n,matin->M,matin->N,&mat);CHKERRQ(ierr);
  ierr = MatSetType(mat,matin->type_name);CHKERRQ(ierr);
  ierr = PetscMemcpy(mat->ops,matin->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  
  mat->factor       = matin->factor; 
  mat->preallocated = PETSC_TRUE;
  mat->assembled    = PETSC_TRUE;
  mat->insertmode   = NOT_SET_VALUES;

  a = (Mat_MPISBAIJ*)mat->data;
  mat->bs  = matin->bs;
  a->bs2   = oldmat->bs2;
  a->mbs   = oldmat->mbs;
  a->nbs   = oldmat->nbs;
  a->Mbs   = oldmat->Mbs;
  a->Nbs   = oldmat->Nbs;
  
  a->rstart       = oldmat->rstart;
  a->rend         = oldmat->rend;
  a->cstart       = oldmat->cstart;
  a->cend         = oldmat->cend;
  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  a->donotstash   = oldmat->donotstash;
  a->roworiented  = oldmat->roworiented;
  a->rowindices   = 0;
  a->rowvalues    = 0;
  a->getrowactive = PETSC_FALSE;
  a->barray       = 0;
  a->rstart_bs    = oldmat->rstart_bs;
  a->rend_bs      = oldmat->rend_bs;
  a->cstart_bs    = oldmat->cstart_bs;
  a->cend_bs      = oldmat->cend_bs;

  /* hash table stuff */
  a->ht           = 0;
  a->hd           = 0;
  a->ht_size      = 0;
  a->ht_flag      = oldmat->ht_flag;
  a->ht_fact      = oldmat->ht_fact;
  a->ht_total_ct  = 0;
  a->ht_insert_ct = 0;
  
  ierr = PetscMemcpy(a->rowners,oldmat->rowners,3*(a->size+2)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = MatStashCreate_Private(matin->comm,1,&mat->stash);CHKERRQ(ierr);
  ierr = MatStashCreate_Private(matin->comm,matin->bs,&mat->bstash);CHKERRQ(ierr);
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
  
  ierr =  VecDuplicate(oldmat->lvec,&a->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->lvec);CHKERRQ(ierr);
  ierr =  VecScatterCopy(oldmat->Mvctx,&a->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->Mvctx);CHKERRQ(ierr);

  ierr =  VecDuplicate(oldmat->slvec0,&a->slvec0);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->slvec0);CHKERRQ(ierr);
  ierr =  VecDuplicate(oldmat->slvec1,&a->slvec1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->slvec1);CHKERRQ(ierr);

  ierr = VecGetLocalSize(a->slvec1,&nt);CHKERRQ(ierr);
  ierr = VecGetArray(a->slvec1,&array);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs*mbs,array,&a->slvec1a);CHKERRQ(ierr); 
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,nt-bs*mbs,array+bs*mbs,&a->slvec1b);CHKERRQ(ierr);
  ierr = VecRestoreArray(a->slvec1,&array);CHKERRQ(ierr);
  ierr = VecGetArray(a->slvec0,&array);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,nt-bs*mbs,array+bs*mbs,&a->slvec0b);CHKERRQ(ierr);
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
  ierr = PetscFListDuplicate(mat->qlist,&matin->qlist);CHKERRQ(ierr);
  *newmat = mat;
  PetscFunctionReturn(0);
}

#include "petscsys.h"

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPISBAIJ"
PetscErrorCode MatLoad_MPISBAIJ(PetscViewer viewer,const MatType type,Mat *newmat)
{
  Mat            A;
  PetscErrorCode ierr;
  PetscInt       i,nz,j,rstart,rend;
  PetscScalar    *vals,*buf;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;
  MPI_Status     status;
  PetscMPIInt    rank,size,tag = ((PetscObject)viewer)->tag,*sndcounts = 0,*browners,maxnz,*rowners;
  PetscInt       header[4],*rowlengths = 0,M,N,m,*cols;
  PetscInt       *locrowlens,*procsnz = 0,jj,*mycols,*ibuf;
  PetscInt       bs=1,Mbs,mbs,extra_rows;
  PetscInt       *dlens,*odlens,*mask,*masked1,*masked2,rowcount,odcount;
  PetscInt       dcount,kmax,k,nzcount,tmp;
  int            fd;
 
  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
    if (header[3] < 0) {
      SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format, cannot load as MPISBAIJ");
    }
  }

  ierr = MPI_Bcast(header+1,3,MPIU_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; N = header[2];

  if (M != N) SETERRQ(PETSC_ERR_SUP,"Can only do square matrices");

  /* 
     This code adds extra rows to make sure the number of rows is 
     divisible by the blocksize
  */
  Mbs        = M/bs;
  extra_rows = bs - M + bs*(Mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  Mbs++;
  if (extra_rows &&!rank) {
    ierr = PetscLogInfo((0,"MatLoad_MPISBAIJ:Padding loaded matrix to match blocksize\n"));CHKERRQ(ierr);
  }

  /* determine ownership of all rows */
  mbs        = Mbs/size + ((Mbs % size) > rank);
  m          = mbs*bs;
  ierr       = PetscMalloc(2*(size+2)*sizeof(PetscMPIInt),&rowners);CHKERRQ(ierr);
  browners   = rowners + size + 1;
  ierr       = MPI_Allgather(&mbs,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) rowners[i] += rowners[i-1];
  for (i=0; i<=size;  i++) browners[i] = rowners[i]*bs;
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 
  
  /* distribute row lengths to all processors */
  ierr = PetscMalloc((rend-rstart)*bs*sizeof(PetscInt),&locrowlens);CHKERRQ(ierr);
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
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");
  }

  /* loop over local rows, determining number of off diagonal entries */
  ierr     = PetscMalloc(2*(rend-rstart+1)*sizeof(PetscInt),&dlens);CHKERRQ(ierr);
  odlens   = dlens + (rend-rstart);
  ierr     = PetscMalloc(3*Mbs*sizeof(PetscInt),&mask);CHKERRQ(ierr);
  ierr     = PetscMemzero(mask,3*Mbs*sizeof(PetscInt));CHKERRQ(ierr);
  masked1  = mask    + Mbs;
  masked2  = masked1 + Mbs;
  rowcount = 0; nzcount = 0;
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
  
  /* create our matrix */
  ierr = MatCreate(comm,m,m,PETSC_DETERMINE,PETSC_DETERMINE,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,type);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(A,bs,0,dlens,0,odlens);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  
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
      ierr = MatSetValues(A,1,&jj,locrowlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr); 
      mycols += locrowlens[i];
      vals   += locrowlens[i];
      jj++;
    }

    /* read in other processors (except the last one) and ship out */
    for (i=1; i<size-1; i++) {
      nz   = procsnz[i];
      vals = buf;
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,A->tag,comm);CHKERRQ(ierr);
    }
    /* the last proc */
    if (size != 1){
      nz   = procsnz[i] - extra_rows;
      vals = buf;
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      for (i=0; i<extra_rows; i++) vals[nz+i] = 1.0;
      ierr = MPI_Send(vals,nz+extra_rows,MPIU_SCALAR,size-1,A->tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(procsnz);CHKERRQ(ierr);

  } else {
    /* receive numeric values */
    ierr = PetscMalloc(nz*sizeof(PetscScalar),&buf);CHKERRQ(ierr);

    /* receive message of values*/
    vals   = buf;
    mycols = ibuf;
    ierr   = MPI_Recv(vals,nz,MPIU_SCALAR,0,A->tag,comm,&status);CHKERRQ(ierr);
    ierr   = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart*bs;
    for (i=0; i<m; i++) {
      ierr    = MatSetValues_MPISBAIJ(A,1,&jj,locrowlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr);
      mycols += locrowlens[i];
      vals   += locrowlens[i];
      jj++;
    }
  }

  ierr = PetscFree(locrowlens);CHKERRQ(ierr);
  ierr = PetscFree(buf);CHKERRQ(ierr);
  ierr = PetscFree(ibuf);CHKERRQ(ierr);
  ierr = PetscFree(rowners);CHKERRQ(ierr);
  ierr = PetscFree(dlens);CHKERRQ(ierr);
  ierr = PetscFree(mask);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *newmat = A;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMPISBAIJSetHashTableFactor"
/*XXXXX@
   MatMPISBAIJSetHashTableFactor - Sets the factor required to compute the size of the HashTable.

   Input Parameters:
.  mat  - the matrix
.  fact - factor

   Collective on Mat

   Level: advanced

  Notes:
   This can also be set by the command line option: -mat_use_hash_table fact

.keywords: matrix, hashtable, factor, HT

.seealso: MatSetOption()
@XXXXX*/


#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMax_MPISBAIJ"
PetscErrorCode MatGetRowMax_MPISBAIJ(Mat A,Vec v)
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
  ierr = MatGetRowMax(a->A,v);CHKERRQ(ierr); 
  ierr = VecGetArray(v,&va);CHKERRQ(ierr);  

  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(A->comm,&rank);CHKERRQ(ierr);

  bs   = A->bs;
  mbs  = a->mbs;
  Mbs  = a->Mbs;
  ba   = b->a;
  bi   = b->i;
  bj   = b->j;

  /* find ownerships */
  rowners_bs = a->rowners_bs;

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
      ierr    = MPI_Send(svalues,count,MPIU_REAL,dest,rank,A->comm);CHKERRQ(ierr);
    }
  }
  
  /* receive values */
  if (rank){
    rvalues = work;
    count   = rowners_bs[rank+1]-rowners_bs[rank];
    for (source=0; source<rank; source++){     
      ierr = MPI_Recv(rvalues,count,MPIU_REAL,MPI_ANY_SOURCE,MPI_ANY_TAG,A->comm,&stat);CHKERRQ(ierr);
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
#define __FUNCT__ "MatRelax_MPISBAIJ"
PetscErrorCode MatRelax_MPISBAIJ(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_MPISBAIJ   *mat = (Mat_MPISBAIJ*)matin->data;
  PetscErrorCode ierr;
  PetscInt       mbs=mat->mbs,bs=matin->bs;
  PetscScalar    mone=-1.0,*x,*b,*ptr,zero=0.0;
  Vec            bb1;
 
  PetscFunctionBegin;
  if (its <= 0 || lits <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (bs > 1)
    SETERRQ(PETSC_ERR_SUP,"SSOR for block size > 1 is not yet implemented");

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP){
    if ( flag & SOR_ZERO_INITIAL_GUESS ) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,lits,lits,xx);CHKERRQ(ierr);
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

      ierr = VecScale(&mone,mat->slvec0);CHKERRQ(ierr);

      /* copy bb into slvec1a */
      ierr = VecGetArray(mat->slvec1,&ptr);CHKERRQ(ierr);
      ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
      ierr = PetscMemcpy(ptr,b,bs*mbs*sizeof(MatScalar));CHKERRQ(ierr);
      ierr = VecRestoreArray(mat->slvec1,&ptr);CHKERRQ(ierr);  

      /* set slvec1b = 0 */
      ierr = VecSet(&zero,mat->slvec1b);CHKERRQ(ierr); 

      ierr = VecScatterBegin(mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD,mat->sMvctx);CHKERRQ(ierr);  
      ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
      ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr); 
      ierr = VecScatterEnd(mat->slvec0,mat->slvec1,ADD_VALUES,SCATTER_FORWARD,mat->sMvctx);CHKERRQ(ierr); 

      /* upper triangular part: bb1 = bb1 - B*x */ 
      ierr = (*mat->B->ops->multadd)(mat->B,mat->slvec1b,mat->slvec1a,bb1);CHKERRQ(ierr);
  
      /* local diagonal sweep */
      ierr = (*mat->A->ops->relax)(mat->A,bb1,omega,SOR_SYMMETRIC_SWEEP,fshift,lits,lits,xx);CHKERRQ(ierr); 
    }
    ierr = VecDestroy(bb1);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"MatSORType is not supported for SBAIJ matrix format");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRelax_MPISBAIJ_2comm"
PetscErrorCode MatRelax_MPISBAIJ_2comm(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_MPISBAIJ   *mat = (Mat_MPISBAIJ*)matin->data;
  PetscErrorCode ierr;
  PetscScalar    mone=-1.0;
  Vec            lvec1,bb1;
 
  PetscFunctionBegin;
  if (its <= 0 || lits <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (matin->bs > 1)
    SETERRQ(PETSC_ERR_SUP,"SSOR for block size > 1 is not yet implemented");

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP){
    if ( flag & SOR_ZERO_INITIAL_GUESS ) {
      ierr = (*mat->A->ops->relax)(mat->A,bb,omega,flag,fshift,lits,lits,xx);CHKERRQ(ierr);
      its--; 
    }

    ierr = VecDuplicate(mat->lvec,&lvec1);CHKERRQ(ierr);
    ierr = VecDuplicate(bb,&bb1);CHKERRQ(ierr);
    while (its--){
      ierr = VecScatterBegin(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr); 
   
      /* lower diagonal part: bb1 = bb - B^T*xx */
      ierr = (*mat->B->ops->multtranspose)(mat->B,xx,lvec1);CHKERRQ(ierr);
      ierr = VecScale(&mone,lvec1);CHKERRQ(ierr); 

      ierr = VecScatterEnd(xx,mat->lvec,INSERT_VALUES,SCATTER_FORWARD,mat->Mvctx);CHKERRQ(ierr); 
      ierr = VecCopy(bb,bb1);CHKERRQ(ierr);
      ierr = VecScatterBegin(lvec1,bb1,ADD_VALUES,SCATTER_REVERSE,mat->Mvctx);CHKERRQ(ierr);

      /* upper diagonal part: bb1 = bb1 - B*x */ 
      ierr = VecScale(&mone,mat->lvec);CHKERRQ(ierr);
      ierr = (*mat->B->ops->multadd)(mat->B,mat->lvec,bb1,bb1);CHKERRQ(ierr);

      ierr = VecScatterEnd(lvec1,bb1,ADD_VALUES,SCATTER_REVERSE,mat->Mvctx);CHKERRQ(ierr); 
  
      /* diagonal sweep */
      ierr = (*mat->A->ops->relax)(mat->A,bb1,omega,SOR_SYMMETRIC_SWEEP,fshift,lits,lits,xx);CHKERRQ(ierr); 
    }
    ierr = VecDestroy(lvec1);CHKERRQ(ierr);
    ierr = VecDestroy(bb1);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"MatSORType is not supported for SBAIJ matrix format");
  }
  PetscFunctionReturn(0);
}

