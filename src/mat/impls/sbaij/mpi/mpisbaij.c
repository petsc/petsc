/*$Id: mpisbaij.c,v 1.32 2000/10/24 20:26:02 bsmith Exp bsmith $*/

#include "src/mat/impls/baij/mpi/mpibaij.h"    /*I "petscmat.h" I*/
#include "src/vec/vecimpl.h"
#include "mpisbaij.h"
#include "src/mat/impls/sbaij/seq/sbaij.h"

extern int MatSetUpMultiply_MPISBAIJ(Mat); 
extern int DisAssemble_MPISBAIJ(Mat);
extern int MatIncreaseOverlap_MPISBAIJ(Mat,int,IS *,int);
extern int MatGetSubMatrices_MPISBAIJ(Mat,int,IS *,IS *,MatReuse,Mat **);
extern int MatGetValues_SeqSBAIJ(Mat,int,int *,int,int *,Scalar *);
extern int MatSetValues_SeqSBAIJ(Mat,int,int *,int,int *,Scalar *,InsertMode);
extern int MatSetValuesBlocked_SeqSBAIJ(Mat,int,int*,int,int*,Scalar*,InsertMode);
extern int MatGetRow_SeqSBAIJ(Mat,int,int*,int**,Scalar**);
extern int MatRestoreRow_SeqSBAIJ(Mat,int,int*,int**,Scalar**);
extern int MatPrintHelp_SeqSBAIJ(Mat);
extern int MatZeroRows_SeqSBAIJ(Mat,IS,Scalar*);
extern int MatZeroRows_SeqBAIJ(Mat,IS,Scalar *);
extern int MatGetRowMax_MPISBAIJ(Mat,Vec);

/*  UGLY, ugly, ugly
   When MatScalar == Scalar the function MatSetValuesBlocked_MPIBAIJ_MatScalar() does 
   not exist. Otherwise ..._MatScalar() takes matrix elements in single precision and 
   inserts them into the single precision data structure. The function MatSetValuesBlocked_MPIBAIJ()
   converts the entries into single precision and then calls ..._MatScalar() to put them
   into the single precision data structures.
*/
#if defined(PETSC_USE_MAT_SINGLE)
extern int MatSetValuesBlocked_SeqSBAIJ_MatScalar(Mat,int,int*,int,int*,MatScalar*,InsertMode);
extern int MatSetValues_MPISBAIJ_MatScalar(Mat,int,int*,int,int*,MatScalar*,InsertMode);
extern int MatSetValuesBlocked_MPISBAIJ_MatScalar(Mat,int,int*,int,int*,MatScalar*,InsertMode);
extern int MatSetValues_MPISBAIJ_HT_MatScalar(Mat,int,int*,int,int*,MatScalar*,InsertMode);
extern int MatSetValuesBlocked_MPISBAIJ_HT_MatScalar(Mat,int,int*,int,int*,MatScalar*,InsertMode);
#else
#define MatSetValuesBlocked_SeqSBAIJ_MatScalar      MatSetValuesBlocked_SeqSBAIJ
#define MatSetValues_MPISBAIJ_MatScalar             MatSetValues_MPISBAIJ
#define MatSetValuesBlocked_MPISBAIJ_MatScalar      MatSetValuesBlocked_MPISBAIJ
#define MatSetValues_MPISBAIJ_HT_MatScalar          MatSetValues_MPISBAIJ_HT
#define MatSetValuesBlocked_MPISBAIJ_HT_MatScalar   MatSetValuesBlocked_MPISBAIJ_HT
#endif

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="MatStoreValues_MPISBAIJ"></a>*/"MatStoreValues_MPISBAIJ"
int MatStoreValues_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ *aij = (Mat_MPISBAIJ *)mat->data;
  int         ierr;

  PetscFunctionBegin;
  ierr = MatStoreValues(aij->A);CHKERRQ(ierr);
  ierr = MatStoreValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="MatRetrieveValues_MPISBAIJ"></a>*/"MatRetrieveValues_MPISBAIJ"
int MatRetrieveValues_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ *aij = (Mat_MPISBAIJ *)mat->data;
  int         ierr;

  PetscFunctionBegin;
  ierr = MatRetrieveValues(aij->A);CHKERRQ(ierr);
  ierr = MatRetrieveValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* 
     Local utility routine that creates a mapping from the global column 
   number to the local number in the off-diagonal part of the local 
   storage of the matrix.  This is done in a non scable way since the 
   length of colmap equals the global matrix length. 
*/
#undef __FUNC__  
#define __FUNC__ /*<a name="CreateColmap_MPISBAIJ_Private"></a>*/"CreateColmap_MPISBAIJ_Private"
static int CreateColmap_MPISBAIJ_Private(Mat mat)
{
  PetscFunctionBegin;
  SETERRQ(1,"Function not yet written for SBAIJ format");
  /* PetscFunctionReturn(0); */
}

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
      else if (a->nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero into matrix"); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        int       new_nz = ai[a->mbs] + CHUNKSIZE,len,*new_i,*new_j; \
        MatScalar *new_a; \
 \
        if (a->nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero in the matrix"); \
 \
        /* malloc new storage space */ \
        len     = new_nz*(sizeof(int)+bs2*sizeof(MatScalar))+(a->mbs+1)*sizeof(int); \
        new_a   = (MatScalar*)PetscMalloc(len);CHKPTRQ(new_a); \
        new_j   = (int*)(new_a + bs2*new_nz); \
        new_i   = new_j + new_nz; \
 \
        /* copy over old data into new slots */ \
        for (ii=0; ii<brow+1; ii++) {new_i[ii] = ai[ii];} \
        for (ii=brow+1; ii<a->mbs+1; ii++) {new_i[ii] = ai[ii]+CHUNKSIZE;} \
        ierr = PetscMemcpy(new_j,aj,(ai[brow]+nrow)*sizeof(int));CHKERRQ(ierr); \
        len = (new_nz - CHUNKSIZE - ai[brow] - nrow); \
        ierr = PetscMemcpy(new_j+ai[brow]+nrow+CHUNKSIZE,aj+ai[brow]+nrow,len*sizeof(int));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a,aa,(ai[brow]+nrow)*bs2*sizeof(MatScalar));CHKERRQ(ierr); \
        ierr = PetscMemzero(new_a+bs2*(ai[brow]+nrow),bs2*CHUNKSIZE*sizeof(Scalar));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a+bs2*(ai[brow]+nrow+CHUNKSIZE), \
                    aa+bs2*(ai[brow]+nrow),bs2*len*sizeof(MatScalar));CHKERRQ(ierr);  \
        /* free up old matrix storage */ \
        ierr = PetscFree(a->a);CHKERRQ(ierr);  \
        if (!a->singlemalloc) { \
          ierr = PetscFree(a->i);CHKERRQ(ierr); \
          ierr = PetscFree(a->j);CHKERRQ(ierr);\
        } \
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j;  \
        a->singlemalloc = PETSC_TRUE; \
 \
        rp   = aj + ai[brow]; ap = aa + bs2*ai[brow]; \
        rmax = aimax[brow] = aimax[brow] + CHUNKSIZE; \
        PLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + bs2*sizeof(MatScalar))); \
        a->s_maxnz += bs2*CHUNKSIZE; \
        a->reallocs++; \
        a->s_nz++; \
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
      else if (b->nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero into matrix"); \
      if (nrow >= rmax) { \
        /* there is no extra room in row, therefore enlarge */ \
        int       new_nz = bi[b->mbs] + CHUNKSIZE,len,*new_i,*new_j; \
        MatScalar *new_a; \
 \
        if (b->nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero in the matrix"); \
 \
        /* malloc new storage space */ \
        len     = new_nz*(sizeof(int)+bs2*sizeof(MatScalar))+(b->mbs+1)*sizeof(int); \
        new_a   = (MatScalar*)PetscMalloc(len);CHKPTRQ(new_a); \
        new_j   = (int*)(new_a + bs2*new_nz); \
        new_i   = new_j + new_nz; \
 \
        /* copy over old data into new slots */ \
        for (ii=0; ii<brow+1; ii++) {new_i[ii] = bi[ii];} \
        for (ii=brow+1; ii<b->mbs+1; ii++) {new_i[ii] = bi[ii]+CHUNKSIZE;} \
        ierr = PetscMemcpy(new_j,bj,(bi[brow]+nrow)*sizeof(int));CHKERRQ(ierr); \
        len  = (new_nz - CHUNKSIZE - bi[brow] - nrow); \
        ierr = PetscMemcpy(new_j+bi[brow]+nrow+CHUNKSIZE,bj+bi[brow]+nrow,len*sizeof(int));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a,ba,(bi[brow]+nrow)*bs2*sizeof(MatScalar));CHKERRQ(ierr); \
        ierr = PetscMemzero(new_a+bs2*(bi[brow]+nrow),bs2*CHUNKSIZE*sizeof(MatScalar));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a+bs2*(bi[brow]+nrow+CHUNKSIZE), \
                    ba+bs2*(bi[brow]+nrow),bs2*len*sizeof(MatScalar));CHKERRQ(ierr);  \
        /* free up old matrix storage */ \
        ierr = PetscFree(b->a);CHKERRQ(ierr);  \
        if (!b->singlemalloc) { \
          ierr = PetscFree(b->i);CHKERRQ(ierr); \
          ierr = PetscFree(b->j);CHKERRQ(ierr); \
        } \
        ba = b->a = new_a; bi = b->i = new_i; bj = b->j = new_j;  \
        b->singlemalloc = PETSC_TRUE; \
 \
        rp   = bj + bi[brow]; ap = ba + bs2*bi[brow]; \
        rmax = bimax[brow] = bimax[brow] + CHUNKSIZE; \
        PLogObjectMemory(B,CHUNKSIZE*(sizeof(int) + bs2*sizeof(MatScalar))); \
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
#undef __FUNC__  
#define __FUNC__ /*<a name="MatSetValues_MPISBAIJ"></a>*/"MatSetValues_MPISBAIJ"
int MatSetValues_MPISBAIJ(Mat mat,int m,int *im,int n,int *in,Scalar *v,InsertMode addv)
{
  Mat_MPISBAIJ *b = (Mat_MPISBAIJ*)mat->data;
  int         ierr,i,N = m*n;
  MatScalar   *vsingle;

  PetscFunctionBegin;  
  if (N > b->setvalueslen) {
    if (b->setvaluescopy) {ierr = PetscFree(b->setvaluescopy);CHKERRQ(ierr);}
    b->setvaluescopy = (MatScalar*)PetscMalloc(N*sizeof(MatScalar));CHKPTRQ(b->setvaluescopy);
    b->setvalueslen  = N;
  }
  vsingle = b->setvaluescopy;

  for (i=0; i<N; i++) {
    vsingle[i] = v[i];
  }
  ierr = MatSetValues_MPISBAIJ_MatScalar(mat,m,im,n,in,vsingle,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ /*<a name="MatSetValuesBlocked_MPISBAIJ"></a>*/"MatSetValuesBlocked_MPISBAIJ"
int MatSetValuesBlocked_MPISBAIJ(Mat mat,int m,int *im,int n,int *in,Scalar *v,InsertMode addv)
{
  Mat_MPIBAIJ *b = (Mat_MPIBAIJ*)mat->data;
  int         ierr,i,N = m*n*b->bs2;
  MatScalar   *vsingle;

  PetscFunctionBegin;  
  if (N > b->setvalueslen) {
    if (b->setvaluescopy) {ierr = PetscFree(b->setvaluescopy);CHKERRQ(ierr);}
    b->setvaluescopy = (MatScalar*)PetscMalloc(N*sizeof(MatScalar));CHKPTRQ(b->setvaluescopy);
    b->setvalueslen  = N;
  }
  vsingle = b->setvaluescopy;
  for (i=0; i<N; i++) {
    vsingle[i] = v[i];
  }
  ierr = MatSetValuesBlocked_MPISBAIJ_MatScalar(mat,m,im,n,in,vsingle,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ /*<a name="MatSetValues_MPISBAIJ_HT"></a>*/"MatSetValues_MPISBAIJ_HT"
int MatSetValues_MPISBAIJ_HT(Mat mat,int m,int *im,int n,int *in,Scalar *v,InsertMode addv)
{
  Mat_MPIBAIJ *b = (Mat_MPIBAIJ*)mat->data;
  int         ierr,i,N = m*n;
  MatScalar   *vsingle;

  PetscFunctionBegin;  
  SETERRQ(1,"Function not yet written for SBAIJ format");
  /* PetscFunctionReturn(0); */
} 

#undef __FUNC__  
#define __FUNC__ /*<a name="MatSetValuesBlocked_MPISBAIJ_HT"></a>*/"MatSetValuesBlocked_MPISBAIJ_HT"
int MatSetValuesBlocked_MPISBAIJ_HT(Mat mat,int m,int *im,int n,int *in,Scalar *v,InsertMode addv)
{
  Mat_MPIBAIJ *b = (Mat_MPIBAIJ*)mat->data;
  int         ierr,i,N = m*n*b->bs2;
  MatScalar   *vsingle;

  PetscFunctionBegin;  
  SETERRQ(1,"Function not yet written for SBAIJ format"); 
  /* PetscFunctionReturn(0); */
} 
#endif

/* Only add/insert a(i,j) with i<=j (blocks). 
   Any a(i,j) with i>j input by user is ingored. 
*/
#undef __FUNC__  
#define __FUNC__ /*<a name="MatSetValues_MPIBAIJ"></a>*/"MatSetValues_MPIBAIJ"
int MatSetValues_MPISBAIJ_MatScalar(Mat mat,int m,int *im,int n,int *in,MatScalar *v,InsertMode addv)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;
  MatScalar    value;
  PetscTruth   roworiented = baij->roworiented;
  int          ierr,i,j,row,col;
  int          rstart_orig=baij->rstart_bs;
  int          rend_orig=baij->rend_bs,cstart_orig=baij->cstart_bs;
  int          cend_orig=baij->cend_bs,bs=baij->bs;

  /* Some Variables required in the macro */
  Mat          A = baij->A;
  Mat_SeqSBAIJ *a = (Mat_SeqSBAIJ*)(A)->data; 
  int          *aimax=a->imax,*ai=a->i,*ailen=a->ilen,*aj=a->j; 
  MatScalar    *aa=a->a;

  Mat          B = baij->B;
  Mat_SeqBAIJ  *b = (Mat_SeqBAIJ*)(B)->data; 
  int          *bimax=b->imax,*bi=b->i,*bilen=b->ilen,*bj=b->j; 
  MatScalar    *ba=b->a;

  int          *rp,ii,nrow,_i,rmax,N,brow,bcol; 
  int          low,high,t,ridx,cidx,bs2=a->bs2; 
  MatScalar    *ap,*bap;

  /* for stash */
  int          n_loc, *in_loc=0;
  MatScalar    *v_loc=0;

  PetscFunctionBegin;

  if(!baij->donotstash){
    in_loc = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(in_loc);
    v_loc  = (MatScalar*)PetscMalloc(n*sizeof(MatScalar));CHKPTRQ(v_loc);
  }

  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_BOPT_g)
    if (im[i] >= mat->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
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
#if defined(PETSC_USE_BOPT_g)
        else if (in[j] >= mat->N) {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Col too large");}
#endif
        else {  /* off-diag entry (B) */
          if (mat->was_assembled) {
            if (!baij->colmap) {
              ierr = CreateColmap_MPISBAIJ_Private(mat);CHKERRQ(ierr);
            }
#if defined (PETSC_USE_CTABLE)
            ierr = PetscTableFind(baij->colmap,in[j]/bs + 1,&col);CHKERRQ(ierr);
            col  = col - 1 + in[j]%bs;
#else
            col = baij->colmap[in[j]/bs] - 1 + in[j]%bs;
#endif
            if (col < 0 && !((Mat_SeqSBAIJ*)(baij->A->data))->nonew) {
              ierr = DisAssemble_MPISBAIJ(mat);CHKERRQ(ierr); 
              col =  in[j]; 
              /* Reinitialize the variables required by MatSetValues_SeqBAIJ_B_Private() */
              B = baij->B;
              b = (Mat_SeqBAIJ*)(B)->data; 
              bimax=b->imax;bi=b->i;bilen=b->ilen;bj=b->j; 
              ba=b->a;
            }
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

#undef __FUNC__  
#define __FUNC__ /*<a name="MatSetValuesBlocked_MPISBAIJ"></a>*/"MatSetValuesBlocked_MPISBAIJ"
int MatSetValuesBlocked_MPISBAIJ_MatScalar(Mat mat,int m,int *im,int n,int *in,MatScalar *v,InsertMode addv)
{
  PetscFunctionBegin;
  SETERRQ(1,"Function not yet written for SBAIJ format"); 
  /* PetscFunctionReturn(0); */
}

#define HASH_KEY 0.6180339887
#define HASH(size,key,tmp) (tmp = (key)*HASH_KEY,(int)((size)*(tmp-(int)tmp)))
/* #define HASH(size,key) ((int)((size)*fmod(((key)*HASH_KEY),1))) */
/* #define HASH(size,key,tmp) ((int)((size)*fmod(((key)*HASH_KEY),1))) */
#undef __FUNC__  
#define __FUNC__ /*<a name="MatSetValues_MPISBAIJ_HT_MatScalar"></a>*/"MatSetValues_MPISBAIJ_HT_MatScalar"
int MatSetValues_MPISBAIJ_HT_MatScalar(Mat mat,int m,int *im,int n,int *in,MatScalar *v,InsertMode addv)
{
  PetscFunctionBegin;
  SETERRQ(1,"Function not yet written for SBAIJ format"); 
  /* PetscFunctionReturn(0); */
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatSetValuesBlocked_MPISBAIJ_HT_MatScalar"
int MatSetValuesBlocked_MPISBAIJ_HT_MatScalar(Mat mat,int m,int *im,int n,int *in,MatScalar *v,InsertMode addv)
{ 
  PetscFunctionBegin;
  SETERRQ(1,"Function not yet written for SBAIJ format"); 
  /* PetscFunctionReturn(0); */
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetValues_MPISBAIJ"
int MatGetValues_MPISBAIJ(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;
  int          bs=baij->bs,ierr,i,j,bsrstart = baij->rstart*bs,bsrend = baij->rend*bs;
  int          bscstart = baij->cstart*bs,bscend = baij->cend*bs,row,col,data;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative row");
    if (idxm[i] >= mat->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (idxm[i] >= bsrstart && idxm[i] < bsrend) {
      row = idxm[i] - bsrstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative column");
        if (idxn[j] >= mat->N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
        if (idxn[j] >= bscstart && idxn[j] < bscend){
          col = idxn[j] - bscstart;
          ierr = MatGetValues_SeqSBAIJ(baij->A,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
        } else {
          if (!baij->colmap) {
            ierr = CreateColmap_MPISBAIJ_Private(mat);CHKERRQ(ierr);
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
            ierr = MatGetValues_SeqSBAIJ(baij->B,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
          } 
        }
      }
    } else {
      SETERRQ(PETSC_ERR_SUP,"Only local values currently supported");
    }
  }
 PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatNorm_MPISBAIJ"
int MatNorm_MPISBAIJ(Mat mat,NormType type,PetscReal *norm)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;
  /* Mat_SeqSBAIJ *amat = (Mat_SeqSBAIJ*)baij->A->data; */
  /* Mat_SeqBAIJ  *bmat = (Mat_SeqBAIJ*)baij->B->data; */
  int        ierr;
  PetscReal  sum[2],*lnorm2;

  PetscFunctionBegin;
  if (baij->size == 1) {
    ierr =  MatNorm(baij->A,type,norm);CHKERRQ(ierr);
  } else {
    if (type == NORM_FROBENIUS) {
      lnorm2 = (double*)PetscMalloc(2*sizeof(double));CHKPTRQ(lnorm2); 
      ierr =  MatNorm(baij->A,type,lnorm2);CHKERRQ(ierr);
      *lnorm2 = (*lnorm2)*(*lnorm2); lnorm2++;            /* squar power of norm(A) */
      ierr =  MatNorm(baij->B,type,lnorm2);CHKERRQ(ierr);
      *lnorm2 = (*lnorm2)*(*lnorm2); lnorm2--;             /* squar power of norm(B) */
      /*
      ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], lnorm2=%g, %g\n",rank,lnorm2[0],lnorm2[1]);
      */
      ierr = MPI_Allreduce(lnorm2,&sum,2,MPI_DOUBLE,MPI_SUM,mat->comm);CHKERRQ(ierr);
      /*
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], sum=%g, %g\n",rank,sum[0],sum[1]);
      PetscSynchronizedFlush(PETSC_COMM_WORLD); */
      
      *norm = sqrt(sum[0] + 2*sum[1]);
      ierr = PetscFree(lnorm2);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"No support for this norm yet");
    }
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
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatCreateHashTable_MPISBAIJ_Private"
int MatCreateHashTable_MPISBAIJ_Private(Mat mat,PetscReal factor)
{
  PetscFunctionBegin;
  SETERRQ(1,"Function not yet written for SBAIJ format");
  /* PetscFunctionReturn(0); */
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatAssemblyBegin_MPISBAIJ"
int MatAssemblyBegin_MPISBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;
  int         ierr,nstash,reallocs;
  InsertMode  addv;

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
  PLogInfo(0,"MatAssemblyBegin_MPISBAIJ:Stash has %d entries,uses %d mallocs.\n",nstash,reallocs);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  PLogInfo(0,"MatAssemblyBegin_MPISBAIJ:Block-Stash has %d entries, uses %d mallocs.\n",nstash,reallocs);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatAssemblyEnd_MPISBAIJ"
int MatAssemblyEnd_MPISBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPISBAIJ *baij=(Mat_MPISBAIJ*)mat->data;
  Mat_SeqSBAIJ  *a=(Mat_SeqSBAIJ*)baij->A->data;
  Mat_SeqBAIJ  *b=(Mat_SeqBAIJ*)baij->B->data;
  int         i,j,rstart,ncols,n,ierr,flg,bs2=baij->bs2;
  int         *row,*col,other_disassembled;
  PetscTruth  r1,r2,r3;
  MatScalar   *val;
  InsertMode  addv = mat->insertmode;
  int         rank;

  PetscFunctionBegin;
  /* remove 2 line below later */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

  if (!baij->donotstash) { 
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      /*
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]: in AssemblyEnd, stash, flg=%d\n",rank,flg);
      PetscSynchronizedFlush(PETSC_COMM_WORLD); 
      */
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
    ierr = MatSetUpMultiply_MPISBAIJ(mat);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(baij->B,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->B,mode);CHKERRQ(ierr);
  
#if defined(PETSC_USE_BOPT_g)
  if (baij->ht && mode== MAT_FINAL_ASSEMBLY) {
    PLogInfo(0,"MatAssemblyEnd_MPISBAIJ:Average Hash Table Search in MatSetValues = %5.2f\n",((double)baij->ht_total_ct)/baij->ht_insert_ct);
    baij->ht_total_ct  = 0;
    baij->ht_insert_ct = 0;
  }
#endif
  if (baij->ht_flag && !baij->ht && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatCreateHashTable_MPISBAIJ_Private(mat,baij->ht_fact);CHKERRQ(ierr);
    mat->ops->setvalues        = MatSetValues_MPISBAIJ_HT;
    mat->ops->setvaluesblocked = MatSetValuesBlocked_MPISBAIJ_HT;
  }

  if (baij->rowvalues) {
    ierr = PetscFree(baij->rowvalues);CHKERRQ(ierr);
    baij->rowvalues = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatView_MPISBAIJ_ASCIIorDraworSocket"
static int MatView_MPISBAIJ_ASCIIorDraworSocket(Mat mat,Viewer viewer)
{
  Mat_MPISBAIJ  *baij = (Mat_MPISBAIJ*)mat->data;
  int          ierr,format,bs = baij->bs,size = baij->size,rank = baij->rank;
  PetscTruth   isascii,isdraw;
  Viewer       sviewer;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  if (isascii) { 
    ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == VIEWER_FORMAT_ASCII_INFO_LONG) {
      MatInfo info;
      ierr = MPI_Comm_rank(mat->comm,&rank);CHKERRQ(ierr);
      ierr = MatGetInfo(mat,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = ViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %d nz %d nz alloced %d bs %d mem %d\n",
              rank,mat->m,(int)info.nz_used*bs,(int)info.nz_allocated*bs,
              baij->bs,(int)info.memory);CHKERRQ(ierr);      
      ierr = MatGetInfo(baij->A,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = ViewerASCIISynchronizedPrintf(viewer,"[%d] on-diagonal part: nz %d \n",rank,(int)info.nz_used*bs);CHKERRQ(ierr);
      ierr = MatGetInfo(baij->B,MAT_LOCAL,&info);CHKERRQ(ierr); 
      ierr = ViewerASCIISynchronizedPrintf(viewer,"[%d] off-diagonal part: nz %d \n",rank,(int)info.nz_used*bs);CHKERRQ(ierr);
      ierr = ViewerFlush(viewer);CHKERRQ(ierr);
      ierr = VecScatterView(baij->Mvctx,viewer);CHKERRQ(ierr);
      PetscFunctionReturn(0); 
    } else if (format == VIEWER_FORMAT_ASCII_INFO) {
      ierr = ViewerASCIIPrintf(viewer,"  block size is %d\n",bs);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }

  if (isdraw) {
    Draw       draw;
    PetscTruth isnull;
    ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  }

  if (size == 1) {
    ierr = MatView(baij->A,viewer);CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    Mat_SeqSBAIJ *Aloc;
    Mat_SeqBAIJ *Bloc;
    int         M = mat->M,N = mat->N,*ai,*aj,col,i,j,k,*rvals,mbs = baij->mbs;
    MatScalar   *a;

    if (!rank) {
      ierr = MatCreateMPISBAIJ(mat->comm,baij->bs,M,N,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);CHKERRQ(ierr);
    } else {
      ierr = MatCreateMPISBAIJ(mat->comm,baij->bs,0,0,M,N,0,PETSC_NULL,0,PETSC_NULL,&A);CHKERRQ(ierr);
    }
    PLogObjectParent(mat,A);

    /* copy over the A part */
    Aloc  = (Mat_SeqSBAIJ*)baij->A->data;
    ai    = Aloc->i; aj = Aloc->j; a = Aloc->a;
    rvals = (int*)PetscMalloc(bs*sizeof(int));CHKPTRQ(rvals);

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
       synchronized across all processors that share the Draw object
    */
    ierr = ViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatView(((Mat_MPISBAIJ*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = ViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatView_MPISBAIJ"
int MatView_MPISBAIJ(Mat mat,Viewer viewer)
{
  int        ierr;
  PetscTruth isascii,isdraw,issocket,isbinary;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,SOCKET_VIEWER,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,BINARY_VIEWER,&isbinary);CHKERRQ(ierr);
  if (isascii || isdraw || issocket || isbinary) { 
    ierr = MatView_MPISBAIJ_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported by MPISBAIJ matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatDestroy_MPISBAIJ"
int MatDestroy_MPISBAIJ(Mat mat)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;
  int         ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PLogObjectState((PetscObject)mat,"Rows=%d,Cols=%d",mat->M,mat->N);
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
  if (baij->rowvalues) {ierr = PetscFree(baij->rowvalues);CHKERRQ(ierr);}
  if (baij->barray) {ierr = PetscFree(baij->barray);CHKERRQ(ierr);}
  if (baij->hd) {ierr = PetscFree(baij->hd);CHKERRQ(ierr);}
#if defined(PETSC_USE_MAT_SINGLE)
  if (baij->setvaluescopy) {ierr = PetscFree(baij->setvaluescopy);CHKERRQ(ierr);}
#endif
  ierr = PetscFree(baij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatMult_MPISBAIJ"
int MatMult_MPISBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)A->data;
  int         ierr,nt;

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

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatMultAdd_MPISBAIJ"
int MatMultAdd_MPISBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)A->data;
  int        ierr;

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

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatMultTranspose_MPISBAIJ"
int MatMultTranspose_MPISBAIJ(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  SETERRQ(1,"Matrix is symmetric. Call MatMult().");
  /* PetscFunctionReturn(0); */
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatMultTransposeAdd_MPISBAIJ"
int MatMultTransposeAdd_MPISBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscFunctionBegin;
  SETERRQ(1,"Matrix is symmetric. Call MatMultAdd().");
  /* PetscFunctionReturn(0); */
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetDiagonal_MPISBAIJ"
int MatGetDiagonal_MPISBAIJ(Mat A,Vec v)
{
  Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)A->data;
  int         ierr;

  PetscFunctionBegin;
  /* if (a->M != a->N) SETERRQ(PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block"); */
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatScale_MPISBAIJ"
int MatScale_MPISBAIJ(Scalar *aa,Mat A)
{
  Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)A->data;
  int         ierr;

  PetscFunctionBegin;
  ierr = MatScale(aa,a->A);CHKERRQ(ierr);
  ierr = MatScale(aa,a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetOwnershipRange_MPISBAIJ"
int MatGetOwnershipRange_MPISBAIJ(Mat matin,int *m,int *n)
{
  Mat_MPISBAIJ *mat = (Mat_MPISBAIJ*)matin->data;

  PetscFunctionBegin;
  if (m) *m = mat->rstart*mat->bs;
  if (n) *n = mat->rend*mat->bs;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetRow_MPISBAIJ"
int MatGetRow_MPISBAIJ(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPISBAIJ *mat = (Mat_MPISBAIJ*)matin->data;
  Scalar     *vworkA,*vworkB,**pvA,**pvB,*v_p;
  int        bs = mat->bs,bs2 = mat->bs2,i,ierr,*cworkA,*cworkB,**pcA,**pcB;
  int        nztot,nzA,nzB,lrow,brstart = mat->rstart*bs,brend = mat->rend*bs;
  int        *cmap,*idx_p,cstart = mat->cstart;

  PetscFunctionBegin;
  if (mat->getrowactive == PETSC_TRUE) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqSBAIJ *Aa = (Mat_SeqSBAIJ*)mat->A->data;
    Mat_SeqBAIJ  *Ba = (Mat_SeqBAIJ*)mat->B->data; 
    int     max = 1,mbs = mat->mbs,tmp;
    for (i=0; i<mbs; i++) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i]; /* row length */
      if (max < tmp) { max = tmp; }
    }
    mat->rowvalues = (Scalar*)PetscMalloc(max*bs2*(sizeof(int)+sizeof(Scalar)));CHKPTRQ(mat->rowvalues);
    mat->rowindices = (int*)(mat->rowvalues + max*bs2);
  }
       
  if (row < brstart || row >= brend) SETERRQ(PETSC_ERR_SUP,"Only local rows")
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
      int imark = -1;
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

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatRestoreRow_MPISBAIJ"
int MatRestoreRow_MPISBAIJ(Mat mat,int row,int *nz,int **idx,Scalar **v)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;

  PetscFunctionBegin;
  if (baij->getrowactive == PETSC_FALSE) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"MatGetRow not called");
  }
  baij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetBlockSize_MPISBAIJ"
int MatGetBlockSize_MPISBAIJ(Mat mat,int *bs)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;

  PetscFunctionBegin;
  *bs = baij->bs;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatZeroEntries_MPISBAIJ"
int MatZeroEntries_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ *l = (Mat_MPISBAIJ*)A->data;
  int         ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetInfo_MPISBAIJ"
int MatGetInfo_MPISBAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)matin->data;
  Mat         A = a->A,B = a->B;
  int         ierr;
  PetscReal   isend[5],irecv[5];

  PetscFunctionBegin;
  info->block_size     = (double)a->bs;
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
    ierr = MPI_Allreduce(isend,irecv,5,MPI_DOUBLE,MPI_MAX,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,5,MPI_DOUBLE,MPI_SUM,matin->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else {
    SETERRQ1(1,"Unknown MatInfoType argument %d",flag);
  }
  info->rows_global       = (double)A->M;
  info->columns_global    = (double)A->N;
  info->rows_local        = (double)A->m;
  info->columns_local     = (double)A->N;
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatSetOption_MPISBAIJ"
int MatSetOption_MPISBAIJ(Mat A,MatOption op)
{
  Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)A->data;
  int         ierr;

  PetscFunctionBegin;
  if (op == MAT_NO_NEW_NONZERO_LOCATIONS ||
      op == MAT_YES_NEW_NONZERO_LOCATIONS ||
      op == MAT_COLUMNS_UNSORTED ||
      op == MAT_COLUMNS_SORTED ||
      op == MAT_NEW_NONZERO_ALLOCATION_ERR ||
      op == MAT_KEEP_ZEROED_ROWS ||
      op == MAT_NEW_NONZERO_LOCATION_ERR) {
        ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
        ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
  } else if (op == MAT_ROW_ORIENTED) {
        a->roworiented = PETSC_TRUE;
        ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
        ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
  } else if (op == MAT_ROWS_SORTED || 
             op == MAT_ROWS_UNSORTED ||
             op == MAT_SYMMETRIC ||
             op == MAT_STRUCTURALLY_SYMMETRIC ||
             op == MAT_YES_NEW_DIAGONALS ||
             op == MAT_USE_HASH_TABLE) {
    PLogInfo(A,"Info:MatSetOption_MPIBAIJ:Option ignored\n");
  } else if (op == MAT_COLUMN_ORIENTED) {
    a->roworiented = PETSC_FALSE;
    ierr = MatSetOption(a->A,op);CHKERRQ(ierr);
    ierr = MatSetOption(a->B,op);CHKERRQ(ierr);
  } else if (op == MAT_IGNORE_OFF_PROC_ENTRIES) {
    a->donotstash = PETSC_TRUE;
  } else if (op == MAT_NO_NEW_DIAGONALS) {
    SETERRQ(PETSC_ERR_SUP,"MAT_NO_NEW_DIAGONALS");
  } else if (op == MAT_USE_HASH_TABLE) {
    a->ht_flag = PETSC_TRUE;
  } else { 
    SETERRQ(PETSC_ERR_SUP,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatTranspose_MPISBAIJ("
int MatTranspose_MPISBAIJ(Mat A,Mat *matout)
{
  PetscFunctionBegin;
  SETERRQ(1,"Matrix is symmetric. MatTranspose() should not be called");
  /* PetscFunctionReturn(0); */
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatDiagonalScale_MPISBAIJ"
int MatDiagonalScale_MPISBAIJ(Mat mat,Vec ll,Vec rr)
{
  Mat_MPISBAIJ *baij = (Mat_MPISBAIJ*)mat->data;
  Mat         a = baij->A,b = baij->B;
  int         ierr,s1,s2,s3;

  PetscFunctionBegin;
  if (ll != rr) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"For symmetric format, left and right scaling vectors must be same\n");
  }
  ierr = MatGetLocalSize(mat,&s2,&s3);CHKERRQ(ierr);
  if (rr) {
    ierr = VecGetLocalSize(rr,&s1);CHKERRQ(ierr);
    if (s1!=s3) SETERRQ(PETSC_ERR_ARG_SIZ,"right vector non-conforming local size");
    /* Overlap communication with computation. */
    ierr = VecScatterBegin(rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD,baij->Mvctx);CHKERRQ(ierr);
    /*} if (ll) { */
    ierr = VecGetLocalSize(ll,&s1);CHKERRQ(ierr);
    if (s1!=s2) SETERRQ(PETSC_ERR_ARG_SIZ,"left vector non-conforming local size");
    ierr = (*b->ops->diagonalscale)(b,ll,PETSC_NULL);CHKERRQ(ierr);
    /* } */
  /* scale  the diagonal block */
  ierr = (*a->ops->diagonalscale)(a,ll,rr);CHKERRQ(ierr);

  /* if (rr) { */
    /* Do a scatter end and then right scale the off-diagonal block */
    ierr = VecScatterEnd(rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD,baij->Mvctx);CHKERRQ(ierr);
    ierr = (*b->ops->diagonalscale)(b,PETSC_NULL,baij->lvec);CHKERRQ(ierr);
  } 
  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatZeroRows_MPISBAIJ"
int MatZeroRows_MPISBAIJ(Mat A,IS is,Scalar *diag)
{
  Mat_MPISBAIJ   *l = (Mat_MPISBAIJ*)A->data;
  int            i,ierr,N,*rows,*owners = l->rowners,size = l->size;
  int            *procs,*nprocs,j,found,idx,nsends,*work,row;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values,bs=l->bs,rstart_bs=l->rstart_bs;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;
  
  PetscFunctionBegin;
  ierr = ISGetSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);
  
  /*  first count number of contributors to each processor */
  nprocs = (int*)PetscMalloc(2*size*sizeof(int));CHKPTRQ(nprocs);
  ierr   = PetscMemzero(nprocs,2*size*sizeof(int));CHKERRQ(ierr);
  procs  = nprocs + size;
  owner  = (int*)PetscMalloc((N+1)*sizeof(int));CHKPTRQ(owner); /* see note*/
  for (i=0; i<N; i++) {
    idx   = rows[i];
    found = 0;
    for (j=0; j<size; j++) {
      if (idx >= owners[j]*bs && idx < owners[j+1]*bs) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = 1; break;
      }
    }
    if (!found) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Index out of range");
  }
  nsends = 0;  for (i=0; i<size; i++) { nsends += procs[i];} 
  
  /* inform other processors of number of messages and max length*/
  work   = (int*)PetscMalloc(2*size*sizeof(int));CHKPTRQ(work);
  ierr   = MPI_Allreduce(nprocs,work,2*size,MPI_INT,PetscMaxSum_Op,comm);CHKERRQ(ierr);
  nmax   = work[rank];
  nrecvs = work[size+rank]; 
  ierr = PetscFree(work);CHKERRQ(ierr);
  
  /* post receives:   */
  rvalues = (int*)PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int));CHKPTRQ(rvalues);
  recv_waits = (MPI_Request*)PetscMalloc((nrecvs+1)*sizeof(MPI_Request));CHKPTRQ(recv_waits);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }
  
  /* do sends:
     1) starts[i] gives the starting index in svalues for stuff going to 
     the ith processor
  */
  svalues    = (int*)PetscMalloc((N+1)*sizeof(int));CHKPTRQ(svalues);
  send_waits = (MPI_Request*)PetscMalloc((nsends+1)*sizeof(MPI_Request));CHKPTRQ(send_waits);
  starts     = (int*)PetscMalloc((size+1)*sizeof(int));CHKPTRQ(starts);
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for (i=0; i<N; i++) {
    svalues[starts[owner[i]]++] = rows[i];
  }
  ierr = ISRestoreIndices(is,&rows);CHKERRQ(ierr);
  
  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (procs[i]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  base = owners[rank]*bs;
  
  /*  wait on receives */
  lens   = (int*)PetscMalloc(2*(nrecvs+1)*sizeof(int));CHKPTRQ(lens);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPI_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen          += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  /* move the data into the send scatter */
  lrows = (int*)PetscMalloc((slen+1)*sizeof(int));CHKPTRQ(lrows);
  count = 0;
  for (i=0; i<nrecvs; i++) {
    values = rvalues + i*nmax;
    for (j=0; j<lens[i]; j++) {
      lrows[count++] = values[j] - base;
    }
  }
  ierr = PetscFree(rvalues);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
    
  /* actually zap the local rows */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);   
  PLogObjectParent(A,istmp);

  /*
        Zero the required rows. If the "diagonal block" of the matrix
     is square and the user wishes to set the diagonal we use seperate
     code so that MatSetValues() is not called for each diagonal allocating
     new memory, thus calling lots of mallocs and slowing things down.

       Contributed by: Mathew Knepley
  */
  /* must zero l->B before l->A because the (diag) case below may put values into l->B*/
  ierr = MatZeroRows_SeqBAIJ(l->B,istmp,0);CHKERRQ(ierr); 
  if (diag && (l->A->M == l->A->N)) {
    ierr = MatZeroRows_SeqSBAIJ(l->A,istmp,diag);CHKERRQ(ierr);
  } else if (diag) {
    ierr = MatZeroRows_SeqSBAIJ(l->A,istmp,0);CHKERRQ(ierr);
    if (((Mat_SeqSBAIJ*)l->A->data)->nonew) {
      SETERRQ(PETSC_ERR_SUP,"MatZeroRows() on rectangular matrices cannot be used with the Mat options \n\
MAT_NO_NEW_NONZERO_LOCATIONS,MAT_NEW_NONZERO_LOCATION_ERR,MAT_NEW_NONZERO_ALLOCATION_ERR");
    }
    for (i=0; i<slen; i++) {
      row  = lrows[i] + rstart_bs;
      ierr = MatSetValues(A,1,&row,1,&row,diag,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    ierr = MatZeroRows_SeqSBAIJ(l->A,istmp,0);CHKERRQ(ierr);
  }

  ierr = ISDestroy(istmp);CHKERRQ(ierr);
  ierr = PetscFree(lrows);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    send_status = (MPI_Status*)PetscMalloc(nsends*sizeof(MPI_Status));CHKPTRQ(send_status);
    ierr        = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr        = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(svalues);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatPrintHelp_MPISBAIJ"
int MatPrintHelp_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ *a   = (Mat_MPISBAIJ*)A->data;
  MPI_Comm    comm = A->comm;
  static int  called = 0; 
  int         ierr;

  PetscFunctionBegin;
  if (!a->rank) {
    ierr = MatPrintHelp_SeqSBAIJ(a->A);CHKERRQ(ierr);
  }
  if (called) {PetscFunctionReturn(0);} else called = 1;
  ierr = (*PetscHelpPrintf)(comm," Options for MATMPISBAIJ matrix format (the defaults):\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_use_hash_table <factor>: Use hashtable for efficient matrix assembly\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatSetUnfactored_MPISBAIJ"
int MatSetUnfactored_MPISBAIJ(Mat A)
{
  Mat_MPISBAIJ *a   = (Mat_MPISBAIJ*)A->data;
  int         ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static int MatDuplicate_MPISBAIJ(Mat,MatDuplicateOption,Mat *);

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatEqual_MPISBAIJ"
int MatEqual_MPISBAIJ(Mat A,Mat B,PetscTruth *flag)
{
  Mat_MPISBAIJ *matB = (Mat_MPISBAIJ*)B->data,*matA = (Mat_MPISBAIJ*)A->data;
  Mat         a,b,c,d;
  PetscTruth  flg;
  int         ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)B,MATMPISBAIJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_ERR_ARG_INCOMP,"Matrices must be same type");
  a = matA->A; b = matA->B;
  c = matB->A; d = matB->B;

  ierr = MatEqual(a,c,&flg);CHKERRQ(ierr);
  if (flg == PETSC_TRUE) {
    ierr = MatEqual(b,d,&flg);CHKERRQ(ierr);
  }
  ierr = MPI_Allreduce(&flg,flag,1,MPI_INT,MPI_LAND,A->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatSetUpPreallocation_MPISBAIJ"></a>*/"MatSetUpPreallocation_MPISBAIJ"
int MatSetUpPreallocation_MPISBAIJ(Mat A)
{
  int        ierr;

  PetscFunctionBegin;
  ierr = MatMPISBAIJSetPreallocation(A,1,PETSC_DEFAULT,0,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {
  MatSetValues_MPISBAIJ,
  MatGetRow_MPISBAIJ,
  MatRestoreRow_MPISBAIJ,
  MatMult_MPISBAIJ,
  MatMultAdd_MPISBAIJ,
  MatMultTranspose_MPISBAIJ,
  MatMultTransposeAdd_MPISBAIJ,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  MatTranspose_MPISBAIJ,
  MatGetInfo_MPISBAIJ,
  MatEqual_MPISBAIJ,
  MatGetDiagonal_MPISBAIJ,
  MatDiagonalScale_MPISBAIJ,
  MatNorm_MPISBAIJ,
  MatAssemblyBegin_MPISBAIJ,
  MatAssemblyEnd_MPISBAIJ,
  0,
  MatSetOption_MPISBAIJ,
  MatZeroEntries_MPISBAIJ,
  MatZeroRows_MPISBAIJ,
  0,
  0,
  0,
  0,
  MatSetUpPreallocation_MPISBAIJ,
  0,
  MatGetOwnershipRange_MPISBAIJ,
  0,
  0,
  0,
  0,
  MatDuplicate_MPISBAIJ,
  0,
  0,
  0,
  0,
  0,
  MatGetSubMatrices_MPISBAIJ,
  MatIncreaseOverlap_MPISBAIJ,
  MatGetValues_MPISBAIJ,
  0,
  MatPrintHelp_MPISBAIJ,
  MatScale_MPISBAIJ,
  0,
  0,
  0,
  MatGetBlockSize_MPISBAIJ,
  0,
  0,
  0,
  0,
  0,
  0,
  MatSetUnfactored_MPISBAIJ,
  0,
  MatSetValuesBlocked_MPISBAIJ,
  0,
  0,
  0,
  MatGetMaps_Petsc,
  0,
  0,
  0,
  0,
  0,
  0,
  MatGetRowMax_MPISBAIJ};


EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetDiagonalBlock_MPISBAIJ"
int MatGetDiagonalBlock_MPISBAIJ(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
{
  PetscFunctionBegin;
  *a      = ((Mat_MPISBAIJ *)A->data)->A;
  *iscopy = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatCreate_MPISBAIJ"
int MatCreate_MPISBAIJ(Mat B)
{
  Mat_MPISBAIJ *b;
  int          ierr;
  PetscTruth   flg;

  PetscFunctionBegin;

  B->data = (void*)(b = PetscNew(Mat_MPISBAIJ));CHKPTRQ(b);
  ierr    = PetscMemzero(b,sizeof(Mat_MPISBAIJ));CHKERRQ(ierr);
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
  b->rowners = (int*)PetscMalloc(3*(b->size+2)*sizeof(int));CHKPTRQ(b->rowners);
  b->cowners    = b->rowners + b->size + 2;
  b->rowners_bs = b->cowners + b->size + 2;
  PLogObjectMemory(B,3*(b->size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPISBAIJ));

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(B->comm,1,&B->stash);CHKERRQ(ierr);
  b->donotstash  = PETSC_FALSE;
  b->colmap      = PETSC_NULL;
  b->garray      = PETSC_NULL;
  b->roworiented = PETSC_TRUE;

#if defined(PEYSC_USE_MAT_SINGLE)
  /* stuff for MatSetValues_XXX in single precision */
  b->lensetvalues     = 0;
  b->setvaluescopy    = PETSC_NULL;
#endif

  /* stuff used in block assembly */
  b->barray       = 0;

  /* stuff used for matrix vector multiply */
  b->lvec         = 0;
  b->Mvctx        = 0;

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

  ierr = OptionsHasName(PETSC_NULL,"-mat_use_hash_table",&flg);CHKERRQ(ierr);
  if (flg) { 
    double fact = 1.39;
    ierr = MatSetOption(B,MAT_USE_HASH_TABLE);CHKERRQ(ierr);
    ierr = OptionsGetDouble(PETSC_NULL,"-mat_use_hash_table",&fact,PETSC_NULL);CHKERRQ(ierr);
    if (fact <= 1.0) fact = 1.39;
    ierr = MatMPIBAIJSetHashTableFactor(B,fact);CHKERRQ(ierr);
    PLogInfo(0,"MatCreateMPISBAIJ:Hash table Factor used %5.2f\n",fact);
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
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatMPISBAIJSetPreallocation"
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
           of the in diagonal portion of the local (possibly different for each block
           row) or PETSC_NULL.  You must leave room for the diagonal entry even if it is zero.
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
   o (off-diagonal) submatrix.  Note that the d and the o submatrices are
   stored simply in the MATSEQBAIJ format for compressed row storage.

   Now d_nz should indicate the number of block nonzeros per row in the d matrix,
   and o_nz should indicate the number of block nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices.

   Level: intermediate

.keywords: matrix, block, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqSBAIJ(), MatSetValues(), MatCreateMPIBAIJ()
@*/

int MatMPISBAIJSetPreallocation(Mat B,int bs,int d_nz,int *d_nnz,int o_nz,int *o_nnz)
{
  Mat_MPISBAIJ *b;
  int          ierr,i,mbs,Mbs=PETSC_DECIDE;
  PetscTruth   flg2;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)B,MATMPISBAIJ,&flg2);CHKERRQ(ierr);
  if (!flg2) PetscFunctionReturn(0);

  ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);

  if (bs < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid block size specified, must be positive");
  if (d_nz < -2) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"d_nz cannot be less than -2: value %d",d_nz);
  if (o_nz < -2) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"o_nz cannot be less than -2: value %d",o_nz);
  if (d_nnz) {
    for (i=0; i<B->m/bs; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than -1: local row %d value %d",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->m/bs; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than -1: local row %d value %d",i,o_nnz[i]);
    }
  }
  B->preallocated = PETSC_TRUE;
  ierr = PetscSplitOwnershipBlock(B->comm,bs,&B->m,&B->M);CHKERRQ(ierr);
  ierr = PetscSplitOwnershipBlock(B->comm,bs,&B->n,&B->N);CHKERRQ(ierr);
  ierr = MapCreateMPI(B->comm,B->m,B->M,&B->rmap);CHKERRQ(ierr);
  ierr = MapCreateMPI(B->comm,B->m,B->M,&B->cmap);CHKERRQ(ierr);

  b   = (Mat_MPISBAIJ*)B->data;
  mbs = B->m/bs;
  Mbs = B->M/bs;
  if (mbs*bs != B->m) {
    SETERRQ2(PETSC_ERR_ARG_SIZ,"No of local rows %d must be divisible by blocksize %d",B->m,bs);
  }

  b->bs  = bs;
  b->bs2 = bs*bs;
  b->mbs = mbs;
  b->nbs = mbs; 
  b->Mbs = Mbs;
  b->Nbs = Mbs; 

  ierr = MPI_Allgather(&b->mbs,1,MPI_INT,b->rowners+1,1,MPI_INT,B->comm);CHKERRQ(ierr);
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
  

  if (d_nz == PETSC_DEFAULT) d_nz = 5;
  ierr = MatCreateSeqSBAIJ(PETSC_COMM_SELF,bs,B->m,B->m,d_nz,d_nnz,&b->A);CHKERRQ(ierr);
  PLogObjectParent(B,b->A);
  if (o_nz == PETSC_DEFAULT) o_nz = 0;
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,B->m,B->M,o_nz,o_nnz,&b->B);CHKERRQ(ierr);
  PLogObjectParent(B,b->B);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(B->comm,bs,&B->bstash);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatCreateMPISBAIJ"
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
           of the in diagonal portion of the local (possibly different for each block
           row) or PETSC_NULL.  You must leave room for the diagonal entry even if it is zero.
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
   o (off-diagonal) submatrix.  Note that the d and the o submatrices are
   stored simply in the MATSEQBAIJ format for compressed row storage.

   Now d_nz should indicate the number of block nonzeros per row in the d matrix,
   and o_nz should indicate the number of block nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices.

   Level: intermediate

.keywords: matrix, block, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqSBAIJ(), MatSetValues(), MatCreateMPIBAIJ()
@*/

int MatCreateMPISBAIJ(MPI_Comm comm,int bs,int m,int n,int M,int N,int d_nz,int *d_nnz,int o_nz,int *o_nnz,Mat *A)
{
  int ierr,size;

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


#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatDuplicate_MPISBAIJ"
static int MatDuplicate_MPISBAIJ(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat          mat;
  Mat_MPISBAIJ *a,*oldmat = (Mat_MPISBAIJ*)matin->data;
  int          ierr,len=0;

  PetscFunctionBegin;
  *newmat       = 0;
  ierr = MatCreate(matin->comm,matin->m,matin->n,matin->M,matin->N,&mat);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATMPISBAIJ);CHKERRQ(ierr);
  mat->preallocated = PETSC_TRUE;
  a = (Mat_MPISBAIJ*)mat->data;
  a->bs  = oldmat->bs;
  a->bs2 = oldmat->bs2;
  a->mbs = oldmat->mbs;
  a->nbs = oldmat->nbs;
  a->Mbs = oldmat->Mbs;
  a->Nbs = oldmat->Nbs;
  
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

  a->rowners = (int*)PetscMalloc(3*(a->size+2)*sizeof(int));CHKPTRQ(a->rowners);
  PLogObjectMemory(mat,3*(a->size+2)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_MPISBAIJ));
  a->cowners    = a->rowners + a->size + 2;
  a->rowners_bs = a->cowners + a->size + 2;
  ierr = PetscMemcpy(a->rowners,oldmat->rowners,3*(a->size+2)*sizeof(int));CHKERRQ(ierr);
  ierr = MatStashCreate_Private(matin->comm,1,&mat->stash);CHKERRQ(ierr);
  ierr = MatStashCreate_Private(matin->comm,oldmat->bs,&mat->bstash);CHKERRQ(ierr);
  if (oldmat->colmap) {
#if defined (PETSC_USE_CTABLE)
  ierr = PetscTableCreateCopy(oldmat->colmap,&a->colmap);CHKERRQ(ierr); 
#else
    a->colmap = (int*)PetscMalloc((a->Nbs)*sizeof(int));CHKPTRQ(a->colmap);
    PLogObjectMemory(mat,(a->Nbs)*sizeof(int));
    ierr      = PetscMemcpy(a->colmap,oldmat->colmap,(a->Nbs)*sizeof(int));CHKERRQ(ierr);
#endif
  } else a->colmap = 0;
  if (oldmat->garray && (len = ((Mat_SeqBAIJ*)(oldmat->B->data))->nbs)) {
    a->garray = (int*)PetscMalloc(len*sizeof(int));CHKPTRQ(a->garray);
    PLogObjectMemory(mat,len*sizeof(int));
    ierr = PetscMemcpy(a->garray,oldmat->garray,len*sizeof(int));CHKERRQ(ierr);
  } else a->garray = 0;
  
  ierr =  VecDuplicate(oldmat->lvec,&a->lvec);CHKERRQ(ierr);
  PLogObjectParent(mat,a->lvec);
  ierr =  VecScatterCopy(oldmat->Mvctx,&a->Mvctx);CHKERRQ(ierr);

  PLogObjectParent(mat,a->Mvctx);
  ierr =  MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  PLogObjectParent(mat,a->A);
  ierr =  MatDuplicate(oldmat->B,cpvalues,&a->B);CHKERRQ(ierr);
  PLogObjectParent(mat,a->B);
  ierr = FListDuplicate(mat->qlist,&matin->qlist);CHKERRQ(ierr);
  *newmat = mat;
  PetscFunctionReturn(0);
}

#include "petscsys.h"

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatLoad_MPISBAIJ"
int MatLoad_MPISBAIJ(Viewer viewer,MatType type,Mat *newmat)
{
  Mat          A;
  int          i,nz,ierr,j,rstart,rend,fd;
  Scalar       *vals,*buf;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;
  int          header[4],rank,size,*rowlengths = 0,M,N,m,*rowners,*browners,maxnz,*cols;
  int          *locrowlens,*sndcounts = 0,*procsnz = 0,jj,*mycols,*ibuf;
  int          tag = ((PetscObject)viewer)->tag,bs=1,Mbs,mbs,extra_rows;
  int          *dlens,*odlens,*mask,*masked1,*masked2,rowcount,odcount;
  int          dcount,kmax,k,nzcount,tmp;
 
  PetscFunctionBegin;
  ierr = OptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = ViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
    if (header[3] < 0) {
      SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format, cannot load as MPISBAIJ");
    }
  }

  ierr = MPI_Bcast(header+1,3,MPI_INT,0,comm);CHKERRQ(ierr);
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
    PLogInfo(0,"MatLoad_MPISBAIJ:Padding loaded matrix to match blocksize\n");
  }

  /* determine ownership of all rows */
  mbs = Mbs/size + ((Mbs % size) > rank);
  m   = mbs*bs;
  rowners = (int*)PetscMalloc(2*(size+2)*sizeof(int));CHKPTRQ(rowners);
  browners = rowners + size + 1;
  ierr = MPI_Allgather(&mbs,1,MPI_INT,rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) rowners[i] += rowners[i-1];
  for (i=0; i<=size;  i++) browners[i] = rowners[i]*bs;
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 
  
  /* distribute row lengths to all processors */
  locrowlens = (int*)PetscMalloc((rend-rstart)*bs*sizeof(int));CHKPTRQ(locrowlens);
  if (!rank) {
    rowlengths = (int*)PetscMalloc((M+extra_rows)*sizeof(int));CHKPTRQ(rowlengths);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
    for (i=0; i<extra_rows; i++) rowlengths[M+i] = 1;
    sndcounts = (int*)PetscMalloc(size*sizeof(int));CHKPTRQ(sndcounts);
    for (i=0; i<size; i++) sndcounts[i] = browners[i+1] - browners[i];
    ierr = MPI_Scatterv(rowlengths,sndcounts,browners,MPI_INT,locrowlens,(rend-rstart)*bs,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscFree(sndcounts);CHKERRQ(ierr);
  } else {
    ierr = MPI_Scatterv(0,0,0,MPI_INT,locrowlens,(rend-rstart)*bs,MPI_INT,0,comm);CHKERRQ(ierr);
  }
  
  if (!rank) {   /* procs[0] */
    /* calculate the number of nonzeros on each processor */
    procsnz = (int*)PetscMalloc(size*sizeof(int));CHKPTRQ(procsnz);
    ierr    = PetscMemzero(procsnz,size*sizeof(int));CHKERRQ(ierr);
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
    cols = (int*)PetscMalloc(maxnz*sizeof(int));CHKPTRQ(cols);

    /* read in my part of the matrix column indices  */
    nz = procsnz[0];
    ibuf = (int*)PetscMalloc(nz*sizeof(int));CHKPTRQ(ibuf);
    mycols = ibuf;
    if (size == 1)  nz -= extra_rows;
    ierr = PetscBinaryRead(fd,mycols,nz,PETSC_INT);CHKERRQ(ierr);
    if (size == 1)  for (i=0; i< extra_rows; i++) { mycols[nz+i] = M+i; }

    /* read in every ones (except the last) and ship off */
    for (i=1; i<size-1; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPI_INT,i,tag,comm);CHKERRQ(ierr);
    }
    /* read in the stuff for the last proc */
    if (size != 1) {
      nz   = procsnz[size-1] - extra_rows;  /* the extra rows are not on the disk */
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      for (i=0; i<extra_rows; i++) cols[nz+i] = M+i;
      ierr = MPI_Send(cols,nz+extra_rows,MPI_INT,size-1,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
  } else {  /* procs[i], i>0 */
    /* determine buffer space needed for message */
    nz = 0;
    for (i=0; i<m; i++) {
      nz += locrowlens[i];
    }
    ibuf   = (int*)PetscMalloc(nz*sizeof(int));CHKPTRQ(ibuf);
    mycols = ibuf;
    /* receive message of column indices*/
    ierr = MPI_Recv(mycols,nz,MPI_INT,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPI_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");
  }

  /* loop over local rows, determining number of off diagonal entries */
  dlens  = (int*)PetscMalloc(2*(rend-rstart+1)*sizeof(int));CHKPTRQ(dlens);
  odlens = dlens + (rend-rstart);
  mask   = (int*)PetscMalloc(3*Mbs*sizeof(int));CHKPTRQ(mask);
  ierr   = PetscMemzero(mask,3*Mbs*sizeof(int));CHKERRQ(ierr);
  masked1 = mask    + Mbs;
  masked2 = masked1 + Mbs;
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
  ierr = MatCreateMPISBAIJ(comm,bs,m,m,PETSC_DETERMINE,PETSC_DETERMINE,0,dlens,0,odlens,newmat); 
  CHKERRQ(ierr);
  A = *newmat;
  ierr = MatSetOption(A,MAT_COLUMNS_SORTED);CHKERRQ(ierr);
  
  if (!rank) {
    buf = (Scalar*)PetscMalloc(maxnz*sizeof(Scalar));CHKPTRQ(buf);
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
    buf = (Scalar*)PetscMalloc(nz*sizeof(Scalar));CHKPTRQ(buf);

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
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatMPISBAIJSetHashTableFactor"
/*@
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
@*/
int MatMPISBAIJSetHashTableFactor(Mat mat,PetscReal fact)
{
  PetscFunctionBegin;
  SETERRQ(1,"Function not yet written for SBAIJ format"); 
  /* PetscFunctionReturn(0); */
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetRowMax_MPISBAIJ"
int MatGetRowMax_MPISBAIJ(Mat A,Vec v)
{
  Mat_MPISBAIJ *a = (Mat_MPISBAIJ*)A->data;
  Mat_SeqBAIJ  *b = (Mat_SeqBAIJ*)(a->B)->data;
  PetscReal    atmp;
  double       *work,*svalues,*rvalues;
  int          ierr,i,bs,mbs,*bi,*bj,brow,j,ncols,krow,kcol,col,row,Mbs,bcol;
  int          rank,size,*rowners_bs,dest,count,source;
  Scalar       *ba,*va;
  MPI_Status   stat;

  PetscFunctionBegin;
  ierr = MatGetRowMax(a->A,v);CHKERRQ(ierr); 
  ierr = VecGetArray(v,&va);CHKERRQ(ierr);  

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  bs   = a->bs;
  mbs  = a->mbs;
  Mbs  = a->Mbs;
  ba   = b->a;
  bi   = b->i;
  bj   = b->j;
  /*
  PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] M: %d, bs: %d, mbs: %d \n",rank,bs*Mbs,bs,mbs); 
  PetscSynchronizedFlush(PETSC_COMM_WORLD);
  */

  /* find ownerships */
  rowners_bs = a->rowners_bs;
  /*
  for (i=0; i<size+1; i++) {
    PetscPrintf(PETSC_COMM_SELF,"rowners_bs: %d\n",i,rowners_bs[i]); 
  } 
  */

  /* each proc creates an array to be distributed */
  work  = (PetscReal*)PetscMalloc(bs*Mbs*sizeof(PetscReal));CHKPTRQ(work); 
  ierr  = PetscMemzero(work,bs*Mbs*sizeof(PetscReal));CHKERRQ(ierr);

  /* row_max for B */
  for (i=0; i<mbs; i++) {
    ncols = bi[1] - bi[0]; bi++;
    brow  = bs*i;
    for (j=0; j<ncols; j++){
      bcol = bs*(*bj); 
      for (kcol=0; kcol<bs; kcol++){
        col = bcol + kcol;                 /* local col index */
        col += rowners_bs[rank] + bs;      /* global col index */
        /* PetscPrintf(PETSC_COMM_SELF,"[%d], col: %d\n",rank,col); */
        for (krow=0; krow<bs; krow++){         
          atmp = PetscAbsScalar(*ba); ba++;         
          row = brow + krow;    /* local row index */
          /* printf("val[%d,%d]: %g\n",row,col,atmp); */
          if (PetscRealPart(va[row]) < atmp) va[row] = atmp;
          if (work[col] < atmp) work[col] = atmp;
        }
      }
      bj++;
    }   
  }
  
  /* send values to its owners */
  if (rank != size-1){
    for (dest=rank+1; dest<size; dest++){
      svalues = work + rowners_bs[dest];
      count = rowners_bs[dest+1]-rowners_bs[dest];
      ierr = MPI_Send(svalues,count,MPI_DOUBLE,dest,rank,PETSC_COMM_WORLD);CHKERRQ(ierr);
      /*
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] sends %d values to [%d]\n",rank,count,dest); 
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      */
    }
  }
  
  /* receive values */
  if (rank){
    rvalues = work;
    count = rowners_bs[rank+1]-rowners_bs[rank];
    for (source=0; source<rank; source++){     
      ierr = MPI_Recv(rvalues,count,MPI_DOUBLE_PRECISION,MPI_ANY_SOURCE,MPI_ANY_TAG,PETSC_COMM_WORLD,&stat);CHKERRQ(ierr);
      /* process values */     
      for (i=0; i<count; i++){
        if (PetscRealPart(va[i]) < rvalues[i]) va[i] = rvalues[i];
      }   
      /*
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] received from [%d] \n",rank,stat.MPI_SOURCE);  
      PetscSynchronizedFlush(PETSC_COMM_WORLD);
      */
    } 
  }

  ierr = VecRestoreArray(v,&va);CHKERRQ(ierr); 
  ierr = PetscFree(work);CHKERRA(ierr);
  PetscFunctionReturn(0);
}
