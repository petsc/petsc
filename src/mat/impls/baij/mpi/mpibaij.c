
#include <../src/mat/impls/baij/mpi/mpibaij.h>   /*I  "petscmat.h"  I*/
#include <petscblaslapack.h>

extern PetscErrorCode MatSetUpMultiply_MPIBAIJ(Mat); 
extern PetscErrorCode MatDisAssemble_MPIBAIJ(Mat);
extern PetscErrorCode MatGetValues_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt [],PetscScalar []);
extern PetscErrorCode MatSetValues_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt [],const PetscScalar [],InsertMode);
extern PetscErrorCode MatSetValuesBlocked_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
extern PetscErrorCode MatGetRow_SeqBAIJ(Mat,PetscInt,PetscInt*,PetscInt*[],PetscScalar*[]);
extern PetscErrorCode MatRestoreRow_SeqBAIJ(Mat,PetscInt,PetscInt*,PetscInt*[],PetscScalar*[]);
extern PetscErrorCode MatZeroRows_SeqBAIJ(Mat,PetscInt,const PetscInt[],PetscScalar,Vec,Vec);

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMaxAbs_MPIBAIJ"
PetscErrorCode MatGetRowMaxAbs_MPIBAIJ(Mat A,Vec v,PetscInt idx[])
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,*idxb = 0;
  PetscScalar    *va,*vb;
  Vec            vtmp;

  PetscFunctionBegin; 
  ierr = MatGetRowMaxAbs(a->A,v,idx);CHKERRQ(ierr); 
  ierr = VecGetArray(v,&va);CHKERRQ(ierr);
  if (idx) {
    for (i=0; i<A->rmap->n; i++) {if (PetscAbsScalar(va[i])) idx[i] += A->cmap->rstart;}
  }

  ierr = VecCreateSeq(PETSC_COMM_SELF,A->rmap->n,&vtmp);CHKERRQ(ierr);
  if (idx) {ierr = PetscMalloc(A->rmap->n*sizeof(PetscInt),&idxb);CHKERRQ(ierr);}
  ierr = MatGetRowMaxAbs(a->B,vtmp,idxb);CHKERRQ(ierr);
  ierr = VecGetArray(vtmp,&vb);CHKERRQ(ierr);

  for (i=0; i<A->rmap->n; i++){
    if (PetscAbsScalar(va[i]) < PetscAbsScalar(vb[i])) {va[i] = vb[i]; if (idx) idx[i] = A->cmap->bs*a->garray[idxb[i]/A->cmap->bs] + (idxb[i] % A->cmap->bs);}
  }

  ierr = VecRestoreArray(v,&va);CHKERRQ(ierr); 
  ierr = VecRestoreArray(vtmp,&vb);CHKERRQ(ierr); 
  ierr = PetscFree(idxb);CHKERRQ(ierr);
  ierr = VecDestroy(&vtmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues_MPIBAIJ"
PetscErrorCode  MatStoreValues_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ    *aij = (Mat_MPIBAIJ *)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatStoreValues(aij->A);CHKERRQ(ierr);
  ierr = MatStoreValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatRetrieveValues_MPIBAIJ"
PetscErrorCode  MatRetrieveValues_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ    *aij = (Mat_MPIBAIJ *)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRetrieveValues(aij->A);CHKERRQ(ierr);
  ierr = MatRetrieveValues(aij->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* 
     Local utility routine that creates a mapping from the global column 
   number to the local number in the off-diagonal part of the local 
   storage of the matrix.  This is done in a non scalable way since the
   length of colmap equals the global matrix length. 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatCreateColmap_MPIBAIJ_Private"
PetscErrorCode MatCreateColmap_MPIBAIJ_Private(Mat mat)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ    *B = (Mat_SeqBAIJ*)baij->B->data;
  PetscErrorCode ierr;
  PetscInt       nbs = B->nbs,i,bs=mat->rmap->bs;

  PetscFunctionBegin;
#if defined (PETSC_USE_CTABLE)
  ierr = PetscTableCreate(baij->nbs,baij->Nbs+1,&baij->colmap);CHKERRQ(ierr); 
  for (i=0; i<nbs; i++){
    ierr = PetscTableAdd(baij->colmap,baij->garray[i]+1,i*bs+1,INSERT_VALUES);CHKERRQ(ierr);
  }
#else
  ierr = PetscMalloc((baij->Nbs+1)*sizeof(PetscInt),&baij->colmap);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(mat,baij->Nbs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(baij->colmap,baij->Nbs*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<nbs; i++) baij->colmap[baij->garray[i]] = i*bs+1;
#endif
  PetscFunctionReturn(0);
}

#define  MatSetValues_SeqBAIJ_A_Private(row,col,value,addv) \
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

#define  MatSetValues_SeqBAIJ_B_Private(row,col,value,addv) \
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
      CHKMEMQ;\
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

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPIBAIJ"
PetscErrorCode MatSetValues_MPIBAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  MatScalar      value;
  PetscBool      roworiented = baij->roworiented;
  PetscErrorCode ierr;
  PetscInt       i,j,row,col;
  PetscInt       rstart_orig=mat->rmap->rstart;
  PetscInt       rend_orig=mat->rmap->rend,cstart_orig=mat->cmap->rstart;
  PetscInt       cend_orig=mat->cmap->rend,bs=mat->rmap->bs;

  /* Some Variables required in the macro */
  Mat            A = baij->A;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)(A)->data; 
  PetscInt       *aimax=a->imax,*ai=a->i,*ailen=a->ilen,*aj=a->j; 
  MatScalar      *aa=a->a;

  Mat            B = baij->B;
  Mat_SeqBAIJ    *b = (Mat_SeqBAIJ*)(B)->data; 
  PetscInt       *bimax=b->imax,*bi=b->i,*bilen=b->ilen,*bj=b->j; 
  MatScalar      *ba=b->a;

  PetscInt       *rp,ii,nrow,_i,rmax,N,brow,bcol; 
  PetscInt       low,high,t,ridx,cidx,bs2=a->bs2; 
  MatScalar      *ap,*bap;

  PetscFunctionBegin;
  if (v) PetscValidScalarPointer(v,6);
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (im[i] >= mat->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i],mat->rmap->N-1);
#endif
    if (im[i] >= rstart_orig && im[i] < rend_orig) {
      row = im[i] - rstart_orig;
      for (j=0; j<n; j++) {
        if (in[j] >= cstart_orig && in[j] < cend_orig){
          col = in[j] - cstart_orig;
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqBAIJ_A_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqBAIJ(baij->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        } else if (in[j] < 0) continue;
#if defined(PETSC_USE_DEBUG)
        else if (in[j] >= mat->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[j],mat->cmap->N-1);
#endif
        else {
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
            if (col < 0 && !((Mat_SeqBAIJ*)(baij->A->data))->nonew) {
              ierr = MatDisAssemble_MPIBAIJ(mat);CHKERRQ(ierr); 
              col =  in[j];
              /* Reinitialize the variables required by MatSetValues_SeqBAIJ_B_Private() */
              B = baij->B;
              b = (Mat_SeqBAIJ*)(B)->data; 
              bimax=b->imax;bi=b->i;bilen=b->ilen;bj=b->j; 
              ba=b->a;
            } else col += in[j]%bs;
          } else col = in[j];
          if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
          MatSetValues_SeqBAIJ_B_Private(row,col,value,addv);
          /* ierr = MatSetValues_SeqBAIJ(baij->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
      }
    } else {
      if (mat->nooffprocentries) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %D even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
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

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlocked_MPIBAIJ"
PetscErrorCode MatSetValuesBlocked_MPIBAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIBAIJ       *baij = (Mat_MPIBAIJ*)mat->data;
  const PetscScalar *value;
  MatScalar         *barray=baij->barray;
  PetscBool         roworiented = baij->roworiented;
  PetscErrorCode    ierr;
  PetscInt          i,j,ii,jj,row,col,rstart=baij->rstartbs;
  PetscInt          rend=baij->rendbs,cstart=baij->cstartbs,stepval;
  PetscInt          cend=baij->cendbs,bs=mat->rmap->bs,bs2=baij->bs2;
  
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
        /* If NumCol = 1 then a copy is not required */
        if ((roworiented) && (n == 1)) {
          barray = (MatScalar*)v + i*bs2;
        } else if((!roworiented) && (m == 1)) {
          barray = (MatScalar*)v + j*bs2;
        } else { /* Here a copy is required */
          if (roworiented) { 
            value = v + (i*(stepval+bs) + j)*bs;
          } else {
	    value = v + (j*(stepval+bs) + i)*bs;
          }
          for (ii=0; ii<bs; ii++,value+=bs+stepval) {
            for (jj=0; jj<bs; jj++) {
              barray[jj]  = value[jj]; 
            }
            barray += bs;
          }
          barray -= bs2;
        }
          
        if (in[j] >= cstart && in[j] < cend){
          col  = in[j] - cstart;
          ierr = MatSetValuesBlocked_SeqBAIJ(baij->A,1,&row,1,&col,barray,addv);CHKERRQ(ierr);
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
              ierr = MatDisAssemble_MPIBAIJ(mat);CHKERRQ(ierr); 
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

#define HASH_KEY 0.6180339887
#define HASH(size,key,tmp) (tmp = (key)*HASH_KEY,(PetscInt)((size)*(tmp-(PetscInt)tmp)))
/* #define HASH(size,key) ((PetscInt)((size)*fmod(((key)*HASH_KEY),1))) */
/* #define HASH(size,key,tmp) ((PetscInt)((size)*fmod(((key)*HASH_KEY),1))) */
#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPIBAIJ_HT"
PetscErrorCode MatSetValues_MPIBAIJ_HT(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  PetscBool      roworiented = baij->roworiented;
  PetscErrorCode ierr;
  PetscInt       i,j,row,col;
  PetscInt       rstart_orig=mat->rmap->rstart;
  PetscInt       rend_orig=mat->rmap->rend,Nbs=baij->Nbs;
  PetscInt       h1,key,size=baij->ht_size,bs=mat->rmap->bs,*HT=baij->ht,idx;
  PetscReal      tmp;
  MatScalar      **HD = baij->hd,value;
#if defined(PETSC_USE_DEBUG)
  PetscInt       total_ct=baij->ht_total_ct,insert_ct=baij->ht_insert_ct;
#endif

  PetscFunctionBegin;
  if (v) PetscValidScalarPointer(v,6);
  for (i=0; i<m; i++) {
#if defined(PETSC_USE_DEBUG)
    if (im[i] < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row");
    if (im[i] >= mat->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i],mat->rmap->N-1);
#endif
      row = im[i];
    if (row >= rstart_orig && row < rend_orig) {
      for (j=0; j<n; j++) {
        col = in[j];          
        if (roworiented) value = v[i*n+j]; else value = v[i+j*m];
        /* Look up PetscInto the Hash Table */
        key = (row/bs)*Nbs+(col/bs)+1;
        h1  = HASH(size,key,tmp);

        
        idx = h1;
#if defined(PETSC_USE_DEBUG)
        insert_ct++;
        total_ct++;
        if (HT[idx] != key) {
          for (idx=h1; (idx<size) && (HT[idx]!=key); idx++,total_ct++);
          if (idx == size) {
            for (idx=0; (idx<h1) && (HT[idx]!=key); idx++,total_ct++);
            if (idx == h1) {
              SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"(%D,%D) has no entry in the hash table", row, col);
            }
          }
        }
#else
        if (HT[idx] != key) {
          for (idx=h1; (idx<size) && (HT[idx]!=key); idx++);
          if (idx == size) {
            for (idx=0; (idx<h1) && (HT[idx]!=key); idx++);
            if (idx == h1) {
              SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"(%D,%D) has no entry in the hash table", row, col);
            }
          }
        }
#endif
        /* A HASH table entry is found, so insert the values at the correct address */
        if (addv == ADD_VALUES) *(HD[idx]+ (col % bs)*bs + (row % bs)) += value;
        else                    *(HD[idx]+ (col % bs)*bs + (row % bs))  = value;
      }
    } else {
      if (!baij->donotstash) {
        if (roworiented) {
          ierr = MatStashValuesRow_Private(&mat->stash,im[i],n,in,v+i*n,PETSC_FALSE);CHKERRQ(ierr);
        } else {
          ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m,PETSC_FALSE);CHKERRQ(ierr);
        }
      }
    }
  }
#if defined(PETSC_USE_DEBUG)
  baij->ht_total_ct = total_ct;
  baij->ht_insert_ct = insert_ct;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlocked_MPIBAIJ_HT"
PetscErrorCode MatSetValuesBlocked_MPIBAIJ_HT(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIBAIJ       *baij = (Mat_MPIBAIJ*)mat->data;
  PetscBool         roworiented = baij->roworiented;
  PetscErrorCode    ierr;
  PetscInt          i,j,ii,jj,row,col;
  PetscInt          rstart=baij->rstartbs;
  PetscInt          rend=mat->rmap->rend,stepval,bs=mat->rmap->bs,bs2=baij->bs2,nbs2=n*bs2;
  PetscInt          h1,key,size=baij->ht_size,idx,*HT=baij->ht,Nbs=baij->Nbs;
  PetscReal         tmp;
  MatScalar         **HD = baij->hd,*baij_a;
  const PetscScalar *v_t,*value;
#if defined(PETSC_USE_DEBUG)
  PetscInt          total_ct=baij->ht_total_ct,insert_ct=baij->ht_insert_ct;
#endif
 
  PetscFunctionBegin;

  if (roworiented) { 
    stepval = (n-1)*bs;
  } else {
    stepval = (m-1)*bs;
  }
  for (i=0; i<m; i++) {
#if defined(PETSC_USE_DEBUG)
    if (im[i] < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %D",im[i]);
    if (im[i] >= baij->Mbs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i],baij->Mbs-1);
#endif
    row   = im[i];
    v_t   = v + i*nbs2;
    if (row >= rstart && row < rend) {
      for (j=0; j<n; j++) {
        col = in[j];          

        /* Look up into the Hash Table */
        key = row*Nbs+col+1;
        h1  = HASH(size,key,tmp);
      
        idx = h1;
#if defined(PETSC_USE_DEBUG)
        total_ct++;
        insert_ct++;
       if (HT[idx] != key) {
          for (idx=h1; (idx<size) && (HT[idx]!=key); idx++,total_ct++);
          if (idx == size) {
            for (idx=0; (idx<h1) && (HT[idx]!=key); idx++,total_ct++);
            if (idx == h1) {
              SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"(%D,%D) has no entry in the hash table", row, col);
            }
          }
        }
#else  
        if (HT[idx] != key) {
          for (idx=h1; (idx<size) && (HT[idx]!=key); idx++);
          if (idx == size) {
            for (idx=0; (idx<h1) && (HT[idx]!=key); idx++);
            if (idx == h1) {
              SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"(%D,%D) has no entry in the hash table", row, col);
            }
          }
        }
#endif
        baij_a = HD[idx];
        if (roworiented) { 
          /*value = v + i*(stepval+bs)*bs + j*bs;*/
          /* value = v + (i*(stepval+bs)+j)*bs; */
          value = v_t;
          v_t  += bs;
          if (addv == ADD_VALUES) {
            for (ii=0; ii<bs; ii++,value+=stepval) {
              for (jj=ii; jj<bs2; jj+=bs) {
                baij_a[jj]  += *value++; 
              }
            }
          } else {
            for (ii=0; ii<bs; ii++,value+=stepval) {
              for (jj=ii; jj<bs2; jj+=bs) {
                baij_a[jj]  = *value++; 
              }
            }
          }
        } else {
          value = v + j*(stepval+bs)*bs + i*bs;
          if (addv == ADD_VALUES) {
            for (ii=0; ii<bs; ii++,value+=stepval,baij_a+=bs) {
              for (jj=0; jj<bs; jj++) {
                baij_a[jj]  += *value++; 
              }
            }
          } else {
            for (ii=0; ii<bs; ii++,value+=stepval,baij_a+=bs) {
              for (jj=0; jj<bs; jj++) {
                baij_a[jj]  = *value++; 
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
#if defined(PETSC_USE_DEBUG)
  baij->ht_total_ct = total_ct;
  baij->ht_insert_ct = insert_ct;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_MPIBAIJ"
PetscErrorCode MatGetValues_MPIBAIJ(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  PetscErrorCode ierr;
  PetscInt       bs=mat->rmap->bs,i,j,bsrstart = mat->rmap->rstart,bsrend = mat->rmap->rend;
  PetscInt       bscstart = mat->cmap->rstart,bscend = mat->cmap->rend,row,col,data;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (idxm[i] < 0) continue; /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %D",idxm[i]);*/
    if (idxm[i] >= mat->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",idxm[i],mat->rmap->N-1);
    if (idxm[i] >= bsrstart && idxm[i] < bsrend) {
      row = idxm[i] - bsrstart;
      for (j=0; j<n; j++) {
        if (idxn[j] < 0) continue; /* SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative column: %D",idxn[j]); */
        if (idxn[j] >= mat->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",idxn[j],mat->cmap->N-1);
        if (idxn[j] >= bscstart && idxn[j] < bscend){
          col = idxn[j] - bscstart;
          ierr = MatGetValues_SeqBAIJ(baij->A,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
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
#define __FUNCT__ "MatNorm_MPIBAIJ"
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
      v = amat->a;
      nz = amat->nz*bs2;
      for (i=0; i<nz; i++) {
#if defined(PETSC_USE_COMPLEX)
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      v = bmat->a;
      nz = bmat->nz*bs2;
      for (i=0; i<nz; i++) {
#if defined(PETSC_USE_COMPLEX)
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
        sum += (*v)*(*v); v++;
#endif
      }
      ierr = MPI_Allreduce(&sum,nrm,1,MPIU_REAL,MPIU_SUM,((PetscObject)mat)->comm);CHKERRQ(ierr);
      *nrm = PetscSqrtReal(*nrm);
    } else if (type == NORM_1) { /* max column sum */
      PetscReal *tmp,*tmp2;
      PetscInt  *jj,*garray=baij->garray,cstart=baij->rstartbs;
      ierr = PetscMalloc2(mat->cmap->N,PetscReal,&tmp,mat->cmap->N,PetscReal,&tmp2);CHKERRQ(ierr);
      ierr = PetscMemzero(tmp,mat->cmap->N*sizeof(PetscReal));CHKERRQ(ierr);
      v = amat->a; jj = amat->j;
      for (i=0; i<amat->nz; i++) {
        for (j=0; j<bs; j++){
          col = bs*(cstart + *jj) + j; /* column index */
          for (row=0; row<bs; row++){
            tmp[col] += PetscAbsScalar(*v);  v++;
          }
        }
        jj++;
      }
      v = bmat->a; jj = bmat->j;
      for (i=0; i<bmat->nz; i++) {
        for (j=0; j<bs; j++){
          col = bs*garray[*jj] + j;
          for (row=0; row<bs; row++){
            tmp[col] += PetscAbsScalar(*v); v++;
          }
        }
        jj++;
      }
      ierr = MPI_Allreduce(tmp,tmp2,mat->cmap->N,MPIU_REAL,MPIU_SUM,((PetscObject)mat)->comm);CHKERRQ(ierr);
      *nrm = 0.0;
      for (j=0; j<mat->cmap->N; j++) {
        if (tmp2[j] > *nrm) *nrm = tmp2[j];
      }
      ierr = PetscFree2(tmp,tmp2);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) { /* max row sum */
      PetscReal *sums;
      ierr = PetscMalloc(bs*sizeof(PetscReal),&sums);CHKERRQ(ierr);
      sum = 0.0;
      for (j=0; j<amat->mbs; j++) {
        for (row=0; row<bs; row++) sums[row] = 0.0;
        v = amat->a + bs2*amat->i[j];
        nz = amat->i[j+1]-amat->i[j];
        for (i=0; i<nz; i++) { 
          for (col=0; col<bs; col++){ 
            for (row=0; row<bs; row++){
              sums[row] += PetscAbsScalar(*v); v++;
            }
          }
        }
        v = bmat->a + bs2*bmat->i[j];
        nz = bmat->i[j+1]-bmat->i[j];
        for (i=0; i<nz; i++) {
          for (col=0; col<bs; col++){ 
            for (row=0; row<bs; row++){
              sums[row] += PetscAbsScalar(*v); v++;
            }
          }
        }
        for (row=0; row<bs; row++){
          if (sums[row] > sum) sum = sums[row];
        }
      }
      ierr = MPI_Allreduce(&sum,nrm,1,MPIU_REAL,MPIU_MAX,((PetscObject)mat)->comm);CHKERRQ(ierr);
      ierr = PetscFree(sums);CHKERRQ(ierr);
    } else SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"No support for this norm yet");
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
#undef __FUNCT__  
#define __FUNCT__ "MatCreateHashTable_MPIBAIJ_Private"
PetscErrorCode MatCreateHashTable_MPIBAIJ_Private(Mat mat,PetscReal factor)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  Mat            A = baij->A,B=baij->B;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ *)A->data,*b=(Mat_SeqBAIJ *)B->data;
  PetscInt       i,j,k,nz=a->nz+b->nz,h1,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  PetscErrorCode ierr;
  PetscInt       ht_size,bs2=baij->bs2,rstart=baij->rstartbs;
  PetscInt       cstart=baij->cstartbs,*garray=baij->garray,row,col,Nbs=baij->Nbs;
  PetscInt       *HT,key;
  MatScalar      **HD;
  PetscReal      tmp;
#if defined(PETSC_USE_INFO)
  PetscInt       ct=0,max=0;
#endif

  PetscFunctionBegin;
  if (baij->ht) PetscFunctionReturn(0);

  baij->ht_size = (PetscInt)(factor*nz);
  ht_size       = baij->ht_size;
  
  /* Allocate Memory for Hash Table */
  ierr = PetscMalloc2(ht_size,MatScalar*,&baij->hd,ht_size,PetscInt,&baij->ht);CHKERRQ(ierr);
  ierr = PetscMemzero(baij->hd,ht_size*sizeof(MatScalar*));CHKERRQ(ierr);
  ierr = PetscMemzero(baij->ht,ht_size*sizeof(PetscInt));CHKERRQ(ierr);
  HD   = baij->hd;
  HT   = baij->ht;

  /* Loop Over A */
  for (i=0; i<a->mbs; i++) {
    for (j=ai[i]; j<ai[i+1]; j++) {
      row = i+rstart;
      col = aj[j]+cstart;
       
      key = row*Nbs + col + 1;
      h1  = HASH(ht_size,key,tmp);
      for (k=0; k<ht_size; k++){
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
      for (k=0; k<ht_size; k++){
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
    if (HT[i]) {j++;}
  }
  ierr = PetscInfo2(mat,"Average Search = %5.2f,max search = %D\n",(!j)? 0.0:((PetscReal)(ct+j))/j,max);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_MPIBAIJ"
PetscErrorCode MatAssemblyBegin_MPIBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
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
  ierr = MatStashGetInfo_Private(&mat->bstash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(mat,"Block-Stash has %D entries, uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPIBAIJ"
PetscErrorCode MatAssemblyEnd_MPIBAIJ(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIBAIJ    *baij=(Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)baij->A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart,ncols,flg,bs2=baij->bs2;
  PetscInt       *row,*col;
  PetscBool      r1,r2,r3,other_disassembled;
  MatScalar      *val;
  InsertMode     addv = mat->insertmode;
  PetscMPIInt    n;

  /* do not use 'b=(Mat_SeqBAIJ*)baij->B->data' as B can be reset in disassembly */
  PetscFunctionBegin;
  if (!baij->donotstash && !mat->nooffprocentries) {
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;

      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        /* Now assemble all these values with a single function call */
        ierr = MatSetValues_MPIBAIJ(mat,1,row+i,ncols,col+i,val+i,addv);CHKERRQ(ierr);
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
    (((Mat_SeqBAIJ*)baij->B->data))->roworiented    = PETSC_FALSE; /* b->roworiented */
    while (1) {
      ierr = MatStashScatterGetMesg_Private(&mat->bstash,&n,&row,&col,&val,&flg);CHKERRQ(ierr);
      if (!flg) break;
      
      for (i=0; i<n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j=i,rstart=row[j]; j<n; j++) { if (row[j] != rstart) break; }
        if (j < n) ncols = j-i;
        else       ncols = n-i;
        ierr = MatSetValuesBlocked_MPIBAIJ(mat,1,row+i,ncols,col+i,val+i*bs2,addv);CHKERRQ(ierr);
        i = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->bstash);CHKERRQ(ierr);
    baij->roworiented = r1;
    a->roworiented    = r2;
    ((Mat_SeqBAIJ*)baij->B->data)->roworiented    = r3; /* b->roworiented */
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
      ierr = MatDisAssemble_MPIBAIJ(mat);CHKERRQ(ierr);
    }
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIBAIJ(mat);CHKERRQ(ierr);
  }
  ierr = MatSetOption(baij->B,MAT_CHECK_COMPRESSED_ROW,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(baij->B,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(baij->B,mode);CHKERRQ(ierr);
  
#if defined(PETSC_USE_INFO)
  if (baij->ht && mode== MAT_FINAL_ASSEMBLY) {
    ierr = PetscInfo1(mat,"Average Hash Table Search in MatSetValues = %5.2f\n",((PetscReal)baij->ht_total_ct)/baij->ht_insert_ct);CHKERRQ(ierr);
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
  baij->rowvalues = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIBAIJ_ASCIIorDraworSocket"
static PetscErrorCode MatView_MPIBAIJ_ASCIIorDraworSocket(Mat mat,PetscViewer viewer)
{
  Mat_MPIBAIJ       *baij = (Mat_MPIBAIJ*)mat->data;
  PetscErrorCode    ierr;
  PetscMPIInt       size = baij->size,rank = baij->rank;
  PetscInt          bs = mat->rmap->bs;
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
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local rows %D nz %D nz alloced %D bs %D mem %D\n",
             rank,mat->rmap->n,(PetscInt)info.nz_used,(PetscInt)info.nz_allocated,mat->rmap->bs,(PetscInt)info.memory);CHKERRQ(ierr);
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
    PetscDraw       draw;
    PetscBool  isnull;
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  }

  if (size == 1) {
    ierr = PetscObjectSetName((PetscObject)baij->A,((PetscObject)mat)->name);CHKERRQ(ierr);
    ierr = MatView(baij->A,viewer);CHKERRQ(ierr);
  } else {
    /* assemble the entire matrix onto first processor. */
    Mat         A;
    Mat_SeqBAIJ *Aloc;
    PetscInt    M = mat->rmap->N,N = mat->cmap->N,*ai,*aj,col,i,j,k,*rvals,mbs = baij->mbs;
    MatScalar   *a;

    /* Here we are creating a temporary matrix, so will assume MPIBAIJ is acceptable */
    /* Perhaps this should be the type of mat? */
    ierr = MatCreate(((PetscObject)mat)->comm,&A);CHKERRQ(ierr);
    if (!rank) {
      ierr = MatSetSizes(A,M,N,M,N);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(A,0,0,M,N);CHKERRQ(ierr);
    }
    ierr = MatSetType(A,MATMPIBAIJ);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(A,mat->rmap->bs,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(mat,A);CHKERRQ(ierr);

    /* copy over the A part */
    Aloc = (Mat_SeqBAIJ*)baij->A->data;
    ai   = Aloc->i; aj = Aloc->j; a = Aloc->a;
    ierr = PetscMalloc(bs*sizeof(PetscInt),&rvals);CHKERRQ(ierr);

    for (i=0; i<mbs; i++) {
      rvals[0] = bs*(baij->rstartbs + i);
      for (j=1; j<bs; j++) { rvals[j] = rvals[j-1] + 1; }
      for (j=ai[i]; j<ai[i+1]; j++) {
        col = (baij->cstartbs+aj[j])*bs;
        for (k=0; k<bs; k++) {
          ierr = MatSetValues_MPIBAIJ(A,bs,rvals,1,&col,a,INSERT_VALUES);CHKERRQ(ierr);
          col++; a += bs;
        }
      }
    } 
    /* copy over the B part */
    Aloc = (Mat_SeqBAIJ*)baij->B->data;
    ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
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
      ierr = PetscObjectSetName((PetscObject)((Mat_MPIBAIJ*)(A->data))->A,((PetscObject)mat)->name);CHKERRQ(ierr);
    /* Set the type name to MATMPIBAIJ so that the correct type can be printed out by PetscObjectPrintClassNamePrefixType() in MatView_SeqBAIJ_ASCII()*/
      PetscStrcpy(((PetscObject)((Mat_MPIBAIJ*)(A->data))->A)->type_name,MATMPIBAIJ);
      ierr = MatView(((Mat_MPIBAIJ*)(A->data))->A,sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIBAIJ_Binary"
static PetscErrorCode MatView_MPIBAIJ_Binary(Mat mat,PetscViewer viewer)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)mat->data;
  Mat_SeqBAIJ*   A = (Mat_SeqBAIJ*)a->A->data;
  Mat_SeqBAIJ*   B = (Mat_SeqBAIJ*)a->B->data;
  PetscErrorCode ierr;
  PetscInt       i,*row_lens,*crow_lens,bs = mat->rmap->bs,j,k,bs2=a->bs2,header[4],nz,rlen;
  PetscInt       *range=0,nzmax,*column_indices,cnt,col,*garray = a->garray,cstart = mat->cmap->rstart/bs,len,pcnt,l,ll;
  int            fd;
  PetscScalar    *column_values;
  FILE           *file;
  PetscMPIInt    rank,size,tag = ((PetscObject)viewer)->tag;
  PetscInt       message_count,flowcontrolcount;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)mat)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)mat)->comm,&size);CHKERRQ(ierr);
  nz   = bs2*(A->nz + B->nz);
  rlen = mat->rmap->n;
  if (!rank) {
    header[0] = MAT_FILE_CLASSID;
    header[1] = mat->rmap->N;
    header[2] = mat->cmap->N;
    ierr = MPI_Reduce(&nz,&header[3],1,MPIU_INT,MPI_SUM,0,((PetscObject)mat)->comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,header,4,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    /* get largest number of rows any processor has */
    range = mat->rmap->range;
    for (i=1; i<size; i++) {
      rlen = PetscMax(rlen,range[i+1] - range[i]);
    }
  } else {
    ierr = MPI_Reduce(&nz,0,1,MPIU_INT,MPI_SUM,0,((PetscObject)mat)->comm);CHKERRQ(ierr);
  }

  ierr  = PetscMalloc((rlen/bs)*sizeof(PetscInt),&crow_lens);CHKERRQ(ierr);
  /* compute lengths of each row  */
  for (i=0; i<a->mbs; i++) {
    crow_lens[i] = A->i[i+1] - A->i[i] + B->i[i+1] - B->i[i];
  }
  /* store the row lengths to the file */
  ierr = PetscViewerFlowControlStart(viewer,&message_count,&flowcontrolcount);CHKERRQ(ierr);
  if (!rank) {
    MPI_Status status;
    ierr  = PetscMalloc(rlen*sizeof(PetscInt),&row_lens);CHKERRQ(ierr);    
    rlen  = (range[1] - range[0])/bs;
    for (i=0; i<rlen; i++) {
      for (j=0; j<bs; j++) {
        row_lens[i*bs+j] = bs*crow_lens[i];
      }
    }
    ierr = PetscBinaryWrite(fd,row_lens,bs*rlen,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    for (i=1; i<size; i++) {
      rlen = (range[i+1] - range[i])/bs;
      ierr = PetscViewerFlowControlStepMaster(viewer,i,message_count,flowcontrolcount);CHKERRQ(ierr);
      ierr = MPI_Recv(crow_lens,rlen,MPIU_INT,i,tag,((PetscObject)mat)->comm,&status);CHKERRQ(ierr);
      for (k=0; k<rlen; k++) {
	for (j=0; j<bs; j++) {
	  row_lens[k*bs+j] = bs*crow_lens[k];
	}
      }
      ierr = PetscBinaryWrite(fd,row_lens,bs*rlen,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlowControlEndMaster(viewer,message_count);CHKERRQ(ierr);
    ierr = PetscFree(row_lens);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerFlowControlStepWorker(viewer,rank,message_count);CHKERRQ(ierr);
    ierr = MPI_Send(crow_lens,mat->rmap->n/bs,MPIU_INT,0,tag,((PetscObject)mat)->comm);CHKERRQ(ierr);
    ierr = PetscViewerFlowControlEndWorker(viewer,message_count);CHKERRQ(ierr);
  }
  ierr = PetscFree(crow_lens);CHKERRQ(ierr);

  /* load up the local column indices. Include for all rows not just one for each block row since process 0 does not have the
     information needed to make it for each row from a block row. This does require more communication but still not more than
     the communication needed for the nonzero values  */
  nzmax = nz; /*  space a largest processor needs */
  ierr = MPI_Reduce(&nz,&nzmax,1,MPIU_INT,MPI_MAX,0,((PetscObject)mat)->comm);CHKERRQ(ierr);
  ierr = PetscMalloc(nzmax*sizeof(PetscInt),&column_indices);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<a->mbs; i++) {
    pcnt = cnt;
    for (j=B->i[i]; j<B->i[i+1]; j++) {
      if ( (col = garray[B->j[j]]) > cstart) break;
      for (l=0; l<bs; l++) {
	column_indices[cnt++] = bs*col+l;
      }
    }
    for (k=A->i[i]; k<A->i[i+1]; k++) {
      for (l=0; l<bs; l++) {
        column_indices[cnt++] = bs*(A->j[k] + cstart)+l;
      }
    }
    for (; j<B->i[i+1]; j++) {
      for (l=0; l<bs; l++) {
        column_indices[cnt++] = bs*garray[B->j[j]]+l;
      }
    }
    len = cnt - pcnt;
    for (k=1; k<bs; k++) {
      ierr = PetscMemcpy(&column_indices[cnt],&column_indices[pcnt],len*sizeof(PetscInt));CHKERRQ(ierr);
      cnt += len;
    }
  }
  if (cnt != nz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Internal PETSc error: cnt = %D nz = %D",cnt,nz);

  /* store the columns to the file */
  ierr = PetscViewerFlowControlStart(viewer,&message_count,&flowcontrolcount);CHKERRQ(ierr);
  if (!rank) {
    MPI_Status status;
    ierr = PetscBinaryWrite(fd,column_indices,nz,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    for (i=1; i<size; i++) {
      ierr = PetscViewerFlowControlStepMaster(viewer,i,message_count,flowcontrolcount);CHKERRQ(ierr);
      ierr = MPI_Recv(&cnt,1,MPIU_INT,i,tag,((PetscObject)mat)->comm,&status);CHKERRQ(ierr);
      ierr = MPI_Recv(column_indices,cnt,MPIU_INT,i,tag,((PetscObject)mat)->comm,&status);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,column_indices,cnt,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlowControlEndMaster(viewer,message_count);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerFlowControlStepWorker(viewer,rank,message_count);CHKERRQ(ierr);
    ierr = MPI_Send(&cnt,1,MPIU_INT,0,tag,((PetscObject)mat)->comm);CHKERRQ(ierr);
    ierr = MPI_Send(column_indices,cnt,MPIU_INT,0,tag,((PetscObject)mat)->comm);CHKERRQ(ierr);
    ierr = PetscViewerFlowControlEndWorker(viewer,message_count);CHKERRQ(ierr);
  }
  ierr = PetscFree(column_indices);CHKERRQ(ierr);

  /* load up the numerical values */
  ierr = PetscMalloc(nzmax*sizeof(PetscScalar),&column_values);CHKERRQ(ierr);
  cnt = 0;
  for (i=0; i<a->mbs; i++) {
    rlen = bs*(B->i[i+1] - B->i[i] + A->i[i+1] - A->i[i]);
    for (j=B->i[i]; j<B->i[i+1]; j++) {
      if ( garray[B->j[j]] > cstart) break;
      for (l=0; l<bs; l++) {
        for (ll=0; ll<bs; ll++) {
	  column_values[cnt + l*rlen + ll] = B->a[bs2*j+l+bs*ll];
        }
      }
      cnt += bs;
    }
    for (k=A->i[i]; k<A->i[i+1]; k++) {
      for (l=0; l<bs; l++) {
        for (ll=0; ll<bs; ll++) {
          column_values[cnt + l*rlen + ll] = A->a[bs2*k+l+bs*ll];
        }
      }
      cnt += bs;
    }
    for (; j<B->i[i+1]; j++) {
      for (l=0; l<bs; l++) {
        for (ll=0; ll<bs; ll++) {
	  column_values[cnt + l*rlen + ll] = B->a[bs2*j+l+bs*ll];
        }
      }
      cnt += bs;
    }
    cnt += (bs-1)*rlen;
  }
  if (cnt != nz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Internal PETSc error: cnt = %D nz = %D",cnt,nz);

  /* store the column values to the file */
  ierr = PetscViewerFlowControlStart(viewer,&message_count,&flowcontrolcount);CHKERRQ(ierr);
  if (!rank) {
    MPI_Status status;
    ierr = PetscBinaryWrite(fd,column_values,nz,PETSC_SCALAR,PETSC_TRUE);CHKERRQ(ierr);
    for (i=1; i<size; i++) {
      ierr = PetscViewerFlowControlStepMaster(viewer,i,message_count,flowcontrolcount);CHKERRQ(ierr);
      ierr = MPI_Recv(&cnt,1,MPIU_INT,i,tag,((PetscObject)mat)->comm,&status);CHKERRQ(ierr);
      ierr = MPI_Recv(column_values,cnt,MPIU_SCALAR,i,tag,((PetscObject)mat)->comm,&status);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,column_values,cnt,PETSC_SCALAR,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlowControlEndMaster(viewer,message_count);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerFlowControlStepWorker(viewer,rank,message_count);CHKERRQ(ierr);
    ierr = MPI_Send(&nz,1,MPIU_INT,0,tag,((PetscObject)mat)->comm);CHKERRQ(ierr);
    ierr = MPI_Send(column_values,nz,MPIU_SCALAR,0,tag,((PetscObject)mat)->comm);CHKERRQ(ierr);
    ierr = PetscViewerFlowControlEndWorker(viewer,message_count);CHKERRQ(ierr);
  }
  ierr = PetscFree(column_values);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
  if (file) {
    fprintf(file,"-matload_block_size %d\n",(int)mat->rmap->bs);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIBAIJ"
PetscErrorCode MatView_MPIBAIJ(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isdraw,issocket,isbinary;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii || isdraw || issocket) { 
    ierr = MatView_MPIBAIJ_ASCIIorDraworSocket(mat,viewer);CHKERRQ(ierr);
  } else if (isbinary) { 
    ierr = MatView_MPIBAIJ_Binary(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Viewer type %s not supported by MPIBAIJ matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPIBAIJ"
PetscErrorCode MatDestroy_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
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
  ierr = PetscFree2(baij->rowvalues,baij->rowindices);CHKERRQ(ierr);
  ierr = PetscFree(baij->barray);CHKERRQ(ierr);
  ierr = PetscFree2(baij->hd,baij->ht);CHKERRQ(ierr);
  ierr = PetscFree(baij->rangebs);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatStoreValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatRetrieveValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatGetDiagonalBlock_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIBAIJSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIBAIJSetPreallocationCSR_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDiagonalScaleLocal_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatSetHashTableFactor_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpibaij_mpisbaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_mpibaij_mpibstrm_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_MPIBAIJ"
PetscErrorCode MatMult_MPIBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A and xx");
  ierr = VecGetLocalSize(yy,&nt);CHKERRQ(ierr);
  if (nt != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible parition of A and yy");
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_MPIBAIJ"
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

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_MPIBAIJ"
PetscErrorCode MatMultTranspose_MPIBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;
  PetscBool      merged;

  PetscFunctionBegin;
  ierr = VecScatterGetMerged(a->Mvctx,&merged);CHKERRQ(ierr);
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  if (!merged) {
    /* send it on its way */
    ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    /* do local part */
    ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
    /* receive remote parts: note this assumes the values are not actually */
    /* inserted in yy until the next line */
    ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else {
    /* do local part */
    ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
    /* send it on its way */
    ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    /* values actually were received in the Begin() but we need to call this nop */
    ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_MPIBAIJ"
PetscErrorCode MatMultTransposeAdd_MPIBAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* do nondiagonal part */
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  /* send it on its way */
  ierr = VecScatterBegin(a->Mvctx,a->lvec,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* do local part */
  ierr = (*a->A->ops->multtransposeadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  /* receive remote parts: note this assumes the values are not actually */
  /* inserted in yy until the next line, which is true for my implementation*/
  /* but is not perhaps always true. */
  ierr = VecScatterEnd(a->Mvctx,a->lvec,zz,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This only works correctly for square matrices where the subblock A->A is the 
   diagonal block
*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_MPIBAIJ"
PetscErrorCode MatGetDiagonal_MPIBAIJ(Mat A,Vec v)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->rmap->N != A->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block");
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_MPIBAIJ"
PetscErrorCode MatScale_MPIBAIJ(Mat A,PetscScalar aa)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(a->A,aa);CHKERRQ(ierr);
  ierr = MatScale(a->B,aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_MPIBAIJ"
PetscErrorCode MatGetRow_MPIBAIJ(Mat matin,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIBAIJ    *mat = (Mat_MPIBAIJ*)matin->data;
  PetscScalar    *vworkA,*vworkB,**pvA,**pvB,*v_p;
  PetscErrorCode ierr;
  PetscInt       bs = matin->rmap->bs,bs2 = mat->bs2,i,*cworkA,*cworkB,**pcA,**pcB;
  PetscInt       nztot,nzA,nzB,lrow,brstart = matin->rmap->rstart,brend = matin->rmap->rend;
  PetscInt       *cmap,*idx_p,cstart = mat->cstartbs;

  PetscFunctionBegin;
  if (row < brstart || row >= brend) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local rows");
  if (mat->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqBAIJ *Aa = (Mat_SeqBAIJ*)mat->A->data,*Ba = (Mat_SeqBAIJ*)mat->B->data; 
    PetscInt     max = 1,mbs = mat->mbs,tmp;
    for (i=0; i<mbs; i++) {
      tmp = Aa->i[i+1] - Aa->i[i] + Ba->i[i+1] - Ba->i[i];
      if (max < tmp) { max = tmp; }
    }
    ierr = PetscMalloc2(max*bs2,PetscScalar,&mat->rowvalues,max*bs2,PetscInt,&mat->rowindices);CHKERRQ(ierr);
  }
  lrow = row - brstart;

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
#define __FUNCT__ "MatRestoreRow_MPIBAIJ"
PetscErrorCode MatRestoreRow_MPIBAIJ(Mat mat,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ*)mat->data;

  PetscFunctionBegin;
  if (!baij->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"MatGetRow not called");
  baij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_MPIBAIJ"
PetscErrorCode MatZeroEntries_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ    *l = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_MPIBAIJ"
PetscErrorCode MatGetInfo_MPIBAIJ(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)matin->data;
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
    SETERRQ1(((PetscObject)matin)->comm,PETSC_ERR_ARG_WRONG,"Unknown MatInfoType argument %d",(int)flag);
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPIBAIJ"
PetscErrorCode MatSetOption_MPIBAIJ(Mat A,MatOption op,PetscBool  flg)
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
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  default: 
    SETERRQ1(((PetscObject)A)->comm,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_MPIBAIJ"
PetscErrorCode MatTranspose_MPIBAIJ(Mat A,MatReuse reuse,Mat *matout)
{ 
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)A->data;
  Mat_SeqBAIJ    *Aloc;
  Mat            B;
  PetscErrorCode ierr;
  PetscInt       M=A->rmap->N,N=A->cmap->N,*ai,*aj,i,*rvals,j,k,col;
  PetscInt       bs=A->rmap->bs,mbs=baij->mbs;
  MatScalar      *a;
  
  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX && A == *matout && M != N) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_SIZ,"Square matrix only for in-place");
  if (reuse == MAT_INITIAL_MATRIX || *matout == A) {
    ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->cmap->n,A->rmap->n,N,M);CHKERRQ(ierr);
    ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
    /* Do not know preallocation information, but must set block size */
    ierr = MatMPIBAIJSetPreallocation(B,A->rmap->bs,PETSC_DECIDE,PETSC_NULL,PETSC_DECIDE,PETSC_NULL);CHKERRQ(ierr);
  } else {
    B = *matout;
  }

  /* copy over the A part */
  Aloc = (Mat_SeqBAIJ*)baij->A->data;
  ai   = Aloc->i; aj = Aloc->j; a = Aloc->a;
  ierr = PetscMalloc(bs*sizeof(PetscInt),&rvals);CHKERRQ(ierr);
  
  for (i=0; i<mbs; i++) {
    rvals[0] = bs*(baij->rstartbs + i);
    for (j=1; j<bs; j++) { rvals[j] = rvals[j-1] + 1; }
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
  ai = Aloc->i; aj = Aloc->j; a = Aloc->a;
  for (i=0; i<mbs; i++) {
    rvals[0] = bs*(baij->rstartbs + i);
    for (j=1; j<bs; j++) { rvals[j] = rvals[j-1] + 1; }
    for (j=ai[i]; j<ai[i+1]; j++) {
      col = baij->garray[aj[j]]*bs;
      for (k=0; k<bs; k++) { 
        ierr = MatSetValues_MPIBAIJ(B,1,&col,bs,rvals,a,INSERT_VALUES);CHKERRQ(ierr);
        col++; a += bs;
      }
    }
  } 
  ierr = PetscFree(rvals);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  if (reuse == MAT_INITIAL_MATRIX || *matout != A) {
    *matout = B;
  } else {
    ierr = MatHeaderMerge(A,B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_MPIBAIJ"
PetscErrorCode MatDiagonalScale_MPIBAIJ(Mat mat,Vec ll,Vec rr)
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*)mat->data;
  Mat            a = baij->A,b = baij->B;
  PetscErrorCode ierr;
  PetscInt       s1,s2,s3;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(mat,&s2,&s3);CHKERRQ(ierr);
  if (rr) {
    ierr = VecGetLocalSize(rr,&s1);CHKERRQ(ierr);
    if (s1!=s3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"right vector non-conforming local size");
    /* Overlap communication with computation. */
    ierr = VecScatterBegin(baij->Mvctx,rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  if (ll) {
    ierr = VecGetLocalSize(ll,&s1);CHKERRQ(ierr);
    if (s1!=s2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"left vector non-conforming local size");
    ierr = (*b->ops->diagonalscale)(b,ll,PETSC_NULL);CHKERRQ(ierr);
  }
  /* scale  the diagonal block */
  ierr = (*a->ops->diagonalscale)(a,ll,rr);CHKERRQ(ierr);

  if (rr) {
    /* Do a scatter end and then right scale the off-diagonal block */
    ierr = VecScatterEnd(baij->Mvctx,rr,baij->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = (*b->ops->diagonalscale)(b,PETSC_NULL,baij->lvec);CHKERRQ(ierr);
  } 
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_MPIBAIJ"
PetscErrorCode MatZeroRows_MPIBAIJ(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_MPIBAIJ       *l = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscMPIInt       imdex,size = l->size,n,rank = l->rank;
  PetscInt          i,*owners = A->rmap->range;
  PetscInt          *nprocs,j,idx,nsends,row;
  PetscInt          nmax,*svalues,*starts,*owner,nrecvs;
  PetscInt          *rvalues,tag = ((PetscObject)A)->tag,count,base,slen,*source,lastidx = -1;
  PetscInt          *lens,*lrows,*values,rstart_bs=A->rmap->rstart;
  MPI_Comm          comm = ((PetscObject)A)->comm;
  MPI_Request       *send_waits,*recv_waits;
  MPI_Status        recv_status,*send_status;
  const PetscScalar *xx;
  PetscScalar       *bb;
#if defined(PETSC_DEBUG)
  PetscBool         found = PETSC_FALSE;
#endif
  
  PetscFunctionBegin;
  /*  first count number of contributors to each processor */
  ierr  = PetscMalloc(2*size*sizeof(PetscInt),&nprocs);CHKERRQ(ierr);
  ierr  = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);
  ierr  = PetscMalloc((N+1)*sizeof(PetscInt),&owner);CHKERRQ(ierr); /* see note*/
  j = 0;
  for (i=0; i<N; i++) {
    if (lastidx > (idx = rows[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++; 
        nprocs[2*j+1] = 1;
        owner[i] = j; 
#if defined(PETSC_DEBUG)
        found = PETSC_TRUE; 
#endif
        break;
      }
    }
#if defined(PETSC_DEBUG)
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index out of range");
    found = PETSC_FALSE;
#endif
  }
  nsends = 0;  for (i=0; i<size; i++) { nsends += nprocs[2*i+1];} 
  
  if (A->nooffproczerorows) {
    if (nsends > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"You called MatSetOption(,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE) but set an off process zero row");
    nrecvs = nsends;
    nmax   = N;
  } else {
    /* inform other processors of number of messages and max length*/
    ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);
  }
  
  /* post receives:   */
  ierr = PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(PetscInt),&rvalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPIU_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }
  
  /* do sends:
     1) starts[i] gives the starting index in svalues for stuff going to 
     the ith processor
  */
  ierr = PetscMalloc((N+1)*sizeof(PetscInt),&svalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(PetscInt),&starts);CHKERRQ(ierr);
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  for (i=0; i<N; i++) {
    svalues[starts[owner[i]]++] = rows[i];
  }
  
  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[2*i],MPIU_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  base = owners[rank];
  
  /*  wait on receives */
  ierr   = PetscMalloc2(nrecvs+1,PetscInt,&lens,nrecvs+1,PetscInt,&source);CHKERRQ(ierr);
  count  = nrecvs; 
  slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen          += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  /* move the data into the send scatter */
  ierr = PetscMalloc((slen+1)*sizeof(PetscInt),&lrows);CHKERRQ(ierr);
  count = 0;
  for (i=0; i<nrecvs; i++) {
    values = rvalues + i*nmax;
    for (j=0; j<lens[i]; j++) {
      lrows[count++] = values[j] - base;
    }
  }
  ierr = PetscFree(rvalues);CHKERRQ(ierr);
  ierr = PetscFree2(lens,source);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
    
  /* fix right hand side if needed */
  if (x && b) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    for (i=0; i<slen; i++) {
      bb[lrows[i]] = diag*xx[lrows[i]];
    }
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
  ierr = MatZeroRows_SeqBAIJ(l->B,slen,lrows,0.0,0,0);CHKERRQ(ierr); 
  if ((diag != 0.0) && (l->A->rmap->N == l->A->cmap->N)) {
    ierr = MatZeroRows_SeqBAIJ(l->A,slen,lrows,diag,0,0);CHKERRQ(ierr);
  } else if (diag != 0.0) {
    ierr = MatZeroRows_SeqBAIJ(l->A,slen,lrows,0.0,0,0);CHKERRQ(ierr);
    if (((Mat_SeqBAIJ*)l->A->data)->nonew) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatZeroRows() on rectangular matrices cannot be used with the Mat options \n\
       MAT_NEW_NONZERO_LOCATIONS,MAT_NEW_NONZERO_LOCATION_ERR,MAT_NEW_NONZERO_ALLOCATION_ERR");
    for (i=0; i<slen; i++) {
      row  = lrows[i] + rstart_bs;
      ierr = MatSetValues(A,1,&row,1,&row,&diag,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    ierr = MatZeroRows_SeqBAIJ(l->A,slen,lrows,0.0,0,0);CHKERRQ(ierr);
  }

  ierr = PetscFree(lrows);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(svalues);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUnfactored_MPIBAIJ"
PetscErrorCode MatSetUnfactored_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ    *a   = (Mat_MPIBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPIBAIJ(Mat,MatDuplicateOption,Mat *);

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_MPIBAIJ"
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
  ierr = MPI_Allreduce(&flg,flag,1,MPI_INT,MPI_LAND,((PetscObject)A)->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy_MPIBAIJ"
PetscErrorCode MatCopy_MPIBAIJ(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ *)A->data;
  Mat_MPIBAIJ    *b = (Mat_MPIBAIJ *)B->data;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if ((str != SAME_NONZERO_PATTERN) || (A->ops->copy != B->ops->copy)) {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  } else {
    ierr = MatCopy(a->A,b->A,str);CHKERRQ(ierr);
    ierr = MatCopy(a->B,b->B,str);CHKERRQ(ierr);  
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUp_MPIBAIJ"
PetscErrorCode MatSetUp_MPIBAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatMPIBAIJSetPreallocation(A,A->rmap->bs,PETSC_DEFAULT,0,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_MPIBAIJ"
PetscErrorCode MatAXPY_MPIBAIJ(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPIBAIJ    *xx=(Mat_MPIBAIJ *)X->data,*yy=(Mat_MPIBAIJ *)Y->data;
  PetscBLASInt   bnz,one=1;
  Mat_SeqBAIJ    *x,*y;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {  
    PetscScalar alpha = a;
    x = (Mat_SeqBAIJ *)xx->A->data;
    y = (Mat_SeqBAIJ *)yy->A->data;
    bnz = PetscBLASIntCast(x->nz);
    BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one);    
    x = (Mat_SeqBAIJ *)xx->B->data;
    y = (Mat_SeqBAIJ *)yy->B->data;
    bnz = PetscBLASIntCast(x->nz);
    BLASaxpy_(&bnz,&alpha,x->a,&one,y->a,&one);
  } else {
    ierr = MatAXPY_Basic(Y,a,X,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRealPart_MPIBAIJ"
PetscErrorCode MatRealPart_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ   *a = (Mat_MPIBAIJ*)A->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatRealPart(a->A);CHKERRQ(ierr);
  ierr = MatRealPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatImaginaryPart_MPIBAIJ"
PetscErrorCode MatImaginaryPart_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ   *a = (Mat_MPIBAIJ*)A->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = MatImaginaryPart(a->A);CHKERRQ(ierr);
  ierr = MatImaginaryPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_MPIBAIJ"
PetscErrorCode MatGetSubMatrix_MPIBAIJ(Mat mat,IS isrow,IS iscol,MatReuse call,Mat *newmat)
{
  PetscErrorCode ierr;
  IS             iscol_local;
  PetscInt       csize;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(iscol,&csize);CHKERRQ(ierr);
  if (call == MAT_REUSE_MATRIX) {
    ierr = PetscObjectQuery((PetscObject)*newmat,"ISAllGather",(PetscObject*)&iscol_local);CHKERRQ(ierr);
    if (!iscol_local) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Submatrix passed in was not used before, cannot reuse");
  } else {
    ierr = ISAllGather(iscol,&iscol_local);CHKERRQ(ierr);
  }
  ierr = MatGetSubMatrix_MPIBAIJ_Private(mat,isrow,iscol_local,csize,call,newmat);CHKERRQ(ierr);
  if (call == MAT_INITIAL_MATRIX) {
    ierr = PetscObjectCompose((PetscObject)*newmat,"ISAllGather",(PetscObject)iscol_local);CHKERRQ(ierr);
    ierr = ISDestroy(&iscol_local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_MPIBAIJ_Private"
/*
    Not great since it makes two copies of the submatrix, first an SeqBAIJ 
  in local and then by concatenating the local matrices the end result.
  Writing it directly would be much like MatGetSubMatrices_MPIBAIJ()
*/
PetscErrorCode MatGetSubMatrix_MPIBAIJ_Private(Mat mat,IS isrow,IS iscol,PetscInt csize,MatReuse call,Mat *newmat)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       i,m,n,rstart,row,rend,nz,*cwork,j,bs;
  PetscInt       *ii,*jj,nlocal,*dlens,*olens,dlen,olen,jend,mglobal;
  Mat            *local,M,Mreuse;
  MatScalar      *vwork,*aa;
  MPI_Comm       comm = ((PetscObject)mat)->comm;
  Mat_SeqBAIJ    *aij;


  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (call ==  MAT_REUSE_MATRIX) {
    ierr = PetscObjectQuery((PetscObject)*newmat,"SubMatrix",(PetscObject *)&Mreuse);CHKERRQ(ierr);
    if (!Mreuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Submatrix passed in was not used before, cannot reuse");
    local = &Mreuse;
    ierr  = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_REUSE_MATRIX,&local);CHKERRQ(ierr);
  } else {
    ierr   = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&local);CHKERRQ(ierr);
    Mreuse = *local;
    ierr   = PetscFree(local);CHKERRQ(ierr);
  }

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
    ierr   = MPI_Scan(&nlocal,&rend,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
    rstart = rend - nlocal;
    if (rank == size - 1 && rend != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local column sizes %D do not add up to total number of columns %D",rend,n);

    /* next, compute all the lengths */
    ierr  = PetscMalloc((2*m+1)*sizeof(PetscInt),&dlens);CHKERRQ(ierr);
    olens = dlens + m;
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
    ierr = PetscFree(dlens);CHKERRQ(ierr);
  } else {
    PetscInt ml,nl;

    M = *newmat;
    ierr = MatGetLocalSize(M,&ml,&nl);CHKERRQ(ierr);
    if (ml != m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Previous matrix must be same size/layout as request");
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
  aij = (Mat_SeqBAIJ*)(Mreuse)->data;
  ii  = aij->i;
  jj  = aij->j;
  aa  = aij->a;
  for (i=0; i<m; i++) {
    row   = rstart/bs + i;
    nz    = ii[i+1] - ii[i];
    cwork = jj;     jj += nz;
    vwork = aa;     aa += nz;
    ierr = MatSetValuesBlocked_MPIBAIJ(M,1,&row,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *newmat = M;

  /* save submatrix used in processor for next request */
  if (call ==  MAT_INITIAL_MATRIX) {
    ierr = PetscObjectCompose((PetscObject)M,"SubMatrix",(PetscObject)Mreuse);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)Mreuse);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPermute_MPIBAIJ"
PetscErrorCode MatPermute_MPIBAIJ(Mat A,IS rowp,IS colp,Mat *B)
{
  MPI_Comm       comm,pcomm;
  PetscInt       first,local_size,nrows;
  const PetscInt *rows;
  PetscMPIInt    size;
  IS             crowp,growp,irowp,lrowp,lcolp,icolp;
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
  /* collect the global row permutation and invert it */
  ierr = ISAllGather(crowp,&growp);CHKERRQ(ierr);
  ierr = ISSetPermutation(growp);CHKERRQ(ierr);
  if (pcomm!=comm) {
    ierr = ISDestroy(&crowp);CHKERRQ(ierr);
  }
  ierr = ISInvertPermutation(growp,PETSC_DECIDE,&irowp);CHKERRQ(ierr);
  /* get the local target indices */
  ierr = MatGetOwnershipRange(A,&first,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&local_size,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISGetIndices(irowp,&rows);CHKERRQ(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,local_size,rows+first,PETSC_COPY_VALUES,&lrowp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(irowp,&rows);CHKERRQ(ierr);
  ierr = ISDestroy(&irowp);CHKERRQ(ierr);
  /* the column permutation is so much easier;
     make a local version of 'colp' and invert it */
  ierr = PetscObjectGetComm((PetscObject)colp,&pcomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(pcomm,&size);CHKERRQ(ierr);
  if (size==1) {
    lcolp = colp;
  } else {
    ierr = ISGetSize(colp,&nrows);CHKERRQ(ierr);
    ierr = ISGetIndices(colp,&rows);CHKERRQ(ierr);
    ierr = ISCreateGeneral(MPI_COMM_SELF,nrows,rows,PETSC_COPY_VALUES,&lcolp);CHKERRQ(ierr);
  }
  ierr = ISSetPermutation(lcolp);CHKERRQ(ierr);
  ierr = ISInvertPermutation(lcolp,PETSC_DECIDE,&icolp);CHKERRQ(ierr);
  ierr = ISSetPermutation(icolp);CHKERRQ(ierr);
  if (size>1) {
    ierr = ISRestoreIndices(colp,&rows);CHKERRQ(ierr);
    ierr = ISDestroy(&lcolp);CHKERRQ(ierr);
  }
  /* now we just get the submatrix */
  ierr = MatGetSubMatrix_MPIBAIJ_Private(A,lrowp,icolp,local_size,MAT_INITIAL_MATRIX,B);CHKERRQ(ierr);
  /* clean up */
  ierr = ISDestroy(&lrowp);CHKERRQ(ierr);
  ierr = ISDestroy(&icolp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetGhosts_MPIBAIJ"
PetscErrorCode  MatGetGhosts_MPIBAIJ(Mat mat,PetscInt *nghosts,const PetscInt *ghosts[])
{
  Mat_MPIBAIJ    *baij = (Mat_MPIBAIJ*) mat->data;
  Mat_SeqBAIJ    *B = (Mat_SeqBAIJ*)baij->B->data;

  PetscFunctionBegin;
  if (nghosts) { *nghosts = B->nbs;}
  if (ghosts) {*ghosts = baij->garray;}
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatCreateColmap_MPIBAIJ_Private(Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringCreate_MPIBAIJ"
/*
    This routine is almost identical to MatFDColoringCreate_MPIBAIJ()!
*/
PetscErrorCode MatFDColoringCreate_MPIBAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  Mat_MPIBAIJ            *baij = (Mat_MPIBAIJ*)mat->data;
  PetscErrorCode        ierr;
  PetscMPIInt           size,*ncolsonproc,*disp,nn;
  PetscInt              bs,i,n,nrows,j,k,m,*rows = 0,*A_ci,*A_cj,ncols,col;
  const PetscInt        *is;
  PetscInt              nis = iscoloring->n,nctot,*cols,*B_ci,*B_cj;
  PetscInt              *rowhit,M,cstart,cend,colb;
  PetscInt              *columnsforrow,l;
  IS                    *isa;
  PetscBool              done,flg;
  ISLocalToGlobalMapping map = mat->cmap->bmapping;
  PetscInt               *ltog = (map ? map->indices : (PetscInt*) PETSC_NULL) ,ctype=c->ctype;

  PetscFunctionBegin;
  if (!mat->assembled) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be assembled first; MatAssemblyBegin/End();");
  if (ctype == IS_COLORING_GHOSTED && !map) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_ARG_INCOMP,"When using ghosted differencing matrix must have local to global mapping provided with MatSetLocalToGlobalMappingBlock");

  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&isa);CHKERRQ(ierr);
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  M                = mat->rmap->n/bs;
  cstart           = mat->cmap->rstart/bs;
  cend             = mat->cmap->rend/bs;
  c->M             = mat->rmap->N/bs;  /* set the global rows and columns and local rows */
  c->N             = mat->cmap->N/bs;
  c->m             = mat->rmap->n/bs;
  c->rstart        = mat->rmap->rstart/bs;

  c->ncolors       = nis;
  ierr             = PetscMalloc(nis*sizeof(PetscInt),&c->ncolumns);CHKERRQ(ierr);
  ierr             = PetscMalloc(nis*sizeof(PetscInt*),&c->columns);CHKERRQ(ierr); 
  ierr             = PetscMalloc(nis*sizeof(PetscInt),&c->nrows);CHKERRQ(ierr);
  ierr             = PetscMalloc(nis*sizeof(PetscInt*),&c->rows);CHKERRQ(ierr);
  ierr             = PetscMalloc(nis*sizeof(PetscInt*),&c->columnsforrow);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(c,5*nis*sizeof(PetscInt));CHKERRQ(ierr);

  /* Allow access to data structures of local part of matrix */
  if (!baij->colmap) {
    ierr = MatCreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
  }
  ierr = MatGetColumnIJ(baij->A,0,PETSC_FALSE,PETSC_FALSE,&ncols,&A_ci,&A_cj,&done);CHKERRQ(ierr); 
  ierr = MatGetColumnIJ(baij->B,0,PETSC_FALSE,PETSC_FALSE,&ncols,&B_ci,&B_cj,&done);CHKERRQ(ierr); 
  
  ierr = PetscMalloc((M+1)*sizeof(PetscInt),&rowhit);CHKERRQ(ierr);
  ierr = PetscMalloc((M+1)*sizeof(PetscInt),&columnsforrow);CHKERRQ(ierr);

  for (i=0; i<nis; i++) {
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);
    c->ncolumns[i] = n;
    if (n) {
      ierr = PetscMalloc(n*sizeof(PetscInt),&c->columns[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(c,n*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(c->columns[i],is,n*sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      c->columns[i]  = 0;
    }

    if (ctype == IS_COLORING_GLOBAL){
      /* Determine the total (parallel) number of columns of this color */
      ierr = MPI_Comm_size(((PetscObject)mat)->comm,&size);CHKERRQ(ierr); 
      ierr = PetscMalloc2(size,PetscMPIInt,&ncolsonproc,size,PetscMPIInt,&disp);CHKERRQ(ierr);

      nn   = PetscMPIIntCast(n);
      ierr = MPI_Allgather(&nn,1,MPI_INT,ncolsonproc,1,MPI_INT,((PetscObject)mat)->comm);CHKERRQ(ierr);
      nctot = 0; for (j=0; j<size; j++) {nctot += ncolsonproc[j];}
      if (!nctot) {
        ierr = PetscInfo(mat,"Coloring of matrix has some unneeded colors with no corresponding rows\n");CHKERRQ(ierr);
      }

      disp[0] = 0;
      for (j=1; j<size; j++) {
        disp[j] = disp[j-1] + ncolsonproc[j-1];
      }

      /* Get complete list of columns for color on each processor */
      ierr = PetscMalloc((nctot+1)*sizeof(PetscInt),&cols);CHKERRQ(ierr);
      ierr = MPI_Allgatherv((void*)is,n,MPIU_INT,cols,ncolsonproc,disp,MPIU_INT,((PetscObject)mat)->comm);CHKERRQ(ierr);
      ierr = PetscFree2(ncolsonproc,disp);CHKERRQ(ierr);
    } else if (ctype == IS_COLORING_GHOSTED){
      /* Determine local number of columns of this color on this process, including ghost points */
      nctot = n;
      ierr = PetscMalloc((nctot+1)*sizeof(PetscInt),&cols);CHKERRQ(ierr);
      ierr = PetscMemcpy(cols,is,n*sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not provided for this MatFDColoring type");
    }

    /*
       Mark all rows affect by these columns
    */
    /* Temporary option to allow for debugging/testing */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(PETSC_NULL,"-matfdcoloring_slow",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (!flg) {/*-----------------------------------------------------------------------------*/
      /* crude, fast version */
      ierr = PetscMemzero(rowhit,M*sizeof(PetscInt));CHKERRQ(ierr);
      /* loop over columns*/
      for (j=0; j<nctot; j++) {
        if (ctype == IS_COLORING_GHOSTED) {
          col = ltog[cols[j]];       
        } else {
          col  = cols[j];
        }
        if (col >= cstart && col < cend) {
          /* column is in diagonal block of matrix */
          rows = A_cj + A_ci[col-cstart]; 
          m    = A_ci[col-cstart+1] - A_ci[col-cstart];
        } else {
#if defined (PETSC_USE_CTABLE)
          ierr = PetscTableFind(baij->colmap,col+1,&colb);CHKERRQ(ierr);
	  colb --;
#else
          colb = baij->colmap[col] - 1;
#endif
          if (colb == -1) {
            m = 0; 
          } else {
            colb = colb/bs;
            rows = B_cj + B_ci[colb]; 
            m    = B_ci[colb+1] - B_ci[colb];
          }
        }
        /* loop over columns marking them in rowhit */
        for (k=0; k<m; k++) {
          rowhit[*rows++] = col + 1;
        }
      }

      /* count the number of hits */
      nrows = 0;
      for (j=0; j<M; j++) {
        if (rowhit[j]) nrows++;
      }
      c->nrows[i]         = nrows;
      ierr                = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->rows[i]);CHKERRQ(ierr);
      ierr                = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->columnsforrow[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(c,2*(nrows+1)*sizeof(PetscInt));CHKERRQ(ierr);
      nrows = 0;
      for (j=0; j<M; j++) {
        if (rowhit[j]) {
          c->rows[i][nrows]           = j;
          c->columnsforrow[i][nrows] = rowhit[j] - 1;
          nrows++;
        }
      }
    } else {/*-------------------------------------------------------------------------------*/
      /* slow version, using rowhit as a linked list */
      PetscInt currentcol,fm,mfm;
      rowhit[M] = M;
      nrows     = 0;
      /* loop over columns*/
      for (j=0; j<nctot; j++) {
        if (ctype == IS_COLORING_GHOSTED) {
          col = ltog[cols[j]];
        } else {
          col  = cols[j];
        }
        if (col >= cstart && col < cend) {
          /* column is in diagonal block of matrix */
          rows = A_cj + A_ci[col-cstart]; 
          m    = A_ci[col-cstart+1] - A_ci[col-cstart];
        } else {
#if defined (PETSC_USE_CTABLE)
          ierr = PetscTableFind(baij->colmap,col+1,&colb);CHKERRQ(ierr);
          colb --;
#else
          colb = baij->colmap[col] - 1;
#endif
          if (colb == -1) {
            m = 0; 
          } else {
            colb = colb/bs;
            rows = B_cj + B_ci[colb]; 
            m    = B_ci[colb+1] - B_ci[colb];
          }
        }

        /* loop over columns marking them in rowhit */
        fm    = M; /* fm points to first entry in linked list */
        for (k=0; k<m; k++) {
          currentcol = *rows++;
	  /* is it already in the list? */
          do {
            mfm  = fm;
            fm   = rowhit[fm];
          } while (fm < currentcol);
          /* not in list so add it */
          if (fm != currentcol) {
            nrows++;
            columnsforrow[currentcol] = col;
            /* next three lines insert new entry into linked list */
            rowhit[mfm]               = currentcol;
            rowhit[currentcol]        = fm;
            fm                        = currentcol; 
            /* fm points to present position in list since we know the columns are sorted */
          } else {
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid coloring of matrix detected");
          }
        }
      }
      c->nrows[i]         = nrows;
      ierr = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->rows[i]);CHKERRQ(ierr);
      ierr = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->columnsforrow[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(c,(nrows+1)*sizeof(PetscInt));CHKERRQ(ierr);
      /* now store the linked list of rows into c->rows[i] */
      nrows = 0;
      fm    = rowhit[M];
      do {
        c->rows[i][nrows]            = fm;
        c->columnsforrow[i][nrows++] = columnsforrow[fm];
        fm                           = rowhit[fm];
      } while (fm < M);
    } /* ---------------------------------------------------------------------------------------*/
    ierr = PetscFree(cols);CHKERRQ(ierr);
  }

  /* Optimize by adding the vscale, and scaleforrow[][] fields */
  /*
       vscale will contain the "diagonal" on processor scalings followed by the off processor
  */
  if (ctype == IS_COLORING_GLOBAL) {
    PetscInt *garray;
    ierr = PetscMalloc(baij->B->cmap->n*sizeof(PetscInt),&garray);CHKERRQ(ierr);
    for (i=0; i<baij->B->cmap->n/bs; i++) {
      for (j=0; j<bs; j++) {
        garray[i*bs+j] = bs*baij->garray[i]+j;
      }
    }
    ierr = VecCreateGhost(((PetscObject)mat)->comm,baij->A->rmap->n,PETSC_DETERMINE,baij->B->cmap->n,garray,&c->vscale);CHKERRQ(ierr);
    ierr = PetscFree(garray);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = PetscMalloc(c->ncolors*sizeof(PetscInt*),&c->vscaleforrow);CHKERRQ(ierr);
    for (k=0; k<c->ncolors; k++) { 
      ierr = PetscMalloc((c->nrows[k]+1)*sizeof(PetscInt),&c->vscaleforrow[k]);CHKERRQ(ierr);
      for (l=0; l<c->nrows[k]; l++) {
        col = c->columnsforrow[k][l];
        if (col >= cstart && col < cend) {
          /* column is in diagonal block of matrix */
          colb = col - cstart;
        } else {
          /* column  is in "off-processor" part */
#if defined (PETSC_USE_CTABLE)
          ierr = PetscTableFind(baij->colmap,col+1,&colb);CHKERRQ(ierr);
          colb --;
#else
          colb = baij->colmap[col] - 1;
#endif
          colb = colb/bs;
          colb += cend - cstart;
        }
        c->vscaleforrow[k][l] = colb;
      }
    } 
  } else if (ctype == IS_COLORING_GHOSTED) {
    /* Get gtol mapping */
    PetscInt N = mat->cmap->N, *gtol;
    ierr = PetscMalloc((N+1)*sizeof(PetscInt),&gtol);CHKERRQ(ierr);
    for (i=0; i<N; i++) gtol[i] = -1;
    for (i=0; i<map->n; i++) gtol[ltog[i]] = i;
   
    c->vscale = 0; /* will be created in MatFDColoringApply() */
    ierr = PetscMalloc(c->ncolors*sizeof(PetscInt*),&c->vscaleforrow);CHKERRQ(ierr);
    for (k=0; k<c->ncolors; k++) { 
      ierr = PetscMalloc((c->nrows[k]+1)*sizeof(PetscInt),&c->vscaleforrow[k]);CHKERRQ(ierr);
      for (l=0; l<c->nrows[k]; l++) {
        col = c->columnsforrow[k][l];      /* global column index */
        c->vscaleforrow[k][l] = gtol[col]; /* local column index */
      }
    }
    ierr = PetscFree(gtol);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&isa);CHKERRQ(ierr);

  ierr = PetscFree(rowhit);CHKERRQ(ierr);
  ierr = PetscFree(columnsforrow);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(baij->A,0,PETSC_FALSE,PETSC_FALSE,&ncols,&A_ci,&A_cj,&done);CHKERRQ(ierr); 
  ierr = MatRestoreColumnIJ(baij->B,0,PETSC_FALSE,PETSC_FALSE,&ncols,&B_ci,&B_cj,&done);CHKERRQ(ierr); 
    CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSeqNonzeroStructure_MPIBAIJ"
PetscErrorCode MatGetSeqNonzeroStructure_MPIBAIJ(Mat A,Mat *newmat)
{
  Mat            B;
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ *)A->data;
  Mat_SeqBAIJ    *ad = (Mat_SeqBAIJ*)a->A->data,*bd = (Mat_SeqBAIJ*)a->B->data;
  Mat_SeqAIJ     *b;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,*recvcounts = 0,*displs = 0;
  PetscInt       sendcount,i,*rstarts = A->rmap->range,n,cnt,j,bs = A->rmap->bs;
  PetscInt       m,*garray = a->garray,*lens,*jsendbuf,*a_jsendbuf,*b_jsendbuf;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)A)->comm,&rank);CHKERRQ(ierr);

  /* ----------------------------------------------------------------
     Tell every processor the number of nonzeros per row
  */
  ierr = PetscMalloc((A->rmap->N/bs)*sizeof(PetscInt),&lens);CHKERRQ(ierr);
  for (i=A->rmap->rstart/bs; i<A->rmap->rend/bs; i++) {
    lens[i] = ad->i[i-A->rmap->rstart/bs+1] - ad->i[i-A->rmap->rstart/bs] + bd->i[i-A->rmap->rstart/bs+1] - bd->i[i-A->rmap->rstart/bs];
  }
  sendcount = A->rmap->rend/bs - A->rmap->rstart/bs;
  ierr = PetscMalloc(2*size*sizeof(PetscMPIInt),&recvcounts);CHKERRQ(ierr);
  displs     = recvcounts + size;
  for (i=0; i<size; i++) {
    recvcounts[i] = A->rmap->range[i+1]/bs - A->rmap->range[i]/bs;
    displs[i]     = A->rmap->range[i]/bs;
  }
#if defined(PETSC_HAVE_MPI_IN_PLACE)
  ierr  = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,lens,recvcounts,displs,MPIU_INT,((PetscObject)A)->comm);CHKERRQ(ierr);
#else
  ierr  = MPI_Allgatherv(lens+A->rmap->rstart/bs,sendcount,MPIU_INT,lens,recvcounts,displs,MPIU_INT,((PetscObject)A)->comm);CHKERRQ(ierr);
#endif
  /* ---------------------------------------------------------------
     Create the sequential matrix of the same type as the local block diagonal
  */
  ierr  = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr  = MatSetSizes(B,A->rmap->N/bs,A->cmap->N/bs,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr  = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  ierr  = MatSeqAIJSetPreallocation(B,0,lens);CHKERRQ(ierr);
  b = (Mat_SeqAIJ *)B->data;

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
  if (cnt != sendcount) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Corrupted PETSc matrix: nz given %D actual nz %D",sendcount,cnt);

  /*--------------------------------------------------------------------
    Gather all column indices to all processors
  */
  for (i=0; i<size; i++) {
    recvcounts[i] = 0;
    for (j=A->rmap->range[i]/bs; j<A->rmap->range[i+1]/bs; j++) {
      recvcounts[i] += lens[j];
    }
  }
  displs[0]  = 0;
  for (i=1; i<size; i++) {
    displs[i] = displs[i-1] + recvcounts[i-1];
  }
#if defined(PETSC_HAVE_MPI_IN_PLACE)
  ierr = MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,b->j,recvcounts,displs,MPIU_INT,((PetscObject)A)->comm);CHKERRQ(ierr);
#else
  ierr = MPI_Allgatherv(jsendbuf,sendcount,MPIU_INT,b->j,recvcounts,displs,MPIU_INT,((PetscObject)A)->comm);CHKERRQ(ierr);
#endif
  /*--------------------------------------------------------------------
    Assemble the matrix into useable form (note numerical values not yet set)
  */
  /* set the b->ilen (length of each row) values */
  ierr = PetscMemcpy(b->ilen,lens,(A->rmap->N/bs)*sizeof(PetscInt));CHKERRQ(ierr);
  /* set the b->i indices */
  b->i[0] = 0;
  for (i=1; i<=A->rmap->N/bs; i++) {
    b->i[i] = b->i[i-1] + lens[i-1];
  }
  ierr = PetscFree(lens);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(recvcounts);CHKERRQ(ierr);

  if (A->symmetric){
    ierr = MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  } else if (A->hermitian) {
    ierr = MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  } else if (A->structurally_symmetric) {
    ierr = MatSetOption(B,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  }
  *newmat = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSOR_MPIBAIJ"
PetscErrorCode MatSOR_MPIBAIJ(Mat matin,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_MPIBAIJ    *mat = (Mat_MPIBAIJ*)matin->data;
  PetscErrorCode ierr; 
  Vec            bb1 = 0;

  PetscFunctionBegin;
  if (its > 1 || ~flag & SOR_ZERO_INITIAL_GUESS) {
    ierr = VecDuplicate(bb,&bb1);CHKERRQ(ierr);
  }

  if (flag == SOR_APPLY_UPPER) {
    ierr = (*mat->A->ops->sor)(mat->A,bb,omega,flag,fshift,lits,1,xx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP){
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
  } else if (flag & SOR_LOCAL_FORWARD_SWEEP){
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
  } else if (flag & SOR_LOCAL_BACKWARD_SWEEP){
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
  } else SETERRQ(((PetscObject)matin)->comm,PETSC_ERR_SUP,"Parallel version of SOR requested not supported");

  ierr = VecDestroy(&bb1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

extern PetscErrorCode  MatFDColoringApply_BAIJ(Mat,MatFDColoring,Vec,MatStructure*,void*);

#undef __FUNCT__  
#define __FUNCT__ "MatInvertBlockDiagonal_MPIBAIJ"
PetscErrorCode  MatInvertBlockDiagonal_MPIBAIJ(Mat A,PetscScalar **values)
{
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatInvertBlockDiagonal(a->A,values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {
       MatSetValues_MPIBAIJ,
       MatGetRow_MPIBAIJ,
       MatRestoreRow_MPIBAIJ,
       MatMult_MPIBAIJ,
/* 4*/ MatMultAdd_MPIBAIJ,
       MatMultTranspose_MPIBAIJ,
       MatMultTransposeAdd_MPIBAIJ,
       0,
       0,
       0,
/*10*/ 0,
       0,
       0,
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
       0,
       0,
       0,
       0,
/*29*/ MatSetUp_MPIBAIJ,
       0,
       0,
       0,
       0,
/*34*/ MatDuplicate_MPIBAIJ,
       0,
       0,
       0,
       0,
/*39*/ MatAXPY_MPIBAIJ,
       MatGetSubMatrices_MPIBAIJ,
       MatIncreaseOverlap_MPIBAIJ,
       MatGetValues_MPIBAIJ,
       MatCopy_MPIBAIJ,
/*44*/ 0,
       MatScale_MPIBAIJ,
       0,
       0,
       0,
/*49*/ 0,
       0,
       0,
       0,
       0,
/*54*/ MatFDColoringCreate_MPIBAIJ,
       0,
       MatSetUnfactored_MPIBAIJ,
       MatPermute_MPIBAIJ,
       MatSetValuesBlocked_MPIBAIJ,
/*59*/ MatGetSubMatrix_MPIBAIJ,
       MatDestroy_MPIBAIJ,
       MatView_MPIBAIJ,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ MatGetRowMaxAbs_MPIBAIJ,
       0,
       0,
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
       MatLoad_MPIBAIJ,
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
       MatRealPart_MPIBAIJ,
       MatImaginaryPart_MPIBAIJ,
       0,
       0,
/*109*/0,
       0,
       0,
       0,
       0,
/*114*/MatGetSeqNonzeroStructure_MPIBAIJ,
       0,
       MatGetGhosts_MPIBAIJ,
       0,
       0,
/*119*/0,
       0,
       0,
       0,
       0,
/*124*/0,
       0,
       MatInvertBlockDiagonal_MPIBAIJ
};

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonalBlock_MPIBAIJ"
PetscErrorCode  MatGetDiagonalBlock_MPIBAIJ(Mat A,Mat *a)
{
  PetscFunctionBegin;
  *a = ((Mat_MPIBAIJ *)A->data)->A;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode  MatConvert_MPIBAIJ_MPISBAIJ(Mat, MatType,MatReuse,Mat*);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMPIBAIJSetPreallocationCSR_MPIBAIJ"
PetscErrorCode MatMPIBAIJSetPreallocationCSR_MPIBAIJ(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[],const PetscScalar V[])
{
  PetscInt       m,rstart,cstart,cend;
  PetscInt       i,j,d,nz,nz_max=0,*d_nnz=0,*o_nnz=0;
  const PetscInt *JJ=0;
  PetscScalar    *values=0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
  ierr = MatMPIBAIJSetPreallocation(B,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
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
    ierr = MatSetValuesBlocked_MPIBAIJ(B,1,&row,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
  }

  if (!V) { ierr = PetscFree(values);CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMPIBAIJSetPreallocationCSR"
/*@C
   MatMPIBAIJSetPreallocationCSR - Allocates memory for a sparse parallel matrix in BAIJ format
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

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatMPIBAIJSetPreallocation_MPIBAIJ"
PetscErrorCode  MatMPIBAIJSetPreallocation_MPIBAIJ(Mat B,PetscInt bs,PetscInt d_nz,PetscInt *d_nnz,PetscInt o_nz,PetscInt *o_nnz)
{
  Mat_MPIBAIJ    *b;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      d_realalloc = PETSC_FALSE,o_realalloc = PETSC_FALSE;

  PetscFunctionBegin;
  if (d_nz >= 0 || d_nnz) d_realalloc = PETSC_TRUE;
  if (o_nz >= 0 || o_nnz) o_realalloc = PETSC_TRUE;

  if (d_nz == PETSC_DEFAULT || d_nz == PETSC_DECIDE) d_nz = 5;
  if (o_nz == PETSC_DEFAULT || o_nz == PETSC_DECIDE) o_nz = 2;
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

  b = (Mat_MPIBAIJ*)B->data;
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

  if (!B->preallocated) {
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQBAIJ);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(B,b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQBAIJ);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(B,b->B);CHKERRQ(ierr);
    ierr = MatStashCreate_Private(((PetscObject)B)->comm,bs,&B->bstash);CHKERRQ(ierr);
  }

  ierr = MatSeqBAIJSetPreallocation(b->A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(b->B,bs,o_nz,o_nnz);CHKERRQ(ierr);
  /* Do not error if the user did not give real preallocation information. Ugly because this would overwrite a previous user call to MatSetOption(). */
  if (!d_realalloc) {ierr = MatSetOption(b->A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);}
  if (!o_realalloc) {ierr = MatSetOption(b->B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);}
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode  MatDiagonalScaleLocal_MPIBAIJ(Mat,Vec);
extern PetscErrorCode  MatSetHashTableFactor_MPIBAIJ(Mat,PetscReal);
EXTERN_C_END


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatConvert_MPIBAIJ_MPIAdj"
PetscErrorCode  MatConvert_MPIBAIJ_MPIAdj(Mat B, const MatType newtype,MatReuse reuse,Mat *adj)
{
  Mat_MPIBAIJ    *b = (Mat_MPIBAIJ*)B->data;
  PetscErrorCode ierr;
  Mat_SeqBAIJ    *d = (Mat_SeqBAIJ*) b->A->data,*o = (Mat_SeqBAIJ*) b->B->data;
  PetscInt       M = B->rmap->n/B->rmap->bs,i,*ii,*jj,cnt,j,k,rstart = B->rmap->rstart/B->rmap->bs;
  const PetscInt *id = d->i, *jd = d->j, *io = o->i, *jo = o->j, *garray = b->garray;

  PetscFunctionBegin;
  ierr = PetscMalloc((M+1)*sizeof(PetscInt),&ii);CHKERRQ(ierr);
  ii[0] = 0;
  CHKMEMQ;
  for (i=0; i<M; i++) {
    if ((id[i+1] - id[i]) < 0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Indices wrong %D %D %D",i,id[i],id[i+1]);
    if ((io[i+1] - io[i]) < 0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Indices wrong %D %D %D",i,io[i],io[i+1]);
    ii[i+1] = ii[i] + id[i+1] - id[i] + io[i+1] - io[i];
    /* remove one from count of matrix has diagonal */
    for (j=id[i]; j<id[i+1]; j++) {   
      if (jd[j] == i) {ii[i+1]--;break;}
    }
  CHKMEMQ;
  }
  ierr = PetscMalloc(ii[M]*sizeof(PetscInt),&jj);CHKERRQ(ierr);
  cnt = 0;
  for (i=0; i<M; i++) {
    for (j=io[i]; j<io[i+1]; j++) {
      if (garray[jo[j]] > rstart) break;
      jj[cnt++] = garray[jo[j]];
  CHKMEMQ;
    }
    for (k=id[i]; k<id[i+1]; k++) {
      if (jd[k] != i) {
        jj[cnt++] = rstart + jd[k];
  CHKMEMQ;
      }
    }
    for (;j<io[i+1]; j++) {
      jj[cnt++] = garray[jo[j]];
  CHKMEMQ;
    }
  }
  ierr = MatCreateMPIAdj(((PetscObject)B)->comm,M,B->cmap->N/B->rmap->bs,ii,jj,PETSC_NULL,adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#include <../src/mat/impls/aij/mpi/mpiaij.h>
EXTERN_C_BEGIN
PetscErrorCode  MatConvert_SeqBAIJ_SeqAIJ(Mat,const MatType,MatReuse,Mat*);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__ 
#define __FUNCT__ "MatConvert_MPIBAIJ_MPIAIJ"
PetscErrorCode  MatConvert_MPIBAIJ_MPIAIJ(Mat A,const MatType newtype,MatReuse reuse,Mat *newmat)
{ 
  PetscErrorCode ierr;
  Mat_MPIBAIJ    *a = (Mat_MPIBAIJ*)A->data;
  Mat            B;
  Mat_MPIAIJ     *b;

  PetscFunctionBegin;
  if (!A->assembled) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Matrix must be assembled");

  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  b = (Mat_MPIAIJ*) B->data; 

  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = MatDestroy(&b->B);CHKERRQ(ierr);
  ierr = MatDisAssemble_MPIBAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_SeqBAIJ_SeqAIJ(a->A, MATSEQAIJ, MAT_INITIAL_MATRIX, &b->A);CHKERRQ(ierr);
  ierr = MatConvert_SeqBAIJ_SeqAIJ(a->B, MATSEQAIJ, MAT_INITIAL_MATRIX, &b->B);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  } else {
   *newmat = B;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
 
EXTERN_C_BEGIN
#if defined(PETSC_HAVE_MUMPS)
extern PetscErrorCode MatGetFactor_baij_mumps(Mat,MatFactorType,Mat*);
#endif
EXTERN_C_END

/*MC
   MATMPIBAIJ - MATMPIBAIJ = "mpibaij" - A matrix type to be used for distributed block sparse matrices.

   Options Database Keys:
+ -mat_type mpibaij - sets the matrix type to "mpibaij" during a call to MatSetFromOptions()
. -mat_block_size <bs> - set the blocksize used to store the matrix
- -mat_use_hash_table <fact>

  Level: beginner

.seealso: MatCreateMPIBAIJ
M*/

EXTERN_C_BEGIN
extern PetscErrorCode MatConvert_MPIBAIJ_MPIBSTRM(Mat,const MatType,MatReuse,Mat*);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIBAIJ"
PetscErrorCode  MatCreate_MPIBAIJ(Mat B)
{
  Mat_MPIBAIJ    *b;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscNewLog(B,Mat_MPIBAIJ,&b);CHKERRQ(ierr);
  B->data = (void*)b;

  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->assembled  = PETSC_FALSE;

  B->insertmode = NOT_SET_VALUES;
  ierr = MPI_Comm_rank(((PetscObject)B)->comm,&b->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)B)->comm,&b->size);CHKERRQ(ierr);

  /* build local table of row and column ownerships */
  ierr = PetscMalloc((b->size+1)*sizeof(PetscInt),&b->rangebs);CHKERRQ(ierr);

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

  ierr = PetscOptionsBegin(((PetscObject)B)->comm,PETSC_NULL,"Options for loading MPIBAIJ matrix 1","Mat");CHKERRQ(ierr);
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

#if defined(PETSC_HAVE_MUMPS)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_mumps_C", "MatGetFactor_baij_mumps",MatGetFactor_baij_mumps);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpibaij_mpiadj_C",
                                     "MatConvert_MPIBAIJ_MPIAdj",
                                      MatConvert_MPIBAIJ_MPIAdj);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpibaij_mpiaij_C",
                                     "MatConvert_MPIBAIJ_MPIAIJ",
                                      MatConvert_MPIBAIJ_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpibaij_mpisbaij_C",
                                     "MatConvert_MPIBAIJ_MPISBAIJ",
                                      MatConvert_MPIBAIJ_MPISBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatStoreValues_C",
                                     "MatStoreValues_MPIBAIJ",
                                     MatStoreValues_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatRetrieveValues_C",
                                     "MatRetrieveValues_MPIBAIJ",
                                     MatRetrieveValues_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetDiagonalBlock_C",
                                     "MatGetDiagonalBlock_MPIBAIJ",
                                     MatGetDiagonalBlock_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIBAIJSetPreallocation_C",
                                     "MatMPIBAIJSetPreallocation_MPIBAIJ",
                                     MatMPIBAIJSetPreallocation_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIBAIJSetPreallocationCSR_C",
				     "MatMPIBAIJSetPreallocationCSR_MPIBAIJ",
				     MatMPIBAIJSetPreallocationCSR_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatDiagonalScaleLocal_C",
                                     "MatDiagonalScaleLocal_MPIBAIJ",
                                     MatDiagonalScaleLocal_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSetHashTableFactor_C",
                                     "MatSetHashTableFactor_MPIBAIJ",
                                     MatSetHashTableFactor_MPIBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpibaij_mpibstrm_C",
                                     "MatConvert_MPIBAIJ_MPIBSTRM",
                                      MatConvert_MPIBAIJ_MPIBSTRM);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIBAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATBAIJ - MATBAIJ = "baij" - A matrix type to be used for block sparse matrices.

   This matrix type is identical to MATSEQBAIJ when constructed with a single process communicator,
   and MATMPIBAIJ otherwise.

   Options Database Keys:
. -mat_type baij - sets the matrix type to "baij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateBAIJ(),MATSEQBAIJ,MATMPIBAIJ, MatMPIBAIJSetPreallocation(), MatMPIBAIJSetPreallocationCSR()
M*/

#undef __FUNCT__  
#define __FUNCT__ "MatMPIBAIJSetPreallocation"
/*@C
   MatMPIBAIJSetPreallocation - Allocates memory for a sparse parallel matrix in block AIJ format
   (block compressed row).  For good matrix assembly performance
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
           row) or PETSC_NULL.  If you plan to factor the matrix you must leave room for the diagonal entry and 
           set it even if it is zero.
.  o_nz  - number of block nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or PETSC_NULL.

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

   You can call MatGetInfo() to get information on how effective the preallocation was;
   for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
   You can also run with the option -info and look for messages with the string 
   malloc in them to see if additional memory allocation was needed.

   Level: intermediate

.keywords: matrix, block, aij, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqBAIJ(), MatSetValues(), MatCreateBAIJ(), MatMPIBAIJSetPreallocationCSR()
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

#undef __FUNCT__  
#define __FUNCT__ "MatCreateBAIJ"
/*@C
   MatCreateBAIJ - Creates a sparse parallel matrix in block AIJ format
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
.  d_nz  - number of nonzero blocks per block row in diagonal portion of local 
           submatrix  (same for all local rows)
.  d_nnz - array containing the number of nonzero blocks in the various block rows 
           of the in diagonal portion of the local (possibly different for each block
           row) or PETSC_NULL.  If you plan to factor the matrix you must leave room for the diagonal entry 
           and set it even if it is zero.
.  o_nz  - number of nonzero blocks per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzero blocks in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each block row) or PETSC_NULL.

   Output Parameter:
.  A - the matrix 

   Options Database Keys:
+   -mat_block_size - size of the blocks to use
-   -mat_use_hash_table <fact>

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly. 
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

.seealso: MatCreate(), MatCreateSeqBAIJ(), MatSetValues(), MatCreateBAIJ(), MatMPIBAIJSetPreallocation(), MatMPIBAIJSetPreallocationCSR()
@*/
PetscErrorCode  MatCreateBAIJ(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIBAIJ);CHKERRQ(ierr);
    ierr = MatMPIBAIJSetPreallocation(*A,bs,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQBAIJ);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetPreallocation(*A,bs,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_MPIBAIJ"
static PetscErrorCode MatDuplicate_MPIBAIJ(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPIBAIJ    *a,*oldmat = (Mat_MPIBAIJ*)matin->data;
  PetscErrorCode ierr;
  PetscInt       len=0;

  PetscFunctionBegin;
  *newmat       = 0;
  ierr = MatCreate(((PetscObject)matin)->comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,matin->rmap->n,matin->cmap->n,matin->rmap->N,matin->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(mat,((PetscObject)matin)->type_name);CHKERRQ(ierr);
  ierr = PetscMemcpy(mat->ops,matin->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  mat->factortype   = matin->factortype;
  mat->preallocated = PETSC_TRUE;
  mat->assembled    = PETSC_TRUE;
  mat->insertmode   = NOT_SET_VALUES;

  a      = (Mat_MPIBAIJ*)mat->data;
  mat->rmap->bs  = matin->rmap->bs;
  a->bs2   = oldmat->bs2;
  a->mbs   = oldmat->mbs;
  a->nbs   = oldmat->nbs;
  a->Mbs   = oldmat->Mbs;
  a->Nbs   = oldmat->Nbs;
  
  ierr = PetscLayoutReference(matin->rmap,&mat->rmap);CHKERRQ(ierr);  
  ierr = PetscLayoutReference(matin->cmap,&mat->cmap);CHKERRQ(ierr);  

  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  a->donotstash   = oldmat->donotstash;
  a->roworiented  = oldmat->roworiented;
  a->rowindices   = 0;
  a->rowvalues    = 0;
  a->getrowactive = PETSC_FALSE;
  a->barray       = 0;
  a->rstartbs     = oldmat->rstartbs;
  a->rendbs       = oldmat->rendbs;
  a->cstartbs     = oldmat->cstartbs;
  a->cendbs       = oldmat->cendbs;

  /* hash table stuff */
  a->ht           = 0;
  a->hd           = 0;
  a->ht_size      = 0;
  a->ht_flag      = oldmat->ht_flag;
  a->ht_fact      = oldmat->ht_fact;
  a->ht_total_ct  = 0;
  a->ht_insert_ct = 0;

  ierr = PetscMemcpy(a->rangebs,oldmat->rangebs,(a->size+1)*sizeof(PetscInt));CHKERRQ(ierr);
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

  ierr = MatDuplicate(oldmat->A,cpvalues,&a->A);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->A);CHKERRQ(ierr);
  ierr = MatDuplicate(oldmat->B,cpvalues,&a->B);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mat,a->B);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)matin)->qlist,&((PetscObject)mat)->qlist);CHKERRQ(ierr);
  *newmat = mat;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPIBAIJ"
PetscErrorCode MatLoad_MPIBAIJ(Mat newmat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  int            fd;
  PetscInt       i,nz,j,rstart,rend;
  PetscScalar    *vals,*buf;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;
  MPI_Status     status;
  PetscMPIInt    rank,size,maxnz;
  PetscInt       header[4],*rowlengths = 0,M,N,m,*rowners,*cols;
  PetscInt       *locrowlens = PETSC_NULL,*procsnz = PETSC_NULL,*browners = PETSC_NULL;
  PetscInt       jj,*mycols,*ibuf,bs=1,Mbs,mbs,extra_rows,mmax;
  PetscMPIInt    tag = ((PetscObject)viewer)->tag;
  PetscInt       *dlens = PETSC_NULL,*odlens = PETSC_NULL,*mask = PETSC_NULL,*masked1 = PETSC_NULL,*masked2 = PETSC_NULL,rowcount,odcount;
  PetscInt       dcount,kmax,k,nzcount,tmp,mend,sizesset=1,grows,gcols;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(comm,PETSC_NULL,"Options for loading MPIBAIJ matrix 2","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
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

  if (M != N) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Can only do square matrices");

  /* 
     This code adds extra rows to make sure the number of rows is 
     divisible by the blocksize
  */
  Mbs        = M/bs;
  extra_rows = bs - M + bs*Mbs;
  if (extra_rows == bs) extra_rows = 0;
  else                  Mbs++;
  if (extra_rows && !rank) {
    ierr = PetscInfo(viewer,"Padding loaded matrix to match blocksize\n");CHKERRQ(ierr);
  }

  /* determine ownership of all rows */
  if (newmat->rmap->n < 0) { /* PETSC_DECIDE */
    mbs        = Mbs/size + ((Mbs % size) > rank);
    m          = mbs*bs;
  } else { /* User set */
    m          = newmat->rmap->n;
    mbs        = m/bs;
  }
  ierr       = PetscMalloc2(size+1,PetscInt,&rowners,size+1,PetscInt,&browners);CHKERRQ(ierr);
  ierr       = MPI_Allgather(&mbs,1,MPIU_INT,rowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);

  /* process 0 needs enough room for process with most rows */
  if (!rank) {
    mmax = rowners[1];
    for (i=2; i<size; i++) {
      mmax = PetscMax(mmax,rowners[i]);
    }
    mmax*=bs;
  } else mmax = m;

  rowners[0] = 0;
  for (i=2; i<=size; i++)  rowners[i] += rowners[i-1];
  for (i=0; i<=size;  i++) browners[i] = rowners[i]*bs;
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ierr = PetscMalloc((mmax+1)*sizeof(PetscInt),&locrowlens);CHKERRQ(ierr);
  if (!rank) {
    mend = m;
    if (size == 1) mend = mend - extra_rows;
    ierr = PetscBinaryRead(fd,locrowlens,mend,PETSC_INT);CHKERRQ(ierr);
    for (j=mend; j<m; j++) locrowlens[j] = 1;
    ierr = PetscMalloc(m*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
    ierr = PetscMalloc(size*sizeof(PetscInt),&procsnz);CHKERRQ(ierr);
    ierr = PetscMemzero(procsnz,size*sizeof(PetscInt));CHKERRQ(ierr);
    for (j=0; j<m; j++) {
      procsnz[0] += locrowlens[j];
    }
    for (i=1; i<size; i++) {
      mend = browners[i+1] - browners[i];
      if (i == size-1) mend = mend - extra_rows;
      ierr = PetscBinaryRead(fd,rowlengths,mend,PETSC_INT);CHKERRQ(ierr);
      for (j=mend; j<browners[i+1] - browners[i]; j++) rowlengths[j] = 1;
      /* calculate the number of nonzeros on each processor */
      for (j=0; j<browners[i+1]-browners[i]; j++) {
        procsnz[i] += rowlengths[j];
      }
      ierr = MPI_Send(rowlengths,browners[i+1]-browners[i],MPIU_INT,i,tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  } else {
    ierr = MPI_Recv(locrowlens,m,MPIU_INT,0,tag,comm,&status);CHKERRQ(ierr);
  }

  if (!rank) {
    /* determine max buffer needed and allocate it */
    maxnz = procsnz[0];
    for (i=1; i<size; i++) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    ierr = PetscMalloc(maxnz*sizeof(PetscInt),&cols);CHKERRQ(ierr);

    /* read in my part of the matrix column indices  */
    nz     = procsnz[0];
    ierr   = PetscMalloc((nz+1)*sizeof(PetscInt),&ibuf);CHKERRQ(ierr);
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
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for (i=0; i<m; i++) {
      nz += locrowlens[i];
    }
    ierr   = PetscMalloc((nz+1)*sizeof(PetscInt),&ibuf);CHKERRQ(ierr);
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
  rowcount = 0; nzcount = 0;
  for (i=0; i<mbs; i++) {
    dcount  = 0;
    odcount = 0;
    for (j=0; j<bs; j++) {
      kmax = locrowlens[rowcount];
      for (k=0; k<kmax; k++) {
        tmp = mycols[nzcount++]/bs;
        if (!mask[tmp]) {
          mask[tmp] = 1;
          if (tmp < rstart || tmp >= rend) masked2[odcount++] = tmp;
          else masked1[dcount++] = tmp;
        }
      }
      rowcount++;
    }
  
    dlens[i]  = dcount;
    odlens[i] = odcount;

    /* zero out the mask elements we set */
    for (j=0; j<dcount; j++) mask[masked1[j]] = 0;
    for (j=0; j<odcount; j++) mask[masked2[j]] = 0; 
  }

 
  if (!sizesset) {
    ierr = MatSetSizes(newmat,m,m,M+extra_rows,N+extra_rows);CHKERRQ(ierr);
  }
  ierr = MatMPIBAIJSetPreallocation(newmat,bs,0,dlens,0,odlens);CHKERRQ(ierr);

  if (!rank) {
    ierr = PetscMalloc((maxnz+1)*sizeof(PetscScalar),&buf);CHKERRQ(ierr);
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
      ierr = MatSetValues_MPIBAIJ(newmat,1,&jj,locrowlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr);
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
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&buf);CHKERRQ(ierr);

    /* receive message of values*/
    vals   = buf;
    mycols = ibuf;
    ierr   = MPI_Recv(vals,nz,MPIU_SCALAR,0,((PetscObject)newmat)->tag,comm,&status);CHKERRQ(ierr);
    ierr   = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"something is wrong with file");

    /* insert into matrix */
    jj      = rstart*bs;
    for (i=0; i<m; i++) {
      ierr    = MatSetValues_MPIBAIJ(newmat,1,&jj,locrowlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr);
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
#define __FUNCT__ "MatMPIBAIJSetHashTableFactor"
/*@
   MatMPIBAIJSetHashTableFactor - Sets the factor required to compute the size of the HashTable.

   Input Parameters:
.  mat  - the matrix
.  fact - factor

   Not Collective, each process can use a different factor

   Level: advanced

  Notes:
   This can also be set by the command line option: -mat_use_hash_table <fact>

.keywords: matrix, hashtable, factor, HT

.seealso: MatSetOption()
@*/
PetscErrorCode  MatMPIBAIJSetHashTableFactor(Mat mat,PetscReal fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(mat,"MatSetHashTableFactor_C",(Mat,PetscReal),(mat,fact));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSetHashTableFactor_MPIBAIJ"
PetscErrorCode  MatSetHashTableFactor_MPIBAIJ(Mat mat,PetscReal fact)
{
  Mat_MPIBAIJ *baij;

  PetscFunctionBegin;
  baij = (Mat_MPIBAIJ*)mat->data;
  baij->ht_fact = fact;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMPIBAIJGetSeqBAIJ"
PetscErrorCode  MatMPIBAIJGetSeqBAIJ(Mat A,Mat *Ad,Mat *Ao,PetscInt *colmap[])
{
  Mat_MPIBAIJ *a = (Mat_MPIBAIJ *)A->data;
  PetscFunctionBegin;
  *Ad     = a->A;
  *Ao     = a->B;
  *colmap = a->garray;
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

#undef __FUNCT__  
#define __FUNCT__ "matmpibiajsetvaluesblocked"
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

  Notes: This has a complete copy of MatSetValuesBlocked_MPIBAIJ() which is terrible code un-reuse.

  Level: advanced

.seealso:   MatSetValuesBlocked()
@*/
PetscErrorCode matmpibaijsetvaluesblocked_(Mat *matin,PetscInt *min,const PetscInt im[],PetscInt *nin,const PetscInt in[],const MatScalar v[],InsertMode *addvin)
{
  /* convert input arguments to C version */
  Mat             mat = *matin;
  PetscInt        m = *min, n = *nin; 
  InsertMode      addv = *addvin;

  Mat_MPIBAIJ     *baij = (Mat_MPIBAIJ*)mat->data;
  const MatScalar *value;
  MatScalar       *barray=baij->barray;
  PetscBool       roworiented = baij->roworiented;
  PetscErrorCode  ierr;
  PetscInt        i,j,ii,jj,row,col,rstart=baij->rstartbs;
  PetscInt        rend=baij->rendbs,cstart=baij->cstartbs,stepval;
  PetscInt        cend=baij->cendbs,bs=mat->rmap->bs,bs2=baij->bs2;
  
  PetscFunctionBegin;
  /* tasks normally handled by MatSetValuesBlocked() */
  if (mat->insertmode == NOT_SET_VALUES) {
    mat->insertmode = addv;
  }
#if defined(PETSC_USE_DEBUG) 
  else if (mat->insertmode != addv) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
  if (mat->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
#endif
  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);


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
        /* If NumCol = 1 then a copy is not required */
        if ((roworiented) && (n == 1)) {
          barray = (MatScalar*)v + i*bs2;
        } else if((!roworiented) && (m == 1)) {
          barray = (MatScalar*)v + j*bs2;
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
          ierr = MatSetValuesBlocked_SeqBAIJ(baij->A,1,&row,1,&col,barray,addv);CHKERRQ(ierr);
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
              ierr = MatDisAssemble_MPIBAIJ(mat);CHKERRQ(ierr); 
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
  
  /* task normally handled by MatSetValuesBlocked() */
  ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPIBAIJWithArrays"
/*@
     MatCreateMPIBAIJWithArrays - creates a MPI BAIJ matrix using arrays that contain in standard
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
PetscErrorCode  MatCreateMPIBAIJWithArrays(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt i[],const PetscInt j[],const PetscScalar a[],Mat *mat)
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
  ierr = MatMPIBAIJSetPreallocationCSR(*mat,bs,i,j,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
