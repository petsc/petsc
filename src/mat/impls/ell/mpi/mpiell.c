
#include <../src/mat/impls/ell/mpi/mpiell.h>   /*I  "petscmat.h"  I*/

#include <petscblaslapack.h>
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "MatStoreValues_MPIELL"
PetscErrorCode  MatStoreValues_MPIELL(Mat mat)
{
  Mat_MPIELL    *ell = (Mat_MPIELL*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatStoreValues(ell->A);CHKERRQ(ierr);
  ierr = MatStoreValues(ell->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRetrieveValues_MPIELL"
PetscErrorCode  MatRetrieveValues_MPIELL(Mat mat)
{
  Mat_MPIELL    *ell = (Mat_MPIELL*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRetrieveValues(ell->A);CHKERRQ(ierr);
  ierr = MatRetrieveValues(ell->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Local utility routine that creates a mapping from the global column
   number to the local number in the off-diagonal part of the local
   storage of the matrix.  This is done in a non scalable way since the
   length of colmap equals the global matrix length.
*/
#undef __FUNCT__
#define __FUNCT__ "MatCreateColmap_MPIELL_Private"
PetscErrorCode MatCreateColmap_MPIELL_Private(Mat mat)
{
  Mat_MPIELL    *ell = (Mat_MPIELL*)mat->data;
  Mat_SeqELL    *B    = (Mat_SeqELL*)ell->B->data;
  PetscErrorCode ierr;
  PetscInt       nbs = B->nbs,i,bs=mat->rmap->bs;

  PetscFunctionBegin;
#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableCreate(ell->nbs,ell->Nbs+1,&ell->colmap);CHKERRQ(ierr);
  for (i=0; i<nbs; i++) {
    ierr = PetscTableAdd(ell->colmap,ell->garray[i]+1,i*bs+1,INSERT_VALUES);CHKERRQ(ierr);
  }
#else
  ierr = PetscMalloc1(ell->Nbs+1,&ell->colmap);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)mat,ell->Nbs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(ell->colmap,ell->Nbs*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<nbs; i++) ell->colmap[ell->garray[i]] = i*bs+1;
#endif
  PetscFunctionReturn(0);
}

#define  MatSetValues_SeqELL_A_Private(row,col,value,addv,orow,ocol)       \
  { \
 \
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
    if (a->nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(A,a->mbs,bs2,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,aimax,a->nonew,MatScalar); \
    N = nrow++ - 1;  \
    /* shift up all the later entries in this row */ \
    for (ii=N; ii>=_i; ii--) { \
      rp[ii+1] = rp[ii]; \
      ierr     = PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(MatScalar));CHKERRQ(ierr); \
    } \
    if (N>=_i) { ierr = PetscMemzero(ap+bs2*_i,bs2*sizeof(MatScalar));CHKERRQ(ierr); }  \
    rp[_i]                      = bcol;  \
    ap[bs2*_i + bs*cidx + ridx] = value;  \
a_noinsert:; \
    ailen[brow] = nrow; \
  }

#define  MatSetValues_SeqELL_B_Private(row,col,value,addv,orow,ocol)       \
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
    if (b->nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column  (%D, %D) into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(B,b->mbs,bs2,nrow,brow,bcol,rmax,ba,bi,bj,rp,ap,bimax,b->nonew,MatScalar); \
    N = nrow++ - 1;  \
    /* shift up all the later entries in this row */ \
    for (ii=N; ii>=_i; ii--) { \
      rp[ii+1] = rp[ii]; \
      ierr     = PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(MatScalar));CHKERRQ(ierr); \
    } \
    if (N>=_i) { ierr = PetscMemzero(ap+bs2*_i,bs2*sizeof(MatScalar));CHKERRQ(ierr);}  \
    rp[_i]                      = bcol;  \
    ap[bs2*_i + bs*cidx + ridx] = value;  \
b_noinsert:; \
    bilen[brow] = nrow; \
  }

#undef __FUNCT__
#define __FUNCT__ "MatSetValues_MPIELL"
PetscErrorCode MatSetValues_MPIELL(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
{
  Mat_MPIELL    *ell = (Mat_MPIELL*)mat->data;
  MatScalar      value;
  PetscBool      roworiented = ell->roworiented;
  PetscErrorCode ierr;
  PetscInt       i,j,row,col;
  PetscInt       rstart_orig=mat->rmap->rstart;
  PetscInt       rend_orig  =mat->rmap->rend,cstart_orig=mat->cmap->rstart;
  PetscInt       cend_orig  =mat->cmap->rend,bs=mat->rmap->bs;

  /* Some Variables required in the macro */
  Mat        A      = ell->A;
  Mat_SeqELL *a     = (Mat_SeqELL*)(A)->data;
  PetscInt   *aimax = a->imax,*asliceidx=a->sliceidx,*ailen=a->ilen,*acolidx=a->colidx;
  MatScalar  *aval  = a->val;

  Mat        B      = ell->B;
  Mat_SeqELL *b     = (Mat_SeqELL*)(B)->data;
  PetscInt   *bimax = b->imax,*bsliceidx=b->sliceidx,*bilen=b->ilen,*bcolidx=b->colidx;
  MatScalar  *bval  = b->val;

  PetscInt  *rp,ii,nrow,_i,rmax,N,brow,bcol;
  PetscInt  low,high,t,ridx,cidx,bs2=a->bs2;
  MatScalar *ap,*bap;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (im[i] >= mat->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i],mat->rmap->N-1);
#endif
    if (im[i] >= rstart_orig && im[i] < rend_orig) {
      row = im[i] - rstart_orig;
      for (j=0; j<n; j++) {
        if (in[j] >= cstart_orig && in[j] < cend_orig) {
          col = in[j] - cstart_orig;
          if (roworiented) value = v[i*n+j];
          else             value = v[i+j*m];
          MatSetValues_SeqELL_A_Private(row,col,value,addv,im[i],in[j]);
          /* ierr = MatSetValues_SeqELL(ell->A,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        } else if (in[j] < 0) continue;
#if defined(PETSC_USE_DEBUG)
        else if (in[j] >= mat->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[j],mat->cmap->N-1);
#endif
        else {
          if (mat->was_assembled) {
            if (!ell->colmap) {
              ierr = MatCreateColmap_MPIELL_Private(mat);CHKERRQ(ierr);
            }
#if defined(PETSC_USE_CTABLE)
            ierr = PetscTableFind(ell->colmap,in[j]/bs + 1,&col);CHKERRQ(ierr);
            col  = col - 1;
#else
            col = ell->colmap[in[j]/bs] - 1;
#endif
            if (col < 0 && !((Mat_SeqELL*)(ell->B->data))->nonew) {
              ierr = MatDisAssemble_MPIELL(mat);CHKERRQ(ierr);
              col  =  in[j];
              /* Reinitialize the variables required by MatSetValues_SeqELL_B_Private() */
              B    = ell->B;
              b    = (Mat_SeqELL*)(B)->data;
              bimax=b->imax;bi=b->i;bilen=b->ilen;bj=b->j;
              ba   =b->a;
            } else if (col < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) into matrix", im[i], in[j]);
            else col += in[j]%bs;
          } else col = in[j];
          if (roworiented) value = v[i*n+j];
          else             value = v[i+j*m];
          MatSetValues_SeqELL_B_Private(row,col,value,addv,im[i],in[j]);
          /* ierr = MatSetValues_SeqELL(ell->B,1,&row,1,&col,&value,addv);CHKERRQ(ierr); */
        }
      }
    } else {
      if (mat->nooffprocentries) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %D even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i]);
      if (!ell->donotstash) {
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
#define __FUNCT__ "MatGetValues_MPIELL"
PetscErrorCode MatGetValues_MPIELL(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],PetscScalar v[])
{
  Mat_MPIELL    *ell = (Mat_MPIELL*)mat->data;
  PetscErrorCode ierr;
  PetscInt       bs       = mat->rmap->bs,i,j,bsrstart = mat->rmap->rstart,bsrend = mat->rmap->rend;
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
        if (idxn[j] >= bscstart && idxn[j] < bscend) {
          col  = idxn[j] - bscstart;
          ierr = MatGetValues_SeqELL(ell->A,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
        } else {
          if (!ell->colmap) {
            ierr = MatCreateColmap_MPIELL_Private(mat);CHKERRQ(ierr);
          }
#if defined(PETSC_USE_CTABLE)
          ierr = PetscTableFind(ell->colmap,idxn[j]/bs+1,&data);CHKERRQ(ierr);
          data--;
#else
          data = ell->colmap[idxn[j]/bs]-1;
#endif
          if ((data < 0) || (ell->garray[data/bs] != idxn[j]/bs)) *(v+i*n+j) = 0.0;
          else {
            col  = data + idxn[j]%bs;
            ierr = MatGetValues_SeqELL(ell->B,1,&row,1,&col,v+i*n+j);CHKERRQ(ierr);
          }
        }
      }
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only local values currently supported");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_MPIELL"
PetscErrorCode MatAssemblyBegin_MPIELL(Mat mat,MatAssemblyType mode)
{
  Mat_MPIELL    *ell = (Mat_MPIELL*)mat->data;
  PetscErrorCode ierr;
  PetscInt       nstash,reallocs;

  PetscFunctionBegin;
  if (ell->donotstash || mat->nooffprocentries) PetscFunctionReturn(0);

  ierr = MatStashScatterBegin_Private(mat,&mat->stash,mat->rmap->range);CHKERRQ(ierr);
  ierr = MatStashScatterBegin_Private(mat,&mat->bstash,ell->rangebs);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(mat,"Stash has %D entries,uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->bstash,&nstash,&reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(mat,"Block-Stash has %D entries, uses %D mallocs.\n",nstash,reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_MPIELL"
PetscErrorCode MatAssemblyEnd_MPIELL(Mat mat,MatAssemblyType mode)
{
  Mat_MPIELL    *ell=(Mat_MPIELL*)mat->data;
  Mat_SeqELL    *a   =(Mat_SeqELL*)ell->A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,rstart,ncols,flg,bs2=ell->bs2;
  PetscInt       *row,*col;
  PetscBool      r1,r2,r3,other_disassembled;
  MatScalar      *val;
  PetscMPIInt    n;

  PetscFunctionBegin;
  /* do not use 'b=(Mat_SeqELL*)ell->B->data' as B can be reset in disassembly */
  if (!ell->donotstash && !mat->nooffprocentries) {
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
        ierr = MatSetValues_MPIELL(mat,1,row+i,ncols,col+i,val+i,mat->insertmode);CHKERRQ(ierr);
        i    = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);
    /* Now process the block-stash. Since the values are stashed column-oriented,
       set the roworiented flag to column oriented, and after MatSetValues()
       restore the original flags */
    r1 = ell->roworiented;
    r2 = a->roworiented;
    r3 = ((Mat_SeqELL*)ell->B->data)->roworiented;

    ell->roworiented = PETSC_FALSE;
    a->roworiented    = PETSC_FALSE;

    (((Mat_SeqELL*)ell->B->data))->roworiented = PETSC_FALSE; /* b->roworiented */
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
        ierr = MatSetValuesBlocked_MPIELL(mat,1,row+i,ncols,col+i,val+i*bs2,mat->insertmode);CHKERRQ(ierr);
        i    = j;
      }
    }
    ierr = MatStashScatterEnd_Private(&mat->bstash);CHKERRQ(ierr);

    ell->roworiented = r1;
    a->roworiented    = r2;

    ((Mat_SeqELL*)ell->B->data)->roworiented = r3; /* b->roworiented */
  }

  ierr = MatAssemblyBegin(ell->A,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ell->A,mode);CHKERRQ(ierr);

  /* determine if any processor has disassembled, if so we must
     also disassemble ourselfs, in order that we may reassemble. */
  /*
     if nonzero structure of submatrix B cannot change then we know that
     no processor disassembled thus we can skip this stuff
  */
  if (!((Mat_SeqELL*)ell->B->data)->nonew) {
    ierr = MPIU_Allreduce(&mat->was_assembled,&other_disassembled,1,MPIU_BOOL,MPI_PROD,PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);
    if (mat->was_assembled && !other_disassembled) {
      ierr = MatDisAssemble_MPIELL(mat);CHKERRQ(ierr);
    }
  }

  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatSetUpMultiply_MPIELL(mat);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(ell->B,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ell->B,mode);CHKERRQ(ierr);

#if defined(PETSC_USE_INFO)
  if (ell->ht && mode== MAT_FINAL_ASSEMBLY) {
    ierr = PetscInfo1(mat,"Average Hash Table Search in MatSetValues = %5.2f\n",((PetscReal)ell->ht_total_ct)/ell->ht_insert_ct);CHKERRQ(ierr);

    ell->ht_total_ct  = 0;
    ell->ht_insert_ct = 0;
  }
#endif
  if (ell->ht_flag && !ell->ht && mode == MAT_FINAL_ASSEMBLY) {
    ierr = MatCreateHashTable_MPIELL_Private(mat,ell->ht_fact);CHKERRQ(ierr);

    mat->ops->setvalues        = MatSetValues_MPIELL_HT;
    mat->ops->setvaluesblocked = MatSetValuesBlocked_MPIELL_HT;
  }

  ierr = PetscFree2(ell->rowvalues,ell->rowindices);CHKERRQ(ierr);

  ell->rowvalues = 0;

  /* if no new nonzero locations are allowed in matrix then only set the matrix state the first time through */
  if ((!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) || !((Mat_SeqELL*)(ell->A->data))->nonew) {
    PetscObjectState state = ell->A->nonzerostate + ell->B->nonzerostate;
    ierr = MPIU_Allreduce(&state,&mat->nonzerostate,1,MPIU_INT64,MPI_SUM,PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPIELL"
PetscErrorCode MatDestroy_MPIELL(Mat mat)
{
  Mat_MPIELL    *ell = (Mat_MPIELL*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%D,Cols=%D",mat->rmap->N,mat->cmap->N);
#endif
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  ierr = MatStashDestroy_Private(&mat->bstash);CHKERRQ(ierr);
  ierr = MatDestroy(&ell->A);CHKERRQ(ierr);
  ierr = MatDestroy(&ell->B);CHKERRQ(ierr);
#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableDestroy(&ell->colmap);CHKERRQ(ierr);
#else
  ierr = PetscFree(ell->colmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(ell->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&ell->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ell->Mvctx);CHKERRQ(ierr);
  ierr = PetscFree2(ell->rowvalues,ell->rowindices);CHKERRQ(ierr);
  ierr = PetscFree(ell->barray);CHKERRQ(ierr);
  ierr = PetscFree2(ell->hd,ell->ht);CHKERRQ(ierr);
  ierr = PetscFree(ell->rangebs);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatStoreValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatRetrieveValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIELLSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMPIELLSetPreallocationCSR_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDiagonalScaleLocal_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_MPIELL"
PetscErrorCode MatMult_MPIELL(Mat A,Vec xx,Vec yy)
{
  Mat_MPIELL    *a = (Mat_MPIELL*)A->data;
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
#define __FUNCT__ "MatMultAdd_MPIELL"
PetscErrorCode MatMultAdd_MPIELL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIELL     *a = (Mat_MPIELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This only works correctly for square matrices where the subblock A->A is the
   diagonal block
*/
#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_MPIELL"
PetscErrorCode MatGetDiagonal_MPIELL(Mat A,Vec v)
{
  Mat_MPIELL    *a = (Mat_MPIELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->rmap->N != A->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Supports only square matrix where A->A is diag block");
  ierr = MatGetDiagonal(a->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_MPIELL"
PetscErrorCode MatScale_MPIELL(Mat A,PetscScalar aa)
{
  Mat_MPIELL    *a = (Mat_MPIELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(a->A,aa);CHKERRQ(ierr);
  ierr = MatScale(a->B,aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreRow_MPIELL"
PetscErrorCode MatRestoreRow_MPIELL(Mat mat,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_MPIELL *ell = (Mat_MPIELL*)mat->data;

  PetscFunctionBegin;
  if (!ell->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"MatGetRow not called");
  ell->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_MPIELL"
PetscErrorCode MatZeroEntries_MPIELL(Mat A)
{
  Mat_MPIELL    *l = (Mat_MPIELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetInfo_MPIELL"
PetscErrorCode MatGetInfo_MPIELL(Mat matin,MatInfoType flag,MatInfo *info)
{
  Mat_MPIELL    *a = (Mat_MPIELL*)matin->data;
  Mat            A  = a->A,B = a->B;
  PetscErrorCode ierr;
  PetscReal      isend[5],irecv[5];

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
    ierr = MPIU_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)matin));CHKERRQ(ierr);

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPIU_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)matin));CHKERRQ(ierr);

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else SETERRQ1(PetscObjectComm((PetscObject)matin),PETSC_ERR_ARG_WRONG,"Unknown MatInfoType argument %d",(int)flag);
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_MPIELL"
PetscErrorCode MatSetOption_MPIELL(Mat A,MatOption op,PetscBool flg)
{
  Mat_MPIELL    *a = (Mat_MPIELL*)A->data;
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
    MatCheckPreallocated(A,1);
    ierr = MatSetOption(a->A,op,flg);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroRowsColumns_MPIELL"
PetscErrorCode MatZeroRowsColumns_MPIELL(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_MPIELL       *l = (Mat_MPIELL*)A->data;
  PetscErrorCode    ierr;
  PetscMPIInt       n = A->rmap->n;
  PetscInt          i,j,k,r,p = 0,len = 0,row,col,count;
  PetscInt          *lrows,*owners = A->rmap->range;
  PetscSFNode       *rrows;
  PetscSF           sf;
  const PetscScalar *xx;
  PetscScalar       *bb,*mask;
  Vec               xmask,lmask;
  Mat_SeqELL       *ell = (Mat_SeqELL*)l->B->data;
  PetscInt           bs = A->rmap->bs, bs2 = ell->bs2;
  PetscScalar       *aa;

  PetscFunctionBegin;
  /* Create SF where leaves are input rows and roots are owned rows */
  ierr = PetscMalloc1(n, &lrows);CHKERRQ(ierr);
  for (r = 0; r < n; ++r) lrows[r] = -1;
  ierr = PetscMalloc1(N, &rrows);CHKERRQ(ierr);
  for (r = 0; r < N; ++r) {
    const PetscInt idx   = rows[r];
    if (idx < 0 || A->rmap->N <= idx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D out of range [0,%D)",idx,A->rmap->N);
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
    count = (ell->i[row/bs +1] - ell->i[row/bs])*bs;
    aa    = ((MatScalar*)(ell->a)) + ell->i[row/bs]*bs2 + (row%bs);
    for (k = 0; k < count; ++k) {
      aa[0] = 0.0;
      aa   += bs;
    }
  }
  /* loop over all elements of off process part of matrix zeroing removed columns*/
  for (i = 0; i < l->B->rmap->N; ++i) {
    row = i/bs;
    for (j = ell->i[row]; j < ell->i[row+1]; ++j) {
      for (k = 0; k < bs; ++k) {
        col = bs*ell->j[j] + k;
        if (PetscAbsScalar(mask[col])) {
          aa = ((MatScalar*)(ell->a)) + j*bs2 + (i%bs) + bs*k;
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
  if (!((Mat_SeqELL*)(l->A->data))->keepnonzeropattern) {
    PetscObjectState state = l->A->nonzerostate + l->B->nonzerostate;
    ierr = MPIU_Allreduce(&state,&A->nonzerostate,1,MPIU_INT64,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUnfactored_MPIELL"
PetscErrorCode MatSetUnfactored_MPIELL(Mat A)
{
  Mat_MPIELL    *a = (Mat_MPIELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(a->A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_MPIELL(Mat,MatDuplicateOption,Mat*);

#undef __FUNCT__
#define __FUNCT__ "MatEqual_MPIELL"
PetscErrorCode MatEqual_MPIELL(Mat A,Mat B,PetscBool  *flag)
{
  Mat_MPIELL    *matB = (Mat_MPIELL*)B->data,*matA = (Mat_MPIELL*)A->data;
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
  ierr = MPIU_Allreduce(&flg,flag,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCopy_MPIELL"
PetscErrorCode MatCopy_MPIELL(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_MPIELL    *a = (Mat_MPIELL*)A->data;
  Mat_MPIELL    *b = (Mat_MPIELL*)B->data;

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
#define __FUNCT__ "MatSetUp_MPIELL"
PetscErrorCode MatSetUp_MPIELL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMPIELLSetPreallocation(A,A->rmap->bs,PETSC_DEFAULT,0,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRealPart_MPIELL"
PetscErrorCode MatRealPart_MPIELL(Mat A)
{
  Mat_MPIELL    *a = (Mat_MPIELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatRealPart(a->A);CHKERRQ(ierr);
  ierr = MatRealPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatImaginaryPart_MPIELL"
PetscErrorCode MatImaginaryPart_MPIELL(Mat A)
{
  Mat_MPIELL    *a = (Mat_MPIELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatImaginaryPart(a->A);CHKERRQ(ierr);
  ierr = MatImaginaryPart(a->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetGhosts_MPIELL"
PetscErrorCode  MatGetGhosts_MPIELL(Mat mat,PetscInt *nghosts,const PetscInt *ghosts[])
{
  Mat_MPIELL *ell = (Mat_MPIELL*) mat->data;
  Mat_SeqELL *B    = (Mat_SeqELL*)ell->B->data;

  PetscFunctionBegin;
  if (nghosts) *nghosts = B->nbs;
  if (ghosts) *ghosts = ell->garray;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvertBlockDiagonal_MPIELL"
PetscErrorCode MatInvertBlockDiagonal_MPIELL(Mat A,const PetscScalar **values)
{
  Mat_MPIELL    *a = (Mat_MPIELL*) A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatInvertBlockDiagonal(a->A,values);CHKERRQ(ierr);
  A->factorerrortype             = a->A->factorerrortype;
  A->factorerror_zeropivot_value = a->A->factorerror_zeropivot_value;
  A->factorerror_zeropivot_row   = a->A->factorerror_zeropivot_row;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatShift_MPIELL"
PetscErrorCode MatShift_MPIELL(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_MPIELL    *mell = (Mat_MPIELL*)Y->data;
  Mat_SeqELL    *ell = (Mat_SeqELL*)mell->A->data;

  PetscFunctionBegin;
  if (!Y->preallocated) {
    ierr = MatMPIELLSetPreallocation(Y,Y->rmap->bs,1,NULL,0,NULL);CHKERRQ(ierr);
  } else if (!ell->nz) {
    PetscInt nonew = ell->nonew;
    ierr = MatSeqELLSetPreallocation(mell->A,Y->rmap->bs,1,NULL);CHKERRQ(ierr);
    ell->nonew = nonew;
  }
  ierr = MatShift_Basic(Y,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMissingDiagonal_MPIELL"
PetscErrorCode MatMissingDiagonal_MPIELL(Mat A,PetscBool  *missing,PetscInt *d)
{
  Mat_MPIELL    *a = (Mat_MPIELL*)A->data;
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

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonalBlock_MPIELL"
PetscErrorCode  MatGetDiagonalBlock_MPIELL(Mat A,Mat *a)
{
  PetscFunctionBegin;
  *a = ((Mat_MPIELL*)A->data)->A;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_MPIELL,
                                       0,
                                       MatRestoreRow_MPIELL,
                                       MatMult_MPIELL,
                                /* 4*/ MatMultAdd_MPIELL,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*10*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*15*/ MatGetInfo_MPIELL,
                                       MatEqual_MPIELL,
                                       MatGetDiagonal_MPIELL,
                                       0,
                                       0,
                                /*20*/ MatAssemblyBegin_MPIELL,
                                       MatAssemblyEnd_MPIELL,
                                       MatSetOption_MPIELL,
                                       MatZeroEntries_MPIELL,
                                /*24*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*29*/ MatSetUp_MPIELL,
                                       0,
                                       0,
                                       MatGetDiagonalBlock_MPIELL,
                                       0,
                                /*34*/ MatDuplicate_MPIELL,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*39*/ 0,
                                       0,
                                       0,
                                       MatGetValues_MPIELL,
                                       MatCopy_MPIELL,
                                /*44*/ 0,
                                       MatScale_MPIELL,
                                       MatShift_MPIELL,
                                       0,
                                       MatZeroRowsColumns_MPIELL,
                                /*49*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*54*/ 0,
                                       0,
                                       MatSetUnfactored_MPIELL,
                                       0,
                                       0,
                                /*59*/ 0,
                                       MatDestroy_MPIELL,
                                       0,
                                       0,
                                       0,
                                /*64*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*69*/ 0,
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
                                       0,
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
                                       MatRealPart_MPIELL,
                                       MatImaginaryPart_MPIELL,
                                       0,
                                       0,
                                /*109*/0,
                                       0,
                                       0,
                                       0,
                                       MatMissingDiagonal_MPIELL,
                                /*114*/0,
                                       0,
                                       MatGetGhosts_MPIELL,
                                       0,
                                       0,
                                /*119*/0,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*124*/0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*129*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*134*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*139*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                                /*144*/0
};


#undef __FUNCT__
#define __FUNCT__ "MatMPIELLSetPreallocationCSR_MPIELL"
PetscErrorCode MatMPIELLSetPreallocationCSR_MPIELL(Mat B,PetscInt bs,const PetscInt ii[],const PetscInt jj[],const PetscScalar V[])
{
  PetscInt       m,rstart,cstart,cend;
  PetscInt       i,j,d,nz,nz_max=0,*d_nnz=0,*o_nnz=0;
  const PetscInt *JJ    =0;
  PetscScalar    *values=0;
  PetscBool      roworiented = ((Mat_MPIELL*)B->data)->roworiented;
  PetscErrorCode ierr;

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

  if (ii[0]) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ii[0] must be 0 but it is %D",ii[0]);
  ierr = PetscMalloc2(m,&d_nnz,m,&o_nnz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nz = ii[i+1] - ii[i];
    if (nz < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local row %D has a negative number of columns %D",i,nz);
    nz_max = PetscMax(nz_max,nz);
    JJ     = jj + ii[i];
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
  ierr = MatMPIELLSetPreallocation(B,bs,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
  ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);

  values = (PetscScalar*)V;
  if (!values) {
    ierr = PetscMalloc1(bs*bs*nz_max,&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,bs*bs*nz_max*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (i=0; i<m; i++) {
    PetscInt          row    = i + rstart;
    PetscInt          ncols  = ii[i+1] - ii[i];
    const PetscInt    *icols = jj + ii[i];
    if (!roworiented) {         /* block ordering matches the non-nested layout of MatSetValues so we can insert entire rows */
      const PetscScalar *svals = values + (V ? (bs*bs*ii[i]) : 0);
      ierr = MatSetValuesBlocked_MPIELL(B,1,&row,ncols,icols,svals,INSERT_VALUES);CHKERRQ(ierr);
    } else {                    /* block ordering does not match so we can only insert one block at a time. */
      PetscInt j;
      for (j=0; j<ncols; j++) {
        const PetscScalar *svals = values + (V ? (bs*bs*(ii[i]+j)) : 0);
        ierr = MatSetValuesBlocked_MPIELL(B,1,&row,1,&icols[j],svals,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  if (!V) { ierr = PetscFree(values);CHKERRQ(ierr); }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMPIELLSetPreallocationCSR"
/*@C
   MatMPIELLSetPreallocationCSR - Allocates memory for a sparse parallel matrix in ELL format
   (the default parallel PETSc format).

   Collective on MPI_Comm

   Input Parameters:
+  B - the matrix
.  bs - the block size
.  i - the indices into j for the start of each local row (starts with zero)
.  j - the column indices for each local row (starts with zero) these must be sorted for each row
-  v - optional values in the matrix

   Level: developer

   Notes: The order of the entries in values is specified by the MatOption MAT_ROW_ORIENTED.  For example, C programs
   may want to use the default MAT_ROW_ORIENTED=PETSC_TRUE and use an array v[nnz][bs][bs] where the second index is
   over rows within a block and the last index is over columns within a block row.  Fortran programs will likely set
   MAT_ROW_ORIENTED=PETSC_FALSE and use a Fortran array v(bs,bs,nnz) in which the first index is over rows within a
   block column and the second index is over columns within a block.

.keywords: matrix, ell, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIELLSetPreallocation(), MatCreateAIJ(), MPIAIJ, MatCreateMPIELLWithArrays(), MPIELL
@*/
PetscErrorCode  MatMPIELLSetPreallocationCSR(Mat B,PetscInt bs,const PetscInt i[],const PetscInt j[], const PetscScalar v[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscTryMethod(B,"MatMPIELLSetPreallocationCSR_C",(Mat,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[]),(B,bs,i,j,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMPIELLSetPreallocation_MPIELL"
PetscErrorCode  MatMPIELLSetPreallocation_MPIELL(Mat B,PetscInt d_nz,const PetscInt *d_nnz,PetscInt o_nz,const PetscInt *o_nnz)
{
  Mat_MPIELL    *b;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);

  if (d_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than -1: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than -1: local row %D value %D",i,o_nnz[i]);
    }
  }

  b      = (Mat_MPIELL*)B->data;
  if (!B->preallocated) {
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQELL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQELL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);
    ierr = MatStashCreate_Private(PetscObjectComm((PetscObject)B),bs,&B->bstash);CHKERRQ(ierr);
  }

  ierr = MatSeqELLSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqELLSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

extern PetscErrorCode  MatDiagonalScaleLocal_MPIELL(Mat,Vec);

#include <../src/mat/impls/ell/mpi/mpiell.h>

/*MC
   MATMPIELL - MATMPIELL = "mpiell" - A matrix type to be used for distributed block sparse matrices.

   Options Database Keys:
+ -mat_type mpiell - sets the matrix type to "mpiell" during a call to MatSetFromOptions()
. -mat_block_size <bs> - set the blocksize used to store the matrix
- -mat_use_hash_table <fact>

  Level: beginner

.seealso: MatCreateMPIELL
M*/

#undef __FUNCT__
#define __FUNCT__ "MatCreate_MPIELL"
PETSC_EXTERN PetscErrorCode MatCreate_MPIELL(Mat B)
{
  Mat_MPIELL    *b;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr    = PetscNewLog(B,&b);CHKERRQ(ierr);
  B->data = (void*)b;

  ierr         = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->assembled = PETSC_FALSE;

  B->insertmode = NOT_SET_VALUES;
  ierr          = MPI_Comm_rank(PetscObjectComm((PetscObject)B),&b->rank);CHKERRQ(ierr);
  ierr          = MPI_Comm_size(PetscObjectComm((PetscObject)B),&b->size);CHKERRQ(ierr);

  /* build local table of row and column ownerships */
  ierr = PetscMalloc1(b->size+1,&b->rangebs);CHKERRQ(ierr);

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(PetscObjectComm((PetscObject)B),1,&B->stash);CHKERRQ(ierr);

  b->donotstash  = PETSC_FALSE;
  b->colmap      = NULL;
  b->garray      = NULL;
  b->roworiented = PETSC_TRUE;

  /* stuff used in block assembly */
  b->barray = 0;

  /* stuff used for matrix vector multiply */
  b->lvec  = 0;
  b->Mvctx = 0;

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

  /* stuff for MatGetSubMatrices_MPIELL_local() */
  b->ijonly = PETSC_FALSE;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_MPIELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_MPIELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPIELLSetPreallocation_C",MatMPIELLSetPreallocation_MPIELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMPIELLSetPreallocationCSR_C",MatMPIELLSetPreallocationCSR_MPIELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDiagonalScaleLocal_C",MatDiagonalScaleLocal_MPIELL);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIELL);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)B),NULL,"Options for loading MPIELL matrix 1","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_use_hash_table","Use hash table to save memory in constructing matrix","MatSetOption",flg,&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    PetscReal fact = 1.39;
    ierr = MatSetOption(B,MAT_USE_HASH_TABLE,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_use_hash_table","Use hash table factor","MatMPIELLSetHashTableFactor",fact,&fact,NULL);CHKERRQ(ierr);
    if (fact <= 1.0) fact = 1.39;
    ierr = MatMPIELLSetHashTableFactor(B,fact);CHKERRQ(ierr);
    ierr = PetscInfo1(B,"Hash table Factor used %5.2f\n",fact);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATELL - MATELL = "ell" - A matrix type to be used for block sparse matrices.

   This matrix type is identical to MATSEQELL when constructed with a single process communicator,
   and MATMPIELL otherwise.

   Options Database Keys:
. -mat_type ell - sets the matrix type to "ell" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateELL(),MATSEQELL,MATMPIELL, MatMPIELLSetPreallocation(), MatMPIELLSetPreallocationCSR()
M*/

#undef __FUNCT__
#define __FUNCT__ "MatMPIELLSetPreallocation"
/*@C
   MatMPIELLSetPreallocation - Allocates memory for a sparse parallel matrix in ELL format
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
   stored simply in the MATSEQELL format for compressed row storage.

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

.keywords: matrix, block, ell, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqELL(), MatSetValues(), MatCreateELL(), MatMPIELLSetPreallocationCSR(), PetscSplitOwnership()
@*/
PetscErrorCode  MatMPIELLSetPreallocation(Mat B,PetscInt bs,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscValidLogicalCollectiveInt(B,bs,2);
  ierr = PetscTryMethod(B,"MatMPIELLSetPreallocation_C",(Mat,PetscInt,PetscInt,const PetscInt[],PetscInt,const PetscInt[]),(B,bs,d_nz,d_nnz,o_nz,o_nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateELL"
/*@C
   MatCreateELL - Creates a sparse parallel matrix in ELL format.
   For good matrix assembly performance the user should preallocate the matrix
   storage by setting the parameters d_nz (or d_nnz) and o_nz (or o_nnz).

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
.  n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
           This value should be the same as the local size used in creating the
           x vector for the matrix-vector product y = Ax.
.  M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.  N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.  d_nz  - number of nonzeros per row in diagonal portion of local submatrix
           (same for all local rows)
.  d_nnz - array containing the number of nonzeros in the various rows of the
           in diagonal portion of the local (possibly different for each row)
           or NULL.  If you plan to factor the matrix you must leave room for
           the diagonal entry and set it even if it is zero.
.  o_nz  - number of nonzeros per block row in the off-diagonal portion of local
           submatrix (same for all local rows).
-  o_nnz - array containing the number of nonzeros in the various block rows of the
           off-diagonal portion of the local submatrix (possibly different for
           each row) or NULL.

   Output Parameter:
.  A - the matrix

   Options Database Keys:

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If the *_nnz parameter is given then the *_nz parameter is ignored

   The user MUST specify either the local or global matrix dimensions (possibly both).

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
   stored simply in the MATSEQELL format for compressed row storage.

   Now d_nz should indicate the number of nonzeros per row in the d matrix,
   and o_nz should indicate the number of nonzeros per row in the o matrix.
   In general, for PDE problems in which most nonzeros are near the diagonal,
   one expects d_nz >> o_nz.   For large problems you MUST preallocate memory
   or you will get TERRIBLE performance; see the users' manual chapter on
   matrices.

   Level: intermediate

.keywords: matrix, ell, ell, sparse, parallel

.seealso: MatCreate(), MatCreateSeqELL(), MatSetValues(), MatCreateELL(), MatMPIELLSetPreallocation(), MatMPIELLSetPreallocationCSR()
@*/
PetscErrorCode  MatCreateELL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIELL);CHKERRQ(ierr);
    ierr = MatMPIELLSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQELL);CHKERRQ(ierr);
    ierr = MatSeqELLSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_MPIELL"
static PetscErrorCode MatDuplicate_MPIELL(Mat matin,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat            mat;
  Mat_MPIELL    *a,*oldmat = (Mat_MPIELL*)matin->data;
  PetscErrorCode ierr;
  PetscInt       len=0;

  PetscFunctionBegin;
  *newmat = 0;
  ierr    = MatCreate(PetscObjectComm((PetscObject)matin),&mat);CHKERRQ(ierr);
  ierr    = MatSetSizes(mat,matin->rmap->n,matin->cmap->n,matin->rmap->N,matin->cmap->N);CHKERRQ(ierr);
  ierr    = MatSetType(mat,((PetscObject)matin)->type_name);CHKERRQ(ierr);
  ierr    = PetscMemcpy(mat->ops,matin->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  mat->factortype   = matin->factortype;
  mat->preallocated = PETSC_TRUE;
  mat->assembled    = PETSC_TRUE;
  mat->insertmode   = NOT_SET_VALUES;

  a             = (Mat_MPIELL*)mat->data;
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
#if defined(PETSC_USE_CTABLE)
    ierr = PetscTableCreateCopy(oldmat->colmap,&a->colmap);CHKERRQ(ierr);
#else
    ierr = PetscMalloc1(a->Nbs,&a->colmap);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)mat,(a->Nbs)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(a->colmap,oldmat->colmap,(a->Nbs)*sizeof(PetscInt));CHKERRQ(ierr);
#endif
  } else a->colmap = 0;

  if (oldmat->garray && (len = ((Mat_SeqELL*)(oldmat->B->data))->nbs)) {
    ierr = PetscMalloc1(len,&a->garray);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)mat,len*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(a->garray,oldmat->garray,len*sizeof(PetscInt));CHKERRQ(ierr);
  } else a->garray = 0;

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

#undef __FUNCT__
#define __FUNCT__ "MatMPIELLGetSeqELL"
PetscErrorCode  MatMPIELLGetSeqELL(Mat A,Mat *Ad,Mat *Ao,const PetscInt *colmap[])
{
  Mat_MPIELL *a = (Mat_MPIELL*)A->data;

  PetscFunctionBegin;
  if (Ad)     *Ad     = a->A;
  if (Ao)     *Ao     = a->B;
  if (colmap) *colmap = a->garray;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateMPIELLWithArrays"
/*@
     MatCreateMPIELLWithArrays - creates a MPI ELL matrix using arrays that contain in standard
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

     The order of the entries in values is the same as the block compressed sparse row storage format; that is, it is
     the same as a three dimensional array in Fortran values(bs,bs,nnz) that contains the first column of the first
     block, followed by the second column of the first block etc etc.  That is, the blocks are contiguous in memory
     with column-major ordering within blocks.

       The i and j indices are 0 based, and i indices are indices corresponding to the local j array.

.keywords: matrix, ell, compressed row, sparse, parallel

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatMPIAIJSetPreallocation(), MatMPIAIJSetPreallocationCSR(),
          MPIAIJ, MatCreateAIJ(), MatCreateMPIAIJWithSplitArrays()
@*/
PetscErrorCode  MatCreateMPIELLWithArrays(MPI_Comm comm,PetscInt bs,PetscInt m,PetscInt n,PetscInt M,PetscInt N,const PetscInt i[],const PetscInt j[],const PetscScalar a[],Mat *mat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (i[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  if (m < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"local number of rows (m) cannot be PETSC_DECIDE, or negative");
  ierr = MatCreate(comm,mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATMPISELL);CHKERRQ(ierr);
  ierr = MatSetOption(*mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatMPIELLSetPreallocationCSR(*mat,bs,i,j,a);CHKERRQ(ierr);
  ierr = MatSetOption(*mat,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
