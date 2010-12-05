#define PETSCMAT_DLL

#include "../src/dm/dd/matdd/matdd.h"          

typedef struct {
  Mat_DDMeta        meta;
  PetscInt          nonew;            /* 1 don't add new nonzero blocks, -1 generate error on new */
  PetscInt          nounused;         /* -1 generate error on unused space */
  PetscBool         singlemalloc;     /* if true a, i, and j have been obtained with one big malloc */
  PetscInt          maxnz;            /* allocated nonzeros */
  PetscInt          *imax;            /* maximum space allocated for each row */
  PetscInt          *ilen;            /* actual length of each row */
  PetscBool         free_imax_ilen;  
  PetscInt          reallocs;         /* number of mallocs done during MatDDAddBlock()
                                        as more blocks are set than were prealloced */
  PetscInt          rmax;             /* max nonzeros in any row */
  PetscBool         free_ij;          /* free the column indices j and row offsets i when the matrix is destroyed */ 
  PetscBool         free_a;           /* free the numerical values when matrix is destroy */ 
  PetscInt          nz;               /* nonzero blocks */                                       
  PetscInt          *i;               /* pointer to beginning of each row */               
  PetscInt          *j;               /* column values: j + i[k] - 1 is start of row k */  
  MatDDMeta_Block   *a;               /* nonzero blocks */                               
} Mat_DDMetaAIJ;


/*
    Frees the a, i, and j arrays from the MatDDMetaAIJ matrix type
*/
#undef __FUNCT__  
#define __FUNCT__ "MatDDMetaAIJ_FreeAIJ"
PETSC_STATIC_INLINE PetscErrorCode MatDDMetaAIJ_FreeAIJ(Mat M, MetaMatDD_Block **a,PetscInt **j,PetscInt **i) 
{
  PetscErrorCode ierr;
                                     Mat_DDMetaAIJ *A = (Mat_DDMetaAIJ*) M->data;
                                     if (A->singlemalloc) {
                                       ierr = PetscFree3(*a,*j,*i);CHKERRQ(ierr);
                                     } else {
                                       if (A->free_a  && *a) {ierr = PetscFree(*a);CHKERRQ(ierr);}
                                       if (A->free_ij && *j) {ierr = PetscFree(*j);CHKERRQ(ierr);}
                                       if (A->free_ij && *i) {ierr = PetscFree(*i);CHKERRQ(ierr);}
                                     }
                                     *a = 0; *j = 0; *i = 0;
                                     return 0;
}

#define CHUNKSIZE 15
/*
    Allocates larger a, i, and j arrays for the DDAIJ matrix type
*/
#define MatDDMetaAIJ_ReallocateAIJ(M,rowcount,current_row_length,row,col,allocated_row_length,aa,ai,aj,j_row_pointer,a_row_pointer,allocated_row_lengths,no_new_block_flag) \
{\
  Mat_DDMetaAIJ *A = (Mat_DDMetaAIJ*)M->data;\
  PetscInt AM = rowcount, NROW = current_row_length, ROW = row, COL = col, RMAX = allocated_row_length;\
  MatDDMeta_Block *AA = aa;\
  PetscInt *AI = ai, *AJ = aj, *RP = j_row_pointer;\
  MatDDMeta_Block *AP = a_row_pointer;\
  PetscInt *AIMAX = allocated_row_lengths;\
  PetscBool NONEW=(no_new_block_flag) ? PETSC_TRUE: PETSC_FALSE;	\
  if (NROW >= RMAX) {\
        /* there is no extra room in row, therefore enlarge */ \
        PetscInt   new_nz = AI[AM] + CHUNKSIZE,len,*new_i=0,*new_j=0; \
        MatDDMeta_Block *new_a; \
 \
        if (NONEW == -2) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"New nonzero at (%D,%D) caused a malloc",ROW,COL); \
        /* malloc new storage space */ \
        ierr = PetscMalloc3(new_nz,MatDDMeta_Block,&new_a,new_nz,PetscInt,&new_j,AM+1,PetscInt,&new_i);CHKERRQ(ierr);\
 \
        /* copy over old data into new slots */ \
        for (ii=0; ii<ROW+1; ii++) {new_i[ii] = AI[ii];} \
        for (ii=ROW+1; ii<AM+1; ii++) {new_i[ii] = AI[ii]+CHUNKSIZE;} \
        ierr = PetscMemcpy(new_j,AJ,(AI[ROW]+NROW)*sizeof(PetscInt));CHKERRQ(ierr); \
        len = (new_nz - CHUNKSIZE - AI[ROW] - NROW); \
        ierr = PetscMemcpy(new_j+AI[ROW]+NROW+CHUNKSIZE,AJ+AI[ROW]+NROW,len*sizeof(PetscInt));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a,AA,(AI[ROW]+NROW)*sizeof(MatDDMeta_Block));CHKERRQ(ierr); \
        ierr = PetscMemzero(new_a+(AI[ROW]+NROW),CHUNKSIZE*sizeof(MatDDMeta_Block));CHKERRQ(ierr);\
        ierr = PetscMemcpy(new_a+(AI[ROW]+NROW+CHUNKSIZE),AA+(AI[ROW]+NROW),len*sizeof(MatDDMeta_Block));CHKERRQ(ierr);  \
        /* free up old matrix storage */ \
        ierr = MatDDMetaAIJ_FreeAIJ(M,&A->a,&A->j,&A->i);CHKERRQ(ierr);\
        AA = new_a; \
        A->a = (MatDDMeta_Block*) new_a;		   \
        AI = A->i = new_i; AJ = A->j = new_j;  \
        A->singlemalloc = PETSC_TRUE; \
 \
        RP          = AJ + AI[ROW]; AP = AA + AI[ROW]; \
        RMAX        = AIMAX[ROW] = AIMAX[ROW] + CHUNKSIZE; \
        A->maxnz += CHUNKSIZE; \
        A->reallocs++; \
      } \
} \

#define MatDDSetup_MetaAIJ(A) \
{\
  Mat_DD        *dd  (Mat_DD*)A->data;\
  Mat_DDMetaAIJ *aij (Mat_DDMetaAIJ*)A->data;\
  if(!dd->setup) {\
    PetscInt i,j,k;                             \
    MetaMatDD_Block *b;                         \
    for (i=0; i<dd->rowblockcount; ++i) {\
      for(k = 0; k < aij->ilen[i]; ++k) {\
        j = *(aij->j + aij->i[i] + k);\
        b = aij->a + aij->i[i] + k;\
        ierr = MatDDMeta_BlockSetUp(A, i,j,*b); CHKERRQ(ierr);\
      }\
    }
    dd->setup = PETSC_TRUE;\
  }\
}\



#undef __FUNCT__  
#define __FUNCT__ "MatDDMetaAIJSetPreallocation"
PetscErrorCode  MatDDMetaAIJSetPreallocation(Mat A, PetscInt nz, PetscInt *nnz)
{
  /* Assume gather and scatter have been set */

  Mat_DDMetaAIJ  *dd = (Mat_DD*)A->data;
  Mat_DDMetaAIJ  *aij = (Mat_DDMetaAIJ*)A->data;
  PetscBool      skipallocation = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  
  if (nz == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  if (nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %d",nz);
  if (nnz) {
    for (i=0; i<dd->rowblockcount; ++i) {
      if (nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: block %d value %d",i,nnz[i]);
      if (nnz[i] > dd->colblockcount) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than the number of column blocks: block %d value %d column block count %d",i,nnz[i],dd->colblockcount);
    }
  }
  if (!skipallocation) {
    if (!aij->imax) {
      ierr = PetscMalloc2(dd->rowblockcount,PetscInt,&aij->imax,dd->rowblockcount,PetscInt,&aij->ilen);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(A,2*dd->rowblockcount*sizeof(PetscInt));CHKERRQ(ierr); 
    }
    if (!nnz) {
      if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 10;
      else if (nz <= 0)        nz = 1;
      for (i=0; i<dd->rowblockcount; ++i) aij->imax[i] = nz;
      nz = nz*dd->rowblockcount;
    } else {
      nz = 0;
      for (i=0; i<dd->rowblockcount; ++i) {aij->imax[i] = nnz[i]; nz += nnz[i];}
    }
    /* aij->ilen will count nonzeros in each row so far. */
    for (i=0; i<dd->rowblockcount; i++) { aij->ilen[i] = 0; }

    /* allocate the block space */
    ierr = MatDDMetaAIJ_FreeAIJ(meta,&aij->a,&aij->j,&aij->i);CHKERRQ(ierr);
    ierr = PetscMalloc3(nz,MatDDMeta_Block,&aij->a,nz,PetscInt,&aij->j,dd->rowblockcount+1,PetscInt,&aij->i);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,(dd->rowblockcount+1)*sizeof(PetscInt)+nz*(sizeof(MatDDMeta_Block)+sizeof(PetscInt)));CHKERRQ(ierr);  
    aij->i[0] = 0;
    for (i=1; i<dd->rowblockcount+1; ++i) {
      aij->i[i] = aij->i[i-1] + aij->imax[i-1];
    }
    aij->singlemalloc = PETSC_TRUE;
    aij->free_a       = PETSC_TRUE;
    aij->free_ij      = PETSC_TRUE;
  } else {
    aij->free_a       = PETSC_FALSE;
    aij->free_ij      = PETSC_FALSE;
  }

  aij->nz                = 0;
  aij->maxnz             = nz;
  A->info.nz_unneeded  = (double)aij->maxnz;
  PetscFunctionReturn(0);
}/* MatDDMetaAIJSetPreallocation() */


#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_DDMetaAIJ"
PetscErrorCode  MatSetUpPreallocation_DDMetaAIJ(Mat A) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatDDMetaAIJSetPreallocation(A,PETSC_DEFAULT,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatSetUpPreallocation_DDMetaAIJ() */


#undef  __FUNCT__
#define __FUNCT__ "MatDDMetaLocateBlock_AIJ"
PetscErrorCode  MatDDMetaLocateBlock_AIJ(Mat A, PetscInt row, PetscInt col, PetscBool insert, MatDDMeta_Block **block_pp) {
  PetscErrorCode        ierr;
  Mat_DD         *dd = (Mat_DD*)A->data;
  Mat_DDMetaAIJ  *a = (Mat_DDMetaAIJ*)A->data;
  PetscInt       *rp,low,high,t,ii,nrow,i,rmax,N;
  PetscInt       *imax = a->imax,*ai = a->i,*ailen = a->ilen;
  PetscInt       *aj = a->j,nonew = a->nonew,lastcol = -1;
  Mat_DDBlock   *ap,*aa = a->a;
  PetscFunctionBegin;

  *block_pp = PETSC_NULL;
  if (row < 0) goto we_are_done;
#if defined(PETSC_USE_DEBUG)  
  if (row >= dd->rowblockcount) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Block row too large: row %D max %D",row,dd->rowblockcount-1);
#endif
  rp   = aj + ai[row]; ap = aa + ai[row];
  rmax = imax[row]; nrow = ailen[row]; 
  low  = 0;
  high = nrow;
  
  if (col < 0) goto we_are_done;
#if defined(PETSC_USE_DEBUG)  
  if (col >= dd->colblockcount) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Block columnt too large: col %D max %D",col,dd->colblockcount-1);
#endif

  if (col <= lastcol) low = 0; else high = nrow;
  lastcol = col;
  while (high-low > 5) {
    t = (low+high)/2;
    if (rp[t] > col) high = t;
    else             low  = t;
  }
  for (i=low; i<high; i++) {
    if (rp[i] > col) break;
    if (rp[i] == col) {
      *block_pp = ap+i;  
      goto we_are_done;
    }
  } 
  if (!insert || nonew == 1) goto we_are_done;
  if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new block at (%D,%D) in the matrix",row,col);
  MatDDMetaAIJ_ReallocateAIJ(A,dd->rowblockcount,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew);
  N = nrow++ - 1; a->nz++; high++;
  /* shift up all the later entries in this row */
  for (ii=N; ii>=i; ii--) {
    rp[ii+1] = rp[ii];
    ap[ii+1] = ap[ii];
  }
  rp[i] = col; 
  *block_pp = ap+i; 
  low   = i + 1;
  ailen[row] = nrow;
  A->same_nonzero = PETSC_FALSE;

  if (A->assembled) {
    A->was_assembled = PETSC_TRUE; 
    A->assembled     = PETSC_FALSE;
  }
 we_are_done:;
  PetscFunctionReturn(0);
}/* MatDDMetaLocateBlock_AIJ() */


#undef  __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_DDMetaAIJ"
PetscErrorCode  MatAssemblyBegin_DDMetaAIJ(Mat A, MatAssemblyType type) {
  Mat_DD         *dd = (Mat_DD*)A->data;
  Mat_DDMetaAIJ  *aij = (Mat_DDMetaAIJ*)A->data;
  PetscInt i,j,k;
  Mat B;
  MatDDMeta_Block *ap;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  for(i = 0; i < dd->rowblockcount; ++i) {
    ap = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k, ++ap) {
      j = aij->j[aij->i[i] + k];
      ierr = MatDDMeta_BlockGetMat(A, i, j, ap,&B); CHKERRQ(ierr);
      ierr = MatAssemblyBegin(B, type); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}/* MatAssemblyBegin_DDMetaAIJ() */

#undef  __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_DDMetaAIJ"
PetscErrorCode  MatAssemblyEnd_DDMetaAIJ(Mat A, MatAssemblyType type) {
  Mat_DD     *dd = (Mat_DD*)A->data;
  Mat_DDAIJ  *aij = (Mat_DDMetaAIJ*)A->data;
  PetscInt i,j,k;
  Mat B;
  MaDDMeta_Block *ap;
  PetscErrorCode ierr; 
  PetscFunctionBegin;
  for(i = 0; i < dd->rowblockcount; ++i) {
    ap = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k, ++ap) {
      j = aij->j[aij->i[i] + k];
      ierr = MatDDMeta_BlockGetMat(A,i,j,ap,&B); CHKERRQ(ierr);
      ierr = MatAssemblyEnd(B, type); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}/* MatAssemblyEnd_DDMetaAIJ() */


#undef  __FUNCT__
#define __FUNCT__ "MatMult_DDMetaAIJ"
PetscErrorCode  MatMult_DDAIJ(Mat A, Vec x, Vec y) {
  Mat_DD  *dd = (Mat_DD*)A->data;
  Mat_DDMetaAIJ *aij = (Mat_DDMetaAIJ*)A->data;
  const PetscInt *aj;
  PetscInt i,j,k;
  PetscInt xoff, yoff;
  MatDDMeta_Block *aa, *b;
  PetscScalar *xarr, *yarr;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  
  MatDDMetaAIJ_SetUp(A);
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarr); CHKERRQ(ierr); 
  for (i=0; i<dd->rowblockcount; ++i) {
    yoff = dd->rowblockoffset[i];
    ierr = VecPlaceArray(b->outvec,yarr+yoff); CHKERRQ(ierr); 
    aj  = aij->j + aij->i[i];
    aa  = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k) {
      j = *(aj+k);
      b = aa+k;
      xoff = dd->colblockoffset[j];
      ierr = VecPlaceArray(b->invec,xarr+xoff); CHKERRQ(ierr); 
      ierr = MatMultAdd(b->mat, b->invec, b->outvec, b->outvec); CHKERRQ(ierr);
      ierr = VecResetArray(b->invec); CHKERRQ(ierr);
    }
    ierr = VecResetArray(b->outvec); CHKERRQ(ierr);
  } 
  ierr = VecRestoreArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yarr); CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}/* MatMult_DDAIJ() */


#undef  __FUNCT__
#define __FUNCT__ "MatMultTranspose_DDAIJ"
PetscErrorCode  MatMultTranspose_DDAIJ(Mat A, Vec x, Vec y) {
  Mat_DD  *dd = (Mat_DD*)A->data;
  Mat_DDAIJ *aij = (Mat_DDAIJ*)dd->data;
  const PetscInt *aj;
  PetscInt i,j,k;
  PetscInt xoff, yoff;
  Mat_DDBlock *aa, *b;
  PetscScalar *xarr, *yarr;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  MatDDAIJ_SetUp(A);
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarr); CHKERRQ(ierr); 
  for (i=0; i<dd->rowblockcount; ++i) {
    xoff = dd->rowblockoffset[i];
    ierr = VecPlaceArray(b->outvec,xarr+xoff); CHKERRQ(ierr);
    aj  = aij->j + aij->i[i];
    aa  = aij->a + aij->i[i];
    for(k = 0; k < aij->ilen[i]; ++k) {
      j = *(aj+k);
      b = aa+k;
      yoff = dd->colblockoffset[j];
      ierr = VecPlaceArray(b->invec,yarr+yoff); CHKERRQ(ierr);
      ierr = MatMultTransposeAdd(b->mat, b->outvec, b->invec, b->invec); CHKERRQ(ierr);
      ierr = VecResetArray(b->invec); CHKERRQ(ierr);
    }
    ierr = VecResetArray(b->outvec); CHKERRQ(ierr);
  } 
  ierr = VecRestoreArray(x,&xarr); CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yarr); CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}/* MatMultTranspose_DDAIJ() */


#undef  __FUNCT__
#define __FUNCT__ "MatCreate_DDAIJ"
PetscErrorCode  MatCreate_DDAIJ(Mat A) {
  Mat_DD     *dd = (Mat_DD *)A->data;
  Mat_DDAIJ  *aij;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(A,Mat_DDAIJ,&aij);CHKERRQ(ierr);
  dd->data             = (void*)aij;

  aij->reallocs         = 0;
  aij->nonew            = 0;
  
  A->ops->assemblybegin = MatAssemblyBegin_DDAIJ;
  A->ops->assemblyend   = MatAssemblyEnd_DDAIJ;
  
  PetscFunctionReturn(0);
}/* MatCreate_DDAIJ() */

#undef  __FUNCT__
#define __FUNCT__ "MatDestroy_DDAIJ"
PetscErrorCode  MatDestroy_DDAIJ(Mat M) {
  Mat_DD     *dd = (Mat_DD *)M->data;
  Mat_DDAIJ  *aij = (Mat_DDAIJ *)dd->data;
  PetscInt    *ai = aij->i, *ailen = aij->ilen;
  Mat_DDBlock *aa = aij->a, *bp;
  PetscInt    i,n,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Destroy blocks */
  for(i = 0; i < dd->rowblockcount; ++i) {
    n = ailen[i];
    bp = aa + ai[i];
    for(j = 0; j < n; ++j) {
      ierr = Mat_DDBlockFinit(M, bp); CHKERRQ(ierr);
      ++bp;
    }
  }
  /* Dealloc the AIJ structures */
  ierr = MatDDXAIJFreeAIJ(M, &(aij->a), &(aij->j), &(aij->i)); CHKERRQ(ierr);
  ierr = PetscFree2(aij->imax,aij->ilen);CHKERRQ(ierr);
  ierr = PetscFree(aij);CHKERRQ(ierr);
  dd->data = 0;
  
  PetscFunctionReturn(0);
}/* MatDestroy_DDAIJ() */

/* CONTINUE: MatAssemblyBegin_DDMeta(), MatAssemblyEnd_DDMeta() */


