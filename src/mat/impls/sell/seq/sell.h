
#if !defined(__SELL_H)
#define __SELL_H

#include <petsc/private/matimpl.h>
#include <petscctable.h>

/*
 Struct header for SeqSELL matrix format
*/
#define SEQSELLHEADER(datatype) \
PetscBool   roworiented;       /* if true, row-oriented input, default */ \
PetscInt    nonew;             /* 1 don't add new nonzeros, -1 generate error on new */ \
PetscInt    nounused;          /* -1 generate error on unused space */ \
PetscBool   singlemalloc;      /* if true a, i, and j have been obtained with one big malloc */ \
PetscInt    maxallocmat;       /* max allocated space for the matrix */ \
PetscInt    maxallocrow;       /* max allocated space for each row */ \
PetscInt    nz;                /* actual nonzeros */  \
PetscInt    rlenmax;           /* max actual row length, rmax cannot exceed maxallocrow */ \
PetscInt    *rlen;             /* actual length of each row (padding zeros excluded) */ \
PetscBool   free_rlen;         /* free rlen array ? */ \
PetscInt    reallocs;          /* number of mallocs done during MatSetValues() \
as more values are set than were prealloced */\
PetscBool   keepnonzeropattern;/* keeps matrix structure same in calls to MatZeroRows()*/ \
PetscBool   ignorezeroentries; \
PetscBool   free_colidx;       /* free the column indices colidx when the matrix is destroyed */ \
PetscBool   free_val;          /* free the numerical values when matrix is destroy */ \
PetscInt    *colidx;           /* column index */ \
PetscInt    *diag;             /* pointers to diagonal elements */ \
PetscInt    nonzerorowcnt;     /* how many rows have nonzero entries */ \
PetscBool   free_diag;         /* free diag ? */ \
datatype    *val;              /* elements including nonzeros and padding zeros */  \
PetscScalar *solve_work;       /* work space used in MatSolve */ \
IS          row, col, icol;    /* index sets, used for reorderings */ \
PetscBool   pivotinblocks;     /* pivot inside factorization of each diagonal block */ \
Mat         parent;            /* set if this matrix was formed with MatDuplicate(...,MAT_SHARE_NONZERO_PATTERN,....);
means that this shares some data structures with the parent including diag, ilen, imax, i, j */ \
PetscInt    *sliidx;           /* slice index */ \
PetscInt    totalslices;       /* total number of slices */ \
PetscInt    *getrowcols;       /* workarray for MatGetRow_SeqSELL */ \
PetscScalar *getrowvals        /* workarray for MatGetRow_SeqSELL */ \

typedef struct {
  SEQSELLHEADER(MatScalar);
  MatScalar   *saved_values;             /* location for stashing nonzero values of matrix */
  PetscScalar *idiag,*mdiag,*ssor_work;  /* inverse of diagonal entries, diagonal values and workspace for Eisenstat trick */
  PetscBool   idiagvalid;                /* current idiag[] and mdiag[] are valid */
  PetscScalar fshift,omega;              /* last used omega and fshift */
  ISColoring  coloring;                  /* set with MatADSetColoring() used by MatADSetValues() */
} Mat_SeqSELL;

/*
 Frees the arrays from the XSELLPACK matrix type
 */
static inline PetscErrorCode MatSeqXSELLFreeSELL(Mat AA,MatScalar **val,PetscInt **colidx)
{
  Mat_SeqSELL    *A = (Mat_SeqSELL*) AA->data;
  PetscErrorCode ierr;
  if (A->singlemalloc) {
    ierr = PetscFree2(*val,*colidx);CHKERRQ(ierr);
  } else {
    if (A->free_val) {ierr = PetscFree(*val);CHKERRQ(ierr);}
    if (A->free_colidx) {ierr = PetscFree(*colidx);CHKERRQ(ierr);}
  }
  return 0;
}

#define MatSeqXSELLReallocateSELL(Amat,AM,BS2,WIDTH,SIDX,SID,ROW,COL,COLIDX,VAL,CP,VP,NONEW,datatype) \
if (WIDTH >= (SIDX[SID+1]-SIDX[SID])/8) { \
Mat_SeqSELL *Ain = (Mat_SeqSELL*)Amat->data; \
/* there is no extra room in row, therefore enlarge 8 elements (1 slice column) */ \
PetscInt new_size=Ain->maxallocmat+8,*new_colidx; \
datatype *new_val; \
\
PetscAssertFalse(NONEW == -2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"New nonzero at (%" PetscInt_FMT ",%" PetscInt_FMT ") caused a malloc\nUse MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) to turn off this check",ROW,COL); \
/* malloc new storage space */ \
ierr = PetscMalloc2(BS2*new_size,&new_val,BS2*new_size,&new_colidx);CHKERRQ(ierr); \
\
/* copy over old data into new slots by two steps: one step for data before the current slice and the other for the rest */ \
ierr = PetscArraycpy(new_val,VAL,SIDX[SID+1]);CHKERRQ(ierr); \
ierr = PetscArraycpy(new_colidx,COLIDX,SIDX[SID+1]);CHKERRQ(ierr); \
ierr = PetscArraycpy(new_val+SIDX[SID+1]+8,VAL+SIDX[SID+1],SIDX[AM>>3]-SIDX[SID+1]);CHKERRQ(ierr); \
ierr = PetscArraycpy(new_colidx+SIDX[SID+1]+8,COLIDX+SIDX[SID+1],SIDX[AM>>3]-SIDX[SID+1]);CHKERRQ(ierr); \
/* update slice_idx */ \
for (ii=SID+1;ii<=AM>>3;ii++) { SIDX[ii] += 8; } \
/* update pointers. Notice that they point to the FIRST postion of the row */ \
CP = new_colidx+SIDX[SID]+(ROW & 0x07); \
VP = new_val+SIDX[SID]+(ROW & 0x07); \
/* free up old matrix storage */ \
ierr              = MatSeqXSELLFreeSELL(A,&Ain->val,&Ain->colidx);CHKERRQ(ierr); \
Ain->val          = (MatScalar*) new_val; \
Ain->colidx       = new_colidx; \
Ain->singlemalloc = PETSC_TRUE; \
Ain->maxallocmat  = new_size; \
Ain->reallocs++; \
if (WIDTH>=Ain->maxallocrow) Ain->maxallocrow++; \
if (WIDTH>=Ain->rlenmax) Ain->rlenmax++; \
} \

#define MatSetValue_SeqSELL_Private(A,row,col,value,addv,orow,ocol,cp,vp,lastcol,low,high) \
{ \
  Mat_SeqSELL  *a=(Mat_SeqSELL*)A->data; \
  found=PETSC_FALSE; \
  if (col <= lastcol) low = 0; \
  else high = a->rlen[row]; \
  lastcol = col; \
  while (high-low > 5) { \
    t = (low+high)/2; \
    if (*(cp+8*t) > col) high = t; \
    else low = t; \
  } \
  for (_i=low; _i<high; _i++) { \
    if (*(cp+8*_i) > col) break; \
    if (*(cp+8*_i) == col) { \
      if (addv == ADD_VALUES)*(vp+8*_i) += value; \
      else *(vp+8*_i) = value; \
      found = PETSC_TRUE; \
      break; \
    } \
  } \
  if (!found) { \
    PetscAssertFalse(a->nonew == -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", orow, ocol); \
    if (a->nonew != 1 && !(value == 0.0 && a->ignorezeroentries) && a->rlen[row] >= (a->sliidx[row/8+1]-a->sliidx[row/8])/8) { \
      /* there is no extra room in row, therefore enlarge 8 elements (1 slice column) */ \
      if (a->maxallocmat < a->sliidx[a->totalslices]+8) { \
        /* allocates a larger array for the XSELL matrix types; only extend the current slice by one more column. */ \
        PetscInt  new_size=a->maxallocmat+8,*new_colidx; \
        MatScalar *new_val; \
        PetscAssertFalse(a->nonew == -2,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"New nonzero at (%" PetscInt_FMT ",%" PetscInt_FMT ") caused a malloc\nUse MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) to turn off this check",orow,ocol); \
        /* malloc new storage space */ \
        ierr = PetscMalloc2(new_size,&new_val,new_size,&new_colidx);CHKERRQ(ierr); \
        /* copy over old data into new slots by two steps: one step for data before the current slice and the other for the rest */ \
        ierr = PetscArraycpy(new_val,a->val,a->sliidx[row/8+1]);CHKERRQ(ierr); \
        ierr = PetscArraycpy(new_colidx,a->colidx,a->sliidx[row/8+1]);CHKERRQ(ierr); \
        ierr = PetscArraycpy(new_val+a->sliidx[row/8+1]+8,a->val+a->sliidx[row/8+1],a->sliidx[a->totalslices]-a->sliidx[row/8+1]);CHKERRQ(ierr);  \
        ierr = PetscArraycpy(new_colidx+a->sliidx[row/8+1]+8,a->colidx+a->sliidx[row/8+1],a->sliidx[a->totalslices]-a->sliidx[row/8+1]);CHKERRQ(ierr); \
        /* update pointers. Notice that they point to the FIRST postion of the row */ \
        cp = new_colidx+a->sliidx[row/8]+(row & 0x07); \
        vp = new_val+a->sliidx[row/8]+(row & 0x07); \
        /* free up old matrix storage */ \
        ierr            = MatSeqXSELLFreeSELL(A,&a->val,&a->colidx);CHKERRQ(ierr); \
        a->val          = (MatScalar*)new_val; \
        a->colidx       = new_colidx; \
        a->singlemalloc = PETSC_TRUE; \
        a->maxallocmat  = new_size; \
        a->reallocs++; \
      } else { \
        /* no need to reallocate, just shift the following slices to create space for the added slice column */ \
        ierr = PetscArraymove(a->val+a->sliidx[row/8+1]+8,a->val+a->sliidx[row/8+1],a->sliidx[a->totalslices]-a->sliidx[row/8+1]);CHKERRQ(ierr);  \
        ierr = PetscArraymove(a->colidx+a->sliidx[row/8+1]+8,a->colidx+a->sliidx[row/8+1],a->sliidx[a->totalslices]-a->sliidx[row/8+1]);CHKERRQ(ierr); \
      } \
      /* update slice_idx */ \
      for (ii=row/8+1;ii<=a->totalslices;ii++) a->sliidx[ii] += 8; \
      if (a->rlen[row]>=a->maxallocrow) a->maxallocrow++; \
      if (a->rlen[row]>=a->rlenmax) a->rlenmax++; \
    } \
    /* shift up all the later entries in this row */ \
    for (ii=a->rlen[row]-1; ii>=_i; ii--) { \
      *(cp+8*(ii+1)) = *(cp+8*ii); \
      *(vp+8*(ii+1)) = *(vp+8*ii); \
    } \
    *(cp+8*_i) = col; \
    *(vp+8*_i) = value; \
    a->nz++; a->rlen[row]++; A->nonzerostate++; \
    low = _i+1; high++; \
  } \
} \

PETSC_INTERN PetscErrorCode MatSeqSELLSetPreallocation_SeqSELL(Mat,PetscInt,const PetscInt[]);
PETSC_INTERN PetscErrorCode MatMult_SeqSELL(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqSELL(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTranspose_SeqSELL(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTransposeAdd_SeqSELL(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMissingDiagonal_SeqSELL(Mat,PetscBool*,PetscInt*);
PETSC_INTERN PetscErrorCode MatMarkDiagonal_SeqSELL(Mat);
PETSC_INTERN PetscErrorCode MatInvertDiagonal_SeqSELL(Mat,PetscScalar,PetscScalar);
PETSC_INTERN PetscErrorCode MatZeroEntries_SeqSELL(Mat);
PETSC_INTERN PetscErrorCode MatDestroy_SeqSELL(Mat);
PETSC_INTERN PetscErrorCode MatSetOption_SeqSELL(Mat,MatOption,PetscBool);
PETSC_INTERN PetscErrorCode MatGetDiagonal_SeqSELL(Mat,Vec v);
PETSC_INTERN PetscErrorCode MatGetValues_SeqSELL(Mat,PetscInt,const PetscInt [],PetscInt,const PetscInt[],PetscScalar[]);
PETSC_INTERN PetscErrorCode MatView_SeqSELL(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatAssemblyEnd_SeqSELL(Mat,MatAssemblyType);
PETSC_INTERN PetscErrorCode MatGetInfo_SeqSELL(Mat,MatInfoType,MatInfo*);
PETSC_INTERN PetscErrorCode MatSetValues_SeqSELL(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode MatCopy_SeqSELL(Mat,Mat,MatStructure);
PETSC_INTERN PetscErrorCode MatSetUp_SeqSELL(Mat);
PETSC_INTERN PetscErrorCode MatSeqSELLGetArray_SeqSELL(Mat,PetscScalar *[]);
PETSC_INTERN PetscErrorCode MatSeqSELLRestoreArray_SeqSELL(Mat,PetscScalar *[]);
PETSC_INTERN PetscErrorCode MatShift_SeqSELL(Mat,PetscScalar);
PETSC_INTERN PetscErrorCode MatSOR_SeqSELL(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode MatCreate_SeqSELL(Mat);
PETSC_INTERN PetscErrorCode MatDuplicate_SeqSELL(Mat,MatDuplicateOption,Mat*);
PETSC_INTERN PetscErrorCode MatEqual_SeqSELL(Mat,Mat,PetscBool*);
PETSC_INTERN PetscErrorCode MatSeqSELLInvalidateDiagonal(Mat);
PETSC_INTERN PetscErrorCode MatConvert_SeqSELL_SeqAIJ(Mat,MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqSELL(Mat,MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatFDColoringCreate_SeqSELL(Mat,ISColoring,MatFDColoring);
PETSC_INTERN PetscErrorCode MatFDColoringSetUp_SeqSELL(Mat,ISColoring,MatFDColoring);
PETSC_INTERN PetscErrorCode MatGetColumnIJ_SeqSELL_Color(Mat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt *[],const PetscInt *[],PetscInt *[],PetscBool*);
PETSC_INTERN PetscErrorCode MatRestoreColumnIJ_SeqSELL_Color(Mat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt *[],const PetscInt *[],PetscInt *[],PetscBool*);
PETSC_INTERN PetscErrorCode MatConjugate_SeqSELL(Mat A);
PETSC_INTERN PetscErrorCode MatScale_SeqSELL(Mat,PetscScalar);
PETSC_INTERN PetscErrorCode MatDiagonalScale_SeqSELL(Mat,Vec,Vec);
#endif
