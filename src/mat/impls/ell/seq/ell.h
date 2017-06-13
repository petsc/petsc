
#if !defined(__ELL_H)
#define __ELL_H

#include <petsc/private/matimpl.h>
#include <petscctable.h>

/*
 Struct header for SeqELL matrix format
 */
#define SEQELLHEADER(datatype) \
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
PetscBool   free_bt;           /* free the bit array */ \
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
char        *bt                /* bit array */

typedef struct {
  SEQELLHEADER(MatScalar);
  MatScalar        *saved_values;             /* location for stashing nonzero values of matrix */

  PetscScalar *idiag,*mdiag,*ssor_work;       /* inverse of diagonal entries, diagonal values and workspace for Eisenstat trick */
  PetscBool   idiagvalid;                     /* current idiag[] and mdiag[] are valid */
  PetscScalar *ibdiag;                        /* inverses of block diagonals */
  PetscBool   ibdiagvalid;                    /* inverses of block diagonals are valid. */
  PetscScalar fshift,omega;                   /* last used omega and fshift */

  ISColoring  coloring;                       /* set with MatADSetColoring() used by MatADSetValues() */
} Mat_SeqELL;

/*
 Frees the arrays from the sliced XELLPACK matrix type
 */
PETSC_STATIC_INLINE PetscErrorCode MatSeqXELLFreeELL(Mat AA,MatScalar **val,PetscInt **colidx,char **bt)
{
  PetscErrorCode ierr;
  Mat_SeqELL     *A = (Mat_SeqELL*) AA->data;
  if (A->singlemalloc) {
    ierr = PetscFree3(*val,*colidx,*bt);CHKERRQ(ierr);
  } else {
    if (A->free_val)  {ierr = PetscFree(*val);CHKERRQ(ierr);}
    if (A->free_colidx) {ierr = PetscFree(*colidx);CHKERRQ(ierr);}
    if (A->free_bt) {ierr = PetscFree(*bt);CHKERRQ(ierr);}
  }
  return 0;
}

/*
 Allocates a larger array for the XELL matrix types; only extend the current slice by one more column.
 This is a macro because it takes the datatype as an argument which can be either a Mat or a MatScalar
 */
#define MatSeqXELLReallocateELL(Amat,AM,BS2,WIDTH,SIDX,SID,ROW,COL,COLIDX,VAL,BT,CP,VP,BP,NONEW,datatype) \
if (WIDTH >= (SIDX[SID+1]-SIDX[SID])/8) { \
Mat_SeqELL *Ain = (Mat_SeqELL*)Amat->data; \
/* there is no extra room in row, therefore enlarge 8 elements (1 slice column) */ \
PetscInt new_size=Ain->maxallocmat+8,*new_colidx; \
char *new_bt; \
datatype *new_val; \
\
if (NONEW == -2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"New nonzero at (%D,%D) caused a malloc\nUse MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) to turn off this check",ROW,COL); \
/* malloc new storage space */ \
ierr = PetscMalloc3(BS2*new_size,&new_val,BS2*new_size,&new_colidx,BS2*new_size/8,&new_bt);CHKERRQ(ierr); \
\
/* copy over old data into new slots by two steps: one step for data before the current slice and the other for the rest */ \
ierr = PetscMemcpy(new_val,VAL,SIDX[SID+1]*sizeof(datatype));CHKERRQ(ierr); \
ierr = PetscMemcpy(new_colidx,COLIDX,SIDX[SID+1]*sizeof(PetscInt));CHKERRQ(ierr); \
ierr = PetscMemcpy(new_bt,BT,SIDX[SID+1]/8*sizeof(char));CHKERRQ(ierr); \
/* set the added mask to zero */ \
*(new_bt+SIDX[SID+1]-1)=0; \
ierr = PetscMemcpy(new_val+SIDX[SID+1]+8,VAL+SIDX[SID+1],(SIDX[AM>>3]-SIDX[SID+1])*sizeof(datatype));CHKERRQ(ierr); \
ierr = PetscMemcpy(new_colidx+SIDX[SID+1]+8,COLIDX+SIDX[SID+1],(SIDX[AM>>3]-SIDX[SID+1])*sizeof(PetscInt));CHKERRQ(ierr); \
ierr = PetscMemcpy(new_bt+SIDX[SID+1]/8+1,BT+SIDX[SID+1]/8,(SIDX[AM>>3]-SIDX[SID+1])/8*sizeof(char));CHKERRQ(ierr); \
/* update slice_idx */ \
for (ii=SID+1;ii<=AM>>3;ii++) { SIDX[ii] += 8; } \
/* update pointers. Notice that they point to the FIRST postion of the row */ \
CP = new_colidx+SIDX[SID]+(ROW & 0x07); \
VP = new_val+SIDX[SID]+(ROW & 0x07); \
BP = new_bt+SIDX[SID]/8; \
/* free up old matrix storage */ \
ierr              = MatSeqXELLFreeELL(A,&Ain->val,&Ain->colidx,&Ain->bt);CHKERRQ(ierr); \
Ain->val          = (MatScalar*) new_val; \
Ain->colidx       = new_colidx; \
Ain->bt           = new_bt; \
Ain->singlemalloc = PETSC_TRUE; \
Ain->maxallocmat  = new_size; \
Ain->reallocs++; \
if (WIDTH>=Ain->maxallocrow) Ain->maxallocrow++; \
if (WIDTH>=Ain->rlenmax) Ain->rlenmax++; \
} \

PETSC_INTERN PetscErrorCode MatSeqELLSetPreallocation_SeqELL(Mat,PetscInt,const PetscInt[]);
PETSC_INTERN PetscErrorCode MatMult_SeqELL(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqELL(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTranspose_SeqELL(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultTransposeAdd_SeqELL(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMissingDiagonal_SeqELL(Mat,PetscBool*,PetscInt*);
PETSC_INTERN PetscErrorCode MatMarkDiagonal_SeqELL(Mat);
PETSC_INTERN PetscErrorCode MatInvertDiagonal_SeqELL(Mat,PetscScalar,PetscScalar);
PETSC_INTERN PetscErrorCode MatZeroEntries_SeqELL(Mat);
PETSC_INTERN PetscErrorCode MatDestroy_SeqELL(Mat);
PETSC_INTERN PetscErrorCode MatSetOption_SeqELL(Mat,MatOption,PetscBool);
PETSC_INTERN PetscErrorCode MatGetDiagonal_SeqELL(Mat,Vec v);
PETSC_INTERN PetscErrorCode MatGetValues_SeqELL(Mat,PetscInt,const PetscInt [],PetscInt,const PetscInt[],PetscScalar[]);
PETSC_INTERN PetscErrorCode MatView_SeqELL(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatAssemblyEnd_SeqELL(Mat,MatAssemblyType);
PETSC_INTERN PetscErrorCode MatGetInfo_SeqELL(Mat,MatInfoType,MatInfo*);
PETSC_INTERN PetscErrorCode MatSetValues_SeqELL(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode MatCopy_SeqELL(Mat,Mat,MatStructure);
PETSC_INTERN PetscErrorCode MatSetUp_SeqELL(Mat);
PETSC_INTERN PetscErrorCode MatSeqELLGetArray_SeqELL(Mat,PetscScalar *[]);
PETSC_INTERN PetscErrorCode MatSeqELLRestoreArray_SeqELL(Mat,PetscScalar *[]);
PETSC_INTERN PetscErrorCode MatShift_SeqELL(Mat,PetscScalar);
PETSC_INTERN PetscErrorCode MatSOR_SeqELL(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);
PETSC_EXTERN PetscErrorCode MatCreate_SeqELL(Mat);
PETSC_INTERN PetscErrorCode MatDuplicate_SeqELL(Mat,MatDuplicateOption,Mat*);
PETSC_INTERN PetscErrorCode MatEqual_SeqELL(Mat,Mat,PetscBool*);
PETSC_INTERN PetscErrorCode MatSeqELLInvalidateDiagonal(Mat);
PETSC_INTERN PetscErrorCode MatConvert_SeqELL_SeqAIJ(Mat,MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqELL(Mat,MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatFDColoringCreate_SeqELL(Mat,ISColoring,MatFDColoring);
PETSC_INTERN PetscErrorCode MatFDColoringSetUp_SeqELL(Mat,ISColoring,MatFDColoring);
PETSC_INTERN PetscErrorCode MatGetColumnIJ_SeqELL_Color(Mat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt *[],const PetscInt *[],PetscInt *[],PetscBool*);
PETSC_INTERN PetscErrorCode MatRestoreColumnIJ_SeqELL_Color(Mat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt *[],const PetscInt *[],PetscInt *[],PetscBool*);
#endif
