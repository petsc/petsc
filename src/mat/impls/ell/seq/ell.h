
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
PetscBool   free_rlen;  \
PetscInt    reallocs;          /* number of mallocs done during MatSetValues() \
as more values are set than were prealloced */\
PetscBool   keepnonzeropattern;/* keeps matrix structure same in calls to MatZeroRows()*/ \
PetscBool   ignorezeroentries; \
PetscBool   free_colidx;       /* free the column indices colidx when the matrix is destroyed */ \
PetscBool   free_val;          /* free the numerical values when matrix is destroy */ \
PetscInt    *colidx;           /* column index */                                   \
PetscInt    *diag;             /* pointers to diagonal elements */                  \
PetscInt    nonzerorowcnt;     /* how many rows have nonzero entries */             \
PetscBool   free_diag;         \
datatype    *val;              /* elements including nonzeros and padding zeros */  \
PetscScalar *solve_work;       /* work space used in MatSolve */                    \
IS          row, col, icol;    /* index sets, used for reorderings */ \
PetscBool   pivotinblocks;     /* pivot inside factorization of each diagonal block */ \
Mat         parent             /* set if this matrix was formed with MatDuplicate(...,MAT_SHARE_NONZERO_PATTERN,....);
means that this shares some data structures with the parent including diag, ilen, imax, i, j */

typedef struct {
  SEQELLHEADER(MatScalar);
  MatScalar        *saved_values;             /* location for stashing nonzero values of matrix */

  PetscScalar *idiag,*mdiag,*ssor_work;       /* inverse of diagonal entries, diagonal values and workspace for Eisenstat trick */
  PetscBool   idiagvalid;                     /* current idiag[] and mdiag[] are valid */
  PetscScalar *ibdiag;                        /* inverses of block diagonals */
  PetscBool   ibdiagvalid;                    /* inverses of block diagonals are valid. */
  PetscScalar fshift,omega;                   /* last used omega and fshift */
} Mat_SeqELL;

/*
 Frees the arrays from the XELL matrix types
 */
#undef __FUNCT__
#define __FUNCT__ "MatSeqXELLFreeELL"
PETSC_STATIC_INLINE PetscErrorCode MatSeqXELLFreeELL(Mat AA,MatScalar **val,PetscInt **colidx)
{
  PetscErrorCode ierr;
  Mat_SeqELL     *A = (Mat_SeqELL*) AA->data;
  if (A->singlemalloc) {
    ierr = PetscFree2(*val,*colidx);CHKERRQ(ierr);
  } else {
    if (A->free_val)  {ierr = PetscFree(*val);CHKERRQ(ierr);}
    if (A->free_colidx) {ierr = PetscFree(*colidx);CHKERRQ(ierr);}
  }
  return 0;
}

/*
 Allocates larger arrays (with WIDTH+1) for the XELL matrix types
 This is a macro because it takes the datatype as an argument which can be either a Mat or a MatScalar
 */
#define MatSeqXELLReallocateELL(Amat,AM,BS2,WIDTH,MAXALLOCROW,ROW,COL,CP,VP,NONEW,datatype) \
if (WIDTH >= MAXALLOCROW) { \
Mat_SeqELL *Ain = (Mat_SeqELL*)Amat->data; \
/* there is no extra room in row, therefore enlarge */ \
PetscInt new_size=Ain->maxallocmat+AM,new_shift,*new_colidx; \
datatype *new_val; \
\
if (NONEW == -2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"New nonzero at (%D,%D) caused a malloc\nUse MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) to turn off this check",ROW,COL); \
/* malloc new storage space */ \
ierr = PetscMalloc2(BS2*new_size,&new_val,BS2*new_size,&new_colidx);CHKERRQ(ierr); \
\
/* copy over old data into new slots */ \
for (ii=0;ii<AM>>3;ii++) { ierr = PetscMemcpy(new_val+ii*WIDTH*8,VP+ii*(WIDTH+1)*8,8*WIDTH*sizeof(datatype));CHKERRQ(ierr);ierr = PetscMemcpy(new_colidx+ii*WIDTH*8,CP+ii*(WIDTH+1)*8,8*WIDTH*sizeof(PetscInt));CHKERRQ(ierr);} \
new_shift = (ROW & ~0x07)*(WIDTH+1)+(ROW & 0x07); \
CP = new_colidx+new_shift; \
VP = new_val+new_shift; \
/* free up old matrix storage */ \
ierr              = MatSeqXELLFreeELL(A,&Ain->val,&Ain->colidx);CHKERRQ(ierr); \
Ain->val          = (MatScalar*) new_val;                   \
Ain->colidx       = new_colidx;   \
Ain->singlemalloc = PETSC_TRUE; \
Ain->maxallocmat  = new_size; \
Ain->reallocs++; \
Ain->maxallocrow++; \
Ain->rlenmax++; \
} \

PETSC_INTERN PetscErrorCode MatSeqELLSetPreallocation_SeqELL(Mat,PetscInt,const PetscInt[]);
PETSC_INTERN PetscErrorCode MatMult_SeqELL(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqELL(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMissingDiagonal_SeqELL(Mat,PetscBool*,PetscInt*);
PETSC_INTERN PetscErrorCode MatMarkDiagonal_SeqELL(Mat);
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
PETSC_EXTERN PetscErrorCode MatCreate_SeqELL(Mat);
PETSC_INTERN PetscErrorCode MatDuplicate_SeqELL(Mat,MatDuplicateOption,Mat*);
PETSC_INTERN PetscErrorCode MatEqual_SeqELL(Mat,Mat,PetscBool*);
PETSC_INTERN PetscErrorCode MatSeqELLInvalidateDiagonal(Mat);

#endif
