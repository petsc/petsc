
#if !defined(__AIJ_H)
#define __AIJ_H

#include <petsc-private/matimpl.h>

/*  
    Struct header shared by SeqAIJ, SeqBAIJ and SeqSBAIJ matrix formats
*/
#define SEQAIJHEADER(datatype)	\
  PetscBool         roworiented;      /* if true, row-oriented input, default */\
  PetscInt          nonew;            /* 1 don't add new nonzeros, -1 generate error on new */\
  PetscInt          nounused;         /* -1 generate error on unused space */\
  PetscBool         singlemalloc;     /* if true a, i, and j have been obtained with one big malloc */\
  PetscInt          maxnz;            /* allocated nonzeros */\
  PetscInt          *imax;            /* maximum space allocated for each row */\
  PetscInt          *ilen;            /* actual length of each row */\
  PetscBool         free_imax_ilen;  \
  PetscInt          reallocs;         /* number of mallocs done during MatSetValues() \
                                        as more values are set than were prealloced */\
  PetscInt          rmax;             /* max nonzeros in any row */\
  PetscBool         keepnonzeropattern;   /* keeps matrix structure same in calls to MatZeroRows()*/\
  PetscBool         ignorezeroentries; \
  PetscInt          *xtoy,*xtoyB;     /* map nonzero pattern of X into Y's, used by MatAXPY() */\
  Mat               XtoY;             /* used by MatAXPY() */\
  PetscBool         free_ij;          /* free the column indices j and row offsets i when the matrix is destroyed */ \
  PetscBool         free_a;           /* free the numerical values when matrix is destroy */ \
  Mat_CompressedRow compressedrow;    /* use compressed row format */                      \
  PetscInt          nz;               /* nonzeros */                                       \
  PetscInt          *i;               /* pointer to beginning of each row */               \
  PetscInt          *j;               /* column values: j + i[k] - 1 is start of row k */  \
  PetscInt          *diag;            /* pointers to diagonal elements */                  \
  PetscBool         free_diag;         \
  datatype          *a;               /* nonzero elements */                               \
  PetscScalar       *solve_work;      /* work space used in MatSolve */                    \
  IS                row, col, icol;   /* index sets, used for reorderings */ \
  PetscBool         pivotinblocks;    /* pivot inside factorization of each diagonal block */ \
  Mat               parent             /* set if this matrix was formed with MatDuplicate(...,MAT_SHARE_NONZERO_PATTERN,....); 
                                         means that this shares some data structures with the parent including diag, ilen, imax, i, j */

typedef struct {
  MatTransposeColoring      matcoloring;
  Mat                       Bt_den;  /* dense matrix of B^T */
  Mat                       ABt_den; /* dense matrix of A*B^T */
  PetscBool                 usecoloring; 
  PetscErrorCode (*destroy)(Mat);
} Mat_MatMatTransMult;

typedef struct {
  PetscInt       *api,*apj;    /* symbolic structure of A*P */
  PetscScalar    *apa;         /* temporary array for storing one row of A*P */
  PetscErrorCode (*destroy)(Mat);
} Mat_PtAP;

typedef struct {
  MatTransposeColoring matcoloring;
  Mat                  Rt;    /* dense matrix of R^T */
  Mat                  RARt;  /* dense matrix of R*A*R^T */
  MatScalar            *work; /* work array to store columns of A*R^T used in MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqDense() */
  PetscErrorCode (*destroy)(Mat);
} Mat_RARt;

/*  
  MATSEQAIJ format - Compressed row storage (also called Yale sparse matrix
  format) or compressed sparse row (CSR).  The i[] and j[] arrays start at 0. For example,
  j[i[k]+p] is the pth column in row k.  Note that the diagonal
  matrix elements are stored with the rest of the nonzeros (not separately).
*/

/* Info about i-nodes (identical nodes) helper class for SeqAIJ */
typedef struct {
  MatScalar   *bdiag,*ibdiag,*ssor_work;      /* diagonal blocks of matrix used for MatSOR_SeqAIJ_Inode() */
  PetscInt    bdiagsize;                       /* length of bdiag and ibdiag */
  PetscBool   ibdiagvalid;                     /* do ibdiag[] and bdiag[] contain the most recent values */

  PetscBool  use;
  PetscInt   node_count;                    /* number of inodes */
  PetscInt   *size;                         /* size of each inode */
  PetscInt   limit;                         /* inode limit */
  PetscInt   max_limit;                     /* maximum supported inode limit */
  PetscBool  checked;                       /* if inodes have been checked for */
} Mat_SeqAIJ_Inode;

extern PetscErrorCode MatView_SeqAIJ_Inode(Mat,PetscViewer);
extern PetscErrorCode MatAssemblyEnd_SeqAIJ_Inode(Mat,MatAssemblyType);
extern PetscErrorCode MatDestroy_SeqAIJ_Inode(Mat);
extern PetscErrorCode MatCreate_SeqAIJ_Inode(Mat);
extern PetscErrorCode MatSetOption_SeqAIJ_Inode(Mat,MatOption,PetscBool );
extern PetscErrorCode MatDuplicate_SeqAIJ_Inode(Mat,MatDuplicateOption,Mat*);
extern PetscErrorCode MatDuplicateNoCreate_SeqAIJ(Mat,Mat,MatDuplicateOption,PetscBool );
extern PetscErrorCode MatLUFactorNumeric_SeqAIJ_Inode_inplace(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode MatLUFactorNumeric_SeqAIJ_Inode(Mat,Mat,const MatFactorInfo*);

typedef struct {
  SEQAIJHEADER(MatScalar);
  Mat_SeqAIJ_Inode inode;
  MatScalar        *saved_values;             /* location for stashing nonzero values of matrix */

  PetscScalar      *idiag,*mdiag,*ssor_work;  /* inverse of diagonal entries, diagonal values and workspace for Eisenstat trick */
  PetscBool        idiagvalid;                     /* current idiag[] and mdiag[] are valid */
  PetscScalar      *ibdiag;                   /* inverses of block diagonals */
  PetscBool        ibdiagvalid;               /* inverses of block diagonals are valid. */
  PetscScalar      fshift,omega;                   /* last used omega and fshift */

  ISColoring       coloring;                  /* set with MatADSetColoring() used by MatADSetValues() */
  
  PetscScalar      *matmult_abdense;     /* used by MatMatMult() */
  Mat_PtAP         *ptap;                /* used by MatPtAP() */
} Mat_SeqAIJ;

/*
  Frees the a, i, and j arrays from the XAIJ (AIJ, BAIJ, and SBAIJ) matrix types
*/
#undef __FUNCT__  
#define __FUNCT__ "MatSeqXAIJFreeAIJ"
PETSC_STATIC_INLINE PetscErrorCode MatSeqXAIJFreeAIJ(Mat AA,MatScalar **a,PetscInt **j,PetscInt **i)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *A = (Mat_SeqAIJ*) AA->data;
  if (A->singlemalloc) {
    ierr = PetscFree3(*a,*j,*i);CHKERRQ(ierr);
  } else {
    if (A->free_a)  {ierr = PetscFree(*a);CHKERRQ(ierr);}
    if (A->free_ij) {ierr = PetscFree(*j);CHKERRQ(ierr);}
    if (A->free_ij) {ierr = PetscFree(*i);CHKERRQ(ierr);}
  }
  return 0;
}
/*
    Allocates larger a, i, and j arrays for the XAIJ (AIJ, BAIJ, and SBAIJ) matrix types
    This is a macro because it takes the datatype as an argument which can be either a Mat or a MatScalar
*/
#define MatSeqXAIJReallocateAIJ(Amat,AM,BS2,NROW,ROW,COL,RMAX,AA,AI,AJ,RP,AP,AIMAX,NONEW,datatype) \
  if (NROW >= RMAX) {\
	Mat_SeqAIJ *Ain = (Mat_SeqAIJ*)Amat->data;\
        /* there is no extra room in row, therefore enlarge */ \
        PetscInt   CHUNKSIZE = 15,new_nz = AI[AM] + CHUNKSIZE,len,*new_i=0,*new_j=0; \
        datatype   *new_a; \
 \
        if (NONEW == -2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"New nonzero at (%D,%D) caused a malloc",ROW,COL); \
        /* malloc new storage space */ \
        ierr = PetscMalloc3(BS2*new_nz,datatype,&new_a,new_nz,PetscInt,&new_j,AM+1,PetscInt,&new_i);CHKERRQ(ierr);\
 \
        /* copy over old data into new slots */ \
        for (ii=0; ii<ROW+1; ii++) {new_i[ii] = AI[ii];} \
        for (ii=ROW+1; ii<AM+1; ii++) {new_i[ii] = AI[ii]+CHUNKSIZE;} \
        ierr = PetscMemcpy(new_j,AJ,(AI[ROW]+NROW)*sizeof(PetscInt));CHKERRQ(ierr); \
        len = (new_nz - CHUNKSIZE - AI[ROW] - NROW); \
        ierr = PetscMemcpy(new_j+AI[ROW]+NROW+CHUNKSIZE,AJ+AI[ROW]+NROW,len*sizeof(PetscInt));CHKERRQ(ierr); \
        ierr = PetscMemcpy(new_a,AA,BS2*(AI[ROW]+NROW)*sizeof(datatype));CHKERRQ(ierr); \
        ierr = PetscMemzero(new_a+BS2*(AI[ROW]+NROW),BS2*CHUNKSIZE*sizeof(datatype));CHKERRQ(ierr);\
        ierr = PetscMemcpy(new_a+BS2*(AI[ROW]+NROW+CHUNKSIZE),AA+BS2*(AI[ROW]+NROW),BS2*len*sizeof(datatype));CHKERRQ(ierr);  \
        /* free up old matrix storage */ \
        ierr = MatSeqXAIJFreeAIJ(A,&Ain->a,&Ain->j,&Ain->i);CHKERRQ(ierr);\
        AA = new_a; \
        Ain->a = (MatScalar*) new_a;		   \
        AI = Ain->i = new_i; AJ = Ain->j = new_j;  \
        Ain->singlemalloc = PETSC_TRUE; \
 \
        RP          = AJ + AI[ROW]; AP = AA + BS2*AI[ROW]; \
        RMAX        = AIMAX[ROW] = AIMAX[ROW] + CHUNKSIZE; \
        Ain->maxnz += BS2*CHUNKSIZE; \
        Ain->reallocs++; \
      } \


EXTERN_C_BEGIN
extern PetscErrorCode MatSeqAIJSetPreallocation_SeqAIJ(Mat,PetscInt,const PetscInt*);
EXTERN_C_END
extern PetscErrorCode MatILUFactorSymbolic_SeqAIJ_inplace(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode MatILUFactorSymbolic_SeqAIJ(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode MatILUFactorSymbolic_SeqAIJ_ilu0(Mat,Mat,IS,IS,const MatFactorInfo*);

extern PetscErrorCode MatICCFactorSymbolic_SeqAIJ_inplace(Mat,Mat,IS,const MatFactorInfo*);
extern PetscErrorCode MatICCFactorSymbolic_SeqAIJ(Mat,Mat,IS,const MatFactorInfo*);
extern PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJ_inplace(Mat,Mat,IS,const MatFactorInfo*);
extern PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJ(Mat,Mat,IS,const MatFactorInfo*);
extern PetscErrorCode MatCholeskyFactorNumeric_SeqAIJ_inplace(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode MatCholeskyFactorNumeric_SeqAIJ(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode MatDuplicate_SeqAIJ(Mat,MatDuplicateOption,Mat*);
extern PetscErrorCode MatCopy_SeqAIJ(Mat,Mat,MatStructure);
extern PetscErrorCode MatMissingDiagonal_SeqAIJ(Mat,PetscBool *,PetscInt*);
extern PetscErrorCode MatMarkDiagonal_SeqAIJ(Mat);
extern PetscErrorCode MatFindZeroDiagonals_SeqAIJ_Private(Mat,PetscInt*,PetscInt**);

extern PetscErrorCode MatMult_SeqAIJ(Mat A,Vec,Vec);
extern PetscErrorCode MatMultAdd_SeqAIJ(Mat A,Vec,Vec,Vec);
extern PetscErrorCode MatMultTranspose_SeqAIJ(Mat A,Vec,Vec);
extern PetscErrorCode MatMultTransposeAdd_SeqAIJ(Mat A,Vec,Vec,Vec);
extern PetscErrorCode MatSOR_SeqAIJ(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);

extern PetscErrorCode MatSetColoring_SeqAIJ(Mat,ISColoring);
extern PetscErrorCode MatSetValuesAdic_SeqAIJ(Mat,void*);
extern PetscErrorCode MatSetValuesAdifor_SeqAIJ(Mat,PetscInt,void*);

extern PetscErrorCode MatGetSymbolicTranspose_SeqAIJ(Mat,PetscInt *[],PetscInt *[]);
extern PetscErrorCode MatGetSymbolicTransposeReduced_SeqAIJ(Mat,PetscInt,PetscInt,PetscInt *[],PetscInt *[]);
extern PetscErrorCode MatRestoreSymbolicTranspose_SeqAIJ(Mat,PetscInt *[],PetscInt *[]);
extern PetscErrorCode MatTransposeSymbolic_SeqAIJ(Mat,Mat*);
extern PetscErrorCode MatToSymmetricIJ_SeqAIJ(PetscInt,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt**,PetscInt**);
extern PetscErrorCode MatLUFactorSymbolic_SeqAIJ_inplace(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode MatLUFactorSymbolic_SeqAIJ(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode MatLUFactorNumeric_SeqAIJ_inplace(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode MatLUFactorNumeric_SeqAIJ(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode MatLUFactorNumeric_SeqAIJ_InplaceWithPerm(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode MatLUFactor_SeqAIJ(Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode MatSolve_SeqAIJ_inplace(Mat,Vec,Vec);
extern PetscErrorCode MatSolve_SeqAIJ(Mat,Vec,Vec);
extern PetscErrorCode MatSolve_SeqAIJ_Inode_inplace(Mat,Vec,Vec);
extern PetscErrorCode MatSolve_SeqAIJ_Inode(Mat,Vec,Vec);
extern PetscErrorCode MatSolve_SeqAIJ_NaturalOrdering_inplace(Mat,Vec,Vec);
extern PetscErrorCode MatSolve_SeqAIJ_NaturalOrdering(Mat,Vec,Vec);
extern PetscErrorCode MatSolve_SeqAIJ_InplaceWithPerm(Mat,Vec,Vec);
extern PetscErrorCode MatSolveAdd_SeqAIJ_inplace(Mat,Vec,Vec,Vec);
extern PetscErrorCode MatSolveAdd_SeqAIJ(Mat,Vec,Vec,Vec);
extern PetscErrorCode MatSolveTranspose_SeqAIJ_inplace(Mat,Vec,Vec);
extern PetscErrorCode MatSolveTranspose_SeqAIJ(Mat,Vec,Vec);
extern PetscErrorCode MatSolveTransposeAdd_SeqAIJ_inplace(Mat,Vec,Vec,Vec);
extern PetscErrorCode MatSolveTransposeAdd_SeqAIJ(Mat,Vec,Vec,Vec);
extern PetscErrorCode MatMatSolve_SeqAIJ_inplace(Mat,Mat,Mat);
extern PetscErrorCode MatMatSolve_SeqAIJ(Mat,Mat,Mat);
extern PetscErrorCode MatEqual_SeqAIJ(Mat A,Mat B,PetscBool * flg);
extern PetscErrorCode MatFDColoringCreate_SeqAIJ(Mat,ISColoring,MatFDColoring);
extern PetscErrorCode MatLoad_SeqAIJ(Mat,PetscViewer);
extern PetscErrorCode RegisterApplyPtAPRoutines_Private(Mat);
extern PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable_fast(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ(Mat,Mat,Mat);
extern PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(Mat,Mat,Mat);

extern PetscErrorCode MatPtAPSymbolic_SeqAIJ(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatPtAPNumeric_SeqAIJ(Mat,Mat,Mat);
extern PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqAIJ(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqAIJ_SparseAxpy2(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ(Mat,Mat,Mat);
extern PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy(Mat,Mat,Mat);
extern PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy2(Mat,Mat,Mat);

extern PetscErrorCode MatRARtSymbolic_SeqAIJ_SeqAIJ(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatRARtNumeric_SeqAIJ_SeqAIJ(Mat,Mat,Mat);

extern PetscErrorCode MatTransposeMatMult_SeqAIJ_SeqAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode MatTransposeMatMultSymbolic_SeqAIJ_SeqAIJ(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatTransposeMatMultNumeric_SeqAIJ_SeqAIJ(Mat,Mat,Mat);
extern PetscErrorCode MatMatTransposeMult_SeqAIJ_SeqAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode MatMatTransposeMultSymbolic_SeqAIJ_SeqAIJ(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatMatTransposeMultNumeric_SeqAIJ_SeqAIJ(Mat,Mat,Mat);
extern PetscErrorCode MatTransposeColoringCreate_SeqAIJ(Mat,ISColoring,MatTransposeColoring);
extern PetscErrorCode MatTransColoringApplySpToDen_SeqAIJ(MatTransposeColoring,Mat,Mat);
extern PetscErrorCode MatTransColoringApplyDenToSp_SeqAIJ(MatTransposeColoring,Mat,Mat);

extern PetscErrorCode MatSetValues_SeqAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
extern PetscErrorCode MatGetRow_SeqAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
extern PetscErrorCode MatRestoreRow_SeqAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
extern PetscErrorCode MatAXPY_SeqAIJ(Mat,PetscScalar,Mat,MatStructure);
extern PetscErrorCode MatGetRowIJ_SeqAIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt*,PetscInt *[],PetscInt *[],PetscBool  *);
extern PetscErrorCode MatRestoreRowIJ_SeqAIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt *,PetscInt *[],PetscInt *[],PetscBool  *);
extern PetscErrorCode MatGetColumnIJ_SeqAIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt*,PetscInt *[],PetscInt *[],PetscBool  *);
extern PetscErrorCode MatRestoreColumnIJ_SeqAIJ(Mat,PetscInt,PetscBool ,PetscBool ,PetscInt *,PetscInt *[],PetscInt *[],PetscBool  *);
extern PetscErrorCode MatDestroy_SeqAIJ(Mat);
extern PetscErrorCode MatSetUp_SeqAIJ(Mat);
extern PetscErrorCode MatView_SeqAIJ(Mat,PetscViewer);

extern PetscErrorCode MatSeqAIJInvalidateDiagonal(Mat);
extern PetscErrorCode MatSeqAIJInvalidateDiagonal_Inode(Mat);
extern PetscErrorCode Mat_CheckInode(Mat,PetscBool );
extern PetscErrorCode Mat_CheckInode_FactorLU(Mat,PetscBool );

extern PetscErrorCode MatAXPYGetPreallocation_SeqAIJ(Mat,Mat,PetscInt*);

EXTERN_C_BEGIN
extern PetscErrorCode  MatConvert_SeqAIJ_SeqSBAIJ(Mat,const MatType,MatReuse,Mat*);
extern PetscErrorCode  MatConvert_SeqAIJ_SeqBAIJ(Mat,const MatType,MatReuse,Mat*);
extern PetscErrorCode  MatConvert_SeqAIJ_SeqAIJPERM(Mat,const MatType,MatReuse,Mat*);
extern PetscErrorCode  MatReorderForNonzeroDiagonal_SeqAIJ(Mat,PetscReal,IS,IS);
extern PetscErrorCode  MatMatMult_SeqAIJ_SeqAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode  MatMatMult_SeqDense_SeqAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode  MatRARt_SeqAIJ_SeqAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode  MatCreate_SeqAIJ(Mat);
EXTERN_C_END
extern PetscErrorCode  MatAssemblyEnd_SeqAIJ(Mat,MatAssemblyType);
extern PetscErrorCode  MatDestroy_SeqAIJ(Mat);


/*
    PetscSparseDenseMinusDot - The inner kernel of triangular solves and Gauss-Siedel smoothing. \sum_i xv[i] * r[xi[i]] for CSR storage

  Input Parameters:
+  nnz - the number of entries
.  r - the array of vector values
.  xv - the matrix values for the row
-  xi - the column indices of the nonzeros in the row

  Output Parameter:
.  sum - negative the sum of results

  PETSc compile flags:
+   PETSC_KERNEL_USE_UNROLL_4 -   don't use this; it changes nnz and hence is WRONG
-   PETSC_KERNEL_USE_UNROLL_2 -

.seealso: PetscSparseDensePlusDot()

*/
#ifdef PETSC_KERNEL_USE_UNROLL_4
#define PetscSparseDenseMinusDot(sum,r,xv,xi,nnz) {\
if (nnz > 0) {\
switch (nnz & 0x3) {\
case 3: sum -= *xv++ * r[*xi++];\
case 2: sum -= *xv++ * r[*xi++];\
case 1: sum -= *xv++ * r[*xi++];\
nnz -= 4;}\
while (nnz > 0) {\
sum -=  xv[0] * r[xi[0]] - xv[1] * r[xi[1]] -\
	xv[2] * r[xi[2]] - xv[3] * r[xi[3]];\
xv  += 4; xi += 4; nnz -= 4; }}}

#elif defined(PETSC_KERNEL_USE_UNROLL_2)
#define PetscSparseDenseMinusDot(sum,r,xv,xi,nnz) {\
PetscInt __i,__i1,__i2;\
for(__i=0;__i<nnz-1;__i+=2) {__i1 = xi[__i]; __i2=xi[__i+1];\
sum -= (xv[__i]*r[__i1] + xv[__i+1]*r[__i2]);}\
if (nnz & 0x1) sum -= xv[__i] * r[xi[__i]];}

#else
#define PetscSparseDenseMinusDot(sum,r,xv,xi,nnz) {\
PetscInt __i;\
for(__i=0;__i<nnz;__i++) sum -= xv[__i] * r[xi[__i]];}
#endif



/*
    PetscSparseDensePlusDot - The inner kernel of matrix-vector product \sum_i xv[i] * r[xi[i]] for CSR storage

  Input Parameters:
+  nnz - the number of entries
.  r - the array of vector values
.  xv - the matrix values for the row
-  xi - the column indices of the nonzeros in the row

  Output Parameter:
.  sum - the sum of results

  PETSc compile flags:
+   PETSC_KERNEL_USE_UNROLL_4 -  don't use this; it changes nnz and hence is WRONG
-   PETSC_KERNEL_USE_UNROLL_2 -

.seealso: PetscSparseDenseMinusDot()

*/
#ifdef PETSC_KERNEL_USE_UNROLL_4
#define PetscSparseDensePlusDot(sum,r,xv,xi,nnz) {\
if (nnz > 0) {\
switch (nnz & 0x3) {\
case 3: sum += *xv++ * r[*xi++];\
case 2: sum += *xv++ * r[*xi++];\
case 1: sum += *xv++ * r[*xi++];\
nnz -= 4;}\
while (nnz > 0) {\
sum +=  xv[0] * r[xi[0]] + xv[1] * r[xi[1]] +\
	xv[2] * r[xi[2]] + xv[3] * r[xi[3]];\
xv  += 4; xi += 4; nnz -= 4; }}}

#elif defined(PETSC_KERNEL_USE_UNROLL_2)
#define PetscSparseDensePlusDot(sum,r,xv,xi,nnz) {\
PetscInt __i,__i1,__i2;\
for(__i=0;__i<nnz-1;__i+=2) {__i1 = xi[__i]; __i2=xi[__i+1];\
sum += (xv[__i]*r[__i1] + xv[__i+1]*r[__i2]);}\
if (nnz & 0x1) sum += xv[__i] * r[xi[__i]];}

#else
#define PetscSparseDensePlusDot(sum,r,xv,xi,nnz) {\
 PetscInt __i;\
for(__i=0;__i<nnz;__i++) sum += xv[__i] * r[xi[__i]];}
#endif

#endif
