
#ifndef __MATIMPL_H
#define __MATIMPL_H

#include "petscmat.h"

/*
  This file defines the parts of the matrix data structure that are 
  shared by all matrix types.
*/

/*
    If you add entries here also add them to the MATOP enum
    in include/petscmat.h and include/finclude/petscmat.h
*/
typedef struct _MatOps *MatOps;
struct _MatOps {
  /* 0*/
  PetscErrorCode (*setvalues)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
  PetscErrorCode (*getrow)(Mat,PetscInt,PetscInt *,PetscInt*[],PetscScalar*[]);
  PetscErrorCode (*restorerow)(Mat,PetscInt,PetscInt *,PetscInt *[],PetscScalar *[]);
  PetscErrorCode (*mult)(Mat,Vec,Vec);
  PetscErrorCode (*multadd)(Mat,Vec,Vec,Vec);
  /* 5*/
  PetscErrorCode (*multtranspose)(Mat,Vec,Vec);
  PetscErrorCode (*multtransposeadd)(Mat,Vec,Vec,Vec);
  PetscErrorCode (*solve)(Mat,Vec,Vec);
  PetscErrorCode (*solveadd)(Mat,Vec,Vec,Vec);
  PetscErrorCode (*solvetranspose)(Mat,Vec,Vec);
  /*10*/
  PetscErrorCode (*solvetransposeadd)(Mat,Vec,Vec,Vec);
  PetscErrorCode (*lufactor)(Mat,IS,IS,const MatFactorInfo*);
  PetscErrorCode (*choleskyfactor)(Mat,IS,const MatFactorInfo*);
  PetscErrorCode (*sor)(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);
  PetscErrorCode (*transpose)(Mat,MatReuse,Mat *);
  /*15*/
  PetscErrorCode (*getinfo)(Mat,MatInfoType,MatInfo*);
  PetscErrorCode (*equal)(Mat,Mat,PetscTruth *);
  PetscErrorCode (*getdiagonal)(Mat,Vec);
  PetscErrorCode (*diagonalscale)(Mat,Vec,Vec);
  PetscErrorCode (*norm)(Mat,NormType,PetscReal*);
  /*20*/
  PetscErrorCode (*assemblybegin)(Mat,MatAssemblyType);
  PetscErrorCode (*assemblyend)(Mat,MatAssemblyType);
  PetscErrorCode (*setoption)(Mat,MatOption,PetscTruth);
  PetscErrorCode (*zeroentries)(Mat);
  /*24*/
  PetscErrorCode (*zerorows)(Mat,PetscInt,const PetscInt[],PetscScalar);
  PetscErrorCode (*lufactorsymbolic)(Mat,Mat,IS,IS,const MatFactorInfo*);
  PetscErrorCode (*lufactornumeric)(Mat,Mat,const MatFactorInfo*);
  PetscErrorCode (*choleskyfactorsymbolic)(Mat,Mat,IS,const MatFactorInfo*);
  PetscErrorCode (*choleskyfactornumeric)(Mat,Mat,const MatFactorInfo*);
  /*29*/
  PetscErrorCode (*setuppreallocation)(Mat);
  PetscErrorCode (*ilufactorsymbolic)(Mat,Mat,IS,IS,const MatFactorInfo*);
  PetscErrorCode (*iccfactorsymbolic)(Mat,Mat,IS,const MatFactorInfo*);
  PetscErrorCode (*getarray)(Mat,PetscScalar**);
  PetscErrorCode (*restorearray)(Mat,PetscScalar**);
  /*34*/
  PetscErrorCode (*duplicate)(Mat,MatDuplicateOption,Mat*);
  PetscErrorCode (*forwardsolve)(Mat,Vec,Vec);
  PetscErrorCode (*backwardsolve)(Mat,Vec,Vec);
  PetscErrorCode (*ilufactor)(Mat,IS,IS,const MatFactorInfo*);
  PetscErrorCode (*iccfactor)(Mat,IS,const MatFactorInfo*);
  /*39*/
  PetscErrorCode (*axpy)(Mat,PetscScalar,Mat,MatStructure);
  PetscErrorCode (*getsubmatrices)(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
  PetscErrorCode (*increaseoverlap)(Mat,PetscInt,IS[],PetscInt);
  PetscErrorCode (*getvalues)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar []);
  PetscErrorCode (*copy)(Mat,Mat,MatStructure);
  /*44*/
  PetscErrorCode (*getrowmax)(Mat,Vec,PetscInt[]);
  PetscErrorCode (*scale)(Mat,PetscScalar);
  PetscErrorCode (*shift)(Mat,PetscScalar);
  PetscErrorCode (*diagonalset)(Mat,Vec,InsertMode);
  PetscErrorCode (*dummy)(void);
  /*49*/
  PetscErrorCode (*setblocksize)(Mat,PetscInt);
  PetscErrorCode (*getrowij)(Mat,PetscInt,PetscTruth,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *);
  PetscErrorCode (*restorerowij)(Mat,PetscInt,PetscTruth,PetscTruth,PetscInt *,PetscInt *[],PetscInt *[],PetscTruth *);
  PetscErrorCode (*getcolumnij)(Mat,PetscInt,PetscTruth,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *);
  PetscErrorCode (*restorecolumnij)(Mat,PetscInt,PetscTruth,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *);
  /*54*/
  PetscErrorCode (*fdcoloringcreate)(Mat,ISColoring,MatFDColoring);
  PetscErrorCode (*coloringpatch)(Mat,PetscInt,PetscInt,ISColoringValue[],ISColoring*);
  PetscErrorCode (*setunfactored)(Mat);
  PetscErrorCode (*permute)(Mat,IS,IS,Mat*);
  PetscErrorCode (*setvaluesblocked)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
  /*59*/
  PetscErrorCode (*getsubmatrix)(Mat,IS,IS,MatReuse,Mat*);
  PetscErrorCode (*destroy)(Mat);
  PetscErrorCode (*view)(Mat,PetscViewer);
  PetscErrorCode (*convertfrom)(Mat, const MatType,MatReuse,Mat*);
  PetscErrorCode (*usescaledform)(Mat,PetscTruth);
  /*64*/
  PetscErrorCode (*scalesystem)(Mat,Vec,Vec);
  PetscErrorCode (*unscalesystem)(Mat,Vec,Vec);
  PetscErrorCode (*setlocaltoglobalmapping)(Mat,ISLocalToGlobalMapping);
  PetscErrorCode (*setvalueslocal)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
  PetscErrorCode (*zerorowslocal)(Mat,PetscInt,const PetscInt[],PetscScalar);
  /*69*/
  PetscErrorCode (*getrowmaxabs)(Mat,Vec,PetscInt[]);
  PetscErrorCode (*getrowminabs)(Mat,Vec,PetscInt[]);
  PetscErrorCode (*convert)(Mat, const MatType,MatReuse,Mat*);
  PetscErrorCode (*setcoloring)(Mat,ISColoring);
  PetscErrorCode (*setvaluesadic)(Mat,void*);
  /*74*/
  PetscErrorCode (*setvaluesadifor)(Mat,PetscInt,void*);
  PetscErrorCode (*fdcoloringapply)(Mat,MatFDColoring,Vec,MatStructure*,void*);
  PetscErrorCode (*setfromoptions)(Mat);
  PetscErrorCode (*multconstrained)(Mat,Vec,Vec);
  PetscErrorCode (*multtransposeconstrained)(Mat,Vec,Vec);
  /*79*/
  PetscErrorCode (*permutesparsify)(Mat, PetscInt, double, double, IS, IS, Mat *);
  PetscErrorCode (*mults)(Mat, Vecs, Vecs);
  PetscErrorCode (*solves)(Mat, Vecs, Vecs);
  PetscErrorCode (*getinertia)(Mat,PetscInt*,PetscInt*,PetscInt*);
  PetscErrorCode (*load)(PetscViewer, const MatType,Mat*);
  /*84*/
  PetscErrorCode (*issymmetric)(Mat,PetscReal,PetscTruth*);
  PetscErrorCode (*ishermitian)(Mat,PetscReal,PetscTruth*);
  PetscErrorCode (*isstructurallysymmetric)(Mat,PetscTruth*);
  PetscErrorCode (*dummy4)(void);
  PetscErrorCode (*getvecs)(Mat,Vec*,Vec*);
  /*89*/
  PetscErrorCode (*matmult)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*matmultsymbolic)(Mat,Mat,PetscReal,Mat*);
  PetscErrorCode (*matmultnumeric)(Mat,Mat,Mat);
  PetscErrorCode (*ptap)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*ptapsymbolic)(Mat,Mat,PetscReal,Mat*); /* double dispatch wrapper routine */
  /*94*/
  PetscErrorCode (*ptapnumeric)(Mat,Mat,Mat);             /* double dispatch wrapper routine */
  PetscErrorCode (*matmulttranspose)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*matmulttransposesymbolic)(Mat,Mat,PetscReal,Mat*);
  PetscErrorCode (*matmulttransposenumeric)(Mat,Mat,Mat);
  PetscErrorCode (*ptapsymbolic_seqaij)(Mat,Mat,PetscReal,Mat*); /* actual implememtation, A=seqaij */
  /*99*/
  PetscErrorCode (*ptapnumeric_seqaij)(Mat,Mat,Mat);             /* actual implememtation, A=seqaij */
  PetscErrorCode (*ptapsymbolic_mpiaij)(Mat,Mat,PetscReal,Mat*); /* actual implememtation, A=mpiaij */
  PetscErrorCode (*ptapnumeric_mpiaij)(Mat,Mat,Mat);             /* actual implememtation, A=mpiaij */
  PetscErrorCode (*conjugate)(Mat);                              /* complex conjugate */
  PetscErrorCode (*setsizes)(Mat,PetscInt,PetscInt,PetscInt,PetscInt);
  /*104*/
  PetscErrorCode (*setvaluesrow)(Mat,PetscInt,const PetscScalar[]);
  PetscErrorCode (*realpart)(Mat);
  PetscErrorCode (*imaginarypart)(Mat);
  PetscErrorCode (*getrowuppertriangular)(Mat);
  PetscErrorCode (*restorerowuppertriangular)(Mat);
  /*109*/
  PetscErrorCode (*matsolve)(Mat,Mat,Mat);
  PetscErrorCode (*getredundantmatrix)(Mat,PetscInt,MPI_Comm,PetscInt,MatReuse,Mat*);
  PetscErrorCode (*getrowmin)(Mat,Vec,PetscInt[]);
  PetscErrorCode (*getcolumnvector)(Mat,Vec,PetscInt);
  PetscErrorCode (*missingdiagonal)(Mat,PetscTruth*,PetscInt*);
  /*114*/
  PetscErrorCode (*getseqnonzerostructure)(Mat,Mat *);
  PetscErrorCode (*create)(Mat);  
  PetscErrorCode (*getghosts)(Mat,PetscInt*,const PetscInt *[]);
  PetscErrorCode (*dummy2)(void);
  PetscErrorCode (*dummy3)(void);
  /*119*/
  PetscErrorCode (*multdiagonalblock)(Mat,Vec,Vec);
  PetscErrorCode (*hermitiantranspose)(Mat,MatReuse,Mat*);
  PetscErrorCode (*multhermitiantranspose)(Mat,Vec,Vec);
  PetscErrorCode (*multhermitiantransposeadd)(Mat,Vec,Vec,Vec);
};
/*
    If you add MatOps entries above also add them to the MATOP enum
    in include/petscmat.h and include/finclude/petscmat.h
*/

/*
   Utility private matrix routines
*/
EXTERN PetscErrorCode MatConvert_Basic(Mat, const MatType,MatReuse,Mat*);
EXTERN PetscErrorCode MatCopy_Basic(Mat,Mat,MatStructure);
EXTERN PetscErrorCode MatView_Private(Mat);

EXTERN PetscErrorCode MatHeaderCopy(Mat,Mat);
EXTERN PetscErrorCode MatHeaderReplace(Mat,Mat);
EXTERN PetscErrorCode MatAXPYGetxtoy_Private(PetscInt,PetscInt*,PetscInt*,PetscInt*, PetscInt*,PetscInt*,PetscInt*, PetscInt**);
EXTERN PetscErrorCode MatPtAP_Basic(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode MatDiagonalSet_Default(Mat,Vec,InsertMode);

/* 
  The stash is used to temporarily store inserted matrix values that 
  belong to another processor. During the assembly phase the stashed 
  values are moved to the correct processor and 
*/

typedef struct _MatStashSpace *PetscMatStashSpace;

struct _MatStashSpace {
  PetscMatStashSpace next;
  PetscScalar        *space_head,*val;
  PetscInt           *idx,*idy;
  PetscInt           total_space_size;
  PetscInt           local_used;
  PetscInt           local_remaining;
};

EXTERN PetscErrorCode PetscMatStashSpaceGet(PetscInt,PetscInt,PetscMatStashSpace *);
EXTERN PetscErrorCode PetscMatStashSpaceContiguous(PetscInt,PetscMatStashSpace *,PetscScalar *,PetscInt *,PetscInt *);
EXTERN PetscErrorCode PetscMatStashSpaceDestroy(PetscMatStashSpace);

typedef struct {
  PetscInt      nmax;                   /* maximum stash size */
  PetscInt      umax;                   /* user specified max-size */
  PetscInt      oldnmax;                /* the nmax value used previously */
  PetscInt      n;                      /* stash size */
  PetscInt      bs;                     /* block size of the stash */
  PetscInt      reallocs;               /* preserve the no of mallocs invoked */    
  PetscMatStashSpace space_head,space;  /* linked list to hold stashed global row/column numbers and matrix values */
  /* The following variables are used for communication */
  MPI_Comm      comm;
  PetscMPIInt   size,rank;
  PetscMPIInt   tag1,tag2;
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  MPI_Status    *send_status;           /* array of send status */
  PetscInt      nsends,nrecvs;          /* numbers of sends and receives */
  PetscScalar   *svalues;               /* sending data */
  PetscInt      *sindices;    
  PetscScalar   **rvalues;              /* receiving data (values) */
  PetscInt      **rindices;             /* receiving data (indices) */
  PetscInt      nprocessed;             /* number of messages already processed */
  PetscMPIInt   *flg_v;                 /* indicates what messages have arrived so far and from whom */
} MatStash;

EXTERN PetscErrorCode MatStashCreate_Private(MPI_Comm,PetscInt,MatStash*);
EXTERN PetscErrorCode MatStashDestroy_Private(MatStash*);
EXTERN PetscErrorCode MatStashScatterEnd_Private(MatStash*);
EXTERN PetscErrorCode MatStashSetInitialSize_Private(MatStash*,PetscInt);
EXTERN PetscErrorCode MatStashGetInfo_Private(MatStash*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode MatStashValuesRow_Private(MatStash*,PetscInt,PetscInt,const PetscInt[],const PetscScalar[],PetscTruth);
EXTERN PetscErrorCode MatStashValuesCol_Private(MatStash*,PetscInt,PetscInt,const PetscInt[],const PetscScalar[],PetscInt,PetscTruth);
EXTERN PetscErrorCode MatStashValuesRowBlocked_Private(MatStash*,PetscInt,PetscInt,const PetscInt[],const PetscScalar[],PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode MatStashValuesColBlocked_Private(MatStash*,PetscInt,PetscInt,const PetscInt[],const PetscScalar[],PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode MatStashScatterBegin_Private(Mat,MatStash*,PetscInt*);
EXTERN PetscErrorCode MatStashScatterGetMesg_Private(MatStash*,PetscMPIInt*,PetscInt**,PetscInt**,PetscScalar**,PetscInt*);

typedef struct {
  PetscInt   dim;
  PetscInt   dims[4];
  PetscInt   starts[4];
  PetscTruth noc;        /* this is a single component problem, hence user will not set MatStencil.c */
} MatStencilInfo;

/* Info about using compressed row format */
typedef struct {
  PetscTruth use;
  PetscInt   nrows;                         /* number of non-zero rows */
  PetscInt   *i;                            /* compressed row pointer  */
  PetscInt   *rindex;                       /* compressed row index               */
  PetscTruth checked;                       /* if compressed row format have been checked for */
} Mat_CompressedRow;
EXTERN PetscErrorCode Mat_CheckCompressedRow(Mat,Mat_CompressedRow*,PetscInt*,PetscInt,PetscReal);

struct _p_Mat {
  PETSCHEADER(struct _MatOps);
  PetscLayout            rmap,cmap;
  void                   *data;            /* implementation-specific data */
  MatFactorType          factor;           /* MAT_FACTOR_LU, or MAT_FACTOR_CHOLESKY */
  PetscTruth             assembled;        /* is the matrix assembled? */
  PetscTruth             was_assembled;    /* new values inserted into assembled mat */
  PetscInt               num_ass;          /* number of times matrix has been assembled */
  PetscTruth             same_nonzero;     /* matrix has same nonzero pattern as previous */
  MatInfo                info;             /* matrix information */
  ISLocalToGlobalMapping mapping;          /* mapping used in MatSetValuesLocal() */
  ISLocalToGlobalMapping bmapping;         /* mapping used in MatSetValuesBlockedLocal() */
  InsertMode             insertmode;       /* have values been inserted in matrix or added? */
  MatStash               stash,bstash;     /* used for assembling off-proc mat emements */
  MatNullSpace           nullsp;
  PetscTruth             preallocated;
  MatStencilInfo         stencil;          /* information for structured grid */
  PetscTruth             symmetric,hermitian,structurally_symmetric;
  PetscTruth             symmetric_set,hermitian_set,structurally_symmetric_set; /* if true, then corresponding flag is correct*/
  PetscTruth             symmetric_eternal;
  void                   *spptr;          /* pointer for special library like SuperLU */
  MatSolverPackage       solvertype;
};

#define MatPreallocated(A)  ((!(A)->preallocated) ? MatSetUpPreallocation(A) : 0)
extern PetscErrorCode MatAXPY_Basic(Mat,PetscScalar,Mat,MatStructure);

/*
    Object for partitioning graphs
*/

typedef struct _MatPartitioningOps *MatPartitioningOps;
struct _MatPartitioningOps {
  PetscErrorCode (*apply)(MatPartitioning,IS*);
  PetscErrorCode (*setfromoptions)(MatPartitioning);
  PetscErrorCode (*destroy)(MatPartitioning);
  PetscErrorCode (*view)(MatPartitioning,PetscViewer);
};

struct _p_MatPartitioning {
  PETSCHEADER(struct _MatPartitioningOps);
  Mat         adj;
  PetscInt    *vertex_weights;
  PetscReal   *part_weights;
  PetscInt    n;                                 /* number of partitions */
  void        *data;
  PetscInt    setupcalled;
};

/*
    MatFDColoring is used to compute Jacobian matrices efficiently
  via coloring. The data structure is explained below in an example.

   Color =   0    1     0    2   |   2      3       0 
   ---------------------------------------------------
            00   01              |          05
            10   11              |   14     15               Processor  0
                       22    23  |          25
                       32    33  | 
   ===================================================
                                 |   44     45     46
            50                   |          55               Processor 1
                                 |   64            66
   ---------------------------------------------------

    ncolors = 4;

    ncolumns      = {2,1,1,0}
    columns       = {{0,2},{1},{3},{}}
    nrows         = {4,2,3,3}
    rows          = {{0,1,2,3},{0,1},{1,2,3},{0,1,2}}
    columnsforrow = {{0,0,2,2},{1,1},{4,3,3},{5,5,5}}
    vscaleforrow  = {{,,,},{,},{,,},{,,}}
    vwscale       = {dx(0),dx(1),dx(2),dx(3)}               MPI Vec
    vscale        = {dx(0),dx(1),dx(2),dx(3),dx(4),dx(5)}   Seq Vec

    ncolumns      = {1,0,1,1}
    columns       = {{6},{},{4},{5}}
    nrows         = {3,0,2,2}
    rows          = {{0,1,2},{},{1,2},{1,2}}
    columnsforrow = {{6,0,6},{},{4,4},{5,5}}
    vscaleforrow =  {{,,},{},{,},{,}}
    vwscale       = {dx(4),dx(5),dx(6)}              MPI Vec
    vscale        = {dx(0),dx(4),dx(5),dx(6)}        Seq Vec

    See the routine MatFDColoringApply() for how this data is used
    to compute the Jacobian.

*/

struct  _p_MatFDColoring{
  PETSCHEADER(int);
  PetscInt       M,N,m;            /* total rows, columns; local rows */
  PetscInt       rstart;           /* first row owned by local processor */
  PetscInt       ncolors;          /* number of colors */
  PetscInt       *ncolumns;        /* number of local columns for a color */ 
  PetscInt       **columns;        /* lists the local columns of each color (using global column numbering) */
  PetscInt       *nrows;           /* number of local rows for each color */
  PetscInt       **rows;           /* lists the local rows for each color (using the local row numbering) */
  PetscInt       **columnsforrow;  /* lists the corresponding columns for those rows (using the global column) */ 
  PetscReal      error_rel;        /* square root of relative error in computing function */
  PetscReal      umin;             /* minimum allowable u'dx value */
  Vec            w1,w2,w3;         /* work vectors used in computing Jacobian */
  PetscErrorCode (*f)(void);       /* function that defines Jacobian */
  void           *fctx;            /* optional user-defined context for use by the function f */
  PetscInt       **vscaleforrow;   /* location in vscale for each columnsforrow[] entry */
  Vec            vscale;           /* holds FD scaling, i.e. 1/dx for each perturbed column */
  Vec            F;                /* current value of user provided function; can set with MatFDColoringSetF() */
  PetscInt       currentcolor;     /* color for which function evaluation is being done now */
  const char     *htype;            /* "wp" or "ds" */
  ISColoringType ctype;            /* IS_COLORING_GLOBAL or IS_COLORING_GHOSTED */

  void           *ftn_func_pointer,*ftn_func_cntx; /* serve the same purpose as *fortran_func_pointers in PETSc objects */
};

/*
   Null space context for preconditioner/operators
*/
struct _p_MatNullSpace {
  PETSCHEADER(int);
  PetscTruth     has_cnst;
  PetscInt       n;
  Vec*           vecs;
  PetscScalar*   alpha;                 /* for projections */
  Vec            vec;                   /* for out of place removals */
  PetscErrorCode (*remove)(MatNullSpace,Vec,void*);  /* for user provided removal function */
  void*          rmctx;                 /* context for remove() function */
};

/* 
   Checking zero pivot for LU, ILU preconditioners.
*/
typedef struct {
  PetscInt       nshift,nshift_max;
  PetscReal      shift_amount,shift_lo,shift_hi,shift_top,shift_fraction;
  PetscTruth     useshift;
  PetscReal      rs;  /* active row sum of abs(offdiagonals) */
  PetscScalar    pv;  /* pivot of the active row */
} FactorShiftCtx;

EXTERN PetscErrorCode MatFactorDumpMatrix(Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatLUCheckShift_inline"
/*@C
   MatLUCheckShift_inline - shift the diagonals when zero pivot is detected on LU factor

   Collective on Mat

   Input Parameters:
+  info - information about the matrix factorization 
.  sctx - pointer to the struct FactorShiftCtx
-  row  - active row index

   Output  Parameter:
+  newshift - 0: shift is unchanged; 1: shft is updated; -1: zeropivot  

   Level: developer
@*/
#define MatLUCheckShift_inline(info,sctx,row,newshift) 0;\
{\
  PetscInt  _newshift;\
  PetscReal _rs   = sctx.rs;\
  PetscReal _zero = info->zeropivot*_rs;\
  if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO && PetscAbsScalar(sctx.pv) <= _zero){\
    /* force |diag| > zeropivot*rs */\
    if (!sctx.nshift){\
      sctx.shift_amount = info->shiftamount;\
    } else {\
      sctx.shift_amount *= 2.0;\
    }\
    sctx.useshift = PETSC_TRUE;\
    (sctx.nshift)++;\
    _newshift = 1;\
  } else if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE && PetscRealPart(sctx.pv) <= _zero){\
    /* force matfactor to be diagonally dominant */\
    if (sctx.nshift > sctx.nshift_max) {\
      ierr = MatFactorDumpMatrix(A);CHKERRQ(ierr);\
      SETERRQ1(PETSC_ERR_CONV_FAILED,"Unable to determine shift to enforce positive definite preconditioner after %d tries",sctx.nshift);\
    } else if (sctx.nshift == sctx.nshift_max) {\
      sctx.shift_fraction = sctx.shift_hi;\
      sctx.useshift        = PETSC_TRUE;\
    } else {\
      sctx.shift_lo = sctx.shift_fraction;\
      sctx.shift_fraction = (sctx.shift_hi+sctx.shift_lo)/2.;\
      sctx.useshift  = PETSC_TRUE;\
    }\
    sctx.shift_amount = sctx.shift_fraction * sctx.shift_top;\
    sctx.nshift++;\
    _newshift = 1;\
  } else if (PetscAbsScalar(sctx.pv) <= _zero){\
    ierr = MatFactorDumpMatrix(A);CHKERRQ(ierr);\
    SETERRQ4(PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot row %D value %G tolerance %G * rowsum %G",row,PetscAbsScalar(sctx.pv),_zero,_rs); \
  } else {\
    _newshift = 0;\
  }\
  newshift = _newshift;\
}

#define MatPivotCheck_nz(info,sctx,row) 0;\
{\
  PetscReal _rs   = sctx.rs;\
  PetscReal _zero = info->zeropivot*_rs;\
  if (PetscAbsScalar(sctx.pv) <= _zero){\
    /* force |diag| > zeropivot*rs */\
    if (!sctx.nshift){\
      sctx.shift_amount = info->shiftamount;\
    } else {\
      sctx.shift_amount *= 2.0;\
    }\
    sctx.useshift = PETSC_TRUE;\
    (sctx.nshift)++;\
    break;          \
  } \
}

#define MatPivotCheck_pd(info,sctx,row) 0;\
{\
  PetscReal _rs   = sctx.rs;\
  PetscReal _zero = info->zeropivot*_rs;\
  if (PetscRealPart(sctx.pv) <= _zero){\
    /* force matfactor to be diagonally dominant */\
    if (sctx.nshift > sctx.nshift_max) {\
      ierr = MatFactorDumpMatrix(A);CHKERRQ(ierr);\
      SETERRQ1(PETSC_ERR_CONV_FAILED,"Unable to determine shift to enforce positive definite preconditioner after %d tries",sctx.nshift);\
    } else if (sctx.nshift == sctx.nshift_max) {\
      sctx.shift_fraction = sctx.shift_hi;\
      sctx.useshift        = PETSC_TRUE;\
    } else {\
      sctx.shift_lo = sctx.shift_fraction;\
      sctx.shift_fraction = (sctx.shift_hi+sctx.shift_lo)/2.;\
      sctx.useshift = PETSC_TRUE;\
    }\
    sctx.shift_amount = sctx.shift_fraction * sctx.shift_top;\
    sctx.nshift++;\
    break; \
  }\
}

#define MatPivotCheck_inblocks(info,sctx,row) 0;\
{\
  PetscReal _zero = info->zeropivot;\
  if (PetscAbsScalar(sctx.pv) <= _zero){\
    sctx.pv          += info->shiftamount;\
    sctx.shift_amount = 0.0;\
    sctx.nshift++;\
  }\
}

#define MatPivotCheck_none(info,sctx,row) 0;\
{\
  PetscReal _zero = info->zeropivot;\
  if (PetscAbsScalar(sctx.pv) <= _zero){\
    ierr = MatFactorDumpMatrix(A);CHKERRQ(ierr);\
    SETERRQ3(PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot row %D value %G tolerance %G",row,PetscAbsScalar(sctx.pv),_zero); \
  } \
}

#define MatPivotCheck(info,sctx,row) 0;\
{\
  if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO){\
    ierr = MatPivotCheck_nz(info,sctx,row);CHKERRQ(ierr);\
  } else if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE){\
    ierr = MatPivotCheck_pd(info,sctx,row);CHKERRQ(ierr);\
  } else if (info->shifttype == (PetscReal)MAT_SHIFT_INBLOCKS){\
    ierr = MatPivotCheck_inblocks(info,sctx,srow);CHKERRQ(ierr);\
  } else {\
    ierr = MatPivotCheck_none(info,sctx,row);CHKERRQ(ierr);\
  }\
}

/* 
   Checking zero pivot for Cholesky, ICC preconditioners.
*/
typedef struct {
  PetscInt       nshift;
  PetscReal      shift_amount;
  PetscTruth     chshift;
  PetscReal      rs;  /* active row sum of abs(offdiagonals) */
  PetscScalar    pv;  /* pivot of the active row */
} ChShift_Ctx;

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyCheckShift_inline"
/*@C
   MatCholeskyCheckShift_inline -  shift the diagonals when zero pivot is detected on Cholesky factor

   Collective on Mat

   Input Parameters:
+  info - information about the matrix factorization 
.  sctx - pointer to the struct CholeskyShift_Ctx
.  row  - pivot row
-  newshift - 0: shift is unchanged; 1: shft is updated; -1: zeropivot  

   Level: developer
   Note: Unlike in the ILU case there is no exit condition on nshift:
       we increase the shift until it converges. There is no guarantee that
       this algorithm converges faster or slower, or is better or worse
       than the ILU algorithm. 
@*/
#define MatCholeskyCheckShift_inline(info,sctx,row,newshift) 0;	\
{\
  PetscInt  _newshift;\
  PetscReal _rs   = sctx.rs;\
  PetscReal _zero = info->zeropivot*_rs;\
  if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO && PetscAbsScalar(sctx.pv) <= _zero){\
    /* force |diag| > zeropivot*sctx.rs */\
    if (!sctx.nshift){\
      sctx.shift_amount = info->shiftamount;\
    } else {\
      sctx.shift_amount *= 2.0;\
    }\
    sctx.chshift = PETSC_TRUE;\
    sctx.nshift++;\
    _newshift = 1;\
  } else if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE && PetscRealPart(sctx.pv) <= _zero){\
    /* calculate a shift that would make this row diagonally dominant */\
    sctx.shift_amount = PetscMax(_rs+PetscAbs(PetscRealPart(sctx.pv)),1.1*sctx.shift_amount);\
    sctx.chshift      = PETSC_TRUE;\
    sctx.nshift++;\
    _newshift = 1;\
  } else if (PetscAbsScalar(sctx.pv) <= _zero){\
    SETERRQ4(PETSC_ERR_MAT_CH_ZRPVT,"Zero pivot row %D value %G tolerance %G * rowsum %G",row,PetscAbsScalar(sctx.pv),_zero,_rs); \
  } else {\
    _newshift = 0; \
  }\
  newshift = _newshift;\
}

/* 
  Create and initialize a linked list 
  Input Parameters:
    idx_start - starting index of the list
    lnk_max   - max value of lnk indicating the end of the list
    nlnk      - max length of the list
  Output Parameters:
    lnk       - list initialized
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscLLCreate(idx_start,lnk_max,nlnk,lnk,bt) \
  (PetscMalloc(nlnk*sizeof(PetscInt),&lnk) || PetscBTCreate(nlnk,bt) || PetscBTMemzero(nlnk,bt) || (lnk[idx_start] = lnk_max,0))

/*
  Add an index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - interger array
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
*/
#define PetscLLAdd(nidx,indices,idx_start,nlnk,lnk,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _entry = indices[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      /* start from the beginning if _entry < previous _entry */\
      if (_k && _entry < _lnkdata) _lnkdata  = idx_start;\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    }\
  }\
}

/*
  Add a permuted index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - interger array
    perm      - permutation of indices
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
*/
#define PetscLLAddPerm(nidx,indices,perm,idx_start,nlnk,lnk,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _entry = perm[indices[_k]];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      /* start from the beginning if _entry < previous _entry */\
      if (_k && _entry < _lnkdata) _lnkdata  = idx_start;\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - sorted interger array 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
*/
#define PetscLLAddSorted(nidx,indices,idx_start,nlnk,lnk,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata;\
  nlnk      = 0;\
  _lnkdata  = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _entry = indices[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list used for LUFactorSymbolic()
  Same as PetscLLAddSorted() with an additional operation:
       count the number of input indices that are no larger than 'diag' 
  Input Parameters:
    indices   - sorted interger array 
    idx_start - starting index of the list, index of pivot row
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
    diag      - index of the active row in LUFactorSymbolic
    nzbd      - number of input indices with indices <= idx_start
    im        - im[idx_start] is initialized as num of nonzero entries in row=idx_start
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
    im        - im[idx_start]: unchanged if diag is not an entry 
                             : num of entries with indices <= diag if diag is an entry
*/
#define PetscLLAddSortedLU(indices,idx_start,nlnk,lnk,bt,diag,nzbd,im) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata,_nidx;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  _nidx = im[idx_start] - nzbd; /* num of entries with idx_start < index <= diag */\
  for (_k=0; _k<_nidx; _k++){\
    _entry = indices[_k];\
    nzbd++;\
    if ( _entry== diag) im[idx_start] = nzbd;\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    }\
  }\
}

/*
  Copy data on the list into an array, then initialize the list 
  Input Parameters:
    idx_start - starting index of the list 
    lnk_max   - max value of lnk indicating the end of the list 
    nlnk      - number of data on the list to be copied
    lnk       - linked list
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    indices   - array that contains the copied data
    lnk       - linked list that is cleaned and initialize
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscLLClean(idx_start,lnk_max,nlnk,lnk,indices,bt) 0;\
{\
  PetscInt _j,_idx=idx_start;\
  for (_j=0; _j<nlnk; _j++){\
    _idx = lnk[_idx];\
    *(indices+_j) = _idx;\
    ierr = PetscBTClear(bt,_idx);CHKERRQ(ierr);\
  }\
  lnk[idx_start] = lnk_max;\
}
/*
  Free memories used by the list
*/
#define PetscLLDestroy(lnk,bt) (PetscFree(lnk) || PetscBTDestroy(bt))

/* Routines below are used for incomplete matrix factorization */
/* 
  Create and initialize a linked list and its levels
  Input Parameters:
    idx_start - starting index of the list
    lnk_max   - max value of lnk indicating the end of the list
    nlnk      - max length of the list
  Output Parameters:
    lnk       - list initialized
    lnk_lvl   - array of size nlnk for storing levels of lnk
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscIncompleteLLCreate(idx_start,lnk_max,nlnk,lnk,lnk_lvl,bt)\
  (PetscMalloc(2*nlnk*sizeof(PetscInt),&lnk) || PetscBTCreate(nlnk,bt) || PetscBTMemzero(nlnk,bt) || (lnk[idx_start] = lnk_max,lnk_lvl = lnk + nlnk,0))

/*
  Initialize a sorted linked list used for ILU and ICC
  Input Parameters:
    nidx      - number of input idx
    idx       - interger array used for storing column indices
    idx_start - starting index of the list
    perm      - indices of an IS
    lnk       - linked list(an integer array) that is created
    lnklvl    - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk     - number of newly added idx
    lnk      - the sorted(increasing order) linked list containing new and non-redundate entries from idx
    lnklvl   - levels of lnk
    bt       - updated PetscBT (bitarray) 
*/
#define PetscIncompleteLLInit(nidx,idx,idx_start,perm,nlnk,lnk,lnklvl,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _entry = perm[idx[_k]];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      if (_k && _entry < _lnkdata) _lnkdata  = idx_start;\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location]  = _entry;\
      lnk[_entry]     = _lnkdata;\
      lnklvl[_entry] = 0;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list for ILU
  Input Parameters:
    nidx      - number of input indices
    idx       - sorted interger array used for storing column indices
    level     - level of fill, e.g., ICC(level)
    idxlvl    - level of idx 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    lnklvl    - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
    prow      - the row number of idx
  output Parameters:
    nlnk     - number of newly added idx
    lnk      - the sorted(increasing order) linked list containing new and non-redundate entries from idx
    lnklvl   - levels of lnk
    bt       - updated PetscBT (bitarray) 

  Note: the level of factor(i,j) is set as lvl(i,j) = min{ lvl(i,j), lvl(i,prow)+lvl(prow,j)+1)
        where idx = non-zero columns of U(prow,prow+1:n-1), prow<i
*/
#define PetscILULLAddSorted(nidx,idx,level,idxlvl,idx_start,nlnk,lnk,lnklvl,bt,lnklvl_prow) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata,_incrlev,_lnklvl_prow=lnklvl[prow];\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _incrlev = idxlvl[_k] + _lnklvl_prow + 1;\
    if (_incrlev > level) continue;\
    _entry = idx[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location]  = _entry;\
      lnk[_entry]     = _lnkdata;\
      lnklvl[_entry] = _incrlev;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    } else { /* existing entry: update lnklvl */\
      if (lnklvl[_entry] > _incrlev) lnklvl[_entry] = _incrlev;\
    }\
  }\
}

/*
  Add a index set into a sorted linked list
  Input Parameters:
    nidx      - number of input idx
    idx   - interger array used for storing column indices
    level     - level of fill, e.g., ICC(level)
    idxlvl - level of idx 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    lnklvl   - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added idx
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from idx
    lnklvl   - levels of lnk
    bt        - updated PetscBT (bitarray) 
*/
#define PetscIncompleteLLAdd(nidx,idx,level,idxlvl,idx_start,nlnk,lnk,lnklvl,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata,_incrlev;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _incrlev = idxlvl[_k] + 1;\
    if (_incrlev > level) continue;\
    _entry = idx[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      if (_k && _entry < _lnkdata) _lnkdata  = idx_start;\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location]  = _entry;\
      lnk[_entry]     = _lnkdata;\
      lnklvl[_entry] = _incrlev;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    } else { /* existing entry: update lnklvl */\
      if (lnklvl[_entry] > _incrlev) lnklvl[_entry] = _incrlev;\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    idx   - sorted interger array used for storing column indices
    level     - level of fill, e.g., ICC(level)
    idxlvl - level of idx 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    lnklvl    - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added idx
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from idx
    lnklvl    - levels of lnk
    bt        - updated PetscBT (bitarray) 
*/
#define PetscIncompleteLLAddSorted(nidx,idx,level,idxlvl,idx_start,nlnk,lnk,lnklvl,bt) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata,_incrlev;\
  nlnk = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _incrlev = idxlvl[_k] + 1;\
    if (_incrlev > level) continue;\
    _entry = idx[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      lnklvl[_entry] = _incrlev;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    } else { /* existing entry: update lnklvl */\
      if (lnklvl[_entry] > _incrlev) lnklvl[_entry] = _incrlev;\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list for ICC
  Input Parameters:
    nidx      - number of input indices
    idx       - sorted interger array used for storing column indices
    level     - level of fill, e.g., ICC(level)
    idxlvl    - level of idx 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    lnklvl    - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
    idxlvl_prow - idxlvl[prow], where prow is the row number of the idx
  output Parameters:
    nlnk   - number of newly added indices
    lnk    - the sorted(increasing order) linked list containing new and non-redundate entries from idx
    lnklvl - levels of lnk
    bt     - updated PetscBT (bitarray) 
  Note: the level of U(i,j) is set as lvl(i,j) = min{ lvl(i,j), lvl(prow,i)+lvl(prow,j)+1)
        where idx = non-zero columns of U(prow,prow+1:n-1), prow<i
*/
#define PetscICCLLAddSorted(nidx,idx,level,idxlvl,idx_start,nlnk,lnk,lnklvl,bt,idxlvl_prow) 0;\
{\
  PetscInt _k,_entry,_location,_lnkdata,_incrlev;\
  nlnk = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _incrlev = idxlvl[_k] + idxlvl_prow + 1;\
    if (_incrlev > level) continue;\
    _entry = idx[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      lnklvl[_entry] = _incrlev;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    } else { /* existing entry: update lnklvl */\
      if (lnklvl[_entry] > _incrlev) lnklvl[_entry] = _incrlev;\
    }\
  }\
}

/*
  Copy data on the list into an array, then initialize the list 
  Input Parameters:
    idx_start - starting index of the list 
    lnk_max   - max value of lnk indicating the end of the list 
    nlnk      - number of data on the list to be copied
    lnk       - linked list
    lnklvl    - level of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    indices - array that contains the copied data
    lnk     - linked list that is cleaned and initialize
    lnklvl  - level of lnk that is reinitialized 
    bt      - PetscBT (bitarray) with all bits set to false
*/
#define PetscIncompleteLLClean(idx_start,lnk_max,nlnk,lnk,lnklvl,indices,indiceslvl,bt) 0;\
{\
  PetscInt _j,_idx=idx_start;\
  for (_j=0; _j<nlnk; _j++){\
    _idx = lnk[_idx];\
    *(indices+_j) = _idx;\
    *(indiceslvl+_j) = lnklvl[_idx];\
    lnklvl[_idx] = -1;\
    ierr = PetscBTClear(bt,_idx);CHKERRQ(ierr);\
  }\
  lnk[idx_start] = lnk_max;\
}
/*
  Free memories used by the list
*/
#define PetscIncompleteLLDestroy(lnk,bt) (PetscFree(lnk) || PetscBTDestroy(bt))

extern PetscLogEvent  MAT_Mult, MAT_MultMatrixFree, MAT_Mults, MAT_MultConstrained, MAT_MultAdd, MAT_MultTranspose;
extern PetscLogEvent  MAT_MultTransposeConstrained, MAT_MultTransposeAdd, MAT_Solve, MAT_Solves, MAT_SolveAdd, MAT_SolveTranspose;
extern PetscLogEvent  MAT_SolveTransposeAdd, MAT_SOR, MAT_ForwardSolve, MAT_BackwardSolve, MAT_LUFactor, MAT_LUFactorSymbolic;
extern PetscLogEvent  MAT_LUFactorNumeric, MAT_CholeskyFactor, MAT_CholeskyFactorSymbolic, MAT_CholeskyFactorNumeric, MAT_ILUFactor;
extern PetscLogEvent  MAT_ILUFactorSymbolic, MAT_ICCFactorSymbolic, MAT_Copy, MAT_Convert, MAT_Scale, MAT_AssemblyBegin;
extern PetscLogEvent  MAT_AssemblyEnd, MAT_SetValues, MAT_GetValues, MAT_GetRow, MAT_GetRowIJ, MAT_GetSubMatrices, MAT_GetColoring, MAT_GetOrdering, MAT_GetRedundantMatrix;
extern PetscLogEvent  MAT_IncreaseOverlap, MAT_Partitioning, MAT_ZeroEntries, MAT_Load, MAT_View, MAT_AXPY, MAT_FDColoringCreate;
extern PetscLogEvent  MAT_FDColoringApply, MAT_Transpose, MAT_FDColoringFunction;
extern PetscLogEvent  MAT_MatMult, MAT_MatSolve,MAT_MatMultSymbolic, MAT_MatMultNumeric,MAT_Getlocalmatcondensed,MAT_GetBrowsOfAcols,MAT_GetBrowsOfAocols;
extern PetscLogEvent  MAT_PtAP, MAT_PtAPSymbolic, MAT_PtAPNumeric,MAT_Seqstompinum,MAT_Seqstompisym,MAT_Seqstompi,MAT_Getlocalmat;

extern PetscLogEvent  MAT_MatMultTranspose, MAT_MatMultTransposeSymbolic, MAT_MatMultTransposeNumeric;
extern PetscLogEvent  MAT_Applypapt, MAT_Applypapt_symbolic, MAT_Applypapt_numeric;
extern PetscLogEvent  MAT_Getsymtranspose, MAT_Transpose_SeqAIJ, MAT_Getsymtransreduced,MAT_GetSequentialNonzeroStructure;

extern PetscLogEvent  MATMFFD_Mult;

#endif
