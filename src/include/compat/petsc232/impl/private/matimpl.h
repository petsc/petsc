
#if !defined(__MATIMPL)
#define __MATIMPL

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
  PetscErrorCode (*setvalues)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const MatScalar[],InsertMode);
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
  PetscErrorCode (*lufactor)(Mat,IS,IS,MatFactorInfo*);
  PetscErrorCode (*choleskyfactor)(Mat,IS,MatFactorInfo*);
  PetscErrorCode (*relax)(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);
  PetscErrorCode (*transpose)(Mat,Mat *);
  /*15*/
  PetscErrorCode (*getinfo)(Mat,MatInfoType,MatInfo*);
  PetscErrorCode (*equal)(Mat,Mat,PetscTruth *);
  PetscErrorCode (*getdiagonal)(Mat,Vec);
  PetscErrorCode (*diagonalscale)(Mat,Vec,Vec);
  PetscErrorCode (*norm)(Mat,NormType,PetscReal*);
  /*20*/
  PetscErrorCode (*assemblybegin)(Mat,MatAssemblyType);
  PetscErrorCode (*assemblyend)(Mat,MatAssemblyType);
  PetscErrorCode (*compress)(Mat);
  PetscErrorCode (*setoption)(Mat,MatOption);
  PetscErrorCode (*zeroentries)(Mat);
  /*25*/
  PetscErrorCode (*zerorows)(Mat,PetscInt,const PetscInt[],PetscScalar);
  PetscErrorCode (*lufactorsymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  PetscErrorCode (*lufactornumeric)(Mat,MatFactorInfo*,Mat*);
  PetscErrorCode (*choleskyfactorsymbolic)(Mat,IS,MatFactorInfo*,Mat*);
  PetscErrorCode (*choleskyfactornumeric)(Mat,MatFactorInfo*,Mat*);
  /*30*/
  PetscErrorCode (*setuppreallocation)(Mat);
  PetscErrorCode (*ilufactorsymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  PetscErrorCode (*iccfactorsymbolic)(Mat,IS,MatFactorInfo*,Mat*);
  PetscErrorCode (*getarray)(Mat,PetscScalar**);
  PetscErrorCode (*restorearray)(Mat,PetscScalar**);
  /*35*/
  PetscErrorCode (*duplicate)(Mat,MatDuplicateOption,Mat*);
  PetscErrorCode (*forwardsolve)(Mat,Vec,Vec);
  PetscErrorCode (*backwardsolve)(Mat,Vec,Vec);
  PetscErrorCode (*ilufactor)(Mat,IS,IS,MatFactorInfo*);
  PetscErrorCode (*iccfactor)(Mat,IS,MatFactorInfo*);
  /*40*/
  PetscErrorCode (*axpy)(Mat,PetscScalar,Mat,MatStructure);
  PetscErrorCode (*getsubmatrices)(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]);
  PetscErrorCode (*increaseoverlap)(Mat,PetscInt,IS[],PetscInt);
  PetscErrorCode (*getvalues)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar []);
  PetscErrorCode (*copy)(Mat,Mat,MatStructure);
  /*45*/
  PetscErrorCode (*dummy)(void);
  PetscErrorCode (*scale)(Mat,PetscScalar);
  PetscErrorCode (*shift)(Mat,PetscScalar);
  PetscErrorCode (*diagonalset)(Mat,Vec,InsertMode);
  PetscErrorCode (*iludtfactor)(Mat,IS,IS,MatFactorInfo*,Mat *);
  /*50*/
  PetscErrorCode (*setblocksize)(Mat,PetscInt);
  PetscErrorCode (*getrowij)(Mat,PetscInt,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *);
  PetscErrorCode (*restorerowij)(Mat,PetscInt,PetscTruth,PetscInt *,PetscInt *[],PetscInt *[],PetscTruth *);
  PetscErrorCode (*getcolumnij)(Mat,PetscInt,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *);
  PetscErrorCode (*restorecolumnij)(Mat,PetscInt,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *);
  /*55*/
  PetscErrorCode (*fdcoloringcreate)(Mat,ISColoring,MatFDColoring);
  PetscErrorCode (*coloringpatch)(Mat,PetscInt,PetscInt,ISColoringValue[],ISColoring*);
  PetscErrorCode (*setunfactored)(Mat);
  PetscErrorCode (*permute)(Mat,IS,IS,Mat*);
  PetscErrorCode (*setvaluesblocked)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
  /*60*/
  PetscErrorCode (*getsubmatrix)(Mat,IS,IS,PetscInt,MatReuse,Mat*);
  PetscErrorCode (*destroy)(Mat);
  PetscErrorCode (*view)(Mat,PetscViewer);
  PetscErrorCode (*convertfrom)(Mat, MatType,MatReuse,Mat*);
  PetscErrorCode (*usescaledform)(Mat,PetscTruth);
  /*65*/
  PetscErrorCode (*scalesystem)(Mat,Vec,Vec);
  PetscErrorCode (*unscalesystem)(Mat,Vec,Vec);
  PetscErrorCode (*setlocaltoglobalmapping)(Mat,ISLocalToGlobalMapping);
  PetscErrorCode (*setvalueslocal)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
  PetscErrorCode (*zerorowslocal)(Mat,PetscInt,const PetscInt[],PetscScalar);
  /*70*/
  PetscErrorCode (*getrowmax)(Mat,Vec);
  PetscErrorCode (*convert)(Mat, MatType,MatReuse,Mat*);
  PetscErrorCode (*setcoloring)(Mat,ISColoring);
  PetscErrorCode (*setvaluesadic)(Mat,void*);
  PetscErrorCode (*setvaluesadifor)(Mat,PetscInt,void*);
  /*75*/
  PetscErrorCode (*fdcoloringapply)(Mat,MatFDColoring,Vec,MatStructure*,void*);
  PetscErrorCode (*setfromoptions)(Mat);
  PetscErrorCode (*multconstrained)(Mat,Vec,Vec);
  PetscErrorCode (*multtransposeconstrained)(Mat,Vec,Vec);
  PetscErrorCode (*ilufactorsymbolicconstrained)(Mat,IS,IS,double,PetscInt,PetscInt,Mat *);
  /*80*/
  PetscErrorCode (*permutesparsify)(Mat, PetscInt, double, double, IS, IS, Mat *);
  PetscErrorCode (*mults)(Mat, Vecs, Vecs);
  PetscErrorCode (*solves)(Mat, Vecs, Vecs);
  PetscErrorCode (*getinertia)(Mat,PetscInt*,PetscInt*,PetscInt*);
  PetscErrorCode (*load)(PetscViewer, MatType,Mat*);
  /*85*/
  PetscErrorCode (*issymmetric)(Mat,PetscReal,PetscTruth*);
  PetscErrorCode (*ishermitian)(Mat,PetscTruth*);
  PetscErrorCode (*isstructurallysymmetric)(Mat,PetscTruth*);
  PetscErrorCode (*pbrelax)(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);
  PetscErrorCode (*getvecs)(Mat,Vec*,Vec*);
  /*90*/
  PetscErrorCode (*matmult)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*matmultsymbolic)(Mat,Mat,PetscReal,Mat*);
  PetscErrorCode (*matmultnumeric)(Mat,Mat,Mat);
  PetscErrorCode (*ptap)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*ptapsymbolic)(Mat,Mat,PetscReal,Mat*); /* double dispatch wrapper routine */
  /*95*/
  PetscErrorCode (*ptapnumeric)(Mat,Mat,Mat);             /* double dispatch wrapper routine */
  PetscErrorCode (*matmulttranspose)(Mat,Mat,MatReuse,PetscReal,Mat*);
  PetscErrorCode (*matmulttransposesymbolic)(Mat,Mat,PetscReal,Mat*);
  PetscErrorCode (*matmulttransposenumeric)(Mat,Mat,Mat);
  PetscErrorCode (*ptapsymbolic_seqaij)(Mat,Mat,PetscReal,Mat*); /* actual implememtation, A=seqaij */
  /*100*/
  PetscErrorCode (*ptapnumeric_seqaij)(Mat,Mat,Mat);             /* actual implememtation, A=seqaij */
  PetscErrorCode (*ptapsymbolic_mpiaij)(Mat,Mat,PetscReal,Mat*); /* actual implememtation, A=mpiaij */
  PetscErrorCode (*ptapnumeric_mpiaij)(Mat,Mat,Mat);             /* actual implememtation, A=mpiaij */
  PetscErrorCode (*conjugate)(Mat);                              /* complex conjugate */
  PetscErrorCode (*setsizes)(Mat,PetscInt,PetscInt,PetscInt,PetscInt);
  /*105*/
  PetscErrorCode (*setvaluesrow)(Mat,PetscInt,const MatScalar[]);
  PetscErrorCode (*realpart)(Mat);
  PetscErrorCode (*imaginarypart)(Mat);
  PetscErrorCode (*getrowuppertriangular)(Mat);
  PetscErrorCode (*restorerowuppertriangular)(Mat);
  /*110*/
  PetscErrorCode (*matsolve)(Mat,Mat,Mat);
};
/*
    If you add MatOps entries above also add them to the MATOP enum
    in include/petscmat.h and include/finclude/petscmat.h
*/

/*
   Utility private matrix routines
*/
EXTERN PetscErrorCode MatConvert_Basic(Mat, MatType,MatReuse,Mat*);
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
#if !defined(_MatStashSpace_h_)
#define _MatStashSpace_h_

#include "petsc.h"

typedef struct _MatStashSpace *PetscMatStashSpace;

struct _MatStashSpace {
  PetscMatStashSpace next;
  MatScalar          *space_head,*val;
  PetscInt           *idx,*idy;
  PetscInt           total_space_size;
  PetscInt           local_used;
  PetscInt           local_remaining;
};

EXTERN PetscErrorCode PetscMatStashSpaceGet(PetscInt,PetscInt,PetscMatStashSpace *);
EXTERN PetscErrorCode PetscMatStashSpaceContiguous(PetscInt,PetscMatStashSpace *,PetscScalar *,PetscInt *,PetscInt *);
EXTERN PetscErrorCode PetscMatStashSpaceDestroy(PetscMatStashSpace);

#endif

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
  MatScalar     *svalues;               /* sending data */
  MatScalar     **rvalues;              /* receiving data (values) */
  PetscInt      **rindices;             /* receiving data (indices) */
  PetscMPIInt   *nprocs;                /* tmp data used both during scatterbegin and end */
  PetscInt      nprocessed;             /* number of messages already processed */
} MatStash;

EXTERN PetscErrorCode MatStashCreate_Private(MPI_Comm,PetscInt,MatStash*);
EXTERN PetscErrorCode MatStashDestroy_Private(MatStash*);
EXTERN PetscErrorCode MatStashScatterEnd_Private(MatStash*);
EXTERN PetscErrorCode MatStashSetInitialSize_Private(MatStash*,PetscInt);
EXTERN PetscErrorCode MatStashGetInfo_Private(MatStash*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode MatStashValuesRow_Private(MatStash*,PetscInt,PetscInt,const PetscInt[],const MatScalar[]);
EXTERN PetscErrorCode MatStashValuesCol_Private(MatStash*,PetscInt,PetscInt,const PetscInt[],const MatScalar[],PetscInt);
EXTERN PetscErrorCode MatStashValuesRowBlocked_Private(MatStash*,PetscInt,PetscInt,const PetscInt[],const MatScalar[],PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode MatStashValuesColBlocked_Private(MatStash*,PetscInt,PetscInt,const PetscInt[],const MatScalar[],PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode MatStashScatterBegin_Private(MatStash*,PetscInt*);
EXTERN PetscErrorCode MatStashScatterGetMesg_Private(MatStash*,PetscMPIInt*,PetscInt**,PetscInt**,MatScalar**,PetscInt*);

#define FACTOR_LU       1
#define FACTOR_CHOLESKY 2

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
  PetscMap               rmap,cmap;
  void                   *data;            /* implementation-specific data */
  PetscInt               factor;           /* 0, FACTOR_LU, or FACTOR_CHOLESKY */
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
  PetscInt       freq;             /* frequency at which new Jacobian is computed */
  Vec            w1,w2,w3;         /* work vectors used in computing Jacobian */
  PetscErrorCode (*f)(void);       /* function that defines Jacobian */
  void           *fctx;            /* optional user-defined context for use by the function f */
  PetscInt       **vscaleforrow;   /* location in vscale for each columnsforrow[] entry */
  Vec            vscale;           /* holds FD scaling, i.e. 1/dx for each perturbed column */
  PetscTruth     usersetsrecompute;/* user determines when Jacobian is recomputed, via MatFDColoringSetRecompute() */
  PetscTruth     recompute;        /* used with usersetrecompute to determine if Jacobian should be recomputed */
  Vec            F;                /* current value of user provided function; can set with MatFDColoringSetF() */
  PetscInt       currentcolor;     /* color for which function evaluation is being done now */
  const char     *htype;            /* "wp" or "ds" */
  ISColoringType ctype;            /* IS_COLORING_LOCAL or IS_COLORING_GHOSTED */
};

/*
   Null space context for preconditioner/operators
*/
struct _p_MatNullSpace {
  PETSCHEADER(int);
  PetscTruth     has_cnst;
  PetscInt       n;
  Vec*           vecs;
  Vec            vec;                   /* for out of place removals */
  PetscErrorCode (*remove)(Vec,void*);  /* for user provided removal function */
  void*          rmctx;                 /* context for remove() function */
};

/* 
   Checking zero pivot for LU, ILU preconditioners.
*/
typedef struct {
  PetscInt       nshift,nshift_max;
  PetscReal      shift_amount,shift_lo,shift_hi,shift_top;
  PetscTruth     lushift;
  PetscReal      rs;  /* active row sum of abs(offdiagonals) */
  PetscScalar    pv;  /* pivot of the active row */
} LUShift_Ctx;

EXTERN PetscErrorCode MatFactorDumpMatrix(Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatLUCheckShift_inline"
/*@C
   MatLUCheckShift_inline - shift the diagonals when zero pivot is detected on LU factor

   Collective on Mat

   Input Parameters:
+  info - information about the matrix factorization 
.  sctx - pointer to the struct LUShift_Ctx
.  row  - active row index
-  idiag - index of diagonals in array aval

   Output  Parameter:
+  newshift - 0: shift is unchanged; 1: shft is updated; -1: zeropivot  

   Level: developer
@*/
#define MatLUCheckShift_inline(info,sctx,row,idiag,newshift) 0;\
{\
  PetscInt  _newshift;\
  PetscReal _rs   = sctx.rs;\
  PetscReal _zero = info->zeropivot*_rs;\
  if (info->shiftnz && PetscAbsScalar(sctx.pv) <= _zero){\
    /* force |diag| > zeropivot*rs */\
    if (!sctx.nshift){\
      sctx.shift_amount = info->shiftnz;\
    } else {\
      sctx.shift_amount *= 2.0;\
    }\
    sctx.lushift = PETSC_TRUE;\
    (sctx.nshift)++;\
    _newshift = 1;\
  } else if (info->shiftpd && PetscRealPart(sctx.pv) <= _zero){\
    /* force matfactor to be diagonally dominant */\
    if (sctx.nshift > sctx.nshift_max) {\
      ierr = MatFactorDumpMatrix(A);CHKERRQ(ierr);\
      SETERRQ1(PETSC_ERR_CONV_FAILED,"Unable to determine shift to enforce positive definite preconditioner after %d tries",sctx.nshift);\
    } else if (sctx.nshift == sctx.nshift_max) {\
      info->shift_fraction = sctx.shift_hi;\
      sctx.lushift        = PETSC_TRUE;\
    } else {\
      sctx.shift_lo = info->shift_fraction;\
      info->shift_fraction = (sctx.shift_hi+sctx.shift_lo)/2.;\
      sctx.lushift  = PETSC_TRUE;\
    }\
    sctx.shift_amount = info->shift_fraction * sctx.shift_top;\
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
  if (info->shiftnz && PetscAbsScalar(sctx.pv) <= _zero){\
    /* force |diag| > zeropivot*sctx.rs */\
    if (!sctx.nshift){\
      sctx.shift_amount = info->shiftnz;\
    } else {\
      sctx.shift_amount *= 2.0;\
    }\
    sctx.chshift = PETSC_TRUE;\
    sctx.nshift++;\
    _newshift = 1;\
  } else if (info->shiftpd && PetscRealPart(sctx.pv) <= _zero){\
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
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
    diag      - index of the active row in LUFactorSymbolic
    nzbd      - number of input indices with indices <= idx_start
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
    im        - im[idx_start] =  num of entries with indices <= diag
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

extern PetscEvent  MAT_Mult, MAT_MultMatrixFree, MAT_Mults, MAT_MultConstrained, MAT_MultAdd, MAT_MultTranspose;
extern PetscEvent  MAT_MultTransposeConstrained, MAT_MultTransposeAdd, MAT_Solve, MAT_Solves, MAT_SolveAdd, MAT_SolveTranspose;
extern PetscEvent  MAT_SolveTransposeAdd, MAT_Relax, MAT_ForwardSolve, MAT_BackwardSolve, MAT_LUFactor, MAT_LUFactorSymbolic;
extern PetscEvent  MAT_LUFactorNumeric, MAT_CholeskyFactor, MAT_CholeskyFactorSymbolic, MAT_CholeskyFactorNumeric, MAT_ILUFactor;
extern PetscEvent  MAT_ILUFactorSymbolic, MAT_ICCFactorSymbolic, MAT_Copy, MAT_Convert, MAT_Scale, MAT_AssemblyBegin;
extern PetscEvent  MAT_AssemblyEnd, MAT_SetValues, MAT_GetValues, MAT_GetRow, MAT_GetSubMatrices, MAT_GetColoring, MAT_GetOrdering;
extern PetscEvent  MAT_IncreaseOverlap, MAT_Partitioning, MAT_ZeroEntries, MAT_Load, MAT_View, MAT_AXPY, MAT_FDColoringCreate;
extern PetscEvent  MAT_FDColoringApply, MAT_Transpose, MAT_FDColoringFunction;
extern PetscEvent  MAT_MatMult, MAT_MatMultSymbolic, MAT_MatMultNumeric;
extern PetscEvent  MAT_PtAP, MAT_PtAPSymbolic, MAT_PtAPNumeric;
extern PetscEvent  MAT_MatMultTranspose, MAT_MatMultTransposeSymbolic, MAT_MatMultTransposeNumeric;

#endif







