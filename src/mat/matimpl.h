
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
  PetscErrorCode (*setvalues)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode),
                 (*getrow)(Mat,PetscInt,PetscInt *,PetscInt*[],PetscScalar*[]),
                 (*restorerow)(Mat,PetscInt,PetscInt *,PetscInt *[],PetscScalar *[]),
                 (*mult)(Mat,Vec,Vec),
/* 4*/           (*multadd)(Mat,Vec,Vec,Vec),
                 (*multtranspose)(Mat,Vec,Vec),
                 (*multtransposeadd)(Mat,Vec,Vec,Vec),
                 (*solve)(Mat,Vec,Vec),
                 (*solveadd)(Mat,Vec,Vec,Vec),
                 (*solvetranspose)(Mat,Vec,Vec),
/*10*/           (*solvetransposeadd)(Mat,Vec,Vec,Vec),
                 (*lufactor)(Mat,IS,IS,MatFactorInfo*),
                 (*choleskyfactor)(Mat,IS,MatFactorInfo*),
                 (*relax)(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec),
                 (*transpose)(Mat,Mat *),
/*15*/           (*getinfo)(Mat,MatInfoType,MatInfo*),
                 (*equal)(Mat,Mat,PetscTruth *),
                 (*getdiagonal)(Mat,Vec),
                 (*diagonalscale)(Mat,Vec,Vec),
                 (*norm)(Mat,NormType,PetscReal *),
/*20*/           (*assemblybegin)(Mat,MatAssemblyType),
                 (*assemblyend)(Mat,MatAssemblyType),
                 (*compress)(Mat),
                 (*setoption)(Mat,MatOption),
                 (*zeroentries)(Mat),
/*25*/           (*zerorows)(Mat,IS,const PetscScalar *),
                 (*lufactorsymbolic)(Mat,IS,IS,MatFactorInfo*,Mat *),
                 (*lufactornumeric)(Mat,Mat *),
                 (*choleskyfactorsymbolic)(Mat,IS,MatFactorInfo*,Mat *),
                 (*choleskyfactornumeric)(Mat,Mat *),
/*30*/           (*setuppreallocation)(Mat),
                 (*ilufactorsymbolic)(Mat,IS,IS,MatFactorInfo*,Mat *),
                 (*iccfactorsymbolic)(Mat,IS,MatFactorInfo*,Mat *),
                 (*getarray)(Mat,PetscScalar **),
                 (*restorearray)(Mat,PetscScalar **),
/*35*/           (*duplicate)(Mat,MatDuplicateOption,Mat *),
                 (*forwardsolve)(Mat,Vec,Vec),
                 (*backwardsolve)(Mat,Vec,Vec),
                 (*ilufactor)(Mat,IS,IS,MatFactorInfo*),
                 (*iccfactor)(Mat,IS,MatFactorInfo*),
/*40*/           (*axpy)(const PetscScalar *,Mat,Mat,MatStructure),
                 (*getsubmatrices)(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat *[]),
                 (*increaseoverlap)(Mat,PetscInt,IS[],PetscInt),
                 (*getvalues)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar []),
                 (*copy)(Mat,Mat,MatStructure),
/*45*/           (*printhelp)(Mat),
                 (*scale)(const PetscScalar *,Mat),
                 (*shift)(const PetscScalar *,Mat),
                 (*diagonalset)(Mat,Vec,InsertMode),
                 (*iludtfactor)(Mat,MatFactorInfo*,IS,IS,Mat *),
/*50*/           (*setblocksize)(Mat,PetscInt),
                 (*getrowij)(Mat,PetscInt,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *),
                 (*restorerowij)(Mat,PetscInt,PetscTruth,PetscInt *,PetscInt *[],PetscInt *[],PetscTruth *),
                 (*getcolumnij)(Mat,PetscInt,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *),
                 (*restorecolumnij)(Mat,PetscInt,PetscTruth,PetscInt*,PetscInt *[],PetscInt *[],PetscTruth *),
/*55*/           (*fdcoloringcreate)(Mat,ISColoring,MatFDColoring),
                 (*coloringpatch)(Mat,PetscInt,PetscInt,ISColoringValue[],ISColoring*),
                 (*setunfactored)(Mat),
                 (*permute)(Mat,IS,IS,Mat*),
                 (*setvaluesblocked)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode),
/*60*/           (*getsubmatrix)(Mat,IS,IS,PetscInt,MatReuse,Mat*),
                 (*destroy)(Mat),
                 (*view)(Mat,PetscViewer),
                 (*getmaps)(Mat,PetscMap*,PetscMap*),
                 (*usescaledform)(Mat,PetscTruth),
/*65*/           (*scalesystem)(Mat,Vec,Vec),
                 (*unscalesystem)(Mat,Vec,Vec),
                 (*setlocaltoglobalmapping)(Mat,ISLocalToGlobalMapping),
                 (*setvalueslocal)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode),
                 (*zerorowslocal)(Mat,IS,const PetscScalar *),
/*70*/           (*getrowmax)(Mat,Vec),
                 (*convert)(Mat,const MatType,Mat*),
                 (*setcoloring)(Mat,ISColoring),
                 (*setvaluesadic)(Mat,void*),
                 (*setvaluesadifor)(Mat,PetscInt,void*),
/*75*/           (*fdcoloringapply)(Mat,MatFDColoring,Vec,MatStructure*,void*),
                 (*setfromoptions)(Mat),
                 (*multconstrained)(Mat,Vec,Vec),
                 (*multtransposeconstrained)(Mat,Vec,Vec),
                 (*ilufactorsymbolicconstrained)(Mat,IS,IS,double,PetscInt,PetscInt,Mat *),
/*80*/           (*permutesparsify)(Mat, PetscInt, double, double, IS, IS, Mat *),
                 (*mults)(Mat, Vecs, Vecs),
                 (*solves)(Mat, Vecs, Vecs),
                 (*getinertia)(Mat,PetscInt*,PetscInt*,PetscInt*),
                 (*load)(PetscViewer,const MatType,Mat*),
/*85*/           (*issymmetric)(Mat,PetscReal,PetscTruth*),
                 (*ishermitian)(Mat,PetscTruth*),
                 (*isstructurallysymmetric)(Mat,PetscTruth*),
                 (*pbrelax)(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec),
                 (*getvecs)(Mat,Vec*,Vec*),
/*90*/           (*matmult)(Mat,Mat,MatReuse,PetscReal,Mat*),
                 (*matmultsymbolic)(Mat,Mat,PetscReal,Mat*),
                 (*matmultnumeric)(Mat,Mat,Mat),
                 (*ptap)(Mat,Mat,MatReuse,PetscReal,Mat*),
                 (*ptapsymbolic)(Mat,Mat,PetscReal,Mat*),
/*95*/           (*ptapnumeric)(Mat,Mat,Mat),
                 (*matmulttranspose)(Mat,Mat,MatReuse,PetscReal,Mat*),
                 (*matmulttransposesymbolic)(Mat,Mat,PetscReal,Mat*),
                 (*matmulttransposenumeric)(Mat,Mat,Mat);
};

/*
   Utility private matrix routines
*/
EXTERN PetscErrorCode MatConvert_Basic(Mat,const MatType,Mat*);
EXTERN PetscErrorCode MatCopy_Basic(Mat,Mat,MatStructure);
EXTERN PetscErrorCode MatView_Private(Mat);
EXTERN PetscErrorCode MatGetPetscMaps_Petsc(Mat,PetscMap *,PetscMap *);
EXTERN PetscErrorCode MatHeaderCopy(Mat,Mat);
EXTERN PetscErrorCode MatAXPYGetxtoy_Private(PetscInt,PetscInt*,PetscInt*,PetscInt*, PetscInt*,PetscInt*,PetscInt*, PetscInt**);

/* 
  The stash is used to temporarily store inserted matrix values that 
  belong to another processor. During the assembly phase the stashed 
  values are moved to the correct processor and 
*/

typedef struct {
  PetscInt      nmax;                   /* maximum stash size */
  PetscInt      umax;                   /* user specified max-size */
  PetscInt      oldnmax;                /* the nmax value used previously */
  PetscInt      n;                      /* stash size */
  PetscInt      bs;                     /* block size of the stash */
  PetscInt      reallocs;               /* preserve the no of mallocs invoked */           
  PetscInt      *idx;                   /* global row numbers in stash */
  PetscInt      *idy;                   /* global column numbers in stash */
  MatScalar     *array;                 /* array to hold stashed values */
  /* The following variables are used for communication */
  MPI_Comm      comm;
  PetscMPIInt   size,rank;
  PetscMPIInt   tag1,tag2;
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  MPI_Status    *send_status;           /* array of send status */
  PetscInt      nsends,nrecvs;         /* numbers of sends and receives */
  MatScalar     *svalues,*rvalues;     /* sending and receiving data */
  PetscInt      rmax;                   /* maximum message length */
  PetscInt      *nprocs;                /* tmp data used both duiring scatterbegin and end */
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
EXTERN PetscErrorCode Mat_CheckCompressedRow(Mat,Mat_CompressedRow*,PetscInt*,PetscReal);

struct _p_Mat {
  PETSCHEADER(struct _MatOps)
  PetscMap               rmap,cmap;
  void                   *data;            /* implementation-specific data */
  PetscInt               factor;           /* 0, FACTOR_LU, or FACTOR_CHOLESKY */
  PetscReal              lupivotthreshold; /* threshold for pivoting */
  PetscTruth             assembled;        /* is the matrix assembled? */
  PetscTruth             was_assembled;    /* new values inserted into assembled mat */
  PetscInt               num_ass;          /* number of times matrix has been assembled */
  PetscTruth             same_nonzero;     /* matrix has same nonzero pattern as previous */
  PetscInt               M,N;             /* global numbers of rows, columns */
  PetscInt               m,n;             /* local numbers of rows, columns */
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
  PetscInt               bs;
  void                   *spptr;          /* pointer for special library like SuperLU */
};

#define MatPreallocated(A) {PetscInt _e;if (!(A)->preallocated) {_e = MatSetUpPreallocation(A);CHKERRQ(_e);}}
extern PetscErrorCode MatAXPY_Basic(const PetscScalar*,Mat,Mat,MatStructure);

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
  PETSCHEADER(struct _MatPartitioningOps)
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
  PETSCHEADER(int)
  PetscInt   M,N,m;            /* total rows, columns; local rows */
  PetscInt   rstart;           /* first row owned by local processor */
  PetscInt   ncolors;          /* number of colors */
  PetscInt   *ncolumns;        /* number of local columns for a color */ 
  PetscInt   **columns;        /* lists the local columns of each color (using global column numbering) */
  PetscInt   *nrows;           /* number of local rows for each color */
  PetscInt   **rows;           /* lists the local rows for each color (using the local row numbering) */
  PetscInt   **columnsforrow;  /* lists the corresponding columns for those rows (using the global column) */ 
  PetscReal  error_rel;        /* square root of relative error in computing function */
  PetscReal  umin;             /* minimum allowable u'dx value */
  PetscInt   freq;             /* frequency at which new Jacobian is computed */
  Vec        w1,w2,w3;         /* work vectors used in computing Jacobian */
  PetscErrorCode (*f)(void);       /* function that defines Jacobian */
  void       *fctx;            /* optional user-defined context for use by the function f */
  PetscInt   **vscaleforrow;   /* location in vscale for each columnsforrow[] entry */
  Vec        vscale;           /* holds FD scaling, i.e. 1/dx for each perturbed column */
  PetscTruth usersetsrecompute;/* user determines when Jacobian is recomputed, via MatFDColoringSetRecompute() */
  PetscTruth recompute;        /* used with usersetrecompute to determine if Jacobian should be recomputed */
  Vec        F;                /* current value of user provided function; can set with MatFDColoringSetF() */
  PetscInt   currentcolor;     /* color for which function evaluation is being done now */
};

/*
   Null space context for preconditioner/operators
*/
struct _p_MatNullSpace {
  PETSCHEADER(int)
  PetscTruth  has_cnst;
  PetscInt    n;
  Vec*        vecs;
  Vec         vec;      /* for out of place removals */
};

#endif






