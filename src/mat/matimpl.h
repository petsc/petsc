/* $Id: matimpl.h,v 1.126 2001/08/21 21:02:01 bsmith Exp $ */

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
  int       (*setvalues)(Mat,int,const int[],int,const int[],const PetscScalar[],InsertMode),
            (*getrow)(Mat,int,int *,int*[],PetscScalar*[]),
            (*restorerow)(Mat,int,int *,int *[],PetscScalar *[]),
            (*mult)(Mat,Vec,Vec),
/* 4*/      (*multadd)(Mat,Vec,Vec,Vec),
            (*multtranspose)(Mat,Vec,Vec),
            (*multtransposeadd)(Mat,Vec,Vec,Vec),
            (*solve)(Mat,Vec,Vec),
            (*solveadd)(Mat,Vec,Vec,Vec),
            (*solvetranspose)(Mat,Vec,Vec),
/*10*/      (*solvetransposeadd)(Mat,Vec,Vec,Vec),
            (*lufactor)(Mat,IS,IS,MatFactorInfo*),
            (*choleskyfactor)(Mat,IS,MatFactorInfo*),
            (*relax)(Mat,Vec,PetscReal,MatSORType,PetscReal,int,int,Vec),
            (*transpose)(Mat,Mat *),
/*15*/      (*getinfo)(Mat,MatInfoType,MatInfo*),
            (*equal)(Mat,Mat,PetscTruth *),
            (*getdiagonal)(Mat,Vec),
            (*diagonalscale)(Mat,Vec,Vec),
            (*norm)(Mat,NormType,PetscReal *),
/*20*/      (*assemblybegin)(Mat,MatAssemblyType),
            (*assemblyend)(Mat,MatAssemblyType),
            (*compress)(Mat),
            (*setoption)(Mat,MatOption),
            (*zeroentries)(Mat),
/*25*/      (*zerorows)(Mat,IS,const PetscScalar *),
            (*lufactorsymbolic)(Mat,IS,IS,MatFactorInfo*,Mat *),
            (*lufactornumeric)(Mat,Mat *),
            (*choleskyfactorsymbolic)(Mat,IS,MatFactorInfo*,Mat *),
            (*choleskyfactornumeric)(Mat,Mat *),
/*30*/      (*setuppreallocation)(Mat),
            (*ilufactorsymbolic)(Mat,IS,IS,MatFactorInfo*,Mat *),
            (*iccfactorsymbolic)(Mat,IS,MatFactorInfo*,Mat *),
            (*getarray)(Mat,PetscScalar **),
            (*restorearray)(Mat,PetscScalar **),
/*35*/      (*duplicate)(Mat,MatDuplicateOption,Mat *),
            (*forwardsolve)(Mat,Vec,Vec),
            (*backwardsolve)(Mat,Vec,Vec),
            (*ilufactor)(Mat,IS,IS,MatFactorInfo*),
            (*iccfactor)(Mat,IS,MatFactorInfo*),
/*40*/      (*axpy)(const PetscScalar *,Mat,Mat,MatStructure),
            (*getsubmatrices)(Mat,int,const IS[],const IS[],MatReuse,Mat *[]),
            (*increaseoverlap)(Mat,int,IS[],int),
            (*getvalues)(Mat,int,const int[],int,const int[],PetscScalar []),
            (*copy)(Mat,Mat,MatStructure),
/*45*/      (*printhelp)(Mat),
            (*scale)(const PetscScalar *,Mat),
            (*shift)(const PetscScalar *,Mat),
            (*diagonalset)(Mat,Vec,InsertMode),
            (*iludtfactor)(Mat,MatFactorInfo*,IS,IS,Mat *),
/*50*/      (*getblocksize)(Mat,int *),
            (*getrowij)(Mat,int,PetscTruth,int*,int *[],int *[],PetscTruth *),
            (*restorerowij)(Mat,int,PetscTruth,int *,int *[],int *[],PetscTruth *),
            (*getcolumnij)(Mat,int,PetscTruth,int*,int *[],int *[],PetscTruth *),
            (*restorecolumnij)(Mat,int,PetscTruth,int*,int *[],int *[],PetscTruth *),
/*55*/      (*fdcoloringcreate)(Mat,ISColoring,MatFDColoring),
            (*coloringpatch)(Mat,int,int,const ISColoringValue[],ISColoring*),
            (*setunfactored)(Mat),
            (*permute)(Mat,IS,IS,Mat*),
            (*setvaluesblocked)(Mat,int,const int[],int,const int[],const PetscScalar[],InsertMode),
/*60*/      (*getsubmatrix)(Mat,IS,IS,int,MatReuse,Mat*),
            (*destroy)(Mat),
            (*view)(Mat,PetscViewer),
            (*getmaps)(Mat,PetscMap*,PetscMap*),
            (*usescaledform)(Mat,PetscTruth),
/*65*/      (*scalesystem)(Mat,Vec,Vec),
            (*unscalesystem)(Mat,Vec,Vec),
            (*setlocaltoglobalmapping)(Mat,ISLocalToGlobalMapping),
            (*setvalueslocal)(Mat,int,const int[],int,const int[],const PetscScalar[],InsertMode),
            (*zerorowslocal)(Mat,IS,const PetscScalar *),
/*70*/      (*getrowmax)(Mat,Vec),
            (*convert)(Mat,MatType,Mat*),
            (*setcoloring)(Mat,ISColoring),
            (*setvaluesadic)(Mat,void*),
            (*setvaluesadifor)(Mat,int,void*),
/*75*/      (*fdcoloringapply)(Mat,MatFDColoring,Vec,MatStructure*,void*),
            (*setfromoptions)(Mat),
            (*multconstrained)(Mat,Vec,Vec),
            (*multtransposeconstrained)(Mat,Vec,Vec),
            (*ilufactorsymbolicconstrained)(Mat,IS,IS,double,int,int,Mat *),
/*80*/      (*serialize)(MPI_Comm, Mat *, PetscViewer, PetscTruth),
            (*permutesparsify)(Mat, int, double, double, IS, IS, Mat *),
            (*mults)(Mat, Vecs, Vecs),
            (*solves)(Mat, Vecs, Vecs),
            (*getinertia)(Mat,int*,int*,int*),
/*85*/      (*load)(PetscViewer,MatType,Mat*);
};

/*
   Utility private matrix routines
*/
EXTERN int MatConvert_Basic(Mat,MatType,Mat*);
EXTERN int MatCopy_Basic(Mat,Mat,MatStructure);
EXTERN int MatView_Private(Mat);
EXTERN int MatGetPetscMaps_Petsc(Mat,PetscMap *,PetscMap *);
EXTERN int MatHeaderCopy(Mat,Mat);
EXTERN int MatAXPYGetxtoy_Private(int,int*,int*,int*, int*,int*,int*, int**);

/* 
  The stash is used to temporarily store inserted matrix values that 
  belong to another processor. During the assembly phase the stashed 
  values are moved to the correct processor and 
*/

typedef struct {
  int           nmax;                   /* maximum stash size */
  int           umax;                   /* user specified max-size */
  int           oldnmax;                /* the nmax value used previously */
  int           n;                      /* stash size */
  int           bs;                     /* block size of the stash */
  int           reallocs;               /* preserve the no of mallocs invoked */           
  int           *idx;                   /* global row numbers in stash */
  int           *idy;                   /* global column numbers in stash */
  MatScalar     *array;                 /* array to hold stashed values */
  /* The following variables are used for communication */
  MPI_Comm      comm;
  int           size,rank;
  int           tag1,tag2;
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  MPI_Status    *send_status;           /* array of send status */
  int           nsends,nrecvs;         /* numbers of sends and receives */
  MatScalar     *svalues,*rvalues;     /* sending and receiving data */
  int           rmax;                   /* maximum message length */
  int           *nprocs;                /* tmp data used both duiring scatterbegin and end */
  int           nprocessed;             /* number of messages already processed */
} MatStash;

EXTERN int MatStashCreate_Private(MPI_Comm,int,MatStash*);
EXTERN int MatStashDestroy_Private(MatStash*);
EXTERN int MatStashScatterEnd_Private(MatStash*);
EXTERN int MatStashSetInitialSize_Private(MatStash*,int);
EXTERN int MatStashGetInfo_Private(MatStash*,int*,int*);
EXTERN int MatStashValuesRow_Private(MatStash*,int,int,const int[],const MatScalar[]);
EXTERN int MatStashValuesCol_Private(MatStash*,int,int,const int[],const MatScalar[],int);
EXTERN int MatStashValuesRowBlocked_Private(MatStash*,int,int,const int[],const MatScalar[],int,int,int);
EXTERN int MatStashValuesColBlocked_Private(MatStash*,int,int,const int[],const MatScalar[],int,int,int);
EXTERN int MatStashScatterBegin_Private(MatStash*,int*);
EXTERN int MatStashScatterGetMesg_Private(MatStash*,int*,int**,int**,MatScalar**,int*);

#define FACTOR_LU       1
#define FACTOR_CHOLESKY 2

typedef struct {
  int        dim;
  int        dims[4];
  int        starts[4];
  PetscTruth noc;        /* this is a single component problem, hence user will not set MatStencil.c */
} MatStencilInfo;

struct _p_Mat {
  PETSCHEADER(struct _MatOps)
  PetscMap               rmap,cmap;
  void                   *data;            /* implementation-specific data */
  int                    factor;           /* 0, FACTOR_LU, or FACTOR_CHOLESKY */
  PetscReal              lupivotthreshold; /* threshold for pivoting */
  PetscTruth             assembled;        /* is the matrix assembled? */
  PetscTruth             was_assembled;    /* new values inserted into assembled mat */
  int                    num_ass;          /* number of times matrix has been assembled */
  PetscTruth             same_nonzero;     /* matrix has same nonzero pattern as previous */
  int                    M,N;             /* global numbers of rows, columns */
  int                    m,n;             /* local numbers of rows, columns */
  MatInfo                info;             /* matrix information */
  ISLocalToGlobalMapping mapping;          /* mapping used in MatSetValuesLocal() */
  ISLocalToGlobalMapping bmapping;         /* mapping used in MatSetValuesBlockedLocal() */
  InsertMode             insertmode;       /* have values been inserted in matrix or added? */
  MatStash               stash,bstash;     /* used for assembling off-proc mat emements */
  MatNullSpace           nullsp;
  PetscTruth             preallocated;
  MatStencilInfo         stencil;          /* information for structured grid */
  PetscTruth             symmetric,structurally_symmetric;
  void                   *spptr;          /* pointer for special library like SuperLU */
  void                   *esimat;
};

#define MatPreallocated(A) {int _e;if (!(A)->preallocated) {_e = MatSetUpPreallocation(A);CHKERRQ(_e);}}
extern int MatAXPY_Basic(const PetscScalar*,Mat,Mat,MatStructure);

/*
    Object for partitioning graphs
*/

typedef struct _MatPartitioningOps *MatPartitioningOps;
struct _MatPartitioningOps {
  int         (*apply)(MatPartitioning,IS*);
  int         (*setfromoptions)(MatPartitioning);
  int         (*destroy)(MatPartitioning);
  int         (*view)(MatPartitioning,PetscViewer);
};

struct _p_MatPartitioning {
  PETSCHEADER(struct _MatPartitioningOps)
  Mat         adj;
  int         *vertex_weights;
  PetscReal   *part_weights;
  int         n;                                 /* number of partitions */
  void        *data;
  int         setupcalled;
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
  int        M,N,m;            /* total rows, columns; local rows */
  int        rstart;           /* first row owned by local processor */
  int        ncolors;          /* number of colors */
  int        *ncolumns;        /* number of local columns for a color */ 
  int        **columns;        /* lists the local columns of each color (using global column numbering) */
  int        *nrows;           /* number of local rows for each color */
  int        **rows;           /* lists the local rows for each color (using the local row numbering) */
  int        **columnsforrow;  /* lists the corresponding columns for those rows (using the global column) */ 
  PetscReal  error_rel;        /* square root of relative error in computing function */
  PetscReal  umin;             /* minimum allowable u'dx value */
  int        freq;             /* frequency at which new Jacobian is computed */
  Vec        w1,w2,w3;         /* work vectors used in computing Jacobian */
  int        (*f)(void);       /* function that defines Jacobian */
  void       *fctx;            /* optional user-defined context for use by the function f */
  int        **vscaleforrow;   /* location in vscale for each columnsforrow[] entry */
  Vec        vscale;           /* holds FD scaling, i.e. 1/dx for each perturbed column */
  PetscTruth usersetsrecompute;/* user determines when Jacobian is recomputed, via MatFDColoringSetRecompute() */
  PetscTruth recompute;        /* used with usersetrecompute to determine if Jacobian should be recomputed */
  Vec        F;                /* current value of user provided function; can set with MatFDColoringSetF() */
  int        currentcolor;     /* color for which function evaluation is being done now */
};

/*
   Null space context for preconditioner/operators
*/
struct _p_MatNullSpace {
  PETSCHEADER(int)
  int         has_cnst;
  int         n;
  Vec*        vecs;
  Vec         vec;      /* for out of place removals */
};


#endif






