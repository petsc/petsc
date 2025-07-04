#pragma once

#include <petscmat.h>
#include <petscmatcoarsen.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      MatRegisterAllCalled;
PETSC_EXTERN PetscBool      MatSeqAIJRegisterAllCalled;
PETSC_EXTERN PetscBool      MatOrderingRegisterAllCalled;
PETSC_EXTERN PetscBool      MatColoringRegisterAllCalled;
PETSC_EXTERN PetscBool      MatPartitioningRegisterAllCalled;
PETSC_EXTERN PetscBool      MatCoarsenRegisterAllCalled;
PETSC_EXTERN PetscErrorCode MatRegisterAll(void);
PETSC_EXTERN PetscErrorCode MatOrderingRegisterAll(void);
PETSC_EXTERN PetscErrorCode MatColoringRegisterAll(void);
PETSC_EXTERN PetscErrorCode MatPartitioningRegisterAll(void);
PETSC_EXTERN PetscErrorCode MatCoarsenRegisterAll(void);
PETSC_EXTERN PetscErrorCode MatSeqAIJRegisterAll(void);

/* Gets the root type of the input matrix's type (e.g., MATAIJ for MATSEQAIJ) */
PETSC_EXTERN PetscErrorCode MatGetRootType_Private(Mat, MatType *);

/* Gets the MPI type corresponding to the input matrix's type (e.g., MATMPIAIJ for MATSEQAIJ) */
PETSC_INTERN PetscErrorCode MatGetMPIMatType_Private(Mat, MatType *);

/*
  This file defines the parts of the matrix data structure that are
  shared by all matrix types.
*/

/*
    If you add entries here also add them to the MATOP enum
    in include/petscmat.h
*/
typedef struct _MatOps *MatOps;
struct _MatOps {
  /* 0*/
  PetscErrorCode (*setvalues)(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
  PetscErrorCode (*getrow)(Mat, PetscInt, PetscInt *, PetscInt *[], PetscScalar *[]);
  PetscErrorCode (*restorerow)(Mat, PetscInt, PetscInt *, PetscInt *[], PetscScalar *[]);
  PetscErrorCode (*mult)(Mat, Vec, Vec);
  PetscErrorCode (*multadd)(Mat, Vec, Vec, Vec);
  /* 5*/
  PetscErrorCode (*multtranspose)(Mat, Vec, Vec);
  PetscErrorCode (*multtransposeadd)(Mat, Vec, Vec, Vec);
  PetscErrorCode (*solve)(Mat, Vec, Vec);
  PetscErrorCode (*solveadd)(Mat, Vec, Vec, Vec);
  PetscErrorCode (*solvetranspose)(Mat, Vec, Vec);
  /*10*/
  PetscErrorCode (*solvetransposeadd)(Mat, Vec, Vec, Vec);
  PetscErrorCode (*lufactor)(Mat, IS, IS, const MatFactorInfo *);
  PetscErrorCode (*choleskyfactor)(Mat, IS, const MatFactorInfo *);
  PetscErrorCode (*sor)(Mat, Vec, PetscReal, MatSORType, PetscReal, PetscInt, PetscInt, Vec);
  PetscErrorCode (*transpose)(Mat, MatReuse, Mat *);
  /*15*/
  PetscErrorCode (*getinfo)(Mat, MatInfoType, MatInfo *);
  PetscErrorCode (*equal)(Mat, Mat, PetscBool *);
  PetscErrorCode (*getdiagonal)(Mat, Vec);
  PetscErrorCode (*diagonalscale)(Mat, Vec, Vec);
  PetscErrorCode (*norm)(Mat, NormType, PetscReal *);
  /*20*/
  PetscErrorCode (*assemblybegin)(Mat, MatAssemblyType);
  PetscErrorCode (*assemblyend)(Mat, MatAssemblyType);
  PetscErrorCode (*setoption)(Mat, MatOption, PetscBool);
  PetscErrorCode (*zeroentries)(Mat);
  /*24*/
  PetscErrorCode (*zerorows)(Mat, PetscInt, const PetscInt[], PetscScalar, Vec, Vec);
  PetscErrorCode (*lufactorsymbolic)(Mat, Mat, IS, IS, const MatFactorInfo *);
  PetscErrorCode (*lufactornumeric)(Mat, Mat, const MatFactorInfo *);
  PetscErrorCode (*choleskyfactorsymbolic)(Mat, Mat, IS, const MatFactorInfo *);
  PetscErrorCode (*choleskyfactornumeric)(Mat, Mat, const MatFactorInfo *);
  /*29*/
  PetscErrorCode (*setup)(Mat);
  PetscErrorCode (*ilufactorsymbolic)(Mat, Mat, IS, IS, const MatFactorInfo *);
  PetscErrorCode (*iccfactorsymbolic)(Mat, Mat, IS, const MatFactorInfo *);
  PetscErrorCode (*getdiagonalblock)(Mat, Mat *);
  PetscErrorCode (*setinf)(Mat);
  /*34*/
  PetscErrorCode (*duplicate)(Mat, MatDuplicateOption, Mat *);
  PetscErrorCode (*forwardsolve)(Mat, Vec, Vec);
  PetscErrorCode (*backwardsolve)(Mat, Vec, Vec);
  PetscErrorCode (*ilufactor)(Mat, IS, IS, const MatFactorInfo *);
  PetscErrorCode (*iccfactor)(Mat, IS, const MatFactorInfo *);
  /*39*/
  PetscErrorCode (*axpy)(Mat, PetscScalar, Mat, MatStructure);
  PetscErrorCode (*createsubmatrices)(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat *[]);
  PetscErrorCode (*increaseoverlap)(Mat, PetscInt, IS[], PetscInt);
  PetscErrorCode (*getvalues)(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], PetscScalar[]);
  PetscErrorCode (*copy)(Mat, Mat, MatStructure);
  /*44*/
  PetscErrorCode (*getrowmax)(Mat, Vec, PetscInt[]);
  PetscErrorCode (*scale)(Mat, PetscScalar);
  PetscErrorCode (*shift)(Mat, PetscScalar);
  PetscErrorCode (*diagonalset)(Mat, Vec, InsertMode);
  PetscErrorCode (*zerorowscolumns)(Mat, PetscInt, const PetscInt[], PetscScalar, Vec, Vec);
  /*49*/
  PetscErrorCode (*setrandom)(Mat, PetscRandom);
  PetscErrorCode (*getrowij)(Mat, PetscInt, PetscBool, PetscBool, PetscInt *, const PetscInt *[], const PetscInt *[], PetscBool *);
  PetscErrorCode (*restorerowij)(Mat, PetscInt, PetscBool, PetscBool, PetscInt *, const PetscInt *[], const PetscInt *[], PetscBool *);
  PetscErrorCode (*getcolumnij)(Mat, PetscInt, PetscBool, PetscBool, PetscInt *, const PetscInt *[], const PetscInt *[], PetscBool *);
  PetscErrorCode (*restorecolumnij)(Mat, PetscInt, PetscBool, PetscBool, PetscInt *, const PetscInt *[], const PetscInt *[], PetscBool *);
  /*54*/
  PetscErrorCode (*fdcoloringcreate)(Mat, ISColoring, MatFDColoring);
  PetscErrorCode (*coloringpatch)(Mat, PetscInt, PetscInt, ISColoringValue[], ISColoring *);
  PetscErrorCode (*setunfactored)(Mat);
  PetscErrorCode (*permute)(Mat, IS, IS, Mat *);
  PetscErrorCode (*setvaluesblocked)(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
  /*59*/
  PetscErrorCode (*createsubmatrix)(Mat, IS, IS, MatReuse, Mat *);
  PetscErrorCode (*destroy)(Mat);
  PetscErrorCode (*view)(Mat, PetscViewer);
  PetscErrorCode (*convertfrom)(Mat, MatType, MatReuse, Mat *);
  PetscErrorCode (*matmatmultsymbolic)(Mat, Mat, Mat, PetscReal, Mat);
  /*64*/
  PetscErrorCode (*matmatmultnumeric)(Mat, Mat, Mat, Mat);
  PetscErrorCode (*setlocaltoglobalmapping)(Mat, ISLocalToGlobalMapping, ISLocalToGlobalMapping);
  PetscErrorCode (*setvalueslocal)(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
  PetscErrorCode (*zerorowslocal)(Mat, PetscInt, const PetscInt[], PetscScalar, Vec, Vec);
  PetscErrorCode (*getrowmaxabs)(Mat, Vec, PetscInt[]);
  /*69*/
  PetscErrorCode (*getrowminabs)(Mat, Vec, PetscInt[]);
  PetscErrorCode (*convert)(Mat, MatType, MatReuse, Mat *);
  PetscErrorCode (*hasoperation)(Mat, MatOperation, PetscBool *);
  PetscErrorCode (*fdcoloringapply)(Mat, MatFDColoring, Vec, void *);
  PetscErrorCode (*setfromoptions)(Mat, PetscOptionItems);
  /*74*/
  PetscErrorCode (*findzerodiagonals)(Mat, IS *);
  PetscErrorCode (*mults)(Mat, Vecs, Vecs);
  PetscErrorCode (*solves)(Mat, Vecs, Vecs);
  PetscErrorCode (*getinertia)(Mat, PetscInt *, PetscInt *, PetscInt *);
  PetscErrorCode (*load)(Mat, PetscViewer);
  /*79*/
  PetscErrorCode (*issymmetric)(Mat, PetscReal, PetscBool *);
  PetscErrorCode (*ishermitian)(Mat, PetscReal, PetscBool *);
  PetscErrorCode (*isstructurallysymmetric)(Mat, PetscBool *);
  PetscErrorCode (*setvaluesblockedlocal)(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
  PetscErrorCode (*getvecs)(Mat, Vec *, Vec *);
  /*84*/
  PetscErrorCode (*matmultsymbolic)(Mat, Mat, PetscReal, Mat);
  PetscErrorCode (*matmultnumeric)(Mat, Mat, Mat);
  PetscErrorCode (*ptapnumeric)(Mat, Mat, Mat); /* double dispatch wrapper routine */
  PetscErrorCode (*mattransposemultsymbolic)(Mat, Mat, PetscReal, Mat);
  PetscErrorCode (*mattransposemultnumeric)(Mat, Mat, Mat);
  /*89*/
  PetscErrorCode (*bindtocpu)(Mat, PetscBool);
  PetscErrorCode (*productsetfromoptions)(Mat);
  PetscErrorCode (*productsymbolic)(Mat);
  PetscErrorCode (*productnumeric)(Mat);
  PetscErrorCode (*conjugate)(Mat); /* complex conjugate */
  /*94*/
  PetscErrorCode (*viewnative)(Mat, PetscViewer);
  PetscErrorCode (*setvaluesrow)(Mat, PetscInt, const PetscScalar[]);
  PetscErrorCode (*realpart)(Mat);
  PetscErrorCode (*imaginarypart)(Mat);
  PetscErrorCode (*getrowuppertriangular)(Mat);
  /*99*/
  PetscErrorCode (*restorerowuppertriangular)(Mat);
  PetscErrorCode (*matsolve)(Mat, Mat, Mat);
  PetscErrorCode (*matsolvetranspose)(Mat, Mat, Mat);
  PetscErrorCode (*getrowmin)(Mat, Vec, PetscInt[]);
  PetscErrorCode (*getcolumnvector)(Mat, Vec, PetscInt);
  /*104*/
  PetscErrorCode (*missingdiagonal)(Mat, PetscBool *, PetscInt *);
  PetscErrorCode (*getseqnonzerostructure)(Mat, Mat *);
  PetscErrorCode (*create)(Mat);
  PetscErrorCode (*getghosts)(Mat, PetscInt *, const PetscInt *[]);
  PetscErrorCode (*getlocalsubmatrix)(Mat, IS, IS, Mat *);
  /*109*/
  PetscErrorCode (*restorelocalsubmatrix)(Mat, IS, IS, Mat *);
  PetscErrorCode (*multdiagonalblock)(Mat, Vec, Vec);
  PetscErrorCode (*hermitiantranspose)(Mat, MatReuse, Mat *);
  PetscErrorCode (*multhermitiantranspose)(Mat, Vec, Vec);
  PetscErrorCode (*multhermitiantransposeadd)(Mat, Vec, Vec, Vec);
  /*114*/
  PetscErrorCode (*getmultiprocblock)(Mat, MPI_Comm, MatReuse, Mat *);
  PetscErrorCode (*findnonzerorows)(Mat, IS *);
  PetscErrorCode (*getcolumnreductions)(Mat, PetscInt, PetscReal *);
  PetscErrorCode (*invertblockdiagonal)(Mat, const PetscScalar **);
  PetscErrorCode (*invertvariableblockdiagonal)(Mat, PetscInt, const PetscInt *, PetscScalar *);
  /*119*/
  PetscErrorCode (*createsubmatricesmpi)(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat **);
  PetscErrorCode (*setvaluesbatch)(Mat, PetscInt, PetscInt, PetscInt *, const PetscScalar *);
  PetscErrorCode (*transposematmultsymbolic)(Mat, Mat, PetscReal, Mat);
  PetscErrorCode (*transposematmultnumeric)(Mat, Mat, Mat);
  PetscErrorCode (*transposecoloringcreate)(Mat, ISColoring, MatTransposeColoring);
  /*124*/
  PetscErrorCode (*transcoloringapplysptoden)(MatTransposeColoring, Mat, Mat);
  PetscErrorCode (*transcoloringapplydentosp)(MatTransposeColoring, Mat, Mat);
  PetscErrorCode (*rartnumeric)(Mat, Mat, Mat); /* double dispatch wrapper routine */
  PetscErrorCode (*setblocksizes)(Mat, PetscInt, PetscInt);
  PetscErrorCode (*residual)(Mat, Vec, Vec, Vec);
  /*129*/
  PetscErrorCode (*fdcoloringsetup)(Mat, ISColoring, MatFDColoring);
  PetscErrorCode (*findoffblockdiagonalentries)(Mat, IS *);
  PetscErrorCode (*creatempimatconcatenateseqmat)(MPI_Comm, Mat, PetscInt, MatReuse, Mat *);
  PetscErrorCode (*destroysubmatrices)(PetscInt, Mat *[]);
  PetscErrorCode (*mattransposesolve)(Mat, Mat, Mat);
  /*134*/
  PetscErrorCode (*getvalueslocal)(Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[], PetscScalar[]);
  PetscErrorCode (*creategraph)(Mat, PetscBool, PetscBool, PetscReal, PetscInt, PetscInt[], Mat *);
  PetscErrorCode (*transposesymbolic)(Mat, Mat *);
  PetscErrorCode (*eliminatezeros)(Mat, PetscBool);
  PetscErrorCode (*getrowsumabs)(Mat, Vec);
  /*139*/
  PetscErrorCode (*getfactor)(Mat, MatSolverType, MatFactorType, Mat *);
  PetscErrorCode (*getblockdiagonal)(Mat, Mat *);  // NOTE: the caller of get{block, vblock}diagonal owns the returned matrix;
  PetscErrorCode (*getvblockdiagonal)(Mat, Mat *); // they must destroy it after use
  PetscErrorCode (*copyhashtoxaij)(Mat, Mat);
};
/*
    If you add MatOps entries above also add them to the MATOP enum
    in include/petscmat.h
*/

#include <petscsys.h>

typedef struct _p_MatRootName *MatRootName;
struct _p_MatRootName {
  char       *rname, *sname, *mname;
  MatRootName next;
};

PETSC_EXTERN MatRootName MatRootNameList;

/*
   Utility private matrix routines used outside Mat
*/
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode MatFindNonzeroRowsOrCols_Basic(Mat, PetscBool, PetscReal, IS *);
PETSC_EXTERN PetscErrorCode                MatShellGetScalingShifts(Mat, PetscScalar *, PetscScalar *, Vec *, Vec *, Vec *, Mat *, IS *, IS *);

#define MAT_SHELL_NOT_ALLOWED (void *)-1

/*
   Utility private matrix routines
*/
PETSC_INTERN PetscErrorCode MatConvert_Basic(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatConvert_Shell(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatConvertFrom_Shell(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatShellSetContext_Immutable(Mat, void *);
PETSC_INTERN PetscErrorCode MatShellSetContextDestroy_Immutable(Mat, PetscCtxDestroyFn *);
PETSC_INTERN PetscErrorCode MatShellSetManageScalingShifts_Immutable(Mat);
PETSC_INTERN PetscErrorCode MatCopy_Basic(Mat, Mat, MatStructure);
PETSC_INTERN PetscErrorCode MatDiagonalSet_Default(Mat, Vec, InsertMode);
#if defined(PETSC_HAVE_SCALAPACK)
PETSC_INTERN PetscErrorCode MatConvert_Dense_ScaLAPACK(Mat, MatType, MatReuse, Mat *);
#endif
PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_Basic(Mat, PetscCount, PetscInt[], PetscInt[]);
PETSC_INTERN PetscErrorCode MatSetValuesCOO_Basic(Mat, const PetscScalar[], InsertMode);

/* This can be moved to the public header after implementing some missing MatProducts */
PETSC_INTERN PetscErrorCode MatCreateFromISLocalToGlobalMapping(ISLocalToGlobalMapping, Mat, PetscBool, PetscBool, MatType, Mat *);

/* these callbacks rely on the old matrix function pointers for
   matmat operations. They are unsafe, and should be removed.
   However, the amount of work needed to clean up all the
   implementations is not negligible */
PETSC_INTERN PetscErrorCode MatProductSymbolic_AB(Mat);
PETSC_INTERN PetscErrorCode MatProductNumeric_AB(Mat);
PETSC_INTERN PetscErrorCode MatProductSymbolic_AtB(Mat);
PETSC_INTERN PetscErrorCode MatProductNumeric_AtB(Mat);
PETSC_INTERN PetscErrorCode MatProductSymbolic_ABt(Mat);
PETSC_INTERN PetscErrorCode MatProductNumeric_ABt(Mat);
PETSC_INTERN PetscErrorCode MatProductNumeric_PtAP(Mat);
PETSC_INTERN PetscErrorCode MatProductNumeric_RARt(Mat);
PETSC_INTERN PetscErrorCode MatProductSymbolic_ABC(Mat);
PETSC_INTERN PetscErrorCode MatProductNumeric_ABC(Mat);

PETSC_INTERN PetscErrorCode MatProductCreate_Private(Mat, Mat, Mat, Mat);
/* this callback handles all the different triple products and
   does not rely on the function pointers; used by cuSPARSE/hipSPARSE and KOKKOS-KERNELS */
PETSC_INTERN PetscErrorCode MatProductSymbolic_ABC_Basic(Mat);

/* CreateGraph is common to AIJ seq and mpi */
PETSC_INTERN PetscErrorCode MatCreateGraph_Simple_AIJ(Mat, PetscBool, PetscBool, PetscReal, PetscInt, PetscInt[], Mat *);

#if defined(PETSC_CLANG_STATIC_ANALYZER)
template <typename Tm>
extern void MatCheckPreallocated(Tm, int);
template <typename Tm>
extern void MatCheckProduct(Tm, int);
#else /* PETSC_CLANG_STATIC_ANALYZER */
  #define MatCheckPreallocated(A, arg) \
    do { \
      if (!(A)->preallocated) PetscCall(MatSetUp(A)); \
    } while (0)

  #if defined(PETSC_USE_DEBUG)
    #define MatCheckProduct(A, arg) \
      do { \
        PetscCheck((A)->product, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Argument %d \"%s\" is not a matrix obtained from MatProductCreate()", (arg), #A); \
      } while (0)
  #else
    #define MatCheckProduct(A, arg) \
      do { \
      } while (0)
  #endif
#endif /* PETSC_CLANG_STATIC_ANALYZER */

/*
  The stash is used to temporarily store inserted matrix values that
  belong to another processor. During the assembly phase the stashed
  values are moved to the correct processor and
*/

typedef struct _MatStashSpace *PetscMatStashSpace;

struct _MatStashSpace {
  PetscMatStashSpace next;
  PetscScalar       *space_head, *val;
  PetscInt          *idx, *idy;
  PetscInt           total_space_size;
  PetscInt           local_used;
  PetscInt           local_remaining;
};

PETSC_EXTERN PetscErrorCode PetscMatStashSpaceGet(PetscInt, PetscInt, PetscMatStashSpace *);
PETSC_EXTERN PetscErrorCode PetscMatStashSpaceContiguous(PetscInt, PetscMatStashSpace *, PetscScalar *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscMatStashSpaceDestroy(PetscMatStashSpace *);

typedef struct {
  PetscInt count;
} MatStashHeader;

typedef struct {
  void    *buffer; /* Of type blocktype, dynamically constructed  */
  PetscInt count;
  char     pending;
} MatStashFrame;

typedef struct _MatStash MatStash;
struct _MatStash {
  PetscInt           nmax;              /* maximum stash size */
  PetscInt           umax;              /* user specified max-size */
  PetscInt           oldnmax;           /* the nmax value used previously */
  PetscInt           n;                 /* stash size */
  PetscInt           bs;                /* block size of the stash */
  PetscInt           reallocs;          /* preserve the no of mallocs invoked */
  PetscMatStashSpace space_head, space; /* linked list to hold stashed global row/column numbers and matrix values */

  PetscErrorCode (*ScatterBegin)(Mat, MatStash *, PetscInt *);
  PetscErrorCode (*ScatterGetMesg)(MatStash *, PetscMPIInt *, PetscInt **, PetscInt **, PetscScalar **, PetscInt *);
  PetscErrorCode (*ScatterEnd)(MatStash *);
  PetscErrorCode (*ScatterDestroy)(MatStash *);

  /* The following variables are used for communication */
  MPI_Comm      comm;
  PetscMPIInt   size, rank;
  PetscMPIInt   tag1, tag2;
  MPI_Request  *send_waits;     /* array of send requests */
  MPI_Request  *recv_waits;     /* array of receive requests */
  MPI_Status   *send_status;    /* array of send status */
  PetscMPIInt   nsends, nrecvs; /* numbers of sends and receives */
  PetscScalar  *svalues;        /* sending data */
  PetscInt     *sindices;
  PetscScalar **rvalues;    /* receiving data (values) */
  PetscInt    **rindices;   /* receiving data (indices) */
  PetscMPIInt   nprocessed; /* number of messages already processed */
  PetscMPIInt  *flg_v;      /* indicates what messages have arrived so far and from whom */
  PetscBool     reproduce;
  PetscMPIInt   reproduce_count;

  /* The following variables are used for BTS communication */
  PetscBool       first_assembly_done; /* Is the first time matrix assembly done? */
  PetscBool       use_status;          /* Use MPI_Status to determine number of items in each message */
  PetscMPIInt     nsendranks;
  PetscMPIInt     nrecvranks;
  PetscMPIInt    *sendranks;
  PetscMPIInt    *recvranks;
  MatStashHeader *sendhdr, *recvhdr;
  MatStashFrame  *sendframes; /* pointers to the main messages */
  MatStashFrame  *recvframes;
  MatStashFrame  *recvframe_active;
  PetscInt        recvframe_i;     /* index of block within active frame */
  PetscInt        recvframe_count; /* Count actually sent for current frame */
  PetscMPIInt     recvcount;       /* Number of receives processed so far */
  PetscMPIInt    *some_indices;    /* From last call to MPI_Waitsome */
  MPI_Status     *some_statuses;   /* Statuses from last call to MPI_Waitsome */
  PetscMPIInt     some_count;      /* Number of requests completed in last call to MPI_Waitsome */
  PetscMPIInt     some_i;          /* Index of request currently being processed */
  MPI_Request    *sendreqs;
  MPI_Request    *recvreqs;
  PetscSegBuffer  segsendblocks;
  PetscSegBuffer  segrecvframe;
  PetscSegBuffer  segrecvblocks;
  MPI_Datatype    blocktype;
  size_t          blocktype_size;
  InsertMode     *insertmode; /* Pointer to check mat->insertmode and set upon message arrival in case no local values have been set. */
};

#if !defined(PETSC_HAVE_MPIUNI)
PETSC_INTERN PetscErrorCode MatStashScatterDestroy_BTS(MatStash *);
#endif
PETSC_INTERN PetscErrorCode MatStashCreate_Private(MPI_Comm, PetscInt, MatStash *);
PETSC_INTERN PetscErrorCode MatStashDestroy_Private(MatStash *);
PETSC_INTERN PetscErrorCode MatStashScatterEnd_Private(MatStash *);
PETSC_INTERN PetscErrorCode MatStashSetInitialSize_Private(MatStash *, PetscInt);
PETSC_INTERN PetscErrorCode MatStashGetInfo_Private(MatStash *, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode MatStashValuesRow_Private(MatStash *, PetscInt, PetscInt, const PetscInt[], const PetscScalar[], PetscBool);
PETSC_INTERN PetscErrorCode MatStashValuesCol_Private(MatStash *, PetscInt, PetscInt, const PetscInt[], const PetscScalar[], PetscInt, PetscBool);
PETSC_INTERN PetscErrorCode MatStashValuesRowBlocked_Private(MatStash *, PetscInt, PetscInt, const PetscInt[], const PetscScalar[], PetscInt, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatStashValuesColBlocked_Private(MatStash *, PetscInt, PetscInt, const PetscInt[], const PetscScalar[], PetscInt, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatStashScatterBegin_Private(Mat, MatStash *, PetscInt *);
PETSC_INTERN PetscErrorCode MatStashScatterGetMesg_Private(MatStash *, PetscMPIInt *, PetscInt **, PetscInt **, PetscScalar **, PetscInt *);
PETSC_INTERN PetscErrorCode MatGetInfo_External(Mat, MatInfoType, MatInfo *);

typedef struct {
  PetscInt  dim;
  PetscInt  dims[4];
  PetscInt  starts[4];
  PetscBool noc; /* this is a single component problem, hence user will not set MatStencil.c */
} MatStencilInfo;

/* Info about using compressed row format */
typedef struct {
  PetscBool use;    /* indicates compressed rows have been checked and will be used */
  PetscInt  nrows;  /* number of non-zero rows */
  PetscInt *i;      /* compressed row pointer  */
  PetscInt *rindex; /* compressed row index               */
} Mat_CompressedRow;
PETSC_EXTERN PetscErrorCode MatCheckCompressedRow(Mat, PetscInt, Mat_CompressedRow *, PetscInt *, PetscInt, PetscReal);

typedef struct { /* used by MatCreateRedundantMatrix() for reusing matredundant */
  PetscInt     nzlocal, nsends, nrecvs;
  PetscMPIInt *send_rank, *recv_rank;
  PetscInt    *sbuf_nz, *rbuf_nz, *sbuf_j, **rbuf_j;
  PetscScalar *sbuf_a, **rbuf_a;
  MPI_Comm     subcomm; /* when user does not provide a subcomm */
  IS           isrow, iscol;
  Mat         *matseq;
} Mat_Redundant;

typedef struct { /* used by MatProduct() */
  MatProductType type;
  char          *alg;
  Mat            A, B, C, Dwork;
  PetscBool      symbolic_used_the_fact_A_is_symmetric; /* Symbolic phase took advantage of the fact that A is symmetric, and optimized e.g. AtB as AB. Then, .. */
  PetscBool      symbolic_used_the_fact_B_is_symmetric; /* .. in the numeric phase, if a new A is not symmetric (but has the same sparsity as the old A therefore .. */
  PetscBool      symbolic_used_the_fact_C_is_symmetric; /* MatMatMult(A,B,MAT_REUSE_MATRIX,..&C) is still legitimate), we need to redo symbolic! */
  PetscObjectParameterDeclare(PetscReal, fill);
  PetscBool api_user; /* used to distinguish command line options and to indicate the matrix values are ready to be consumed at symbolic phase if needed */
  PetscBool setfromoptionscalled;

  /* Some products may display the information on the algorithm used */
  PetscErrorCode (*view)(Mat, PetscViewer);

  /* many products have intermediate data structures, each specific to Mat types and product type */
  PetscBool clear;                   /* whether or not to clear the data structures after MatProductNumeric has been called */
  void     *data;                    /* where to stash those structures */
  PetscErrorCode (*destroy)(void *); /* destroy routine */
} Mat_Product;

struct _p_Mat {
  PETSCHEADER(struct _MatOps);
  PetscLayout      rmap, cmap;
  void            *data;                                    /* implementation-specific data */
  MatFactorType    factortype;                              /* MAT_FACTOR_LU, ILU, CHOLESKY or ICC */
  PetscBool        trivialsymbolic;                         /* indicates the symbolic factorization doesn't actually do a symbolic factorization, it is delayed to the numeric factorization */
  PetscBool        canuseordering;                          /* factorization can use ordering provide to routine (most PETSc implementations) */
  MatOrderingType  preferredordering[MAT_FACTOR_NUM_TYPES]; /* what is the preferred (or default) ordering for the matrix solver type */
  PetscBool        assembled;                               /* is the matrix assembled? */
  PetscBool        was_assembled;                           /* new values inserted into assembled mat */
  PetscInt         num_ass;                                 /* number of times matrix has been assembled */
  PetscObjectState nonzerostate;                            /* each time new nonzeros locations are introduced into the matrix this is updated */
  PetscObjectState ass_nonzerostate;                        /* nonzero state at last assembly */
  MatInfo          info;                                    /* matrix information */
  InsertMode       insertmode;                              /* have values been inserted in matrix or added? */
  MatStash         stash, bstash;                           /* used for assembling off-proc mat emements */
  MatNullSpace     nullsp;                                  /* null space (operator is singular) */
  MatNullSpace     transnullsp;                             /* null space of transpose of operator */
  MatNullSpace     nearnullsp;                              /* near null space to be used by multigrid methods */
  PetscInt         congruentlayouts;                        /* are the rows and columns layouts congruent? */
  PetscBool        preallocated;
  MatStencilInfo   stencil; /* information for structured grid */
  PetscBool3       symmetric, hermitian, structurally_symmetric, spd;
  PetscBool        symmetry_eternal, structural_symmetry_eternal, spd_eternal;
  PetscBool        nooffprocentries, nooffproczerorows;
  PetscBool        assembly_subset; /* set by MAT_SUBSET_OFF_PROC_ENTRIES */
  PetscBool        submat_singleis; /* for efficient PCSetUp_ASM() */
  PetscBool        structure_only;
  PetscBool        sortedfull;      /* full, sorted rows are inserted */
  PetscBool        force_diagonals; /* set by MAT_FORCE_DIAGONAL_ENTRIES */
#if defined(PETSC_HAVE_DEVICE)
  PetscOffloadMask offloadmask; /* a mask which indicates where the valid matrix data is (GPU, CPU or both) */
  PetscBool        boundtocpu;
  PetscBool        bindingpropagates;
#endif
  char                *defaultrandtype;
  void                *spptr; /* pointer for special library like SuperLU */
  char                *solvertype;
  PetscBool            checksymmetryonassembly, checknullspaceonassembly;
  PetscReal            checksymmetrytol;
  Mat                  schur;                            /* Schur complement matrix */
  MatFactorSchurStatus schur_status;                     /* status of the Schur complement matrix */
  Mat_Redundant       *redundant;                        /* used by MatCreateRedundantMatrix() */
  PetscBool            erroriffailure;                   /* Generate an error if detected (for example a zero pivot) instead of returning */
  MatFactorError       factorerrortype;                  /* type of error in factorization */
  PetscReal            factorerror_zeropivot_value;      /* If numerical zero pivot was detected this is the computed value */
  PetscInt             factorerror_zeropivot_row;        /* Row where zero pivot was detected */
  PetscInt             nblocks, *bsizes;                 /* support for MatSetVariableBlockSizes() */
  PetscInt             p_cstart, p_rank, p_cend, n_rank; /* Information from parallel MatComputeVariableBlockEnvelope() */
  PetscBool            p_parallel;
  char                *defaultvectype;
  Mat_Product         *product;
  PetscBool            form_explicit_transpose; /* hint to generate an explicit mat tranpsose for operations like MatMultTranspose() */
  PetscBool            transupdated;            /* whether or not the explicitly generated transpose is up-to-date */
  char                *factorprefix;            /* the prefix to use with factored matrix that is created */
  PetscBool            hash_active;             /* indicates MatSetValues() is being handled by hashing */
};

PETSC_INTERN PetscErrorCode MatAXPY_Basic(Mat, PetscScalar, Mat, MatStructure);
PETSC_INTERN PetscErrorCode MatAXPY_BasicWithPreallocation(Mat, Mat, PetscScalar, Mat, MatStructure);
PETSC_INTERN PetscErrorCode MatAXPY_Basic_Preallocate(Mat, Mat, Mat *);
PETSC_INTERN PetscErrorCode MatAXPY_Dense_Nest(Mat, PetscScalar, Mat);

PETSC_INTERN PetscErrorCode MatSetUp_Default(Mat);

/*
    Utility for MatZeroRows
*/
PETSC_INTERN PetscErrorCode MatZeroRowsMapLocal_Private(Mat, PetscInt, const PetscInt *, PetscInt *, PetscInt **);

/*
    Utility for MatView/MatLoad
*/
PETSC_INTERN PetscErrorCode MatView_Binary_BlockSizes(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatLoad_Binary_BlockSizes(Mat, PetscViewer);

/*
    Object for partitioning graphs
*/

typedef struct _MatPartitioningOps *MatPartitioningOps;
struct _MatPartitioningOps {
  PetscErrorCode (*apply)(MatPartitioning, IS *);
  PetscErrorCode (*applynd)(MatPartitioning, IS *);
  PetscErrorCode (*setfromoptions)(MatPartitioning, PetscOptionItems);
  PetscErrorCode (*destroy)(MatPartitioning);
  PetscErrorCode (*view)(MatPartitioning, PetscViewer);
  PetscErrorCode (*improve)(MatPartitioning, IS *);
};

struct _p_MatPartitioning {
  PETSCHEADER(struct _MatPartitioningOps);
  Mat        adj;
  PetscInt  *vertex_weights;
  PetscReal *part_weights;
  PetscInt   n;    /* number of partitions */
  PetscInt   ncon; /* number of vertex weights per vertex */
  void      *data;
  PetscBool  use_edge_weights; /* A flag indicates whether or not to use edge weights */
};

/* needed for parallel nested dissection by ParMetis and PTSCOTCH */
PETSC_INTERN PetscErrorCode MatPartitioningSizesToSep_Private(PetscInt, PetscInt[], PetscInt[], PetscInt[]);

/*
    Object for coarsen graphs
*/
typedef struct _MatCoarsenOps *MatCoarsenOps;
struct _MatCoarsenOps {
  PetscErrorCode (*apply)(MatCoarsen);
  PetscErrorCode (*setfromoptions)(MatCoarsen, PetscOptionItems);
  PetscErrorCode (*destroy)(MatCoarsen);
  PetscErrorCode (*view)(MatCoarsen, PetscViewer);
};

#define MAT_COARSEN_STRENGTH_INDEX_SIZE 3
struct _p_MatCoarsen {
  PETSCHEADER(struct _MatCoarsenOps);
  Mat   graph;
  void *subctx;
  /* */
  PetscBool         strict_aggs;
  IS                perm;
  PetscCoarsenData *agg_lists;
  PetscInt          max_it;    /* number of iterations in HEM */
  PetscReal         threshold; /* HEM can filter interim graphs */
  PetscInt          strength_index_size;
  PetscInt          strength_index[MAT_COARSEN_STRENGTH_INDEX_SIZE];
};

PETSC_EXTERN PetscErrorCode MatCoarsenMISKSetDistance(MatCoarsen, PetscInt);
PETSC_EXTERN PetscErrorCode MatCoarsenMISKGetDistance(MatCoarsen, PetscInt *);

/*
    Used in aijdevice.h
*/
typedef struct {
  PetscInt    *i;
  PetscInt    *j;
  PetscScalar *a;
  PetscInt     n;
  PetscInt     ignorezeroentries;
} PetscCSRDataStructure;

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
    vwscale       = {dx(0),dx(1),dx(2),dx(3)}               MPI Vec
    vscale        = {dx(0),dx(1),dx(2),dx(3),dx(4),dx(5)}   Seq Vec

    ncolumns      = {1,0,1,1}
    columns       = {{6},{},{4},{5}}
    nrows         = {3,0,2,2}
    rows          = {{0,1,2},{},{1,2},{1,2}}
    vwscale       = {dx(4),dx(5),dx(6)}              MPI Vec
    vscale        = {dx(0),dx(4),dx(5),dx(6)}        Seq Vec

    See the routine MatFDColoringApply() for how this data is used
    to compute the Jacobian.

*/
typedef struct {
  PetscInt     row;
  PetscInt     col;
  PetscScalar *valaddr; /* address of value */
} MatEntry;

typedef struct {
  PetscInt     row;
  PetscScalar *valaddr; /* address of value */
} MatEntry2;

struct _p_MatFDColoring {
  PETSCHEADER(int);
  PetscInt         M, N, m;                       /* total rows, columns; local rows */
  PetscInt         rstart;                        /* first row owned by local processor */
  PetscInt         ncolors;                       /* number of colors */
  PetscInt        *ncolumns;                      /* number of local columns for a color */
  PetscInt       **columns;                       /* lists the local columns of each color (using global column numbering) */
  IS              *isa;                           /* these are the IS that contain the column values given in columns */
  PetscInt        *nrows;                         /* number of local rows for each color */
  MatEntry        *matentry;                      /* holds (row, column, address of value) for Jacobian matrix entry */
  MatEntry2       *matentry2;                     /* holds (row, address of value) for Jacobian matrix entry */
  PetscScalar     *dy;                            /* store a block of F(x+dx)-F(x) when J is in BAIJ format */
  PetscReal        error_rel;                     /* square root of relative error in computing function */
  PetscReal        umin;                          /* minimum allowable u'dx value */
  Vec              w1, w2, w3;                    /* work vectors used in computing Jacobian */
  PetscBool        fset;                          /* indicates that the initial function value F(X) is set */
  MatFDColoringFn *f;                             /* function that defines Jacobian */
  void            *fctx;                          /* optional user-defined context for use by the function f */
  Vec              vscale;                        /* holds FD scaling, i.e. 1/dx for each perturbed column */
  PetscInt         currentcolor;                  /* color for which function evaluation is being done now */
  const char      *htype;                         /* "wp" or "ds" */
  ISColoringType   ctype;                         /* IS_COLORING_GLOBAL or IS_COLORING_LOCAL */
  PetscInt         brows, bcols;                  /* number of block rows or columns for speedup inserting the dense matrix into sparse Jacobian */
  PetscBool        setupcalled;                   /* true if setup has been called */
  PetscBool        viewed;                        /* true if the -mat_fd_coloring_view has been triggered already */
  void (*ftn_func_pointer)(void), *ftn_func_cntx; /* serve the same purpose as *fortran_func_pointers in PETSc objects */
  PetscObjectId matid;                            /* matrix this object was created with, must always be the same */
};

typedef struct _MatColoringOps *MatColoringOps;
struct _MatColoringOps {
  PetscErrorCode (*destroy)(MatColoring);
  PetscErrorCode (*setfromoptions)(MatColoring, PetscOptionItems);
  PetscErrorCode (*view)(MatColoring, PetscViewer);
  PetscErrorCode (*apply)(MatColoring, ISColoring *);
  PetscErrorCode (*weights)(MatColoring, PetscReal **, PetscInt **);
};

struct _p_MatColoring {
  PETSCHEADER(struct _MatColoringOps);
  Mat                   mat;
  PetscInt              dist;         /* distance of the coloring */
  PetscInt              maxcolors;    /* the maximum number of colors returned, maxcolors=1 for MIS */
  void                 *data;         /* inner context */
  PetscBool             valid;        /* check to see if what is produced is a valid coloring */
  MatColoringWeightType weight_type;  /* type of weight computation to be performed */
  PetscReal            *user_weights; /* custom weights and permutation */
  PetscInt             *user_lperm;
  PetscBool             valid_iscoloring; /* check to see if matcoloring is produced a valid iscoloring */
};

struct _p_MatTransposeColoring {
  PETSCHEADER(int);
  PetscInt       M, N, m;      /* total rows, columns; local rows */
  PetscInt       rstart;       /* first row owned by local processor */
  PetscInt       ncolors;      /* number of colors */
  PetscInt      *ncolumns;     /* number of local columns for a color */
  PetscInt      *nrows;        /* number of local rows for each color */
  PetscInt       currentcolor; /* color for which function evaluation is being done now */
  ISColoringType ctype;        /* IS_COLORING_GLOBAL or IS_COLORING_LOCAL */

  PetscInt *colorforrow, *colorforcol; /* pointer to rows and columns */
  PetscInt *rows;                      /* lists the local rows for each color (using the local row numbering) */
  PetscInt *den2sp;                    /* maps (row,color) in the dense matrix to index of sparse matrix array a->a */
  PetscInt *columns;                   /* lists the local columns of each color (using global column numbering) */
  PetscInt  brows;                     /* number of rows for efficient implementation of MatTransColoringApplyDenToSp() */
  PetscInt *lstart;                    /* array used for loop over row blocks of Csparse */
};

/*
   Null space context for preconditioner/operators
*/
struct _p_MatNullSpace {
  PETSCHEADER(int);
  PetscBool             has_cnst;
  PetscInt              n;
  Vec                  *vecs;
  PetscScalar          *alpha;  /* for projections */
  MatNullSpaceRemoveFn *remove; /* for user provided removal function */
  void                 *rmctx;  /* context for remove() function */
};

/*
   Internal data structure for MATMPIDENSE
*/
typedef struct {
  Mat A; /* local submatrix */

  /* The following variables are used for matrix assembly */
  PetscBool    donotstash;        /* Flag indicating if values should be stashed */
  MPI_Request *send_waits;        /* array of send requests */
  MPI_Request *recv_waits;        /* array of receive requests */
  PetscInt     nsends, nrecvs;    /* numbers of sends and receives */
  PetscScalar *svalues, *rvalues; /* sending and receiving data */
  PetscInt     rmax;              /* maximum message length */

  /* The following variables are used for matrix-vector products */
  Vec       lvec;        /* local vector */
  PetscSF   Mvctx;       /* for mat-mult communications */
  PetscBool roworiented; /* if true, row-oriented input (default) */

  /* Support for MatDenseGetColumnVec and MatDenseGetSubMatrix */
  Mat                cmat;     /* matrix representation of a given subset of columns */
  Vec                cvec;     /* vector representation of a given column */
  const PetscScalar *ptrinuse; /* holds array to be restored (just a placeholder) */
  PetscInt           vecinuse; /* if cvec is in use (col = vecinuse-1) */
  PetscInt           matinuse; /* if cmat is in use (cbegin = matinuse-1) */
  /* if this is from MatDenseGetSubMatrix, which columns and rows does it correspond to? */
  PetscInt sub_rbegin;
  PetscInt sub_rend;
  PetscInt sub_cbegin;
  PetscInt sub_cend;
} Mat_MPIDense;

/*
   Checking zero pivot for LU, ILU preconditioners.
*/
typedef struct {
  PetscInt    nshift, nshift_max;
  PetscReal   shift_amount, shift_lo, shift_hi, shift_top, shift_fraction;
  PetscBool   newshift;
  PetscReal   rs; /* active row sum of abs(off-diagonals) */
  PetscScalar pv; /* pivot of the active row */
} FactorShiftCtx;

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode MatTransposeCheckNonzeroState_Private(Mat, Mat);

/*
 Used by MatTranspose() and potentially other functions to track the matrix used in the generation of another matrix
*/
typedef struct {
  PetscObjectId    id;
  PetscObjectState state;
  PetscObjectState nonzerostate;
} MatParentState;

PETSC_EXTERN PetscErrorCode MatFactorDumpMatrix(Mat);
PETSC_INTERN PetscErrorCode MatSetBlockSizes_Default(Mat, PetscInt, PetscInt);

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode MatShift_Basic(Mat, PetscScalar);

static inline PetscErrorCode MatPivotCheck_nz(PETSC_UNUSED Mat mat, const MatFactorInfo *info, FactorShiftCtx *sctx, PETSC_UNUSED PetscInt row)
{
  PetscReal _rs   = sctx->rs;
  PetscReal _zero = info->zeropivot * _rs;

  PetscFunctionBegin;
  if (PetscAbsScalar(sctx->pv) <= _zero && !PetscIsNanScalar(sctx->pv)) {
    /* force |diag| > zeropivot*rs */
    if (!sctx->nshift) sctx->shift_amount = info->shiftamount;
    else sctx->shift_amount *= 2.0;
    sctx->newshift = PETSC_TRUE;
    (sctx->nshift)++;
  } else {
    sctx->newshift = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode MatPivotCheck_pd(PETSC_UNUSED Mat mat, const MatFactorInfo *info, FactorShiftCtx *sctx, PETSC_UNUSED PetscInt row)
{
  PetscReal _rs   = sctx->rs;
  PetscReal _zero = info->zeropivot * _rs;

  PetscFunctionBegin;
  if (PetscRealPart(sctx->pv) <= _zero && !PetscIsNanScalar(sctx->pv)) {
    /* force matfactor to be diagonally dominant */
    if (sctx->nshift == sctx->nshift_max) {
      sctx->shift_fraction = sctx->shift_hi;
    } else {
      sctx->shift_lo       = sctx->shift_fraction;
      sctx->shift_fraction = (sctx->shift_hi + sctx->shift_lo) / (PetscReal)2.;
    }
    sctx->shift_amount = sctx->shift_fraction * sctx->shift_top;
    sctx->nshift++;
    sctx->newshift = PETSC_TRUE;
  } else {
    sctx->newshift = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode MatPivotCheck_inblocks(PETSC_UNUSED Mat mat, const MatFactorInfo *info, FactorShiftCtx *sctx, PETSC_UNUSED PetscInt row)
{
  PetscReal _zero = info->zeropivot;

  PetscFunctionBegin;
  if (PetscAbsScalar(sctx->pv) <= _zero && !PetscIsNanScalar(sctx->pv)) {
    sctx->pv += info->shiftamount;
    sctx->shift_amount = 0.0;
    sctx->nshift++;
  }
  sctx->newshift = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode MatPivotCheck_none(Mat fact, Mat mat, const MatFactorInfo *info, FactorShiftCtx *sctx, PetscInt row)
{
  PetscReal _zero = info->zeropivot;

  PetscFunctionBegin;
  sctx->newshift = PETSC_FALSE;
  if (PetscAbsScalar(sctx->pv) <= _zero && !PetscIsNanScalar(sctx->pv)) {
    PetscCheck(!mat->erroriffailure, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Zero pivot row %" PetscInt_FMT " value %g tolerance %g", row, (double)PetscAbsScalar(sctx->pv), (double)_zero);
    PetscCall(PetscInfo(mat, "Detected zero pivot in factorization in row %" PetscInt_FMT " value %g tolerance %g\n", row, (double)PetscAbsScalar(sctx->pv), (double)_zero));
    fact->factorerrortype             = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    fact->factorerror_zeropivot_value = PetscAbsScalar(sctx->pv);
    fact->factorerror_zeropivot_row   = row;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode MatPivotCheck(Mat fact, Mat mat, const MatFactorInfo *info, FactorShiftCtx *sctx, PetscInt row)
{
  PetscFunctionBegin;
  if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) PetscCall(MatPivotCheck_nz(mat, info, sctx, row));
  else if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) PetscCall(MatPivotCheck_pd(mat, info, sctx, row));
  else if (info->shifttype == (PetscReal)MAT_SHIFT_INBLOCKS) PetscCall(MatPivotCheck_inblocks(mat, info, sctx, row));
  else PetscCall(MatPivotCheck_none(fact, mat, info, sctx, row));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscbt.h>
/*
  Create and initialize a linked list
  Input Parameters:
    idx_start - starting index of the list
    lnk_max   - max value of lnk indicating the end of the list
    nlnk      - max length of the list
  Output Parameters:
    lnk       - list initialized
    bt        - PetscBT (bitarray) with all bits set to false
    lnk_empty - flg indicating the list is empty
*/
#define PetscLLCreate(idx_start, lnk_max, nlnk, lnk, bt) ((PetscErrorCode)(PetscMalloc1(nlnk, &lnk) || PetscBTCreate(nlnk, &(bt)) || (lnk[idx_start] = lnk_max, PETSC_SUCCESS)))

#define PetscLLCreate_new(idx_start, lnk_max, nlnk, lnk, bt, lnk_empty) ((PetscErrorCode)(PetscMalloc1(nlnk, &lnk) || PetscBTCreate(nlnk, &(bt)) || (lnk_empty = PETSC_TRUE, 0) || (lnk[idx_start] = lnk_max, PETSC_SUCCESS)))

static inline PetscErrorCode PetscLLInsertLocation_Private(PetscBool assume_sorted, PetscInt k, PetscInt idx_start, PetscInt entry, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnkdata, PetscInt *PETSC_RESTRICT lnk)
{
  PetscInt location;

  PetscFunctionBegin;
  /* start from the beginning if entry < previous entry */
  if (!assume_sorted && k && entry < *lnkdata) *lnkdata = idx_start;
  /* search for insertion location */
  do {
    location = *lnkdata;
    *lnkdata = lnk[location];
  } while (entry > *lnkdata);
  /* insertion location is found, add entry into lnk */
  lnk[location] = entry;
  lnk[entry]    = *lnkdata;
  ++(*nlnk);
  *lnkdata = entry; /* next search starts from here if next_entry > entry */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscLLAdd_Private(PetscInt nidx, const PetscInt *PETSC_RESTRICT indices, PetscInt idx_start, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscBT bt, PetscBool assume_sorted)
{
  PetscFunctionBegin;
  *nlnk = 0;
  for (PetscInt k = 0, lnkdata = idx_start; k < nidx; ++k) {
    const PetscInt entry = indices[k];

    if (!PetscBTLookupSet(bt, entry)) PetscCall(PetscLLInsertLocation_Private(assume_sorted, k, idx_start, entry, nlnk, &lnkdata, lnk));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Add an index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - integer array
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray)
*/
static inline PetscErrorCode PetscLLAdd(PetscInt nidx, const PetscInt *PETSC_RESTRICT indices, PetscInt idx_start, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscBT bt)
{
  PetscFunctionBegin;
  PetscCall(PetscLLAdd_Private(nidx, indices, idx_start, nlnk, lnk, bt, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Add a SORTED ascending index set into a sorted linked list - same as PetscLLAdd() bus skip 'if (_k && _entry < _lnkdata) _lnkdata  = idx_start;'
  Input Parameters:
    nidx      - number of input indices
    indices   - sorted integer array
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray)
*/
static inline PetscErrorCode PetscLLAddSorted(PetscInt nidx, const PetscInt *PETSC_RESTRICT indices, PetscInt idx_start, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscBT bt)
{
  PetscFunctionBegin;
  PetscCall(PetscLLAdd_Private(nidx, indices, idx_start, nlnk, lnk, bt, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Add a permuted index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - integer array
    perm      - permutation of indices
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray)
*/
static inline PetscErrorCode PetscLLAddPerm(PetscInt nidx, const PetscInt *PETSC_RESTRICT indices, const PetscInt *PETSC_RESTRICT perm, PetscInt idx_start, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscBT bt)
{
  PetscFunctionBegin;
  *nlnk = 0;
  for (PetscInt k = 0, lnkdata = idx_start; k < nidx; ++k) {
    const PetscInt entry = perm[indices[k]];

    if (!PetscBTLookupSet(bt, entry)) PetscCall(PetscLLInsertLocation_Private(PETSC_FALSE, k, idx_start, entry, nlnk, &lnkdata, lnk));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
/* this appears to be unused? */
static inline PetscErrorCode PetscLLAddSorted_new(PetscInt nidx, PetscInt *indices, PetscInt idx_start, PetscBool *lnk_empty, PetscInt *nlnk, PetscInt *lnk, PetscBT bt)
{
  PetscInt lnkdata = idx_start;

  PetscFunctionBegin;
  if (*lnk_empty) {
    for (PetscInt k = 0; k < nidx; ++k) {
      const PetscInt entry = indices[k], location = lnkdata;

      PetscCall(PetscBTSet(bt,entry)); /* mark the new entry */
      lnkdata       = lnk[location];
      /* insertion location is found, add entry into lnk */
      lnk[location] = entry;
      lnk[entry]    = lnkdata;
      lnkdata       = entry; /* next search starts from here */
    }
    /* lnk[indices[nidx-1]] = lnk[idx_start];
       lnk[idx_start]       = indices[0];
       PetscCall(PetscBTSet(bt,indices[0]));
       for (_k=1; _k<nidx; _k++) {
       PetscCall(PetscBTSet(bt,indices[_k]));
       lnk[indices[_k-1]] = indices[_k];
       }
    */
    *nlnk      = nidx;
    *lnk_empty = PETSC_FALSE;
  } else {
    *nlnk = 0;
    for (PetscInt k = 0; k < nidx; ++k) {
      const PetscInt entry = indices[k];

      if (!PetscBTLookupSet(bt,entry)) PetscCall(PetscLLInsertLocation_Private(PETSC_TRUE,k,idx_start,entry,nlnk,&lnkdata,lnk));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*
  Add a SORTED index set into a sorted linked list used for LUFactorSymbolic()
  Same as PetscLLAddSorted() with an additional operation:
       count the number of input indices that are no larger than 'diag'
  Input Parameters:
    indices   - sorted integer array
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
static inline PetscErrorCode PetscLLAddSortedLU(const PetscInt *PETSC_RESTRICT indices, PetscInt idx_start, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscBT bt, PetscInt diag, PetscInt nzbd, PetscInt *PETSC_RESTRICT im)
{
  const PetscInt nidx = im[idx_start] - nzbd; /* num of entries with idx_start < index <= diag */

  PetscFunctionBegin;
  *nlnk = 0;
  for (PetscInt k = 0, lnkdata = idx_start; k < nidx; ++k) {
    const PetscInt entry = indices[k];

    ++nzbd;
    if (entry == diag) im[idx_start] = nzbd;
    if (!PetscBTLookupSet(bt, entry)) PetscCall(PetscLLInsertLocation_Private(PETSC_TRUE, k, idx_start, entry, nlnk, &lnkdata, lnk));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
static inline PetscErrorCode PetscLLClean(PetscInt idx_start, PetscInt lnk_max, PetscInt nlnk, PetscInt *PETSC_RESTRICT lnk, PetscInt *PETSC_RESTRICT indices, PetscBT bt)
{
  PetscFunctionBegin;
  for (PetscInt j = 0, idx = idx_start; j < nlnk; ++j) {
    idx        = lnk[idx];
    indices[j] = idx;
    PetscCall(PetscBTClear(bt, idx));
  }
  lnk[idx_start] = lnk_max;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Free memories used by the list
*/
#define PetscLLDestroy(lnk, bt) ((PetscErrorCode)(PetscFree(lnk) || PetscBTDestroy(&(bt))))

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
#define PetscIncompleteLLCreate(idx_start, lnk_max, nlnk, lnk, lnk_lvl, bt) \
  ((PetscErrorCode)(PetscIntMultError(2, nlnk, NULL) || PetscMalloc1(2 * nlnk, &lnk) || PetscBTCreate(nlnk, &(bt)) || (lnk[idx_start] = lnk_max, lnk_lvl = lnk + nlnk, PETSC_SUCCESS)))

static inline PetscErrorCode PetscIncompleteLLInsertLocation_Private(PetscBool assume_sorted, PetscInt k, PetscInt idx_start, PetscInt entry, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnkdata, PetscInt *PETSC_RESTRICT lnk, PetscInt *PETSC_RESTRICT lnklvl, PetscInt newval)
{
  PetscFunctionBegin;
  PetscCall(PetscLLInsertLocation_Private(assume_sorted, k, idx_start, entry, nlnk, lnkdata, lnk));
  lnklvl[entry] = newval;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Initialize a sorted linked list used for ILU and ICC
  Input Parameters:
    nidx      - number of input idx
    idx       - integer array used for storing column indices
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
static inline PetscErrorCode PetscIncompleteLLInit(PetscInt nidx, const PetscInt *PETSC_RESTRICT idx, PetscInt idx_start, const PetscInt *PETSC_RESTRICT perm, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscInt *PETSC_RESTRICT lnklvl, PetscBT bt)
{
  PetscFunctionBegin;
  *nlnk = 0;
  for (PetscInt k = 0, lnkdata = idx_start; k < nidx; ++k) {
    const PetscInt entry = perm[idx[k]];

    if (!PetscBTLookupSet(bt, entry)) PetscCall(PetscIncompleteLLInsertLocation_Private(PETSC_FALSE, k, idx_start, entry, nlnk, &lnkdata, lnk, lnklvl, 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscIncompleteLLAdd_Private(PetscInt nidx, const PetscInt *PETSC_RESTRICT idx, PetscReal level, const PetscInt *PETSC_RESTRICT idxlvl, PetscInt idx_start, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscInt *PETSC_RESTRICT lnklvl, PetscBT bt, PetscInt prow_offset, PetscBool assume_sorted)
{
  PetscFunctionBegin;
  *nlnk = 0;
  for (PetscInt k = 0, lnkdata = idx_start; k < nidx; ++k) {
    const PetscInt incrlev = idxlvl[k] + prow_offset + 1;

    if (incrlev <= level) {
      const PetscInt entry = idx[k];

      if (!PetscBTLookupSet(bt, entry)) PetscCall(PetscIncompleteLLInsertLocation_Private(assume_sorted, k, idx_start, entry, nlnk, &lnkdata, lnk, lnklvl, incrlev));
      else if (lnklvl[entry] > incrlev) lnklvl[entry] = incrlev; /* existing entry */
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Add a SORTED index set into a sorted linked list for ICC
  Input Parameters:
    nidx      - number of input indices
    idx       - sorted integer array used for storing column indices
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
static inline PetscErrorCode PetscICCLLAddSorted(PetscInt nidx, const PetscInt *PETSC_RESTRICT idx, PetscReal level, const PetscInt *PETSC_RESTRICT idxlvl, PetscInt idx_start, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscInt *PETSC_RESTRICT lnklvl, PetscBT bt, PetscInt idxlvl_prow)
{
  PetscFunctionBegin;
  PetscCall(PetscIncompleteLLAdd_Private(nidx, idx, level, idxlvl, idx_start, nlnk, lnk, lnklvl, bt, idxlvl_prow, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Add a SORTED index set into a sorted linked list for ILU
  Input Parameters:
    nidx      - number of input indices
    idx       - sorted integer array used for storing column indices
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
static inline PetscErrorCode PetscILULLAddSorted(PetscInt nidx, const PetscInt *PETSC_RESTRICT idx, PetscInt level, const PetscInt *PETSC_RESTRICT idxlvl, PetscInt idx_start, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscInt *PETSC_RESTRICT lnklvl, PetscBT bt, PetscInt prow)
{
  PetscFunctionBegin;
  PetscCall(PetscIncompleteLLAdd_Private(nidx, idx, level, idxlvl, idx_start, nlnk, lnk, lnklvl, bt, lnklvl[prow], PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Add a index set into a sorted linked list
  Input Parameters:
    nidx      - number of input idx
    idx   - integer array used for storing column indices
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
static inline PetscErrorCode PetscIncompleteLLAdd(PetscInt nidx, const PetscInt *PETSC_RESTRICT idx, PetscReal level, const PetscInt *PETSC_RESTRICT idxlvl, PetscInt idx_start, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscInt *PETSC_RESTRICT lnklvl, PetscBT bt)
{
  PetscFunctionBegin;
  PetscCall(PetscIncompleteLLAdd_Private(nidx, idx, level, idxlvl, idx_start, nlnk, lnk, lnklvl, bt, 0, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Add a SORTED index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    idx   - sorted integer array used for storing column indices
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
static inline PetscErrorCode PetscIncompleteLLAddSorted(PetscInt nidx, const PetscInt *PETSC_RESTRICT idx, PetscReal level, const PetscInt *PETSC_RESTRICT idxlvl, PetscInt idx_start, PetscInt *PETSC_RESTRICT nlnk, PetscInt *PETSC_RESTRICT lnk, PetscInt *PETSC_RESTRICT lnklvl, PetscBT bt)
{
  PetscFunctionBegin;
  PetscCall(PetscIncompleteLLAdd_Private(nidx, idx, level, idxlvl, idx_start, nlnk, lnk, lnklvl, bt, 0, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
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
static inline PetscErrorCode PetscIncompleteLLClean(PetscInt idx_start, PetscInt lnk_max, PetscInt nlnk, PetscInt *PETSC_RESTRICT lnk, PetscInt *PETSC_RESTRICT lnklvl, PetscInt *PETSC_RESTRICT indices, PetscInt *PETSC_RESTRICT indiceslvl, PetscBT bt)
{
  PetscFunctionBegin;
  for (PetscInt j = 0, idx = idx_start; j < nlnk; ++j) {
    idx           = lnk[idx];
    indices[j]    = idx;
    indiceslvl[j] = lnklvl[idx];
    lnklvl[idx]   = -1;
    PetscCall(PetscBTClear(bt, idx));
  }
  lnk[idx_start] = lnk_max;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Free memories used by the list
*/
#define PetscIncompleteLLDestroy(lnk, bt) ((PetscErrorCode)(PetscFree(lnk) || PetscBTDestroy(&(bt))))

#if !defined(PETSC_CLANG_STATIC_ANALYZER)
  #define MatCheckSameLocalSize(A, ar1, B, ar2) \
    do { \
      PetscCheckSameComm(A, ar1, B, ar2); \
      PetscCheck(((A)->rmap->n == (B)->rmap->n) && ((A)->cmap->n == (B)->cmap->n), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Incompatible matrix local sizes: parameter # %d (%" PetscInt_FMT " x %" PetscInt_FMT ") != parameter # %d (%" PetscInt_FMT " x %" PetscInt_FMT ")", ar1, \
                 (A)->rmap->n, (A)->cmap->n, ar2, (B)->rmap->n, (B)->cmap->n); \
    } while (0)
  #define MatCheckSameSize(A, ar1, B, ar2) \
    do { \
      PetscCheck(((A)->rmap->N == (B)->rmap->N) && ((A)->cmap->N == (B)->cmap->N), PetscObjectComm((PetscObject)(A)), PETSC_ERR_ARG_INCOMP, "Incompatible matrix global sizes: parameter # %d (%" PetscInt_FMT " x %" PetscInt_FMT ") != parameter # %d (%" PetscInt_FMT " x %" PetscInt_FMT ")", ar1, \
                 (A)->rmap->N, (A)->cmap->N, ar2, (B)->rmap->N, (B)->cmap->N); \
      MatCheckSameLocalSize(A, ar1, B, ar2); \
    } while (0)
#else
template <typename Tm>
extern void MatCheckSameLocalSize(Tm, int, Tm, int);
template <typename Tm>
extern void MatCheckSameSize(Tm, int, Tm, int);
#endif

#define VecCheckMatCompatible(M, x, ar1, b, ar2) \
  do { \
    PetscCheck((M)->cmap->N == (x)->map->N, PetscObjectComm((PetscObject)(M)), PETSC_ERR_ARG_SIZ, "Vector global length incompatible with matrix: parameter # %d global size %" PetscInt_FMT " != matrix column global size %" PetscInt_FMT, ar1, (x)->map->N, \
               (M)->cmap->N); \
    PetscCheck((M)->rmap->N == (b)->map->N, PetscObjectComm((PetscObject)(M)), PETSC_ERR_ARG_SIZ, "Vector global length incompatible with matrix: parameter # %d global size %" PetscInt_FMT " != matrix row global size %" PetscInt_FMT, ar2, (b)->map->N, \
               (M)->rmap->N); \
  } while (0)

/*
  Create and initialize a condensed linked list -
    same as PetscLLCreate(), but uses a scalable array 'lnk' with size of max number of entries, not O(N).
    Barry suggested this approach (Dec. 6, 2011):
      I've thought of an alternative way of representing a linked list that is efficient but doesn't have the O(N) scaling issue
      (it may be faster than the O(N) even sequentially due to less crazy memory access).

      Instead of having some like  a  2  -> 4 -> 11 ->  22  list that uses slot 2  4 11 and 22 in a big array use a small array with two slots
      for each entry for example  [ 2 1 | 4 3 | 22 -1 | 11 2]   so the first number (of the pair) is the value while the second tells you where
      in the list the next entry is. Inserting a new link means just append another pair at the end. For example say we want to insert 13 into the
      list it would then become [2 1 | 4 3 | 22 -1 | 11 4 | 13 2 ] you just add a pair at the end and fix the point for the one that points to it.
      That is 11 use to point to the 2 slot, after the change 11 points to the 4th slot which has the value 13. Note that values are always next
      to each other so memory access is much better than using the big array.

  Example:
     nlnk_max=5, lnk_max=36:
     Initial list: [0, 0 | 36, 2 | 0, 0 | 0, 0 | 0, 0 | 0, 0 | 0, 0]
     here, head_node has index 2 with value lnk[2]=lnk_max=36,
           0-th entry is used to store the number of entries in the list,
     The initial lnk represents head -> tail(marked by 36) with number of entries = lnk[0]=0.

     Now adding a sorted set {2,4}, the list becomes
     [2, 0 | 36, 4 |2, 6 | 4, 2 | 0, 0 | 0, 0 | 0, 0 ]
     represents head -> 2 -> 4 -> tail with number of entries = lnk[0]=2.

     Then adding a sorted set {0,3,35}, the list
     [5, 0 | 36, 8 | 2, 10 | 4, 12 | 0, 4 | 3, 6 | 35, 2 ]
     represents head -> 0 -> 2 -> 3 -> 4 -> 35 -> tail with number of entries = lnk[0]=5.

  Input Parameters:
    nlnk_max  - max length of the list
    lnk_max   - max value of the entries
  Output Parameters:
    lnk       - list created and initialized
    bt        - PetscBT (bitarray) with all bits set to false. Note: bt has size lnk_max, not nln_max!
*/
static inline PetscErrorCode PetscLLCondensedCreate(PetscInt nlnk_max, PetscInt lnk_max, PetscInt **lnk, PetscBT *bt)
{
  PetscInt *llnk, lsize = 0;

  PetscFunctionBegin;
  PetscCall(PetscIntMultError(2, nlnk_max + 2, &lsize));
  PetscCall(PetscMalloc1(lsize, lnk));
  PetscCall(PetscBTCreate(lnk_max, bt));
  llnk    = *lnk;
  llnk[0] = 0;       /* number of entries on the list */
  llnk[2] = lnk_max; /* value in the head node */
  llnk[3] = 2;       /* next for the head node */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Add a SORTED ascending index set into a sorted linked list. See PetscLLCondensedCreate() for detailed description.
  Input Parameters:
    nidx      - number of input indices
    indices   - sorted integer array
    lnk       - condensed linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    lnk       - the sorted(increasing order) linked list containing previous and newly added non-redundate indices
    bt        - updated PetscBT (bitarray)
*/
static inline PetscErrorCode PetscLLCondensedAddSorted(PetscInt nidx, const PetscInt indices[], PetscInt lnk[], PetscBT bt)
{
  PetscInt location = 2;      /* head */
  PetscInt nlnk     = lnk[0]; /* num of entries on the input lnk */

  PetscFunctionBegin;
  for (PetscInt k = 0; k < nidx; k++) {
    const PetscInt entry = indices[k];
    if (!PetscBTLookupSet(bt, entry)) { /* new entry */
      PetscInt next, lnkdata;

      /* search for insertion location */
      do {
        next     = location + 1;  /* link from previous node to next node */
        location = lnk[next];     /* idx of next node */
        lnkdata  = lnk[location]; /* value of next node */
      } while (entry > lnkdata);
      /* insertion location is found, add entry into lnk */
      const PetscInt newnode = 2 * (nlnk + 2); /* index for this new node */
      lnk[next]              = newnode;        /* connect previous node to the new node */
      lnk[newnode]           = entry;          /* set value of the new node */
      lnk[newnode + 1]       = location;       /* connect new node to next node */
      location               = newnode;        /* next search starts from the new node */
      nlnk++;
    }
  }
  lnk[0] = nlnk; /* number of entries in the list */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscLLCondensedClean(PetscInt lnk_max, PETSC_UNUSED PetscInt nidx, PetscInt *indices, PetscInt lnk[], PetscBT bt)
{
  const PetscInt nlnk = lnk[0]; /* num of entries on the list */
  PetscInt       next = lnk[3]; /* head node */

  PetscFunctionBegin;
  for (PetscInt k = 0; k < nlnk; k++) {
    indices[k] = lnk[next];
    next       = lnk[next + 1];
    PetscCall(PetscBTClear(bt, indices[k]));
  }
  lnk[0] = 0;       /* num of entries on the list */
  lnk[2] = lnk_max; /* initialize head node */
  lnk[3] = 2;       /* head node */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscLLCondensedView(PetscInt *lnk)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "LLCondensed of size %" PetscInt_FMT ", (val,  next)\n", lnk[0]));
  for (PetscInt k = 2; k < lnk[0] + 2; ++k) PetscCall(PetscPrintf(PETSC_COMM_SELF, " %" PetscInt_FMT ": (%" PetscInt_FMT ", %" PetscInt_FMT ")\n", 2 * k, lnk[2 * k], lnk[2 * k + 1]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Free memories used by the list
*/
static inline PetscErrorCode PetscLLCondensedDestroy(PetscInt *lnk, PetscBT bt)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(lnk));
  PetscCall(PetscBTDestroy(&bt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 Same as PetscLLCondensedCreate(), but does not use non-scalable O(lnk_max) bitarray
  Input Parameters:
    nlnk_max  - max length of the list
  Output Parameters:
    lnk       - list created and initialized
*/
static inline PetscErrorCode PetscLLCondensedCreate_Scalable(PetscInt nlnk_max, PetscInt **lnk)
{
  PetscInt *llnk, lsize = 0;

  PetscFunctionBegin;
  PetscCall(PetscIntMultError(2, nlnk_max + 2, &lsize));
  PetscCall(PetscMalloc1(lsize, lnk));
  llnk    = *lnk;
  llnk[0] = 0;             /* number of entries on the list */
  llnk[2] = PETSC_INT_MAX; /* value in the head node */
  llnk[3] = 2;             /* next for the head node */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscLLCondensedExpand_Scalable(PetscInt nlnk_max, PetscInt **lnk)
{
  PetscInt lsize = 0;

  PetscFunctionBegin;
  PetscCall(PetscIntMultError(2, nlnk_max + 2, &lsize));
  PetscCall(PetscRealloc(lsize * sizeof(PetscInt), lnk));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscLLCondensedAddSorted_Scalable(PetscInt nidx, const PetscInt indices[], PetscInt lnk[])
{
  PetscInt location = 2;      /* head */
  PetscInt nlnk     = lnk[0]; /* num of entries on the input lnk */

  for (PetscInt k = 0; k < nidx; k++) {
    const PetscInt entry = indices[k];
    PetscInt       next, lnkdata;

    /* search for insertion location */
    do {
      next     = location + 1;  /* link from previous node to next node */
      location = lnk[next];     /* idx of next node */
      lnkdata  = lnk[location]; /* value of next node */
    } while (entry > lnkdata);
    if (entry < lnkdata) {
      /* insertion location is found, add entry into lnk */
      const PetscInt newnode = 2 * (nlnk + 2); /* index for this new node */
      lnk[next]              = newnode;        /* connect previous node to the new node */
      lnk[newnode]           = entry;          /* set value of the new node */
      lnk[newnode + 1]       = location;       /* connect new node to next node */
      location               = newnode;        /* next search starts from the new node */
      nlnk++;
    }
  }
  lnk[0] = nlnk; /* number of entries in the list */
  return PETSC_SUCCESS;
}

static inline PetscErrorCode PetscLLCondensedClean_Scalable(PETSC_UNUSED PetscInt nidx, PetscInt *indices, PetscInt *lnk)
{
  const PetscInt nlnk = lnk[0];
  PetscInt       next = lnk[3]; /* head node */

  for (PetscInt k = 0; k < nlnk; k++) {
    indices[k] = lnk[next];
    next       = lnk[next + 1];
  }
  lnk[0] = 0; /* num of entries on the list */
  lnk[3] = 2; /* head node */
  return PETSC_SUCCESS;
}

static inline PetscErrorCode PetscLLCondensedDestroy_Scalable(PetscInt *lnk)
{
  return PetscFree(lnk);
}

/*
      lnk[0]   number of links
      lnk[1]   number of entries
      lnk[3n]  value
      lnk[3n+1] len
      lnk[3n+2] link to next value

      The next three are always the first link

      lnk[3]    PETSC_INT_MIN+1
      lnk[4]    1
      lnk[5]    link to first real entry

      The next three are always the last link

      lnk[6]    PETSC_INT_MAX - 1
      lnk[7]    1
      lnk[8]    next valid link (this is the same as lnk[0] but without the decreases)
*/

static inline PetscErrorCode PetscLLCondensedCreate_fast(PetscInt nlnk_max, PetscInt **lnk)
{
  PetscInt *llnk;
  PetscInt  lsize = 0;

  PetscFunctionBegin;
  PetscCall(PetscIntMultError(3, nlnk_max + 3, &lsize));
  PetscCall(PetscMalloc1(lsize, lnk));
  llnk    = *lnk;
  llnk[0] = 0;                 /* nlnk: number of entries on the list */
  llnk[1] = 0;                 /* number of integer entries represented in list */
  llnk[3] = PETSC_INT_MIN + 1; /* value in the first node */
  llnk[4] = 1;                 /* count for the first node */
  llnk[5] = 6;                 /* next for the first node */
  llnk[6] = PETSC_INT_MAX - 1; /* value in the last node */
  llnk[7] = 1;                 /* count for the last node */
  llnk[8] = 0;                 /* next valid node to be used */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscLLCondensedAddSorted_fast(PetscInt nidx, const PetscInt indices[], PetscInt lnk[])
{
  for (PetscInt k = 0, prev = 3 /* first value */; k < nidx; k++) {
    const PetscInt entry = indices[k];
    PetscInt       next  = lnk[prev + 2];

    /* search for insertion location */
    while (entry >= lnk[next]) {
      prev = next;
      next = lnk[next + 2];
    }
    /* entry is in range of previous list */
    if (entry < lnk[prev] + lnk[prev + 1]) continue;
    lnk[1]++;
    /* entry is right after previous list */
    if (entry == lnk[prev] + lnk[prev + 1]) {
      lnk[prev + 1]++;
      if (lnk[next] == entry + 1) { /* combine two contiguous strings */
        lnk[prev + 1] += lnk[next + 1];
        lnk[prev + 2] = lnk[next + 2];
        next          = lnk[next + 2];
        lnk[0]--;
      }
      continue;
    }
    /* entry is right before next list */
    if (entry == lnk[next] - 1) {
      lnk[next]--;
      lnk[next + 1]++;
      prev = next;
      next = lnk[prev + 2];
      continue;
    }
    /*  add entry into lnk */
    lnk[prev + 2] = 3 * ((lnk[8]++) + 3); /* connect previous node to the new node */
    prev          = lnk[prev + 2];
    lnk[prev]     = entry; /* set value of the new node */
    lnk[prev + 1] = 1;     /* number of values in contiguous string is one to start */
    lnk[prev + 2] = next;  /* connect new node to next node */
    lnk[0]++;
  }
  return PETSC_SUCCESS;
}

static inline PetscErrorCode PetscLLCondensedClean_fast(PETSC_UNUSED PetscInt nidx, PetscInt *indices, PetscInt *lnk)
{
  const PetscInt nlnk = lnk[0];
  PetscInt       next = lnk[5]; /* first node */

  for (PetscInt k = 0, cnt = 0; k < nlnk; k++) {
    for (PetscInt j = 0; j < lnk[next + 1]; j++) indices[cnt++] = lnk[next] + j;
    next = lnk[next + 2];
  }
  lnk[0] = 0;                 /* nlnk: number of links */
  lnk[1] = 0;                 /* number of integer entries represented in list */
  lnk[3] = PETSC_INT_MIN + 1; /* value in the first node */
  lnk[4] = 1;                 /* count for the first node */
  lnk[5] = 6;                 /* next for the first node */
  lnk[6] = PETSC_INT_MAX - 1; /* value in the last node */
  lnk[7] = 1;                 /* count for the last node */
  lnk[8] = 0;                 /* next valid location to make link */
  return PETSC_SUCCESS;
}

static inline PetscErrorCode PetscLLCondensedView_fast(const PetscInt *lnk)
{
  const PetscInt nlnk = lnk[0];
  PetscInt       next = lnk[5]; /* first node */

  for (PetscInt k = 0; k < nlnk; k++) {
#if 0 /* Debugging code */
    printf("%d value %d len %d next %d\n", next, lnk[next], lnk[next + 1], lnk[next + 2]);
#endif
    next = lnk[next + 2];
  }
  return PETSC_SUCCESS;
}

static inline PetscErrorCode PetscLLCondensedDestroy_fast(PetscInt *lnk)
{
  return PetscFree(lnk);
}

PETSC_EXTERN PetscErrorCode PetscCDCreate(PetscInt, PetscCoarsenData **);
PETSC_EXTERN PetscErrorCode PetscCDDestroy(PetscCoarsenData *);
PETSC_EXTERN PetscErrorCode PetscCDIntNdSetID(PetscCDIntNd *, PetscInt);
PETSC_EXTERN PetscErrorCode PetscCDIntNdGetID(const PetscCDIntNd *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscCDAppendID(PetscCoarsenData *, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscCDMoveAppend(PetscCoarsenData *, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscCDAppendNode(PetscCoarsenData *, PetscInt, PetscCDIntNd *);
PETSC_EXTERN PetscErrorCode PetscCDRemoveNextNode(PetscCoarsenData *, PetscInt, PetscCDIntNd *);
PETSC_EXTERN PetscErrorCode PetscCDCountAt(const PetscCoarsenData *, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscCDIsEmptyAt(const PetscCoarsenData *, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscCDSetChunkSize(PetscCoarsenData *, PetscInt);
PETSC_EXTERN PetscErrorCode PetscCDPrint(const PetscCoarsenData *, PetscInt, MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscCDGetNonemptyIS(PetscCoarsenData *, IS *);
PETSC_EXTERN PetscErrorCode PetscCDGetMat(PetscCoarsenData *, Mat *);
PETSC_EXTERN PetscErrorCode PetscCDSetMat(PetscCoarsenData *, Mat);
PETSC_EXTERN PetscErrorCode PetscCDClearMat(PetscCoarsenData *);
PETSC_EXTERN PetscErrorCode PetscCDRemoveAllAt(PetscCoarsenData *, PetscInt);
PETSC_EXTERN PetscErrorCode PetscCDCount(const PetscCoarsenData *, PetscInt *_sz);

PETSC_EXTERN PetscErrorCode PetscCDGetHeadPos(const PetscCoarsenData *, PetscInt, PetscCDIntNd **);
PETSC_EXTERN PetscErrorCode PetscCDGetNextPos(const PetscCoarsenData *, PetscInt, PetscCDIntNd **);
PETSC_EXTERN PetscErrorCode PetscCDGetASMBlocks(const PetscCoarsenData *, const PetscInt, PetscInt *, IS **);

PETSC_SINGLE_LIBRARY_VISIBILITY_INTERNAL PetscErrorCode MatFDColoringApply_AIJ(Mat, MatFDColoring, Vec, void *);

PETSC_EXTERN PetscLogEvent MAT_Mult;
PETSC_EXTERN PetscLogEvent MAT_MultAdd;
PETSC_EXTERN PetscLogEvent MAT_MultTranspose;
PETSC_EXTERN PetscLogEvent MAT_MultHermitianTranspose;
PETSC_EXTERN PetscLogEvent MAT_MultTransposeAdd;
PETSC_EXTERN PetscLogEvent MAT_MultHermitianTransposeAdd;
PETSC_EXTERN PetscLogEvent MAT_Solve;
PETSC_EXTERN PetscLogEvent MAT_Solves;
PETSC_EXTERN PetscLogEvent MAT_SolveAdd;
PETSC_EXTERN PetscLogEvent MAT_SolveTranspose;
PETSC_EXTERN PetscLogEvent MAT_SolveTransposeAdd;
PETSC_EXTERN PetscLogEvent MAT_SOR;
PETSC_EXTERN PetscLogEvent MAT_ForwardSolve;
PETSC_EXTERN PetscLogEvent MAT_BackwardSolve;
PETSC_EXTERN PetscLogEvent MAT_LUFactor;
PETSC_EXTERN PetscLogEvent MAT_LUFactorSymbolic;
PETSC_EXTERN PetscLogEvent MAT_LUFactorNumeric;
PETSC_EXTERN PetscLogEvent MAT_QRFactor;
PETSC_EXTERN PetscLogEvent MAT_QRFactorSymbolic;
PETSC_EXTERN PetscLogEvent MAT_QRFactorNumeric;
PETSC_EXTERN PetscLogEvent MAT_CholeskyFactor;
PETSC_EXTERN PetscLogEvent MAT_CholeskyFactorSymbolic;
PETSC_EXTERN PetscLogEvent MAT_CholeskyFactorNumeric;
PETSC_EXTERN PetscLogEvent MAT_ILUFactor;
PETSC_EXTERN PetscLogEvent MAT_ILUFactorSymbolic;
PETSC_EXTERN PetscLogEvent MAT_ICCFactorSymbolic;
PETSC_EXTERN PetscLogEvent MAT_Copy;
PETSC_EXTERN PetscLogEvent MAT_Convert;
PETSC_EXTERN PetscLogEvent MAT_Scale;
PETSC_EXTERN PetscLogEvent MAT_AssemblyBegin;
PETSC_EXTERN PetscLogEvent MAT_AssemblyEnd;
PETSC_EXTERN PetscLogEvent MAT_SetValues;
PETSC_EXTERN PetscLogEvent MAT_GetValues;
PETSC_EXTERN PetscLogEvent MAT_GetRow;
PETSC_EXTERN PetscLogEvent MAT_GetRowIJ;
PETSC_EXTERN PetscLogEvent MAT_CreateSubMats;
PETSC_EXTERN PetscLogEvent MAT_GetOrdering;
PETSC_EXTERN PetscLogEvent MAT_RedundantMat;
PETSC_EXTERN PetscLogEvent MAT_IncreaseOverlap;
PETSC_EXTERN PetscLogEvent MAT_Partitioning;
PETSC_EXTERN PetscLogEvent MAT_PartitioningND;
PETSC_EXTERN PetscLogEvent MAT_Coarsen;
PETSC_EXTERN PetscLogEvent MAT_ZeroEntries;
PETSC_EXTERN PetscLogEvent MAT_Load;
PETSC_EXTERN PetscLogEvent MAT_View;
PETSC_EXTERN PetscLogEvent MAT_AXPY;
PETSC_EXTERN PetscLogEvent MAT_FDColoringCreate;
PETSC_EXTERN PetscLogEvent MAT_TransposeColoringCreate;
PETSC_EXTERN PetscLogEvent MAT_FDColoringSetUp;
PETSC_EXTERN PetscLogEvent MAT_FDColoringApply;
PETSC_EXTERN PetscLogEvent MAT_Transpose;
PETSC_EXTERN PetscLogEvent MAT_FDColoringFunction;
PETSC_EXTERN PetscLogEvent MAT_CreateSubMat;
PETSC_EXTERN PetscLogEvent MAT_MatSolve;
PETSC_EXTERN PetscLogEvent MAT_MatTrSolve;
PETSC_EXTERN PetscLogEvent MAT_MatMultSymbolic;
PETSC_EXTERN PetscLogEvent MAT_MatMultNumeric;
PETSC_EXTERN PetscLogEvent MAT_Getlocalmatcondensed;
PETSC_EXTERN PetscLogEvent MAT_GetBrowsOfAcols;
PETSC_EXTERN PetscLogEvent MAT_GetBrowsOfAocols;
PETSC_EXTERN PetscLogEvent MAT_PtAPSymbolic;
PETSC_EXTERN PetscLogEvent MAT_PtAPNumeric;
PETSC_EXTERN PetscLogEvent MAT_Seqstompinum;
PETSC_EXTERN PetscLogEvent MAT_Seqstompisym;
PETSC_EXTERN PetscLogEvent MAT_Seqstompi;
PETSC_EXTERN PetscLogEvent MAT_Getlocalmat;
PETSC_EXTERN PetscLogEvent MAT_RARtSymbolic;
PETSC_EXTERN PetscLogEvent MAT_RARtNumeric;
PETSC_EXTERN PetscLogEvent MAT_MatTransposeMultSymbolic;
PETSC_EXTERN PetscLogEvent MAT_MatTransposeMultNumeric;
PETSC_EXTERN PetscLogEvent MAT_TransposeMatMultSymbolic;
PETSC_EXTERN PetscLogEvent MAT_TransposeMatMultNumeric;
PETSC_EXTERN PetscLogEvent MAT_MatMatMultSymbolic;
PETSC_EXTERN PetscLogEvent MAT_MatMatMultNumeric;
PETSC_EXTERN PetscLogEvent MAT_Getsymtransreduced;
PETSC_EXTERN PetscLogEvent MAT_GetSeqNonzeroStructure;
PETSC_EXTERN PetscLogEvent MATMFFD_Mult;
PETSC_EXTERN PetscLogEvent MAT_GetMultiProcBlock;
PETSC_EXTERN PetscLogEvent MAT_CUSPARSECopyToGPU;
PETSC_EXTERN PetscLogEvent MAT_CUSPARSECopyFromGPU;
PETSC_EXTERN PetscLogEvent MAT_CUSPARSEGenerateTranspose;
PETSC_EXTERN PetscLogEvent MAT_CUSPARSESolveAnalysis;
PETSC_EXTERN PetscLogEvent MAT_HIPSPARSECopyToGPU;
PETSC_EXTERN PetscLogEvent MAT_HIPSPARSECopyFromGPU;
PETSC_EXTERN PetscLogEvent MAT_HIPSPARSEGenerateTranspose;
PETSC_EXTERN PetscLogEvent MAT_HIPSPARSESolveAnalysis;
PETSC_EXTERN PetscLogEvent MAT_SetValuesBatch;
PETSC_EXTERN PetscLogEvent MAT_CreateGraph;
PETSC_EXTERN PetscLogEvent MAT_ViennaCLCopyToGPU;
PETSC_EXTERN PetscLogEvent MAT_DenseCopyToGPU;
PETSC_EXTERN PetscLogEvent MAT_DenseCopyFromGPU;
PETSC_EXTERN PetscLogEvent MAT_Merge;
PETSC_EXTERN PetscLogEvent MAT_Residual;
PETSC_EXTERN PetscLogEvent MAT_SetRandom;
PETSC_EXTERN PetscLogEvent MAT_FactorFactS;
PETSC_EXTERN PetscLogEvent MAT_FactorInvS;
PETSC_EXTERN PetscLogEvent MAT_PreallCOO;
PETSC_EXTERN PetscLogEvent MAT_SetVCOO;
PETSC_EXTERN PetscLogEvent MATCOLORING_Apply;
PETSC_EXTERN PetscLogEvent MATCOLORING_Comm;
PETSC_EXTERN PetscLogEvent MATCOLORING_Local;
PETSC_EXTERN PetscLogEvent MATCOLORING_ISCreate;
PETSC_EXTERN PetscLogEvent MATCOLORING_SetUp;
PETSC_EXTERN PetscLogEvent MATCOLORING_Weights;
PETSC_EXTERN PetscLogEvent MAT_H2Opus_Build;
PETSC_EXTERN PetscLogEvent MAT_H2Opus_Compress;
PETSC_EXTERN PetscLogEvent MAT_H2Opus_Orthog;
PETSC_EXTERN PetscLogEvent MAT_H2Opus_LR;
PETSC_EXTERN PetscLogEvent MAT_CUDACopyToGPU;
PETSC_EXTERN PetscLogEvent MAT_HIPCopyToGPU;
