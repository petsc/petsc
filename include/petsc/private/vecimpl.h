
/*
   This private file should not be included in users' code.
   Defines the fields shared by all vector implementations.

*/

#ifndef __VECIMPL_H
#define __VECIMPL_H

#include <petscvec.h>
#include <petsc/private/petscimpl.h>
#include <petscviewer.h>

PETSC_EXTERN PetscBool VecRegisterAllCalled;
PETSC_EXTERN PetscErrorCode VecRegisterAll(void);

/* ----------------------------------------------------------------------------*/

typedef struct _VecOps *VecOps;
struct _VecOps {
  PetscErrorCode (*duplicate)(Vec,Vec*);         /* get single vector */
  PetscErrorCode (*duplicatevecs)(Vec,PetscInt,Vec**);     /* get array of vectors */
  PetscErrorCode (*destroyvecs)(PetscInt,Vec[]);           /* free array of vectors */
  PetscErrorCode (*dot)(Vec,Vec,PetscScalar*);             /* z = x^H * y */
  PetscErrorCode (*mdot)(Vec,PetscInt,const Vec[],PetscScalar*); /* z[j] = x dot y[j] */
  PetscErrorCode (*norm)(Vec,NormType,PetscReal*);        /* z = sqrt(x^H * x) */
  PetscErrorCode (*tdot)(Vec,Vec,PetscScalar*);             /* x'*y */
  PetscErrorCode (*mtdot)(Vec,PetscInt,const Vec[],PetscScalar*);/* z[j] = x dot y[j] */
  PetscErrorCode (*scale)(Vec,PetscScalar);                 /* x = alpha * x   */
  PetscErrorCode (*copy)(Vec,Vec);                     /* y = x */
  PetscErrorCode (*set)(Vec,PetscScalar);                        /* y = alpha  */
  PetscErrorCode (*swap)(Vec,Vec);                               /* exchange x and y */
  PetscErrorCode (*axpy)(Vec,PetscScalar,Vec);                   /* y = y + alpha * x */
  PetscErrorCode (*axpby)(Vec,PetscScalar,PetscScalar,Vec);      /* y = alpha * x + beta * y*/
  PetscErrorCode (*maxpy)(Vec,PetscInt,const PetscScalar*,Vec*); /* y = y + alpha[j] x[j] */
  PetscErrorCode (*aypx)(Vec,PetscScalar,Vec);                   /* y = x + alpha * y */
  PetscErrorCode (*waxpy)(Vec,PetscScalar,Vec,Vec);         /* w = y + alpha * x */
  PetscErrorCode (*axpbypcz)(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);   /* z = alpha * x + beta *y + gamma *z*/
  PetscErrorCode (*pointwisemult)(Vec,Vec,Vec);        /* w = x .* y */
  PetscErrorCode (*pointwisedivide)(Vec,Vec,Vec);      /* w = x ./ y */
  PetscErrorCode (*setvalues)(Vec,PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
  PetscErrorCode (*assemblybegin)(Vec);                /* start global assembly */
  PetscErrorCode (*assemblyend)(Vec);                  /* end global assembly */
  PetscErrorCode (*getarray)(Vec,PetscScalar**);            /* get data array */
  PetscErrorCode (*getsize)(Vec,PetscInt*);
  PetscErrorCode (*getlocalsize)(Vec,PetscInt*);
  PetscErrorCode (*restorearray)(Vec,PetscScalar**);        /* restore data array */
  PetscErrorCode (*max)(Vec,PetscInt*,PetscReal*);      /* z = max(x); idx=index of max(x) */
  PetscErrorCode (*min)(Vec,PetscInt*,PetscReal*);      /* z = min(x); idx=index of min(x) */
  PetscErrorCode (*setrandom)(Vec,PetscRandom);         /* set y[j] = random numbers */
  PetscErrorCode (*setoption)(Vec,VecOption,PetscBool );
  PetscErrorCode (*setvaluesblocked)(Vec,PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
  PetscErrorCode (*destroy)(Vec);
  PetscErrorCode (*view)(Vec,PetscViewer);
  PetscErrorCode (*placearray)(Vec,const PetscScalar*);     /* place data array */
  PetscErrorCode (*replacearray)(Vec,const PetscScalar*);     /* replace data array */
  PetscErrorCode (*dot_local)(Vec,Vec,PetscScalar*);
  PetscErrorCode (*tdot_local)(Vec,Vec,PetscScalar*);
  PetscErrorCode (*norm_local)(Vec,NormType,PetscReal*);
  PetscErrorCode (*mdot_local)(Vec,PetscInt,const Vec[],PetscScalar*);
  PetscErrorCode (*mtdot_local)(Vec,PetscInt,const Vec[],PetscScalar*);
  PetscErrorCode (*load)(Vec,PetscViewer);
  PetscErrorCode (*reciprocal)(Vec);
  PetscErrorCode (*conjugate)(Vec);
  PetscErrorCode (*setlocaltoglobalmapping)(Vec,ISLocalToGlobalMapping);
  PetscErrorCode (*setvalueslocal)(Vec,PetscInt,const PetscInt *,const PetscScalar *,InsertMode);
  PetscErrorCode (*resetarray)(Vec);      /* vector points to its original array, i.e. undoes any VecPlaceArray() */
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,Vec);
  PetscErrorCode (*maxpointwisedivide)(Vec,Vec,PetscReal*);      /* m = max abs(x ./ y) */
  PetscErrorCode (*pointwisemax)(Vec,Vec,Vec);
  PetscErrorCode (*pointwisemaxabs)(Vec,Vec,Vec);
  PetscErrorCode (*pointwisemin)(Vec,Vec,Vec);
  PetscErrorCode (*getvalues)(Vec,PetscInt,const PetscInt[],PetscScalar[]);
  PetscErrorCode (*sqrt)(Vec);
  PetscErrorCode (*abs)(Vec);
  PetscErrorCode (*exp)(Vec);
  PetscErrorCode (*log)(Vec);
  PetscErrorCode (*shift)(Vec);
  PetscErrorCode (*create)(Vec);
  PetscErrorCode (*stridegather)(Vec,PetscInt,Vec,InsertMode);
  PetscErrorCode (*stridescatter)(Vec,PetscInt,Vec,InsertMode);
  PetscErrorCode (*dotnorm2)(Vec,Vec,PetscScalar*,PetscScalar*);
  PetscErrorCode (*getsubvector)(Vec,IS,Vec*);
  PetscErrorCode (*restoresubvector)(Vec,IS,Vec*);
  PetscErrorCode (*getarrayread)(Vec,const PetscScalar**);
  PetscErrorCode (*restorearrayread)(Vec,const PetscScalar**);
  PetscErrorCode (*stridesubsetgather)(Vec,PetscInt,const PetscInt[],const PetscInt[],Vec,InsertMode);
  PetscErrorCode (*stridesubsetscatter)(Vec,PetscInt,const PetscInt[],const PetscInt[],Vec,InsertMode);
  PetscErrorCode (*viewnative)(Vec,PetscViewer);
  PetscErrorCode (*loadnative)(Vec,PetscViewer);
  PetscErrorCode (*getlocalvector)(Vec,Vec);
  PetscErrorCode (*restorelocalvector)(Vec,Vec);
  PetscErrorCode (*getlocalvectorread)(Vec,Vec);
  PetscErrorCode (*restorelocalvectorread)(Vec,Vec);
};

/*
    The stash is used to temporarily store inserted vec values that
  belong to another processor. During the assembly phase the stashed
  values are moved to the correct processor and
*/

typedef struct {
  PetscInt      nmax;                   /* maximum stash size */
  PetscInt      umax;                   /* max stash size user wants */
  PetscInt      oldnmax;                /* the nmax value used previously */
  PetscInt      n;                      /* stash size */
  PetscInt      bs;                     /* block size of the stash */
  PetscInt      reallocs;               /* preserve the no of mallocs invoked */
  PetscInt      *idx;                   /* global row numbers in stash */
  PetscScalar   *array;                 /* array to hold stashed values */
  /* The following variables are used for communication */
  MPI_Comm      comm;
  PetscMPIInt   size,rank;
  PetscMPIInt   tag1,tag2;
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  MPI_Status    *send_status;           /* array of send status */
  PetscInt      nsends,nrecvs;          /* numbers of sends and receives */
  PetscScalar   *svalues,*rvalues;      /* sending and receiving data */
  PetscInt      *sindices,*rindices;
  PetscInt      rmax;                   /* maximum message length */
  PetscInt      *nprocs;                /* tmp data used both during scatterbegin and end */
  PetscInt      nprocessed;             /* number of messages already processed */
  PetscBool     donotstash;
  PetscBool     ignorenegidx;           /* ignore negative indices passed into VecSetValues/VetGetValues */
  InsertMode    insertmode;
  PetscInt      *bowners;
} VecStash;

struct _p_Vec {
  PETSCHEADER(struct _VecOps);
  PetscLayout            map;
  void                   *data;     /* implementation-specific data */
  PetscBool              array_gotten;
  VecStash               stash,bstash; /* used for storing off-proc values during assembly */
  PetscBool              petscnative;  /* means the ->data starts with VECHEADER and can use VecGetArrayFast()*/
  PetscInt               lock;   /* vector is locked to read only */
#if defined(PETSC_HAVE_CUSP)
  PetscCUSPFlag          valid_GPU_array;    /* indicates where the most recently modified vector data is (GPU or CPU) */
  void                   *spptr; /* if we're using CUSP, then this is the special pointer to the array on the GPU */
#elif defined(PETSC_HAVE_VIENNACL)
  PetscViennaCLFlag      valid_GPU_array;    /* indicates where the most recently modified vector data is (GPU or CPU) */
  void                   *spptr; /* if we're using ViennaCL, then this is the special pointer to the array on the GPU */
#elif defined(PETSC_HAVE_VECCUDA)
  PetscCUDAFlag          valid_GPU_array;    /* indicates where the most recently modified vector data is (GPU or CPU) */
  void                   *spptr; /* if we're using CUDA, then this is the special pointer to the array on the GPU */
#endif
};

PETSC_EXTERN PetscLogEvent VEC_View, VEC_Max, VEC_Min, VEC_DotBarrier, VEC_Dot, VEC_MDotBarrier, VEC_MDot, VEC_TDot, VEC_MTDot;
PETSC_EXTERN PetscLogEvent VEC_Norm, VEC_Normalize, VEC_Scale, VEC_Copy, VEC_Set, VEC_AXPY, VEC_AYPX, VEC_WAXPY, VEC_MAXPY;
PETSC_EXTERN PetscLogEvent VEC_AssemblyEnd, VEC_PointwiseMult, VEC_SetValues, VEC_Load, VEC_ScatterBarrier, VEC_ScatterBegin, VEC_ScatterEnd;
PETSC_EXTERN PetscLogEvent VEC_SetRandom, VEC_ReduceArithmetic, VEC_ReduceBarrier, VEC_ReduceCommunication;
PETSC_EXTERN PetscLogEvent VEC_ReduceBegin,VEC_ReduceEnd;
PETSC_EXTERN PetscLogEvent VEC_Swap, VEC_AssemblyBegin, VEC_NormBarrier, VEC_DotNormBarrier, VEC_DotNorm, VEC_AXPBYPCZ, VEC_Ops;
PETSC_EXTERN PetscLogEvent VEC_CUSPCopyToGPU, VEC_CUSPCopyFromGPU;
PETSC_EXTERN PetscLogEvent VEC_CUSPCopyToGPUSome, VEC_CUSPCopyFromGPUSome;
PETSC_EXTERN PetscLogEvent VEC_ViennaCLCopyToGPU,     VEC_ViennaCLCopyFromGPU;
PETSC_EXTERN PetscLogEvent VEC_CUDACopyToGPU, VEC_CUDACopyFromGPU;
PETSC_EXTERN PetscLogEvent VEC_CUDACopyToGPUSome, VEC_CUDACopyFromGPUSome;

PETSC_EXTERN PetscErrorCode VecView_Seq(Vec,PetscViewer);
#if defined(PETSC_HAVE_CUSP)
PETSC_EXTERN PetscErrorCode VecCUSPAllocateCheckHost(Vec v);
PETSC_EXTERN PetscErrorCode VecCUSPCopyFromGPU(Vec v);
#elif defined(PETSC_HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode VecViennaCLAllocateCheckHost(Vec v);
PETSC_EXTERN PetscErrorCode VecViennaCLCopyFromGPU(Vec v);
#elif defined(PETSC_HAVE_VECCUDA)
PETSC_EXTERN PetscErrorCode VecCUDAAllocateCheckHost(Vec v);
PETSC_EXTERN PetscErrorCode VecCUDACopyFromGPU(Vec v);
#endif


/*
     Common header shared by array based vectors,
   currently Vec_Seq and Vec_MPI
*/
#define VECHEADER                          \
  PetscScalar *array;                      \
  PetscScalar *array_allocated;                        /* if the array was allocated by PETSc this is its pointer */  \
  PetscScalar *unplacedarray;                           /* if one called VecPlaceArray(), this is where it stashed the original */

/* Default obtain and release vectors; can be used by any implementation */
PETSC_INTERN PetscErrorCode VecDuplicateVecs_Default(Vec,PetscInt,Vec *[]);
PETSC_INTERN PetscErrorCode VecDestroyVecs_Default(PetscInt,Vec []);
PETSC_INTERN PetscErrorCode VecLoad_Binary(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Default(Vec, PetscViewer);

PETSC_EXTERN PetscInt  NormIds[7];  /* map from NormType to IDs used to cache/retreive values of norms */

/* --------------------------------------------------------------------*/
/*                                                                     */
/* Defines the data structures used in the Vec Scatter operations      */

typedef enum { VEC_SCATTER_SEQ_GENERAL,VEC_SCATTER_SEQ_STRIDE,
               VEC_SCATTER_MPI_GENERAL,VEC_SCATTER_MPI_TOALL,
               VEC_SCATTER_MPI_TOONE} VecScatterType;

#define VECSCATTER_IMPL_HEADER \
      VecScatterType type;

typedef struct {
  VECSCATTER_IMPL_HEADER
} VecScatter_Common;

/*
   These scatters are for the purely local case.
*/
typedef struct {
  VECSCATTER_IMPL_HEADER
  PetscInt       n;                    /* number of components to scatter */
  PetscInt       *vslots;              /* locations of components */
  /*
       The next three fields are used in parallel scatters, they contain
       optimization in the special case that the "to" vector and the "from"
       vector are the same, so one only needs copy components that truly
       copies instead of just y[idx[i]] = y[jdx[i]] where idx[i] == jdx[i].
  */
  PetscBool      nonmatching_computed;
  PetscInt       n_nonmatching;        /* number of "from"s  != "to"s */
  PetscInt       *slots_nonmatching;   /* locations of "from"s  != "to"s */
  PetscBool      is_copy;
  PetscInt       copy_start;   /* local scatter is a copy starting at copy_start */
  PetscInt       copy_length;
} VecScatter_Seq_General;

typedef struct {
  VECSCATTER_IMPL_HEADER
  PetscInt       n;
  PetscInt       first;
  PetscInt       step;
} VecScatter_Seq_Stride;

/*
   This scatter is for a global vector copied (completely) to each processor (or all to one)
*/
typedef struct {
  VECSCATTER_IMPL_HEADER
  PetscMPIInt    *count;        /* elements of vector on each processor */
  PetscMPIInt    *displx;
  PetscScalar    *work1;
  PetscScalar    *work2;
} VecScatter_MPI_ToAll;

/*
   This is the general parallel scatter
*/
typedef struct {
  VECSCATTER_IMPL_HEADER
  PetscInt               n;        /* number of processors to send/receive */
  PetscInt               *starts;  /* starting point in indices and values for each proc*/
  PetscInt               *indices; /* list of all components sent or received */
  PetscMPIInt            *procs;   /* processors we are communicating with in scatter */
  MPI_Request            *requests,*rev_requests;
  PetscScalar            *values;  /* buffer for all sends or receives */
  VecScatter_Seq_General local;    /* any part that happens to be local */
  MPI_Status             *sstatus,*rstatus;
  PetscBool              use_readyreceiver;
  PetscInt               bs;
  PetscBool              sendfirst;
  PetscBool              contiq;
  /* for MPI_Alltoallv() approach */
  PetscBool              use_alltoallv;
  PetscMPIInt            *counts,*displs;
  /* for MPI_Alltoallw() approach */
  PetscBool              use_alltoallw;
#if defined(PETSC_HAVE_MPI_ALLTOALLW)
  PetscMPIInt            *wcounts,*wdispls;
  MPI_Datatype           *types;
#endif
  PetscBool              use_window;
#if defined(PETSC_HAVE_MPI_WIN_CREATE)
  MPI_Win                window;
  PetscInt               *winstarts;    /* displacements in the processes I am putting to */
#endif
} VecScatter_MPI_General;


PETSC_INTERN PetscErrorCode VecScatterGetTypes_Private(VecScatter,VecScatterType*,VecScatterType*);
PETSC_INTERN PetscErrorCode VecScatterIsSequential_Private(VecScatter_Common*,PetscBool*);

typedef struct _VecScatterOps *VecScatterOps;
struct _VecScatterOps {
  PetscErrorCode (*begin)(VecScatter,Vec,Vec,InsertMode,ScatterMode);
  PetscErrorCode (*end)(VecScatter,Vec,Vec,InsertMode,ScatterMode);
  PetscErrorCode (*copy)(VecScatter,VecScatter);
  PetscErrorCode (*destroy)(VecScatter);
  PetscErrorCode (*view)(VecScatter,PetscViewer);
  PetscErrorCode (*viewfromoptions)(VecScatter,const char prefix[],const char name[]); 
  PetscErrorCode (*remap)(VecScatter,PetscInt *,PetscInt*);
  PetscErrorCode (*getmerged)(VecScatter,PetscBool *);
};

struct _p_VecScatter {
  PETSCHEADER(struct _VecScatterOps);
  PetscInt       to_n,from_n;
  PetscBool      inuse;                /* prevents corruption from mixing two scatters */
  PetscBool      beginandendtogether;  /* indicates that the scatter begin and end  function are called together, VecScatterEnd()
                                          is then treated as a nop */
  PetscBool      packtogether;         /* packs all the messages before sending, same with receive */
  PetscBool      reproduce;            /* always receive the ghost points in the same order of processes */
  void           *fromdata,*todata;
  void           *spptr;
};

PETSC_INTERN PetscErrorCode VecStashCreate_Private(MPI_Comm,PetscInt,VecStash*);
PETSC_INTERN PetscErrorCode VecStashDestroy_Private(VecStash*);
PETSC_INTERN PetscErrorCode VecStashExpand_Private(VecStash*,PetscInt);
PETSC_INTERN PetscErrorCode VecStashScatterEnd_Private(VecStash*);
PETSC_INTERN PetscErrorCode VecStashSetInitialSize_Private(VecStash*,PetscInt);
PETSC_INTERN PetscErrorCode VecStashGetInfo_Private(VecStash*,PetscInt*,PetscInt*);
PETSC_INTERN PetscErrorCode VecStashScatterBegin_Private(VecStash*,PetscInt*);
PETSC_INTERN PetscErrorCode VecStashScatterGetMesg_Private(VecStash*,PetscMPIInt*,PetscInt**,PetscScalar**,PetscInt*);
PETSC_INTERN PetscErrorCode VecStashSortCompress_Private(VecStash*);
PETSC_INTERN PetscErrorCode VecStashGetOwnerList_Private(VecStash*,PetscLayout,PetscMPIInt*,PetscMPIInt**);

/*
  VecStashValue_Private - inserts a single value into the stash.

  Input Parameters:
  stash  - the stash
  idx    - the global of the inserted value
  values - the value inserted
*/
PETSC_STATIC_INLINE PetscErrorCode VecStashValue_Private(VecStash *stash,PetscInt row,PetscScalar value)
{
  PetscErrorCode ierr;
  /* Check and see if we have sufficient memory */
  if (((stash)->n + 1) > (stash)->nmax) {
    ierr = VecStashExpand_Private(stash,1);CHKERRQ(ierr);
  }
  (stash)->idx[(stash)->n]   = row;
  (stash)->array[(stash)->n] = value;
  (stash)->n++;
  return 0;
}

/*
  VecStashValuesBlocked_Private - inserts 1 block of values into the stash.

  Input Parameters:
  stash  - the stash
  idx    - the global block index
  values - the values inserted
*/
PETSC_STATIC_INLINE PetscErrorCode VecStashValuesBlocked_Private(VecStash *stash,PetscInt row,PetscScalar *values)
{
  PetscInt       jj,stash_bs=(stash)->bs;
  PetscScalar    *array;
  PetscErrorCode ierr;
  if (((stash)->n+1) > (stash)->nmax) {
    ierr = VecStashExpand_Private(stash,1);CHKERRQ(ierr);
  }
  array = (stash)->array + stash_bs*(stash)->n;
  (stash)->idx[(stash)->n]   = row;
  for (jj=0; jj<stash_bs; jj++) { array[jj] = values[jj];}
  (stash)->n++;
  return 0;
}

PETSC_INTERN PetscErrorCode VecStrideGather_Default(Vec,PetscInt,Vec,InsertMode);
PETSC_INTERN PetscErrorCode VecStrideScatter_Default(Vec,PetscInt,Vec,InsertMode);
PETSC_INTERN PetscErrorCode VecReciprocal_Default(Vec);
PETSC_INTERN PetscErrorCode VecStrideSubSetGather_Default(Vec,PetscInt,const PetscInt[],const PetscInt[],Vec,InsertMode);
PETSC_INTERN PetscErrorCode VecStrideSubSetScatter_Default(Vec,PetscInt,const PetscInt[],const PetscInt[],Vec,InsertMode);

#if defined(PETSC_HAVE_MATLAB_ENGINE)
PETSC_EXTERN PetscErrorCode VecMatlabEnginePut_Default(PetscObject,void*);
PETSC_EXTERN PetscErrorCode VecMatlabEngineGet_Default(PetscObject,void*);
#endif

PETSC_EXTERN PetscErrorCode PetscSectionGetField_Internal(PetscSection, PetscSection, Vec, PetscInt, PetscInt, PetscInt, IS *, Vec *);
PETSC_EXTERN PetscErrorCode PetscSectionRestoreField_Internal(PetscSection, PetscSection, Vec, PetscInt, PetscInt, PetscInt, IS *, Vec *);

/* Reset __FUNCT__ in case the user does not define it themselves */
#undef __FUNCT__
#define __FUNCT__ "User provided function"

#endif
