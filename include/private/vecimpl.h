
/* 
   This private file should not be included in users' code.
   Defines the fields shared by all vector implementations.

*/

#ifndef __VECIMPL_H
#define __VECIMPL_H

#include "petscvec.h"
PETSC_EXTERN_CXX_BEGIN

/*S
     PetscLayout - defines layout of vectors and matrices across processes (which rows are owned by which processes)

   Level: developer


.seealso:  PetscLayoutCreate(), PetscLayoutDestroy()
S*/
typedef struct _n_PetscLayout* PetscLayout;
struct _n_PetscLayout{
  MPI_Comm  comm;
  PetscInt  n,N;         /* local, global vector size */
  PetscInt  rstart,rend; /* local start, local end + 1 */
  PetscInt  *range;      /* the offset of each processor */
  PetscInt  bs;          /* number of elements in each block (generally for multi-component problems) Do NOT multiply above numbers by bs */
  PetscInt  refcnt;      /* MPI Vecs obtained with VecDuplicate() and from MatGetVecs() reuse map of input object */
};

EXTERN PetscErrorCode PetscLayoutCreate(MPI_Comm,PetscLayout*);
EXTERN PetscErrorCode PetscLayoutSetUp(PetscLayout);
EXTERN PetscErrorCode PetscLayoutDestroy(PetscLayout);
EXTERN PetscErrorCode PetscLayoutCopy(PetscLayout,PetscLayout*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutSetLocalSize(PetscLayout,PetscInt);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutGetLocalSize(PetscLayout,PetscInt *);
PetscPolymorphicFunction(PetscLayoutGetLocalSize,(PetscLayout m),(m,&s),PetscInt,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutSetSize(PetscLayout,PetscInt);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutGetSize(PetscLayout,PetscInt *);
PetscPolymorphicFunction(PetscLayoutGetSize,(PetscLayout m),(m,&s),PetscInt,s)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutSetBlockSize(PetscLayout,PetscInt);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutGetBlockSize(PetscLayout,PetscInt*);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutGetRange(PetscLayout,PetscInt *,PetscInt *);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutGetRanges(PetscLayout,const PetscInt *[]);

/* ----------------------------------------------------------------------------*/

typedef struct _VecOps *VecOps;
struct _VecOps {
  PetscErrorCode (*duplicate)(Vec,Vec*);         /* get single vector */
  PetscErrorCode (*duplicatevecs)(Vec,PetscInt,Vec**);     /* get array of vectors */
  PetscErrorCode (*destroyvecs)(Vec[],PetscInt);           /* free array of vectors */
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
  PetscErrorCode (*setfromoptions)(Vec);
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

#if defined(PETSC_HAVE_CUDA)
/* Defines the flag structure that the CUDA arch uses. */
typedef enum {PETSC_CUDA_UNALLOCATED,PETSC_CUDA_GPU,PETSC_CUDA_CPU,PETSC_CUDA_BOTH} PetscCUDAFlag;
#endif

struct _p_Vec {
  PETSCHEADER(struct _VecOps);
  PetscLayout            map;
  void                   *data;     /* implementation-specific data */
  ISLocalToGlobalMapping mapping;   /* mapping used in VecSetValuesLocal() */
  ISLocalToGlobalMapping bmapping;  /* mapping used in VecSetValuesBlockedLocal() */
  PetscBool              array_gotten;
  VecStash               stash,bstash; /* used for storing off-proc values during assembly */
  PetscBool              petscnative;  /* means the ->data starts with VECHEADER and can use VecGetArrayFast()*/
#if defined(PETSC_HAVE_CUDA)
  PetscCUDAFlag          valid_GPU_array;    /* indicates where the most recently modified vector data is (GPU or CPU) */
  void                   *spptr; /* if we're using CUDA, then this is the special pointer to the array on the GPU */
#endif
};

extern PetscLogEvent VEC_View, VEC_Max, VEC_Min, VEC_DotBarrier, VEC_Dot, VEC_MDotBarrier, VEC_MDot, VEC_TDot, VEC_MTDot;
extern PetscLogEvent VEC_Norm, VEC_Normalize, VEC_Scale, VEC_Copy, VEC_Set, VEC_AXPY, VEC_AYPX, VEC_WAXPY, VEC_MAXPY;
extern PetscLogEvent VEC_AssemblyEnd, VEC_PointwiseMult, VEC_SetValues, VEC_Load, VEC_ScatterBarrier, VEC_ScatterBegin, VEC_ScatterEnd;
extern PetscLogEvent VEC_SetRandom, VEC_ReduceArithmetic, VEC_ReduceBarrier, VEC_ReduceCommunication;
extern PetscLogEvent VEC_Swap, VEC_AssemblyBegin, VEC_NormBarrier, VEC_DotNormBarrier, VEC_DotNorm, VEC_AXPBYPCZ, VEC_Ops;
extern PetscLogEvent VEC_CUDACopyToGPU, VEC_CUDACopyFromGPU;
extern PetscLogEvent VEC_CUDACopyToGPUSome, VEC_CUDACopyFromGPUSome;

#if defined(PETSC_HAVE_CUDA)
EXTERN PetscErrorCode VecCUDACopyFromGPU(Vec v);
#endif

/*
    These are for use only in the Vec implementations. They DO NOT increase any vectors state. The increase of the vector state
   is always handled by the outter vector operation, for example VecAXPY()
*/
#undef __FUNCT__
#define __FUNCT__ "VecGetArrayPrivate"
PETSC_STATIC_INLINE PetscErrorCode VecGetArrayPrivate(Vec x, PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (x->petscnative){
#if defined(PETSC_HAVE_CUDA)
    ierr = VecCUDACopyFromGPU(x);CHKERRQ(ierr);
#endif
    *a = *((PetscScalar **)x->data);
  } else {
    ierr = VecGetArray_Private(x,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecRestoreArrayPrivate"
PETSC_STATIC_INLINE PetscErrorCode VecRestoreArrayPrivate(Vec x, PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (x->petscnative){
#if defined(PETSC_HAVE_CUDA)
    if (x->valid_GPU_array != PETSC_CUDA_UNALLOCATED) {
      x->valid_GPU_array = PETSC_CUDA_CPU;
    }
#endif
  } else {
    ierr = VecRestoreArray_Private(x,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   These do not increase the vector state because we know the vector cannot be changed
*/
#undef __FUNCT__
#define __FUNCT__ "VecGetArrayRead"
PETSC_STATIC_INLINE PetscErrorCode VecGetArrayRead(Vec x, const PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayPrivate(x,(PetscScalar**)a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecRestoreArrayRead"
PETSC_STATIC_INLINE PetscErrorCode VecRestoreArrayRead(Vec x, const PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* do not mark the vector as owned by the CPU since it may be shared between the CPU and GPU */
  if (!x->petscnative){
    ierr = VecRestoreArray_Private(x,(PetscScalar**)a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecGetArray"
PETSC_STATIC_INLINE PetscErrorCode VecGetArray(Vec x, PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayPrivate(x,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecRestoreArray"
PETSC_STATIC_INLINE PetscErrorCode VecRestoreArray(Vec x, PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecRestoreArrayPrivate(x,a);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecGetArrayPrivate2"
PETSC_STATIC_INLINE PetscErrorCode VecGetArrayPrivate2(Vec x, PetscScalar *xx[], Vec y, PetscScalar *yy[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecGetArrayPrivate(x,xx);CHKERRQ(ierr);
  if (x == y) {
    *yy = *xx;
  } else {
    ierr = VecGetArrayPrivate(y,yy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecRestoreArrayPrivate2"
PETSC_STATIC_INLINE PetscErrorCode VecRestoreArrayPrivate2(Vec x, PetscScalar *xx[], Vec y, PetscScalar *yy[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecRestoreArrayPrivate(x,xx);CHKERRQ(ierr);
  if (x != y) {
    ierr = VecRestoreArrayPrivate(y,yy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetArrayPrivate3"
PETSC_STATIC_INLINE PetscErrorCode VecGetArrayPrivate3(Vec x, PetscScalar *xx[], Vec y, PetscScalar *yy[], Vec w, PetscScalar *ww[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecGetArrayPrivate(x,xx);CHKERRQ(ierr);
  if (x == y) {
    *yy = *xx;
  } else {
    ierr = VecGetArrayPrivate(y,yy);CHKERRQ(ierr);
  }
  if (w == x) {
    *ww = *xx;
  } else if(w == y) {
    *ww = *yy;
  } else {
    ierr = VecGetArrayPrivate(w,ww);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecRestoreArrayPrivate3"
/*
    Does not increase the state of the vectors
*/
PETSC_STATIC_INLINE PetscErrorCode VecRestoreArrayPrivate3(Vec x, PetscScalar *xx[], Vec y, PetscScalar *yy[], Vec w, PetscScalar *ww[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecRestoreArrayPrivate(x,xx);CHKERRQ(ierr);
  if (x != y){
    ierr = VecRestoreArrayPrivate(y,yy);CHKERRQ(ierr);
  }
  if (w != x && w != y){
    ierr = VecRestoreArrayPrivate(w,ww);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
     Common header shared by array based vectors, 
   currently Vec_Seq and Vec_MPI
*/
#define VECHEADER                          \
  PetscScalar *array;                      \
  PetscScalar *array_allocated;                        /* if the array was allocated by PETSc this is its pointer */  \
  PetscScalar *unplacedarray;                           /* if one called VecPlaceArray(), this is where it stashed the original */

/* Default obtain and release vectors; can be used by any implementation */
EXTERN PetscErrorCode VecDuplicateVecs_Default(Vec,PetscInt,Vec *[]);
EXTERN PetscErrorCode VecDestroyVecs_Default(Vec [],PetscInt);
EXTERN PetscErrorCode VecLoad_Binary(Vec, PetscViewer);
EXTERN PetscErrorCode VecLoad_Default(Vec, PetscViewer);

extern PetscInt NormIds[7];  /* map from NormType to IDs used to cache/retreive values of norms */

/* --------------------------------------------------------------------*/
/*                                                                     */
/* Defines the data structures used in the Vec Scatter operations      */

typedef enum { VEC_SCATTER_SEQ_GENERAL,VEC_SCATTER_SEQ_STRIDE,
               VEC_SCATTER_MPI_GENERAL,VEC_SCATTER_MPI_TOALL,
               VEC_SCATTER_MPI_TOONE} VecScatterType;

/* 
   These scatters are for the purely local case.
*/
typedef struct {
  VecScatterType type;
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
  VecScatterType type;
  PetscInt       n;
  PetscInt       first;
  PetscInt       step;           
} VecScatter_Seq_Stride;

/*
   This scatter is for a global vector copied (completely) to each processor (or all to one)
*/
typedef struct {
  VecScatterType type;
  PetscMPIInt    *count;        /* elements of vector on each processor */
  PetscMPIInt    *displx;        
  PetscScalar    *work1;
  PetscScalar    *work2;        
} VecScatter_MPI_ToAll;

/*
   This is the general parallel scatter
*/
typedef struct { 
  VecScatterType         type;
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

struct _p_VecScatter {
  PETSCHEADER(int);
  PetscInt       to_n,from_n;
  PetscBool      inuse;                /* prevents corruption from mixing two scatters */
  PetscBool      beginandendtogether;  /* indicates that the scatter begin and end  function are called together, VecScatterEnd()
                                          is then treated as a nop */
  PetscBool      packtogether;         /* packs all the messages before sending, same with receive */
  PetscBool      reproduce;            /* always receive the ghost points in the same order of processes */
  PetscErrorCode (*begin)(VecScatter,Vec,Vec,InsertMode,ScatterMode);
  PetscErrorCode (*end)(VecScatter,Vec,Vec,InsertMode,ScatterMode);
  PetscErrorCode (*copy)(VecScatter,VecScatter);
  PetscErrorCode (*destroy)(VecScatter);
  PetscErrorCode (*view)(VecScatter,PetscViewer);
  void           *fromdata,*todata;
  void           *spptr;
};

EXTERN PetscErrorCode VecStashCreate_Private(MPI_Comm,PetscInt,VecStash*);
EXTERN PetscErrorCode VecStashDestroy_Private(VecStash*);
EXTERN PetscErrorCode VecStashExpand_Private(VecStash*,PetscInt);
EXTERN PetscErrorCode VecStashScatterEnd_Private(VecStash*);
EXTERN PetscErrorCode VecStashSetInitialSize_Private(VecStash*,PetscInt);
EXTERN PetscErrorCode VecStashGetInfo_Private(VecStash*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode VecStashScatterBegin_Private(VecStash*,PetscInt*);
EXTERN PetscErrorCode VecStashScatterGetMesg_Private(VecStash*,PetscMPIInt*,PetscInt**,PetscScalar**,PetscInt*);

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

EXTERN PetscErrorCode VecStrideGather_Default(Vec,PetscInt,Vec,InsertMode);
EXTERN PetscErrorCode VecStrideScatter_Default(Vec,PetscInt,Vec,InsertMode);
EXTERN PetscErrorCode VecReciprocal_Default(Vec);

#if defined(PETSC_HAVE_MATLAB_ENGINE)
EXTERN_C_BEGIN
EXTERN PetscErrorCode VecMatlabEnginePut_Default(PetscObject,void*);
EXTERN PetscErrorCode VecMatlabEngineGet_Default(PetscObject,void*);
EXTERN_C_END
#endif


PETSC_EXTERN_CXX_END
#endif

