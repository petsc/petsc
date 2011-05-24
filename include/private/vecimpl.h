
/* 
   This private file should not be included in users' code.
   Defines the fields shared by all vector implementations.

*/

#ifndef __VECIMPL_H
#define __VECIMPL_H

#include <petscvec.h>
PETSC_EXTERN_CXX_BEGIN

/*S
     PetscLayout - defines layout of vectors and matrices across processes (which rows are owned by which processes)

   Level: developer


.seealso:  PetscLayoutCreate(), PetscLayoutDestroy()
S*/
typedef struct _n_PetscLayout* PetscLayout;
struct _n_PetscLayout{
  MPI_Comm               comm;
  PetscInt               n,N;         /* local, global vector size */
  PetscInt               rstart,rend; /* local start, local end + 1 */
  PetscInt               *range;      /* the offset of each processor */
  PetscInt               bs;          /* number of elements in each block (generally for multi-component problems) Do NOT multiply above numbers by bs */
  PetscInt               refcnt;      /* MPI Vecs obtained with VecDuplicate() and from MatGetVecs() reuse map of input object */
  ISLocalToGlobalMapping mapping;     /* mapping used in Vec/MatSetValuesLocal() */
  ISLocalToGlobalMapping bmapping;    /* mapping used in Vec/MatSetValuesBlockedLocal() */
};

extern PetscErrorCode PetscLayoutCreate(MPI_Comm,PetscLayout*);
extern PetscErrorCode PetscLayoutSetUp(PetscLayout);
extern PetscErrorCode PetscLayoutDestroy(PetscLayout*);
extern PetscErrorCode PetscLayoutCopy(PetscLayout,PetscLayout*);
extern PetscErrorCode PetscLayoutReference(PetscLayout,PetscLayout*);
extern PetscErrorCode  PetscLayoutSetLocalSize(PetscLayout,PetscInt);
extern PetscErrorCode  PetscLayoutGetLocalSize(PetscLayout,PetscInt *);
PetscPolymorphicFunction(PetscLayoutGetLocalSize,(PetscLayout m),(m,&s),PetscInt,s)
extern PetscErrorCode  PetscLayoutSetSize(PetscLayout,PetscInt);
extern PetscErrorCode  PetscLayoutGetSize(PetscLayout,PetscInt *);
PetscPolymorphicFunction(PetscLayoutGetSize,(PetscLayout m),(m,&s),PetscInt,s)
extern PetscErrorCode  PetscLayoutSetBlockSize(PetscLayout,PetscInt);
extern PetscErrorCode  PetscLayoutGetBlockSize(PetscLayout,PetscInt*);
extern PetscErrorCode  PetscLayoutGetRange(PetscLayout,PetscInt *,PetscInt *);
extern PetscErrorCode  PetscLayoutGetRanges(PetscLayout,const PetscInt *[]);
extern PetscErrorCode  PetscLayoutSetISLocalToGlobalMapping(PetscLayout,ISLocalToGlobalMapping);
extern PetscErrorCode  PetscLayoutSetISLocalToGlobalMappingBlock(PetscLayout,ISLocalToGlobalMapping);

#undef __FUNCT__
#define __FUNCT__ "PetscLayoutFindOwner"
/*@C
     PetscLayoutFindOwner - Find the owning rank for a global index

    Not Collective

   Input Parameters:
+    map - the layout
-    idx - global index to find the owner of

   Output Parameter:
.    owner - the owning rank

   Level: developer

    Fortran Notes:
      Not available from Fortran

@*/
PETSC_STATIC_INLINE PetscErrorCode PetscLayoutFindOwner(PetscLayout map,PetscInt idx,PetscInt *owner)
{
  PetscErrorCode ierr;
  PetscMPIInt    lo = 0,hi,t;
  PetscInt       bs = map->bs;

  PetscFunctionBegin;
  if (!((map->n >= 0) && (map->N >= 0) && (map->range))) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscLayoutSetUp() must be called first");
  if (idx < 0 || idx > map->N) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D is out of range",idx);
  ierr = MPI_Comm_size(map->comm,&hi);CHKERRQ(ierr);
  while (hi - lo > 1) {
    t = lo + (hi - lo) / 2;
    if (idx < map->range[t]/bs) hi = t;
    else                     lo = t;
  }
  *owner = lo;
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------*/
typedef struct _n_PetscUniformSection *PetscUniformSection;
struct _n_PetscUniformSection {
  MPI_Comm comm;
  PetscInt pStart, pEnd; /* The chart: all points are contained in [pStart, pEnd) */
  PetscInt numDof;       /* Describes layout of storage, point --> (constant # of values, (p - pStart)*constant # of values) */
};

/*S
  PetscSection - This is a mapping from DMMESH points to sets of values, which is
  our presentation of a fibre bundle.

  Level: developer

.seealso:  PetscSectionCreate(), PetscSectionDestroy()
S*/
typedef struct _n_PetscSection *PetscSection;
struct _n_PetscSection {
  struct _n_PetscUniformSection atlasLayout;  /* Layout for the atlas */
  PetscInt                     *atlasDof;     /* Describes layout of storage, point --> # of values */
  PetscInt                     *atlasOff;     /* Describes layout of storage, point --> offset into storage */
  PetscSection                  bc;           /* Describes constraints, point --> # local dofs which are constrained */
  PetscInt                     *bcIndices;    /* Local indices for constrained dofs */
  PetscInt                      refcnt;       /* Vecs obtained with VecDuplicate() and from MatGetVecs() reuse map of input object */
};

extern PetscErrorCode PetscSectionCreate(MPI_Comm,PetscSection*);
extern PetscErrorCode PetscSectionGetChart(PetscSection, PetscInt *, PetscInt *);
extern PetscErrorCode PetscSectionSetChart(PetscSection, PetscInt, PetscInt);
extern PetscErrorCode PetscSectionGetDof(PetscSection, PetscInt, PetscInt*);
extern PetscErrorCode PetscSectionSetDof(PetscSection, PetscInt, PetscInt);
extern PetscErrorCode PetscSectionGetConstraintDof(PetscSection, PetscInt, PetscInt*);
extern PetscErrorCode PetscSectionSetConstraintDof(PetscSection, PetscInt, PetscInt);
extern PetscErrorCode PetscSectionGetConstraintIndices(PetscSection, PetscInt, PetscInt**);
extern PetscErrorCode PetscSectionSetConstraintIndices(PetscSection, PetscInt, PetscInt*);
extern PetscErrorCode PetscSectionSetUp(PetscSection);
extern PetscErrorCode PetscSectionDestroy(PetscSection*);

extern PetscErrorCode VecGetValuesSection(Vec, PetscSection, PetscInt, PetscScalar **);
extern PetscErrorCode VecSetValuesSection(Vec, PetscSection, PetscInt, PetscScalar [], InsertMode);

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
  PetscErrorCode (*getsubvector)(Vec,IS,Vec*);
  PetscErrorCode (*restoresubvector)(Vec,IS,Vec*);
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

#if defined(PETSC_HAVE_CUSP)
/* Defines the flag structure that the CUSP arch uses. */
typedef enum {PETSC_CUSP_UNALLOCATED,PETSC_CUSP_GPU,PETSC_CUSP_CPU,PETSC_CUSP_BOTH} PetscCUSPFlag;
#endif

struct _p_Vec {
  PETSCHEADER(struct _VecOps);
  PetscLayout            map;
  void                   *data;     /* implementation-specific data */
  PetscBool              array_gotten;
  VecStash               stash,bstash; /* used for storing off-proc values during assembly */
  PetscBool              petscnative;  /* means the ->data starts with VECHEADER and can use VecGetArrayFast()*/
#if defined(PETSC_HAVE_CUSP)
  PetscCUSPFlag          valid_GPU_array;    /* indicates where the most recently modified vector data is (GPU or CPU) */
  void                   *spptr; /* if we're using CUSP, then this is the special pointer to the array on the GPU */
#endif
};

extern PetscLogEvent VEC_View, VEC_Max, VEC_Min, VEC_DotBarrier, VEC_Dot, VEC_MDotBarrier, VEC_MDot, VEC_TDot, VEC_MTDot;
extern PetscLogEvent VEC_Norm, VEC_Normalize, VEC_Scale, VEC_Copy, VEC_Set, VEC_AXPY, VEC_AYPX, VEC_WAXPY, VEC_MAXPY;
extern PetscLogEvent VEC_AssemblyEnd, VEC_PointwiseMult, VEC_SetValues, VEC_Load, VEC_ScatterBarrier, VEC_ScatterBegin, VEC_ScatterEnd;
extern PetscLogEvent VEC_SetRandom, VEC_ReduceArithmetic, VEC_ReduceBarrier, VEC_ReduceCommunication;
extern PetscLogEvent VEC_Swap, VEC_AssemblyBegin, VEC_NormBarrier, VEC_DotNormBarrier, VEC_DotNorm, VEC_AXPBYPCZ, VEC_Ops;
extern PetscLogEvent VEC_CUSPCopyToGPU, VEC_CUSPCopyFromGPU;
extern PetscLogEvent VEC_CUSPCopyToGPUSome, VEC_CUSPCopyFromGPUSome;

#if defined(PETSC_HAVE_CUSP)
extern PetscErrorCode VecCUSPCopyFromGPU(Vec v);
#endif

#undef __FUNCT__
#define __FUNCT__ "VecGetArrayRead"
PETSC_STATIC_INLINE PetscErrorCode VecGetArrayRead(Vec x,const PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (x->petscnative){
#if defined(PETSC_HAVE_CUSP)
    if (x->valid_GPU_array == PETSC_CUSP_GPU || !*((PetscScalar**)x->data)){
      ierr = VecCUSPCopyFromGPU(x);CHKERRQ(ierr);
    }
#endif
    *a = *((PetscScalar **)x->data);
  } else {
    ierr = (*x->ops->getarray)(x,(PetscScalar**)a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecRestoreArrayRead"
PETSC_STATIC_INLINE PetscErrorCode VecRestoreArrayRead(Vec x,const PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (x->petscnative){
#if defined(PETSC_HAVE_CUSP)
    if (x->valid_GPU_array != PETSC_CUSP_UNALLOCATED) {
      x->valid_GPU_array = PETSC_CUSP_BOTH;
    }
#endif
  } else {
    ierr = (*x->ops->restorearray)(x,(PetscScalar**)a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetArray"
PETSC_STATIC_INLINE PetscErrorCode VecGetArray(Vec x,PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (x->petscnative){
#if defined(PETSC_HAVE_CUSP)
    if (x->valid_GPU_array == PETSC_CUSP_GPU || !*((PetscScalar**)x->data)){
      ierr = VecCUSPCopyFromGPU(x);CHKERRQ(ierr);
    }
#endif
    *a = *((PetscScalar **)x->data);
  } else {
    ierr = (*x->ops->getarray)(x,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecRestoreArray"
PETSC_STATIC_INLINE PetscErrorCode VecRestoreArray(Vec x,PetscScalar *a[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (x->petscnative){
#if defined(PETSC_HAVE_CUSP)
    if (x->valid_GPU_array != PETSC_CUSP_UNALLOCATED) {
      x->valid_GPU_array = PETSC_CUSP_CPU;
    }
#endif
  } else {
    ierr = (*x->ops->restorearray)(x,a);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)x);CHKERRQ(ierr);
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
extern PetscErrorCode VecDuplicateVecs_Default(Vec,PetscInt,Vec *[]);
extern PetscErrorCode VecDestroyVecs_Default(PetscInt,Vec []);
extern PetscErrorCode VecLoad_Binary(Vec, PetscViewer);
extern PetscErrorCode VecLoad_Default(Vec, PetscViewer);

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

extern PetscErrorCode VecStashCreate_Private(MPI_Comm,PetscInt,VecStash*);
extern PetscErrorCode VecStashDestroy_Private(VecStash*);
extern PetscErrorCode VecStashExpand_Private(VecStash*,PetscInt);
extern PetscErrorCode VecStashScatterEnd_Private(VecStash*);
extern PetscErrorCode VecStashSetInitialSize_Private(VecStash*,PetscInt);
extern PetscErrorCode VecStashGetInfo_Private(VecStash*,PetscInt*,PetscInt*);
extern PetscErrorCode VecStashScatterBegin_Private(VecStash*,PetscInt*);
extern PetscErrorCode VecStashScatterGetMesg_Private(VecStash*,PetscMPIInt*,PetscInt**,PetscScalar**,PetscInt*);

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

extern PetscErrorCode VecStrideGather_Default(Vec,PetscInt,Vec,InsertMode);
extern PetscErrorCode VecStrideScatter_Default(Vec,PetscInt,Vec,InsertMode);
extern PetscErrorCode VecReciprocal_Default(Vec);

#if defined(PETSC_HAVE_MATLAB_ENGINE)
EXTERN_C_BEGIN
extern PetscErrorCode VecMatlabEnginePut_Default(PetscObject,void*);
extern PetscErrorCode VecMatlabEngineGet_Default(PetscObject,void*);
EXTERN_C_END
#endif

PETSC_EXTERN_CXX_END

/* Reset __FUNCT__ in case the user does not define it themselves */
#undef __FUNCT__
#define __FUNCT__ "User provided function"

#endif
