
/* $Id: vecimpl.h,v 1.89 2001/09/19 16:07:46 bsmith Exp $ */

/* 
   This private file should not be included in users' code.
   Defines the fields shared by all vector implementations.
*/

#ifndef __VECIMPL_H
#define __VECIMPL_H

#include "petscvec.h"

struct _PetscMapOps {
  int (*setfromoptions)(PetscMap),
      (*destroy)(PetscMap);
};

struct _p_PetscMap {
  PETSCHEADER(struct _PetscMapOps)
  int  n,N;         /* local, global vector size */
  int  rstart,rend; /* local start, local end + 1 */
  int *range;       /* the offset of each processor */
};

/* ----------------------------------------------------------------------------*/

typedef struct _VecOps *VecOps;
struct _VecOps {
  int  (*duplicate)(Vec,Vec*),              /* get single vector */
       (*duplicatevecs)(Vec,int,Vec**),     /* get array of vectors */
       (*destroyvecs)(const Vec[],int),     /* free array of vectors */
       (*dot)(Vec,Vec,PetscScalar*),             /* z = x^H * y */
       (*mdot)(int,Vec,const Vec[],PetscScalar*), /* z[j] = x dot y[j] */
       (*norm)(Vec,NormType,PetscReal*),        /* z = sqrt(x^H * x) */
       (*tdot)(Vec,Vec,PetscScalar*),             /* x'*y */
       (*mtdot)(int,Vec,const Vec[],PetscScalar*),/* z[j] = x dot y[j] */
       (*scale)(const PetscScalar*,Vec),          /* x = alpha * x   */
       (*copy)(Vec,Vec),                     /* y = x */
       (*set)(const PetscScalar*,Vec),            /* y = alpha  */
       (*swap)(Vec,Vec),                     /* exchange x and y */
       (*axpy)(const PetscScalar*,Vec,Vec),       /* y = y + alpha * x */
       (*axpby)(const PetscScalar*,const PetscScalar*,Vec,Vec), /* y = y + alpha * x + beta * y*/
       (*maxpy)(int,const PetscScalar*,Vec,Vec*), /* y = y + alpha[j] x[j] */
       (*aypx)(const PetscScalar*,Vec,Vec),       /* y = x + alpha * y */
       (*waxpy)(const PetscScalar*,Vec,Vec,Vec),  /* w = y + alpha * x */
       (*pointwisemult)(Vec,Vec,Vec),        /* w = x .* y */
       (*pointwisedivide)(Vec,Vec,Vec),      /* w = x ./ y */
       (*setvalues)(Vec,int,const int[],const PetscScalar[],InsertMode),
       (*assemblybegin)(Vec),                /* start global assembly */
       (*assemblyend)(Vec),                  /* end global assembly */
       (*getarray)(Vec,PetscScalar**),            /* get data array */
       (*getsize)(Vec,int*),
       (*getlocalsize)(Vec,int*),
       (*restorearray)(Vec,PetscScalar**),        /* restore data array */
       (*max)(Vec,int*,PetscReal*),      /* z = max(x); idx=index of max(x) */
       (*min)(Vec,int*,PetscReal*),      /* z = min(x); idx=index of min(x) */
       (*setrandom)(PetscRandom,Vec),        /* set y[j] = random numbers */
       (*setoption)(Vec,VecOption),
       (*setvaluesblocked)(Vec,int,const int[],const PetscScalar[],InsertMode),
       (*destroy)(Vec),
       (*view)(Vec,PetscViewer),
       (*placearray)(Vec,const PetscScalar*),     /* place data array */
       (*replacearray)(Vec,const PetscScalar*),     /* replace data array */
       (*dot_local)(Vec,Vec,PetscScalar*),
       (*tdot_local)(Vec,Vec,PetscScalar*),
       (*norm_local)(Vec,NormType,PetscReal*),
       (*loadintovector)(PetscViewer,Vec),
       (*reciprocal)(Vec),
       (*viewnative)(Vec,PetscViewer),
       (*conjugate)(Vec),
       (*setlocaltoglobalmapping)(Vec,ISLocalToGlobalMapping),
       (*setvalueslocal)(Vec,int,const int *,const PetscScalar *,InsertMode),
       (*resetarray)(Vec),      /* vector points to its original array, i.e. undoes any VecPlaceArray() */
       (*setfromoptions)(Vec),
       (*maxpointwisedivide)(Vec,Vec,PetscScalar*);      /* m = max abs(x ./ y) */
};

/* 
    The stash is used to temporarily store inserted vec values that 
  belong to another processor. During the assembly phase the stashed 
  values are moved to the correct processor and 
*/

typedef struct {
  int           nmax;                   /* maximum stash size */
  int           umax;                   /* max stash size user wants */
  int           oldnmax;                /* the nmax value used previously */
  int           n;                      /* stash size */
  int           bs;                     /* block size of the stash */
  int           reallocs;               /* preserve the no of mallocs invoked */           
  int           *idx;                   /* global row numbers in stash */
  PetscScalar   *array;                 /* array to hold stashed values */
  /* The following variables are used for communication */
  MPI_Comm      comm;
  int           size,rank;
  int           tag1,tag2;
  MPI_Request   *send_waits;            /* array of send requests */
  MPI_Request   *recv_waits;            /* array of receive requests */
  MPI_Status    *send_status;           /* array of send status */
  int           nsends,nrecvs;          /* numbers of sends and receives */
  PetscScalar   *svalues,*rvalues;      /* sending and receiving data */
  int           rmax;                   /* maximum message length */
  int           *nprocs;                /* tmp data used both duiring scatterbegin and end */
  int           nprocessed;             /* number of messages already processed */
  PetscTruth    donotstash;
  InsertMode    insertmode;
  int           *bowners;
} VecStash;

struct _p_Vec {
  PETSCHEADER(struct _VecOps)
  PetscMap               map;
  void                   *data;     /* implementation-specific data */
  int                    N,n;      /* global, local vector size */
  int                    bs;
  ISLocalToGlobalMapping mapping;   /* mapping used in VecSetValuesLocal() */
  ISLocalToGlobalMapping bmapping;  /* mapping used in VecSetValuesBlockedLocal() */
  PetscTruth             array_gotten;
  VecStash               stash,bstash; /* used for storing off-proc values during assembly */
  PetscTruth             petscnative;  /* means the ->data starts with VECHEADER and can use VecGetArrayFast()*/
  void                   *esivec;      /* ESI wrapper of vector */
};

#define VecGetArrayFast(x,a)     ((x)->petscnative ? (*(a) = *((PetscScalar **)(x)->data),0) : VecGetArray((x),(a)))
#define VecRestoreArrayFast(x,a) ((x)->petscnative ? 0 : VecRestoreArray((x),(a)))

/*
     Common header shared by array based vectors, 
   currently Vec_Seq and Vec_MPI
*/
#define VECHEADER                         \
  PetscScalar *array;                          \
  PetscScalar *array_allocated;            

/* Default obtain and release vectors; can be used by any implementation */
EXTERN int VecDuplicateVecs_Default(Vec,int,Vec *[]);
EXTERN int VecDestroyVecs_Default(const Vec [],int);

EXTERN int VecLoadIntoVector_Default(PetscViewer,Vec);

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
  int            n;                    /* number of components to scatter */
  int            *slots;               /* locations of components */
  /*
       The next three fields are used in parallel scatters, they contain 
       optimization in the special case that the "to" vector and the "from" 
       vector are the same, so one only needs copy components that truly 
       copies instead of just y[idx[i]] = y[jdx[i]] where idx[i] == jdx[i].
  */
  PetscTruth     nonmatching_computed;
  int            n_nonmatching;        /* number of "from"s  != "to"s */
  int            *slots_nonmatching;   /* locations of "from"s  != "to"s */
  PetscTruth     is_copy;
  int            copy_start;   /* local scatter is a copy starting at copy_start */
  int            copy_length;
} VecScatter_Seq_General;

typedef struct {
  VecScatterType type;
  int            n;
  int            first;
  int            step;           
} VecScatter_Seq_Stride;

/*
   This scatter is for a global vector copied (completely) to each processor (or all to one)
*/
typedef struct {
  VecScatterType type;
  int            *count;        /* elements of vector on each processor */
  PetscScalar    *work1;
  PetscScalar    *work2;        
} VecScatter_MPI_ToAll;

/*
   This is the general parallel scatter
*/
typedef struct { 
  VecScatterType         type;
  int                    n;        /* number of processors to send/receive */
  int                    *starts;  /* starting point in indices and values for each proc*/ 
  int                    *indices; /* list of all components sent or received */
  int                    *procs;   /* processors we are communicating with in scatter */
  MPI_Request            *requests,*rev_requests;
  PetscScalar            *values;  /* buffer for all sends or receives */
  VecScatter_Seq_General local;    /* any part that happens to be local */
  MPI_Status             *sstatus,*rstatus;
  PetscTruth             use_readyreceiver;
  int                    bs;
  PetscTruth             sendfirst;
} VecScatter_MPI_General;

struct _p_VecScatter {
  PETSCHEADER(int)
  int        to_n,from_n;
  PetscTruth inuse;   /* prevents corruption from mixing two scatters */
  PetscTruth beginandendtogether;         /* indicates that the scatter begin and end
                                          function are called together, VecScatterEnd()
                                          is then treated as a nop */
  PetscTruth packtogether; /* packs all the messages before sending, same with receive */
  int        (*postrecvs)(Vec,Vec,InsertMode,ScatterMode,VecScatter);
  int        (*begin)(Vec,Vec,InsertMode,ScatterMode,VecScatter);
  int        (*end)(Vec,Vec,InsertMode,ScatterMode,VecScatter);
  int        (*copy)(VecScatter,VecScatter);
  int        (*destroy)(VecScatter);
  int        (*view)(VecScatter,PetscViewer);
  void       *fromdata,*todata;
};

EXTERN int VecStashCreate_Private(MPI_Comm,int,VecStash*);
EXTERN int VecStashDestroy_Private(VecStash*);
EXTERN int VecStashExpand_Private(VecStash*,int);
EXTERN int VecStashScatterEnd_Private(VecStash*);
EXTERN int VecStashSetInitialSize_Private(VecStash*,int);
EXTERN int VecStashGetInfo_Private(VecStash*,int*,int*);
EXTERN int VecStashScatterBegin_Private(VecStash*,int*);
EXTERN int VecStashScatterGetMesg_Private(VecStash*,int*,int**,PetscScalar**,int*);

/* 
   The following are implemented as macros to avoid the function
   call overhead.

   extern int VecStashValue_Private(VecStash*,int,PetscScalar);
   extern int VecStashValuesBlocked_Private(VecStash*,int,PetscScalar*);
*/

/*
  VecStashValue_Private - inserts a single values into the stash.

  Input Parameters:
  stash  - the stash
  idx    - the global of the inserted value
  values - the value inserted
*/
#define VecStashValue_Private(stash,row,value) \
{  \
  /* Check and see if we have sufficient memory */ \
  if (((stash)->n + 1) > (stash)->nmax) { \
    ierr = VecStashExpand_Private(stash,1);CHKERRQ(ierr); \
  } \
  (stash)->idx[(stash)->n]   = row; \
  (stash)->array[(stash)->n] = value; \
  (stash)->n++; \
}

/*
  VecStashValuesBlocked_Private - inserts 1 block of values into the stash. 

  Input Parameters:
  stash  - the stash
  idx    - the global block index
  values - the values inserted
*/
#define VecStashValuesBlocked_Private(stash,row,values) \
{ \
  int    jj,stash_bs=(stash)->bs; \
  PetscScalar *array; \
  if (((stash)->n+1) > (stash)->nmax) { \
    ierr = VecStashExpand_Private(stash,1);CHKERRQ(ierr); \
  } \
  array = (stash)->array + stash_bs*(stash)->n; \
  (stash)->idx[(stash)->n]   = row; \
  for (jj=0; jj<stash_bs; jj++) { array[jj] = values[jj];} \
  (stash)->n++; \
}

EXTERN int VecReciprocal_Default(Vec);

#if defined(PETSC_HAVE_MATLAB_ENGINE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN_C_BEGIN
EXTERN int VecMatlabEnginePut_Default(PetscObject,void*);
EXTERN int VecMatlabEngineGet_Default(PetscObject,void*);
EXTERN_C_END
#endif


#endif

