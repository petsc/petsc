
/* $Id: vecimpl.h,v 1.43 1998/01/28 21:00:24 bsmith Exp bsmith $ */

/* 
   This private file should not be included in users' code.
   Defines the fields shared by all vector implementations.
*/

#ifndef __VECIMPL 
#define __VECIMPL
#include "vec.h"

/* vector operations */
struct _VeOps {
  int  (*duplicate)(Vec,Vec*),           /* get single vector */
       (*duplicatevecs)(Vec,int,Vec**),  /* get array of vectors */
       (*destroyvecs)(Vec*,int),         /* free array of vectors */
       (*dot)(Vec,Vec,Scalar*),          /* z = x^H * y */
       (*mdot)(int,Vec,Vec*,Scalar*),    /* z[j] = x dot y[j] */
       (*norm)(Vec,NormType,double*),    /* z = sqrt(x^H * x) */
       (*tdot)(Vec,Vec,Scalar*),         /* x'*y */
       (*mtdot)(int,Vec,Vec*,Scalar*),   /* z[j] = x dot y[j] */
       (*scale)(Scalar*,Vec),            /* x = alpha * x   */
       (*copy)(Vec,Vec),                 /* y = x */
       (*set)(Scalar*,Vec),              /* y = alpha  */
       (*swap)(Vec,Vec),                 /* exchange x and y */
       (*axpy)(Scalar*,Vec,Vec),         /* y = y + alpha * x */
       (*axpby)(Scalar*,Scalar*,Vec,Vec),/* y = y + alpha * x + beta * y*/
       (*maxpy)(int,Scalar*,Vec,Vec*),   /* y = y + alpha[j] x[j] */
       (*aypx)(Scalar*,Vec,Vec),         /* y = x + alpha * y */
       (*waxpy)(Scalar*,Vec,Vec,Vec),    /* w = y + alpha * x */
       (*pointwisemult)(Vec,Vec,Vec),    /* w = x .* y */
       (*pointwisedivide)(Vec,Vec,Vec),  /* w = x ./ y */
       (*setvalues)(Vec,int,int*,Scalar*,InsertMode),
       (*assemblybegin)(Vec),            /* start global assembly */
       (*assemblyend)(Vec),              /* end global assembly */
       (*getarray)(Vec,Scalar**),        /* get data array */
       (*getsize)(Vec,int*),(*getlocalsize)(Vec,int*),
       (*getownershiprange)(Vec,int*,int*),
       (*restorearray)(Vec,Scalar**),    /* restore data array */
       (*max)(Vec,int*,double*),         /* z = max(x); idx=index of max(x) */
       (*min)(Vec,int*,double*),         /* z = min(x); idx=index of min(x) */
       (*setrandom)(PetscRandom,Vec),    /* set y[j] = random numbers */
       (*setoption)(Vec,VecOption),
       (*setvaluesblocked)(Vec,int,int*,Scalar*,InsertMode);
};

struct _p_Vec {
  PETSCHEADER(struct _VeOps ops)
  void                   *data;     /* implementation-specific data */
  int                    N, n;      /* global, local vector size */
  int                    bs;
  ISLocalToGlobalMapping mapping;   /* mapping used in VecSetValuesLocal() */
  ISLocalToGlobalMapping bmapping;  /* mapping used in VecSetValuesBlockedLocal() */
};

/*
     Common header shared by array based vectors, 
   currently Vec_Seq and Vec_MPI
*/
#define VECHEADER                         \
  int    n;                               \
  Scalar *array;                          \
  Scalar *array_allocated;            

typedef struct {
  VECHEADER
} Vec_ArrayBased;

/*
    Macros for accessing array-based vector fields quickly to
  avoid function call overhead.
*/
#define VecGetArray_Fast(x,a)     a = ((Vec_ArrayBased *)(x->data))->array
#define VecPlaceArray_Fast(x,a)   ((Vec_ArrayBased *)(x->data))->array = a
#define VecRestoreArray_Fast(x,a)
#define VecGetLocalSize_Fast(x,a) a = x->n;

/* Default obtain and release vectors; can be used by any implementation */
extern int     VecDuplicateVecs_Default(Vec, int, Vec **);
extern int     VecDestroyVecs_Default(Vec *,int);

/* --------------------------------------------------------------------*/
/*                                                                     */
/* Defines the data structures used in the Vec Scatter operations      */

typedef enum { VEC_SCATTER_SEQ_GENERAL, VEC_SCATTER_SEQ_STRIDE, 
               VEC_SCATTER_MPI_GENERAL, VEC_SCATTER_MPI_TOALL,
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
  int            nonmatching_computed;
  int            n_nonmatching;        /* number of "from"s  != "to"s */
  int            *slots_nonmatching;   /* locations of "from"s  != "to"s */
  int            is_copy,copy_start;   /* local scatter is a copy starting at copy_start */
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
  Scalar         *work1;
  Scalar         *work2;        
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
  Scalar                 *values;  /* buffer for all sends or receives */
  VecScatter_Seq_General local;    /* any part that happens to be local */
  MPI_Status             *sstatus;
  int                    use_readyreceiver,bs;
} VecScatter_MPI_General;

struct _p_VecScatter {
  PETSCHEADER(int dummy)
  int     to_n,from_n;
  int     inuse;   /* prevents corruption from mixing two scatters */
  int     (*postrecvs)(Vec,Vec,InsertMode,ScatterMode,VecScatter);
  int     (*begin)(Vec,Vec,InsertMode,ScatterMode,VecScatter);
  int     (*end)(Vec,Vec,InsertMode,ScatterMode,VecScatter);
  int     (*copy)(VecScatter,VecScatter);
  void    *fromdata,*todata;
};

#endif
