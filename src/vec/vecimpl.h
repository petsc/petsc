
/* $Id: matimpl.h,v 1.60 1996/07/11 04:05:13 balay Exp $ */

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
       (*getvecs)(Vec,int,Vec**),        /* get array of vectors */
       (*destroyvecs)(Vec*,int),            /* free array of vectors */
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
       (*setrandom)(PetscRandom,Vec);       /* set y[j] = random numbers */
};

struct _Vec {
  PETSCHEADER                            /* general PETSc header */
  struct _VeOps ops;                     /* vector operations */
  void          *data;                   /* implementation-specific data */
  int           N, n;                    /* global, local vector size */
};

/* Default obtain and release vectors; can be used by any implementation */
extern int     VecGetVecs_Default(Vec, int, Vec **);
extern int     VecDestroyVecs_Private(Vec *,int);

/* --------------------------------------------------------------------*/
/*                                                                     */
/* Defines the data structures used in the Vec Scatter operations      */

typedef enum { VEC_SCATTER_SEQ_GENERAL, VEC_SCATTER_SEQ_VEC, 
               VEC_SCATTER_MPI_GENERAL, VEC_SCATTER_MPI_TOALL} VecScatterType;

/* 
   These scatters are for the purely local case.
*/

typedef struct {
  VecScatterType type;
  int            n;         /* number of components to scatter */
  int            *slots;    /* locations of components */
} VecScatter_Seq_General;

typedef struct {
  VecScatterType type;
  int            n;
  int            first;
  int            step;           
} VecScatter_Seq_Stride;

/*
   This scatter is for a global vector copied (completely) to each processor
*/

typedef struct {
  VecScatterType type;
  int            *count;        /* elements of vector on each processor */
  Scalar         *work, *work2;        
} VecScatter_MPI_ToAll;

/*
   This is the general parallel scatter
*/
typedef struct { 
  VecScatterType     type;
  int                n;        /* number of processors to send/receive */
  int                nbelow;   /* number with lower process id */
  int                nself;    /* number sending to self */
  int                *starts;  /* starting point in indices and values for each proc*/ 
  int                *indices; /* list of all components sent or received */
  int                *procs;   /* processors we are communicating with in scatter */
  MPI_Request        *requests;
  Scalar             *values;  /* buffer for all sends or receives */
                               /* note that we pack/unpack ourselves;
                                   we do not use MPI packing */
  VecScatter_Seq_General local;    /* any part that happens to be local */
  MPI_Status         *sstatus;
  int                local_is_matching; 
                               /* Indicates that the slots[] of the in and out
                                  local "scatters" are identical, this is used
                                  to avoid  an unnecessary copy when the input
                                  and output vectors are identical */
} VecScatter_MPI;

struct _VecScatter {
  PETSCHEADER
  int     inuse;           /* prevents corruption from mixing two scatters */
  int     (*scatterbegin)(Vec,Vec,InsertMode,int,VecScatter);
  int     (*scatterend)(Vec,Vec,InsertMode,int,VecScatter);
  int     (*copy)(VecScatter,VecScatter);
  void    *fromdata,*todata;
};


#endif
