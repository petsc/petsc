/* $Id: vecimpl.h,v 1.18 1995/11/01 19:08:23 bsmith Exp bsmith $ */
/* 
   This should not be included in users code.
*/

#ifndef __VECIMPL 
#define __VECIMPL
#include "vec.h"

struct _VeOps {
  int  (*duplicate)(Vec,Vec*),           /*  Get single vector */
       (*getvecs)(Vec,int,Vec**),        /*  Get array of vectors */
       (*freevecs)(Vec*,int),            /* Free array of vectors */
       (*dot)(Vec,Vec,Scalar*),          /* z = x^H * y */
       (*mdot)(int,Vec,Vec*,Scalar*),    /*   z[j] = x dot y[j] */
       (*norm)(Vec,NormType,double*),    /* z = sqrt(x^H * x) */
       (*tdot)(Vec,Vec,Scalar*),         /* x'*y */
       (*mtdot)(int,Vec,Vec*,Scalar*),   /*   z[j] = x dot y[j] */
       (*scale)(Scalar*,Vec),            /*  x = alpha * x   */
       (*copy)(Vec,Vec),                 /*  y = x */
       (*set)(Scalar*,Vec),              /*  y = alpha  */
       (*swap)(Vec,Vec),                 /* exchange x and y */
       (*axpy)(Scalar*,Vec,Vec),         /*  y = y + alpha * x */
       (*maxpy)(int,Scalar*,Vec,Vec*),   /*   y = y + alpha[j] x[j] */
       (*aypx)(Scalar*,Vec,Vec),         /*  y = x + alpha * y */
       (*waxpy)(Scalar*,Vec,Vec,Vec),    /*  w = y + alpha * x */
       (*pmult)(Vec,Vec,Vec),            /*  w = x .* y */
       (*pdiv)(Vec,Vec,Vec),             /*  w = x ./ y */
       (*setvalues)(Vec,int,int*,Scalar*,InsertMode),
       (*assemblybegin)(Vec),
       (*assemblyend)(Vec),
       (*getarray)(Vec,Scalar**),
       (*getsize)(Vec,int*),(*getlocalsize)(Vec,int*),
       (*getownershiprange)(Vec,int*,int*),
       (*restorearray)(Vec,Scalar**),
       (*max)(Vec,int*,double*),
       (*min)(Vec,int*,double*);          /* z = min(x); idx = index of min(x) */
};

struct _Vec {
  PETSCHEADER
  struct _VeOps ops;
  void          *data;
};

/* 
   These scatters are for purely local.
*/

typedef struct {
  int n,*slots;                /* number of components and their locations */
} VecScatter_General;

typedef struct {
  int n,first,step;           
} VecScatter_Stride;

/*
   This scatter is for a global vector copied (completely) to each processor
*/

typedef struct {
  int    *count;              /* elements of vector on each processor */
  Scalar *work,*work2;        
} VecScatter_MPIToAll;

/*
   This is the parallel scatter
*/
typedef struct { 
  int                n;         /* number of processors to send/receive */
  int                nbelow;    /* number with lower process id */
  int                nself;     /* number sending to self */
  int                *starts;   /* The starting point in indices and values for each proc*/ 
  int                *indices;  /* List of all components sent or received */
  int                *procs;    /* Processors we are communicating with in scatter */
  MPI_Request        *requests;
  Scalar             *values;   /* buffer for all sends or receives */
                                /* note that we pack/unpack ourself,do not use MPI packing */
  VecScatter_General local;     /* any part that happens to be local */
  MPI_Status         *sstatus;
} VecScatter_MPI;

struct _VecScatter {
  PETSCHEADER
  int     inuse;               /* prevents corruption from mixing two scatters */
  int     (*scatterbegin)(Vec,Vec,InsertMode,int,VecScatter);
  int     (*scatterend)(Vec,Vec,InsertMode,int,VecScatter);
  int     (*pipelinebegin)(Vec,Vec,InsertMode,PipelineMode,VecScatter);
  int     (*pipelineend)(Vec,Vec,InsertMode,PipelineMode,VecScatter);
  int     (*copy)(VecScatter,VecScatter);
  void    *fromdata,*todata;
};

/* Default obtain and release vectors; can be used by any implementation */
extern int     Veiobtain_vectors(Vec, int, Vec **);
extern int     Veirelease_vectors(Vec *,int);

#endif
