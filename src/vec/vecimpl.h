/* $Id: vecimpl.h,v 1.13 1995/07/17 03:53:25 bsmith Exp bsmith $ */
/* 
   This should not be included in users code.
*/

#ifndef __VECIMPL 
#define __VECIMPL
#include "vec.h"

struct _VeOps {
  int  (*duplicate)(Vec,Vec*),       /*  Get single vector */
       (*getvecs)(Vec,int,Vec**), /*  Get array of vectors */
       (*freevecs)(Vec*,int),     /* Free array of vectors */
       (*dot)(Vec,Vec,Scalar*),          /* z = x^H * y */
       (*mdot)(int,Vec,Vec*,Scalar*),    /*   z[j] = x dot y[j] */
       (*norm)(Vec,double*),             /* z = sqrt(x^H * x) */
       (*amax)(Vec,int*,double*),  /* z = max(|x|); idx = index of max(|x|) */
       (*asum)(Vec,double*),             /*  z = sum |x| */
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
       (*min)(Vec,int*,double*);  /* z = min(x); idx = index of min(x) */
};

struct _Vec {
  PETSCHEADER
  struct _VeOps *ops;
  void          *data;
};

/* 
   These scatters are for purely local.
*/

typedef struct {
  int n,*slots;
} VecScatterGeneral;

typedef struct {
  int n,first,step;
} VecScatterStride;

/*
   This is the parallel scatter
*/
typedef struct { 
  int               n,nbelow,nself;  /* number of processors */
  int               *starts,*indices,*procs;
  MPI_Request       *requests;
  Scalar            *values;
  VecScatterGeneral local;
} VecScatterMPI;

struct _VecScatterCtx {
  PETSCHEADER
  int     inuse;
  int     (*scatterbegin)(Vec,Vec,InsertMode,int,VecScatterCtx);
  int     (*scatterend)(Vec,Vec,InsertMode,int,VecScatterCtx);
  int     (*pipelinebegin)(Vec,Vec,InsertMode,PipelineMode,VecScatterCtx);
  int     (*pipelineend)(Vec,Vec,InsertMode,PipelineMode,VecScatterCtx);
  int     (*copy)(VecScatterCtx,VecScatterCtx);
  void    *fromdata,*todata;
};

/* Default obtain and release vectors; can be used by any implementation */
extern int     Veiobtain_vectors(Vec, int, Vec **);
extern int     Veirelease_vectors(Vec *,int);

#endif
