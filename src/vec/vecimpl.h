/* 
   This should not be included in users code.
*/

#ifndef __VECIMPL 
#define __VECIMPL
#include "ptscimpl.h"
#include "vec.h"

struct _VeOps {
  int  (*create_vector)(Vec,Vec*),       /*  Get single vector */
       (*obtain_vectors)(Vec,int,Vec**), /*  Get array of vectors */
       (*release_vectors)(Vec*,int),     /* Free array of vectors */
       (*dot)(Vec,Vec,Scalar*),          /* z = x^H * y */
       (*mdot)(int,Vec,Vec*,Scalar*),    /*   z[j] = x dot y[j] */
       (*norm)(Vec,double*),             /* z = sqrt(x^H * x) */
       (*amax)(Vec,int*,double*),  /* z = max(|x|); idx = index of max(|x|) */
       (*asum)(Vec,double*),             /*  z = sum |x| */
       (*tdot)(Vec,Vec,Scalar*),         /* x'*y */
       (*mtdot)(int,Vec,Vec*,Scalar*),   /*   z[j] = x dot y[j] */
       (*scal)(Scalar*,Vec),             /*  x = alpha * x   */
       (*copy)(Vec,Vec),                 /*  y = x */
       (*set)(Scalar*,Vec),              /*  y = alpha  */
       (*swap)(Vec,Vec),                 /* exchange x and y */
       (*axpy)(Scalar*,Vec,Vec),         /*  y = y + alpha * x */
       (*maxpy)(int,Scalar*,Vec,Vec*),   /*   y = y + alpha[j] x[j] */
       (*aypx)(Scalar*,Vec,Vec),         /*  y = x + alpha * y */
       (*waxpy)(Scalar*,Vec,Vec,Vec),    /*  w = y + alpha * x */
       (*pmult)(Vec,Vec,Vec),            /*  w = x .* y */
       (*pdiv)(Vec,Vec,Vec),             /*  w = x ./ y */
       (*insertvalues)(Vec,int,int*,Scalar*,InsertMode),
       (*beginassm)(Vec),
       (*endassm)(Vec),
       (*getarray)(Vec,Scalar**),
       (*getsize)(Vec,int*),(*localsize)(Vec,int*),
       (*getrange)(Vec,int*,int*),
       (*restorearray)(Vec,Scalar**),
       (*max)(Vec,int*,double*),
       (*min)(Vec,int*,double*);  /* z = min(x); idx = index of min(x) */
};

/* Vector types, must be powers of 2 */
#define SEQVECTOR               0
#define MPIVECTOR               1

struct _Vec {
  PETSCHEADER
  struct _VeOps *ops;
  void          *data;
};


typedef struct {
  int n,*slots;
} VecScatterGeneral;

typedef struct {
  int n,first,step;
} VecScatterStride;

/*
    I would like to use MPI_Type_indexed() but MPI doesn't seem
  to provide any direct support for ScatterAdd, thus we pack our own instead.
*/
typedef struct { 
  int          n,nbelow,nself;  /* number of processors (including self) */
  int          *starts,*indices,*procs;
  MPI_Request  *requests;
  Scalar       *values;
} VecScatterMPI;

struct _VecScatterCtx {
  PETSCHEADER
  int     (*begin)(Vec,Vec,VecScatterCtx,InsertMode,int);
  int     (*end)(Vec,Vec,VecScatterCtx,InsertMode,int);
  int     (*beginpipe)(Vec,Vec,VecScatterCtx,InsertMode,int);
  int     (*endpipe)(Vec,Vec,VecScatterCtx,InsertMode,int);
  int     (*copy)(VecScatterCtx,VecScatterCtx);
  void    *fromdata,*todata;
};

/* Default obtain and release vectors; can be used by any implementation */
extern int     Veiobtain_vectors(Vec, int, Vec **);
extern int     Veirelease_vectors(Vec *,int);

#endif
