/* $Id: matimpl.h,v 1.20 1995/07/06 17:19:25 bsmith Exp bsmith $ */

#if !defined(__MATIMPL)
#define __MATIMPL
#include "ptscimpl.h"
#include "mat.h"

struct _MatOps {
  int       (*setvalues)(Mat,int,int*,int,int*,Scalar*,InsertMode),
            (*getrow)(Mat,int,int*,int**,Scalar**),
            (*restorerow)(Mat,int,int*,int**,Scalar**),
            (*mult)(Mat,Vec,Vec),(*multadd)(Mat,Vec,Vec,Vec),
            (*multtrans)(Mat,Vec,Vec),(*multtransadd)(Mat,Vec,Vec,Vec),
            (*solve)(Mat,Vec,Vec),(*solveadd)(Mat,Vec,Vec,Vec),
            (*solvetrans)(Mat,Vec,Vec),(*solvetransadd)(Mat,Vec,Vec,Vec),
            (*lufactor)(Mat,IS,IS,double),(*choleskyfactor)(Mat,IS,double),
            (*relax)(Mat,Vec,double,MatSORType,double,int,Vec),
            (*transpose)(Mat,Mat*),
            (*getinfo)(Mat,MatInfoType,int*,int*,int*),(*equal)(Mat,Mat),
            (*getdiagonal)(Mat,Vec),(*scale)(Mat,Vec,Vec),
            (*norm)(Mat,int,double*),
            (*assemblybegin)(Mat,MatAssemblyType),
            (*assemblyend)(Mat,MatAssemblyType),(*compress)(Mat),
            (*setoption)(Mat,MatOption),(*zeroentries)(Mat),
            (*zerorows)(Mat,IS,Scalar *),
            (*getreordering)(Mat,int,IS*,IS*),
            (*lufactorsymbolic)(Mat,IS,IS,double,Mat *),
            (*lufactornumeric)(Mat,Mat* ),
            (*choleskyfactorsymbolic)(Mat,IS,double,Mat *),
            (*choleskyfactornumeric)(Mat,Mat* ),
            (*getsize)(Mat,int*,int*),(*getlocalsize)(Mat,int*,int*),
            (*getownershiprange)(Mat,int*,int*),
            (*ilufactorsymbolic)(Mat,IS,IS,double,int,Mat *),
            (*incompletecholeskyfactorsymbolic)(Mat,IS,double,int,Mat *),
            (*getarray)(Mat,Scalar **),(*restorearray)(Mat,Scalar **),
            (*convert)(Mat,MatType,Mat *),
            (*getsubmatrix)(Mat,IS,IS,Mat*),
            (*getsubmatrixinplace)(Mat,IS,IS),
            (*copyprivate)(Mat,Mat *);
};

#define FACTOR_LU       1
#define FACTOR_CHOLESKY 2

struct _Mat {
  PETSCHEADER
  struct _MatOps *ops;
  void           *data;
  int            factor;   /* 0, FACTOR_LU or FACTOR_CHOLESKY */
};

/* Since most (all?) of the parallel matrix assemblies use this stashing,
   we move it to a common place. Perhaps it ultimately belongs elsewhere. */

typedef struct {
  int    nmax, n, *idx, *idy; 
  Scalar *array;
} Stash;

extern int StashValues_Private(Stash*,int,int,int*,Scalar*,InsertMode);
extern int StashInitialize_Private(Stash*);
extern int StashBuild_Private(Stash*);
extern int StashDestroy_Private(Stash*);

typedef struct { 
  int               n;  /* number of processors */
  int               *starts,*rows,*procs;
  MPI_Request       *requests;
  Scalar            *values;
} MatScatterCtx_MPISendRows;

typedef struct { 
  int               n;  /* number of processors */
  int               *procs;
  MPI_Request       *requests;
  Scalar            *values;
} MatScatterCtx_MPIRecvRows;

struct _MatScatterCtx {
  PETSCHEADER
  int              inuse;
  int              (*begin)(Mat,Mat,InsertMode,MatScatterCtx);
  int              (*end)(Mat,Mat,InsertMode,MatScatterCtx);
  void             *send,*receive;
};

#endif


