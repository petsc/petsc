/* $Id: matimpl.h,v 1.27 1995/09/21 20:10:03 bsmith Exp bsmith $ */

#if !defined(__MATIMPL)
#define __MATIMPL
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
            (*norm)(Mat,MatNormType,double*),
            (*assemblybegin)(Mat,MatAssemblyType),
            (*assemblyend)(Mat,MatAssemblyType),(*compress)(Mat),
            (*setoption)(Mat,MatOption),(*zeroentries)(Mat),
            (*zerorows)(Mat,IS,Scalar *),
            (*getreordering)(Mat,MatOrdering,IS*,IS*),
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
            (*copyprivate)(Mat,Mat *),
            (*forwardsolve)(Mat,Vec,Vec),(*backwardsolve)(Mat,Vec,Vec),
            (*ilufactor)(Mat,IS,IS,double),
            (*incompletecholeskyfactor)(Mat,IS,double);
};

/*   
     Each matrix has to know how to set up its own (matrix specific) preconditioners
     Note that this introduces a loop in the "inheritence" tree. 
*/
#include "pc.h"

struct _PCSetUps  {
  int (*icc)(PC);
  int (*ilu)(PC);
  int (*bjacobi)(PC);
};

#define FACTOR_LU       1
#define FACTOR_CHOLESKY 2

struct _Mat {
  PETSCHEADER
  struct _MatOps   ops;
  void             *data;
  int              factor;   /* 0, FACTOR_LU or FACTOR_CHOLESKY */
  double           lupivotthreshold;
  struct _PCSetUps pcsetups;
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

/*
   Does reorderings for sequential IJ format. By default uses 
  SparsePak routines.
*/
extern int MatGetReordering_IJ(int,int*,int*,MatOrdering,IS *,IS*);

#endif


