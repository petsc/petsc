
#if !defined(__MATIMPL)
#define __MATIMPL
#include "ptscimpl.h"
#include "mat.h"

struct _MatOps {
  int       (*insert)(Mat,int,int*,int,int*,Scalar*,InsertMode),
            (*getrow)(Mat,int,int*,int**,Scalar**),
            (*restorerow)(Mat,int,int*,int**,Scalar**),
            (*mult)(Mat,Vec,Vec),(*multadd)(Mat,Vec,Vec,Vec),
            (*multtrans)(Mat,Vec,Vec),(*multtransadd)(Mat,Vec,Vec,Vec),
            (*solve)(Mat,Vec,Vec),(*solveadd)(Mat,Vec,Vec,Vec),
            (*solvetrans)(Mat,Vec,Vec),(*solvetransadd)(Mat,Vec,Vec,Vec),
            (*lufactor)(Mat,IS,IS),(*chfactor)(Mat,IS),
            (*relax)(Mat,Vec,double,int,double,int,Vec),
            (*trans)(Mat),
            (*info)(Mat,int,int*,int*,int*),(*equal)(Mat,Mat),
            (*copy)(Mat,Mat*),
            (*getdiag)(Mat,Vec),(*scale)(Mat,Vec,Vec),(*norm)(Mat,int,double*),
            (*bassembly)(Mat,int),(*eassembly)(Mat,int),(*compress)(Mat),
            (*insopt)(Mat,int),(*zeroentries)(Mat),
            (*zerorow)(Mat,IS,Scalar *),
            (*order)(Mat,int,IS*,IS*),
            (*lufactorsymbolic)(Mat,IS,IS,Mat *),
            (*lufactornumeric)(Mat,Mat* ),
            (*chfactorsymbolic)(Mat,IS,Mat *),
            (*chfactornumeric)(Mat,Mat* ),
            (*size)(Mat,int*,int*),(*lsize)(Mat,int*,int*),
            (*range)(Mat,int*,int*),
            (*ilufactorsymbolic)(Mat,IS,IS,int,Mat *),
            (*ichfactorsymbolic)(Mat,IS,int,Mat *),
            (*getarray)(Mat,Scalar **),
            (*convert)(Mat,MATTYPE,Mat *);
};


#define FACTOR_LU       1
#define FACTOR_CHOLESKY 2

struct _Mat {
  PETSCHEADER
  struct _MatOps *ops;
  void           *data;
  int            factor;   /* 0, FACTOR_LU or FACTOR_CHOLESKY */
  IS             row, col; /* possible row or column mappings */
};

#endif


