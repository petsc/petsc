
/*
      An index set is essentially a subset of the integers
*/
#if !defined(__IS_PACKAGE)
#define __IS_PACKAGE
#include "petsc.h"

typedef struct _IS* IS;

int    ISCreateSequential(int,int *,IS *);
int    ISCreateSequentialPermutation(int,int *,IS *);
int    ISCreateStrideSequential(int,int,int,IS *);
int    ISCreateRangeSequential(int,int,int,IS *);

int   ISGetIndices(IS,int **);
int   ISRestoreIndices(IS,int **);
int   ISGetSize(IS,int *);
int   ISGetLocalSize(IS,int *);
int   ISGetPosition(IS,int,int *);
int   ISDestroy(IS);
int   ISIsPermutation(IS); 
int   ISInvertPermutation(IS,IS*);
int   ISView(IS,Viewer);

#endif
