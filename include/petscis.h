
/*
      An index set is essentially a subset of the integers
*/
#if !defined(__IS_PACKAGE)
#define __IS_PACKAGE
#include "petsc.h"

typedef struct _IS* IS;
typedef struct _ISScatterCtx* ISScatterCtx;

extern int    ISCreateSequential(int,int *,IS *);
#if defined(USING_MPI)
extern int    ISCreateParallel(int,int,int *,MPI_Comm,IS *);
#endif
extern int    ISCreateStrideSequential(int,int,int,IS *);
extern int    ISCreateRangeSequential(int,int,int,IS *);

extern int   ISSetPermutation(IS);
extern int   ISIsPermutation(IS); 
extern int   ISGetIndices(IS,int **);
extern int   ISRestoreIndices(IS,int **);
extern int   ISGetSize(IS,int *);
extern int   ISGetLocalSize(IS,int *);
extern int   ISDestroy(IS);
extern int   ISInvertPermutation(IS,IS*);
extern int   ISView(IS,Viewer);

extern int   ISSetUpScatterBegin(IS,IS,ISScatterCtx*);
extern int   ISSetUpScatterEnd(IS,IS,ISScatterCtx*);
extern int   ISFreeScatterCtx(ISScatterCtx);

#endif
