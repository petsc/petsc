
/*
      An index set is essentially a subset of the integers
*/
#if !defined(__IS_PACKAGE)
#define __IS_PACKAGE
#include "petsc.h"

typedef struct _IS* IS;

extern int    ISCreateSequential(int,int *,IS *);
#if defined(USING_MPI)
extern int    ISCreateMPI(int,int,int *,MPI_Comm,IS *);
#endif
extern int    ISCreateStrideSequential(int,int,int,IS *);
extern int    ISStrideGetInfo(IS,int *,int*);

extern int   ISSetPermutation(IS);
extern int   ISIsPermutation(IS); 
extern int   ISGetIndices(IS,int **);
extern int   ISRestoreIndices(IS,int **);
extern int   ISGetSize(IS,int *);
extern int   ISGetLocalSize(IS,int *);
extern int   ISDestroy(IS);
extern int   ISInvertPermutation(IS,IS*);
extern int   ISView(IS,Viewer);

#endif
