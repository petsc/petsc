/* $Id: is.h,v 1.16 1995/10/11 17:59:04 curfman Exp bsmith $ */

/*
      An index set is essentially a subset of the integers
*/
#if !defined(__IS_PACKAGE)
#define __IS_PACKAGE
#include "petsc.h"

#define IS_COOKIE PETSC_COOKIE+2

typedef struct _IS* IS;

extern int   ISCreateSeq(MPI_Comm,int,int *,IS *);
extern int   ISCreateStrideSeq(MPI_Comm,int,int,int,IS *);
extern int   ISAddStrideSeq(IS*,int,int,int);
extern int   ISStrideGetInfo(IS,int *,int*);

extern int   ISSetPermutation(IS);
extern int   ISIsPermutation(IS); 
extern int   ISSetIdentity(IS);
extern int   ISIsIdentity(IS);

extern int   ISGetIndices(IS,int **);
extern int   ISRestoreIndices(IS,int **);
extern int   ISGetSize(IS,int *);
extern int   ISGetLocalSize(IS,int *);
extern int   ISDestroy(IS);
extern int   ISInvertPermutation(IS,IS*);
extern int   ISView(IS,Viewer);

typedef enum {IS_SEQ=0, IS_STRIDE_SEQ=2} IndexSetType;

#endif
