/* $Id: is.h,v 1.18 1995/11/19 00:50:48 bsmith Exp bsmith $ */

/*
      An index set is essentially a subset of the integers. They are used
   for defining scatters and gathers.
*/
#if !defined(__IS_PACKAGE)
#define __IS_PACKAGE
#include "petsc.h"

typedef enum {IS_SEQ=0, IS_STRIDE_SEQ=2} IndexSetType;

#define IS_COOKIE PETSC_COOKIE+2

typedef struct _IS* IS;

extern int   ISCreateSeq(MPI_Comm,int,int *,IS *);
extern int   ISCreateStrideSeq(MPI_Comm,int,int,int,IS *);

extern int   ISDestroy(IS);

extern int   ISStrideGetInfo(IS,int *,int*);

extern int   ISSetPermutation(IS);
extern int   ISIsPermutation(IS); 
extern int   ISSetIdentity(IS);
extern int   ISIsIdentity(IS);

extern int   ISGetIndices(IS,int **);
extern int   ISRestoreIndices(IS,int **);
extern int   ISGetSize(IS,int *);
extern int   ISGetLocalSize(IS,int *);
extern int   ISInvertPermutation(IS,IS*);
extern int   ISView(IS,Viewer);

#endif
