
/*
      An index set is essentially a subset of the integers
*/
#if !defined(__IS_PACKAGE)
#define __IS_PACKAGE
#include "petsc.h"

typedef struct _IS* IS;

int    ISCreateSequential       ANSI_ARGS((int,int *,IS *));
int    ISCreateStrideSequential ANSI_ARGS((int,int,int,IS *));
int    ISCreateRangeSequential  ANSI_ARGS((int,int,int,IS *));

#if defined(MPI_COMPONENT)
int    ISCreateMPI              ANSI_ARGS((void *,int,int *,IS *)); 
#endif

int   ISGetIndices             ANSI_ARGS((IS,int **));
int   ISRestoreIndices         ANSI_ARGS((IS,int **));
int   ISGetSize                ANSI_ARGS((IS,int *));
int   ISGetLocalSize           ANSI_ARGS((IS,int *));
int   ISGetPosition            ANSI_ARGS((IS,int,int *));
int   ISDestroy                ANSI_ARGS((IS));

#endif
