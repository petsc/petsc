/* $Id: aoimpl.h,v 1.4 1997/09/20 23:57:14 bsmith Exp bsmith $ */
/* 
   This private file should not be included in users' code.
*/

#ifndef __AOIMPL 
#define __AOIMPL
#include "ao.h"

/*
    Defines the abstract AO operations
*/
struct _AOOps {
  int  (*petsctoapplication)(AO,int,int*),  /* map a set of integers to application order */
       (*applicationtopetsc)(AO,int,int*);   
};

struct _p_AO {
  PETSCHEADER                            /* general PETSc header */
  struct _AOOps ops;                     /* AO operations */
  void          *data;                   /* implementation-specific data */
  int           N, n;                    /* global, local vector size */
};

/*
    Defines the abstract AOData operations
*/
struct _AODataOps {
  int (*add)(AOData,char *,int,int,int*,void*,PetscDataType);
  int (*get)(AOData,char *,int,int*,void**);
  int (*restore)(AOData,char *,int,int*,void**);
};

typedef struct {
  void              *data;                   /* implementation-specific data */
  char              *name;
  int               N;                       /* global size of data*/
  int               bs;                      /* block size of basic chunk */
  PetscDataType     datatype;                /* type of data item, int, double etc */
} AODataSegment;

struct _p_AOData {
  PETSCHEADER                                /* general PETSc header */
  struct _AODataOps ops;                     /* AOData operations */
  int               nsegments;               /* number of items allocated for */
  int               nc;                      /* current number of items */
  AODataSegment     *segments;
  void              *data;
};

#endif
