/* $Id: aoimpl.h,v 1.3 1997/05/23 16:04:46 balay Exp bsmith $ */
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
  int joe;
};

struct _p_AOData {
  PETSCHEADER                                /* general PETSc header */
  struct _AODataOps ops;                     /* AOData operations */
  void              *data;                   /* implementation-specific data */
  int               N;                       /* global size of data*/
  int               bs;                      /* block size of basic chunk */
  PetscDataType     datatype;                /* type of data item, int, double etc */
};

#endif
