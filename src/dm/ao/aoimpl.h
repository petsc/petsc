/* $Id: aoimpl.h,v 1.1 1996/06/25 19:17:36 bsmith Exp bsmith $ */
/* 
   This private file should not be included in users' code.
*/

#ifndef __AOIMPL 
#define __AOIMPL
#include "ao.h"

/* vector operations */
struct _AOOps {
  int  (*petsctoapplication)(AO,int,int*),
       (*applicationtopetsc)(AO,int,int*);   
};

struct _AO {
  PETSCHEADER                            /* general PETSc header */
  struct _AOOps ops;                     /* AO operations */
  void          *data;                   /* implementation-specific data */
  int           N, n;                    /* global, local vector size */
};


#endif
