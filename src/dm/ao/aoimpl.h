/* $Id: aoimpl.h,v 1.2 1996/07/02 18:09:18 bsmith Exp balay $ */
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

struct _p_AO {
  PETSCHEADER                            /* general PETSc header */
  struct _AOOps ops;                     /* AO operations */
  void          *data;                   /* implementation-specific data */
  int           N, n;                    /* global, local vector size */
};


#endif
