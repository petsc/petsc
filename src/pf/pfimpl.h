/* $Id: pfimpl.h,v 1.7 2001/01/15 21:49:41 bsmith Exp bsmith $ */

#ifndef _PFIMPL
#define _PFIMPL

#include "petscpf.h"

typedef struct _PFOps *PFOps;
struct _PFOps {
  int          (*apply)(void*,int,PetscScalar*,PetscScalar*);
  int          (*applyvec)(void*,Vec,Vec);
  int          (*destroy)(void*);
  int          (*view)(void*,PetscViewer);
  int          (*setfromoptions)(PF);
};

struct _p_PF {
  PETSCHEADER(struct _PFOps)
  int    dimin,dimout;             /* dimension of input and output spaces */
  void   *data;
};

#endif
