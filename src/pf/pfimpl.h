/* $Id: pfimpl.h,v 1.5 2000/05/05 22:20:08 balay Exp bsmith $ */

#ifndef _PFIMPL
#define _PFIMPL

#include "petscpf.h"

typedef struct _PFOps *PFOps;
struct _PFOps {
  int          (*apply)(void*,int,Scalar*,Scalar*);
  int          (*applyvec)(void*,Vec,Vec);
  int          (*destroy)(void*);
  int          (*view)(void*,Viewer);
  int          (*setfromoptions)(PF);
};

struct _p_PF {
  PETSCHEADER(struct _PFOps)
  int    dimin,dimout;             /* dimension of input and output spaces */
  void   *data;
};

#endif
