/* $Id: pfimpl.h,v 1.4 2000/04/09 03:11:41 bsmith Exp balay $ */

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
  int          (*printhelp)(PF,char *);
};

struct _p_PF {
  PETSCHEADER(struct _PFOps)
  int    dimin,dimout;             /* dimension of input and output spaces */
  void   *data;
};

#endif
