#ifndef _MMIMPL
#define _MMIMPL

#include "petsc.h"
#include "mm.h"

/*
   Multi-model context
      Question: Where to put MM_COOKIE??  It doesn't really belong here, but
      we need to stash this someplace for use in testing whether the MM context
      remains valid.
*/
struct _p_MM {
  PETSCHEADER(int)
  int         MM_COOKIE;
  int         setupcalled;
  int         ncomponents;
  int         (*printhelp)(MM,char*);
  int         (*setfromoptions)(MM);
  int         (*destroy)(MM);
  int         (*view)(MM,Viewer);
  void        *data;
};

#endif
