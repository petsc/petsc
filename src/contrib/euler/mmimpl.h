#ifndef _MMIMPL
#define _MMIMPL

#include "petsc.h"
#include "mm.h"

/*
   Multi-model context
      Question: Where to put MM_COOKIE??  Not here ...
*/
struct _p_MM {
  PETSCHEADER(int)
  int          MM_COOKIE;
  int          ncomponents;
  int          setupcalled;
  int          (*printhelp)(MM,char*),(*setfrom)(MM);
  void         *data;
};

#endif
