
#if !defined(_SLESIMPL_H)
#define _SLESIMPL_H
#include "sles.h"

struct _SLES {
  PETSCHEADER
  char *prefix;
  int setupcalled;
  PC  pc;
  KSP ksp;
};

#endif
