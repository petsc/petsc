
#if !defined(_SLESIMPL_H)
#define _SLESIMPL_H
#include "sles.h"

struct _p_SLES {
  PETSCHEADER(int dummy)
  int setupcalled;
  PC  pc;
  KSP ksp;
};

#endif
