
#if !defined(_SLESIMPL_H)
#define _SLESIMPL_H
#include "ptscimpl.h"
#include "sles.h"

struct _SLES {
  PETSCHEADER
  int setupcalled;
  PC  pc;
  KSP ksp;
};

#endif
