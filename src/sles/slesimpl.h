
#if !defined(_SLESIMPL_H)
#define _SLESIMPL_H
#include "ptscimpl.h"
#include "sles.h"

#define SLES_COOKIE 0x70707070

struct _SLES {
  PETSCHEADER
  Mat mat;
  PC  pc;
  KSP ksp;
};

#endif
