
#if !defined(__ESMF_H)
#define __ESMF_H

#include "petsc.h"
#include "petscf90.h"

typedef struct _p_ESMFBase *ESMFBase;
struct _p_ESMFBase {
  char typename[64];
  int  refcount;
  int  fcomm;
  int threadcount;
};

#endif
