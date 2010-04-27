#if !defined(__MATIM_H)
#define __MATIM_H

#include "private/matimpl.h"
#include "../src/vec/vec/impls/dvecimpl.h"




typedef struct {
  IS in, out;
  VecScatter scatter;
} Mat_IM;



#endif
