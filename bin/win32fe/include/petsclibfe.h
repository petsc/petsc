/* $Id:$ */
#ifndef PETScLibFE_h
#define PETScLibFE_h

#include "petscfe.h"

namespace PETScFE {
  class lib : public archiver {
  public:
    lib() {}
    ~lib() {}
    void Execute(void);
  };

}

#endif
