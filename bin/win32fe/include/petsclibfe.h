/* $Id: petsclibfe.h,v 1.1 2001/03/06 23:57:40 buschelm Exp $ */
#ifndef PETScLibFE_h
#define PETScLibFE_h

#include "petscfe.h"

namespace PETScFE {
  class lib : public archiver {
  public:
    lib() {}
    ~lib() {}
    virtual void Execute(void);
  };

}

#endif
