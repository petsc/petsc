/* $Id: libfe.h,v 1.4 2001/04/17 20:52:49 buschelm Exp buschelm $ */
#ifndef PETScLibFE_h
#define PETScLibFE_h

#include "petscarchiverfe.h"

namespace PETScFE {
  class lib : public archiver {
  public:
    lib() {}
    ~lib() {}
    virtual void Execute(void);
    virtual void Help(void);
  };

}

#endif
