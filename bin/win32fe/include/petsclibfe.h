/* $Id: petsclibfe.h,v 1.2 2001/03/23 19:33:28 buschelm Exp $ */
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
