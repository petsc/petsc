/* $Id: libfe.h,v 1.3 2001/04/11 07:51:10 buschelm Exp $ */
#ifndef PETScLibFE_h
#define PETScLibFE_h

#include "archiverfe.h"

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
