/* $Id: petsctlibfe.h,v 1.1 2001/03/06 23:57:40 buschelm Exp $ */
#ifndef PETScTlibFE_h
#define PETScTlibFE_h

#include "petscfe.h"

namespace PETScFE {

  class tlib : public archiver {
  public:
    tlib() {}
    ~tlib() {}
  protected:
    virtual void Execute(void);
    virtual void FoundFile(LI &);
    virtual void FoundFlag(LI &);
  };

}

#endif
