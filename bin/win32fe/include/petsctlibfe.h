/* $Id: petsctlibfe.h,v 1.5 2001/04/17 21:10:46 buschelm Exp $ */
#ifndef PETScTlibFE_h
#define PETScTlibFE_h

#include "petscarchiverfe.h"

namespace PETScFE {

  class tlib : public archiver {
  public:
    tlib() {}
    ~tlib() {}
    virtual void Execute(void);
  protected:
    virtual void Help(void);
    virtual void FoundFlag(LI &);
  };

}

#endif
