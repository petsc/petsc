/* $Id: petsctlibfe.h,v 1.2 2001/03/23 19:33:28 buschelm Exp $ */
#ifndef PETScTlibFE_h
#define PETScTlibFE_h

#include "petscarchiverfe.h"

namespace PETScFE {

  class tlib : public archiver {
  public:
    tlib() {}
    ~tlib() {}
  protected:
    virtual void Execute(void);
    virtual void Help(void);
    virtual void FoundFile(LI &);
    virtual void FoundFlag(LI &);
  };

}

#endif
