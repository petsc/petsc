/* $Id: petsctlibfe.h,v 1.6 2001/05/03 11:03:46 buschelm Exp $ */
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
    virtual void FoundFile(LI &);
  };

}

#endif
