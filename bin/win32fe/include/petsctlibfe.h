/* $Id: tlibfe.h,v 1.4 2001/04/17 20:52:49 buschelm Exp buschelm $ */
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
