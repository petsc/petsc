/* $Id: tlibfe.h,v 1.3 2001/04/11 07:51:10 buschelm Exp $ */
#ifndef PETScTlibFE_h
#define PETScTlibFE_h

#include "archiverfe.h"

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
