/* $Id:$ */
#ifndef PETScTlibFE_h
#define PETScTlibFE_h

#include "petscfe.h"

namespace PETScFE {

  class tlib : public archiver {
  public:
    tlib() {}
    ~tlib() {}
  protected:
    void FoundFile(int &,string);
    void FoundFlag(int &,string);
  };

}

#endif
