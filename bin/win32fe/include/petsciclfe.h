/* $Id:$ */
#ifndef PETScIclFE_h
#define PETScIclFE_h

#include "petscclfe.h"

namespace PETScFE {

  class icl : public cl {
  public:
    icl () {}
    ~icl() {}
    virtual void Parse(void);
  protected:
    virtual void Help(void);

    virtual void FindInstallation(void);
    virtual void AddSystemInclude(void);
    virtual void AddSystemLib(void);
  };

  class ifl : public icl {
  public:
    ifl () {}
    ~ifl() {}
  protected:
    virtual void Help(void);
  };

}

#endif

