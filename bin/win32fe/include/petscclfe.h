/* $Id: petscclfe.h,v 1.9 2001/04/17 21:09:36 buschelm Exp $ */
#ifndef PETScClFE_h
#define PETScClFE_h

#include "petsccompilerfe.h"

namespace PETScFE {

  class cl : public compiler {
  public:
    cl();
    ~cl() {}
    virtual void Parse(void);
  protected:
    virtual void Help(void);

    virtual void FoundL(LI &);
  };

}

#endif

