/* $Id: petscbccfe.h,v 1.7 2001/04/17 21:09:54 buschelm Exp $ */
#ifndef PETScBccFE_h
#define PETScBccFE_h

#include "petsccompilerfe.h"

namespace PETScFE {

  class bcc : public compiler {
  public:
    bcc();
    ~bcc() {}
    virtual void Parse(void);
  protected:
    virtual void Link(void);
    virtual void Help(void);

    virtual void Foundl(LI &);
  };

}

#endif
