/* $Id: petscclfe.h,v 1.1 2001/03/06 23:57:40 buschelm Exp $ */
#ifndef PETScClFE_h
#define PETScClFE_h

#include <map>
#include "petscfe.h"

namespace PETScFE {
  class cl : public compiler {
  public:
    cl();
    ~cl() {}
    virtual void Parse(void);
    virtual void GetArgs(int argc,char *argv[]);
  protected:
    virtual void Compile(void);

    virtual void FoundD(LI &);
    virtual void Foundl(LI &);
    virtual void Foundo(LI &);
    virtual void FixFx(void);

    LI OutputFlag;
  };

  class df : public cl {
  protected:
    virtual void Foundo(LI &);
  };
}

#endif

