/* $Id: clfe.h,v 1.7 2001/04/11 07:51:10 buschelm Exp $ */
#ifndef PETScClFE_h
#define PETScClFE_h

#include "compilerfe.h"

namespace PETScFE {

  class cl : public compiler {
  public:
    cl();
    ~cl() {}
    virtual void Parse(void);
    virtual void GetArgs(int argc,char *argv[]);
  protected:
    virtual void Compile(void);
    virtual void Help(void);

    virtual void FoundD(LI &);
    virtual void FoundL(LI &);
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

