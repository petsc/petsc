/* $Id: petscbccfe.h,v 1.4 2001/03/28 21:05:04 buschelm Exp $ */
#ifndef PETScBccFE_h
#define PETScBccFE_h

#include "petsccompilerfe.h"

namespace PETScFE {

  class bcc : public compiler {
  public:
    bcc();
    ~bcc() {}
    virtual void GetArgs(int,char *[]);
    virtual void Parse(void);
  protected:
    virtual void Compile(void);
    virtual void Link(void);
    virtual void Help(void);

    virtual void FoundD(LI &);
    virtual void Foundl(LI &);
    virtual void Foundo(LI &);
    virtual void FixOutput(void);
    LI OutputFlag;
  };

}

#endif
