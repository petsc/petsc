/* $Id: petsccompilerfe.h,v 1.8 2001/05/04 21:30:26 buschelm Exp $ */
#ifndef PETScCompilerFE_h_
#define PETScCompilerFE_h_

#include "petsctoolfe.h"

namespace PETScFE {

  class compiler : public tool {
  public:
    compiler();
    virtual ~compiler() {}
    virtual void Parse(void);
    virtual void Execute(void);
  protected:
    virtual void Compile(void);
    virtual void Link(void);
    virtual void Help(void);

    virtual void FoundD(LI &);
    virtual void FoundI(LI &);
    virtual void FoundL(LI &);
    virtual void Foundh(LI &);
    virtual void Foundc(LI &);
    virtual void Foundl(LI &);
    virtual void Foundo(LI &);
    virtual void FoundUnknown(LI &);

    virtual void FixOutput(void);
    virtual void AddSystemInfo(void);
    virtual void AddPaths(void) {}
    virtual void AddSystemInclude(void);
    virtual void AddSystemLib(void);

    string OptionTags;
    list<string> compilearg;
    list<string> linkarg;

    string compileoutflag;
    string linkoutflag;
    LI OutputFlag;
  private:
    typedef void (PETScFE::compiler::*ptm)(LI &);
    map<char,ptm> Options;
  };

}

#endif
