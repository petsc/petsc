/* $Id: petsccompilerfe.h,v 1.9 2001/05/05 02:16:52 buschelm Exp buschelm $ */
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

    virtual void AddSystemInfo(void);
    virtual void AddPaths(void) {}
    virtual void AddSystemInclude(void);
    virtual void AddSystemLib(void);
    virtual void DisplayVersion(void);
    virtual bool IsAKnownTool(void);
    virtual void FixOutput(void);

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
