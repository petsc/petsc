/* $Id: compilerfe.h,v 1.2 2001/04/17 20:52:49 buschelm Exp buschelm $ */
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
    virtual void GetArgs(int argc,char *argv[]);
  protected:
    virtual void Compile(void);
    virtual void Link(void);
    virtual void Help(void);

    virtual void FoundFile(LI &);
    virtual void FoundD(LI &);
    virtual void FoundI(LI &);
    virtual void FoundL(LI &);
    virtual void Foundhelp(LI &);
    virtual void Foundc(LI &);
    virtual void Foundl(LI &);
    virtual void Foundo(LI &);
    virtual void FoundUnknown(LI &);

    string OptionTags;

    list<string> compilearg;
    list<string> file;
    list<string> linkarg;
  private:
    typedef void (PETScFE::compiler::*ptm)(LI &);
    map<char,ptm> Options;
  };

}

#endif
