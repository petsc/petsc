/* $Id:$ */
#ifndef PETScBccFE_h
#define PETScBccFE_h

#include <map>
#include "petscfe.h"

namespace PETScFE {
  class bcc : public compiler {
  public:
    bcc();
    ~bcc() {}
    void GetArgs(int,char *[]);
    void Parse(void);
  protected:
    void Compile(void);
    void Link(void);

    void FoundI(int &,string);
    void FoundL(int &,string);
    void Foundl(int &,string);
    void Foundo(int &,string);
  private:
    void FixOutput(void);
    typedef void (PETScFE::bcc::*ptm)(int &,string);
    map<char,ptm> Options;
    int OutputFlag;
  };

}

#endif
