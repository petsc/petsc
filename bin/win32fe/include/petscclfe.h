/* $Id:$ */
#ifndef PETScClFE_h
#define PETScClFE_h

#include <map>
#include "petscfe.h"

namespace PETScFE {
  class cl : public compiler {
  public:
    cl();
    ~cl() {}
    void Parse(void);
    void GetArgs(int argc,char *argv[]);
  protected:
    void Compile(void);

    void FoundD(int &,string);
    void FoundL(int &,string);
    void Foundl(int &,string);
    void Foundo(int &,string);
  private:
    void FixFx(void);
    typedef void (PETScFE::cl::*ptm)(int &,string);
    map<char,ptm> Options;
    int OutputFlag;
  };

}

#endif

