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

    virtual void FindInstallation(void);
    virtual void AddPaths(void);

    string VisualStudioDir;
    string VSVersion;
  };

  class df : public cl {
  public:
    df () {}
    ~df() {}
  protected:
    virtual void Help(void);
    virtual void AddPaths(void);
    virtual void AddSystemLib(void);
  };

}

#endif

