/* $Id: petsclibfe.h,v 1.6 2001/05/03 11:03:46 buschelm Exp $ */
#ifndef PETScLibFE_h
#define PETScLibFE_h

#include "petscarchiverfe.h"

namespace PETScFE {
  class lib : public archiver {
  public:
    lib() {}
    ~lib() {}
    virtual void Parse(void);
    virtual void Execute(void);
  protected:
    virtual void Archive(void);
    virtual void Help(void);

    virtual void FindInstallation(void);
    virtual void AddPaths(void);

    string VisualStudioDir;
    string VSVersion;
  };

}

#endif
