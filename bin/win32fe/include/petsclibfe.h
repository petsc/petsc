/* $Id: petsclibfe.h,v 1.5 2001/04/17 21:11:14 buschelm Exp $ */
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
  };

}

#endif
