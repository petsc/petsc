/* $Id: petscarchiverfe.h,v 1.6 2001/05/03 11:03:46 buschelm Exp buschelm $ */
#ifndef PETScArchiverFE_h_
#define PETScArchiverFE_h_

#include "petsctoolfe.h"

namespace PETScFE {

  class archiver : public tool {
  public:
    archiver() {}
    virtual ~archiver() {}
    virtual void Parse(void);
    virtual void Execute(void);
  protected:
    virtual void Archive(void);
    virtual void Help(void);

    virtual void FoundFlag(LI &);
    virtual void DisplayVersion(void);
    virtual bool IsAKnownTool(void);
    virtual void Merge(string &,list<string> &,LI &);

    list<string> archivearg;
  };

}

#endif
