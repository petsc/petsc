/* $Id: petscarchiverfe.h,v 1.5 2001/04/17 21:10:12 buschelm Exp $ */
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
    virtual void Merge(string &,list<string> &,LI &);

    list<string> archivearg;
  };

}

#endif
