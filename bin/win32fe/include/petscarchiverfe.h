/* $Id: archiverfe.h,v 1.3 2001/04/16 22:17:30 buschelm Exp $ */
#ifndef PETScArchiverFE_h_
#define PETScArchiverFE_h_

#include "toolfe.h"

namespace PETScFE {

  class archiver : public tool {
  public:
    archiver() {}
    virtual ~archiver() {}
    virtual void Parse(void);
    virtual void Execute(void);
    virtual void GetArgs(int argc,char *argv[]);
  protected:
    virtual void Help(void);
    virtual void FoundFile(LI &);
    virtual void FoundFlag(LI &);
    virtual void Merge(string &,list<string> &,LI &);

    list<string> archivearg;
    list<string> file;
  };

}

#endif
