/* $Id: petscarchiverfe.h,v 1.2 2001/04/16 22:12:48 buschelm Exp buschelm $ */
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
