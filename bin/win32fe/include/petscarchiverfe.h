/* $Id$ */
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

    list<string> archivearg;
    list<string> file;
  };

}

#endif
