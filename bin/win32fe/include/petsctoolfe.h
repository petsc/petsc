/* $Id: petsctoolfe.h,v 1.9 2001/05/04 02:40:31 buschelm Exp $ */
#ifndef PETScToolFE_h_
#define PETScToolFE_h_

#include "petscfe.h"

namespace PETScFE {

  class tool {
  public:
    static int Create(tool*&,string);
    void GetArgs(int argc,char *argv[]);
    virtual void Parse(void);
    virtual void Execute(void);
    virtual void Destroy(void) {delete this;}
  protected:
    tool();
    virtual ~tool() {}
    virtual void Help(void);

    void FoundHelp(LI &);
    void FoundPath(LI &);
    void FoundUse(LI &);
    void FoundVerbose(LI &);

    virtual void FoundFile(LI &);
    virtual void AddSystemInfo(void);
    virtual void FindInstallation(void);
    virtual void AddPaths(void);

    void PrintListString(list<string> &);
    int GetShortPath(string &);
    virtual void ProtectQuotes(string &);
    virtual void ReplaceSlashWithBackslash(string &);
    virtual void Merge(string &,list<string> &,LI &);

    list<string> arg;
    list<string> file;
    string InstallDir;
    bool verbose;
    bool helpfound;
    bool inpath;
  private:
    string OptionTags;
    typedef void (PETScFE::tool::*ptm)(LI &);
    map<char,ptm> Options;
  };

}
#endif
