/* $Id: petsctoolfe.h,v 1.10 2001/05/05 02:16:52 buschelm Exp buschelm $ */
#ifndef PETScToolFE_h_
#define PETScToolFE_h_

#include "petscfe.h"

#define UNKNOWN '*'

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
    void FoundNoDetect(LI &);
    void FoundPath(LI &);
    void FoundUse(LI &);
    void FoundV(LI &);
    void FoundWoff(LI &);
    void FoundUnknown(LI &);

    virtual void FoundFile(LI &);
    virtual void AddSystemInfo(void);
    virtual void FindInstallation(void);
    virtual void AddPaths(void);
    virtual void DisplayVersion(void);
    virtual bool IsAKnownTool(void);

    void PrintListString(list<string> &);
    int GetShortPath(string &);
    int GetLongPath(string &,string &);
    virtual void ProtectQuotes(string &);
    virtual void ReplaceSlashWithBackslash(string &);
    virtual void Merge(string &,list<string> &,LI &);

    list<string> arg;
    list<string> file;
    string InstallDir;
    string version_string;
    bool detect;
    bool inpath;
    bool helpfound;
    bool verbose;
    bool versionfound;
    bool woff;
  private:
    string OptionTags;
    typedef void (PETScFE::tool::*ptm)(LI &);
    map<char,ptm> Options;
  };

}
#endif
