/* $Id: petscfe.h,v 1.1 2001/03/06 23:57:40 buschelm Exp $ */
#ifndef PETScFE_h_
#define PETScFE_h_

#include <list>
#include <string>
#include <map>

using namespace std;
typedef list<string>::iterator LI;

namespace PETScFE {

  class tool {
  public:
    static void Create(tool*&,string);
    virtual void Parse(void);
    virtual void Execute(void);
    virtual void GetArgs(int argc,char *argv[]);
    virtual void Destroy(void) {delete this;}
  protected:
    tool();
    virtual ~tool() {}
    void PrintListString(list<string> &);
    virtual void ProtectQuotes(string &);
    virtual void ReplaceSlashWithBackslash(string &);
    void Merge(string &,list<string> &,LI);

    list<string> arg;
    int verbose;
  private:
    void FoundUse(LI &);
    void FoundVerbose(LI &);
/*      void FoundVersion(LI &); */

    string OptionTags;
    typedef void (PETScFE::tool::*ptm)(LI &);
    map<char,ptm> Options;
  };

  class compiler : public tool {
  public:
    compiler();
    virtual ~compiler() {}
    virtual void Parse(void);
    virtual void Execute(void);
    virtual void GetArgs(int argc,char *argv[]);
  protected:
    virtual void Compile(void);
    virtual void Link(void);

    virtual void FoundFile(LI &);
    virtual void FoundD(LI &);
    virtual void FoundI(LI &);
    virtual void FoundL(LI &);
    virtual void Foundc(LI &);
    virtual void Foundl(LI &);
    virtual void Foundo(LI &);
    virtual void FoundUnknown(LI &);

    string OptionTags;

    list<string> compilearg;
    list<string> file;
    list<string> linkarg;
  private:
    typedef void (PETScFE::compiler::*ptm)(LI &);
    map<char,ptm> Options;
  };

  class archiver : public tool {
  public:
    archiver() {}
    virtual ~archiver() {}
    virtual void Parse(void);
    virtual void Execute(void);
    virtual void GetArgs(int argc,char *argv[]);
  protected:
    virtual void FoundFile(LI &);
    virtual void FoundFlag(LI &);

    list<string> archivearg;
    list<string> file;
  };

}

#endif
