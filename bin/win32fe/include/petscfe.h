/* $Id:$ */
#ifndef PETScFE_h_
#define PETScFE_h_

#include <vector>
#include <string>
#include <map>

using namespace std;

namespace PETScFE {

  class tool {
  public:
    static void Create(tool*&,char *);
    virtual void Parse(void) {}
    virtual void Parse(int,char*[]);
    virtual void Execute(void);
    virtual void GetArgs(int argc,char *argv[]);
    virtual void Destroy(void) {delete this;}
  protected:
    tool();
    virtual ~tool() {}
    virtual void ReplaceSlashWithBackslash(string &);
    void PrintStringVector(vector<string> &);
    void Squeeze(vector<string> &);
    void Merge(string &,vector<string> &,int);

    vector<string> arg;
    int quiet;
  private:
    void FoundArg(int &,string);
    void FoundUse(int &,string);
    void FoundVersion(int &,string);
    void FoundQuiet(int &,string);

    string OptionTags;
    typedef void (PETScFE::tool::*ptm)(int &,string);
    map<string,ptm> Options;
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

    virtual void FoundFile(int &,string);
    virtual void FoundD(int &,string);
    virtual void FoundI(int &,string);
    virtual void FoundL(int &,string);
    virtual void Foundc(int &,string);
    virtual void Foundl(int &,string);
    virtual void Foundo(int &,string);
    virtual void FoundUnknown(int &,string);

    void Squeeze(void);
    string OptionTags;

    vector<string> compilearg;
    vector<string> file;
    vector<string> linkarg;
  private:
    typedef void (PETScFE::compiler::*ptm)(int &,string);
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
    virtual void FoundFile(int &,string);
    virtual void FoundFlag(int &,string);

    void Squeeze(void);

    vector<string> archivearg;
    vector<string> file;
  };

}

#endif
