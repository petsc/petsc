#ifndef included_ALE_ALE_hh
#define included_ALE_ALE_hh

#include <petsc.h>

#include <map>
#include <set>
#include <vector>
#include <string>
#include <sstream>

typedef std::basic_ostringstream<char> ostringstream;

namespace ALE {

  class Exception {
    const char *_msg;
  public:
    Exception(const char *msg)    {this->_msg = msg;};
    //Exception(const Exception& e) {this->_msg = e._msg;};
    const char *message() const   {return this->_msg;};
  };

  // A helper function that throws an ALE::Exception with a message identifying the function that returned the given error code, 
  // including the function and the line where the error occured.
  void ERROR(PetscErrorCode ierr, const char *func, int line, const char *msg);
  // A helper function that allocates and assembles an error message from a format string 
  const char *ERRORMSG(const char *fmt, ...);
  // A helper function for converting MPI errors to exception
  void MPIERROR(PetscErrorCode ierr, const char *func, int line, const char *msg);
}// namespace ALE

// A helper macro that passes __FUNCT__ and __LINE__ with the error msg to the ERROR routine
#define CHKERROR(ierr, msg) \
  ERROR(ierr, __FUNCT__,  __LINE__, msg);

// A helper macro that passes __FUNCT__ and __LINE__ with the error msg to the MPIERROR routine
#define CHKMPIERROR(ierr, msg) \
  MPIERROR(ierr, __FUNCT__,  __LINE__, msg);

#include <ALE_mem.hh>
#include <ALE_containers.hh>

namespace ALE {

  // Base class for all distributed ALE classes
  class Coaster {
  protected:
    //
    MPI_Comm                              comm;
    int32_t                               commRank;
    int32_t                               commSize;
    PetscObject                           petscObj;
    int32_t                               verbosity;
    int                                   _lock;
    void __checkLock(){if(this->_lock > 0) {throw(ALE::Exception("Mutating method attempted on a locked Coaster"));}};
  public:
    //
    Coaster() : petscObj(NULL) {this->clear();};
    Coaster(MPI_Comm c) : petscObj(NULL) {this->clear(); this->setComm(c);};
    virtual ~Coaster(){this->clear();};
    //
    virtual Coaster&          clear();
    virtual Coaster&          getLock();
    virtual Coaster&          releaseLock();
    virtual bool              isLocked(){return (this->_lock > 0);};
    virtual void              assertLock(bool status);
    //
    virtual void              setComm(MPI_Comm comm);
    MPI_Comm                  getComm() const{ return this->comm;};
    int32_t                   getCommSize() const {return this->commSize;};
    int32_t                   getCommRank() const {return this->commRank;};
    void                      setVerbosity(int32_t v){this->verbosity = v;};
    int32_t                   getVerbosity() const {return this->verbosity;};
    virtual void              view(const char *name);
    //
    friend void               CHKCOMM(Coaster& obj);
    friend void               CHKCOMMS(Coaster& obj1, Coaster& obj2);
  };
} // namespace ALE

#endif
