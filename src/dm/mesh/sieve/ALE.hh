#ifndef included_ALE_ALE_hh
#define included_ALE_ALE_hh

#include <petsc.h>

#ifndef  included_ALE_exception_hh
#include <ALE_exception.hh>
#endif
#ifndef  included_ALE_mem_hh
#include <ALE_mem.hh>
#endif
#ifndef  included_ALE_containers_hh
#include <ALE_containers.hh>
#endif

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
