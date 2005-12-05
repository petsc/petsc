#define ALE_Coaster_cxx

#include <Coaster.hh>

namespace ALE {

  #undef  __FUNCT__
  #define __FUNCT__ "Coaster::Coaster"
  Coaster::Coaster(const Coaster &c) {
    this->comm       = c.comm;
    this->commRank   = c.commRank;
    this->commSize   = c.commSize;
    this->verbosity  = c.verbosity;
    this->_lock      = 0;
    if(this->petscObj != NULL) {
      PetscErrorCode ierr;
      ierr = PetscObjectReference(this->petscObj); CHKERROR(ierr, "Failed on PetscObjectReference");
    }
  }// Coaster::Coaster()


  #undef  __FUNCT__
  #define __FUNCT__ "Coaster::setComm"
  void Coaster::setComm(MPI_Comm c) {
    if (this->comm == c) {
      return;
    }
    if ((this->comm != MPI_COMM_NULL) && (this->comm != c)) {
      throw ALE::Exception("Cannot reset the communicator");
    }
    PetscErrorCode ierr;
    this->comm = c;
    ierr = MPI_Comm_rank(this->comm, &this->commRank); CHKERROR(ierr, "Error in MPI_Comm_rank");
    ierr = MPI_Comm_size(this->comm, &this->commSize); CHKERROR(ierr, "Error in MPI_Comm_rank"); 
    ierr = PetscObjectCreate(this->comm, &this->petscObj); CHKERROR(ierr, "Failed on PetscObjectCreate");
  }// Coaster::setComm()
  
  #undef  __FUNCT__
  #define __FUNCT__ "Coaster::clear"
  Coaster& Coaster::clear(){
    this->comm     = MPI_COMM_NULL; 
    this->commRank = -1; 
    this->commSize = 0; 
    if(this->petscObj != NULL) {
      PetscErrorCode ierr;
      ierr = PetscObjectDestroy(this->petscObj);  CHKERROR(ierr, "Error in PetscObjectDestroy");
    }
    this->petscObj = NULL;
    this->verbosity = 0;
    this->_lock = 0;
    return *this;
  }// Coaster::clear()

  #undef  __FUNCT__
  #define __FUNCT__ "Coaster::getLock"
  Coaster&  Coaster::getLock(){
    CHKCOMM(*this);
    this->_lock++;
    PetscErrorCode ierr = MPI_Barrier(this->getComm()); CHKERROR(ierr, "Error in MPI_Barrier");
    return *this;
  }//Coaster::getLock()
  
  #undef  __FUNCT__
  #define __FUNCT__ "Coaster::releaseLock"
  Coaster&  Coaster::releaseLock(){
    CHKCOMM(*this);
    if(this->_lock == 0){
      throw ALE::Exception("Cannot release non-existent lock");
    }
    PetscErrorCode ierr = MPI_Barrier(this->getComm()); CHKERROR(ierr, "Error in MPI_Barrier");      
    this->_lock--;
    return *this;
  }//Coaster::releaseLock()

  #undef  __FUNCT__
  #define __FUNCT__ "Coaster::assertLock"
  void    Coaster::assertLock(bool status){
    if(status != this->isLocked()) {
      throw Exception("Lock status assertion failed");
    }
  }// Coaster::assertLock()


  #undef  __FUNCT__
  #define __FUNCT__ "Coaster::view"
  void Coaster::view(const char *name){}


  // A helper friend function that checks whether the communicator is set on an ALE::Coaster object,
  // and throws an exception otherwise.
  void CHKCOMM(Coaster& obj) {
    if(obj.comm == MPI_COMM_NULL){     
      throw ALE::Exception("ALE: Communicator not set");
    }
  }// CHKCOMM()

  // A helper friend function that checks whether the communicators are set on a pair of  ALE::Coaster objects,
  // that the communicators are the same, and throws an exception otherwise.
  void CHKCOMMS(Coaster& obj1, Coaster& obj2) {
    if((obj1.comm == MPI_COMM_NULL) || (obj2.comm == MPI_COMM_NULL)){     
      throw ALE::Exception("ALE: Communicator not set");
    }
    int result;
    MPI_Comm_compare(obj1.comm, obj2.comm, &result);
    if(result == MPI_UNEQUAL) {
      throw ALE::Exception("ALE: Incompatible communicators");
    }
  }// CHKCOMMS()

}

#undef ALE_Coaster_cxx
