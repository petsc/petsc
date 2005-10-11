#define ALE_ALE_cxx

#include <ALE.hh>

namespace ALE {

  #undef  __FUNCT__
  #define __FUNCT__ "Coaster::setComm"
  void Coaster::setComm(MPI_Comm comm) {
    if (this->comm == comm) {
      return;
    }
    if ((this->comm != -1) && (this->comm != comm)) {
      throw ALE::Exception("Cannot reset the communicator");
    }
    PetscErrorCode ierr;
    this->comm = comm;
    ierr = MPI_Comm_rank(this->comm, &this->commRank); CHKERROR(ierr, "Error in MPI_Comm_rank");
    ierr = MPI_Comm_size(this->comm, &this->commSize); CHKERROR(ierr, "Error in MPI_Comm_rank"); 
    ierr = PetscObjectCreate(this->comm, &this->petscObj); CHKERROR(ierr, "Failed on PetscObjectCreate");
  }// Coaster::setComm()
  
  #undef  __FUNCT__
  #define __FUNCT__ "Coaster::clear"
  Coaster& Coaster::clear(){
    this->comm     = -1; 
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
    if(obj.comm == -1){     
      throw ALE::Exception("ALE: Communicator not set");
    }
  }// CHKCOMM()

  // A helper friend function that checks whether the communicators are set on a pair of  ALE::Coaster objects,
  // that the communicators are the same, and throws an exception otherwise.
  void CHKCOMMS(Coaster& obj1, Coaster& obj2) {
    if((obj1.comm == -1) || (obj2.comm == -1)){     
      throw ALE::Exception("ALE: Communicator not set");
    }
    if(obj1.comm != obj2.comm) {
      throw ALE::Exception("ALE: Incompatible communicators");
    }
  }// CHKCOMMS()

  // A helper function that throws an ALE::Exception with a message identifying the function that returned the given error code, 
  // including the function and the line where the error occured.
  void ERROR(PetscErrorCode ierr, const char *func, int line, const char *msg) {
    if(ierr) {
        int32_t buf_size = 2*1024;
        char *mess = (char *)malloc(sizeof(char)*(buf_size+1));
        snprintf(mess, buf_size, "%s: line %d: error %d: %s:\n", func, line, (int)ierr, msg);
        throw ALE::Exception(mess);
    }
  }// ERROR()

  const char *ERRORMSG(const char *fmt, ...);

  // A helper function for converting MPI errors to exception
  void MPIERROR(PetscErrorCode ierr, const char *func, int line, const char *msg) {
    if(ierr) {
      char mpi_error[MPI_MAX_ERROR_STRING+1];
      int32_t len = MPI_MAX_ERROR_STRING;
      PetscErrorCode ie = MPI_Error_string(ierr, mpi_error, &len);
      char *mess;
      if(!ie) {
        mess = (char *)malloc(sizeof(char)*(strlen(msg)+len+1));
        sprintf(mess, "%s: %s", msg, mpi_error);
      }
      else {
        mess = (char *)malloc(sizeof(char)*(strlen(msg)));
        sprintf(mess, "%s: <unknown error>", msg);
      }
      ERROR(ierr, func, line, mess);
    }
  }// MPIERROR()

  // A helper function that allocates and assembles an error message from a format string 
  const char *ERRORMSG(const char *fmt, ...) {
    va_list Argp;
    int32_t buf_size = 2*MPI_MAX_ERROR_STRING;
    if(fmt) {
      va_start(Argp, fmt);
      char *msg = (char *)malloc(sizeof(char)*(buf_size+1));
      snprintf(msg, buf_size, fmt, Argp);
      va_end(Argp);
      return msg;
    }
    return fmt;
  }// ERRORMSG()
}

#undef ALE_ALE_cxx
