#define ALE_ALE_cxx

#include <ALE.hh>

namespace ALE {
  //
  // Package-wide verbosity
  //
  static int verbosity = 0;

  int  getVerbosity() {return ALE::verbosity;};
  void setVerbosity(const int& verbosity) {ALE::verbosity = verbosity;};

  //
  //  Memory handling stuff (ALE_mem.hh).
  // 
  
  // static instance of a standard char allocator;  this is the only allocator used by ALE; 
  // it is defined here -- in an .cxx file -- to ensure that exactly one copy exists;
  // its services are presented through a static interface defined in universal_allocator.

  std::allocator<char> _alloc;

  char *universal_allocator::allocate(const universal_allocator::size_type& sz) {
    return _alloc.allocate(sz);
  }

  void universal_allocator::deallocate(char *p, const universal_allocator::size_type& sz) {
    return _alloc.deallocate(p,sz);
  }

  universal_allocator::size_type universal_allocator::max_size() {
    return _alloc.max_size();
  }

  //
  // Error/exception handling helper functions (ALE_exception.hh).
  //

  // A helper function for converting PETSc errors to exception
  // including the function and the line where the error occured.
  void ERROR(PetscErrorCode ierr, const char *func, int line, const char *msg) {
    if(ierr) {
      ostringstream mess;
      mess << func << ": line " << line << ": error " << ierr << ": " << msg << ":\n";
      throw ALE::Exception(mess);
    }
  }// ERROR()

  const char *ERRORMSG(const char *fmt, ...);

  // A helper function for converting MPI errors to exception
  void MPIERROR(PetscErrorCode ierr, const char *func, int line, const char *msg) {
    if(ierr) {
      char mpi_error[MPI_MAX_ERROR_STRING+1];
      int len = MPI_MAX_ERROR_STRING;
      PetscErrorCode ie = MPI_Error_string(ierr, mpi_error, &len);
      ostringstream mess;

      if (!ie) {
        mess << func << ": line " << line << ": error " << ierr << ": " << msg << ": " << mpi_error << "\n";
      } else {
        mess << func << ": line " << line << ": error " << ierr << ": " << msg << ": <unknown error>\n";
      }
      throw ALE::Exception(mess);
    }
  }// MPIERROR()

  // A helper function that allocates and assembles an error message from a format string 
  const char *ERRORMSG(const char *fmt, ...) {
    va_list Argp;
    int32_t buf_size = 2*MPI_MAX_ERROR_STRING;
    if(fmt) {
      va_start(Argp, fmt);
      char *msg = (char *)malloc(sizeof(char)*(buf_size+1));
#ifdef PETSC_HAVE_SNPRINTF
      snprintf(msg, buf_size, fmt, Argp);
#else
      sprintf(msg, fmt, Argp);
#endif
      va_end(Argp);
      return msg;
    }
    return fmt;
  }// ERRORMSG()

  //
  // Logging helper functions
  //

  static std::map<std::string, LogStage> _log_stage;  // a map from stage names to stage numbers

  #undef  __FUNCT__
  #define __FUNCT__ "LogCookieRegister"
  LogCookie LogCookieRegister(const char *name){
    LogCookie cookie;
    PetscErrorCode ierr = PetscClassIdRegister(name, &cookie);
   CHKERROR(ierr, "PetscClassIdRegister failed");
    return cookie;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "LogStageRegister"
  LogStage LogStageRegister(const char *name){
    int stage = 0;
    std::string stage_name(name);
    if(_log_stage.find(stage_name) == _log_stage.end()) {    
      // stage by that name not yet registered, so we register it and store its registration number.
      PetscErrorCode ierr = PetscLogStageRegister(name, &stage);CHKERROR(ierr, "PetscLogStageRegister failed");
      _log_stage[stage_name] = stage;                   
    }                                                        
    else {                                                   
      // stage by that name already registered, so we retrieve its registration number.
      stage = _log_stage[stage_name];                   
    }                                                        
    return stage;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "LogStagePush"
  void LogStagePush(int s){
    PetscErrorCode ierr;
    ierr = PetscLogStagePush(s);CHKERROR(ierr, "PetscLogStagePush failed");
  }//LogStagePush()

  #undef  __FUNCT__
  #define __FUNCT__ "LogStagePop"
  void LogStagePop(int s){
    // A future implementation may use 's' to check for the correct order of stage push/pop events.
    PetscErrorCode ierr;
    ierr = PetscLogStagePop();CHKERROR(ierr, "PetscLogStagePop failed");
  }//LogStagePop()

  static std::map<std::string, LogEvent> _log_event;  // a map from event names to event numbers

  #undef  __FUNCT__
  #define __FUNCT__ "LogEventRegister"
  LogEvent LogEventRegister(LogCookie cookie, const char *name) {
    std::string event_name(name);
    LogEvent event = 0;

    if (_log_event.find(event_name) == _log_event.end()) {
      PetscErrorCode ierr = PetscLogEventRegister(name, cookie, &event);CHKERROR(ierr, "PetscLogEventRegister failed");
      _log_event[event_name] = event;
    } else {
      // event by that name already registered, so we retrieve its registration number.
      event = _log_event[event_name];
    }
    return event;
  }

  #undef  __FUNCT__
  #define __FUNCT__ "LogEventRegister"
  LogEvent LogEventRegister(const char *name){
    return LogEventRegister(PETSC_SMALLEST_CLASSID, name);
  }

  #undef  __FUNCT__
  #define __FUNCT__ "LogEventBegin"
  void LogEventBegin(LogEvent e){
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(e, 0, 0, 0, 0);CHKERROR(ierr, "PetscLogEventBegin failed");
  }//LogEventBegin()

  #undef  __FUNCT__
  #define __FUNCT__ "LogEventEnd"
  void LogEventEnd(LogEvent e){
    PetscErrorCode ierr;
    ierr = PetscLogEventEnd(e, 0, 0, 0, 0);CHKERROR(ierr, "PetscLogEventEnd failed");
  }//LogEventEnd()

}

#undef ALE_ALE_cxx
