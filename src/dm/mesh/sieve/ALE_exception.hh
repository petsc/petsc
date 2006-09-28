#ifndef included_ALE_exception_hh
#define included_ALE_exception_hh

#include <string>
#include <sstream>

typedef std::basic_ostringstream<char> ostringstream;
typedef std::string string;

namespace ALE {
  class Exception {
    string _msg;
  public:
    //    explicit Exception(const char         * msg) : _msg(msg){};
    explicit Exception(const string&        msg) : _msg(msg){};
    explicit Exception(const ostringstream& txt) : _msg(txt.str()){};
    Exception(const Exception& e)      : _msg(e._msg) {};
    const string& msg()     const  {return this->_msg;};
    const char   *message() const  {return this->_msg.c_str();};
    // Printing
    template <typename Stream_>
    friend Stream_& operator<<(Stream_& os, const Exception& e) {
      os << "ERROR: " << e.message() << std::endl;
      return os;
    };
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

#endif
