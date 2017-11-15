#ifndef included_ALE_exception_hh
#define included_ALE_exception_hh

#include <stdexcept>
#include <string>
#include <sstream>

typedef std::basic_ostringstream<char> ostringstream;
typedef std::basic_ostringstream<char> ostrstr;
typedef std::string                    string;


namespace ALE {
  class Exception : public std::runtime_error {
  public:
    explicit Exception(const char         * msg) : std::runtime_error(msg){};
    explicit Exception(const string&        msg) : std::runtime_error(msg){};
    explicit Exception(const ostringstream& txt) : std::runtime_error(txt.str()){};
    Exception(const Exception& e)      : std::runtime_error(e.what()) {};
    string msg()     const  {return std::string(this->what());};
    const char   *message() const  {return this->what();};
    // Printing
    template <typename Stream_>
    friend Stream_& operator<<(Stream_& os, const Exception& e) {
      os << "ERROR: " << e.message() << std::endl;
      return os;
    }
  };

  class XException {
    ostrstr _txt;
  public:
    XException(){};
    explicit
    XException(const string& msg)   {this->_txt << msg;};
    explicit
    XException(const ostrstr& txt)  {this->_txt << txt.str();};
    XException(const XException& e) {this->_txt << e._txt.str();};
    //
    const string msg()     const {return this->_txt.str();};
    const char   *message() const {return this->_txt.str().c_str();};
    // Message input
    template<typename Input_>
    XException& operator<<(const Input_& in) {
      this->_txt << in;
      return *this;
    }
    // Printing
    template <typename Stream_>
    friend Stream_& operator<<(Stream_& os, const XException& e) {
      os << "ERROR: " << e.message() << std::endl;
      return os;
    }
  };// class XException


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
  ::ALE::ERROR(ierr, __FUNCT__,  __LINE__, msg);

// A helper macro that passes __FUNCT__ and __LINE__ with the error msg to the MPIERROR routine
#define CHKMPIERROR(ierr, msg) \
  ::ALE::MPIERROR(ierr, __FUNCT__,  __LINE__, msg);

#endif
