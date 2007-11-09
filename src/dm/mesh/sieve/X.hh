#ifndef included_ALE_X_hh
#define included_ALE_X_hh

// BEGIN: these includes come from boost/multi_index/mem_fun.hpp
#include <boost/config.hpp> /* keep it first to prevent nasty warns in MSVC */
#include <boost/mpl/if.hpp>
#include <boost/type_traits/remove_reference.hpp>
// END


#include <boost/multi_index_container.hpp>
#include <boost/multi_index/key_extractors.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>

#include <boost/lambda/lambda.hpp>
using namespace ::boost::lambda;

#include <iostream>


#include <ALE.hh>

namespace ALE { 
  //
  class XObject {
  protected:
    int      _debug;
  public:
    XObject(const int debug = 0)    : _debug(debug) {};
    XObject(const XObject& xobject) : _debug(xobject._debug) {};
    //
    int      debug(const int& debug = -1) {if(debug >= 0) {this->_debug = debug;} return this->_debug;};
  };// class XObject

  class XParallelObject : public XObject {
  protected:
    MPI_Comm _comm;
    int      _commRank, _commSize;
  protected:
    void __setupComm(const MPI_Comm& comm) {
      this->_comm = comm;
      PetscErrorCode ierr;
      ierr = MPI_Comm_size(this->_comm, &this->_commSize); CHKERROR(ierr, "Error in MPI_Comm_size");
      ierr = MPI_Comm_rank(this->_comm, &this->_commRank); CHKERROR(ierr, "Error in MPI_Comm_rank");
    }
  public:
    XParallelObject(const MPI_Comm& comm, const int debug)   : XObject(debug) {this->__setupComm(comm);};
    XParallelObject(const MPI_Comm& comm = PETSC_COMM_WORLD) : XObject()      {this->__setupComm(comm);};
    XParallelObject(const XParallelObject& xpobject)         : XObject(xpobject), _comm(xpobject._comm) {};
    //
    MPI_Comm comm()     {return this->_comm;};
    int      commSize() {return this->_commSize;};
    int      commRank() {return this->_commRank;};
  };// class XParallelObject
  
  //
  // Key extractors
  //
  // 
  // The following member function return a const result.
  // It is best used through the macro ALE_CONST_MEM_FUN which takes only three arguments: 
  //  Class, ResultType, MemberFunctionPtr (see below).
  // OutputType (the actual return type) is different from the ResultType for somewhat obscure reasons.
  // Once I (have time to) understand the issue better, the usage pattern may get simplified.
  template<class InputType_, typename ResultType_, typename OutputType_, OutputType_ (InputType_::*PtrToMemberFunction)()const>
  struct const_const_mem_fun
  {
    typedef InputType_                                            input_type;
    typedef typename ::boost::remove_reference<ResultType_>::type result_type;
    typedef OutputType_                                           output_type;
    //
    // Main interface
    //
    template<typename ChainedPtrTarget>
    output_type operator()(const ChainedPtrTarget*& x)const
    {
      return operator()((*x));
    }
    
    output_type operator()(const input_type& x)const
    {
      return (x.*PtrToMemberFunction)();
    }
    
    output_type operator()(const ::boost::reference_wrapper<const input_type>& x)const
    { 
      return operator()(x.get());
    }
    
    output_type operator()(const ::boost::reference_wrapper<input_type>& x,int=0)const
    { 
      return operator()(x.get());
    }
  };// struct const_const_mem_fun
#define ALE_CONST_MEM_FUN(CLASS, RESULT_TYPE, FUN) ::ALE::const_const_mem_fun<CLASS, RESULT_TYPE, const RESULT_TYPE, FUN>
    static int Xdebug   = 0;
    static int Xcodebug = 0;
    // Debugging works from the top: setting ALE::XSifterDef::debug to n will 'uncover' the *last* (newest) n layers of debugging.
    // Thus, the functions with the n heighest __ALE_DEBUG__ markers will produce debugging output.
    // Co-debugging works from the bottom: setting ALE::XSifterDef::codebug to n will 'uncover' the *first* (oldest) n layers of debugging.
    // Thus, the functions with the n lowest __ALE_DEBUG__ markers will produce debugging output.
#define ALE_XDEBUG_HEIGHT 7
#define ALE_XDEBUG_LEVEL(n)  ((ALE::Xcodebug >= n) || (n > ALE_XDEBUG_HEIGHT - ALE::Xdebug))
#define ALE_XDEBUG           (ALE_XDEBUG_LEVEL(__ALE_XDEBUG__))
}//namespace ALE

#endif
