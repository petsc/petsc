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
  public:
    XObject() {};
    XObject(const XObject& xobject) {};
  };// class XObject
  //
  class XParallelObject : public XObject {
  protected:
    MPI_Comm _comm;
    int      _commRank, _commSize;
  protected:
    void __setupComm(const MPI_Comm& comm) {
      this->_comm = comm;
      PetscErrorCode ierr;
      ierr = MPI_Comm_size(this->_comm, &this->_commSize);CHKERROR(ierr, "Error in MPI_Comm_size");
      ierr = MPI_Comm_rank(this->_comm, &this->_commRank);CHKERROR(ierr, "Error in MPI_Comm_rank");
    }
  public:
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

#ifdef ALE_USE_DEBUGGING
  //   Xdebug and Xcodebug might have confusing interpretations, but the usage should be relatively transparent.
  // X sets the number of debugging layers laid so far -- ALE_XDEBUG_HEIGHT -- laid chronologically -- 
  // with the oldest layers laid first and having the lowest numbers: 1,2,etc.
  //   Now, debug works from the top of the debugging layers, and codebug works from the bottom:
  // Setting '--debug N' will activate N LAST layers of debuggin: if a routine has its __ALE_XDEBUG__ 
  // within N of the top, its debugging will be activated.
  // Likewise, setting '--codebug N' will activate the first N layers of debugging,
  // which is not what the developer usually want.
  // Hence, 'debug' is made to have the more common meaning.
  static int Xdebug   = 0;
  static int Xcodebug = 0;
  // Debugging works from the top: setting ALE::XSifterDef::debug to n will 'uncover' the *last* (newest) n layers of debugging.
  // Thus, the functions with the n heighest __ALE_DEBUG__ markers will produce debugging output.
  // Co-debugging works from the bottom: setting ALE::XSifterDef::codebug to n will 'uncover' the *first* (oldest) n layers of debugging.
  // Thus, the functions with the n lowest __ALE_DEBUG__ markers will produce debugging output.
#endif 

#define ALE_XDEBUG_HEIGHT 7
#define ALE_XDEBUG_LEVEL(n)  ((ALE::Xcodebug >= n) || (n > ALE_XDEBUG_HEIGHT - ALE::Xdebug))
#define ALE_XDEBUG           (ALE_XDEBUG_LEVEL(__ALE_XDEBUG__))

  

  template <typename Element_>
  struct SetElementTraits {
    typedef Element_ element_type;
    typedef typename std::template less<element_type> less_than;
  };

  template <typename Argument_>
  struct NoOp {
    typedef Argument_ argument_type;
    void operator()(const argument_type& arg) const{};
  };// struct NoOp

  template <typename Element_, typename Traits_ = SetElementTraits<Element_> , typename Allocator_ = ALE_ALLOCATOR<Element_> >
  class Set : public std::set<Element_, typename Traits_::less_than, Allocator_ > {
  public:
    // Encapsulated types
    typedef typename std::set<Element_, typename Traits_::less_than, Allocator_>       super;
    typedef Set                                                                        set_type;
    typedef Element_                                                                   element_type;
    typedef Traits_                                                                    element_traits;
    typedef typename super::iterator                                                   iterator;
    typedef element_type                                                               value_type;
    //
    // Standard
    //
    // making constructors explicit may prevent ambiguous application of operators, such as operator<<
    Set()                               : super(){};
    explicit Set(const element_type& e) : super() {insert(e);}     //
    template<typename ElementSequence_>
    explicit Set(const ElementSequence_& eseq) : super(eseq.begin(), eseq.end()){};
    // 
    // Main
    // 
    // Redirection: 
    // FIX: it is a little weird that 'insert' methods aren't inherited
    // but perhaps can be fixed by calling insert<Element_> (i.e., insert<Point> etc)?
    std::pair<iterator, bool> 
    inline insert(const Element_& e) { return super::insert(e); };
    //
    iterator 
    inline insert(iterator position, const Element_& e) {return super::insert(position,e);};
    //
    template <class InputIterator>
    void 
    inline insert(InputIterator b, InputIterator e) { return super::insert(b,e);};
    // 
    // Extended interface
    //
    inline iterator last() {
      return this->rbegin();
    };// last()
    //    
    inline bool contains(const Element_& e) {return (this->find(e) != this->end());};
    //
    inline void join(const Set& s) {
      for(iterator s_itor = s.begin(); s_itor != s.end(); s_itor++) {
        this->insert(*s_itor);
      }
    };
    inline void join(const Obj<Set>& s) { this->join(s->object());};
    //
    inline void meet(const Set& s) {// this should be called 'intersect' (the verb)
      Set removal;
      for(iterator self_itor = this->begin(); self_itor != this->end(); self_itor++) {
        Element_ e = *self_itor;
        if(!s.contains(e)){
          removal.insert(e);
        }
      }
      for(iterator rem_itor = removal.begin(); rem_itor != removal.end(); rem_itor++) {
        Element_ ee = *rem_itor;
        this->erase(ee);
      }
    };
    inline void meet(const Obj<Set>& s) { this->meet(s.object());};
    //
    inline void subtract(const Set& s) {
      Set removal;
      for(iterator self_itor = this->begin(); self_itor != this->end(); self_itor++) {
        Element_ e = *self_itor;
        if(s->contains(e)){
          removal.insert(e);
        }
      }
      for(iterator rem_itor = removal.begin(); rem_itor != removal.end(); rem_itor++) {
        Element_ ee = *rem_itor;
        this->erase(ee);
      }
    };
    inline void subtract(const Obj<Set>& s) {this->subtract(s.object());};
    //
    template <typename Op_>
    inline void traverse(const Op_& op) {
      for(iterator iter = this->begin(); iter!= this->end(); ++iter) {
        op(*iter);
      }
    };
    //
    template <typename ostream_type>
    friend ostream_type& operator<<(ostream_type& os, const Set& s) {
      os << "[[ ";
      for(iterator s_itor = s.begin(); s_itor != s.end(); s_itor++) {
        Element_ e = *s_itor;
        os << e << " ";
      }
      os << " ]]";
      return os;
    };
    //
    template <typename ostream_type>
    void view(ostream_type& os, const char *name = NULL) {
      os << "Viewing set";
      if(name != NULL) {
        os << " " << name;
      }
      os << " of size " << (int) this->size() << std::endl;
      os << *this << "\n";
    };
  };
}//namespace ALE

#endif
