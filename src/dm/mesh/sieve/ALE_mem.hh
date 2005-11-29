#ifndef included_ALE_mem_hh
#define included_ALE_mem_hh
// This should be included indirectly -- only by including ALE.hh


#include <memory>
#include <typeinfo>
#include <petsc.h>

#ifdef ALE_HAVE_CXX_ABI
#include <cxxabi.h>
#endif

namespace ALE {

  // General allocator; use it when no logging is necessary; create/del methods act as new/delete.
  template <class _T>
  class constructor_allocator {
  public:
    typedef typename std::allocator<_T> Allocator; 
  private:
    Allocator _allocator; // The underlying allocator
  public:
    typedef typename Allocator::size_type       size_type;
    typedef typename Allocator::difference_type difference_type;
    typedef typename Allocator::pointer         pointer;
    typedef typename Allocator::const_pointer   const_pointer;
    typedef typename Allocator::reference       reference;
    typedef typename Allocator::const_reference const_reference;
    typedef typename Allocator::value_type      value_type;

    constructor_allocator()                                    {};    
    constructor_allocator(const constructor_allocator& a)      {};
    template <class _TT> 
    constructor_allocator(const constructor_allocator<_TT>& aa){};
    ~constructor_allocator() {};

    // Reproducing the standard allocator interface
    pointer       address(reference _x) const          { return _allocator.address(_x); };
    const_pointer address(const_reference _x) const    { return _allocator.address(_x); };
    _T*           allocate(size_type _n)               { return _allocator.allocate(_n);};
    void          deallocate(pointer _p, size_type _n) { _allocator.deallocate(_p, _n); };
    void          construct(pointer _p, const _T& _val){ _allocator.construct(_p, _val);};
    void          destroy(pointer _p)                  { _allocator.destroy(_p);        };
    size_type     max_size() const                     { return _allocator.max_size();  };
    // conversion typedef
    template <class _TT>
    struct rebind { typedef constructor_allocator<_TT> other;};

    _T* create(const _T& _val = _T());
    void del(_T* _p);
  };

  template <class _T> 
  _T* constructor_allocator<_T>::create(const _T& _val) {
    // First, allocate space for a single object
    _T* _p = this->_allocator.allocate(1);
    // Construct an object in the provided space using the provided initial value
    this->_allocator.construct(_p,  _val);
    return _p;
  }

  template <class _T> 
  void constructor_allocator<_T>::del(_T* _p) {
    this->_allocator.destroy(_p);
    this->_allocator.deallocate(_p, 1);
  }



  // An allocator all of whose events (allocation, deallocation, new, delete) are logged using PetscLogging facilities.
  template <class _T>
  class logged_allocator : public constructor_allocator<_T> {
  private:
    static bool        _log_initialized;
    static PetscCookie _cookie;
    static int         _allocate_event;
    static int         _deallocate_event;
    static int         _construct_event;
    static int         _destroy_event;
    static int         _create_event;
    static int         _del_event;
    static void __log_initialize();
    static void __log_event_register(const char *class_name, const char *event_name, PetscEvent *event_ptr);
  public:
    typedef typename constructor_allocator<_T>::size_type size_type;
    logged_allocator()                                   : constructor_allocator<_T>()  {__log_initialize();};    
    logged_allocator(const logged_allocator& a)          : constructor_allocator<_T>(a) {__log_initialize();};
    template <class _TT> 
    logged_allocator(const logged_allocator<_TT>& aa)    : constructor_allocator<_T>(aa){__log_initialize();};
    ~logged_allocator() {};
    // conversion typedef
    template <class _TT>
    struct rebind { typedef logged_allocator<_TT> other;};

    _T*  allocate(size_type _n);
    void deallocate(_T*  _p, size_type _n);
    void construct(_T* _p, const _T& _val);
    void destroy(_T* _p);

    _T*  create(const _T& _val = _T());
    void del(_T*  _p);    
  };

  template <class _T>
  bool logged_allocator<_T>::_log_initialized(false);
  template <class _T>
  PetscCookie logged_allocator<_T>::_cookie(0);
  template <class _T>
  int logged_allocator<_T>::_allocate_event(0);
  template <class _T>
  int logged_allocator<_T>::_deallocate_event(0);
  template <class _T>
  int logged_allocator<_T>::_construct_event(0);
  template <class _T>
  int logged_allocator<_T>::_destroy_event(0);
  template <class _T>
  int logged_allocator<_T>::_create_event(0);
  template <class _T>
  int logged_allocator<_T>::_del_event(0);
  
  template <class _T>
  void logged_allocator<_T>::__log_initialize() {
    if(!logged_allocator::_log_initialized) {
      // Get a new cookie based on _T's typeid name
      const std::type_info& id = typeid(_T);
      const char *id_name;
#ifdef ALE_HAVE_CXX_ABI
      // If the C++ ABI API is available, we can use it to demangle the class name provided by type_info.
      // Here we assume the industry standard C++ ABI as described in http://www.codesourcery.com/cxx-abi/abi.html.
      int status;
      char *id_name_demangled = abi::__cxa_demangle(id.name(), NULL, NULL, &status);
      if(status != 0) {
        // Demangling failed, we use the mangled name.
        id_name = id.name();
      }
      else {
        // Use the demangled name to register a cookie.
        id_name = id_name_demangled;
      }
#else
      // If demangling is not available, use the class name returned by typeid directly.
      id_name = id.name();
#endif
      // Use id_name to register a cookie and events.
      PetscErrorCode ierr = PetscLogClassRegister(&logged_allocator::_cookie, id_name); 
      CHKERROR(ierr, "PetscLogClassRegister failed");
      // Register the basic allocator methods' invocations as events; use the mangled class name.
      logged_allocator::__log_event_register(id_name, "allocate", &logged_allocator::_allocate_event);
      logged_allocator::__log_event_register(id_name, "deallocate", &logged_allocator::_deallocate_event);
      logged_allocator::__log_event_register(id_name, "construct", &logged_allocator::_construct_event);
      logged_allocator::__log_event_register(id_name, "destroy", &logged_allocator::_destroy_event);
      logged_allocator::__log_event_register(id_name, "create", &logged_allocator::_create_event);
      logged_allocator::__log_event_register(id_name, "del", &logged_allocator::_del_event);
#ifdef ALE_HAVE_CXX_ABI
      // Free the name malloc'ed by __cxa_demangle
      free(id_name_demangled);
#endif
      logged_allocator::_log_initialized = true;

    }// if(!!logged_allocator::_log_initialized)
  }


  template <class _T> 
  void logged_allocator<_T>::__log_event_register(const char *class_name, const char *event_name, PetscEvent *event_ptr){
    // This routine assumes a cookie has been obtained.
    ostringstream txt;
    txt << class_name << ": " << event_name;
    PetscErrorCode ierr = PetscLogEventRegister(event_ptr, txt.str().c_str(), logged_allocator::_cookie);
    CHKERROR(ierr, "PetscLogEventRegister failed");
  }

  template <class _T>
  _T*  logged_allocator<_T>::allocate(size_type _n) {
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(logged_allocator::_allocate_event, 0, 0, 0, 0); CHKERROR(ierr, "Event begin failed");
    _T* _p = constructor_allocator<_T>::allocate(_n);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "logged_allocator: allocate called\n"); CHKERROR(ierr, "PetscPrintf failed");
    ierr = PetscLogEventEnd(logged_allocator::_allocate_event, 0, 0, 0, 0); CHKERROR(ierr, "Event end failed");
    return _p;
  }

  template <class _T>
  void logged_allocator<_T>::deallocate(_T* _p, size_type _n) {
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(logged_allocator::_deallocate_event, 0, 0, 0, 0); CHKERROR(ierr, "Event begin failed");
    constructor_allocator<_T>::deallocate(_p, _n);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "logged_allocator: deallocate called\n"); CHKERROR(ierr, "PetscPrintf failed");
    ierr = PetscLogEventEnd(logged_allocator::_deallocate_event, 0, 0, 0, 0); CHKERROR(ierr, "Event end failed");
  }

  template <class _T>
  void logged_allocator<_T>::construct(_T* _p, const _T& _val) {
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(logged_allocator::_construct_event, 0, 0, 0, 0); CHKERROR(ierr, "Event begin failed");
    constructor_allocator<_T>::construct(_p, _val);
    ierr = PetscLogEventEnd(logged_allocator::_construct_event, 0, 0, 0, 0); CHKERROR(ierr, "Event end failed");
  }

  template <class _T>
  void logged_allocator<_T>::destroy(_T* _p) {
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(logged_allocator::_destroy_event, 0, 0, 0, 0); CHKERROR(ierr, "Event begin failed");
    constructor_allocator<_T>::destroy(_p);
    ierr = PetscLogEventEnd(logged_allocator::_destroy_event, 0, 0, 0, 0); CHKERROR(ierr, "Event end failed");
  }

  template <class _T>
  _T* logged_allocator<_T>::create(const _T& _val) {
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(logged_allocator::_create_event, 0, 0, 0, 0); CHKERROR(ierr, "Event begin failed");
    _T* _p = constructor_allocator<_T>::create(_val);
    ierr = PetscLogEventEnd(logged_allocator::_create_event, 0, 0, 0, 0); CHKERROR(ierr, "Event end failed");
    return _p;
  }

  template <class _T>
  void logged_allocator<_T>::del(_T* _p) {
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(logged_allocator::_del_event, 0, 0, 0, 0); CHKERROR(ierr, "Event begin failed");
    constructor_allocator<_T>::del(_p);
    ierr = PetscLogEventEnd(logged_allocator::_del_event, 0, 0, 0, 0); CHKERROR(ierr, "Event end failed");
  }

#ifdef ALE_USE_LOGGING
#define ALE_ALLOCATOR logged_allocator
#else
#define ALE_ALLOCATOR constructor_allocator
#endif

  //
  // The following classes define smart pointer behavior.  
  // They rely on allocators for memory pooling and logging (if logging is on).
  //

  // This is an Obj<X>-specific exception that is thrown when incompatible object conversion is attempted.
  class BadCast : public Exception {
  public:
    BadCast(const char  *msg)   : Exception(msg) {};
    BadCast(const BadCast& e)   : Exception(e) {};
  };

  // This is the main smart pointer class.
  template<class X> 
  class Obj {
  public:
    // Types 
    typedef ALE_ALLOCATOR<X>   Allocator;
    typedef ALE_ALLOCATOR<int> Allocator_int;
  public:
    // These are intended to be private
    X*       objPtr;         // object pointer
    int32_t *refCnt;         // reference count
    bool     allocator_used; // indicates that the object was allocated using create<X>, and should be released using del<X>.
    // allocators
    Allocator_int      int_allocator;
    Allocator          allocator;
    // Constructor; this can be made private, if we move operator Obj<Y> outside this class definition and make it a friend.
    Obj(X *xx, int32_t *refCnt, bool allocator_used);
  public:
    // Constructors & a destructor
    Obj() : objPtr((X *)NULL), refCnt((int32_t*)NULL), allocator_used(false) {};
    Obj(X x);
    Obj(X *xx);
    Obj(const Obj& obj);
    ~Obj();

    // "Factory" methods
    Obj& create(){return Obj(X());};
    Obj& create(const X& x){return Obj(X(x));};

    // predicates & assertions
    bool isNull() {return (this->objPtr == NULL);};
    void assertNull(bool flag) { if(this->isNull() != flag){ throw(Exception("Null assertion failed"));}};

    // comparison operators
    bool operator==(const Obj& obj) { return (this->objPtr == obj.objPtr);};
    bool operator!=(const Obj& obj) { return (this->objPtr != obj.objPtr);};
    
    // assignment/conversion operators
    Obj& operator=(const Obj& obj);
    template <class Y> operator Obj<Y> const();
    template <class Y> Obj& operator=(const Obj<Y>& obj);

    // dereference operators
    X*   operator->() {return objPtr;};
    
    // "exposure" methods: expose the underlying object or object pointer
    operator X*() {return objPtr;};
    X operator*() {assertNull(false); return *objPtr;};
    operator X()  {assertNull(false); return *objPtr;};
    template<class Y> Obj& copy(Obj<Y>& obj); // this operator will copy the underlying objects: USE WITH CAUTION
    

    // depricated methods/operators
    X* ptr()      {return objPtr;};
    X* pointer()  {return objPtr;};
    X  obj()      {assertNull(false); return *objPtr;};
    X  object()   {assertNull(false); return *objPtr;};
    
  };// class Obj<X>


  // Constructors 
  template <class X>
  Obj<X>::Obj(X x) { 
    // allocate and copy object
    this->objPtr = Obj<X>::allocator.create(x);
    //this->objPtr = new X(x);
    refCnt = Obj<X>::int_allocator.create(1);
    //this->refCnt = new int(1);
    this->allocator_used = true;
    //this->allocator_used = false;
  }
  
  template <class X>
  Obj<X>::Obj(X *xx){// such an object will be destroyed by calling 'delete' on its pointer 
                     // (e.g., we assume the pointer was obtained with new)
    this->objPtr = xx; 
    refCnt = Obj<X>::int_allocator.create(1);
    //refCnt   = new int(1);
    allocator_used = false;
  }
  
  template <class X>
  Obj<X>::Obj(X *xx, int32_t *refCnt, bool allocator_used) {  // This is intended to be private.
    this->objPtr = xx;
    this->refCnt = refCnt;  // we assume that all refCnt pointers are obtained using an int_allocator
    (*this->refCnt)++;
    this->allocator_used = allocator_used;
  }
  
  template <class X>
  Obj<X>::Obj(const Obj& obj) {
    this->objPtr = obj.objPtr;
    this->refCnt = obj.refCnt;
    (*this->refCnt)++;
    this->allocator_used = obj.allocator_used;
  }

  // Destructor
  template <class X>
  Obj<X>::~Obj(){
    if((this->refCnt != (int32_t *)NULL) && (--(this->refCnt) == 0)) {  
      // If  allocator has been used to create an objPtr, we use it to delete it as well.
      if(this->allocator_used) {
        Obj<X>::allocator.del(objPtr);
      }
      else { // otherwise we use 'delete'
        delete objPtr;
      }
      // refCnt is always created/delete using the int_allocator.
      Obj<X>::int_allocator.del(refCnt);
    }
  }


  // assignment operator
  template <class X>
  Obj<X>& Obj<X>::operator=(const Obj<X>& obj) {
    if(this->objPtr == obj.objPtr) {return *this;}
    // Destroy 'this' Obj -- it will properly release the underlying object if the reference count is exhausted.
    this->~Obj<X>();
    // Now copy the data from obj.
    this->objPtr = obj.objPtr;
    this->refCnt = obj.refCnt;
    if(this->refCnt!= NULL) {
      (*this->refCnt)++;
    }
    this->allocator_used = obj.allocator_used;
    return *this;
  }

  // conversion operator, preserves 'this'
  template<class X> template<class Y> 
  Obj<X>::operator Obj<Y> const() {
    // We attempt to cast X* objPtr to Y* using dynamic_cast
    Y* yObjPtr = dynamic_cast<Y*>(this->objPtr);
    // If the cast failed, throw an exception
    if(yObjPtr == NULL) {
      throw ALE::Exception("Bad cast Obj<X> --> Obj<Y>");
    }
    // Okay, we can proceed 
    // TROUBLE: potentially a different allocator was used to allocate *yObjPtr.
    return Obj<Y>(yObjPtr, this->refCnt, this->allocator_used);
  }

  // assignment-conversion operator
  template<class X> template<class Y> 
  Obj<X>& Obj<X>::operator=(const Obj<Y>& obj) {
    // We attempt to cast Y* obj.objPtr to X* using dynamic_cast
    X* xObjPtr = dynamic_cast<X*>(obj.objPtr);
    // If the cast failed, throw an exception
    if(xObjPtr == NULL) {
      throw BadCast("Bad cast Obj<Y> --> Obj<X>");
    }
    // Okay, we can proceed with the assignment
    if(this->objPtr == obj.objPtr) {return *this;}
    // Destroy 'this' Obj -- it will properly release the underlying object if the reference count is exhausted.
    this->~Obj<X>();
    // Now copy the data from obj.
    // TROUBLE: potentially a different allocator was used to allocate *xObjPtr.
    this->objPtr = xObjPtr;
    this->refCnt = obj.refCnt;
    (*this->refCnt)++;
    this->allocator_used = obj.allocator_used;
    return *this;
  }
 
  // copy operator (USE WITH CAUTION)
  template<class X> template<class Y> 
  Obj<X>& Obj<X>::copy(Obj<Y>& obj) {
    if(this->isNull() || obj.isNull()) {
      throw(Exception("Copying to or from a null Obj"));
    }
    *(this->objPtr) = *(obj.objPtr);
    return *this;
  }


} // namespace ALE

#endif
