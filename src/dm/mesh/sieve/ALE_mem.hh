#ifndef included_ALE_mem_hh
#define included_ALE_mem_hh
// This should be included indirectly -- only by including ALE.hh


#include <memory>
#include <typeinfo>
#include <petsc.h>
#include <ALE_log.hh>

#ifdef ALE_HAVE_CXX_ABI
#include <cxxabi.h>
#endif

namespace ALE {

  // This UNIVERSAL allocator class is static and provides allocation/deallocation services to all allocators defined below.
  class universal_allocator {
  public: 
    typedef std::size_t size_type;
    static char*     allocate(const size_type& sz);
    static void      deallocate(char *p, const size_type& sz);
    static size_type max_size();
  };

  // This allocator implements create and del methods, that act roughly as new and delete in that they invoke a constructor/destructor
  // in addition to memory allocation/deallocation.
  // An additional (and potentially dangerous) feature allows an object of any type to be deleted so long as its size has been provided.
  template <class T>
  class polymorphic_allocator {
  public:
    typedef typename std::allocator<T> Alloc;
    // A specific allocator -- alloc -- of type Alloc is used to define the correct types and implement methods
    // that do not allocate/deallocate memory themselves -- the universal _alloc is used for that (and only that).
    // The relative size sz is used to calculate the amount of memory to request from _alloc to satisfy a request to alloc.
    typedef typename Alloc::size_type       size_type;
    typedef typename Alloc::difference_type difference_type;
    typedef typename Alloc::pointer         pointer;
    typedef typename Alloc::const_pointer   const_pointer;
    typedef typename Alloc::reference       reference;
    typedef typename Alloc::const_reference const_reference;
    typedef typename Alloc::value_type      value_type;

    static Alloc alloc;                            // The underlying specific allocator
    static typename Alloc::size_type sz;           // The size of T universal units of char

    polymorphic_allocator()                                    {};    
    polymorphic_allocator(const polymorphic_allocator& a)      {};
    template <class TT> 
    polymorphic_allocator(const polymorphic_allocator<TT>& aa){};
    ~polymorphic_allocator() {};

    // Reproducing the standard allocator interface
    pointer       address(reference _x) const          { return alloc.address(_x);                                    };
    const_pointer address(const_reference _x) const    { return alloc.address(_x);                                    };
    T*            allocate(size_type _n)               { return (T*)universal_allocator::allocate(_n*sz);            };
    void          deallocate(pointer _p, size_type _n) { universal_allocator::deallocate((char*)_p, _n*sz);           };
    void          construct(pointer _p, const T& _val) { alloc.construct(_p, _val);                                   };
    void          destroy(pointer _p)                  { alloc.destroy(_p);                                           };
    size_type     max_size() const                     { return (size_type)floor(universal_allocator::max_size()/sz); };
    // conversion typedef
    template <class TT>
    struct rebind { typedef polymorphic_allocator<TT> other;};
    
    T*  create(const T& _val = T());
    void del(T* _p);
    template<class TT> void del(TT* _p, size_type _sz);
  };

  template <class T>
  typename polymorphic_allocator<T>::Alloc polymorphic_allocator<T>::alloc;

  //IMPORTANT: allocator 'sz' calculation takes place here
  template <class T>
  typename polymorphic_allocator<T>::size_type polymorphic_allocator<T>::sz = 
    (typename polymorphic_allocator<T>::size_type)(ceil(sizeof(T)/sizeof(char)));

  template <class T> 
  T* polymorphic_allocator<T>::create(const T& _val) {
    // First, allocate space for a single object
    T* _p = (T*)universal_allocator::allocate(sz);
    // Construct an object in the provided space using the provided initial value
    this->alloc.construct(_p,  _val);
    return _p;
  }

  template <class T>
  void polymorphic_allocator<T>::del(T* _p) {
    _p->~T();
    universal_allocator::deallocate((char*)_p, polymorphic_allocator<T>::sz);
  }

  template <class T> template <class TT>
  void polymorphic_allocator<T>::del(TT* _p, size_type _sz) {
    _p->~TT();
    universal_allocator::deallocate((char*)_p, _sz);
  }


  // An allocator all of whose events (allocation, deallocation, new, delete) are logged using ALE_log facilities.
  // O is true if this is an Obj allocator (that's the intended use, anyhow).
  template <class T, bool O = false>
  class logged_allocator : public polymorphic_allocator<T> {
  private:
    static bool        _log_initialized;
    static LogCookie   _cookie;
    static int         _allocate_event;
    static int         _deallocate_event;
    static int         _construct_event;
    static int         _destroy_event;
    static int         _create_event;
    static int         _del_event;
    //
    static void     __log_initialize();
    static LogEvent __log_event_register(const char *class_name, const char *event_name);
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    // FIX: should PETSc memory logging machinery be wrapped by ALE_log like the rest of the logging stuff?
    PetscObject _petscObj; // this object is used to log memory in PETSc
#endif
    void __alloc_initialize(); 
    void __alloc_finalize();
  public:
    // Present the correct allocator interface
    typedef typename polymorphic_allocator<T>::size_type       size_type;
    typedef typename polymorphic_allocator<T>::difference_type difference_type;
    typedef typename polymorphic_allocator<T>::pointer         pointer;
    typedef typename polymorphic_allocator<T>::const_pointer   const_pointer;
    typedef typename polymorphic_allocator<T>::reference       reference;
    typedef typename polymorphic_allocator<T>::const_reference const_reference;
    typedef typename polymorphic_allocator<T>::value_type      value_type;
    //
    logged_allocator()                                   : polymorphic_allocator<T>()  {__log_initialize(); __alloc_initialize();};    
    logged_allocator(const logged_allocator& a)          : polymorphic_allocator<T>(a) {__log_initialize(); __alloc_initialize();};
    template <class TT> 
    logged_allocator(const logged_allocator<TT>& aa)    : polymorphic_allocator<T>(aa){__log_initialize(); __alloc_initialize();};
    ~logged_allocator() {__alloc_finalize();};
    // conversion typedef
    template <class TT>
    struct rebind { typedef logged_allocator<TT> other;};

    T*   allocate(size_type _n);
    void deallocate(T*  _p, size_type _n);
    void construct(T* _p, const T& _val);
    void destroy(T* _p);

    T*  create(const T& _val = T());
    void del(T*  _p);    
    template <class TT> void del(TT* _p, size_type _sz);
  };

  template <class T, bool O>
  bool logged_allocator<T, O>::_log_initialized(false);
  template <class T, bool O>
  LogCookie logged_allocator<T,O>::_cookie(0);
  template <class T, bool O>
  int logged_allocator<T, O>::_allocate_event(0);
  template <class T, bool O>
  int logged_allocator<T, O>::_deallocate_event(0);
  template <class T, bool O>
  int logged_allocator<T, O>::_construct_event(0);
  template <class T, bool O>
  int logged_allocator<T, O>::_destroy_event(0);
  template <class T, bool O>
  int logged_allocator<T, O>::_create_event(0);
  template <class T, bool O>
  int logged_allocator<T, O>::_del_event(0);
  
  template <class T, bool O>
  void logged_allocator<T, O>::__log_initialize() {
    if(!logged_allocator::_log_initialized) {
      // First of all we make sure PETSc is initialized
      PetscTruth flag;
      PetscErrorCode ierr = PetscInitialized(&flag); CHKERROR(ierr, "Error in PetscInitialized");
      if(!flag) {
        // I guess it would be nice to initialize PETSc here, but we'd need argv/argc here
        throw ALE::Exception("PETSc not initialized");
      }
      // Get a new cookie based on T's typeid name
      const std::type_info& id = typeid(T);
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
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
      // Use id_name to register a cookie and events.
      logged_allocator::_cookie = LogCookieRegister(id_name); 
      // Register the basic allocator methods' invocations as events; use the mangled class name.
      logged_allocator::_allocate_event   = logged_allocator::__log_event_register(id_name, "allocate");
      logged_allocator::_deallocate_event = logged_allocator::__log_event_register(id_name, "deallocate");
      logged_allocator::_construct_event  = logged_allocator::__log_event_register(id_name, "construct");
      logged_allocator::_destroy_event    = logged_allocator::__log_event_register(id_name, "destroy");
      logged_allocator::_create_event     = logged_allocator::__log_event_register(id_name, "create");
      logged_allocator::_del_event        = logged_allocator::__log_event_register(id_name, "del");
#endif
#ifdef ALE_HAVE_CXX_ABI
      // Free the name malloc'ed by __cxa_demangle
      free(id_name_demangled);
#endif
      logged_allocator::_log_initialized = true;

    }// if(!!logged_allocator::_log_initialized)
  }// logged_allocator<T,O>::__log_initialize()

  template <class T, bool O>
  void logged_allocator<T, O>::__alloc_initialize() {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    PetscErrorCode ierr = PetscObjectCreate(PETSC_COMM_WORLD, &this->_petscObj); 
    CHKERROR(ierr, "Failed in PetscObjectCreate");
#endif
  }// logged_allocator<T,O>::__alloc_initialize

  template <class T, bool O>
  void logged_allocator<T, O>::__alloc_finalize() {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    // FIX: destroying these PetscObjects will cause a meltdown
    //PetscErrorCode ierr = PetscObjectDestroy(this->_petscObj); 
    //CHKERROR(ierr, "Failed in PetscObjectDestroy");
#endif
  }// logged_allocator<T,O>::__alloc_finalize

  template <class T, bool O> 
  LogEvent logged_allocator<T, O>::__log_event_register(const char *class_name, const char *event_name){
    // This routine assumes a cookie has been obtained.
    ostringstream txt;
    if(O) {
      txt << "Obj: ";
    }
#ifdef ALE_LOGGING_VERBOSE
    txt << class_name;
#else
    txt << "<allocator>";
#endif
    txt << ": " << event_name;
    return LogEventRegister(logged_allocator::_cookie, txt.str().c_str());
  }

  template <class T, bool O>
  T*  logged_allocator<T, O>::allocate(size_type _n) {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventBegin(logged_allocator::_allocate_event); 
#endif
    T* _p = polymorphic_allocator<T>::allocate(_n);
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    PetscErrorCode ierr = PetscLogObjectMemory(this->_petscObj, _n*polymorphic_allocator<T>::sz); 
    CHKERROR(ierr, "Error in PetscLogObjectMemory");
    LogEventEnd(logged_allocator::_allocate_event); 
#endif
    return _p;
  }
  
  template <class T, bool O>
  void logged_allocator<T, O>::deallocate(T* _p, size_type _n) {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventBegin(logged_allocator::_deallocate_event);
#endif
    polymorphic_allocator<T>::deallocate(_p, _n);
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventEnd(logged_allocator::_deallocate_event);
#endif
  }
  
  template <class T, bool O>
  void logged_allocator<T, O>::construct(T* _p, const T& _val) {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventBegin(logged_allocator::_construct_event);
#endif
    polymorphic_allocator<T>::construct(_p, _val);
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventEnd(logged_allocator::_construct_event);
#endif
  }
  
  template <class T, bool O>
  void logged_allocator<T, O>::destroy(T* _p) {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventBegin(logged_allocator::_destroy_event);
#endif
    polymorphic_allocator<T>::destroy(_p);
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventEnd(logged_allocator::_destroy_event);
#endif
  }
  
  template <class T, bool O>
  T* logged_allocator<T, O>::create(const T& _val) {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventBegin(logged_allocator::_create_event); 
#endif
    T* _p = polymorphic_allocator<T>::create(_val);
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    PetscErrorCode ierr = PetscLogObjectMemory(this->_petscObj, polymorphic_allocator<T>::sz); 
    CHKERROR(ierr, "Error in PetscLogObjectMemory");
    LogEventEnd(logged_allocator::_create_event);
#endif
    return _p;
  }

  template <class T, bool O>
  void logged_allocator<T, O>::del(T* _p) {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventBegin(logged_allocator::_del_event);
#endif
    polymorphic_allocator<T>::del(_p);
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventEnd(logged_allocator::_del_event);
#endif
  }

  template <class T, bool O> template <class TT>
  void logged_allocator<T, O>::del(TT* _p, size_type _sz) {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventBegin(logged_allocator::_del_event);
#endif
    polymorphic_allocator<T>::del(_p, _sz);
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventEnd(logged_allocator::_del_event);
#endif
  }

#ifdef ALE_USE_LOGGING
#define ALE_ALLOCATOR logged_allocator
#else
#define ALE_ALLOCATOR polymorphic_allocator
#endif

  //
  // The following classes define smart pointer behavior.  
  // They rely on allocators for memory pooling and logging (if logging is on).
  //

  // This is an Obj<X>-specific exception that is thrown when incompatible object conversion is attempted.
  class BadCast : public Exception {
  public:
    explicit BadCast(const string&        msg) : Exception(msg) {};
    explicit BadCast(const ostringstream& txt) : Exception(txt) {};
    //  It actually looks like passing txt as an argument to Exception(ostringstream) performs a copy of txt, 
    //  which is disallowed due to the ostringstream constructor being private; must use a string constructor.
    BadCast(const BadCast& e)        : Exception(e) {};
  };

  // This is the main smart pointer class.
  template<class X> 
  class Obj {
  public:
    // Types 
#ifdef ALE_USE_LOGGING
    typedef logged_allocator<X,true>      Allocator;
    typedef logged_allocator<int,true>    Allocator_int;
#else
    typedef polymorphic_allocator<X>      Allocator;
    typedef polymorphic_allocator<int>    Allocator_int;
#endif
    typedef typename Allocator::size_type size_type;
  public:
    // These are intended to be private or at least protected
    // allocators
    Allocator_int          int_allocator;
    Allocator              allocator;
    //
    X*                     objPtr; // object pointer
    int*                   refCnt; // reference count
    size_type              sz;     // Size of underlying object (universal units) allocated with an allocator; indicates allocator use.
    // Constructor; this can be made private, if we move operator Obj<Y> outside this class definition and make it a friend.
    Obj(X *xx, int *refCnt, size_type sz);
  public:
    // Constructors & a destructor
    Obj() : objPtr((X *)NULL), refCnt((int*)NULL), sz(0) {};
    Obj(const X& x);
    Obj(X *xx);
    Obj(const Obj& obj);
    virtual ~Obj();

    // "Factory" methods
    Obj& create(const X& x = X());

    // predicates & assertions
    bool isNull() const {return (this->objPtr == NULL);};
    void assertNull(bool flag) const { if(this->isNull() != flag){ throw(Exception("Null assertion failed"));}};

    // comparison operators
    bool operator==(const Obj& obj) { return (this->objPtr == obj.objPtr);};
    bool operator!=(const Obj& obj) { return (this->objPtr != obj.objPtr);};
    
    // assignment/conversion operators
    Obj& operator=(const Obj& obj);
    template <class Y> operator Obj<Y> const();
    template <class Y> Obj& operator=(const Obj<Y>& obj);

    // dereference operators
    X*   operator->() const {return objPtr;};
    
    // "exposure" methods: expose the underlying object or object pointer
    operator X*() {return objPtr;};
    X& operator*() {assertNull(false); return *objPtr;};
    operator X()  {assertNull(false); return *objPtr;};
    template<class Y> Obj& copy(const Obj<Y>& obj); // this operator will copy the underlying objects: USE WITH CAUTION
    

    // depricated methods/operators
    X* ptr()      {return objPtr;};
    X* pointer()  {return objPtr;};
    X  obj()      {assertNull(false); return *objPtr;};
    X  object()   {assertNull(false); return *objPtr;};
    
  };// class Obj<X>


  // Constructors 
  // New reference
  template <class X>
  Obj<X>::Obj(const X& x) {
    this->refCnt = NULL;
    this->create(x);
  }
  
  // Stolen reference
  template <class X>
  Obj<X>::Obj(X *xx){// such an object will be destroyed by calling 'delete' on its pointer 
                     // (e.g., we assume the pointer was obtained with new)
    if (xx) {
      this->objPtr = xx; 
      this->refCnt = this->int_allocator.create(1);
      //this->refCnt   = new int(1);
      this->sz = 0;
    } else {
      this->objPtr = NULL; 
      this->refCnt = NULL;
      this->sz = 0;
    }
  }
  
  template <class X>
  Obj<X>::Obj(X *_xx, int *_refCnt, size_type _sz) {  // This is intended to be private.
    if (!_xx) {
      throw ALE::Exception("Making an Obj with a NULL objPtr");
    }
    this->objPtr = _xx;
    this->refCnt = _refCnt;  // we assume that all refCnt pointers are obtained using an int_allocator
    (*this->refCnt)++;
    this->sz = _sz;
    //if (!this->sz) {
    //  throw ALE::Exception("Making an Obj with zero size");
    //}
  }
  
  template <class X>
  Obj<X>::Obj(const Obj& obj) {
    this->objPtr = obj.objPtr;
    this->refCnt = obj.refCnt;
    if (obj.refCnt) {
      (*this->refCnt)++;
    }
    this->sz = obj.sz;
    //if (!this->sz) {
    //  throw ALE::Exception("Making an Obj with zero size");
    //}
  }

  // Destructor
  template <class X>
  Obj<X>::~Obj(){
    if(ALE::getVerbosity() > 3) {
#ifdef ALE_USE_DEBUGGING
      const char *id_name;
      const std::type_info& id = typeid(X);
#ifdef ALE_HAVE_CXX_ABI
      int status;
      char *id_name_demangled = abi::__cxa_demangle(id.name(), NULL, NULL, &status);
      id_name = id_name_demangled;
#else 
      id_name = id.name();
#endif
      printf("~Obj<X>: Calling destructor for Obj<%s>", id_name);
      if (!this->refCnt) {
        printf(" with no refCnt\n");
      } else {
        printf(" with refCnt %d\n", *this->refCnt);
      }
#ifdef ALE_HAVE_CXX_ABI
      free(id_name_demangled);
#endif
#endif
    }
    if (this->refCnt != NULL) {
      (*this->refCnt)--;
      if (*this->refCnt == 0) {
        // If  allocator has been used to create an objPtr, as indicated by 'sz', we use the allocator to delete objPtr, using 'sz'.
        if(this->sz != 0) {
#ifdef ALE_USE_DEBUGGING
          if(ALE::getVerbosity() > 3) {
            printf("  Calling deallocator on %p with size %d\n", this->objPtr, (int) this->sz);
          }
#endif
          this->allocator.del(this->objPtr, this->sz);
          this->sz = 0;
        }
        else { // otherwise we use 'delete'
#ifdef ALE_USE_DEBUGGING
          if(ALE::getVerbosity() > 3) {
            printf("  Calling delete on %p\n", this->objPtr);
          }
#endif
          if (!this->objPtr) {
            throw ALE::Exception("Trying to free NULL pointer");
          }
          delete this->objPtr;
        }
        // refCnt is always created/delete using the int_allocator.
        this->int_allocator.del(this->refCnt);
      }
    }
  }

  template <class X>
  Obj<X>& Obj<X>::create(const X& x) {
    // Destroy the old state
    this->~Obj<X>();
    // Create the new state
    this->objPtr = this->allocator.create(x); 
    this->refCnt = this->int_allocator.create(1);
    this->sz     = this->allocator.sz;
    if (!this->sz) {
      throw ALE::Exception("Making an Obj with zero size obtained from allocator");
    }
    return *this;
  }

  // assignment operator
  template <class X>
  Obj<X>& Obj<X>::operator=(const Obj<X>& obj) {
    if(this->objPtr == obj.objPtr) {return *this;}
    // Destroy 'this' Obj -- it will properly release the underlying object if the reference count is exhausted.
    if(this->objPtr) {
      this->~Obj<X>();
    }
    // Now copy the data from obj.
    this->objPtr = obj.objPtr;
    this->refCnt = obj.refCnt;
    if(this->refCnt!= NULL) {
      (*this->refCnt)++;
    }
    this->sz = obj.sz;
    return *this;
  }

  // conversion operator, preserves 'this'
  template<class X> template<class Y> 
  Obj<X>::operator Obj<Y> const() {
    // We attempt to cast X* objPtr to Y* using dynamic_
#ifdef ALE_USE_DEBUGGING
    if(ALE::getVerbosity() > 1) {
      printf("Obj<X>::operator Obj<Y>: attempting a dynamic_cast on objPtr %p\n", this->objPtr);
    }
#endif
    Y* yObjPtr = dynamic_cast<Y*>(this->objPtr);
    // If the cast failed, throw an exception
    if(yObjPtr == NULL) {
#ifdef ALE_HAVE_CXX_ABI 
      int status;
      const char *Xname = abi::__cxa_demangle(typeid(X).name(), NULL, NULL, &status);
      const char *Yname = abi::__cxa_demangle(typeid(Y).name(), NULL, NULL, &status);
#else
      const char *Xname = typeid(X).name();
      const char *Yname = typeid(Y).name();
#endif
      std::string msg("Bad cast Obj<");
      msg += Xname;
      msg += "> --> Obj<";
      msg += Yname;
      msg += ">";
      throw BadCast(msg.c_str());
    }
    // Okay, we can proceed 
    return Obj<Y>(yObjPtr, this->refCnt, this->sz);
  }

  // assignment-conversion operator
  template<class X> template<class Y> 
  Obj<X>& Obj<X>::operator=(const Obj<Y>& obj) {
    // We attempt to cast Y* obj.objPtr to X* using dynamic_cast
    X* xObjPtr = dynamic_cast<X*>(obj.objPtr);
    // If the cast failed, throw an exception
    if(xObjPtr == NULL) {
#ifdef ALE_HAVE_CXX_ABI 
      int status;
      const char *Xname = abi::__cxa_demangle(typeid(X).name(), NULL, NULL, &status);
      const char *Yname = abi::__cxa_demangle(typeid(Y).name(), NULL, NULL, &status);
#else
      const char *Xname = typeid(X).name();
      const char *Yname = typeid(Y).name();
#endif
      std::string msg("Bad assignment cast Obj<");
      msg += Yname;
      msg += "> --> Obj<";
      msg += Xname;
      msg += ">";
      throw BadCast(msg.c_str());
    }
    // Okay, we can proceed with the assignment
    if(this->objPtr == obj.objPtr) {return *this;}
    // Destroy 'this' Obj -- it will properly release the underlying object if the reference count is exhausted.
    this->~Obj<X>();
    // Now copy the data from obj.
    this->objPtr = xObjPtr;
    this->refCnt = obj.refCnt;
    (*this->refCnt)++;
    this->sz = obj.sz;
    return *this;
  }
 
  // copy operator (USE WITH CAUTION)
  template<class X> template<class Y> 
  Obj<X>& Obj<X>::copy(const Obj<Y>& obj) {
    if(this->isNull() || obj.isNull()) {
      throw(Exception("Copying to or from a null Obj"));
    }
    *(this->objPtr) = *(obj.objPtr);
    return *this;
  }


} // namespace ALE

#endif
