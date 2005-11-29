#ifndef included_ALE_mem_hh
#define included_ALE_mem_hh
// This should be included indirectly -- only by including ALE.hh


#include <memory>
#include <typeinfo>
#include <petsc.h>

namespace ALE {

  // General allocator; use it when no logging is necessary; create/del methods act as new/delete.
  template <class _T>
  class constructor_allocator : public std::allocator<_T> {
  public:
    constructor_allocator()                                    : std::allocator<_T>()  {};    
    constructor_allocator(const constructor_allocator& a)      : std::allocator<_T>(a) {};
    template <class _TT> 
    constructor_allocator(const constructor_allocator<_TT>& aa): std::allocator<_T>(aa){};
    ~constructor_allocator() {};

    _T* create();
    _T* create(const _T& val);
    void del(_T* p);
    // conversion typedef
    template <class _TT>
    struct rebind { typedef constructor_allocator<_TT> other;};
  };
  
  template <class _T> 
  _T* constructor_allocator<_T>::create(const _T& val) {
    // First, allocate space for a single object
    _T* p = std::allocator<_T>::allocate(1);
    // Construct an object in the provided space using the provided initial value
    std::allocator<_T>::construct(p,  val);
  }

  template <class _T> 
  _T* constructor_allocator<_T>::create() {
    // First, allocate space for a single object
    _T* p = std::allocator<_T>::allocate(1);
    // Construct an object in the provided space using the default initial value
    std::allocator<_T>::construct(p, _T());
  }

  template <class _T> 
  void constructor_allocator<_T>::del(_T* _p) {
    std::allocator<_T>::destroy(_p);
    std::allocator<_T>::deallocate(_p, 1);
  }



  // An allocator all of whose events (allocation, deallocation, new, delete) are logged using PetscLogging facilities.
  template <class _T>
  class logged_allocator : public constructor_allocator<_T> {
  private:
    static PetscCookie _cookie;
    static int         _allocate_event;
    static int         _deallocate_event;
    static int         _construct_event;
    static int         _destroy_event;
    static int         _create_event;
    static int         _del_event;
    static void __log_initialize();
    static void __log_event_register(const char *event_name, PetscEvent *event_ptr);
  public:
    typedef typename constructor_allocator<_T>::size_type size_type;
    logged_allocator()                                   : constructor_allocator<_T>()  {__log_initialize();};    
    logged_allocator(const logged_allocator& a)          : constructor_allocator<_T>(a) {__log_initialize();};
    template <class _TT> 
    logged_allocator(const logged_allocator<_TT>& aa)    : constructor_allocator<_T>(aa){__log_initialize();};
    ~logged_allocator() {};

    _T*  allocate(size_type _n);
    void deallocate(_T*  _p, size_type _n);
    void construct(_T* _p, const _T& _val);
    void destroy(_T* _p);
    _T*  create();
    _T*  create(const _T& _val);
    void del(_T*  _p);    
    // conversion typedef
    template <class _TT>
    struct rebind { typedef logged_allocator<_TT> other;};
  };

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
    // Get a new cookie based on _T's typeid name
    const std::type_info& id = typeid(_T);
    PetscErrorCode ierr = PetscLogClassRegister(&logged_allocator::_cookie, id.name()); 
    CHKERROR(ierr, "PetscLogClassRegister failed");
    // Register the basic allocator methods' invocations as events
    logged_allocator::__log_event_register("allocate", &logged_allocator::_allocate_event);
    logged_allocator::__log_event_register("deallocate", &logged_allocator::_deallocate_event);
    logged_allocator::__log_event_register("construct", &logged_allocator::_construct_event);
    logged_allocator::__log_event_register("destroy", &logged_allocator::_destroy_event);
    logged_allocator::__log_event_register("create", &logged_allocator::_create_event);
    logged_allocator::__log_event_register("del", &logged_allocator::_del_event);
  }


  template <class _T> 
  void logged_allocator<_T>::__log_event_register(const char *event_name, PetscEvent *event_ptr){
    const std::type_info& id = typeid(_T);
    ostringstream txt;
    txt << id.name() << ": " << event_name;
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
  _T* logged_allocator<_T>::create() {
    PetscErrorCode ierr;
    ierr = PetscLogEventBegin(logged_allocator::_create_event, 0, 0, 0, 0); CHKERROR(ierr, "Event begin failed");
    _T* _p = constructor_allocator<_T>::create();
    ierr = PetscLogEventEnd(logged_allocator::_create_event, 0, 0, 0, 0); CHKERROR(ierr, "Event end failed");
    return _p;
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

  // This should be inside an #ifdef clause and controlled through configure.
#define ALE_ALLOCATOR logged_allocator

  class BadCast : public Exception {
  public:
    BadCast(const char  *msg)   : Exception(msg) {};
    BadCast(const BadCast& e)   : Exception(e) {};
  };

  template<class X> 
  class Obj {
  public:
    X*       objPtr;       // object pointer
    int32_t *refCnt;       // reference count
    int      borrowed;     // indicates that the object should not be released
  public:
    // constructors
    Obj() : objPtr((X *)NULL), refCnt((int32_t*)NULL), borrowed(0) {};
    Obj(X x) { // such an object won't be destroyed (e.g., an object allocated on the stack)
      //this->objPtr = &x;
      this->objPtr = new X(x);
      this->borrowed=0; 
      refCnt = new int(1);
    };
    Obj(X *xx){// such an object will be destroyed by calling 'delete' on its pointer (e.g., pointer obtained with new)
      this->objPtr = xx; 
      this->borrowed=0;
      refCnt = new int(1);
    }

    Obj(X *xx, int32_t *refCnt) {
      this->objPtr = xx;
      this->borrowed=0;
      this->refCnt = refCnt;
      (*this->refCnt)++;
    };

    Obj(const Obj& obj) {
      //// We disallow constructors from borrowed objects, to prevent their being returned from function calls.
      //if(obj.borrowed) {
      //  throw Exception("Cannot clone a borrowed object");
      //}
      this->objPtr = obj.objPtr;
      this->refCnt = obj.refCnt;
      (*this->refCnt)++;
      this->borrowed = obj.borrowed;
    };
    
    // check whether Obj points to a NULL object
    int isNull() {
      return (this->objPtr == NULL);
    };

    // assertion that throws an exception if objPtr is/isn't null
    void assertNull(bool flag) {
      if(this->isNull() != flag){
        throw(Exception("Null assertion failed"));
      }
    };

    // comparison operators
    int operator==(const Obj& obj) {
      return (this->objPtr == obj.objPtr);
    };
    int operator!=(const Obj& obj) {
      return (this->objPtr != obj.objPtr);
    };

    // assignment operator
    Obj& operator=(const Obj& obj) {
      if (borrowed) {
        throw ALE::Exception("Borrowed should never be nonzero");
      }
      if(this->objPtr == obj.objPtr) {return *this;}
      // We are letting go of objPtr, so need to check whether we are the last reference holder.
      if((this->refCnt != (int32_t *)NULL) && (--(*this->refCnt) == 0) && !this->borrowed)  {
        delete this->objPtr;
        delete this->refCnt;
      }
      this->objPtr = obj.objPtr;
      this->refCnt = obj.refCnt;
      if(this->refCnt!= NULL) {
        (*this->refCnt)++;
      }
      this->borrowed = obj.borrowed;
      return *this;
    };

    // conversion operator
    template<class Y> operator Obj<Y>() {
      // We attempt to cast X* objPtr to Y* using dynamic_cast
      Y* yObjPtr = dynamic_cast<Y*>(this->objPtr);
      // If the cast failed, throw an exception
      if(yObjPtr == NULL) {
        throw ALE::Exception("Bad cast Obj<X> --> Obj<Y>");
      }
      // Okay, we can proceed 
      return Obj<Y>(yObjPtr, this->refCnt);
    }

    // another conversion operator
    template<class Y> Obj& operator=(const Obj<Y>& obj) {
      // We attempt to cast Y* obj.objPtr to X* using dynamic_cast
      X* xObjPtr = dynamic_cast<X*>(obj.objPtr);
      // If the cast failed, throw an exception
      if(xObjPtr == NULL) {
        throw BadCast("Bad cast Obj<Y> --> Obj<X>");
      }
      // Okay, we can proceed with the assignment
      if(this->objPtr == obj.objPtr) {return *this;}
      // We are letting go of objPtr, so need to check whether we are the last reference holder.
      if((this->refCnt != (int32_t *)NULL) && (--(*this->refCnt) == 0) ) {
        delete this->objPtr;
        delete this->refCnt;
      }
      this->objPtr = xObjPtr;
      this->refCnt = obj.refCnt;
      (*this->refCnt)++;
      this->borrowed = obj.borrowed;
      return *this;
    }


    // dereference operators
    X*   operator->() {return objPtr;};
    //
    template<class Y> Obj& copy(Obj<Y>& obj) {
      if(this->isNull() || obj.isNull()) {
        throw(Exception("Copying to or from a null Obj"));
      }
      *(this->objPtr) = *(obj.objPtr);
      return *this;
    }
    

    // "peeling" (off the shell) methods
    X* ptr()      {return objPtr;};
    X* pointer()  {return objPtr;};
    operator X*() {return this->pointer();};
    X  obj()      {assertNull(0); return *objPtr;};
    X  object()   {assertNull(0); return *objPtr;};
    X operator*() {assertNull(0); return *objPtr;};
    operator X()  {return this->object();};
    

    // destructor
    ~Obj(){
      if((this->refCnt != (int32_t *)NULL) && (--(this->refCnt) == 0) && !this->borrowed) {  
        delete objPtr; 
        delete refCnt;
      }
    };
    
  };// class Obj<X>


} // namespace ALE

#endif
