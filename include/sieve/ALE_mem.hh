#ifndef included_ALE_mem_hh
#define included_ALE_mem_hh
// This should be included indirectly -- only by including ALE.hh

#include <assert.h>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <cstdlib>
#include <typeinfo>
#include <petscsys.h>
#include <sieve/ALE_log.hh>

#ifdef ALE_HAVE_CXX_ABI
#include <cxxabi.h>
#endif

namespace ALE {
  class MemoryLogger;
}
extern ALE::MemoryLogger Petsc_MemoryLogger;

namespace ALE {
  class MemoryLogger {
  public:
    struct Log {
      long long num;
      long long total;
      std::map<std::string, long long> items;

      Log(): num(0), total(0) {};
    };
    typedef std::map<std::string, std::pair<Log, Log> > stageLog;
    typedef std::deque<std::string>                     names;
  protected:
    int      _debug;
    MPI_Comm _comm;
    int      rank;
    names    stageNames;
    stageLog stages;
  public:
    MemoryLogger(): _debug(0), _comm(MPI_COMM_NULL), rank(-1) {
      stageNames.push_front("default");
    };
  public:
    ~MemoryLogger() {};
    static MemoryLogger& singleton() {
      if (Petsc_MemoryLogger.comm() == MPI_COMM_NULL) {
        Petsc_MemoryLogger.setComm(PETSC_COMM_WORLD);
      }
      return Petsc_MemoryLogger;
    };
    int  debug() {return _debug;};
    void setDebug(int debug) {_debug = debug;};
    MPI_Comm comm() {return _comm;};
    void     setComm(MPI_Comm comm) {
      _comm = comm;
      MPI_Comm_rank(_comm, &rank);
    };
  public:
    void stagePush(const std::string& name) {
      for(names::const_iterator s_iter = stageNames.begin(); s_iter != stageNames.end(); ++s_iter) {
        if (*s_iter == name) throw ALE::Exception(std::string("Cannot push duplicate stage name '")+name+std::string("'."));
      }
      stageNames.push_front(name);
      if (_debug) {
        std::cout << "["<<rank<<"]Pushing stage " << name << ":" << std::endl;
        for(names::const_iterator s_iter = stageNames.begin(); s_iter != stageNames.end(); ++s_iter) {
          std::cout << "["<<rank<<"]  " << *s_iter << ": " << stages[*s_iter].first.num  << " acalls  " << stages[*s_iter].first.total  << " bytes" << std::endl;
          std::cout << "["<<rank<<"]  " << *s_iter << ": " << stages[*s_iter].second.num << " dcalls  " << stages[*s_iter].second.total << " bytes" << std::endl;
        }
      }
    };
    void stagePop() {
      if (_debug) {
        std::cout << "["<<rank<<"]Popping stage " << stageNames.front() << ":" << std::endl;
        for(names::const_iterator s_iter = stageNames.begin(); s_iter != stageNames.end(); ++s_iter) {
          std::cout << "["<<rank<<"]  " << *s_iter << ": " << stages[*s_iter].first.num  << " acalls  " << stages[*s_iter].first.total  << " bytes" << std::endl;
          std::cout << "["<<rank<<"]  " << *s_iter << ": " << stages[*s_iter].second.num << " dcalls  " << stages[*s_iter].second.total << " bytes" << std::endl;
        }
      }
      stageNames.pop_front();
    };
    void logAllocation(const std::string& className, int bytes) {
      for(names::const_iterator s_iter = stageNames.begin(); s_iter != stageNames.end(); ++s_iter) {
        logAllocation(*s_iter, className, bytes);
      }
    };
    void logAllocation(const std::string& stage, const std::string& className, int bytes) {
      if (_debug > 1) {std::cout << "["<<rank<<"]Allocating " << bytes << " bytes for class " << className << std::endl;}
      stages[stage].first.num++;
      stages[stage].first.total += bytes;
      stages[stage].first.items[className] += bytes;
    };
    void logDeallocation(const std::string& className, int bytes) {
      for(names::const_iterator s_iter = stageNames.begin(); s_iter != stageNames.end(); ++s_iter) {
        logDeallocation(*s_iter, className, bytes);
      }
    };
    void logDeallocation(const std::string& stage, const std::string& className, int bytes) {
      if (_debug > 1) {std::cout << "["<<rank<<"]Deallocating " << bytes << " bytes for class " << className << std::endl;}
      stages[stage].second.num++;
      stages[stage].second.total += bytes;
      stages[stage].second.items[className] += bytes;
    };
  public:
    long long getNumAllocations() {return getNumAllocations(stageNames.front());};
    long long getNumAllocations(const std::string& stage) {return stages[stage].first.num;};
    long long getNumDeallocations() {return getNumDeallocations(stageNames.front());};
    long long getNumDeallocations(const std::string& stage) {return stages[stage].second.num;};
    long long getAllocationTotal() {return getAllocationTotal(stageNames.front());};
    long long getAllocationTotal(const std::string& stage) {return stages[stage].first.total;};
    long long getDeallocationTotal() {return getDeallocationTotal(stageNames.front());};
    long long getDeallocationTotal(const std::string& stage) {return stages[stage].second.total;};
  public:
    void show() {
      std::cout << "["<<rank<<"]Memory Stages:" << std::endl;
      for(stageLog::const_iterator s_iter = stages.begin(); s_iter != stages.end(); ++s_iter) {
        std::cout << "["<<rank<<"]  " << s_iter->first << ": " << s_iter->second.first.num  << " acalls  " << s_iter->second.first.total  << " bytes" << std::endl;
        for(std::map<std::string, long long>::const_iterator i_iter = s_iter->second.first.items.begin(); i_iter != s_iter->second.first.items.end(); ++i_iter) {
          std::cout << "["<<rank<<"]    " << i_iter->first << ": " << i_iter->second << " bytes" << std::endl;
        }
        std::cout << "["<<rank<<"]  " << s_iter->first << ": " << s_iter->second.second.num << " dcalls  " << s_iter->second.second.total << " bytes" << std::endl;
        for(std::map<std::string, long long>::const_iterator i_iter = s_iter->second.second.items.begin(); i_iter != s_iter->second.second.items.end(); ++i_iter) {
          std::cout << "["<<rank<<"]    " << i_iter->first << ": " << i_iter->second << " bytes" << std::endl;
        }
      }
    };
  public:
    template<typename T>
    static const char *getClassName() {
      const std::type_info& id      = typeid(T);
      char                 *id_name = const_cast<char *>(id.name());

#ifdef ALE_HAVE_CXX_ABI
      // If the C++ ABI API is available, we can use it to demangle the class name provided by type_info.
      // Here we assume the industry standard C++ ABI as described in http://www.codesourcery.com/cxx-abi/abi.html.
      int   status;
      char *id_name_demangled = abi::__cxa_demangle(id.name(), NULL, NULL, &status);

      if (!status) {
        id_name = id_name_demangled;
      }
#endif
      return id_name;
    }
    static void restoreClassName(const char * /* className */) {};
  };

  template<class T>
  class malloc_allocator
  {
  public:
    typedef T                 value_type;
    typedef value_type*       pointer;
    typedef const value_type* const_pointer;
    typedef value_type&       reference;
    typedef const value_type& const_reference;
    typedef std::size_t       size_type;
    typedef std::ptrdiff_t    difference_type;
  public:
    template <class U>
    struct rebind {typedef malloc_allocator<U> other;};
  protected:
    int         numAllocs;
    const char *className;
  public:
    int sz;
  public:
#ifdef ALE_MEM_LOGGING
    malloc_allocator() : numAllocs(0) {className = ALE::MemoryLogger::getClassName<T>();sz = sizeof(value_type);}
    malloc_allocator(const malloc_allocator&) : numAllocs(0) {className = ALE::MemoryLogger::getClassName<T>();sz = sizeof(value_type);}
    template <class U> 
    malloc_allocator(const malloc_allocator<U>&) : numAllocs(0) {className = ALE::MemoryLogger::getClassName<T>();sz = sizeof(value_type);}
    ~malloc_allocator() {ALE::MemoryLogger::restoreClassName(className);}
#else
    malloc_allocator() : numAllocs(0) {sz = sizeof(value_type);}
    malloc_allocator(const malloc_allocator&) : numAllocs(0) {sz = sizeof(value_type);}
    template <class U> 
    malloc_allocator(const malloc_allocator<U>&) : numAllocs(0) {sz = sizeof(value_type);}
    ~malloc_allocator() {}
#endif
  public:
    pointer address(reference x) const {return &x;}
    // For some reason the goddamn MS compiler does not like this function
    //const_pointer address(const_reference x) const {return &x;}

    pointer allocate(size_type n, const_pointer = 0) {
      assert(n >= 0);
#ifdef ALE_MEM_LOGGING
      ALE::MemoryLogger::singleton().logAllocation(className, n * sizeof(T));
#endif
      numAllocs++;
      void *p = std::malloc(n * sizeof(T));
      if (!p) throw std::bad_alloc();
      return static_cast<pointer>(p);
    }

#ifdef ALE_MEM_LOGGING
    void deallocate(pointer p, size_type n) {
      ALE::MemoryLogger::singleton().logDeallocation(className, n * sizeof(T));
      std::free(p);
    }
#else
    void deallocate(pointer p, size_type) {
      std::free(p);
    }
#endif

    size_type max_size() const {return static_cast<size_type>(-1) / sizeof(T);}

    void construct(pointer p, const value_type& x) {new(p) value_type(x);}

    void destroy(pointer p) {p->~value_type();}
  public:
    pointer create(const value_type& x = value_type()) {
      pointer p = (pointer) allocate(1);
      construct(p, x);
      return p;
    };

    void del(pointer p) {
      destroy(p);
      deallocate(p, 1);
    };

    // This is just to be compatible with Dmitry's weird crap for now
    void del(pointer p, size_type size) {
      if (size != sizeof(value_type)) throw std::exception();
      destroy(p);
      deallocate(p, 1);
    };
  private:
    void operator=(const malloc_allocator&);
  };

  template<> class malloc_allocator<void>
  {
    typedef void        value_type;
    typedef void*       pointer;
    typedef const void* const_pointer;

    template <class U> 
    struct rebind {typedef malloc_allocator<U> other;};
  };

  template <class T>
  inline bool operator==(const malloc_allocator<T>&, const malloc_allocator<T>&) {
    return true;
  };

  template <class T>
  inline bool operator!=(const malloc_allocator<T>&, const malloc_allocator<T>&) {
    return false;
  };

  template <class T>
  static const char *getClassName() {
    const std::type_info& id = typeid(T);
    const char *id_name;

#ifdef ALE_HAVE_CXX_ABI
      // If the C++ ABI API is available, we can use it to demangle the class name provided by type_info.
      // Here we assume the industry standard C++ ABI as described in http://www.codesourcery.com/cxx-abi/abi.html.
      int   status;
      char *id_name_demangled = abi::__cxa_demangle(id.name(), NULL, NULL, &status);

      if (status != 0) {
        id_name = id.name();
      } else {
        id_name = id_name_demangled;
      }
#else
      id_name = id.name();
#endif
      return id_name;
  };
  template <class T>
  static const char *getClassName(const T * /* obj */) {
    return getClassName<T>();
  };
#ifdef ALE_HAVE_CXX_ABI
  template<class T>
  static void restoreClassName(const char *id_name) {
    // Free the name malloc'ed by __cxa_demangle
    free((char *) id_name);
  };
#else
  template<class T>
  static void restoreClassName(const char *) {};
#endif
  template<class T>
  static void restoreClassName(const T * /* obj */, const char *id_name) {restoreClassName<T>(id_name);};

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
    polymorphic_allocator(const polymorphic_allocator<TT>& aa){}
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
    logged_allocator(const logged_allocator<TT>& aa)    : polymorphic_allocator<T>(aa){__log_initialize(); __alloc_initialize();}
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
      PetscBool      flag;
      PetscErrorCode ierr = PetscInitialized(&flag);CHKERROR(ierr, "Error in PetscInitialized");
      if(!flag) {
        // I guess it would be nice to initialize PETSc here, but we'd need argv/argc here
        throw ALE::Exception("PETSc not initialized");
      }
      // Get a new cookie based on the class name
      const char *id_name = ALE::getClassName<T>();
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
      ALE::restoreClassName<T>(id_name);
      logged_allocator::_log_initialized = true;
    }// if(!!logged_allocator::_log_initialized)
  }// logged_allocator<T,O>::__log_initialize()

  template <class T, bool O>
  void logged_allocator<T, O>::__alloc_initialize() {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    const char *id_name = ALE::getClassName<T>();
    ALE::restoreClassName<T>(id_name);
#endif
  }// logged_allocator<T,O>::__alloc_initialize

  template <class T, bool O>
  void logged_allocator<T, O>::__alloc_finalize() {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
#endif
  }// logged_allocator<T,O>::__alloc_finalize

  template <class T, bool O> 
  LogEvent logged_allocator<T, O>::__log_event_register(const char *class_name, const char *event_name){
    // This routine assumes a cookie has been obtained.
    ostringstream txt;
    if(O) {
      txt << "Obj:";
    }
#ifdef ALE_LOGGING_VERBOSE
    txt << class_name;
#else
    txt << "<allocator>";
#endif
    txt << ":" << event_name;
    return LogEventRegister(logged_allocator::_cookie, txt.str().c_str());
  }

  template <class T, bool O>
  T*  logged_allocator<T, O>::allocate(size_type _n) {
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
    LogEventBegin(logged_allocator::_allocate_event); 
#endif
    T* _p = polymorphic_allocator<T>::allocate(_n);
#if defined ALE_USE_LOGGING && defined ALE_LOGGING_LOG_MEM
//     PetscErrorCode ierr = PetscLogObjectMemory(this->_petscObj, _n*polymorphic_allocator<T>::sz); 
//     CHKERROR(ierr, "Error in PetscLogObjectMemory");
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
//     PetscErrorCode ierr = PetscLogObjectMemory(this->_petscObj, polymorphic_allocator<T>::sz); 
//     CHKERROR(ierr, "Error in PetscLogObjectMemory");
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
#define ALE_ALLOCATOR ::ALE::logged_allocator
#else
#if 1
#define ALE_ALLOCATOR ::ALE::malloc_allocator
#else
#define ALE_ALLOCATOR ::ALE::polymorphic_allocator
#endif
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
  template<class X, typename A = malloc_allocator<X> > 
  class Obj {
  public:
    // Types
#if 1
    typedef A                                               Allocator;
    typedef typename Allocator::template rebind<int>::other Allocator_int;
#else
#ifdef ALE_USE_LOGGING
    typedef logged_allocator<X,true>      Allocator;
    typedef logged_allocator<int,true>    Allocator_int;
#else
    typedef polymorphic_allocator<X>      Allocator;
    typedef polymorphic_allocator<int>    Allocator_int;
#endif
#endif
    typedef typename Allocator::size_type size_type;
  protected:
    Allocator& allocator() {
      static Allocator _allocator;

      return _allocator;
    };
    Allocator_int& int_allocator() {
      static Allocator_int _allocator;

      return _allocator;
    };
  public:
    X*        objPtr; // object pointer
    int*      refCnt; // reference count
    size_type sz;     // Size of underlying object (universal units) allocated with an allocator; indicates allocator use.
    // Constructor; this can be made private, if we move operator Obj<Y> outside this class definition and make it a friend.
    Obj(X *xx, int *refCnt, size_type sz);
  public:
    // Constructors & a destructor
    Obj() : objPtr((X *)NULL), refCnt((int*)NULL), sz(0) {};
    Obj(const X& x);
    Obj(X *xx);
    Obj(X *xx, size_type sz);
    Obj(const Obj& obj);
    virtual ~Obj();

    // "Factory" methods
    Obj& create(const X& x = X());
    void destroy();

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
    X& operator*() const {assertNull(false); return *objPtr;};
    operator X()  {assertNull(false); return *objPtr;};
    template<class Y> Obj& copy(const Obj<Y>& obj); // this operator will copy the underlying objects: USE WITH CAUTION
    

    // depricated methods/operators
    X* ptr() const     {return objPtr;};
    X* pointer() const {return objPtr;};
    X  obj() const     {assertNull(false); return *objPtr;};
    X  object() const  {assertNull(false); return *objPtr;};

    void addRef() {if (refCnt) {(*refCnt)++;}}
  };// class Obj<X>

  // Constructors 
  // New reference
  template <class X, typename A>
  Obj<X,A>::Obj(const X& x) {
    this->refCnt = NULL;
    this->create(x);
  }
  
  // Stolen reference
  template <class X, typename A>
  Obj<X,A>::Obj(X *xx){// such an object will be destroyed by calling 'delete' on its pointer 
                     // (e.g., we assume the pointer was obtained with new)
    if (xx) {
      this->objPtr = xx; 
      this->refCnt = int_allocator().create(1);
      //this->refCnt   = new int(1);
      this->sz = 0;
    } else {
      this->objPtr = NULL; 
      this->refCnt = NULL;
      this->sz = 0;
    }
  }

  // Work around for thing allocated with an allocator
  template <class X, typename A>
  Obj<X,A>::Obj(X *xx, size_type sz){// such an object will be destroyed by the allocator
    if (xx) {
      this->objPtr = xx; 
      this->refCnt = int_allocator().create(1);
      this->sz     = sz;
    } else {
      this->objPtr = NULL; 
      this->refCnt = NULL;
      this->sz = 0;
    }
  }
  
  template <class X, typename A>
  Obj<X,A>::Obj(X *_xx, int *_refCnt, size_type _sz) {  // This is intended to be private.
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
  
  template <class X, typename A>
  Obj<X,A>::Obj(const Obj& obj) {
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
  template <class X, typename A>
  Obj<X,A>::~Obj(){
    this->destroy();
  }

  template <class X, typename A>
  Obj<X,A>& Obj<X,A>::create(const X& x) {
    // Destroy the old state
    this->destroy();
    // Create the new state
    this->objPtr = allocator().create(x); 
    this->refCnt = int_allocator().create(1);
    this->sz     = allocator().sz;
    if (!this->sz) {
      throw ALE::Exception("Making an Obj with zero size obtained from allocator");
    }
    return *this;
  }

  template <class X, typename A>
  void Obj<X,A>::destroy() {
    if(ALE::getVerbosity() > 3) {
#ifdef ALE_USE_DEBUGGING
      const char *id_name = ALE::getClassName<X>();

      printf("Obj<X>.destroy: Destroying Obj<%s>", id_name);
      if (!this->refCnt) {
        printf(" with no refCnt\n");
      } else {
        printf(" with refCnt %d\n", *this->refCnt);
      }
      ALE::restoreClassName<X>(id_name);
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
          allocator().del(this->objPtr, this->sz);
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
        int_allocator().del(this->refCnt);
        this->objPtr = NULL;
        this->refCnt = NULL;
      }
    }
  }

  // assignment operator
  template <class X, typename A>
  Obj<X,A>& Obj<X,A>::operator=(const Obj<X,A>& obj) {
    if(this->objPtr == obj.objPtr) {return *this;}
    // Destroy 'this' Obj -- it will properly release the underlying object if the reference count is exhausted.
    if(this->objPtr) {
      this->destroy();
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
  template<class X, typename A> template<class Y> 
  Obj<X,A>::operator Obj<Y> const() {
    // We attempt to cast X* objPtr to Y* using dynamic_
#ifdef ALE_USE_DEBUGGING
    if(ALE::getVerbosity() > 1) {
      printf("Obj<X>::operator Obj<Y>: attempting a dynamic_cast on objPtr %p\n", this->objPtr);
    }
#endif
    Y* yObjPtr = dynamic_cast<Y*>(this->objPtr);
    // If the cast failed, throw an exception
    if(yObjPtr == NULL) {
      const char *Xname = ALE::getClassName<X>();
      const char *Yname = ALE::getClassName<Y>();
      std::string msg("Bad cast Obj<");
      msg += Xname;
      msg += "> --> Obj<";
      msg += Yname;
      msg += ">";
      ALE::restoreClassName<X>(Xname);
      ALE::restoreClassName<X>(Yname);
      throw BadCast(msg.c_str());
    }
    // Okay, we can proceed 
    return Obj<Y>(yObjPtr, this->refCnt, this->sz);
  }

  // assignment-conversion operator
  template<class X, typename A> template<class Y> 
  Obj<X,A>& Obj<X,A>::operator=(const Obj<Y>& obj) {
    // We attempt to cast Y* obj.objPtr to X* using dynamic_cast
    X* xObjPtr = dynamic_cast<X*>(obj.objPtr);
    // If the cast failed, throw an exception
    if(xObjPtr == NULL) {
      const char *Xname = ALE::getClassName<X>();
      const char *Yname = ALE::getClassName<Y>();
      std::string msg("Bad assignment cast Obj<");
      msg += Yname;
      msg += "> --> Obj<";
      msg += Xname;
      msg += ">";
      ALE::restoreClassName<X>(Xname);
      ALE::restoreClassName<X>(Yname);
      throw BadCast(msg.c_str());
    }
    // Okay, we can proceed with the assignment
    if(this->objPtr == obj.objPtr) {return *this;}
    // Destroy 'this' Obj -- it will properly release the underlying object if the reference count is exhausted.
    this->destroy();
    // Now copy the data from obj.
    this->objPtr = xObjPtr;
    this->refCnt = obj.refCnt;
    (*this->refCnt)++;
    this->sz = obj.sz;
    return *this;
  }
 
  // copy operator (USE WITH CAUTION)
  template<class X, typename A> template<class Y> 
  Obj<X,A>& Obj<X,A>::copy(const Obj<Y>& obj) {
    if(this->isNull() || obj.isNull()) {
      throw(Exception("Copying to or from a null Obj"));
    }
    *(this->objPtr) = *(obj.objPtr);
    return *this;
  }


} // namespace ALE

#endif
