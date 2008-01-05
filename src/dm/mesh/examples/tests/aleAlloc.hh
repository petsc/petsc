#define ALE_HAVE_CXX_ABI

#ifdef ALE_HAVE_CXX_ABI
#include <cxxabi.h>
#endif

namespace ALE {
  class MemoryLogger {
  public:
    struct Log {
      int num;
      int total;
      std::map<std::string, int> items;

      Log(): num(0), total(0) {};
    };
    typedef std::map<std::string, std::pair<Log, Log> > stageLog;
  protected:
    stageLog stages;
  protected:
    MemoryLogger() {};
  public:
    ~MemoryLogger() {};
    static MemoryLogger& singleton() {
      static MemoryLogger singleton;

      return singleton;
    };
  public:
    void logAllocation(const std::string& className, int bytes) {logAllocation("default", className, bytes);};
    void logAllocation(const std::string& stage, const std::string& className, int bytes) {
      stages[stage].first.num++;
      stages[stage].first.total += bytes;
      stages[stage].first.items[className] += bytes;
    };
    void logDeallocation(const std::string& className, int bytes) {logDeallocation("default", className, bytes);};
    void logDeallocation(const std::string& stage, const std::string& className, int bytes) {
      stages[stage].second.num++;
      stages[stage].second.total += bytes;
      stages[stage].second.items[className] += bytes;
    };
  public:
    int getNumAllocations() {return getNumAllocations("default");};
    int getNumAllocations(const std::string& stage) {return stages[stage].first.num;};
    int getNumDeallocations() {return getNumDeallocations("default");};
    int getNumDeallocations(const std::string& stage) {return stages[stage].second.num;};
    int getAllocationTotal() {return getAllocationTotal("default");};
    int getAllocationTotal(const std::string& stage) {return stages[stage].first.total;};
    int getDeallocationTotal() {return getDeallocationTotal("default");};
    int getDeallocationTotal(const std::string& stage) {return stages[stage].second.total;};
  public:
    template<typename T>
    static const char *getClassName() {
      const std::type_info& id = typeid(T);

#ifdef ALE_HAVE_CXX_ABI
      // If the C++ ABI API is available, we can use it to demangle the class name provided by type_info.
      // Here we assume the industry standard C++ ABI as described in http://www.codesourcery.com/cxx-abi/abi.html.
      char *id_name;
      int   status;
      char *id_name_demangled = abi::__cxa_demangle(id.name(), NULL, NULL, &status);

      if (status != 0) {
        id_name = new char[strlen(id.name())+1];
        strcpy(id_name, id.name());
      } else {
        id_name = id_name_demangled;
      }
#else
      const char *id_name;

      id_name = id.name();
#endif
      return id_name;
    };
    static void restoreClassName(const char *className) {
#ifdef ALE_HAVE_CXX_ABI
      // Free the name malloc'ed by __cxa_demangle
      free(const_cast<char *>(className));
#endif
    };
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
    const char *className;
  public:
    malloc_allocator() {className = ALE::MemoryLogger::getClassName<T>();}
    malloc_allocator(const malloc_allocator&) {className = ALE::MemoryLogger::getClassName<T>();}
    template <class U> 
    malloc_allocator(const malloc_allocator<U>&) {className = ALE::MemoryLogger::getClassName<T>();}
    ~malloc_allocator() {ALE::MemoryLogger::restoreClassName(className);}
  public:
    pointer address(reference x) const {return &x;}
    const_pointer address(const_reference x) const {return x;}

    pointer allocate(size_type n, const_pointer = 0) {
      ALE::MemoryLogger::singleton().logAllocation(className, n * sizeof(T));
      void *p = std::malloc(n * sizeof(T));
      if (!p) throw std::bad_alloc();
      return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type n) {
      ALE::MemoryLogger::singleton().logDeallocation(className, n * sizeof(T));
      std::free(p);
    }

    size_type max_size() const {return static_cast<size_type>(-1) / sizeof(T);}

    void construct(pointer p, const value_type& x) {new(p) value_type(x);}

    void destroy(pointer p) {p->~value_type();}
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
  }

  template <class T>
  inline bool operator!=(const malloc_allocator<T>&, const malloc_allocator<T>&) {
    return false;
  }
}
