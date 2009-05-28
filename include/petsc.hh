#if !defined(__PETSC_HH)
#define __PETSC_HH

#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_EXTERN_CXX)
#include<map>

namespace PETSc {
  class Exception : public std::exception {
    std::ostringstream _txt;
  public:
    Exception() : std::exception() {}
    explicit Exception(const std::string& msg) : std::exception() {this->_txt << msg;}
    explicit Exception(const std::ostringstream& txt) : std::exception() {this->_txt << txt.str();}
    Exception(const Exception& e) : std::exception() {this->_txt << e._txt.str();}
    ~Exception() throw () {}
  public:
    const std::string msg() const {return this->_txt.str();}
    const char *message() const {return this->_txt.str().c_str();}
    /* Message input */
    template<typename Input>
    Exception& operator<<(const Input& in) {
      this->_txt << in;
      return *this;
    }
    /* Printing */
    template<typename Stream>
    friend Stream& operator<<(Stream& os, const Exception& e) {
      os << e.message() << std::endl;
      return os;
    }
  };

  template<int dim, typename Value_ = double>
  struct Point {
    typedef Value_ value_type;
    value_type x[dim];

    Point() {for(int d = 0; d < dim; ++d) {x[d] = 0.0;}}
    Point(const value_type p) {for(int d = 0; d < dim; ++d) {x[d] = p;}}
    Point(const value_type p[]) {for(int d = 0; d < dim; ++d) {x[d] = p[d];}}
    Point(const Point& p) {for(int d = 0; d < dim; ++d) {x[d] = p.x[d];}}
    template<typename Value2_>
    Point(const Point<dim,Value2_>& p) {for(int d = 0; d < dim; ++d) {x[d] = (Value2_) p.x[d];}}

    void operator=(value_type v) {for(int d = 0; d < dim; ++d) {x[d] = v;}}
    void operator=(const Point& p) {for(int d = 0; d < dim; ++d) {x[d] = p.x[d];}}
    template<int d>
    static bool lessThan(const Point& a, const Point &b) {
      return a.x[d] < b.x[d];
    }
    value_type operator[](const int i) const {return this->x[i];};
    value_type& operator[](const int i) {return this->x[i];};
    friend std::ostream& operator<<(std::ostream& stream, const Point& p) {
      for(int d = 0; d < dim; ++d) {
        if (d > 0) stream << ",";
        stream << p.x[d];
      }
      return stream;
    }
    Point& operator-() {
      for(int d = 0; d < dim; ++d) {x[d] = -x[d];}
      return *this;
    }
    friend Point operator+ (const Point& a, const Point &b) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] + b.x[d];}
      return tmp;
    }
    friend Point operator+ (const Point& a, const double c) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] + c;}
      return tmp;
    }
    friend Point operator- (const Point& a, const Point &b) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] - b.x[d];}
      return tmp;
    }
    friend Point operator* (const Point& a, const Point &b) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] * b.x[d];}
      return tmp;
    }
    friend Point operator* (const Point& a, const double c) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] * c;}
      return tmp;
    }
    friend Point operator/ (const Point& a, const Point &b) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] / b.x[d];}
      return tmp;
    }
  };

  class LogEvent {
  protected:
    std::string   name;
    PetscLogEvent id;
  public:
    LogEvent() : name(), id() {};
    LogEvent(const std::string& name, PetscLogEvent id) : name(name), id(id) {};
    LogEvent(const LogEvent& event) : name(event.name), id(event.id) {};
    void begin(PetscObject o1 = PETSC_NULL, PetscObject o2 = PETSC_NULL, PetscObject o3 = PETSC_NULL, PetscObject o4 = PETSC_NULL) {
      PetscErrorCode ierr = PetscLogEventBegin(this->id, o1, o2, o3, o4); CHKERRXX(ierr);
    };
    void end(PetscObject o1 = PETSC_NULL, PetscObject o2 = PETSC_NULL, PetscObject o3 = PETSC_NULL, PetscObject o4 = PETSC_NULL) {
      PetscErrorCode ierr = PetscLogEventEnd(this->id, o1, o2, o3, o4); CHKERRXX(ierr);
    };
    void barrierBegin(PetscObject o1 = PETSC_NULL, PetscObject o2 = PETSC_NULL, PetscObject o3 = PETSC_NULL, PetscObject o4 = PETSC_NULL, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) comm = PETSC_COMM_WORLD;
      PetscErrorCode ierr = PetscLogEventBarrierBegin(this->id, o1, o2, o3, o4, comm); CHKERRXX(ierr);
    };
    void barrierEnd(PetscObject o1 = PETSC_NULL, PetscObject o2 = PETSC_NULL, PetscObject o3 = PETSC_NULL, PetscObject o4 = PETSC_NULL, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) comm = PETSC_COMM_WORLD;
      PetscErrorCode ierr = PetscLogEventBarrierEnd(this->id, o1, o2, o3, o4, comm); CHKERRXX(ierr);
    };
  };

  class Log {
    static std::map<std::string,LogEvent> event_registry;
  public:
    static LogEvent& Event(const std::string& name, PetscCookie cookie = PETSC_OBJECT_COOKIE) {
      if (event_registry.find(name) == event_registry.end()) {
        PetscLogEvent  id;
        PetscErrorCode ierr;

        // Should check for already registered events
        ierr = PetscLogEventRegister(name.c_str(), cookie, &id);CHKERRXX(ierr);
        event_registry[name] = LogEvent(name, id);
      }
      return event_registry[name];
    };
  };
}
#endif /* PETSC_CLANGUAGE_CXX */

#endif /* __PETSC_HH */
