#if !defined(__PETSC_HH)
#define __PETSC_HH

#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_EXTERN_CXX)

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
    template<typename Value2_>
    Point(const Point<dim,Value2_>& p) {for(int d = 0; d < dim; ++d) {x[d] = (Value2_) p.x[d];}}

    inline int size() const {return dim;};

    inline operator value_type *() {return x;};

    inline void operator=(value_type v) {for(int d = 0; d < dim; ++d) {x[d] = v;}}
    inline void operator=(const Point& p) {for(int d = 0; d < dim; ++d) {x[d] = p.x[d];}}
    inline bool operator==(const Point& p) {for(int d = 0; d < dim; ++d) {if (x[d] != p.x[d]) return false;} return true;}
    inline void operator+=(const Point& p) {for(int d = 0; d < dim; ++d) {x[d] += p.x[d];}}
    inline void operator-=(const Point& p) {for(int d = 0; d < dim; ++d) {x[d] -= p.x[d];}}
    template<int d>
    static bool lessThan(const Point& a, const Point &b) {
      return a.x[d] < b.x[d];
    }
    inline value_type operator[](const int i) const {return this->x[i];};
    inline value_type& operator[](const int i) {return this->x[i];};
    friend std::ostream& operator<<(std::ostream& stream, const Point& p) {
      for(int d = 0; d < dim; ++d) {
        if (d > 0) stream << ",";
        stream << p.x[d];
      }
      return stream;
    }
    inline Point& operator-() {
      for(int d = 0; d < dim; ++d) {x[d] = -x[d];}
      return *this;
    }
    inline friend Point operator+ (const Point& a, const Point &b) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] + b.x[d];}
      return tmp;
    }
    inline friend Point operator+ (const Point& a, const double c) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] + c;}
      return tmp;
    }
    inline friend Point operator- (const Point& a, const Point &b) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] - b.x[d];}
      return tmp;
    }
    inline friend Point operator* (const Point& a, const Point &b) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] * b.x[d];}
      return tmp;
    }
    inline friend Point operator* (const Point& a, const double c) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] * c;}
      return tmp;
    }
    inline friend Point operator/ (const Point& a, const Point &b) {
      Point tmp;
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] / b.x[d];}
      return tmp;
    }
  };
}
#endif /* PETSC_CLANGUAGE_CXX */

#endif /* __PETSC_HH */
