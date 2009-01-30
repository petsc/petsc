#if !defined(__PETSC_HH)
#define __PETSC_HH

#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_EXTERN_CXX)
namespace PETSc {
  class Exception : public std::exception {
    std::ostringstream _txt;
  public:
    Exception() : std::exception() {};
    explicit Exception(const std::string& msg) : std::exception() {this->_txt << msg;};
    explicit Exception(const std::ostringstream& txt) : std::exception() {this->_txt << txt.str();};
    Exception(const Exception& e) : std::exception() {this->_txt << e._txt.str();};
    ~Exception() throw () {};
  public:
    const std::string msg() const {return this->_txt.str();};
    const char *message() const {return this->_txt.str().c_str();};
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

  template<int dim>
  struct Point {
    double x[dim];
    Point() {for(int d = 0; d < dim; ++d) {x[d] = 0.0;}};
    Point(const int p) {for(int d = 0; d < dim; ++d) {x[d] = (double) p;}};
    Point(const double p) {for(int d = 0; d < dim; ++d) {x[d] = p;}};
    Point(const double p[]) {for(int d = 0; d < dim; ++d) {x[d] = p[d];}};
    Point(const Point& p) {for(int d = 0; d < dim; ++d) {x[d] = p.x[d];}};

    void operator=(const Point& p) {for(int d = 0; d < dim; ++d) {x[d] = p.x[d];}};
    template<int d>
    static bool lessThan(const Point& a, const Point &b) {
      return a.x[d] < b.x[d];
    }
    double operator[](const int i) const {return this->x[i];};
    friend std::ostream& operator<<(std::ostream& stream, const Point& p) {
      for(int d = 0; d < dim; ++d) {
        if (d > 0) stream << ",";
        stream << p.x[d];
      }
      return stream;
    };
    Point& operator-() {
      for(int d = 0; d < dim; ++d) {x[d] = -x[d];}
      return *this;
    };
    friend Point operator- (const Point& a, const Point &b) {
      double tmp[dim];
      for(int d = 0; d < dim; ++d) {tmp[d] = a.x[d] - b.x[d];}
      return Point(tmp);
    };
  };
}
#endif /* PETSC_CLANGUAGE_CXX */

#endif /* __PETSC_HH */
