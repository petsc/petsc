#ifndef included_ALE_ALE_hh
#define included_ALE_ALE_hh

#include <petsc.h>

#include <map>
#include <set>
#include <vector>
#include <string>
#include <sstream>

typedef std::basic_ostringstream<char> ostringstream;

namespace ALE {

//   class Exception {
//   protected:
//     ostringstream *_msg;
//   public:
//     // WARNING: this will leak memory, if exceptions are handled;  should use Obj<X> as a base class for Exception.
//     Exception() {this->_msg = new ostringstream;};
//     Exception(const char  *msg) {Exception(); *(this->_msg) << msg;};
//     Exception(const Exception& e){Exception(e.message(););};
//     ~Exception() {delete this->_msg;};
//     //
//     const char *message(){return this->_msg->str().c_str();};
//     Exception& operator<<(const int32_t i){*(this->_msg) << i; return *this;};
//     Exception& operator<<(const double  i){*(this->_msg) << i; return *this;};
//     Exception& operator<<(const char   *i){*(this->_msg) << i; return *this;};
//   };

//   class Exception {
//     static const int LEN = 1024;
//     char *_msg;
//   public:
//     Exception()  {this->_msg = (char *)malloc(sizeof(char)*(LEN+1));};
//     ~Exception() {free((void*)this->_msg);};
//     Exception(const char *msg) {this->_msg = (char *)malloc(sizeof(char)*(LEN+1)); strncpy(this->_msg, msg, LEN);};
//     Exception(const ostringstream& s) {Exception(s.str().c_str());};
//     Exception(const Exception& e) {Exception(e.message());};
//     //
//     const char *message() const {return this->_msg;};
//     Exception& operator<<(const char   *i){
//       snprintf(this->_msg, LEN, "%s%s", this->_msg, i); 
//       return *this;
//     };
//     Exception& operator<<(const int32_t i){
//       snprintf(this->_msg, LEN, "%s%d", this->_msg, i); 
//       return *this;
//     };
//     Exception& operator<<(const double i){
//       snprintf(this->_msg, LEN, "%s%g", this->_msg, i); 
//       return *this;
//     };
//  };

  class Exception {
    const char *_msg;
  public:
    Exception(const char *msg)    {this->_msg = msg;};
    //Exception(const Exception& e) {this->_msg = e._msg;};
    const char *message() const   {return this->_msg;};
  };

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
    Obj() : objPtr((X *)NULL), refCnt((int32_t*)NULL){};               
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
    };

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
    

    // subscription operator
    X& operator[](int32_t i) {assertNull(0); return objPtr[i];};

    // destructor
    ~Obj(){
      if((this->refCnt != (int32_t *)NULL) && (--(this->refCnt) == 0) && !this->borrowed) {  
        delete objPtr; 
        delete refCnt;
      }
    };
    
  };// class Obj<X>



  class Point {
  public:
    int32_t prefix;
    int32_t index;
    Point() : prefix(0), index(0){};
    Point(int32_t p, int32_t i) : prefix(p), index(i){};
    bool operator==(const Point& q) const {
      return ( (this->prefix == q.prefix) && (this->index == q.index) );
    };
    bool operator!=(const Point& q) const {
      return ( (this->prefix != q.prefix) || (this->index != q.index) );
    };
    class Cmp {
    public: 
      bool operator()(const Point& p, const Point& q) const {
        return( (p.prefix < q.prefix) || ((p.prefix == q.prefix) && (p.index < q.index)));
      };
    };
  };

  class Point_array : public std::vector<Point> {
  public:
    Point_array()             : std::vector<Point>(){};
    Point_array(int32_t size) : std::vector<Point>(size){};
    //
    void view(const char *name = NULL) {
      printf("Viewing Point_array");
      if(name != NULL) {
        printf(" %s", name);
      }
      printf(" of size %d\n", (int) this->size());
      for(unsigned int cntr = 0; cntr < this->size(); cntr++) {
        Point p = (*this)[cntr];
        printf("element[%d]: (%d,%d)\n", cntr++, p.prefix, p.index);
      }
      
    };
  };

  class Point_set : public std::set<Point, Point::Cmp > {
  public:
    Point_set()         : std::set<Point, Point::Cmp>(){};
    Point_set(Point& p) : std::set<Point, Point::Cmp>(){insert(p);};
    //
    void join(Obj<Point_set> s) {
      for(Point_set::iterator s_itor = s->begin(); s_itor != s->end(); s_itor++) {
        this->insert(*s_itor);
      }
    };
    void meet(Obj<Point_set> s) {
      Point_set removal;
      for(Point_set::iterator self_itor = this->begin(); self_itor != this->end(); self_itor++) {
        Point p = *self_itor;
        if(s->find(p) == s->end()){
          removal.insert(p);
        }
      }
      for(Point_set::iterator rem_itor = removal.begin(); rem_itor != removal.end(); rem_itor++) {
        Point q = *rem_itor;
        this->erase(q);
      }
    };
    void view(const char *name = NULL) {
      printf("Viewing Point_set");
      if(name != NULL) {
        printf(" %s", name);
      }
      printf(" of size %d\n", (int) this->size());
      int32_t cntr = 0;
      for(Point_set::iterator s_itor = this->begin(); s_itor != this->end(); s_itor++) {
        Point p = *s_itor;
        printf("element[%d]: (%d,%d)\n", cntr++, p.prefix, p.index);
      }
      
    };
  };

  class PointIterator {
    Obj<Point_set>      _set;
    Point_set::iterator  _itor;
  public:
    PointIterator(Obj<Point_set> set) : _set(set), _itor(set->begin()) {};
    PointIterator& advance() { this->_itor++; return *this;                 };
    int atBeginning()        { return (this->_itor == this->_set->begin()); };
    int atEnd()              { return (this->_itor == this->_set->end());   };
    int32_t prefix()         { return this->_itor->prefix;                  };
    int32_t index()          { return this->_itor->index;                   };
  };


  class PointSet {
    Obj<Point_set> _set;
    Obj<int32_t> _array;
    int      _arrayInvalid;
  public:
    PointSet() : _set(new Point_set()), _array((int32_t *)NULL), _arrayInvalid(1) {};
    int contains(Point point)        {return (this->_set->find(point) !=  this->_set->end());      };
    PointSet& insert(const Point& p) {this->_set->insert(p); this->_arrayInvalid = 1; return *this;};
    PointSet& erase(const Point& p)  {this->_set->erase(p);  this->_arrayInvalid = 1; return *this;};
    int32_t   size()                 {return this->_set->size();                                   };
    PointSet& clear()                {this->_set->clear(); return *this;                           };
    Obj<int32_t> getArray(){
      if(this->_arrayInvalid) {
        this->_array = Obj<int32_t>(new int32_t[2*this->size()]);
        int32_t i = 0;
        for(Point_set::iterator itor = this->_set->begin(); itor != this->_set->end(); itor++) {
          this->_array[i++] = itor->prefix;
          this->_array[i++] = itor->index;
        }
      }
      return this->_array;
    }// getArray()
    Obj<PointIterator> beginning(){return Obj<PointIterator>(new PointIterator(this->_set));};
  };
  

  typedef std::map<int32_t, Point >               int__Point;
  typedef std::map<Point, int32_t, Point::Cmp >   Point__int;
  typedef std::map<Point, Point, Point::Cmp >     Point__Point;
  typedef std::map<Point, Point_set, Point::Cmp > Point__Point_set;
  typedef std::map<Point, PointSet, Point::Cmp >  Point__PointSet;

  typedef std::pair<int32_t, int32_t> int_pair;
  typedef std::set<int32_t> int_set;
  typedef std::set<int_pair> int_pair_set;
  typedef std::map<int32_t, int32_t> int__int;
  typedef std::map<int32_t, int_set> int__int_set;
 

  // Base class for all distributed ALE classes
  class Coaster {
  protected:
    //
    MPI_Comm                              comm;
    int32_t                               commRank;
    int32_t                               commSize;
    PetscObject                           petscObj;
    int32_t                               verbosity;
    int                                   _lock;
    void __checkLock(){if(this->_lock > 0) {throw(ALE::Exception("Mutating method attempted on a locked Coaster"));}};
  public:
    //
    Coaster() : petscObj(NULL) {this->clear();};
    Coaster(MPI_Comm c) : petscObj(NULL) {this->clear(); this->setComm(c);};
    virtual ~Coaster(){this->clear();};
    //
    virtual Coaster&          clear();
    virtual Coaster&          getLock();
    virtual Coaster&          releaseLock();
    virtual bool              isLocked(){return (this->_lock > 0);};
    virtual void              assertLock(bool status);
    //
    virtual void              setComm(MPI_Comm comm);
    MPI_Comm                  getComm() const{ return this->comm;};
    int32_t                   getCommSize() const {return this->commSize;};
    int32_t                   getCommRank() const {return this->commRank;};
    void                      setVerbosity(int32_t v){this->verbosity = v;};
    int32_t                   getVerbosity() const {return this->verbosity;};
    virtual void              view(const char *name);
    //
    friend void               CHKCOMM(Coaster& obj);
    friend void               CHKCOMMS(Coaster& obj1, Coaster& obj2);
  };





  // A helper function that throws an ALE::Exception with a message identifying the function that returned the given error code, 
  // including the function and the line where the error occured.
  void ERROR(PetscErrorCode ierr, const char *func, int line, const char *msg);
  // A helper function that allocates and assembles an error message from a format string 
  const char *ERRORMSG(const char *fmt, ...);
  // A helper function for converting MPI errors to exception
  void MPIERROR(PetscErrorCode ierr, const char *func, int line, const char *msg);


} // namespace ALE

  // A helper macro that passes __FUNCT__ and __LINE__ with the error msg to the ERROR routine
#define CHKERROR(ierr, msg) \
  ERROR(ierr, __FUNCT__,  __LINE__, msg);

  // A helper macro that passes __FUNCT__ and __LINE__ with the error msg to the MPIERROR routine
#define CHKMPIERROR(ierr, msg) \
  MPIERROR(ierr, __FUNCT__,  __LINE__, msg);

#endif
