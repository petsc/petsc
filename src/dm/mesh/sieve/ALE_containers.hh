#ifndef included_ALE_containers_hh
#define included_ALE_containers_hh
// This should be included indirectly -- only by including ALE.hh

#include <map>
#include <set>
#include <vector>

#ifndef  included_ALE_exception_hh
#include <ALE_exception.hh>
#endif
#ifndef  included_ALE_mem_hh
#include <ALE_mem.hh>
#endif

namespace ALE {

  class point {
  public:
    typedef ALE_ALLOCATOR<point> allocator;
    int32_t prefix;
    int32_t index;
    point() : prefix(0), index(0){};
    point(const int32_t& p, const int32_t& i) : prefix(p), index(i){};
    bool operator==(const point& q) const {
      return ( (this->prefix == q.prefix) && (this->index == q.index) );
    };
    bool operator!=(const point& q) const {
      return ( (this->prefix != q.prefix) || (this->index != q.index) );
    };
    class less_than {
    public: 
      bool operator()(const point& p, const point& q) const {
        return( (p.prefix < q.prefix) || ((p.prefix == q.prefix) && (p.index < q.index)));
      };
    };
    typedef less_than Cmp;
  };
  
  namespace def {
    //
    // This is a set of abstract classes describing an interface to point sets.
    //
    
    // const_point_iterator 
    class const_point_iterator {
      virtual ~const_point_iterator();
      virtual void          operator++() = 0;
      virtual void          operator++(int) = 0;
      virtual bool          operator==(const const_point_iterator& itor) = 0;
      virtual bool          operator!=(const const_point_iterator& itor) = 0;
      virtual const point&  operator*()  = 0;
    };

    // const_point_interval
    class const_point_interval {
      typedef const_point_iterator iterator;
      virtual ~const_point_interval();
      virtual const_point_iterator& begin() = 0;
      virtual const_point_iterator& end()   = 0;
      virtual std::size_t           size()  = 0;
    };

    // const_point_set is a sequence of points in no particular a priori order delineated by the begin() & end() iterators
    class const_point_set : public const_point_interval {
      virtual ~const_point_set();
      virtual bool                 contains(const point& p) = 0;
      virtual void                 view(const char*name) = 0;
    };

    // point_set extends const_point_set to allow point addition and removal
    class point_set : public const_point_set {
      // copy constructor
      point_set(const const_point_set& s);
      // destructor
      virtual ~point_set();
      // mutating methods
      virtual void   insert(const point& p) = 0;              // postcondition: contains(p) == true 
      virtual void   remove(const point& p) = 0;              // postcondition: contains(p) == false
      virtual void   add(const const_point_set& s) = 0;       // postcondition: contains points from s and *this before the call.
      virtual void   intersect(const const_point_set& s) = 0; // postcondition: contains points common to s and *this before the call.
      virtual void   subtract(const const_point_set&  s) = 0; // postcondition: contains points of *this before call that are not in s.
    };
    
  } // namespace def

  namespace future {
    //
    // algorithms on point sets
    //

    // join == 'union'; cannot use 'union' directly as it is a C/C++ reserved word
    template<bool future_is_here>
    Obj<ALE::def::point_set> join(const ALE::def::const_point_set& s1, const ALE::def::const_point_set& s2){
      Obj<ALE::def::point_set> _s(s1);
      for(ALE::def::point_set::iterator s2_itor = s2.begin(); s2_itor != s2.end(); s2_itor++) {
        _s->insert(*s2_itor);
      }
      return _s;
    }; // join()

    // meet == 'intersection'
    template<bool future_is_here>
    Obj<ALE::def::point_set> meet(const ALE::def::const_point_set& s1, const ALE::def::const_point_set& s2){
      Obj<ALE::def::point_set> _s(ALE::def::point_set());
      ALE::def::const_point_set& ss1, ss2;
      // Iterate over the smaller set
      if(s1.size() < s2.size()) {
        ss1 = s1;
        ss2 = s2;
      }
      else
      {
        ss1 = s2;
        ss2 = s1;
      }
      for(ALE::def::point_set::iterator ss1_itor = ss1.begin(); ss1_itor != ss1.end(); ss1_itor++) {
        if(ss2.contains(*ss1_itor)) {
          _s->insert(*ss1_itor);
        }
      }
      return _s;
    }; // meet()

    // diff == 'difference' == 's1\s2'
    template<bool future_is_here>
    Obj<ALE::def::point_set> diff(const ALE::def::const_point_set& s1, const ALE::def::const_point_set& s2){
      Obj<ALE::def::point_set> _s(ALE::def::point_set());
      for(ALE::def::const_point_set::iterator s1_itor = s1.begin(); s1_itor != s1.end(); s1_itor++) {
        if(!s2.contains(*s1_itor)) {
          _s->insert(*s1_itor);
        }
      }
      return _s;
    }; // diff()

  } // namespace future

  typedef point Point;
  class Point_array : public std::vector<Point, Point::allocator > {
  public:
    Point_array()             : std::vector<Point, Point::allocator >(){};
    Point_array(int32_t size) : std::vector<Point, Point::allocator >(size){};
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



  class Point_set : public std::set<Point, Point::less_than, Point::allocator > {
  public:
    Point_set()        : std::set<Point, Point::less_than, Point::allocator>(){};
    Point_set(Point p) : std::set<Point, Point::less_than, Point::allocator>(){insert(p);};
    //
    void join(Obj<Point_set> s) {
      for(Point_set::iterator s_itor = s->begin(); s_itor != s->end(); s_itor++) {
        this->insert(*s_itor);
      }
    };
    void meet(Obj<Point_set> s) {// this should be called 'intersect' (the verb)
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
    void subtract(Obj<Point_set> s) {
      Point_set removal;
      for(Point_set::iterator self_itor = this->begin(); self_itor != this->end(); self_itor++) {
        Point p = *self_itor;
        if(s->find(p) != s->end()){
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

  typedef std::map<int32_t, Point >                       int__Point;
  typedef std::map<Point, int32_t,   Point::less_than >   Point__int;
  typedef std::map<Point, Point,     Point::less_than >   Point__Point;
  typedef std::map<Point, Point_set, Point::less_than >   Point__Point_set;

  typedef std::pair<int32_t, int32_t> int_pair;
  typedef std::set<int32_t> int_set;
  typedef std::set<int_pair> int_pair_set;
  typedef std::map<int32_t, int32_t> int__int;
  typedef std::map<int32_t, int_set> int__int_set;



} // namespace ALE


#endif
