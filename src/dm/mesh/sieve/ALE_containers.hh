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

  namespace def {
    //
    // This is a set of abstract classes describing an interface to point containers.
    //

    // Basic object
    class point {
    public:
      int32_t prefix, index;
    };
    
    // point_iterator interface
    class point_iterator {
      virtual ~point_iterator() = 0;
      //
      virtual void                  operator++();
      virtual void                  operator++(int);
      virtual bool                  operator==(const point_iterator& itor);
      virtual bool                  operator!=(const point_iterator& itor);
      virtual const point&          operator*()  = 0;
    };

    // const_point_sequence interface:
    // a constant sequence of (not necesserily unique) points delineated by begin() & end() iterators; can be traversed linearly
    class const_point_sequence {
      typedef point_iterator iterator;
      virtual ~const_point_sequence() = 0;
      //
      virtual point_iterator& begin();
      virtual point_iterator& end();
      virtual std::size_t     size();
    };

    // const_point_collection interface:
    // a constant collection no particular order; can queried for containment of a given point
    class const_point_collection {
      virtual ~const_point_collection() = 0;
      //
      virtual bool contains(const point& p);
    };

    // point_set interface:
    // extends const_point_sequence & const_point_collection and allows point addition and removal
    class point_set : public const_point_sequence, const_point_collection {
      // conversion constructors
      point_set(const const_point_sequence&);
      point_set(const Obj<const_point_sequence>&);
      point_set(const const_point_collection&);
      point_set(const Obj<const_point_collection>&);
      // destructor
      virtual ~point_set();
      // mutating methods
      virtual void insert(const point& p);                     // post: contains(p) == true 
      virtual void remove(const point& p);                     // post: contains(p) == false
      virtual void add(const const_point_sequence& s);         // post: contains points from s and '*this before the call'
      virtual void add(const const_point_collection& s);       // post: contains points from s and '*this before the call'
      virtual void intersect(const const_point_sequence& s);   // post: contains points common to s and '*this before the call'
      virtual void intersect(const const_point_collection& s); // post: contains points common to s and '*this before the call'
      virtual void subtract(const const_point_sequence&  s);   // post: contains points of '*this before call' that are not in s
      virtual void subtract(const const_point_collection&  s); // post: contains points of '*this before call' that are not in s
    };
    
  } // namespace def

  class Point {
  public:
    int32_t prefix, index;
    typedef ALE_ALLOCATOR<Point> allocator;
    Point() : prefix(0), index(0){};
    Point(const int32_t& p, const int32_t& i) : prefix(p), index(i){};
    //Point(const Point& q) : prefix(q.prefix), index(q.index) {};
    bool operator==(const Point& q) const {
      return ( (this->prefix == q.prefix) && (this->index == q.index) );
    };
    bool operator!=(const Point& q) const {
      return ( (this->prefix != q.prefix) || (this->index != q.index) );
    };
    class less_than {
    public: 
      bool operator()(const Point& p, const Point& q) const {
        return( (p.prefix < q.prefix) || ((p.prefix == q.prefix) && (p.index < q.index)));
      };
    };
    typedef less_than Cmp;
  };
  



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
