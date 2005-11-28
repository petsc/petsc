#ifndef included_ALE_containers_hh
#define included_ALE_containers_hh
// This should be included indirectly -- only by including ALE.hh

#include <ALE_mem.hh>

namespace ALE {

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
    typedef logged_allocator<Point> allocator;
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

  class Point_set : public std::set<Point, Point::Cmp, Point::allocator > {
  public:
    Point_set()        : std::set<Point, Point::Cmp, Point::allocator>(){};
    Point_set(Point p) : std::set<Point, Point::Cmp, Point::allocator>(){insert(p);};
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

  typedef std::map<int32_t, Point >               int__Point;
  typedef std::map<Point, int32_t, Point::Cmp >   Point__int;
  typedef std::map<Point, Point, Point::Cmp >     Point__Point;
  typedef std::map<Point, Point_set, Point::Cmp > Point__Point_set;

  typedef std::pair<int32_t, int32_t> int_pair;
  typedef std::set<int32_t> int_set;
  typedef std::set<int_pair> int_pair_set;
  typedef std::map<int32_t, int32_t> int__int;
  typedef std::map<int32_t, int_set> int__int_set;
 

} // namespace ALE


#endif
