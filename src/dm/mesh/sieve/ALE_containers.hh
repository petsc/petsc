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
#ifndef  included_ALE_log_hh
#include <ALE_log.hh>
#endif

namespace ALE {


  //
  // This is a set of classes and class templates describing an interface to point containers.
  //
  
  // Basic object
  class Point {
  public:
    typedef ALE_ALLOCATOR<Point> allocator;
    int32_t prefix, index;
    // Constructors
    Point() : prefix(0), index(0){};
    Point(int p, int i) : prefix(p), index(i){};
    Point(const Point& p) : prefix(p.prefix), index(p.index){};
    // Comparisons
    class less_than {
    public: 
      bool operator()(const Point& p, const Point& q) const {
        return( (p.prefix < q.prefix) || ((p.prefix == q.prefix) && (p.index < q.index)));
      };
    };
    typedef less_than Cmp;
    
    bool operator==(const Point& q) const {
      return ( (this->prefix == q.prefix) && (this->index == q.index) );
    };
    bool operator!=(const Point& q) const {
      return ( (this->prefix != q.prefix) || (this->index != q.index) );
    };
    bool operator<(const Point& q) const {
      return( (this->prefix < q.prefix) || ((this->prefix == q.prefix) && (this->index < q.index)));
    };
    // Printing
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
      os << "(" << p.prefix << ", "<< p.index << ")";
      return os;
    };
  };

  template <typename Element_>
  class array : public std::vector<Element_, ALE_ALLOCATOR<Element_> > {
  public:
    array()             : std::vector<Element_, ALE_ALLOCATOR<Element_> >(){};
    array(int32_t size) : std::vector<Element_, ALE_ALLOCATOR<Element_> >(size){};
    //
    template <typename ostream_type>
    void view(ostream_type& os, const char *name = NULL) {
      os << "Viewing array";
      if(name != NULL) {
        os << " " << name;
      }
      os << " of size " << (int) this->size() << std::endl;
      os << "[";
      for(unsigned int cntr = 0; cntr < this->size(); cntr++) {
        Element_ e = (*this)[cntr];
        os << e;
      }
      os << " ]" << std::endl;
      
    };
  };


  template <typename Element_>
  class set : public std::set<Element_, typename Element_::less_than,  ALE_ALLOCATOR<Element_> > {
  public:
    // Encapsulated types
    typedef std::set<Element_, typename Element_::less_than, ALE_ALLOCATOR<Element_> > base_type;
    typedef typename base_type::iterator                                               iterator;
    // Basic interface
    set()        : std::set<Element_, typename Element_::less_than, ALE_ALLOCATOR<Element_> >(){};
    set(Point p) : std::set<Element_, typename Element_::less_than, ALE_ALLOCATOR<Element_> >(){insert(p);};
    // Redirection: 
    // FIX: it is a little weird that methods aren't inheritec, 
    //      but perhaps can be fixed by calling insert<Element_> (i.e., insert<Point> etc)?

    std::pair<iterator, bool> 
    insert(const Element_& e) { return base_type::insert(e); };

    iterator 
    insert(iterator position, const Element_& e) {return base_type::insert(position,e);};

    template <class InputIterator>
    void 
    insert(InputIterator b, InputIterator e) { return base_type::insert(b,e);};

    
    // Extensions to std::set interface
    bool contains(const Element_& e) {return (this->find(e) != this->end());};
    void join(Obj<set> s) {
      for(iterator s_itor = s->begin(); s_itor != s->end(); s_itor++) {
        this->insert(*s_itor);
      }
    };
    void meet(Obj<set> s) {// this should be called 'intersect' (the verb)
      set removal;
      for(iterator self_itor = this->begin(); self_itor != this->end(); self_itor++) {
        Element_ e = *self_itor;
        if(!s->contains(e)){
          removal.insert(e);
        }
      }
      for(iterator rem_itor = removal.begin(); rem_itor != removal.end(); rem_itor++) {
        Element_ ee = *rem_itor;
        this->erase(ee);
      }
    };
    void subtract(Obj<set> s) {
      set removal;
      for(iterator self_itor = this->begin(); self_itor != this->end(); self_itor++) {
        Element_ e = *self_itor;
        if(s->contains(e)){
          removal.insert(e);
        }
      }
      for(iterator rem_itor = removal.begin(); rem_itor != removal.end(); rem_itor++) {
        Element_ ee = *rem_itor;
        this->erase(ee);
      }
    };

    //
    template <typename ostream_type>
    void view(ostream_type& os, const char *name = NULL) {
      os << "Viewing set";
      if(name != NULL) {
        os << " " << name;
      }
      os << " of size " << (int) this->size() << std::endl;
      os << "[";
      for(iterator s_itor = this->begin(); s_itor != this->end(); s_itor++) {
        Element_ e = *s_itor;
        os << e;
      }
      os << " ]" << std::endl;
    };
  };

  typedef set<Point>   PointSet;
  typedef array<Point> PointArray;

  template <typename X, typename Y>
  struct pair : public std::pair<X,Y> {
    pair() : std::pair<X,Y>(){};
    pair(const pair& p) : std::pair<X,Y>(p.first, p.second) {};
    pair(const X& x, const Y& y) : std::pair<X,Y>(x,y) {};
    ~pair(){};
    friend std::ostream& operator<<(std::ostream& os, const pair& p) {
      os << "<" << p.first << ", "<< p.second << ">";
      return os;
    };
  };// struct pair
  

} // namespace ALE


#endif
