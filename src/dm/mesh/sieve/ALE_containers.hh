#ifndef included_ALE_containers_hh
#define included_ALE_containers_hh
// This should be included indirectly -- only by including ALE.hh

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>

#include <iostream>
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
    Point(int p) : prefix(p){};
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

  template <typename X>
  struct singleton {
    X first;
    //
    singleton(const X& x)         : first(x) {};
    singleton(const singleton& s) : first(s.first) {};
  };

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

  // 
  // Arrow definitions
  // 
  template<typename Source_, typename Target_, typename Color_>
  struct  Arrow { //: public ALE::def::Arrow<Source_, Target_, Color_> {
    typedef Arrow   arrow_type;
    typedef Source_ source_type;
    typedef Target_ target_type;
    typedef Color_  color_type;
    source_type source;
    target_type target;
    color_type  color;
    // Arrow modifiers
    struct sourceChanger {
      sourceChanger(const source_type& newSource) : _newSource(newSource) {};
      void operator()(arrow_type& a) {a.source = this->_newSource;}
    private:
      source_type _newSource;
    };
    
    struct targetChanger {
      targetChanger(const target_type& newTarget) : _newTarget(newTarget) {};
      void operator()(arrow_type& a) { a.target = this->_newTarget;}
    private:
      const target_type _newTarget;
    };
    // Flipping
    template <typename OtherSource_, typename OtherTarget_, typename OtherColor_>
    struct rebind {
      typedef Arrow<OtherSource_, OtherTarget_, OtherColor_> type;
    };
    struct flip {
      typedef Arrow<target_type, source_type, color_type> type;
      type arrow(const arrow_type& a) { return type(a.target, a.source, a.color);};
    };
  public:
    //
    // Basic interface
    Arrow(const source_type& s, const target_type& t, const color_type& c) : source(s), target(t), color(c) {};
    Arrow(const Arrow& a) : source(a.source), target(a.target), color(a.color) {};
    ~Arrow(){};
    //
    // Extended interface
    // Printing
    template <typename Stream_>
    friend Stream_& operator<<(Stream_& os, const Arrow& a) {
      os << a.source << " --(" << a.color << ")--> " << a.target;
      return os;
    }
  };// struct Arrow

  // Defines a sequence representing a subset of a multi_index container Index_.
  // A sequence defines output (input in std terminology) iterators for traversing an Index_ object.
  // Upon dereferencing values are extracted from each result record using a ValueExtractor_ object.
  template <typename Index_, typename ValueExtractor_ = ::boost::multi_index::identity<typename Index_::value_type> >
  struct IndexSequence {
    typedef Index_                                   index_type;
    typedef ValueExtractor_                          extractor_type;
    //
    template <typename Sequence_ = IndexSequence>
    class iterator {
    public:
      // Parent sequence type
      typedef Sequence_                              sequence_type;
      // Standard iterator typedefs
      typedef std::input_iterator_tag                iterator_category;
      typedef typename extractor_type::result_type   value_type;
      typedef int                                    difference_type;
      typedef value_type*                            pointer;
      typedef value_type&                            reference;
      // Underlying iterator type
      typedef typename index_type::iterator          itor_type;
    protected:
      // Parent sequence
      sequence_type&  _sequence;
      // Underlying iterator 
      itor_type      _itor;
      // Member extractor
      extractor_type _ex;
    public:
      iterator(sequence_type& sequence, itor_type itor)       : _sequence(sequence),_itor(itor) {};
      iterator(const iterator& iter)                          : _sequence(iter._sequence),_itor(iter._itor) {}
      virtual ~iterator() {};
      virtual bool              operator==(const iterator& iter) const {return this->_itor == iter._itor;};
      virtual bool              operator!=(const iterator& iter) const {return this->_itor != iter._itor;};
      // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
      virtual const value_type  operator*() const {return _ex(*(this->_itor));};
      virtual iterator   operator++() {++this->_itor; return *this;};
      virtual iterator   operator++(int n) {iterator tmp(*this); ++this->_itor; return tmp;};
    };// class iterator
  protected:
    index_type& _index;
  public:
    //
    // Basic interface
    //
    IndexSequence(const IndexSequence& seq)  : _index(seq._index) {};
    IndexSequence(index_type& index)         : _index(index) {};
    virtual ~IndexSequence() {};
    //
    // Extended interface
    //
    virtual bool         empty() {return this->_index.empty();};

    virtual typename index_type::size_type  size()  {
      typename index_type::size_type sz = 0;
      for(typename index_type::iterator itor = this->_index.begin(); itor != this->_index.end(); itor++) {
        ++sz;
      }
      return sz;
    };
    template<typename ostream_type>
    void view(ostream_type& os, const char* label = NULL){
      if(label != NULL) {
        os << "Viewing " << label << " sequence:" << std::endl;
      } 
      os << "[";
      for(iterator<> i = this->begin(); i != this->end(); i++) {
        os << " "<< *i;
      }
      os << " ]" << std::endl;
    };
  };// class IndexSequence    

} // namespace ALE


#endif
