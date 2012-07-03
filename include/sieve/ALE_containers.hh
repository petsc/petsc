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
#include <sieve/ALE_exception.hh>
#endif
#ifndef  included_ALE_mem_hh
#include <sieve/ALE_mem.hh>
#endif
#ifndef  included_ALE_log_hh
#include <sieve/ALE_log.hh>
#endif

namespace ALE {
  class ParallelObject {
  protected:
    int         _debug;
    MPI_Comm    _comm;
    int         _commRank;
    int         _commSize;
    std::string _name;
    std::string _className;
  protected:
    void __init(MPI_Comm comm) {
      PetscErrorCode ierr;

      this->_comm = comm;
      ierr = MPI_Comm_rank(this->_comm, &this->_commRank);CHKERROR(ierr, "Error in MPI_Comm_rank");
      ierr = MPI_Comm_size(this->_comm, &this->_commSize);CHKERROR(ierr, "Error in MPI_Comm_size");
      const char *class_name = ALE::getClassName(this);
      this->_className = class_name;
      ALE::restoreClassName(this, class_name);
    };
  public:
    ParallelObject(MPI_Comm comm = PETSC_COMM_SELF, const int debug = 0) : _debug(debug) {__init(comm);}
    ~ParallelObject() {};
  public:
    int                debug()    const {return this->_debug;};
    void               setDebug(const int debug) {this->_debug = debug;};
    MPI_Comm           comm()     const {return this->_comm;};
    int                commSize() const {return this->_commSize;};
    int                commRank() const {return this->_commRank;}
    const std::string& getName() const {return this->_name;};
    void               setName(const std::string& name) {this->_name = name;};
    const std::string& getClassName() const {return this->_className;};
    void               setClassName(const std::string& name) {this->_className = name;};
  public:
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
      ostringstream txt;
      int           rank;

      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
        rank = this->commRank();
      } else {
        MPI_Comm_rank(comm, &rank);
      }
      if (name == "") {
        if(rank == 0) {
          txt << "viewing a " << this->getClassName() << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing " << this->getClassName() << " '" << name << "'" << std::endl;
        }
      }
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };

  // Use for ArrowSections
  template<typename Source_, typename Target_>
  struct MinimalArrow {
    typedef Source_ source_type;
    typedef Target_ target_type;
    source_type source;
    target_type target;
    MinimalArrow() {};
    MinimalArrow(const source_type& source, const target_type& target) : source(source), target(target) {};
    MinimalArrow(const MinimalArrow& a) : source(a.source), target(a.target) {};
    friend std::ostream& operator<<(std::ostream& os, const MinimalArrow& a) {
      os << a.source << " ----> " << a.target;
      return os;
    }
    // Comparisons
    class less_than {
    public: 
      bool operator()(const MinimalArrow& p, const MinimalArrow& q) const {
        return((p.source < q.source) || ((p.source == q.source) && (p.target < q.target)));
      };
    };
    typedef less_than Cmp;
    bool operator==(const MinimalArrow& q) const {
      return((this->source == q.source) && (this->target == q.target));
    };
    bool operator!=(const MinimalArrow& q) const {
      return((this->source != q.source) || (this->target != q.target));
    };
    bool operator<(const MinimalArrow& q) const {
      return((this->source < q.source) || ((this->source == q.source) && (this->target < q.target)));
    };
  };

  //
  // This is a set of classes and class templates describing an interface to point containers.
  //
  
  // Basic object
  class Point {
  public:
    typedef ALE_ALLOCATOR<Point> allocator;
    typedef int32_t prefix_type;
    typedef int32_t index_type;
    prefix_type prefix;
    index_type  index;
    // Constructors
    Point() : prefix(0), index(0){};
    Point(int p) : prefix(p), index(0){};
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
    void operator+=(const Point& q) {
      this->prefix += q.prefix;
      this->index  += q.index;
    };
    // Printing
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
      os << "(" << p.prefix << ", "<< p.index << ")";
      return os;
    };
    // Reading
    friend std::istream& operator>>(std::istream& is, Point& p) {
      // Read "("
      char tmp = '\0';

      while (tmp != '(') {is.get(tmp);}
      is >> p.prefix;
      // Read ","
      while (tmp != ',') {is.get(tmp);}
      is >> p.index;
      // Read ")"
      while (tmp != ')') {is.get(tmp);}
      return is;
    };
  };
  template<typename FirstType, typename SecondType>
  class Pair {
  public:
    typedef FirstType  first_type;
    typedef SecondType second_type;
    first_type  first;
    second_type second;
    // Constructors
    Pair() : first(0), second(0) {};
    Pair(FirstType f) : first(f), second(0) {};
    Pair(FirstType f, SecondType s) : first(f), second(s) {};
    Pair(const Pair& p) : first(p.first), second(p.second) {};
    // Comparisons
    class less_than {
    public: 
      bool operator()(const Pair& p, const Pair& q) const {
        return( (p.first < q.first) || ((p.first == q.first) && (p.second < q.second)));
      };
    };
    typedef less_than Cmp;
    
    bool operator==(const Pair& q) const {
      return((this->first == q.first) && (this->second == q.second));
    };
    bool operator!=(const Pair& q) const {
      return((this->first != q.first) || (this->second != q.second));
    };
    bool operator<(const Pair& q) const {
      return((this->first < q.first) || ((this->first == q.first) && (this->second < q.second)));
    };
    void operator+=(const Pair& q) {
      this->first  += q.first;
      this->second += q.second;
    };
    // Printing
    friend std::ostream& operator<<(std::ostream& os, const Pair& p) {
      os << "(" << p.first << ", "<< p.second << ")";
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
      
    }
  };


  template <typename Element_>
  class set : public std::set<Element_, typename Element_::less_than,  ALE_ALLOCATOR<Element_> > {
  public:
    // Encapsulated types
    typedef std::set<Element_, typename Element_::less_than, ALE_ALLOCATOR<Element_> > super;
    typedef typename super::iterator                                                   iterator;
    typedef Element_                                                                   element_type;
    typedef element_type                                                               value_type;
    //
    // Basic interface
    //
    set()        : super(){};
    // FIX: this is a little weird that there is a specific constructor with Point
    set(Point p) : super(){insert(p);};
    //
    set(const element_type& e) : super() {insert(e);}
    //
    template<typename ElementSequence_>
    set(const ElementSequence_& eseq) : super(eseq.begin(), eseq.end()){}
    // 
    // Standard interface
    // 
    // Redirection: 
    // FIX: it is a little weird that 'insert' methods aren't inherited
    //      but perhaps can be fixed by calling insert<Element_> (i.e., insert<Point> etc)?
        std::pair<iterator, bool> 
    inline insert(const Element_& e) { return super::insert(e); };
    //
    iterator 
    inline insert(iterator position, const Element_& e) {return super::insert(position,e);};
    //
    template <class InputIterator>
    void 
    inline insert(InputIterator b, InputIterator e) { return super::insert(b,e);}
    // 
    // Extended interface
    //
    inline iterator last() {
      return this->rbegin();
    };// last()
    //    
    inline bool contains(const Element_& e) {return (this->find(e) != this->end());};
    //
    inline void join(Obj<set> s) {
      for(iterator s_itor = s->begin(); s_itor != s->end(); s_itor++) {
        this->insert(*s_itor);
      }
    };
    //
    inline void meet(Obj<set> s) {// this should be called 'intersect' (the verb)
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
    //
    inline void subtract(Obj<set> s) {
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
    }
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
      for(iterator<> i = this->_index.begin(); i != this->_index.end(); i++) {
        os << " "<< *i;
      }
      os << " ]" << std::endl;
    }
  };// class IndexSequence    

} // namespace ALE


#endif
