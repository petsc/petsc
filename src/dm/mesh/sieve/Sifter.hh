#ifndef included_ALE_Sifter_hh
#define included_ALE_Sifter_hh

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <iostream>

// ALE extensions

#ifndef  included_ALE_hh
#include <ALE.hh>
#endif

namespace ALE {

  namespace def {
    //
    // This is a set of abstract classes describing an interface to point containers.
    //

    // Basic object
    class Point {
    public:
      int32_t prefix, index;
      // Constructors
      Point() : prefix(0), index(0){};
      Point(const int32_t& p, const int32_t& i) : prefix(p), index(i){};
      Point(const Point& p) : prefix(p.prefix), index(p.index){};
      // Comparisons
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
    
    // iterator interface
    template <typename Data>
    class const_iterator {
    public:
      virtual ~const_iterator() {};
      //
      virtual void                  operator++() {std::cout << "Faux preincrement" << std::endl;};
      virtual void                  operator++(int) {std::cout << "Faux postincrement" << std::endl;};
      virtual bool                  operator==(const const_iterator& itor) const {return false;};
      virtual bool                  operator!=(const const_iterator& itor) const {std::cout << "Faux !=" << std::endl;return false;};
      virtual const Data&           operator*() const {return Data();};
    };

    // const_sequence interface:
    // a constant sequence of (not necesserily unique) colors delineated by begin() & end() iterators; can be traversed linearly
    template <typename Data>
    class const_sequence {
    public:
      typedef const_iterator<Data> iterator;
      virtual ~const_sequence() {};
      //
      virtual const_iterator<Data> begin() {std::cout << "Faux iteration start" << std::endl; };
      virtual const_iterator<Data> end() {std::cout << "Faux iteration end" << std::endl; };
      virtual std::size_t          size() {return -1;};
    };

    // const_collection interface:
    // a constant collection no particular order; can queried for containment of a given color
    template <typename Data>
    class const_collection {
      virtual ~const_collection() = 0;
      //
      virtual bool contains(const Data& p);
    };

    // const_set interface
    // combines const_sequence & const_collection interfaces
    template <typename Data>
    class const_set : public const_sequence<Data>, public const_collection<Data> {
      virtual ~const_set();
    };

    // set interface:
    // extends const_set interface to allows point addition and removal
    template <typename Data>
    class set : public const_set<Data> {
      const std::set<Data>& delegate;
    public:
      set() {this->delegate = new std::set<Data>;};
      set(const std::set<Data>& delegate) {this->delegate = delegate;};
      // destructor
      virtual ~set() {delete this->delegate;};
      // mutating methods
      virtual void insert(const Data& p);                            // post: contains(p) == true 
      virtual void remove(const Data& p);                            // post: contains(p) == false
      virtual void add(const const_sequence<Data>& s);               // post: contains colors from s and '*this before the call'
      virtual void add(const const_collection<Data>& s);             // post: contains colors from s and '*this before the call'
      virtual void intersect(const const_sequence<Data>& s);         // post: contains colors common to s and '*this before the call'
      virtual void intersect(const const_collection<Data>& s);       // post: contains colors common to s and '*this before the call'
      virtual void subtract(const const_sequence<Data>&  s);         // post: contains colors of '*this before call' that are not in s
      virtual void subtract(const const_collection<Data>&  s);       // post: contains colors of '*this before call' that are not in s
    };

    class PointIterator : public const_iterator<Point> {
    public:
      Point *point;
      PointIterator(const Point *p) : point((Point *) p) {};
      virtual ~PointIterator() {};
      //
      virtual void                  operator++() {
        std::cout << "Before preincrement: " << this->point << std::endl;
        ++this->point;
        std::cout << "After preincrement: " << this->point << std::endl;
      };
      virtual void                  operator++(int n) {
        std::cout << "Before postincrement: " << this->point << std::endl;
        ++this->point;
        std::cout << "After postincrement: " << this->point << std::endl;
      };
      virtual bool                  operator==(const const_iterator<Point>& itor) const {
        return point == dynamic_cast<const PointIterator&>(itor).point;};
      virtual bool                  operator!=(const const_iterator<Point>& itor) const {
        std::cout << "Comparing " << point << " with " << dynamic_cast<const PointIterator&>(itor).point << std::endl;
        return point != dynamic_cast<const PointIterator&>(itor).point;};
      virtual const Point&          operator*() const {return *point;};
    };

    class PointSequence : public const_sequence<Point> {
      const Point& point;
    public:
      typedef PointIterator iterator;
      PointSequence(const Point& p) : point(p) {std::cout << "Made PointSequence from " << p << std::endl;};
      virtual ~PointSequence() {};
      virtual const_iterator<Point> begin() {
        PointIterator iter = PointIterator(&point);
        std::cout << "Iteration start: " << iter.point << std::endl;
        return iter;};
      virtual const_iterator<Point> end() {
        PointIterator iter = PointIterator(&point);
        ++iter;
        std::cout << "Iteration end: " << iter.point << std::endl;
        return iter;};
      virtual std::size_t           size() {return sizeof(Point);};
    };

    template <typename Color>
    struct Arrow {
      Point source;
      Point target;
      Color color;

      Arrow(Point s, Point t, Color c) : source(s), target(t), color(c) {};
      friend std::ostream& operator<<(std::ostream& os, const Arrow& a) {
        os << a.source << " --" << a.color << "--> " << a.target << std::endl;
        return os;
      }
    };

    //
    // Sieve:
    //   A Sieve is a set of {\emph arrows} connecting {\emph points} of type Data. Thus we
    // could realize a sieve, for instance, as a directed graph. In addition, we will often
    // impose an acyclicity constraint, arriving at a DAG. Each arrow may also carry a label,
    // or {\emph color}, and the interface allows sets of arrows to be filtered by color.
    //
    template <typename Data, typename Color>
    class Sieve {
      // tags for accessing the corresponding indices of employee_set
      struct source{};
      struct target{};
      struct color{};
      typedef Arrow<Color> SieveArrow;
      typedef ::boost::multi_index::multi_index_container<
        SieveArrow,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<source>,  BOOST_MULTI_INDEX_MEMBER(SieveArrow,Point,source)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<target>,  BOOST_MULTI_INDEX_MEMBER(SieveArrow,Point,target)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<color>,  BOOST_MULTI_INDEX_MEMBER(SieveArrow,Color,color)>
        >
      > ArrowSet;

      class coneIterator : public const_iterator<Data> {
      public:
        //const ArrowSet::index<target>::type::iterator arrowIter;
        typename boost::multi_index::index<ArrowSet,target>::type::iterator arrowIter;
        coneIterator(typename boost::multi_index::index<ArrowSet,target>::type::iterator& iter) : arrowIter(iter) {};
        coneIterator(const typename boost::multi_index::index<ArrowSet,target>::type::iterator& iter) {
          this->arrowIter = typename boost::multi_index::index<ArrowSet,target>::type::iterator(iter);
        };
        virtual ~coneIterator() {};
        //
        virtual void                  operator++() {++this->arrowIter;};
        virtual void                  operator++(int n) {++this->arrowIter;};
        virtual bool                  operator==(const const_iterator<Data>& itor) const {
          return this->arrowIter == dynamic_cast<const coneIterator&>(itor).arrowIter;};
        virtual bool                  operator!=(const const_iterator<Data>& itor) const {
          return this->arrowIter != dynamic_cast<const coneIterator&>(itor).arrowIter;};
        virtual const Data&           operator*() const {return this->arrowIter->source;};
      };

      class coneSequence : public const_sequence<Data> {
        const typename ::boost::multi_index::index<ArrowSet,target>::type& coneIndex;
        const Point& key;
      public:
        coneSequence(const typename ::boost::multi_index::index<ArrowSet,target>::type& cone, const Point& p) : coneIndex(cone), key(p) {};
        virtual ~coneSequence() {};
        virtual const_iterator<Data> begin() {return coneIterator(this->coneIndex.lower_bound(key));};
        virtual const_iterator<Data> end()   {return coneIterator(this->coneIndex.upper_bound(key));};
        virtual std::size_t          size()  {return sizeof(Point);};
      };
      //
      ArrowSet        arrows;
      std::set<Color> colors;
    public:
      Sieve() {};
      // The basic Sieve interface
      Obj<const_sequence<Data> > cone(const Obj<const_sequence<Data> >& p) {
        //FIX: return coneSequence(this->arrows.get<2>(), p);
        return coneSequence(::boost::multi_index::get<target>(this->arrows), *p->begin());
      };
      Obj<const_sequence<Data> > cone(const Obj<const_sequence<Data> >& p, const Color& color);
      Obj<const_sequence<Data> > nCone(const Obj<const_sequence<Data> >& p, const int& n);
      Obj<const_sequence<Data> > nCone(const Obj<const_sequence<Data> >& p, const int& n, const Color& color);
      Obj<const_sequence<Data> > support(const Obj<const_sequence<Data> >& p);
      Obj<const_sequence<Data> > support(const Obj<const_sequence<Data> >& p, const Color& color);
      Obj<const_sequence<Data> > nSupport(const Obj<const_sequence<Data> >& p, const int& n);
      Obj<const_sequence<Data> > nSupport(const Obj<const_sequence<Data> >& p, const int& n, const Color& color);
      // Iterated versions
      Obj<const_sequence<Data> > closure(const Obj<const_sequence<Data> >& p);
      Obj<const_sequence<Data> > closure(const Obj<const_sequence<Data> >& p, const int& n, const Color& color);
      Obj<const_sequence<Data> > nClosure(const Obj<const_sequence<Data> >& p, const int& n);
      Obj<const_sequence<Data> > nClosure(const Obj<const_sequence<Data> >& p, const int& n, const int& n, const Color& color);
      Obj<const_sequence<Data> > star(const Obj<const_sequence<Data> >& p);
      Obj<const_sequence<Data> > star(const Obj<const_sequence<Data> >& p, const int& n, const Color& color);
      Obj<const_sequence<Data> > nStar(const Obj<const_sequence<Data> >& p, const int& n);
      Obj<const_sequence<Data> > nStar(const Obj<const_sequence<Data> >& p, const int& n, const int& n, const Color& color);
      // Lattice methods
      Obj<const_sequence<Data> > meet(const const_sequence<Data>& pp);
      Obj<const_sequence<Data> > meet(const const_sequence<Data>& pp, const Color& color);
      Obj<const_sequence<Data> > nMeet(const const_sequence<Data>& pp, const int& n);
      Obj<const_sequence<Data> > nMeet(const const_sequence<Data>& pp, const int& n, const Color& color);
      Obj<const_sequence<Data> > join(const const_sequence<Data>& pp);
      Obj<const_sequence<Data> > join(const const_sequence<Data>& pp, const Color& color);
      Obj<const_sequence<Data> > nJoin(const const_sequence<Data>& pp, const int& n);
      Obj<const_sequence<Data> > nJoin(const const_sequence<Data>& pp, const int& n, const Color& color);
      // Manipulation
      void                       addArrow(const Data& p, const Data& q) {
        this->arrows.insert(SieveArrow(p, q, Color()));
        std::cout << "Added " << SieveArrow(p, q, Color());
      };
      void                       addArrow(const Data& p, const Data& q, const Color& color);
      void                       setColor(const Data& p, const Data& q, const Color& color);
      void                       addCone(const Obj<const_sequence<Data> >& points,  const Data& p){
        std::cout << "Adding a cone " << std::endl;
        for(typename const_sequence<Data>::iterator iter = points->begin(); iter != points->end(); ++iter) {
          std::cout << "Adding arrow from " << *iter << " to " << p << std::endl;
          this->addArrow(*iter, p);
        }
      };
      void                       addSupport(const Data& p,const Obj<const_sequence<Data> >& points);
      void                       add(const Obj<Sieve<Data,Color> >& sieve);
      // Parallelism
      Obj<std::map<Point, Sieve<Data,Color> > > completion(const Sieve<Data,Color>& base);
      // These methods are meaningful only for acyclic sieves
      int depth(const Data& p);
      int height(const Data& p);
      int diameter(const Data& p);
      int diameter();
      Obj<const_sequence<Data> > depthStratum(const int& depth);
      Obj<const_sequence<Data> > heightStratum(const int& height);
      void                       setStratification(bool on);
      bool                       getStratification();
      void                       stratify();
    };

    //
    // CoSieve:
    //   Value is the type of function space (PreSheaf?) for fibers over the mesh
    //
    template <typename Data, typename Value>
    class CoSieve {
      const Value *restrict(const Data& support, const const_sequence<Data>& chain);
      void         assemble(const Data& support, const const_sequence<Data>& chain, const Value values[]);

      // \rho
      //     This needs some sort of policy argument
      void reduce();

      // Refine the support chain
      void refine();
      void refine(const const_sequence<Data>& chain);
      // Coarsen the support chain
      void coarsen();
      void coarsen(const const_sequence<Data>& chain);
    };
  } // namespace def



} // namespace ALE

#endif
