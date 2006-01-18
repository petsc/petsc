#ifndef included_ALE_Sifter_hh
#define included_ALE_Sifter_hh

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <iostream>

// ALE extensions

#ifndef  included_ALE_hh
#include <ALE.hh>
#endif

namespace ALE {

  namespace def {
    //
    // This is a set of classes and class templates describing an interface to point containers.
    //

    // Basic object
    class Point {
    public:
      int32_t prefix, index;
      // Constructors
      Point() : prefix(0), index(0){};
      Point(int p, int i) : prefix(p), index(i){};
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
    typedef std::set<Point> PointSet;

    template <typename Source_, typename Target_, typename Color_>
    struct Arrow {
      Source_ source;
      Target_ target;
      Color_ color;

      Arrow(Source_ s, Target_ t, Color_ c) : source(s), target(t), color(c) {};
      friend std::ostream& operator<<(std::ostream& os, const Arrow& a) {
        os << a.source << " --" << a.color << "--> " << a.target << std::endl;
        return os;
      }
    };

    //
    // Sieve:
    //   A Sieve is a set of {\emph arrows} connecting {\emph points} of type Point_. Thus we
    // could realize a sieve, for instance, as a directed graph. In addition, we will often
    // assume an acyclicity constraint, arriving at a DAG. Each arrow may also carry a label,
    // or {\emph color} of type Color_, and the interface allows sets of arrows to be filtered 
    // by color.
    //
    template <typename Point_, typename Color_>
    class Sieve { // class Sieve
    public:
      typedef Color_  color_type;
      typedef Point_  point_type;
      typedef int     marker_type;
      int debug;
    private:
      // Arrow query tags
      struct source{};
      struct target{};
      struct color{};
      struct sourceColor{};
      struct targetColor{};      
      // Arrow structures
      typedef Arrow<Point,Point,Color_> Arrow_;
      typedef ::boost::multi_index::multi_index_container<
        Arrow_,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<source>, BOOST_MULTI_INDEX_MEMBER(Arrow_,Point_,source)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<target>, BOOST_MULTI_INDEX_MEMBER(Arrow_,Point_,target)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<color>,  BOOST_MULTI_INDEX_MEMBER(Arrow_,Color_,color)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<sourceColor>,
            ::boost::multi_index::composite_key<
              Arrow_, BOOST_MULTI_INDEX_MEMBER(Arrow_,Point_,source), BOOST_MULTI_INDEX_MEMBER(Arrow_,Color_,color)>
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<targetColor>,
            ::boost::multi_index::composite_key<
              Arrow_, BOOST_MULTI_INDEX_MEMBER(Arrow_,Point_,target), BOOST_MULTI_INDEX_MEMBER(Arrow_,Color_,color)>
          >
        >,
        ALE_ALLOCATOR<Arrow_>
      > ArrowSet;
      ArrowSet arrows;

      // Stratum query tags
      struct point{};
      struct depthTag{};
      struct heightTag{};
      struct indegree{};
      struct outdegree{};
      struct marker{};
      // Stratum stuctures
      struct StratumPoint {
        Point_  point;
        int   depth;
        int   height;
        int   indegree;
        int   outdegree;
        marker_type marker;

        StratumPoint() : depth(0), height(0), indegree(0), outdegree(0), marker(marker_type()) {};
        StratumPoint(const Point_& p) : point(p), depth(0), height(0), indegree(0), outdegree(0), marker(marker_type()) {};
        // Printing
        friend std::ostream& operator<<(std::ostream& os, const StratumPoint& p) {
          os << "[" << p.point << ", "<< p.depth << ", "<< p.height << ", "<< p.indegree << ", "<< p.outdegree << ", " << p.marker << "]";
          return os;
        };
      };
      typedef ::boost::multi_index::multi_index_container<
        StratumPoint,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<point>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,Point_,point)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<depthTag>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,int,depth)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<heightTag>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,int,height)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<indegree>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,int,indegree)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<outdegree>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,int,outdegree)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<marker>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,marker_type,marker)>
        >,
        ALE_ALLOCATOR<StratumPoint>
      > StratumSet;
      StratumSet strata;
      bool       stratification; 
      int        maxDepth;
      int        maxHeight;
      int        graphDiameter;
    public:
      //
      // Output sequence types
      //

      class baseSequence {
        const typename ::boost::multi_index::index<StratumSet,indegree>::type& baseIndex;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef Point_  value_type;
          typedef int   difference_type;
          typedef Point_* pointer;
          typedef Point_& reference;
          typename boost::multi_index::index<StratumSet,indegree>::type::iterator pointIter;
          
          iterator(const typename boost::multi_index::index<StratumSet,indegree>::type::iterator& iter) {
            this->pointIter = typename boost::multi_index::index<StratumSet,indegree>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->pointIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->pointIter); ++this->pointIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->pointIter == itor.pointIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->pointIter != itor.pointIter;};
          virtual const Point_& operator*() const {return this->pointIter->point;};
        };
        baseSequence(const typename ::boost::multi_index::index<StratumSet,indegree>::type& base) : baseIndex(base) {};
        virtual ~baseSequence() {};
        virtual iterator    begin() {return iterator(this->baseIndex.upper_bound(0));};
        virtual iterator    end()   {return iterator(this->baseIndex.end());};
        virtual std::size_t size()  {return -1;};
      };

      class capSequence {
        const typename ::boost::multi_index::index<StratumSet,outdegree>::type& capIndex;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef Point_  value_type;
          typedef int   difference_type;
          typedef Point_* pointer;
          typedef Point_& reference;
          typename boost::multi_index::index<StratumSet,outdegree>::type::iterator pointIter;
          
          iterator(const typename boost::multi_index::index<StratumSet,outdegree>::type::iterator& iter) {
            this->pointIter = typename boost::multi_index::index<StratumSet,outdegree>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->pointIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->pointIter); ++this->pointIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->pointIter == itor.pointIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->pointIter != itor.pointIter;};
          virtual const Point_& operator*() const {return this->pointIter->point;};
        };
        capSequence(const typename ::boost::multi_index::index<StratumSet,outdegree>::type& cap) : capIndex(cap) {};
        virtual ~capSequence() {};
        virtual iterator    begin() {return iterator(this->capIndex.upper_bound(0));};
        virtual iterator    end()   {return iterator(this->capIndex.end());};
        virtual std::size_t size()  {return -1;};
      };

      class rootSequence {
        const typename ::boost::multi_index::index<StratumSet,indegree>::type& rootIndex;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef Point_  value_type;
          typedef int   difference_type;
          typedef Point_* pointer;
          typedef Point_& reference;
          typename boost::multi_index::index<StratumSet,indegree>::type::iterator pointIter;
          
          iterator(const typename boost::multi_index::index<StratumSet,indegree>::type::iterator& iter){
            this->pointIter = typename boost::multi_index::index<StratumSet,indegree>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->pointIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->pointIter); ++this->pointIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->pointIter == itor.pointIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->pointIter != itor.pointIter;};
          virtual const Point_& operator*() const {return this->pointIter->point;};
        };
        rootSequence(const typename ::boost::multi_index::index<StratumSet,indegree>::type& root) : rootIndex(root) {};
        virtual ~rootSequence() {};
        virtual iterator    begin() {return iterator(this->rootIndex.lower_bound(0));};
        virtual iterator    end()   {return iterator(this->rootIndex.upper_bound(0));};
        virtual std::size_t size()  {return this->rootIndex.count(0);};
      };  
      
      class leafSequence {
        const typename ::boost::multi_index::index<StratumSet,outdegree>::type& leafIndex;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef Point_  value_type;
          typedef int   difference_type;
          typedef Point_* pointer;
          typedef Point_& reference;
          typename boost::multi_index::index<StratumSet,outdegree>::type::iterator pointIter;
          
          iterator(const typename boost::multi_index::index<StratumSet,outdegree>::type::iterator& iter) {
            this->pointIter = typename boost::multi_index::index<StratumSet,outdegree>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->pointIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->pointIter); ++this->pointIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->pointIter == itor.pointIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->pointIter != itor.pointIter;};
          virtual const Point_& operator*() const {return this->pointIter->point;};
        };
        
        leafSequence(const typename ::boost::multi_index::index<StratumSet,outdegree>::type& leaf) : leafIndex(leaf) {};
        virtual ~leafSequence() {};
        virtual iterator    begin() {return iterator(this->leafIndex.lower_bound(0));};
        virtual iterator    end()   {return iterator(this->leafIndex.upper_bound(0));};
        virtual std::size_t size()  {return this->leafIndex.count(0);};
      };
    
      class coneSequence {
        const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& coneIndex;
        const Point_ key;
        const Color_ color;
        const bool useColor;
      public:
        class iterator {
        public:
          typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator arrowIter;
          
          iterator(const typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator& iter) {
            this->arrowIter = typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->arrowIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->arrowIter); ++this->arrowIter; return tmp;};
          virtual iterator    operator--() {--this->arrowIter; return *this;};
          virtual iterator    operator--(int n) {iterator tmp(this->arrowIter); --this->arrowIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->arrowIter == itor.arrowIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->arrowIter != itor.arrowIter;};
          virtual const Point_& operator*() const {return this->arrowIter->source;};
        };
        class reverse_iterator {
        public:
          typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator arrowIter;
          
          reverse_iterator(const typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator& iter) {
            this->arrowIter = typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator(iter);
          };
          virtual ~reverse_iterator() {};
          //
          virtual reverse_iterator operator++() {--this->arrowIter; return *this;};
          virtual reverse_iterator operator++(int n) {reverse_iterator tmp(this->arrowIter); --this->arrowIter; return tmp;};
          virtual bool             operator==(const reverse_iterator& itor) const {return this->arrowIter == itor.arrowIter;};
          virtual bool             operator!=(const reverse_iterator& itor) const {return this->arrowIter != itor.arrowIter;};
          virtual const Point_&      operator*() const {return this->arrowIter->source;};
        };
        
        coneSequence(const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& cone, const Point_& p) : 
          coneIndex(cone), key(p), color(Color_()), useColor(0) {};
        coneSequence(const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& cone, const Point_& p, const Color_& c) : 
          coneIndex(cone), key(p), color(c), useColor(1) {};
        virtual ~coneSequence() {};
        virtual iterator    begin() {
          if (useColor) {
            return iterator(this->coneIndex.lower_bound(::boost::make_tuple(key,color)));
          } else {
            return iterator(this->coneIndex.lower_bound(::boost::make_tuple(key)));
          }
        };
        virtual iterator    end()   {
          if (useColor) {
            return iterator(this->coneIndex.upper_bound(::boost::make_tuple(key,color)));
          } else {
            return iterator(this->coneIndex.upper_bound(::boost::make_tuple(key)));
          }
        };
        virtual reverse_iterator rbegin() {
          if (useColor) {
            return reverse_iterator(--this->coneIndex.upper_bound(::boost::make_tuple(key,color)));
          } else {
            return reverse_iterator(--this->coneIndex.upper_bound(::boost::make_tuple(key)));
          }
        };
        virtual reverse_iterator rend()   {
          typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator i;
          
          if (useColor) {
            return reverse_iterator(--this->coneIndex.lower_bound(::boost::make_tuple(key,color)));
          } else {
            return reverse_iterator(--this->coneIndex.lower_bound(::boost::make_tuple(key)));
          }
        };
        virtual std::size_t size()  {
          if (useColor) {
            return this->coneIndex.count(::boost::make_tuple(key,color));
          } else {
            return this->coneIndex.count(::boost::make_tuple(key));
          }
        };
      }; // class Sieve<Point_,Color_>::coneSequence
      
      class supportSequence {
        const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& supportIndex;
        const Point_ key;
        const Color_ color;
        const bool useColor;
      public:
        class iterator {
        public:
          typename boost::multi_index::index<ArrowSet,sourceColor>::type::iterator arrowIter;
          
          iterator(const typename boost::multi_index::index<ArrowSet,sourceColor>::type::iterator& iter) {
            this->arrowIter = typename boost::multi_index::index<ArrowSet,sourceColor>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->arrowIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->arrowIter); ++this->arrowIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->arrowIter == itor.arrowIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->arrowIter != itor.arrowIter;};
          virtual const Point_& operator*() const {return this->arrowIter->target;};
        };
        
        supportSequence(const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& support, const Point_& p)
          : supportIndex(support), key(p), color(Color_()), useColor(0) {};
        supportSequence(const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& support,const Point_& p,const Color_& c)
          : supportIndex(support), key(p), color(c), useColor(1) {};
        virtual ~supportSequence() {};
        virtual iterator    begin() {
          if (useColor) {
            return iterator(this->supportIndex.lower_bound(::boost::make_tuple(key,color)));
          } else {
            return iterator(this->supportIndex.lower_bound(::boost::make_tuple(key)));
          }
        };
        virtual iterator    end()   {
          if (useColor) {
            return iterator(this->supportIndex.upper_bound(::boost::make_tuple(key,color)));
          } else {
            return iterator(this->supportIndex.upper_bound(::boost::make_tuple(key)));
          }
        };
        virtual std::size_t size()  {
          if (useColor) {
            return this->supportIndex.count(::boost::make_tuple(key,color));
          } else {
            return this->supportIndex.count(::boost::make_tuple(key));
          }
        };
      };


      class depthSequence {
        const typename ::boost::multi_index::index<StratumSet,depthTag>::type& depthIndex;
        const int d;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef Point_  value_type;
          typedef int     difference_type;
          typedef Point_* pointer;
          typedef Point_& reference;
          typename boost::multi_index::index<StratumSet,depthTag>::type::iterator pointIter;

          iterator(const typename boost::multi_index::index<StratumSet,depthTag>::type::iterator& iter) {
            this->pointIter = typename boost::multi_index::index<StratumSet,depthTag>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator      operator++() {++this->pointIter; return *this;};
          virtual iterator      operator++(int n) {iterator tmp(this->pointIter); ++this->pointIter; return tmp;};
          virtual bool          operator==(const iterator& itor) const {return this->pointIter == itor.pointIter;};
          virtual bool          operator!=(const iterator& itor) const {return this->pointIter != itor.pointIter;};
          virtual const Point_& operator*() const {return this->pointIter->point;};
          marker_type           getMarker() const {return this->pointIter->marker;};
        };

        depthSequence(const typename ::boost::multi_index::index<StratumSet,depthTag>::type& depthIndex, const int d) : depthIndex(depthIndex), d(d) {};
        virtual ~depthSequence() {};
        virtual iterator    begin() {return iterator(this->depthIndex.lower_bound(d));};
        virtual iterator    end()   {return iterator(this->depthIndex.upper_bound(d));};
        virtual std::size_t size()  {return this->depthIndex.count(d);};
      };

      class heightSequence {
        const typename ::boost::multi_index::index<StratumSet,heightTag>::type& heightIndex;
        const int h;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef Point_  value_type;
          typedef int   difference_type;
          typedef Point_* pointer;
          typedef Point_& reference;
          typename boost::multi_index::index<StratumSet,heightTag>::type::iterator pointIter;
          
          iterator(const typename boost::multi_index::index<StratumSet,heightTag>::type::iterator& iter) {
            this->pointIter = typename boost::multi_index::index<StratumSet,heightTag>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->pointIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->pointIter); ++this->pointIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->pointIter == itor.pointIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->pointIter != itor.pointIter;};
          virtual const Point_& operator*() const {return this->pointIter->point;};
        };
        
        heightSequence(const typename ::boost::multi_index::index<StratumSet,heightTag>::type& height, const int h) : 
          heightIndex(height), h(h) {};
        virtual ~heightSequence() {};
        virtual iterator    begin() {return iterator(this->heightIndex.lower_bound(h));};
        virtual iterator    end()   {return iterator(this->heightIndex.upper_bound(h));};
        virtual std::size_t size()  {return this->heightIndex.count(h);};
      };
      
      class markerSequence {
        const typename ::boost::multi_index::index<StratumSet,marker>::type& markerIndex;
        const marker_type m;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef Point_  value_type;
          typedef int     difference_type;
          typedef Point_* pointer;
          typedef Point_& reference;
          typename boost::multi_index::index<StratumSet,marker>::type::iterator pointIter;

          iterator(const typename boost::multi_index::index<StratumSet,marker>::type::iterator& iter) {
            this->pointIter = typename boost::multi_index::index<StratumSet,marker>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator      operator++() {++this->pointIter; return *this;};
          virtual iterator      operator++(int n) {iterator tmp(this->pointIter); ++this->pointIter; return tmp;};
          virtual bool          operator==(const iterator& itor) const {return this->pointIter == itor.pointIter;};
          virtual bool          operator!=(const iterator& itor) const {return this->pointIter != itor.pointIter;};
          virtual const Point_& operator*() const {return this->pointIter->point;};
        };

        markerSequence(const typename ::boost::multi_index::index<StratumSet,marker>::type& marker, const int m) : markerIndex(marker), m(m) {};
        virtual ~markerSequence() {};
        virtual iterator    begin() {return iterator(this->markerIndex.lower_bound(m));};
        virtual iterator    end()   {return iterator(this->markerIndex.upper_bound(m));};
        virtual std::size_t size()  {return this->markerIndex.count(m);};
      };

    public:
      Sieve() : debug(0), stratification(false), maxDepth(-1), maxHeight(-1), graphDiameter(-1) {};
      // Printing
      friend std::ostream& operator<<(std::ostream& os, Obj<Sieve<Point_,Color_> > s) { 
        os << *s; 
        return os;
      };
    
      friend std::ostream& operator<<(std::ostream& os, Sieve<Point_,Color_>& s) {
        Obj<baseSequence> base = s.base();
        for(typename baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          Obj<coneSequence> cone = s.cone(*b_iter);
          os << "Base point " << *b_iter << " with cone:" << std::endl;
          for(typename coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            os << "  " << *c_iter << std::endl;
          }
        }
        return os;
      };

      //friend std::ostream& operator<<(std::ostream& os, Obj<Sieve<Point_,Color_> > s);
      //friend std::ostream& operator<<(std::ostream& os, Sieve<Point_,Color_>& s);

    public:
      //
      // The basic Sieve interface
      //
      void clear();

      Obj<coneSequence> cone(const Point_& p);

      template<class InputSequence> 
      Obj<PointSet>     cone(const Obj<InputSequence>& points);

      Obj<coneSequence> cone(const Point_& p, const Color_& color);

      template<class InputSequence> 
      Obj<PointSet> cone(const Obj<InputSequence>& points, const Color_& color);

      Obj<PointSet> nCone(const Point_& p, int n);

      Obj<PointSet> nCone(const Point_& p, int n, const Color_& color, bool useColor = true);

      template<class InputSequence> 
      Obj<PointSet> nCone(const Obj<InputSequence>& points, int n);

      template<class InputSequence> 
      Obj<PointSet> nCone(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor = true);

      bool          coneContains(const Point_& p, const Point_& q);

      Obj<supportSequence> support(const Point_& p);

      template<class InputSequence> 
      Obj<PointSet>        support(const Obj<InputSequence>& points);

      Obj<supportSequence> support(const Point_& p, const Color_& color);

      template<class InputSequence> 
      Obj<PointSet>        support(const Obj<InputSequence>& points, const Color_& color);

      template<class InputSequence> 
      Obj<PointSet>        nSupport(const Obj<InputSequence>& points, int n);

      template<class InputSequence> 
      Obj<PointSet>        nSupport(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor = true);

    private:
      template<class InputSequence> Obj<PointSet> 
      __nCone(Obj<InputSequence>& cone, int n, const Color_& color, bool useColor);
      
    public:
      //
      // Iterated versions
      //
      Obj<PointSet> closure(const Point_& p);

      Obj<PointSet> closure(const Point_& p, const Color_& color);

      template<class InputSequence> 
      Obj<PointSet> closure(const Obj<InputSequence>& points);

      template<class InputSequence> 
      Obj<PointSet> closure(const Obj<InputSequence>& points, const Color_& color);

      Obj<PointSet> nClosure(const Point_& p, int n);

      Obj<PointSet> nClosure(const Point_& p, int n, const Color_& color, bool useColor = true);

      template<class InputSequence> 
      Obj<PointSet> nClosure(const Obj<InputSequence>& points, int n);

      template<class InputSequence> 
      Obj<PointSet> nClosure(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor = true);

      Obj<Sieve<Point_,Color_> > closureSieve(const Point_& p);

      Obj<Sieve<Point_,Color_> > closureSieve(const Point_& p, const Color_& color);

      template<class InputSequence> 
      Obj<Sieve<Point_,Color_> > closureSieve(const Obj<InputSequence>& points);

      template<class InputSequence> 
      Obj<Sieve<Point_,Color_> > closureSieve(const Obj<InputSequence>& points, const Color_& color);

      Obj<Sieve<Point_,Color_> > nClosureSieve(const Point_& p, int n);

      Obj<Sieve<Point_,Color_> > nClosureSieve(const Point_& p, int n, const Color_& color, bool useColor = true);

      template<class InputSequence> 
      Obj<Sieve<Point_,Color_> > nClosureSieve(const Obj<InputSequence>& points, int n);

      template<class InputSequence> 
      Obj<Sieve<Point_,Color_> > nClosureSieve(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor = true);

      Obj<PointSet> star(const Point_& p);

      Obj<PointSet> star(const Point_& p, const Color_& color);

      template<class InputSequence> 
      Obj<PointSet> star(const Obj<InputSequence>& points);

      template<class InputSequence> 
      Obj<PointSet> star(const Obj<InputSequence>& points, const Color_& color);

      Obj<PointSet> nStar(const Point_& p, int n);

      Obj<PointSet> nStar(const Point_& p, int n, const Color_& color, bool useColor = true);

      template<class InputSequence> 
      Obj<PointSet> nStar(const Obj<InputSequence>& points, int n);

      template<class InputSequence> 
      Obj<PointSet> nStar(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor = true);

    private:
      template<class InputSequence> 
      Obj<PointSet> __nClosure(Obj<InputSequence>& cone, int n, const Color_& color, bool useColor);

      template<class InputSequence> 
      Obj<Sieve<Point_,Color_> > __nClosureSieve(Obj<InputSequence>& cone, int n, const Color_& color, bool useColor);

      template<class InputSequence> 
      Obj<PointSet> __nStar(Obj<InputSequence>& support, int n, const Color_& color, bool useColor);

    public:
      //
      // Lattice methods
      //
      Obj<PointSet> meet(const Point_& p, const Point_& q);

      Obj<PointSet> meet(const Point_& p, const Point_& q, const Color_& color);

      template<class InputSequence> 
      Obj<PointSet> meet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1);

      template<class InputSequence> 
      Obj<PointSet> meet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, const Color_& color);

      Obj<PointSet> nMeet(const Point_& p, const Point_& q, int n);

      Obj<PointSet> nMeet(const Point_& p, const Point_& q, int n, const Color_& color, bool useColor = true);

      template<class InputSequence> 
      Obj<PointSet> nMeet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n);

      template<class InputSequence> 
      Obj<PointSet> nMeet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n, 
                          const Color_& color, bool useColor = true);

      Obj<PointSet> join(const Point_& p, const Point_& q);

      Obj<PointSet> join(const Point_& p, const Point_& q, const Color_& color);

      template<class InputSequence> 
      Obj<PointSet> join(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1);

      template<class InputSequence> 
      Obj<PointSet> join(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, const Color_& color);

      Obj<PointSet> nJoin(const Point_& p, const Point_& q, int n);

      Obj<PointSet> nJoin(const Point_& p, const Point_& q, int n, const Color_& color, bool useColor = true);

      template<class InputSequence> 
      Obj<PointSet> nJoin(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n);

      template<class InputSequence> 
      Obj<PointSet> nJoin(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n, const Color_& color, bool useColor = true);

    public:
      //
      // Manipulation
      //
      void addPoint(const Point_& p);

      void addArrow(const Point_& p, const Point_& q);

      void addArrow(const Point_& p, const Point_& q, const Color_& color);

      template<class InputSequence> 
      void addCone(const Obj<InputSequence >& points, const Point_& p);

      template<class InputSequence> 
      void addCone(const Obj<InputSequence >& points, const Point_& p, const Color_& color);

      template<class InputSequence> 
      void addSupport(const Point_& p, const Obj<InputSequence >& points);

      template<class InputSequence> 
      void addSupport(const Point_& p, const Obj<InputSequence >& points, const Color_& color);

      void add(const Obj<Sieve<Point_,Color_> >& sieve);

    public:
      //
      // Parallelism
      //
      Obj<std::map<Point_, Sieve<Point_,Color_> > > completion(const Sieve<Point_,Color_>& base);

    public:
      //
      // Views
      //
      Obj<baseSequence> base() {
        return baseSequence(::boost::multi_index::get<indegree>(this->strata));
      };
      Obj<capSequence>  cap() {
        return capSequence(::boost::multi_index::get<outdegree>(this->strata));
      };
      Obj<rootSequence> roots() {
        return rootSequence(::boost::multi_index::get<indegree>(this->strata));
      };
      Obj<leafSequence> leaves() {
        return leafSequence(::boost::multi_index::get<outdegree>(this->strata));
      };

    public:
      //
      // Structural queries
      //
      int depth(); 
      int depth(const Point_& p);

      template<typename InputSequence> 
      int depth(const Obj<InputSequence>& points);

      int height();

      int height(const Point_& p);

      template<typename InputSequence> 
      int height(const Obj<InputSequence>& points);

      int diameter();

      int diameter(const Point_& p);

      Obj<depthSequence> depthStratum(int d);

      Obj<heightSequence> heightStratum(int h);

      Obj<markerSequence> markerStratum(int m);
 
      void setStratification(bool doStratify);

      bool getStratification();

      void stratify(bool show = false);


    public:
      //
      // Structural manipulation
      //

      struct changeMarker {
        changeMarker(int newMarker) : newMarker(newMarker) {};

        void operator()(StratumPoint& p) {
          p.marker = newMarker;
        }
      private:
        marker_type newMarker;
      };

      void setMarker(const point_type& p, const marker_type& marker);

      template<class InputSequence> 
      void setMarker(const Obj<InputSequence>& points, const marker_type& marker);

    private:
      struct changeIndegree {
        changeIndegree(int newIndegree) : newIndegree(newIndegree) {};

        void operator()(StratumPoint& p) {
          p.indegree = newIndegree;
        }
      private:
        int newIndegree;
      };

      struct changeOutdegree {
        changeOutdegree(int newOutdegree) : newOutdegree(newOutdegree) {};

        void operator()(StratumPoint& p) {
          p.outdegree = newOutdegree;
        }
      private:
        int newOutdegree;
      };

      void __computeDegrees();

      struct changeHeight {
        changeHeight(int newHeight) : newHeight(newHeight) {};

        void operator()(StratumPoint& p) {
          p.height = newHeight;
        }
      private:
        int newHeight;
      };
      
      template<class InputSequence> 
      void __computeClosureHeights(const Obj<InputSequence>& points);

      struct changeDepth {
        changeDepth(int newDepth) : newDepth(newDepth) {};

        void operator()(StratumPoint& p) {
          p.depth = newDepth;
        }
      private:
        int newDepth;
      };

      template<class InputSequence> 
      void __computeStarDepths(const Obj<InputSequence>& points);

    };//class Sieve


    // 
    // Sieve printing
    //
 
//     template <typename Point_, typename Color_> 
//     std::ostream& operator<<(std::ostream& os, Obj<Sieve<Point_,Color_> > s) { 
//       os << *s; 
//       return os;
//     };
    
//     template <typename Point_, typename Color_> 
//     std::ostream& operator<<(std::ostream& os, Sieve<Point_,Color_>& s) {
//       Obj<Sieve<Point_,Color_>::baseSequence> base = s.base();
      
//       for(typename baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
//         Obj<Sieve<Point_,Color_>::coneSequence> cone = s.cone(*b_iter);
        
//         os << "Base point " << *b_iter << " with cone:" << std::endl;
//         for(typename Sieve<Point_,Color_>::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
//           os << "  " << *c_iter << std::endl;
//         }
//       }
//       return os;
//     };
        
    //
    // The basic Sieve interface
    //
    template <typename Point_, typename Color_> 
    void Sieve<Point_, Color_>::clear() { 
      this->arrows.clear(); this->strata.clear();
    };

    template <typename Point_, typename Color_> 
    Obj<typename Sieve<Point_, Color_>::coneSequence> Sieve<Point_, Color_>::cone(const Point_& p) { 
      return coneSequence(::boost::multi_index::get<targetColor>(this->arrows), p);
    }
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_, Color_>::cone(const Obj<InputSequence>& points) {
      return this->nCone(points, 1);
    };
    
    template <typename Point_, typename Color_> 
    Obj<typename Sieve<Point_, Color_>::coneSequence> Sieve<Point_, Color_>::cone(const Point_& p, const Color_& color) {
      return coneSequence(::boost::multi_index::get<targetColor>(this->arrows), p, color);
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_, Color_>::cone(const Obj<InputSequence>& points, const Color_& color) {
      return this->nCone(points, 1, color);
    };
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_, Color_>::nCone(const Point_& p, int n) {
      return this->nCone(p, n, Color_(), false);
    };
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_, Color_>::nCone(const Point_& p, int n, const Color_& color, bool useColor) {
      Obj<PointSet> cone = PointSet();
      cone->insert(p);
      return this->__nCone(cone, n, color, useColor);
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_, Color_>::nCone(const Obj<InputSequence>& points, int n) {
      return this->nCone(points, n, Color_(), false);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_, Color_>::nCone(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor ) {
      Obj<PointSet> cone = PointSet();
      cone->insert(points->begin(), points->end());
      return this->__nCone(cone, n, color, useColor);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_, Color_>::__nCone(Obj<InputSequence>& cone, int n, const Color_& color, bool useColor) {
      Obj<PointSet> base = PointSet();

      for(int i = 0; i < n; ++i) {
        Obj<PointSet> tmp = cone; cone = base; base = tmp;
        
        cone->clear();
        for(PointSet::iterator b_itor = base->begin(); b_itor != base->end(); ++b_itor) {
          Obj<coneSequence> pCone;
          
          if (useColor) {
            pCone = this->cone(*b_itor, color);
          } else {
            pCone = this->cone(*b_itor);
          }
          cone->insert(pCone->begin(), pCone->end());
        }
      }
      return cone;
    };

    template <typename Point_, typename Color_> 
    bool Sieve<Point_, Color_>::coneContains(const Point_& p, const Point_& q) {
      //FIX: Shouldn't we just be able to query an arrow?
      Obj<coneSequence> cone = this->cone(p);
      
      for(typename Sieve<Point_, Color_>::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); c_iter++) {
        if (*c_iter == q) return true;
      }
      return false;
    }

    template <typename Point_, typename Color_> 
    Obj<typename Sieve<Point_,Color_>::supportSequence> Sieve<Point_, Color_>::support(const Point_& p) {
      return supportSequence(::boost::multi_index::get<sourceColor>(this->arrows), p);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_, Color_>::support(const Obj<InputSequence>& points) {
      return this->nSupport(points, 1);
    };

    template <typename Point_, typename Color_> 
    Obj<typename Sieve<Point_,Color_>::supportSequence> Sieve<Point_, Color_>::support(const Point_& p, const Color_& color) {
      return supportSequence(::boost::multi_index::get<sourceColor>(this->arrows), p, color);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_, Color_>::support(const Obj<InputSequence>& points, const Color_& color) {
      return this->nSupport(points, 1, color);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_, Color_>::nSupport(const Obj<InputSequence>& points, int n) {
      return this->nSupport(points, n, Color_(), false);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_, Color_>::nSupport(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor ) {
      Obj<PointSet> support = PointSet();
      Obj<PointSet> cap = PointSet();
      
      support->insert(points->begin(), points->end());
      for(int i = 0; i < n; ++i) {
        Obj<PointSet> tmp = support; support = cap; cap = tmp;
        
        support->clear();
        for(PointSet::iterator c_itor = cap->begin(); c_itor != cap->end(); ++c_itor) {
          Obj<supportSequence> pSupport;
          
          if (useColor) {
            pSupport = this->support(*c_itor, color);
          } else {
            pSupport = this->support(*c_itor);
          }
          support->insert(pSupport->begin(), pSupport->end());
        }
      }
      return support;
    };

    //
    // Iterated versions
    //
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::closure(const Point_& p) {
      return nClosure(p, this->depth());
    };
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::closure(const Point_& p, const Color_& color) {
      return nClosure(p, this->depth(), color);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::closure(const Obj<InputSequence>& points) {
      return nClosure(points, this->depth());
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::closure(const Obj<InputSequence>& points, const Color_& color) {
      return nClosure(points, this->depth(), color);
    };
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::nClosure(const Point_& p, int n) {
      return this->nClosure(p, n, Color_(), false);
    };

    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::nClosure(const Point_& p, int n, const Color_& color, bool useColor ) {
      Obj<PointSet> cone = PointSet();
      
      cone->insert(p);
      return this->__nClosure(cone, n, color, useColor);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::nClosure(const Obj<InputSequence>& points, int n) {
      return this->nClosure(points, n, Color_(), false);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::nClosure(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor ) {
      Obj<PointSet> cone = PointSet();
      cone->insert(points->begin(), points->end());
      return this->__nClosure(cone, n, color, useColor);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::__nClosure(Obj<InputSequence>& cone, int n, const Color_& color, bool useColor) {
      Obj<PointSet> base = PointSet();
      Obj<PointSet> closure = PointSet();
      
      for(int i = 0; i < n; ++i) {
        Obj<PointSet> tmp = cone; cone = base; base = tmp;
        
        cone->clear();
        for(PointSet::iterator b_itor = base->begin(); b_itor != base->end(); ++b_itor) {
          Obj<coneSequence> pCone;
          
          if (useColor) {
            pCone = this->cone(*b_itor, color);
          } else {
            pCone = this->cone(*b_itor);
          }
          cone->insert(pCone->begin(), pCone->end());
          closure->insert(pCone->begin(), pCone->end());
        }
      }
      return closure;
    };

    template <typename Point_, typename Color_> 
    Obj<Sieve<Point_,Color_> > Sieve<Point_,Color_>::closureSieve(const Point_& p) {
      return nClosureSieve(p, this->depth());
    };

    template <typename Point_, typename Color_> 
    Obj<Sieve<Point_,Color_> > Sieve<Point_,Color_>::closureSieve(const Point_& p, const Color_& color) {
      return nClosureSieve(p, this->depth(), color);
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<Sieve<Point_,Color_> > Sieve<Point_,Color_>::closureSieve(const Obj<InputSequence>& points) {
      return nClosureSieve(points, this->depth());
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<Sieve<Point_,Color_> > Sieve<Point_,Color_>::closureSieve(const Obj<InputSequence>& points, const Color_& color) {
      return nClosureSieve(points, this->depth(), color);
    };
    
    template <typename Point_, typename Color_> 
    Obj<Sieve<Point_,Color_> > Sieve<Point_,Color_>::nClosureSieve(const Point_& p, int n) {
      return this->nClosureSieve(p, n, Color_(), false);
    };
    
    template <typename Point_, typename Color_> 
    Obj<Sieve<Point_,Color_> > Sieve<Point_,Color_>::nClosureSieve(const Point_& p, int n, const Color_& color, bool useColor ) {
      Obj<PointSet> cone = PointSet();
      
      cone->insert(p);
      return this->__nClosureSieve(cone, n, color, useColor);
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<Sieve<Point_,Color_> > Sieve<Point_,Color_>::nClosureSieve(const Obj<InputSequence>& points, int n) {
      return this->nClosureSieve(points, n, Color_(), false);
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<Sieve<Point_,Color_> > Sieve<Point_,Color_>::nClosureSieve(const Obj<InputSequence>& points, int n, 
                                                                   const Color_& color, bool useColor ) {
      Obj<PointSet> cone = PointSet();
      
      cone->insert(points->begin(), points->end());
      return this->__nClosureSieve(cone, n, color, useColor);
    };
 
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<Sieve<Point_,Color_> > Sieve<Point_,Color_>::__nClosureSieve(Obj<InputSequence>& cone,int n,const Color_& color,bool useColor) {
      Obj<PointSet> base = PointSet();
      Obj<Sieve<Point_,Color_> > closure = Sieve<Point_,Color_>();
      
      for(int i = 0; i < n; ++i) {
        Obj<PointSet> tmp = cone; cone = base; base = tmp;
        
        cone->clear();
        for(PointSet::iterator b_itor = base->begin(); b_itor != base->end(); ++b_itor) {
          Obj<coneSequence> pCone;
          
          if (useColor) {
            pCone = this->cone(*b_itor, color);
          } else {
            pCone = this->cone(*b_itor);
          }
          cone->insert(pCone->begin(), pCone->end());
          closure->addCone(pCone, *b_itor);
        }
      }
      return closure;
    };
      
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::star(const Point_& p) {
      return nStar(p, this->height());
    };
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::star(const Point_& p, const Color_& color) {
      return nStar(p, this->depth(), color);
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::star(const Obj<InputSequence>& points) {
      return nStar(points, this->height());
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::star(const Obj<InputSequence>& points, const Color_& color) {
      return nStar(points, this->height(), color);
    };

    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::nStar(const Point_& p, int n) {
      return this->nStar(p, n, Color_(), false);
    };
      
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::nStar(const Point_& p, int n, const Color_& color, bool useColor ) {
      Obj<PointSet> support = PointSet();
      
      support->insert(p);
      return this->__nStar(support, n, color, useColor);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::nStar(const Obj<InputSequence>& points, int n) {
      return this->nStar(points, n, Color_(), false);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::nStar(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor ) {
      Obj<PointSet> support = PointSet();
      
      support->insert(points->begin(), points->end());
      return this->__nStar(cone, n, color, useColor);
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::__nStar(Obj<InputSequence>& support, int n, const Color_& color, bool useColor) {
      Obj<PointSet> cap = PointSet();
      Obj<PointSet> star = PointSet();
      
      for(int i = 0; i < n; ++i) {
        Obj<PointSet> tmp = support; support = base; base = tmp;
        
        support->clear();
        for(PointSet::iterator b_itor = base->begin(); b_itor != base->end(); ++b_itor) {
          Obj<supportSequence> pSupport;
          
          if (useColor) {
            pSupport = this->support(*b_itor, color);
          } else {
            pSupport = this->support(*b_itor);
          }
          support->insert(pSupport->begin(), pSupport->end());
          star->insert(pSupport->begin(), pSupport->end());
        }
      }
      return star;
    };
    
    //
    // Lattice methods
    //

    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::meet(const Point_& p, const Point_& q) {
      return nMeet(p, q, this->depth());
    };
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::meet(const Point_& p, const Point_& q, const Color_& color) {
      return nMeet(p, q, this->depth(), color);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::meet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1) {
      return nMeet(chain0, chain1, this->depth());
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::meet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, const Color_& color) {
        return nMeet(chain0, chain1, this->depth(), color);
    };

    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::nMeet(const Point_& p, const Point_& q, int n) {
        return nMeet(p, q, n, Color_(), false);
    };

    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::nMeet(const Point_& p, const Point_& q, int n, const Color_& color, bool useColor ) {
      Obj<PointSet> chain0 = PointSet();
      Obj<PointSet> chain1 = PointSet();
      
      chain0->insert(p);
      chain1->insert(q);
      return this->nMeet(chain0, chain1, n, color, useColor);
    };

    template <typename Point_, typename Color_>     
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::nMeet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n) {
      return this->nMeet(chain0, chain1, n, Color_(), false);
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::nMeet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1,int n,const Color_& color, bool useColor){
      // The strategy is to compute the intersection of cones over the chains, remove the intersection 
      // and use the remaining two parts -- two disjoined components of the symmetric difference of cones -- as the new chains.
      // The intersections at each stage are accumulated and their union is the meet.
      // The iteration stops after n steps in addition to the meet of the initial chains or sooner if at least one of the chains is empty.
      Obj<PointSet> meet = PointSet(); 
      Obj<PointSet> cone;
      
      if((chain0->size() != 0) && (chain1->size() != 0)) {
        for(int i = 0; i <= n; ++i) {
          // Compute the intersection of chains and put it in meet at the same time removing it from c and cc
          //std::set_intersection(chain0->begin(), chain0->end(), chain1->begin(), chain1->end(), std::insert_iterator<PointSet>(meet, meet->begin()));
          //chain0->erase(meet->begin(), meet->end());
          //chain1->erase(meet->begin(), meet->end());
          for(typename InputSequence::iterator iter = chain0->begin(); iter != chain0->end(); ++iter) {
            if (chain1->find(*iter) != chain1->end()) {
              meet->insert(*iter);
              chain0->erase(*iter);
              chain1->erase(*iter);
            }
          }
          // Replace each of the cones with a cone over it, and check if either is empty; if so, return what's in meet at the moment.
          cone = this->cone(chain0);
          chain0->insert(cone->begin(), cone->end());
          if(chain0->size() == 0) {
            break;
          }
          cone = this->cone(chain1);
          chain1->insert(cone->begin(), cone->end());
          if(chain1->size() == 0) {
            break;
          }
        }
      }
      return meet;
    };
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::join(const Point_& p, const Point_& q) {
      return this->nJoin(p, q, this->depth());
    };
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::join(const Point_& p, const Point_& q, const Color_& color) {
      return this->nJoin(p, q, this->depth(), color);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::join(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1) {
      return this->nJoin(chain0, chain1, this->depth());
    };
    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::join(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, const Color_& color) {
      return this->nJoin(chain0, chain1, this->depth(), color);
    };
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::nJoin(const Point_& p, const Point_& q, int n) {
      return this->nJoin(p, q, n, Color_(), false);
    };
    
    template <typename Point_, typename Color_> 
    Obj<PointSet> Sieve<Point_,Color_>::nJoin(const Point_& p, const Point_& q, int n, const Color_& color, bool useColor) {
      Obj<PointSet> chain0 = PointSet();
      Obj<PointSet> chain1 = PointSet();
      
      chain0->insert(p);
      chain1->insert(q);
      return this->nJoin(chain0, chain1, n, color, useColor);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::nJoin(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n) {
      return this->nJoin(chain0, chain1, n, Color_(), false);
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    Obj<PointSet> Sieve<Point_,Color_>::nJoin(const Obj<InputSequence>& chain0,const Obj<InputSequence>& chain1,int n,const Color_& color,bool useColor){
      // The strategy is to compute the intersection of supports over the chains, remove the intersection 
      // and use the remaining two parts -- two disjoined components of the symmetric difference of supports -- as the new chains.
      // The intersections at each stage are accumulated and their union is the join.
      // The iteration stops after n steps in addition to the join of the initial chains or sooner if at least one of the chains is empty.
      Obj<PointSet> join = PointSet(); 
      Obj<PointSet> support;
      
      if((chain0->size() != 0) && (chain1->size() != 0)) {
        for(int i = 0; i <= n; ++i) {
          // Compute the intersection of chains and put it in meet at the same time removing it from c and cc
          //std::set_intersection(chain0->begin(), chain0->end(), chain1->begin(), chain1->end(), std::insert_iterator<PointSet>(join.obj(), join->begin()));
          //chain0->erase(join->begin(), join->end());
          //chain1->erase(join->begin(), join->end());
          for(typename InputSequence::iterator iter = chain0->begin(); iter != chain0->end(); ++iter) {
            if (chain1->find(*iter) != chain1->end()) {
              join->insert(*iter);
              chain0->erase(*iter);
              chain1->erase(*iter);
            }
          }
          // Replace each of the supports with the support over it, and check if either is empty; if so, return what's in join at the moment.
          support = this->support(chain0);
          chain0->insert(support->begin(), support->end());
          if(chain0->size() == 0) {
            break;
          }
          support = this->support(chain1);
          chain1->insert(support->begin(), support->end());
          if(chain1->size() == 0) {
            break;
          }
        }
      }
      return join;
    };
    
    //
    // Manipulation
    //
  
    template <typename Point_, typename Color_> 
    void Sieve<Point_,Color_>::addPoint(const Point_& p) {
      this->strata(StratumPoint(p));
    };

    template <typename Point_, typename Color_> 
    void Sieve<Point_,Color_>::addArrow(const Point_& p, const Point_& q) {
      this->addArrow(p, q, Color_());
    };

    template <typename Point_, typename Color_> 
    void Sieve<Point_,Color_>::addArrow(const Point_& p, const Point_& q, const Color_& color) {
      this->arrows.insert(Arrow_(p, q, color));
      if (debug) {std::cout << "Added " << Arrow_(p, q, color);}
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    void Sieve<Point_,Color_>::addCone(const Obj<InputSequence >& points, const Point_& p) {
      this->addCone(points, p, Color_());
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    void Sieve<Point_,Color_>::addCone(const Obj<InputSequence >& points, const Point_& p, const Color_& color){
      if (debug) {std::cout << "Adding a cone " << std::endl;}
      for(typename InputSequence::iterator iter = points->begin(); iter != points->end(); ++iter) {
        if (debug) {std::cout << "Adding arrow from " << *iter << " to " << p << "(" << color << ")" << std::endl;}
        this->addArrow(*iter, p, color);
      }
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    void Sieve<Point_,Color_>::addSupport(const Point_& p, const Obj<InputSequence >& points) {
      this->addSupport(p, points, Color_());
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    void Sieve<Point_,Color_>::addSupport(const Point_& p, const Obj<InputSequence >& points, const Color_& color) {
      if (debug) {std::cout << "Adding a support " << std::endl;}
      for(typename InputSequence::iterator iter = points->begin(); iter != points->end(); ++iter) {
        if (debug) {std::cout << "Adding arrow from " << p << " to " << *iter << std::endl;}
        this->addArrow(p, *iter, color);
      }
    };

    template <typename Point_, typename Color_> 
    void Sieve<Point_,Color_>::add(const Obj<Sieve<Point_,Color_> >& sieve) {
      const typename ::boost::multi_index::index<ArrowSet,target>::type& cones = ::boost::multi_index::get<target>(sieve.arrows);
      
      for(typename ::boost::multi_index::index<ArrowSet,target>::type::iterator iter = cones.begin(); iter != cones.end(); ++iter) {
        this->addArrow(*iter);
      }
    };
    

    //
    // Structural queries
    //
    template <typename Point_, typename Color_> 
    int Sieve<Point_,Color_>::depth() {
      return this->maxDepth;
    };

    template <typename Point_, typename Color_> 
    int Sieve<Point_,Color_>::depth(const Point_& p) {
        return ::boost::multi_index::get<point>(this->strata).find(p)->depth;
    };
    
    template <typename Point_, typename Color_> 
    template<typename InputSequence> 
    int Sieve<Point_,Color_>::depth(const Obj<InputSequence>& points) {
      const typename ::boost::multi_index::index<StratumSet,point>::type& index = 
        ::boost::multi_index::get<point>(this->strata);
      int maxDepth = -1;
      
      for(typename InputSequence::iterator iter = points->begin(); iter != points->end(); ++iter) {
        maxDepth = std::max(maxDepth, index.find(*iter)->depth);
      }
      return maxDepth;
    };

    template <typename Point_, typename Color_> 
    int Sieve<Point_,Color_>::height() {
      return this->maxHeight;
    };
    
    template <typename Point_, typename Color_> 
    int Sieve<Point_,Color_>::height(const Point_& p) {
      return ::boost::multi_index::get<point>(this->strata).find(p)->height;
    };
    
    template <typename Point_, typename Color_> 
    template<typename InputSequence> 
    int Sieve<Point_,Color_>::height(const Obj<InputSequence>& points) {
      const typename ::boost::multi_index::index<StratumSet,point>::type& index = ::boost::multi_index::get<point>(this->strata);
      int maxHeight = -1;
      
      for(typename InputSequence::iterator iter = points->begin(); iter != points->end(); ++iter) {
        maxHeight = std::max(maxHeight, index.find(*iter)->height);
      }
      return maxHeight;
    };
    
    template <typename Point_, typename Color_> 
    int Sieve<Point_,Color_>::diameter() {
      int globalDiameter;
      int ierr = MPI_Allreduce(&this->graphDiameter, &globalDiameter, 1, MPI_INT, MPI_MAX, this->comm);
      CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Allreduce"));
      return globalDiameter;
    };
    
    template <typename Point_, typename Color_> 
    int Sieve<Point_,Color_>::diameter(const Point_& p) {
      return this->depth(p) + this->height(p);
    };
    
    template <typename Point_, typename Color_> 
    Obj<typename Sieve<Point_,Color_>::depthSequence> Sieve<Point_,Color_>::depthStratum(int d) {
      return depthSequence(::boost::multi_index::get<depthTag>(this->strata), d);
    };
    
    template <typename Point_, typename Color_> 
    Obj<typename Sieve<Point_,Color_>::heightSequence> Sieve<Point_,Color_>::heightStratum(int h) {
      return heightSequence(::boost::multi_index::get<heightTag>(this->strata), h);
    };
    
    template <typename Point_, typename Color_> 
    Obj<typename Sieve<Point_,Color_>::markerSequence> Sieve<Point_,Color_>::markerStratum(int m) {
      return markerSequence(::boost::multi_index::get<marker>(this->strata), m);
    };

    template <typename Point_, typename Color_> 
    void Sieve<Point_,Color_>::setStratification(bool doStratify) {
      this->stratification = doStratify;
    };
    
    template <typename Point_, typename Color_> 
    bool Sieve<Point_,Color_>::getStratification() {
      return this->stratification;
    };
    
    //
    // Structural manipulation
    // 
    
    template <typename Point_, typename Color_> 
    void Sieve<Point_,Color_>::setMarker(const point_type& p, const marker_type& marker) {
      typename ::boost::multi_index::index<StratumSet,point>::type& index = ::boost::multi_index::get<point>(this->strata);
      typename ::boost::multi_index::index<StratumSet,point>::type::iterator i = index.find(p);
      if (i != index.end()) {
        index.modify(i, changeMarker(marker));
      }
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    void Sieve<Point_,Color_>::setMarker(const Obj<InputSequence>& points, const marker_type& marker) {
      typename ::boost::multi_index::index<StratumSet,point>::type& index = ::boost::multi_index::get<point>(this->strata);
      changeMarker changer(marker);
      
      for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
        typename ::boost::multi_index::index<StratumSet,point>::type::iterator i = index.find(*p_itor);
        if (i != index.end()) {
          index.modify(i, changer);
        }
      }
    };

    // __FUNCT__ used by the logging macros
    #undef __FUNCT__
    #define __FUNCT__ "Sieve::stratify"
    template <typename Point_, typename Color_> 
    void Sieve<Point_,Color_>::stratify(bool show) {
      ALE_LOG_EVENT_BEGIN;
      this->__computeDegrees();
      // FIX: We would like to avoid the copy here with cone() and support()
      this->__computeClosureHeights(this->cone(this->leaves()));
      this->__computeStarDepths(this->support(this->roots()));
      
      if (debug || show) {
        const typename ::boost::multi_index::index<StratumSet,point>::type& points = ::boost::multi_index::get<point>(this->strata);
        for(typename ::boost::multi_index::index<StratumSet,point>::type::iterator i = points.begin(); i != points.end(); i++) {
          std::cout << *i << std::endl;
        }
      }
      ALE_LOG_EVENT_END;
    };


    template <typename Point_, typename Color_> 
    void Sieve<Point_,Color_>::__computeDegrees() {
      const typename ::boost::multi_index::index<ArrowSet,target>::type& cones = ::boost::multi_index::get<target>(this->arrows);
      const typename ::boost::multi_index::index<ArrowSet,source>::type& supports = ::boost::multi_index::get<source>(this->arrows);
      typename ::boost::multi_index::index<StratumSet,point>::type& points = ::boost::multi_index::get<point>(this->strata);
      
      for(typename ::boost::multi_index::index<ArrowSet,target>::type::iterator c_iter = cones.begin(); c_iter != cones.end();++c_iter){
        if (points.find(c_iter->target) != points.end()) {
          typename ::boost::multi_index::index<StratumSet,point>::type::iterator i = points.find(c_iter->target);
          
          points.modify(i, changeIndegree(cones.count(c_iter->target)));
        } else {
          StratumPoint p;
          
          p.point    = c_iter->target;
          p.indegree = cones.count(c_iter->target);
          this->strata.insert(p);
        }
      }
      
      for(typename ::boost::multi_index::index<ArrowSet,source>::type::iterator s_iter = supports.begin(); s_iter != supports.end(); ++s_iter) {
        if (points.find(s_iter->source) != points.end()) {
          typename ::boost::multi_index::index<StratumSet,point>::type::iterator i = points.find(s_iter->source);
          
          points.modify(i, changeOutdegree(supports.count(s_iter->source)));
        } else {
          StratumPoint p;
          
          p.point     = s_iter->source;
          p.outdegree = supports.count(s_iter->source);
          this->strata.insert(p);
        }
      }
    };

    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    void Sieve<Point_,Color_>::__computeClosureHeights(const Obj<InputSequence>& points) {
      typename ::boost::multi_index::index<StratumSet,point>::type& index = ::boost::multi_index::get<point>(this->strata);
      Obj<PointSet> modifiedPoints = PointSet();
      
      for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
        // Compute the max height of the points in the support of p, and add 1
        int h0 = this->height(*p_itor);
        int h1 = this->height(this->support(*p_itor)) + 1;
        if(h1 != h0) {
          typename ::boost::multi_index::index<StratumSet,point>::type::iterator i = index.find(*p_itor);
          index.modify(i, changeHeight(h1));
          if (h1 > this->maxHeight) this->maxHeight = h1;
          modifiedPoints->insert(*p_itor);
        }
      }
      // FIX: We would like to avoid the copy here with cone()
      if(modifiedPoints->size() > 0) {
        this->__computeClosureHeights(this->cone(modifiedPoints));
      }
    };

    
    template <typename Point_, typename Color_> 
    template<class InputSequence> 
    void Sieve<Point_,Color_>::__computeStarDepths(const Obj<InputSequence>& points) {
      typename ::boost::multi_index::index<StratumSet,point>::type& index = ::boost::multi_index::get<point>(this->strata);
      Obj<PointSet> modifiedPoints = PointSet();
      
      for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
        // Compute the max depth of the points in the support of p, and add 1
        int d0 = this->depth(*p_itor);
        int d1 = this->depth(this->cone(*p_itor)) + 1;
        if(d1 != d0) {
          index.modify(index.find(*p_itor), changeDepth(d1));
          if (d1 > this->maxDepth) this->maxDepth = d1;
          modifiedPoints->insert(*p_itor);
        }
      }
      // FIX: We would like to avoid the copy here with cone()
      if(modifiedPoints->size() > 0) {
        this->__computeStarDepths(this->support(modifiedPoints));
      }
    };
  } // namespace def
  
} // namespace ALE

#endif
