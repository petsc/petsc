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
    typedef std::set<Point> PointSet;

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
      struct sourceColor{};
      struct targetColor{};
      typedef Arrow<Color> SieveArrow;
      typedef ::boost::multi_index::multi_index_container<
        SieveArrow,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<source>, BOOST_MULTI_INDEX_MEMBER(SieveArrow,Data,source)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<target>, BOOST_MULTI_INDEX_MEMBER(SieveArrow,Data,target)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<color>,  BOOST_MULTI_INDEX_MEMBER(SieveArrow,Color,color)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<sourceColor>,
            ::boost::multi_index::composite_key<
              SieveArrow, BOOST_MULTI_INDEX_MEMBER(SieveArrow,Data,source), BOOST_MULTI_INDEX_MEMBER(SieveArrow,Color,color)>
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<targetColor>,
            ::boost::multi_index::composite_key<
              SieveArrow, BOOST_MULTI_INDEX_MEMBER(SieveArrow,Data,target), BOOST_MULTI_INDEX_MEMBER(SieveArrow,Color,color)>
          >
        >,
        ALE_ALLOCATOR<SieveArrow>
      > ArrowSet;
      ArrowSet arrows;

      struct point {};
      struct depth {};
      struct height {};
      struct StratumPoint {
        Data  point;
        Color color;
        int   depth;
        int   height;
      };
      typedef ::boost::multi_index::multi_index_container<
        StratumPoint,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<point>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,Data,point)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<color>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,Color,color)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<depth>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,int,depth)>,
          ::boost::multi_index::ordered_non_unique<
          ::boost::multi_index::tag<height>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,int,height)>
        >,
        ALE_ALLOCATOR<StratumPoint>
      > StratumSet;
      StratumSet strata;
      bool       stratification; 
      int        maxDepth;
      int        maxHeight;
      int        sieveDiameter;
      std::set<Point> _roots;
      std::set<Point> _leaves;
    public:
      class baseSequence {
        const typename ::boost::multi_index::index<ArrowSet,target>::type& baseIndex;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef Data  value_type;
          typedef int   difference_type;
          typedef Data* pointer;
          typedef Data& reference;
          typename boost::multi_index::index<ArrowSet,target>::type::iterator arrowIter;

          iterator(const typename boost::multi_index::index<ArrowSet,target>::type::iterator& iter) {
            this->arrowIter = typename boost::multi_index::index<ArrowSet,target>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->arrowIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->arrowIter); ++this->arrowIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->arrowIter == itor.arrowIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->arrowIter != itor.arrowIter;};
          virtual const Data& operator*() const {return this->arrowIter->target;};
        };

        baseSequence(const typename ::boost::multi_index::index<ArrowSet,target>::type& base) : baseIndex(base) {};
        virtual ~baseSequence() {};
        virtual iterator    begin() {return iterator(this->baseIndex.begin());};
        virtual iterator    end()   {return iterator(this->baseIndex.end());};
        virtual std::size_t size()  {return this->baseIndex.size();};
      };

      class coneSequence {
        const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& coneIndex;
        const Data key;
        const Color color;
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
          virtual bool        operator==(const iterator& itor) const {return this->arrowIter == itor.arrowIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->arrowIter != itor.arrowIter;};
          virtual const Data& operator*() const {return this->arrowIter->source;};
        };

        coneSequence(const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& cone, const Data& p) : coneIndex(cone), key(p), color(Color()), useColor(0) {};
        coneSequence(const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& cone, const Data& p, const Color& c) : coneIndex(cone), key(p), color(c), useColor(1) {};
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
        virtual std::size_t size()  {
          if (useColor) {
            return this->coneIndex.count(::boost::make_tuple(key,color));
          } else {
            return this->coneIndex.count(::boost::make_tuple(key));
          }
        };
      };

      class supportSequence {
        const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& supportIndex;
        const Data key;
        const Color color;
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
          virtual const Data& operator*() const {return this->arrowIter->target;};
        };

        supportSequence(const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& support, const Data& p) : supportIndex(support), key(p), color(Color()), useColor(0) {};
        supportSequence(const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& support, const Data& p, const Color& c) : supportIndex(support), key(p), color(c), useColor(1) {};
        virtual ~supportSequence() {};
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
        virtual std::size_t size()  {
          if (useColor) {
            return this->coneIndex.count(::boost::make_tuple(key,color));
          } else {
            return this->coneIndex.count(::boost::make_tuple(key));
          }
        };
      };

      class depthSequence {
        const typename ::boost::multi_index::index<StratumSet,depth>::type& depthIndex;
        const int d;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef Data  value_type;
          typedef int   difference_type;
          typedef Data* pointer;
          typedef Data& reference;
          typename boost::multi_index::index<StratumSet,target>::type::iterator arrowIter;

          iterator(const typename boost::multi_index::index<StratumSet,target>::type::iterator& iter) {
            this->arrowIter = typename boost::multi_index::index<StratumSet,target>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->arrowIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->arrowIter); ++this->arrowIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->arrowIter == itor.arrowIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->arrowIter != itor.arrowIter;};
          virtual const Data& operator*() const {return this->arrowIter->target;};
        };

        depthSequence(const typename ::boost::multi_index::index<StratumSet,target>::type& depth, const int d) : depthIndex(depth), d(d) {};
        virtual ~depthSequence() {};
        virtual iterator    begin() {return iterator(this->depthIndex.lower_bound(d));};
        virtual iterator    end()   {return iterator(this->depthIndex.upper_bound(d));};
        virtual std::size_t size()  {return this->depthIndex.count(d);};
      };

      class heightSequence {
        const typename ::boost::multi_index::index<StratumSet,height>::type& heightIndex;
        const int h;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef Data  value_type;
          typedef int   difference_type;
          typedef Data* pointer;
          typedef Data& reference;
          typename boost::multi_index::index<StratumSet,target>::type::iterator arrowIter;

          iterator(const typename boost::multi_index::index<StratumSet,target>::type::iterator& iter) {
            this->arrowIter = typename boost::multi_index::index<StratumSet,target>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->arrowIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->arrowIter); ++this->arrowIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->arrowIter == itor.arrowIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->arrowIter != itor.arrowIter;};
          virtual const Data& operator*() const {return this->arrowIter->target;};
        };

        heightSequence(const typename ::boost::multi_index::index<StratumSet,target>::type& height, const int h) : heightIndex(height), h(h) {};
        virtual ~heightSequence() {};
        virtual iterator    begin() {return iterator(this->heightIndex.lower_bound(h));};
        virtual iterator    end()   {return iterator(this->heightIndex.upper_bound(h));};
        virtual std::size_t size()  {return this->heightIndex.count(h);};
      };

      Sieve() : stratification(false), maxDepth(-1), maxHeight(-1), sieveDiameter(-1) {};
      // The basic Sieve interface
      void clear() {
        this->arrows.clear();
      };
      Obj<coneSequence> cone(const Data& p) {
        return coneSequence(::boost::multi_index::get<targetColor>(this->arrows), p);
      }
      template<class InputSequence> Obj<PointSet> cone(const Obj<InputSequence>& points) {
        return this->nCone(points, 1);
      };
      Obj<coneSequence> cone(const Data& p, const Color& color) {
        return coneSequence(::boost::multi_index::get<targetColor>(this->arrows), p, color);
      }
      template<class InputSequence> Obj<PointSet> cone(const Obj<InputSequence>& points, const Color& color) {
        return this->nCone(points, 1, color);
      };
      template<class InputSequence> Obj<PointSet> nCone(const Obj<InputSequence>& points, const int& n) {
        return this->nCone(points, n, Color(), false);
      };
      template<class InputSequence> Obj<PointSet> nCone(const Obj<InputSequence>& points, const int& n, const Color& color, bool useColor = true) {
        Obj<PointSet> cone = PointSet();
        Obj<PointSet> base = PointSet();

        cone->insert(points->begin(), points->end());
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
      Obj<supportSequence> support(const Data& p) {
        return supportSequence(::boost::multi_index::get<sourceColor>(this->arrows), p);
      };
      template<class InputSequence> Obj<supportSequence> support(const Obj<InputSequence>& points) {
        return this->nSupport(points, 1);
      };
      Obj<supportSequence> support(const Data& p, const Color& color) {
        return supportSequence(::boost::multi_index::get<sourceColor>(this->arrows), p, color);
      };
      template<class InputSequence> Obj<supportSequence> support(const Obj<InputSequence>& points, const Color& color) {
        return this->nSupport(points, 1, color);
      };
      template<class InputSequence> Obj<supportSequence> nSupport(const Obj<InputSequence>& points, const int& n) {
        return this->nSupport(points, n, Color(), false);
      };
      template<class InputSequence> Obj<supportSequence> nSupport(const Obj<InputSequence>& points, const int& n, const Color& color, bool useColor = true) {
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
      // Iterated versions
      Obj<PointSet> closure(const Data& p) {
        return nClosure(p, this->depth());
      };
      Obj<PointSet> closure(const Data& p, const Color& color) {
        return nClosure(p, this->depth(), color);
      };
      template<class InputSequence> Obj<PointSet> closure(const Obj<InputSequence>& points) {
        return nClosure(points, this->depth());
      };
      template<class InputSequence> Obj<PointSet> closure(const Obj<InputSequence>& points, const Color& color) {
        return nClosure(points, this->depth(), color);
      };
      Obj<PointSet> nClosure(const Data& p, const int& n) {
        return this->nClosure(p, n, Color(), false);
      };
      Obj<PointSet> nClosure(const Data& p, const int& n, const Color& color, bool useColor = true) {
        Obj<PointSet> cone = PointSet();

        cone->insert(p);
        return this->__nClosure(cone, n, color, useColor);
      };
      template<class InputSequence> Obj<PointSet> nClosure(const Obj<InputSequence>& points, const int& n) {
        return this->nClosure(points, n, Color(), false);
      };
      template<class InputSequence> Obj<PointSet> nClosure(const Obj<InputSequence>& points, const int& n, const Color& color, bool useColor = true) {
        Obj<PointSet> cone = PointSet();

        cone->insert(points->begin(), points->end());
        return this->__nClosure(cone, n, color, useColor);
      }
    private:
      template<class InputSequence> Obj<PointSet> __nClosure(Obj<InputSequence>& cone, const int& n, const Color& color, bool useColor) {
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
    public:
      Obj<PointSet> star(const Data& p) {
        return nStar(p, this->height());
      };
      Obj<PointSet> star(const Data& p, const Color& color) {
        return nStar(p, this->depth(), color);
      };
      template<class InputSequence> Obj<PointSet> star(const Obj<InputSequence>& points) {
        return nStar(points, this->height());
      };
      template<class InputSequence> Obj<PointSet> star(const Obj<InputSequence>& points, const Color& color) {
        return nStar(points, this->height(), color);
      };
      Obj<PointSet> nStar(const Data& p, const int& n) {
        return this->nStar(p, n, Color(), false);
      };
      Obj<PointSet> nStar(const Data& p, const int& n, const Color& color, bool useColor = true) {
        Obj<PointSet> support = PointSet();

        support->insert(p);
        return this->__nStar(support, n, color, useColor);
      };
      template<class InputSequence> Obj<PointSet> nStar(const Obj<InputSequence>& points, const int& n) {
        return this->nStar(points, n, Color(), false);
      };
      template<class InputSequence> Obj<PointSet> nStar(const Obj<InputSequence>& points, const int& n, const Color& color, bool useColor = true) {
        Obj<PointSet> support = PointSet();

        support->insert(points->begin(), points->end());
        return this->__nStar(cone, n, color, useColor);
      };
    private:
      template<class InputSequence> Obj<PointSet> __nStar(Obj<InputSequence>& support, const int& n, const Color& color, bool useColor) {
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
    public:
      // Lattice methods
      template<class InputSequence> Obj<coneSequence> meet(const Obj<InputSequence>& pp);
      template<class InputSequence> Obj<coneSequence> meet(const Obj<InputSequence>& pp, const Color& color);
      template<class InputSequence> Obj<coneSequence> nMeet(const Obj<InputSequence>& pp, const int& n);
      template<class InputSequence> Obj<coneSequence> nMeet(const Obj<InputSequence>& pp, const int& n, const Color& color);
      template<class InputSequence> Obj<supportSequence> join(const Obj<InputSequence>& pp);
      template<class InputSequence> Obj<supportSequence> join(const Obj<InputSequence>& pp, const Color& color);
      template<class InputSequence> Obj<supportSequence> nJoin(const Obj<InputSequence>& pp, const int& n);
      template<class InputSequence> Obj<supportSequence> nJoin(const Obj<InputSequence>& pp, const int& n, const Color& color);
      // Manipulation
      void addArrow(const Data& p, const Data& q) {
        this->addArrow(p, q, Color());
      };
      void addArrow(const Data& p, const Data& q, const Color& color) {
        this->arrows.insert(SieveArrow(p, q, color));
        std::cout << "Added " << SieveArrow(p, q, color);
      };
      template<class InputSequence> void addCone(const Obj<InputSequence >& points, const Data& p) {
        this->addCone(points, p, Color());
      };
      template<class InputSequence> void addCone(const Obj<InputSequence >& points, const Data& p, const Color& color){
        std::cout << "Adding a cone " << std::endl;
        for(typename InputSequence::iterator iter = points->begin(); iter != points->end(); ++iter) {
          std::cout << "Adding arrow from " << *iter << " to " << p << "(" << color << ")" << std::endl;
          this->addArrow(*iter, p, color);
        }
      };
      template<class InputSequence> void addSupport(const Data& p, const Obj<InputSequence >& points) {
        this->addSupport(p, points, Color());
      };
      template<class InputSequence> void addSupport(const Data& p, const Obj<InputSequence >& points, const Color& color) {
        std::cout << "Adding a support " << std::endl;
        for(typename InputSequence::iterator iter = points->begin(); iter != points->end(); ++iter) {
          std::cout << "Adding arrow from " << p << " to " << *iter << std::endl;
          this->addArrow(p, *iter, color);
        }
      };
      void add(const Obj<Sieve<Data,Color> >& sieve) {
        const typename ::boost::multi_index::index<ArrowSet,target>::type& cones = ::boost::multi_index::get<target>(sieve.arrows);

        for(typename ::boost::multi_index::index<ArrowSet,target>::type::iterator iter = cones.begin(); iter != cones.end(); ++iter) {
          this->addArrow(*iter);
        }
      };
      // Parallelism
      Obj<std::map<Data, Sieve<Data,Color> > > completion(const Sieve<Data,Color>& base);
      // Views
      Obj<baseSequence> base() {
        // Could probably use height 0
        return baseSequence(::boost::multi_index::get<target>(this->arrows));
      };
      // Structural methods
      int depth() {return this->maxDepth;};
      int depth(const Data& p) {
        return ::boost::multi_index::get<point>(this->strata).find(p)->depth;
      };
      int height() {return this->maxHeight;};
      int height(const Data& p) {
        return ::boost::multi_index::get<point>(this->strata).find(p)->height;
      };
      int diameter() {
        int globalDiameter;
        int ierr = MPI_Allreduce(&this->sieveDiameter, &globalDiameter, 1, MPI_INT, MPI_MAX, this->comm);
        CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Allreduce"));
        ALE_LOG_STAGE_END;
        return globalDiameter;
      };
      int diameter(const Data& p) {
        return this->depth(p) + this->height(p);
      };
      Obj<depthSequence> depthStratum(const int& d) {
        return depthSequence(::boost::multi_index::get<depth>(this->strata), d);
      };
      Obj<heightSequence> heightStratum(const int& h) {
        return heightSequence(::boost::multi_index::get<height>(this->strata), h);
      };
      void setStratification(bool doStratify) {this->stratification = doStratify;};
      bool getStratification() {return this->stratification;};
      void stratify() {
        // FIX: We would like to avoid the copy here with cone() and support()
        this->__computeClosureHeights(this->cone(this->_leaves));
        this->__computeStarDepths(this->support(this->_roots));
      };
    private:
      void __computeRootsAndLeaves() {
        const typename ::boost::multi_index::index<ArrowSet,target>::type& cones = ::boost::multi_index::get<target>(this->arrows);
        const typename ::boost::multi_index::index<ArrowSet,source>::type& supports = ::boost::multi_index::get<source>(this->arrows);

        this->_roots.clear();
        for(typename ::boost::multi_index::index<ArrowSet,source>::type::iterator s_iter = supports.begin(); s_iter != supports.end(); ++s_iter) {
          if (cones.find(s_iter->source) == cones.end()) {
            this->_roots.insert(s_iter->source);
          }
        }

        this->_leavas.clear();
        for(typename ::boost::multi_index::index<ArrowSet,target>::type::iterator c_iter = cones.begin(); c_iter != cones.end(); ++c_iter) {
          if (supports.find(c_iter->target) == supports.end()) {
            this->_leaves.insert(c_iter->target);
          }
        }
      };

      template<class InputSequence> void __computeClosureHeights(const Obj<InputSequence>& points) {
        const typename ::boost::multi_index::index<StratumSet,point>::type& index = ::boost::multi_index::get<point>(this->strata);
        Obj<PointSet> modifiedPoints = PointSet();

        for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
          // Compute the max height of the points in the support of p, and add 1
          int h0 = this->height(*p_itor);
          int h1 = this->height(this->support(*p_itor)) + 1;
          if(h1 != h0) {
            index->find(*p_itor)->height = h1;
            modifiedPoints->insert(*p_itor);
          }
        }
        // FIX: We would like to avoid the copy here with cone()
        if(modifiedPoints->size() > 0) {
          this->__computeClosureHeights(this->cone(modifiedPoints));
        }
      };

      template<class InputSequence> void __computeStarDepths(const Obj<PointSet>& points) {
        const typename ::boost::multi_index::index<StratumSet,point>::type& index = ::boost::multi_index::get<point>(this->strata);
        Obj<PointSet> modifiedPoints = PointSet();

        for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
          // Compute the max depth of the points in the support of p, and add 1
          int d0 = this->depth(*p_itor);
          int d1 = this->maxDepth(this->cone(*p_itor)) + 1;
          if(d1 != d0) {
            index->find(*p_itor)->depth = d1;
            modifiedPoints->insert(*p_itor);
          }
        }
        // FIX: We would like to avoid the copy here with cone()
        if(modifiedPoints->size() > 0) {
          this->__computeStarDepths(this->support(modifiedPoints));
        }
      };
    public:
    };

  } // namespace def



} // namespace ALE

#endif
