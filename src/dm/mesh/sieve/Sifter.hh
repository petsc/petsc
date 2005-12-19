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
            ::boost::multi_index::tag<source>, BOOST_MULTI_INDEX_MEMBER(SieveArrow,Point,source)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<target>, BOOST_MULTI_INDEX_MEMBER(SieveArrow,Point,target)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<color>,  BOOST_MULTI_INDEX_MEMBER(SieveArrow,Color,color)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<sourceColor>,
            ::boost::multi_index::composite_key<
              SieveArrow, BOOST_MULTI_INDEX_MEMBER(SieveArrow,Point,source), BOOST_MULTI_INDEX_MEMBER(SieveArrow,Color,color)>
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<targetColor>,
            ::boost::multi_index::composite_key<
              SieveArrow, BOOST_MULTI_INDEX_MEMBER(SieveArrow,Point,target), BOOST_MULTI_INDEX_MEMBER(SieveArrow,Color,color)>
          >
        >
      > ArrowSet;
      //
      ArrowSet        arrows;
      std::set<Color> colors;
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

        coneSequence(const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& cone, const Point& p) : coneIndex(cone), key(p), color(Color()), useColor(0) {};
        coneSequence(const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& cone, const Point& p, const Color& c) : coneIndex(cone), key(p), color(c), useColor(1) {};
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

        supportSequence(const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& support, const Point& p) : supportIndex(support), key(p), color(Color()), useColor(0) {};
        supportSequence(const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& support, const Point& p, const Color& c) : supportIndex(support), key(p), color(c), useColor(1) {};
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

      Sieve() {};
      // The basic Sieve interface
      void clear() {
        this->arrows.clear();
      };
      Obj<coneSequence> cone(const Point& p) {
        return coneSequence(::boost::multi_index::get<targetColor>(this->arrows), p);
      }
      template<class InputSequence> Obj<PointSet> cone(const Obj<InputSequence>& points) {
        return this->nCone(points, 1);
      };
      Obj<coneSequence> cone(const Point& p, const Color& color) {
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
      Obj<supportSequence> support(const Point& p) {
        return supportSequence(::boost::multi_index::get<sourceColor>(this->arrows), p);
      };
      template<class InputSequence> Obj<supportSequence> support(const Obj<InputSequence>& points) {
        return this->nSupport(points, 1);
      };
      Obj<supportSequence> support(const Point& p, const Color& color) {
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
      Obj<PointSet> closure(const Point& p) {
        return nClosure(p, this->depth());
      };
      Obj<PointSet> closure(const Point& p, const Color& color) {
        return nClosure(p, this->depth(), color);
      };
      template<class InputSequence> Obj<PointSet> closure(const Obj<InputSequence>& points) {
        return nClosure(points, this->depth());
      };
      template<class InputSequence> Obj<PointSet> closure(const Obj<InputSequence>& points, const Color& color) {
        return nClosure(points, this->depth(), color);
      };
      Obj<PointSet> nClosure(const Point& p, const int& n) {
        return this->nClosure(p, n, Color(), false);
      };
      Obj<PointSet> nClosure(const Point& p, const int& n, const Color& color, bool useColor = true) {
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
      Obj<PointSet> star(const Point& p) {
        return nStar(p, this->height());
      };
      Obj<PointSet> star(const Point& p, const Color& color) {
        return nStar(p, this->depth(), color);
      };
      template<class InputSequence> Obj<PointSet> star(const Obj<InputSequence>& points) {
        return nStar(points, this->height());
      };
      template<class InputSequence> Obj<PointSet> star(const Obj<InputSequence>& points, const Color& color) {
        return nStar(points, this->height(), color);
      };
      Obj<PointSet> nStar(const Point& p, const int& n) {
        return this->nStar(p, n, Color(), false);
      };
      Obj<PointSet> nStar(const Point& p, const int& n, const Color& color, bool useColor = true) {
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
      Obj<std::map<Point, Sieve<Data,Color> > > completion(const Sieve<Data,Color>& base);
      // Views
      Obj<baseSequence> base() {
        // Could probably use height 0
        return baseSequence(::boost::multi_index::get<target>(this->arrows));
      };
      // Structural methods
      int depth();
      int depth(const Data& p);
      int height();
      int height(const Data& p);
      int diameter();
      int diameter(const Data& p);
      Obj<coneSequence> depthStratum(const int& depth);
      Obj<coneSequence> heightStratum(const int& height);
      void              setStratification(bool on);
      bool              getStratification();
      void              stratify();
    };

    //
    // CoSieve:
    //   Value is the type of function space (PreSheaf?) for fibers over the mesh
    //
    template <typename Data, typename Value>
    class CoSieve {
      template<class InputSequence> const Value *restrict(const Data& support, const Obj<InputSequence>& chain);
      template<class InputSequence> void         assemble(const Data& support, const Obj<InputSequence>& chain, const Value values[]);

      // \rho
      //     This needs some sort of policy argument
      void reduce();

      // Refine the support chain
      void refine();
      template<class InputSequence> void refine(const Obj<InputSequence>& chain);
      // Coarsen the support chain
      void coarsen();
      template<class InputSequence> void coarsen(const Obj<InputSequence>& chain);
    };
  } // namespace def



} // namespace ALE

#endif
