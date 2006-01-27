#ifndef included_ALE_BiGraph_hh
#define included_ALE_BiGraph_hh

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <iostream>

// ALE extensions

#ifndef  included_Sifter_hh
#include <Sifter.hh>
#endif

namespace ALE {

  // The original implementation of BiGraph is in namespace One.
  namespace One {
    //
    // BiGraph (short for BipartiteGraph) implements a sequential interface similar to that of Sieve (below),
    // except the source and target points may have different types and iterated operations (e.g., nCone, closure)
    // are not available.
    // 
    template<typename Source_, typename Target_, typename Color_>
    class BiGraph {
    public:
      typedef Source_ source_type;
      typedef Target_ target_type;
      typedef Color_  color_type;
      int debug;
    private:
      // Arrow storage
      struct source{};
      struct target{};
      struct color{};
      struct sourceColor{};
      struct targetColor{};
      typedef ALE::def::Arrow<source_type,target_type,color_type> Arrow_;
      typedef ::boost::multi_index::multi_index_container<
        Arrow_,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<source>, BOOST_MULTI_INDEX_MEMBER(Arrow_,source_type,source)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<target>, BOOST_MULTI_INDEX_MEMBER(Arrow_,target_type,target)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<color>,  BOOST_MULTI_INDEX_MEMBER(Arrow_,color_type,color)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<sourceColor>,
            ::boost::multi_index::composite_key<
              Arrow_, BOOST_MULTI_INDEX_MEMBER(Arrow_,source_type,source), BOOST_MULTI_INDEX_MEMBER(Arrow_,color_type,color)>
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<targetColor>,
            ::boost::multi_index::composite_key<
              Arrow_, BOOST_MULTI_INDEX_MEMBER(Arrow_,target_type,target), BOOST_MULTI_INDEX_MEMBER(Arrow_,color_type,color)>
          >
        >,
        ALE_ALLOCATOR<Arrow_>
      > ArrowSet;
      ArrowSet arrows;

      struct point{};
      struct depthTag{};
      struct heightTag{};
      struct indegree{};
      struct outdegree{};
      struct StratumPoint {
        target_type point;
        int depth;
        int height;
        int indegree;
        int outdegree;

        StratumPoint() : depth(0), height(0), indegree(0), outdegree(0) {};
        StratumPoint(const target_type& p) : point(p), depth(0), height(0), indegree(0), outdegree(0) {};
        // Printing
        friend std::ostream& operator<<(std::ostream& os, const StratumPoint& p) {
          os << "[" << p.point << ", "<< p.depth << ", "<< p.height << ", "<< p.indegree << ", "<< p.outdegree << "]";
          return os;
        };
      };
      typedef ::boost::multi_index::multi_index_container<
        StratumPoint,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<point>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,target_type,point)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<depthTag>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,int,depth)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<heightTag>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,int,height)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<indegree>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,int,indegree)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<outdegree>, BOOST_MULTI_INDEX_MEMBER(StratumPoint,int,outdegree)>
        >,
        ALE_ALLOCATOR<StratumPoint>
      > StratumSet;
      StratumSet strata;
      bool       stratification; 
      int        maxDepth;
      int        maxHeight;
      int        graphDiameter;
    public:
      BiGraph() : debug(0), stratification(false), maxDepth(-1), maxHeight(-1), graphDiameter(-1) {};
      // Return types
      class baseSequence {
        const typename ::boost::multi_index::index<StratumSet,indegree>::type& baseIndex;
      public:
        class iterator {
        public:
          typedef std::input_iterator_tag iterator_category;
          typedef target_type  value_type;
          typedef int          difference_type;
          typedef target_type* pointer;
          typedef target_type& reference;
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
          virtual const target_type& operator*() const {return this->pointIter->point;};
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
          typedef target_type  value_type;
          typedef int          difference_type;
          typedef target_type* pointer;
          typedef target_type& reference;
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
          virtual const target_type& operator*() const {return this->pointIter->point;};
        };

        capSequence(const typename ::boost::multi_index::index<StratumSet,outdegree>::type& cap) : capIndex(cap) {};
        virtual ~capSequence() {};
        virtual iterator    begin() {return iterator(this->capIndex.upper_bound(0));};
        virtual iterator    end()   {return iterator(this->capIndex.end());};
        virtual std::size_t size()  {return -1;};
      };
      class coneSequence {
        typename ::boost::multi_index::index<ArrowSet,targetColor>::type& coneIndex;
        const target_type key;
        const color_type  color;
        const bool        useColor;
        struct changeColor {
          changeColor(color_type newColor) : newColor(newColor) {};

          void operator()(Arrow_& p) {
            p.color = newColor;
          }
        private:
          color_type newColor;
        };
      public:
        class iterator {
          typename ::boost::multi_index::index<ArrowSet,targetColor>::type& index;
        public:
          typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator arrowIter;

          iterator(typename ::boost::multi_index::index<ArrowSet,targetColor>::type& index, const typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator& iter) : index(index) {
            this->arrowIter = typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator(iter);
          };
          virtual ~iterator() {};
          //
          virtual iterator    operator++() {++this->arrowIter; return *this;};
          virtual iterator    operator++(int n) {iterator tmp(this->index, this->arrowIter); ++this->arrowIter; return tmp;};
          virtual iterator    operator--() {--this->arrowIter; return *this;};
          virtual iterator    operator--(int n) {iterator tmp(this->index, this->arrowIter); --this->arrowIter; return tmp;};
          virtual bool        operator==(const iterator& itor) const {return this->arrowIter == itor.arrowIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->arrowIter != itor.arrowIter;};
          virtual const source_type& operator*() const {return this->arrowIter->source;};
          void                setColor(int newColor) {this->index.modify(this->arrowIter, changeColor(newColor));};
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
          virtual const source_type& operator*() const {return this->arrowIter->source;};
        };

        coneSequence(typename ::boost::multi_index::index<ArrowSet,targetColor>::type& cone, const target_type& p) : coneIndex(cone), key(p), color(color_type()), useColor(0) {};
        coneSequence(typename ::boost::multi_index::index<ArrowSet,targetColor>::type& cone, const target_type& p, const color_type& c) : coneIndex(cone), key(p), color(c), useColor(1) {};
        virtual ~coneSequence() {};
        virtual iterator    begin() {
          if (useColor) {
            return iterator(this->coneIndex, this->coneIndex.lower_bound(::boost::make_tuple(key,color)));
          } else {
            return iterator(this->coneIndex, this->coneIndex.lower_bound(::boost::make_tuple(key)));
          }
        };
        virtual iterator    end()   {
          if (useColor) {
            return iterator(this->coneIndex, this->coneIndex.upper_bound(::boost::make_tuple(key,color)));
          } else {
            return iterator(this->coneIndex, this->coneIndex.upper_bound(::boost::make_tuple(key)));
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
      };
      class supportSequence {
        const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& supportIndex;
        const source_type key;
        const color_type  color;
        const bool        useColor;
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
          virtual const target_type& operator*() const {return this->arrowIter->target;};
        };

        supportSequence(const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& support, const source_type& p) : supportIndex(support), key(p), color(color_type()), useColor(0) {};
        supportSequence(const typename ::boost::multi_index::index<ArrowSet,sourceColor>::type& support, const source_type& p, const color_type& c) : supportIndex(support), key(p), color(c), useColor(1) {};
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
      typedef std::set<source_type, std::less<source_type>, ALE_ALLOCATOR<source_type> > coneSet;
      typedef std::set<target_type, std::less<target_type>, ALE_ALLOCATOR<target_type> > supportSet;
      // Completion types (some redundant)
      //   Color of completions; encodes the color of the completed BiGraph as well as the rank of the process that contributed arrow.
      typedef std::pair<color_type, int> completion_color_type;
      typedef BiGraph<source_type, target_type, completion_color_type> completion_type;
      // Query methods
      Obj<capSequence>     cap() {
        return capSequence(::boost::multi_index::get<outdegree>(this->strata));
      };
      Obj<baseSequence>    base() {
        return baseSequence(::boost::multi_index::get<indegree>(this->strata));
      };
      Obj<coneSequence> cone(const target_type& p) {
        return coneSequence(::boost::multi_index::get<targetColor>(this->arrows), p);
      }
      template<class InputSequence> Obj<coneSet> cone(const Obj<InputSequence>& points) {
        return this->cone(points, color_type(), false);
      };
      Obj<coneSequence>    cone(const target_type& p, const color_type& color) {
        return coneSequence(::boost::multi_index::get<targetColor>(this->arrows), p, color);
      };
      template<class InputSequence>
      Obj<coneSet>         cone(const Obj<InputSequence>& points, const color_type& color, bool useColor = true) {
        Obj<coneSet> cone = coneSet();

        for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
          Obj<coneSequence> pCone;

          if (useColor) {
            pCone = this->cone(*p_itor, color);
          } else {
            pCone = this->cone(*p_itor);
          }
          cone->insert(pCone->begin(), pCone->end());
        }
        return cone;
      };
      Obj<supportSequence> support(const source_type& p) {
        return supportSequence(::boost::multi_index::get<sourceColor>(this->arrows), p);
      };
      Obj<supportSequence> support(const source_type& p, const color_type& color) {
        return supportSequence(::boost::multi_index::get<sourceColor>(this->arrows), p, color);
      };
      template<class sourceInputSequence>
      Obj<supportSet>      support(const Obj<sourceInputSequence>& sources);
      template<class sourceInputSequence>
      Obj<supportSet>      support(const Obj<sourceInputSequence>& sources, const color_type& color);
      bool                 supportContains(const source_type& p, const target_type& q) {
        //FIX: Shouldn't we just be able to query an arrow?
        Obj<supportSequence> support = this->support(p);
      
        for(typename supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); s_iter++) {
          if (*s_iter == q) return true;
        }
        return false;
      }
      // Lattice queries
      template<class targetInputSequence> 
      Obj<coneSequence> meet(const Obj<targetInputSequence>& targets);
      template<class targetInputSequence> 
      Obj<coneSequence> meet(const Obj<targetInputSequence>& targets, const color_type& color);
      template<class sourceInputSequence> 
      Obj<coneSequence> join(const Obj<sourceInputSequence>& sources);
      template<class sourceInputSequence> 
      Obj<coneSequence> join(const Obj<sourceInputSequence>& sources, const color_type& color);
      // Manipulation
      void clear() {
        this->arrows.clear();
        this->strata.clear();
      };
      void addArrow(const source_type& p, const target_type& q) {
        this->addArrow(p, q, color_type());
      };
      void addArrow(const source_type& p, const target_type& q, const color_type& color) {
        this->arrows.insert(Arrow_(p, q, color));
        //std::cout << "Added " << Arrow_(p, q, color);
      };
      void addCone(const source_type& source, const target_type& target){
        this->addArrow(source, target);
      };
      template<class sourceInputSequence> 
      void addCone(const Obj<sourceInputSequence>& sources, const target_type& target) {
        this->addCone(sources, target, color_type());
      };
      void addCone(const source_type& source, const target_type& target, const color_type& color) {
        this->addArrow(source, target, color);
      };
      template<class sourceInputSequence> 
      void addCone(const Obj<sourceInputSequence>& sources, const target_type& target, const color_type& color) {
        if (debug) {std::cout << "Adding a cone " << std::endl;}
        for(typename sourceInputSequence::iterator iter = sources->begin(); iter != sources->end(); ++iter) {
          if (debug) {std::cout << "Adding arrow from " << *iter << " to " << target << "(" << color << ")" << std::endl;}
          this->addArrow(*iter, target, color);
        }
      };
    private:
      void __clearCone(const target_type& p, const color_type& color, bool useColor) {
        typename ::boost::multi_index::index<ArrowSet,targetColor>::type& coneIndex = ::boost::multi_index::get<targetColor>(this->arrows);
        if (useColor) {
          coneIndex.erase(coneIndex.lower_bound(::boost::make_tuple(p,color)), coneIndex.upper_bound(::boost::make_tuple(p,color)));
        } else {
          coneIndex.erase(coneIndex.lower_bound(::boost::make_tuple(p)),       coneIndex.upper_bound(::boost::make_tuple(p)));
        }
      }
    public:
      void setCone(const source_type& source, const target_type& target){
        this->__clearCone(target, color_type(), false);
        this->addCone(source, target);
      };
      template<class sourceInputSequence> 
      void setCone(const Obj<sourceInputSequence>& sources, const target_type& target) {
        this->__clearCone(target, color_type(), false);
        this->addCone(sources, target, color_type());
      };
      void setCone(const source_type& source, const target_type& target, const color_type& color) {
        this->__clearCone(target, color, true);
        this->addCone(source, target, color);
      };
      template<class sourceInputSequence> 
      void setCone(const Obj<sourceInputSequence>& sources, const target_type& target, const color_type& color) {
        this->__clearCone(target, color, true);
        this->addCone(sources, target, color);
      };
      template<class targetInputSequence> 
      void addSupport(const source_type& source, const Obj<targetInputSequence >& targets);
      template<class targetInputSequence> 
      void addSupport(const source_type& source, const Obj<targetInputSequence >& targets, const color_type& color);
      void add(const Obj<BiGraph<source_type, target_type, color_type> >& bigraph);
    private:
      struct changeSource {
        changeSource(source_type newSource) : newSource(newSource) {};

        void operator()(Arrow_& p) {
          p.source = newSource;
        }
      private:
        source_type newSource;
      };
    public:
      void replaceSource(const source_type& s, const source_type& new_s) {
        typename ::boost::multi_index::index<ArrowSet,source>::type& index = ::boost::multi_index::get<source>(this->arrows);
        typename ::boost::multi_index::index<ArrowSet,source>::type::iterator i = index.find(s);
        if (i != index.end()) {
          index.modify(i, changeSource(new_s));
        } else {
          std::cout << "ERROR: Could not replace source " << s << " with " << new_s << std::endl;
        }
      };
    private:
      struct changeTarget {
        changeTarget(target_type newTarget) : newTarget(newTarget) {};

        void operator()(Arrow_& p) {
          p.target = newTarget;
        }
      private:
        target_type newTarget;
      };
    public:
      void replaceTarget(const target_type& t, const target_type& new_t) {
        typename ::boost::multi_index::index<ArrowSet,target>::type& index = ::boost::multi_index::get<target>(this->arrows);
        typename ::boost::multi_index::index<ArrowSet,target>::type::iterator i = index.find(t);
        if (i != index.end()) {
          index.modify(i, changeTarget(new_t));
        } else {
          std::cout << "ERROR: Could not replace target " << t << " with " << new_t << std::endl;
        }
      };
      void replaceSourceOfTarget(const target_type& t, const target_type& new_s) {
        typename ::boost::multi_index::index<ArrowSet,target>::type& index = ::boost::multi_index::get<target>(this->arrows);
        typename ::boost::multi_index::index<ArrowSet,target>::type::iterator i = index.find(t);
        if (i != index.end()) {
          index.modify(i, changeSource(new_s));
        } else {
          std::cout << "ERROR: Could not replace source of target" << t << " with " << new_s << std::endl;
        }
      }
    private:
      struct changeColor {
        changeColor(color_type newColor) : newColor(newColor) {};

        void operator()(Arrow_& p) {
          p.color = newColor;
        }
      private:
        color_type newColor;
      };
    public:
      bool replaceSourceColor(const source_type& p, const color_type& newColor) {
        typename ::boost::multi_index::index<ArrowSet,source>::type& index = ::boost::multi_index::get<source>(this->arrows);
        typename ::boost::multi_index::index<ArrowSet,source>::type::iterator i = index.find(p);
        if (i != index.end()) {
          index.modify(i, changeColor(newColor));
        } else {
          return false;
        }
        return true;
      };
      // Structural methods
      #undef __FUNCT__
      #define __FUNCT__ "BiGraph::stratify"
      void stratify() {
        ALE_LOG_EVENT_BEGIN;
        std::cout << "Stratifying" << std::endl;
        this->__computeDegrees();
        ALE_LOG_EVENT_END;
      }
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
      void __computeDegrees() {
        const typename ::boost::multi_index::index<ArrowSet,target>::type& cones = ::boost::multi_index::get<target>(this->arrows);
#if 0
        const typename ::boost::multi_index::index<ArrowSet,source>::type& supports = ::boost::multi_index::get<source>(this->arrows);
#endif
        typename ::boost::multi_index::index<StratumSet,point>::type& points = ::boost::multi_index::get<point>(this->strata);

        for(typename ::boost::multi_index::index<ArrowSet,target>::type::iterator c_iter = cones.begin(); c_iter != cones.end(); ++c_iter) {
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

#if 0
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
#endif
      };
    public:
      // Parallelism
      // Compute the cone completion and return it in a separate BiGraph with arrows labeled by completion_color_type colors.
      Obj<completion_type> coneCompletion();
      // Compute the support completion and return it in a separate BiGraph with arrows labeled by completion_color_type colors.
      Obj<completion_type> supportCompletion();
      // Merge the BiGraph with the provided completion; do nothing if the completion is Null
      void complete(const Obj<completion_type>& completion);
      // Merge the BiGraph with the provided completion or compute a cone completion and merge with it, if a Null completion provided
      void coneComplete(const Obj<completion_type>& completion = Obj<completion_type>());
      // Merge the BiGraph with the provided completion or compute a support completion and merge with it, if a Null completion provided
      void supportComplete(const Obj<completion_type>& completion = Obj<completion_type>());
    };
  } // namespace One


  // A more templated implementation of BiGraph is in namespace Two.
  namespace Two {
    
    // A result_iterator defines the generic way of iterating over a query result.
    // Members of type Value_ are extracted from each result record.
    template <typename Iterator_, typename MemberExtractor_>
    class result_iterator {
    public:
      // Standard iterator typedefs
      typedef std::input_iterator_tag iterator_category;
      typedef typename MemberExtractor_::result_type value_type;
      typedef MemberExtractor_                       extractor_type;
      typedef int                     difference_type;
      typedef value_type*             pointer;
      typedef value_type&             reference;
      
      // Underlying iterator type
      typedef Iterator_ itor_type;
    protected:
      // Underlying iterator 
      itor_type      _itor;
      // Member extractor
      extractor_type _ex;
    public:
      result_iterator(const itor_type& itor) {
        this->_itor = itor_type(itor);
      };
      virtual ~result_iterator() {};
      //
      virtual result_iterator   operator++() {++this->_itor; return *this;};
      virtual result_iterator   operator++(int n) {result_iterator tmp(this->_itor); ++this->_itor; return tmp;};
      virtual bool              operator==(const result_iterator& iter) const {return this->_itor == iter._itor;};
      virtual bool              operator!=(const result_iterator& iter) const {return this->_itor != iter._itor;};
      virtual const reference   operator*() const {return _ex(*(this->_itor));};
    };

    // reverse_result_iterator is the reverse of result_iterator
    template <typename Iterator_, typename Value_>
    class reverse_result_iterator : public result_iterator<Iterator_, Value_> {
    public:
      typedef Iterator_ itor_type;
    public:
      reverse_result_iterator(const itor_type& itor) {
        this->_itor = itor_type(itor);
      };
      virtual ~reverse_result_iterator() {};
      //
      virtual reverse_result_iterator     operator++() {--this->_itor; return *this;};
      virtual reverse_result_iterator     operator++(int n) {reverse_result_iterator tmp(this->_itor); --this->_itor; return tmp;};
      virtual bool                        operator==(const reverse_result_iterator& iter) const {return this->_itor == iter._itor;};
      virtual bool                        operator!=(const reverse_result_iterator& iter) const {return this->_itor != iter._itor;};
    };

    // OutputSequence defines a generic encapsulation of a result of a on a MultiIndexSet defined by the index with the Tag_.  
    // The result can be traversed yielding a Value_ object upon dereferencing.  To this end OutputSequence encapsulates the type 
    // of the iterator used to traverse the result, as well as the methods returning the extrema -- the beginning and ending iterators.
    // Specializations of OutputSequence will rely on specializations of iterator.
    template <typename MultiIndexSet_, typename Tag_, typename MemberExtractor_>
    class OutputSequence {
    public:
      // Basic encapsulated types
      typedef MultiIndexSet_                                                   set_type;
      typedef Tag_                                                             tag;
      typedef typename ::boost::multi_index::index<set_type, tag>::type        index_type;
      typedef result_iterator<typename index_type::iterator, MemberExtractor_> iterator;
    protected:
      index_type _index;
    public:
      // Basic interface
      OutputSequence(const OutputSequence& seq) : _index(seq._index) {};
      OutputSequence(const set_type& set) : _index(::boost::multi_index::get<tag>(set)) {};
      virtual ~OutputSequence() {};
      virtual iterator    begin() = 0;
      virtual iterator    end()   = 0;
      virtual std::size_t size();
    };

    // ReversibleOutputSequence extends OutputSequence to allow reverse traversals.
    // Specializations of ReversibleOutputSequence will rely on specializations of result_iterator and reverse_result_iterator,
    // which dereference to objects of Value_ type.
    template <typename MultiIndexSet_, typename Tag_, typename MemberExtractor_>
    class ReversibleOutputSequence : public OutputSequence<MultiIndexSet_, Tag_, MemberExtractor_> {
    public:
      typedef MemberExtractor_                                                          extractor_type;
      typedef MultiIndexSet_                                                            set_type;
      typedef Tag_                                                                      tag;
      typedef typename OutputSequence<MultiIndexSet_,Tag_,MemberExtractor_>::index_type index_type;
    public:
      // Encapsulated reverse_result_iterator type
      typedef typename OutputSequence<MultiIndexSet_,Tag_,MemberExtractor_>::iterator   iterator;
      typedef reverse_result_iterator<typename index_type::iterator, MemberExtractor_>  reverse_iterator;
      // Generic ReversibleOutputSequence interface
      ReversibleOutputSequence(const ReversibleOutputSequence& seq) : OutputSequence<MultiIndexSet_, Tag_, MemberExtractor_>(seq){};
      ReversibleOutputSequence(const set_type& set) : OutputSequence<MultiIndexSet_,Tag_,MemberExtractor_>(set) {};
      virtual ~ReversibleOutputSequence() {};
      virtual reverse_iterator    rbegin() = 0;
      virtual reverse_iterator    rend()   = 0;
    };


    //
    // BiGraph (short for BipartiteGraph) implements a sequential interface similar to that of Sieve (below),
    // except the source and target points may have different types and iterated operations (e.g., nCone, closure)
    // are not available.
    // 
    template<typename Source_, typename Target_, typename Color_>
    class BiGraph {
    public:
      typedef Source_ source_type;
      typedef Target_ target_type;
      typedef Color_  color_type;
      typedef ALE::def::Arrow<source_type,target_type,color_type> Arrow_;
      int debug;
    private:

      //
      // Arrow storage
      //
      struct sourceTag{};
      struct targetTag{};
      struct colorTag{};
      struct targetSourceTag{};
      struct sourceColorTag{};
      struct colorSourceTag{};
      struct targetColorTag{};
      // Arrow record set
      typedef ::boost::multi_index::multi_index_container<
        Arrow_,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<sourceTag>, BOOST_MULTI_INDEX_MEMBER(Arrow_,source_type,source)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<targetTag>, BOOST_MULTI_INDEX_MEMBER(Arrow_,target_type,target)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<colorTag>,  BOOST_MULTI_INDEX_MEMBER(Arrow_,color_type,color)>,
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<targetSourceTag>,
            ::boost::multi_index::composite_key<
              Arrow_, BOOST_MULTI_INDEX_MEMBER(Arrow_,target_type,target), BOOST_MULTI_INDEX_MEMBER(Arrow_,source_type,source)>
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<sourceColorTag>,
            ::boost::multi_index::composite_key<
              Arrow_, BOOST_MULTI_INDEX_MEMBER(Arrow_,source_type,source), BOOST_MULTI_INDEX_MEMBER(Arrow_,color_type,color)>
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<targetColorTag>,
            ::boost::multi_index::composite_key<
              Arrow_, BOOST_MULTI_INDEX_MEMBER(Arrow_,target_type,target), BOOST_MULTI_INDEX_MEMBER(Arrow_,color_type,color)>
          >
        >,
        ALE_ALLOCATOR<Arrow_>
      > ArrowSet;      
      ArrowSet _arrows;
      
      //
      // Point storage
      //
      struct pointTag{};
      struct degreeTag{};
      // Base/Cap point record
      template <typename Point_>
      struct PointRecord {
        typedef Point_ point_type;
        point_type point;
        int        degree;
        // Basic interface
        PointRecord() : degree(0){};
        PointRecord(const point_type& p) : point(p), degree(0) {};
        PointRecord(const point_type& p, const int d) : point(p), degree(d) {};
        // Printing
        friend std::ostream& operator<<(std::ostream& os, const PointRecord& p) {
          os << "[" << p.point << ", "<< p.degree << "]";
          return os;
        };
      };
      // Base point records are of BasePoint type stored in a BasePointSet
      typedef PointRecord<target_type> BasePoint;
      typedef ::boost::multi_index::multi_index_container<
        BasePoint,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<pointTag>, BOOST_MULTI_INDEX_MEMBER(BasePoint, target_type, point)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<degreeTag>, BOOST_MULTI_INDEX_MEMBER(BasePoint,int,degree)>
        >,
        ALE_ALLOCATOR<BasePoint>
      > BasePointSet;
      // Cap point records are of CapPoint type stored in a CapPointSet
      typedef PointRecord<source_type> CapPoint;    
      typedef ::boost::multi_index::multi_index_container<
        CapPoint,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<pointTag>, BOOST_MULTI_INDEX_MEMBER(CapPoint, source_type, point)>,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<degreeTag>, BOOST_MULTI_INDEX_MEMBER(CapPoint,int,degree)>
        >,
        ALE_ALLOCATOR<CapPoint>
      > CapPointSet;
      BasePointSet _base;
      CapPointSet  _cap;      

    public:
      //
      // Return types
      //

      // base specialization of OutputSequence and related iterators and methods
      class baseSequence : public OutputSequence<BasePointSet, degreeTag, BOOST_MULTI_INDEX_MEMBER(BasePoint,const typename BasePoint::point_type,point)> {
      public:
          typedef typename BOOST_MULTI_INDEX_MEMBER(BasePoint,typename BasePoint::point_type,point) extractor_type;
          typedef typename OutputSequence<BasePointSet,degreeTag,extractor_type>::iterator iterator;

        baseSequence(const baseSequence& seq) : OutputSequence<CapPointSet, degreeTag, extractor_type>(seq){};
        baseSequence(const basePointSet& set) : OutputSequence<BasePointSet, degreeTag, extractor_type>(set){};
        virtual void ~baseSequence();

        virtual iterator begin() {
          // Retrieve the beginning iterator to the sequence of points with indegree >= 1
          return iterator(this->_index.lower_bound(1));
        };
        virtual iterator end() {
          // Retrieve the ending iterator to the sequence of points with indegree >= 1
          // Since the elements in this index are ordered by degree, this amounts to the end() of the index.
          return iterator(this->_index.end());
        };
      };


      // cap specialization of OutputSequence and related iterators and  methods
      class capSequence  : public OutputSequence<CapPointSet, degreeTag, BOOST_MULTI_INDEX_MEMBER(CapPoint,typename CapPoint::point_type,point)> {
      public:
        typedef typename BOOST_MULTI_INDEX_MEMBER(CapPoint,typename CapPoint::point_type,point) extractor_type;
        typedef typename OutputSequence<CapPointSet, degreeTag, extractor_type>::iterator iterator;

        capSequence(const capSequence& seq) : OutputSequence<CapPointSet, degreeTag, extractor_type>(seq){};
        capSequence(const capPointSet& set) : OutputSequence<CapPointSet, degreeTag, extractor_type>(set){};
        virtual void ~capSequence();

        virtual iterator begin() {
          // Retrieve the beginning iterator to the sequence of points with outdegree >= 1
          return iterator(this->_index.lower_bound(1));
        };
        virtual iterator end() {
          // Retrieve the ending iterator to the sequence of points with outdegree >= 1
          // Since the elements in this index are ordered by degree, this amounts to the end() of the index.
          return iterator(this->_index.end());
        };
      };

      // (Partial) Arrow specialization of ReversibleOutputSequence and related iterators and methods.
      // Provides basis for cone and support specializations
      // ArrowSequence iterates over an index of ArrowSet defined by Tag_, Key_ and an optional color_type color argument,
      // returning Value_ objects upon dereferencing.
      template <typename Tag_, typename Key_, typename MemberExtractor_>
      class ArrowSequence : public ReversibleOutputSequence<ArrowSet, Tag_, MemberExtractor_> {
      public:
        typedef typename ReversibleOutputSequence<ArrowSet, Tag_, MemberExtractor_>::tag              tag;
        typedef typename ReversibleOutputSequence<ArrowSet, Tag_, MemberExtractor_>::set_type         set_type;
        typedef typename ReversibleOutputSequence<ArrowSet, Tag_, MemberExtractor_>::index_type       index_type;
        typedef Key_                                                                                  key_type;
      protected:
        const key_type      key;
        const color_type    color;  // color_type is defined by BiGraph
        const bool          useColor;
      public:
        // Need to extend the inherited iterators to be able to extract arrow color
        class iterator : public ReversibleOutputSequence<ArrowSet, Tag_, MemberExtractor_>::iterator {
        public:
          virtual const color_type& color() const {return this->_itor->color;};
        };
        class reverse_iterator : public ReversibleOutputSequence<ArrowSet, Tag_, MemberExtractor_>::reverse_iterator {
        public:
          virtual const color_type& color() const {return this->_itor->color;};
        };

      public:
        //
        // Basic interface
        //
        ArrowSequence(const ArrowSequence& seq)
          : ReversibleOutputSequence<ArrowSet,Tag_,MemberExtractor_>(seq), key(seq.p), color(seq.color), useColor(seq.useColor) {};
        ArrowSequence(const set_type& set, const key_type& p)
          : ReversibleOutputSequence<ArrowSet,Tag_,MemberExtractor_>(set), key(p), color(color_type()), useColor(0) {};
        ArrowSequence(const set_type& set, const key_type& p, const color_type& c)
          : ReversibleOutputSequence<ArrowSet,Tag_,MemberExtractor_>(set), key(p), color(c), useColor(1) {};
        virtual ~ArrowSequence() {};

        virtual iterator begin() {
          if (this->useColor) {
            return iterator(this->_index.lower_bound(::boost::make_tuple(this->key,this->color)));
          } else {
            return iterator(this->_index.lower_bound(::boost::make_tuple(this->key)));
          }
        };

        virtual iterator end() {
          if (this->useColor) {
            return iterator(this->_index.upper_bound(::boost::make_tuple(this->key,this->color)));
          } else {
            return iterator(this->_index.upper_bound(::boost::make_tuple(this->key)));
          }
        };

        virtual reverse_iterator rbegin() {
          if (this->useColor) {
            return reverse_iterator(--this->_index.upper_bound(::boost::make_tuple(this->key,this->color)));
          } else {
            return reverse_iterator(--this->_index.upper_bound(::boost::make_tuple(this->key)));
          }
        };

        virtual reverse_iterator rend() {
          //typename boost::multi_index::index<ArrowSet,targetColor>::type::iterator i;
          if (this->useColor) {
            return reverse_iterator(--this->_index.lower_bound(::boost::make_tuple(this->key,this->color)));
          } else {
            return reverse_iterator(--this->_index.lower_bound(::boost::make_tuple(this->key)));
          }
        };

        virtual std::size_t size() {
          if (this->useColor) {
            return this->_index.count(::boost::make_tuple(this->key,color));
          } else {
            return this->_index.count(::boost::make_tuple(this->key));
          }
        };
      };

      // coneSequence specializes ArrowSequence with Tag_ == targetColorTag, Key_ == target_type, MemberExtractor_ == XXX_MEMBER(source)
      typedef ArrowSequence<targetColorTag, target_type, BOOST_MULTI_INDEX_MEMBER(Arrow_,source_type,source)> coneSequence;
  
      // supportSequence specializes ArrowSequence with Tag_==sourceColorTag,Key_==source_type,MemberExtractor_==XXX_MEMBER(target_type)
      typedef ArrowSequence<sourceColorTag, source_type, BOOST_MULTI_INDEX_MEMBER(Arrow_,target_type,target)> supportSequence;

      typedef std::set<source_type, std::less<source_type>, ALE_ALLOCATOR<source_type> > coneSet;
      typedef std::set<target_type, std::less<target_type>, ALE_ALLOCATOR<target_type> > supportSet;
      
    public:
      // 
      // Basic interface
      //
      BiGraph(int debug = 0) : debug(debug) {};

      //
      // Query methods
      //
      Obj<capSequence>     cap() {
        return capSequence(this->_cap);
      };
      Obj<baseSequence>    base() {
        return baseSequence(this->_base);
      };
      Obj<coneSequence> cone(const target_type& p) {
        return coneSequence(this->_arrows, p);
      }
      template<class InputSequence> Obj<coneSet> cone(const Obj<InputSequence>& points) {
        return this->cone(points, color_type(), false);
      };
      Obj<coneSequence>    cone(const target_type& p, const color_type& color) {
        return coneSequence(this->_arrows, p, color);
      };
      template<class InputSequence>
      Obj<coneSet>         cone(const Obj<InputSequence>& points, const color_type& color, bool useColor = true);
      // implementation follows BiGraph declaration

      Obj<supportSequence> support(const source_type& p) {
        return supportSequence(this->_arrows, p);
      };
      Obj<supportSequence> support(const source_type& p, const color_type& color) {
        return supportSequence(this->_arrows, p, color);
      };
      template<class sourceInputSequence>
      Obj<supportSet>      support(const Obj<sourceInputSequence>& sources);
      // unimplemented
      template<class sourceInputSequence>
      Obj<supportSet>      support(const Obj<sourceInputSequence>& sources, const color_type& color);
      // unimplemented
      const color_type&    getColor(const source_type& s, const target_type& t) {
        typename ::boost::multi_index::index<ArrowSet,targetSourceTag>::type& index = ::boost::multi_index::get<targetSourceTag>(this->_arrows);
        typename ::boost::multi_index::index<ArrowSet,targetSourceTag>::type::iterator i = index.find(::boost::make_tuple(t,s));
        return (*i).color;
      };

      //
      // Lattice queries
      //
      template<class targetInputSequence> 
      Obj<coneSequence> meet(const Obj<targetInputSequence>& targets);
      // unimplemented

      template<class targetInputSequence> 
      Obj<coneSequence> meet(const Obj<targetInputSequence>& targets, const color_type& color);
      // unimplemented

      template<class sourceInputSequence> 
      Obj<coneSequence> join(const Obj<sourceInputSequence>& sources);
      // unimplemented

      template<class sourceInputSequence> 
      Obj<coneSequence> join(const Obj<sourceInputSequence>& sources, const color_type& color);

      template<typename ostream_type>
      void view(ostream_type& ostream, const char* label = NULL);

      //
      // Structural manipulation
      //
    private:
      // Manipulator objects

      struct changeSource {
        changeSource(source_type newSource) : _newSource(newSource) {};
        void operator()(Arrow_& a) { a.source = this->_newSource;}
      private:
        source_type _newSource;
      };

      struct changeTarget {
        changeTarget(target_type newTarget) : _newTarget(newTarget) {};
        void operator()(Arrow_& a) { a.target = this->_newTarget;}
      private:
        target_type _newTarget;
      };

      template <typename PointRecord_>
      struct adjustDegree {
        adjustDegree(int degreeDelta) : _degreeDelta(degreeDelta) {};
        void operator()(PointRecord_& p) { 
          int newDegree = p.degree + this->_degreeDelta;
          if(newDegree < 0) {
            ostringstream ss;
            ss << "adjustDegree: Adjustment by " << this->_degreeDelta << " would result in negative degree: " << newDegree;
            throw Exception(ss.str().c_str());
          }
          p.degree = newDegree;
        }
      private:
        int _degreeDelta;
      };

    private:
      void __adjustIndegree(const target_type& t, int delta); 
      void __adjustOutdegree(const source_type& s, int delta); 
    public:
      void clear() {
        this->arrows.clear(); this->strata.clear();
      };
      void addArrow(const source_type& p, const target_type& q) {
        this->addArrow(p, q, color_type());
      };
      void addArrow(const source_type& p, const target_type& q, const color_type& color) {
        this->addArrow(Arrow_(p, q, color));
        //std::cout << "Added " << Arrow_(p, q, color);
      };
      void addArrow(const Arrow_& a) {
        this->_arrows.insert(a); this->__adjustIndegree(a.target,1); this->__adjustOutdegree(a.source,1);
        //std::cout << "Added " << Arrow_(p, q, color);
      };
      void addCone(const source_type& source, const target_type& target){
        this->addArrow(source, target);
      };
      template<class sourceInputSequence> 
      void addCone(const Obj<sourceInputSequence>& sources, const target_type& target) {
        this->addCone(sources, target, color_type());
      };
      void addCone(const source_type& source, const target_type& target, const color_type& color) {
        this->addArrow(source, target, color);
      };
      template<class sourceInputSequence> 
      void addCone(const Obj<sourceInputSequence>& sources, const target_type& target, const color_type& color);

    private:
      void __clearCone(const target_type& p, const color_type& color, bool useColor);
    public:
      void setCone(const source_type& source, const target_type& target){
        this->__clearCone(target, color_type(), false); this->addCone(source, target);
      };

      template<class sourceInputSequence> 
      void setCone(const Obj<sourceInputSequence>& sources, const target_type& target) {
        this->__clearCone(target, color_type(), false); this->addCone(sources, target, color_type());
      };

      void setCone(const source_type& source, const target_type& target, const color_type& color) {
        this->__clearCone(target, color, true); this->addCone(source, target, color);
      };

      template<class sourceInputSequence> 
      void setCone(const Obj<sourceInputSequence>& sources, const target_type& target, const color_type& color) {
        this->__clearCone(target, color, true); this->addCone(sources, target, color);
      };

      template<class targetInputSequence> 
      void addSupport(const source_type& source, const Obj<targetInputSequence >& targets);
      // Unimplemented

      template<class targetInputSequence> 
      void addSupport(const source_type& source, const Obj<targetInputSequence >& targets, const color_type& color);
      // Unimplemented

      void add(const Obj<BiGraph<source_type, target_type, color_type> >& bigraph);
      // Unimplemented

      void replaceSource(const source_type& s, const source_type& new_s);

      void replaceTarget(const target_type& t, const target_type& new_t);

      template<typename Changer>
      void modifyColor(const source_type& s, const target_type& t, Changer changeColor) {
        typename ::boost::multi_index::index<ArrowSet,targetSourceTag>::type& index = ::boost::multi_index::get<targetSourceTag>(this->_arrows);
        typename ::boost::multi_index::index<ArrowSet,targetSourceTag>::type::iterator i = index.find(::boost::make_tuple(t,s));
        if (i != index.end()) {
          index.modify(i, changeColor);
        } else {
//           std::cout << "ERROR: Could not change color for " << s << " to " << t << std::endl;
//           for(typename ::boost::multi_index::index<ArrowSet,targetSourceTag>::type::iterator v = index.begin(); v != index.end(); ++v) {
//             std::cout << "  Arrow " << *v << std::endl;
//           }
          Arrow_ a(s, t, color_type());
          changeColor(a);
          this->addArrow(a);
        }
      };

      void replaceSourceOfTarget(const target_type& t, const source_type& new_s);

    }; // BiGraph

    //
    // BiGraph methods
    //

    template <typename Source_, typename Target_, typename Color_>
    template<class InputSequence>
    Obj<typename BiGraph<Source_,Target_,Color_>::coneSet>  
    BiGraph<Source_,Target_,Color_>::cone(const Obj<InputSequence>& points, const color_type& color, bool useColor) {
      Obj<coneSet> cone = coneSet();
      
      for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
        Obj<coneSequence> pCone;
        
        if (useColor) {
          pCone = this->cone(*p_itor, color);
        } else {
          pCone = this->cone(*p_itor);
        }
        cone->insert(pCone->begin(), pCone->end());
      }
      return cone;
    };

    template <typename Source_, typename Target_, typename Color_>
    template<typename ostream_type>
    void BiGraph<Source_,Target_,Color_>::view(ostream_type& os, const char* label) {
      if(label != NULL) {
        os << "Viewing BiGraph '" << label << "':" << std::endl;
      } 
      else {
        os << "Viewing a BiGraph:" << std::endl;
      }
      capSequence cap = this->cap();
      for(typename capSequence::iterator capi = cap.begin(); capi != cap.end(); capi++) {
        supportSequence supp = this->support(*capi);
        for(typename supportSequence::iterator suppi = supp.begin(); suppi != supp.end(); suppi++) {
          os << *capi << "--" << capi.color() << "-->" << *suppi << std::endl;
        }
      }
      baseSequence base = this->base();
      for(typename baseSequence::iterator basei = base.begin(); basei != base.end(); basei++) {
        coneSequence cone = this->cone(*basei);
        for(typename coneSequence::iterator conei = cone.begin(); conei != cone.end(); conei++) {
          os << *basei <<  "<--" << conei.color() << "--" << *conei << std::endl;
        }
      }
    };


    template <typename Source_, typename Target_, typename Color_>
    template<class sourceInputSequence> 
    void BiGraph<Source_,Target_,Color_>::addCone(const Obj<sourceInputSequence>& sources, 
                                                  const BiGraph<Source_,Target_,Color_>::target_type& target, 
                                                  const BiGraph<Source_,Target_,Color_>::color_type& color) {
      if (debug) {std::cout << "Adding a cone " << std::endl;}
      for(typename sourceInputSequence::iterator iter = sources->begin(); iter != sources->end(); ++iter) {
        if (debug) {std::cout << "Adding arrow from " << *iter << " to " << target << "(" << color << ")" << std::endl;}
        this->addArrow(*iter, target, color);
      }
    };

    template <typename Source_, typename Target_, typename Color_>
    void BiGraph<Source_,Target_,Color_>::__adjustIndegree(const BiGraph<Source_,Target_,Color_>::target_type& p, int delta) {
      typename ::boost::multi_index::index<BasePointSet,pointTag>::type& index = ::boost::multi_index::get<pointTag>(this->_base);
      typename ::boost::multi_index::index<BasePointSet,pointTag>::type::iterator i = index.find(p);
      if (i == index.end()) { // No such point exists
        if(delta < 0) { // Cannot decrease degree of a non-existent point
          ostringstream err;
          err << "ERROR: BiGraph::__adjustIndegree: Non-existent point " << p;
          std::cout << err << std::endl;
          throw(Exception(err.str().c_str()));
        }
        else { // We CAN INCREASE the degree of a non-existent point: simply insert a new element with degree == delta
          std::pair<typename ::boost::multi_index::index<BasePointSet,pointTag>::type::iterator, bool> ii;
          BasePoint pp(p,delta);
          ii = index.insert(pp);
          if(ii.second == false) {
            ostringstream err;
            err << "ERROR: BiGraph::__adjustIndegree: Failed to insert a CapPoint " << pp;
            std::cout << err << std::endl;
            throw(Exception(err.str().c_str()));
          }
        }
      }
      else { // Point exists, so we modify its degree
        index.modify(i, adjustDegree<BasePoint>(delta));
      }
    }// BiGraph::__adjustIndegree

    template <typename Source_, typename Target_, typename Color_>
    void BiGraph<Source_,Target_,Color_>::__adjustOutdegree(const typename BiGraph<Source_,Target_,Color_>::source_type& p, int delta) {
      typename ::boost::multi_index::index<CapPointSet,pointTag>::type& index = ::boost::multi_index::get<pointTag>(this->_cap);
      typename ::boost::multi_index::index<CapPointSet,pointTag>::type::iterator i = index.find(p);
      if (i == index.end()) { // No such point exists
        if(delta < 0) { // Cannot decrease degree of a non-existent point
          ostringstream err;
          err << "ERROR: BiGraph::__adjustOutdegree: Non-existent point " << p;
          std::cout << err << std::endl;
          throw(Exception(err.str().c_str()));
        }
        else { // We CAN INCREASE the degree of a non-existent point: simply insert a new element with degree == delta
          std::pair<typename ::boost::multi_index::index<CapPointSet,pointTag>::type::iterator, bool> ii;
          CapPoint pp(p,delta);
          ii = index.insert(pp);
          if(ii.second == false) {
            ostringstream err;
            err << "ERROR: BiGraph::__adjustOutdegree: Failed to insert a CapPoint " << pp;
            std::cout << err << std::endl;
            throw(Exception(err.str().c_str()));
          }
        }
      }
      else { // Point exists, so we modify its degree
        index.modify(i, adjustDegree<CapPoint>(delta));
      }
    }// BiGraph::__adjustOutdegree

    template <typename Source_, typename Target_, typename Color_>
    void BiGraph<Source_,Target_,Color_>::__clearCone(const BiGraph<Source_,Target_,Color_>::target_type& p, 
                                                      const BiGraph<Source_,Target_,Color_>::color_type& color, bool useColor) {
      typename ::boost::multi_index::index<ArrowSet,targetColorTag>::type& coneIndex = 
        ::boost::multi_index::get<targetColorTag>(this->_arrows);
      typename ::boost::multi_index::index<ArrowSet,targetColorTag>::type::iterator i, ii;
      if (this->useColor) {
        i = coneIndex.lower_bound(::boost::make_tuple(p,color));
        ii = coneIndex.upper_bound(::boost::make_tuple(p,color));
      } else {
        i = coneIndex.lower_bound(::boost::make_tuple(p));
        ii = coneIndex.upper_bound(::boost::make_tuple(p));
      }
      for(; i != ii; i++){
        // Adjust the degrees before removing the arrow
        this->__adjustOutdegree(i->source, -1);
        this->__adjustIndegree(i->target, -1);
        coneIndex.erase(i);
      }
    }

    template <typename Source_, typename Target_, typename Color_>
    void BiGraph<Source_,Target_,Color_>::replaceSource(const BiGraph<Source_,Target_,Color_>::source_type& s, 
                                                        const BiGraph<Source_,Target_,Color_>::source_type& new_s) {
      typename ::boost::multi_index::index<ArrowSet,sourceTag>::type& index = ::boost::multi_index::get<sourceTag>(this->_arrows);
      typename ::boost::multi_index::index<ArrowSet,sourceTag>::type::iterator i = index.find(s);
      if (i != index.end()) {
        // The replacement of source amounts to removing an arrow and adding another arrow, hence, two sets of 
        // degree adjustments are necessary.  However, since the target of the two arrows is the same, its degree is unchanged.
        // Caution: degrees may become inconsistent if an exception is thrown (e.g., by index.modify(...)).
        this->__adjustOutdegree(s,-1);
        index.modify(i, changeSource(new_s));
        this->__adjustOutdegree(new_s,1);
      } else {
        std::cout << "ERROR: Could not replace source " << s << " with " << new_s << std::endl;
      }
    }

    template <typename Source_, typename Target_, typename Color_>
    void BiGraph<Source_,Target_,Color_>::replaceTarget(const BiGraph<Source_,Target_,Color_>::target_type& t, 
                                                        const BiGraph<Source_,Target_,Color_>::target_type& new_t) {
      typename ::boost::multi_index::index<ArrowSet,targetTag>::type& index = ::boost::multi_index::get<targetTag>(this->_arrows);
      typename ::boost::multi_index::index<ArrowSet,targetTag>::type::iterator i = index.find(t);
      if (i != index.end()) {
        // The replacement of target amounts to removing an arrow and adding another arrow, hence, two sets of 
        // degree adjustments are necessary.  However, since the source of the two arrows is the same, its degree is unchanged.
        // Caution: degrees may become inconsistent if an exception is thrown (e.g., by index.modify(...)).
        this->__adjustIndegree(t,-1);
        index.modify(i, changeTarget(new_t));
        this->__adjustIndegree(new_t,1);
      } else {
        std::cout << "ERROR: Could not replace target " << t << " with " << new_t << std::endl;
      }
    }

    template <typename Source_, typename Target_, typename Color_>
    void BiGraph<Source_,Target_,Color_>::replaceSourceOfTarget(const BiGraph<Source_,Target_,Color_>::target_type& t, 
                                                                const BiGraph<Source_,Target_,Color_>::source_type& new_s) {
      typename ::boost::multi_index::index<ArrowSet,targetTag>::type& index = ::boost::multi_index::get<targetTag>(this->_arrows);
      typename ::boost::multi_index::index<ArrowSet,targetTag>::type::iterator i = index.find(t);
      if (i != index.end()) {
        // The replacement of source amounts to removing an arrow and adding another arrow, hence, two sets of 
        // degree adjustments are necessary.  However, since the target of the two arrows is the same, its degree is unchanged.
        // Caution: degrees may become inconsistent if an exception is thrown (e.g., by index.modify(...)).
        this->__adjustOutdegree(i->source,-1);
        index.modify(i, changeSource(new_s));
        this->__adjustOutdegree(new_s,1);
      } else {
        std::cout << "ERROR: Could not replace source of target" << t << " with " << new_s << std::endl;
      }
    }


  } // namespace Two
  


} // namespace ALE

#endif
