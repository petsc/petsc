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
    // Upon dereferencing values are extracted from each result record using a ValueExtractor_ object.
    template <typename Iterator_, typename ValueExtractor_>
    class result_iterator_base {
    public:
      // Standard iterator typedefs
      typedef ValueExtractor_                        extractor_type;
      typedef std::input_iterator_tag                iterator_category;
      typedef typename extractor_type::result_type   value_type;
      typedef int                                    difference_type;
      typedef value_type*                            pointer;
      typedef value_type&                            reference;
      
      // Underlying iterator type
      typedef Iterator_ itor_type;
    protected:
      // Underlying iterator 
      itor_type      _itor;
      // Member extractor
      extractor_type _ex;
    public:
      result_iterator_base(itor_type itor) {
        this->_itor = itor_type(itor);
      };
      virtual ~result_iterator_base() {};
      virtual bool              operator==(const result_iterator_base& iter) const {return this->_itor == iter._itor;};
      virtual bool              operator!=(const result_iterator_base& iter) const {return this->_itor != iter._itor;};
      // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
      virtual const value_type  operator*() const {return _ex(*(this->_itor));};
    };

    template <typename Iterator_, typename ValueExtractor_>
    class result_iterator : public result_iterator_base<Iterator_, ValueExtractor_>{
    public:
      // Standard iterator typedefs
      typedef result_iterator_base<Iterator_, ValueExtractor_> base_type;
      typedef typename base_type::iterator_category  iterator_category;
      typedef typename base_type::value_type         value_type;
      typedef typename base_type::extractor_type     extractor_type;
      typedef typename base_type::difference_type    difference_type;
      typedef typename base_type::pointer            pointer;
      typedef typename base_type::reference          reference;
      // Underlying iterator type
      typedef typename base_type::itor_type          itor_type;
    public:
      result_iterator(const itor_type& itor) : base_type(itor) {};
      virtual ~result_iterator() {};
      //
      virtual result_iterator   operator++() {++this->_itor; return *this;};
      virtual result_iterator   operator++(int n) {result_iterator tmp(this->_itor); ++this->_itor; return tmp;};
//       virtual bool              operator==(const result_iterator& iter) const {return this->_itor == iter._itor;};
//       virtual bool              operator!=(const result_iterator& iter) const {return this->_itor != iter._itor;};
//       // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
//       virtual const value_type  operator*() const {return _ex(*(this->_itor));};
    };

    // reverse_result_iterator is the reverse of result_iterator
    template <typename Iterator_, typename ValueExtractor_>
    class reverse_result_iterator : public result_iterator_base<Iterator_, ValueExtractor_> {
    public:
      // Standard iterator typedefs
      typedef result_iterator_base<Iterator_, ValueExtractor_> base_type;
      typedef typename base_type::iterator_category  iterator_category;
      typedef typename base_type::value_type         value_type;
      typedef typename base_type::extractor_type     extractor_type;
      typedef typename base_type::difference_type    difference_type;
      typedef typename base_type::pointer            pointer;
      typedef typename base_type::reference          reference;
      // Underlying iterator type
      typedef typename base_type::itor_type          itor_type;
    public:
      reverse_result_iterator(const itor_type& itor) : base_type(itor) {};
      virtual ~reverse_result_iterator() {};
      //
      virtual reverse_result_iterator     operator++() {--this->_itor; return *this;};
      virtual reverse_result_iterator     operator++(int n) {reverse_result_iterator tmp(this->_itor); --this->_itor; return tmp;};
//       virtual bool                        operator==(const reverse_result_iterator& iter) const {return this->_itor == iter._itor;};
//       virtual bool                        operator!=(const reverse_result_iterator& iter) const {return this->_itor != iter._itor;};
    };

    // OutputSequence defines a generic encapsulation of a result of a query on an index Index_.
    // The result can be traversed yielding a value extracted by the ValueExtractor_ upon dereferencing.  
    // To this end OutputSequence encapsulates the type of the iterator used to traverse the result, 
    // as well as the methods returning the extrema -- the beginning and ending iterators.
    // Specializations of OutputSequence will rely on specializations of iterator.
    template <typename Index_, typename ValueExtractor_>
    class OutputSequence {
    public:
      // Basic encapsulated types
      typedef Index_                                                           index_type;
      typedef ValueExtractor_                                                  extractor_type;
      typedef result_iterator<typename index_type::iterator, ValueExtractor_>  iterator;
    protected:
      const index_type&   _index;
      const index_type&   index()const {return _index;};
    public:
      // Basic interface
      OutputSequence(const OutputSequence& seq) : _index(seq._index) {};
      OutputSequence(const index_type& index)   : _index(index)      {};
      ~OutputSequence(){};
    };

    // ReversibleOutputSequence extends OutputSequence to allow reverse traversals.
    // Specializations of ReversibleOutputSequence will rely on specializations of result_iterator and reverse_result_iterator,
    // which dereference to objects of Value_ type.
    template <typename Index_, typename ValueExtractor_>
    class ReversibleOutputSequence : public OutputSequence<Index_, ValueExtractor_> {
    public:
      // Basic encapsulated types
      typedef typename OutputSequence<Index_, ValueExtractor_>::index_type     index_type;
      typedef typename OutputSequence<Index_, ValueExtractor_>::extractor_type extractor_type;
    public:
      // Encapsulated reverse_result_iterator type
      typedef typename OutputSequence<index_type, extractor_type>::iterator               iterator;
      typedef reverse_result_iterator<typename index_type::iterator, extractor_type>      reverse_iterator;
      // Generic ReversibleOutputSequence interface
      ReversibleOutputSequence(const ReversibleOutputSequence& seq) : OutputSequence<index_type, extractor_type>(seq){};
      ReversibleOutputSequence(const index_type& index)             : OutputSequence<index_type, extractor_type>(index) {};
      ~ReversibleOutputSequence(){};
    };

    // Arrow
    template<typename Source_, typename Target_, typename Color_>
    struct  Arrow { //: public ALE::def::Arrow<Source_, Target_, Color_> {
      typedef Source_ source_type;
      typedef Target_ target_type;
      typedef Color_  color_type;
      source_type source;
      target_type target;
      color_type  color;
      Arrow(const source_type& s, const target_type& t, const color_type& c) : source(s), target(t), color(c) {};
    };
    
    //
    // ArrowContainer definitions and some related specializations of (Reversible)OutputSequence.
    //
    template<typename Source_, typename Target_, typename Color_>
    struct ArrowContainer {
      // Encapsulated types
      typedef Arrow<Source_,Target_,Color_> arrow_type;
      // Index tags
      struct sourceColorTag{};
      struct targetColorTag{};
      struct sourceTargetTag{};

      // multi-index set type -- arrow set
      typedef ::boost::multi_index::multi_index_container<
        arrow_type,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<sourceTargetTag>,
            ::boost::multi_index::composite_key<
              arrow_type, 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename arrow_type::source_type, source), 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename arrow_type::target_type, target)
            >
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<sourceColorTag>,
            ::boost::multi_index::composite_key<
              arrow_type, 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename arrow_type::source_type, source), 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename arrow_type::color_type,  color)
            >
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<targetColorTag>,
            ::boost::multi_index::composite_key<
              arrow_type, 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename arrow_type::target_type, target), 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename arrow_type::color_type,  color)
            >
          >
        >,
        ALE_ALLOCATOR<arrow_type>
      > arrow_set_type;      

      // multi-index arrow set 
      arrow_set_type arrows;

      //
      // ArrowContainer return types
      //  

      // (Partial) Arrow specialization of ReversibleOutputSequence and related iterators and methods.
      // ArrowSequence iterates over the subset of an ArrowSet Index_ with a ValueExtractor_ object 
      // used to extract the default value on dereferencing.  A Key_ object, other than color, is used
      // to extract the subset.
      template <typename Index_, typename Key_, typename ValueExtractor_>
      class ArrowSequence : public ReversibleOutputSequence<Index_, ValueExtractor_> {
      public:
        typedef ReversibleOutputSequence<Index_, ValueExtractor_>   rout_seq_type;      
        typedef Index_                                              index;
        typedef Key_                                                key_type;
      protected:
        const key_type                                              key;
        const typename arrow_type::color_type                       color;  // color_type is defined by BiGraph
        const bool                                                  useColor;
      public:
        // Need to extend the inherited iterators to be able to extract arrow color
        class iterator : public rout_seq_type::iterator {
        public:
          iterator(const typename rout_seq_type::iterator::itor_type& itor) : rout_seq_type::iterator(itor) {};
          virtual const typename arrow_type::color_type& color() const {return this->_itor->color;};
        };
        class reverse_iterator : public rout_seq_type::reverse_iterator {
        public:
          reverse_iterator(const typename rout_seq_type::reverse_iterator::itor_type& itor) : rout_seq_type::reverse_iterator(itor) {};
          virtual const typename arrow_type::color_type& color() const {return this->_itor->color;};
        };
      public:
        //
        // Basic ArrowSequence interface
        //
        ArrowSequence(const ArrowSequence& seq) : rout_seq_type(seq), key(seq.key), color(seq.color), useColor(seq.useColor) {};
        ArrowSequence(const typename rout_seq_type::index_type& index, const key_type& k) : 
          rout_seq_type(index), key(k), color(typename arrow_type::color_type()), useColor(0) {};
        ArrowSequence(const typename rout_seq_type::index_type& index, const key_type& k, const typename arrow_type::color_type& c) : 
          rout_seq_type(index), key(k), color(c), useColor(1){};
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
      };// class ArrowSequence
    
    protected:
      //
      // Container modifiers
      //
      struct sourceChanger {
        sourceChanger(const typename arrow_type::source_type& newSource) : _newSource(newSource) {};
        void operator()(arrow_type& a) {a.source = this->_newSource;}
      private:
        typename arrow_type::source_type _newSource;
      };

      struct targetChanger {
        targetChanger(const typename arrow_type::target_type& newTarget) : _newTarget(newTarget) {};
        void operator()(arrow_type& a) { a.target = this->_newTarget;}
      private:
        typename arrow_type::target_type _newTarget;
      };
    };// class ArrowContainer
  

    //
    // RecContainer definitions & related OutputSequence specializations.
    // Rec denotes a point record.
    // 
    template <typename Point_>
    struct Rec {
      typedef Point_ point_type;
      point_type     point;
      int            degree;
      // Basic interface
      Rec() : degree(0){};
      Rec(const Rec& r) : point(r.point), degree(r.degree) {}
      Rec(const point_type& p) : point(p), degree(0) {};
      Rec(const point_type& p, const int d) : point(p), degree(d) {};
      // Printing
      friend std::ostream& operator<<(std::ostream& os, const Rec& p) {
        os << "[" << p.point << ", "<< p.degree << "]";
        return os;
      };
      
      struct degreeAdjuster {
        degreeAdjuster(int degreeDelta) : _degreeDelta(degreeDelta) {};
        void operator()(Rec& r) { 
          int newDegree = r.degree + this->_degreeDelta;
          if(newDegree < 0) {
            ostringstream ss;
            ss << "degreeAdjuster: Adjustment by " << this->_degreeDelta << " would result in negative degree: " << newDegree;
            throw Exception(ss.str().c_str());
          }
          r.degree = newDegree;
        }
      private:
        int _degreeDelta;
      };

    };// class Rec

    template <typename Point_>
    struct RecContainer {
      typedef Rec<Point_> rec_type;
      // Index tags
      struct pointTag{};
      struct degreeTag{};
      // Rec set definition
      typedef ::boost::multi_index::multi_index_container<
        rec_type,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<pointTag>, BOOST_MULTI_INDEX_MEMBER(rec_type, typename rec_type::point_type, point)
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<degreeTag>, BOOST_MULTI_INDEX_MEMBER(rec_type, typename rec_type::point_type, point)
          >
        >,
        ALE_ALLOCATOR<rec_type>
      > rec_set_type; 
      rec_set_type recs;

      //
      // Return types
      //
      // Rec specialization of OutputSequence and related iterators and methods
      class DegreeSequence : public OutputSequence<typename ::boost::multi_index::index<rec_set_type, degreeTag>::type,
                                                   BOOST_MULTI_INDEX_MEMBER(rec_type, typename rec_type::point_type,point) > {
      public:
        typedef OutputSequence<typename ::boost::multi_index::index<rec_set_type, degreeTag>::type,
                                 BOOST_MULTI_INDEX_MEMBER(rec_type, typename rec_type::point_type,point) > out_seq_type;
        typedef typename out_seq_type::index_type                                                          index_type;
        typedef typename out_seq_type::iterator                                                            iterator;
        //
        DegreeSequence(const DegreeSequence& seq) : out_seq_type(seq){};
        DegreeSequence(index_type& index)         : out_seq_type(index){};
        virtual ~DegreeSequence(){};

        virtual iterator begin() {
          // Retrieve the beginning iterator to the sequence of points with indegree >= 1
          return iterator(this->_index.lower_bound(1));
        };
        virtual iterator end() {
          // Retrieve the ending iterator to the sequence of points with indegree >= 1
          // Since the elements in this index are ordered by degree, this amounts to the end() of the index.
          return iterator(this->_index.end());
        };
      }; // class DegreeSequence

      void adjustDegree(const typename rec_type::point_type& p, int delta) {
        typename ::boost::multi_index::index<rec_set_type, pointTag>::type& index = ::boost::multi_index::get<pointTag>(this->recs);
        typename ::boost::multi_index::index<rec_set_type, pointTag>::type::iterator i = index.find(p);
        if (i == index.end()) { // No such point exists
          if(delta < 0) { // Cannot decrease degree of a non-existent point
            ostringstream err;
            err << "ERROR: BiGraph::adjustDegree: Non-existent point " << p;
            std::cout << err << std::endl;
            throw(Exception(err.str().c_str()));
          }
          else { // We CAN INCREASE the degree of a non-existent point: simply insert a new element with degree == delta
            std::pair<typename ::boost::multi_index::index<rec_set_type, pointTag>::type::iterator, bool> ii;
            rec_type r(p,delta);
            ii = index.insert(r);
            if(ii.second == false) {
              ostringstream err;
              err << "ERROR: BiGraph::adjustDegree: Failed to insert a rec " << r;
              std::cout << err << std::endl;
              throw(Exception(err.str().c_str()));
            }
          }
        }
        else { // Point exists, so we modify its degree
          index.modify(i, typename rec_type::degreeAdjuster(delta));
        }
      }; // adjustDegree()
    }; // class RecContainer

    //
    // BiGraph (short for BipartiteGraph) implements a sequential interface similar to that of Sieve (below),
    // except the source and target points may have different types and iterated operations (e.g., nCone, closure)
    // are not available.
    // 
    template<typename Source_, typename Target_, typename Color_>
    class BiGraph {
    public:
      // Encapsulated types
      typedef ArrowContainer<Source_,Target_,Color_>      arrow_container_type;
      typedef RecContainer<Source_>                       cap_container_type;
      typedef RecContainer<Target_>                       base_container_type;
      typedef typename arrow_container_type::arrow_type   arrow_type;
      typedef typename arrow_type::source_type            source_type;
      typedef typename arrow_type::target_type            target_type;
      typedef typename arrow_type::color_type             color_type;
      // Convenient tag names
      typedef typename arrow_container_type::sourceColorTag  supportTag;
      typedef typename arrow_container_type::targetColorTag  coneTag;
      typedef typename arrow_container_type::sourceTargetTag arrowTag;
      // Debug level
      int debug;
    protected:
      arrow_container_type _body;
      base_container_type  _base;
      cap_container_type   _cap;
    public:
      //
      // Return types
      //
      typedef typename
      arrow_container_type::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::arrow_set_type, 
                                                                                        arrowTag>::type,
                                                   source_type,
                                                   BOOST_MULTI_INDEX_MEMBER(arrow_type,color_type,color)> 
      arrowSequence;

      typedef typename 
      arrow_container_type::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::arrow_set_type, 
                                                                                        coneTag>::type,
                                                   target_type,
                                                   BOOST_MULTI_INDEX_MEMBER(arrow_type, source_type, source)> 
      coneSequence;

      typedef typename 
      arrow_container_type::template ArrowSequence<typename
                                                   ::boost::multi_index::index<typename arrow_container_type::arrow_set_type, 
                                                                               supportTag>::type,
                                                   source_type,
                                                   BOOST_MULTI_INDEX_MEMBER(arrow_type, target_type, target)> 
      supportSequence;

      typedef typename cap_container_type::DegreeSequence  capSequence;
      typedef typename base_container_type::DegreeSequence baseSequence;
      typedef std::set<source_type>                        coneSet;
      typedef std::set<target_type>                        supportSet;
      typedef std::set<arrow_type>                         arrowSet;
    public:
      // 
      // Basic interface
      //
      BiGraph(int debug = 0) : debug(debug) {};

      //
      // Query methods
      //
      Obj<capSequence>   
      cap() {
        return capSequence(::boost::multi_index::get<typename cap_container_type::degreeTag>(_cap.recs));
      };

      Obj<baseSequence>    
      base() {
        return baseSequence(::boost::multi_index::get<typename base_container_type::degreeTag>(_base.recs));
      };

      Obj<arrowSequence> 
      arrows(const source_type& s, const target_type& t) {
        return arrowSequence(::boost::multi_index::get<arrowTag>(this->_body.arrows), 
                             ::boost::make_tuple(s,t));
      };

      Obj<coneSequence> 
      cone(const target_type& p) {
        return coneSequence(::boost::multi_index::get<coneTag>(this->_body.arrows), p);
      };

      template<class InputSequence> 
      Obj<coneSet> 
      cone(const Obj<InputSequence>& points) {
        return this->cone(points, color_type(), false);
      };

      Obj<coneSequence> 
      cone(const target_type& p, const color_type& color) {
        return coneSequence(::boost::multi_index::get<coneTag>(this->_body.arrows), p, color);
      };

      template<class InputSequence>
      Obj<coneSet> 
      cone(const Obj<InputSequence>& points, const color_type& color, bool useColor = true) {
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

      Obj<supportSequence> 
      support(const source_type& p) {
        return supportSequence(::boost::multi_index::get<supportTag>(this->_body.arrows), p);
      };

      Obj<supportSequence> 
      support(const source_type& p, const color_type& color) {
        return supportSequence(::boost::multi_index::get<supportTag>(this->_body.arrows), p, color);
      };

      template<class sourceInputSequence>
      Obj<supportSet>      
      support(const Obj<sourceInputSequence>& sources);
      // unimplemented

      template<class sourceInputSequence>
      Obj<supportSet>      
      support(const Obj<sourceInputSequence>& sources, const color_type& color);
      // unimplemented

      const color_type&    
      getColor(const source_type& s, const target_type& t) {
        return *(arrows(s,t));
      };

      template<typename ostream_type>
      void view(ostream_type& os, const char* label = NULL){
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
            os << *capi << "--" << suppi.color() << "-->" << *suppi << std::endl;
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
    public:
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

    public:
      //
      // Structural manipulation
      //
      void clear() {
        this->arrows.clear(); this->strata.clear();
      };
      void addArrow(const source_type& p, const target_type& q) {
        this->addArrow(p, q, color_type());
      };
      void addArrow(const source_type& p, const target_type& q, const color_type& color) {
        this->addArrow(arrow_type(p, q, color));
        //std::cout << "Added " << arrow_type(p, q, color);
      };
      void addArrow(const arrow_type& a) {
        this->_body.arrows.insert(a); _base.adjustDegree(a.target,1); _cap.adjustDegree(a.source,1);
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
      void 
      addCone(const Obj<sourceInputSequence>& sources, const target_type& target, const color_type& color) {
        if (debug) {std::cout << "Adding a cone " << std::endl;}
        for(typename sourceInputSequence::iterator iter = sources->begin(); iter != sources->end(); ++iter) {
          if (debug) {std::cout << "Adding arrow from " << *iter << " to " << target << "(" << color << ")" << std::endl;}
          this->addArrow(*iter, target, color);
        }
      };

    private:
      void 
      __clearCone(const target_type& t, const color_type&  color, bool useColor = false) {
        // Use the cone sequence types to clear the cone
        coneSequence cone;
        typename coneSequence::index& coneIndex = ::boost::multi_index::get<coneTag>(this->_body.arrows);
        typename coneSequence::index::iterator i, ii;
        if (this->useColor) {
          i = coneIndex.lower_bound(::boost::make_tuple(t,color));
          ii = coneIndex.upper_bound(::boost::make_tuple(t,color));
        } else {
          i = coneIndex.lower_bound(::boost::make_tuple(t));
          ii = coneIndex.upper_bound(::boost::make_tuple(t));
        }
        for(; i != ii; i++){
          // Adjust the degrees before removing the arrow
          cap.adjustDegree(i->source, -1);
          base.adjustDegree(i->target, -1);
          coneIndex.erase(i);
        }
      };
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
      void setCone(const Obj<sourceInputSequence>& sources, const target_type& target, const color_type& color){
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

    }; // class BiGraph

  } // namespace Two

} // namespace ALE

#endif
