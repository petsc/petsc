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

    // Defines the traits of a sequence representing a subset of a multi_index container Index_.
    // A sequence defines output (input in std terminology) iterators for traversing an Index_ object.
    // Upon dereferencing values are extracted from each result record using a ValueExtractor_ object.
    template <typename Index_, typename ValueExtractor_>
    struct IndexSequenceTraits {
      typedef Index_ index_type;
      class iterator_base {
      public:
        // Standard iterator typedefs
        typedef ValueExtractor_                        extractor_type;
        typedef std::input_iterator_tag                iterator_category;
        typedef typename extractor_type::result_type   value_type;
        typedef int                                    difference_type;
        typedef value_type*                            pointer;
        typedef value_type&                            reference;
        
        // Underlying iterator type
        typedef typename index_type::iterator          itor_type;
      protected:
        // Underlying iterator 
        itor_type      _itor;
        // Member extractor
        extractor_type _ex;
      public:
        iterator_base(itor_type itor) {
          this->_itor = itor_type(itor);
        };
        virtual ~iterator_base() {};
        virtual bool              operator==(const iterator_base& iter) const {return this->_itor == iter._itor;};
        virtual bool              operator!=(const iterator_base& iter) const {return this->_itor != iter._itor;};
        // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
        virtual const value_type  operator*() const {return _ex(*(this->_itor));};
      };// class iterator_base
      class iterator : public iterator_base {
      public:
        // Standard iterator typedefs
        typedef iterator_base                   base_type;
        typedef typename base_type::iterator_category  iterator_category;
        typedef typename base_type::value_type         value_type;
        typedef typename base_type::extractor_type     extractor_type;
        typedef typename base_type::difference_type    difference_type;
        typedef typename base_type::pointer            pointer;
        typedef typename base_type::reference          reference;
        // Underlying iterator type
        typedef typename base_type::itor_type          itor_type;
      public:
        iterator(const itor_type& itor) : base_type(itor) {};
        virtual ~iterator() {};
        //
        virtual iterator   operator++() {++this->_itor; return *this;};
        virtual iterator   operator++(int n) {iterator tmp(this->_itor); ++this->_itor; return tmp;};
      };// class iterator
    }; // struct IndexSequenceTraits
    
    template <typename Index_, typename ValueExtractor_>
    struct ReversibleIndexSequenceTraits {
      typedef IndexSequenceTraits<Index_, ValueExtractor_> base_traits;
      typedef typename base_traits::iterator_base   iterator_base;
      typedef typename base_traits::iterator        iterator;
      typedef typename base_traits::index_type      index_type;

      // reverse_iterator is the reverse of iterator
      class reverse_iterator : public iterator_base {
      public:
        // Standard iterator typedefs
        typedef iterator_base                          base_type;
        typedef typename base_type::iterator_category  iterator_category;
        typedef typename base_type::value_type         value_type;
        typedef typename base_type::extractor_type     extractor_type;
        typedef typename base_type::difference_type    difference_type;
        typedef typename base_type::pointer            pointer;
        typedef typename base_type::reference          reference;
        // Underlying iterator type
        typedef typename base_type::itor_type          itor_type;
      public:
        reverse_iterator(const itor_type& itor) : base_type(itor) {};
        virtual ~reverse_iterator() {};
        //
        virtual reverse_iterator     operator++() {--this->_itor; return *this;};
        virtual reverse_iterator     operator++(int n) {reverse_iterator tmp(this->_itor); --this->_itor; return tmp;};
      };
    }; // class ReversibleIndexSequenceTraits


    //
    // Rec & RecContainer definitions.
    // Rec is intended to denote a graph point record.
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
      };// degreeAdjuster()
    };// class Rec

    template <typename Point_>
    struct RecContainerTraits {
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
            ::boost::multi_index::tag<degreeTag>, BOOST_MULTI_INDEX_MEMBER(rec_type, int, degree)
          >
        >,
        ALE_ALLOCATOR<rec_type>
      > set_type; 
      //
      // Return types
      //
      // Rec specialization of OutputSequence and related iterators and methods
      class DegreeSequence {
      public:
        typedef IndexSequenceTraits<typename ::boost::multi_index::index<set_type, degreeTag>::type,
                                    BOOST_MULTI_INDEX_MEMBER(rec_type, typename rec_type::point_type,point)>
        traits;
      protected:
        const typename traits::index_type& _index;
      public:
        
        // Need to extend the inherited iterator to be able to extract the degree
        class iterator : public traits::iterator {
        public:
          iterator(const typename traits::iterator::itor_type& itor) : traits::iterator(itor) {};
          virtual const int& degree() const {return this->_itor->degree;};
        };

        DegreeSequence(const DegreeSequence& seq)           : _index(seq._index) {};
        DegreeSequence(typename traits::index_type& index)  : _index(index)     {};
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
    };// struct RecContainerTraits

    template <typename Point_>
    struct RecContainer {
      typedef RecContainerTraits<Point_> traits;
      typedef typename traits::set_type set_type;
      set_type set;
      void adjustDegree(const typename traits::rec_type::point_type& p, int delta) {
        typename ::boost::multi_index::index<set_type, typename traits::pointTag>::type& index = 
          ::boost::multi_index::get<typename traits::pointTag>(this->set);
        typename ::boost::multi_index::index<set_type, typename traits::pointTag>::type::iterator i = index.find(p);
        if (i == index.end()) { // No such point exists
          if(delta < 0) { // Cannot decrease degree of a non-existent point
            ostringstream err;
            err << "ERROR: BiGraph::adjustDegree: Non-existent point " << p;
            std::cout << err << std::endl;
            throw(Exception(err.str().c_str()));
          }
          else { // We CAN INCREASE the degree of a non-existent point: simply insert a new element with degree == delta
            std::pair<typename ::boost::multi_index::index<set_type, typename traits::pointTag>::type::iterator, bool> ii;
            typename traits::rec_type r(p,delta);
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
          index.modify(i, typename traits::rec_type::degreeAdjuster(delta));
        }
      }; // adjustDegree()
    }; // struct RecContainer

    // 
    // Arrow & ArrowContainer definitions
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
      Arrow(const source_type& s, const target_type& t, const color_type& c) : source(s), target(t), color(c) {};
      //
      // Arrow modifiers
      //
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
    };// struct Arrow
    

    template<typename Source_, typename Target_, typename Color_>
    struct ArrowContainerTraits {
    public:
      //
      // Encapsulated types
      //
      typedef Arrow<Source_,Target_,Color_>    arrow_type;
      typedef typename arrow_type::source_type source_type;
      typedef typename arrow_type::target_type target_type;
      typedef typename arrow_type::color_type  color_type;
      // Index tags
      struct                                   sourceColorTag{};
      struct                                   targetColorTag{};
      struct                                   sourceTargetTag{};      

      // Sequence type
      template <typename Index_, typename Key_, typename SubKey_, typename ValueExtractor_>
      class ArrowSequence {
        // ArrowSequence implements ReversibleIndexSequencTraits with Index_ and ValueExtractor_ types.
        // A Key_ object and an optional SubKey_ object are used to extract the index subset.
      public:
        typedef ReversibleIndexSequenceTraits<Index_, ValueExtractor_>  traits;
        typedef Key_                                                    key_type;
        typedef SubKey_                                                 subkey_type;
      protected:
        const typename traits::index_type&                              _index;
        const key_type                                                  key;
        const subkey_type                                               subkey;
        const bool                                                      useSubkey;
      public:
        // Need to extend the inherited iterators to be able to extract arrow color
        class iterator : public traits::iterator {
        public:
          iterator(const typename traits::iterator::itor_type& itor) : traits::iterator(itor) {};
          virtual const color_type& color() const {return this->_itor->color;};
        };
        class reverse_iterator : public traits::reverse_iterator {
        public:
          reverse_iterator(const typename traits::reverse_iterator::itor_type& itor) : traits::reverse_iterator(itor) {};
          virtual const color_type& color() const {return this->_itor->color;};
        };
      public:
        //
        // Basic ArrowSequence interface
        //
        ArrowSequence(const ArrowSequence& seq) : _index(seq._index), key(seq.key), subkey(seq.subkey), useSubkey(seq.useSubkey) {};
        ArrowSequence(const typename traits::index_type& index, const key_type& k) : 
          _index(index), key(k), subkey(subkey_type()), useSubkey(0) {};
        ArrowSequence(const typename traits::index_type& index, const key_type& k, const subkey_type& kk) : 
          _index(index), key(k), subkey(kk), useSubkey(1){};
        virtual ~ArrowSequence() {};
        
        virtual iterator begin() {
          if (this->useSubkey) {
            return iterator(this->_index.lower_bound(::boost::make_tuple(this->key,this->subkey)));
          } else {
            return iterator(this->_index.lower_bound(::boost::make_tuple(this->key)));
          }
        };
        
        virtual iterator end() {
          if (this->useSubkey) {
            return iterator(this->_index.upper_bound(::boost::make_tuple(this->key,this->subkey)));
          } else {
            return iterator(this->_index.upper_bound(::boost::make_tuple(this->key)));
          }
        };
        
        virtual reverse_iterator rbegin() {
          if (this->useSubkey) {
            return reverse_iterator(--this->_index.upper_bound(::boost::make_tuple(this->key,this->subkey)));
          } else {
            return reverse_iterator(--this->_index.upper_bound(::boost::make_tuple(this->key)));
          }
        };
        
        virtual reverse_iterator rend() {
          if (this->useSubkey) {
            return reverse_iterator(--this->_index.lower_bound(::boost::make_tuple(this->key,this->subkey)));
          } else {
            return reverse_iterator(--this->_index.lower_bound(::boost::make_tuple(this->key)));
          }
        };
        
        virtual std::size_t size() {
          if (this->useSubkey) {
            return this->_index.count(::boost::make_tuple(this->key,subkey));
          } else {
            return this->_index.count(::boost::make_tuple(this->key));
          }
        };

        template<typename ostream_type>
        void view(ostream_type& os, const bool& useColor = false, const char* label = NULL){
          if(label != NULL) {
            os << "Viewing " << label << " sequence:" << std::endl;
          } 
          os << "[";
          for(iterator i = this->begin(); i != this->end(); i++) {
            os << " (" << *i;
            if(useColor) {
              os << "," << i.color();
            }
            os  << ")";
          }
          os << " ]" << std::endl;
        };
      };// class ArrowSequence    
    };// class ArrowContainerTraits
  

    // The specialized ArrowContainer types distinguish the cases of unique and multiple colors of arrows on 
    // for each (source,target) pair (i.e., a single arrow, or multiple arrows between each pair of points).
    typedef enum {multiColor, uniColor} ColorMultiplicity;

    template<typename Source_, typename Target_, typename Color_, ColorMultiplicity colorMultiplicity> 
    struct ArrowContainer {};
    
    template<typename Source_, typename Target_, typename Color_>
    struct ArrowContainer<Source_, Target_, Color_, multiColor> {
      // Define container's encapsulated types
      typedef ArrowContainerTraits<Source_, Target_, Color_> traits;
      // need to def arrow_type locally, since BOOST_MULTI_INDEX_MEMBER barfs when first template parameter starts with 'typename'
      typedef typename traits::arrow_type                         arrow_type; 

      // Container set type
      typedef ::boost::multi_index::multi_index_container<
        typename traits::arrow_type,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<typename traits::sourceTargetTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type, 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source), 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target)
            >
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<typename traits::sourceColorTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type, 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source), 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::color_type,  color)
            >
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<typename traits::targetColorTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type, 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target), 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::color_type,  color)
            >
          >
        >,
        ALE_ALLOCATOR<typename traits::arrow_type>
      > set_type;      
      // multi-index set of multicolor arrows
      set_type set;
    }; // class ArrowContainer<multiColor>
    
    template<typename Source_, typename Target_, typename Color_>
    struct ArrowContainer<Source_, Target_, Color_, uniColor> {
      // Define container's encapsulated types
      typedef ArrowContainerTraits<Source_, Target_, Color_>      traits;
      // need to def arrow_type locally, since BOOST_MULTI_INDEX_MEMBER barfs when first template parameter starts with 'typename'
      typedef typename traits::arrow_type                         arrow_type; 

      // multi-index set type -- arrow set
      typedef ::boost::multi_index::multi_index_container<
        typename traits::arrow_type,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<typename traits::sourceTargetTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type, 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source), 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target)
            >
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<typename traits::sourceColorTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type, 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source), 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::color_type,  color)
            >
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<typename traits::targetColorTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type, 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target), 
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::color_type,  color)
            >
          >
        >,
        ALE_ALLOCATOR<typename traits::arrow_type>
      > set_type;      
      // multi-index set of unicolor arrow records 
      set_type set;
    }; // class ArrowContainer<uniColor>



    //
    // ColorBiGraph (short for ColorBipartiteGraph) implements a sequential interface similar to that of Sieve,
    // except the source and target points may have different types and iterated operations (e.g., nCone, closure)
    // are not available.
    // 
    template<typename Source_, typename Target_, typename Color_, ColorMultiplicity colorMultiplicity>
    class ColorBiGraph { // class ColorBiGraph
    public:
      typedef struct {
        // Encapsulated container types
        typedef ArrowContainer<Source_, Target_, Color_, colorMultiplicity>      arrow_container_type;
        typedef RecContainer<Source_>                                            cap_container_type;
        typedef RecContainer<Target_>                                            base_container_type;
        // Types associated with records held in containers
        typedef typename arrow_container_type::traits::arrow_type                arrow_type;
        typedef typename arrow_container_type::traits::source_type               source_type;
        typedef typename arrow_container_type::traits::target_type               target_type;
        typedef typename arrow_container_type::traits::color_type                color_type;
        // Convenient tag names
        typedef typename arrow_container_type::traits::sourceColorTag            supportInd;
        typedef typename arrow_container_type::traits::targetColorTag            coneInd;
        typedef typename arrow_container_type::traits::sourceTargetTag           arrowInd;
        typedef typename base_container_type::traits::degreeTag                  baseInd;
        typedef typename cap_container_type::traits::degreeTag                   capInd;
        //
        // Return types
        //
        typedef typename
        arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,arrowInd>::type, source_type, target_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, color_type, color)> 
        arrowSequence;

        typedef typename 
        arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,coneInd>::type, target_type, color_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, source_type, source)> 
        coneSequence;

        typedef typename 
        arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type, supportInd>::type, source_type, color_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, target_type, target)> 
        supportSequence;
     
        typedef typename base_container_type::traits::DegreeSequence baseSequence;
        typedef typename cap_container_type::traits::DegreeSequence  capSequence;
        typedef std::set<source_type> coneSet;
        typedef std::set<target_type> supportSet;

      } traits;
    public:
      // Debug level
      int debug;
    protected:
      typename traits::arrow_container_type _arrows;
      typename traits::base_container_type  _base;
      typename traits::cap_container_type   _cap;
    public:
      // 
      // Basic interface
      //
      ColorBiGraph(int debug = 0) : debug(debug) {};
      virtual ~ColorBiGraph(){};
      //
      // Query methods
      //
      Obj<typename traits::capSequence>   
      cap() {
        return typename traits::capSequence(::boost::multi_index::get<typename traits::capInd>(_cap.set));
      };
      Obj<typename traits::baseSequence>    
      base() {
        return typename traits::baseSequence(::boost::multi_index::get<typename traits::baseInd>(_base.set));
      };
      Obj<typename traits::arrowSequence> 
      arrows(const typename traits::source_type& s, const typename traits::target_type& t) {
        return typename traits::arrowSequence(::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set), s, t);
      };
      Obj<typename traits::coneSequence> 
      cone(const typename traits::target_type& p) {
        return typename traits::coneSequence(::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set), p);
      };
      template<class InputSequence> 
      Obj<typename traits::coneSet> 
      cone(const Obj<InputSequence>& points) {
        return this->cone(points, typename traits::color_type(), false);
      };
      Obj<typename traits::coneSequence> 
      cone(const typename traits::target_type& p, const typename traits::color_type& color) {
        return typename traits::coneSequence(::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set), p, color);
      };
      template<class InputSequence>
      Obj<typename traits::coneSet> 
      cone(const Obj<InputSequence>& points, const typename traits::color_type& color, bool useColor = true) {
        Obj<typename traits::coneSet> cone = typename traits::coneSet();
        for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
          Obj<typename traits::coneSequence> pCone;
          if (useColor) {
            pCone = this->cone(*p_itor, color);
          } else {
            pCone = this->cone(*p_itor);
          }
          cone->insert(pCone->begin(), pCone->end());
        }
        return cone;
      };
      Obj<typename traits::supportSequence> 
      support(const typename traits::source_type& p) {
        return typename traits::supportSequence(::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set), p);
      };
      Obj<typename traits::supportSequence> 
      support(const typename traits::source_type& p, const typename traits::color_type& color) {
        return typename traits::supportSequence(::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set), p, color);
      };
      template<class sourceInputSequence>
      Obj<typename traits::supportSet>      
      support(const Obj<sourceInputSequence>& sources);
      // unimplemented
      template<class sourceInputSequence>
      Obj<typename traits::supportSet>      
      support(const Obj<sourceInputSequence>& sources, const typename traits::color_type& color);
      // unimplemented
 
      template<typename ostream_type>
      void view(ostream_type& os, const char* label = NULL){
        if(label != NULL) {
          os << "Viewing BiGraph '" << label << "':" << std::endl;
        } 
        else {
          os << "Viewing a BiGraph:" << std::endl;
        }
        os << "cap --> base:" << std::endl;
        typename traits::capSequence cap = this->cap();
        for(typename traits::capSequence::iterator capi = cap.begin(); capi != cap.end(); capi++) {
          typename traits::supportSequence supp = this->support(*capi);
          for(typename traits::supportSequence::iterator suppi = supp.begin(); suppi != supp.end(); suppi++) {
            os << *capi << "--" << suppi.color() << "-->" << *suppi << std::endl;
          }
        }
        os << "base <-- cap:" << std::endl;
        typename traits::baseSequence base = this->base();
        for(typename traits::baseSequence::iterator basei = base.begin(); basei != base.end(); basei++) {
          typename traits::coneSequence cone = this->cone(*basei);
          for(typename traits::coneSequence::iterator conei = cone.begin(); conei != cone.end(); conei++) {
            os << *basei <<  "<--" << conei.color() << "--" << *conei << std::endl;
          }
        }
        os << "cap --> outdegrees:" << std::endl;
        for(typename traits::capSequence::iterator capi = cap.begin(); capi != cap.end(); capi++) {
            os << *capi <<  "-->" << capi.degree() << std::endl;
        }
        os << "base <-- indegrees:" << std::endl;
        for(typename traits::baseSequence::iterator basei = base.begin(); basei != base.end(); basei++) {
            os << *basei <<  "<--" << basei.degree() << std::endl;
        }
      };
    public:
      //
      // Lattice queries
      //
      template<class targetInputSequence> 
      Obj<typename traits::coneSequence> meet(const Obj<targetInputSequence>& targets);
      // unimplemented
      template<class targetInputSequence> 
      Obj<typename traits::coneSequence> meet(const Obj<targetInputSequence>& targets, const typename traits::color_type& color);
      // unimplemented
      template<class sourceInputSequence> 
      Obj<typename traits::coneSequence> join(const Obj<sourceInputSequence>& sources);
      // unimplemented
      template<class sourceInputSequence> 
      Obj<typename traits::coneSequence> join(const Obj<sourceInputSequence>& sources, const typename traits::color_type& color);
    public:
      //
      // Structural manipulation
      //
      void clear() {
        this->_arrows.set.clear(); this->_base.set.clear(); this->_cap.set.clear();
      };
      void addArrow(const typename traits::source_type& p, const typename traits::target_type& q) {
        this->addArrow(p, q, typename traits::color_type());
      };
      void addArrow(const typename traits::source_type& p, const typename traits::target_type& q, const typename traits::color_type& color) {
        this->addArrow(typename traits::arrow_type(p, q, color));
        //std::cout << "Added " << arrow_type(p, q, color);
      };
      void addArrow(const typename traits::arrow_type& a) {
        this->_arrows.set.insert(a); _base.adjustDegree(a.target,1); _cap.adjustDegree(a.source,1);
        //std::cout << "Added " << Arrow_(p, q, color);
      };
      void addCone(const typename traits::source_type& source, const typename traits::target_type& target){
        this->addArrow(source, target);
      };
      template<class sourceInputSequence> 
      void addCone(const Obj<sourceInputSequence>& sources, const typename traits::target_type& target) {
        this->addCone(sources, target, typename traits::color_type());
      };
      void addCone(const typename traits::source_type& source, const typename traits::target_type& target, const typename traits::color_type& color) {
        this->addArrow(source, target, color);
      };
      template<class sourceInputSequence> 
      void 
      addCone(const Obj<sourceInputSequence>& sources, const typename traits::target_type& target, const typename traits::color_type& color) {
        if (debug) {std::cout << "Adding a cone " << std::endl;}
        for(typename sourceInputSequence::iterator iter = sources->begin(); iter != sources->end(); ++iter) {
          if (debug) {std::cout << "Adding arrow from " << *iter << " to " << target << "(" << color << ")" << std::endl;}
          this->addArrow(*iter, target, color);
        }
      };
    private:
      void 
      __clearCone(const typename traits::target_type& t, const typename traits::color_type&  color, bool useColor = false) {
        // Use the cone sequence types to clear the cone
        typename traits::coneSequence cone;
        typename traits::coneSequence::index& coneIndex = ::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set);
        typename traits::coneSequence::index::iterator i, ii;
        if (this->useColor) {
          i = coneIndex.lower_bound(::boost::make_tuple(t,color));
          ii = coneIndex.upper_bound(::boost::make_tuple(t,color));
        } else {
          i = coneIndex.lower_bound(::boost::make_tuple(t));
          ii = coneIndex.upper_bound(::boost::make_tuple(t));
        }
        for(; i != ii; i++){
          // Adjust the degrees before removing the arrow
          this->_cap.adjustDegree(i->source, -1);
          this->_base.adjustDegree(i->target, -1);
          coneIndex.erase(i);
        }
      };
    public:
      void setCone(const typename traits::source_type& source, const typename traits::target_type& target){
        this->__clearCone(target, typename traits::color_type(), false); this->addCone(source, target);
      };
      template<class sourceInputSequence> 
      void setCone(const Obj<sourceInputSequence>& sources, const typename traits::target_type& target) {
        this->__clearCone(target, typename traits::color_type(), false); this->addCone(sources, target, typename traits::color_type());
      };
      void setCone(const typename traits::source_type& source, const typename traits::target_type& target, const typename traits::color_type& color) {
        this->__clearCone(target, color, true); this->addCone(source, target, color);
      };
      template<class sourceInputSequence> 
      void setCone(const Obj<sourceInputSequence>& sources, const typename traits::target_type& target, const typename traits::color_type& color){
        this->__clearCone(target, color, true); this->addCone(sources, target, color);
      };
      template<class targetInputSequence> 
      void addSupport(const typename traits::source_type& source, const Obj<targetInputSequence >& targets);
      // Unimplemented
      template<class targetInputSequence> 
      void addSupport(const typename traits::source_type& source, const Obj<targetInputSequence>& targets, const typename traits::color_type& color);
        
      void add(const Obj<ColorBiGraph<typename traits::source_type, typename traits::target_type, const typename traits::color_type, colorMultiplicity> >& cbg);
      // Unimplemented

      //
      // Delta interface
      //
      struct delta { // struct delta
        // A direct product out-of-place cone fuser
        template <typename OtherColorBiGraph_>
        struct DefaultConeFuser {
          typedef ColorBiGraph<Source_, Target_, Color_, colorMultiplicity> left_type;
          typedef OtherColorBiGraph_                                        right_type;
          
          struct traits {
            typedef std::pair<typename left_type::traits::source_type,typename right_type::traits::source_type> source_type;
            typedef typename left_type::traits::target_type                                                     target_type;
            typedef std::pair<typename left_type::traits::color_type,typename right_type::traits::color_type>   color_type;
            typedef Arrow<source_type, target_type, color_type>                                                 arrow_type;
          };
          
          typename traits::arrow_type 
          operator()(const typename left_type::traits::arrow_type&  left_arrow, const typename right_type::traits::arrow_type& right_arrow) {
            // Form the arrow from two pairs and a target
            return typename traits::arrow_type(typename traits::source_type(left_arrow.source,right_arrow.source),
                                               left_arrow.target,
                                               typename traits::color_type(left_arrow.color, right_arrow.color));
          };
        }; // struct DefaultConeFuser

        // cone delta operates over the base overlap of two ColorBiGraphs
        template <typename OtherColorBiGraph_, typename Fuser_ = DefaultConeFuser<OtherColorBiGraph_> >
        struct cone { // delta::cone
          typedef ColorBiGraph<Source_, Target_, Color_, colorMultiplicity> left_type;
          typedef OtherColorBiGraph_                                        right_type;
          typedef Fuser_                                                    fuser_type;
          typedef ColorBiGraph<typename fuser_type::traits::source_type, 
                               typename fuser_type::traits::target_type, 
                               typename fuser_type::traits::color_type, 
                               colorMultiplicity>                           delta_type;
          // Ideally we should check the equality of left::traits::target_type & right::traits::target_type,
          // although instantiation will fail somewhere (e.g., when the base intersection is computed).
          class BaseOverlapSequence {
          public:
            typedef typename left_type::traits::base_container::traits::rec_type rec_type;
            typedef 
            IndexSequenceTraits<typename ::boost::multi_index::index<typename left_type::traits::base_container::traits::set_type, 
                                                                    typename left_type::traits::base_container::traits::pointTag>::type,
                                 BOOST_MULTI_INDEX_MEMBER(rec_type, 
                                                        typename left_type::traits::base_container::traits::rec_type::point_type,point)>
            traits;
            virtual typename traits::iterator begin() {
              //return typename traits::iterator();
            };
            virtual typename traits::iterator end() {
              //return typename traits::iterator();
            };            
            BaseOverlapSequence(const left_type& l, const right_type& r){
            };
          protected:
            typename 
            ::boost::multi_index::index<typename left_type::traits::base_container::traits::set_type, 
                                        typename left_type::traits::base_container::traits::pointTag>::type
            _left_index;
            typename 
            ::boost::multi_index::index<typename left_type::traits::base_container::traits::set_type, 
                                        typename left_type::traits::base_container::traits::pointTag>::type
            _right_index;
          };

          // delta::cone::operator() in various versions 
          void 
          operator()(const left_type& l, const right_type& r, delta_type& d, const fuser_type& f = fuser_type()) {
            // Compute the overlap of left and right bases and then call a 'based' version of the operator
            operator()(overlap(l,r), l,r,d,f);
          };
          void 
          operator()(const BaseOverlapSequence& overlap,const left_type& l,const right_type& r, delta_type& d, const fuser_type& f = fuser_type()) {
            // We assume that the fuser knows how to fuse a left_type::traits::arrow_type and a right_type::traits::arrow_type
            // to produce a delta_type::traits::arrow_type
            for(typename BaseOverlapSequence::iterator i = overlap.begin(); i != overlap.end(); i++) {
              // At each overlap point we have to use a local n^2 algorithm.  The alternative would be to have the fuser
              // produce the whole delta cone in a set or a sequence, which would have to be copied into d.
              // That would amount to putting almost all of the delta complexity into the fuser, which might not be so bad,
              // after all, but needs to be thought through.
              // However, in the case of a UniColorBiGraph, both cones contain at most one point
              typename left_type::traits::coneSequence lcone = l.cone(*i);
              typename right_type::traits::coneSequence rcone = r.cone(*i);
              for(typename left_type::traits::coneSequence::iterator lc = lcone.begin(); lc != lcone.end(); lc++) {
                for(typename right_type::traits::coneSequence::iterator rc = rcone.begin(); rc != rcone.end(); rc++) {
                  d.addArrow(f(lc.arrow(),rc.arrow()));
                }
              }
            }            
          };
          Obj<delta_type> 
          operator()(const left_type& l, const right_type& r, const fuser_type& f = fuser_type()) {
            Obj<delta_type> d = delta_type();
            operator()(l,r,d,f);
            return d;
          };
          Obj<delta_type> 
          operator()(const BaseOverlapSequence& overlap, const left_type& l, const right_type& r, const fuser_type& f = fuser_type()) {
            Obj<delta_type> d = delta_type();
            operator()(overlap,l,r,d,f);
            return d;
          };
        public:
        }; // struct cone  
      }; // struct delta
    }; // class ColorBiGraph


    // The parameterizing types of a base ColorBiGraph type can be reintroduced into a descendant type by using a ColorBiGraphTraits 
    // template class.  ColorBiGraphTraits uses the traits of the template parameter in typedef redefinition of these types.
    template<typename ColorBiGraph_>
    struct ColorBiGraphTraits {
      typedef ColorBiGraph_                                    base_type;
      // Container type names
      typedef typename base_type::traits::arrow_container_type arrow_container_type;
      typedef typename base_type::traits::base_container_type  base_container_type;
      typedef typename base_type::traits::cap_container_type   cap_container_type;
      // Arrow type names
      typedef typename base_type::traits::arrow_type           arrow_type;
      typedef typename base_type::traits::source_type          source_type;
      typedef typename base_type::traits::target_type          target_type;
      typedef typename base_type::traits::color_type           color_type;
      // Convenient index tag names
      typedef typename base_type::traits::supportInd           supportInd;
      typedef typename base_type::traits::coneInd              coneInd;
      typedef typename base_type::traits::arrowInd             arrowInd;
      typedef typename base_type::traits::baseInd              baseInd;
      typedef typename base_type::traits::capInd               capInd;
      // Output sequence type names
      typedef typename base_type::traits::baseSequence         baseSequence;
      typedef typename base_type::traits::capSequence          capSequence;
      typedef typename base_type::traits::arrowSequence        arrowSequence;
      typedef typename base_type::traits::coneSequence         coneSequence;
      typedef typename base_type::traits::supportSequence      supportSequence;
      typedef typename base_type::traits::coneSet              coneSet;
      typedef typename base_type::traits::supportSet           supportSet;

    };

    // A UniColorBiGraph aka BiGraph
    template <typename Source_, typename Target_, typename Color_>
    class BiGraph : public ColorBiGraph<Source_, Target_, Color_, uniColor> {
    public:
      typedef ColorBiGraphTraits<ColorBiGraph<Source_, Target_, Color_, uniColor> >   traits;
      // Re-export some typedefs expected by CoSifter
      typedef typename traits::arrow_type                                             Arrow_;
      typedef typename traits::coneSequence                                           coneSequence;
      typedef typename traits::supportSequence                                        supportSequence;
      typedef typename traits::baseSequence                                           baseSequence;
      typedef typename traits::capSequence                                            capSequence;
      // Basic interface
      BiGraph(const int& debug = 0) : traits::base_type(debug) {};
      
      const typename traits::color_type&
      getColor(const typename traits::source_type& s, const typename traits::target_type& t, bool fail = true) {
        typename traits::arrowSequence arr = this->arrows(s,t);
        if(arr.begin() != arr.end()) {
          return arr.begin().color();
        }
        if (fail) {
          ostringstream o;
          o << "Arrow " << s << " --> " << t << " not present";
          throw ALE::Exception(o.str().c_str());
        } else {
          static typename traits::color_type c;
          return c;
        }
      };

      template<typename ColorChanger>
      void modifyColor(const typename traits::source_type& s, const typename traits::target_type& t, const ColorChanger& changeColor) {
        typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type, typename traits::arrowInd>::type& index = 
          ::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set);
        typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type, typename traits::arrowInd>::type::iterator i = 
          index.find(::boost::make_tuple(t,s));
        if (i != index.end()) {
          index.modify(i, changeColor);
        } else {
          typename traits::arrow_type a(s, t, typename traits::color_type());
          changeColor(a);
          this->addArrow(a);
        }
      };
    };// class BiGraph

  } // namespace Two

} // namespace ALE

#endif
