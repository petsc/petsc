#ifndef included_ALE_CoSifter_hh
#define included_ALE_CoSifter_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif
#include <stack>
#include <queue>

// Dmitry's explanation:
//
// Okay, check out what I have put there.
// It's a rather high-level interface, but I think it sketches out the implementation idea.  I have also become a master of switching from 'public' to 'private' and back.

// The idea is to put more power into BiGraphs (BipartiteGraphs).  They are like Sieves but with two point types (source and target) and no recursive operations (nCone, closure, etc).
// I claim they should be parallel, so cone/support completions should be computable for them.  The footprint is incorporated into the color of the new BiGraph, which is returned as a completion.
// It would be very natural to have Sieve<Point_, Color_> to extend BiGraph<Point_, Point_, Color_> with the recursive operations.

// The reason for putting the completion functionality into BiGraphs is that patches and indices under and over a topology Sieve are BiGraphs and have to be completed:
// the new overlap_patches has to encode patch pairs along with the rank of the second patch (first is always local); likewise, overlap_indices must encode a pair of intervals with a rank
// -- the attached to the same Sieve point by two different processes -- one local and one (possibly) remote.  At any rate, the support completion of 'patches' contains all the information
// needed for 'overlap_patches' -- remember that struct with a triple {point, patch, number} you had on the board?   Likewise for 'overlap_indices' built out of the cone completion of 'indices'.

// Once the 'overlap_XXX' are computed, we can allocate the storage for the Delta data and post sends receives.
// We should be able to reuse the completion subroutine from the old Sieve.
// So you are right that perhaps Sieve completion gets us to the CoSieve completion, except I think it's the BiGraph completion this time.
// I can do the completion when I come back if you get the serial BiGraph/Sieve stuff going.
//
namespace ALE {
  namespace def {
    //
    // BiGraph (short for BipartiteGraph) implements a sequential interface similar to that of Sieve,
    // except the source and target points may have different types and iterated operations (e.g., nCone, closure)
    // are not available.
    // 
    template<typename Source_, typename Target_, typename Color_>
    class BiGraph {
    public:
      typedef Source_ source_type;
      typedef Target_ target_type;
      typedef Color_  color_type;
    private:
      // Arrow storage
      struct source{};
      struct target{};
      struct color{};
      struct sourceColor{};
      struct targetColor{};
      typedef Arrow<source_type,target_type,color_type> Arrow_;
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
        const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& coneIndex;
        const target_type key;
        const color_type  color;
        const bool        useColor;
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
          virtual const source_type& operator*() const {return this->arrowIter->source;};
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

        coneSequence(const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& cone, const target_type& p) : coneIndex(cone), key(p), color(color_type()), useColor(0) {};
        coneSequence(const typename ::boost::multi_index::index<ArrowSet,targetColor>::type& cone, const target_type& p, const color_type& c) : coneIndex(cone), key(p), color(c), useColor(1) {};
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
        std::cout << "Adding a cone " << std::endl;
        for(typename sourceInputSequence::iterator iter = sources->begin(); iter != sources->end(); ++iter) {
          std::cout << "Adding arrow from " << *iter << " to " << target << "(" << color << ")" << std::endl;
          this->addArrow(*iter, target, color);
        }
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
      // Structural methods
      void stratify() {
        this->__computeDegrees();
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
        const typename ::boost::multi_index::index<ArrowSet,source>::type& supports = ::boost::multi_index::get<source>(this->arrows);
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


    //
    // CoSieve:
    // This object holds the data layout and the data over a Sieve patitioned into support patches.
    //
    template <typename Sieve_, typename Patch_, typename Index_, typename Value_>
    class CoSieve {
    public:
      //     Basic types
      typedef Sieve_ Sieve;
      typedef Value_ value_type;
      typedef Patch_ patch_type;
      typedef Index_ index_type;
      typedef std::vector<index_type> IndexArray;
      typedef std::map<typename Sieve::point_type, index_type> IndexMap;
    private:
      // Base topology
      Obj<Sieve>   _topology;
    public:
      // Breakdown of the base Sieve into patches
      //   the colors (int) order the points (Sieve::point_type) over a patch (patch_type).
      typedef BiGraph<typename Sieve::point_type, patch_type, int> patches_type;
    private:
      patches_type _patches; 
    public:
      // Attachment of fiber dimension intervals to Sieve points
      //   fields are encoded by colors, which double as the field ordering index
      //   colors (<patch_type, int>) order the indices (index_type, usually an interval) over a point (Sieve::point_type).
      typedef std::pair<patch_type, int> index_color;
      typedef BiGraph<index_type, typename Sieve::point_type, index_color> indices_type;
    private:
      indices_type _indices; 
    private:
      // Holds values for each patch
      std::map<patch_type, value_type *> _storage;

      void __clear() {
        Obj<typename patches_type::baseSequence> patches = this->_patches.base();

        for(typename patches_type::baseSequence::iterator p_itor = patches->begin(); p_itor != patches->end(); ++p_itor) {
          delete [] this->_storage[*p_itor];
        }
        this->_patches.clear();
        this->_indices.clear();
        this->_storage.clear();
      };
    public:
      //     Topology Manipulation
      void           setTopology(const Obj<Sieve>& topology) {this->_topology = topology;};
      Obj<Sieve>     getTopology() {return this->_topology;};
      //     Patch manipulation
    private:
      template <typename InputSequence>
      void order(const patch_type& patch, Obj<InputSequence> base, std::map<typename Sieve::point_type, typename Sieve::point_type>& seen, int& offset) {
        // To enable the depth-first order traversal without recursion, we employ a stack.
        std::stack<typename Sieve::point_type> stk;

        // We traverse the sub-bundle over base
        for(typename InputSequence::reverse_iterator b_ritor = base->rbegin(); b_ritor != base->rend(); ++b_ritor) {
          std::cout << "Init Pushing " << *b_ritor << " on stack" << std::endl;
          stk.push(*b_ritor);
        }
        while(1) {
          if (stk.empty()) break;

          typename Sieve::point_type p = stk.top(); stk.pop();
          int p_dim = this->getIndexDimension(patch, p);
          int p_off;

          if(seen.find(p) != seen.end()){
            // If p has already been seen, we use the stored offset.
            p_off = seen[p].prefix;
          } else {
            // The offset is only incremented when we encounter a point not yet seen
            p_off   = offset;
            seen[p] = typename Sieve::point_type(p_off, 0);
            offset += p_dim;
          }
          std::cout << "  Point " << p << " with dimension " << p_dim << " and offset " << p_off << std::endl;

          Obj<typename Sieve::coneSequence> cone = this->getTopology()->cone(p);
          for(typename InputSequence::iterator s_itor = base->begin(); s_itor != base->end(); ++s_itor) {
            // I THINK THIS IS ALWAYS TRUE NOW
            if (*s_itor == p) {
              std::cout << "  Found p in base" << std::endl;
              // If s (aka p) has a nonzero dimension but has not been indexed yet
              if((p_dim > 0) && (seen[p].index == 0)) {
                typename Sieve::point_type newIndex(p_off, p_dim);

                seen[p] = newIndex;
                this->_indices.replaceSourceOfTarget(p, newIndex);
                std::cout << "    Assigned new index " << newIndex << std::endl;
              }
              std::cout << "  now ordering cone" << std::endl;
              for(typename Sieve::coneSequence::iterator p_itor = cone->begin(); p_itor != cone->end(); ++p_itor) {
                std::cout << "  cone point " << *p_itor << std::endl;
              }
              this->order(patch, cone, seen, offset);
              break;
            }
          }
        }
      };

      void allocateAndOrderPatch(const patch_type& patch) {
        std::map<typename Sieve::point_type, typename Sieve::point_type> seen;
        int                                            offset = 0;

        std::cout << "Ordering patch " << patch << std::endl;
        this->order(patch, this->getPatch(patch), seen, offset);

        if (this->_storage.find(patch) != this->_storage.end()) {
          delete [] this->_storage[patch];
        }
        std::cout << "Allocated patch " << patch << " of size " << offset << std::endl;
        this->_storage[patch] = new value_type[offset];
      };
    public:
      // This attaches Sieve::point_type points to a patch in the order prescribed by the sequence; 
      // care must be taken not to assign duplicates (or should it?)
      template<typename pointInputSequence>
      void                            setPatch(const Obj<pointInputSequence>& points, const patch_type& patch) {
        this->_patches.addCone(points, patch);
        this->_patches.stratify();
      };
      // This retrieves the Sieve::point_type points attached to a given patch
      Obj<typename patches_type::coneSequence> getPatch(const patch_type& patch) {
        return this->_patches.cone(patch);
      };
      //     Index manipulation
      void orderPatches() {
        Obj<typename patches_type::baseSequence> base = this->_patches.base();

        for(typename patches_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          this->allocateAndOrderPatch(*b_iter);
        }
      };
      // These attach index_type indices of a given color to a Sieve::point_type point
      void  addIndices(const patch_type& patch, const index_type& indx, typename Sieve::color_type color, const typename Sieve::point_type& p) {
        this->_indices.addCone(indx, p, index_color(patch, color));
      };
      template<typename indexInputSequence>
      void  addIndices(const patch_type& patch, const Obj<indexInputSequence>& indices, typename Sieve::color_type color, const typename Sieve::point_type& p) {
        this->_indices.addCone(indices, p, index_color(patch, color));
      };
      template<typename indexInputSequence>
      void  setIndices(const patch_type& patch, const Obj<indexInputSequence>& indices, typename Sieve::color_type color, const typename Sieve::point_type& p) {
        this->_indices.setCone(indices, p, index_color(patch, color));
      };
      // This retrieves the index_type indices of a given color attached to a Sieve::point_type point
      const index_type getIndex(const patch_type& patch, const typename Sieve::point_type& p) {
        return *this->_indices.cone(p)->begin();
      };
      Obj<typename indices_type::coneSequence> getIndices(const patch_type& patch, const typename Sieve::point_type& p) {
        return this->_indices.cone(p);
      };
      template<typename pointSequence>
      Obj<IndexMap> getIndices(const patch_type& patch, Obj<pointSequence> points) {
        Obj<IndexMap> indices = IndexMap();

        for(typename pointSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          (*indices)[*p_iter] = *this->getIndices(patch, *p_iter)->begin();
        }
        return indices;
      }
    private:
      template<typename orderSequence>
      std::map<int, typename Sieve::point_type> __checkOrderChain(Obj<orderSequence> order, int& minDepth, int& maxDepth) {
        Obj<Sieve> topology = this->getTopology();
        std::map<int, typename Sieve::point_type> dElement;
        minDepth = 0;
        maxDepth = 0;

        // A topology cell-tuple contains one element per dimension, so we order the points by depth.
        for(typename orderSequence::iterator ord_itor = order->begin(); ord_itor != order->end(); ord_itor++) {
          int depth = topology->depth(*ord_itor);

          if (depth < 0) {
            throw Exception("Invalid element: negative depth returned"); 
          }
          if (depth > maxDepth) {
            maxDepth = depth;
          }
          if (depth < minDepth) {
            minDepth = depth;
          }
          dElement[depth] = *ord_itor;
        }
        // Verify that the chain is a "baricentric chain", i.e. it starts at depth 0
        //   and has an element of every depth between 0 and maxDepth
        //   and that each element at depth d is in the cone of the element at depth d+1
        if(minDepth != 0) {
          throw Exception("Invalid order chain: minimal depth is nonzero");
        }
        for(int d = 0; d <= maxDepth; d++) {
          typename std::map<int, typename Sieve::point_type>::iterator d_itor = dElement.find(d);

          if(d_itor == dElement.end()){
            ostringstream ex;
            //FIX: ex << "[" << this->getCommRank() << "]: " << "Missing Point at depth " << d;
            ex << "Missing Point at depth " << d;
            throw ALE::Exception(ex.str().c_str());
          }
          if(d > 0) {
            if(!topology->coneContains(dElement[d], dElement[d-1])){
              ostringstream ex;
              // FIX: ex << "[" << this->getCommRank() << "]: ";
              ex << "point (" << dElement[d-1].prefix << ", " << dElement[d-1].index << ") at depth " << d-1 << " not in the cone of ";
              ex << "point (" << dElement[d].prefix << ", " << dElement[d].index << ") at depth " << d;
              throw ALE::Exception(ex.str().c_str());
            }
          }
        }
        return dElement;
      };
      void __orderElement(int dim, typename Sieve::point_type element, std::map<int, std::queue<index_type> >& ordered, ALE::Obj<ALE::def::PointSet> elementsOrdered) {
        if (elementsOrdered->find(element) != elementsOrdered->end()) return;
        ordered[dim].push(element);
        elementsOrdered->insert(element);
        std::cout << "  ordered element " << element << " dim " << dim << std::endl;
      };
      typename Sieve::point_type __orderCell(int dim, std::map<int, typename Sieve::point_type>& orderChain, std::map<int, std::queue<index_type> >& ordered, Obj<PointSet> elementsOrdered) {
        typename Sieve::point_type last;

        std::cout << "Ordering cell " << orderChain[dim] << " dim " << dim << std::endl;
        for(int d = 0; d < dim; d++) {
          std::cout << "  orderChain["<<d<<"] " << orderChain[d] << std::endl;
        }
        if (dim == 0) {
          last = orderChain[0];
          this->__orderElement(0, last, ordered, elementsOrdered);
          return last;
        } else if (dim == 1) {
          Obj<typename Sieve::coneSequence> flip = this->_topology->cone(orderChain[1]);
          bool found = false;

          if (flip->size() != 2) throw ALE::Exception("Last edge did not separate two faces");
          for(typename Sieve::coneSequence::iterator c_iter = flip->begin(); c_iter != flip->end(); ++c_iter) {
            if (*c_iter != orderChain[dim-1]) {
              last = *c_iter;
              found = true;
              break;
            }
          }
          if (!found) throw ALE::Exception("Inconsistent edge separation");
          this->__orderElement(0, orderChain[0], ordered, elementsOrdered);
          this->__orderElement(0, last, ordered, elementsOrdered);
          this->__orderElement(1, orderChain[1], ordered, elementsOrdered);
          orderChain[dim-1] = last;
          return last;
        }
        Obj<Sieve> closure = this->_topology->closureSieve(orderChain[dim]);
        do {
          last = this->__orderCell(dim-1, orderChain, ordered, elementsOrdered);
          std::cout << "    last " << last << std::endl;
          Obj<typename Sieve::supportSequence> faces = closure->support(last);
          bool found = false;

          if (faces->size() != 2) throw ALE::Exception("Last edge did not separate two faces");
          for(typename Sieve::supportSequence::iterator s_iter = faces->begin(); s_iter != faces->end(); ++s_iter) {
            if (*s_iter != orderChain[dim-1]) {
              last = orderChain[dim-1];
              orderChain[dim-1] = *s_iter;
              found = true;
              break;
            }
          }
          if (!found) throw ALE::Exception("Inconsistent edge separation");
        } while(elementsOrdered->find(orderChain[dim-1]) == elementsOrdered->end());
        std::cout << "Finish ordering for cell " << orderChain[dim] << std::endl;
        std::cout << "  with last " << last << std::endl;
        orderChain[dim-1] = last;
        this->__orderElement(dim, orderChain[dim], ordered, elementsOrdered);
        return last;
      };
    public:
      template<typename orderSequence>
      Obj<IndexArray> getOrderedIndices(const patch_type& patch, Obj<orderSequence> order) {
        // We store the elements ordered in each dimension
        std::map<int, std::queue<index_type> > ordered;
        // Set of the elements already ordered
        Obj<PointSet> elementsOrdered = PointSet();
        Obj<IndexArray> indexArray = IndexArray();
        int minDepth, maxDepth;

        std::map<int, typename Sieve::point_type> dElement = this->__checkOrderChain(order, minDepth, maxDepth);
        std::cout << "Ordering " << dElement[maxDepth] << std::endl;
        // Could share the closure between these methods
        Obj<IndexMap> indices = this->getIndices(patch, this->_topology->closure(dElement[maxDepth]));
        typename Sieve::point_type last = this->__orderCell(maxDepth, dElement, ordered, elementsOrdered);
        for(int d = minDepth; d <= maxDepth; d++) {
          while(!ordered[d].empty()) {
            index_type ind = (*indices)[ordered[d].front()];

            ordered[d].pop();
            std::cout << "  indices " << ind << std::endl;
            if (ind.index > 0) {
              indexArray->push_back(ind);
            }
          }
        }
        return indexArray;
      };
      int getIndexDimension(const patch_type& patch, const typename Sieve::point_type& p) {
        Obj<typename indices_type::coneSequence> cone = this->_indices.cone(p);
        int dim = 0;

        std::cout << "  getting dimension of " << p << " in patch " << patch << std::endl;
        for(typename indices_type::coneSequence::iterator iter = cone->begin(); iter != cone->end(); ++iter) {
          std::cout << "    adding " << *iter << std::endl;
          dim += (*iter).index;
        }
        return dim;
      };
      // Attach indexDim indices to each element of a certain depth in the topology
      void setIndexDimensionByDepth(int depth, int indexDim) {
        this->setIndexDimensionByDepth(depth, typename Sieve::color_type(), indexDim);
      }
      void setIndexDimensionByDepth(int depth, typename Sieve::color_type color, int indexDim) {
        Obj<typename patches_type::baseSequence> base = this->_patches.base();
        Obj<typename Sieve::depthSequence> stratum = this->getTopology()->depthStratum(depth);

        std::cout << "Setting all points of depth " << depth << " to have dimension " << indexDim << std::endl;
        for(typename patches_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          std::cout << "  traversing patch " << *b_iter << std::endl;
          for(typename Sieve::depthSequence::iterator iter = stratum->begin(); iter != stratum->end(); ++iter) {
            std::cout << "  setting dimension of " << *iter << " to " << indexDim << std::endl;
            this->addIndices(*b_iter, typename Sieve::point_type(-1, indexDim), color, *iter);
          }
        }
      };
      const value_type *restrict(const patch_type& patch) {
        return this->_storage[patch];
      };
      const value_type *restrict(const patch_type& patch, const typename Sieve::point_type& p) {
        Obj<typename indices_type::coneSequence> indices = this->getIndices(patch, p);

        if (indices->size() != 1) {
          throw ALE::Exception("Invalid indices for requested point");
        }
        return &(this->_storage[patch][(*indices->begin()).prefix]);
      }
      template<typename InputSequence>
      const value_type *restrict(const patch_type& patch, const InputSequence& pointSequence) {
        throw ALE::Exception("Not implemented");
      };
      // Insert values into the specified patch
      void update(const patch_type& patch, const typename Sieve::point_type& p, value_type values[]) {
        Obj<typename indices_type::coneSequence> indices = this->getIndices(patch, p);

        for(typename indices_type::coneSequence::iterator ind = indices->begin(); ind != indices->end(); ++ind) {
          int offset = (*ind).prefix;

          for(int i = 0; i < (*ind).index; ++i) {
            std::cout << "Set a[" << offset+i << "] = " << values[i] << " on patch " << patch << std::endl;
            this->_storage[patch][offset+i] = values[i];
          }
        }
      }
      template<typename InputSequence>
      void update(const patch_type& patch, const InputSequence& pointSequence, value_type values[]) {
        throw ALE::Exception("Not implemented");
      };
    public:
      //      Reduction types
      //   The overlap types must be defined in terms of the patch_type and index types;
      // they serve as Patch_ and Index_ for the delta CoSieve;
      // there may be a better way than encoding then as nested std::pairs;
      // the int member encodes the rank that contributed the second member of the inner pair (patch or index).
      typedef std::pair<std::pair<patch_type, patch_type>, int> overlap_patch_type;
      typedef std::pair<std::pair<index_type, index_type>, int> overlap_index_type;
      //   The delta CoSieve uses the overlap_patch_type and the overlap_index_type;  it should be impossible to change
      // the structure of a delta CoSieve -- only queries are allowed, so it is a sort of const_CoSieve
      typedef CoSieve<Sieve, overlap_patch_type, overlap_index_type, value_type> delta_type;

      //      Reduction methods (by stage)
      // Compute the overlap patches and indices that determine the structure of the delta CoSieve
      Obj<overlap_patch_type> computeOverlapPatches();  // use support completion on _patches and reorganize (?)
      Obj<overlap_index_type> computeOverlapIndices();  // use cone completion on    _indices and reorganize (?)
      // Compute the delta CoSieve
      Obj<delta_type>           computeDelta();           // compute overlap patches and overlap indices and use them
      // Reduce the CoSieve by computing the delta first and then fixing up the data on the overlap;
      // note: a 'zero' delta corresponds to the identical data attached from either side of each overlap patch
      // To implement a given policy, override this method and utilize the result of computeDelta.
      void                      reduce();
    }; // class CoSifter
  } // namespace def
} // namespace ALE

#endif
