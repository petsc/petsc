#ifndef included_ALE_CoSifter_hh
#define included_ALE_CoSifter_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif
#include <stack>

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
          virtual bool        operator==(const iterator& itor) const {return this->arrowIter == itor.arrowIter;};
          virtual bool        operator!=(const iterator& itor) const {return this->arrowIter != itor.arrowIter;};
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
      void addArrow(const source_type& p, const target_type& q);
      void addArrow(const source_type& p, const target_type& q, const color_type& color);
      template<class sourceInputSequence> 
      void addCone(const Obj<sourceInputSequence>& sources, const target_type& target);
      template<class sourceInputSequence> 
      void addCone(const Obj<sourceInputSequence>& sources, const target_type& target, const color_type& color);
      template<class targetInputSequence> 
      void addSupport(const source_type& source, const Obj<targetInputSequence >& targets);
      template<class targetInputSequence> 
      void addSupport(const source_type& source, const Obj<targetInputSequence >& targets, const color_type& color);
      void add(const Obj<BiGraph<source_type, target_type, color_type> >& bigraph);
      void replaceSource(const source_type& s, const source_type& new_s) {
        typename ::boost::multi_index::index<ArrowSet,source>::type& index = ::boost::multi_index::get<source>(this->arrows);
        typename ::boost::multi_index::index<ArrowSet,source>::type::iterator i = index.find(s);
        index.replace(i, new_s);
      };
      void replaceTarget(const target_type& t, const target_type& new_t) {
        typename ::boost::multi_index::index<ArrowSet,target>::type& index = ::boost::multi_index::get<target>(this->arrows);
        typename ::boost::multi_index::index<ArrowSet,target>::type::iterator i = index.find(t);
        index.replace(i, new_t);
      };
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
      typedef BiGraph<index_type, typename Sieve::point_type, std::pair<patch_type, int> > indices_type;
    private:
      indices_type _indices; 
    private:
      // Holds values for each patch
      std::set<patch_type, value_type *> _storage;

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
      void           setTopology(const Obj<Sieve>& topology) {this->topology = topology;};
      Obj<Sieve>     getTopology() {return this->topology;};
      //     Patch manipulation
    private:
      template <typename InputSequence>
      void order(InputSequence base, std::map<typename Sieve::point_type, typename Sieve::point_type>& seen, int& offset) {
        // To enable the depth-first order traversal without recursion, we employ a stack.
        std::stack<typename Sieve::point_type> stk;

        // We traverse the sub-bundle over base
        for(typename InputSequence::reverse_iterator b_ritor = base->rbegin(); b_ritor != base->rend(); ++b_ritor) {
          stk.push(*b_ritor);
        }
        while(1) {
          if (stk.empty()) break;

          typename Sieve::point_type p = stk.top(); stk.pop();
          int p_dim = this->getFiberDimension(p);
          int p_off;

          if(seen.find(p) != seen.end()){
            // If p has already been seen, we use the stored offset.
            p_off = seen[p].prefix;
          } else {
            // The offset is only incremented when we encounter a point not yet seen
            p_off   = offset;
            seen[p] = Sieve::point_type(p_off, 0);
            offset += p_dim;
          }

          Obj<typename Sieve::coneSequence> cone = this->getTopology()->cone(p);
          Point_set::iterator      s_itor = base->find(p);
          if (s_itor != base->end()) {
            // If s (aka p) has a nonzero dimension but has not been indexed yet
            if((p_dim > 0) && (seen[p].index == 0)) {
              typename Sieve::point_type newIndex(p_off, p_dim);

              seen[p] = newIndex;
              this->_indices.replaceSource(p, newIndex);
            }
            this->order(cone, seen, offset);
          }

          for(typename Sieve::coneSequence::reverse_iterator p_ritor = cone->rbegin(); p_ritor != cone->rend(); ++p_ritor) {
            stk.push(*p_ritor);
          }
        }
      };

      void allocateAndOrderPatch(const patch_type& patch) {
        std::map<typename Sieve::point_type, typename Sieve::point_type> seen;
        int                                            offset = 0;

        this->order(this->getPatch(patch), seen, offset);

        if (this->_storage.find(patch) != this->_storage.end()) {
          delete [] this->_storage[patch];
        }
        this->_storage[patch] = new value_type[offset];
      };
    public:
      // This attaches Sieve::point_type points to a patch in the order prescribed by the sequence; 
      // care must be taken not to assign duplicates (or should it?)
      template<typename pointInputSequence>
      void                            setPatch(const Obj<pointInputSequence>& points, const patch_type& patch) {
        this->_patches.addCone(points, patch);
        this->allocateAndOrderPatch(patch);
      };
      // This retrieves the Sieve::point_type points attached to a given patch
      Obj<typename patches_type::coneSequence> getPatch(const patch_type& patch) {
        return this->_patches.cone(patch);
      };
      //     Index manipulation
      // These attach index_type indices of a given color to a Sieve::point_type point
      template<typename indexInputSequence>
      void  addIndices(const Obj<indexInputSequence>& indices, typename Sieve::color_type index_color, const typename Sieve::point_type& p) {
        this->_indices.addCone(indices, p);
      };
      template<typename indexInputSequence>
      void  setIndices(const Obj<indexInputSequence>& indices, typename Sieve::color_type index_color, const typename Sieve::point_type& p) {
        this->_indices.setCone(indices, p);
      };
      // This retrieves the index_type indices of a given color attached to a Sieve::point_type point
      Obj<typename indices_type::coneSequence> getIndices(const typename Sieve::point_type& p) {
        return this->_indices.cone(p);
      };
      // Attach indexDim indices to each element of a certain depth in the topology
      void setIndexDimensionByDepth(int depth, int indexDim) {
        this->setIndexDimensionByDepth(depth, Sieve::color_type(), indexDim);
      }
      void setIndexDimensionByDepth(int depth, typename Sieve::color_type index_color, int indexDim) {
        Obj<typename Sieve::depthSequence> stratum = this->getTopology()->depthStratum(depth);

        for(typename Sieve::depthSequence::iterator iter = stratum.begin(); iter != stratum.end(); ++iter) {
          this->addIndices(Sieve::point_type(-1, indexDim), index_color, *iter);
        }
      };
      // Insert values into the specified patch
      void update(const patch_type& patch, const typename Sieve::point_type& p, value_type values[]) {
        Obj<typename indices_type::coneSequence> indices = this->getIndices(p);

        for(typename indices_type::coneSequence::iterator ind = indices->begin(); ind != indices->end(); ++ind) {
          int offset = ind->prefix;

          for(int i = 0; i < ind->index; ++i) {
            this->_storage[offset+i] = values[i];
          }
        }
      }
      template<typename InputSequence>
      void update(const patch_type& patch, const InputSequence& pointSequence, value_type values[]) {
        
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
