#ifndef included_ALE_CoSifter_hh
#define included_ALE_CoSifter_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif

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
      typedef Source_ source_point;
      typedef Target_ target_point;
      typedef Color_  color_type;
      //    Return types
      class baseSequence;
      class capSequence;
      class coneSequence;
      class supportSequence;
      typedef std::set<source_point, std::less<source_point>, ALE_ALLOCATOR<source_point> > coneSet;
      typedef std::set<target_point, std::less<target_point>, ALE_ALLOCATOR<target_point> > supportSet;
      //    Completion types (some redundant)
      // Color of completions; encodes the color of the completed BiGraph as well as the rank of the process that contributed arrow.
      typedef std::pair<color_type, int> completion_color_type;
      typedef BiGraph<source_point, target_point, completion_color_type> completion_type;
      //     Query methods
      Obj<capSequence>     cap();
      Obj<baseSequence>    base();
      Obj<coneSequence>    cone(const target_point& target);
      Obj<supportSequence> support(const source_point& source);
      Obj<coneSequence>    cone(const target_point& target, const color_type& color);
      Obj<supportSequence> support(const source_point& source, const color_type& color);
      template<class targetInputSequence>
      Obj<coneSet>         cone(const Obj<targetInputSequence>& targets);
      template<class sourceInputSequence>
      Obj<supportSet>      support(const Obj<sourceInputSequence>& sources);
      template<class targetInputSequence>
      Obj<coneSet>         cone(const Obj<targetInputSequence>& targets, const color_type& color);
      template<class sourceInputSequence>
      Obj<supportSet>      support(const Obj<sourceInputSequence>& sources, const color_type& color);
      //     Lattice queries
      template<clase targetInputSequence> 
      Obj<coneSequence> meet(const Obj<targetInputSequence>& targets);
      template<class targetInputSequence> 
      Obj<coneSequence> meet(const Obj<InputSequence>& targets, const color_type& color);
      template<clase sourceInputSequence> 
      Obj<coneSequence> join(const Obj<sourceInputSequence>& sources);
      template<class sourceInputSequence> 
      Obj<coneSequence> meet(const Obj<InputSequence>& sources, const color_type& color);
      //     Manipulation
      void addArrow(const source_point& p, const target_point& q);
      void addArrow(const source_point& p, const target_point& q, const color_type& color);
      template<class sourceInputSequence> 
      void addCone(const Obj<sourceInputSequence >& sources, const target_point& target);
      template<class sourceInputSequence> 
      void addCone(const Obj<sourceInputSequence >& sources, const target_point& target, const color_type& color);
      template<class targetInputSequence> 
      void addSupport(const source_point& source, const Obj<targetInputSequence >& targets);
      template<class targetInputSequence> 
      void addSupport(const source_point& source, const Obj<targetInputSequence >& targets, const color_type& color);
      void add(const Obj<BiGraph<source_point, target_point, color_type> >& bigraph);
      //     Parallelism
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
      Obj<Sieve>     _topology;
    public:
      typedef BiGraph<Sieve::point_type,  patch_type, int> patches_type;
      // Breakdown of the base Sieve into patches; the int color numbers the Sieve::point_type points over a patch_type patch.
    private:
      patches_type   _patches; 
    public:
      // Attachment of fiber dimension intervals to Sieve points; fields encoded by colors, which double as the field ordering index;
      // int color numbers the index_type index (usually an interval) over a Sieve::point_type point.
      typedef BiGraph<index_type, Sieve::point_type,  int> indices_type;
    private:
      indices_type   _indices; 
    private:
      void __clear();
    public:
      //     Topology Manipulation
      void           setTopology(const Obj<Sieve>& topology);
      Obj<Sieve>     getTopology();
      //     Patch manipulation
      // This attaches Sieve::point_type points to a patch in the order prescribed by the sequence; 
      // care must be taken not to assign duplicates (or should it?)
      template<pointInputSequence>
      void                            setPatch(const Obj<pointInputSequence>& points, const patch_type& patch);
      // This retrieves the Sieve::point_type points attached to a given patch
      Obj<patches_type::coneSequence> getPatch(const patch_type& patch);
      //     Index manipulation
      // These attach index_type indices of a given color to a Sieve::point_type point
      template<indexInputSequence>
      void  addIndices(const Obj<indexInputSequence>& indices, indices_type::color_type index_color, const Sieve::point_type& p);
      template<indexInputSequence>
      void  setIndices(const Obj<indexInputSequence>& indices, indices_type::color_type index_color, const Sieve::point_type& p);
      // This retrieves the index_type indices of a given color attached to a Sieve::point_type point
      Obj<indices_type::coneSequence> getIndices(const Sieve::point_type& point);
     
    public:
      //      Reduction types
      //   The overlap types must be defined in terms of the patch_type and index types;
      // they serve as Patch_ and Index_ for the delta CoSieve;
      // there may be a better way than encoding then as nested std::pairs;
      // the int member encodes the rank that contributed the second member of the inner pair (patch or index).
      typedef std::pair<std::pair<patch_type, patch_type> int> overlap_patch_type;
      typedef std::pair<std::pair<index_type, index_type> int> overlap_index_type;
      //   The delta CoSieve uses the overlap_patch_type and the overlap_index_type;  it should be impossible to change
      // the structure of a delta CoSieve -- only queries are allowed, so it is a sort of const_CoSieve
      typedef CoSieve<sieve_type, overlap_patch_type, overlap_index_type> delta_type;

      //      Reduction methods (by stage)
      // Compute the overlap patches and indices that determine the structure of the delta CoSieve
      Obj<overlap_patches_type> computeOverlapPatches();  // use support completion on _patches and reorganize (?)
      Obj<overlap_indices_type> computeOverlapIndices();  // use cone completion on    _indices and reorganize (?)
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
