#ifndef included_ALE_Delta_hh
#define included_ALE_Delta_hh

#ifndef  included_ALE_BiGraph_hh
#include <BiGraph.hh>
#endif

//
// This file contains classes and methods implementing  the delta operation on a pair of ColorBiGraphs or similar objects.
//
namespace ALE {

  namespace Two {


//       template <typename LeftBiGraph_, typename RightBiGraph_, typename DeltaBiGraph_>
//       class ProductConeFuser {
//       public:
//         //Encapsulated types
//         struct traits {
//           typedef LeftBiGraph_  left_type;
//           typedef RightBiGraph_ right_type;
//           typedef DeltaBiGraph_ delta_type;
//           typedef std::pair<typename left_type::traits::source_type,typename right_type::traits::source_type> source_type;
//           typedef typename left_type::traits::target_type                                                     target_type;
//           typedef std::pair<typename left_type::traits::color_type,typename right_type::traits::color_type>   color_type;
//         };        
//         void
//         fuseCones(const typename traits::left_type::traits::coneSequence&  lcone, 
//                   const typename traits::right_type::traits::coneSequence& rcone, 
//                   typename typename traits::delta_type& delta) {
//           // This Fuser traverses both left cone and right cone, forming an arrow from each pair of arrows -- 
//           // one from each of the cones --  and inserting it into the delta BiGraph.
//           for(typename left_type::traits::coneSequence::iterator lci = lcone.begin(); lci != lcone.end(); lci++) {
//             for(typename left_type::traits::coneSequence::iterator lci = lcone.begin(); lci != lcone.end(); lci++) {
//               delta.addArrow(this->fuseArrows(lci.arrow(), rci.arrow()));
//             }
//           }
//         }
//         typename traits::delta_type::arrow_type
//         fuseArrows(const typename traits::left_type::traits::arrow_type& larrow, 
//                    const typename traits::right_type::traits::arrow_type& rarrow) {
//           return typename traits::arrow_type(traits::source_type(*lci,*rci), lci.target(), 
//                                              typename traits::color_type(lci.color(),rci.color()));
//         }
//       }; // struct ProductConeFuser
      
      
//       template <typename LeftBiGraph_, typename RightBiGraph_, typename DeltaBiGraph_>
//       class Cone {
//         // Cone::operator() in various forms
//         void 
//         operator()(const left_type& l, const right_type& r, delta_type& d, const fuser_type& f = fuser_type()) {
//           // Compute the overlap of left and right bases and then call a 'based' version of the operator
//           operator()(overlap(l,r), l,r,d,f);
//         };
//         void 
//         operator()(const BaseOverlapSequence& overlap,const left_type& l,const right_type& r, delta_type& d, 
//                    const fuser_type& f = fuser_type()) {
//           for(typename BaseOverlapSequence::iterator i = overlap.begin(); i != overlap.end(); i++) {
//             typename left_type::traits::coneSequence lcone = l.cone(*i);
//             typename right_type::traits::coneSequence rcone = r.cone(*i);
//           }
//         }
//         Obj<delta_type> 
//         operator()(const left_type& l, const right_type& r, const fuser_type& f = fuser_type()) {
//           Obj<delta_type> d = delta_type();
//           operator()(l,r,d,f);
//           return d;
//         };
//         Obj<delta_type> 
//         operator()(const BaseOverlapSequence& overlap, const left_type& l, const right_type& r, const fuser_type& f = fuser_type()) {
//           Obj<delta_type> d = delta_type();
//           operator()(overlap,l,r,d,f);
//           return d;
//         };
//       }; // class Cone
    
      template <typename LeftBiGraph_, typename RightBiGraph_>
      class BaseOverlapSequence : public LeftBiGraph_::traits::baseSequence {
        // There is an assumption that LeftBiGraph_ and RightBiGraph_ have equivalent baseSequence types
      public:
        //
        // Encapsulted types
        //
        typedef LeftBiGraph_  left_type;
        typedef RightBiGraph_ right_type;
        typedef typename left_type::traits::baseSequence::traits traits;

        // Overloaded iterator
        class iterator : public traits::iterator {
        };
        //
        // Basic interface
        //
        BaseOverlapSequence(const left_type& l, const right_type& r) : left_type::traits::baseSequence(l.base()), _left(l), _right(r){};

      protected:
        const typename traits::left_type&  _left;
        const typename traits::right_type& _right;

      };// class BaseOverlapSequence


    template <typename ParBiGraph_>
    class ParDelta {
      // Here we specialize to BiGraphs based on Points in order to enable parallel overlap discovery.
      // We also assume that the Points in the base are ordered appropriately so we can use baseSequence.begin() and 
      // baseSequence.end() as the extrema for global reduction.
      typedef ParBiGraph graph_type;
      typedef typename bigraph_type::rebind<ALE::def::Point, unsigned int, unsigned int> overlap_type;

      class Overlap {
      public:
        Overlap(const graph_type& pbgraph) : _graph(pbgraph);
        ~Overlap(){};
      protected:
        _pgraph;
        void __determinePointOwners(const Obj<graph_type>& graph, Obj<overlap_type> overlap) {
          int  size = graph->commSize();
          int  rank = graph->commRank();
          
          // We need to partition global nodes among lessors, which we do by global prefix
          // First we determine the extent of global prefices and the bounds on the indices with each global prefix.
          int minGlobalPrefix = 0;
          // Determine the local extent of global domains
          for(Point_set::iterator point_itor = points.begin(); point_itor != points.end(); point_itor++) {
            Point p = (*point_itor);
            if((p.prefix < 0) && (p.prefix < minGlobalPrefix)) {
              minGlobalPrefix = p.prefix;
            }
          }
          int MinGlobalPrefix;
          ierr = MPI_Allreduce(&minGlobalPrefix, &MinGlobalPrefix, 1, MPIU_INT, MPI_MIN, this->comm); 
          CHKERROR(ierr, "Error in MPI_Allreduce");
          
          int__int BaseLowerBound, BaseUpperBound; // global quantities computed from the local quantities below
          int__int BaseMaxSize;                    // the maximum size of the global base index space by global prefix
          int__int BaseSliceScale, BaseSliceSize, BaseSliceOffset;
          
          if(MinGlobalPrefix < 0) { // if we actually do have global base points
            // Determine the upper and lower bounds on the indices of base points with each global prefix.
            // We use maps to keep track of these quantities with different global prefices.
            int__int baseLowerBound, baseUpperBound; // local quantities
            // Initialize local bound maps with the upper below lower so we can later recognize omitted prefices.
            for(int d = -1; d >= MinGlobalPrefix; d--) {
              baseLowerBound[d] = 0; baseUpperBound[d] = -1;
            }
            // Compute local bounds
            for(Point_set::iterator point_itor = points.begin(); point_itor != points.end(); point_itor++) {
              Point p = (*point_itor);
              int d = p.prefix;
              int i = p.index;
              if(d < 0) { // it is indeed a global prefix
                if (i < baseLowerBound[d]) {
                  baseLowerBound[d] = i;
                }
                if (i > baseUpperBound[d]) {
                  baseUpperBound[d] = i;
                }
              }
            }
            // Compute global bounds
            for(int d = -1; d >= MinGlobalPrefix; d--){
              int lowerBound, upperBound, maxSize;
              ierr   = MPI_Allreduce(&baseLowerBound[d],&lowerBound,1,MPIU_INT,MPI_MIN,this->comm); 
              CHKERROR(ierr, "Error in MPI_Allreduce");
              ierr   = MPI_Allreduce(&baseUpperBound[d],&upperBound,1,MPIU_INT,MPI_MAX,this->comm); 
              CHKERROR(ierr, "Error in MPI_Allreduce");
              maxSize = upperBound - lowerBound + 1;
              if(maxSize > 0) { // there are actually some indices in this global prefix
                BaseLowerBound[d] = lowerBound;
                BaseUpperBound[d] = upperBound;
                BaseMaxSize[d]    = maxSize;
                
                // Each processor (at least potentially) owns a slice of the base indices with each global indices.
                // The size of the slice with global prefix d is BaseMaxSize[d]/size + 1 (except if rank == size-1, 
                // where the slice size can be smaller; +1 is for safety).
                
                // For a non-empty domain d we compute and store the slice size in BaseSliceScale[d] (the 'typical' slice size) and 
                // BaseSliceSize[d] (the 'actual' slice size, which only differs from 'typical' for processor with rank == size -1 ).
                // Likewise, each processor has to keep track of the index offset for each slice it owns and stores it in BaseSliceOffset[d].
                BaseSliceScale[d]  = BaseMaxSize[d]/size + 1;
                BaseSliceSize[d]   = BaseSliceScale[d]; 
                if (rank == size-1) {
                  BaseSliceSize[d] =  BaseMaxSize[d] - BaseSliceScale[d]*(size-1); 
                }
                BaseSliceSize[d]   = PetscMax(1,BaseSliceSize[d]);
                BaseSliceOffset[d] = BaseLowerBound[d] + BaseSliceScale[d]*rank;
              }// for(int d = -1; d >= MinGlobalPrefix; d--){
            }// 
          }// if(MinGlobalDomain < 0) 

          for (Point_set::iterator point_itor = points.begin(); point_itor != points.end(); point_itor++) {
            Point p = (*point_itor);
            // Determine which slice p falls into
            // ASSUMPTION on Point type
            int d = p.prefix;
            int i = p.index;
            int proc;
            if(d < 0) { // global domain -- determine the owner by which slice p falls into
              proc = (i-BaseLowerBound[d])/BaseSliceScale[d];
            }
            else { // local domain -- must refer to a rank within the comm
              if(d >= size) {
                throw ALE::Exception("Local domain outside of comm size");
              }
              proc = d;
            }
            owner[p]     = proc;              
            LeaseData[2*proc+1] = 1;                 // processor owns at least one of ours (i.e., the number of leases from proc is 1)
            LeaseData[2*proc]++;                     // count of how many we lease from proc
          }
        };
      };// ParOverlap

    }; // class ParDelta

  } // namespace Two
    
} // namespace ALE

#endif
