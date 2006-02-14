#ifndef included_ALE_Delta_hh
#define included_ALE_Delta_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif

//
// This file contains classes and methods implementing  the fusion operation on a pair of ColorBiGraphs or similar objects.
//
namespace ALE {

  namespace Two {


      
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
    


//       template <typename LeftBiGraph_, typename RightBiGraph_, typename DeltaBiGraph_>
//       class ConeProductFuser {
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
//       }; // struct ConeProductFuser


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
    

    template <typename LeftBiGraph_, typename RightBiGraph_>
    class RightConeDuplicationFuser {
      // Replicate the cone of the graph on the right in the overlap graph.
    public:
      //Encapsulated types
      struct traits {
        typedef LeftBiGraph_                                                                     left_type;
        typedef RightBiGraph_                                                                    right_type;
        //
        typedef typename right_type::traits::source_type                                         source_type;
        typedef typename right_type::traits::sourceRec_type                                      sourceRec_type;
        typedef typename right_type::traits::target_type                                         target_type;
        typedef typename right_type::traits::targetRec_type                                      targetRec_type;
        typedef typename right_type::traits::color_type                                          color_type;
        // we are using 'rebind' here to illustrate the general point, 
        // even though a simple 'typedef right_type delta_type' would be enough
        typedef typename right_type::template rebind<source_type, sourceRec_type, target_type, targetRec_type, color_type>::type fusion_type;
        typedef typename fusion_type::traits::coneSequence                                       coneSequence;
      };        
      
      static typename traits::coneSequence
      coneFusion(const typename traits::left_type::traits::coneSequence&  lcone, 
             const typename traits::right_type::traits::coneSequence& rcone){
        // Simply return the right cone sequence
        return rcone;
      };
      static void
      fuseCones(const typename traits::left_type::traits::coneSequence&  lcone, 
                const typename traits::right_type::traits::coneSequence& rcone, 
                typename traits::fusion_type& fusion) {
        // This Fuser traverses the right cone and inserts it into the overlap, 
        // duplicating the right graph's cone in the fusion graph.
        for(typename traits::right_type::traits::coneSequence::iterator rci = rcone.begin(); rci != rcone.end(); rci++) {
          fusion.addArrow(rci.arrow());
        }
      }
    }; // struct RightConeDuplicationFuser


    template <typename Source_, typename Target_, typename Color_>
    class TargetArrowArraySequence {
      // TargetArrowArraySequence wraps a raw byte array of (Source_,Color_) pairs
      // presenting it as a sequence of arrows to a fixed target.
    public:
      typedef Arrow<Source_, Target_, Color_> arrow_type;
      typedef Source_ source_type;
      typedef Target_ target_type;
      typedef Color_  color_type;
      //
      struct target_arrow_type { 
        source_type source; color_type color; 
        target_arrow_type(const source_type& s, const color_type& c) : source(s), color(c)  {};
        target_arrow_type(const target_arrow_type& ta) : source(ta.source), color(ta.color) {};
      };
      typedef target_arrow_type* target_arrow_array;
    protected:
      target_type        _target;
      target_arrow_array _arr_ptr;
      size_t             _seq_size;
    public:
      class iterator {
        target_type        _target;
        target_arrow_type* _ptr;
      public:
        iterator(const target_type& target, const target_arrow_array& ptr) : _target(target),     _ptr(ptr)     {};
        iterator(const iterator& it)                                       : _target(it._target), _ptr(it._ptr) {};
        virtual ~iterator() {};
        //
        virtual arrow_type                  operator*() const {
          return arrow_type(this->_ptr->source, this->_target, this->_ptr->color);
        };
        virtual iterator           operator++()      {this->_ptr++; return *this;};
        virtual iterator           operator++(int n) {iterator tmp(this->_target, this->_ptr); this->_ptr++; return tmp;};
        virtual bool               operator!=(const iterator& it) {return ((it._target != this->_target)||(it._ptr != this->_ptr));};
        //
        virtual const source_type& source() const    {return (this->_ptr)->source;};
        virtual const color_type&  color()  const    {return (this->_ptr)->color; };
        virtual const target_type& target() const    {return this->_target;       };
        virtual const arrow_type   arrow()  const    {
          return arrow_type(this->_ptr->source,this->_target,this->_ptr->color);
        };
      }; 
      // Basic interface
      TargetArrowArraySequence(const target_type& target, target_arrow_array arr_ptr, const size_t& seq_size) : 
        _target(target), _arr_ptr(arr_ptr), _seq_size(seq_size) {};
      TargetArrowArraySequence(const TargetArrowArraySequence& seq) :
        _target(seq._target), _arr_ptr(seq._arr_ptr), _seq_size(seq._seq_size) {};
      virtual ~TargetArrowArraySequence() {};
      //
      virtual iterator begin() { return iterator(this->_target, this->_arr_ptr); };
      virtual iterator end()   { return iterator(this->_target, this->_arr_ptr+this->_seq_size); };
      virtual size_t   size()  { return this->_seq_size; };
      virtual bool     empty() { return (this->size() == 0); };
    };// class TargetArrowArraySequence

    template <typename ParBiGraph_, typename Fuser_>
    class ParDelta { // class ParDelta
    public:
      // Here we specialize to BiGraphs based on (capped by) Points in order to enable parallel overlap discovery.
      // We also assume that the Points in the base (cap) are ordered appropriately so we can use baseSequence.begin() and 
      // baseSequence.end() as the extrema for global reduction.
      typedef Fuser_      fuser_type;
      typedef ParBiGraph_ graph_type;
      typedef ColorBiGraph<int, ALE::Two::Rec<int>, ALE::def::Point, ALE::Two::Rec<ALE::def::Point>, ALE::def::Point, uniColor> overlap_type;
      typedef typename fuser_type::traits::fusion_type                      fusion_type;
      //
      ParDelta(Obj<graph_type> graph, int debug = 0) : 
        _graph(graph), comm(_graph->comm()), size(_graph->commSize()), rank(_graph->commRank()), debug(debug) {
        PetscErrorCode ierr = PetscObjectCreate(this->comm, &this->petscObj); CHKERROR(ierr, "Failed in PetscObjectCreate");
      };
      ~ParDelta(){};

      Obj<overlap_type> overlap(){
        Obj<overlap_type> overlap = overlap_type();
        // If this is a serial object, we return an empty overlap
        if((this->comm != PETSC_COMM_SELF) && (this->size > 1)) {
          __determineNeighbors(overlap);
        }
        return overlap;
      };

      Obj<fusion_type> fusion(const Obj<overlap_type>& overlap) {
        Obj<fusion_type> fusion = fusion_type();
        // If this is a serial object, we return an empty delta
        if((this->comm != PETSC_COMM_SELF) && (this->size > 1)) {
          __computeFusion(overlap, fusion);
        }
        return fusion;
      };
    protected:
      // FIX:  need to have _graph of const graph_type& type, but that requires const_cap, const_base etc (see ColorBiGraph)
      Obj<graph_type>           _graph;
      MPI_Comm                  comm;
      int                       size;
      int                       rank;
      int                       debug;
      PetscObject               petscObj;
      // Internal type definitions to ensure compatibility with the legacy code in the parallel subroutines
      typedef ALE::def::Point                                Point;
      typedef int                                            int32_t;
      typedef std::pair<int32_t, int32_t>                    int_pair;
      typedef std::set<std::pair<int32_t, int32_t> >         int_pair_set;
      typedef std::map<int32_t,int32_t>                      int__int;
      typedef std::map<Point, int32_t>                       Point__int;
      typedef std::map<Point, std::pair<int32_t,int32_t> >   Point__int_int;
      typedef std::map<Point, int_pair_set>                  Point__int_pair_set;
      //
      template <typename Sequence>
      void __determinePointOwners(const Obj<Sequence>& points, int32_t *LeaseData, Point__int& owner) {
        PetscErrorCode ierr;
        // The Sequence points will be referred to as 'base' throughout, although it may in fact represent a cap.
        int  size = this->_graph->commSize();
        int  rank = this->_graph->commRank();
        // Make sure the base is not empty 
        if(points->begin() != points->end()) {
          // We need to partition global nodes among lessors, which we do by global prefix
          // First we determine the extent of global prefices and the bounds on the indices with each global prefix.
          int minGlobalPrefix = 0;
          // Determine the local extent of global domains
          for(typename Sequence::iterator point_itor = points->begin(); point_itor != points->end(); point_itor++) {
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
            for(typename Sequence::iterator point_itor = points->begin(); point_itor != points->end(); point_itor++) {
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
          
          for (typename Sequence::iterator point_itor = points->begin(); point_itor != points->end(); point_itor++) {
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
        }// base not empty
      }; // __determinePointOwners()


      //-----------------------------------------------------
      void __determineNeighbors(Obj<overlap_type>& overlap) {

        typedef typename graph_type::baseSequence Sequence;
        PetscErrorCode ierr;
        bool debug = this->debug > 0;
        bool debug2 = this->debug > 1;

        // Allocate space for the ownership data
        int32_t *LeaseData; // 2 ints per processor: number of leased nodes and number of leases (0 or 1).
        ierr = PetscMalloc(2*size*sizeof(PetscInt),&LeaseData);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMemzero(LeaseData,2*size*sizeof(PetscInt));CHKERROR(ierr, "Error in PetscMemzero");
        
        // The base we are going to work with
        Obj<Sequence> points = this->_graph->base();

        // determine owners of each base node and save it in a map
        Point__int owner;
        this->__determinePointOwners(this->_graph->base(), LeaseData, owner);
    
        // Now we accumulate the max lease size and the total number of renters
        // Determine the owners of base nodes and collect the lease data for each processor:
        // the number of nodes leased and the number of leases (0 or 1).
        int32_t MaxLeaseSize, RenterCount;
        ierr = PetscMaxSum(this->comm,LeaseData,&MaxLeaseSize,&RenterCount); 
        CHKERROR(ierr,"Error in PetscMaxSum");
        ierr = PetscInfo1(0,"ParDelta::__determineNeighbors: Number of renters %d\n", RenterCount); 
        CHKERROR(ierr,"Error in PetscInfo");

        if(debug) { /* -------------------------------------------------------------- */
          ierr = PetscSynchronizedPrintf(this->comm, "[%d]: ParDelta::__determineNeighbors: RenterCount = %d, MaxLeaseSize = %d\n", rank, RenterCount, MaxLeaseSize);
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm);
          CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        } /* ----------------------------------------------------------------------- */
        
        // post receives for all Rented nodes; we will be receiving 3 data items per rented node, 
        // and at most MaxLeaseSize of nodes per renter
        PetscMPIInt    tag1;
        ierr = PetscObjectGetNewTag(this->petscObj, &tag1); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        int32_t *RentedNodes;
        MPI_Request *Renter_waits;
        if(RenterCount){
          ierr = PetscMalloc((RenterCount)*(3*MaxLeaseSize+1)*sizeof(int32_t),&RentedNodes);  CHKERROR(ierr,"Error in PetscMalloc");
          ierr = PetscMemzero(RentedNodes,(RenterCount)*(3*MaxLeaseSize+1)*sizeof(int32_t));  CHKERROR(ierr,"Error in PetscMemzero");
          ierr = PetscMalloc((RenterCount)*sizeof(MPI_Request),&Renter_waits);                CHKERROR(ierr,"Error in PetscMalloc");
        }
        for (int32_t i=0; i<RenterCount; i++) {
          ierr = MPI_Irecv(RentedNodes+3*MaxLeaseSize*i,3*MaxLeaseSize,MPIU_INT,MPI_ANY_SOURCE,tag1,this->comm,Renter_waits+i);
          CHKERROR(ierr,"Error in MPI_Irecv");
        }
        
        int32_t LessorCount;
        LessorCount = 0; for (int32_t i=0; i<size; i++) LessorCount += LeaseData[2*i+1];
        ierr = PetscInfo1(0,"ParDelta::__determineNeighbors: Number of lessors %d\n",LessorCount);
        CHKERROR(ierr,"Error in PetscInfo");
        if(debug) { /* -------------------------------------------------------------- */
          ierr = PetscSynchronizedPrintf(this->comm, "[%d]: ParDelta::__determineNeighbors: LessorCount = %d\n", rank, LessorCount);
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm);
          CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        } /* ----------------------------------------------------------------------- */
        
        // We keep only the data about the real lessors -- those that own the nodes we lease
        int32_t *LeaseSizes, *Lessors;
        if(LessorCount) {
          ierr = PetscMalloc(sizeof(int32_t)*(LessorCount), &LeaseSizes); CHKERROR(ierr, "Error in PetscMalloc");
          ierr = PetscMalloc(sizeof(int32_t)*(LessorCount), &Lessors);    CHKERROR(ierr, "Error in PetscMalloc");
        }
        // We also need to compute the inverse to the Lessors array, since we need to be able to convert i into cntr
        // after using the owner array.  We use a map LessorIndex; it is likely to be small -- ASSUMPTION
        int__int LessorIndex;
        // Traverse all processes in ascending order
        int32_t cntr = 0; // keep track of entered records
        for(int32_t i = 0; i < size; i++) {
          if(LeaseData[2*i]) { // if there are nodes leased from process i, record it
            LeaseSizes[cntr] = LeaseData[2*i];
            Lessors[cntr] = i;
            LessorIndex[i] = cntr;
            cntr++;
          }
        }
        ierr = PetscFree(LeaseData); CHKERROR(ierr, "Error in PetscFree");
        if(debug2) { /* ----------------------------------- */
          ostringstream txt;
          txt << "[" << rank << "]: ParDelta::__determineNeighbors: lessor data [index, rank, lease size]: ";
          for(int32_t i = 0; i < LessorCount; i++) {
            txt << "[" << i << ", " << Lessors[i] << ", " << LeaseSizes[i] << "] ";
          }
          txt << "\n";
          ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }/* -----------------------------------  */
        if(debug2) { /* ----------------------------------- */
          ostringstream txt;
          txt << "[" << rank << "]: ParDelta::__determineNeighbors: LessorIndex: ";
          for(int__int::iterator li_itor = LessorIndex.begin(); li_itor!= LessorIndex.end(); li_itor++) {
            int32_t i = (*li_itor).first;
            int32_t j = (*li_itor).second;
            txt << i << "-->" << j << "; ";
          }
          txt << "\n";
          ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }/* -----------------------------------  */
        
        
        // pack messages containing lists of leased base nodes and their cone sizes to the lessors
        int32_t LeasedNodeCount = points->size(); // all points are considered leased from someone
        int32_t *LeasedNodes;
        int32_t *LessorOffsets;
        // We need 3 ints per leased node -- 2 per Point and 1 for the cone size
        if(LeasedNodeCount) {
          ierr       = PetscMalloc((3*LeasedNodeCount)*sizeof(PetscInt),&LeasedNodes); CHKERROR(ierr,"Error in PetscMalloc");
        }
        if(LessorCount) {
          ierr       = PetscMalloc((LessorCount)*sizeof(PetscInt),&LessorOffsets);     CHKERROR(ierr,"Error in PetscMalloc");
          LessorOffsets[0] = 0; 
        }
        for (int32_t i=1; i<LessorCount; i++) { LessorOffsets[i] = LessorOffsets[i-1] + 3*LeaseSizes[i-1];} 
        for (typename Sequence::iterator point_itor = points->begin(); point_itor != points->end(); point_itor++) {
          Point p = (*point_itor);
          int32_t ow = owner[p];
          int32_t ind  = LessorIndex[ow];
          LeasedNodes[LessorOffsets[ind]++] = p.prefix;
          LeasedNodes[LessorOffsets[ind]++] = p.index;      
          LeasedNodes[LessorOffsets[ind]++] = this->_graph->cone(p)->size();
        }
        if(LessorCount) {
          LessorOffsets[0] = 0; 
        }
        for (int32_t i=1; i<LessorCount; i++) { LessorOffsets[i] = LessorOffsets[i-1] + 3*LeaseSizes[i-1];} 
        
        // send the messages to the lessors
        MPI_Request *Lessor_waits;
        if(LessorCount) {
          ierr = PetscMalloc((LessorCount)*sizeof(MPI_Request),&Lessor_waits);CHKERROR(ierr,"Error in PetscMalloc");
        }
        for (int32_t i=0; i<LessorCount; i++) {
          ierr      = MPI_Isend(LeasedNodes+LessorOffsets[i],3*LeaseSizes[i],MPIU_INT,Lessors[i],tag1,this->comm,&Lessor_waits[i]);
          CHKERROR(ierr,"Error in MPI_Isend");
        }
        
        // wait on receive request and prepare to record the identities of the renters responding to the request and their lease sizes
        int__int Renters, RenterLeaseSizes;
        // Prepare to compute the set of renters of each owned node along with the cone sizes held by those renters over the node.
        // Since we don't have a unique ordering on the owned nodes a priori, we will utilize a map.
        Point__int_pair_set NodeRenters;
        cntr  = RenterCount; 
        while (cntr) {
          int32_t arrivalNumber;
          MPI_Status Renter_status;
          ierr = MPI_Waitany(RenterCount,Renter_waits,&arrivalNumber,&Renter_status);  
          CHKMPIERROR(ierr,ERRORMSG("Error in MPI_Waitany"));
          int32_t renter = Renter_status.MPI_SOURCE;
          Renters[arrivalNumber] = renter;
          ierr = MPI_Get_count(&Renter_status,MPIU_INT,&RenterLeaseSizes[arrivalNumber]); CHKERROR(ierr,"Error in MPI_Get_count");
          // Since there are 3 ints per leased node, the lease size is computed by dividing the received count by 3;
          RenterLeaseSizes[arrivalNumber] = RenterLeaseSizes[arrivalNumber]/3;
          // Record the renters for each node
          for (int32_t i=0; i<RenterLeaseSizes[arrivalNumber]; i++) {
            // Compute the offset into the RentedNodes array for the arrived lease.
            int32_t LeaseOffset = arrivalNumber*3*MaxLeaseSize;
            // ASSUMPTION on Point type
            Point node = Point(RentedNodes[LeaseOffset + 3*i], RentedNodes[LeaseOffset + 3*i+1]);
            int32_t coneSize = RentedNodes[LeaseOffset + 3*i + 2];
            NodeRenters[node].insert(int_pair(renter,coneSize));  
          }
          cntr--;
        }
        
        if (debug) { /* -----------------------------------  */
          // We need to collect all the data to be submitted to PetscSynchronizedPrintf
          // We use a C++ string streams for that
          ostringstream txt;
          for (Point__int_pair_set::iterator nodeRenters_itor=NodeRenters.begin();nodeRenters_itor!= NodeRenters.end();nodeRenters_itor++) {
            Point node = (*nodeRenters_itor).first;
            int_pair_set renterSet   = (*nodeRenters_itor).second;
            // ASSUMPTION on point type
            txt << "[" << rank << "]: ParDelta::__determineNeighbors: node (" << node.prefix << "," << node.index << ") is rented by " << renterSet.size() << " renters (renter, cone size):  ";
            for (int_pair_set::iterator renterSet_itor = renterSet.begin(); renterSet_itor != renterSet.end(); renterSet_itor++) 
            {
              txt << "(" << (*renterSet_itor).first << "," << (*renterSet_itor).second << ") ";
            }
            txt << "\n";
          }
          // Now send the C-string behind txt to PetscSynchronizedPrintf
          ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }/* -----------------------------------  */
        
        // wait on the original sends to the lessors
        MPI_Status *Lessor_status;
        if (LessorCount) {
          ierr = PetscMalloc((LessorCount)*sizeof(MPI_Status),&Lessor_status); CHKERROR(ierr,"Error in PetscMalloc");
          ierr = MPI_Waitall(LessorCount,Lessor_waits,Lessor_status);          CHKERROR(ierr,"Error in MPI_Waitall");
        }
        
        
        // Neighbor counts: here the renters receive from the lessors the number of other renters sharing each leased node.
        // Prepare to receive three integers per leased node: two for the node itself and one for the number of neighbors over that node.
        // The buffer has the same structure as LeasedNodes, hence LessorOffsets can be reused.
        // IMPROVE: can probably reduce the message size by a factor of 3 if we assume an ordering on the nodes received from each lessor.
        // ASSUMPTION on Point type
        int32_t *NeighborCounts;
        if(LeasedNodeCount) {
          ierr = PetscMalloc(3*(LeasedNodeCount)*sizeof(PetscInt),&NeighborCounts); CHKERROR(ierr,"Error in PetscMalloc");
        }
        // Post receives for NeighbornCounts
        PetscMPIInt    tag2;
        ierr = PetscObjectGetNewTag(this->petscObj, &tag2); CHKERROR(ierr, "Failded on PetscObjectGetNewTag");
        for (int32_t i=0; i<LessorCount; i++) {
          ierr = MPI_Irecv(NeighborCounts+LessorOffsets[i],3*LeaseSizes[i],MPIU_INT,Lessors[i],tag2,this->comm,&Lessor_waits[i]);
          CHKERROR(ierr,"Error in MPI_Irecv");
        }
        // pack and send messages back to renters; we need to send 3 integers per rental (2 for Point, 1 for sharer count) 
        // grouped by the renter
        // ASSUMPTION on Point type
        // first we compute the total number of rentals
        int32_t TotalRentalCount = 0;
        for(Point__int_pair_set::iterator nodeRenters_itor=NodeRenters.begin();nodeRenters_itor!=NodeRenters.end();nodeRenters_itor++){
          TotalRentalCount += (*nodeRenters_itor).second.size();
        }
        if(debug2) {
          ierr = PetscSynchronizedPrintf(this->comm, "[%d]: TotalRentalCount %d\n", rank, TotalRentalCount); 
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm); CHKERROR(ierr, "PetscSynchronizedFlush");
        }/* -----------------------------------  */
        
        // Allocate sharer counts array for all rentals
        int32_t *SharerCounts;
        if(TotalRentalCount) {
          ierr = PetscMalloc(3*(TotalRentalCount)*sizeof(int32_t),&SharerCounts); CHKERROR(ierr,"Error in PetscMalloc");
        }
        // Renters are traversed in the order of their original arrival index by arrival number a
        int32_t RenterOffset = 0;
        cntr = 0;
        for(int32_t a = 0; a < RenterCount; a++) {
          // traverse the nodes leased by the renter
          int32_t RenterLeaseOffset = a*3*MaxLeaseSize;
          for(int32_t i = 0; i < RenterLeaseSizes[a]; i++) {
            // ASSUMPTION on Point type
            Point node;
            node.prefix = RentedNodes[RenterLeaseOffset + 3*i];
            node.index  = RentedNodes[RenterLeaseOffset + 3*i + 1];
            SharerCounts[cntr++]   = node.prefix;
            SharerCounts[cntr++]   = node.index;        
            // Decrement the sharer count by one not to count the current renter itself (with arrival number a).
            SharerCounts[cntr++] = NodeRenters[node].size()-1;
          }
          // Send message to renter
          ierr      = MPI_Isend(SharerCounts+RenterOffset,3*RenterLeaseSizes[a],MPIU_INT,Renters[a],tag2,this->comm,Renter_waits+a);
          CHKERROR(ierr, "Error in MPI_Isend");
          // Offset is advanced by thrice the number of leased nodes, since we store 3 integers per leased node: Point and cone size
          RenterOffset = cntr;
        }
        // Wait on receives from lessors with the neighbor counts
        if (LessorCount) {
          ierr = MPI_Waitall(LessorCount,Lessor_waits,Lessor_status); CHKERROR(ierr,"Error in MPI_Waitall");
        }
        // Wait on the original sends to the renters
        MPI_Status *Renter_status;
        ierr = PetscMalloc((RenterCount)*sizeof(MPI_Status),&Renter_status);CHKERROR(ierr,"Error in PetscMalloc");
        if(RenterCount) {
          ierr = MPI_Waitall(RenterCount, Renter_waits, Renter_status);CHKERROR(ierr,"Error in MPI_Waitall");
        }
        
        if (debug) { /* -----------------------------------  */
          // Use a C++ string stream to report the numbers of shared nodes leased from each lessor
          ostringstream txt;
          cntr = 0;
          txt << "[" << rank << "]: ParDelta::__determineNeighbors: neighbor counts by lessor-node [lessor rank, (node), neighbor count]:  ";
          for(int32_t i = 0; i < LessorCount; i++) {
            // ASSUMPTION on point type
            for(int32_t j = 0; j < LeaseSizes[i]; j++) 
            {
              int32_t prefix, index, sharerCount;
              prefix      = NeighborCounts[cntr++];
              index       = NeighborCounts[cntr++];
              sharerCount = NeighborCounts[cntr++];
              txt << "[" << Lessors[i] <<", (" << prefix << "," << index << "), " << sharerCount << "] ";
            }
          }
          txt << "\n";
          ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm); CHKERROR(ierr, "PetscSynchronizedFlush");
        }/* -----------------------------------  */
        
        
        // Now we allocate an array to receive the neighbor ranks and the remote cone sizes for each leased node,
        // hence, the total array size is 2*TotalNeighborCount.
        // Note that the lessor offsets must be recalculated, since they are no longer based on the number of nodes 
        // leased from that lessor, but on the number of neighbor over the nodes leased from that lessor.
        
        // First we compute the numbers of neighbors over the nodes leased from a given lessor.
        int32_t TotalNeighborCount = 0;
        int32_t *NeighborCountsByLessor;
        if(LessorCount) {
          ierr = PetscMalloc((LessorCount)*sizeof(int32_t), &NeighborCountsByLessor); CHKERROR(ierr, "Error in PetscMalloc");
        }
        cntr = 0;
        for(int32_t i = 0; i < LessorCount; i++) {
          for(int32_t j = 0; j < LeaseSizes[i]; j++) {
            //ASSUMPTION on Point type affects NeighborCountsOffset size
            cntr += 2;
            TotalNeighborCount += NeighborCounts[cntr++];
          }
          if(i == 0) {
            NeighborCountsByLessor[i] = TotalNeighborCount;
          }
          else {
            NeighborCountsByLessor[i] = TotalNeighborCount - NeighborCountsByLessor[i-1];
          }
        }
        if (debug2) { /* -----------------------------------  */
          // Use a C++ string stream to report the numbers of shared nodes leased from each lessor
          ostringstream txt;
          cntr = 0;
          txt << "[" << rank << "]: ParDelta::__determineNeighbors: NeighborCountsByLessor [rank, count]:  ";
          for(int32_t i = 0; i < LessorCount; i++) {
            txt << "[" << Lessors[i] <<","  <<  NeighborCountsByLessor[i] << "]; ";
          }
          txt << "\n";
          ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm); CHKERROR(ierr, "PetscSynchronizedFlush");
        }/* -----------------------------------  */
        int32_t *Neighbors = 0;
        if(TotalNeighborCount) {
          ierr = PetscMalloc((2*TotalNeighborCount)*sizeof(int32_t),&Neighbors); CHKERROR(ierr,"Error in PetscMalloc");
        }
        
        // Post receives for Neighbors
        PetscMPIInt    tag3;
        ierr = PetscObjectGetNewTag(this->petscObj, &tag3); CHKERROR(ierr, "Failded on PetscObjectGetNewTag");
        int32_t lessorOffset = 0;
        for(int32_t i=0; i<LessorCount; i++) {
          if(NeighborCountsByLessor[i]) { // We expect messages from lessors with a non-zero NeighborCountsByLessor entry only
            ierr = MPI_Irecv(Neighbors+lessorOffset,2*NeighborCountsByLessor[i],MPIU_INT,Lessors[i],tag3,this->comm,&Lessor_waits[i]);
            CHKERROR(ierr,"Error in MPI_Irecv");
            lessorOffset += 2*NeighborCountsByLessor[i];
          }
        }
        // Pack and send messages back to renters. 
        // For each node p and each renter r (hence for each rental (p,r)) we must send to r a segment consisting of the list of all
        // (rr,cc) such that (p,rr) is a share and cc is the cone size over p at rr.
        // ALTERNATIVE, SCALABILITY: 
        //             1. allocate an array capable of holding all messages to all renters and send one message per renter (more memory)
        //             2. allocate an array capable of holding all rentals for all nodes and send one message per share (more messages).
        // Here we choose 1 since we assume that the memory requirement is modest and the communication time is much more expensive, 
        // however, this is likely to be application-dependent, and a switch should be introduced to change this behavior at will.
        // The rental segments are grouped by the renter recepient and within the renter by the node in the same order as SharerCounts.
        
        // We need to compute the send buffer size using the SharerCounts array.
        // Traverse the renters in order of their original arrival, indexed by the arrival number a, and then by the nodes leased by a.
        // Add up all entries equal to 2 mod 3 in SharerCounts (0 & 1 mod 3 are node IDs, ASSUMPTION on Point type) and double that number 
        // to account for sharer ranks AND the cone sizes we are sending.
        int32_t SharersSize = 0; // 'Sharers' buffer size
        cntr = 0;
        for(int32_t a = 0; a < RenterCount; a++) {
          // traverse the number of nodes leased by the renter
          for(int32_t i = 0; i < RenterLeaseSizes[a]; i++) {
            SharersSize += SharerCounts[3*cntr+2];
            cntr++;
          }
        }
        SharersSize *= 2;
        // Allocate the Sharers array
        int32_t *Sharers;
        if(SharersSize) {
          ierr = PetscMalloc(SharersSize*sizeof(int32_t),&Sharers); CHKERROR(ierr,"Error in PetscMalloc");
        }
        // Now pack the messages and send them off.
        // Renters are traversed in the order of their original arrival index by arrival number a
        ostringstream txt; // DEBUG
        if(debug2) {
          txt << "[" << rank << "]: ParDelta::__determineNeighbors: RenterCount = " << RenterCount << "\n";
        }
        RenterOffset = 0; // this is the current offset into Sharers needed for the send statement
        for(int32_t a = 0; a < RenterCount; a++) {//
          int32_t r = Renters[a];
          int32_t RenterLeaseOffset = a*3*MaxLeaseSize;
          int32_t SegmentSize = 0;
          // traverse the nodes leased by the renter
          for(int32_t i = 0; i < RenterLeaseSizes[a]; i++) {
            // Get a node p rented to r
            // ASSUMPTION on Point type
            Point p;
            p.prefix = RentedNodes[RenterLeaseOffset + 3*i];
            p.index  = RentedNodes[RenterLeaseOffset + 3*i + 1];
            if(debug) {
              txt << "[" << rank << "]: ParDelta::__determineNeighbors: renters sharing with " << r << " of node  (" << p.prefix << "," << p.index << ")  [rank, cone size]:  ";
            }
            // now traverse the set of all the renters of p
            for(int_pair_set::iterator pRenters_itor=NodeRenters[p].begin(); pRenters_itor!=NodeRenters[p].end(); pRenters_itor++) {
              int32_t rr = (*pRenters_itor).first;  // rank of a pRenter 
              int32_t cc = (*pRenters_itor).second; // cone size over p at rr
              // skip r itself
              if(rr != r){
                Sharers[RenterOffset+SegmentSize++] = rr;
                Sharers[RenterOffset+SegmentSize++] = cc;
                if(debug) {
                  txt << "[" << rr << ","  << cc << "]; ";
                }
              }
            }// for(int_pair_set::iterator pRenters_itor=NodeRenters[p].begin(); pRenters_itor!=NodeRenters[p].end(); pRenters_itor++) {
            if(debug) {
              txt << "\n";
            }
          }// for(int32_t i = 0; i < RenterLeaseSizes[a]; i++) {
          // Send message to renter only if the segment size is positive
          if(SegmentSize > 0) {
            ierr      = MPI_Isend(Sharers+RenterOffset,SegmentSize,MPIU_INT,Renters[a],tag3,this->comm,Renter_waits+a);
            CHKERROR(ierr, "Error in MPI_Isend");
          }
          // Offset is advanced by the segmentSize
          RenterOffset += SegmentSize;
        }//  for(int32_t a = 0; a < RenterCount; a++) {
        if(debug) {
          ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm); CHKERROR(ierr, "PetscSynchronizedFlush");      
        }
        
        // Wait on receives from lessors with the neighbor counts
        if (LessorCount) {
          ierr = MPI_Waitall(LessorCount,Lessor_waits,Lessor_status);CHKERROR(ierr,"Error in MPI_Waitall");
        }
        if (debug) { /* -----------------------------------  */
          // To report the neighbors at each lessor we use C++ a string stream
          ostringstream txt;
          int32_t cntr1 = 0;
          int32_t cntr2 = 0;
          for(int32_t i = 0; i < LessorCount; i++) {
            // ASSUMPTION on point type
            txt << "[" <<rank<< "]: ParDelta::__determineNeighbors: neighbors over nodes leased from " <<Lessors[i]<< ":\n";
            int32_t activeLessor = 0;
            for(int32_t j = 0; j < LeaseSizes[i]; j++) 
            {
              int32_t prefix, index, sharerCount;
              prefix = NeighborCounts[cntr1++];
              index = NeighborCounts[cntr1++];
              sharerCount = NeighborCounts[cntr1++];
              if(sharerCount > 0) {
                txt <<"[" << rank << "]:\t(" << prefix <<","<<index<<"):  [rank, coneSize]: ";
                activeLessor++;
              }
              for(int32_t k = 0; k < sharerCount; k++) {
                int32_t sharer = Neighbors[cntr2++];
                int32_t coneSize = Neighbors[cntr2++];
                txt << "[" <<sharer <<", "<< coneSize << "] ";
              }
            }// for(int32_t j = 0; j < LeaseSizes[i]; j++) 
            if(!activeLessor) {
              txt <<"[" << rank << "]:\tnone";
            }
            txt << "\n";
          }// for(int32_t i = 0; i < LessorCount; i++)
          ierr = PetscSynchronizedPrintf(this->comm,txt.str().c_str());CHKERROR(ierr,"Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }/* -----------------------------------  */
        
        // This concludes the interaction of lessors and renters, and the exchange is completed by a peer-to-peer neighbor cone swap
        // (except we still have to wait on our last sends to the renters -- see below).
        // However, we don't free all of the arrays associated with the lessor-renter exchanges, since some of the data
        // still use those structures.  Here are the arrays we can get rid of:
        if(RenterCount) {
          ierr = PetscFree(RentedNodes);  CHKERROR(ierr, "Error in PetscFree");
        }
        if(SharersSize) {ierr = PetscFree(Sharers); CHKERROR(ierr, "Error in PetscFree");}
        if(LessorCount) {
          ierr = PetscFree(NeighborCountsByLessor); CHKERROR(ierr, "Error in PetscFree");
          ierr = PetscFree(Lessor_status);          CHKERROR(ierr,"Error in PetscFree");
          ierr = PetscFree(Lessor_waits);           CHKERROR(ierr,"Error in PetscFree");
          ierr = PetscFree(LessorOffsets);          CHKERROR(ierr,"Error in PetscFree");
          ierr = PetscFree(LeaseSizes);             CHKERROR(ierr,"Error in PetscFree");
          ierr = PetscFree(Lessors);                CHKERROR(ierr,"Error in PetscFree");
        }
        if(LeasedNodeCount) {
          ierr = PetscFree(LeasedNodes); CHKERROR(ierr,"Error in PetscFree");
        }
        
        // First we determine the neighbors and the cones over each node to be received from or sent to each neigbor.
        std::map<int32_t, Point__int > NeighborPointConeSizeIn, NeighborPointConeSizeOut;
        // Traverse Neighbors and separate the data by neighbor over each point: NeighborPointConeSizeIn/Out.
        // cntr keeps track of the current position within the Neighbors array, node boundaries are delineated using NeighborCounts.
        // ASSUMPTION: 'Neighbors' stores node renter segments in the same order as NeighborCounts stores the node data.
        cntr = 0;
        for(int32_t i = 0; i < LeasedNodeCount; i++) {
          // ASSUMPTION on Point type
          Point p;
          p.prefix = NeighborCounts[3*i];
          p.index  = NeighborCounts[3*i+1]; 
          int32_t pNeighborsCount = NeighborCounts[3*i+2]; // recall that NeighborCounts lists the number of neighbors after each node
          // extract the renters of p from Neighbors
          for(int32_t j = 0; j < pNeighborsCount; j++) {
            int32_t neighbor = Neighbors[cntr++];
            int32_t coneSize = Neighbors[cntr++];
            // Record the size of the cone over p coming in from neighbor
            NeighborPointConeSizeIn[neighbor][p]= coneSize;
            // Record the size of the cone over p going out to neighbor
            NeighborPointConeSizeOut[neighbor][p] = this->_graph->cone(p)->size();
            overlap->addArrow(neighbor, p, Point(coneSize, this->_graph->cone(p)->size())); 
          }
        }// for(int32_t i = 0; i < LeasedNodeCount; i++)
        
        
        // Compute total incoming cone sizes by neighbor and the total incomping cone size.
        // Also count the total number of neighbors we will be communicating with
        int32_t  NeighborCount = 0;
        int__int NeighborConeSizeIn;
        int32_t  ConeSizeIn = 0;
        ostringstream txt3;
        // Traverse all of the neighbors  from whom we will be receiving cones.
        for(std::map<int32_t, Point__int>::iterator np_itor = NeighborPointConeSizeIn.begin();np_itor!=NeighborPointConeSizeIn.end(); np_itor++){
          int32_t neighbor = (*np_itor).first;
          NeighborConeSizeIn[neighbor] = 0;
          if(debug2) {
            txt3 << "[" << rank << "]: " << "ParDelta::__determineNeighbors: NeighborPointConeSizeIn[" << neighbor << "]: ";
          }
          // Traverse all the points the cones over which we are receiving and add the cone sizes
          for(Point__int::iterator pConeSize_itor = (*np_itor).second.begin(); pConeSize_itor != (*np_itor).second.end(); pConeSize_itor++){
            NeighborConeSizeIn[neighbor] = NeighborConeSizeIn[neighbor] + (*pConeSize_itor).second;
            txt3 << "(" << (*pConeSize_itor).first.prefix << "," << (*pConeSize_itor).first.index << ")" << "->" << (*pConeSize_itor).second << "; ";
          }
          // Accumulate the total cone size
          ConeSizeIn += NeighborConeSizeIn[neighbor];
          NeighborCount++;
          txt3 << "NeighborConeSizeIn[" << neighbor << "]: " << NeighborConeSizeIn[neighbor] << "\n";
        }//for(std::map<int32_t, Point__int>::iterator np_itor=NeighborPointConeSizeIn.begin();np_itor!=NeighborPointConeSizeIn.end(); np_itor++)
        if(debug2) {
          ierr = PetscSynchronizedPrintf(this->comm,txt3.str().c_str());CHKERROR(ierr,"Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }
        if(debug) {/* --------------------------------------------------------------------------------------------- */
          ostringstream txt;
          txt << "[" << rank << "]: ParDelta::__determineNeighbors: total size of incoming cone: " << ConeSizeIn << "\n";
          for(int__int::iterator np_itor = NeighborConeSizeIn.begin();np_itor!=NeighborConeSizeIn.end();np_itor++)
          {
            int32_t neighbor = (*np_itor).first;
            int32_t coneSize = (*np_itor).second;
            txt << "[" << rank << "]: ParDelta::__determineNeighbors: size of cone from " << neighbor << ": " << coneSize << "\n";
            
          }//int__int::iterator np_itor=NeighborConeSizeIn.begin();np_itor!=NeighborConeSizeIn.end();np_itor++)
          ierr = PetscSynchronizedPrintf(this->comm, txt.str().c_str());
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(this->comm);
          CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        }/* --------------------------------------------------------------------------------------------- */
        
        // Wait on the original sends to the renters (the last vestige of the lessor-renter exchange epoch; we delayed it to afford the
        // greatest opportunity for a communication-computation overlap).
        if(RenterCount) {
          ierr = MPI_Waitall(RenterCount, Renter_waits, Renter_status); CHKERROR(ierr,"Error in MPI_Waitall");
        }
        if(RenterCount) {
          ierr = PetscFree(Renter_waits); CHKERROR(ierr, "Error in PetscFree");
          ierr = PetscFree(Renter_status); CHKERROR(ierr, "Error in PetscFree");
        }        
      };// __determinePointOwners()

      
      void __computeFusion(const Obj<overlap_type>& overlap, Obj<fusion_type> fusion) {
        
      };

    }; // class ParDelta

  } // namespace Two
    
} // namespace ALE

#endif
