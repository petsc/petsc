#ifndef included_ALE_ParDelta_hh
#define included_ALE_ParDelta_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif



//
// Classes and methods implementing  the parallel Overlap and Fusion algorithms on ASifter-like objects.
//
namespace ALE {

    template <typename RightConeSequence_>
    class RightSequenceDuplicator {
      // Replicate the cone sequence on the right in the overlap graph.
      int debug;
    public:
      //Encapsulated types
      typedef RightConeSequence_                            right_sequence_type;
      typedef typename right_sequence_type::target_type     right_target_type;
      //
      typedef typename right_sequence_type::source_type     fusion_source_type;   
      typedef typename right_sequence_type::target_type     fusion_target_type;   
      typedef typename right_sequence_type::color_type      fusion_color_type;   
    public:
      //
      // Basic interface
      //
      RightSequenceDuplicator(int debug = 0) : debug(debug) {};
      RightSequenceDuplicator(const RightSequenceDuplicator& f) {};
      virtual ~RightSequenceDuplicator() {};

      template <typename left_target_type>
      fusion_target_type
      fuseBasePoints(const left_target_type&  ltarget, const right_target_type& rtarget) {
        return rtarget;
      };

      // FIX: need to have const left_sequence& and const right_sequence& , but begin() and end() aren't const methods 
      template <typename left_sequence_type, typename fusion_sequence_type>
      void
      fuseCones(left_sequence_type&  lcone, right_sequence_type& rcone, const Obj<fusion_sequence_type>& fcone) {
        for(typename right_sequence_type::iterator rci = rcone.begin(); rci != rcone.end(); rci++) {
          fcone->addArrow(rci.arrow());
        }
      };
    }; // struct RightSequenceDuplicator


    template <typename Arrow_>
    class ConeArraySequence {
      // ConeArraySequence wraps a raw byte array of (Source_,Color_) pairs
      // presenting it as a cone sequence for a given target.
    public:
      typedef Arrow_                           arrow_type;
      typedef typename arrow_type::source_type source_type;
      typedef typename arrow_type::target_type target_type;
      typedef typename arrow_type::color_type  color_type;
      //
      struct cone_arrow_type { 
        source_type source; 
        color_type color; 
        //
        cone_arrow_type(const arrow_type&  a) : source(a.source), color(a.color)          {};
        cone_arrow_type(const source_type& s, const color_type& c) : source(s), color(c)  {};
        cone_arrow_type(const cone_arrow_type& ca) : source(ca.source), color(ca.color)   {};
        // 
        static void place(cone_arrow_type* ca_ptr, const arrow_type& a) {
          // WARNING: an unsafe method in that it has no way of checking the validity of ca_ptr
          ca_ptr->source = a.source;
          ca_ptr->color  = a.color;
        };
        static void place(cone_arrow_type* ca_ptr, const source_type& s, const color_type& c) {
          // WARNING: an unsafe method in that it has no way of checking the validity of ca_ptr
          ca_ptr->source = s;
          ca_ptr->color  = c;
        };
      };
    protected:
      typedef cone_arrow_type* cone_arrow_array;
      target_type        _target;
      cone_arrow_array   _arr_ptr;
      size_t             _seq_size;
    public:
      class iterator {
        target_type        _target;
        cone_arrow_type*   _ptr;
      public:
        iterator(const target_type& target, const cone_arrow_array& ptr) : _target(target),     _ptr(ptr)     {};
        iterator(const iterator& it)                                       : _target(it._target), _ptr(it._ptr) {};
        virtual ~iterator() {};
        //
        virtual source_type        operator*() const { return this->_ptr->source;};
        virtual iterator           operator++()      {this->_ptr++; return *this;};
        virtual iterator           operator++(int n) {iterator tmp(this->_target, this->_ptr); this->_ptr++; return tmp;};
        virtual bool               operator!=(const iterator& it) {return ((it._target != this->_target)||(it._ptr != this->_ptr));};
        //
        virtual const source_type& source() const    {return this->_ptr->source;};
        virtual const color_type&  color()  const    {return this->_ptr->color; };
        virtual const target_type& target() const    {return this->_target;     };
        virtual const arrow_type   arrow()  const    {
          return arrow_type(this->_ptr->source,this->_target,this->_ptr->color);
        };
      }; 
      // Basic interface
      ConeArraySequence(cone_arrow_array arr_ptr, const size_t& seq_size, const target_type& target) : 
        _target(target), _arr_ptr(arr_ptr), _seq_size(seq_size) {};
      ConeArraySequence(const ConeArraySequence& seq) :
        _target(seq._target), _arr_ptr(seq._arr_ptr), _seq_size(seq._seq_size) {};
      virtual ~ConeArraySequence() {};
      //
      virtual iterator begin() { return iterator(this->_target, this->_arr_ptr); };
      virtual iterator end()   { return iterator(this->_target, this->_arr_ptr+this->_seq_size); };
      virtual size_t   size()  { return this->_seq_size; };
      virtual bool     empty() { return (this->size() == 0); };

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
    };// class ConeArraySequence


    template <typename ParSifter_,
              typename Fuser_ = RightSequenceDuplicator<ConeArraySequence<typename ParSifter_::traits::arrow_type> >,
              typename FusionSifter_ = typename ParSifter_::template rebind<typename Fuser_::fusion_source_type, 
                                                                              typename Fuser_::fusion_target_type, 
                                                                              typename Fuser_::fusion_color_type>::type
    >    
    class ParConeDelta { // class ParConeDelta
    public:
      // Here we specialize to Sifters based on Points in order to enable parallel overlap discovery.
      // We also assume that the Points in the base are ordered appropriately so we can use baseSequence.begin() and 
      // baseSequence.end() as the extrema for global reduction.
      typedef ParConeDelta<ParSifter_, Fuser_, FusionSifter_>                                   delta_type;
      typedef ParSifter_                                                                        graph_type;
      typedef Fuser_                                                                            fuser_type;
      // These are default "return" types, although methods are templated on their main input/return types
      typedef ASifter<int, ALE::Point, ALE::pair<ALE::Point, ALE::pair<int,int> >, uniColor>    overlap_type;
      typedef ASifter<int, ALE::pair<int,ALE::Point>, ALE::pair<ALE::Point, ALE::pair<int,int> >, uniColor>    bioverlap_type;
      typedef FusionSifter_                                                                     fusion_type;

      //
      static Obj<overlap_type> 
      overlap(const Obj<graph_type> graph) {
        Obj<overlap_type> overlap = overlap_type(graph->comm());
        // If this is a serial object, we return an empty overlap
        if((graph->comm() != PETSC_COMM_SELF) && (graph->commSize() > 1)) {
          computeOverlap(graph, overlap);
        }
        return overlap;
      };

      template <typename Overlap_>
      static void computeOverlap(const Obj<graph_type>& graph, Obj<Overlap_>& overlap){
        __computeOverlapNew(graph, overlap);
      };

      static Obj<bioverlap_type> 
      overlap(const Obj<graph_type> graphA, const Obj<graph_type> graphB) {
        Obj<bioverlap_type> overlap = bioverlap_type(graphA->comm());
        PetscMPIInt         comp;

        MPI_Comm_compare(graphA->comm(), graphB->comm(), &comp);
        if (comp != MPI_IDENT) {
          throw ALE::Exception("Non-matching communicators for overlap");
        }
        computeOverlap(graphA, graphB, overlap);
        return overlap;
      };

      template <typename Overlap_>
      static void computeOverlap(const Obj<graph_type>& graphA, const Obj<graph_type>& graphB, Obj<Overlap_>& overlap){
        __computeOverlapNew(graphA, graphB, overlap);
      };

      template <typename Overlap_>
      static Obj<fusion_type> 
      fusion(const Obj<graph_type>& graph, const Obj<Overlap_>& overlap, const Obj<fuser_type>& fuser = fuser_type()) {
        Obj<fusion_type> fusion = fusion_type(graph->comm());
        // If this is a serial object, we return an empty delta
        if((graph->comm() != PETSC_COMM_SELF) && (graph->commSize() > 1)) {
          computeFusion(graph, overlap, fusion, fuser);
        }
        return fusion;
      };

      template <typename Overlap_, typename Fusion_>
      static void computeFusion(const Obj<graph_type>& graph, const Obj<Overlap_>& overlap, Obj<Fusion_>& fusion, const Obj<fuser_type>& fuser = fuser_type()){
        __computeFusionNew(graph, overlap, fusion, fuser);
      };

      template <typename Overlap_>
      static Obj<fusion_type> 
      fusion(const Obj<graph_type>& graphA, const Obj<graph_type>& graphB, const Obj<Overlap_>& overlap, const Obj<fuser_type>& fuser = fuser_type()) {
        Obj<fusion_type> fusion = fusion_type(graphA->comm());
        PetscMPIInt       comp;

        MPI_Comm_compare(graphA->comm(), graphB->comm(), &comp);
        if (comp != MPI_IDENT) {
          throw ALE::Exception("Non-matching communicators for overlap");
        }
        computeFusion(graphA, graphB, overlap, fusion, fuser);
        return fusion;
      };

      template <typename Overlap_, typename Fusion_>
      static void computeFusion(const Obj<graph_type>& graphA, const Obj<graph_type>& graphB, const Obj<Overlap_>& overlap, Obj<Fusion_>& fusion, const Obj<fuser_type>& fuser = fuser_type()){
        PetscMPIInt       comp;

        MPI_Comm_compare(graphA->comm(), graphB->comm(), &comp);
        if (comp != MPI_IDENT) {
          throw ALE::Exception("Non-matching communicators for overlap");
        }
        __computeFusionNew(graphA, graphB, overlap, fusion, fuser);
      };

    protected:
      static int                debug;
      // Internal type definitions to ensure compatibility with the legacy code in the parallel subroutines
      typedef ALE::Point                                Point;
      typedef int                                            int32_t;
      typedef std::pair<int32_t, int32_t>                    int_pair;
      typedef std::set<std::pair<int32_t, int32_t> >         int_pair_set;
      typedef std::map<int32_t,int32_t>                      int__int;
      typedef std::map<Point, int32_t>                       Point__int;
      typedef std::map<Point, std::pair<int32_t,int32_t> >   Point__int_int;
      typedef std::map<Point, int_pair_set>                  Point__int_pair_set;

    protected:
      //--------------------------------------------------------------------------------------------------------
      template <typename Sequence>
      static void __determinePointOwners(const Obj<graph_type> _graph, const Obj<Sequence>& points, int32_t *LeaseData, Point__int& owner) {
        PetscErrorCode ierr;
        // The Sequence points will be referred to as 'base' throughout, although it may in fact represent a cap.
        MPI_Comm comm = _graph->comm();
        int  size     = _graph->commSize();
        int  rank     = _graph->commRank();

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
        ierr = MPI_Allreduce(&minGlobalPrefix, &MinGlobalPrefix, 1, MPIU_INT, MPI_MIN, comm); 
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
            ierr   = MPI_Allreduce(&baseLowerBound[d],&lowerBound,1,MPIU_INT,MPI_MIN,comm); 
            CHKERROR(ierr, "Error in MPI_Allreduce");
            ierr   = MPI_Allreduce(&baseUpperBound[d],&upperBound,1,MPIU_INT,MPI_MAX,comm); 
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
          }
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

        // Base was empty 
        if(points->begin() == points->end()) {
          for(int p = 0; p < size; p++) {
            LeaseData[2*p+0] = 0;
            LeaseData[2*p+1] = 0;
          }
        }
      }; // __determinePointOwners()


      //-------------------------------------------------------------------------------------------------------
      #undef  __FUNCT__
      #define __FUNCT__ "__computeOverlapNew"
      template <typename Overlap_>
      static void __computeOverlapNew(const Obj<graph_type>& _graph, Obj<Overlap_>& overlap) {
        typedef typename graph_type::traits::baseSequence Sequence;
        MPI_Comm       comm = _graph->comm();
        int            size = _graph->commSize();
        int            rank = _graph->commRank();
        PetscObject    petscObj = _graph->petscObj();
        PetscMPIInt    tag1, tag2, tag3;
        PetscErrorCode ierr;
        // The base we are going to work with
        Obj<Sequence> points = _graph->base();
        // 2 ints per processor: number of points we buy and number of sales (0 or 1).
        int *BuyData;
        ierr = PetscMalloc(2*size * sizeof(int), &BuyData);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMemzero(BuyData, 2*size * sizeof(int));CHKERROR(ierr, "Error in PetscMemzero");
        // Map from points to the process managing its bin (seller)
        Point__int owner;

        // determine owners of each base node and save it in a map
        __determinePointOwners(_graph, points, BuyData, owner);

        int  msgSize = 3;  // A point is 2 ints, and the cone size is 1
        int  BuyCount = 0; // The number of sellers with which this process (buyer) communicates
        int *BuySizes;     // The number of points to buy from each seller
        int *Sellers;      // The process for each seller
        int *offsets = new int[size];
        for(int p = 0; p < size; ++p) {BuyCount += BuyData[2*p+1];}
        ierr = PetscMalloc2(BuyCount,int,&BuySizes,BuyCount,int,&Sellers);CHKERROR(ierr, "Error in PetscMalloc");
        for(int p = 0, buyNum = 0; p < size; ++p) {
          if (BuyData[2*p]) {
            Sellers[buyNum]    = p;
            BuySizes[buyNum++] = BuyData[2*p];
          }
          if (p == 0) {
            offsets[p] = 0;
          } else {
            offsets[p] = offsets[p-1] + msgSize*BuyData[2*(p-1)];
          }
        }

        // All points are bought from someone
        int32_t *BuyPoints;
        ierr = PetscMalloc(msgSize*points->size() *sizeof(int32_t),&BuyPoints);CHKERROR(ierr,"Error in PetscMalloc");
        for (typename Sequence::iterator p_itor = points->begin(); p_itor != points->end(); p_itor++) {
          BuyPoints[offsets[owner[*p_itor]]++] = (*p_itor).prefix;
          BuyPoints[offsets[owner[*p_itor]]++] = (*p_itor).index;      
          BuyPoints[offsets[owner[*p_itor]]++] = _graph->cone(*p_itor)->size();
        }
        for(int b = 0, o = 0; b < BuyCount; ++b) {
          if (offsets[Sellers[b]] - o != msgSize*BuySizes[b]) {
            throw ALE::Exception("Invalid point size");
          }
          o += msgSize*BuySizes[b];
        }
        delete [] offsets;

        int  SellCount;      // The number of buyers with which this process (seller) communicates
        int *SellSizes;      // The number of points to sell to each buyer
        int *Buyers;         // The process for each buyer
        int  MaxSellSize;    // The maximum number of messages to be sold to any buyer
        int32_t *SellPoints = PETSC_NULL; // The points and cone sizes from all buyers
        ierr = PetscMaxSum(comm, BuyData, &MaxSellSize, &SellCount);CHKERROR(ierr,"Error in PetscMaxSum");
        ierr = PetscMalloc2(SellCount,int,&SellSizes,SellCount,int,&Buyers);CHKERROR(ierr, "Error in PetscMalloc");
        for(int s = 0; s < SellCount; s++) {
          SellSizes[s] = MaxSellSize;
          Buyers[s]    = MPI_ANY_SOURCE;
        }

        if (debug) {
          ostringstream txt;

          for(int p = 0; p < (int) points->size(); p++) {
            txt << "["<<rank<<"]: BuyPoints["<<p<<"]: ("<<BuyPoints[p*msgSize]<<", "<<BuyPoints[p*msgSize+1]<<") coneSize "<<BuyPoints[p*msgSize+2]<<std::endl;
          }
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }

        // First tell sellers which points we want to buy
        ierr = PetscObjectGetNewTag(petscObj, &tag1); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag1, msgSize, BuyCount, BuySizes, Sellers, BuyPoints, SellCount, SellSizes, Buyers, &SellPoints);

        if (debug) {
          ostringstream txt;

          if (!rank) {txt << "Unsquished" << std::endl;}
          for(int p = 0; p < SellCount*MaxSellSize; p++) {
            txt << "["<<rank<<"]: SellPoints["<<p<<"]: ("<<SellPoints[p*msgSize]<<", "<<SellPoints[p*msgSize+1]<<") coneSize "<<SellPoints[p*msgSize+2]<<std::endl;
          }
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }

        // Since we gave maximum sizes, we need to squeeze SellPoints
        for(int s = 0, offset = 0; s < SellCount; s++) {
          if (offset != s*MaxSellSize*msgSize) {
            ierr = PetscMemmove(&SellPoints[offset], &SellPoints[s*MaxSellSize*msgSize], SellSizes[s]*msgSize*sizeof(int32_t));CHKERROR(ierr,"Error in PetscMemmove");
          }
          offset += SellSizes[s]*msgSize;
        }

        if (debug) {
          ostringstream txt;
          int SellSize = 0;

          if (!rank) {txt << "Squished" << std::endl;}
          for(int s = 0; s < SellCount; s++) {
            SellSize += SellSizes[s];
            txt << "SellSizes["<<s<<"]: "<<SellSizes[s]<< std::endl;
          }
          for(int p = 0; p < SellSize; p++) {
            txt << "["<<rank<<"]: SellPoints["<<p<<"]: ("<<SellPoints[p*msgSize]<<", "<<SellPoints[p*msgSize+1]<<") coneSize "<<SellPoints[p*msgSize+2]<<std::endl;
          }
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }

        // SellSizes, Buyers, and SellPoints are output
        Point__int_pair_set BillOfSale;

        for(int s = 0, offset = 0; s < SellCount; s++) {
          for(int m = 0; m < SellSizes[s]; m++) {
            Point point = Point(SellPoints[offset], SellPoints[offset+1]);

            BillOfSale[point].insert(int_pair(Buyers[s], SellPoints[offset+2]));
            offset += msgSize;
          }
        }
        for(int s = 0, offset = 0; s < SellCount; s++) {
          for(int m = 0; m < SellSizes[s]; m++) {
            Point point = Point(SellPoints[offset], SellPoints[offset+1]);

            // Decrement the buyer count so as not to count the current buyer itself
            SellPoints[offset+2] = BillOfSale[point].size()-1;
            offset += msgSize;
          }
        }

        // Then tell buyers how many other buyers there were
        ierr = PetscObjectGetNewTag(petscObj, &tag2); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag2, msgSize, SellCount, SellSizes, Buyers, SellPoints, BuyCount, BuySizes, Sellers, &BuyPoints);

        int      BuyConesSize  = 0;
        int      SellConesSize = 0;
        int     *BuyConesSizes;  // The number of points to buy from each seller
        int     *SellConesSizes; // The number of points to sell to each buyer
        int32_t *SellCones;      // The (rank, cone size) for each point from all other buyers
        int32_t *overlapInfo = PETSC_NULL; // The (rank, cone size) for each point from all other buyers
        ierr = PetscMalloc2(BuyCount,int,&BuyConesSizes,SellCount,int,&SellConesSizes);CHKERROR(ierr, "Error in PetscMalloc");
        for(int s = 0, offset = 0; s < SellCount; s++) {
          SellConesSizes[s] = 0;

          for(int m = 0; m < SellSizes[s]; m++) {
            SellConesSizes[s] += SellPoints[offset+2]+1;
            offset            += msgSize;
          }
          SellConesSize += SellConesSizes[s];
        }

        for(int b = 0, offset = 0; b < BuyCount; b++) {
          BuyConesSizes[b] = 0;

          for(int m = 0; m < BuySizes[b]; m++) {
            BuyConesSizes[b] += BuyPoints[offset+2]+1;
            offset           += msgSize;
          }
          BuyConesSize += BuyConesSizes[b];
        }

        int cMsgSize = 2;
        ierr = PetscMalloc(SellConesSize*cMsgSize * sizeof(int32_t), &SellCones);CHKERROR(ierr, "Error in PetscMalloc");
        for(int s = 0, offset = 0, cOffset = 0, SellConeSize = 0; s < SellCount; s++) {
          for(int m = 0; m < SellSizes[s]; m++) {
            Point point(SellPoints[offset],SellPoints[offset+1]);

            for(typename int_pair_set::iterator p_iter = BillOfSale[point].begin(); p_iter != BillOfSale[point].end(); ++p_iter) {
              SellCones[cOffset+0] = (*p_iter).first;
              SellCones[cOffset+1] = (*p_iter).second;
              cOffset += cMsgSize;
            }
            offset += msgSize;
          }
          if (cOffset - cMsgSize*SellConeSize != cMsgSize*SellConesSizes[s]) {
            throw ALE::Exception("Nonmatching sizes");
          }
          SellConeSize += SellConesSizes[s];
        }

        // Then send buyers a (rank, cone size) for all buyers of the same points
        ierr = PetscObjectGetNewTag(petscObj, &tag3); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag3, cMsgSize, SellCount, SellConesSizes, Buyers, SellCones, BuyCount, BuyConesSizes, Sellers, &overlapInfo);

        // Finally build the overlap sifter
        //   (remote rank) ---(base overlap point, remote cone size, local cone size)---> (base overlap point)
        for(int b = 0, offset = 0, cOffset = 0; b < BuyCount; b++) {
          for(int m = 0; m < BuySizes[b]; m++) {
            Point p(BuyPoints[offset],BuyPoints[offset+1]);

            for(int n = 0; n <= BuyPoints[offset+2]; n++) {
              int neighbor = overlapInfo[cOffset+0];
              int coneSize = overlapInfo[cOffset+1];

              if (neighbor != rank) {
                // Record the point, size of the cone over p coming in from neighbor, and going out to the neighbor for the arrow color
                overlap->addArrow(neighbor, p, ALE::pair<Point,ALE::pair<int,int> >(p, ALE::pair<int,int>(coneSize, _graph->cone(p)->size())) );
              }
              cOffset += cMsgSize;
            }
            offset += msgSize;
          }
        }
      };

      #undef  __FUNCT__
      #define __FUNCT__ "__computeOverlapNew"
      template <typename Overlap_>
      static void __computeOverlapNew(const Obj<graph_type>& _graphA, const Obj<graph_type>& _graphB, Obj<Overlap_>& overlap) {
        typedef typename graph_type::traits::baseSequence Sequence;
        MPI_Comm       comm = _graphA->comm();
        int            size = _graphA->commSize();
        int            rank = _graphA->commRank();
        PetscObject    petscObj = _graphA->petscObj();
        PetscMPIInt    tag1, tag2, tag3;
        PetscErrorCode ierr;
        // The bases we are going to work with
        Obj<Sequence> pointsA = _graphA->base();
        Obj<Sequence> pointsB = _graphB->base();

        // We MUST have the same sellers for points in A and B (same point owner determination)
        int *BuyDataA; // 2 ints per processor: number of A base points we buy and number of sales (0 or 1).
        int *BuyDataB; // 2 ints per processor: number of B base points we buy and number of sales (0 or 1).
        ierr = PetscMalloc2(2*size,int,&BuyDataA,2*size,int,&BuyDataB);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMemzero(BuyDataA, 2*size * sizeof(int));CHKERROR(ierr, "Error in PetscMemzero");
        ierr = PetscMemzero(BuyDataB, 2*size * sizeof(int));CHKERROR(ierr, "Error in PetscMemzero");
        // Map from points to the process managing its bin (seller)
        Point__int ownerA, ownerB;

        // determine owners of each base node and save it in a map
        __determinePointOwners(_graphA, pointsA, BuyDataA, ownerA);
        __determinePointOwners(_graphB, pointsB, BuyDataB, ownerB);

        int  msgSize = 3;   // A point is 2 ints, and the cone size is 1
        int  BuyCountA = 0; // The number of sellers with which this process (A buyer) communicates
        int  BuyCountB = 0; // The number of sellers with which this process (B buyer) communicates
        int *BuySizesA;     // The number of A points to buy from each seller
        int *BuySizesB;     // The number of B points to buy from each seller
        int *SellersA;      // The process for each seller of A points
        int *SellersB;      // The process for each seller of B points
        int *offsetsA = new int[size];
        int *offsetsB = new int[size];
        for(int p = 0; p < size; ++p) {
          BuyCountA += BuyDataA[2*p+1];
          BuyCountB += BuyDataB[2*p+1];
        }
        ierr = PetscMalloc2(BuyCountA,int,&BuySizesA,BuyCountA,int,&SellersA);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMalloc2(BuyCountB,int,&BuySizesB,BuyCountB,int,&SellersB);CHKERROR(ierr, "Error in PetscMalloc");
        for(int p = 0, buyNumA = 0, buyNumB = 0; p < size; ++p) {
          if (BuyDataA[2*p+1]) {
            SellersA[buyNumA]    = p;
            BuySizesA[buyNumA++] = BuyDataA[2*p];
          }
          if (BuyDataB[2*p+1]) {
            SellersB[buyNumB]    = p;
            BuySizesB[buyNumB++] = BuyDataB[2*p];
          }
          if (p == 0) {
            offsetsA[p] = 0;
            offsetsB[p] = 0;
          } else {
            offsetsA[p] = offsetsA[p-1] + msgSize*BuyDataA[2*(p-1)];
            offsetsB[p] = offsetsB[p-1] + msgSize*BuyDataB[2*(p-1)];
          }
        }

        // All points are bought from someone
        int32_t *BuyPointsA; // (point, coneSize) for each A point boung from a seller
        int32_t *BuyPointsB; // (point, coneSize) for each B point boung from a seller
        ierr = PetscMalloc2(msgSize*pointsA->size(),int32_t,&BuyPointsA,msgSize*pointsB->size(),int32_t,&BuyPointsB);CHKERROR(ierr,"Error in PetscMalloc");
        for (typename Sequence::iterator p_itor = pointsA->begin(); p_itor != pointsA->end(); p_itor++) {
          BuyPointsA[offsetsA[ownerA[*p_itor]]++] = (*p_itor).prefix;
          BuyPointsA[offsetsA[ownerA[*p_itor]]++] = (*p_itor).index;      
          BuyPointsA[offsetsA[ownerA[*p_itor]]++] = _graphA->cone(*p_itor)->size();
        }
        for (typename Sequence::iterator p_itor = pointsB->begin(); p_itor != pointsB->end(); p_itor++) {
          BuyPointsB[offsetsB[ownerB[*p_itor]]++] = (*p_itor).prefix;
          BuyPointsB[offsetsB[ownerB[*p_itor]]++] = (*p_itor).index;      
          BuyPointsB[offsetsB[ownerB[*p_itor]]++] = _graphB->cone(*p_itor)->size();
        }
        for(int b = 0; b < BuyCountA; ++b) {
          if (offsetsA[SellersA[b]] != msgSize*BuySizesA[b]) {
            throw ALE::Exception("Invalid point size");
          }
        }
        for(int b = 0; b < BuyCountB; ++b) {
          if (offsetsB[SellersB[b]] != msgSize*BuySizesB[b]) {
            throw ALE::Exception("Invalid point size");
          }
        }
        delete [] offsetsA;
        delete [] offsetsB;

        int  SellCountA;     // The number of A point buyers with which this process (seller) communicates
        int  SellCountB;     // The number of B point buyers with which this process (seller) communicates
        int *SellSizesA;     // The number of A points to sell to each buyer
        int *SellSizesB;     // The number of B points to sell to each buyer
        int *BuyersA;        // The process for each A point buyer
        int *BuyersB;        // The process for each B point buyer
        int  MaxSellSizeA;   // The maximum number of messages to be sold to any A point buyer
        int  MaxSellSizeB;   // The maximum number of messages to be sold to any B point buyer
        int32_t *SellPointsA = PETSC_NULL; // The points and cone sizes from all buyers
        int32_t *SellPointsB = PETSC_NULL; // The points and cone sizes from all buyers
        ierr = PetscMaxSum(comm, BuyDataA, &MaxSellSizeA, &SellCountA);CHKERROR(ierr,"Error in PetscMaxSum");
        ierr = PetscMaxSum(comm, BuyDataB, &MaxSellSizeB, &SellCountB);CHKERROR(ierr,"Error in PetscMaxSum");
        ierr = PetscMalloc2(SellCountA,int,&SellSizesA,SellCountA,int,&BuyersA);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMalloc2(SellCountB,int,&SellSizesB,SellCountB,int,&BuyersB);CHKERROR(ierr, "Error in PetscMalloc");
        for(int s = 0; s < SellCountA; s++) {
          SellSizesA[s] = MaxSellSizeA;
          BuyersA[s]    = MPI_ANY_SOURCE;
        }
        for(int s = 0; s < SellCountB; s++) {
          SellSizesB[s] = MaxSellSizeB;
          BuyersB[s]    = MPI_ANY_SOURCE;
        }

        if (debug) {
          ostringstream txt;

          for(int s = 0; s < BuyCountA; s++) {
            txt << "BuySizesA["<<s<<"]: "<<BuySizesA[s]<<" from seller "<<SellersA[s]<< std::endl;
          }
          for(int p = 0; p < (int) pointsA->size(); p++) {
            txt << "["<<rank<<"]: BuyPointsA["<<p<<"]: ("<<BuyPointsA[p*msgSize]<<", "<<BuyPointsA[p*msgSize+1]<<") coneSize "<<BuyPointsA[p*msgSize+2]<<std::endl;
          }
          for(int s = 0; s < BuyCountB; s++) {
            txt << "BuySizesB["<<s<<"]: "<<BuySizesB[s]<<" from seller "<<SellersB[s]<< std::endl;
          }
          for(int p = 0; p < (int) pointsB->size(); p++) {
            txt << "["<<rank<<"]: BuyPointsB["<<p<<"]: ("<<BuyPointsB[p*msgSize]<<", "<<BuyPointsB[p*msgSize+1]<<") coneSize "<<BuyPointsB[p*msgSize+2]<<std::endl;
          }
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }

        // First tell sellers which points we want to buy
        //   SellSizes, Buyers, and SellPoints are output
        ierr = PetscObjectGetNewTag(petscObj, &tag1); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag1, msgSize, BuyCountA, BuySizesA, SellersA, BuyPointsA, SellCountA, SellSizesA, BuyersA, &SellPointsA);
        commCycle(comm, tag1, msgSize, BuyCountB, BuySizesB, SellersB, BuyPointsB, SellCountB, SellSizesB, BuyersB, &SellPointsB);

        if (debug) {
          ostringstream txt;

          if (!rank) {txt << "Unsquished" << std::endl;}
          for(int p = 0; p < SellCountA*MaxSellSizeA; p++) {
            txt << "["<<rank<<"]: SellPointsA["<<p<<"]: ("<<SellPointsA[p*msgSize]<<", "<<SellPointsA[p*msgSize+1]<<") coneSize "<<SellPointsA[p*msgSize+2]<<std::endl;
          }
          for(int p = 0; p < SellCountB*MaxSellSizeB; p++) {
            txt << "["<<rank<<"]: SellPointsB["<<p<<"]: ("<<SellPointsB[p*msgSize]<<", "<<SellPointsB[p*msgSize+1]<<") coneSize "<<SellPointsB[p*msgSize+2]<<std::endl;
          }
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }

        // Since we gave maximum sizes, we need to squeeze SellPoints
        for(int s = 0, offset = 0; s < SellCountA; s++) {
          if (offset != s*MaxSellSizeA*msgSize) {
            ierr = PetscMemmove(&SellPointsA[offset], &SellPointsA[s*MaxSellSizeA*msgSize], SellSizesA[s]*msgSize*sizeof(int32_t));CHKERROR(ierr,"Error in PetscMemmove");
          }
          offset += SellSizesA[s]*msgSize;
        }
        for(int s = 0, offset = 0; s < SellCountB; s++) {
          if (offset != s*MaxSellSizeB*msgSize) {
            ierr = PetscMemmove(&SellPointsB[offset], &SellPointsB[s*MaxSellSizeB*msgSize], SellSizesB[s]*msgSize*sizeof(int32_t));CHKERROR(ierr,"Error in PetscMemmove");
          }
          offset += SellSizesB[s]*msgSize;
        }

        if (debug) {
          ostringstream txt;
          int SellSizeA = 0, SellSizeB = 0;

          if (!rank) {txt << "Squished" << std::endl;}
          for(int s = 0; s < SellCountA; s++) {
            SellSizeA += SellSizesA[s];
            txt << "SellSizesA["<<s<<"]: "<<SellSizesA[s]<<" from buyer "<<BuyersA[s]<< std::endl;
          }
          for(int p = 0; p < SellSizeA; p++) {
            txt << "["<<rank<<"]: SellPointsA["<<p<<"]: ("<<SellPointsA[p*msgSize]<<", "<<SellPointsA[p*msgSize+1]<<") coneSize "<<SellPointsA[p*msgSize+2]<<std::endl;
          }
          for(int s = 0; s < SellCountB; s++) {
            SellSizeB += SellSizesB[s];
            txt << "SellSizesB["<<s<<"]: "<<SellSizesB[s]<<" from buyer "<<BuyersB[s]<< std::endl;
          }
          for(int p = 0; p < SellSizeB; p++) {
            txt << "["<<rank<<"]: SellPointsB["<<p<<"]: ("<<SellPointsB[p*msgSize]<<", "<<SellPointsB[p*msgSize+1]<<") coneSize "<<SellPointsB[p*msgSize+2]<<std::endl;
          }
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }

        // Map from A base points to (B process, B coneSize) pairs
        Point__int_pair_set BillOfSaleAtoB;
        // Map from B base points to (A process, A coneSize) pairs
        Point__int_pair_set BillOfSaleBtoA;

        // Find the A points being sold to B buyers and record the B cone size
        for(int s = 0, offset = 0; s < SellCountA; s++) {
          for(int m = 0; m < SellSizesA[s]; m++) {
            Point point = Point(SellPointsA[offset], SellPointsA[offset+1]);
            // Just insert the point
            int size = BillOfSaleAtoB[point].size();
            // Avoid unused variable warning
            if (!size) offset += msgSize;
          }
        }
        for(int s = 0, offset = 0; s < SellCountB; s++) {
          for(int m = 0; m < SellSizesB[s]; m++) {
            Point point = Point(SellPointsB[offset], SellPointsB[offset+1]);

            if (BillOfSaleAtoB.find(point) != BillOfSaleAtoB.end()) {
              BillOfSaleAtoB[point].insert(int_pair(BuyersB[s], SellPointsB[offset+2]));
            }
            offset += msgSize;
          }
        }
        // Find the B points being sold to A buyers and record the A cone size
        for(int s = 0, offset = 0; s < SellCountB; s++) {
          for(int m = 0; m < SellSizesB[s]; m++) {
            Point point = Point(SellPointsB[offset], SellPointsB[offset+1]);
            // Just insert the point
            int size = BillOfSaleBtoA[point].size();
            // Avoid unused variable warning
            if (!size) offset += msgSize;
          }
        }
        for(int s = 0, offset = 0; s < SellCountA; s++) {
          for(int m = 0; m < SellSizesA[s]; m++) {
            Point point = Point(SellPointsA[offset], SellPointsA[offset+1]);

            if (BillOfSaleBtoA.find(point) != BillOfSaleBtoA.end()) {
              BillOfSaleBtoA[point].insert(int_pair(BuyersA[s], SellPointsA[offset+2]));
            }
            offset += msgSize;
          }
        }
        // Calculate number of B buyers for A base points
        for(int s = 0, offset = 0; s < SellCountA; s++) {
          for(int m = 0; m < SellSizesA[s]; m++) {
            Point point = Point(SellPointsA[offset], SellPointsA[offset+1]);

            SellPointsA[offset+2] = BillOfSaleAtoB[point].size();
            offset += msgSize;
          }
        }
        // Calculate number of A buyers for B base points
        for(int s = 0, offset = 0; s < SellCountB; s++) {
          for(int m = 0; m < SellSizesB[s]; m++) {
            Point point = Point(SellPointsB[offset], SellPointsB[offset+1]);

            SellPointsB[offset+2] = BillOfSaleBtoA[point].size();
            offset += msgSize;
          }
        }

        // Tell A buyers how many B buyers there were (contained in BuyPointsA)
        // Tell B buyers how many A buyers there were (contained in BuyPointsB)
        ierr = PetscObjectGetNewTag(petscObj, &tag2); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag2, msgSize, SellCountA, SellSizesA, BuyersA, SellPointsA, BuyCountA, BuySizesA, SellersA, &BuyPointsA);
        commCycle(comm, tag2, msgSize, SellCountB, SellSizesB, BuyersB, SellPointsB, BuyCountB, BuySizesB, SellersB, &BuyPointsB);

        if (debug) {
          ostringstream txt;
          int BuySizeA = 0, BuySizeB = 0;

          if (!rank) {txt << "Got other B and A buyers" << std::endl;}
          for(int s = 0; s < BuyCountA; s++) {
            BuySizeA += BuySizesA[s];
            txt << "BuySizesA["<<s<<"]: "<<BuySizesA[s]<<" from seller "<<SellersA[s]<< std::endl;
          }
          for(int p = 0; p < BuySizeA; p++) {
            txt << "["<<rank<<"]: BuyPointsA["<<p<<"]: ("<<BuyPointsA[p*msgSize]<<", "<<BuyPointsA[p*msgSize+1]<<") B buyers "<<BuyPointsA[p*msgSize+2]<<std::endl;
          }
          for(int s = 0; s < BuyCountB; s++) {
            BuySizeB += BuySizesB[s];
            txt << "BuySizesB["<<s<<"]: "<<BuySizesB[s]<<" from seller "<<SellersB[s]<< std::endl;
          }
          for(int p = 0; p < BuySizeB; p++) {
            txt << "["<<rank<<"]: BuyPointsB["<<p<<"]: ("<<BuyPointsB[p*msgSize]<<", "<<BuyPointsB[p*msgSize+1]<<") A buyers "<<BuyPointsB[p*msgSize+2]<<std::endl;
          }
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }

        int      BuyConesSizeA  = 0;
        int      BuyConesSizeB  = 0;
        int      SellConesSizeA = 0;
        int      SellConesSizeB = 0;
        int     *BuyConesSizesA;  // The number of A points to buy from each seller
        int     *BuyConesSizesB;  // The number of B points to buy from each seller
        int     *SellConesSizesA; // The number of A points to sell to each buyer
        int     *SellConesSizesB; // The number of B points to sell to each buyer
        int32_t *SellConesA;      // The (rank, B cone size) for each A point from all other B buyers
        int32_t *SellConesB;      // The (rank, A cone size) for each B point from all other A buyers
        int32_t *overlapInfoA = PETSC_NULL; // The (rank, B cone size) for each A point from all other B buyers
        int32_t *overlapInfoB = PETSC_NULL; // The (rank, A cone size) for each B point from all other A buyers
        ierr = PetscMalloc2(BuyCountA,int,&BuyConesSizesA,SellCountA,int,&SellConesSizesA);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMalloc2(BuyCountB,int,&BuyConesSizesB,SellCountB,int,&SellConesSizesB);CHKERROR(ierr, "Error in PetscMalloc");
        for(int s = 0, offset = 0; s < SellCountA; s++) {
          SellConesSizesA[s] = 0;

          for(int m = 0; m < SellSizesA[s]; m++) {
            SellConesSizesA[s] += SellPointsA[offset+2];
            offset             += msgSize;
          }
          SellConesSizeA += SellConesSizesA[s];
        }
        for(int s = 0, offset = 0; s < SellCountB; s++) {
          SellConesSizesB[s] = 0;

          for(int m = 0; m < SellSizesB[s]; m++) {
            SellConesSizesB[s] += SellPointsB[offset+2];
            offset             += msgSize;
          }
          SellConesSizeB += SellConesSizesB[s];
        }

        for(int b = 0, offset = 0; b < BuyCountA; b++) {
          BuyConesSizesA[b] = 0;

          for(int m = 0; m < BuySizesA[b]; m++) {
            BuyConesSizesA[b] += BuyPointsA[offset+2];
            offset            += msgSize;
          }
          BuyConesSizeA += BuyConesSizesA[b];
        }
        for(int b = 0, offset = 0; b < BuyCountB; b++) {
          BuyConesSizesB[b] = 0;

          for(int m = 0; m < BuySizesB[b]; m++) {
            BuyConesSizesB[b] += BuyPointsB[offset+2];
            offset            += msgSize;
          }
          BuyConesSizeB += BuyConesSizesB[b];
        }

        int cMsgSize = 2;
        ierr = PetscMalloc2(SellConesSizeA*cMsgSize,int32_t,&SellConesA,SellConesSizeB*cMsgSize,int32_t,&SellConesB);CHKERROR(ierr, "Error in PetscMalloc");
        for(int s = 0, offset = 0, cOffset = 0, SellConeSize = 0; s < SellCountA; s++) {
          for(int m = 0; m < SellSizesA[s]; m++) {
            Point point(SellPointsA[offset],SellPointsA[offset+1]);

            for(typename int_pair_set::iterator p_iter = BillOfSaleAtoB[point].begin(); p_iter != BillOfSaleAtoB[point].end(); ++p_iter) {
              SellConesA[cOffset+0] = (*p_iter).first;
              SellConesA[cOffset+1] = (*p_iter).second;
              cOffset += cMsgSize;
            }
            offset += msgSize;
          }
          if (cOffset - cMsgSize*SellConeSize != cMsgSize*SellConesSizesA[s]) {
            throw ALE::Exception("Nonmatching sizes");
          }
          SellConeSize += SellConesSizesA[s];
        }
        for(int s = 0, offset = 0, cOffset = 0, SellConeSize = 0; s < SellCountB; s++) {
          for(int m = 0; m < SellSizesB[s]; m++) {
            Point point(SellPointsB[offset],SellPointsB[offset+1]);

            for(typename int_pair_set::iterator p_iter = BillOfSaleBtoA[point].begin(); p_iter != BillOfSaleBtoA[point].end(); ++p_iter) {
              SellConesB[cOffset+0] = (*p_iter).first;
              SellConesB[cOffset+1] = (*p_iter).second;
              cOffset += cMsgSize;
            }
            offset += msgSize;
          }
          if (cOffset - cMsgSize*SellConeSize != cMsgSize*SellConesSizesB[s]) {
            throw ALE::Exception("Nonmatching sizes");
          }
          SellConeSize += SellConesSizesB[s];
        }

        // Then send A buyers a (rank, cone size) for all B buyers of the same points
        ierr = PetscObjectGetNewTag(petscObj, &tag3); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag3, cMsgSize, SellCountA, SellConesSizesA, BuyersA, SellConesA, BuyCountA, BuyConesSizesA, SellersA, &overlapInfoA);
        commCycle(comm, tag3, cMsgSize, SellCountB, SellConesSizesB, BuyersB, SellConesB, BuyCountB, BuyConesSizesB, SellersB, &overlapInfoB);

        // Finally build the A-->B overlap sifter
        //   (remote rank) ---(base A overlap point, remote cone size, local cone size)---> (base A overlap point)
        for(int b = 0, offset = 0, cOffset = 0; b < BuyCountA; b++) {
          for(int m = 0; m < BuySizesA[b]; m++) {
            Point p(BuyPointsA[offset],BuyPointsA[offset+1]);

            for(int n = 0; n < BuyPointsA[offset+2]; n++) {
              int neighbor = overlapInfoA[cOffset+0];
              int coneSize = overlapInfoA[cOffset+1];

              // Record the point, size of the cone over p coming in from neighbor, and going out to the neighbor for the arrow color
              overlap->addArrow(neighbor, ALE::pair<int,Point>(0, p), ALE::pair<Point,ALE::pair<int,int> >(p, ALE::pair<int,int>(coneSize, _graphA->cone(p)->size())) );
              cOffset += cMsgSize;
            }
            offset += msgSize;
          }
        }

        // Finally build the B-->A overlap sifter
        //   (remote rank) ---(base B overlap point, remote cone size, local cone size)---> (base B overlap point)
        for(int b = 0, offset = 0, cOffset = 0; b < BuyCountB; b++) {
          for(int m = 0; m < BuySizesB[b]; m++) {
            Point p(BuyPointsB[offset],BuyPointsB[offset+1]);

            for(int n = 0; n < BuyPointsB[offset+2]; n++) {
              int neighbor = overlapInfoB[cOffset+0];
              int coneSize = overlapInfoB[cOffset+1];

              // Record the point, size of the cone over p coming in from neighbor, and going out to the neighbor for the arrow color
              overlap->addArrow(neighbor, ALE::pair<int,Point>(1, p), ALE::pair<Point,ALE::pair<int,int> >(p, ALE::pair<int,int>(coneSize, _graphB->cone(p)->size())) );
              cOffset += cMsgSize;
            }
            offset += msgSize;
          }
        }
      };

      #undef  __FUNCT__
      #define __FUNCT__ "__computeOverlap"
      template <typename Overlap_>
      static void __computeOverlap(const Obj<graph_type>& _graph, Obj<Overlap_>& overlap) {
        typedef typename graph_type::traits::baseSequence Sequence;
        PetscErrorCode ierr;
        MPI_Comm comm = _graph->comm();
        int      size = _graph->commSize();
        int      rank = _graph->commRank();
        PetscObject petscObj = _graph->petscObj();

        bool debug  = delta_type::debug > 0;
        bool debug2 = delta_type::debug > 1;

        // Allocate space for the ownership data
        int32_t *LeaseData; // 2 ints per processor: number of leased nodes and number of leases (0 or 1).
        ierr = PetscMalloc(2*size*sizeof(PetscInt),&LeaseData);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMemzero(LeaseData,2*size*sizeof(PetscInt));CHKERROR(ierr, "Error in PetscMemzero");
        
        // The base we are going to work with
        Obj<Sequence> points = _graph->base();

        // determine owners of each base node and save it in a map
        Point__int owner;
        __determinePointOwners(_graph, _graph->base(), LeaseData, owner);
    
        // Now we accumulate the max lease size and the total number of renters
        // Determine the owners of base nodes and collect the lease data for each processor:
        // the number of nodes leased and the number of leases (0 or 1).
        int32_t MaxLeaseSize, RenterCount;
        ierr = PetscMaxSum(comm,LeaseData,&MaxLeaseSize,&RenterCount);CHKERROR(ierr,"Error in PetscMaxSum");
        //ierr = PetscInfo1(0,"%s: Number of renters %d\n", __FUNCT__, RenterCount); 
        //CHKERROR(ierr,"Error in PetscInfo");

        if(debug) { /* -------------------------------------------------------------- */
          ierr = PetscSynchronizedPrintf(comm, "[%d]: %s: RenterCount = %d, MaxLeaseSize = %d\n", 
                                         rank, __FUNCT__, RenterCount, MaxLeaseSize);
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);
          CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        } /* ----------------------------------------------------------------------- */
        
        // post receives for all Rented nodes; we will be receiving 3 data items per rented node, 
        // and at most MaxLeaseSize of nodes per renter
        PetscMPIInt    tag1;
        ierr = PetscObjectGetNewTag(petscObj, &tag1); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        int32_t *RentedNodes;
        MPI_Request *Renter_waits;
        if(RenterCount){
          ierr = PetscMalloc((RenterCount)*(3*MaxLeaseSize+1)*sizeof(int32_t),&RentedNodes);  CHKERROR(ierr,"Error in PetscMalloc");
          ierr = PetscMemzero(RentedNodes,(RenterCount)*(3*MaxLeaseSize+1)*sizeof(int32_t));  CHKERROR(ierr,"Error in PetscMemzero");
          ierr = PetscMalloc((RenterCount)*sizeof(MPI_Request),&Renter_waits);                CHKERROR(ierr,"Error in PetscMalloc");
        }
        for (int32_t i=0; i<RenterCount; i++) {
          ierr = MPI_Irecv(RentedNodes+3*MaxLeaseSize*i,3*MaxLeaseSize,MPIU_INT,MPI_ANY_SOURCE,tag1,comm,Renter_waits+i);
          CHKERROR(ierr,"Error in MPI_Irecv");
        }
        
        int32_t LessorCount;
        LessorCount = 0; for (int32_t i=0; i<size; i++) LessorCount += LeaseData[2*i+1];
        //ierr = PetscInfo1(0,"%s: Number of lessors %d\n",__FUNCT__, LessorCount);
        //CHKERROR(ierr,"Error in PetscInfo");
        if(debug) { /* -------------------------------------------------------------- */
          ierr = PetscSynchronizedPrintf(comm, "[%d]: %s: LessorCount = %d\n", rank, __FUNCT__, LessorCount);
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);
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
          txt << "[" << rank << "]: " << __FUNCT__ << ": lessor data [index, rank, lease size]: ";
          for(int32_t i = 0; i < LessorCount; i++) {
            txt << "[" << i << ", " << Lessors[i] << ", " << LeaseSizes[i] << "] ";
          }
          txt << "\n";
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }/* -----------------------------------  */
        if(debug2) { /* ----------------------------------- */
          ostringstream txt;
          txt << "[" << rank << "]: " << __FUNCT__ << ": LessorIndex: ";
          for(int__int::iterator li_itor = LessorIndex.begin(); li_itor!= LessorIndex.end(); li_itor++) {
            int32_t i = (*li_itor).first;
            int32_t j = (*li_itor).second;
            txt << i << "-->" << j << "; ";
          }
          txt << "\n";
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
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
          LeasedNodes[LessorOffsets[ind]++] = _graph->cone(p)->size();
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
          ierr      = MPI_Isend(LeasedNodes+LessorOffsets[i],3*LeaseSizes[i],MPIU_INT,Lessors[i],tag1,comm,&Lessor_waits[i]);
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
            txt << "[" << rank << "]: " << __FUNCT__ << ": node (" << node.prefix << "," << node.index << ") is rented by " << renterSet.size() << " renters (renter, cone size):  ";
            for (int_pair_set::iterator renterSet_itor = renterSet.begin(); renterSet_itor != renterSet.end(); renterSet_itor++) 
            {
              txt << "(" << (*renterSet_itor).first << "," << (*renterSet_itor).second << ") ";
            }
            txt << "\n";
          }
          // Now send the C-string behind txt to PetscSynchronizedPrintf
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
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
        ierr = PetscObjectGetNewTag(petscObj, &tag2); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        for (int32_t i=0; i<LessorCount; i++) {
          ierr = MPI_Irecv(NeighborCounts+LessorOffsets[i],3*LeaseSizes[i],MPIU_INT,Lessors[i],tag2,comm,&Lessor_waits[i]);
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
          ierr = PetscSynchronizedPrintf(comm, "[%d]: TotalRentalCount %d\n", rank, TotalRentalCount); 
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm); CHKERROR(ierr, "PetscSynchronizedFlush");
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
          ierr      = MPI_Isend(SharerCounts+RenterOffset,3*RenterLeaseSizes[a],MPIU_INT,Renters[a],tag2,comm,Renter_waits+a);
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
          txt << "[" << rank << "]: " << __FUNCT__ << ": neighbor counts by lessor-node [lessor rank, (node), neighbor count]:  ";
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
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm); CHKERROR(ierr, "PetscSynchronizedFlush");
        }/* -----------------------------------  */
        
        
        // Now we allocate an array to receive the neighbor ranks and the remote cone sizes for each leased node,
        // hence, the total array size is 2*TotalNeighborCount.
        // Note that the lessor offsets must be recalculated, since they are no longer based on the number of nodes 
        // leased from that lessor, but on the number of neighbor over the nodes leased from that lessor.
        
        // First we compute the numbers of neighbors over the nodes leased from a given lessor.
        // NeighborCountsByLessor[lessor] = # of neighbors on that lessor
        int32_t TotalNeighborCount = 0;
        int32_t *NeighborCountsByLessor;
        if(LessorCount) {
          ierr = PetscMalloc((LessorCount)*sizeof(int32_t), &NeighborCountsByLessor); CHKERROR(ierr, "Error in PetscMalloc");
        }
        cntr = 0;
        for(int32_t i = 0; i < LessorCount; i++) {
          int32_t neighborCountByLessor = 0;        
          for(int32_t j = 0; j < LeaseSizes[i]; j++) {
            //ASSUMPTION on Point type affects NeighborCountsOffset size
            cntr += 2;
            neighborCountByLessor += NeighborCounts[cntr++];
          }
          NeighborCountsByLessor[i] = neighborCountByLessor;
          TotalNeighborCount       += neighborCountByLessor; 
        }
        if (debug2) { /* -----------------------------------  */
          // Use a C++ string stream to report the numbers of shared nodes leased from each lessor
          ostringstream txt;
          cntr = 0;
          txt << "[" << rank << "]: " << __FUNCT__ << ": NeighborCountsByLessor [rank, count]:  ";
          for(int32_t i = 0; i < LessorCount; i++) {
            txt << "[" << Lessors[i] <<","  <<  NeighborCountsByLessor[i] << "]; ";
          }
          txt << std::endl;
          txt << "[" << rank << "]: " << __FUNCT__ << ": TotalNeighborCount: " << TotalNeighborCount << std::endl;
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm); CHKERROR(ierr, "PetscSynchronizedFlush");
        }/* -----------------------------------  */
        int32_t *Neighbors = 0;
        if(TotalNeighborCount) {
          ierr = PetscMalloc((2*TotalNeighborCount)*sizeof(int32_t),&Neighbors); CHKERROR(ierr,"Error in PetscMalloc");
        }
        
        // Post receives for Neighbors
        PetscMPIInt    tag3;
        ierr = PetscObjectGetNewTag(petscObj, &tag3); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        int32_t lessorOffset = 0;
        for(int32_t i=0; i<LessorCount; i++) {
          if(NeighborCountsByLessor[i]) { // We expect messages from lessors with a non-zero NeighborCountsByLessor entry only
            ierr = MPI_Irecv(Neighbors+lessorOffset,2*NeighborCountsByLessor[i],MPIU_INT,Lessors[i],tag3,comm,&Lessor_waits[i]);
            CHKERROR(ierr,"Error in MPI_Irecv");
            lessorOffset += 2*NeighborCountsByLessor[i];
          }
        }
        if (lessorOffset != 2*TotalNeighborCount) {
          ostringstream msg;

          msg << "["<<rank<<"]Invalid lessor offset " << lessorOffset << " should be " << 2*TotalNeighborCount << std::endl;
          throw ALE::Exception(msg.str().c_str());
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
          txt << "[" << rank << "]: " << __FUNCT__ << ": RenterCount = " << RenterCount << "\n";
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
              txt << "[" << rank << "]: " << __FUNCT__ << ": renters sharing with " << r << " of node  (" << p.prefix << "," << p.index << ")  [rank, cone size]:  ";
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
            ierr      = MPI_Isend(Sharers+RenterOffset,SegmentSize,MPIU_INT,Renters[a],tag3,comm,Renter_waits+a);
            CHKERROR(ierr, "Error in MPI_Isend");
          }
          // Offset is advanced by the segmentSize
          RenterOffset += SegmentSize;
        }//  for(int32_t a = 0; a < RenterCount; a++) {
        if(debug) {
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm); CHKERROR(ierr, "PetscSynchronizedFlush");      
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
            txt << "[" <<rank<< "]: " << __FUNCT__ << ": neighbors over nodes leased from " <<Lessors[i]<< ":\n";
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
          ierr = PetscSynchronizedPrintf(comm,txt.str().c_str());CHKERROR(ierr,"Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
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
        
        // Now we record the neighbors and the cones over each node to be received from or sent to each neigbor.
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
            // Record the size of the cone over p coming in from neighbor and going out to the neighbor as a pair of integers
            // which is the color of the overlap arrow from neighbor to p
            overlap->addArrow(neighbor, p, ALE::pair<Point,ALE::pair<int,int> >(p, ALE::pair<int,int>(coneSize, _graph->cone(p)->size())) ); 
          }
        }// for(int32_t i = 0; i < LeasedNodeCount; i++)

        // Wait on the original sends to the renters (the last vestige of the lessor-renter exchange epoch; we delayed it to afford the
        // greatest opportunity for a communication-computation overlap).
        if(RenterCount) {
          ierr = MPI_Waitall(RenterCount, Renter_waits, Renter_status); CHKERROR(ierr,"Error in MPI_Waitall");
        }
        if(RenterCount) {
          ierr = PetscFree(Renter_waits); CHKERROR(ierr, "Error in PetscFree");
          ierr = PetscFree(Renter_status); CHKERROR(ierr, "Error in PetscFree");
        }        

        if(LeasedNodeCount) {ierr = PetscFree(NeighborCounts); CHKERROR(ierr,"Error in PetscFree");}
        if(TotalNeighborCount) {ierr = PetscFree(Neighbors);   CHKERROR(ierr, "Error in PetscFree");}    
        if(TotalRentalCount){ierr = PetscFree(SharerCounts);   CHKERROR(ierr, "Error in PetscFree");}

      };// __computeOverlap()

      /*
        Seller:       A possessor of data
        Buyer:        A requestor of data

        Note that in this routine, the caller functions as BOTH a buyer and seller.

        When we post receives, we use a buffer of the maximum size for each message
        in order to simplify the size calculations (less communication).

        BuyCount:     The number of sellers with which this process (buyer) communicates
                      This is calculated locally
        BuySizes:     The number of messages to buy from each seller
        Sellers:      The process for each seller
        BuyData:      The data to be bought from each seller. There are BuySizes[p] messages
                      to be purchased from each process p, in order of rank.
        SellCount:    The number of buyers with which this process (seller) communicates
                      This requires communication
        SellSizes:    The number of messages to be sold to each buyer
        Buyers:       The process for each buyer
        msgSize:      The number of integers in each message
        SellData:     The data to be sold to each buyer. There are SellSizes[p] messages
                      to be sold to each process p, in order of rank.
      */
      static void commCycle(MPI_Comm comm, PetscMPIInt tag, int msgSize, int BuyCount, int BuySizes[], int Sellers[], int32_t BuyData[], int SellCount, int SellSizes[], int Buyers[], int32_t *SellData[]) {
        int32_t     *locSellData; // Messages to sell to buyers (received from buyers)
        int          SellSize = 0;
        int         *BuyOffsets, *SellOffsets;
        MPI_Request *buyWaits,  *sellWaits;
        MPI_Status  *buyStatus;
        PetscErrorCode ierr;

        // Allocation
        ierr = PetscMallocValidate(__LINE__,__FUNCT__,__FILE__,__SDIR__);CHKERROR(ierr,"Memory corruption");
        for(int s = 0; s < SellCount; s++) {SellSize += SellSizes[s];}
        ierr = PetscMalloc2(BuyCount,int,&BuyOffsets,SellCount,int,&SellOffsets);CHKERROR(ierr,"Error in PetscMalloc");
        ierr = PetscMalloc3(BuyCount,MPI_Request,&buyWaits,SellCount,MPI_Request,&sellWaits,BuyCount,MPI_Status,&buyStatus);
        CHKERROR(ierr,"Error in PetscMalloc");
        if (*SellData) {
          locSellData = *SellData;
        } else {
          ierr = PetscMalloc(msgSize*SellSize * sizeof(int32_t), &locSellData);CHKERROR(ierr,"Error in PetscMalloc");
        }
        // Initialization
        for(int b = 0; b < BuyCount; b++) {
          if (b == 0) {
            BuyOffsets[0] = 0;
          } else {
            BuyOffsets[b] = BuyOffsets[b-1] + msgSize*BuySizes[b-1];
          }
        }
        for(int s = 0; s < SellCount; s++) {
          if (s == 0) {
            SellOffsets[0] = 0;
          } else {
            SellOffsets[s] = SellOffsets[s-1] + msgSize*SellSizes[s-1];
          }
        }
        ierr = PetscMemzero(locSellData, msgSize*SellSize * sizeof(int32_t));CHKERROR(ierr,"Error in PetscMemzero");

        // Post receives for bill of sale (data request)
        for(int s = 0; s < SellCount; s++) {
          ierr = MPI_Irecv(&locSellData[SellOffsets[s]], msgSize*SellSizes[s], MPIU_INT, Buyers[s], tag, comm, &sellWaits[s]);
          CHKERROR(ierr,"Error in MPI_Irecv");
        }
        // Post sends with bill of sale (data request)
        for(int b = 0; b < BuyCount; b++) {
          ierr = MPI_Isend(&BuyData[BuyOffsets[b]], msgSize*BuySizes[b], MPIU_INT, Sellers[b], tag, comm, &buyWaits[b]);
          CHKERROR(ierr,"Error in MPI_Isend");
        }
        // Receive the bill of sale from buyer
        for(int s = 0; s < SellCount; s++) {
          MPI_Status sellStatus;
          int        num;

          ierr = MPI_Waitany(SellCount, sellWaits, &num, &sellStatus);CHKMPIERROR(ierr,ERRORMSG("Error in MPI_Waitany"));
          // OUTPUT: Overwriting input buyer process
          Buyers[num] = sellStatus.MPI_SOURCE;
          // OUTPUT: Overwriting input sell size
          ierr = MPI_Get_count(&sellStatus, MPIU_INT, &SellSizes[num]);CHKERROR(ierr,"Error in MPI_Get_count");
          SellSizes[num] /= msgSize;
        }
        // Wait on send for bill of sale
        if (BuyCount) {
          ierr = MPI_Waitall(BuyCount, buyWaits, buyStatus); CHKERROR(ierr,"Error in MPI_Waitall");
        }

        ierr = PetscFree2(BuyOffsets, SellOffsets);CHKERROR(ierr,"Error in PetscFree");
        ierr = PetscFree3(buyWaits, sellWaits, buyStatus);CHKERROR(ierr,"Error in PetscFree");
        // OUTPUT: Providing data out
        *SellData = locSellData;
      }

      // -------------------------------------------------------------------------------------------------------------------
      #undef __FUNCT__
      #define __FUNCT__ "__computeFusion"
      template <typename Overlap_, typename Fusion_>
      static void __computeFusion(const Obj<graph_type>& _graph, const Obj<Overlap_>& overlap, Obj<Fusion_> fusion, const Obj<fuser_type>& fuser) {
        //
        typedef ConeArraySequence<typename graph_type::traits::arrow_type> cone_array_sequence;
        typedef typename cone_array_sequence::cone_arrow_type              cone_arrow_type;
        PetscErrorCode ierr;
        MPI_Comm comm = _graph->comm();
        int      rank = _graph->commRank();
        PetscObject petscObj = _graph->petscObj();

        bool debug = delta_type::debug > 0;
        bool debug2 = delta_type::debug > 1;

        // Compute total incoming cone sizes by neighbor and the total incomping cone size.
        // Also count the total number of neighbors we will be communicating with
        int32_t  NeighborCountIn = 0;
        int__int NeighborConeSizeIn;
        int32_t  ConeSizeIn = 0;
        ostringstream txt3;
        // Traverse all of the neighbors  from whom we will be receiving cones -- the cap of the overlap.
        typename Overlap_::traits::capSequence overlapCap = overlap->cap();
        for(typename Overlap_::traits::capSequence::iterator ci  = overlapCap.begin(); ci != overlapCap.end(); ci++) 
        { // traversing overlap.cap()
          int32_t neighborIn = *ci;
          // Traverse the supports of the overlap graph under each neighbor rank, count cone sizes to be received and add the cone sizes
          typename Overlap_::traits::supportSequence supp = overlap->support(*ci);
          if(debug2) {
            //txt3 << "[" << rank << "]: " << __FUNCT__ << ": overlap: support of rank " << neighborIn << ": " << std::endl;
            //txt3 << supp;
          }
          int32_t coneSizeIn = 0;
          for(typename Overlap_::traits::supportSequence::iterator si = supp.begin(); si != supp.end(); si++) {
            // FIX: replace si.color() type: Point --> ALE::pair
            //coneSizeIn += si.color().prefix;
            coneSizeIn += si.color().second.first;
          }
          if(coneSizeIn > 0) {
            // Accumulate the total cone size
            ConeSizeIn += coneSizeIn;
            NeighborConeSizeIn[neighborIn] = coneSizeIn;
            NeighborCountIn++;
            txt3  << "[" << rank << "]: " << "NeighborConeSizeIn[" << neighborIn << "]: " << NeighborConeSizeIn[neighborIn] << "\n";
          }
        }
        if(debug2) {
          if(NeighborCountIn == 0) {
            txt3  << "[" << rank << "]: no incoming Neighbors" << std::endl;
          }
          ierr = PetscSynchronizedPrintf(comm,txt3.str().c_str());CHKERROR(ierr,"Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }
        if(debug) {/* --------------------------------------------------------------------------------------------- */
          ostringstream txt;
          txt << "[" << rank << "]: " << __FUNCT__ << ": total size of incoming cone: " << ConeSizeIn << "\n";
          for(int__int::iterator np_itor = NeighborConeSizeIn.begin();np_itor!=NeighborConeSizeIn.end();np_itor++)
          {
            int32_t neighbor = (*np_itor).first;
            int32_t coneSize = (*np_itor).second;
            txt << "[" << rank << "]: " << __FUNCT__ << ": size of cone from " << neighbor << ": " << coneSize << "\n";
            
          }//int__int::iterator np_itor=NeighborConeSizeIn.begin();np_itor!=NeighborConeSizeIn.end();np_itor++)
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str());
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);
          CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        }/* --------------------------------------------------------------------------------------------- */
        // Compute the size of a cone element
        size_t cone_arrow_size = sizeof(cone_arrow_type);
        // Now we can allocate a receive buffer to receive all of the remote cones from neighbors
        cone_arrow_type *ConesIn;
        if(ConeSizeIn) {
          ierr = PetscMalloc(ConeSizeIn*cone_arrow_size,&ConesIn); CHKERROR(ierr,"Error in PetscMalloc");
        }
        // Allocate receive requests
        MPI_Request *NeighborsIn_waits;
        if(NeighborCountIn) {
          ierr = PetscMalloc((NeighborCountIn)*sizeof(MPI_Request),&NeighborsIn_waits);CHKERROR(ierr,"Error in PetscMalloc");
        }
        // Post receives for ConesIn
        PetscMPIInt    tag4;
        ierr = PetscObjectGetNewTag(petscObj, &tag4); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        // Traverse all neighbors from whom we are receiving cones
        cone_arrow_type *NeighborOffsetIn = ConesIn;
        if(debug2) {
          ierr = PetscSynchronizedPrintf(comm, "[%d]: %s: NeighborConeSizeIn.size() = %d\n",rank, __FUNCT__, NeighborConeSizeIn.size());
          CHKERROR(ierr, "Error in PetscSynchornizedPrintf");
          ierr = PetscSynchronizedFlush(comm);
          CHKERROR(ierr, "Error in PetscSynchornizedFlush");
          if(NeighborConeSizeIn.size()) {
            ierr=PetscSynchronizedPrintf(comm, "[%d]: %s: *NeighborConeSizeIn.begin() = (%d,%d)\n",
                                         rank, __FUNCT__, (*NeighborConeSizeIn.begin()).first, (*NeighborConeSizeIn.begin()).second);
            CHKERROR(ierr, "Error in PetscSynchornizedPrintf");
            ierr = PetscSynchronizedFlush(comm);
            CHKERROR(ierr, "Error in PetscSynchornizedFlush");
            
          }
        }
        int32_t n = 0;
        for(std::map<int32_t, int32_t>::iterator n_itor = NeighborConeSizeIn.begin(); n_itor!=NeighborConeSizeIn.end(); n_itor++) {
          int32_t neighborIn = (*n_itor).first;
          int32_t coneSizeIn = (*n_itor).second;
          ierr = MPI_Irecv(NeighborOffsetIn,cone_arrow_size*coneSizeIn,MPI_BYTE,neighborIn,tag4,comm, NeighborsIn_waits+n);
          CHKERROR(ierr, "Error in MPI_Irecv");
          NeighborOffsetIn += coneSizeIn;
          n++;
        }
        
        // Compute the total outgoing cone sizes by neighbor and the total outgoing cone size.
        int__int NeighborConeSizeOut;
        int32_t  ConeSizeOut = 0;
        int32_t NeighborCountOut = 0;
        for(typename Overlap_::traits::capSequence::iterator ci  = overlapCap.begin(); ci != overlapCap.end(); ci++) 
        { // traversing overlap.cap()
          int32_t neighborOut = *ci;
          // Traverse the supports of the overlap graph under each neighbor rank, count cone sizes to be sent and add the cone sizes
          typename Overlap_::traits::supportSequence supp = overlap->support(*ci);
          if(debug2) {
            //txt3 << "[" << rank << "]: " << __FUNCT__ << ": overlap: support of rank " << neighborOut << ": " << std::endl;
            //txt3 << supp;
          }
          int32_t coneSizeOut = 0;
          for(typename Overlap_::traits::supportSequence::iterator si = supp.begin(); si != supp.end(); si++) {
            // FIX: replace si.color() Point --> ALE::pair
            //coneSizeOut += si.color().index;
            coneSizeOut += si.color().second.second;
          }
          if(coneSizeOut > 0) {
            // Accumulate the total cone size
            ConeSizeOut += coneSizeOut;
            NeighborConeSizeOut[neighborOut] = coneSizeOut;
            NeighborCountOut++;
          }
        }//traversing overlap.cap()
        
        if(debug) {/* --------------------------------------------------------------------------------------------- */
          ostringstream txt;
          txt << "[" << rank << "]: " << __FUNCT__ << ": total size of outgoing cone: " << ConeSizeOut << "\n";
          for(int__int::iterator np_itor = NeighborConeSizeOut.begin();np_itor!=NeighborConeSizeOut.end();np_itor++)
          {
            int32_t neighborOut = (*np_itor).first;
            int32_t coneSizeOut = (*np_itor).second;
            txt << "[" << rank << "]: " << __FUNCT__ << ": size of cone to " << neighborOut << ": " << coneSizeOut << "\n";
            
          }//int__int::iterator np_itor=NeighborConeSizeOut.begin();np_itor!=NeighborConeSizeOut.end();np_itor++)
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str());
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);
          CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        }/* --------------------------------------------------------------------------------------------- */
        
        // Now we can allocate a send buffer to send all of the remote cones to neighbors
        cone_arrow_type *ConesOut;
        if(ConeSizeOut) {
          ierr = PetscMalloc(cone_arrow_size*ConeSizeOut,&ConesOut); CHKERROR(ierr,"Error in PetscMalloc");
        }
        // Allocate send requests
        MPI_Request *NeighborsOut_waits;
        if(NeighborCountOut) {
          ierr = PetscMalloc((NeighborCountOut)*sizeof(MPI_Request),&NeighborsOut_waits);CHKERROR(ierr,"Error in PetscMalloc");
        }
        
        // Pack and send messages
        cone_arrow_type *NeighborOffsetOut = ConesOut;
        int32_t cntr = 0; // arrow counter
        n = 0;    // neighbor counter
        ostringstream txt2;
        // Traverse all neighbors to whom we are sending cones
        for(typename Overlap_::traits::capSequence::iterator ci  = overlapCap.begin(); ci != overlapCap.end(); ci++) 
        { // traversing overlap.cap()
          int32_t neighborOut = *ci;

          // Make sure we have a cone going out to this neighbor
          if(NeighborConeSizeOut.find(neighborOut) != NeighborConeSizeOut.end()) { // if there is anything to send
            if(debug) { /* ------------------------------------------------------------ */
              txt2  << "[" << rank << "]: " << __FUNCT__ << ": outgoing cones destined for " << neighborOut << "\n";
            }/* ----------------------------------------------------------------------- */
            int32_t coneSizeOut = NeighborConeSizeOut[neighborOut];          
            // ASSUMPTION: all overlap supports are "symmetric" with respect to swapping processes,so we safely can assume that 
            //             the receiver will be expecting points in the same order as they appear in the support here.
            // Traverse all the points within the overlap with this neighbor 
            typename Overlap_::traits::supportSequence supp = overlap->support(*ci);
            for(typename Overlap_::traits::supportSequence::iterator si = supp.begin(); si != supp.end(); si++) {
              Point p = *si;
              if(debug) { /* ------------------------------------------------------------ */
                txt2  << "[" << rank << "]: \t cone over " << p << ":  ";
              }/* ----------------------------------------------------------------------- */
              // Traverse the cone over p in the local _graph and place corresponding TargetArrows in ConesOut
              typename graph_type::traits::coneSequence cone = _graph->cone(p);
              for(typename graph_type::traits::coneSequence::iterator cone_itor = cone.begin(); cone_itor != cone.end(); cone_itor++) {
                // Place a TargetArrow into the ConesOut buffer 
                // WARNING: pointer arithmetic involving ConesOut takes place here
                //cone_arrow_type::place(ConesOut+cntr, cone_itor.arrow()); 
                cone_arrow_type::place(ConesOut+cntr, typename graph_type::traits::arrow_type(*cone_itor,p,cone_itor.color()));
                cntr++;
                if(debug) { /* ------------------------------------------------------------ */
                  txt2  << " " << *cone_itor;
                }/* ----------------------------------------------------------------------- */
              }
              if(debug) { /* ------------------------------------------------------------ */
                txt2  << std::endl;
              }/* ----------------------------------------------------------------------- */
            }
            ierr = MPI_Isend(NeighborOffsetOut,cone_arrow_size*coneSizeOut,MPI_BYTE,neighborOut,tag4,comm, NeighborsOut_waits+n);
            CHKERROR(ierr, "Error in MPI_Isend");
            // WARNING: pointer arithmetic involving NeighborOffsetOut takes place here
            NeighborOffsetOut += coneSizeOut; // keep track of offset
            n++;  // count neighbors
          }// if there is anything to send
        }// traversing overlap.cap()
        if(debug && NeighborCountOut) {/* --------------------------------------------------------------- */
          ierr = PetscSynchronizedPrintf(comm, txt2.str().c_str());
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);
          CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        }/* --------------------------------------------------------------------------------------------- */
        
        // Allocate an In status array
        MPI_Status *NeighborIn_status;
        if(NeighborCountIn) {
          ierr = PetscMalloc((NeighborCountIn)*sizeof(MPI_Status),&NeighborIn_status);CHKERROR(ierr,"Error in PetscMalloc");
        }
        
        // Wait on the receives
        if(NeighborCountIn) {
          ostringstream txt;
          txt << "[" << _graph->commRank() << "]: Error in MPI_Waitall";
          ierr = MPI_Waitall(NeighborCountIn, NeighborsIn_waits, NeighborIn_status); CHKERROR(ierr,txt.str().c_str());
        }
        
        // Now we unpack the received cones, fuse them with the local cones and store the result in the completion graph.
        // Traverse all neighbors  from whom we are expecting cones
        cntr = 0; // arrow counter
        NeighborOffsetIn = ConesIn;
        ostringstream txt;
        for(typename Overlap_::traits::capSequence::iterator ci  = overlapCap.begin(); ci != overlapCap.end(); ci++) 
        { // traversing overlap.cap()
          // Traverse all the points within the overlap with this neighbor 
          // ASSUMPTION: points are sorted within each neighbor, so we are expecting points in the same order as they arrived in ConesIn
          typename Overlap_::traits::supportSequence supp = overlap->support(*ci);
          for(typename Overlap_::traits::supportSequence::iterator si = supp.begin(); si != supp.end(); si++)
          {
            Point p = *si;
            //int32_t coneSizeIn = si.color().prefix; // FIX: color() type Point --> ALE::Two::pair
            int32_t coneSizeIn = si.color().second.first;
            // NOTE: coneSizeIn may be 0, which is legal, since the fuser in principle can operate on an empty cone.
            // Extract the local cone into a coneSequence
            typename graph_type::traits::coneSequence lcone = _graph->cone(p);
            // Wrap the arrived cone in a cone_array_sequence
            cone_array_sequence rcone(NeighborOffsetIn, coneSizeIn, p);
            if(debug) { /* ---------------------------------------------------------------------------------------*/
              txt << "[" << rank << "]: "<<__FUNCT__<< ": received a cone over " << p << " of size " << coneSizeIn << " from rank "<<*ci<< ":" << std::endl;
              rcone.view(txt, true);
            }/* --------------------------------------------------------------------------------------------------*/
            // Fuse the cones
            fuser->fuseCones(lcone, rcone, fusion->cone(fuser->fuseBasePoints(p,p)));
            if(debug) {
              //ostringstream txt;
              //txt << "[" << rank << "]: ... after fusing the cone over" << p << std::endl;
              //fusion->view(std::cout, txt.str().c_str());
            }
            NeighborOffsetIn += coneSizeIn;
          }
        }
        if(debug) { /* ---------------------------------------------------------------------------------------*/
          if(NeighborCountIn == 0) {
            txt << "[" << rank << "]: no cones to fuse in" << std::endl;
          } 
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str());
          CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
          ierr = PetscSynchronizedFlush(comm);
          CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        }

        // Wait on the original sends
        // Allocate an Out status array
        MPI_Status *NeighborOut_status;
        if(NeighborCountOut) {
          ierr = PetscMalloc((NeighborCountOut)*sizeof(MPI_Status),&NeighborOut_status);CHKERROR(ierr,"Error in PetscMalloc");
          ierr = MPI_Waitall(NeighborCountOut, NeighborsOut_waits, NeighborOut_status); CHKERROR(ierr,"Error in MPI_Waitall");
        }
        
        // Computation complete; freeing memory.
        // Some of these can probably be freed earlier, if memory is needed.
        // However, be careful while freeing memory that may be in use implicitly.  
        // For instance, ConesOut is a send buffer and should probably be retained until all send requests have been waited on.
        if(NeighborCountOut){
          ierr = PetscFree(NeighborsOut_waits); CHKERROR(ierr, "Error in PetscFree");
          ierr = PetscFree(NeighborOut_status); CHKERROR(ierr, "Error in PetscFree");
        }
        if(NeighborCountIn){
          ierr = PetscFree(NeighborsIn_waits);  CHKERROR(ierr, "Error in PetscFree");
          ierr = PetscFree(NeighborIn_status); CHKERROR(ierr, "Error in PetscFree");
        }
        
        if(ConeSizeIn) {ierr = PetscFree(ConesIn);           CHKERROR(ierr, "Error in PetscFree");}
        if(ConeSizeOut){ierr = PetscFree(ConesOut);          CHKERROR(ierr, "Error in PetscFree");}
        
        // Done!  
      };// __computeFusion()

      #undef __FUNCT__
      #define __FUNCT__ "__computeFusionNew"
      template <typename Overlap_, typename Fusion_>
      static void __computeFusionNew(const Obj<graph_type>& _graph, const Obj<Overlap_>& overlap, Obj<Fusion_> fusion, const Obj<fuser_type>& fuser) {
        typedef ConeArraySequence<typename graph_type::traits::arrow_type> cone_array_sequence;
        typedef typename cone_array_sequence::cone_arrow_type              cone_arrow_type;
        MPI_Comm       comm = _graph->comm();
        int            rank = _graph->commRank();
        int            size = _graph->commSize();
        PetscObject    petscObj = _graph->petscObj();
        PetscMPIInt    tag1;
        PetscErrorCode ierr;

        Obj<typename Overlap_::traits::capSequence> overlapCap = overlap->cap();
        int msgSize = sizeof(cone_arrow_type)/sizeof(int); // Messages are arrows

        int NeighborCount = overlapCap->size();
        int *Neighbors, *NeighborByProc; // Neighbor processes and the reverse map
        int *SellSizes, *BuySizes;    // Sizes of the cones to transmit and receive
        int *SellCones = PETSC_NULL, *BuyCones = PETSC_NULL;    //
        int n, offset;
        ierr = PetscMalloc2(NeighborCount,int,&Neighbors,size,int,&NeighborByProc);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMalloc2(NeighborCount,int,&SellSizes,NeighborCount,int,&BuySizes);CHKERROR(ierr, "Error in PetscMalloc");

        n = 0;
        for(typename Overlap_::traits::capSequence::iterator neighbor = overlapCap->begin(); neighbor != overlapCap->end(); ++neighbor) {
          Neighbors[n] = *neighbor;
          NeighborByProc[*neighbor] = n;
          BuySizes[n] = 0;
          SellSizes[n] = 0;
          n++;
        }

        n = 0;
        offset = 0;
        for(typename Overlap_::traits::capSequence::iterator neighbor = overlapCap->begin(); neighbor != overlapCap->end(); ++neighbor) {
          Obj<typename Overlap_::traits::supportSequence> support = overlap->support(*neighbor);

          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            BuySizes[n] += p_iter.color().second.first;
            SellSizes[n] += p_iter.color().second.second;
            offset += _graph->cone(*p_iter)->size();
          }
          n++;
        }

        ierr = PetscMalloc(offset*msgSize * sizeof(int), &SellCones);CHKERROR(ierr, "Error in PetscMalloc");
        cone_arrow_type *ConesOut = (cone_arrow_type *) SellCones;
        offset = 0;
        for(typename Overlap_::traits::capSequence::iterator neighbor = overlapCap->begin(); neighbor != overlapCap->end(); ++neighbor) {
          Obj<typename Overlap_::traits::supportSequence> support = overlap->support(*neighbor);
          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            Obj<typename graph_type::traits::coneSequence> cone = _graph->cone(*p_iter);

            for(typename graph_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              if (debug) {
                ostringstream txt;

                txt << "["<<rank<<"]Packing arrow for " << *neighbor << "  " << *c_iter << "--" << c_iter.color() << "-->" << *p_iter << std::endl;
                ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
              }
              cone_arrow_type::place(ConesOut+offset, typename graph_type::traits::arrow_type(*c_iter, *p_iter, c_iter.color()));
              offset++;
            }
            if (p_iter.color().second.second != (int) cone->size()) {
              throw ALE::Exception("Non-matching sizes");
            }
          }
        }
        if (debug) {
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }

        // Send and retrieve cones of the base overlap
        ierr = PetscObjectGetNewTag(petscObj, &tag1); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag1, msgSize, NeighborCount, SellSizes, Neighbors, SellCones, NeighborCount, BuySizes, Neighbors, &BuyCones);

        cone_arrow_type *ConesIn = (cone_arrow_type *) BuyCones;
        offset = 0;
        for(typename Overlap_::traits::capSequence::iterator neighbor = overlapCap->begin(); neighbor != overlapCap->end(); ++neighbor) {
          Obj<typename Overlap_::traits::supportSequence> support = overlap->support(*neighbor);

          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            typename graph_type::traits::coneSequence localCone = _graph->cone(*p_iter);
            int remoteConeSize = p_iter.color().second.first;
            cone_array_sequence remoteCone(&ConesIn[offset], remoteConeSize, *p_iter);
            if (debug) {
              ostringstream txt;

              txt << "["<<rank<<"]Unpacking cone for " << *p_iter << std::endl;
              remoteCone.view(txt, true);
              ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
            }
            // Fuse in received cones
            fuser->fuseCones(localCone, remoteCone, fusion->cone(fuser->fuseBasePoints(*p_iter, *p_iter)));
            offset += remoteConeSize;
          }
        }
        if (debug) {
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }
      };

      #undef __FUNCT__
      #define __FUNCT__ "__computeFusionNew"
      template <typename Overlap_, typename Fusion_>
      static void __computeFusionNew(const Obj<graph_type>& _graphA, const Obj<graph_type>& _graphB, const Obj<Overlap_>& overlap, Obj<Fusion_> fusion, const Obj<fuser_type>& fuser) {
        typedef ConeArraySequence<typename graph_type::traits::arrow_type> cone_array_sequence;
        typedef typename cone_array_sequence::cone_arrow_type              cone_arrow_type;
        MPI_Comm       comm = _graphA->comm();
        int            rank = _graphA->commRank();
        PetscObject    petscObj = _graphA->petscObj();
        PetscMPIInt    tag1;
        PetscErrorCode ierr;

        Obj<typename Overlap_::traits::capSequence> overlapCap = overlap->cap();
        int msgSize = sizeof(cone_arrow_type)/sizeof(int); // Messages are arrows

        int NeighborCountA = 0, NeighborCountB = 0;
        for(typename Overlap_::traits::capSequence::iterator neighbor = overlapCap->begin(); neighbor != overlapCap->end(); ++neighbor) {
          Obj<typename Overlap_::traits::supportSequence> support = overlap->support(*neighbor);

          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            if ((*p_iter).first == 0) {
              NeighborCountA++;
              break;
            }
          }
          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            if ((*p_iter).first == 1) {
              NeighborCountB++;
              break;
            }
          } 
        }

        int *NeighborsA, *NeighborsB; // Neighbor processes
        int *SellSizesA, *BuySizesA;  // Sizes of the A cones to transmit and B cones to receive
        int *SellSizesB, *BuySizesB;  // Sizes of the B cones to transmit and A cones to receive
        int *SellConesA = PETSC_NULL, *BuyConesA = PETSC_NULL;
        int *SellConesB = PETSC_NULL, *BuyConesB = PETSC_NULL;
        int nA, nB, offsetA, offsetB;
        ierr = PetscMalloc2(NeighborCountA,int,&NeighborsA,NeighborCountB,int,&NeighborsB);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMalloc2(NeighborCountA,int,&SellSizesA,NeighborCountA,int,&BuySizesA);CHKERROR(ierr, "Error in PetscMalloc");
        ierr = PetscMalloc2(NeighborCountB,int,&SellSizesB,NeighborCountB,int,&BuySizesB);CHKERROR(ierr, "Error in PetscMalloc");

        nA = 0;
        nB = 0;
        for(typename Overlap_::traits::capSequence::iterator neighbor = overlapCap->begin(); neighbor != overlapCap->end(); ++neighbor) {
          Obj<typename Overlap_::traits::supportSequence> support = overlap->support(*neighbor);

          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            if ((*p_iter).first == 0) {
              NeighborsA[nA] = *neighbor;
              BuySizesA[nA] = 0;
              SellSizesA[nA] = 0;
              nA++;
              break;
            }
          }
          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            if ((*p_iter).first == 1) {
              NeighborsB[nB] = *neighbor;
              BuySizesB[nB] = 0;
              SellSizesB[nB] = 0;
              nB++;
              break;
            }
          } 
        }
        if ((nA != NeighborCountA) || (nB != NeighborCountB)) {
          throw ALE::Exception("Invalid neighbor count");
        }

        nA = 0;
        offsetA = 0;
        nB = 0;
        offsetB = 0;
        for(typename Overlap_::traits::capSequence::iterator neighbor = overlapCap->begin(); neighbor != overlapCap->end(); ++neighbor) {
          Obj<typename Overlap_::traits::supportSequence> support = overlap->support(*neighbor);
          int foundA = 0, foundB = 0;

          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            if ((*p_iter).first == 0) {
              BuySizesA[nA] += p_iter.color().second.first;
              SellSizesA[nA] += p_iter.color().second.second;
              offsetA += _graphA->cone((*p_iter).second)->size();
              foundA = 1;
            } else {
              BuySizesB[nB] += p_iter.color().second.first;
              SellSizesB[nB] += p_iter.color().second.second;
              offsetB += _graphB->cone((*p_iter).second)->size();
              foundB = 1;
            }
          }
          if (foundA) nA++;
          if (foundB) nB++;
        }

        ierr = PetscMalloc2(offsetA*msgSize,int,&SellConesA,offsetB*msgSize,int,&SellConesB);CHKERROR(ierr, "Error in PetscMalloc");
        cone_arrow_type *ConesOutA = (cone_arrow_type *) SellConesA;
        cone_arrow_type *ConesOutB = (cone_arrow_type *) SellConesB;
        offsetA = 0;
        offsetB = 0;
        for(typename Overlap_::traits::capSequence::iterator neighbor = overlapCap->begin(); neighbor != overlapCap->end(); ++neighbor) {
          Obj<typename Overlap_::traits::supportSequence> support = overlap->support(*neighbor);

          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            Obj<typename graph_type::traits::coneSequence> cone;
            const Point& p = (*p_iter).second;

            if ((*p_iter).first == 0) {
              cone = _graphA->cone(p);
              for(typename graph_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
                if (debug) {
                  ostringstream txt;

                  txt << "["<<rank<<"]Packing A arrow for " << *neighbor << "  " << *c_iter << "--" << c_iter.color() << "-->" << p << std::endl;
                  ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
                }
                cone_arrow_type::place(ConesOutA+offsetA, typename graph_type::traits::arrow_type(*c_iter, p, c_iter.color()));
                offsetA++;
              }
            } else {
              cone = _graphB->cone(p);
              for(typename graph_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
                if (debug) {
                  ostringstream txt;

                  txt << "["<<rank<<"]Packing B arrow for " << *neighbor << "  " << *c_iter << "--" << c_iter.color() << "-->" << p << std::endl;
                  ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
                }
                cone_arrow_type::place(ConesOutB+offsetB, typename graph_type::traits::arrow_type(*c_iter, p, c_iter.color()));
                offsetB++;
              }
            }
            if (p_iter.color().second.second != (int) cone->size()) {
              std::cout << "["<<rank<<"] " << p_iter.color() << " does not match cone size " << cone->size() << std::endl;
              throw ALE::Exception("Non-matching sizes");
            }
          }
        }
        if (debug) {
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }

        // Send and retrieve cones of the base overlap
        ierr = PetscObjectGetNewTag(petscObj, &tag1); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag1, msgSize, NeighborCountA, SellSizesA, NeighborsA, SellConesA, NeighborCountA, BuySizesA, NeighborsA, &BuyConesA);
        commCycle(comm, tag1, msgSize, NeighborCountB, SellSizesB, NeighborsB, SellConesB, NeighborCountB, BuySizesB, NeighborsB, &BuyConesB);

        // Must unpack with the BtoA overlap
        cone_arrow_type *ConesInA = (cone_arrow_type *) BuyConesA;
        cone_arrow_type *ConesInB = (cone_arrow_type *) BuyConesB;
        offsetA = 0;
        offsetB = 0;
        for(typename Overlap_::traits::capSequence::iterator neighbor = overlapCap->begin(); neighbor != overlapCap->end(); ++neighbor) {
          Obj<typename Overlap_::traits::supportSequence> support = overlap->support(*neighbor);

          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            Obj<typename graph_type::traits::coneSequence> localCone;
            const Point& p = (*p_iter).second;
            int remoteConeSize = p_iter.color().second.first;

            // Right now we only provide the A->B fusion
            if ((*p_iter).first == 0) {
#if 0
              cone_array_sequence remoteCone(&ConesInB[offsetB], remoteConeSize, p);

              localCone = _graphA->cone(p);
              offsetB += remoteConeSize;
              if (debug) {
                ostringstream txt;

                txt << "["<<rank<<"]Unpacking B cone for " << p << " from " << *neighbor << std::endl;
                remoteCone.view(txt, true);
                ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
              }
              // Fuse in received cones
              fuser->fuseCones(localCone, remoteCone, fusion->cone(fuser->fuseBasePoints(p, p)));
#endif
            } else {
              cone_array_sequence remoteCone(&ConesInA[offsetA], remoteConeSize, p);

              localCone = _graphB->cone(p);
              offsetA += remoteConeSize;
              if (debug) {
                ostringstream txt;

                txt << "["<<rank<<"]Unpacking A cone for " << p <<  " from " << *neighbor << std::endl;
                remoteCone.view(txt, true);
                ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
              }
              // Fuse in received cones
              fuser->fuseCones(localCone, remoteCone, fusion->cone(fuser->fuseBasePoints(p, p)));
            }
          }
        }
        if (debug) {
          ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
        }
      };

    public:
      static void setDebug(int debug) {ParConeDelta::debug = debug;};
      static int  getDebug() {return ParConeDelta::debug;};
    }; // class ParConeDelta
  
    template <typename ParSifter_, typename Fuser_, typename FusionSifter_>
    int ParConeDelta<ParSifter_, Fuser_, FusionSifter_>::debug = 0;
    

    //
    // Auxiliary type
    //
    template <typename Sifter_>
    class Flip { // class Flip
    public:
      typedef Sifter_       graph_type;
      typedef Flip<Sifter_> flip_type;
    protected:
      Obj<graph_type> _graph;
    public:
      //
      struct traits {
        // Basic types
        typedef typename graph_type::traits::arrow_type::flip::type                 arrow_type;
        typedef typename arrow_type::source_type                                    source_type;
        typedef typename arrow_type::target_type                                    target_type;
        typedef typename arrow_type::color_type                                     color_type;
        // Sequences
        // Be careful: use only a limited set of iterator methods: NO arrow(), source(), target() etc; operator*() and color() are OK.
        typedef typename graph_type::traits::coneSequence                           supportSequence;
        typedef typename graph_type::traits::supportSequence                        coneSequence;
        typedef typename graph_type::traits::baseSequence                           capSequence;
        typedef typename graph_type::traits::capSequence                            baseSequence;
      };
      // Basic interface
      Flip(const Obj<graph_type>& graph) : _graph(graph) {};
      Flip(const Flip& flip) : _graph(flip._graph) {};
      virtual ~Flip() {};
      // Redirect 
      // Only a limited set of methods is redirected: simple cone, support, base, cap and arrow insertion.
      //
      // Query methods
      //
      MPI_Comm    comm()     const {return this->_graph->comm();};
      int         commSize() const {return this->_graph->commSize();};
      int         commRank() const {return this->_graph->commRank();}
      PetscObject petscObj() const {return this->_graph->petscObj();};
      
      // FIX: need const_cap, const_base returning const capSequence etc, but those need to have const_iterators, const_begin etc.
      Obj<typename traits::capSequence> cap() {
        return this->_graph->base();
      };
      Obj<typename traits::baseSequence> base() {
        return this->_graph->cap();
      };
      
      Obj<typename traits::coneSequence> 
      cone(const typename traits::target_type& p) {
        return this->_graph->support(p);
      };
      
      Obj<typename traits::coneSequence> 
      cone(const typename traits::target_type& p, const typename traits::color_type& color) {
        return this->_graph->support(p, color);
      };
      
      Obj<typename traits::supportSequence> 
      support(const typename traits::source_type& p) {
        return this->_graph->cone(p);
      };
      
      Obj<typename traits::supportSequence> 
      support(const typename traits::source_type& p, const typename traits::color_type& color) {
        return this->_graph->cone(p,color);
      };
      
      virtual void addArrow(const typename traits::source_type& p, const typename traits::target_type& q) {
        this->_graph->addArrow(q, p);
      };
      
      virtual void addArrow(const typename traits::source_type& p, const typename traits::target_type& q, const typename traits::color_type& color) {
        this->_graph->addArrow(q, p, color);
      };
      
      virtual void addArrow(const typename traits::arrow_type& a) {
        this->_graph->addArrow(a.target, a.source, a.color);
      };
      
    };// class Flip


    // WARNING: must pass in a 'flipped' Fuser, that is a fuser that acts on cones instead of supports 
    template<typename ParSifter_,
             typename Fuser_ = RightSequenceDuplicator<ConeArraySequence<typename ParSifter_::traits::arrow_type::flip::type> >,
             typename FusionSifter_ = typename ParSifter_::template rebind<typename Fuser_::fusion_target_type, 
                                                                             typename Fuser_::fusion_source_type, 
                                                                             typename Fuser_::fusion_color_type>::type>    
    class ParSupportDelta {
    public:
      // Here we specialize to Sifters based on Points in order to enable parallel overlap discovery.
      // We also assume that the Points in the base are ordered appropriately so we can use baseSequence.begin() and 
      // baseSequence.end() as the extrema for global reduction.
      typedef ParSupportDelta<ParSifter_, Fuser_, FusionSifter_>                                delta_type;
      typedef ParSifter_                                                                        graph_type;
      typedef Fuser_                                                                            fuser_type;
      typedef ASifter<ALE::Point, int, ALE::pair<ALE::Point, ALE::pair<int,int> >, uniColor>    overlap_type;
      typedef ASifter<ALE::pair<int,ALE::Point>, int, ALE::pair<ALE::Point, ALE::pair<int,int> >, uniColor>    bioverlap_type;
      typedef FusionSifter_                                                                     fusion_type;
      //

      //
      // FIX: Is there a way to inherit this from ParConeDelta?  Right now it is a verbatim copy.
      static Obj<overlap_type> 
      overlap(const Obj<graph_type> graph) {
        Obj<overlap_type> overlap = overlap_type(graph->comm());
        // If this is a serial object, we return an empty overlap
        if((graph->comm() != PETSC_COMM_SELF) && (graph->commSize() > 1)) {
          computeOverlap(graph, overlap);
        }
        return overlap;
      };

      template <typename Overlap_>
      static void computeOverlap(const Obj<graph_type>& graph, Obj<Overlap_>& overlap){
        // Flip the graph and the overlap and use ParConeDelta's method
        Obj<Flip<graph_type> >   graph_flip   = Flip<graph_type>(graph);
        Obj<Flip<Overlap_> > overlap_flip     = Flip<Overlap_>(overlap);
        ParConeDelta<Flip<graph_type>, fuser_type, Flip<fusion_type> >::computeOverlap(graph_flip, overlap_flip);
      };

      static Obj<bioverlap_type> 
      overlap(const Obj<graph_type> graphA, const Obj<graph_type> graphB) {
        Obj<bioverlap_type> overlap = bioverlap_type(graphA->comm());
        PetscMPIInt         comp;

        MPI_Comm_compare(graphA->comm(), graphB->comm(), &comp);
        if (comp != MPI_IDENT) {
          throw ALE::Exception("Non-matching communicators for overlap");
        }
        Obj<Flip<graph_type> >   graphA_flip   = Flip<graph_type>(graphA);
        Obj<Flip<graph_type> >   graphB_flip   = Flip<graph_type>(graphB);
        Obj<Flip<bioverlap_type> > overlap_flip     = Flip<bioverlap_type>(overlap);

        ParConeDelta<Flip<graph_type>, fuser_type, Flip<fusion_type> >::computeOverlap(graphA_flip, graphB_flip, overlap_flip);
        return overlap;
      };

      // FIX: Is there a way to inherit this from ParConeDelta?  Right now it is a verbatim copy.
      template <typename Overlap_>
      static Obj<fusion_type> 
      fusion(const Obj<graph_type>& graph, const Obj<Overlap_>& overlap, const Obj<fuser_type>& fuser = fuser_type()) {
        Obj<fusion_type> fusion = fusion_type(graph->comm());
        // If this is a serial object, we return an empty delta
        if((graph->comm() != PETSC_COMM_SELF) && (graph->commSize() > 1)) {
          computeFusion(graph, overlap, fusion, fuser);
        }
        return fusion;
      };

      template <typename Overlap_, typename Fusion_>
      static void computeFusion(const Obj<graph_type>& graph, const Obj<Overlap_>& overlap, Obj<Fusion_> fusion, const Obj<fuser_type>& fuser = fuser_type()){
        // Flip the graph, the overlap and the fusion, and the use ParConeDelta's method
        Obj<Flip<graph_type> > graph_flip   = Flip<graph_type>(graph);
        Obj<Flip<Overlap_> >   overlap_flip = Flip<Overlap_>(overlap);
        Obj<Flip<Fusion_> >    fusion_flip  = Flip<Fusion_>(fusion);
        ParConeDelta<Flip<graph_type>, fuser_type, Flip<fusion_type> >::computeFusion(graph_flip, overlap_flip, fusion_flip);
      };      
    public:
      static void setDebug(int debug) {ParConeDelta<Flip<graph_type>, fuser_type, Flip<fusion_type> >::setDebug(debug);};
      static int  getDebug() {return ParConeDelta<Flip<graph_type>, fuser_type, Flip<fusion_type> >::getDebug();};
    }; // class ParSupportDelta
  
} // namespace ALE

#endif
