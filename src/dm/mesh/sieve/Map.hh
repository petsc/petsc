#ifndef included_ALE_Map_hh
#define included_ALE_Map_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif



//
// Classes and methods implementing  the parallel Overlap and Fusion algorithms on ASifter-like objects.
//
namespace ALE {
  namespace X {

    template <typename Point_, typename Chart_, typename Index_>
    class Atlas : public ASifter<Point_, Chart_, typename ALE::pair<Index_,Index_>, SifterDef::uniColor> {
    public:
      // 
      // Encapsulated types
      //
      typedef Point_  point_type;
      typedef Chart_  chart_type;
      typedef Index_  index_type;
      typedef ASifter<Point_, Chart_, typename ALE::pair<int,int>, SifterDef::uniColor> sifter_type;
    public:
      //
      // Basic interface
      //
      Atlas(MPI_Comm comm = PETSC_COMM_SELF, const int& debug = 0) : sifter_type(comm, debug) {};
      virtual ~Atlas(){};
      //
      // Extended interface
      //
      index_type size(){
        index_type sz = 0;
        // Here we simply look at each chart's cone and add up the sizes
        // In fact, we assume that within a chart indices are contiguous, 
        // so we simply add up the offset to the size of the last point of each chart's cone and sum them up.
        baseSequence base = this->base();
        for(typename base::iterator bitor = base->begin(); bitor != base->end(); bitor++) {
          ALE::pair<index_type,index_type> ii = this->cone(*bitor)->rbegin()->color();
          sz += ii.first + ii.second;
        }
      };
    };// class Atlas

    template <typename Atlas_, typename Data_>
    class CoSifter {
    public:
      // 
      // Encapsulated types
      //
      typedef Atlas_                           atlas_type;
      typedef Data_                            data_type;      
      typedef typename atlas_type::point_type  point_type;
      typedef typename atlas_type::chart_type  chart_type;
      typedef typename atlas_type::index_type  index_type;
      //
      // Perhaps the most important incapsulated type: sequence of data elements over a sequence of AtlasArrows.
      // of the sequence.
      template <typename AtlasArrowSequence_>
      class DataSequence {
      // The AtlasArrowSequence_ encodes the arrows over a chart or a chart-point pair.
      // The crucial assumption is that the begin() and rbegin() of the AtlasArrowSequence_ contain the extremal indices
      public:
        //
        // Encapsulated types
        // 
        typedef AtlasArrowSequence_ atlas_arrow_sequence_type;
        //
        // Encapsulated iterators
        class iterator {
        public:
          typedef std::input_iterator_tag     iterator_category;
          typedef data_type                   value_type;
          typedef int                         difference_type;
          typedef value_type*                 pointer;
          typedef value_type&                 reference;
        protected:
          // Encapsulates a data_type pointer
          data_type* _ptr;
        public:
          iterator(const iterator& iter) : _ptr(iter._ptr) {};
          iterator(data_type* ptr)       : _ptr(ptr) {};
          data_type&         operator*(){return *(this->_ptr);};
          virtual iterator   operator++() {++this->_ptr; return *this;};
          virtual iterator   operator++(int n) {iterator tmp(this->_ptr); ++this->_ptr; return tmp;};
          virtual bool       operator==(const iterator& iter) const {return this->_ptr == iter._ptr;};
          virtual bool       operator!=(const iterator& iter) const {return this->_ptr != iter._ptr;};
        };
        //
        class reverse_iterator {
        public:
          typedef std::input_iterator_tag     iterator_category;
          typedef data_type                   value_type;
          typedef int                         difference_type;
          typedef value_type*                 pointer;
          typedef value_type&                 reference;
        protected:
          // Encapsulates a data_type pointer
          data_type* _ptr;
        public:
          reverse_iterator(const reverse_iterator& iter) : _ptr(iter._ptr) {};
          reverse_iterator(data_type* ptr)       : _ptr(ptr) {};
          data_type&                operator*(){return *(this->_ptr);};
          virtual reverse_iterator  operator++() {--this->_ptr; return *this;};
          virtual reverse_iterator  operator++(int n) {reverse_iterator tmp(this->_ptr); --this->_ptr; return tmp;};
          virtual bool              operator==(const reverse_iterator& iter) const {return this->_ptr == iter._ptr;};
          virtual bool              operator!=(const reverse_iterator& iter) const {return this->_ptr != iter._ptr;};
        };
      protected:
        const atlas_arrow_sequence_type& _arrows;
        data_type *_base_ptr;
        index      _size;
      public:
        //
        // Basic interface
        DataSequence(data_type *arr, const atlas_arrow_sequence_type& arrows) : _arrows(arrows) {
          // We immediately calculate the base pointer into the array and the size of the data sequence.
          // To compute the index of the base pointer look at the beginning of the _arrows sequence.
          this->_base_ptr = arr + this->_arrows->begin()->color().first;
          // To compute the total size of the array segement, we look at the end of the _arrows sequence.
          ALE::pair<index_type, index_type> ii = this->_arrows->rbegin()->color();
          this->_size = ii.first + ii.second;
        };
       ~DataSequence(){};
        // 
        // Extended interface
        index_type size() {return this->_size;};
        iterator begin()  {return iterator(this->_base_ptr);};
        iterator end()    {return iterator(this->_base_ptr+this->_size+1);};
        iterator rbegin() {return reverse_iterator(this->_base_ptr+this->_size);};
        iterator rend()   {return reverse_iterator(this->_base_ptr-1);};
      }; // class CoSifter::DataSequence
    protected:
      Obj<atlas_type> _atlas;
      data_type*      _data;
      bool            _allocated;
    public:
      //
      // Basic interface
      //
      CoSifter(const Obj<atlas_type> atlas, const (data_type*)& data) {this->setAtlas(atlas, false); this->_data = data;};
      CoSifter(const Obj<atlas_type> atlas = Obj<atlas_type>()) {this->_data = NULL; this->setAtlas(atlas);};
      ~CoSifter(){if((this->_data != NULL)&&(this->_allocated)) {ierr = PetscFree(this->_data); CHKERROR(ierr, "Error in PetscFree");}};
      //
      // Extended interface
      //
      void setAtlas(const Obj<atlas_type>& atlas, bool allocate = true) {
        if(!this->_atlas.isNull()) {
          throw ALE::Exception("Cannot reset nonempty atlas"); 
        }
        else {
          if(atlas.isNull()) {
            throw ALE::Exception("Cannot set a nonempty atlas");  
          }
          else {
            this->_atlas = atlas;
            this->_allocated = allocate;
            if(allocate) {
              // Allocate data
              ierr = PetscMalloc(this->_atlas->size()*sizeof(data_type), &this->_data); CHKERROR(ierr, "Error in PetscMalloc");
            }
          }
        }
      };// setAtlas()
      Obj<atlas_type> getAtlas() {return this->_atlas;};
      Obj<atlas_type> atlas()    {return getAtlas();};
      //
      DataSequence<typename atlas_type::coneSequence> 
      restrict(const chart_type& chart) { 
        return DataSequence<typename atlas_type::coneSequence>(this->_data, this->_atlas->cone(chart));
      };
      DataSequence<typename atlas_type::arrowSequence> 
      restrict(const chart_type& chart, const point_type& point) { 
        return DataSequence<typename atlas_type::coneSequence>(this->_data, this->_atlas->arrows(point, chart));
      };
    };// class CoSifter



    template <typename InAtlas_, typename OutAtlas_, typename Data_, typename Crossbar_>
    class Map { // class Map
    public:
      //
      // Encapsulated types
      // 
      // An InAtlas_ (and likewise for an OutAtlas_) has arrows from points to charts decorated with indices
      // into Data_ storage of the corresponding CoSifter.
      typedef InAtlas_   in_atlas_type;
      typedef OutAtlas_  out_atlas_type;
      typedef Crossbar_  crossbar_type;
      typedef Data_      data_type;
      //
      // A ProjectionAtlas/FusionAtlas has the same types for points and colors as the In/OutAtlas_, 
      // but its charts are prefixed with the process rank and the color carries indices into different Data_ storage -- 
      // that encapsulated in Map.
      //
      typedef ASifter<typename in_atlas_type::source_type, 
                      typename in_atlas_type::color_type, 
                      typename ALE::pair<MPI_Int, in_atlas_type::target_type>, 
                      SifterDef::uniColor>                                         fusion_atlas_type;
      typedef ASifter<typename out_atlas_type::source_type, 
                      typename out_atlas_type::color_type, 
                      typename ALE::pair<MPI_Int, out_atlas_type::target_type>, 
                      SifterDef::uniColor>                                         projection_atlas_type;
      
    protected:
      Obj<in_atlas_type>         _in_atlas;
      Obj<out_atlas_type>        _out_atlas;
      Obj<crossbar_type>         _crossbar;
      //
      Obj<projection_atlas_type> _projection_atlas;
      Obj<fusion_atlas_type>     _fusion_atlas;
    public:
      //
      // Basic interface
      //
      Map(const Obj<in_atlas_type>& in_atlas, const Obj<out_atlas_type>& out_atlas, 
          const Obj<crossbar_type>& crossbar = Obj<crossbar_type>()): _in_atlas(in_atlas), _out_atlas(out_atlas), _crossbar(crossbar) 
      {};
      ~Map(){};
      
      //
      // Extended interface
      //
      int  getDebug() {return this->debug;}
      void setDebug(const int& d) {this->debug = d;};

      // THE FOLLOWING RELATES TO OVERLAP/LIGHT-CONE COMPUTATION:
      // Here we specialize to Atlases with int like-prefixed source points to enable parallel overlap discovery.
      // We also assume that the atlas bases are ordered appropriately so we can use baseSequence.begin() and 
      // baseSequence.end() as the extrema for global reduction.
      #undef __FUNCT__
      #define __FUNCT__ "Map::computeOverlap"
      void computeOverlap(const Obj<graph_type> graphA, const Obj<graph_type> graphB) {
        ALE_LOG_EVENT_BEGIN;
        Obj<bioverlap_type> overlap = bioverlap_type(graphA->comm());
        PetscMPIInt         comp;

        MPI_Comm_compare(graphA->comm(), graphB->comm(), &comp);
        if (comp != MPI_IDENT) {
          throw ALE::Exception("Non-matching communicators for overlap");
        }
        __computeOverlapNew(graphA, graphB, overlap);
        ALE_LOG_EVENT_END;
      };
    protected:
      int                debug;
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


      #undef  __FUNCT__
      #define __FUNCT__ "__computeOverlapNew"
      template <typename Overlap_>
      static void __computeOverlapNew(const Obj<graph_type>& _graphA, const Obj<graph_type>& _graphB, Obj<Overlap_>& overlap) {
        typedef typename graph_type::traits::baseSequence Sequence;
        MPI_Comm       comm = _graphA->comm();
        int            size = _graphA->commSize();
        int            rank = _graphA->commRank();
        PetscObject    petscObj = _graphA->petscObj();
        PetscMPIInt    tag1, tag2, tag3, tag4, tag5, tag6;
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
        for(int b = 0, o = 0; b < BuyCountA; ++b) {
          if (offsetsA[SellersA[b]] - o != msgSize*BuySizesA[b]) {
            throw ALE::Exception("Invalid A point size");
          }
          o += msgSize*BuySizesA[b];
        }
        for(int b = 0, o = 0; b < BuyCountB; ++b) {
          if (offsetsB[SellersB[b]] - o != msgSize*BuySizesB[b]) {
            throw ALE::Exception("Invalid B point size");
          }
          o += msgSize*BuySizesB[b];
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
        ierr = PetscObjectGetNewTag(petscObj, &tag2); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag2, msgSize, BuyCountB, BuySizesB, SellersB, BuyPointsB, SellCountB, SellSizesB, BuyersB, &SellPointsB);

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
        ierr = PetscObjectGetNewTag(petscObj, &tag3); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag3, msgSize, SellCountA, SellSizesA, BuyersA, SellPointsA, BuyCountA, BuySizesA, SellersA, &BuyPointsA);
        ierr = PetscObjectGetNewTag(petscObj, &tag4); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag4, msgSize, SellCountB, SellSizesB, BuyersB, SellPointsB, BuyCountB, BuySizesB, SellersB, &BuyPointsB);

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
        ierr = PetscObjectGetNewTag(petscObj, &tag5); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag5, cMsgSize, SellCountA, SellConesSizesA, BuyersA, SellConesA, BuyCountA, BuyConesSizesA, SellersA, &overlapInfoA);
        ierr = PetscObjectGetNewTag(petscObj, &tag6); CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag6, cMsgSize, SellCountB, SellConesSizesB, BuyersB, SellConesB, BuyCountB, BuyConesSizesB, SellersB, &overlapInfoB);

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
      #define __FUNCT__ "commCycle"
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
          // Stupid, stupid, stupid fucking MPICH fails with 0-length storage
          ierr = PetscMalloc(PetscMax(1, msgSize*SellSize) * sizeof(int32_t), &locSellData);CHKERROR(ierr,"Error in PetscMalloc");
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
      }// commCycle()



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
            if (((*p_iter).first == 0) && (p_iter.color().second.first || p_iter.color().second.second)) {
              NeighborCountA++;
              break;
            }
          }
          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            if (((*p_iter).first == 1) && (p_iter.color().second.first || p_iter.color().second.second)) {
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
            if (((*p_iter).first == 0) && (p_iter.color().second.first || p_iter.color().second.second)) {
              NeighborsA[nA] = *neighbor;
              BuySizesA[nA] = 0;
              SellSizesA[nA] = 0;
              nA++;
              break;
            }
          }
          for(typename Overlap_::traits::supportSequence::iterator p_iter = support->begin(); p_iter != support->end(); ++p_iter) {
            if (((*p_iter).first == 1) && (p_iter.color().second.first || p_iter.color().second.second)) {
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
            if (((*p_iter).first == 0) && (p_iter.color().second.first || p_iter.color().second.second)) {
              BuySizesA[nA] += p_iter.color().second.first;
              SellSizesA[nA] += p_iter.color().second.second;
              offsetA += _graphA->cone((*p_iter).second)->size();
              foundA = 1;
            } else if (((*p_iter).first == 1) && (p_iter.color().second.first || p_iter.color().second.second)) {
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
        commCycle(comm, tag1, msgSize, NeighborCountA, SellSizesA, NeighborsA, SellConesA, NeighborCountB, BuySizesB, NeighborsB, &BuyConesB);
        commCycle(comm, tag1, msgSize, NeighborCountB, SellSizesB, NeighborsB, SellConesB, NeighborCountA, BuySizesA, NeighborsA, &BuyConesA);

        // Must unpack with the BtoA overlap
        //cone_arrow_type *ConesInA = (cone_arrow_type *) BuyConesA;
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
              cone_array_sequence remoteCone(&ConesInA[offsetA], remoteConeSize, p);

              localCone = _graphA->cone(p);
              offsetA += remoteConeSize;
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
              cone_array_sequence remoteCone(&ConesInB[offsetB], remoteConeSize, p);

              localCone = _graphB->cone(p);
              offsetB += remoteConeSize;
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
    }; // class Map
  

} // namespace X  
} // namespace ALE

#endif
