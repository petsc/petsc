#ifndef included_ALE_Map_hh
#define included_ALE_Map_hh

#ifndef  included_ALE_Sifter_hh
#include <Sifter.hh>
#endif


// Concepts.
// -----------
// Because of the use of templating and generic programming techniques,
// many fundamental ALE types cannot be defined by making progressively specific 
// class declarations within a given hierarchy. Instead, they must satisfy certain
// Concept requirements that make them acceptable inputs to various Algorithms.
// This reflects a shortcoming of the current compiler technology that some defining
// features of Concepts have to be specified in documentation, rather than within the 
// language.  Sometimes, however, conceptual types can be viewed as themselves 
// encapsulating Algorithms acting on other types implementing certain concepts.
// This allows to define the structure of these algorithmic types using generic 
// programming techniques available in C++, for example.


// Atlas & Sec
// -----------------------
// In the past we have considered the Atlas concept, which, for the given
// Sifter, chart and ind types computed the assignment of indices
// to the points in the underlying Sieve (relative to a given chart).
// This mimics a system of local coordinate systems on a manifold or some
// such space. 

// Essentially, Atlas can be thought of as a Sifter, with the 
// underlying Sifter's points (base or cap?) in the cap, charts in the base
// and indices being the color on the edges.  However, any type that responds
// to the fundamental requests -- setting the number of indices over a point 
// (fiber dimension), (re)ordering the indices after some dimension modifications
// have been made, and retrieving the indices -- is an Atlas.

// An object that assigns data of a given type to the (point, chart) pairs of an Atlas 
// is called its section or Sec.  If an Atlas is viewed as a discrete model of a structure
// bundle over the point space, a Sec is a discrete model of a section of that bundle.
// Sec is required to define a restrict method, which, given a (point,chart) pair returns
// a data iterator.  

// If the Sifter underling the Atlas is a Sieve, we assume that to each
// covering arrow p --> q (within the same chart), corresponds a mapping of the data d(p) <-- d(q),
// reflecting the idea that d(q) can be 'restricted' to d(p) within the same chart (or perhaps 
// in any chart?).  This sort of behavior is certainly impossible to guarantee through
// an interface specification, but it remains a conceptual requirement on Sec that 
// the data over p are somehow "included" in the data over q.  This "inclusion" is partly specified
// in the Atlas (e.g., by ensuring that the indices over p are included in those over q),
// and partly in Sec itself.


// Map & ParMap concepts.
// -----------------------
// A Map is thought of as a type of mapping from a Sec class to another Sec class.
// Maps and ParMaps can be viewed as algorithmic types acting on the input/output Sec types.  
// Most importantly, a Map must advertise the Atlases of the input and output Sec
// types.  Furthermore, a Map acts essentially as a Sec relative to the output
// atlas: giving a fixed input Sec S, a Map extends the Sec interface by defining restrictions 
// to output  (point,chart) pairs relative to S.  Alternatively, a Map can return an output Sec T,
// containing the result of applying the map to S, which can be queried independently.

// With these features of a Map hardly define any implementation structure (rather an interface), 
// since the particular behavior of restrictions is unconstrained by this specification.  
// Particular Maps can impose further constraints on the input/output atlases
// and expand the interface, so long as they conform to the Map concept outlined above.
// For example, we can require that each output chart data depend only on the explicitly specified 
// input charts' data.  This is done by specifying the 'lightcone' Sifter or a Sieve, connecting 
// the input charts to the output charts they depend on.  Taking the cone (or the closure, if it's 
// a Sieve) of a given output chart in the lightcone returns all of the input charts necessary
// to compute the data over the output chart.

// The specification of a Map's lightcone is very necessary to enable preallocation of internal 
// data structures, such as the matrix storage, for linear maps.  In the distributed context
// it also enables the setup of communication structures.  This behavior is encapsulated in 
// a ParMap, which is itself a conceptual type that extends Map.  ParMap encapsulates (among other things) 
// three conceptual Map objects: Gather,Scatter and the Transform. A ParMap is then an algorithm that orchestrates 
// the action of other maps.

// ***
//   The ParMap algorithm is a most clear illustration of the locazation principle underlying Sieves and computation over
// them: restrict-compute-assemble.  To compute the action of a ParMap on a distributed Sec the necessary data must be
// communicate to and from the processes requiring and holding the data.  The total overlap of a processes domain with the
// rest of the communicator is naturally covered by the individual overlaps indexed by the remote ranks. 
//   To communicate the local input Sec data to the remote processes, the Sec is first restricted to each of the overlap pieces, 
// forming another Sec, whose charts are the overlaps indexed by indices, and whose points are the (in_point,in_chart)
// pairs on which in the input Sec is defined.  This Sec can be viewed as multisheeted coverings of the overlap porition
// of the input Sec, and the multiplexing process forming the new Sec will be called Scatter.  It is a map between two Secs
// with the input atlas and the new rank-indexed atlas.
//   After Scatter maps the input Sec to a multisheeted covering, the multisheeted data are communicated to the processes according
// to the rank in each chart. This can be viewed as a map between two such Secs -- communication is certainly a mapping in
// the distributed context -- done 'locally' over each chart.  Thus, the data from the input Sec are first localized onto
// each rank-chart, then mapped, and finally must be assembled. 
//   To obtain a Sec over the local domain, the Scatter process must be reversed, using the a map called Gather.  Gather
// takes in the multisheeted covering of the overlap obtained after the communication phase, and obtains a single
// (in_point,in_chart) data point from the collection of all such points over all the rank charts.  During this reduction
// the overlap portion of the communicated input Sec is unified with the local data over the same (in_point,in_chart) pair,
// completing the assembly of the input Sec.
//   Once the input data have been communicate to the consuming processes, Transform locally maps the data from the input
// Sec to the output Sec.  The Gather/Scatter maps involved in the communication stage depend on the GatherAtlas, which
// describes the multisheeted covering of InAtlas.  Gather/Scatter maps can in principle operate on a Sec of any Atlas,
// unifying the data over the same point in different charts, producing a Sec over a single-charted atlas.  Different
// implementations of Gather/Scatter lead to different multiplexing/reduction procedures, while the atlas structure stays
// the same.  Gather/Scatter maps can be used locally as well and need not act on the result of a communication.
// ***

// The Gather output atlas has the same structure as the ParMap input atlas,
// while the Gather input atlas  -- GatherAtlas -- combines the ParMap input (point,chart) pairs into 
// the source and puts the communicator rank in the target. The Scatter input/output atlases have the 
// structure of the output/input atlases of Gather respectively. The Transform atlases have the same 
// structure as the ParMap atlases.
//                   (Gather input)                  (Gather output == Transform input)            
                                       
//                           index                            index                       
//             (point,chart) -----> rank     <==>       point -----> chart   

//                   (Scatter output)                (Scatter input == Transform input)            

// GatherAtlas is constructed from ParMap's input atlas using the lightcone.
// The Gather input/Scatter output atlas essentially applies the idea of a chart recursively:
// Transform input charts are distributed among different processes.  Given a single process, its overlap
// with other processes can be indexed by their remote communicator ranks.  All (point_in,chart_in) pairs shared
// with a given rank are part of a single rank-chart.  This way a single in-chart is "blown up"
// into a "multisheeted" covering by rank-charts; each (point_in,chart_in) pair becomes a rank-point within 
// one or many rank-charts.  

// The data over this rank-atlas are essentially the data in the send/receive buffers,
// and the Scatter map is responsible for (multiplexing) packing and moving the data from the input Sec into the rank-Sec 
// encapsulating these buffers. Once this has been done, ParMap executes the communication code (send/recv),
// and the rank-multisheeted data are transfered to the required processes.  

//                                     Scatter                                     send/recv
//                                                                    ... rank_0
//              point_in --> chart_in    ==>      (point_in,chart_in) --> rank_k      ==> 
//                                                                    ... rank_K

// Then Gather reduces the data over a single (point_in,chart_in) pair in all of the rank-charts.  This can be thought 
// of as gluing all of the partial sections over the overlaps with remote processes into a single "remote" Sec and then
// gluing it with the "local" Sec.  Once the remote data have been assimilated into the local input Sec, Transform does
// its thing.


//                                      rank_0    Gather                        Transform
//                                 ...
//             (point_in,chart_in) -->  rank_n     ==>    point_in --> chart_in    ==>      point_out --> chart_out
//                                 ...
//                                      rank_N

// Observe that the structure of the GatherAtlas is essentially the same as the structure of the
// Overlap Sifter in ParDelta, therefore the Overlap code can be reused.  However, that code is not customizable,
// while we may want to allow the GatherAtlas  constructor the flexibility to massage the atlas (e.g., to keep only 
// s single rank for a given (point,chart) pair, thereby implementing the 'owner' concept).  Making GatherAtlas a class
// a class will allow this flexibility by exposing the input atlas computation method to overloading.
// The prototypical GatherAtlas object will be implemented to keep all of the ranks in the remote overlap under
// a given (point_in, chart_in) pair.  Custom GatherAtlas objects may prune that so that the number and amount
// of data sent/recv'd by ParMap is only as required.  Here we assume that the overlap is small and computed only once
// or infrequently, while ParMap mappings are frequent.

         


//
// Atlas, Sec and Map classes
//
namespace ALE {
  namespace X {

    // We require that any class implementing the Atlas concept extending Sifter.
    // FIX: should Atlas depend on a Sieve type?  A Sifter type?
    template <typename Ind_, typename Point_, typename Chart_>
    class Atlas : public ASifter<Ind_, Point_, Chart_, SifterDef::multiColor> {
    public:
      // 
      // Encapsulated types
      //
      typedef Point_                                 point_type;
      typedef Chart_                                 chart_type;
      typedef Ind_                                   ind_type;
      typedef ALE::pair<ind_type, ind_type>          index_type;
      //
      typedef ASifter<index_type, point_type, chart_type, SifterDef::multiColor> sifter_type;
    public:
      //
      // Basic interface
      //
      Atlas(MPI_Comm comm = PETSC_COMM_SELF, const int& debug = 0) : sifter_type(comm, debug) {};
      virtual ~Atlas(){};
      //
      // Extended interface
      //
      ind_type size(){
        ind_type sz = 0;
        // Here we simply look at each chart's cone and add up the sizes
        // In fact, we assume that within a chart indices are contiguous, 
        // so we simply add up the offset to the size of the last point of each chart's cone and sum them up.
        baseSequence base = this->base();
        for(typename baseSequence::iterator bitor = base->begin(); bitor != base->end(); bitor++) {
          index_type ii = this->cone(*bitor)->rbegin()->color();
          sz += ii.first + ii.second;
        }
        return sz;
      };
      ind_type size(const chart_type& c) {
        // Here we simply look at the chart's cone and add up the sizes.
        // In fact, we assume that within a chart indices are contiguous, 
        // so we simply return the sum of the offset to the size of the chart's last point.
        index_type ii = this->cone(c).rbegin()->color();
        ind_type sz = ii.first + ii.second;
        return sz;
      };
      ind_type size(const chart_type& c, const point_type& p) {
        // Here we assume that at most a single arrow between p and c exists
        arrowSequence arrows = this->arrows(p,c);
        ind_type sz = 0;
        if(arrows.begin() != arrows.end()) {
          sz = arrows.begin()->first;
        }
        return sz;
      };
      
      ind_type offset(const chart_type& c) {
        // We assume that within a chart indices are contiguous, so the offset of the chart
        // is the offset of the first element in its cone.
        ind_type off = this->cone(c).begin()->color().first;
        return off;
      };
      ind_type offset(const chart_type& c, const point_type& p) {
        // Here we assume that at most a single arrow between p and c exists
        arrowSequence arrows = this->arrows(p,c);
        // CONTINUE: what's the offset in case p is not in c
        ind_type sz = 0;
        if(arrows.begin() != arrows.end()) {
          sz = arrows.begin()->first;
        }
        return sz;
      };
      
    };// class Atlas

    

    // FIX: should Sec depend on a Sieve?  Perhaps Atlas should encapsulate a Sieve type?
    template <typename Data_, typename Atlas_>
    class Sec {
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
      }; // class Sec::DataSequence
    protected:
      Obj<atlas_type> _atlas;
      data_type*      _data;
      bool            _allocated;
    public:
      //
      // Basic interface
      //
      Sec(const Obj<atlas_type> atlas, const (data_type*)& data) {this->setAtlas(atlas, false); this->_data = data;};
      Sec(const Obj<atlas_type> atlas = Obj<atlas_type>()) {this->_data = NULL; this->setAtlas(atlas);};
     ~Sec(){if((this->_data != NULL)&&(this->_allocated)) {ierr = PetscFree(this->_data);CHKERROR(ierr, "Error in PetscFree");}};
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
              ierr = PetscMalloc(this->_atlas->size()*sizeof(data_type), &this->_data);CHKERROR(ierr, "Error in PetscMalloc");
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
    };// class Sec


    template <typename Data_, typename Atlas_>
    class ArraySec : public Sec<Data_, Atlas_> {
    public:
      // 
      // Encapsulated types
      //
      typedef Sec<Data_,Atlas_>                sec_type;
      typedef Atlas_                           atlas_type;
      typedef Data_                            data_type;      
      typedef typename atlas_type::point_type  point_type;
      typedef typename atlas_type::chart_type  chart_type;
      typedef typename atlas_type::index_type  index_type;
      //
      // Basic interface
      //
      ArraySec(const Obj<atlas_type> atlas, const (data_type*)& data) {this->setAtlas(atlas, false); this->_data = data;};
      ArraySec(const Obj<atlas_type> atlas = Obj<atlas_type>()) {this->_data = NULL; this->setAtlas(atlas);};
     ~ArraySec(){if((this->_data != NULL)&&(this->_allocated)) {ierr = PetscFree(this->_data);CHKERROR(ierr, "Error in PetscFree");}};
      //
      // Extended interface
      // 
      data_type*
      restrict(const chart_type& chart) { 
        return this->_data + this->_atlas->offset(chart);
      };
      data_type*
      restrict(const chart_type& chart, const point_type& point) { 
        return this->_data + this->_atlas->offset(chart,point);
      };
    }; // class ArraySec


    // Overlap is a container class that declares GatherAtlas and ScatterAtlas types as well as 
    // defines their construction procedures.
    // InAtlas_ and OutAtlas_ are Atlases, Lightcone_ is a Sifter.
    // Lightcone connects the charts of some ParMap's OutAtlas_ to the ParMap's InAtlas_.
    // GatherAtlas and ScatterAtlas have (point_in,chart_in) pairs as points and process ranks as charts.
    // Preconditions: InAtlas, Lightcone, and OutAtlas share communicator; we require that chart type be Point
    // FIX: should GatherAtlas/ScatterAtlas depend on an underlying Topology Sieve?
    template <typename Data_, typename InAtlas_, typename OutAtlas_, typename Lightcone_>
    class Overlap {
    public:
      // encapsulated types
      typedef Data_                                                 data_type;
      typedef InAtlas_                                              in_atlas_type;
      typedef Lightcone_                                            lightcone_type;
      typedef OutAtlas_                                             out_atlas_type;
      typedef MPI_Int                                               chart_type;
      typedef ALE::pair<in_atlas_type::point_type, chart_type>      point_type;
      typedef typename in_atlas_type::ind_type                      ind_type;
      typedef typename in_atlas_type::index_type                    index_type;
      //
      typedef typename Atlas<point_type, chart_type, ind_type>      gather_scatter_atlas_type;
    protected:
      Obj<in_atlas_type>             _in_atlas;
      Obj<out_atlas_type>            _out_atlas;
      Obj<lightcone_type>            _lightcone;
      //
      Obj<gather_scatter_atlas_type> _gather_atlas;
      Obj<gather_scatter_atlas_type> _scatter_atlas;
    public:
      //
      Overlap(const Obj<in_atlas_type>& in_atlas,const Obj<out_atlas_type>& out_atlas, const Obj<lightcone_type>& lightcone = Obj<lightcone_type>()) : atlas_type(in_atlas->comm()), _in_atlas(in_atlas),_out_atlas(out_atlas),_lightcone(lightcone) {
        this->computeAtlases(this->_gather_atlas, this->_scatter_atlas);
      };
     ~Overlap(){};
      //
      // Extended interface
      Obj<in_atlas_type>  inAtlas()  {return this->_in_atlas();};
      Obj<out_atlas_type> outAtlas() {return this->_out_atlas();};
      Obj<lightcone_type> lightcone(){return this->_lightcone();}
      //
      void computeAtlases(Obj<gather_atlas_type> gather_atlas, Obj<scatter_atlas_type> scatter_atlas) {
        // This function computes the gather and scatter atlases necessary for exchanging and fusing the input data lying over the 
        // overlap with the remote processes into the local data over the overlap points.
        // The Lightcone sifter encodes the dependence between input and output charts of some map: each out-chart in the base 
        // of the Lightcone depends on the in-chart in its Lightcone cone (depends for the computation of the map values).
        // A Null Lightcone Obj is interpreted as an identity Lightcone, hence the if-else dichotomy in the code below.
        //

        // In order to retrieve the remote points that OutAtlas depends on, we compute the overlap of the local bases of InAtlas 
        // restricted to a suitable subset of Lightcone's cap.  The idea is that at most the charts the in the cap of the Lightcone 
        // are required locally by the OutAtlas.
        // Furthermore, at most the cone of OutAtlas' base in Lightcone is required, which is the set we use to compute the overlap.
        // If the Lightcone Obj is Null, we take all of the OutAtlas base as the base of the overlap in InAtlas.
        //
        if(gather_atlas.isNull()) {
          gather_atlas  = gather_atlas_type(this->_in_atlas->comm());
        }
        if(scatter_atlas.isNull()){
          scatter_atlas = scatter_atlas_type(this->_in_atlas->comm());
        }
        //
        if(!lightcone.isNull()) {
          // Take all of out-charts & compute their lightcone closure; this will give all of the in-charts required locally
          typename out_atlas_type::capSequence out_base = this->_out_atlas->base();
          typename lightcone_type::coneSet in_charts = this->_lightcone->closure(out_base);
          // Now we compute the "overlap" of in_atlas with these local in-charts; this will be the gather atlas
          this->__pullbackAtlases(in_charts, gather_atlas, scatter_atlas);

        }// if(!lightcone.isNull())
        else {
          // FIX: handle the Null lightcone case
        }
      };// computeAtlases()

    protected:
      // Internal type definitions to ensure compatibility with the legacy code in the parallel subroutines
      typedef ALE::Point                                Point;
      typedef int                                            int32_t;
      typedef ALE::pair<int32_t, int32_t>                    int_pair;
      typedef ALE::set<std::pair<int32_t, int32_t> >         int_pair_set;
      typedef ALE::map<int32_t,int32_t>                      int__int;
      typedef ALE::map<Point, int32_t>                       Point__int;
      typedef ALE::map<Point, std::pair<int32_t,int32_t> >   Point__int_int;
      typedef ALE::map<Point, int_pair_set>                  Point__int_pair_set;

      template <typename Sequence>
      svoid __determinePointOwners(const Obj<Sequence>& points, int32_t *LeaseData, Point__int& owner) {
        PetscErrorCode ierr;
        // The Sequence points will be referred to as 'base' throughout, although it may in fact represent a cap.
        MPI_Comm comm = this->comm();
        MPI_Int  rank = this->commRank();
        MPI_Int  size = this->commSize();

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


      template <typename BaseSequence_>
      void __pullbackAtlases(const BaseSequence& pointsB, Obj<gather_atlas_type>& gather_atlas, Obj<scatter_atlas_type>& scatter_atlas){
        typedef typename in_atlas_type::traits::baseSequence Sequence;
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
        ierr = PetscObjectGetNewTag(petscObj, &tag1);CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag1, msgSize, BuyCountA, BuySizesA, SellersA, BuyPointsA, SellCountA, SellSizesA, BuyersA, &SellPointsA);
        ierr = PetscObjectGetNewTag(petscObj, &tag2);CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
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
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str());CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
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
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str());CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
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
        ierr = PetscObjectGetNewTag(petscObj, &tag3);CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag3, msgSize, SellCountA, SellSizesA, BuyersA, SellPointsA, BuyCountA, BuySizesA, SellersA, &BuyPointsA);
        ierr = PetscObjectGetNewTag(petscObj, &tag4);CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
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
          ierr = PetscSynchronizedPrintf(comm, txt.str().c_str());CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
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
        ierr = PetscObjectGetNewTag(petscObj, &tag5);CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
        commCycle(comm, tag5, cMsgSize, SellCountA, SellConesSizesA, BuyersA, SellConesA, BuyCountA, BuyConesSizesA, SellersA, &overlapInfoA);
        ierr = PetscObjectGetNewTag(petscObj, &tag6);CHKERROR(ierr, "Failed on PetscObjectGetNewTag");
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
      }; // __pullbackAtlases()

    public:
    }; // class Overlap

    template <typename Data_, typename Map_, typename Overlap_, typename Fusion_>
    class ParMap { // class ParMap
    public:
      //
      // Encapsulated types
      // 
      //  Overlap is an object that encapsulates GatherAtlas & ScatterAtlas objects, while Fusion encapsulates and the corresponding 
      // Gather & Scatter map objects. The idea is that Gather & Scatter depend on the structure of the corresponding atlases, but
      // do not necessarily control their construction.  Therefore the two types of objects and/or their constructors can be overloaded
      // separately.
      //  For convenience Overlap encapsulates the data it is constructed from: two atlases, InAtlas and OutAtlas, and Lightcone, 
      // which is a Sifter.
      // InAtlas refers to the input Sec (the argument of the ParMap), while OutAtlas refers to the output Sec (the result of ParMap).
      // InAtlas_ and OutAtlas_ have arrows from points to charts decorated with indices into Data_ storage of the input/output Sec 
      // respectively.  
      //  GatherAtlas is an Atlas lifting InAtlas into a rank-indexed covering by overlaps with remote processes.  
      // The overlap is computed accroding to Lightcone, which connects the charts of OutAtlas_ to InAtlas_, which implies the two 
      // atlases have charts of the same type.  The idea is that data dependencies are initially expressed at the chart level; 
      // a refinement of this can be achieved by subclassing (or implementing a new) GatherAtlas.
      //  Gather is  a Map that reduce a Sec over GatherAtlas with data over each ((in_point,in_chart),rank) pair to a Sec over InAtlas,
      // with data over each (in_point, in_chart) pair.  Scatter maps in the opposite direction by "multiplexing" the data onto a 
      // rank-indexed covering.
      //  Map is a Map sending an InAtlas Sec obtained from Gather, into an OutAtlas Sec.
      //
      typedef Data_                                            data_type;
      typedef Overlap_                                         overlap_type;
      typedef typename overlap_type::in_atlas_type             in_atlas_type;
      typedef typename overlap_type::out_atlas_type            out_atlas_type;
      typedef typename lightcone_type::out_atlas_type          lightcone_type;
      //
      typedef Fusion_                                          fusion_type;
      typedef typename fusion_type::gather_type                gather_type;
      typedef typename fusion_type::scatter_type               scatter_type;
      typedef Map_                                             map_type;
      //
    protected:
      int                             _debug;
      //
      Obj<map_type>                   _map;
      Obj<overlap_type>               _overlap_type;
      Obj<fusion_type>                _fusion;
    protected:
      void __init(MPI_Comm comm) {    
        PetscErrorCode ierr;
        this->_comm = comm;
        ierr = MPI_Comm_rank(this->_comm, &this->_commRank);CHKERROR(ierr, "Error in MPI_Comm_rank");
        ierr = MPI_Comm_size(this->_comm, &this->_commSize);CHKERROR(ierr, "Error in MPI_Comm_rank"); 
      };
    public:
      //
      // Basic interface
      //
      ParMap(const Obj<map_type>& map, const Obj<overlap_type>& overlap, const Obj<fusion_type>& fusion) 
        : _debug(0), _map(map), _overlap(overlap), _fusion(fusion) {};
     ~ParMap(){};      
      //
      // Extended interface
      //
      int  getDebug() {return this->_debug;}
      void setDebug(const int& d) {this->_debug = d;};
      MPI_Comm comm() {return this->_comm;};
      MPI_Int  commRank() {return this->_commRank;};
      MPI_Int  commSize() {return this->_commSize;};
    }; // class ParMap
  

} // namespace X  
} // namespace ALE

#endif
