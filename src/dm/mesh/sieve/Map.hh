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
      typedef ASifter<Point_, Chart_, typename ALE::pair<index_type,index_type>, SifterDef::uniColor> sifter_type;
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
      // Crossbar is a Sieve connecting the charts of OutAtlas_ to InAtlas_, which implies the two atlases have
      // charts of the same type. The idea is that data dependencies are initially expressed at the chart level.
      // A refinement of this can be achieved by overriding Map methods.
      // Semantics: InAtlas refers to the data coming in (to enable the computation; the argument of the Map), while
      //            OutAtlas refers to the data going out (the result of the computation; the output of the Map).
      typedef InAtlas_   in_atlas_type;
      typedef OutAtlas_  out_atlas_type;
      typedef Crossbar_  crossbar_type;
      typedef Data_      data_type;
      //
      // A FusionAtlas has the same types for points and colors and charts as the InAtlas_.
      // A GatherAtlas combines InAtlas points and charts in its point and has process ranks for sources; 
      // color is of the same type as in InAtlas_ except it carries indices into potentially different Data_ storage -- 
      // that encapsulated in Map.
      //
      typedef typename ALE::pair<MPI_Int, in_atlas_type::target_type> rank_chart_type;
      typedef ASifter<typename in_atlas_type::source_type, 
                      typename in_atlas_type::color_type, 
                      typename ALE::pair<MPI_Int, in_atlas_type::target_type>, 
                      SifterDef::uniColor>                                         fusion_atlas_type;
    protected:
      Obj<in_atlas_type>              _in_atlas;
      Obj<out_atlas_type>             _out_atlas;
      Obj<crossbar_type>              _crossbar;
      //
      Obj<fusion_atlas_type>          _fusion_atlas;
    protected:
      void __init(MPI_Comm comm) {    
        PetscErrorCode ierr;
        this->_comm = comm;
        ierr = MPI_Comm_rank(this->_comm, &this->_commRank); CHKERROR(ierr, "Error in MPI_Comm_rank");
        ierr = MPI_Comm_size(this->_comm, &this->_commSize); CHKERROR(ierr, "Error in MPI_Comm_rank"); 
      };
    public:
      //
      // Basic interface
      //
      Map(const Obj<in_atlas_type>& in_atlas, const Obj<out_atlas_type>& out_atlas, 
          const Obj<crossbar_type>& crossbar = Obj<crossbar_type>()): _in_atlas(in_atlas), _out_atlas(out_atlas), _crossbar(crossbar)
      {
        this->__init(crossbar->comm());
      };
     ~Map(){};
      
      //
      // Extended interface
      //
      int  getDebug() {return this->debug;}
      void setDebug(const int& d) {this->debug = d;};
      MPI_Comm comm() {return this->_comm;};
      MPI_Int  commRank() {return this->_commRank;};
      MPI_Int  commSize() {return this->_commSize;};
    protected:
      static void computePullbackAtlas(const Obj<crossbar_type>& crossbar, const Obj<in_atlas_type>& in_atlas, 
                                       const Obj<pullback_atlas_type>& pullback_atlas) 
      {
        // This function computes the pullback atlas necessary for calculating the output data (result) from the rearranged input data.
        // The crossbar sieve relates the charts in its base to the charts in its closure: each out chart in the base depends on the 
        // in chart in its crossbar closure for the computation of the map values.
        //
        // A Null crossbar Obj is interpreted as an identity crossbar, hence the if-else dichotomy in the code below
        // Loop over the base of crossbar
        if(!crossbar.isNull()) {
          typename crossbar_type::baseSequence base = crossbar->base();
          for(typename crossbar_type::baseSequence::iterator b_itor = base.begin(); b_itor != base.end(); b_itor++) {
            // Take the crossbar closure of each base point and compute the cone of 
            typename crossbar_type::coneSet closure = crossbar->closure(*b_itor);
            // Take the in_atlas support of 'closure' 
            typename crossbar_type::coneSet in = in_atlas->cone(closure);
            // 'in' contains the dependent in_atlas charts and points  of the *b_itor out chart. 
            //   The indices in 'in' are relative to some storage associated with in_atlas and can be used directly in a serial setting.
            //   In a parallel setting, some of the data must be moved first into a local storage encapsulated in Map.
            //   To combine the local serial data and the proxy parallel data a more sophisticated Atlas type is needed.  
            //   This can be done by subclassing Atlas and overriding some of its methods (i.e., those that retrieve indices).
            //   For instance, local rank prepended to the chart can be interpreted as direct access to storage, while a foreigh
            //   rank signals proxy storage.  Ordering may be affected by that.
            //   For the moment we use direct serial access.
            //
            // Loop over 'in' and insert all of its arrows, prepending the local rank to the chart.
            for(typename crossbar_type::coneSet::iterator in_iter = in.begin(); in_iter != in.end(); in_iter++) {
              pullback_atlas->addArrow(in_itor->source(),rank_chart_type(this->commRank(),in_itor->target()),in_itor->color());
            }
          }
        }// if(!crossbar.isNull())
      };// computePullbackAtlas()
    }; // class Map
  

} // namespace X  
} // namespace ALE

#endif
