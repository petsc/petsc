#ifndef included_ALE_Predicate_hh
#define included_ALE_Predicate_hh

#ifndef  included_ALE_hh
#include <ALE.hh>
#endif

#include <boost/multi_index/mem_fun.hpp>

namespace ALE {

  // 
  // Various Filter concepts and definitions
  // 
  namespace FilterDef {
    //
    // PredicateTraits encapsulates Predicate types encoding object subsets with a given Predicate value or within a value range.
    template<typename Predicate_> 
    struct PredicateTraits {};
    // Traits of different predicate types are defined via specialization of PredicateTraits.
    // We require that the predicate type act like unsigned int.
    template<>
    struct PredicateTraits<unsigned int> {
      typedef      unsigned int  predicate_type;
      //
      static predicate_type zero_predicate() {return 0;};
      static predicate_type max_predicate()  {return UINT_MAX;};
      predicate_type& inc(predicate_type& p) {return(++p);};
    };
    //
    template<>
    struct PredicateTraits<char> {
      typedef char predicate_type;
      //
      static predicate_type zero_predicate() {return 0;};
      static predicate_type max_predicate()  {return CHAR_MAX;};
      predicate_type& inc(predicate_type& p) {return (++p);};
    };
    //
    template <typename Predicate_, typename PredicateTraits_ = PredicateTraits<Predicate_> >
    struct PredicateRec {
      typedef  PredicateRec<Predicate_, PredicateTraits_> predicate_rec_type;
      typedef Predicate_                                  predicate_type;
      typedef PredicateTraits_                            predicate_traits;
      //
      predicate_type     predicate;
      struct PredicateAdjuster {
        PredicateAdjuster(const predicate_type& newPredicate = predicate_traits::zero_predicate()) : 
          _newPredicate(newPredicate) {};
        void operator()(predicate_rec_type& r) {r.predicate = this->_newPredicate;};
      private:
        predicate_type _newPredicate; // assume predicate is cheap to copy,no more expensive than a pointer -- else ref would be used
      };
      //
      // Basic interface
      PredicateRec(const predicate_type& p = predicate_traits::zero_predicate()) : predicate(p) {};
      PredicateRec(const PredicateRec& r) : predicate(r.p) {};
      ~PredicateRec() {};
    };

    template <typename PredicateSet_, typename FilterTag_>
    struct FilterContainer {
    public:
      //
      // Encapsulated types
      typedef PredicateSet_                                                  predicate_set_type;
      typedef FilterTag_                                                     filter_tag;
      typedef typename predicate_set_type::template index<filter_tag>::type  filter_index_type;
      //
      typedef typename predicate_set_type::value_type                        predicate_rec_type;
      typedef typename predicate_rec_type::predicate_type                    predicate_type;
      typedef typename predicate_rec_type::predicate_traits                  predicate_traits;
      //
      //
      // 
      enum FilterClearingPolicy {doNotClear = 0, clearOnEntry, clearOnExit};
      //
      // Filter
      struct Filter : public IndexSequence<filter_index_type> {
      public:
        //
        // Encapsulated types
        typedef FilterContainer                          container_type;
        typedef IndexSequence<filter_index_type>         index_sequence_type;
        typedef typename index_sequence_type::index_type index_type;
        //
      protected:
        container_type&      _container;
        predicate_type       _high;
        predicate_type       _low;
        FilterClearingPolicy _policy;
      public:
        //
        // Basic interface
        Filter(container_type& container, index_type& index, const predicate_type& high = predicate_traits::zero_predicate(), 
               const predicate_type& low = predicate_traits::zero_predicate(), FilterClearingPolicy policy = doNotClear) : 
          index_sequence_type(index), _container(container), _high(high), _low(low), _policy(policy)      
        {
          this->_container.applyFilter();
          if((this->_policy == clearOnEntry)) {
            // Upon entering Window must clear out the predicate values.
            this->clear(this->_high, this->_low);
          }
        };
        //
        // Basic interface
        Filter(const Filter& f) : 
          index_sequence_type(f._index), _container(f._container), _high(f._high), _low(f._low), _policy(f._policy) {};
       ~Filter() {
          if((this->_policy == clearOnExit)) {
            // Upon entering Window must clear out the predicate values.
            this->clear(this->_high, this->_low);
          }       
          this->_container.removeFilter();
        };
        //
        // Extended interface
        predicate_type low() {return this->_low;};
        void           newLow(const predicate_type& low) {this->_low = low;};
        predicate_type high() {return this->_high;};
        void           newHigh(const predicate_type& high) {this->_low = high;};
        //
        FilterClearingPolicy clearingPolicy() {return this->_policy;};
        void clear() {this->clear(this->high(), this->low());};
        void clear(const predicate_type& high, const predicate_type& low) {
          if( (high >= low) && (high > predicate_traits::zero_predicate())) {
            // Clear out all predicates with values between low and high (inclusive).
            // We must be careful, since we are modifying the index we are traversing.
            // The good news is that the modification only lowers the rank of the modified element's predicate(sends to zero_predicate),
            // and we are only interested in the elements that are greater than zero_predicate.
            typename predicate_rec_type::PredicateAdjuster zeroOut;
            // We traverse the interval [low, high].
            for(typename index_type::iterator itor = this->_index.lower_bound(::boost::make_tuple(low)); 
                itor != this->_index.upper_bound(::boost::make_tuple(high));){
              // We advance storing the current itor in mod_itor
              // The reason is that mod_itor will be modified and its position will change
              typename index_type::iterator mod_itor = itor; ++itor;
              // Now we modify mod_itor with the default PredicateAdjuster, which will make the predicate zero.
              this->_index.modify(mod_itor, zeroOut);
            }
          }
        }; // Filter::clear()        
      };// class FilterContainer::Filter
      //
      template <typename FilterIndex_, typename ValueExtractor_>
      class FilterSequence : public IndexSequence<FilterIndex_, ValueExtractor_> { // class FilterSequence
      public:
        typedef FilterContainer                              container_type;
        typedef IndexSequence<FilterIndex_, ValueExtractor_> index_sequence_type;
        typedef typename index_sequence_type::index_type     index_type;
        typedef typename index_sequence_type::extractor_type extractor_type;
        //
        // Basic iterator over records with predicates
        class iterator : public index_sequence_type::iterator { // class iterator
        public:
          typedef FilterSequence sequence_type;
        public:
          iterator(sequence_type& sequence, typename index_type::iterator& itor) : 
            index_sequence_type::iterator(sequence, itor) {};
          iterator(const iterator& iter) : index_sequence_type::iterator(iter) {};
          //
          void            setPredicate(const predicate_type& p)      {
            this->_sequence._index.modify(this->_itor, typename predicate_rec_type::PredicateAdjuster(p));
          };
          predicate_type  getPredicate()                       const {
            return this->_itor->predicate;
          };
        };// class iterator
      protected:
        container_type&      _container;
        Obj<Filter>          _filter;
      public:
        //
        // Basic interface
        FilterSequence(const FilterSequence& seq) : index_sequence_type(seq),_container(seq._container), _filter(seq._filter) {};
        FilterSequence(container_type& container, index_type& index, const Obj<Filter>& filter = Obj<Filter>()) : 
          index_sequence_type(index), _container(container), _filter(filter) {};
        ~FilterSequence(){};
        //
        // Extended interface
        virtual typename index_type::size_type size()  {
          typename index_type::size_type sz = 0;
          predicate_type low, high;
          if(!this->_filter.isNull()) {
            low = this->_filter->low();
            high = this->_filter->high();
          }
          for(predicate_type p = low; p != high; p++) {
            sz += this->_index.count(::boost::make_tuple(p));
          }
          return sz;
        };
      }; // class FilterSequence
    protected:
      predicate_set_type _set;
      bool _filterSet;
    public:
      //
      // Basic interface
      FilterContainer() : _filterSet(false) {};
     ~FilterContainer() {
        if(this->_filterSet) {
          throw ALE::Exception("Destructor attempted on FilterContainer with a Filter set");
        }
      };
      //
      // Extended interface
      void setFilter() {
        if(this->_filterSet) {
          throw ALE::Exception("setFilter attempted on a FilterContainer with a Filter already set");
        }
      };
      void removeFilter() {
        if(!this->_filterSet) {
          throw ALE::Exception("removeFilter attempted on a FilterContainer without a Filter set");
        }
      };
    };// class FilterContainer
  }// namespace FilterDef

  namespace Experimental {
    using namespace ALE::FilterDef;
    namespace SifterDef { // namespace SifterDef
      // 
      // Various ArrowContainer definitions
      // 

      // Index tags
      struct SourceColorTag{};
      struct TargetColorTag{};
      struct SourceTargetTag{}; 
      
      // Arrow record 'concept' -- conceptual structure names the fields expected in such an ArrowRec
      template <typename Predicate_, typename Arrow_>
      struct ArrowRec : public PredicateRec<Predicate_> {
        typedef PredicateRec<Predicate_>                      predicate_rec_type;
        typedef typename predicate_rec_type::predicate_type   predicate_type;
        typedef typename predicate_rec_type::predicate_traits predicate_traits;
        //
        typedef Arrow_                                        arrow_type;
        typedef  typename arrow_type::source_type             source_type;
        typedef  typename arrow_type::target_type             target_type;
        typedef  typename arrow_type::color_type              color_type;
      public:
        //
        arrow_type arrow;
      public:
        //
        // Basic interface
        ArrowRec(const source_type& s, const target_type& t, const color_type& c, const predicate_type& p = predicate_type()) : 
          predicate_rec_type(p), arrow(s,t,c) {};
        ArrowRec(const ArrowRec& r) : 
          predicate_rec_type(r.predicate), arrow(r.arrow.source, r.arrow.target, r.arrow.color)  {};
        ~ArrowRec(){};
        // 
        // Extended interface
        const predicate_type& getPredicate() const {return this->predicate;};
        const arrow_type&     getArrow()     const {return this->arrow;};
        const source_type&    getSource()    const {return this->arrow.source;};
        const target_type&    getTarget()    const {return this->arrow.target;};
        const color_type&     getColor()     const {return this->arrow.color;};
      };// class ArrowRec
      
      template<typename ArrowRecSet_, typename ArrowFilterTag_ = SourceColorTag>
      struct ArrowContainer : public FilterContainer<ArrowRecSet_, ArrowFilterTag_> { // class ArrowContainer
      public:
        //
        // Encapsulated types
        typedef FilterContainer<ArrowRecSet_, ArrowFilterTag_> filter_container_type;
        typedef typename filter_container_type::Filter         filter_type;
        //
        typedef ArrowRecSet_                               arrow_rec_set_type; //must have correct tags
        typedef typename arrow_rec_set_type::value_type    arrow_rec_type;     // must (conceptually) extend ArrowContainerDef::ArrowRec
        typedef typename arrow_rec_type::arrow_type        arrow_type;         // arrow_type must have 'source','target','color' fields
        typedef typename arrow_rec_type::predicate_type    predicate_type;
        typedef typename arrow_rec_type::predicate_traits  predicate_traits;
        //
        typedef typename arrow_type::source_type           source_type;
        typedef typename arrow_type::target_type           target_type;
        typedef typename arrow_type::color_type            color_type;
        //
        template <typename Index_, typename Key_, typename SubKey_, typename ValueExtractor_>
        class ArrowSequence : public filter_container_type::template FilterSequence<Index_, ValueExtractor_> { // class ArrowSequence
        public:
          //
          // Encapsulated types
          typedef ArrowContainer                                                                   arrow_container_type;
          typedef typename filter_container_type::template FilterSequence<Index_, ValueExtractor_> filter_sequence_type;            
          typedef typename filter_sequence_type::index_type                                        index_type;
          typedef typename filter_sequence_type::extractor_type                                    extractor_type;
          typedef Key_                                                                             key_type;
          typedef SubKey_                                                                          subkey_type;
          // Need to extend the inherited iterator to be able to extract arrow attributes
          class iterator : public filter_sequence_type::iterator {
          public:
            typedef ArrowSequence                          sequence_type;
            typedef typename sequence_type::index_type     index_type;
            //
            iterator(sequence_type& sequence, typename index_type::iterator itor) : 
              filter_sequence_type::iterator(sequence, itor) {};
            virtual const source_type& source()  const {return this->_itor->arrow.source;};
            virtual const color_type&  color ()  const {return this->_itor->arrow.color;};
            virtual const target_type& target()  const {return this->_itor->arrow.target;};
            virtual const arrow_type&  arrow ()  const {return this->_itor->arrow;};
          };// class iterator
        protected:
          arrow_container_type& _container;
          const key_type        key;
          const subkey_type     subkey;
          const bool            useSubkey;
        public:
          //
          // Basic interface
          ArrowSequence(const ArrowSequence& seq) : 
            filter_sequence_type(seq), _container(seq._container), key(seq.key), subkey(seq.subkey), useSubkey(seq.useSubkey) {};
          ArrowSequence(arrow_container_type& container,index_type& index,const key_type& k,Obj<filter_type> filter=Obj<filter_type>()):
            filter_sequence_type(container,index, filter), _container(container), key(k), subkey(subkey_type()), useSubkey(0) {};
          ArrowSequence(arrow_container_type& container,index_type& index,const key_type& k,const subkey_type& kk,Obj<filter_type> filter=Obj<filter_type>()) : filter_sequence_type(container,index, filter), _container(container), key(k), subkey(kk), useSubkey(1){};
         ~ArrowSequence() {};
          //
          // Extended interface
          virtual typename index_type::size_type  size()  {
            typename index_type::size_type sz = 0;
            predicate_type low, high;
            if(!(this->_filter.isNull())) {
              low = this->_filter->low();
              high = this->_filter->high();
            }
            if (this->useSubkey) {
              for(predicate_type p = low; p <= high; ++p) {
                sz += this->_index.count(::boost::make_tuple(p, this->key, this->subkey));
              }
            } else {
              for(predicate_type p = low; p <= high; ++p) {
                sz += this->_index.count(::boost::make_tuple(p, this->key));
              }
            }
            return sz;
          };
          virtual iterator begin() {
            predicate_type low, high;
            if(!(this->_filter.isNull())) {
              low = this->_filter->low();
              high = this->_filter->high();
            }
            if (this->useSubkey) {
              return iterator(*this, this->_index.lower_bound(::boost::make_tuple(low, this->key,this->subkey)));
            } else {
              return iterator(*this, this->_index.lower_bound(::boost::make_tuple(low, this->key)));
            }
          };
          virtual iterator end() {
            predicate_type low, high;
            if(!(this->_filter.isNull())) {
              low = this->_filter->low();
              high = this->_filter->high();
            }
            if (this->useSubkey) {
              return iterator(*this, this->_index.upper_bound(::boost::make_tuple(high, this->key,this->subkey)));
            } else {
              return iterator(*this, this->_index.upper_bound(::boost::make_tuple(high, this->key)));
            }
          };
          template<typename ostream_type>
          void view(ostream_type& os, const char* label = NULL, const bool& useColor = true){
            if(label != NULL) {
              os << "Viewing " << label << " arrow sequence:";
              if(useColor) {
                os << " with color";
              }
              os << std::endl;
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
        //
        typedef ArrowSequence<typename ::boost::multi_index::index<arrow_rec_set_type,TargetColorTag>::type, target_type, color_type,  typename ::boost::multi_index::const_mem_fun<arrow_rec_type, const source_type&, &arrow_rec_type::getSource> >
        ConeSequence;
      public:
        //
        // Basic interface
        ArrowContainer() {};
        ArrowContainer(const ArrowContainer& container) : filter_container_type(container) {};      
        ~ArrowContainer(){};
        //
        // Extended interface
        void addArrow(const source_type& s, const target_type& t, const color_type& c) {
          this->_set.insert(arrow_rec_type(s,t,c));
        };
        ConeSequence cone(const target_type& t) {
          return ConeSequence(*this, ::boost::multi_index::get<TargetColorTag>(this->_set), t);
        };
      };// class ArrowContainer
    };// namespace SifterDef 

    typedef SifterDef::ArrowRec<unsigned int, ALE::Arrow<int, int, int> > ArrowRec;

    // multi-index set type -- arrow set
    typedef  ::boost::multi_index::multi_index_container<
               ArrowRec,
               ::boost::multi_index::indexed_by<
                 ::boost::multi_index::ordered_non_unique<
                   ::boost::multi_index::tag<SifterDef::SourceTargetTag>,
                   ::boost::multi_index::composite_key<
                     ArrowRec, 
                     ::boost::multi_index::const_mem_fun<ArrowRec, const ArrowRec::predicate_type&, &ArrowRec::getPredicate>,
                     ::boost::multi_index::const_mem_fun<ArrowRec, const ArrowRec::source_type&,    &ArrowRec::getSource>, 
                     ::boost::multi_index::const_mem_fun<ArrowRec, const ArrowRec::target_type&,    &ArrowRec::getTarget>
                  >
                 >,
                 ::boost::multi_index::ordered_non_unique<
                   ::boost::multi_index::tag<SifterDef::SourceColorTag>,
                   ::boost::multi_index::composite_key<
                     ArrowRec, 
                     ::boost::multi_index::const_mem_fun<ArrowRec, const ArrowRec::source_type&, &ArrowRec::getSource>, 
                     ::boost::multi_index::const_mem_fun<ArrowRec, const ArrowRec::color_type&,  &ArrowRec::getColor>
                   >
                 >,
                 ::boost::multi_index::ordered_non_unique<
                   ::boost::multi_index::tag<SifterDef::TargetColorTag>,
                   ::boost::multi_index::composite_key<
                     ArrowRec, 
                     ::boost::multi_index::const_mem_fun<ArrowRec, const ArrowRec::target_type&, &ArrowRec::getTarget>, 
                     ::boost::multi_index::const_mem_fun<ArrowRec, const ArrowRec::color_type&,  &ArrowRec::getColor>
                   >
                 >
               >,
               ALE_ALLOCATOR<ArrowRec>
    > UnicolorArrowSet;
    
  }; // namespace Experimental
}; // namespace ALE

#endif
