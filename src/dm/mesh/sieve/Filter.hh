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
      typedef      unsigned int  printable_type;
      //
      static const predicate_type half;
      static const predicate_type max;
    };
    const PredicateTraits<unsigned int>::predicate_type PredicateTraits<unsigned int>::half = UINT_MAX/4;
    const PredicateTraits<unsigned int>::predicate_type PredicateTraits<unsigned int>::max  = UINT_MAX;
    //
    template<>
    struct PredicateTraits<unsigned short> {
      typedef      unsigned short  predicate_type;
      typedef      unsigned short  printable_type;
      //
      static const predicate_type half;
      static const predicate_type max;
    };
    const PredicateTraits<unsigned short>::predicate_type PredicateTraits<unsigned short>::half = USHRT_MAX/4;
    const PredicateTraits<unsigned short>::predicate_type PredicateTraits<unsigned short>::max  = USHRT_MAX;
    //
    template<>
    struct PredicateTraits<unsigned char> {
      typedef unsigned char  predicate_type;
      typedef unsigned short printable_type;
      //
      static const predicate_type half;
      static const predicate_type max;
    };
    const PredicateTraits<unsigned char>::predicate_type PredicateTraits<unsigned char>::half = UCHAR_MAX/4;
    const PredicateTraits<unsigned char>::predicate_type PredicateTraits<unsigned char>::max  = UCHAR_MAX;
    //
    template <typename Predicate_, typename PredicateTraits_ = PredicateTraits<Predicate_> >
    struct PredicateRec {
      typedef  PredicateRec<Predicate_, PredicateTraits_> predicate_rec_type;
      typedef Predicate_                                  predicate_type;
      typedef PredicateTraits_                            predicate_traits;
      //
      predicate_type     predicate;
      struct PredicateAdjuster {
        PredicateAdjuster(const predicate_type& newPredicate = 0) : 
          _newPredicate(newPredicate) {};
        void operator()(predicate_rec_type& r) {r.predicate = this->_newPredicate;};
      private:
        predicate_type _newPredicate; // assume predicate is cheap to copy,no more expensive than a pointer -- else ref would be used
      };
      //
      // Basic interface
      PredicateRec(const predicate_type& p = 0) : predicate(p) {};
      PredicateRec(const PredicateRec& r) : predicate(r.p) {};
      ~PredicateRec() {};
    };

    class FilterError : public ALE::Exception {
    public:
      FilterError(const string&  msg) : ALE::Exception(msg){}
      FilterError(const ostringstream& txt) : ALE::Exception(txt.str()) {};
     ~FilterError(){};
    };

    template <typename PredicateSet_>
    struct FilterContainer {
    public:
      //
      // Encapsulated types
      typedef FilterContainer                                                filter_container_type;
      typedef PredicateSet_                                                  predicate_set_type;
      typedef typename predicate_set_type::value_type                        predicate_rec_type;
      typedef typename predicate_rec_type::predicate_type                    predicate_type;
      typedef typename predicate_rec_type::predicate_traits                  predicate_traits;
      //
      class Filter : public std::pair<predicate_type, predicate_type> {
      protected:
        filter_container_type& _container;
      public:
        Filter(filter_container_type& container) : std::pair<predicate_type, predicate_type>(0,0),  _container(container) {};
        Filter(filter_container_type& container, const predicate_type& left, const predicate_type& right) : 
          std::pair<predicate_type, predicate_type>(left, right), _container(container) {};
        Filter(const Filter& f) : std::pair<predicate_type, predicate_type>(f.left(), f.right()),_container(f._container) {};
        ~Filter(){this->_container.returnFilter(*this);};
        //
        predicate_type         left()      const {return this->first;};
        predicate_type         right()     const {return this->second;};
        template <typename Stream_>
        friend Stream_& operator<<(Stream_& os, const Filter& f) {
          os << "[";
          os << ((typename predicate_traits::printable_type)(f.left())) << ",";
          os << ((typename predicate_traits::printable_type)(f.right())); 
          os << "]";
          return os;
        };
        template <typename Stream_>
        friend Stream_& operator<<(Stream_& os, const Obj<Filter>& f) {
          os << "[";
          os << ((typename predicate_traits::printable_type)(f->left())) << ",";
          os << ((typename predicate_traits::printable_type)(f->right())); 
          os << "]";
          return os;
        };
      };
      typedef Filter                                                          filter_type;
      typedef Obj<Filter>                                                     filter_object_type;
      
      //
      //
      // FilterSequence extends IndexSequence:
      //       0) elements are 'PredicateRec's
      //       1) holds the parent FilterContainer and an Obj<Filter>
      //       2) elements' predicates can be modified through the index iterator
      //       3) intent: in subclasses iterator construction (begin(), end()) and other queries should depend on the Filter(Object).
      template <typename FilterContainer_, typename Index_, typename ValueExtractor_>
      class FilterSequence : public IndexSequence<Index_, ValueExtractor_> { // class FilterSequence
      public:
        typedef FilterContainer_                             container_type;
        typedef IndexSequence<Index_, ValueExtractor_>       index_sequence_type;
        typedef typename index_sequence_type::index_type     index_type;
        typedef typename index_sequence_type::extractor_type extractor_type;
        typedef typename container_type::filter_type         filter_type;
        typedef typename container_type::filter_object_type  filter_object_type;
        //
        // Basic iterator over records with predicates
        template <typename Sequence_ = FilterSequence>
        class iterator : public index_sequence_type::template iterator<Sequence_> { // class iterator
        public:
          typedef typename index_sequence_type::template iterator<Sequence_> index_sequence_iterator;
          typedef typename index_sequence_iterator::sequence_type            sequence_type;
        public:
          iterator(sequence_type& sequence, typename index_type::iterator& itor) : index_sequence_iterator(sequence, itor) {};
          iterator(const iterator& iter)                                         : index_sequence_iterator(iter) {};
          //
          predicate_type  predicate()               const {return this->_itor->predicate;};
          bool            markUp(const predicate_type& p) {
            // We only allow predicate to increase above the filter boundary
            // In either case the iterator is advanced
            filter_object_type f = this->_sequence.filter();
            predicate_type high = 0;
            bool result = false;
            if(!f.isNull()) {
              high = f->right();
            }
            if(p > high){
              // We keep a copy of the inderlying _itor of the location following *this prior to the mark up.
              // This becomes the new value of *this at the end, since the marked up *this will be out of the filter.
              iterator it(*this); ++it; 
              // We have to deal with the following "corner case":
              // ++it points to this->_sequence.end(), but after having been marked Up *this becomes the new this->_sequence.end().
              // There are two ways to deal with the situation:
              // (1) the situation arises only if *this <= it, hence we can check for this using the ordering of this->_sequence --
              //    the ordering of the underlying index;
              // (2) the situation arises only if until the markUp ++it == this->_sequence.end(), so we can check for this 
              //    and at the end of the markUp enforce *this == this->_sequence.end() regardless of the actual relationship 
              //    between ++it and *this;
              // We choose option (2).
              bool atEnd = (it == this->_sequence.end());
              this->_sequence.mark(this->_itor, p);
              // now we advance this->_itor relative to ++it, not to the new marked up *this
              // can't use 'iterator' assignments because of the reference member _sequence
              if(atEnd) {
                this->_itor = this->_sequence.end()._itor;
              }
              else {
                this->_itor =  it._itor; 
              }
              result = true;
            }// if(p > high)
            return result; // return true if markUp successful
          };// markUp()
        };// class iterator
      protected:
        container_type&        _container;
        filter_object_type     _filter;
      public:
        //
        // Basic interface
        FilterSequence(const FilterSequence& seq) : index_sequence_type(seq),_container(seq._container), _filter(seq._filter) {};
        FilterSequence(container_type& container, index_type& index, const filter_object_type& filter = filter_object_type()) : 
          index_sequence_type(index), _container(container), _filter(filter) {};
        ~FilterSequence(){};
        //
        // Extended interface
        filter_object_type filter() {return _filter;};
        void               mark(typename index_type::iterator itor, const predicate_type& p)      {
          this->_index.modify(itor, typename predicate_rec_type::PredicateAdjuster(p));
        };
      }; // class FilterSequence
    protected:
      predicate_set_type _set;
      predicate_type     _top;
      predicate_type     _occupancy[3];
    public:
      //
      // Basic interface
      FilterContainer() : _top(1) {this->_occupancy[0] = 0; this->_occupancy[1] = 0; this->_occupancy[2] = 0;};
     ~FilterContainer() {};
      //
      // Extended interface
      predicate_type   top()                   const {return this->_top;};
      predicate_type   occupancy(const int& i) const {
        if((i < 0) || (i > 2)){throw FilterError("Invalid interval index");} 
        return this->_occupancy[i];
      };
      filter_object_type newFilter(const predicate_type& width) {
        // This routine will allocate a new filter prescribed by a (left,right) predicate pair.
        // Filter endpoints are chosen from three different intervals of size predicate_traits::half ('half' here) each:
        // 'left' is picked from [1,half] or [half+1,2*half], while 'right' is picked from [1,half],[half+1,2*half],[2*half+1,3*half].
        // The container keeps track of the number of filters intersecting each interval as well as the 'top' of the allocated
        // predicate range: 'top' is the next available predicate -- the next 'left'. Here are the rules for picking 'left' and 'right':
        //   (1) 'left' = 'top', 'right' = 'left' + width - 1,  and allocation of 'width' = 'right' - 'left' + 1 >= 'half' fails.
        //       Hence, if 'left' is from [1,half] then 'right' can be from [1,half] or [half+1,2*half] (depending on 'left' & 'width').
        //       Likewise, if 'left' is from [half+1,2*half] then 'right' can be from [half+1,2*half] or [2*half+1,3*half].
        //   (2) When 'top' ends up in [2*half+1, 3*half] it is shifted to 1 so that subsequent allocations are from [1,half]
        //       thereby completing the 'circle'.
        //   (3) Any time a filter crosses into a new interval, that interval is required to have no intersections with previously
        //       allocated filters, or allocation fails.
        // The rules can be intuitively summed up by imagining a cycle of intervals such that all old filters are deallocated from
        // an interval before new filters move in there again.  The working assumption is that the life span of filters is short enough
        // relative to the rate of new filter creation.  Rule (2) ensures that the interval endpoints always satisfy left <= right:
        // [2*half+1, 3*half] forms an 'overlflow buffer' for [half+1,2*half] and is little used.
        // 
        if(width <= 0) {
          ostringstream txt; txt << "Invalid filter: width = " << width << " (" << *this << ")";
          throw FilterError(txt);
        }
        if(width > predicate_traits::half){
          ostringstream txt; txt << "Too big a filter requested: width = " << width << " (" << *this << ")";
          throw FilterError(txt);
        }
        predicate_type lastRight = this->_top-1;      // 'right' of the preceeding interval
        predicate_type left      = this->_top;        
        predicate_type right     = this->_top+width-1;
        // Before we return the new filter, internal bookkeeping and checks must be done to enforce the rules above.
        for(int i = 0; i < 2; i++) {
          // Check if crossing from the (i-1)-st interval  [(i-1)*half+1,i*half] to the i-th interval [i*half+1,i*half] has occured.
          if((lastRight <= i*predicate_traits::half) && (right > i*predicate_traits::half)) {
            // Line from [(i-1)*half+1,i*half] to [i*half+1,(i+1)*half] has been crossed
            // Check if [i*half+1,(i+1)*half] has been fully vacated
            if(this->_occupancy[i] > 0){
              ostringstream txt; 
              txt << "Interval " << i << " not fully vacated when new filter requested: width = ";
              txt << (typename predicate_traits::printable_type)width;
              txt << " (" << *this << ")"; 
              throw FilterError(txt);
            }
          }
        }// for(int i = 0; i < 2; i++) {
        // Adjust occupancy of intervals
        // Find the interval that 'left' lies in
        for(int i = 0; i < 2; i++) {
          if((i*predicate_traits::half < left) && (left <= (i+1)*predicate_traits::half)) {
            ++(this->_occupancy[i]);
            // Now check whether the interval spans two intervals (i.e., 'right' lies in the following interval)
            if(right > (i+1)*predicate_traits::half) {
              ++(this->_occupancy[i+1]);
            }
          }
        }// for(int i = 0; i < 2; i++)
        //
        // Now we calculate the new top and set it to 0 if it is greater than 2*half (i.e. within [2*half+1, 3*half]).
        this->_top = right+1;
        if(this->_top > 2*predicate_traits::half) {
          this->_top = 1;
        }
        // Finally, we return the newly allocated filter;
        filter_object_type f;
        f.create(filter_type(*this));
        f->first = left; f->second = right;
        return f;
      }; // newFilter()
      //
      void returnFilter(const filter_type& f)      {
        predicate_type left = f.left(), right = f.right();
        // We only deregister non-zero filters
        if((left == 0) && (right == 0)) {
          return;
        }
        predicate_type width = right-left+1;
        // DEBUG
        // Validate filter: size etc.
        if((left == 0) || (left > 2*predicate_traits::half) || (right < left) || (width > predicate_traits::half)) {
          ostringstream txt; txt << "Cannot close invalid filter " << f << " (" << *this << ")";
          throw FilterError(txt);
        }
        // Find the interval that 'left' lies in
        for(int i = 0; i < 2; i++) {
          if((i*predicate_traits::half < left) && (left <= (i+1)*predicate_traits::half)) {
            if(this->_occupancy[i] == 0) {
              ostringstream txt; 
              txt << "Occupancy of interval " << i << " will be negative upon closing of filter [" << left << ", " << right << "]";
              throw FilterError(txt);
            }
            --(this->_occupancy[i]);
            // Now check whether the interval spans two intervals (i.e., 'right' lies in the following interval)
            if(right > (i+1)*predicate_traits::half) {
              if(this->_occupancy[i+1] == 0) {
                ostringstream txt; 
                txt << "Occupancy of interval " << i+1 << " will be negative upon closing of filter [" << left << ", " << right << "]";
                throw FilterError(txt);
              }
              --(this->_occupancy[i+1]);
            }
          }
        }// for(int i = 0; i < 2; i++)
      }; // returnFilter()
      // Printing
      template <typename Stream_>
      friend Stream_& operator<<(Stream_& os, const FilterContainer& fc) {
        os << "top = " << (typename predicate_traits::printable_type)(fc.top()) << ", occupancy = [";
        os << (typename predicate_traits::printable_type)(fc.occupancy(0)) << ",";
        os << (typename predicate_traits::printable_type)(fc.occupancy(1)) << ",";
        os << (typename predicate_traits::printable_type)(fc.occupancy(2));
        os << "]";
        return os;
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
      struct ConeTag{};

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
        template <typename Stream_>
        friend Stream_& operator<<(Stream_& os, const ArrowRec& r) {
          os << r.getArrow() << " <" << (typename predicate_traits::printable_type)(r.getPredicate()) << "> ";
          return os;
        };
      };// class ArrowRec
      
      template<typename ArrowRecSet_>
      struct ArrowContainer : public FilterContainer<ArrowRecSet_> { // class ArrowContainer
      public:
        //
        // Encapsulated types
        typedef FilterContainer<ArrowRecSet_>                        filter_container_type;
        typedef typename filter_container_type::filter_type          filter_type;
        typedef typename filter_container_type::filter_object_type   filter_object_type;
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
        //
        template <typename ArrowContainer_, typename Index_, typename Key_, typename ValueExtractor_>
        class UniKeyArrowSequence : public filter_container_type::template FilterSequence<ArrowContainer_, Index_, ValueExtractor_> 
        { // class TopFilterArrowSequence
        public:
          //
          // Encapsulated types
          typedef typename filter_container_type::template FilterSequence<ArrowContainer_,Index_,ValueExtractor_> filter_sequence_type; 
          typedef typename filter_sequence_type::container_type                                                   container_type;
          typedef typename filter_sequence_type::index_sequence_type                                              index_sequence_type;
          typedef typename filter_sequence_type::index_type                                                       index_type;
          typedef typename filter_sequence_type::extractor_type                                                   extractor_type;
          typedef typename filter_sequence_type::filter_type                                                      filter_type;
          typedef typename filter_sequence_type::filter_object_type                                               filter_object_type;
          typedef Key_                                                                                            key_type;
          // Need to extend the inherited iterator to be able to extract arrow attributes
          template <typename Sequence_ = UniKeyArrowSequence>
          class iterator : public filter_sequence_type::template iterator<Sequence_> {
          public:
            typedef typename filter_sequence_type::template iterator<Sequence_> filter_sequence_iterator;
            typedef typename filter_sequence_iterator::sequence_type            sequence_type;
            typedef typename sequence_type::index_type                          index_type;
            //
            iterator(sequence_type& sequence, typename index_type::iterator itor) : filter_sequence_iterator(sequence, itor) {};
            virtual const source_type& source()  const {return this->_itor->arrow.source;};
            virtual const color_type&  color ()  const {return this->_itor->arrow.color;};
            virtual const target_type& target()  const {return this->_itor->arrow.target;};
            virtual const arrow_type&  arrow ()  const {return this->_itor->arrow;};
          };// class iterator
        protected:
          const key_type         _key;
        public:
          //
          // Basic interface
          UniKeyArrowSequence(const UniKeyArrowSequence& seq) : filter_sequence_type(seq), _key(seq._key) {};
          UniKeyArrowSequence(container_type& container, index_type& index,const key_type& key, filter_object_type filter=filter_object_type()) : filter_sequence_type(container, index, filter), _key(key) {};
         ~UniKeyArrowSequence() {};
          //
          // Extended interface
          virtual typename index_type::size_type  size()  {
            typename index_type::size_type sz = 0;
            predicate_type low  = 0;
            predicate_type high = 0;
            if(!(this->_filter.isNull())) {
              low  = this->_filter->left();
              high = this->_filter->right();
            }
            for(predicate_type p = low; p != high; p++) {              
              sz += this->_index.count(::boost::make_tuple(this->_key,p));
            }
            return sz;
          };
          virtual iterator<> begin() {
            predicate_type low = 0;
            if(!(this->_filter.isNull())) {
              low = this->_filter->left();
            }
            return iterator<>(*this, this->_index.lower_bound(::boost::make_tuple(this->_key,low)));
          };
          virtual iterator<> end() {
            predicate_type high = 0;
            if(!(this->_filter.isNull())) {
              high = this->_filter->right();
            }
            return iterator<>(*this, this->_index.upper_bound(::boost::make_tuple(this->_key,high)));
          };
          virtual iterator<> beginAll() {
            return iterator<>(*this, this->_index.lower_bound(::boost::make_tuple(this->_key)));
          };
          virtual iterator<> endAll() {
            return iterator<>(*this, this->_index.upper_bound(::boost::make_tuple(this->_key)));
          };
          template<typename ostream_type>
          void view(ostream_type& os, const char* label = NULL){
            os << "Viewing";
            if(label != NULL) {
              os << " " << label;
            }
            if(!this->_filter.isNull()) {
              os << " filtered";
            }
            os << " sequence";
            if(!this->_filter.isNull()) {
              os << ", filter " << this->_filter;
            }
            os << ":" << std::endl;
            os << "[";
            for(iterator<> i = this->begin(); i != this->end(); i++) {
              if(i != this->begin()) {
                os << ", " << i.arrow();
              }
              else {
                os  << i.arrow();
              }
              os << " <" << (typename predicate_traits::printable_type)(i.predicate()) << ">";              
            };
            os << "]" << std::endl;
          };// view()
        };// class UniKeyArrowSequence
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
        //
        typedef UniKeyArrowSequence<ArrowContainer, 
                                    typename ::boost::multi_index::index<arrow_rec_set_type,ConeTag>::type, 
                                    target_type, 
                                    typename ::boost::multi_index::const_mem_fun<arrow_rec_type, const source_type&, &arrow_rec_type::getSource> >
        ConeSequence;
        ConeSequence cone(const target_type& t, typename ConeSequence::filter_object_type f = filter_object_type()) {
          return ConeSequence(*this, ::boost::multi_index::get<ConeTag>(this->_set), t, f);
        };
      };// class ArrowContainer
    };// namespace SifterDef 

    typedef SifterDef::ArrowRec<unsigned char, ALE::Arrow<int, int, int> > MyArrowRec;

    // multi-index set type -- arrow set
    typedef  ::boost::multi_index::multi_index_container<
               MyArrowRec,
               ::boost::multi_index::indexed_by<
                 ::boost::multi_index::ordered_non_unique<
                   ::boost::multi_index::tag<SifterDef::ConeTag>,
                   ::boost::multi_index::composite_key<
                     MyArrowRec, 
                     ::boost::multi_index::const_mem_fun<MyArrowRec, const MyArrowRec::target_type&,    &MyArrowRec::getTarget>, 
                     ::boost::multi_index::const_mem_fun<MyArrowRec, const MyArrowRec::predicate_type&, &MyArrowRec::getPredicate>, 
                     ::boost::multi_index::const_mem_fun<MyArrowRec, const MyArrowRec::color_type&,     &MyArrowRec::getColor>
                   >
                 >
               >,
               ALE_ALLOCATOR<MyArrowRec>
    > TopFilterUniColorArrowSet;

    // multi-index set type -- arrow set
    typedef  ::boost::multi_index::multi_index_container<
               MyArrowRec,
               ::boost::multi_index::indexed_by<
                 ::boost::multi_index::ordered_non_unique<
                   ::boost::multi_index::tag<SifterDef::ConeTag>,
                   ::boost::multi_index::composite_key<
                     MyArrowRec, 
                     ::boost::multi_index::const_mem_fun<MyArrowRec, const MyArrowRec::color_type&,     &MyArrowRec::getColor>,
                     ::boost::multi_index::const_mem_fun<MyArrowRec, const MyArrowRec::target_type&,    &MyArrowRec::getTarget>, 
                     ::boost::multi_index::const_mem_fun<MyArrowRec, const MyArrowRec::predicate_type&, &MyArrowRec::getPredicate>
                   >
                 >
               >,
               ALE_ALLOCATOR<MyArrowRec>
    > BottomFilterUniColorArrowSet;
    
  }; // namespace Experimental
}; // namespace ALE

#endif
