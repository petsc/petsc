#ifndef included_ALE_Sifter_hh
#define included_ALE_Sifter_hh


#include <boost/multi_index_container.hpp>
#include <boost/multi_index/key_extractors.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>

#include <iostream>

// ALE extensions

#ifndef  included_ALE_hh
#include <ALE.hh>
#endif


namespace ALE_X { 
  
  namespace SifterDef {
    
    
    // 
    // Arrow definition
    // 
    template<typename Source_, typename Target_, typename Color_>
    struct  Arrow { 
      typedef Arrow   arrow_type;
      typedef Source_ source_type;
      typedef Target_ target_type;
      typedef Color_  color_type;
      source_type _source;
      target_type _target;
      color_type  _color;
      //
      source_type source() const {return this->_source;};
      target_type target() const {return this->_target;};
      color_type  color()  const {return this->_color;};
      // Basic
      Arrow(const source_type& s, const target_type& t, const color_type& c) : _source(s), _target(t), _color(c) {};
      // Rebinding
      template <typename OtherSource_, typename OtherTarget_, typename OtherColor_>
      struct rebind {
        typedef Arrow<OtherSource_, OtherTarget_, OtherColor_> type;
      };
      // Flipping
      struct flip {
        typedef Arrow<target_type, source_type, color_type> type;
        type arrow(const arrow_type& a) { return type(a.target, a.source, a.color);};
      };
      // Printing
      friend std::ostream& operator<<(std::ostream& os, const Arrow& a) {
        os << a.source << " --(" << a.color << ")--> " << a.target;
        return os;
      }
      // Modifying
      struct sourceChanger {
        sourceChanger(const source_type& newSource) : _newSource(newSource) {};
        void operator()(arrow_type& a) {a._source = this->_newSource;}
      private:
        source_type _newSource;
      };
      //
      struct targetChanger {
        targetChanger(const target_type& newTarget) : _newTarget(newTarget) {};
        void operator()(arrow_type& a) { a._target = this->_newTarget;}
      private:
        const target_type _newTarget;
      };
      //
      struct colorChanger {
        colorChanger(const color_type& newColor) : _newColor(newColor) {};
        void operator()(arrow_type& a) { a._color = this->_newColor;}
      private:
        const color_type _newColor;
      };
    };// struct Arrow
    
    //
    // Order and Compare definitions
    //
    // Definitions of wrappers for comparison predicates:
    // The most basic predicates, called 'orders', for our purposes is a lexicographical order on one or two keys.
    //
    // lex1 simply delegates to the underlying Order_; defined purely for aesthetic purposes only
    template <typename Key_, typename Order_ = std::less<Key_> >
    struct lex1 {
    private:
      Order_ _less;
    public:
      bool operator()(const Key_& keyA, const Key_& keyB) 
      {
        return  (_less(keyA,keyB));
      };
    };
    //
    template <typename Key1_, typename Key2_,typename Order1_ = std::less<Key1_>, typename Order2_ = std::less<Key2_> >
    struct lex2 {
    private:
      Order1_ _less1;
      Order2_ _less2;
    public:
      bool operator()(const Key1_& key1A, const Key2_& key2A, const Key1_& key1B, const Key2_& key2B)
      {
        // In the following (key1A < key1B) || ((key1A == key1B)&&(key2A < key2B)) is computed.
        // Since we don't have equivalence '==' explicitly, it is defined by !(key1A < key1B) && !(key1B < key1A).
        // Furthermore, the expression to the right of '||' is evaluated only if that to the left of '||' fails (C semantics),
        // which means that !(key1A < key1B) is true, and we only need to test the other possibility to establish 
        // key equivalence key1A == key1B
        return  (_less1(key1A,key1B) || 
                 (!_less1(key1B,key1A) &&  _less2(key2A,key2B)) );
      };
    };

    //
    // Rec orders
    //
    // RecKeyOrder compares records by comparing keys of type Key_ extracted from arrows using a KeyExtractor_.
    // In addition, a recordcan be compared to a single Key_ or another CompatibleKey_.
    template<typename Rec_, typename KeyExtractor_, typename KeyOrder_ = std::less<typename KeyExtractor_::result_type> >
    struct RecKeyOrder {
      typedef Rec_                                rec_type;
      typedef typename KeyExtractor_::result_type Key_;
    protected:
      KeyOrder_ _key_order;
    public:
      bool operator()(const Rec_& rec1, const Rec_& rec2) {
        return _key_order(_key(rec1), _key(rec2));
      };
      template <typename CompatibleKey_>
      bool operator()(const Rec_& rec, const CompatibleKey_ key) {
        return _key_order(_key(rec), key);
      };
    };// RecKeyOrder

    //
    // Composite Rec ordering operators (e.g., used to generate cone and support indices for Arrows).
    // An RecKeyXXXOrder first orders on a single key using KeyOrder (e.g., Target or Source for cone and support respectively),
    // and then on the whole Rec, using an additional predicate XXXOrder.
    // These are then specialized (with Rec = Arrow) to SupportCompare & ConeCompare, using the order operators supplied by the user:
    // SupportOrder = (SourceOrder, SupportXXXOrder), 
    // ConeOrder    = (TargetOrder, ConeXXXOrder), etc
    template <typename Rec_, typename KeyExtractor_, typename KeyOrder_, typename XXXOrder_>
    struct RecKeyXXXOrder {
      typedef Rec_                                                             rec_type;
      typedef KeyExtractor_                                                    key_extractor_type;
      typedef KeyOrder_                                                        key_order_type;
      typedef typename key_extractor_type::result_type                         key_type;
      typedef XXXOrder_                                                        xxx_order_type;
      //
      typedef lex1<key_type, key_order_type>                                   order1_type;
      typedef lex2<key_type, rec_type, key_order_type, xxx_order_type>         order2_type;
    private:
    public:
      bool operator()(const rec_type& rec1, const rec_type& rec2) { 
        static order2_type        _order2;
        static key_extractor_type _kex;
        return _order2(_kex(rec1),rec1,_kex(rec2),rec2);
      };
      template <typename CompatibleKey_>
      bool operator()(const rec_type& rec1, const CompatibleKey_& key) {
        // We want key to be less than any (key, ...)
        return !_compare1(key,_kex(rec1));
      };
      template <typename CompatibleKey_, typename CompatibleXXXKey_>
      bool operator()(const rec_type& rec1, const ALE::pair<CompatibleKey_, CompatibleXXXKey_>& keyPair) {
        // We want (key,xxxkey) to be less than any (key, xxxkey, ...)
        return !_compare2(keyPair.first,keyPair.second,_kex(rec1),rec1);
      };
    };// class RecKeyXXXOrder

    //
    // Default orders.
    //
    template<typename Arrow_, 
             typename SourceOrder_ = std::less<typename Arrow_::source_type>,
             typename ColorOrder_  = std::less<typename Arrow_::color_type> >
    struct SourceColorOrder : 
      public RecKeyXXXOrder<Arrow_, 
                            ::boost::multi_index::const_mem_fun<Arrow_,typename Arrow_::source_type, &Arrow_::source>, 
                            SourceOrder_, 
                            RecKeyOrder<Arrow_, 
                                        ::boost::multi_index::const_mem_fun<Arrow_, typename Arrow_::color_type, &Arrow_::color>, 
                                        ColorOrder_>
      >
    {};
    
    //
    template<typename Arrow_,
             typename ColorOrder_  = std::less<typename Arrow_::color_type>,
             typename SourceOrder_ = std::less<typename Arrow_::source_type>
    >
    struct ColorSourceOrder : 
      public RecKeyXXXOrder<Arrow_, 
                            ::boost::multi_index::const_mem_fun<Arrow_,typename Arrow_::color_type, &Arrow_::source>, 
                            ColorOrder_,
                            RecKeyOrder<Arrow_, 
                                        ::boost::multi_index::const_mem_fun<Arrow_, typename Arrow_::source_type, &Arrow_::source>, 
                                        SourceOrder_>
      >
    {};
    //
    template<typename Arrow_, 
             typename TargetOrder_ = std::less<typename Arrow_::source_type>,
             typename ColorOrder_  = std::less<typename Arrow_::color_type> >
    struct TargetColorOrder : 
      public RecKeyXXXOrder<Arrow_, 
                            ::boost::multi_index::const_mem_fun<Arrow_,typename Arrow_::source_type, &Arrow_::source>, 
                            TargetOrder_,
                            RecKeyOrder<Arrow_, 
                                        ::boost::multi_index::const_mem_fun<Arrow_, typename Arrow_::color_type, &Arrow_::color>, 
                                        ColorOrder_>
      >
    {};
    //
    template<typename Arrow_, 
             typename ColorOrder_  = std::less<typename Arrow_::color_type>,
             typename TargetOrder_ = std::less<typename Arrow_::source_type> >
    struct ColorTargetOrder : 
      public RecKeyXXXOrder<Arrow_, 
                            ::boost::multi_index::const_mem_fun<Arrow_,typename Arrow_::color_type, &Arrow_::source>, 
                            ColorOrder_,
                            RecKeyOrder<Arrow_, 
                                        ::boost::multi_index::const_mem_fun<Arrow_, typename Arrow_::source_type, &Arrow_::source>, 
                                        TargetOrder_>
      >
    {};
  
    //
    // StridedIndexSequence definition
    // 
    // Defines a sequence representing a subset of a multi_index container defined by its Index_ which is ordered lexicographically 
    // starting with an OuterKey_ (obtained from an OuterKeyExtractor_) and then by an InnerKey_ (obtained from an InnerKeyExtractor_).
    // A sequence defines output iterators (input iterators in std terminology) for traversing an Index_ object.
    // This particular sequence traverses all OuterKey_ segements within the given bounds, and within each segment traverses all 
    // with a given Key_. In other words, the sequence iterates over the (OuterKey_, InnerKey_) value pairs with 
    // the outer keys from a given range and a fixed inner key.
    // Upon dereferencing values are extracted from each result record using a ValueExtractor_ object.
    template <typename Index_, typename OuterKeyExtractor_, typename InnerKeyExtractor_, 
              typename ValueExtractor_ = ::boost::multi_index::identity<typename Index_::value_type>, bool inner_strided_flag = false >
    struct StridedIndexSequence {
      typedef Index_                                           index_type;
      typedef OuterKeyExtractor_                               outer_key_extractor_type;
      typedef typename outer_key_extractor_type::result_type   outer_key_type;
      typedef InnerKeyExtractor_                               inner_key_extractor_type;
      typedef typename inner_key_extractor_type::result_type   inner_key_type;
      typedef ValueExtractor_                                  value_extractor_type;
      typedef typename value_extractor_type::result_type       value_type;
      typedef typename index_type::size_type                   size_type;
      typedef typename index_type::iterator                    itor_type;
      //
      class iterator {
      public:
        // Parent sequence type
        typedef StridedIndexSequence                   sequence_type;
        // Standard iterator typedefs
        typedef std::input_iterator_tag                iterator_category;
        typedef int                                    difference_type;
        typedef value_type*                            pointer;
        typedef value_type&                            reference;
        /* value_type defined in the containing StridedIndexSequence */
      protected:
        // Parent sequence
        sequence_type&  _sequence;
        // Underlying iterator & segment boundary
        itor_type       _itor;
        itor_type       _segBndry;
        //
        // Value extractor
        outer_key_extractor_type _okex;
        inner_key_extractor_type _ikex;
        value_extractor_type     _ex;
      public:
        iterator(sequence_type& sequence, const itor_type& itor, const itor_type& segBndry) : 
          _sequence(sequence), _itor(itor), _segBndry(segBndry) {};
        iterator(const iterator& iter):_sequence(iter._sequence), _itor(iter._itor),_segBndry(iter.segBndry){};
        virtual ~iterator() {};
        virtual bool              operator==(const iterator& iter) const {return this->_itor == iter._itor;};
        virtual bool              operator!=(const iterator& iter) const {return this->_itor != iter._itor;};
        // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
        virtual const value_type  operator*() const {return _ex(*(this->_itor));};
        virtual iterator   operator++() {
          this->_sequence.next(this->_itor, this->_segBndry, inner_strided_flag);
          return *this;
        };
        virtual iterator   operator++(int n) {iterator tmp(*this); ++(*this); return tmp;};
      };// class iterator
    protected:
      index_type&     _index;
      outer_key_type  _ohigh, _olow;
      bool            _have_olow, _have_ohigh;
      outer_key_type  _ihigh, _ilow;
      bool            _have_ilow, _have_ihigh;
      //
      outer_key_extractor_type _okex;
      inner_key_extractor_type _ikex;
    public:
      //
      // Basic interface
      //
      StridedIndexSequence(const StridedIndexSequence& seq) : _index(seq._index), _olow(seq._olow), _ohigh(seq._ohigh), _have_olow(seq._have_olow), _have_ohigh(seq._have_ohigh), _ilow(seq._ilow), _ihigh(seq._ihigh), _have_ilow(seq._have_ilow), _have_ihigh(seq._have_ihigh)
      {};
      StridedIndexSequence(index_type& index)  :  _index(index) {
        this->_have_olow = false; this->_have_ohigh = false;
        this->_have_ilow = false; this->_have_ihigh = false;
      };
      virtual ~StridedIndexSequence() {};
      void copy(const StridedIndexSequence& seq, StridedIndexSequence cseq) {
        cseq._index = seq._index; 
        cseq._have_olow  = seq._have_olow;
        cseq._have_ohigh = seq._have_ohigh;
        cseq._olow       = seq._olow;
        cseq._ohigh      = seq._ohigh;
        //
        cseq._have_ilow  = seq._have_ilow;
        cseq._have_ihigh = seq._have_ihigh;
        cseq._ilow       = seq._ilow;
        cseq._ihigh      = seq._ihigh;
      };
      StridedIndexSequence& operator=(const StridedIndexSequence& seq) {
        copy(seq,*this); return *this;
      };
      void reset(index_type& index) {
        this->_index      = index;
        this->_have_olow  = false;
        this->_have_ohigh = false;
        this->_olow       = outer_key_type();
        this->_ohigh      = outer_key_type();
        this->_have_ilow  = false;
        this->_have_ihigh = false;
        this->_ilow       = inner_key_type();
        this->_ihigh      = inner_key_type();
      };
      //
      // Extended interface
      //
      virtual bool         
      empty() {return (this->begin() == this->end());};
      //
      virtual size_type  
      size()  {
        size_type sz = 0;
        for(iterator it = this->begin(); it != this->end(); it++) {
          ++sz;
        }
        return sz;
      };
      iterator begin() {
        static itor_type itor;
        // Determine the lower outer limit iterator
        if(this->_have_olow()) {
          // ASSUMPTION: index ordering operator can compare against outer_keys
          this->_itor = this->_index.lower_bound(this->_olow);
        }
        else {
          this->_itor = this->_index.begin();
        }
        // Now determine the inner lower limit and set the iterator to that limit within the first segment
        if(this->_have_ilow) {
          // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
          itor = this->_index.lower_bound(ALE::pair<outer_key_type, inner_key_type>(this->_okex(*itor),this->_ilow));
        }
        else {
          // the itor is already in the right place: nothing to do
        }  
        // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
        static itor_type segBndry;
        segBndry = this->_index.upper_bound(ALE::pair<outer_key_type, inner_key_type>(this->_okex(*itor),_ikex(*itor)));
        return iterator(*this, itor, segBndry);
      }; // begin()
      //
      void next(itor_type& itor, itor_type& segBndry, bool inner_strided = false) {
        outer_key_type olow;
        inner_key_type ilow;
        // If iteration over inner keys is to be strided as well, we advance directly to the segment boundary.
        // Effectively, we iterate over segments.
        if(inner_strided) {
          itor = segBndry;
          // Finally, compute the new segment's boundary
          // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
          olow = this->_okex(*itor);
          ilow = this->_ikex(*itor);
          segBndry = this->_index.upper_bound(ALE::pair<outer_key_type, inner_key_type>(olow,ilow));
        }// inner strided
        // Otherwise, we iterate *within* a segment until its end is reached; then the following segment is started.
        else {
          // See if our advance would lead to breaching the segment boundary:
          itor_type tmp_itor = ++(itor);
          if(tmp_itor != segBndry) { 
            // if not breached the segment boundary, simply use the advanced iterator
            itor = tmp_itor;
          }
          else {
            // Obtain the current outer key from itor:
            olow = this->_okex(*itor);
            // Compute the lower boundary of the new segment
            // ASSUMPTION: index ordering operator can compare against outer_keys
            itor = this->_index.upper_bound(olow);
            // Extract the new outer key
            olow = this->_okex(*itor);
            // Now determine the inner lower limit and set the iterator to that limit within the new segment
            if(this->_have_ilow) {
              ilow = this->_ilow;
              // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
              itor = this->_index.lower_bound(ALE::pair<outer_key_type, inner_key_type>(olow,ilow));
            }
            else {
              // the itor is already in the right place; need to extract the ilow key
              ilow = this->_ikex(*itor);
            }
            // Finally, compute the new segment's boundary
            // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
            segBndry = this->_index.upper_bound(ALE::pair<outer_key_type, inner_key_type>(olow,ilow));
          }
        }// inner not strided
      };// next()
      //
      iterator end() {
        itor_type itor;
        // Determine the upper outer limit
        outer_key_type ohigh;
        if(!this->_have_ohigh) {
          itor = this->_index.rbegin();
          ohigh = this->_okex(*itor);
        }
        // Determine the inner outer limit
        inner_key_type ihigh;
        if(this->_have_ihigh) {
          ihigh = this->_ihigh;
          // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
          itor = this->_index.upper_bound(ALE::pair<outer_key_type, inner_key_type>(ohigh,ihigh));
        }
        else {
          // ASSUMPTION: index ordering operator can compare against outer_keys
          itor = this->_index.upper_bound(ohigh);
        }
        // use segBndry == itor
        return iterator(*this, itor, itor); 
      };// end()
      //
      void setInnerLow(const inner_key_type& ilow) {
        this->_ilow = ilow; this->_have_ilow = true;
      };
      //
      void setInnerHigh(const inner_key_type& ihigh) {
        this->_ihigh = ihigh; this->_have_ihigh = true;
      };
      //
      void setInnerLimits(const inner_key_type& ilow, const inner_key_type& ihigh) {
        this->_ilow = ilow; this->_have_ilow = true; this->_ihigh = ihigh; this->_have_ihigh = true;
      };
      //
      void setOuterLimits(const outer_key_type& olow, const outer_key_type& ohigh) {
        this->_olow = olow; this->_have_olow = true; this->_ohigh = ohigh; this->_have_ohigh = true;
      };
      //
      void setOuterLow(const outer_key_type& olow) {
        this->_olow = olow; this->_have_olow = true;
      };
      //
      void setOuterHigh(const outer_key_type& ohigh) {
        this->_ohigh = ohigh; this->_have_ohigh = true;
      };
      //
      template<typename ostream_type>
      void view(ostream_type& os, const char* label = NULL){
        if(label != NULL) {
          os << "Viewing " << label << " sequence:" << std::endl;
        } 
        os << "[";
        for(iterator i = this->begin(); i != this->end(); i++) {
          os << " "<< *i;
        }
        os << " ]" << std::endl;
      };
    };// class StridedIndexSequence    
  }; // namespace SifterDef
  
  //
  // Sifter definition
  template<typename Arrow_, 
           typename ArrowSupportOrder_= SifterDef::ColorTargetOrder<Arrow_>, 
           typename ArrowConeOrder_   = SifterDef::ColorSourceOrder<Arrow_>, 
           typename Predicate_ = int, typename PredicateOrder_ = std::less<Predicate_> >
  struct Sifter { // struct Sifter
    //
    // Encapsulated types
    //
    typedef Arrow_                           arrow_type;
    typedef typename arrow_type::source_type source_type;
    typedef typename arrow_type::target_type target_type;
    typedef typename arrow_type::color_type  color_type;
    //
    // Internal types
    //
    // Predicates and Rec
    typedef Predicate_                       predicate_type;
    typedef PredicateOrder_                  predicate_order_type;
    struct Rec : public arrow_type {
    public:
      //
      // Re-export typedefs
      //
      typedef typename arrow_type::source_type        source_type;
      typedef typename arrow_type::target_type        target_type;
      typedef typename arrow_type::color_type         color_type;
    public:
      // Predicate stored alongside the arrow data
      predicate_type _predicate;
      predicate_type predicate() const{return this->_predicate;};
      source_type    source() const {return this->arrow_type::source();};
      target_type    target() const {return this->arrow_type::target();};
    };
    //
    typedef Rec                              rec_type;
    //
    // Compound orders are assembled here
    //
    typedef std::less<typename rec_type::source_type> source_order_type; 
    typedef std::less<typename rec_type::target_type> target_order_type;
    //
    // Rec 'downward' order type: first order by predicate, then source, then support
    struct downward_order_type : public 
    SifterDef::RecKeyXXXOrder<rec_type, 
                              typename ::boost::multi_index::const_mem_fun<rec_type, predicate_type, &rec_type::predicate>, 
                              predicate_order_type, 
                              SifterDef::RecKeyXXXOrder<rec_type,
                                                        ::boost::multi_index::const_mem_fun<rec_type,
                                                                                            typename rec_type::source_type,
                                                                                            &rec_type::source>,
                                                        source_order_type, ArrowSupportOrder_> > {};
    
    //
    // Rec Cone order
    struct upward_order_type : public 
    SifterDef::RecKeyXXXOrder<rec_type, 
                              typename ::boost::multi_index::const_mem_fun<rec_type, 
                                                                           predicate_type, 
                                                                           &rec_type::predicate>, 
                              predicate_order_type,
                              SifterDef::RecKeyXXXOrder<rec_type,
                                                        ::boost::multi_index::const_mem_fun<rec_type,
                                                                                            typename rec_type::target_type,
                                                                                            &rec_type::target>,
                                                        target_order_type, ArrowConeOrder_> >
    {};
    
    //
    // Index tags
    //
    struct                                   DownwardTag{};
    struct                                   UpwardTag{};
    
    // Rec set type
    typedef ::boost::multi_index::multi_index_container< 
      rec_type,
      ::boost::multi_index::indexed_by< 
        ::boost::multi_index::ordered_non_unique<
          ::boost::multi_index::tag<UpwardTag>, ::boost::multi_index::identity<rec_type>, downward_order_type
        >,
        ::boost::multi_index::ordered_non_unique<
          ::boost::multi_index::tag<DownwardTag>, ::boost::multi_index::identity<rec_type>, upward_order_type
        > 
      >,
      ALE_ALLOCATOR<rec_type> > 
    rec_set_type;
    //
    // Sequence types
    template <typename Index_, 
              typename OuterKeyExtractor_, typename InnerKeyExtractor_, typename ValueExtractor_, bool inner_strided_flag = false>
    class ArrowSequence : 
      public SifterDef::StridedIndexSequence<Index_, OuterKeyExtractor_, InnerKeyExtractor_, ValueExtractor_, inner_strided_flag> {
      // ArrowSequence extends StridedIndexSequence with extra iterator methods.
    public:
      typedef SifterDef::StridedIndexSequence<Index_, OuterKeyExtractor_, InnerKeyExtractor_, ValueExtractor_> super;
      typedef Sifter                                                                                           container_type;
      typedef typename super::index_type                                                                       index_type;
      typedef typename super::outer_key_type                                                                   outer_key_type;
      typedef typename super::inner_key_type                                                                   inner_key_type;
      
      // Need to extend the inherited iterators to be able to extract arrow color
      class iterator : public super::iterator {
      public:
        iterator(const typename super::iterator& super_iter) : super::iterator(super_iter) {};
        virtual const source_type& source() const {return this->_itor->_source;};
        virtual const color_type&  color()  const {return this->_itor->_color;};
        virtual const target_type& target() const {return this->_itor->_target;};
        virtual const arrow_type&  arrow()  const {return *(this->_itor);};
      };
    protected:
      container_type _container;
    public:
      //
      // Basic ArrowSequence interface
      //
      ArrowSequence(const ArrowSequence& seq) : super(seq), _container(seq._container) {};
      ArrowSequence(const container_type& container, index_type& index) : super(index), _container(container) {};
      virtual ~ArrowSequence() {};
      void copy(const ArrowSequence& seq, ArrowSequence& cseq) {
        super::copy(seq,cseq);
        cseq._container = seq._container;
      };
      ArrowSequence& operator=(const ArrowSequence& seq) {
        copy(seq,*this); return *this;
      };
      void reset(const container_type& container, index_type& index) {
        this->super::reset(index);
        this->_container = container;
      };
      //
      // Extended ArrowSequence interface
      //
      
      virtual iterator begin() {
        return super::begin();
      };
      //
      virtual iterator end() {
        return super::end();
      };
      //
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
      void addArrow(const arrow_type& a) {
        this->_container.addArrow(a);
      };
      //
      virtual bool contains(const outer_key_type& ok, const inner_key_type& ik) {
        return (this->_index.find(ALE::pair<outer_key_type,inner_key_type>(ok,ik)) != this->_index.end());
      };
    };// class ArrowSequence    
      
    //
    // Specialized sequence types
    //
    typedef ArrowSequence<typename ::boost::multi_index::index<rec_set_type, UpwardTag>::type,
                          ::boost::multi_index::const_mem_fun<rec_type, predicate_type, &rec_type::predicate>,
                          ::boost::multi_index::identity<rec_type>,
                          ::boost::multi_index::const_mem_fun<rec_type, target_type, &rec_type::target>, 
                          true>                                                       
    BaseSequence;

    typedef ArrowSequence<typename ::boost::multi_index::index<rec_set_type, UpwardTag>::type,
                          ::boost::multi_index::const_mem_fun<rec_type, predicate_type, &rec_type::predicate>,
                          ::boost::multi_index::identity<rec_type>,
                          ::boost::multi_index::const_mem_fun<rec_type, source_type, &rec_type::source> >     
    ConeSequence;
    //
    // Extended interface
    //
    void addArrow(const arrow_type& a) {
      this->_rec_set.insert(a);
    };
    void cone(const target_type& t, ConeSequence& seq) {
      seq.reset(*this, ::boost::multi_index::get<UpwardTag>(this->_rec_set));
      seq.setInnerLimits(t,t);
    };
    ConeSequence& cone(const target_type& t) {
      static ConeSequence cseq;
      this->cone(t,cseq);
      return cseq;
    };
    void base(BaseSequence& seq) {
      seq.reset(*this, ::boost::multi_index::get<UpwardTag>(this->_rec_set));
    };
    BaseSequence& base() {
      static BaseSequence bseq;
      this->base(bseq);
      return bseq;
    };
    
  protected:
    // set of arrow records
    rec_set_type _rec_set;
    
  }; // class Sifter


} // namespace ALE_X

#endif
