#ifndef included_ALE_Sifter_hh
#define included_ALE_Sifter_hh


#include <boost/multi_index_container.hpp>
#include <boost/multi_index/key_extractors.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>

#include <iostream>


#ifndef  included_ALE_hh
#include <ALE.hh>
#endif


namespace ALE { 

  class XObject {
  protected:
    int      _debug;
  public:
    XObject(const int debug = 0)    : _debug(debug) {};
    XObject(const XObject& xobject) : _debug(xobject._debug) {};
    //
    int      debug() {return this->_debug;};
  };// class XObject

  class XParallelObject : public XObject {
  protected:
    MPI_Comm _comm;
    int      _commRank, _commSize;
  protected:
    void __setupComm(const MPI_Comm& comm) {
      this->_comm = comm;
      PetscErrorCode ierr;
      ierr = MPI_Comm_size(this->_comm, &this->_commSize); CHKERROR(ierr, "Error in MPI_Comm_size");
      ierr = MPI_Comm_rank(this->_comm, &this->_commRank); CHKERROR(ierr, "Error in MPI_Comm_rank");
    }
  public:
    XParallelObject(const MPI_Comm& comm, const int debug)   : XObject(debug) {this->__setupComm(comm);};
    XParallelObject(const MPI_Comm& comm = PETSC_COMM_WORLD) : XObject()      {this->__setupComm(comm);};
    XParallelObject(const XParallelObject& xpobject)         : XObject(xpobject), _comm(xpobject._comm) {};
    //
    MPI_Comm comm()     {return this->_comm;};
    int      commSize() {return this->_commSize;};
    int      commRank() {return this->_commRank;};
  };// class XParallelObject
  
  namespace XSifterDef {
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
        os << a._source << " --(" << a._color << ")--> " << a._target;
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
    // Rec orders
    //
    // RecKeyOrder compares records by comparing keys of type Key_ extracted from arrows using a KeyExtractor_.
    // In addition, a recordcan be compared to a single Key_ or another CompatibleKey_.
    template<typename Rec_, typename KeyExtractor_, typename KeyOrder_ = std::less<typename KeyExtractor_::result_type> >
    struct RecKeyOrder {
      typedef Rec_                                     rec_type;
      typedef KeyExtractor_                            key_extractor_type;
      typedef typename key_extractor_type::result_type key_type;
      typedef KeyOrder_                                key_order_type;
    protected:
      key_order_type     _key_order;
      key_extractor_type _kex;
    public:
      bool operator()(const rec_type& rec1, const rec_type& rec2) const {
        return this->_key_order(this->_kex(rec1), this->_kex(rec2));
      };
      template <typename CompatibleKey_>
      bool operator()(const rec_type& rec, const ALE::singleton<CompatibleKey_> keySingleton) const {
        // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
        // and must be unwrapped before the ordering operator is applied
        return this->_key_order(this->_kex(rec), keySingleton.first);
      };
      template <typename CompatibleKey_>
      bool operator()(const ALE::singleton<CompatibleKey_> keySingleton, const rec_type& rec) const {
        // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
        // and must be unwrapped before the ordering operator is applied
        return this->_key_order(keySingleton.first, this->_kex(rec));
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
    private:
      key_order_type     _compare_keys;
      xxx_order_type     _compare_xxx;
      key_extractor_type _kex;
    public:
      bool operator()(const rec_type& rec1, const rec_type& rec2) const { 
        //
        return this->_compare_keys(_kex(rec1),_kex(rec2)) ||
          (!this->_compare_keys(_kex(rec2),_kex(rec1)) && this->_compare_xxx(rec1,rec2));
      };
      template <typename CompatibleKey_>
      bool operator()(const ALE::singleton<CompatibleKey_>& keySingleton, const rec_type& rec1) const {
        // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
        // and must be unwrapped before the ordering operator is applied
        //
        // We want key to be less than any (key, ...)
        return this->_compare_keys(keySingleton.first, this->_kex(rec1));
      };
      template <typename CompatibleKey_>
      bool operator()(const rec_type& rec1, const ALE::singleton<CompatibleKey_>& keySingleton) const {
        // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
        // and must be unwrapped before the ordering operator is applied
        //
        // We want key to be less than any (key, ...)
        return !this->_compare_keys(keySingleton.first, this->_kex(rec1));
      };
      template <typename CompatibleKey_, typename CompatibleXXXKey_>
      bool operator()(const ALE::pair<CompatibleKey_, CompatibleXXXKey_>& keyPair, const rec_type& rec) const {
        // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
        // and must be unwrapped before the ordering operator is applied
        //
        // We want (key,xxxkey) to be less than any (key, xxxkey, ...)
        return this->_compare_keys(keyPair.first, _kex(rec)) ||
          (!this->_compare_keys(_kex(rec), keyPair.first) && this->_compare_xxx(ALE::singleton<CompatibleXXXKey_>(keyPair.second),rec));
        // Note that CompatibleXXXKey_ -- the second key in the pair -- must be wrapped up as a singleton before being passed for comparison against rec
        // to compare_xxx.  This is necessary for compare_xxx to disamiguate comparison of recs to elements of differents types.  In particular, 
        // this is necessary if compare_xxx is of the RecKeyXXXOrder type. Specialization doesn't work, or I don't know how to make it work in this context.
      };
      template <typename CompatibleKey_, typename CompatibleXXXKey_>
      bool operator()(const rec_type& rec, const ALE::pair<CompatibleKey_, CompatibleXXXKey_>& keyPair) const {
        // We want (key,xxxkey) to be less than any (key, xxxkey, ...)
        return _compare_keys(_kex(rec), keyPair.first) ||
          (!this->_compare_keys(keyPair.first, _kex(rec)) && this->_compare_xxx(rec, ALE::singleton<CompatibleXXXKey_>(keyPair.second)));
        // Note that CompatibleXXXKey_ -- the second key in the pair -- must be wrapped up as a singleton before being passed for comparison against rec
        // to compare_xxx.  This is necessary for compare_xxx to disamiguate comparison of recs to elements of differents types.  In particular, 
        // this is necessary if compare_xxx is of the RecKeyXXXOrder type. Specialization doesn't work, or I don't know how to make it work in this context.
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
    #undef  __CLASS__
    #define __CLASS__ "StridedIndexSequence"
    template <typename Index_, typename OuterKeyExtractor_, typename InnerKeyExtractor_, 
              typename ValueExtractor_ = ::boost::multi_index::identity<typename Index_::value_type>, bool inner_strided_flag = false >
    struct StridedIndexSequence : XObject {
      typedef Index_                                           index_type;
      typedef OuterKeyExtractor_                               outer_key_extractor_type;
      typedef typename outer_key_extractor_type::result_type   outer_key_type;
      typedef InnerKeyExtractor_                               inner_key_extractor_type;
      typedef typename inner_key_extractor_type::result_type   inner_key_type;
      typedef ValueExtractor_                                  value_extractor_type;
      typedef typename value_extractor_type::result_type       value_type;
      typedef typename index_type::size_type                   size_type;
      typedef typename index_type::iterator                    itor_type;
      typedef typename index_type::reverse_iterator            ritor_type;
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
        sequence_type  *_sequence;
        // Underlying iterator & segment boundary
        itor_type       _itor;
        itor_type       _segBndry;
        //
        // Value extractor
        outer_key_extractor_type _okex;
        inner_key_extractor_type _ikex;
        value_extractor_type     _ex;
      public:
        iterator() : _sequence(NULL) {};
        iterator(sequence_type *sequence, const itor_type& itor, const itor_type& segBndry) : 
          _sequence(sequence), _itor(itor), _segBndry(segBndry) {};
        iterator(const iterator& iter):_sequence(iter._sequence), _itor(iter._itor),_segBndry(iter._segBndry){};
        virtual ~iterator() {};
        virtual bool              operator==(const iterator& iter) const {return this->_itor == iter._itor;};
        virtual bool              operator!=(const iterator& iter) const {return this->_itor != iter._itor;};
        // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
        virtual const value_type  operator*() const {return _ex(*(this->_itor));};
        virtual iterator   operator++() {
          this->_sequence->next(this->_itor, this->_segBndry, inner_strided_flag);
          return *this;
        };
        virtual iterator   operator++(int n) {iterator tmp(*this); ++(*this); return tmp;};
      };// class iterator
    protected:
      index_type      *_index;
      //
      outer_key_type      _olow, _ohigh;
      bool                _have_olow, _have_ohigh;
      inner_key_type      _ilow, _ihigh;
      bool                _have_ilow, _have_ihigh;
      //
      outer_key_extractor_type _okex;
      inner_key_extractor_type _ikex;
    public:
      //
      // Basic interface
      //
      StridedIndexSequence(index_type *index, const int& debug = 0)  : XObject(debug), _index(index) {
        this->_have_olow = false; this->_have_ohigh = false;
        this->_have_ilow = false; this->_have_ihigh = false;
      };
      StridedIndexSequence(const int& debug = 0) : XObject(debug),_index(NULL) {};
      StridedIndexSequence(const StridedIndexSequence& seq) : XObject(seq), _index(seq._index), _olow(seq._olow), _ohigh(seq._ohigh), _have_olow(seq._have_olow), _have_ohigh(seq._have_ohigh), _ilow(seq._ilow), _ihigh(seq._ihigh), _have_ilow(seq._have_ilow), _have_ihigh(seq._have_ihigh)
      {};
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
      void reset(index_type *index) {
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
      #undef  __FUNCT__
      #define __FUNCT__ "begin"
      iterator begin() {
        if(this->debug()) {
          std::cout << std::endl << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "(olow,ohigh): (";
          if(this->_have_olow){
            std::cout << this->_olow << ", ";
          }
          else {
            std::cout << "none, ";
          }
          if(this->_have_ohigh){
            std::cout << this->_ohigh;
          }
          else {
            std::cout << "none";
          }
          std::cout << "), ";
          //
          std::cout << "(ilow,ihigh): (";
          if(this->_have_ilow){
            std::cout << this->_ilow << ", ";
          }
          else {
            std::cout << "none, ";
          }
          if(this->_have_ihigh){
            std::cout << this->_ihigh;
          }
          else {
            std::cout << "none";
          }
          std::cout << ")" << std::endl;
        }
        static itor_type itor;
        // Determine the lower outer limit iterator
        if(this->_have_olow) {
          // ASSUMPTION: index ordering operator can compare against outer_key singleton
          itor = this->_index->lower_bound(ALE::singleton<outer_key_type>(this->_olow));
        }
        else {
          itor = this->_index->begin();
        }
        // Now determine the inner lower limit and set the iterator to that limit within the first segment
        if(this->_have_ilow) {
          // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
          itor = this->_index->lower_bound(ALE::pair<outer_key_type, inner_key_type>(this->_okex(*itor),this->_ilow));
        }
        else {
          // the itor is already in the right place: nothing to do
        }  
        // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
        static itor_type segBndry;
        // Segment boundary set to just above the current (outer_key, inner_key) pair.
        segBndry = this->_index->upper_bound(ALE::pair<outer_key_type, inner_key_type>(this->_okex(*itor),this->_ikex(*itor)));
        if(this->debug()){
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*itor " << *itor; 
          std::cout << ", (okey, ikey): (" << this->_okex(*itor) << ", " << this->_ikex(*itor) << ") " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*segBndry " << *segBndry;
          std::cout << ", (okey, ikey): (" << this->_okex(*segBndry) << ", " << this->_ikex(*segBndry) << ") " << std::endl;
        }
        return iterator(this, itor, segBndry);
      }; // begin()
      //
      #undef  __FUNCT__
      #define __FUNCT__ "next"
      void next(itor_type& itor, itor_type& segBndry, bool inner_strided = false) {
        if(this->debug()) {
          std::cout << std::endl << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "(olow,ohigh): (";
          if(this->_have_olow){
            std::cout << this->_olow << ", ";
          }
          else {
            std::cout << "none, ";
          }
          if(this->_have_ohigh){
            std::cout << this->_ohigh;
          }
          else {
            std::cout << "none";
          }
          std::cout << "), ";
          //
          std::cout << "(ilow,ihigh): (";
          if(this->_have_ilow){
            std::cout << this->_ilow << ", ";
          }
          else {
            std::cout << "none, ";
          }
          if(this->_have_ihigh){
            std::cout << this->_ihigh;
          }
          else {
            std::cout << "none";
          }
          std::cout << ")" << std::endl;
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "starting with *itor " << *itor; 
          std::cout << ", (okey, ikey): (" << this->_okex(*itor) << ", " << this->_ikex(*itor) << ") " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "starting with *segBndry " << *segBndry;
          std::cout << ", (okey, ikey): (" << this->_okex(*segBndry) << ", " << this->_ikex(*segBndry) << ") " << std::endl;
        }
        outer_key_type olow;
        inner_key_type ilow;
        // If iteration over inner keys is to be strided as well, we advance directly to the segment boundary.
        // Effectively, we iterate over segments.
        if(inner_strided) {
          if(this->debug()) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "inner key strided " << std::endl;
          }
          // Advance itor to the segment boundary
          itor = segBndry;
          // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
          olow = this->_okex(*itor);
          ilow = this->_ikex(*itor);
          // Compute the new segment's boundary
          segBndry = this->_index->upper_bound(ALE::pair<outer_key_type, inner_key_type>(olow,ilow));
        }// inner strided
        // Otherwise, we iterate *within* a segment until its end is reached; then the following segment is started.
        else {
          if(this->debug()) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "inner key not strided " << std::endl;
          }
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
            itor = this->_index->upper_bound(ALE::singleton<outer_key_type>(olow));
            // Extract the new outer key
            olow = this->_okex(*itor);
            // Now determine the inner lower limit and set the iterator to that limit within the new segment
            if(this->_have_ilow) {
              ilow = this->_ilow;
              // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
              itor = this->_index->lower_bound(ALE::pair<outer_key_type, inner_key_type>(olow,ilow));
            }
            else {
              // the itor is already in the right place; need to extract the ilow key
              ilow = this->_ikex(*itor);
            }
            // Finally, compute the new segment's boundary
            // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
            segBndry = this->_index->upper_bound(ALE::pair<outer_key_type, inner_key_type>(olow,ilow));
          }
        }// inner not strided
        if(this->debug()) {
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "new *itor " << *itor; 
          std::cout << ", (okey, ikey): (" << this->_okex(*itor) << ", " << this->_ikex(*itor) << ") " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "new *segBndry " << *segBndry;
          std::cout << ", (okey, ikey): (" << this->_okex(*segBndry) << ", " << this->_ikex(*segBndry) << ") " << std::endl;
        }
      };// next()
      //
      #undef  __FUNCT__
      #define __FUNCT__ "end"
      iterator end() {
        if(this->debug()) {
          std::cout << std::endl << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "(olow,ohigh): (";
          if(this->_have_olow){
            std::cout << this->_olow << ", ";
          }
          else {
            std::cout << "none, ";
          }
          if(this->_have_ohigh){
            std::cout << this->_ohigh;
          }
          else {
            std::cout << "none";
          }
          std::cout << "), ";
          //
          std::cout << "(ilow,ihigh): (";
          if(this->_have_ilow){
            std::cout << this->_ilow << ", ";
          }
          else {
            std::cout << "none, ";
          }
          if(this->_have_ihigh){
            std::cout << this->_ihigh;
          }
          else {
            std::cout << "none";
          }
          std::cout << ")" << std::endl;
        }
        static itor_type itor;
        static ritor_type ritor;
        // Determine the upper outer limit
        static outer_key_type ohigh;
        if(this->_have_ohigh) {
          ohigh = this->_ohigh;
        }
        else {
          ritor = this->_index->rbegin();
          ohigh = this->_okex(*ritor);
        }
        // Determine the inner outer limit
        static inner_key_type ihigh;
        if(this->_have_ihigh) {
          ihigh = this->_ihigh;
          // ASSUMPTION: index ordering operator can compare against (outer_key, inner_key) pairs
          itor = this->_index->upper_bound(ALE::pair<outer_key_type, inner_key_type>(ohigh,ihigh));
        }
        else {
          // ASSUMPTION: index ordering operator can compare against outer_key singletons.
          itor = this->_index->upper_bound(ALE::singleton<outer_key_type>(ohigh));
        }
        // use segBndry == itor
        if(this->debug()){
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*itor " << *itor; 
          std::cout << ", (okey, ikey): (" << this->_okex(*itor) << ", " << this->_ikex(*itor) << ") " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": segBndry == itor" << std::endl; 
        }
        return iterator(this, itor, itor); 
      };// end()
      //
      virtual bool contains(const outer_key_type& ok, const inner_key_type& ik) {
        // FIX: This has to be implemented correctly, using the index ordering operator.
        //return (this->_index->find(ALE::pair<outer_key_type,inner_key_type>(ok,ik)) != this->_index->end());
        return true;
      };
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
  }; // namespace XSifterDef
  
  //
  // XSifter definition
  template<typename Arrow_, 
           typename ArrowSupportOrder_= XSifterDef::TargetColorOrder<Arrow_>, 
           typename ArrowConeOrder_   = XSifterDef::SourceColorOrder<Arrow_>, 
           typename Predicate_ = int, typename PredicateOrder_ = std::less<Predicate_> >
  struct XSifter : XObject { // struct XSifter
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
      // Basic interface
      Rec(const arrow_type& a) : arrow_type(a) {};
      Rec(const arrow_type& a, const predicate_type& p) : arrow_type(a), _predicate(p) {};
      // Extended interface
      predicate_type predicate() const{return this->_predicate;};
      source_type    source() const {return this->arrow_type::source();};
      target_type    target() const {return this->arrow_type::target();};
      // Printing
      friend std::ostream& operator<<(std::ostream& os, const Rec& r) {
        os << "[" << (arrow_type)r << "]<" << r._predicate << ">";
        return os;
      }

      // Modifier objects
      struct predicateChanger {
        predicateChanger(const predicate_type& newPredicate) : _newPredicate(newPredicate) {};
        void operator()(Rec& r) { r._predicate = this->_newPredicate;}
      private:
        const predicate_type _newPredicate;
      };
    };// struct Rec
    //
    typedef Rec                              rec_type;
    //
    // Compound orders are assembled here
    //
    typedef std::less<typename rec_type::source_type> source_order_type; 
    typedef std::less<typename rec_type::target_type> target_order_type;
    //
    // Rec 'downward' order type: first order by predicate, then source, then target, etc (Support order)
    struct downward_order_type : public 
    XSifterDef::RecKeyXXXOrder<rec_type, 
                              typename ::boost::multi_index::const_mem_fun<rec_type, predicate_type, &rec_type::predicate>, 
                              predicate_order_type, 
                              XSifterDef::RecKeyXXXOrder<rec_type,
                                                        ::boost::multi_index::const_mem_fun<rec_type, typename rec_type::source_type, &rec_type::source>,
                                                         source_order_type, ArrowSupportOrder_> > {};
    
    //
    // Rec 'upward' order type: first order by predicate, then target, then source, etc (Cone order)
    struct upward_order_type : public 
    XSifterDef::RecKeyXXXOrder<rec_type, 
                              typename ::boost::multi_index::const_mem_fun<rec_type,predicate_type, &rec_type::predicate>, 
                              predicate_order_type,
                              XSifterDef::RecKeyXXXOrder<rec_type,
                                                        ::boost::multi_index::const_mem_fun<rec_type, typename rec_type::target_type, &rec_type::target>,
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
          ::boost::multi_index::tag<DownwardTag>, ::boost::multi_index::identity<rec_type>, downward_order_type
        >,
        ::boost::multi_index::ordered_non_unique<
          ::boost::multi_index::tag<UpwardTag>, ::boost::multi_index::identity<rec_type>, upward_order_type
        > 
      >,
      ALE_ALLOCATOR<rec_type> > 
    rec_set_type;
    //
    // Index types
    typedef typename ::boost::multi_index::index<rec_set_type, UpwardTag>::type   upward_index_type;
    typedef typename ::boost::multi_index::index<rec_set_type, DownwardTag>::type downward_index_type;
    //
    // Sequence types
    template <typename Index_, 
              typename OuterKeyExtractor_, typename InnerKeyExtractor_, typename ValueExtractor_, bool inner_strided_flag = false>
    class ArrowSequence : 
      public XSifterDef::StridedIndexSequence<Index_, OuterKeyExtractor_, InnerKeyExtractor_, ValueExtractor_, inner_strided_flag> {
      // ArrowSequence extends StridedIndexSequence with extra iterator methods.
    public:
      typedef XSifterDef::StridedIndexSequence<Index_, OuterKeyExtractor_, InnerKeyExtractor_, ValueExtractor_, inner_strided_flag> super;
      typedef XSifter                                                                                                              container_type;
      typedef typename super::index_type                                                                                           index_type;
      typedef typename super::outer_key_type                                                                                       outer_key_type;
      typedef typename super::inner_key_type                                                                                       inner_key_type;
      
      // Need to extend the inherited iterators to be able to extract arrow color
      class iterator : public super::iterator {
      public:
        iterator() : super::iterator() {};
        iterator(const typename super::iterator& super_iter) : super::iterator(super_iter) {};
        virtual const source_type& source() const {return this->_itor->_source;};
        virtual const color_type&  color()  const {return this->_itor->_color;};
        virtual const target_type& target() const {return this->_itor->_target;};
        virtual const arrow_type&  arrow()  const {return *(this->_itor);};
      };
    protected:
      container_type *_container;
    public:
      //
      // Basic ArrowSequence interface
      //
      ArrowSequence(const int& debug = 0) : super(debug), _container(NULL) {};
      ArrowSequence(const ArrowSequence& seq) : super(seq), _container(seq._container) {};
      ArrowSequence(container_type *container, index_type *index, const int& debug = 0) : super(index, debug), _container(container) {};
      virtual ~ArrowSequence() {};
      void copy(const ArrowSequence& seq, ArrowSequence& cseq) {
        super::copy(seq,cseq);
        cseq._container = seq._container;
      };
      ArrowSequence& operator=(const ArrowSequence& seq) {
        copy(seq,*this); return *this;
      };
      void reset(container_type *container, index_type *index) {
        this->super::reset(index);
        this->_container = container;
      };
      //
      // Extended ArrowSequence interface
      //
      virtual iterator begin() {
        return this->super::begin();
      };
      //
      virtual iterator end() {
        return this->super::end();
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
        this->_container->addArrow(a);
      };
      //
    };// class ArrowSequence    
      
    //
    // Specialized sequence types
    //
    typedef ArrowSequence<typename ::boost::multi_index::index<rec_set_type, UpwardTag>::type,
                          ::boost::multi_index::const_mem_fun<rec_type, predicate_type, &rec_type::predicate>,
                          ::boost::multi_index::const_mem_fun<rec_type, target_type, &rec_type::target>,
                          ::boost::multi_index::const_mem_fun<rec_type, target_type, &rec_type::target>, 
                          true>                                                       
    BaseSequence;

    typedef ArrowSequence<typename ::boost::multi_index::index<rec_set_type, UpwardTag>::type,
                          ::boost::multi_index::const_mem_fun<rec_type, predicate_type, &rec_type::predicate>,
                          ::boost::multi_index::const_mem_fun<rec_type, target_type, &rec_type::target>,
                          ::boost::multi_index::const_mem_fun<rec_type, source_type, &rec_type::source> >     
    ConeSequence;
    //
    // Basic interface
    //
    XSifter(const MPI_Comm comm, int debug = 0) : XObject(debug) {};// FIXIT: Should really inherit from XParallelObject
    //
    // Extended interface
    //
    void addArrow(const arrow_type& a) {
#ifdef ALE_USE_DEBUGGING
      rec_type r(a);
      this->_rec_set.insert(r);
#else
      this->_rec_set.insert(rec_type(a));
#endif
    };
    void cone(const target_type& t, ConeSequence& seq) {
      seq.reset(this, &::boost::multi_index::get<UpwardTag>(this->_rec_set));
      seq.setInnerLimits(t,t);
    };
    ConeSequence& cone(const target_type& t) {
      static ConeSequence cseq(this->debug());
      this->cone(t,cseq);
      return cseq;
    };
    void base(BaseSequence& seq) {
      seq.reset(this, &::boost::multi_index::get<UpwardTag>(this->_rec_set));
    };
    BaseSequence& base() {
      static BaseSequence bseq(this->debug());
      this->base(bseq);
      return bseq;
    };
    //
    template<typename ostream_type>
    void view(ostream_type& os, const char* label = NULL){
      if(label != NULL) {
        os << "Viewing " << label << " XSifter (debug: " << this->debug() << "): " << std::endl;
      } 
      else {
        os << "Viewing a XSifter (debug: " << this->debug() << "): " << std::endl;
      } 
      upward_index_type&   upward_index   = ::boost::multi_index::get<UpwardTag>(this->_rec_set);
      downward_index_type& downward_index = ::boost::multi_index::get<DownwardTag>(this->_rec_set);
      os << "Downward index: (";
        for(typename downward_index_type::iterator itor = downward_index.begin(); itor != downward_index.end(); ++itor) {
          os << *itor << " ";
        }
      os << ")" << std::endl;
      os << "Upward index: (";
        for(typename upward_index_type::iterator itor = upward_index.begin(); itor != upward_index.end(); ++itor) {
          os << *itor << " ";
        }
      os << ")" << std::endl;
    };
  protected:
    // set of arrow records
    rec_set_type _rec_set;
    
  }; // class XSifter


} // namespace ALE

#endif
