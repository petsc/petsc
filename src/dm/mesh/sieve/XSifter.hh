#ifndef included_ALE_Sifter_hh
#define included_ALE_Sifter_hh


#include <boost/multi_index_container.hpp>
#include <boost/multi_index/key_extractors.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>

#include <boost/lambda/lambda.hpp>
using namespace ::boost::lambda;

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
    int      debug(const int& debug = -1) {if(debug >= 0) {this->_debug = debug;} return this->_debug;};
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
    static int debug   = 0;
    static int codebug = 0;
#define ALE_XDEBUG_DEPTH 5
#define ALE_XDEBUG(n)      ((ALE_XDEBUG_DEPTH - ALE::XSifterDef::debug < n)              || (ALE::XSifterDef::codebug >= n))
#define ALE_XDEBUG_AUTO    ((ALE_XDEBUG_DEPTH - ALE::XSifterDef::debug < __ALE_XDEBUG__) || (ALE::XSifterDef::codebug >= __ALE_XDEBUG__))


    //
    // Key orders
    //
    template<typename OuterKey_, typename InnerKey_, typename OuterKeyOrder_, typename InnerKeyOrder_>
    struct OuterInnerKeyOrder {
      typedef OuterKey_                                 outer_key_type;
      typedef InnerKey_                                 inner_key_type;
      typedef ALE::pair<outer_key_type, inner_key_type> key_pair_type;
      //
      typedef OuterKeyOrder_                            outer_key_order_type;
      typedef InnerKeyOrder_                            inner_key_order_type;
      //
      bool operator()(const outer_key_type ok1, const inner_key_type ik1, const outer_key_type& ok2, const inner_key_type& ik2) {
        static outer_key_order_type okCompare;
        static inner_key_order_type ikCompare;
        return (okCompare(ok1,ok2) || (!okCompare(ok2,ok1) && ikCompare(ik1,ik2)));
      };
    };

    //
    // Rec orders
    //
    // RecKeyOrder compares records by comparing keys of type Key_ extracted from arrows using a KeyExtractor_.
    // In addition, a record can be compared to a single Key_ or another CompatibleKey_.
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
        // and must be unwrapped before the ordering operator is applied.
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
        //        return this->_compare_keys(this->_kex(rec1),this->_kex(rec2)) ||
        //  (!this->_compare_keys(this->_kex(rec2),this->_kex(rec1)) && this->_compare_xxx(rec1,rec2));
        if(this->_compare_keys(this->_kex(rec1), this->_kex(rec2))) 
          return true;
        if(this->_compare_keys(this->_kex(rec2), this->_kex(rec1))) 
          return false;
        if(this->_compare_xxx(rec1,rec2))
          return true;
        return false;           
      };
      template <typename CompatibleKey_>
      bool operator()(const ALE::singleton<CompatibleKey_>& keySingleton, const rec_type& rec1) const {
        // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
        // and must be unwrapped before the ordering operator is applied
        return this->_compare_keys(keySingleton.first, this->_kex(rec1));
      };
      template <typename CompatibleKey_>
      bool operator()(const rec_type& rec1, const ALE::singleton<CompatibleKey_>& keySingleton) const {
        // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
        // and must be unwrapped before the ordering operator is applied
        return this->_compare_keys(this->_kex(rec1), keySingleton.first);
      };
      template <typename CompatibleKey_, typename CompatibleXXXKey_>
      bool operator()(const ALE::pair<CompatibleKey_, CompatibleXXXKey_>& keyPair, const rec_type& rec) const {
        // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
        // and must be unwrapped before the ordering operator is applied
        //
        // We want (key,xxxkey) to be no greater than any (key, xxxkey, ...)
        return this->_compare_keys(keyPair.first, _kex(rec)) ||
          (!this->_compare_keys(_kex(rec), keyPair.first) && this->_compare_xxx(ALE::singleton<CompatibleXXXKey_>(keyPair.second), rec));
        // Note that CompatibleXXXKey_ -- the second key in the pair -- must be wrapped up as a singleton before being passed for comparison against rec
        // to compare_xxx.  This is necessary for compare_xxx to disamiguate comparison of recs to elements of differents types.  In particular, 
        // this is necessary if compare_xxx is of the RecKeyXXXOrder type. Specialization doesn't work, or I don't know how to make it work in this context.
      };
      template <typename CompatibleKey_, typename CompatibleXXXKey_>
      bool operator()(const rec_type& rec, const ALE::pair<CompatibleKey_, CompatibleXXXKey_>& keyPair) const {
        // We want (key,xxxkey) to be no greater than any (key, xxxkey, ...)
        return _compare_keys(_kex(rec), keyPair.first) ||
          (!this->_compare_keys(keyPair.first, _kex(rec)) && this->_compare_xxx(rec,ALE::singleton<CompatibleXXXKey_>(keyPair.second)));
        // Note that CompatibleXXXKey_ -- the second key in the pair -- must be wrapped up as a singleton before being passed for comparison against rec
        // to compare_xxx.  This is necessary for compare_xxx to disamiguate comparison of recs to elements of differents types.  In particular, 
        // this is necessary if compare_xxx is of the RecKeyXXXOrder type. Specialization doesn't work, or I don't know how to make it work in this context.
      };
    };// class RecKeyXXXOrder


    //
    // PredicateTraits encapsulates Predicate types encoding object subsets with a given Predicate value or within a value range.
    template<typename Predicate_> 
    struct PredicateTraits {};
    // Traits of different predicate types are defined via specialization of PredicateTraits.
    // We require that the predicate type act like an int (signed or unsigned).
    //
    template<>
    struct PredicateTraits<int> {
      typedef      int  predicate_type;
      typedef      int  printable_type;
      static const predicate_type default_value;
      static printable_type printable(const predicate_type& p) {return (printable_type)p;};
    };
    const PredicateTraits<int>::predicate_type PredicateTraits<int>::default_value = 0;
    //
    template<>
    struct PredicateTraits<unsigned int> {
      typedef      unsigned int  predicate_type;
      typedef      unsigned int  printable_type;
      static const predicate_type default_value;
      static printable_type printable(const predicate_type& p) {return (printable_type)p;};
    };
    const PredicateTraits<unsigned int>::predicate_type PredicateTraits<unsigned int>::default_value = 0;
    //
    template<>
    struct PredicateTraits<short> {
      typedef      short  predicate_type;
      typedef      short  printable_type;
      static const predicate_type default_value;
      static printable_type printable(const predicate_type& p) {return (printable_type)p;};
    };
    const PredicateTraits<short>::predicate_type PredicateTraits<short>::default_value = 0;
    //
    template<>
    struct PredicateTraits<char> {
      typedef char  predicate_type;
      typedef short printable_type;
      static const predicate_type default_value;
      static printable_type printable(const predicate_type& p) {return (printable_type)p;};
    };
    const PredicateTraits<char>::predicate_type PredicateTraits<char>::default_value = '\0';


    //
    // RangeFilter defines a subset of Index_ based on the high and low value of a key.
    // The type and ordering of the key are defined by KeyExtractor_ and KeyCompare_ respectively.
    // 
    #undef  __CLASS__
    #define __CLASS__ "RangeFilter"
    template <typename Index_, typename KeyExtractor_, typename KeyCompare_, bool Strided = false>
    class RangeFilter : XObject {
    public:
      typedef Index_                                           index_type;
      typedef KeyExtractor_                                    key_extractor_type;
      typedef KeyCompare_                                      key_compare_type;
      typedef typename key_extractor_type::result_type         key_type;
      typedef PredicateTraits<key_type>                        key_traits;
      typedef typename index_type::iterator                    iterator;
    protected:
      index_type*                           _index; // a pointer rather than a reference is used for use in default constructor
      bool                                  _have_low, _have_high;
      key_type                              _low, _high;
    public:
      // Basic interface
      RangeFilter(index_type* index) : 
        XObject(), _index(index), _have_low(false), _have_high(false) {};
      RangeFilter(index_type* index, const key_type& low, const key_type& high) : 
        XObject(), _index(index), _have_low(true), _have_high(true), _low(low), _high(high)  {};
      RangeFilter(const RangeFilter& f) : 
        XObject(f), _index(f._index), _have_low(f._have_low), _have_high(f._have_high), _low(f._low), _high(f._high) {};
      ~RangeFilter(){};
      //
      void setLow(const key_type& low)   {this->_low  = low;  this->_have_low  = true;};
      void setHigh(const key_type& high) {this->_high = high; this->_have_high = true;};
      //
      key_type         low()       const {return this->_low;};
      key_type         high()      const {return this->_high;};
      bool             haveLow()   const {return this->_have_low;};
      bool             haveHigh()  const {return this->_have_high;};
      //
      static key_type  key(const iterator& iter)  { static key_extractor_type kex; return kex(*iter);};
//       //
//       #undef  __FUNCT__
//       #define __FUNCT__ "begin"
//       // Returns the start of the first allowed segment.
//       void begin(iterator& iter) const {         
//         if(ALE_XDEBUG(1)) {
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//           std::cout << "filter: " << *this << std::endl;
//         }
//         if(this->_have_low) {
//           // ASSUMPTION: index ordering operator can compare against key_type singleton
//           iter = this->_index->lower_bound(ALE::singleton<key_type>(this->_low));
//         }
//         else {
//           iter = this->_index->begin();
//         }
//         if(ALE_XDEBUG(1)){
//           //
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter " << *iter << std::endl; 
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
//         }
//       };//begin()

//       //
//       #undef  __FUNCT__
//       #define __FUNCT__ "begin<OuterFilter_>"
//       #undef  __ALE_XSIFTER_DEBUG__
//       #define __ALE_XSIFTER_DEBUG__ 5
//       // Returns the start of the first allowed subsegment following current_iter within the same segment (defined by the outer key).
//       // The outer key is extracted using a OuterFilter, although only the key extraction capabilities of OuterFilter are used.
//       template<typename OuterFilter_>
//       void begin(const iterator& current_iter, const iterator& outerEnd, const OuterFilter_& outer_filter, iterator& iter) const { 
//         typedef typename OuterFilter_::key_type         outer_key_type;
//         typedef typename OuterFilter_::key_compare_type outer_key_compare_type;
//         static OuterInnerKeyOrder<outer_key_type, key_type, outer_key_compare_type, key_compare_type> oiKeyCompare;
//         if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//           std::cout << "filter: " << *this << ", outer filter: " << outer_filter  << ", *current_iter: " << *current_iter << std::endl;
//         }
//         // If current_iter precedes inner _low, go to inner _low
//         if(this->_have_low &&  oiKeyCompare(outer_filter.key(current_iter),this->key(current_iter), outer_filter.key(current_iter),this->_low)) {
//           if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
//             std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//             std::cout << "looking for lower bound of pair (okey,ikey): (" << outer_filter.key(current_iter) << ", " << this->_low << ")" << std::endl;
//           }
//           // ASSUMPTION: index ordering operator can compare against (outer_key_type,key_type) pairs
//           iter = this->_index->lower_bound(ALE::pair<outer_key_type,key_type>(outer_filter.key(current_iter),this->_low));
//         }
//         else {
//           // If there is no inner _low or current_iter doesn't precede it, return current_iter unchanged.
//           iter = current_iter;
//         }
//         if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)){
//           //
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter " << *iter << std::endl; 
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
//         }
//       };//begin<OuterFilter_>()
//       //
//       #undef  __FUNCT__
//       #define __FUNCT__ "next(current_iterator)"
//       // Returns the start of the first allowed segment following current_iter.
//       void next(const iterator& current_iter, iterator& iter) const {         
//         static key_compare_type keyCompare;
//         if(ALE_XDEBUG(1)) {
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//           std::cout << "filter: " << *this << std::endl;
//         }
//         // If current_iter precedes _low, go to _low;  
//         if(this->_have_low && keyCompare(this->key(current_iter),this->_low)) {
//           // ASSUMPTION: index ordering operator can compare against key_type singleton
//           iter = this->_index->lower_bound(ALE::singleton<key_type>(this->_low));
//         }
//         // If there is no _low or current_iter does not precede it, go to segment end
//         else {
//           this->end(iter);
//         }
//         if(ALE_XDEBUG(1)){
//           //
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter " << *iter << std::endl; 
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
//         }
//       };//next(current_iterator)
//       //
//       #undef  __FUNCT__
//       #define __FUNCT__ "next<OuterFilter_>"
//       #undef  __ALE_XSIFTER_DEBUG__
//       #define __ALE_XSIFTER_DEBUG__ 5
//       // Returns the start of the first allowed subsegment following current_iter within the same segment (defined by the outer key).
//       // The outer key is extracted using a OuterFilter, although only the key extraction capabilities of OuterFilter are used.
//       template<typename OuterFilter_>
//       void next(const iterator& current_iter, const iterator& outerSegmentEnd, const OuterFilter_& outer_filter, iterator& iter) const { 
//         typedef typename OuterFilter_::key_type         outer_key_type;
//         typedef typename OuterFilter_::key_compare_type outer_key_compare_type;
//         static OuterInnerKeyOrder<outer_key_type, key_type, outer_key_compare_type, key_compare_type> oiKeyCompare;
//         if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//           std::cout << "filter: " << *this << ", outer filter: " << outer_filter  << ", *current_iter: " << *current_iter;
//           std::cout << "*outerSegmentEnd: " << *outerSegmentEnd << std::endl;
//         }
//         // If current_iter precedes inner _low, go to inner _low
//         if(this->_have_low &&  oiKeyCompare(outer_filter.key(current_iter),this->key(current_iter), outer_filter.key(current_iter),this->_low)) {
//           if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
//             std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//             std::cout << "looking for lower bound of pair (okey,ikey): (" << outer_filter.key(current_iter) << ", " << this->_low << ")" << std::endl;
//           }
//           // ASSUMPTION: index ordering operator can compare against (outer_key_type,key_type) pairs
//           iter = this->_index->lower_bound(ALE::pair<outer_key_type,key_type>(outer_filter.key(current_iter),this->_low));
//         }
//         else {
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//           std::cout << "no inner _low following current_iter; going to the end of subsegment" << std::endl;
//           // If there is no inner _low or current_iter doesn't precede it, go to the end of the subsegment.
//           // CONTINUE:
//           // IMPROVE: should pass innerSegmentEnd in (perhaps instead of outerSegmentEnd).
//           this->end(current_iter, outerSegmentEnd, outer_filter, iter);
//         }
//         if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)){
//           //
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter " << *iter << std::endl; 
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
//         }
//       };//next<OuterFilter_>()
//       //
//       #undef  __FUNCT__
//       #define __FUNCT__ "end"
//       #undef  __ALE_XDEBUG__ 
//       #define __ALE_XDEBUG__ 5
//       // Returns the end of the last allowed segment within the index.
//       void end(iterator& iter) const {
//         if(ALE_XDEBUG(__ALE_XDEBUG__)) {
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl; 
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//           std::cout << "filter: " << *this << std::endl;
//         }
//         // Determine the upper limit
//         if(this->_have_high) {
//           if(ALE_XDEBUG(__ALE_XDEBUG__)) {
//             std::cout << __CLASS__ << "::" << __FUNCT__ << ": have_high, looking for upper bound of " << this->high() << std::endl;
//           }
//           // ASSUMPTION: index ordering operator can compare against (key_type) singletons
//           iter = this->_index->upper_bound(ALE::singleton<key_type>(this->high()));
//         }
//         else {
//           if(ALE_XDEBUG(__ALE_XDEBUG__)) {
//             std::cout << __CLASS__ << "::" << __FUNCT__ << ": !have_high, looking for index end" << std::endl;
//           }
//           iter = this->_index->end();
//         }
//         if(ALE_XDEBUG(__ALE_XDEBUG__)){
//           //
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter " << *iter << std::endl; 
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
//         }
//       };//end()
//       //
//       #undef  __FUNCT__
//       #define __FUNCT__ "end<OuterFilter_>"
//       #undef  __ALE_XSIFTER_DEBUG__
//       #define __ALE_XSIFTER_DEBUG__ 5
//       // Returns the end of the allowed subsegment following current_iter within the same allowable segment.
//       // The outer key is extracted using a OuterFilter, although only the key extraction capabilities of OuterFilter are used.
//       template <typename OuterFilter_>
//       void end(const iterator& current_iter, const iterator& outerEnd, const OuterFilter_& outer_filter, iterator& iter) const {
//         typedef typename OuterFilter_::key_type         outer_key_type;
//         typedef typename OuterFilter_::key_compare_type outer_key_compare_type;
//         static OuterInnerKeyOrder<outer_key_type, key_type, outer_key_compare_type, key_compare_type> oiKeyCompare;
//         if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//           std::cout << "filter: " << *this << ", outer filter: " << outer_filter << ", *current_iter: " << *current_iter;
//           std::cout << "*outerEnd: " << *outerEnd << std::endl;
//         }
//         if(current_iter == this->_index->end()) {
//           iter = current_iter;
//         }
//         else {
//           // If there is a high
//           if(this->_have_high) {
//             // and if current_iter precedes _high within the segment, return an upper bound on _high within the segment.
//             if(oiKeyCompare(outer_filter.key(current_iter), this->key(current_iter), outer_filter.key(current_iter), this->_high)){
//               iter = this->_index->upper_bound(ALE::pair<outer_key_type,key_type>(outer_filter.key(current_iter),this->_high));
//             }
//             // else, current_iter does not precede _high and is returned unchanged
//             else { 
//               iter = current_iter;
//             }
//           }
//           // If the inner filter is unbounded, return the end of the outer segment
//           else {
//             iter = outerEnd;
//           }
//         }
//         if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)){
//           //
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter " << *iter << std::endl; 
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
//         }
//       };//end<OuterFilter_>()
//       //
//       #undef  __FUNCT__
//       #define __FUNCT__ "last"
//       #undef  __ALE_XDEBUG__ 
//       #define __ALE_XDEBUG__ 5
//       // Returns the last iterator of the last allowed segment within the index.
//       void last(iterator& iter) const {
//         if(ALE_XDEBUG(__ALE_XDEBUG__)) {
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl; 
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//           std::cout << "filter: " << *this << std::endl;
//         }
//         iterator filterEnd;
//         this->end(filterEnd);
//         if(ALE_XDEBUG(__ALE_XDEBUG__)) {
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//           std::cout << "*filterEnd: " << *filterEnd << std::endl;
//         }
//         if(filterEnd != this->_index->begin()) {
//           if(ALE_XDEBUG(__ALE_XDEBUG__)) {
//             std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//             std::cout << "filterEnd not at index beginning" << std::endl;
//             if(filterEnd == this->_index->end()){
//               std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//               std::cout << "filterEnd at index end" << std::endl;
//             }
//             else {
//               std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
//               std::cout << "filterEnd not at index end: " << std::endl;
//             }
//           }
//             iter = --filterEnd;
//         }
//         if(ALE_XDEBUG(__ALE_XDEBUG__)){
//           //
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter " << *iter << std::endl; 
//           std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
//         }
//       };//last()
      //
      #undef  __FUNCT__
      #define __FUNCT__ "firstSegment"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 5
      // Returns the first allowed segment.
      void firstSegment(iterator& segmentBegin, iterator& segmentEnd) const {         
        if(ALE_XDEBUG(__ALE_XDEBUG__)) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << std::endl;
        }
        if(this->_have_low) {
          segmentBegin = this->_index->lower_bound(ALE::singleton<key_type>(this->_low));
        }
        else {
          segmentBegin = this->_index->begin();
        }
        if(Strided) {
          if(ALE_XDEBUG(__ALE_XDEBUG__)) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": strided filter" << std::endl;
          }
          if(this->_have_high) {
            segmentEnd = this->_index->upper_bound(ALE::singleton<key_type>(this->_high));
          }
          else {
            segmentEnd = this->_index->end();
          }
        }
        else {
          if(ALE_XDEBUG(__ALE_XDEBUG__)) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": non-strided filter" << std::endl;
          }
          segmentEnd   = this->_index->upper_bound(ALE::singleton<key_type>(this->key(segmentBegin)));
        }
        //
        if(ALE_XDEBUG(__ALE_XDEBUG__)){
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*segmentBegin: " << *segmentBegin;
          std::cout << ", *segmentEnd: " << *segmentEnd << std::endl; 
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        }
      };//firstSegment()
      //
      #undef  __FUNCT__
      #define __FUNCT__ "firstSegment<OuterFilter_>"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 5
      // Returns the first allowed subsegment within the segment containing current_iter.
      template<typename OuterFilter_>
      void firstSegment(const OuterFilter_& outer_filter, const iterator& current_iter, const iterator& outerEnd, iterator& segmentBegin, iterator& segmentEnd) const {
        typedef typename OuterFilter_::key_type         outer_key_type;
        if(ALE_XDEBUG(__ALE_XDEBUG__)) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << ", outer filter: " << outer_filter << ", *current_iter: " << *current_iter << std::endl;
        }
        if(this->_have_low) {
          segmentBegin = this->_index->lower_bound(ALE::pair<outer_key_type, key_type>(outer_filter.key(current_iter), this->_low));
        }
        else {
          segmentBegin = current_iter;
        }
        if(Strided) {
          if(ALE_XDEBUG(__ALE_XDEBUG__)) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": strided filter" << std::endl;
          }
          if(this->_have_high) {
            segmentEnd = this->_index->upper_bound(ALE::pair<outer_key_type, key_type>(outer_filter.key(current_iter),this->_high));
          }
          else {
            segmentEnd = this->_index->end();
          }
        }
        else {
          if(ALE_XDEBUG(__ALE_XDEBUG__)) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": non-strided filter" << std::endl;
          }
          segmentEnd = this->_index->upper_bound(ALE::pair<outer_key_type, key_type>(outer_filter.key(current_iter), this->key(segmentBegin)));
        }
        //
        if(ALE_XDEBUG(__ALE_XDEBUG__)){
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*segmentBegin: " << *segmentBegin;
          std::cout << ", *segmentEnd: " << *segmentEnd << std::endl; 
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        }
      };//firstSegment<OuterFilter_>()
      //
      #undef  __FUNCT__
      #define __FUNCT__ "nextSegment"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 5
      // Returns the allowed segment immediately following current_iter.
      void nextSegment(const iterator& current_iter, iterator& segmentBegin, iterator& segmentEnd) const {         
        static key_compare_type keyCompare;
        if(ALE_XDEBUG(__ALE_XDEBUG__)) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << std::endl;
        }
        // Go to the segmentEnd --- iterator with the following key;  
        segmentBegin = segmentEnd;
        // Check for an overshoot
        if(this->_have_high && keyCompare(this->_high, this->key(segmentBegin))) {// overshoot
          // Go to the end of index
          segmentBegin = this->_index->end(); 
          segmentEnd = this->_index->end();
        }
        else { // no overshoot
          segmentEnd= this->_index->upper_bound(ALE::singleton<key_type>(this->key(segmentBegin)));
        }
        if(ALE_XDEBUG(__ALE_XDEBUG__)){
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*segmentBegin: " << *segmentBegin;
          std::cout << ", *segmentEnd: " << *segmentEnd << std::endl; 
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        }
      };//nextSegment
      //
      #undef  __FUNCT__
      #define __FUNCT__ "nextSegment<OuterFilter_>"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 5
      // Returns the allowed subsegment immediately following current_iter within the same segment.
      template <typename OuterFilter_>
      void nextSegment(const OuterFilter_& outer_filter, const iterator& current_iter, const iterator& outerEnd, iterator& segmentBegin, iterator& segmentEnd) const {         
        typedef typename OuterFilter_::key_type         outer_key_type;
        typedef typename OuterFilter_::key_compare_type outer_key_compare_type;
        static outer_key_compare_type oKeyCompare;
        static OuterInnerKeyOrder<outer_key_type, key_type, outer_key_compare_type, key_compare_type> oiKeyCompare;
        if(ALE_XDEBUG(__ALE_XDEBUG__)) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << ", outer filter: " << outer_filter << ", *current_iter: " << *current_iter << std::endl;
        }
        // Check if current_iter is at the end of the index already
        if(current_iter == this->_index->end()) {
          segmentBegin = this->_index->end();
          segmentEnd   = this->_index->end();
        }
        else {// if current_iter is not at index end
          // Go to segmentEnd -- iterator with the following key
          segmentBegin = segmentEnd;
          // Check for an overshot: whether segmentBegin follows inner _high;  
          if( !oKeyCompare(outer_filter.key(segmentBegin), outer_filter.key(outerEnd)) /* outerEnd overshot */ ||
              (this->_have_high && 
               oiKeyCompare(outer_filter.key(current_iter), this->_high, outer_filter.key(segmentBegin), this->key(segmentBegin))) /* inner high overshot */
            )
          {// overshoot
            // go to the outer end
            segmentBegin = outerEnd; segmentEnd = outerEnd;
          }
          else {// no overshoot
            segmentEnd = this->_index->upper_bound(ALE::pair<outer_key_type, key_type>(outer_filter.key(segmentBegin), this->key(segmentBegin)));
          }
        }// if current_iter is not at index end
        if(ALE_XDEBUG(__ALE_XDEBUG__)){
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*segmentBegin " << *segmentBegin;
          std::cout << ", *segmentEnd: " << *segmentEnd << std::endl; 
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        }
      };//nextSegment<OuterFilter_>
      //
      friend std::ostream& operator<<(std::ostream& os, const RangeFilter& f) {
        os << "[low, high] = [";
        if(f.haveLow()){
          os << ((typename key_traits::printable_type)(f.low())) << ",";
        }
        else {
          os << "none, ";
        }
        if(f.haveHigh()) {
          os << ((typename key_traits::printable_type)(f.high())); 
        }
        else {
          os << "none";
        }
        os << "]";
        return os;
      };
      friend std::ostream& operator<<(std::ostream& os, const Obj<RangeFilter>& f) {
        return (os << f.object());
      };
    };// RangeFilter

    //
    // FilteredIndexSequence definition
    // 
    //   Defines a sequence representing a subset of a multi_index container defined by its Index_ which is ordered lexicographically.
    // The ordering is controlled by a pair of filters (OuterFilter_ and InnerFilter_ types).
    // The elements of the sequence are a subset of an Index_ ordered lexicographically by (OuterKey_,InnerKey_) pairs and such that 
    // each key lies in the range of OuterFilter_/InnerFilter_ respectively.  
    // A sequence defines output iterators (input iterators in std terminology) for traversing a subset of an Index_ object.
    // Upon dereferencing values are extracted from each result record using a ValueExtractor_ object.
    //   More precisely, the index can be viewed as oredered by (OuterKey_,InnerKey_,RemainderKey_) triples, where the RemainderKey_
    // type is not explicitly known to FilteredIndexSequence.  The Filters only restrict the allowable values of the first two keys, 
    // while all RemainderKey_ values are allowed. By default a FilteredIndexSequence will traverse ALL entries with a given leading 
    // (OuterKey_,InnerKey_) pair.  However, setting the template parameter 'Strided = true' will cause only the first elements of 
    // each segment with the same (OuterKey_,InnerKey_) pair to be traversed.  In other words, each (OuterKey_,InnerKey_) pair will 
    // be seen once only.  
    //   Ideally, 'Strided' should parameterize different implementations, but right now there is an 'if' test. An opporunity for improvement.
    // 
    #undef  __CLASS__
    #define __CLASS__ "FilteredIndexSequence"
    template <typename Index_, typename OuterFilter_, typename InnerFilter_, 
              typename ValueExtractor_ = ::boost::multi_index::identity<typename Index_::value_type>, bool Strided = false >
    struct FilteredIndexSequence : XObject {
      typedef Index_                                           index_type;
      typedef InnerFilter_                                     inner_filter_type;
      typedef OuterFilter_                                     outer_filter_type;
      //
      typedef typename outer_filter_type::key_extractor_type   outer_key_extractor_type;
      typedef typename outer_key_extractor_type::result_type   outer_key_type;
      typedef typename inner_filter_type::key_extractor_type   inner_key_extractor_type;
      typedef typename inner_key_extractor_type::result_type   inner_key_type;
      //
      typedef ValueExtractor_                                  value_extractor_type;
      typedef typename value_extractor_type::result_type       value_type;
      typedef typename index_type::size_type                   size_type;
      //
      typedef typename index_type::iterator                    itor_type;
      typedef typename index_type::reverse_iterator            ritor_type;
      //
      class iterator {
      public:
        // Parent sequence type
        typedef FilteredIndexSequence                   sequence_type;
        // Standard iterator typedefs
        typedef std::input_iterator_tag                iterator_category;
        typedef int                                    difference_type;
        typedef value_type*                            pointer;
        typedef value_type&                            reference;
        /* value_type defined in the containing FilteredIndexSequence */
      protected:
        // Parent sequence
        sequence_type  *_sequence;
        // Underlying iterator & segment boundary
        itor_type       _itor;
        itor_type       _outerEnd, _innerEnd;
        //
        // Key and Value extractors
        outer_key_extractor_type _okex;
        inner_key_extractor_type _ikex;
        value_extractor_type     _ex;
      public:
        iterator() : _sequence(NULL) {};
        iterator(sequence_type *sequence, const itor_type& itor, const itor_type& outerEnd, const itor_type& innerEnd) : 
          _sequence(sequence), _itor(itor), _outerEnd(outerEnd), _innerEnd(innerEnd) {};
        iterator(const iterator& iter):_sequence(iter._sequence), _itor(iter._itor), _outerEnd(iter._outerEnd), _innerEnd(iter._innerEnd) {};
        virtual ~iterator() {};
        virtual bool              operator==(const iterator& iter) const {return this->_itor == iter._itor;};
        virtual bool              operator!=(const iterator& iter) const {return this->_itor != iter._itor;};
        // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
        virtual const value_type  operator*() const {return _ex(*(this->_itor));};
        virtual iterator   operator++() {
          this->_sequence->next(this->_itor, this->_outerEnd, this->_innerEnd);
          return *this;
        };
        virtual iterator   operator++(int n) {iterator tmp(*this); ++(*this); return tmp;};
      };// class iterator
    protected:
      //
      index_type      *_index;
      //
      outer_filter_type   _outer_filter;
      inner_filter_type   _inner_filter;
      //
      outer_key_extractor_type _okex;
      inner_key_extractor_type _ikex;
    public:
      //
      // Basic interface
      //
      FilteredIndexSequence() : XObject(), _index(NULL), _outer_filter(NULL), _inner_filter(NULL) {};
      FilteredIndexSequence(index_type *index, const outer_filter_type& outer_filter, const inner_filter_type& inner_filter) : 
        XObject(), _index(index), _outer_filter(outer_filter), _inner_filter(inner_filter){};
      FilteredIndexSequence(const FilteredIndexSequence& seq) : 
        XObject(seq), _index(seq._index), _outer_filter(seq._outer_filter), _inner_filter(seq._inner_filter) {};
      virtual ~FilteredIndexSequence() {};
      // 
      void copy(const FilteredIndexSequence& seq, FilteredIndexSequence cseq) {
        cseq._index = seq._index; 
        cseq._inner_filter = seq._inner_filter;
        cseq._outer_filter = seq._outer_filter;
      };
      FilteredIndexSequence& operator=(const FilteredIndexSequence& seq) {
        copy(seq,*this); return *this;
      };
      void reset(index_type *index, const outer_filter_type& outer_filter, const inner_filter_type& inner_filter) {
        this->_index         = index;
        this->_inner_filter  = inner_filter;
        this->_outer_filter  = outer_filter;
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
      // 
      inner_filter_type& innerFilter(){return this->_inner_filter;};
      outer_filter_type& outerFilter(){return this->_outer_filter;};
      void setInnerFilter(const inner_filter_type& inner_filter) {this->_inner_filter = inner_filter;};
      void setOuterFilter(const outer_filter_type& outer_filter) {this->_outer_filter = outer_filter;};
      //
      #undef  __FUNCT__
      #define __FUNCT__ "begin"
      iterator begin() {
        if(ALE_XDEBUG(1)) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "outer filter: " << this->outerFilter() << ", ";
          std::cout << "inner filter: " << this->innerFilter() << std::endl;
        }
        static itor_type itor, outerEnd, innerEnd;
        this->outerFilter().firstSegment(itor, outerEnd);
        this->innerFilter().firstSegment(this->outerFilter(), itor, outerEnd, itor, innerEnd);
        if(ALE_XDEBUG(1)){
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*itor: " << *itor;
          std::cout << ", (okey, ikey): (" << this->outerFilter().key(itor) << ", " << this->innerFilter().key(itor) << ") ";
          std::cout << ", *outerEnd: " << *outerEnd << ", *innerEnd: " << *innerEnd; 
          std::cout << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        }
        return iterator(this, itor, outerEnd, innerEnd);
      }; // begin()
      //
      #undef  __FUNCT__
      #define __FUNCT__ "next"
      #undef  __ALE_XSIFTER_DEBUG__
      #define __ALE_XSIFTER_DEBUG__ 5
      void next(itor_type& itor, itor_type& outerEnd, itor_type& innerEnd) {
        if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "outer filter: " << this->outerFilter() << ", ";
          std::cout << "inner filter: " << this->innerFilter() << std::endl;
          if(Strided) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "strided sequence" << std::endl;
          }
          else {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "non-strided sequence" << std::endl;
          }
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "starting with *itor " << *itor; 
          std::cout << ", (okey, ikey): (" << this->outerFilter().key(itor) << ", " << this->innerFilter().key(itor) << ") ";
          std::cout << ", *outerEnd: " << *outerEnd << ", *innerEnd: " << *innerEnd; 
          std::cout << std::endl;
        }
        // We assume that it is safe to advance the iterator first and then check for segment ends.
        // If iteration is to be strided we skip the remainder of the current subsegment and go over to the following subsegment within the same segment:
        // effectively, we iterate over subsegments.
        if(Strided) {
          if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "strided sequence" << std::endl;
          }
          this->innerFilter().nextSegment(this->outerFilter(), itor, outerEnd, itor, innerEnd);
        }// Strided
        // Otherwise, we iterate *within* a segment until its end is reached; then the following segment is started.
        else {
          if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "non-strided sequence" << std::endl;
          }
          ++itor; 
        }// not Strided

        // Check to see if we have reached the subsegment end
        if(itor == innerEnd) { // at subsegment end
          if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": at inner end ..." << std::endl;
          }
          // Check whether the end of the outer segment has been reached
          if(itor != outerEnd) {
            if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
              std::cout << __CLASS__ << "::" << __FUNCT__ << ": but not at outer end" << std::endl;
            }
            // go to the next inner segment
            this->innerFilter().nextSegment(this->outerFilter(), itor, outerEnd, itor, innerEnd);
          }
          else {// go to the next outer segment
            if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
              std::cout << __CLASS__ << "::" << __FUNCT__ << ": and at outer end" << std::endl;
            }
            this->outerFilter().nextSegment(itor, itor, outerEnd);
            // Select the first inner segment within the new outer segment
            this->innerFilter().firstSegment(this->outerFilter(), itor, outerEnd, itor, innerEnd);
          }
        }
        else { // not at subsegment end
          if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": not at inner end" << std::endl;
          }
        }// not at subsegment end
        if(ALE_XDEBUG(__ALE_XSIFTER_DEBUG__)) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "new *itor " << *itor; 
          std::cout << ", (okey, ikey): (" << this->outerFilter().key(itor) << ", " << this->innerFilter().key(itor) << ") ";
          std::cout << ", *outerEnd: " << *outerEnd << ", *innerEnd: " << *innerEnd; 
          std::cout << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        }
      };// next()
      //
      #undef  __FUNCT__
      #define __FUNCT__ "end"
      #undef  __ALE_XDEBUG__
      #define __ALE_XDEBUG__ 4
      iterator end() {
        if(ALE_XDEBUG(__ALE_XDEBUG__)) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>>" << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "outer filter: " << this->outerFilter() << ", ";
          std::cout << "inner filter: " << this->innerFilter() << std::endl;
          if(Strided) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "strided sequence" << std::endl;
          }
          else {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "non-strided sequence" << std::endl;
          }
        }
        static itor_type itor, outerEnd, innerEnd;
        itor  = this->_index->end();
        outerEnd = itor;
        innerEnd = itor;
        if(ALE_XDEBUG(__ALE_XDEBUG__)){
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*itor: " << *itor; 
          std::cout << ", (okey, ikey): (" << this->outerFilter().key(itor) << ", " << this->innerFilter().key(itor) << ") ";
          std::cout << ", *outerEnd: " << *outerEnd << ", *innerEnd: " << *innerEnd;           
          std::cout << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<<" << std::endl;
        }
        return iterator(this, itor, outerEnd, innerEnd); 
      };// end()
      //
      virtual bool contains(const outer_key_type& ok, const inner_key_type& ik) {
        // FIX: This has to be implemented correctly, using the index ordering operator.
        //return (this->_index->find(ALE::pair<outer_key_type,inner_key_type>(ok,ik)) != this->_index->end());
        return true;
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
    };// class FilteredIndexSequence    
    

    // Definitions of typical XSifter usage of records, orderings, etc.

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
    // Arrow definition: a concrete arrow; other Arrow definitions are possible, since orders above are templated on it; must have expected const_mem_funs
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

  }; // namespace XSifterDef
  

  //
  // XSifter definition
  //
  template<typename Arrow_, 
           typename ArrowSupportOrder_= XSifterDef::TargetColorOrder<Arrow_>, 
           typename ArrowConeOrder_   = XSifterDef::SourceColorOrder<Arrow_>, 
           typename Predicate_ = unsigned int, typename PredicateOrder_ = std::less<Predicate_> >
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
    typedef Predicate_                                        predicate_type;
    typedef ALE::XSifterDef::PredicateTraits<predicate_type>  predicate_traits;
    typedef PredicateOrder_                                   predicate_order_type;
    struct Rec : public arrow_type {
    public:
      //
      // Re-export typedefs
      //
      typedef typename arrow_type::source_type        source_type;
      typedef typename arrow_type::target_type        target_type;
      typedef typename arrow_type::color_type         color_type;
    protected:
      // Predicate stored alongside the arrow data
      predicate_type _predicate;
    public:
      // Basic interface
      Rec(const arrow_type& a) : arrow_type(a), _predicate(predicate_traits::default_value) {};
      Rec(const arrow_type& a, const predicate_type& p) : arrow_type(a), _predicate(p) {};
      // Extended interface
      predicate_type predicate() const{return this->_predicate;};
      source_type    source() const {return this->arrow_type::source();};
      target_type    target() const {return this->arrow_type::target();};
      color_type     color()  const {return this->arrow_type::color();};
      // Printing
      friend std::ostream& operator<<(std::ostream& os, const Rec& r) {
        os << "<" << predicate_traits::printable(r._predicate) << ">" << "[" << (arrow_type)r << "]";
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
    // Key extractors are defined here
    //
    typedef ::boost::multi_index::const_mem_fun<rec_type, source_type,    &rec_type::source>    source_extractor_type;
    typedef ::boost::multi_index::const_mem_fun<rec_type, target_type,    &rec_type::target>    target_extractor_type;
    typedef ::boost::multi_index::const_mem_fun<rec_type, color_type,     &rec_type::color>     color_extractor_type;
    typedef ::boost::multi_index::const_mem_fun<rec_type, predicate_type, &rec_type::predicate> predicate_extractor_type;
    //
    // Orders are defined here
    //
    typedef std::less<typename rec_type::source_type> source_order_type; 
    typedef std::less<typename rec_type::target_type> target_order_type;
    //
    // Rec 'Base' order type: first order by predicate, then target
    struct base_order_type : public 
    XSifterDef::RecKeyXXXOrder<rec_type, 
                              typename ::boost::multi_index::const_mem_fun<rec_type,predicate_type, &rec_type::predicate>, 
                              predicate_order_type,
                              XSifterDef::RecKeyOrder<rec_type,
                                                      ::boost::multi_index::const_mem_fun<rec_type, typename rec_type::target_type, &rec_type::target>,
                                                      target_order_type> >
    {};
    // Rec 'Cone' order type: first by target, then predicate
    struct cone_order_type : public 
    XSifterDef::RecKeyXXXOrder<rec_type, 
                              typename ::boost::multi_index::const_mem_fun<rec_type,target_type, &rec_type::target>, 
                              target_order_type,
                              XSifterDef::RecKeyXXXOrder<rec_type,
                                                         ::boost::multi_index::const_mem_fun<rec_type, predicate_type,&rec_type::predicate>,
                                                       predicate_order_type, 
                                                       ArrowConeOrder_> >
    {};
    
    //
    // Index tags
    //
    struct                                   BaseTag{};
    struct                                   ConeTag{};
    
    // Rec set type
    typedef ::boost::multi_index::multi_index_container< 
      rec_type,
      ::boost::multi_index::indexed_by< 
        ::boost::multi_index::ordered_non_unique<
          ::boost::multi_index::tag<BaseTag>, ::boost::multi_index::identity<rec_type>, base_order_type
        >,
        ::boost::multi_index::ordered_non_unique<
          ::boost::multi_index::tag<ConeTag>, ::boost::multi_index::identity<rec_type>, cone_order_type
        > 
      >,
      ALE_ALLOCATOR<rec_type> > 
    rec_set_type;
    //
    // Index types
    //
    typedef typename ::boost::multi_index::index<rec_set_type, BaseTag>::type base_index_type;
    typedef typename ::boost::multi_index::index<rec_set_type, ConeTag>::type cone_index_type;
    //
    // Sequence types
    //
    template <typename Index_, 
              typename OuterFilter_, typename InnerFilter_, typename ValueExtractor_, bool Strided = false>
    class ArrowSequence : 
      public XSifterDef::FilteredIndexSequence<Index_, OuterFilter_, InnerFilter_, ValueExtractor_, Strided> {
      // ArrowSequence extends FilteredIndexSequence with extra iterator methods.
    public:
      typedef XSifterDef::FilteredIndexSequence<Index_, OuterFilter_, InnerFilter_, ValueExtractor_, Strided> super;
      typedef XSifter                                                                                                               container_type;
      typedef typename super::index_type                                                                                            index_type;
      typedef typename super::outer_filter_type                                                                                     outer_filter_type;
      typedef typename super::inner_filter_type                                                                                     inner_filter_type;
      typedef typename super::outer_key_type                                                                                        outer_key_type;
      typedef typename super::inner_key_type                                                                                        inner_key_type;
      
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
      ArrowSequence() : super(), _container(NULL) {};
      ArrowSequence(const ArrowSequence& seq) : super(seq), _container(seq._container) {};
      ArrowSequence(container_type *container, index_type *index, const outer_filter_type& outer_filter, const inner_filter_type& inner_filter) : 
        super(index, outer_filter, inner_filter), _container(container) {};
      virtual ~ArrowSequence() {};
      void copy(const ArrowSequence& seq, ArrowSequence& cseq) {
        super::copy(seq,cseq);
        cseq._container = seq._container;
      };
      void reset(container_type *container, index_type *index, const outer_filter_type& outer_filter, const inner_filter_type& inner_filter) {
        this->super::reset(index, outer_filter, inner_filter);
        this->_container = container;
      };
      ArrowSequence& operator=(const ArrowSequence& seq) {
        copy(seq,*this); return *this;
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
    // Specialized RangeFilters
    //
    //typedef ALE::XSifterDef::RangeFilter<base_index_type, predicate_extractor_type, predicate_order_type> base_predicate_filter_type;
    //typedef ALE::XSifterDef::RangeFilter<base_index_type, target_extractor_type, target_order_type>       base_target_filter_type;
    typedef ALE::XSifterDef::RangeFilter<cone_index_type, target_extractor_type, target_order_type>       cone_target_filter_type;
    typedef ALE::XSifterDef::RangeFilter<cone_index_type, predicate_extractor_type, predicate_order_type> cone_predicate_filter_type;
    //
    // Specialized sequence types
    //
    //typedef ArrowSequence<base_index_type, base_predicate_filter_type, base_target_filter_type, target_extractor_type, true>  BaseSequence;
    typedef ArrowSequence<cone_index_type, cone_target_filter_type, cone_predicate_filter_type, target_extractor_type, true>  BaseSequence;

    typedef ArrowSequence<cone_index_type, cone_target_filter_type, cone_predicate_filter_type, source_extractor_type>        ConeSequence;
    //
    // Basic interface
    //
    XSifter(const MPI_Comm comm, int debug = 0) : // FIXIT: Should really inherit from XParallelObject
      XObject(debug), _rec_set(), 
      _base_index(::boost::multi_index::get<BaseTag>(this->_rec_set)), 
      _cone_index(::boost::multi_index::get<ConeTag>(this->_rec_set))
    {};
    //
    // Extended interface
    //
    void addArrow(const arrow_type& a, const predicate_type& p) {
      this->_rec_set.insert(rec_type(a,p));
    };
    void addArrow(const arrow_type& a) {
      this->_rec_set.insert(rec_type(a));
    };
    //
    void cone(const target_type& t, ConeSequence& seq) {
      seq.reset(this, &this->_cone_index,cone_target_filter_type(&this->_cone_index,t,t),cone_predicate_filter_type(&this->_cone_index));
    };
    ConeSequence& cone(const target_type& t) {
      static ConeSequence cseq;
      this->cone(t,cseq);
      return cseq;
    };
    //
    void base(BaseSequence& seq) {
      //seq.reset(this, &this->_base_index,base_predicate_filter_type(&this->_base_index),base_target_filter_type(&this->_base_index));
      seq.reset(this, &this->_cone_index,cone_target_filter_type(&this->_cone_index),cone_predicate_filter_type(&this->_cone_index));
    };
    BaseSequence& base() {
      static BaseSequence bseq;
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
      // os << "Base index: (";
//         for(typename base_index_type::iterator itor = this->_base_index.begin(); itor != this->_base_index.end(); ++itor) {
//           os << *itor << " ";
//         }
//       os << ")" << std::endl;
      os << "Cone index: (";
        for(typename cone_index_type::iterator itor = this->_cone_index.begin(); itor != this->_cone_index.end(); ++itor) {
          os << *itor << " ";
        }
      os << ")" << std::endl;
    };
  protected:
    // set of arrow records
    rec_set_type     _rec_set;
    base_index_type& _base_index;
    cone_index_type& _cone_index;
  }; // class XSifter


} // namespace ALE

#endif
