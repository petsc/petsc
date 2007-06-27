#ifndef included_ALE_XSifter_hh
#define included_ALE_XSifter_hh

// BEGIN: these includes come from boost/multi_index/mem_fun.hpp
#include <boost/config.hpp> /* keep it first to prevent nasty warns in MSVC */
#include <boost/mpl/if.hpp>
#include <boost/type_traits/remove_reference.hpp>
// END


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
  //
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
  
  // 
  // The following member function return a const result.
  // It is best used through the macro ALE_CONST_MEM_FUN which takes only three arguments: 
  //  Class, ReturnType, MemberFunctionPtr (see below).
  template<class Class,typename Type, typename const_Type, const_Type (Class::*PtrToMemberFunction)()const>
  struct const_const_mem_fun
  {
    typedef typename ::boost::remove_reference<Type>::type result_type;
    //typedef Type                                    result_type;
    template<typename ChainedPtr>
    const_Type operator()(const ChainedPtr& x)const
    {
      return operator()(*x);
    }
    
    const_Type operator()(const Class& x)const
    {
      return (x.*PtrToMemberFunction)();
    }
    
    const_Type operator()(const ::boost::reference_wrapper<const Class>& x)const
    { 
      return operator()(x.get());
    }
    
    const_Type operator()(const ::boost::reference_wrapper<Class>& x,int=0)const
    { 
      return operator()(x.get());
    }
  };// struct const_const_mem_fun
#define ALE_CONST_MEM_FUN(CLASS, TYPE, FUN) ::ALE::const_const_mem_fun<CLASS, TYPE, const TYPE, FUN>
  
  namespace XSifterDef {
    static int debug   = 0;
    static int codebug = 0;
    // Debugging works the top: setting ALE::XSifterDef::debug to n will 'uncover' the *last* (newest) n layers of debugging.
    // Thus, the functions with the n heighst __ALE_DEBUG__ markers will spew out debugging prints.
    // Co-debugging works from the bottom: setting ALE::XSifterDef::codebug to n will 'uncover' the *first* (oldest) n layers of debugging.
#define ALE_XDEBUG_HEIGHT 6
#define ALE_XDEBUG_LEVEL(n)  ((ALE::XSifterDef::codebug >= n) || (n > ALE_XDEBUG_HEIGHT - ALE::XSifterDef::debug))
#define ALE_XDEBUG           (ALE_XDEBUG_LEVEL(__ALE_XDEBUG__))

    //
    // Rec compares
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
      key_order_type& keyCompare(){return this->_key_order;};
      //
      bool operator()(const rec_type& rec1, const rec_type& rec2) const {
        return this->_key_order(this->_kex(rec1), this->_kex(rec2));
      };
      //
      bool operator()(const key_type& key1, const key_type& key2) const {
        return this->_key_order(key1, key2);
      };

//       template <typename CompatibleKey_>
//       bool operator()(const rec_type& rec, const ALE::singleton<CompatibleKey_> keySingleton) const {
//         // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
//         // and must be unwrapped before the ordering operator is applied.
//         return this->_key_order(this->_kex(rec), keySingleton.first);
//       };
//       template <typename CompatibleKey_>
//       bool operator()(const ALE::singleton<CompatibleKey_> keySingleton, const rec_type& rec) const {
//         // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
//         // and must be unwrapped before the ordering operator is applied
//         return this->_key_order(keySingleton.first, this->_kex(rec));
//       };
//       template <typename CompatibleKey_>
//       bool operator()(const ALE::singleton<CompatibleKey_> keySingleton1, const ALE::singleton<CompatibleKey_> keySingleton2) const {
//         // In order to disamiguate calls such as this from (rec&,rec&) calls, compatible keys are passed in wrapped as singletons, 
//         // and must be unwrapped before the ordering operator is applied
//         return this->_key_order(keySingleton1.first, keySingleton2.first);
//       };

      // Do we really need to wrap keys as Singletons?
      template <typename CompatibleKey_>
      bool operator()(const rec_type& rec, const CompatibleKey_ key) const {
        return this->_key_order(this->_kex(rec), key);
      };
      //
      template <typename CompatibleKey_>
      bool operator()(const CompatibleKey_ key, const rec_type& rec) const {
        return this->_key_order(key, this->_kex(rec));
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
      typedef RecKeyXXXOrder                              pre_compare_type;
      typedef xxx_order_type                              pos_compare_type;
      typedef key_extractor_type                          pre_extractor_type;
      typedef typename xxx_order_type::key_extractor_type pos_extractor_type;
    private:
      key_order_type     _compare_keys;
      xxx_order_type     _compare_xxx;
      key_extractor_type _kex;
    public:
      inline const pre_compare_type& preCompare() const {return *this;};
      inline const pos_compare_type& posCompare() const {return this->_compare_xxx;};
      //
      inline bool operator()(const rec_type& rec1, const rec_type& rec2) const { 
        if(this->_compare_keys(this->_kex(rec1), this->_kex(rec2))) 
          return true;
        if(this->_compare_keys(this->_kex(rec2), this->_kex(rec1))) 
          return false;
        if(this->_compare_xxx(rec1,rec2))
          return true;
        return false;           
      };
      // In order to disambiguate calls such as (key, rec) with key_type==rec_type, from (rec,rec) calls, 
      // we provide a call interface where compatible keys are passed in wrapped as singletons, 
      // and must be unwrapped before the ordering operator is applied.  (When is this actually encountered?)
      // Unwrapped keys are allowed as well (see below).
//       template <typename CompatibleKey_>
//       inline bool operator()(const ALE::singleton<CompatibleKey_>& keySingleton, const rec_type& rec1) const {
//         return this->_compare_keys(keySingleton.first, this->_kex(rec1));
//       };
//       template <typename CompatibleKey_>
//       inline bool operator()(const rec_type& rec1, const ALE::singleton<CompatibleKey_>& keySingleton) const {
//         return this->_compare_keys(this->_kex(rec1), keySingleton.first);
//       };
//       template <typename CompatibleKey_>
//       inline bool operator()(const ALE::singleton<CompatibleKey_>& keySingleton1, const ALE::singleton<CompatibleKey_>& keySingleton2) const {
//         return this->_compare_keys(keySingleton1.first, keySingleton2.first);
//       };
//       //
//       template <typename CompatibleKey_, typename CompatibleXXX_>
//       inline bool operator()(const ALE::pair<CompatibleKey_, CompatibleXXX_>& pair, const rec_type& rec) const {
//         // We want (key,xxx) to be no greater than any (key, xxx, ...)
//         return this->_compare_keys(pair.first, _kex(rec)) ||
//           (!this->_compare_keys(_kex(rec), pair.first) && this->_compare_xxx(ALE::singleton<CompatibleXXX_>(pair.second), rec));
//         // Note that CompatibleXXX_ -- the second element of the pair -- must be wrapped up as a singleton before being passed for comparison against rec
//         // to compare_xxx.  This is necessary for compare_xxx to disamiguate comparison of recs to elements of differents types.  In particular, 
//         // this is necessary if compare_xxx is of the RecKeyXXXOrder type. Specialization doesn't work, or I don't know how to make it work in this context.
//       };
//       template <typename CompatibleKey_, typename CompatibleXXX_>
//       inline bool operator()(const rec_type& rec, const ALE::pair<CompatibleKey_, CompatibleXXX_>& pair) const {
//         // We want (key,xxx) to be no greater than any (key, xxx, ...)
//         return _compare_keys(_kex(rec), pair.first) ||
//           (!this->_compare_keys(pair.first, _kex(rec)) && this->_compare_xxx(rec,ALE::singleton<CompatibleXXX_>(pair.second)));
//         // Note that CompatibleXXX_ -- the second element of the pair -- must be wrapped up as a singleton before being passed for comparison against rec
//         // to compare_xxx.  This is necessary for compare_xxx to disamiguate comparison of recs to elements of differents types.  In particular, 
//         // this is necessary if compare_xxx is of the RecKeyXXXOrder type. Specialization doesn't work, or I don't know how to make it work in this context.
//       };
      //
      // Calls with unwrapped keys are defined here.
      template <typename CompatibleKey_>
      inline bool operator()(const rec_type& rec, const CompatibleKey_ key) const {
        return this->_compare_keys(this->_kex(rec), key);
      };
      //
      template <typename CompatibleKey_>
      inline bool operator()(const CompatibleKey_ key, const rec_type& rec) const {
        return this->_compare_keys(key, this->_kex(rec));
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
      static const predicate_type zero;
      static printable_type printable(const predicate_type& p) {return (printable_type)p;};
    };
    const PredicateTraits<int>::predicate_type PredicateTraits<int>::default_value = 0;
    const PredicateTraits<int>::predicate_type PredicateTraits<int>::zero          = 0;
    //
    template<>
    struct PredicateTraits<unsigned int> {
      typedef      unsigned int  predicate_type;
      typedef      unsigned int  printable_type;
      static const predicate_type default_value;
      static const predicate_type zero;
      static printable_type printable(const predicate_type& p) {return (printable_type)p;};
    };
    const PredicateTraits<unsigned int>::predicate_type PredicateTraits<unsigned int>::default_value = 0;
    const PredicateTraits<unsigned int>::predicate_type PredicateTraits<unsigned int>::zero          = 0;
    //
    template<>
    struct PredicateTraits<short> {
      typedef      short  predicate_type;
      typedef      short  printable_type;
      static const predicate_type default_value;
      static const predicate_type zero;
      static printable_type printable(const predicate_type& p) {return (printable_type)p;};
    };
    const PredicateTraits<short>::predicate_type PredicateTraits<short>::default_value = 0;
    const PredicateTraits<short>::predicate_type PredicateTraits<short>::zero          = 0;

    //
    template<>
    struct PredicateTraits<char> {
      typedef char  predicate_type;
      typedef short printable_type;
      static const predicate_type default_value;
      static const predicate_type zero;
      static printable_type printable(const predicate_type& p) {return (printable_type)p;};
    };
    const PredicateTraits<char>::predicate_type PredicateTraits<char>::default_value = '\0';
    const PredicateTraits<char>::predicate_type PredicateTraits<char>::zero          = '\0';


    //
    // IndexFilterAdapter wraps a multi_index container index of type Index_ and makes it behave like a filter.
    // This class can (almost) be taken as the interface of the concept Index. 
    //
    #undef  __CLASS__
    #define __CLASS__ "IndexFilterAdapter"
    template <typename Index_>
    class IndexFilterAdapter {
    public:
      typedef Index_                                           index_type;
      // also need: key_type, key_traits in a general Filter
      //typedef typename index_type::iterator                  iterator;
      typedef typename index_type::const_iterator              iterator;
      typedef typename index_type::key_type                    key_type;
      typedef typename index_type::key_from_value              key_extractor_type;
      typedef typename index_type::key_compare                 key_compare_type;
      //
      struct cookie_type {
        iterator _segmentEnd;
      public:
        //
        cookie_type(){};
        explicit cookie_type(const iterator& iter) : _segmentEnd(iter) {};
        cookie_type(const cookie_type& cookie)     : _segmentEnd(cookie._segmentEnd){};
       
        const iterator&  segmentEnd() const {return this->_segmentEnd;};
        //
        template <typename Stream_>
        friend Stream_& operator<<(Stream_& os, const cookie_type& cookie) {
          return os << (*(cookie.segmentEnd()));
        };
      };// struct cookie
      //      
    protected:
      index_type* _index;
    public:
      //
      // Basic interface
      //IndexFilterAdapter(){};
      IndexFilterAdapter(index_type* index) : _index(index){};
      ~IndexFilterAdapter(){};
      //
      // Extended interface
      //
      inline static key_type  key(const iterator& iter) { static key_extractor_type kex; return kex(*iter);};
      static        bool      compare(const key_type& k1, const key_type& k2) {
        static key_compare_type comp;
        return comp(k1,k2);
      };
      inline static        bool      compare(const iterator& i1, const iterator& i2) {
        return compare(key(i1),key(i2));
      };
      //
      #undef  __FUNCT__
      #define __FUNCT__ "firstSegment"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 5
      void firstSegment(iterator& iter, cookie_type& cookie) const {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << std::endl;
        };
#endif

        iter = this->_index->begin();
        cookie._segmentEnd = this->_index->end();
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << "*iter: " << *iter << ", cookie: " << cookie << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        };
#endif
      };
      //
      #undef  __FUNCT__
      #define __FUNCT__ "nextSegment"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 5
      void nextSegment(iterator& iter, cookie_type& cookie) const {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "*iter: " << *iter << ", cookie: " << cookie << std::endl;
        };
#endif

        iter = this->_index->end();
        cookie._segmentEnd = this->_index->end();
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << "*iter: " << *iter << ", cookie: " << cookie << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        };
#endif
      };
      //
      void next(iterator& iter, cookie_type& cookie) const {
        iter = this->nextSegment(iter, cookie);
      };
      //
      void end(iterator& iter, cookie_type& cookie) const {
        iter = this->veryEnd(); cookie(iter);
      };
      //
      iterator veryEnd()       const { return this->_index->end();  };
      //
      template <typename Stream_>
      friend Stream_& operator<<(Stream_& os, const IndexFilterAdapter& f) {
        os << "[*index_begin, *index_end] = [" << *(f._index->begin()) << ", " << *(f._index->end()) << "]";
        return os;
      };
      template <typename Stream_>
      friend Stream_& operator<<(Stream_& os, const Obj<IndexFilterAdapter>& f) {
        return (os << f.object());
      };
    };// class IndexFilterAdapter

    //
    // RangeFilter defines a subset of an OuterFilter_ based on the high and low value of a key.
    // The type and ordering of the key are defined by KeyExtractor_ and KeyCompare_ respectively.
    // Nested filter types are in one-to-one correspondence with lexicographical nests of their KeyCompare_s.
    // Namely, the outermost Filter_ can compare nested pairs of keys from the outermost 'ko', to the innermost 'ki':
    // (ko,(kj,...,(ki))).
    // 
    #undef  __CLASS__
    #define __CLASS__ "RangeFilter"
    template <typename OuterFilter_, typename KeyExtractor_, typename KeyCompare_, bool Strided = false>
    class RangeFilter : public OuterFilter_ {
    public:
      typedef OuterFilter_                                     outer_filter_type;
      typedef outer_filter_type                                outer_type;
      typedef KeyExtractor_                                    key_extractor_type;
      typedef KeyCompare_                                      key_compare_type;
      typedef typename key_extractor_type::result_type         key_type; 
      typedef PredicateTraits<key_type>                        key_traits;
      typedef typename outer_type::iterator                    iterator;
      typedef typename outer_type::cookie_type                 outer_cookie_type;
      //
      struct cookie_type: public outer_cookie_type {
        // we store an iterator pointing to the end of the segment
        iterator _subsegmentEnd;
        iterator _segmentEnd;
        //
        cookie_type(){};
        cookie_type(const cookie_type& cookie) : _segmentEnd(cookie._segmentEnd) {};
        explicit cookie_type(const iterator& segmentEnd) : outer_cookie_type(segmentEnd), _segmentEnd(segmentEnd) {};
        //
        inline const iterator&  outerEnd()   const {return this->outer_cookie_type::segmentEnd();};
        inline const iterator&  segmentEnd() const {return this->_segmentEnd;};
        inline const iterator&  subsegmentEnd() const {return this->_subsegmentEnd;};
        //
        template <typename Stream_>
        friend Stream_& operator<<(Stream_& os, const cookie_type& cookie) {
          return os /* << ((outer_cookie_type) cookie) << "; " */ << *cookie.segmentEnd();
        };
      };// struct cookie_type
      //
      template <typename SubkeyExtractor_>
      struct superkey_extractor_type {
        typedef SubkeyExtractor_                           subkey_extractor_type;
        typedef typename subkey_extractor_type::value_type subkey_type;
        typedef ALE::pair<key_type, subkey_type>           super_key_type;
        //
        superkey_extractor_type(){};
        //
        inline const super_key_type& operator()(const iterator& iter) { 
          return _super_key(_ex(iter), _subex(iter));
        };
        inline const super_key_type& operator()(const iterator& iter, const subkey_type& subkey) {
          return _super_key(_ex(iter), subkey);
        };
      protected:
        super_key_type        _super_key;  
        subkey_extractor_type _subex;
        key_extractor_type    _ex;
      };// struct superkey_extractor_type
      //      
    protected:
      bool                                  _have_low, _have_high;
      key_type                              _low, _high;
    public:
      //
      // Basic interface
      //
      //RangeFilter(){};
      RangeFilter(const outer_filter_type& outer_filter) : 
       outer_filter_type(outer_filter), _have_low(false), _have_high(false) {};
      RangeFilter(const outer_filter_type& outer_filter, const key_type& low, const key_type& high) : 
       outer_filter_type(outer_filter), _have_low(true), _have_high(true), _low(low), _high(high)  {};
      RangeFilter(const RangeFilter& f) : 
        outer_filter_type(f), _have_low(f._have_low), _have_high(f._have_high), _low(f._low), _high(f._high) {};
      virtual ~RangeFilter(){};
      //
      // Extended interface
      //
      inline void setLow(const key_type& low)   {this->_low  = low;  this->_have_low  = true;};
      inline void setHigh(const key_type& high) {this->_high = high; this->_have_high = true;};
      inline void setLowAndHigh(const key_type& low, const key_type& high) {this->setLow(low); this->setHigh(high);};
      //
      inline static key_type  key(const iterator& iter) { static key_extractor_type kex; return kex(*iter);};
      static        bool      compare(const key_type& k1, const key_type& k2) {
        static key_compare_type comp;
        return comp(k1,k2);
      };
      inline static        bool      compare(const iterator& i1, const iterator& i2) {
        return compare(key(i1),key(i2));
      };
      //
      inline key_type         low()       const {return this->_low;};
      inline key_type         high()      const {return this->_high;};
      inline bool             haveLow()   const {return this->_have_low;};
      inline bool             haveHigh()  const {return this->_have_high;};
      //
    public:
      //
      // We can search on a SubKey subkey within the segment bound by the low/high iterators.
      // The subkey will be combined with any key_type key from the [low, high] range using a pair combinator: 
      // comb(key, subkey) producing a compatible key.
      template <typename SubKeyExtractor_>
      inline iterator lower_bound(const typename SubKeyExtractor_::value_type& subkey, const iterator& low, const iterator& high) const {
        typedef superkey_extractor_type<SubKeyExtractor_> supex_type;
        typedef SubKeyExtractor_                          subex_type;
        static   supex_type      supex;
        static   subex_type      subex;
        typename supex_type::key_type suplow = supex(low, subkey), suphigh = supex(high, subkey);
        iterator iter = this->_index->ale_lower_bound(subex, subkey, supex, suplow, suphigh);
        return (iter==this->veryEnd())?low:iter;
      };
      //
      #undef  __FUNCT__
      #define __FUNCT__ "firstSegment"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 5
      // Returns the first segment allowed by this filter.
      void firstSegment(iterator& iter, cookie_type& cookie) const {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << std::endl;
        };
#endif
        cookie._segmentEnd = iter;
        while((iter == cookie._segmentEnd) && (iter != this->veryEnd())) {
          if(iter == cookie.outerEnd()) {
            this->outer_type::nextSegment(iter, cookie);
            if(iter == this->veryEnd()) break;
          }
          // from here until the end of the while loop the outer segment can be assumed to be non-empty
          if(this->haveLow()) {
            iter = this->lower_bound(this->low(),iter,cookie.outerEnd());
          }
          if(this->haveHigh()) {
            cookie._segmentEnd = this->upper_bound(this->high(),iter,cookie.outerEnd());
          }
        }//while
        if(Strided) {
          // now find the boundary of a subsegment
          if(iter != this->veryEnd()) {
            cookie._subsegmentEnd = this->upper_bound(this->high(),iter,cookie.segmentEnd());
          }
        }
        //
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter: " << *iter;
          std::cout << ", cookie: " << cookie << std::endl; 
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        };
#endif
      };//firstSegment()
      //
    public:
      //
      #undef  __FUNCT__
      #define __FUNCT__ "first"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 5
      // Returns the first segment allowed by this filter.
      void first(iterator& iter, cookie_type& cookie) const {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << std::endl;
        };
#endif
        this->outer_type::first(iter, cookie);
        this->firstSegment(iter, cookie);
        //
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter: " << *iter;
          std::cout << ", cookie: " << cookie << std::endl; 
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        };
#endif
      };//first()
      //
      #undef  __FUNCT__
      #define __FUNCT__ "nextSegment"
      #undef  __ALE_XDEBUG__
      #define __ALE_XDEBUG__ 5
      void nextSegment(iterator& iter, cookie_type& cookie) {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "*iter: " << *iter << ", cookie: " << cookie << std::endl;
        };
#endif
        if(Strided){
          iter = cookie._subsegmentEnd;
          if(iter == cookie._segmentEnd){
            this->outer_type::nextSegment(iter, cookie);
            this->firstSegment(iter,cookie);
          }
          else if(iter != this->veryEnd()) {
            cookie._subsegmentEnd = this->_segment_upper_bound(this->key(iter), iter, cookie);
            // assuming a nonempty segment, there cannot be an overshoot
          }
        }// Strided
        else {// !Strided
            this->outer_type::nextSegment(iter,cookie);
            this->firstSegment(iter,cookie);
        }// !Strided

#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter: " << *iter;
          std::cout << ", cookie: " << cookie << std::endl; 
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        };
#endif
      };// nextSegment()
      //
      #undef  __FUNCT__
      #define __FUNCT__ "next"
      #undef  __ALE_XDEBUG__
      #define __ALE_XDEBUG__ 5
      void next(iterator& iter, cookie_type& cookie) {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "*iter: " << *iter << ", cookie: " << cookie << std::endl;
        };
#endif
        if(!Strided) {
        // If filter is not strided, we iterate through each subsegment until the end of the segment is reached
#ifdef ALE_USE_DEBUGGING
          if(ALE_XDEBUG) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "non-strided filter" << std::endl;
          }
#endif
          if(iter != cookie.segmentEnd()) {
            ++iter;
          }
          else {
            this->nextSegment(iter,cookie);
          }
        }// !Strided
        else {
          // If the filter is strided, we skip the remainder of the current segment; effectively, we iterate over subsegments.
#ifdef ALE_USE_DEBUGGING
          if(ALE_XDEBUG) {
            std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "strided filter" << std::endl;
          }
#endif
          // RangeFilter has multiple inner segments outer segment: one inner segment per key.  
          // A Strided RangeFilter's next iterates over inner segments.
          this->nextSegment(iter,cookie); // really means 'nextSubsegment'
        }// Strided
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter: " << *iter;
          std::cout << ", cookie: " << cookie << std::endl; 
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        };
#endif
      };// next()
      //
      #undef  __FUNCT__
      #define __FUNCT__ "end"
      #undef  __ALE_XDEBUG__
      #define __ALE_XDEBUG__ 5
      void end(iterator& iter, cookie_type& cookie) {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << *this << std::endl;
        };
#endif
        iter = this->veryEnd(); 
        cookie._segmentEnd = iter;
        if(Strided) {
          cookie._subsegmentEnd = iter;
        }
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*iter: " << *iter;
          std::cout << ", cookie: " << cookie << std::endl; 
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        };
#endif
      };// end()
      //
      template <typename Stream_>
      friend Stream_& operator<<(Stream_& os, const RangeFilter& f) {
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
        os << "] ";
        if(Strided) {
          os << "strided";
        }
        else {
          os << "non-strided";
        }
        return os;
      };
      template <typename Stream_>
      friend Stream_& operator<<(Stream_& os, const Obj<RangeFilter>& f) {
        return (os << f.object());
      };
    };// RangeFilter




    //
    // FilteredSequence definition
    // 
    //   Defines a sequence representing a subset of a sequence ordered lexicographically.
    // The ordering is controlled by a nested sequence of filters with the topmost of  Filter_ type.
    //   A sequence defines output iterators (input iterators in std terminology) for traversing a subsequence.
    // Upon dereferencing values are extracted from each result record using a ValueExtractor_ object.
    //   More precisely, the sequence can be viewed as oredered by (Key_, RemainderKey_) pairs, where the RemainderKey_
    // type is not explicitly known to FilteredSequence.  The Filters only restrict the allowable values of Key_, 
    // while all RemainderKey_ values are allowed. 
    // 
    #undef  __CLASS__
    #define __CLASS__ "FilteredSequence"
    template <typename Filter_, typename ValueExtractor_>
    struct FilteredSequence : XObject {
      typedef Filter_                                          filter_type;
      typedef typename filter_type::cookie_type                cookie_type;
      //
      typedef ValueExtractor_                                  value_extractor_type;
      typedef typename value_extractor_type::result_type       value_type;
      typedef typename filter_type::iterator                   itor_type;
      //
      class iterator {
      public:
        // Parent sequence type
        typedef FilteredSequence                               sequence_type;
        // Standard iterator typedefs
        typedef std::input_iterator_tag                        iterator_category;
        typedef int                                            difference_type;
        typedef value_type*                                    pointer;
        typedef value_type&                                    reference;
        /* value_type defined in the containing FilteredIndexSequence */
      protected:
        // Parent sequence
        sequence_type  *_sequence;
        // Underlying iterator & segment filter cookies
        itor_type       _itor;
        cookie_type     _cookie;
        //
        // Value extractors
        value_extractor_type     _ex;
      public:
        iterator() : _sequence(NULL) {};
        iterator(sequence_type *sequence, const itor_type& itor, const cookie_type& cookie) : _sequence(sequence), _itor(itor), _cookie(cookie) {};
        iterator(const iterator& iter):_sequence(iter._sequence), _itor(iter._itor), _cookie(iter._cookie) {};
        virtual ~iterator() {};
        virtual bool              operator==(const iterator& iter) const {return this->_itor == iter._itor;}; // comparison ignores cookies; only increment uses cookies
        virtual bool              operator!=(const iterator& iter) const {return this->_itor != iter._itor;}; // comparison ignores cookies; only increment uses cookies
        //
        // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
        virtual const value_type  operator*() const {return _ex(*(this->_itor));};
        //
        virtual iterator   operator++() { // comparison ignores cookies; only increment uses cookies
          this->_sequence->next(this->_itor, this->_cookie);
          return *this;
        };
        virtual iterator   operator++(int n) {iterator tmp(*this); ++(*this); return tmp;};
      };// class iterator
    protected:
      //
      Obj<filter_type>       _filter;
    public:
      //
      // Basic interface
      //
      FilteredSequence() : XObject() {};
      FilteredSequence(const Obj<filter_type>& filter) : XObject(), _filter(filter){};
      FilteredSequence(const FilteredSequence& seq) : XObject(seq), _filter(seq._filter) {};
      virtual ~FilteredSequence() {};
      // 
      void copy(const FilteredSequence& seq, FilteredSequence cseq) {
        cseq._filter = seq._filter;
      };
      FilteredSequence& operator=(const FilteredSequence& seq) {
        copy(seq,*this); return *this;
      };
      //
      // Extended interface
      //
      virtual bool         
      empty() {return (this->begin() == this->end());};
      //
      Obj<filter_type> filter(){return ((this->_filter));};
      void reset(const Obj<filter_type>& filter) {this->_filter = filter;};
      //
      //
      #undef  __FUNCT__
      #define __FUNCT__ "begin"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 6
      iterator begin() {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << this->filter() << std::endl;
        }
#endif
        static itor_type itor;
        static cookie_type cookie;
        this->filter()->firstSegment(itor, cookie);
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*itor: " << *itor;
          std::cout << ", cookie: " << cookie; 
          std::cout << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        }
#endif
        return iterator(this, itor, cookie);
      }; // begin()
      //
      //
      #undef  __FUNCT__
      #define __FUNCT__ "next"
      #undef  __ALE_XDEBUG__
      #define __ALE_XDEBUG__ 6
      void next(itor_type& itor, cookie_type& cookie) {
        itor_type end = cookie.segmentEnd();
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>> " << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << this->filter() << std::endl;
          //
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "starting with *itor " << *itor; 
          std::cout << ", cookie: " << cookie; 
          std::cout << std::endl;
        }
#endif
        this->filter()->next(itor, cookie);
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "new *itor " << *itor; 
          std::cout << ", cookie: " << cookie; 
          std::cout << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<< " << std::endl;
        }
#endif
      };// next()
      //
      //
      #undef  __FUNCT__
      #define __FUNCT__ "end"
      #undef  __ALE_XDEBUG__
      #define __ALE_XDEBUG__ 6
      iterator end() {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>>" << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "filter: " << this->filter() << std::endl;
        }
#endif
        static itor_type itor;
        static cookie_type cookie;
        this->filter()->end(itor, cookie);
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          //
          //std::cout << __CLASS__ << "::" << __FUNCT__ << ": " << "*itor: " << *itor; 
          //std::cout << ", cookie: " << cookie;
          //std::cout << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<<" << std::endl;
        }
#endif
        return iterator(this, itor, cookie); 
      };// end()
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
    };// class FilteredSequence    


    // Definitions of typical XSifter usage of records, orderings, etc.
    template<typename Arrow_, 
             typename SourceOrder_ = std::less<typename Arrow_::source_type>,
             typename ColorOrder_  = std::less<typename Arrow_::color_type> >
    struct SourceColorOrder : 
      public RecKeyXXXOrder<Arrow_, 
                            ALE_CONST_MEM_FUN(Arrow_, typename Arrow_::source_type&, &Arrow_::source), 
                            SourceOrder_, 
                            RecKeyOrder<Arrow_, 
                                        ALE_CONST_MEM_FUN(Arrow_, typename Arrow_::color_type&, &Arrow_::color), 
                                        ColorOrder_>
      >
    {};
    
    // 
    // Arrow definition: a concrete arrow; other Arrow definitions are possible, since orders above are templated on it
    // To be an Arrow a struct must contain the relevant member functions: source, target, tail and head.
    // This one also has color :-)
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
      const source_type& source() const {return this->_source;};
      const target_type& target() const {return this->_target;};
      const color_type&  color()  const {return this->_color;};
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

    // Arrow + Predicate = ArrowRec (used in the multi-index container underlying the XSifter)
    template <typename Arrow_, typename Predicate_>
    struct ArrowRec : public Arrow_ {
    public:
      //
      // Re-export typedefs
      //
      typedef Arrow_                                  arrow_type;
      typedef typename arrow_type::source_type        source_type;
      typedef typename arrow_type::target_type        target_type;
      //
      typedef Predicate_                              predicate_type;
      typedef PredicateTraits<predicate_type>         predicate_traits;
    protected:
      // Predicate stored alongside the arrow data
      predicate_type _predicate;
      // Slice pointer
      // HACK: ArrowRec should really reside at a lower level
      void * _slice_ptr;
      predicate_type _slice_marker;
    public:
      // Basic interface
      ArrowRec(const arrow_type& a) : arrow_type(a), _predicate(predicate_traits::zero), _slice_ptr(NULL), _slice_marker(predicate_traits::zero) {};
      ArrowRec(const arrow_type& a, const predicate_type& p) : arrow_type(a), _predicate(p), _slice_ptr(NULL), _slice_marker(predicate_traits::zero){};
      // Extended interface
      const predicate_type&    predicate() const {return this->_predicate;};
      const source_type&       source()    const {return this->arrow_type::source();};
      const target_type&       target()    const {return this->arrow_type::target();};
      const arrow_type&        arrow()     const {return *this;};
      //
      // HACK: this should be implemented at the lower level of a multi_index container
      const predicate_type slice_marker() const {return this->_slice_marker;};
      const void*          slice_ptr()    const {return this->_slice_ptr;};
      // Printing
      friend std::ostream& operator<<(std::ostream& os, const ArrowRec& r) {
        os << "<" << predicate_traits::printable(r._predicate) << ">" << "[" << (arrow_type)r << "]";
        return os;
      }
      // Modifier objects
      struct PredicateChanger {
        PredicateChanger(const predicate_type& newPredicate) : _newPredicate(newPredicate) {};
        void operator()(ArrowRec& r) { r._predicate = this->_newPredicate;}
      private:
        const predicate_type _newPredicate;
      };
      //
      struct SliceChanger {
        SliceChanger(const predicate_type& new_slice_marker, const void *& new_slice_ptr) : _new_slice_marker(new_slice_marker), _new_slice_ptr(new_slice_ptr) {};
        void setSlice(const predicate_type& new_slice_marker, const void *& new_slice_ptr){
          this->_new_slice_marker = new_slice_marker;
          this->_new_slice_pointer = new_slice_ptr;
        }
        void operator()(ArrowRec& r) {r._slice_marker = this->_new_slice_marker; r._slice_ptr = this->_new_slice_ptr;};
      private:
        void *_new_slice_ptr;
        predicate_type _new_slice_marker;
      };// struct SliceChanger
    };// struct ArrowRec

    //
    // Arrow Sequence type
    //
    template <typename XSifter_, typename Filter_, typename ValueExtractor_>
    class ArrowSequence : 
      public XSifterDef::FilteredSequence<Filter_, ValueExtractor_> {
      // ArrowSequence extends FilteredSequence with extra iterator methods.
    public:
      typedef XSifter_                                                                            xsifter_type;
      typedef XSifterDef::FilteredSequence<Filter_, ValueExtractor_>                              super;
      typedef Filter_                                                                             filter_type;
      //
      typedef typename xsifter_type::rec_type                                                     rec_type;
      typedef typename xsifter_type::arrow_type                                                   arrow_type;
      typedef typename arrow_type::source_type                                                    source_type;
      typedef typename arrow_type::target_type                                                    target_type;
      //
      // Need to extend the inherited iterators to be able to extract arrow components in addition to what comes out of ValueExtractor_
      class iterator : public super::iterator {
      public:
        iterator() : super::iterator() {};
        iterator(const typename super::iterator& super_iter) : super::iterator(super_iter) {};
        virtual const source_type& source() const {return this->_itor->source();};
        virtual const target_type& target() const {return this->_itor->target();};
        virtual const arrow_type&  arrow()  const {return *(this->_itor);};
        virtual const rec_type&    rec()    const {return *(this->_itor);};
      };
    protected:
      xsifter_type *_xsifter;
    public:
      //
      // Basic ArrowSequence interface
      //
      ArrowSequence() : super(), _xsifter(NULL) {};
      ArrowSequence(const ArrowSequence& seq) : super(seq), _xsifter(seq._xsifter) {};
      ArrowSequence(xsifter_type *xsifter, const filter_type& filter) : super(filter), _xsifter(xsifter) {};
      virtual ~ArrowSequence() {};
      void copy(const ArrowSequence& seq, ArrowSequence& cseq) {
        super::copy(seq,cseq);
        cseq._xsifter = seq._xsifter;
      };
      void reset(xsifter_type* xsifter, const filter_type& filter) {
        this->super::reset(filter);
        this->_xsifter = xsifter;
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
      void view(ostream_type& os, const char* label = NULL){
        if(label != NULL) {
          os << "Viewing " << label << " sequence:" << std::endl;
        } 
        os << "[";
        for(iterator i = this->begin(); i != this->end(); i++) {
          os << " (" << *i << ")";
        }
        os << " ]" << std::endl;
      };
      void addArrow(const arrow_type& a) {
        this->_xsifter->addArrow(a);
      };
      //
    };// class ArrowSequence  
  }; // namespace XSifterDef
  
  //
  // XSifter definition
  //
  template<typename Arrow_, 
           typename TailOrder_   = XSifterDef::SourceColorOrder<Arrow_>, 
           typename Predicate_ = unsigned int>
  struct XSifter : XObject { // struct XSifter
    //
    typedef XSifter xsifter_type;
    //
    // Encapsulated types: re-export types and/or bind parameterized types
    //
    //
    typedef Arrow_                                                 arrow_type;
    typedef typename arrow_type::source_type                       source_type;
    typedef typename arrow_type::target_type                       target_type;
    //
    typedef Predicate_                                             predicate_type;
    typedef ALE::XSifterDef::PredicateTraits<predicate_type>       predicate_traits;
    //
    typedef ALE::XSifterDef::ArrowRec<arrow_type, predicate_type>  rec_type;
    // 
    // Key extractors
    //
    typedef ::ALE::const_const_mem_fun<rec_type, source_type&, const source_type&, &rec_type::source>    source_extractor_type;
    typedef ALE_CONST_MEM_FUN(rec_type, target_type&,    &rec_type::target)    target_extractor_type;
    typedef ALE_CONST_MEM_FUN(rec_type, predicate_type&, &rec_type::predicate) predicate_extractor_type;
    typedef ALE_CONST_MEM_FUN(rec_type, arrow_type&,     &rec_type::arrow)     arrow_extractor_type;

    //
    // Comparison operators
    //
    typedef std::less<typename rec_type::target_type> target_order_type;
    typedef TailOrder_                                tail_order_type;
    //
    // Rec 'Cone' order type: first order by target then using a custom arrow_cone order
    //struct cone_order_type : public 
    typedef 
    XSifterDef::RecKeyXXXOrder<rec_type, 
                               ALE_CONST_MEM_FUN(rec_type, target_type&, &rec_type::target), target_order_type,
                               XSifterDef::RecKeyOrder<rec_type,
                                                       ALE_CONST_MEM_FUN(rec_type, arrow_type&, &rec_type::arrow), tail_order_type> >
    cone_order_type;
    //
    // Index tags
    //
    struct                                   ConeTag{};
    //
    // Rec set type
    //
    typedef ::boost::multi_index::multi_index_container< 
      rec_type,
      ::boost::multi_index::indexed_by< 
        ::boost::multi_index::ordered_non_unique<
          ::boost::multi_index::tag<ConeTag>, ::boost::multi_index::identity<rec_type>, cone_order_type
        > 
      >,
      ALE_ALLOCATOR<rec_type> > 
    rec_set_type;
    //
    // Index types extracted from the Rec set
    //
    typedef typename ::boost::multi_index::index<rec_set_type, ConeTag>::type cone_index_type;
    //
    // Filter types
    //
    typedef ALE::XSifterDef::IndexFilterAdapter<cone_index_type>                                               cone_index_filter_type;
    typedef ALE::XSifterDef::RangeFilter<cone_index_filter_type, target_extractor_type, target_order_type>      cone_filter_type;
    //
    // Sequence types
    //
    typedef ALE::XSifterDef::ArrowSequence<xsifter_type, cone_filter_type,  target_extractor_type>      BaseSequence;
    typedef ALE::XSifterDef::ArrowSequence<xsifter_type, cone_filter_type,  source_extractor_type>      ConeSequence;
    //
    //
    // Basic interface
    //
    //
    XSifter(const MPI_Comm comm, int debug = 0) : // FIXIT: Should really inherit from XParallelObject
      XObject(debug), _rec_set(), _cone_index(::boost::multi_index::get<ConeTag>(this->_rec_set)),
      _cone_index_filter(&_cone_index)
    {
      //this->_cone_index_filter = cone_index_filter_type(this->_cone_index);
    };
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
      //OPTIMIZE: filter object creation
      Obj<cone_filter_type> cone_filter(cone_filter_type(this->_cone_index_filter,t,t));
      seq.reset(this, cone_filter);
    };
    ConeSequence& cone(const target_type& t) {
      static ConeSequence cseq;
      this->cone(t,cseq);
      return cseq;
    };
    //
    void base(BaseSequence& seq) {
      static Obj<cone_filter_type> base_filter(cone_filter_type(this->_cone_index_filter));
      seq.reset(this, base_filter);
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
      os << "Cone index: (";
        for(typename cone_index_type::iterator itor = this->_cone_index.begin(); itor != this->_cone_index.end(); ++itor) {
          os << *itor << " ";
        }
      os << ")" << std::endl;
    };
    //
    // Direct access (a kind of hack)
    //
    // Whole container begin/end
    typedef typename cone_index_type::iterator iterator;
    iterator begin() const {return this->_cone_index.begin();};
    iterator end() const {return this->_cone_index.end();};
    //
    // Slice class
    //
    struct Slice {
    //
    public:
      typedef XSifter xsifter_type;
      //
      // Encapsulated types: re-export types and/or bind parameterized types
      //
      typedef typename xsifter_type::rec_type   rec_type;
      // 
      class iterator {
      protected:
        typename xsifter_type::iterator _itor;
      protected:
        iterator() {};
        iterator(const xsifter_type::iterator itor) : _itor(itor) {};
      public:
        virtual iterator   operator=(const iterator& iter)  const {this->_itor = iter._itor; return *this;};
        virtual bool       operator==(const iterator& iter) const {return this->_itor == iter._itor;};
        virtual bool       operator!=(const iterator& iter) const {return this->_itor != iter._itor;};
        virtual iterator   operator++() {
          this->_itor = (xsifter_type::iterator)(this->_itor->rec().slice_ptr());
          return *this;
        };
        virtual iterator   operator++(int n) {iterator tmp(*this); ++(*this); return tmp;};
        //
        virtual const source_type& source() const {return this->_itor->_source;};
        virtual const target_type& target() const {return this->_itor->_target;};
        virtual const arrow_type&  arrow()  const {return *(this->_itor);};
        //
      };
    protected:
      xsifter_type   _xsifter;
      predicate_type _slice_marker;
      void* _slice_head;
      void* _slice_tail;
      bool _clean_on_destruction;
    public:
      // 
      // Basic interface
      //
      Slice(xsifter_type *xsifter, const predicate_type& marker = predicate_traits::zero(), const bool& clean = false) : 
        _slice_marker(marker), _slice_head(NULL), _slice_tail(NULL), _clean_on_destruction(clean) {};
      ~Slice() {
        if(this->_clean_on_destruction) {
          iterator tmp;
          iterator sitor = this->begin(); 
          iterator send  = this->end();
          for(;sitor != send();) {
            tmp = ++sitor;
            // CONTINUE: modify (*itor)'s _slice_marker and _next_in_slice pointer.
            sitor = tmp;
          }
        }// if(this->_clean_on_destruction)
      };// ~Slice()
      iterator begin(){ return iterator((typename xsifter_type::iterator)this->_slice_head);}; 
      iterator end(){ return iterator((typename xsifter_type::iterator)((xsifter_type::iterator)(this->_slice_tail)->rec().slice_ptr()));}; 
      void add(const xsifter_type::iterator& itor) {
        // add *itor at the tail of the slice linked list
        static typename xsifter_type::rec_type::SliceChanger  tailSlicer(this->_slice_marker, NULL), nextSlicer;
        nextSlicer.setSlice(this->_slice_marker, (void*)itor);
        // set the current slice tail to point to itor 
        //   CAUTION: hoping for no cycles; can check itor's slice_marker, but will hurt performance
        //   However, the caller should really be checking that itor is not in the slice yet; otherwise the 'set' semantics are violated.
        this->_xsifter.update((typename xsifter_type::iterator)(this->_slice_tail), nextSlicer);
        // mark itor with the slice_marker and set it -- the new tail -- to point to NULL
        this->_xsifter.update(itor, tailSlicer);
      };
      //
    };// Slice
    
  public:
  // set of arrow records
    rec_set_type     _rec_set;
    cone_index_type& _cone_index;
    cone_index_filter_type _cone_index_filter;
  public:
    //
  }; // class XSifter

  
} // namespace ALE

#endif
