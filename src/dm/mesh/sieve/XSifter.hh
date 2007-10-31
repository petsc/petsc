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
  // Key extractors
  //
  // 
  // The following member function return a const result.
  // It is best used through the macro ALE_CONST_MEM_FUN which takes only three arguments: 
  //  Class, ResultType, MemberFunctionPtr (see below).
  // OutputType (the actual return type) is different from the ResultType for somewhat obscure reasons.
  // Once I (have time to) understand the issue better, the usage pattern may get simplified.
  template<class InputType_, typename ResultType_, typename OutputType_, OutputType_ (InputType_::*PtrToMemberFunction)()const>
  struct const_const_mem_fun
  {
    typedef InputType_                                            input_type;
    typedef typename ::boost::remove_reference<ResultType_>::type result_type;
    typedef OutputType_                                           output_type;
    //
    // Main interface
    //
    template<typename ChainedPtrTarget>
    output_type operator()(const ChainedPtrTarget*& x)const
    {
      return operator()((*x));
    }
    
    output_type operator()(const input_type& x)const
    {
      return (x.*PtrToMemberFunction)();
    }
    
    output_type operator()(const ::boost::reference_wrapper<const input_type>& x)const
    { 
      return operator()(x.get());
    }
    
    output_type operator()(const ::boost::reference_wrapper<input_type>& x,int=0)const
    { 
      return operator()(x.get());
    }
  };// struct const_const_mem_fun
#define ALE_CONST_MEM_FUN(CLASS, RESULT_TYPE, FUN) ::ALE::const_const_mem_fun<CLASS, RESULT_TYPE, const RESULT_TYPE, FUN>

  
  namespace XSifterDef {
    static int debug   = 0;
    static int codebug = 0;
    // Debugging works from the top: setting ALE::XSifterDef::debug to n will 'uncover' the *last* (newest) n layers of debugging.
    // Thus, the functions with the n heighest __ALE_DEBUG__ markers will produce debugging output.
    // Co-debugging works from the bottom: setting ALE::XSifterDef::codebug to n will 'uncover' the *first* (oldest) n layers of debugging.
    // Thus, the functions with the n lowest __ALE_DEBUG__ markers will produce debugging output.
#define ALE_XDEBUG_HEIGHT 6
#define ALE_XDEBUG_LEVEL(n)  ((ALE::XSifterDef::codebug >= n) || (n > ALE_XDEBUG_HEIGHT - ALE::XSifterDef::debug))
#define ALE_XDEBUG           (ALE_XDEBUG_LEVEL(__ALE_XDEBUG__))

    //
    // Rec compares
    //
    // RecKeyOrder compares records by comparing keys of type Key_ extracted from arrows using a KeyExtractor_.
    // In addition, a record can be compared to a single Key_ or another CompatibleKey_.
    #undef  __CLASS__
    #define __CLASS__ "RecKeyOrder"
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
//       bool operator()(const key_type& key1, const key_type& key2) const {
//         return this->_key_order(key1, key2);
//       };
      //
      #undef  __FUNCT__
      #define __FUNCT__ "operator()"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 4
      template <typename CompatibleKey_>
      bool operator()(const rec_type& rec, const CompatibleKey_ key) const {
        static bool res;
        res = this->_key_order(this->_kex(rec), key);
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(res) {
            std::cout << rec << " < " << key << "\n";
          }
          else {
            std::cout << rec << " >= " << key << "\n";
          }
        };
#endif
        return res;
      };
      //
      #undef  __FUNCT__
      #define __FUNCT__ "operator()"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 4
      template <typename CompatibleKey_>
      bool operator()(const CompatibleKey_ key, const rec_type rec) const {
        static bool res;
        res = this->_key_order(key,this->_kex(rec));
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(res) {
            std::cout << key << " < " << rec << "\n";
          }
          else {
            std::cout << key << " >= " << rec << "\n";
          }
        };
#endif
        return res;
      };
      //
      #undef  __FUNCT__
      #define __FUNCT__ "operator()"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 4
      template <typename CompatibleKey_>
      bool operator()(const CompatibleKey_ key, const rec_type& rec) const {
        static bool res;
        res = this->_key_order(key, this->_kex(rec));
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(res) {
            std::cout << rec << " < " << key << "\n";
          }
          else {
            std::cout << rec << " >= " << key << "\n";
          }
        };
#endif
        return res;
      };
    };// RecKeyOrder

    //
    // Composite Rec ordering operators (e.g., used to generate cone and support indices for Arrows).
    // An RecKeyXXXOrder first orders on a single key using KeyOrder (e.g., Target or Source for cone and support respectively),
    // and then on the whole Rec, using an additional predicate XXXOrder.
    // These are then specialized (with Rec = Arrow) to SupportCompare & ConeCompare, using the order operators supplied by the user:
    // SupportOrder = (SourceOrder, SupportXXXOrder), 
    // ConeOrder    = (TargetOrder, ConeXXXOrder), etc
    #undef  __CLASS__
    #define __CLASS__ "RecKeyXXXOrder"
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
      #undef  __FUNCT__
      #define __FUNCT__ "operator()"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 4
      inline bool operator()(const rec_type& rec1, const rec_type& rec2) const { 
        static bool res;
        if(this->_compare_keys(this->_kex(rec1), this->_kex(rec2))) {
          res = true;
        }
        else if(this->_compare_keys(this->_kex(rec2), this->_kex(rec1))) {
          res = false;
        }
        else if(this->_compare_xxx(rec1,rec2)){
          res = true;
        }
        else {
          res = false;           
        }
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(res) {
            std::cout << rec1 << " < " << rec2 << "\n";
          }
          else {
            std::cout << rec2 << " >= " << rec1 << "\n";
          }
        };
#endif        
        return res;
      };
      //
      // Comparisons with individual keys; XXX-part is ignored
      #undef  __FUNCT__
      #define __FUNCT__ "operator()"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 4
      template <typename CompatibleKey_>
      inline bool operator()(const rec_type& rec, const CompatibleKey_ key) const {
        static bool res;
        res = this->_compare_keys(this->_kex(rec), key);
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(res) {
            std::cout << rec << " < " << key << "\n";
          }
          else {
            std::cout << rec << " >= " << key << "\n";
          }
        };
#endif    
        return res;
      };
      //
      #undef  __FUNCT__
      #define __FUNCT__ "operator()"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 4
      template <typename CompatibleKey_>
      inline bool operator()(const CompatibleKey_ key, const rec_type& rec) const {
        static bool res;
        res = this->_compare_keys(key, this->_kex(rec));
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(res) {
            std::cout << key << " < " << rec << "\n";
          }
          else {
            std::cout << key << " >= " << rec << "\n";
          }
        };
#endif    
        return res;
      };
      //
      // Comparisons with key pairs; XXX-part is compared against the second key of the pair
      #undef  __FUNCT__
      #define __FUNCT__ "operator()"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 4
      template <typename CompatibleKey_, typename CompatibleSubKey_>
      inline bool operator()(const rec_type& rec, const ::ALE::pair<CompatibleKey_, CompatibleSubKey_>& keypair) const {
        static bool res;
        if(this->_compare_keys(this->_kex(rec), keypair.first)) {
          res = true;
        }
        else if(this->_compare_keys(keypair.first, this->_kex(rec))) {
          res = false;
        }
        else if(this->_compare_xxx(rec,keypair.second)){
          res = true;
        }
        else {
          res = false;           
        }
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(res) {
            std::cout << rec << " < " << keypair << "\n";
          }
          else {
            std::cout << keypair << " >= " << rec << "\n";
          }
        };
#endif        
        return res;
      };
      //
      #undef  __FUNCT__
      #define __FUNCT__ "operator()"
      #undef  __ALE_XDEBUG__ 
      #define __ALE_XDEBUG__ 4
      template <typename CompatibleKey_, typename CompatibleSubKey_>
      inline bool operator()(const ::ALE::pair<CompatibleKey_, CompatibleSubKey_>& keypair, const rec_type& rec) const {
        static bool res;
        if(this->_compare_keys(keypair.first, this->_kex(rec))) {
          res = true;
        }
        else if(this->_compare_keys(this->_kex(rec), keypair.first)) {
          res = false;
        }
        else if(this->_compare_xxx(keypair.second, rec)){
          res = true;
        }
        else {
          res = false;           
        }
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(res) {
            std::cout << keypair << " >= " << rec << "\n";
          }
          else {
            std::cout << rec << " < " << keypair << "\n";
          }
        };
#endif        
        return res;
      };
      //
    };// class RecKeyXXXOrder


    // Bounder.
    // Here we define auxiliary classes that can be derived from comparion operators.
    // These convert a compare into a bounder (lower or upper).
    //
    template <typename Compare_, typename Bound_>
    class LowerBounder {
    public:
      typedef Compare_    compare_type;
      typedef Bound_      bound_type;
    protected:
      compare_type _compare;
      bound_type   _bound;
    public:
      // Basic
      LowerBounder(){};
      LowerBounder(const bound_type& bound) : _bound(bound) {};
      inline void reset(const bound_type& bound) {this->_bound = bound;};
      // Extended
      inline const compare_type& compare(){return this->_compare;};
      inline const bound_type&   bound(){return this->_bound;};
      // Main
      template <typename Key_>
      inline bool operator()(const Key_& key){return !this->compare()(key, this->bound());};
    };// class LowerBounder
    //
    template <typename Compare_, typename Bound_>
    class UpperBounder {
    public:
      typedef Compare_    compare_type;
      typedef Bound_      bound_type;
    protected:
      compare_type _compare;
      bound_type   _bound;
    public:
      // Basic
      UpperBounder(){};
      UpperBounder(const bound_type& bound) : _bound(bound) {};
      inline void reset(const bound_type& bound) {this->_bound = bound;};
      // Extended
      const compare_type& compare(){return this->_compare;};
      const bound_type&   bound(){return this->_bound;};
      // Main
      template <typename Key_>
      bool operator()(const Key_& key){return !this->compare()(this->bound(),key);};
    };// class UpperBounder

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
      const arrow_type&  arrow()  const {return *this;};
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
        os << "<" << a._source << "--(" << a._color << ")-->" << a._target << ">";
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
    // Arrow Sequence type
    //
    #undef  __CLASS__
    #define __CLASS__ "ArrowSequence"
    template <typename XSifter_, typename Index_, typename KeyExtractor_, typename ValueExtractor_>
    class ArrowSequence {
    public:
      typedef ArrowSequence                              arrow_sequence_type;
      typedef XSifter_                                   xsifter_type;
      typedef Index_                                     index_type;
      typedef KeyExtractor_                              key_extractor_type;
      typedef ValueExtractor_                            value_extractor_type;
      //
      typedef typename key_extractor_type::result_type   key_type;
      typedef typename value_extractor_type::result_type value_type;
      //
      typedef typename xsifter_type::rec_type            rec_type;
      typedef typename xsifter_type::arrow_type          arrow_type;
      typedef typename arrow_type::source_type           source_type;
      typedef typename arrow_type::target_type           target_type;
      //
      typedef typename index_type::key_compare           index_compare_type;
      typedef typename index_type::iterator              itor_type;
      typedef typename index_type::const_iterator        const_itor_type;
      //
      // cookie_type
      //
      struct cookie_type {
        itor_type        segment_end;
        cookie_type(){};
        // Printing
        friend std::ostream& operator<<(std::ostream& os, const cookie_type& cookie) {
          os << "[...," << *(cookie.segment_end) << "]";
          return os;
        }

      };
      //
      // iterator_type
      //
      friend class iterator;
      class iterator {
      public:
        // Parent sequence type
        friend class ArrowSequence;
        typedef ArrowSequence                                  sequence_type;
        typedef typename sequence_type::itor_type              itor_type;
        typedef typename sequence_type::cookie_type            cookie_type;
        // Value types
        typedef typename sequence_type::value_extractor_type   value_extractor_type;
        typedef typename value_extractor_type::result_type     value_type;
        // Standard iterator typedefs
        typedef std::input_iterator_tag                        iterator_category;
        typedef int                                            difference_type;
        typedef value_type*                                    pointer;
        typedef value_type&                                    reference;
      protected:
        // Parent sequence
        sequence_type  *_sequence;
        // Underlying iterator
        itor_type       _itor;
        cookie_type _cookie;
        //cookie_type& cookie() {return _cookie;};
      public:
        iterator() : _sequence(NULL) {};
        iterator(sequence_type* sequence, const itor_type& itor, const cookie_type& cookie) : 
          _sequence(sequence), _itor(itor), _cookie(cookie) {};
        iterator(const iterator& iter) : _sequence(iter._sequence), _itor(iter._itor), _cookie(iter._cookie) {};
        ~iterator() {};
        //
        itor_type& itor(){return this->_itor;};
        //
        inline const source_type& source() const {return this->_itor->source();};
        inline const target_type& target() const {return this->_itor->target();};
        inline const arrow_type&  arrow()  const {return *(this->_itor);};
        inline const rec_type&    rec()    const {return *(this->_itor);};
        //
        inline bool              operator==(const iterator& iter) const {return this->_itor == iter._itor;}; 
        inline bool              operator!=(const iterator& iter) const {return this->_itor != iter._itor;}; 
        //
        // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
        inline const value_type  operator*() const {return this->_sequence->value(this->_itor);};
        //
        #undef  __FUNCT__
        #define __FUNCT__ "iterator::operator++"
        #undef  __ALE_XDEBUG__
        #define __ALE_XDEBUG__ 6
        inline iterator   operator++() { // comparison ignores cookies; only increment uses cookies
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>>" << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "*itor: " << *(this->_itor) << ", cookie: " << (this->_cookie);
          std::cout << std::endl;
        }
#endif
          this->_sequence->next(this->_itor, this->_cookie);
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << "itor: " << *(this->_itor) << ", cookie: " << (this->_cookie) << "\n";
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<<" << std::endl;
        }
#endif
          return *this;
        };
        inline iterator   operator++(int n) {iterator tmp(*this); ++(*this); return tmp;};
      };// class iterator
    protected:
      xsifter_type*        _xsifter;
      index_type*          _index;
      bool                 _keyless;
      key_type             _key;
      key_extractor_type   _kex;
      value_extractor_type _vex;
    public:
      //
      // Basic interface
      //
      ArrowSequence() : _xsifter(NULL), _index(NULL), _keyless(true) {};
      ArrowSequence(const ArrowSequence& seq) {if(seq._keyless) {reset(seq._xsifter, seq._index);} else {reset(seq._xsifter, seq._index, seq._key);};};
      ArrowSequence(xsifter_type *xsifter, index_type *index) {reset(xsifter, index);};
      ArrowSequence(xsifter_type *xsifter, index_type *index, const key_type& key){reset(xsifter, index, key);};
      virtual ~ArrowSequence() {};
      //
      void copy(const ArrowSequence& seq, ArrowSequence& cseq) {
        cseq._xsifter = seq._xsifter; cseq._index = seq._index; cseq._keyless = seq._keyless; cseq._key = seq._key;
      };
      void reset(xsifter_type *xsifter, index_type* index) {
        this->_xsifter = xsifter; this->_index = index; this->_keyless = true;
      };
      void reset(xsifter_type *xsifter, index_type* index, const key_type& key) {
        this->_xsifter = xsifter; this->_index = index; this->_key = key; this->_keyless = false;
      };
      ArrowSequence& operator=(const arrow_sequence_type& seq) {
        copy(seq,*this); return *this;
      };
      const value_type value(const itor_type itor) {return _vex(*itor);};
      //
      // Extended interface
      //
      const xsifter_type&       xsifter()                    const {return *this->_xsifter;};
      const index_type&         index()                      const {return *this->_index;};
      const bool&               keyless()                    const {return this->_keyless;};
      const key_type&           key()                        const {return this->_key;};
      const value_type&         value(const itor_type& itor) const {this->_vex(*itor);};
      //
      // Main interface
      //
      #undef  __FUNCT__
      #define __FUNCT__ "begin"
      #undef  __ALE_XDEBUG__
      #define __ALE_XDEBUG__ 6
      virtual iterator begin() {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>>" << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(this->keyless()) {
            std::cout << "keyless";
          }
          else {
            std::cout << "key: " << this->key();
          }
          std::cout << std::endl;
        }
#endif
        static itor_type itor;
        cookie_type cookie;
        if(!this->keyless()) {
          itor = this->index().lower_bound(this->key());
        }
        else {
          static std::pair<const_itor_type, const_itor_type> range;
          static LowerBounder<index_compare_type, key_type> lower;
          static UpperBounder<index_compare_type, key_type> upper;
          if(this->index().begin() != this->index().end()){
            lower.reset(this->_kex(*(this->index().begin())));
            upper.reset(this->_kex(*(this->index().begin())));
            range = this->index().range(lower, upper);
            itor = range.first; cookie.segment_end = range.second;        
          }
          else {
            itor = this->index().end(); cookie.segment_end = this->index().end();
          }
        }
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "*itor: " << *itor << ", cookie: " << cookie << "\n";
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<<" << "\n";
        }
#endif
        return iterator(this, itor, cookie);
      };// begin()
    protected:
      //
      #undef  __FUNCT__
      #define __FUNCT__ "next"
      #undef  __ALE_XDEBUG__
      #define __ALE_XDEBUG__ 6
      virtual void next(itor_type& itor, cookie_type& cookie) {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>>" << "\n";
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(this->keyless()) {
            std::cout << "keyless";
          }
          else {
            std::cout << "key: " << this->key();
          }
          std::cout << "\n";
        }
#endif
        if(!this->keyless()) {
          ++(itor); // FIX: use the record's 'next' method
        }
        else {
          if(this->_index->begin() != this->_index->end()){
            itor = cookie.segment_end;
            cookie.segment_end = this->index().upper_bound(this->_kex(*itor));
          }
        }
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "*itor: " << *itor << ", cookie: " << cookie << "\n";
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<<" << "\n";
        }
#endif
      };// next()
    public:
      //
      #undef  __FUNCT__
      #define __FUNCT__ "end"
      #undef  __ALE_XDEBUG__
      #define __ALE_XDEBUG__ 6
      virtual iterator end() {
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>>" << "\n";
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          if(this->keyless()) {
            std::cout << "keyless";
          }
          else {
            std::cout << "key: " << this->key();
          }
          std::cout << "\n";
        }
#endif
        static itor_type itor;
        cookie_type cookie;
        if(!this->keyless()) {
          itor = this->index().upper_bound(this->key());
        }
        else {
          itor = this->index().end(); 
          cookie.segment_end = itor;
        }
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "*itor: " << *itor << ", cookie: " << cookie << "\n";
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<<" << "\n";
        }
#endif
        return iterator(this, itor, cookie);
      };// end()
      //
      template<typename ostream_type>
      void view(ostream_type& os, const char* label = NULL){
        if(label != NULL) {
          os << "Viewing " << label << " sequence: ";
          if(this->keyless()) {
            os << "(keyless)";
          }
          else {
            os << "(key = " << this->key()<<")";
          }
          os << "\n";
        } 
        os << "[";
        for(iterator i = this->begin(); i != this->end(); i++) {
          os << " (" << *i << ")";
        }
        os << " ]" << "\n";
      };
      void addArrow(const arrow_type& a) {
        this->_xsifter->addArrow(a);
      };
      //
    };// class ArrowSequence  


    //
    // Slicing
    //
    namespace Slicing {
      //
      // NoSlices exception is thrown where no new slice can be allocated
      //
      class NoSlices : public ::ALE::Exception {
      };
      
      //
      // A Slice Rec<n> encodes n levels of pointers and markers above the base Arrow_ type
      // 
      template <typename Arrow_, typename Marker_, int n>
      struct Rec : public Rec<Arrow_,Marker_,n-1> {
        typedef Arrow_                                  arrow_type;
        typedef Marker_                                 marker_type;
        typedef Rec<Arrow_,Marker_,n-1>                 pre_rec_type;
      protected:
        // Slice pointer
        Rec         * _next;
        // Marker stored alongside the arrow data
        marker_type _marker;
      public:
        // Basic interface
        Rec(const arrow_type& arr) : pre_rec_type(arr), _next(NULL), _marker(marker_type()) {};
        Rec(const arrow_type& arr, const marker_type& m) : pre_rec_type(arr), _next(NULL), _marker(marker_type()){};
        //
        const marker_type    marker()  const {return this->_marker;};
        const Rec*           next()    const {return this->_next;};
        // Printing
        friend std::ostream& operator<<(std::ostream& os, const Rec& r) {
          os << "<" << r._marker << ">" << "[" << (pre_rec_type)r << "]";
          return os;
        }
      };// struct Rec
      //
      // A Slice Rec<0> inherits from Arrow_ but doesn't extend it: the base case of the Rec<n> type hierarchy.
      template <typename Arrow_, typename Marker_>
      struct Rec<Arrow_, Marker_, 0> : public Arrow_ {
        typedef Arrow_                                  arrow_type;
        typedef Marker_                                 marker_type;
        //
        Rec(const Rec& rec) : arrow_type(rec){};
      };

      //
      // Slicer<n> is capabale of allocating n slices, wrapped up in Rec_.
      // Rec_ is assumed to inherit from Rec<n>, Marker_ is essentially anything and n > 0.
      //
      template <typename Rec_, typename Marker_, int n>
      struct Slicer : public Slicer<Rec_, Marker_,n-1> {
        //
      public:
        typedef Slicer<Rec_, Marker_,n-1>                 pre_slicer_type;
        typedef typename pre_slicer_type::the_slice_type  the_slice_type;
        typedef Rec_                                      the_rec_type;
        typedef Marker_                                   marker_type;
        typedef Rec<the_rec_type::arrow_type, Marker_,n>  rec_type;
        //
        // Slice extends the canonical the_slicer_type; 
        // Slice is essentially a Sequence, in particular, it defines an iterator
        //
        // CONTINUE: changing over to the new Rec<n>/the_rec_type paradigm
        struct Slice : public the_slice_type {
          typedef Slicer                                  slicer_type;
          typedef typename slicer_type::rec_type          rec_type;
          typedef Rec_                                    the_rec_type;
          typedef typename Marker_                        marker_type;
          typedef typename the_slice_type::iterator       iterator;
        protected:
          slicer_type& _slicer;
          rec_type* _head;
          rec_type* _tail;
        public:
          // 
          // Basic interface
          //
          Slice(slicer_type& slicer) : _slicer(slicer), _head(NULL), _tail(NULL){};
          ~Slice() { this->_slicer.give_back(*this);};
          //
          // Main interface
          //
          virtual iterator begin(){ return iterator(this->_head);}; 
          virtual iterator end(){ return iterator(NULL);}; 
          virtual void add(the_rec_type& therec, marker_type marker) {
            rec_type& rec = therec; // cast to rec_type
            // add &rec at the tail of the slice linked list
            // we assume that rec is not transient;
            // set the current slice tail to point to rec 
            //   CAUTION: hoping for no cycles; can check itor's slice_marker, but will hurt performance
            //   However, the caller should really be checking that rec is not in the slice yet; otherwise the 'set' semantics are violated.
            // Also, this in principle violates multi_index record update rules, 
            //   since we are directly manipulating the record, but that's okay, since 
            //   slice data that we are updating is not used in the container ordering.
            rec._next = NULL;
            rec._marker = marker;
            this->_tail->next = &rec;
            this->_tail = &rec;
          };
          virtual marker_type marker(const the_rec_type& therec) {
            rec_type& rec = therec; // cast to rec_type 
            return rec._marker;
          };
          virtual void next(iterator& iter) {
            rec_type& rec = iter.rec(); // cast to rec_type
            iter.reset(*rec._next);
          };
          virtual void clean() {
            iterator iter = this->begin(); 
            iterator end = this->end();
            for(;iter != end;) {
              iterator tmp = iter; // cast to rec_type
              ++iter;
              rec_type& rec = iter.rec();
              rec._marker = marker_type();
              rec._next   = NULL;
            }
          };//clean()
        };// struct Slice
        typedef Slice slice_type;
      public:
        //
        // Basic interface
        //
        Slicer() : _slice(Slice(*this)), _taken(false) {};
        ~Slicer(){};
        //
        // Main
        //
        inline slice_type& take() {
          if(!this->_taken) {
            this->_taken = true;
            return *(this->_slice);
          }
          else {
            return this->pre_slicer_type::take();
          }
        };// take()
        //
        inline void give_back(slice_type& slice) {// this cannot be virtual, since slice must be returned to the correct slicer
          slice.clean();
        };// give_back()
      protected:
        slice_type _slice;
        bool _taken;
      };// struct Slicer<n>
      
      
      //
      // Slicer<0> allocates no Slices, throws a NoSlices exception,
      // when a slice is being 'taken' or 'returned'.
      // However, it defines the base calsses for the Slice and iterator
      // hierarchy that are used by more sophisticated Slicers.
      // Slicer/Slice is a hierarchical construction that dispatches further
      // along the hierarchy, if a slice cannot be allocated locally.
      // Still, we need to return a common iterator class, which is defined
      // in Slicer<0>.  Slicer<0> also terminates the forwarding hierarchy.
      //
      template <typename Rec_, typename Marker_>
      struct Slicer<Rec_, Marker_, 0> {
        typedef Rec_                          rec_type;
        typedef Marker_                       marker_type;
        typedef typename rec_type::arrow_type arrow_type;
        typedef Slicer                        proto_slicer_type;
        //
        struct Slice {
          typedef Slice slice_type;
          typedef Rec_  rec_type;
          //
          // iterator dispatches to Slice for the fundamental 'next' operation.
          // That method is virtual in Slice and is overloaded by Slice's descendants.
          //
          class iterator {
          public:
            // Standard iterator typedefs
            typedef arrow_type                                     value_type;
            typedef std::input_iterator_tag                        iterator_category;
            typedef int                                            difference_type;
            typedef value_type*                                    pointer;
            typedef value_type&                                    reference;
          protected:
            slice_type  *_slice;
            rec_type    *_ptr;
          public:
            iterator() :_slice(NULL), _ptr(NULL) {};
            iterator(slice_type *slice, rec_type* ptr) : _slice(slice), _ptr(ptr) {};
            iterator(const iterator& iter)             : _slice(iter._slice), _ptr(iter._ptr) {};
          public:
            rec_type& rec(){return *(this->_ptr);};
            void      reset(rec_type& rec) {this->_ptr = &rec;};
          public:
            iterator   operator=(const  iterator& iter) const {this->_ptr = iter._ptr; return *this;};
            bool       operator==(const iterator& iter) const {return this->_ptr == iter._ptr;};
            bool       operator!=(const iterator& iter) const {return this->_ptr != iter._ptr;};
            iterator   operator++() {
              // We don't want to make the increment operator virtual, since this entails
              // carrying of a vtable point around with each iterator object.
              // instead, we shift the indirection (of a vtable lookup) to the Slice object.
              this->_slice->next(*this);
              return *this;
            };
            iterator   operator++(int n) {iterator tmp(*this); ++(*this); return tmp;};
            //
            arrow_type&       arrow() {return *(this->_ptr);};
            const arrow_type& operator*() const {return this->arrow();};
          };// iterator
          //
          // Basic
          //
          Slice(){};
          virtual ~Slice(){};
          //
          // Main
          // 
          virtual iterator    begin()=0; 
          virtual iterator    end(); 
          virtual void        next(iterator& iter);
          //
          virtual void        add(rec_type& rec);
          virtual marker_type marker(rec_type& rec);
          virtual void        clean();
        };// Slice
        //
        typedef Slice slice_type;
        // Canonical Slice type, a pointer to which is returned by all Slicers
        typedef Slice    the_slice_type;
      public:
        //
        // Basic
        //
        Slicer(){};
        virtual ~Slicer(){};
        //
        // Main
        //
        Slice& take() {NoSlices e; /* e << "Slicer<0> has no slices"; */ throw e; return Slice();};
        void   give_back(Slice& slice) {NoSlices e; /* e << "ProtoSlicer has no slices"; */ throw e;};
      };// class Slicer<0>
    }//namespace Slicing

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

  }; // namespace XSifterDef
  
  //
  // XSifter definition
  //
  template<typename Arrow_, 
           typename TailOrder_  = XSifterDef::SourceColorOrder<Arrow_>,
           int SliceDepth = 1>
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
    // Right now we assume that arrows have color;  this allows for splitting cones on color.
    // It is not necessary in general, but it will require two different XSfiter types: with and without color.
    // This is because an XSifter with color has a wider interface: retrieve the subcone with a given color.
    typedef typename arrow_type::color_type                        color_type;
    //
    // Slicing
    //
    typedef ::ALE::XSifterDef::Slicer<arrow_type, int, SliceDepth> slicer_type;
    typedef typename slicer_type::the_slice_type                   slice_type;
    typedef typename slicer_type::rec_type                         rec_type;
    //struct rec_type : public arrow_type{};
    // 
    // Key extractors
    //
    // Despite the fact that extractors will operate on rec_type objects, they must be defined as operating on arrow_type objects.
    // This will work because rec_type is assumed to inherit from arrow_type.
    // If rec_type with the methods inherited from arrow_type were used to define extractors, there would be a conversion problem:
    // an arrow_type member function (even as inherited by rec_type) cannot be converted to (the identical) rec_type member function.
    // Just try this, if you want to see what happens:
    //           typedef ALE_CONST_MEM_FUN(rec_type, source_type&,    &rec_type::source)  source_extractor_type;
    //
    typedef ALE_CONST_MEM_FUN(arrow_type, source_type&,    &arrow_type::source)  source_extractor_type;
    typedef ALE_CONST_MEM_FUN(arrow_type, target_type&,    &arrow_type::target)    target_extractor_type;
    typedef ALE_CONST_MEM_FUN(arrow_type, color_type&,     &arrow_type::color)     color_extractor_type;
    typedef ALE_CONST_MEM_FUN(arrow_type, arrow_type&,     &arrow_type::arrow)     arrow_extractor_type;

    //
    // Comparison operators
    //
    typedef std::less<typename rec_type::target_type> target_order_type;
    typedef TailOrder_                                tail_order_type;
    //
    // Rec 'Cone' order type: first order by target then using a custom tail_order
    typedef 
    XSifterDef::RecKeyXXXOrder<rec_type, 
                               target_extractor_type, target_order_type,
                               XSifterDef::RecKeyOrder<rec_type, arrow_extractor_type, tail_order_type> >
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
    // Sequence types
    //
    typedef ALE::XSifterDef::ArrowSequence<xsifter_type, cone_index_type, target_extractor_type, target_extractor_type>   BaseSequence;
    typedef ALE::XSifterDef::ArrowSequence<xsifter_type, cone_index_type, target_extractor_type, source_extractor_type>   ConeSequence;
    typedef ALE::XSifterDef::ArrowSequence<xsifter_type, cone_index_type, arrow_extractor_type,  source_extractor_type>   ColorConeSequence;
    //
    //
    // Basic interface
    //
    //
    XSifter(const MPI_Comm comm, int debug = 0) : // FIX: Should really inherit from XParallelObject
      XObject(debug), _rec_set(), 
      _cone_index(::boost::multi_index::get<ConeTag>(this->_rec_set))
    {};
    //
    // Extended interface
    //
    void addArrow(const arrow_type& a) {
      this->_rec_set.insert(rec_type(a));
    };
    //
    void cone(const target_type& t, ConeSequence& cseq) {
      cseq.reset(this, &_cone_index, t);
    };
    ConeSequence& cone(const target_type& t) {
      static ConeSequence cseq;
      this->cone(t,cseq);
      return cseq;
    };
    //
    void cone(const target_type& t, const color_type& c, ColorConeSequence& ccseq) {
      static ALE::pair<target_type, color_type> comb_key;
      comb_key.first = t; comb_key.second = c;
      ccseq.reset(this, &_cone_index, comb_key);
    };
    //
    ColorConeSequence& cone(const target_type& t, const color_type& c) {
      static ColorConeSequence ccseq;
      this->cone(t,c,ccseq);
      return ccseq;
    };
    //
    void base(BaseSequence& seq) {
      seq.reset(this, &_cone_index);
    };
    BaseSequence& base() {
      static BaseSequence bseq;
      this->base(bseq);
      return bseq;
    };
    // 
    slice_type& slice(){ return this->_slicer.take();};
    //
    template<typename ostream_type>
    void view(ostream_type& os, const char* label = NULL){
      if(label != NULL) {
        os << "Viewing " << label << " XSifter (debug: " << this->debug() << "): " << "\n";
      } 
      else {
        os << "Viewing a XSifter (debug: " << this->debug() << "): " << "\n";
      } 
      os << "Cone index: (";
        for(typename cone_index_type::iterator itor = this->_cone_index.begin(); itor != this->_cone_index.end(); ++itor) {
          os << *itor << " ";
        }
      os << ")" << "\n";
    };
    //
    // Direct access (a kind of hack)
    //
    // Whole container begin/end
    typedef typename cone_index_type::iterator iterator;
    iterator begin() const {return this->_cone_index.begin();};
    iterator end() const {return this->_cone_index.end();};

  public:
  // set of arrow records
    rec_set_type     _rec_set;
    cone_index_type& _cone_index;
    slicer_type      _slicer;
  public:
    //
  }; // class XSifter

  
} // namespace ALE

#endif
