#ifndef included_ALE_XSifter_hh
#define included_ALE_XSifter_hh

#include <X.hh>

namespace ALE {
  
  namespace XSifterDef {

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
        if(this->keyless()) {
          if(this->_index->begin() != this->_index->end()){
            itor = cookie.segment_end;
            cookie.segment_end = this->index().upper_bound(this->_kex(*itor));
          }
        }
        else {
          ++(itor); // FIX: use the record's 'next' method
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
    // Arrow-Link Sequence type
    //
    #undef  __CLASS__
    #define __CLASS__ "ArrowLinkSequence"
    template <typename XSifter_, typename Index_, typename KeyExtractor_, typename NextExtractor_, typename ValueExtractor_>
    class ArrowLinkSequence {
    public:
      typedef ArrowLinkSequence                          arrow_link_sequence_type;
      typedef XSifter_                                   xsifter_type;
      typedef Index_                                     index_type;
      typedef KeyExtractor_                              key_extractor_type;
      typedef NextExtractor_                             next_extractor_type;
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
      // iterator_type
      //
      friend class iterator;
      class iterator {
      public:
        // Parent sequence type
        friend class ArrowLinkSequence;
        typedef ArrowLinkSequence                              sequence_type;
        typedef typename sequence_type::itor_type              itor_type;
        typedef typename sequence_type::rec_type               rec_type;
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
        // Underlying record
        rec_type *_rec, *_seg; // seg == "segment_end"
      public:
        iterator() : _sequence(NULL) {};
        iterator(sequence_type* sequence, rec_type* rec, rec_type* seg) : 
          _sequence(sequence), _rec(rec), _seg(seg) {};
        iterator(const iterator& iter) : 
          _sequence(iter._sequence), _rec(iter._rec), _seg(iter._seg) {};
        ~iterator() {};
        //
        inline const source_type& source() const {return this->_rec->source();};
        inline const target_type& target() const {return this->_rec->target();};
        inline const arrow_type&  arrow()  const {return *(this->_rec);};
        inline const rec_type&    rec()    const {return *(this->_rec);};
        //
        inline bool              operator==(const iterator& iter) const {bool res; res = (this->_rec == iter._rec); return res;}; 
        inline bool              operator!=(const iterator& iter) const {bool res; res = (this->_rec != iter._rec); return res;}; 
        //
        // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
        inline const value_type  operator*() const {return this->_sequence->value(this->_rec);};
        //
        #undef  __FUNCT__
        #define __FUNCT__ "iterator::operator++"
        #undef  __ALE_XDEBUG__
        #define __ALE_XDEBUG__ 6
        inline iterator   operator++() { 
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":>>>" << std::endl;
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "rec: ";
          if(this->_rec!=NULL){std::cout << *(this->_rec);}else{std::cout << "NULL";};
          std::cout << "seg: ";
          if(this->_seg!=NULL){std::cout << *(this->_seg);}else{std::cout << "NULL";};
          std::cout << std::endl;
        }
#endif
          this->_sequence->next(this->_rec, this->_seg);
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << "rec: ";
          if(this->_rec!=NULL){std::cout << *(this->_rec);}else{std::cout << "NULL";};
          std::cout << "seg: ";
          if(this->_seg!=NULL){std::cout << *(this->_seg);}else{std::cout << "NULL";};
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
      next_extractor_type  _nex;
      value_extractor_type _vex;
    public:
      //
      // Basic interface
      //
      ArrowLinkSequence() : _xsifter(NULL), _index(NULL), _keyless(true) {};
      ArrowLinkSequence(const ArrowLinkSequence& seq) {if(seq._keyless) {reset(seq._xsifter, seq._index);} else {reset(seq._xsifter, seq._index, seq._key);};};
      ArrowLinkSequence(xsifter_type *xsifter, index_type *index) {reset(xsifter, index);};
      ArrowLinkSequence(xsifter_type *xsifter, index_type *index, const key_type& key){reset(xsifter, index, key);};
      virtual ~ArrowLinkSequence() {};
      //
      void copy(const ArrowLinkSequence& seq, ArrowLinkSequence& cseq) {
        cseq._xsifter = seq._xsifter; cseq._index = seq._index; cseq._keyless = seq._keyless; cseq._key = seq._key;
      };
      void reset(xsifter_type *xsifter, index_type* index) {
        this->_xsifter = xsifter; this->_index = index; this->_keyless = true;
      };
      void reset(xsifter_type *xsifter, index_type* index, const key_type& key) {
        this->_xsifter = xsifter; this->_index = index; this->_key = key; this->_keyless = false;
      };
      ArrowLinkSequence& operator=(const arrow_link_sequence_type& seq) {
        copy(seq,*this); return *this;
      };
      const value_type value(rec_type const* _rec) {return _vex(*_rec);};
      //
      // Extended interface
      //
      const xsifter_type&       xsifter()                    const {return *this->_xsifter;};
      const index_type&         index()                      const {return *this->_index;};
      const bool&               keyless()                    const {return this->_keyless;};
      const key_type&           key()                        const {return this->_key;};
      const value_type&         value(const rec_type*& rec)  const {this->_vex(*rec);};
    protected:
      // aux
      inline rec_type* itor_to_rec_ptr(const itor_type& itor) {
        const rec_type& crec = *(itor);
        return const_cast<rec_type*>(&crec);
      };
      //
      inline rec_type* itor_to_rec_ptr_safe(const itor_type& itor) {
        rec_type* _rec;
        if(itor == this->_index->end()){
          _rec = NULL;
        }
        else {
          const rec_type& crec = *(itor);
          _rec = const_cast<rec_type*>(&crec);
        }
        return _rec;
      };

    public:
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
        static rec_type *_rec, *_seg;
        if(this->keyless()) {
          static std::pair<const_itor_type, const_itor_type> range;
          static LowerBounder<index_compare_type, key_type> lower;
          static UpperBounder<index_compare_type, key_type> upper;
          if(this->index().begin() != this->index().end()){
            lower.reset(this->_kex(*(this->index().begin())));
            upper.reset(this->_kex(*(this->index().begin())));
            range = this->index().range(lower, upper);
            _rec = itor_to_rec_ptr(range.first); 
            _seg = itor_to_rec_ptr_safe(range.second);
          }
          else {
            _rec = NULL; _seg = NULL;
          }
        }
        else {
          itor = this->index().lower_bound(this->key());
          _rec = itor_to_rec_ptr_safe(itor);
          _seg = NULL;
        }
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "rec: ";
          if(_rec!=NULL){std::cout << *(_rec);}else{std::cout << "NULL";};
          std::cout << "seg: ";
          if(_seg!=NULL){std::cout << *(_seg);}else{std::cout << "NULL";};
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<<" << "\n";
        }
#endif
        return iterator(this, _rec, _seg);
      };// begin()
    protected:
      //
      #undef  __FUNCT__
      #define __FUNCT__ "next"
      #undef  __ALE_XDEBUG__
      #define __ALE_XDEBUG__ 6
      virtual void next(rec_type*& _rec, rec_type*& _seg) {
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
        if(this->keyless()) {
          if(this->_index->begin() != this->_index->end()){
            _rec = _seg;
            if(_rec != NULL) {
              itor_type itor = this->index().upper_bound(this->_kex(*_rec)); 
              _seg = itor_to_rec_ptr_safe(itor);
            }
            // else _seg is already NULL
          }
        }
        else {
          _rec = this->_nex(*_rec);
        }
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "rec: ";
          if(_rec!=NULL){std::cout << *(_rec);}else{std::cout << "NULL";};
          std::cout << "seg: ";
          if(_seg!=NULL){std::cout << *(_seg);}else{std::cout << "NULL";};
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
        static rec_type *_rec, *_seg;
        if(this->keyless()){
          _rec = NULL; _seg = NULL;
        }
        else {
          itor_type itor = this->index().upper_bound(this->key()); 
          _rec = itor_to_rec_ptr_safe(itor);
        }
#ifdef ALE_USE_DEBUGGING
        if(ALE_XDEBUG) {
          std::cout << __CLASS__ << "::" << __FUNCT__ << ": ";
          std::cout << "rec: ";
          if(_rec!=NULL){std::cout << *(_rec);}else{std::cout << "NULL";};
          std::cout << "seg: ";
          if(_seg!=NULL){std::cout << *(_seg);}else{std::cout << "NULL";};
          std::cout << __CLASS__ << "::" << __FUNCT__ << ":<<<" << "\n";
        }
#endif
        return iterator(this, _rec, _seg);
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
    };// class ArrowLinkSequence  


    //
    // Slicing
    //
    //
    // NoSlices exception is thrown where no new slice can be allocated
    //
    struct NoSlices : public ::ALE::XException {
      NoSlices() : ALE::XException("No slices left"){};
    };
    namespace Slicing {  
      //
      // A Slice Rec<n> encodes n levels of pointers and markers above the base Arrow_ type
      // 
      template <typename Arrow_, typename Marker_, int n>
      struct Rec : public Rec<Arrow_,Marker_,n-1> {
        typedef Arrow_                                  arrow_type;
        typedef Marker_                                 marker_type;
        typedef Rec<Arrow_,Marker_,n-1>                 pre_rec_type;
        // Slice pointer
        Rec         * next;
        // Marker stored alongside the arrow data
        marker_type marker;
      public:
        // Basic interface
        Rec(const Rec& rec)        : pre_rec_type(rec), next(NULL), marker(marker_type()) {};
        Rec(const arrow_type& arr) : pre_rec_type(arr), next(NULL), marker(marker_type()) {};
        Rec(const arrow_type& arr, const marker_type& m) : pre_rec_type(arr), next(NULL), marker(marker_type()){};
        // Printing
        friend std::ostream& operator<<(std::ostream& os, const Rec& r) {
          os << "<" << r.marker << ">" << "[" << (pre_rec_type)r << "]";
          return os;
        }
      };// struct Rec
      //
      // A Slice Rec<0> inherits from Arrow_ but doesn't extend it: the base case of the Rec<n> type hierarchy.
      // It must have construction interface expected by the descendant Rec<1> type.
      template <typename Arrow_, typename Marker_>
      struct Rec<Arrow_, Marker_, 0> : public Arrow_ {
        typedef Arrow_                                  arrow_type;
        typedef Marker_                                 marker_type;
        //
        Rec(const Rec& rec)        : arrow_type(rec) {};
        Rec(const arrow_type& arr) : arrow_type(arr) {};
        Rec(const arrow_type& arr, const marker_type& m) : arrow_type(arr) {};
      };

      //
      //   Slicer<n> is capabale of allocating n slices, wrapped up in Rec_, which must extend Rec<n>.
      // Marker_ is essentially anything and n > 0.
      //   The presentation of the interface is done in terms of two canonical types: 
      // the_rec_type and the_slice_type: the_slice_type objects contain the_rec_type records.
      // While the_rec_type == Rec_ descends from the 'top', the_slice_type comes up from the 'bottom' by being 
      // inherited from the ancestral pre_slicer_type.
      //   Rationale: the easiest way to construct a record with several levels of data -- markers
      // and sequence pointers -- is via a class hierarchy, each adding the required extra data members.
      // Hence, the hierarchy of record classes topped by the canonical or universal record type -- the_rec_type.
      // Each slice is defined by the sequence pointers at a given level, hence, a separate slice type that 
      // operates on the data members at the correct level by casting the_rec_type objects to an ancestor rec_type,
      // which defines a single (incremental) level of data members only.
      //   Since the user wants just an unused slice at any level, without having to worry about the different 
      // types that go with different levels.  Thus, the slices returned to the user must be of some canonical 
      // type -- the_slice_type.  The same applies to the iterator type that the_slice_presents presents.
      // The easiest way to construct a dispatch mechanism, which traverses the hierarchy in search of any free
      // slice, is via a corresponding hierarchy of slicers, which either hold a free slice that can be returned,
      // or forward the call down the hierarchy.  The top-level slicer class is presented to the user and initiates
      // the forwarding sequence;  the hierarchy bottoms out at the dummy slicer that returns nothing and throws
      // an exception.
      //   Note that objects of the_slice_type are RETURNED by a slicer, hence, each particular slice_type
      // object must be castable to the_slice_type.  At the same time, the_rec_objects are ACCEPTED by 
      // slice_type objects, hence, must be castable to each particular rec_type.  This explains the direction
      // of inheritance: from the_slicer_type and to the_rec_type.
      //   Note finally that slice destruction must be virtualized, since the slice must be released in the appropriate
      // slicer, while the user only has a reference to an object of the_slice_type.
      //   Also see the construction of the_slice_type in Slicer<0>
      //
      template <typename Rec_, typename Marker_, int n>
      struct Slicer : public Slicer<Rec_, Marker_,n-1> {
        //
      public:
        // ancestor
        typedef Slicer<Rec_, Marker_,n-1>                 pre_slicer_type;
        // canonical types
        typedef typename pre_slicer_type::the_slice_type  the_slice_type;
        typedef Rec_                                      the_rec_type;
        typedef Marker_                                   marker_type;
        // local types
        typedef Rec<typename the_rec_type::arrow_type, Marker_,n>  rec_type;
        //
        // Slice extends the canonical the_slicer_type; 
        // Slice is essentially a Sequence, in particular, it defines an iterator
        struct Slice : public the_slice_type {
          // canonical types
          typedef Rec_                                    the_rec_type;
          typedef Marker_                                 marker_type;
          typedef typename the_slice_type::iterator       iterator;
          // local types
          typedef Slicer                                  slicer_type;
          typedef typename slicer_type::rec_type          rec_type;
        protected:
          slicer_type& _slicer;
          rec_type* _head;
          rec_type* _tail;
        public:
          // 
          // Basic interface
          //
          Slice(slicer_type& slicer) : _slicer(slicer), _head(NULL), _tail(NULL){};
          virtual ~Slice() { this->_slicer.give_back(this);};
          //
          // Main interface
          //
          virtual iterator begin(){ return iterator(this, this->_head);}; 
          virtual iterator end(){ return iterator(this, NULL);}; 
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
            rec.next = NULL;
            rec.marker = marker;
            if(this->_tail != NULL) { // slice already has recs in it
              this->_tail->next = &rec;
            }
            else { // first rec in the slice
              this->_head = &rec;
            }
            this->_tail = &rec;
            
          };
          virtual marker_type marker(const the_rec_type& therec) {
            const rec_type& rec = therec; // cast to rec_type 
            return rec.marker;
          };
          virtual marker_type marker(const iterator& iter) {
            rec_type& rec = the_slice_type::rec(iter); // cast to rec_type
            return rec.marker;
          };
          virtual void next(iterator& iter) {
            rec_type& rec = the_slice_type::rec(iter); // cast to rec_type
            the_slice_type::reset_rec(iter,*(rec.next)); // make pointer point to the next rec
          };
          virtual void clean() {
            iterator iter = this->begin(); 
            iterator end = this->end();
            for(;iter != end;) {
              iterator tmp = iter; 
              ++iter;
              rec_type& rec = the_slice_type::rec(tmp);
              rec.marker = marker_type();
              /*rec.next   = NULL;*/ // FIX: this is, in principle, unnecessary, as long as the head and tail are wiped out
            }
            this->_head = NULL; this->_tail = NULL;
          };//clean()
        };// struct Slice
        typedef Slice slice_type;
      public:
        //
        // Basic interface
        //
        Slicer() : _taken(false) {};
       ~Slicer() {};
        //
        // Main
        //
        // take() must return Obj<the_slice_type> rather than the_slice_type*,
        // since the latter need not be automatically destroyed and cleaned.
        // It would be unnatural (and possibly unallowed by the Obj interface)
        // to return a the_slice_type* to be wrapped as Obj later.
        inline Obj<the_slice_type> take() { 
          if(!this->_taken) {
            this->_taken = true;
            return Obj<the_slice_type>(new slice_type(*this));
          }
          else {
            return this->pre_slicer_type::take();
          }
        };// take()
        //
        // give_back() cannot accept Obj<the_slice_type>, since it is intended
        // to be called from the_slice_type's destructor, which has no Obj.
        inline void give_back(the_slice_type* slice) {
          slice->clean();
          this->_taken = false;
        };// give_back()
      protected:
        bool _taken;
      };// struct Slicing::Slicer<n>
      
      
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
        // canonical types
        typedef Rec_                              the_rec_type;
        typedef Marker_                           marker_type;
        typedef typename the_rec_type::arrow_type arrow_type;
        // local types
        typedef Rec_                              rec_type;
        //
        struct Slice {
          // canonical types
          typedef Slice the_slice_type;
          typedef Rec_  the_rec_type;
          // local types
          typedef Slice slice_type;
          typedef Rec_  rec_type;
          //
          // iterator dispatches to Slice for the fundamental 'next' operation.
          // That method is virtual in Slice and is overloaded by Slice's descendants.
          //
          class iterator {
            friend class Slice;
          public:
            // Standard iterator typedefs
            typedef arrow_type                                     value_type;
            typedef std::input_iterator_tag                        iterator_category;
            typedef int                                            difference_type;
            typedef value_type*                                    pointer;
            typedef value_type&                                    reference;
          protected:
            the_slice_type  *_slice;
            the_rec_type    *_rec;
          public:
            iterator() :_slice(NULL), _rec(NULL) {};
            iterator(the_slice_type *_slice, the_rec_type* _rec) : _slice(_slice), _rec(_rec) {};
            iterator(const iterator& iter)             : _slice(iter._slice), _rec(iter._rec) {};
          public:
            iterator   operator=(const  iterator& iter) {
              this->_slice = iter._slice;
              this->_rec   = iter._rec; 
              return *this;
            };
            //   Can equality comparison be done across different slices?  
            // In principle, same rec pointer can be used by several different slices at different levels.
            // Is it the right idea to say that two iterators from different slices pointing to these two
            // different recs are equal?  For the moment, we treat them as different for safety
            bool       operator==(const iterator& iter) const {return this->_slice == iter._slice && this->_rec == iter._rec;};
            bool       operator!=(const iterator& iter) const {return this->_slice != iter._slice || this->_rec != iter._rec;};
            iterator   operator++() {
              // We don't want to make the increment operator virtual, since this entails
              // carrying of a vtable point around with each iterator object.
              // instead, we shift the indirection (of a vtable lookup) to the Slice object.
              this->_slice->next(*this);
              return *this;
            };
            iterator   operator++(int n) {iterator tmp(*this); ++(*this); return tmp;};
            //
            const arrow_type& arrow()     const {return *(this->_rec);}; // we assume Rec_ implements the Arrow concept.
            const arrow_type& operator*() const {return this->arrow();};            
          };// iterator
          //
          // Basic
          //
          Slice(){};
          Slice(const Slice& slice){};
          virtual ~Slice(){};
          //
          // Main (an abstract interface that never gets invoked, but defines the interface for descendants).
          // 
          virtual iterator    begin()=0; 
          virtual iterator    end()=0; 
          virtual void        next(iterator& iter)=0;
          //
          virtual void        add(the_rec_type& rec, marker_type marker)=0;
          virtual marker_type marker(const the_rec_type& rec)=0;
          virtual marker_type marker(const iterator& iter)=0;
          virtual void        clean()=0;
        protected:
          the_rec_type& rec(const iterator& iter) const {return *(iter._rec);};
          void          reset_rec(iterator& iter, the_rec_type& rec) const {iter._rec = &rec;};
        };// Slice
        // canonical slice type
        typedef Slice    the_slice_type;
        // local slice type
        typedef Slice slice_type;
      public:
        //
        // Basic
        //
        Slicer(){};
        virtual ~Slicer(){};
        //
        // Main
        //
        Obj<the_slice_type> take()                          {ALE::XSifterDef::NoSlices e; throw e; return Obj<the_slice_type>();};
        void                give_back(the_slice_type* slice){ALE::XSifterDef::NoSlices e; throw e;};
      };// class Slicing::Slicer<0>
    }//namespace Slicing

    //
    // Slicer: this class is used by XSifter to implement slicing.
    // 
    // If confused by the presence of another Slicer class -- Slicing::Slicer<Rec_,Marker_,n> -- read this.
    // Otherwise, forget about Slicing::Slicer and proceed directly to use Slicer: Slicer<ArrowModel, Marker, n>.
    //
    //   The problem with Slicing::Slicer<n> is that in principle it doesn't know the total depth of the hierarchy
    // it lives in.  In particular, it expects a nebulous Rec_ that extends Rec<n> and implements Arrow, but may
    // also extend Rec<n+1> etc.  The user must decide what the depth of the hierarchy is and produce 
    // an appropriate Rec_. So direct usage of Slicing::Slicer might also be confusing.  
    //   To aleviate this confusion, Slicer takes an Arrow model, a Marker and the total desired slicing depth n
    // and produces a usable Slicer. We feel that having two Slicer classes may be less confusing, as the user 
    // should (ideally) never have a need to use Slicing::Slicer directly.
    //   Finally, having a separate Slicer class allows us to wrap some of the raw functionality
    // of Slicing::Slicer, such as wrapping up Obj<the_slice_type> as a SliceSequence class with 
    // the usual Sequence interface.  We also define templated iterators with custom dereference operators
    // based on extractor template parameters; these could be useful in implementing different Slice-based
    // sequences.  
    //
    template<typename Arrow_, typename Marker_, int n>
    class Slicer : public Slicing::Slicer< Slicing::Rec<Arrow_, Marker_, n>, Marker_, n> {
    public:
      typedef Slicing::Slicer< Slicing::Rec<Arrow_, Marker_, n>, Marker_, n> super;
      //
      typedef typename super::the_slice_type            the_slice_type;
      //
      typedef typename super::the_rec_type              rec_type;
      typedef Marker_                                   marker_type;
      typedef Obj<the_slice_type>                       slice_type;
      //
      // Since slice_type is actually an Obj type, we wrap it in yet another type,
      // SliceSequence, which forwards the calls to begin(), end() etc bypassing 
      // the '->' intermediary and making slice_type look more like a sequence. 
      // SliceSequence is also templated over an extractor type, which is used
      // in an overloaded iterator.
      //
      template <typename Extractor_>
      class SliceSequence : public slice_type {
      public:
        typedef typename the_slice_type::the_rec_type rec_type;
        typedef typename rec_type::marker_type        marker_type;
        typedef typename rec_type::arrow_type         arrow_type;
        //
        class iterator : public the_slice_type::iterator {
        public:
          typedef Extractor_                                     extractor_type;
          typedef typename the_slice_type::iterator              the_iterator;
          // Standard iterator typedefs
          typedef typename extractor_type::result_type           value_type;
          typedef std::input_iterator_tag                        iterator_category;
          typedef int                                            difference_type;
          typedef value_type*                                    pointer;
          typedef value_type&                                    reference;
        public:
          iterator() :the_iterator() {};
          iterator(const the_iterator& theiter)  : the_iterator(theiter) {};
          const value_type& operator*() const {
            static extractor_type ex;
            return ex(this->arrow());
          };
        };// class iterator
      public:
        SliceSequence(const slice_type& slice) : slice_type(slice) {};
        //
        iterator    begin() {return this->pointer()->begin();};
        iterator    end()   {return this->pointer()->end();};
        void        add(const rec_type& crec, marker_type marker = marker_type()){
          // crec is typically obtained by referencing a multi_index index iterator,
          // hence it is likely to be 'const'.
          // It is okay to cast away this const, since we won't be modifying 
          // any of its elements that affect the index.
          // Otherwise we'd need to use a multi_index record modifier, 
          // which might be costly.
          rec_type& rec = const_cast<rec_type&>(crec);
          this->pointer()->add(rec,marker);
        };
        template <typename RecSequence_>
        void add(const RecSequence_ in, const marker_type& add_marker, const marker_type& in_marker = marker_type()) {
          for(typename RecSequence_::iterator iter = in.begin(); iter!=in.end(); ++iter) {
            if(this->marker(*iter) == in_marker) {
              this->add(*iter, add_marker);
            }
          }
        };// add(seq)
        marker_type marker(const rec_type& rec){return this->pointer()->marker(rec);};
        marker_type marker(const iterator& iter){return this->pointer()->marker(iter);};
        void        clean(){this->pointer()->clean();};
        //
        template<typename ostream_type>
        void view(ostream_type& os, const char* label = NULL){
          os << "Viewing SliceSequence";
          if(label != NULL) {
            os << " " << label;
          } 
          os << ":\n[[ ";
          iterator sbegin = this->begin();
          iterator send = this->end();
          iterator iter;
          for(iter = sbegin; iter!=send; ++iter) {
            os << "<" << this->marker(iter) << ">[" << *iter << "] ";
          }
          os << "]]\n";
        };//view()
      };//SliceSequence()
    }; // class Slicer


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
    // Key extractors
    //
    // Despite the fact that extractors will operate on rec_type objects, they must be defined as operating on arrow_type objects.
    // This will work because rec_type is assumed to inherit from arrow_type.
    // If rec_type with the methods inherited from arrow_type were used to define extractors, there would be a conversion problem:
    // an arrow_type member function (even as inherited by rec_type) cannot be converted to (the identical) rec_type member function.
    // Just try this, if you want to see what happens:
    //           typedef ALE_CONST_MEM_FUN(rec_type, source_type&,    &rec_type::source)  source_extractor_type;
    //
    typedef ALE_CONST_MEM_FUN(arrow_type, source_type&,    &arrow_type::source)    source_extractor_type;
    typedef ALE_CONST_MEM_FUN(arrow_type, target_type&,    &arrow_type::target)    target_extractor_type;
    typedef ALE_CONST_MEM_FUN(arrow_type, color_type&,     &arrow_type::color)     color_extractor_type;
    typedef ALE_CONST_MEM_FUN(arrow_type, arrow_type&,     &arrow_type::arrow)     arrow_extractor_type;

    //
    // Slicing
    //
    typedef ::ALE::XSifterDef::Slicer<arrow_type, int, SliceDepth>       slicer_type;
    typedef typename slicer_type::slice_type                             slice_type;

#ifdef ALE_XSIFTER_USE_ARROW_LINKS
    // 
    // Proto-rec contains a pointer to the next proto-rec for each index, 
    // creating a linked list through each index.
    //
    struct proto_rec_type : slicer_type::rec_type {
      typedef typename xsifter_type::arrow_type arrow_type;
      typedef typename arrow_type::source_type  source_type;
      typedef typename arrow_type::target_type  target_type;
      typedef typename arrow_type::color_type   color_type;
      proto_rec_type* cone_next;
      //proto_rec_type* support_next;
      proto_rec_type(const arrow_type& a) : slicer_type::rec_type(a){};
    };
    typedef ::boost::multi_index::member<proto_rec_type, proto_rec_type*, &proto_rec_type::cone_next>  cone_next_extractor_type;
    typedef proto_rec_type                                               rec_type;
#else
    typedef typename slicer_type::rec_type                               rec_type;
#endif

    // Pre-defined SliceSequences
    typedef typename slicer_type::template SliceSequence<source_extractor_type>   SourceSlice;
    typedef typename slicer_type::template SliceSequence<target_extractor_type>   TargetSlice;
    typedef typename slicer_type::template SliceSequence<color_extractor_type>    ColorSlice;
    typedef typename slicer_type::template SliceSequence<arrow_extractor_type>    ArrowSlice;

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
    struct                                   SupportTag{};
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
#ifdef ALE_XSIFTER_USE_ARROW_LINKS
    typedef ALE::XSifterDef::ArrowLinkSequence<xsifter_type, cone_index_type, target_extractor_type, cone_next_extractor_type, target_extractor_type>   BaseSequence;
    typedef ALE::XSifterDef::ArrowLinkSequence<xsifter_type, cone_index_type, target_extractor_type, cone_next_extractor_type, source_extractor_type>   ConeSequence;
    typedef ALE::XSifterDef::ArrowLinkSequence<xsifter_type, cone_index_type, arrow_extractor_type,  cone_next_extractor_type, source_extractor_type>   ColorConeSequence;
#else
    typedef ALE::XSifterDef::ArrowSequence<xsifter_type, cone_index_type, target_extractor_type, target_extractor_type>   BaseSequence;
    typedef ALE::XSifterDef::ArrowSequence<xsifter_type, cone_index_type, target_extractor_type, source_extractor_type>   ConeSequence;
    typedef ALE::XSifterDef::ArrowSequence<xsifter_type, cone_index_type, arrow_extractor_type,  source_extractor_type>   ColorConeSequence;
#endif
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
      static std::pair<typename cone_index_type::iterator, bool> res;
      rec_type r = rec_type(a);
      res = this->_cone_index.insert(r);
      if(res.second) {// successful insertion
#ifdef ALE_XSIFTER_USE_ARROW_LINKS
        typename cone_index_type::iterator iter = res.first;
        // The following two-stage case seems to be necessary because *iter returns 
        // a strange thing that cannot be const_cast directly;  
        // this pattern repeats twice more below for next and prev iterators.
        const rec_type& ci_rec = (*iter);
        rec_type& i_rec = const_cast<rec_type&>(ci_rec);
        if(iter != this->_cone_index.begin()) {// not the first arrow in index
          typename cone_index_type::iterator prev = --iter;
          const rec_type& cp_rec = (*prev);
          rec_type& p_rec = const_cast<rec_type&>(cp_rec);
          // insert i_rec between p_rec and its successor
          i_rec.proto_rec_type::cone_next = p_rec.proto_rec_type::cone_next;
          p_rec.proto_rec_type::cone_next = &i_rec;;
         
        }
        else {
          typename cone_index_type::iterator next = ++iter;
          if(next != _cone_index.end()){
            const rec_type& cn_rec = (*next);
            rec_type& n_rec = const_cast<rec_type&>(cn_rec);
            i_rec.proto_rec_type::cone_next = &n_rec;
          }
          else {
            i_rec.cone_next = NULL;
          }
        }// first arrow in index
#endif
      }
      else {
        ALE::XException e;
        e << "addArrow of " << a << " failed";
        throw e;
      }
    };// addArrow()
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
    slice_type slice(){return this->_slicer.take();};
    //
    template<typename ostream_type>
    void view(ostream_type& os, const char* label = NULL){
      if(label != NULL) {
        os << "Viewing " << label << " XSifter (debug: " << this->debug() << "): " << "\n";
      } 
      else {
        os << "Viewing a XSifter (debug: " << this->debug() << "): " << "\n";
      } 
      os << "Cone index: [[ ";
        for(typename cone_index_type::iterator itor = this->_cone_index.begin(); itor != this->_cone_index.end(); ++itor) {
          os << *itor << " ";
        }
      os << "]]" << "\n";
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
