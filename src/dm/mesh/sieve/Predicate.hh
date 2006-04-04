#ifndef included_ALE_Predicate_hh
#define included_ALE_Predicate_hh

#ifndef  included_ALE_containers_hh
#include <ALE_containers.hh>
#endif

namespace ALE {

  // 
  // Various Window concepts and definitions
  // 
  namespace PredicateDef {
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
      const static predicate_type zero_predicate = 0;
      const static predicate_type max_predicate  = UINT_MAX;
      //
      predicate_type& inc(predicate_type& p) {return(++p);};
    };
    //
    template<>
    struct PredicateTraits<char> {
      typedef char predicate_type;
      //
      const static predicate_type zero_predicate = 0;
      const static predicate_type max_predicate = CHAR_MAX;
      predicate_type& inc(predicate_type& p) {return (++p);};
    };
    //
    template <typename Predicate_, typename PredicateTraits_=PredicateTraits<Predicate_> >
    struct PredicateRec {
      typedef  PredicateRec<Predicate_, PredicateTraits_> rec_type;
      typedef Predicate_                                  predicate_type;
      typedef PredicateTraits_                            predicate_traits;
      //
      predicate_type           predicate;
      struct PredicateAdjuster {
        PredicateAdjuster(rec_type& r, const predicate_type& newPredicate = predicate_traits::zero_predicate) : _newPredicate(newPredicate) {};
        void operator()(& r) {r.predicate = this->_newPredicate;};
      private:
        predicate_type _newPredicate; // assume predicate is cheap to copy,no more expensive than a pointer -- else ref would be used
      };      
    };
  };// namespace PredicateDef


  template <typename PredicateSet_>
  struct WindowContainer {
  public:
    //
    // Encapsulated types
    typedef PredicateSet_                       predicate_set_type;
    typedef typename set_type::value_type       predicate_rec_type;
    typedef typename rec_type::predicate_type   predicate_type;
    typedef typename rec_type::predicate_traits predicate_traits;
    //
    // Window
    template<typename PredicateIndex_>
    struct Window {
    public:
      //
      // Encapsulated types
      typedef WindowContainer  container_type;
      typedef PredicateIndex_  index_type;
    protected:
      container_type&       _container;
      index_type&           _index;
      const predicate_type  _low, _high;
      void __clear(const predicate_type& high, const predicate_type& low) {
        if( (high >= low) && (high > predicate_traits::zero_predicate)) {
          // Clear out all predicates with values between low and high (inclusive).
          // We must be careful, since we are modifying the index we are traversing.
          // The good news is that the modification only lowers the rank of the modified element's predicate (sends to zero_predicate),
          // and we are only interested in the elements that are greater than zero_predicate.
          predicate_index_type mod_itor;
          typename rec_type::PredicateAdjuster zeroOut;
          // We traverse the interval [low, high].
          for(index_type::iterator itor = this->_index.lower_bound(low); itor != this->_index.upper_bound(high);/*++itor inside loop*/){
            // We advance storing the current itor in mod_itor
            // The reason is that mod_itor will be modified and its position will change
            mod_itor = itor; ++itor;
            // Now we modify mod_itor with the default PredicateAdjuster, which will make the predicate zero.
            this->_index.modify(mod_itor, zeroOut);
          }
        }
      };
    public:
      //
      // Basic interface
      Window(container_type& container, const index_type& index, const predicate_type& high = predicate_traits::zero_predicate, const predicate_type& low = predicate_traits::zero_predicate, bool clear = false) : _container(container), _index(index), _high(high), _low(low)      {
        if(clear) {
          // Upon entering Window must clear out the predicate values.
          this->__clear(this->_high, this->_low);
        }
      };
      Window(const Window& win) : _container(win._container), _index(win._win), _high(win._high), _low(win._low) {};
      ~Window() {};
      //
      // Extended interface
      virtual typename index_type::size_type size()  {
        index_type::size_type sz = 0;
        for(predicate_type p = this->_low; p != this->_high; p++) {
            sz += this->_index.count(::boost::multi_index::make_tuple(p));
        }
        return sz;
      };
      virtual typename index_type::iterator  begin() {
        return typename index_type::iterator(this->_index.lower_bound(::boost::make_tuple(this->_low)));
      };
      virtual typename index_type::iterator  end()   {
        return typename index_type::iterator(this->_index.upper_bound(::boost::make_tuple(this->_high)));
      };
      template<typename ostream_type>
      void view(ostream_type& os, const char* label = NULL){
        if(label != NULL) {
          os << "Viewing window" << label << " :" << std::endl;
        } 
        else {
          os << "Viewing a window:" << std::endl;
        }
        os << "[";
        for(typename index_type::iterator i = this->begin(); i != this->end(); i++) {
          os << " " << *i;
        }
        os << " ]" << std::endl;
      };
    };// class WindowContainer::Window
  protected:
    predicate_set_type& _set;
    bool _windowOpen;
  public:
    //
    // Basic interface
    WindowContainer(const predicate_set_type& set) : _set(set), _windowOpen(false) {};
    ~WindowContainer() {
      if(this->_windowOpen) {
        throw ALE::Exception("Destructor attempted on WindowContainer with an open window");
      }
    };
    //
    // Extended interface
    void openWindow() {
      if(this->_windowOpen) {
        throw ALE::Exception("openWindow attempted on a WindowContainer with an open window");
      }
    };
    void closeWindow() {
      if(!this->_windowOpen) {
        throw ALE::Exception("closeWindow attempted on a WindowContainer without an open window");
      }
    };
  };// class WindowContainer



  namespace Experimental {
    namespace SifterDef { // namespace SifterDef
      // 
      // Various ArrowContainer definitions
      // 

      // Index tags
      struct SourceColorTag{};
      struct TargetColorTag{};
      struct sourceTargetTag{}; 
      
      // Arrow record 'concept' -- conceptual structure names the fields expected in such an ArrowRec
      template <typename Arrow_, typename Predicate_>
      struct ArrowRec : public ALE::PredicateDef::PredicateRec<Predicate_> {
        typedef PredicateRec<Predicate_>            rec_type;
        typedef typename rec_type::predicate_type   predicate_type
        typedef typename rec_type::predicate_traits predicate_traits;
        //
        typedef Arrow_                   arrow_type;
        typedef  arrow_type::source_type source_type;
        typedef  arrow_type::target_type target_type;
        typedef  arrow_type::color_type  color_type;
        //
      protected:
        arrow_type         arrow;
      public:
        source_type source() {return arrow.source;}
        target_type target() {return arrow.target;}
        color_type color()   {return arrow.color;}
      };
      
      template<typename ArrowRecSet_>
      struct ArrowContainer : public WindowContainer<ArrowRecSet_> {
      public:
        //
        // Encapsulated types
        typedef WindowContainer<ArrowRecSet_>       window_container_type;
        //
        typedef ArrowRecSet_                        rec_set_type;    // must have tags as in ArrowContainerDef & PredicateContainerDef
        typedef typename rec_set_type::value_type   rec_type;        // must (conceptually) extend ArrowContainerDef::ArrowRec
        typedef typename rec_type::arrow_type       arrow_type;      // arrow_type must have 'source','target','color' fields
        typedef typename rec_type::predicate_type   predicate_type;
        typedef typename rec_type::predicate_traits predicate_traits;
        //
        typedef typename arrow_type::source_type    source_type;
        typedef typename arrow_type::target_type    target_type;
        typedef typename arrow_type::color_type     color_type;
        //
        template <typename Index_, typename Key_, typename SubKey_>
        class ArrowSequence : public window_container_type::Window<Index_> {
          // ArrowSequence extends the Window sequence and constraints the predicate values to _high==_low
        public:
          //
          // Encapsulated types
          typedef window_container_type::Window           window_type;
          typedef typename base_sequence_type::index_type index_type;
          typedef Key_                                    key_type;
          typedef SubKey_                                 subkey_type;
          //
          // Need to extend the inherited iterator to be able to extract arrow attributes
          friend class iterator {
          protected:  
            typedef ArrowSequence sequence_type;
            sequence_type& _sequence;
          public:
            iterator(sequence_type& sequence, const typename index_type::iterator::itor_type& itor) : 
              typename index_type::iterator(itor), _sequence(sequence) {};
            virtual const source_type& source()  const {return this->_itor->arrow.source;};
            virtual const color_type&  color ()  const {return this->_itor->arrow.color;};
            virtual const target_type& target()  const {return this->_itor->arrow.target;};
            virtual const arrow_type&  arrow ()  const {return *(this->_itor->arrow);};
            //
            void            setPredicate(const predicate_type& p)      {
              this->_sequence._index.modify(i, typename rec_type::PredicateAdjuster(p));
            };
            predicate_type  getPredicate()                       const {
              return this->_itor->predicate;
            };
          };// class iterator
        protected:
          const key_type       key;
          const subkey_type    subkey;
          const bool           useSubkey;
          BasicArrowContainer& _container;
        public:
          //
          // Basic interface
          ArrowSequence(const ArrowSequence& seq) : 
            window_type(seq), _container(seq._container), key(seq.key), subkey(seq.subkey), useSubkey(seq.useSubkey) {};
          ArrowSequence(ArrowContainer& container, index_type& index, const key_type& k) : 
            window_type(container, index), key(k), subkey(subkey_type()), useSubkey(0) {};
          ArrowSequence(ArrowContainer& container, index_type& index, const key_type& k, const subkey_type& kk) : 
            window_type(container, index), key(k), subkey(kk), useSubkey(1){};
          virtual ~ArrowSequence() {};
          //
          // Extended interface
          virtual typename index_type::size_type  size()  {
            if (this->useSubkey) {
              return this->_index.count(::boost::make_tuple(this->_mask,this->key,this->subkey));
            } else {
              return this->_index.count(::boost::make_tuple(this->mask,this->key));
            }
          };
          virtual iterator begin() {
            if (this->useSubkey) {
              return iterator(this->_index.lower_bound(::boost::make_tuple(this->_mask,this->key,this->subkey)));
            } else {
              return iterator(this->_index.lower_bound(::boost::make_tuple(this->_mask,this->key)));
            }
          };
          virtual iterator end() {
            if (this->useSubkey) {
              return iterator(this->_index.upper_bound(::boost::make_tuple(this->_mask,this->key,this->subkey)));
            } else {
              return iterator(this->_index.upper_bound(::boost::make_tuple(this->_mask,this->key)));
            }
          };
          template<typename ostream_type>
          void view(ostream_type& os, const bool& useColor = false, const char* label = NULL){
            if(label != NULL) {
              os << "Viewing " << label << " sequence with mask " << this->_mask <<" :" << std::endl;
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
        // CONTINUE: inherit also from SifterContainers::IndexSequence (only Traits exist as of now).
        template <typename Index_, typename Key_, typename SubKey_, typename ValueExtractor_>
        class ArrowWindowSequence : public window_container_type::Window<Index_> {
          // ArrowWindowSequence extends the Window sequence and constraints the predicate values to _high==_low
        public:
          //
          // Encapsulated types
          typedef window_container_type::Window              window_type;
          typedef typename window_sequence_type::index_type  index_type;
          typedef Key_                                       key_type;
          typedef SubKey_                                    subkey_type;
          typedef ValueExtractor_                            value_extractor_type;
          typedef typename value_extractor_type::result_type value_type;
          //
          // Need to extend the inherited iterator to be able to extract arrow attributes
          friend class iterator {
          protected:  
            typedef ArrowWindowSequence sequence_type;
            sequence_type& _sequence;
          public:
            iterator(sequence_type& sequence, const typename index_type::iterator::itor_type& itor) : 
              typename index_type::iterator(itor), _sequence(sequence) {};
            virtual const source_type& source()  const {return this->_itor->arrow.source;};
            virtual const color_type&  color ()  const {return this->_itor->arrow.color;};
            virtual const target_type& target()  const {return this->_itor->arrow.target;};
            virtual const arrow_type&  arrow ()  const {return *(this->_itor->arrow);};
            //
            void            setPredicate(const predicate_type& p)      {
              this->_sequence._index.modify(i, typename rec_type::PredicateAdjuster(p));
            };
            predicate_type  getPredicate()                       const {
              return this->_itor->predicate;
            };
          };// class iterator
        protected:
          const key_type       key;
          const subkey_type    subkey;
          const bool           useSubkey;
          BasicArrowContainer& _container;
        public:
          //
          // Basic interface
          ArrowSequence(const ArrowSequence& seq) : 
            window_type(seq), _container(seq._container), key(seq.key), subkey(seq.subkey), useSubkey(seq.useSubkey) {};
          ArrowSequence(ArrowContainer& container, index_type& index, const key_type& k) : 
            window_type(container, index), key(k), subkey(subkey_type()), useSubkey(0) {};
          ArrowSequence(ArrowContainer& container, index_type& index, const key_type& k, const subkey_type& kk) : 
            window_type(container, index), key(k), subkey(kk), useSubkey(1){};
          virtual ~ArrowSequence() {};
          //
          // Extended interface
          virtual typename index_type::size_type  size()  {
            if (this->useSubkey) {
              return this->_index.count(::boost::make_tuple(this->_mask,this->key,this->subkey));
            } else {
              return this->_index.count(::boost::make_tuple(this->mask,this->key));
            }
          };
          virtual iterator begin() {
            if (this->useSubkey) {
              return iterator(this->_index.lower_bound(::boost::make_tuple(this->_mask,this->key,this->subkey)));
            } else {
              return iterator(this->_index.lower_bound(::boost::make_tuple(this->_mask,this->key)));
            }
          };
          virtual iterator end() {
            if (this->useSubkey) {
              return iterator(this->_index.upper_bound(::boost::make_tuple(this->_mask,this->key,this->subkey)));
            } else {
              return iterator(this->_index.upper_bound(::boost::make_tuple(this->_mask,this->key)));
            }
          };
          template<typename ostream_type>
          void view(ostream_type& os, const bool& useColor = false, const char* label = NULL){
            if(label != NULL) {
              os << "Viewing " << label << " sequence with mask " << this->_mask <<" :" << std::endl;
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
      protected:
        arrow_set_type _set;
      public:
        //
        // Basic interface
        ArrowContainer(const ArrowContainer& container) : _arrow_set(container._set) {};      
        ~ArrowContainer(){};
      };// class ArrowContainer
  };// namespace SifterDef 
    

    // multi-index set type -- arrow set
    class  UnicolorArrowSet: public 
                             <::boost::multi_index::multi_index_container<
                               ArrowRec_,
                               ::boost::multi_index::indexed_by<
                                 ::boost::multi_index::ordered_unique<
                                   ::boost::multi_index::tag<sourceTargetTag>,
                                   ::boost::multi_index::composite_key<
                                     ArrowRec<Arrow_>, 
                                     const_mem_fun<ArrowRec_, typename ArrowRec_::source_type, &source>, 
                                     const_mem_fun<ArrowRec_, typename ArrowRec_::target_type, &target>
                                   >
                                 >,
                                 ::boost::multi_index::ordered_non_unique<
                                   ::boost::multi_index::tag<sourceColorTag>,
                                   ::boost::multi_index::composite_key<
                                     ArrowRec_, 
                                     const_mem_fun<ArrowRec_, ArrowRec_::source_type, &source>, 
                                     const_mem_fun<ArrowRec_, ArrowRec_::color_type,  &color>
                                   >
                                 >,
                                 ::boost::multi_index::ordered_non_unique<
                                   ::boost::multi_index::tag<targetColorTag>,
                                   ::boost::multi_index::composite_key<
                                     typename traits::arrow_type, 
                                     const_mem_fun<ArrowRec_, ArrowRec_::target_type, &target>, 
                                     const_mem_fun<ArrowRec_, ArrowRec_::color_type,  &color>
                                   >
                                 >
                               >,
                               ALE_ALLOCATOR<typename traits::arrow_type>
                             > 
                            > 
    {
    public:
      // 
      // Encapsulated types
    };
    
  }; // namespace Experimental
}; // namespace ALE
