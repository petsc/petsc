#ifndef included_ALE_LabelSifter_hh
#define included_ALE_LabelSifter_hh

#include <iostream>

#ifndef  included_ALE_hh
#include <sieve/ALE.hh>
#endif

namespace ALE {
  namespace NewSifterDef {
    // Defines the traits of a sequence representing a subset of a multi_index container Index_.
    // A sequence defines output (input in std terminology) iterators for traversing an Index_ object.
    // Upon dereferencing values are extracted from each result record using a ValueExtractor_ object.
    template <typename Index_, typename ValueExtractor_>
    struct IndexSequenceTraits {
      typedef Index_ index_type;
      class iterator_base {
      public:
        // Standard iterator typedefs
        typedef ValueExtractor_                        extractor_type;
        typedef std::input_iterator_tag                iterator_category;
        typedef typename extractor_type::result_type   value_type;
        typedef int                                    difference_type;
        typedef value_type*                            pointer;
        typedef value_type&                            reference;

        // Underlying iterator type
        typedef typename index_type::iterator          itor_type;
      protected:
        // Underlying iterator
        itor_type      _itor;
        // Member extractor
        extractor_type _ex;
      public:
        iterator_base(itor_type itor) {
          this->_itor = itor_type(itor);
        };
        virtual ~iterator_base() {};
        virtual bool              operator==(const iterator_base& iter) const {return this->_itor == iter._itor;};
        virtual bool              operator!=(const iterator_base& iter) const {return this->_itor != iter._itor;};
        // FIX: operator*() should return a const reference, but it won't compile that way, because _ex() returns const value_type
        virtual const value_type  operator*() const {return _ex(*(this->_itor));};
      };// class iterator_base
      class iterator : public iterator_base {
      public:
        // Standard iterator typedefs
        typedef typename iterator_base::iterator_category  iterator_category;
        typedef typename iterator_base::value_type         value_type;
        typedef typename iterator_base::extractor_type     extractor_type;
        typedef typename iterator_base::difference_type    difference_type;
        typedef typename iterator_base::pointer            pointer;
        typedef typename iterator_base::reference          reference;
        // Underlying iterator type
        typedef typename iterator_base::itor_type          itor_type;
      public:
        iterator(const itor_type& itor) : iterator_base(itor) {};
        virtual ~iterator() {};
        //
        virtual iterator   operator++() {++this->_itor; return *this;};
        virtual iterator   operator++(int n) {iterator tmp(this->_itor); ++this->_itor; return tmp;};
      };// class iterator
    }; // struct IndexSequenceTraits

    template <typename Index_, typename ValueExtractor_>
    struct ReversibleIndexSequenceTraits {
      typedef IndexSequenceTraits<Index_, ValueExtractor_> base_traits;
      typedef typename base_traits::iterator_base   iterator_base;
      typedef typename base_traits::iterator        iterator;
      typedef typename base_traits::index_type      index_type;

      // reverse_iterator is the reverse of iterator
      class reverse_iterator : public iterator_base {
      public:
        // Standard iterator typedefs
        typedef typename iterator_base::iterator_category  iterator_category;
        typedef typename iterator_base::value_type         value_type;
        typedef typename iterator_base::extractor_type     extractor_type;
        typedef typename iterator_base::difference_type    difference_type;
        typedef typename iterator_base::pointer            pointer;
        typedef typename iterator_base::reference          reference;
        // Underlying iterator type
        typedef typename iterator_base::itor_type          itor_type;
      public:
        reverse_iterator(const itor_type& itor) : iterator_base(itor) {};
        virtual ~reverse_iterator() {};
        //
        virtual reverse_iterator     operator++() {--this->_itor; return *this;};
        virtual reverse_iterator     operator++(int n) {reverse_iterator tmp(this->_itor); --this->_itor; return tmp;};
      };
    }; // class ReversibleIndexSequenceTraits

    //
    // Arrow & ArrowContainer definitions
    //
    template<typename Source_, typename Target_>
    struct  Arrow { //: public ALE::def::Arrow<Source_, Target_, Color_> {
      typedef Arrow   arrow_type;
      typedef Source_ source_type;
      typedef Target_ target_type;
      source_type source;
      target_type target;
      Arrow(const source_type& s, const target_type& t) : source(s), target(t) {};
      // Flipping
      template <typename OtherSource_, typename OtherTarget_>
      struct rebind {
        typedef Arrow<OtherSource_, OtherTarget_> type;
      };
      struct flip {
        typedef Arrow<target_type, source_type> type;
        type arrow(const arrow_type& a) { return type(a.target, a.source);};
      };

      // Printing
      friend std::ostream& operator<<(std::ostream& os, const Arrow& a) {
        os << a.source << " ----> " << a.target;
        return os;
      }

      // Arrow modifiers
      struct sourceChanger {
        sourceChanger(const source_type& newSource) : _newSource(newSource) {};
        void operator()(arrow_type& a) {a.source = this->_newSource;}
      private:
        source_type _newSource;
      };

      struct targetChanger {
        targetChanger(const target_type& newTarget) : _newTarget(newTarget) {};
        void operator()(arrow_type& a) { a.target = this->_newTarget;}
      private:
        const target_type _newTarget;
      };
    };// struct Arrow


    template<typename Source_, typename Target_>
    struct ArrowContainerTraits {
    public:
      //
      // Encapsulated types
      //
      typedef Arrow<Source_,Target_>           arrow_type;
      typedef typename arrow_type::source_type source_type;
      typedef typename arrow_type::target_type target_type;
      // Index tags
      struct                                   sourceTargetTag{};
      struct                                   targetSourceTag{};

      // Sequence traits and sequence types
      template <typename Index_, typename Key_, typename SubKey_, typename ValueExtractor_>
      class ArrowSequence {
        // ArrowSequence implements ReversibleIndexSequencTraits with Index_ and ValueExtractor_ types.
        // A Key_ object and an optional SubKey_ object are used to extract the index subset.
      public:
        typedef ReversibleIndexSequenceTraits<Index_, ValueExtractor_>  traits;
        //typedef source_type                                             source_type;
        //typedef target_type                                             target_type;
        //typedef arrow_type                                              arrow_type;
        //
        typedef Key_                                                    key_type;
        typedef SubKey_                                                 subkey_type;
      protected:
        typename traits::index_type&                                    _index;
        key_type                                                  key;
        subkey_type                                               subkey;
        bool                                                      useSubkey;
      public:
        // Need to extend the inherited iterators to be able to extract arrow color
        class iterator : public traits::iterator {
        public:
          iterator(const typename traits::iterator::itor_type& itor) : traits::iterator(itor) {};
          virtual const source_type& source() const {return this->_itor->source;};
          virtual const target_type& target() const {return this->_itor->target;};
          virtual const arrow_type&  arrow()  const {return *(this->_itor);};
        };
        class reverse_iterator : public traits::reverse_iterator {
        public:
          reverse_iterator(const typename traits::reverse_iterator::itor_type& itor) : traits::reverse_iterator(itor) {};
          virtual const source_type& source() const {return this->_itor->source;};
          virtual const target_type& target() const {return this->_itor->target;};
          virtual const arrow_type&  arrow()  const {return *(this->_itor);};
        };
      public:
        //
        // Basic ArrowSequence interface
        //
        ArrowSequence(const ArrowSequence& seq) : _index(seq._index), key(seq.key), subkey(seq.subkey), useSubkey(seq.useSubkey) {};
        ArrowSequence(typename traits::index_type& index, const key_type& k) :
          _index(index), key(k), subkey(subkey_type()), useSubkey(0) {};
        ArrowSequence(typename traits::index_type& index, const key_type& k, const subkey_type& kk) :
          _index(index), key(k), subkey(kk), useSubkey(1){};
        virtual ~ArrowSequence() {};

        void setKey(const key_type& key) {this->key = key;};
        void setSubkey(const subkey_type& subkey) {this->subkey = subkey;};
        void setUseSubkey(const bool& useSubkey) {this->useSubkey = useSubkey;};

        virtual bool         empty() {return this->_index.empty();};

        virtual typename traits::index_type::size_type  size()  {
          if (this->useSubkey) {
            return this->_index.count(::boost::make_tuple(this->key,this->subkey));
          } else {
            return this->_index.count(::boost::make_tuple(this->key));
          }
        };

        virtual iterator begin() {
          if (this->useSubkey) {
            return iterator(this->_index.lower_bound(::boost::make_tuple(this->key,this->subkey)));
          } else {
            return iterator(this->_index.lower_bound(::boost::make_tuple(this->key)));
          }
        };

        virtual iterator end() {
          if (this->useSubkey) {
            return iterator(this->_index.upper_bound(::boost::make_tuple(this->key,this->subkey)));
          } else {
            return iterator(this->_index.upper_bound(::boost::make_tuple(this->key)));
          }
        };

        virtual reverse_iterator rbegin() {
          if (this->useSubkey) {
            return reverse_iterator(--this->_index.upper_bound(::boost::make_tuple(this->key,this->subkey)));
          } else {
            return reverse_iterator(--this->_index.upper_bound(::boost::make_tuple(this->key)));
          }
        };

        virtual reverse_iterator rend() {
          if (this->useSubkey) {
            return reverse_iterator(--this->_index.lower_bound(::boost::make_tuple(this->key,this->subkey)));
          } else {
            return reverse_iterator(--this->_index.lower_bound(::boost::make_tuple(this->key)));
          }
        };

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
        }
      };// class ArrowSequence
    };// class ArrowContainerTraits


    // The specialized ArrowContainer types distinguish the cases of unique and multiple colors of arrows on
    // for each (source,target) pair (i.e., a single arrow, or multiple arrows between each pair of points).
    template<typename Source_, typename Target_, typename Alloc_ = ALE_ALLOCATOR<typename ArrowContainerTraits<Source_, Target_>::arrow_type> >
    struct ArrowContainer {
      // Define container's encapsulated types
      typedef ArrowContainerTraits<Source_, Target_> traits;
      // need to def arrow_type locally, since BOOST_MULTI_INDEX_MEMBER barfs when first template parameter starts with 'typename'
      typedef typename traits::arrow_type                                   arrow_type;
      typedef Alloc_ alloc_type;

      // multi-index set type -- arrow set
      typedef ::boost::multi_index::multi_index_container<
        typename traits::arrow_type,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<typename traits::sourceTargetTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type,
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target)
            >
          >,
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<typename traits::targetSourceTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type,
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source)
            >
          >
        >,
        Alloc_
      > set_type;
      // multi-index set of arrow records
      set_type set;

      ArrowContainer() {};
      ArrowContainer(const alloc_type& allocator) {this->set = set_type(typename set_type::ctor_args_list(), allocator);};
    }; // class ArrowContainer
  }; // namespace NewSifterDef

  template<typename Source_, typename Target_, typename Alloc_ = ALE_ALLOCATOR<typename NewSifterDef::ArrowContainer<Source_, Target_>::traits::arrow_type> >
  class LabelSifter { // class Sifter
  public:
    typedef struct {
      typedef LabelSifter<Source_, Target_, Alloc_> graph_type;
      // Encapsulated container types
      typedef NewSifterDef::ArrowContainer<Source_, Target_, Alloc_>                 arrow_container_type;
      // Types associated with records held in containers
      typedef typename arrow_container_type::traits::arrow_type                      arrow_type;
      typedef typename arrow_container_type::traits::source_type                     source_type;
      typedef typename arrow_container_type::traits::target_type                     target_type;
      // Convenient tag names
      typedef typename arrow_container_type::traits::sourceTargetTag                 supportInd;
      typedef typename arrow_container_type::traits::targetSourceTag                 coneInd;
      typedef typename arrow_container_type::traits::sourceTargetTag                 arrowInd;
      //
      // Return types
      //
      typedef typename
      arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,arrowInd>::type, source_type, target_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, source_type, source)>
      arrowSequence;

      // FIX: This is a temp fix to include addArrow into the interface; should probably be pushed up to ArrowSequence
      struct coneSequence : public arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,coneInd>::type, target_type, source_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, source_type, source)> {
      protected:
        graph_type& _graph;
      public:
        typedef typename
          arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,coneInd>::type, target_type, source_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, source_type, source)> base_type;
        // Encapsulated types
        typedef typename base_type::traits traits;
        typedef typename base_type::iterator iterator;
        typedef typename base_type::reverse_iterator reverse_iterator;
        // Basic interface
        coneSequence(const coneSequence& seq) : base_type(seq), _graph(seq._graph) {};
          coneSequence(graph_type& graph, typename traits::index_type& index, const typename base_type::key_type& k) : base_type(index, k), _graph(graph){};
            coneSequence(graph_type& graph, typename traits::index_type& index, const typename base_type::key_type& k, const typename base_type::subkey_type& kk) : base_type(index, k, kk), _graph(graph) {};
              virtual ~coneSequence() {};

        // Extended interface
        void addArrow(const arrow_type& a) {
          // if(a.target != this->key) {
          //               throw ALE::Exception("Arrow target mismatch in a coneSequence");
          //             }
          this->_graph.addArrow(a);
        };
        void addArrow(const source_type& s){
          this->_graph.addArrow(arrow_type(s,this->key));
        };

        virtual bool contains(const source_type& s) {
          // Check whether a given point is in the index
          typename ::boost::multi_index::index<typename LabelSifter::traits::arrow_container_type::set_type,typename LabelSifter::traits::arrowInd>::type& index = ::boost::multi_index::get<typename LabelSifter::traits::arrowInd>(this->_graph._arrows.set);
          return (index.find(::boost::make_tuple(s,this->key)) != index.end());
        };
      };// struct coneSequence

      // FIX: This is a temp fix to include addArrow into the interface; should probably be pushed up to ArrowSequence
      struct supportSequence : public arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,supportInd>::type, source_type, target_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, target_type, target)> {
      protected:
        graph_type& _graph;
      public:
        typedef typename
          arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,supportInd>::type, source_type, target_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, target_type, target)> base_type;
        // Encapsulated types
        typedef typename base_type::traits traits;
        typedef typename base_type::iterator iterator;
        typedef typename base_type::iterator const_iterator;
        typedef typename base_type::reverse_iterator reverse_iterator;
        // Basic interface
        supportSequence(const supportSequence& seq) : base_type(seq), _graph(seq._graph) {};
        supportSequence(graph_type& graph, typename traits::index_type& index, const typename base_type::key_type& k) : base_type(index, k), _graph(graph){};
        supportSequence(graph_type& graph, typename traits::index_type& index, const typename base_type::key_type& k, const typename base_type::subkey_type& kk) : base_type(index, k, kk), _graph(graph) {};
        virtual ~supportSequence() {};

        // FIX: WARNING: (or a HACK?): we flip the arrow on addition here.
        // Fancy interface
        void addArrow(const typename arrow_type::flip::type& af) {
          this->_graph.addArrow(af.target, af.source);
        };
        void addArrow(const target_type& t){
          this->_graph.addArrow(arrow_type(this->key,t));
        };
      };// struct supportSequence

      typedef std::set<source_type, std::less<source_type>, typename Alloc_::template rebind<source_type>::other> coneSet;
      typedef ALE::array<source_type> coneArray;
      typedef std::set<target_type, std::less<target_type>, typename Alloc_::template rebind<source_type>::other> supportSet;
      typedef ALE::array<target_type> supportArray;
    } traits;

    template <typename OtherSource_, typename OtherTarget_>
    struct rebind {
      typedef LabelSifter<OtherSource_, OtherTarget_> type;
    };

    typedef Alloc_                           alloc_type;
    typedef typename traits::source_type     source_type;
    typedef typename traits::target_type     target_type;
    typedef typename traits::coneSequence    coneSequence;
    typedef typename traits::supportSequence supportSequence;
    typedef std::set<int>                    capSequence;
  public:
    // Debug level
    int _debug;
    //protected:
    typename traits::arrow_container_type _arrows;
  protected:
    MPI_Comm    _comm;
    int         _commRank;
    int         _commSize;
    void __init(MPI_Comm comm) {
      static PetscClassId sifterType = -1;
      //const char        *id_name = ALE::getClassName<T>();
      const char        *id_name = "LabelSifter";
      PetscErrorCode     ierr;

      if (sifterType < 0) {
        ierr = PetscClassIdRegister(id_name,&sifterType);CHKERROR(ierr, "Error in MPI_Comm_rank");
      }
      this->_comm = comm;
      ierr = MPI_Comm_rank(this->_comm, &this->_commRank);CHKERROR(ierr, "Error in MPI_Comm_rank");
      ierr = MPI_Comm_size(this->_comm, &this->_commSize);CHKERROR(ierr, "Error in MPI_Comm_rank");
      //ALE::restoreClassName<T>(id_name);
    };
    // We store these sequence objects to avoid creating them each query
    Obj<typename traits::coneSequence> _coneSeq;
    Obj<typename traits::supportSequence> _supportSeq;
  public:
    //
    // Basic interface
    //
    LabelSifter(MPI_Comm comm = PETSC_COMM_SELF, const int& debug = 0) : _debug(debug) {
      __init(comm);
      this->_coneSeq    = new typename traits::coneSequence(*this, ::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set), typename traits::target_type());
      this->_supportSeq = new typename traits::supportSequence(*this, ::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set), typename traits::source_type());
    };
    LabelSifter(MPI_Comm comm, Alloc_& allocator, const int& debug) : _debug(debug), _arrows(allocator) {
      __init(comm);
      this->_coneSeq    = new typename traits::coneSequence(*this, ::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set), typename traits::target_type());
      this->_supportSeq = new typename traits::supportSequence(*this, ::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set), typename traits::source_type());
    };
    virtual ~LabelSifter() {};
    //
    // Query methods
    //
    int         debug()    const {return this->_debug;};
    void        setDebug(const int debug) {this->_debug = debug;};
    MPI_Comm    comm()     const {return this->_comm;};
    int         commSize() const {return this->_commSize;};
    int         commRank() const {return this->_commRank;}

    // FIX: should probably have cone and const_cone etc, since arrows can be modified through an iterator (modifyColor).
    Obj<typename traits::arrowSequence>
    arrows(const typename traits::source_type& s, const typename traits::target_type& t) {
      return typename traits::arrowSequence(::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set), s, t);
    };
    Obj<typename traits::arrowSequence>
    arrows(const typename traits::source_type& s) {
      return typename traits::arrowSequence(::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set), s);
    };
#ifdef SLOW
    Obj<typename traits::coneSequence>
    cone(const typename traits::target_type& p) {
      return typename traits::coneSequence(*this, ::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set), p);
    };
#else
    const Obj<typename traits::coneSequence>&
    cone(const typename traits::target_type& p) {
      this->_coneSeq->setKey(p);
      this->_coneSeq->setUseSubkey(false);
      return this->_coneSeq;
    };
#endif
    template<class InputSequence>
    Obj<typename traits::coneSet>
    cone(const Obj<InputSequence>& points) {
      Obj<typename traits::coneSet> cone = typename traits::coneSet();

      for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
        const Obj<typename traits::coneSequence>& pCone = this->cone(*p_itor);
        cone->insert(pCone->begin(), pCone->end());
      }
      return cone;
    }
    int getConeSize(const typename traits::target_type& p) {
      return this->cone(p)->size();
    };
    template<typename PointCheck>
    bool coneContains(const typename traits::target_type& p, const PointCheck& checker) {
      typename traits::coneSequence cone(*this, ::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set), p);

      for(typename traits::coneSequence::iterator c_iter = cone.begin(); c_iter != cone.end(); ++c_iter) {
        if (checker(*c_iter, p)) return true;
      }
      return false;
    }
    template<typename PointProcess>
    void coneApply(const typename traits::target_type& p, PointProcess& processor) {
      typename traits::coneSequence cone(*this, ::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set), p);

      for(typename traits::coneSequence::iterator c_iter = cone.begin(); c_iter != cone.end(); ++c_iter) {
        processor(*c_iter, p);
      }
    }
#ifdef SLOW
    Obj<typename traits::supportSequence>
    support(const typename traits::source_type& p) {
      return typename traits::supportSequence(*this, ::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set), p);
    };
#else
    const Obj<typename traits::supportSequence>&
    support(const typename traits::source_type& p) {
      this->_supportSeq->setKey(p);
      this->_supportSeq->setUseSubkey(false);
      return this->_supportSeq;
    };
#endif
    template<class InputSequence>
    Obj<typename traits::supportSet>
    support(const Obj<InputSequence>& points){
      Obj<typename traits::supportSet> supp = typename traits::supportSet();
      for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
        const Obj<typename traits::supportSequence>& pSupport = this->support(*p_itor);
        supp->insert(pSupport->begin(), pSupport->end());
      }
      return supp;
    }
    template<typename PointCheck>
    bool supportContains(const typename traits::source_type& p, const PointCheck& checker) {
      typename traits::supportSequence support(*this, ::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set), p);

      for(typename traits::supportSequence::iterator s_iter = support.begin(); s_iter != support.end(); ++s_iter) {
        if (checker(*s_iter, p)) return true;
      }
      return false;
    }
    template<typename PointProcess>
    void supportApply(const typename traits::source_type& p, PointProcess& processor) {
      typename traits::supportSequence support(*this, ::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set), p);

      for(typename traits::supportSequence::iterator s_iter = support.begin(); s_iter != support.end(); ++s_iter) {
        processor(*s_iter, p);
      }
    }

    template<typename ostream_type>
    void view(ostream_type& os, const char* label = NULL, bool rawData = false){
      const int rank = this->commRank();

      if(label != NULL) {
        os << "["<<rank<<"]Viewing LabelSifter '" << label << "':" << std::endl;
      }
      else {
        os << "["<<rank<<"]Viewing a LabelSifter:" << std::endl;
      }
      os << "'raw' arrow set:" << std::endl;
      for(typename traits::arrow_container_type::set_type::iterator ai = _arrows.set.begin(); ai != _arrows.set.end(); ai++) {
        os << *ai << std::endl;
      }
    }
    // A parallel viewer
    #undef __FUNCT__
    #define __FUNCT__ "view"
    PetscErrorCode view(const char* label = NULL, bool raw = false){
      PetscErrorCode ierr;
      ostringstream txt;
      PetscFunctionBegin;
      if(this->_debug) {
        std::cout << "viewing a LabelSifter, comm = " << this->comm() << ", PETSC_COMM_SELF = " << PETSC_COMM_SELF << ", commRank = " << this->commRank() << std::endl;
      }
      if(label != NULL) {
        PetscPrintf(this->comm(), "viewing LabelSifter: '%s'\n", label);
      } else {
        PetscPrintf(this->comm(), "viewing a LabelSifter: \n");
      }
      if(!raw) {
        ostringstream txt;
        if(this->commRank() == 0) {
          txt << "cap --> base:\n";
        }
        if(_arrows.set.empty()) {
          txt << "[" << this->commRank() << "]: empty" << std::endl;
        }
        for(typename traits::arrow_container_type::set_type::iterator ai = _arrows.set.begin(); ai != _arrows.set.end(); ai++) {
          txt << "[" << this->commRank() << "]: " << ai->source << "---->" << ai->target << std::endl;
        }
        ierr = PetscSynchronizedPrintf(this->comm(), txt.str().c_str());CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        ierr = PetscSynchronizedFlush(this->comm()); CHKERROR(ierr, "Error in PetscSynchronizedFlush");
      }
      else { // if(raw)
        ostringstream txt;
        if(this->commRank() == 0) {
          txt << "'raw' arrow set:" << std::endl;
        }
        for(typename traits::arrow_container_type::set_type::iterator ai = _arrows.set.begin(); ai != _arrows.set.end(); ai++)
        {
          typename traits::arrow_type arr = *ai;
          txt << "[" << this->commRank() << "]: " << arr << std::endl;
        }
        ierr = PetscSynchronizedPrintf(this->comm(), txt.str().c_str());CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        ierr = PetscSynchronizedFlush(this->comm()); CHKERROR(ierr, "Error in PetscSynchronizedFlush");
      }// if(raw)

      PetscFunctionReturn(0);
    }
  public:
    //
    // Lattice queries
    //
    template<class targetInputSequence>
    Obj<typename traits::coneSequence> meet(const Obj<targetInputSequence>& targets);
    // unimplemented
    template<class sourceInputSequence>
    Obj<typename traits::coneSequence> join(const Obj<sourceInputSequence>& sources);
  public:
    //
    // Structural manipulation
    //
    void clear() {
      this->_arrows.set.clear();
    };
    // This is necessary to work with Completion right now
    virtual void addArrow(const typename traits::source_type& p, const typename traits::target_type& q, const int dummy) {
      this->addArrow(p, q);
    };
    virtual void addArrow(const typename traits::source_type& p, const typename traits::target_type& q) {
      this->addArrow(typename traits::arrow_type(p, q));
      //std::cout << "Added " << arrow_type(p, q);
    };
    virtual void addArrow(const typename traits::arrow_type& a) {
      this->_arrows.set.insert(a);
    };
    virtual void removeArrow(const typename traits::arrow_type& a) {
      // First, produce an arrow sequence for the given source, target combination.
      typename traits::arrowSequence::traits::index_type& arrowIndex =
        ::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set);
      typename traits::arrowSequence::traits::index_type::iterator i,ii,j;
      i = arrowIndex.lower_bound(::boost::make_tuple(a.source,a.target));
      ii = arrowIndex.upper_bound(::boost::make_tuple(a.source, a.target));
      if (this->_debug) {
        std::cout << "removeArrow: attempting to remove arrow:" << a << std::endl;
        std::cout << "removeArrow: candidate arrows are:" << std::endl;
      }
      for(j = i; j != ii; j++) {
        if (this->_debug) {
          std::cout << " " << *j;
        }
        // Find the arrow of right color and remove it
        if (this->_debug) {
          std::cout << std::endl << "removeArrow: found:" << *j << std::endl;
        }
        arrowIndex.erase(j);
        break;
      }
    };

    void addCone(const typename traits::source_type& source, const typename traits::target_type& target){
      this->addArrow(source, target);
    };
    template<class sourceInputSequence>
    void
    addCone(const Obj<sourceInputSequence>& sources, const typename traits::target_type& target){
      if (this->_debug > 1) {std::cout << "Adding a cone " << std::endl;}
      for(typename sourceInputSequence::iterator iter = sources->begin(); iter != sources->end(); ++iter) {
        if (this->_debug > 1) {std::cout << "Adding arrow from " << *iter << " to " << target << std::endl;}
        this->addArrow(*iter, target);
      }
    }
    void clearCone(const typename traits::target_type& t) {
      // Use the cone sequence types to clear the cone
      typename traits::coneSequence::traits::index_type& coneIndex =
        ::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set);
      typename traits::coneSequence::traits::index_type::iterator i, ii, j;
      if (this->_debug > 20) {
        std::cout << "clearCone: removing cone over " << t;
        std::cout << std::endl;
        const Obj<typename traits::coneSequence>& cone = this->cone(t);
        std::cout << "[";
        for(typename traits::coneSequence::iterator ci = cone->begin(); ci != cone->end(); ci++) {
          std::cout << "  " << ci.arrow();
        }
        std::cout << "]" << std::endl;
      }
      i = coneIndex.lower_bound(::boost::make_tuple(t));
      ii = coneIndex.upper_bound(::boost::make_tuple(t));
      coneIndex.erase(i,ii);
    }// clearCone()

    void clearSupport(const typename traits::source_type& s) {
      // Use the cone sequence types to clear the cone
      typename
        traits::supportSequence::traits::index_type& suppIndex = ::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set);
      typename traits::supportSequence::traits::index_type::iterator i, ii, j;
      i = suppIndex.lower_bound(::boost::make_tuple(s));
      ii = suppIndex.upper_bound(::boost::make_tuple(s));
      suppIndex.erase(i,ii);
    }
    void setCone(const typename traits::source_type& source, const typename traits::target_type& target){
      this->clearCone(target); this->addCone(source, target);
    }
    template<class sourceInputSequence>
    void setCone(const Obj<sourceInputSequence>& sources, const typename traits::target_type& target) {
      this->clearCone(target); this->addCone(sources, target);
    }
    template<class targetInputSequence>
    void addSupport(const typename traits::source_type& source, const Obj<targetInputSequence >& targets) {
      if (this->_debug > 1) {std::cout << "Adding a support " << std::endl;}
      for(typename targetInputSequence::iterator iter = targets->begin(); iter != targets->end(); ++iter) {
        if (this->_debug > 1) {std::cout << "Adding arrow from " << source << " to " << *iter << std::endl;}
        this->addArrow(source, *iter);
      }
    }
    template<typename Sifter_, typename AnotherSifter_>
    void add(const Obj<Sifter_>& cbg, const Obj<AnotherSifter_>& baseRestriction = NULL) {
      typename ::boost::multi_index::index<typename Sifter_::traits::arrow_container_type::set_type, typename Sifter_::traits::arrowInd>::type& aInd = ::boost::multi_index::get<typename Sifter_::traits::arrowInd>(cbg->_arrows.set);
      bool baseRestrict = !baseRestriction.isNull();

      for(typename ::boost::multi_index::index<typename Sifter_::traits::arrow_container_type::set_type, typename Sifter_::traits::arrowInd>::type::iterator a_iter = aInd.begin(); a_iter != aInd.end(); ++a_iter) {
        if (baseRestrict) {
          if (!baseRestriction->getSupportSize(a_iter->target) && !baseRestriction->getConeSize(a_iter->target)) continue;
        }
        this->addArrow(*a_iter);
      }
    }
    template<typename Sifter_, typename AnotherSifter_, typename Renumbering_>
    void add(const Obj<Sifter_>& cbg, const Obj<AnotherSifter_>& baseRestriction, Renumbering_& renumbering) {
      typename ::boost::multi_index::index<typename Sifter_::traits::arrow_container_type::set_type, typename Sifter_::traits::arrowInd>::type& aInd = ::boost::multi_index::get<typename Sifter_::traits::arrowInd>(cbg->_arrows.set);

      for(typename ::boost::multi_index::index<typename Sifter_::traits::arrow_container_type::set_type, typename Sifter_::traits::arrowInd>::type::iterator a_iter = aInd.begin(); a_iter != aInd.end(); ++a_iter) {
        if (renumbering.find(a_iter->target) == renumbering.end()) continue;
        target_type target = renumbering[a_iter->target];

        if (!baseRestriction->getSupportSize(target) && !baseRestriction->getConeSize(target)) continue;
        this->addArrow(a_iter->source, target);
      }
    }
    template<typename Labeling, typename AnotherSifter>
    void relabel(Labeling& relabeling, AnotherSifter& newLabel) {
      typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type, typename traits::arrowInd>::type& aInd = ::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set);

      for(typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type, typename traits::arrowInd>::type::iterator a_iter = aInd.begin(); a_iter != aInd.end(); ++a_iter) {
	const typename traits::target_type newTarget = relabeling.restrictPoint(a_iter->target)[0];

        newLabel.addArrow(a_iter->source, newTarget);
      }
    }

    int size() const {return _arrows.set.size();};
    int getCapSize() const {
      std::set<source_type> cap;
      for(typename traits::arrow_container_type::set_type::iterator a_iter = _arrows.set.begin(); a_iter != _arrows.set.end(); ++a_iter) {
        cap.insert(a_iter->source);
      }
      return cap.size();
    };
    capSequence cap() const {
      std::set<source_type> cap;
      for(typename traits::arrow_container_type::set_type::iterator a_iter = _arrows.set.begin(); a_iter != _arrows.set.end(); ++a_iter) {
        cap.insert(a_iter->source);
      }
      return cap;
    };
    int getBaseSize() const {
      std::set<target_type> base;
      for(typename traits::arrow_container_type::set_type::iterator a_iter = _arrows.set.begin(); a_iter != _arrows.set.end(); ++a_iter) {
        base.insert(a_iter->target);
      }
      return base.size();
    };
  public: // Compatibility with fixed storage variants
    typedef Interval<target_type> chart_type;
    chart_type& getChart() {static chart_type chart(0, 0); return chart;}
    template<typename chart_type>
    void setChart(const chart_type& chart) {}
    void setConeSize(target_type p, int s) {}
    void setSupportSize(source_type p, int s) {}
    void allocate() {}
    void recalculateLabel() {}
  }; // class LabelSifter

  class LabelSifterSerializer {
  public:
    template<typename LabelSifter>
    static void writeLabel(std::ofstream& fs, LabelSifter& label) {
      if (label.commRank() == 0) {
        // Write local
        fs << label._arrows.set.size() << std::endl;
        for(typename LabelSifter::traits::arrow_container_type::set_type::iterator ai = label._arrows.set.begin(); ai != label._arrows.set.end(); ai++) {
          fs << ai->source << " " << ai->target << std::endl;
        }
        // Receive and write remote
        for(int p = 1; p < label.commSize(); ++p) {
          PetscInt       size;
          PetscInt      *arrows;
          MPI_Status     status;
          PetscErrorCode ierr;

          ierr = MPI_Recv(&size, 1, MPIU_INT, p, 1, label.comm(), &status);CHKERRXX(ierr);
          fs << size << std::endl;
          ierr = PetscMalloc(size*2 * sizeof(PetscInt), &arrows);CHKERRXX(ierr);
          ierr = MPI_Recv(arrows, size*2, MPIU_INT, p, 1, label.comm(), &status);CHKERRXX(ierr);
          for(PetscInt a = 0; a < size; ++a) {
            fs << arrows[a*2+0] << " " << arrows[a*2+1] << std::endl;
          }
          ierr = PetscFree(arrows);CHKERRXX(ierr);
        }
      } else {
        // Send remote
        PetscInt       size = label._arrows.set.size();
        PetscInt       a    = 0;
        PetscInt      *arrows;
        PetscErrorCode ierr;

        ierr = MPI_Send(&size, 1, MPIU_INT, 0, 1, label.comm());CHKERRXX(ierr);
        // There is no nice way to make a generic MPI type here. Really sucky
        ierr = PetscMalloc(size*2 * sizeof(PetscInt), &arrows);CHKERRXX(ierr);
        for(typename LabelSifter::traits::arrow_container_type::set_type::iterator ai = label._arrows.set.begin(); ai != label._arrows.set.end(); ai++, ++a) {
          arrows[a*2+0] = ai->source;
          arrows[a*2+1] = ai->target;
        }
        ierr = MPI_Send(arrows, size*2, MPIU_INT, 0, 1, label.comm());CHKERRXX(ierr);
        ierr = PetscFree(arrows);CHKERRXX(ierr);
      }
    }
    template<typename LabelSifter>
    static void loadLabel(std::ifstream& fs, LabelSifter& label) {
      if (label.commRank() == 0) {
        // Load local
        size_t numArrows;

        fs >> numArrows;
        for(size_t a = 0; a < numArrows; ++a) {
          typename LabelSifter::traits::arrow_type::source_type source;
          typename LabelSifter::traits::arrow_type::target_type target;

          fs >> source;
          fs >> target;
          label.addArrow(typename LabelSifter::traits::arrow_type(source, target));
        }
        // Load and send remote
        for(int p = 1; p < label.commSize(); ++p) {
          PetscInt       size;
          PetscInt      *arrows;
          PetscErrorCode ierr;

          fs >> size;
          ierr = MPI_Send(&size, 1, MPIU_INT, p, 1, label.comm());CHKERRXX(ierr);
          ierr = PetscMalloc(size*2 * sizeof(PetscInt), &arrows);CHKERRXX(ierr);
          for(PetscInt a = 0; a < size; ++a) {
            fs >> arrows[a*2+0];
            fs >> arrows[a*2+1];
          }
          ierr = MPI_Send(arrows, size*2, MPIU_INT, p, 1, label.comm());CHKERRXX(ierr);
          ierr = PetscFree(arrows);CHKERRXX(ierr);
        }
      } else {
        // Load remote
        PetscInt       size;
        PetscInt      *arrows;
        MPI_Status     status;
        PetscErrorCode ierr;

        ierr = MPI_Recv(&size, 1, MPIU_INT, 0, 1, label.comm(), &status);CHKERRXX(ierr);
        ierr = PetscMalloc(size*2 * sizeof(PetscInt), &arrows);CHKERRXX(ierr);
        ierr = MPI_Recv(arrows, size*2, MPIU_INT, 0, 1, label.comm(), &status);CHKERRXX(ierr);
        for(PetscInt a = 0; a < size; ++a) {
          label.addArrow(typename LabelSifter::traits::arrow_type(arrows[a*2+0], arrows[a*2+1]));
        }
        ierr = PetscFree(arrows);CHKERRXX(ierr);
      }
    }
  };
} // namespace ALE

#endif // ifdef included_ALE_LabelSifter_hh
