#ifndef included_ALE_Sifter_hh
#define included_ALE_Sifter_hh

/*
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>
*/
#include <iostream>

// ALE extensions

#ifndef  included_ALE_hh
#include <sieve/ALE.hh>
#endif

extern PetscErrorCode PetscObjectDestroy_PetscObject(PetscObject*);

namespace ALE {

  namespace SifterDef {
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
    // Rec & RecContainer definitions.
    // Rec is intended to denote a graph point record.
    //
    template <typename Point_>
    struct Rec {
      typedef Point_ point_type;
      template<typename OtherPoint_>
      struct rebind {
        typedef Rec<OtherPoint_> type;
      };
      point_type     point;
      int            degree;
      // Basic interface
      Rec() : degree(0){};
      Rec(const Rec& r) : point(r.point), degree(r.degree) {}
      //Rec(const point_type& p) : point(p), degree(0) {};
      Rec(const point_type& p, const int d) : point(p), degree(d) {};
      // Printing
      friend std::ostream& operator<<(std::ostream& os, const Rec& p) {
        os << "<" << p.point << ", "<< p.degree << ">";
        return os;
      };

      struct degreeAdjuster {
        degreeAdjuster(int newDegree) : _newDegree(newDegree) {};
        void operator()(Rec& r) { r.degree = this->_newDegree; }
      private:
        int _newDegree;
      };// degreeAdjuster()

    };// class Rec

    template <typename Point_, typename Rec_>
    struct RecContainerTraits {
      typedef Rec_ rec_type;
      // Index tags
      struct pointTag{};
      // Rec set definition
      typedef ::boost::multi_index::multi_index_container<
        rec_type,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<pointTag>, BOOST_MULTI_INDEX_MEMBER(rec_type, typename rec_type::point_type, point)
          >
        >,
        ALE_ALLOCATOR<rec_type>
      > set_type;
      //
      // Return types
      //

     class PointSequence {
     public:
        typedef IndexSequenceTraits<typename ::boost::multi_index::index<set_type, pointTag>::type,
                                    BOOST_MULTI_INDEX_MEMBER(rec_type, typename rec_type::point_type,point)>
        traits;
      protected:
        const typename traits::index_type& _index;
      public:

       // Need to extend the inherited iterator to be able to extract the degree
       class iterator : public traits::iterator {
       public:
         iterator(const typename traits::iterator::itor_type& itor) : traits::iterator(itor) {};
         virtual const int& degree() const {return this->_itor->degree;};
       };

       PointSequence(const PointSequence& seq)            : _index(seq._index) {};
       PointSequence(typename traits::index_type& index) : _index(index)     {};
       virtual ~PointSequence(){};

       virtual bool empty(){return this->_index.empty();};

       virtual typename traits::index_type::size_type size() {return this->_index.size();};

       virtual iterator begin() {
         // Retrieve the beginning iterator of the index
         return iterator(this->_index.begin());
       };
       virtual iterator end() {
         // Retrieve the ending iterator of the index
         // Since the elements in this index are ordered by degree, this amounts to the end() of the index.
         return iterator(this->_index.end());
       };
       virtual bool contains(const typename rec_type::point_type& p) {
         // Check whether a given point is in the index
         return (this->_index.find(p) != this->_index.end());
       }
     }; // class PointSequence
    };// struct RecContainerTraits


    template <typename Point_, typename Rec_>
    struct RecContainer {
      typedef RecContainerTraits<Point_, Rec_> traits;
      typedef typename traits::set_type set_type;
      template <typename OtherPoint_, typename OtherRec_>
      struct rebind {
        typedef RecContainer<OtherPoint_, OtherRec_> type;
      };
      set_type set;
      //
      void removePoint(const typename traits::rec_type::point_type& p) {
        /*typename ::boost::multi_index::index<set_type, typename traits::pointTag>::type& index =
          ::boost::multi_index::get<typename traits::pointTag>(this->set);
        typename ::boost::multi_index::index<set_type, typename traits::pointTag>::type::iterator i = index.find(p);
        if (i != index.end()) { // Point exists
          i = index.erase(i);
        }*/
        this->set.erase(p);
      };
      //
      void adjustDegree(const typename traits::rec_type::point_type& p, int delta) {
        typename ::boost::multi_index::index<set_type, typename traits::pointTag>::type& index =
          ::boost::multi_index::get<typename traits::pointTag>(this->set);
        typename ::boost::multi_index::index<set_type, typename traits::pointTag>::type::iterator i = index.find(p);
        if (i == index.end()) { // No such point exists
          if(delta < 0) { // Cannot decrease degree of a non-existent point
            ostringstream err;
            err << "ERROR: adjustDegree: Non-existent point " << p;
            std::cout << err << std::endl;
            throw(Exception(err.str().c_str()));
          }
          else { // We CAN INCREASE the degree of a non-existent point: simply insert a new element with degree == delta
            std::pair<typename ::boost::multi_index::index<set_type, typename traits::pointTag>::type::iterator, bool> ii;
            typename traits::rec_type r(p,delta);
            ii = index.insert(r);
            if(ii.second == false) {
              ostringstream err;
              err << "ERROR: adjustDegree: Failed to insert a rec " << r;
              std::cout << err << std::endl;
              throw(Exception(err.str().c_str()));
            }
          }
        }
        else { // Point exists, so we try to modify its degree
          // If the adjustment is zero, there is nothing to do, otherwise ...
          if(delta != 0) {
            int newDegree = i->degree + delta;
            if(newDegree < 0) {
              ostringstream ss;
              ss << "adjustDegree: Adjustment of " << *i << " by " << delta << " would result in negative degree: " << newDegree;
              throw Exception(ss.str().c_str());
            }
            index.modify(i, typename traits::rec_type::degreeAdjuster(newDegree));
          }
        }
      }; // adjustDegree()
    }; // struct RecContainer

    //
    // Arrow & ArrowContainer definitions
    //
    template<typename Source_, typename Target_, typename Color_>
    struct  Arrow { //: public ALE::def::Arrow<Source_, Target_, Color_> {
      typedef Arrow   arrow_type;
      typedef Source_ source_type;
      typedef Target_ target_type;
      typedef Color_  color_type;
      source_type source;
      target_type target;
      color_type  color;
      Arrow(const source_type& s, const target_type& t, const color_type& c) : source(s), target(t), color(c) {};
      // Flipping
      template <typename OtherSource_, typename OtherTarget_, typename OtherColor_>
      struct rebind {
        typedef Arrow<OtherSource_, OtherTarget_, OtherColor_> type;
      };
      struct flip {
        typedef Arrow<target_type, source_type, color_type> type;
        type arrow(const arrow_type& a) { return type(a.target, a.source, a.color);};
      };

      // Printing
      friend std::ostream& operator<<(std::ostream& os, const Arrow& a) {
        os << a.source << " --(" << a.color << ")--> " << a.target;
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


    template<typename Source_, typename Target_, typename Color_, typename SupportCompare_>
    struct ArrowContainerTraits {
    public:
      //
      // Encapsulated types
      //
      typedef Arrow<Source_,Target_,Color_>    arrow_type;
      typedef typename arrow_type::source_type source_type;
      typedef typename arrow_type::target_type target_type;
      typedef typename arrow_type::color_type  color_type;
      typedef SupportCompare_                  support_compare_type;
      // Index tags
      struct                                   sourceColorTag{};
      struct                                   targetColorTag{};
      struct                                   sourceTargetTag{};

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
          virtual const color_type&  color()  const {return this->_itor->color;};
          virtual const target_type& target() const {return this->_itor->target;};
          virtual const arrow_type&  arrow()  const {return *(this->_itor);};
        };
        class reverse_iterator : public traits::reverse_iterator {
        public:
          reverse_iterator(const typename traits::reverse_iterator::itor_type& itor) : traits::reverse_iterator(itor) {};
          virtual const source_type& source() const {return this->_itor->source;};
          virtual const color_type&  color()  const {return this->_itor->color;};
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
        }
      };// class ArrowSequence
    };// class ArrowContainerTraits


    // The specialized ArrowContainer types distinguish the cases of unique and multiple colors of arrows on
    // for each (source,target) pair (i.e., a single arrow, or multiple arrows between each pair of points).
    typedef enum {multiColor, uniColor} ColorMultiplicity;

    template<typename Source_, typename Target_, typename Color_, ColorMultiplicity colorMultiplicity, typename SupportCompare_>
    struct ArrowContainer {};

    template<typename Source_, typename Target_, typename Color_, typename SupportCompare_>
    struct ArrowContainer<Source_, Target_, Color_, multiColor, SupportCompare_> {
      // Define container's encapsulated types
      typedef ArrowContainerTraits<Source_, Target_, Color_, SupportCompare_>      traits;
      // need to def arrow_type locally, since BOOST_MULTI_INDEX_MEMBER barfs when first template parameter starts with 'typename'
      typedef typename traits::arrow_type                         arrow_type;
      // Container set type
      typedef ::boost::multi_index::multi_index_container<
        typename traits::arrow_type,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<typename traits::sourceTargetTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type,
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::color_type,  color)
            >
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<typename traits::sourceColorTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type,
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::color_type,  color),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target)
            >
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<typename traits::targetColorTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type,
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::color_type,  color),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source)
            >
          >
        >,
        ALE_ALLOCATOR<typename traits::arrow_type>
      > set_type;
     // multi-index set of multicolor arrows
      set_type set;
    }; // class ArrowContainer<multiColor>

    template<typename Source_, typename Target_, typename Color_, typename SupportCompare_>
    struct ArrowContainer<Source_, Target_, Color_, uniColor, SupportCompare_> {
      // Define container's encapsulated types
      typedef ArrowContainerTraits<Source_, Target_, Color_, SupportCompare_> traits;
      // need to def arrow_type locally, since BOOST_MULTI_INDEX_MEMBER barfs when first template parameter starts with 'typename'
      typedef typename traits::arrow_type                                   arrow_type;

      // multi-index set type -- arrow set
      typedef ::boost::multi_index::multi_index_container<
        typename traits::arrow_type,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<typename traits::sourceTargetTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type,
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::color_type,  color)
            >
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<typename traits::sourceColorTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type,
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::color_type,  color),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target)
            >,
            SupportCompare_
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<typename traits::targetColorTag>,
            ::boost::multi_index::composite_key<
              typename traits::arrow_type,
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::target_type, target),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::color_type,  color),
              BOOST_MULTI_INDEX_MEMBER(arrow_type, typename traits::source_type, source)
            >
          >
        >,
        ALE_ALLOCATOR<typename traits::arrow_type>
      > set_type;
      // multi-index set of unicolor arrow records
      set_type set;
    }; // class ArrowContainer<uniColor>
  }; // namespace SifterDef

  //
  // ASifter (short for Abstract Sifter, structurally a bipartite graph with colored arrows) implements a sequential interface
  // similar to that of Sieve, except the source and target points may have different types and iterated operations (e.g., nCone,
  // closure) are not available.
  //
template<typename Source_, typename Target_, typename Color_, SifterDef::ColorMultiplicity colorMultiplicity, typename SupportCompare_ = ::boost::multi_index::composite_key_compare<std::less<Source_>, std::less<Color_>, std::less<Target_> >, typename SourceCtnr_ = SifterDef::RecContainer<Source_, SifterDef::Rec<Source_> >, typename TargetCtnr_ = SifterDef::RecContainer<Target_, SifterDef::Rec<Target_> > >
  class ASifter { // class ASifter
  public:
    typedef struct {
      typedef ASifter<Source_, Target_, Color_, colorMultiplicity, SupportCompare_, SourceCtnr_, TargetCtnr_> graph_type;
      // Encapsulated container types
      typedef SifterDef::ArrowContainer<Source_, Target_, Color_, colorMultiplicity, SupportCompare_> arrow_container_type;
      typedef SourceCtnr_                                                            cap_container_type;
      typedef TargetCtnr_                                                            base_container_type;
      // Types associated with records held in containers
      typedef typename arrow_container_type::traits::arrow_type                      arrow_type;
      typedef typename arrow_container_type::traits::source_type                     source_type;
      typedef typename cap_container_type::traits::rec_type                          sourceRec_type;
      typedef typename arrow_container_type::traits::target_type                     target_type;
      typedef typename base_container_type::traits::rec_type                         targetRec_type;
      typedef typename arrow_container_type::traits::color_type                      color_type;
      typedef typename arrow_container_type::traits::support_compare_type            support_compare_type;
      // Convenient tag names
      typedef typename arrow_container_type::traits::sourceColorTag                  supportInd;
      typedef typename arrow_container_type::traits::targetColorTag                  coneInd;
      typedef typename arrow_container_type::traits::sourceTargetTag                 arrowInd;
      typedef typename base_container_type::traits::pointTag                         baseInd;
      typedef typename cap_container_type::traits::pointTag                          capInd;
      //
      // Return types
      //
      typedef typename
      arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,arrowInd>::type, source_type, target_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, color_type, color)>
      arrowSequence;

      // FIX: This is a temp fix to include addArrow into the interface; should probably be pushed up to ArrowSequence
      struct coneSequence : public arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,coneInd>::type, target_type, color_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, source_type, source)> {
      protected:
        graph_type& _graph;
      public:
        typedef typename
          arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,coneInd>::type, target_type, color_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, source_type, source)> base_type;
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
        void addArrow(const source_type& s, const color_type& c){
          this->_graph.addArrow(arrow_type(s,this->key,c));
        };

        virtual bool contains(const source_type& s) {
          // Check whether a given point is in the index
          typename ::boost::multi_index::index<typename ASifter::traits::arrow_container_type::set_type,typename ASifter::traits::arrowInd>::type& index = ::boost::multi_index::get<typename ASifter::traits::arrowInd>(this->_graph._arrows.set);
          return (index.find(::boost::make_tuple(s,this->key)) != index.end());
        };
      };// struct coneSequence

      // FIX: This is a temp fix to include addArrow into the interface; should probably be pushed up to ArrowSequence
      struct supportSequence : public arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,supportInd>::type, source_type, color_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, target_type, target)> {
      protected:
        graph_type& _graph;
      public:
        typedef typename
          arrow_container_type::traits::template ArrowSequence<typename ::boost::multi_index::index<typename arrow_container_type::set_type,supportInd>::type, source_type, color_type, BOOST_MULTI_INDEX_MEMBER(arrow_type, target_type, target)> base_type;
        // Encapsulated types
        typedef typename base_type::traits traits;
        typedef typename base_type::iterator iterator;
        typedef typename base_type::reverse_iterator reverse_iterator;
        // Basic interface
        supportSequence(const supportSequence& seq) : base_type(seq), _graph(seq._graph) {};
        supportSequence(graph_type& graph, typename traits::index_type& index, const typename base_type::key_type& k) : base_type(index, k), _graph(graph){};
        supportSequence(graph_type& graph, typename traits::index_type& index, const typename base_type::key_type& k, const typename base_type::subkey_type& kk) : base_type(index, k, kk), _graph(graph) {};
        virtual ~supportSequence() {};

        // FIX: WARNING: (or a HACK?): we flip the arrow on addition here.
        // Fancy interface
        void addArrow(const typename arrow_type::flip::type& af) {
          this->_graph.addArrow(af.target, af.source, af.color);
        };
        void addArrow(const target_type& t, const color_type& c){
          this->_graph.addArrow(arrow_type(this->key,t,c));
        };
      };// struct supportSequence


      typedef typename base_container_type::traits::PointSequence baseSequence;
      typedef typename cap_container_type::traits::PointSequence  capSequence;
      typedef std::set<source_type>   coneSet;
      typedef ALE::array<source_type> coneArray;
      typedef std::set<target_type>   supportSet;
      typedef ALE::array<target_type> supportArray;
    } traits;

    template <typename OtherSource_, typename OtherTarget_, typename OtherColor_, SifterDef::ColorMultiplicity otherColorMultiplicity,
              typename OtherSupportCompare_  = ::boost::multi_index::composite_key_compare<std::less<OtherSource_>, std::less<OtherColor_>, std::less<OtherTarget_> >,
              typename OtherSourceCtnr_ = SifterDef::RecContainer<OtherSource_, SifterDef::Rec<OtherSource_> >,
              typename OtherTargetCtnr_ = SifterDef::RecContainer<OtherTarget_, SifterDef::Rec<OtherTarget_> > >
    struct rebind {
      typedef ASifter<OtherSource_, OtherTarget_, OtherColor_, otherColorMultiplicity, OtherSupportCompare_, OtherSourceCtnr_, OtherTargetCtnr_> type;
    };

  public:
    // Debug level
    int _debug;
    //protected:
    typename traits::arrow_container_type _arrows;
    typename traits::base_container_type  _base;
    typename traits::cap_container_type   _cap;
  protected:
    MPI_Comm    _comm;
    int         _commRank;
    int         _commSize;
    PetscObject _petscObj;
    void __init(MPI_Comm comm) {
      static PetscClassId sifterType = -1;
      //const char        *id_name = ALE::getClassName<T>();
      const char        *id_name = "Sifter";
      PetscErrorCode     ierr;

      if (sifterType < 0) {
        ierr = PetscClassIdRegister(id_name,&sifterType);CHKERROR(ierr, "Error in MPI_Comm_rank");
      }
      this->_comm = comm;
      ierr = MPI_Comm_rank(this->_comm, &this->_commRank);CHKERROR(ierr, "Error in MPI_Comm_rank");
      ierr = MPI_Comm_size(this->_comm, &this->_commSize);CHKERROR(ierr, "Error in MPI_Comm_rank");
#ifdef USE_PETSC_OBJ
      ierr = PetscObjectCreateGeneric(this->_comm, sifterType, id_name, &this->_petscObj);CHKERROR(ierr, "Error in PetscObjectCreate");
#endif
      //ALE::restoreClassName<T>(id_name);
    };
    // We store these sequence objects to avoid creating them each query
    Obj<typename traits::coneSequence> _coneSeq;
    Obj<typename traits::supportSequence> _supportSeq;
  public:
    //
    // Basic interface
    //
    ASifter(MPI_Comm comm = PETSC_COMM_SELF, const int& debug = 0) : _debug(debug), _petscObj(NULL) {
      __init(comm);
      this->_coneSeq    = new typename traits::coneSequence(*this, ::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set), typename traits::target_type());
      this->_supportSeq = new typename traits::supportSequence(*this, ::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set), typename traits::source_type());
   }
    virtual ~ASifter() {
#ifdef USE_PETSC_OBJ
      PetscErrorCode ierr;
      ierr = PetscObjectDestroy(&this->_petscObj);CHKERROR(ierr, "Failed in PetscObjectDestroy");
#endif
    };
    //
    // Query methods
    //
    int         debug()    const {return this->_debug;};
    void        setDebug(const int debug) {this->_debug = debug;};
    MPI_Comm    comm()     const {return this->_comm;};
    int         commSize() const {return this->_commSize;};
    int         commRank() const {return this->_commRank;}
#ifdef USE_PETSC_OBJ
    PetscObject petscObj() const {return this->_petscObj;};
#endif

    // Added to allow optimized versions
    void assemble() {};
    void assemblePoints() {};
    // FIX: need const_cap, const_base returning const capSequence etc, but those need to have const_iterators, const_begin etc.
    Obj<typename traits::capSequence> cap() {
      return typename traits::capSequence(::boost::multi_index::get<typename traits::capInd>(this->_cap.set));
    };
    Obj<typename traits::baseSequence> base() {
      return typename traits::baseSequence(::boost::multi_index::get<typename traits::baseInd>(this->_base.set));
    };
    typename traits::capSequence::iterator capBegin() {
      return this->cap()->begin();
    };
    typename traits::capSequence::iterator capEnd() {
      return this->cap()->end();
    };
    typename traits::baseSequence::iterator baseBegin() {
      return this->base()->begin();
    };
    typename traits::baseSequence::iterator baseEnd() {
      return this->base()->end();
    };
    int getBaseSize() {return this->base()->size();};
    void setBaseSize(int size) {};
    int getCapSize() {return this->cap()->size();};
    void setCapSize(int size) {};
    bool capContains(const typename traits::source_type& p) {
      typename traits::capSequence cap(::boost::multi_index::get<typename traits::capInd>(this->_cap.set));

      //for(typename traits::capSequence::iterator c_iter = cap.begin(); c_iter != cap.end(); ++c_iter) {
      //}
      return cap.contains(p);
    };
    bool baseContains(const typename traits::target_type& p) {
      typename traits::baseSequence base(::boost::multi_index::get<typename traits::baseInd>(this->_base.set));

      //for(typename traits::capSequence::iterator c_iter = cap.begin(); c_iter != cap.end(); ++c_iter) {
      //}
      return base.contains(p);
    };
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
    const typename traits::coneSequence::iterator
    coneBegin(const typename traits::target_type& p) {
      return this->cone(p)->begin();
    };
    const typename traits::coneSequence::iterator
    coneEnd(const typename traits::target_type& p) {
      return this->cone(p)->end();
    };
    void setConeSize(const typename traits::target_type& p, int size) {};
#endif
    template<class InputSequence>
    Obj<typename traits::coneSet>
    cone(const Obj<InputSequence>& points) {
      return this->cone(points, typename traits::color_type(), false);
    }
#ifdef SLOW
    Obj<typename traits::coneSequence>
    cone(const typename traits::target_type& p, const typename traits::color_type& color) {
      return typename traits::coneSequence(*this, ::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set), p, color);
    }
#else
    const Obj<typename traits::coneSequence>&
    cone(const typename traits::target_type& p, const typename traits::color_type& color) {
      this->_coneSeq->setKey(p);
      this->_coneSeq->setSubkey(color);
      this->_coneSeq->setUseSubkey(true);
      return this->_coneSeq;
    };
#endif
    template<class InputSequence>
    Obj<typename traits::coneSet>
    cone(const Obj<InputSequence>& points, const typename traits::color_type& color, bool useColor = true) {
      Obj<typename traits::coneSet> cone = typename traits::coneSet();
      for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
        Obj<typename traits::coneSequence> pCone;
        if (useColor) {
          pCone = this->cone(*p_itor, color);
        } else {
          pCone = this->cone(*p_itor);
        }
        cone->insert(pCone->begin(), pCone->end());
      }
      return cone;
    }
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
    int getConeSize(const typename traits::target_type& p) {
      return this->cone(p)->size();
    }
#ifdef SLOW
    Obj<typename traits::supportSequence>
    support(const typename traits::source_type& p) {
      return typename traits::supportSequence(*this, ::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set), p);
    }
#else
    const Obj<typename traits::supportSequence>&
    support(const typename traits::source_type& p) {
      this->_supportSeq->setKey(p);
      this->_supportSeq->setUseSubkey(false);
      return this->_supportSeq;
    };
    const typename traits::supportSequence::iterator
    supportBegin(const typename traits::source_type& p) {
      return this->support(p)->begin();
    };
    const typename traits::supportSequence::iterator
    supportEnd(const typename traits::source_type& p) {
      return this->support(p)->end();
    };
    const typename traits::supportSequence::iterator
    supportBegin(const typename traits::source_type& p, const typename traits::color_type& color) {
      return this->support(p, color)->begin();
    };
    const typename traits::supportSequence::iterator
    supportEnd(const typename traits::source_type& p, const typename traits::color_type& color) {
      return this->support(p, color)->end();
    };
    void setSupportSize(const typename traits::source_type& p, int size) {};
#endif
#ifdef SLOW
    Obj<typename traits::supportSequence>
    support(const typename traits::source_type& p, const typename traits::color_type& color) {
      return typename traits::supportSequence(*this, ::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set), p, color);
    };
#else
    const Obj<typename traits::supportSequence>&
    support(const typename traits::source_type& p, const typename traits::color_type& color) {
      this->_supportSeq->setKey(p);
      this->_supportSeq->setSubkey(color);
      this->_supportSeq->setUseSubkey(true);
      return this->_supportSeq;
    };
#endif
    template<class InputSequence>
    Obj<typename traits::supportSet>
    support(const Obj<InputSequence>& sources) {
      return this->support(sources, typename traits::color_type(), false);
    }
    template<class InputSequence>
    Obj<typename traits::supportSet>
    support(const Obj<InputSequence>& points, const typename traits::color_type& color, bool useColor = true){
      Obj<typename traits::supportSet> supp = typename traits::supportSet();
      for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
        Obj<typename traits::supportSequence> pSupport;
        if (useColor) {
          pSupport = this->support(*p_itor, color);
        } else {
          pSupport = this->support(*p_itor);
        }
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
    int getSupportSize(const typename traits::source_type& p) {
      return this->support(p)->size();
    }
    int getSupportSize(const typename traits::source_type& p, const typename traits::color_type& color) {
      return this->support(p, color)->size();
    }

    template<typename ostream_type>
    void view(ostream_type& os, const char* label = NULL, bool rawData = false){
      int rank = this->commRank();

      if(label != NULL) {
        os << "["<<rank<<"]Viewing Sifter '" << label << "':" << std::endl;
      }
      else {
        os << "["<<rank<<"]Viewing a Sifter:" << std::endl;
      }
      if(!rawData) {
        os << "cap --> base:" << std::endl;
        Obj<typename traits::capSequence> cap = this->cap();
        for(typename traits::capSequence::iterator capi = cap->begin(); capi != cap->end(); capi++) {
          const Obj<typename traits::supportSequence>& supp = this->support(*capi);

          for(typename traits::supportSequence::iterator suppi = supp->begin(); suppi != supp->end(); suppi++) {
            os << *capi << "--(" << suppi.color() << ")-->" << *suppi << std::endl;
          }
        }
        os << "base <-- cap:" << std::endl;
        Obj<typename traits::baseSequence> base = this->base();
        for(typename traits::baseSequence::iterator basei = base->begin(); basei != base->end(); basei++) {
          const Obj<typename traits::coneSequence>& cone = this->cone(*basei);

          for(typename traits::coneSequence::iterator conei = cone->begin(); conei != cone->end(); conei++) {
            os << *basei <<  "<--(" << conei.color() << ")--" << *conei << std::endl;
          }
        }
        os << "cap --> outdegrees:" << std::endl;
        for(typename traits::capSequence::iterator capi = cap->begin(); capi != cap->end(); capi++) {
          os << *capi <<  "-->" << capi.degree() << std::endl;
        }
        os << "base <-- indegrees:" << std::endl;
        for(typename traits::baseSequence::iterator basei = base->begin(); basei != base->end(); basei++) {
          os << *basei <<  "<--" << basei.degree() << std::endl;
        }
      }
      else {
        os << "'raw' arrow set:" << std::endl;
        for(typename traits::arrow_container_type::set_type::iterator ai = _arrows.set.begin(); ai != _arrows.set.end(); ai++)
        {
          typename traits::arrow_type arr = *ai;
          os << arr << std::endl;
        }
        os << "'raw' base set:" << std::endl;
        for(typename traits::base_container_type::set_type::iterator bi = _base.set.begin(); bi != _base.set.end(); bi++)
        {
          typename traits::base_container_type::traits::rec_type bp = *bi;
          os << bp << std::endl;
        }
        os << "'raw' cap set:" << std::endl;
        for(typename traits::cap_container_type::set_type::iterator ci = _cap.set.begin(); ci != _cap.set.end(); ci++)
        {
          typename traits::cap_container_type::traits::rec_type cp = *ci;
          os << cp << std::endl;
        }
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
        std::cout << "viewing a Sifter, comm = " << this->comm() << ", PETSC_COMM_SELF = " << PETSC_COMM_SELF << ", commRank = " << this->commRank() << std::endl;
      }
      if(label != NULL) {
        PetscPrintf(this->comm(), "viewing Sifter: '%s'\n", label);
      } else {
        PetscPrintf(this->comm(), "viewing a Sifter: \n");
      }
      if(!raw) {
        ostringstream txt;
        if(this->commRank() == 0) {
          txt << "cap --> base:\n";
        }
        typename traits::capSequence cap   = this->cap();
        typename traits::baseSequence base = this->base();
        if(cap.empty()) {
          txt << "[" << this->commRank() << "]: empty" << std::endl;
        }
        for(typename traits::capSequence::iterator capi = cap.begin(); capi != cap.end(); capi++) {
          const Obj<typename traits::supportSequence>& supp = this->support(*capi);
          for(typename traits::supportSequence::iterator suppi = supp->begin(); suppi != supp->end(); suppi++) {
            txt << "[" << this->commRank() << "]: " << *capi << "--(" << suppi.color() << ")-->" << *suppi << std::endl;
          }
        }
        //
        ierr = PetscSynchronizedPrintf(this->comm(), txt.str().c_str());CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        ierr = PetscSynchronizedFlush(this->comm()); CHKERROR(ierr, "Error in PetscSynchronizedFlush");
#if 0
        //
        ostringstream txt1;
        if(this->commRank() == 0) {
          //txt1 << "cap <point,degree>:\n";
          txt1 << "cap:\n";
        }
        txt1 << "[" << this->commRank() << "]:  [";
        for(typename traits::capSequence::iterator capi = cap.begin(); capi != cap.end(); capi++) {
          //txt1 << " <" << *capi << "," << capi.degree() << ">";
          txt1 << "  " << *capi;
        }
        txt1 << " ]" << std::endl;
        //
        ierr = PetscSynchronizedPrintf(this->comm(), txt1.str().c_str());CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        ierr = PetscSynchronizedFlush(this->comm()); CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        //
        ostringstream txt2;
        if(this->commRank() == 0) {
          //txt2 << "base <point,degree>:\n";
          txt2 << "base:\n";
        }
        txt2 << "[" << this->commRank() << "]:  [";
        for(typename traits::baseSequence::iterator basei = base.begin(); basei != base.end(); basei++) {
          txt2 << "  " << *basei;
          //txt2 << " <" << *basei << "," << basei.degree() << ">";
        }
        txt2 << " ]" << std::endl;
        //
        ierr = PetscSynchronizedPrintf(this->comm(), txt2.str().c_str());CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        ierr = PetscSynchronizedFlush(this->comm()); CHKERROR(ierr, "Error in PetscSynchronizedFlush");
#endif
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
        //
        ostringstream txt1;
        if(this->commRank() == 0) {
          txt1 << "'raw' base set:" << std::endl;
        }
        for(typename traits::base_container_type::set_type::iterator bi = _base.set.begin(); bi != _base.set.end(); bi++)
        {
          typename traits::base_container_type::traits::rec_type bp = *bi;
          txt1 << "[" << this->commRank() << "]: " << bp << std::endl;
        }
        ierr = PetscSynchronizedPrintf(this->comm(), txt1.str().c_str());CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        ierr = PetscSynchronizedFlush(this->comm()); CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        //
        ostringstream txt2;
        if(this->commRank() == 0) {
          txt2 << "'raw' cap set:" << std::endl;
        }
        for(typename traits::cap_container_type::set_type::iterator ci = _cap.set.begin(); ci != _cap.set.end(); ci++)
        {
          typename traits::cap_container_type::traits::rec_type cp = *ci;
          txt2 << "[" << this->commRank() << "]: " << cp << std::endl;
        }
        ierr = PetscSynchronizedPrintf(this->comm(), txt2.str().c_str());CHKERROR(ierr, "Error in PetscSynchronizedFlush");
        ierr = PetscSynchronizedFlush(this->comm()); CHKERROR(ierr, "Error in PetscSynchronizedFlush");
      }// if(raw)

      PetscFunctionReturn(0);
    };
  public:
    void copy(ASifter *newSifter) {
      const typename traits::baseSequence::iterator sBegin = this->baseBegin();
      const typename traits::baseSequence::iterator sEnd   = this->baseEnd();

      for(typename traits::baseSequence::iterator r_iter = sBegin; r_iter != sEnd; ++r_iter) {
        const typename traits::coneSequence::iterator pBegin = this->coneBegin(*r_iter);
        const typename traits::coneSequence::iterator pEnd   = this->coneEnd(*r_iter);

        for(typename traits::coneSequence::iterator p_iter = pBegin; p_iter != pEnd; ++p_iter) {
          const Obj<typename traits::supportSequence>&              support  = this->support(*p_iter);
          const typename traits::supportSequence::iterator supBegin = support->begin();
          const typename traits::supportSequence::iterator supEnd   = support->end();

          for(typename traits::supportSequence::iterator s_iter = supBegin; s_iter != supEnd; ++s_iter) {
            newSifter->addArrow(*p_iter, *s_iter, s_iter.color());
          }
        }
      }
    };
    //
    // Lattice queries
    //
    template<class targetInputSequence>
    Obj<typename traits::coneSequence> meet(const Obj<targetInputSequence>& targets);
    // unimplemented
    template<class targetInputSequence>
    Obj<typename traits::coneSequence> meet(const Obj<targetInputSequence>& targets, const typename traits::color_type& color);
    // unimplemented
    template<class sourceInputSequence>
    Obj<typename traits::coneSequence> join(const Obj<sourceInputSequence>& sources);
    // unimplemented
    template<class sourceInputSequence>
    Obj<typename traits::coneSequence> join(const Obj<sourceInputSequence>& sources, const typename traits::color_type& color);
  public:
    //
    // Structural manipulation
    //
    void clear() {
      this->_arrows.set.clear(); this->_base.set.clear(); this->_cap.set.clear();
    };
    void addBasePoint(const typename traits::target_type t) {
      /* // Increase degree by 0, which won't affect an existing point and will insert a new point, if necessery
         this->_base.adjustDegree(t,0); */
      this->_base.set.insert(typename traits::targetRec_type(t,0));
    };
    void addBasePoint(const typename traits::targetRec_type b) {
      this->_base.set.insert(b);
    };
    void removeBasePoint(const typename traits::target_type t) {
      if (this->_debug) {std::cout << " Removing " << t << " from the base" << std::endl;}
      // Clear the cone and remove the point from _base
      this->clearCone(t);
      this->_base.removePoint(t);
    };
    void addCapPoint(const typename traits::source_type s) {
      /* // Increase degree by 0, which won't affect an existing point and will insert a new point, if necessery
         this->_cap.adjustDegree(s,0); */
      this->_cap.set.insert(typename traits::sourceRec_type(s,0));
    };
    void addCapPoint(const typename traits::sourceRec_type c) {
      this->_cap.set.insert(c);
    };
    void removeCapPoint(const typename traits::source_type s) {
      if (this->_debug) {std::cout << " Removing " << s << " from the cap" << std::endl;}
      // Clear the support and remove the point from _cap
      this->clearSupport(s);
      this->_cap.removePoint(s);
    };
    virtual void addArrow(const typename traits::source_type& p, const typename traits::target_type& q) {
      this->addArrow(p, q, typename traits::color_type());
    };
    virtual void addArrow(const typename traits::source_type& p, const typename traits::target_type& q, const typename traits::color_type& color) {
      this->addArrow(typename traits::arrow_type(p, q, color));
      //std::cout << "Added " << arrow_type(p, q, color);
    };
    virtual bool checkArrow(const typename traits::arrow_type& a) {
      if (this->_cap.set.find(a.source) == this->_cap.set.end()) return false;
      if (this->_base.set.find(a.target) == this->_base.set.end()) return false;
      return true;
    };
    virtual void addArrow(const typename traits::arrow_type& a, bool noNewPoints = false) {
      if (noNewPoints && !this->checkArrow(a)) return;
      this->_arrows.set.insert(a);
      this->addBasePoint(a.target);
      this->addCapPoint(a.source);
    };
    virtual void removeArrow(const typename traits::source_type& p, const typename traits::target_type& q) {
      this->removeArrow(typename traits::arrow_type(p, q, typename traits::color_type()));
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
        if(j->color == a.color) {
          if (this->_debug) {
            std::cout << std::endl << "removeArrow: found:" << *j << std::endl;
          }
          /* this->_base.adjustDegree(a.target, -1); this->_cap.adjustDegree(a.source,-1); */
          arrowIndex.erase(j);
          break;
        }
      }
    };

    void addCone(const typename traits::source_type& source, const typename traits::target_type& target){
      this->addArrow(source, target);
    }
    template<class sourceInputSequence>
    void addCone(const Obj<sourceInputSequence>& sources, const typename traits::target_type& target) {
      this->addCone(sources, target, typename traits::color_type());
    }
    void addCone(const typename traits::source_type& source, const typename traits::target_type& target, const typename traits::color_type& color) {
      this->addArrow(source, target, color);
    }
    template<class sourceInputSequence>
    void
    addCone(const Obj<sourceInputSequence>& sources, const typename traits::target_type& target, const typename traits::color_type& color){
      if (this->_debug > 1) {std::cout << "Adding a cone " << std::endl;}
      for(typename sourceInputSequence::iterator iter = sources->begin(); iter != sources->end(); ++iter) {
        if (this->_debug > 1) {std::cout << "Adding arrow from " << *iter << " to " << target << "(" << color << ")" << std::endl;}
        this->addArrow(*iter, target, color);
      }
    }
    void clearCone(const typename traits::target_type& t) {
      clearCone(t, typename traits::color_type(), false);
    };

    void clearCone(const typename traits::target_type& t, const typename traits::color_type&  color, bool useColor = true) {
      // Use the cone sequence types to clear the cone
      typename traits::coneSequence::traits::index_type& coneIndex =
        ::boost::multi_index::get<typename traits::coneInd>(this->_arrows.set);
      typename traits::coneSequence::traits::index_type::iterator i, ii, j;
      if (this->_debug > 20) {
        std::cout << "clearCone: removing cone over " << t;
        if(useColor) {
          std::cout << " with color" << color << std::endl;
          const Obj<typename traits::coneSequence>& cone = this->cone(t,color);
          std::cout << "[";
          for(typename traits::coneSequence::iterator ci = cone->begin(); ci != cone->end(); ci++) {
            std::cout << "  " << ci.arrow();
          }
          std::cout << "]" << std::endl;
        }
        else {
          std::cout << std::endl;
          const Obj<typename traits::coneSequence>& cone = this->cone(t);
          std::cout << "[";
          for(typename traits::coneSequence::iterator ci = cone->begin(); ci != cone->end(); ci++) {
            std::cout << "  " << ci.arrow();
          }
          std::cout << "]" << std::endl;
        }
      }
      if (useColor) {
        i = coneIndex.lower_bound(::boost::make_tuple(t,color));
        ii = coneIndex.upper_bound(::boost::make_tuple(t,color));
      } else {
        i = coneIndex.lower_bound(::boost::make_tuple(t));
        ii = coneIndex.upper_bound(::boost::make_tuple(t));
      }
      for(j = i; j != ii; j++){
        // Adjust the degrees before removing the arrow; use a different iterator, since we'll need i,ii to do the arrow erasing.
        if (this->_debug) {
          std::cout << "clearCone: adjusting degrees for endpoints of arrow: " << *j << std::endl;
        }
        /* this->_cap.adjustDegree(j->source, -1);
           this->_base.adjustDegree(j->target, -1); */
      }
      coneIndex.erase(i,ii);
    };// clearCone()

    template<class InputSequence>
    void
    restrictBase(const Obj<InputSequence>& points) {
      typename traits::baseSequence base = this->base();
      typename std::set<typename traits::target_type> remove;

      for(typename traits::baseSequence::iterator bi = base.begin(); bi != base.end(); bi++) {
        // Check whether *bi is in points, if it is NOT, remove it
        //           if (!points->contains(*bi)) {
        if (points->find(*bi) == points->end()) {
          //             this->removeBasePoint(*bi);
          remove.insert(*bi);
        }
      }
      //FIX
      for(typename std::set<typename traits::target_type>::iterator r_iter = remove.begin(); r_iter != remove.end(); ++r_iter) {
        this->removeBasePoint(*r_iter);
      }
    }

    template<class InputSequence>
    void
    excludeBase(const Obj<InputSequence>& points) {
      for(typename InputSequence::iterator pi = points->begin(); pi != points->end(); pi++) {
        this->removeBasePoint(*pi);
      }
    }

    template<class InputSequence>
    void
    restrictCap(const Obj<InputSequence>& points) {
      typename traits::capSequence cap = this->cap();
      for(typename traits::capSequence::iterator ci = cap.begin(); ci != cap.end(); ci++) {
        // Check whether *ci is in points, if it is NOT, remove it
        if(points->find(*ci) == points->end()) {
          this->removeCapPoint(*ci);
        }
      }
    }

    template<class InputSequence>
    void
    excludeCap(const Obj<InputSequence>& points) {
      for(typename InputSequence::iterator pi = points->begin(); pi != points->end(); pi++) {
        this->removeCapPoint(*pi);
      }
    }

    void clearSupport(const typename traits::source_type& s) {
      clearSupport(s, typename traits::color_type(), false);
    };
    void clearSupport(const typename traits::source_type& s, const typename traits::color_type&  color, bool useColor = true) {
      // Use the cone sequence types to clear the cone
      typename
        traits::supportSequence::traits::index_type& suppIndex = ::boost::multi_index::get<typename traits::supportInd>(this->_arrows.set);
      typename traits::supportSequence::traits::index_type::iterator i, ii, j;
      if (useColor) {
        i = suppIndex.lower_bound(::boost::make_tuple(s,color));
        ii = suppIndex.upper_bound(::boost::make_tuple(s,color));
      } else {
        i = suppIndex.lower_bound(::boost::make_tuple(s));
        ii = suppIndex.upper_bound(::boost::make_tuple(s));
      }
      for(j = i; j != ii; j++){
        // Adjust the degrees before removing the arrow
        /* this->_cap.adjustDegree(j->source, -1);
           this->_base.adjustDegree(j->target, -1); */
      }
      suppIndex.erase(i,ii);
    }
    void setCone(const typename traits::source_type& source, const typename traits::target_type& target){
      this->clearCone(target, typename traits::color_type(), false); this->addCone(source, target);
    }
    template<class sourceInputSequence>
    void setCone(const Obj<sourceInputSequence>& sources, const typename traits::target_type& target) {
      this->clearCone(target, typename traits::color_type(), false); this->addCone(sources, target, typename traits::color_type());
    }
    void setCone(const typename traits::source_type& source, const typename traits::target_type& target, const typename traits::color_type& color) {
      this->clearCone(target, color, true); this->addCone(source, target, color);
    }
    template<class sourceInputSequence>
    void setCone(const Obj<sourceInputSequence>& sources, const typename traits::target_type& target, const typename traits::color_type& color){
      this->clearCone(target, color, true); this->addCone(sources, target, color);
    }
    template<class targetInputSequence>
    void addSupport(const typename traits::source_type& source, const Obj<targetInputSequence >& targets) {
      this->addSupport(source, targets, typename traits::color_type());
    }
    template<class targetInputSequence>
    void addSupport(const typename traits::source_type& source, const Obj<targetInputSequence>& targets, const typename traits::color_type& color) {
      const typename targetInputSequence::iterator end = targets->end();

      for(typename targetInputSequence::iterator iter = targets->begin(); iter != end; ++iter) {
        this->addArrow(source, *iter, color);
      }
    }
    template<typename Sifter_>
    void add(const Obj<Sifter_>& cbg, bool noNewPoints = false) {
      typename ::boost::multi_index::index<typename Sifter_::traits::arrow_container_type::set_type, typename Sifter_::traits::arrowInd>::type& aInd = ::boost::multi_index::get<typename Sifter_::traits::arrowInd>(cbg->_arrows.set);

      for(typename ::boost::multi_index::index<typename Sifter_::traits::arrow_container_type::set_type, typename Sifter_::traits::arrowInd>::type::iterator a_iter = aInd.begin(); a_iter != aInd.end(); ++a_iter) {
        this->addArrow(*a_iter, noNewPoints);
      }
      if (!noNewPoints) {
        typename ::boost::multi_index::index<typename Sifter_::traits::base_container_type::set_type, typename Sifter_::traits::baseInd>::type& bInd = ::boost::multi_index::get<typename Sifter_::traits::baseInd>(this->_base.set);

        for(typename ::boost::multi_index::index<typename Sifter_::traits::base_container_type::set_type, typename Sifter_::traits::baseInd>::type::iterator b_iter = bInd.begin(); b_iter != bInd.end(); ++b_iter) {
          this->addBasePoint(*b_iter);
        }
        typename ::boost::multi_index::index<typename Sifter_::traits::cap_container_type::set_type, typename Sifter_::traits::capInd>::type& cInd = ::boost::multi_index::get<typename Sifter_::traits::capInd>(this->_cap.set);

        for(typename ::boost::multi_index::index<typename Sifter_::traits::cap_container_type::set_type, typename Sifter_::traits::capInd>::type::iterator c_iter = cInd.begin(); c_iter != cInd.end(); ++c_iter) {
          this->addCapPoint(*c_iter);
        }
      }
    }
  }; // class ASifter

  // A UniSifter aka Sifter
  template <typename Source_, typename Target_, typename Color_,
            typename SupportCompare_ = ::boost::multi_index::composite_key_compare<std::less<Source_>, std::less<Color_>, std::less<Target_> >,
            typename SourceCtnr_ = SifterDef:: RecContainer<Source_, SifterDef::Rec<Source_> >, typename TargetCtnr_= SifterDef::RecContainer<Target_, SifterDef::Rec<Target_> > >
  class Sifter : public ASifter<Source_, Target_, Color_, SifterDef::uniColor, SupportCompare_, SourceCtnr_, TargetCtnr_> {
  public:
      typedef typename ASifter<Source_, Target_, Color_, SifterDef::uniColor, SupportCompare_, SourceCtnr_, TargetCtnr_>::traits       traits;
    template <typename OtherSource_, typename OtherTarget_, typename OtherColor_,
              typename OtherSupportCompare_  = ::boost::multi_index::composite_key_compare<std::less<OtherSource_>, std::less<OtherColor_>, std::less<OtherTarget_> >,
              typename OtherSourceCtnr_ = SifterDef::RecContainer<OtherSource_, SifterDef::Rec<OtherSource_> >,
              typename OtherTargetCtnr_ = SifterDef::RecContainer<OtherTarget_, SifterDef::Rec<OtherTarget_> >      >
    struct rebind {
      typedef Sifter<OtherSource_, OtherTarget_, OtherColor_, OtherSupportCompare_, OtherSourceCtnr_, OtherTargetCtnr_> type;
    };
    // Re-export some typedefs expected by CoSifter
    typedef typename traits::source_type                                            source_type;
    typedef typename traits::target_type                                            target_type;
    typedef typename traits::color_type                                             color_type;
    typedef typename traits::arrow_type                                             Arrow_;
    typedef typename traits::coneSequence                                           coneSequence;
    typedef typename traits::supportSequence                                        supportSequence;
    typedef typename traits::baseSequence                                           baseSequence;
    typedef typename traits::capSequence                                            capSequence;
    // Basic interface
    Sifter(MPI_Comm comm = PETSC_COMM_SELF, const int& debug = 0) :
      ASifter<Source_, Target_, Color_, SifterDef::uniColor, SupportCompare_, SourceCtnr_, TargetCtnr_>(comm, debug) {};

    const typename traits::color_type&
    getColor(const typename traits::source_type& s, const typename traits::target_type& t, bool fail = true) {
      typedef typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type,typename traits::arrowInd>::type index_type;

      const index_type& _index = ::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set);
#if 0
      ::boost::tuple<typename traits::source_type, typename traits::target_type> key = ::boost::make_tuple(s, t);
      typename index_type::iterator begin = _index.lower_bound(key);
      if(begin != _index.upper_bound(key)) {
        return begin->color;
      }
#else
      const typename index_type::iterator begin = _index.find(::boost::make_tuple(s, t));
      if (begin != _index.end()) {
        return begin->color;
      }
#endif
//       typename traits::arrowSequence arr(::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set), s, t);
//       if(arr.begin() != arr.end()) {
//         return arr.begin().color();
//       }
      if (fail) {
        ostringstream o;
        o << "Arrow " << s << " --> " << t << " not present";
        throw ALE::Exception(o.str().c_str());
      } else {
        static typename traits::color_type c;
        return c;
      }
    };

    template<typename ColorChanger>
    void modifyColor(const typename traits::source_type& s, const typename traits::target_type& t, const ColorChanger& changeColor) {
      typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type, typename traits::arrowInd>::type& index =
        ::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set);
      typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type, typename traits::arrowInd>::type::iterator i =
        index.find(::boost::make_tuple(s,t));
      if (i != index.end()) {
        index.modify(i, changeColor);
      } else {
        typename traits::arrow_type a(s, t, typename traits::color_type());
        changeColor(a);
        this->addArrow(a);
      }
    }

    struct ColorSetter {
      ColorSetter(const typename traits::color_type& color) : _color(color) {};
      void operator()(typename traits::arrow_type& p) const {
        p.color = _color;
      }
    private:
      const typename traits::color_type& _color;
    };

    void setColor(const typename traits::source_type& s, const typename traits::target_type& t, const typename traits::color_type& color) {
      ColorSetter colorSetter(color);
      typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type, typename traits::arrowInd>::type& index =
        ::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set);
      typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type, typename traits::arrowInd>::type::iterator i =
        index.find(::boost::make_tuple(s,t));
      if (i != index.end()) {
        index.modify(i, colorSetter);
      } else {
        typename traits::arrow_type a(s, t, color);
        this->addArrow(a);
      }
    };
    template<typename Labeling, typename AnotherSifter>
    void relabel(Labeling& relabeling, AnotherSifter& newLabel) {
      typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type, typename traits::arrowInd>::type& aInd = ::boost::multi_index::get<typename traits::arrowInd>(this->_arrows.set);

      for(typename ::boost::multi_index::index<typename traits::arrow_container_type::set_type, typename traits::arrowInd>::type::iterator a_iter = aInd.begin(); a_iter != aInd.end(); ++a_iter) {
        const typename traits::source_type newSource = relabeling.restrictPoint(a_iter->source)[0];
        const typename traits::target_type newTarget = relabeling.restrictPoint(a_iter->target)[0];

        newLabel.addArrow(newSource, newTarget);
      }
    }
  };// class Sifter

  class SifterSerializer {
  public:
    template<typename Sifter>
    static void writeSifter(std::ofstream& fs, Sifter& sifter) {
      typename Sifter::traits::arrow_container_type::set_type::size_type numArrows;

      if (sifter.commRank() == 0) {
        // Write local
        fs << sifter._arrows.set.size() << std::endl;
        for(typename Sifter::traits::arrow_container_type::set_type::iterator ai = sifter._arrows.set.begin(); ai != sifter._arrows.set.end(); ai++) {
          fs << ai->source << " " << ai->target << " " << ai->color << std::endl;
        }
        // Receive and write remote
        for(int p = 1; p < sifter.commSize(); ++p) {
          PetscInt       size;
          PetscInt      *arrows;
          MPI_Status     status;
          PetscErrorCode ierr;

          ierr = MPI_Recv(&size, 1, MPIU_INT, p, 1, sifter.comm(), &status);CHKERRXX(ierr);
          numArrows = size;
          fs << numArrows << std::endl;
          ierr = PetscMalloc(size*3 * sizeof(PetscInt), &arrows);CHKERRXX(ierr);
          ierr = MPI_Recv(arrows, size*3, MPIU_INT, p, 1, sifter.comm(), &status);CHKERRXX(ierr);
          for(PetscInt a = 0; a < size; ++a) {
            typename Sifter::traits::arrow_type::source_type source = arrows[a*3+0];
            typename Sifter::traits::arrow_type::target_type target = arrows[a*3+1];
            typename Sifter::traits::arrow_type::color_type  color  = arrows[a*3+2];

            fs << source << " " << target << " " << color << std::endl;
          }
          ierr = PetscFree(arrows);CHKERRXX(ierr);
        }
      } else {
        // Send remote
        PetscInt       size = sifter._arrows.set.size();
        PetscInt       a    = 0;
        PetscInt      *arrows;
        PetscErrorCode ierr;

        ierr = MPI_Send(&size, 1, MPIU_INT, 0, 1, sifter.comm());CHKERRXX(ierr);
        // There is no nice way to make a generic MPI type here. Really sucky
        ierr = PetscMalloc(size*3 * sizeof(PetscInt), &arrows);CHKERRXX(ierr);
        for(typename Sifter::traits::arrow_container_type::set_type::iterator ai = sifter._arrows.set.begin(); ai != sifter._arrows.set.end(); ai++, ++a) {
          arrows[a*3+0] = ai->source;
          arrows[a*3+1] = ai->target;
          arrows[a*3+2] = ai->color;
        }
        ierr = MPI_Send(arrows, size*3, MPIU_INT, 0, 1, sifter.comm());CHKERRXX(ierr);
        ierr = PetscFree(arrows);CHKERRXX(ierr);
      }
    }
    template<typename Sifter>
    static void loadSifter(std::ifstream& fs, Sifter& sifter) {
      typedef typename Sifter::traits::arrow_container_type::set_type::size_type size_type;
      if (sifter.commRank() == 0) {
        // Load local
        size_type numArrows;

        fs >> numArrows;
        for(size_type a = 0; a < numArrows; ++a) {
          typename Sifter::traits::arrow_type::source_type source;
          typename Sifter::traits::arrow_type::target_type target;
          typename Sifter::traits::arrow_type::color_type  color;

          fs >> source;
          fs >> target;
          fs >> color;
          sifter.addArrow(typename Sifter::traits::arrow_type(source, target, color));
        }
        // Load and send remote
        for(int p = 1; p < sifter.commSize(); ++p) {
          PetscInt       size;
          PetscInt      *arrows;
          PetscErrorCode ierr;

          fs >> numArrows;
          size = numArrows;
          ierr = MPI_Send(&size, 1, MPIU_INT, p, 1, sifter.comm());CHKERRXX(ierr);
          ierr = PetscMalloc(size*3 * sizeof(PetscInt), &arrows);CHKERRXX(ierr);
          for(PetscInt a = 0; a < size; ++a) {
            typename Sifter::traits::arrow_type::source_type source;
            typename Sifter::traits::arrow_type::target_type target;
            typename Sifter::traits::arrow_type::color_type  color;

            fs >> source;
            fs >> target;
            fs >> color;
            arrows[a*3+0] = source;
            arrows[a*3+1] = target;
            arrows[a*3+2] = color;
          }
          ierr = MPI_Send(arrows, size*3, MPIU_INT, p, 1, sifter.comm());CHKERRXX(ierr);
          ierr = PetscFree(arrows);CHKERRXX(ierr);
        }
      } else {
        // Load remote
        PetscInt       size;
        PetscInt      *arrows;
        MPI_Status     status;
        PetscErrorCode ierr;

        ierr = MPI_Recv(&size, 1, MPIU_INT, 0, 1, sifter.comm(), &status);CHKERRXX(ierr);
        ierr = PetscMalloc(size*3 * sizeof(PetscInt), &arrows);CHKERRXX(ierr);
        ierr = MPI_Recv(arrows, size*3, MPIU_INT, 0, 1, sifter.comm(), &status);CHKERRXX(ierr);
        for(PetscInt a = 0; a < size; ++a) {
          sifter.addArrow(typename Sifter::traits::arrow_type(arrows[a*3+0], arrows[a*3+1], arrows[a*3+2]));
        }
        ierr = PetscFree(arrows);CHKERRXX(ierr);
      }
    }
  };
} // namespace ALE

#endif
