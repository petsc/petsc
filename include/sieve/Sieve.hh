
#ifndef included_ALE_Sieve_hh
#define included_ALE_Sieve_hh

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <iostream>

#ifndef  included_ALE_Sifter_hh
#include <sieve/Sifter.hh>
#endif


namespace ALE {

  namespace SieveDef {
    //
    // Rec & RecContainer definitions.
    // Rec is intended to denote a graph point record.
    // 
    template <typename Point_, typename Marker_>
    struct Rec {
      typedef Point_  point_type;
      typedef Marker_ marker_type;
      template<typename OtherPoint_, typename OtherMarker_ = Marker_>
      struct rebind {
        typedef Rec<OtherPoint_, OtherMarker_> type;
      };
      point_type  point;
      int         degree;
      int         depth;
      int         height;
      marker_type marker;
      // Basic interface
      Rec() : point(point_type()), degree(0), depth(0), height(0), marker(marker_type()) {};
      Rec(const Rec& r) : point(r.point), degree(r.degree), depth(r.depth), height(r.height), marker(r.marker) {}
      Rec(const point_type& p) : point(p), degree(0), depth(0), height(0), marker(marker_type()) {};
      Rec(const point_type& p, const int degree) : point(p), degree(degree), depth(0), height(0), marker(marker_type()) {};
      Rec(const point_type& p, const int degree, const int depth, const int height, const marker_type marker) : point(p), degree(degree), depth(depth), height(height), marker(marker) {};
      // Printing
      friend std::ostream& operator<<(std::ostream& os, const Rec& p) {
        os << "<" << p.point << ", "<< p.degree << ", "<< p.depth << ", "<< p.height << ", "<< p.marker << ">";
        return os;
      };

      struct degreeAdjuster {
        degreeAdjuster(int newDegree) : _newDegree(newDegree) {};
        void operator()(Rec& r) { r.degree = this->_newDegree; }
      private:
        int _newDegree;
      };
    };// class Rec

    template <typename Point_, typename Rec_>
    struct RecContainerTraits {
      typedef Rec_ rec_type;
      typedef typename rec_type::marker_type marker_type;
      // Index tags
      struct pointTag{};
      struct degreeTag{};
      struct markerTag{};
      struct depthMarkerTag{};
      struct heightMarkerTag{};
      // Rec set definition
      typedef ::boost::multi_index::multi_index_container<
        rec_type,
        ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
            ::boost::multi_index::tag<pointTag>, BOOST_MULTI_INDEX_MEMBER(rec_type, typename rec_type::point_type, point)
          >,
//           ::boost::multi_index::ordered_non_unique<
//             ::boost::multi_index::tag<degreeTag>, BOOST_MULTI_INDEX_MEMBER(rec_type, int, degree)
//           >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<markerTag>, BOOST_MULTI_INDEX_MEMBER(rec_type, marker_type, marker)
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<depthMarkerTag>,
            ::boost::multi_index::composite_key<
              rec_type, BOOST_MULTI_INDEX_MEMBER(rec_type,int,depth), BOOST_MULTI_INDEX_MEMBER(rec_type,marker_type,marker)>
          >,
          ::boost::multi_index::ordered_non_unique<
            ::boost::multi_index::tag<heightMarkerTag>,
            ::boost::multi_index::composite_key<
              rec_type, BOOST_MULTI_INDEX_MEMBER(rec_type,int,height), BOOST_MULTI_INDEX_MEMBER(rec_type,marker_type,marker)>
          >
        >,
        ALE_ALLOCATOR<rec_type>
      > set_type; 
      //
      // Return types
      //

     class PointSequence {
     public:
       typedef ALE::SifterDef::IndexSequenceTraits<typename ::boost::multi_index::index<set_type, pointTag>::type,
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
         virtual const int& marker() const {return this->_itor->marker;};
         virtual const int& depth()  const {return this->_itor->depth;};
         virtual const int& height() const {return this->_itor->height;};
         //void setDegree(const int degree) {this->_index.modify(this->_itor, typename traits::rec_type::degreeAdjuster(degree));};
       };
       
       PointSequence(const PointSequence& seq)            : _index(seq._index) {};
       PointSequence(typename traits::index_type& index) : _index(index)     {};
       virtual ~PointSequence(){};
       
       virtual bool empty(){return this->_index.empty();};
       
       virtual typename traits::index_type::size_type size() {return this->_index.size();};

       virtual iterator begin() const {
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
       };
     }; // class PointSequence

     template<typename Tag_, typename Value_>
     class ValueSequence {
     public:
       typedef Value_ value_type;
       typedef ALE::SifterDef::IndexSequenceTraits<typename ::boost::multi_index::index<set_type, Tag_>::type,
                                   BOOST_MULTI_INDEX_MEMBER(rec_type, typename rec_type::point_type,point)>
       traits;
     protected:
       const typename traits::index_type& _index;
       const value_type _value;
     public:
       // Need to extend the inherited iterator to be able to extract the degree
       class iterator : public traits::iterator {
       public:
         iterator(const typename traits::iterator::itor_type& itor) : traits::iterator(itor) {};
         virtual const int& degree()  const {return this->_itor->degree;};
         virtual const int& marker()  const {return this->_itor->marker;};
         virtual const int& depth()   const {return this->_itor->depth;};
         virtual const int& height()  const {return this->_itor->height;};
       };
       
       ValueSequence(const ValueSequence& seq) : _index(seq._index), _value(seq._value) {};
       ValueSequence(typename traits::index_type& index, const value_type& value) : _index(index), _value(value) {};
       virtual ~ValueSequence(){};
       
       virtual bool empty(){return this->_index.empty();};
       
       virtual typename traits::index_type::size_type size() {return this->_index.count(this->_value);};

       virtual iterator begin() {
         return iterator(this->_index.lower_bound(this->_value));
       };
       virtual iterator end() {
         return iterator(this->_index.upper_bound(this->_value));
       };
     }; // class ValueSequence

     template<typename Tag_, typename Value_>
     class TwoValueSequence {
     public:
       typedef Value_ value_type;
       typedef ALE::SifterDef::IndexSequenceTraits<typename ::boost::multi_index::index<set_type, Tag_>::type,
                                   BOOST_MULTI_INDEX_MEMBER(rec_type, typename rec_type::point_type,point)>
       traits;
     protected:
       const typename traits::index_type& _index;
       const value_type _valueA, _valueB;
       const bool _useTwoValues;
     public:
       // Need to extend the inherited iterator to be able to extract the degree
       class iterator : public traits::iterator {
       public:
         iterator(const typename traits::iterator::itor_type& itor) : traits::iterator(itor) {};
         virtual const int& degree()  const {return this->_itor->degree;};
         virtual const int& marker()  const {return this->_itor->marker;};
       };
       
       TwoValueSequence(const TwoValueSequence& seq) : _index(seq._index), _valueA(seq._valueA), _valueB(seq._valueB), _useTwoValues(seq._useTwoValues) {};
       TwoValueSequence(typename traits::index_type& index, const value_type& valueA) : _index(index), _valueA(valueA), _valueB(value_type()), _useTwoValues(false) {};
       TwoValueSequence(typename traits::index_type& index, const value_type& valueA, const value_type& valueB) : _index(index), _valueA(valueA), _valueB(valueB), _useTwoValues(true) {};
       virtual ~TwoValueSequence(){};
       
       virtual bool empty(){return this->_index.empty();};
       
       virtual typename traits::index_type::size_type size() {
         if (this->_useTwoValues) {
           return this->_index.count(::boost::make_tuple(this->_valueA,this->_valueB));
         } else {
           return this->_index.count(::boost::make_tuple(this->_valueA));
         }
       };

       virtual iterator begin() {
         if (this->_useTwoValues) {
           return iterator(this->_index.lower_bound(::boost::make_tuple(this->_valueA,this->_valueB)));
         } else {
           return iterator(this->_index.lower_bound(::boost::make_tuple(this->_valueA)));
         }
       };
       virtual iterator end() {
         if (this->_useTwoValues) {
           return iterator(this->_index.upper_bound(::boost::make_tuple(this->_valueA,this->_valueB)));
         } else {
           return iterator(this->_index.upper_bound(::boost::make_tuple(this->_valueA)));
         }
       };
     }; // class TwoValueSequence
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

      void removePoint(const typename traits::rec_type::point_type& p) {
        /*typename ::boost::multi_index::index<set_type, typename traits::pointTag>::type& index = 
          ::boost::multi_index::get<typename traits::pointTag>(this->set);
        typename ::boost::multi_index::index<set_type, typename traits::pointTag>::type::iterator i = index.find(p);
        if (i != index.end()) { // Point exists
          index.erase(i);
        }*/
        this->set.erase(p);
      };
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
  };// namespace SieveDef

    //
    // Sieve:
    //   A Sieve is a set of {\emph arrows} connecting {\emph points} of type Point_. Thus we
    // could realize a sieve, for instance, as a directed graph. In addition, we will often
    // assume an acyclicity constraint, arriving at a DAG. Each arrow may also carry a label,
    // or {\emph color} of type Color_, and the interface allows sets of arrows to be filtered 
    // by color.
    //
    template <typename Point_, typename Marker_, typename Color_>
    class Sieve : public ALE::Sifter<Point_, Point_, Color_, ::boost::multi_index::composite_key_compare<std::less<Point_>, std::less<Color_>, std::less<Point_> >, SieveDef::RecContainer<Point_, SieveDef::Rec<Point_, Marker_> >, SieveDef::RecContainer<Point_, SieveDef::Rec<Point_, Marker_> > > {
    public:
      typedef Color_  color_type;
      typedef Point_  point_type;
      typedef Marker_ marker_type;
      typedef struct {
        typedef ALE::Sifter<Point_, Point_, Color_, ::boost::multi_index::composite_key_compare<std::less<Point_>, std::less<Color_>, std::less<Point_> >, SieveDef::RecContainer<Point_, SieveDef::Rec<Point_, Marker_> >, SieveDef::RecContainer<Point_, SieveDef::Rec<Point_, Marker_> > > baseType;
        // Encapsulated container types
        typedef typename baseType::traits::arrow_container_type arrow_container_type;
        typedef typename baseType::traits::cap_container_type   cap_container_type;
        typedef typename baseType::traits::base_container_type  base_container_type;
        // Types associated with records held in containers
        typedef typename baseType::traits::arrow_type           arrow_type;
        typedef typename baseType::traits::source_type          source_type;
        typedef typename baseType::traits::sourceRec_type       sourceRec_type;
        typedef typename baseType::traits::target_type          target_type;
        typedef typename baseType::traits::targetRec_type       targetRec_type;
        typedef typename baseType::traits::color_type           color_type;
        typedef Point_                                          point_type;
        // Convenient tag names
        typedef typename baseType::traits::supportInd           supportInd;
        typedef typename baseType::traits::coneInd              coneInd;
        typedef typename baseType::traits::arrowInd             arrowInd;
        typedef typename baseType::traits::baseInd              baseInd;
        typedef typename baseType::traits::capInd               capInd;

        //
        // Return types
        //
        typedef typename baseType::traits::arrowSequence        arrowSequence;
        typedef typename baseType::traits::coneSequence         coneSequence;
        typedef typename baseType::traits::supportSequence      supportSequence;
        typedef typename baseType::traits::baseSequence         baseSequence;
        typedef typename baseType::traits::capSequence          capSequence;
        typedef typename base_container_type::traits::template TwoValueSequence<typename base_container_type::traits::depthMarkerTag,int> depthSequence;
        typedef typename cap_container_type::traits::template TwoValueSequence<typename cap_container_type::traits::heightMarkerTag,int> heightSequence;
        typedef typename cap_container_type::traits::template ValueSequence<typename cap_container_type::traits::markerTag,marker_type> markerSequence;
      } traits;
      typedef std::set<point_type>    pointSet;
      typedef ALE::array<point_type>  pointArray;
      typedef std::set<marker_type>   markerSet;
      typedef pointSet                coneSet;
      typedef pointSet                supportSet;
      typedef pointArray              coneArray;
      typedef pointArray              supportArray;
    public:
      Sieve(MPI_Comm comm = PETSC_COMM_SELF, const int& debug = 0) : ALE::Sifter<Point_, Point_, Color_, ::boost::multi_index::composite_key_compare<std::less<Point_>, std::less<Color_>, std::less<Point_> >, SieveDef::RecContainer<Point_, SieveDef::Rec<Point_, Marker_> >, SieveDef::RecContainer<Point_, SieveDef::Rec<Point_, Marker_> > >(comm, debug), doStratify(false), maxDepth(-1), maxHeight(-1), graphDiameter(-1) {
        this->_markers = new markerSet();
        this->_meetSet = new coneSet();
        //std::cout << "["<<this->commRank()<<"]: Creating an ALE::Sieve" << std::endl;
      };
      virtual ~Sieve() {
        //std::cout << "["<<this->commRank()<<"]: Destroying an ALE::Sieve" << std::endl;
      };
      // Printing
      friend std::ostream& operator<<(std::ostream& os, Obj<Sieve<Point_,Marker_,Color_> > s) { 
        os << *s; 
        return os;
      };
    
      friend std::ostream& operator<<(std::ostream& os, Sieve<Point_,Marker_,Color_>& s) {
        Obj<typename traits::baseSequence> base = s.base();
        for(typename traits::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          Obj<typename traits::coneSequence> cone = s.cone(*b_iter);
          os << "Base point " << *b_iter << " with cone:" << std::endl;
          for(typename traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            os << "  " << *c_iter << std::endl;
          }
        }
        return os;
      };

      template<typename ostream_type>
      void view(ostream_type& os, const char* label = NULL, bool rawData = false);
      void view(const char* label = NULL, MPI_Comm comm = MPI_COMM_NULL);

      Obj<Sieve> copy() {
        Obj<Sieve> s = Sieve(this->comm(), this->debug);
        Obj<typename traits::capSequence>  cap  = this->cap();
        Obj<typename traits::baseSequence> base = this->base();

        for(typename traits::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
          s->addCapPoint(*c_iter);
        }
        for(typename traits::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          Obj<typename traits::coneSequence> cone = this->cone(*b_iter);

          for(typename traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            s->addArrow(*c_iter, *b_iter, c_iter.color());
          }
        }
        s->stratify();
        return s;
      };
      bool hasPoint(const point_type& point) {
        if (this->baseContains(point) || this->capContains(point)) return true;
        return false;
      };
    private:
      template<class InputSequence> Obj<coneSet> __nCone(Obj<InputSequence>& cone, int n, const Color_& color, bool useColor);
      template<class pointSequence> void __nCone(const Obj<pointSequence>& points, int n, const Color_& color, bool useColor, Obj<coneArray> cone, Obj<coneSet> seen);
      template<class pointSequence> void __nSupport(const Obj<pointSequence>& points, int n, const Color_& color, bool useColor, Obj<supportArray> cone, Obj<supportSet> seen);
    public:
      //
      // The basic Sieve interface (extensions to Sifter)
      //
      Obj<coneArray> nCone(const Point_& p, int n);
      Obj<coneArray> nCone(const Point_& p, int n, const Color_& color, bool useColor = true);
      template<class InputSequence> Obj<coneSet> nCone(const Obj<InputSequence>& points, int n);
      template<class InputSequence> Obj<coneSet> nCone(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor = true);

      Obj<supportArray> nSupport(const Point_& p, int n);
      Obj<supportArray> nSupport(const Point_& p, int n, const Color_& color, bool useColor = true);
      template<class InputSequence> Obj<supportSet> nSupport(const Obj<InputSequence>& points, int n);
      template<class InputSequence> Obj<supportSet> nSupport(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor = true);
    public:
      virtual bool checkArrow(const typename traits::arrow_type& a) {
        if ((this->_cap.set.find(a.source) == this->_cap.set.end()) && (this->_base.set.find(a.source) == this->_base.set.end())) return false;
        if ((this->_cap.set.find(a.target) == this->_cap.set.end()) && (this->_base.set.find(a.target) == this->_base.set.end())) return false;
        return true;
      };
      //
      // Iterated versions
      //
      Obj<supportSet> star(const Point_& p);

      Obj<supportSet> star(const Point_& p, const Color_& color);

      template<class InputSequence> 
      Obj<supportSet> star(const Obj<InputSequence>& points);

      template<class InputSequence> 
      Obj<supportSet> star(const Obj<InputSequence>& points, const Color_& color);

      Obj<supportSet> nStar(const Point_& p, int n);

      Obj<supportSet> nStar(const Point_& p, int n, const Color_& color, bool useColor = true);

      template<class InputSequence> 
      Obj<supportSet> nStar(const Obj<InputSequence>& points, int n);

      template<class InputSequence> 
      Obj<supportSet> nStar(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor = true);

    private:
      template<class InputSequence> 
      Obj<supportSet> __nStar(Obj<InputSequence>& support, int n, const Color_& color, bool useColor);

    public:
      //
      // Lattice methods
      //
      const Obj<coneSet>& meet(const Point_& p, const Point_& q);

      const Obj<coneSet>& meet(const Point_& p, const Point_& q, const Color_& color);

      template<class InputSequence> 
      const Obj<coneSet>& meet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1);

      template<class InputSequence> 
      const Obj<coneSet>& meet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, const Color_& color);

      const Obj<coneSet>& nMeet(const Point_& p, const Point_& q, int n);

      const Obj<coneSet>& nMeet(const Point_& p, const Point_& q, int n, const Color_& color, bool useColor = true);

      template<class InputSequence> 
      const Obj<coneSet>& nMeet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n);

      template<class InputSequence> 
      const Obj<coneSet>& nMeet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n, 
                                const Color_& color, bool useColor = true);

      Obj<supportSet> join(const Point_& p, const Point_& q);

      Obj<supportSet> join(const Point_& p, const Point_& q, const Color_& color);

      template<class InputSequence> 
      Obj<supportSet> join(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1);

      template<class InputSequence> 
      Obj<supportSet> join(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, const Color_& color);

      template<class InputSequence> 
      Obj<supportSet> nJoin1(const Obj<InputSequence>& chain);

      Obj<supportSet> nJoin(const Point_& p, const Point_& q, int n);

      Obj<supportSet> nJoin(const Point_& p, const Point_& q, int n, const Color_& color, bool useColor = true);

      template<class InputSequence> 
      Obj<supportSet> nJoin(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n);

      template<class InputSequence> 
      Obj<supportSet> nJoin(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n, const Color_& color, bool useColor = true);

      template<class InputSequence> 
      Obj<supportSet> nJoin(const Obj<InputSequence>& chain, const int depth);

    public:
      Obj<typename traits::depthSequence> roots() {
        return this->depthStratum(0);
      };
      Obj<typename traits::heightSequence> leaves() {
        return this->heightStratum(0);
      };
    private:
      bool doStratify;
      int  maxDepth, maxHeight, graphDiameter;
    public:
      //
      // Structural queries
      //
      int depth(); 
      int depth(const point_type& p);
      template<typename InputSequence> int depth(const Obj<InputSequence>& points);

      int height();
      int height(const point_type& p);
      template<typename InputSequence> int height(const Obj<InputSequence>& points);

      int diameter();
      int diameter(const point_type& p);

      Obj<typename traits::depthSequence> depthStratum(int d) {
        if (d == 0) {
          return typename traits::depthSequence(::boost::multi_index::get<typename traits::cap_container_type::traits::depthMarkerTag>(this->_cap.set), d);
        } else {
          return typename traits::depthSequence(::boost::multi_index::get<typename traits::base_container_type::traits::depthMarkerTag>(this->_base.set), d);
        }
      };
      Obj<typename traits::depthSequence> depthStratum(int d, marker_type m) {
        if (d == 0) {
          return typename traits::depthSequence(::boost::multi_index::get<typename traits::cap_container_type::traits::depthMarkerTag>(this->_cap.set), d, m);
        } else {
          return typename traits::depthSequence(::boost::multi_index::get<typename traits::base_container_type::traits::depthMarkerTag>(this->_base.set), d, m);
        }
      };

      Obj<typename traits::heightSequence> heightStratum(int h) {
        if (h == 0) {
          return typename traits::heightSequence(::boost::multi_index::get<typename traits::base_container_type::traits::heightMarkerTag>(this->_base.set), h);
        } else {
          return typename traits::heightSequence(::boost::multi_index::get<typename traits::cap_container_type::traits::heightMarkerTag>(this->_cap.set), h);
        }
      };
      Obj<typename traits::heightSequence> heightStratum(int h, marker_type m) {
        if (h == 0) {
          return typename traits::heightSequence(::boost::multi_index::get<typename traits::base_container_type::traits::heightMarkerTag>(this->_base.set), h, m);
        } else {
          return typename traits::heightSequence(::boost::multi_index::get<typename traits::cap_container_type::traits::heightMarkerTag>(this->_cap.set), h, m);
        }
      };

      Obj<typename traits::markerSequence> markerStratum(marker_type m);
 
      void setStratification(bool doStratify) {this->doStratify = doStratify;};

      bool getStratification() {return this->doStratify;};

      void stratify(bool show = false);
    protected:
      Obj<markerSet> _markers;
      Obj<coneSet>   _meetSet;
    public:
      //
      // Structural manipulation
      //

      struct changeMarker {
        changeMarker(int newMarker) : newMarker(newMarker) {};

        void operator()(typename traits::base_container_type::traits::rec_type& p) {
          p.marker = newMarker;
        }
      private:
        marker_type newMarker;
      };

      void setMarker(const point_type& p, const marker_type& marker);
      template<class InputSequence> void setMarker(const Obj<InputSequence>& points, const marker_type& marker);

      void clearMarkers() {this->_markers.clear();};
      Obj<markerSet> markers() {return this->_markers;};
    private:
      struct changeHeight {
        changeHeight(int newHeight) : newHeight(newHeight) {};

        void operator()(typename traits::base_container_type::traits::rec_type& p) {
          p.height = newHeight;
        }
      private:
        int newHeight;
      };
      
      template<class InputSequence> void __computeClosureHeights(const Obj<InputSequence>& points);

      struct changeDepth {
        changeDepth(int newDepth) : newDepth(newDepth) {};

        void operator()(typename traits::base_container_type::traits::rec_type& p) {
          p.depth = newDepth;
        }
      private:
        int newDepth;
      };

      template<class InputSequence> void __computeStarDepths(const Obj<InputSequence>& points);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::coneArray> Sieve<Point_,Marker_,Color_>::nCone(const Point_& p, int n) {
      return this->nCone(p, n, Color_(), false);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::coneArray> Sieve<Point_,Marker_,Color_>::nCone(const Point_& p, int n, const Color_& color, bool useColor) {
      Obj<coneArray> cone = new coneArray();
      Obj<coneSet>   seen = new coneSet();

      this->__nCone(this->cone(p), n-1, color, useColor, cone, seen);
      return cone;
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class pointSequence> 
    void Sieve<Point_,Marker_,Color_>::__nCone(const Obj<pointSequence>& points, int n, const Color_& color, bool useColor, Obj<coneArray> cone, Obj<coneSet> seen) {
      if (n == 0) {
        for(typename pointSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
          if (seen->find(*p_itor) == seen->end()) {
            cone->push_back(*p_itor);
            seen->insert(*p_itor);
          }
        }
      } else {
        typename pointSequence::iterator end = points->end();
        for(typename pointSequence::iterator p_itor = points->begin(); p_itor != end; ++p_itor) {
          if (useColor) {
            this->__nCone(this->cone(*p_itor, color), n-1, color, useColor, cone, seen);
          } else {
            this->__nCone(this->cone(*p_itor), n-1, color, useColor, cone, seen);
          }
        }
      }
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::coneSet> Sieve<Point_,Marker_,Color_>::nCone(const Obj<InputSequence>& points, int n) {
      return this->nCone(points, n, Color_(), false);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::coneSet> Sieve<Point_,Marker_,Color_>::nCone(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor ) {
      Obj<coneSet> cone = new coneSet();
      cone->insert(points->begin(), points->end());
      return this->__nCone(cone, n, color, useColor);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::coneSet> Sieve<Point_,Marker_,Color_>::__nCone(Obj<InputSequence>& cone, int n, const Color_& color, bool useColor) {
      Obj<coneSet> base = new coneSet();

      for(int i = 0; i < n; ++i) {
        Obj<coneSet> tmp = cone; cone = base; base = tmp;
        
        cone->clear();
        for(typename coneSet::iterator b_itor = base->begin(); b_itor != base->end(); ++b_itor) {
          Obj<typename traits::coneSequence> pCone;
          
          if (useColor) {
            pCone = this->cone(*b_itor, color);
          } else {
            pCone = this->cone(*b_itor);
          }
          cone->insert(pCone->begin(), pCone->end());
        }
      }
      return cone;
    };

    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportArray> Sieve<Point_,Marker_,Color_>::nSupport(const Point_& p, int n) {
      return this->nSupport(p, n, Color_(), false);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportArray> Sieve<Point_,Marker_,Color_>::nSupport(const Point_& p, int n, const Color_& color, bool useColor) {
      Obj<supportArray> cone = new supportArray();
      Obj<supportSet>   seen = new supportSet();

      this->__nSupport(this->support(p), n-1, color, useColor, cone, seen);
      return cone;
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class pointSequence> 
    void Sieve<Point_,Marker_,Color_>::__nSupport(const Obj<pointSequence>& points, int n, const Color_& color, bool useColor, Obj<supportArray> support, Obj<supportSet> seen) {
      if (n == 0) {
        for(typename pointSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
          if (seen->find(*p_itor) == seen->end()) {
            support->push_back(*p_itor);
            seen->insert(*p_itor);
          }
        }
      } else {
        typename pointSequence::iterator end = points->end();
        for(typename pointSequence::iterator p_itor = points->begin(); p_itor != end; ++p_itor) {
          if (useColor) {
            this->__nSupport(this->support(*p_itor, color), n-1, color, useColor, support, seen);
          } else {
            this->__nSupport(this->support(*p_itor), n-1, color, useColor, support, seen);
          }
        }
      }
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nSupport(const Obj<InputSequence>& points, int n) {
      return this->nSupport(points, n, Color_(), false);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nSupport(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor ) {
      Obj<supportSet> support = supportSet();
      Obj<supportSet> cap = supportSet();
      
      support->insert(points->begin(), points->end());
      for(int i = 0; i < n; ++i) {
        Obj<supportSet> tmp = support; support = cap; cap = tmp;
        
        support->clear();
        for(typename supportSet::iterator c_itor = cap->begin(); c_itor != cap->end(); ++c_itor) {
          Obj<typename traits::supportSequence> pSupport;
          
          if (useColor) {
            pSupport = this->support(*c_itor, color);
          } else {
            pSupport = this->support(*c_itor);
          }
          support->insert(pSupport->begin(), pSupport->end());
        }
      }
      return support;
    };
    //
    // Iterated versions
    //
    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::star(const Point_& p) {
      return nStar(p, this->height());
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::star(const Point_& p, const Color_& color) {
      return nStar(p, this->depth(), color);
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::star(const Obj<InputSequence>& points) {
      return nStar(points, this->height());
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::star(const Obj<InputSequence>& points, const Color_& color) {
      return nStar(points, this->height(), color);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nStar(const Point_& p, int n) {
      return this->nStar(p, n, Color_(), false);
    };
      
    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nStar(const Point_& p, int n, const Color_& color, bool useColor ) {
      Obj<supportSet> support = supportSet();
      support->insert(p);
      return this->__nStar(support, n, color, useColor);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nStar(const Obj<InputSequence>& points, int n) {
      return this->nStar(points, n, Color_(), false);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nStar(const Obj<InputSequence>& points, int n, const Color_& color, bool useColor ) {
      Obj<supportSet> support = supportSet();
      support->insert(points->begin(), points->end());
      return this->__nStar(support, n, color, useColor);
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::__nStar(Obj<InputSequence>& support, int n, const Color_& color, bool useColor) {
      Obj<supportSet> cap = supportSet();
      Obj<supportSet> star = supportSet();
      star->insert(support->begin(), support->end());
      for(int i = 0; i < n; ++i) {
        Obj<supportSet> tmp = support; support = cap; cap = tmp;
        support->clear();
        for(typename supportSet::iterator c_itor = cap->begin(); c_itor != cap->end(); ++c_itor) {
          Obj<typename traits::supportSequence> pSupport;          
          if (useColor) {
            pSupport = this->support(*c_itor, color);
          } else {
            pSupport = this->support(*c_itor);
          }
          support->insert(pSupport->begin(), pSupport->end());
          star->insert(pSupport->begin(), pSupport->end());
        }
      }
      return star;
    };

    //
    // Lattice methods
    //

    template <typename Point_, typename Marker_, typename Color_> 
    const Obj<typename Sieve<Point_,Marker_,Color_>::coneSet>& Sieve<Point_,Marker_,Color_>::meet(const Point_& p, const Point_& q) {
      return nMeet(p, q, this->depth());
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    const Obj<typename Sieve<Point_,Marker_,Color_>::coneSet>& Sieve<Point_,Marker_,Color_>::meet(const Point_& p, const Point_& q, const Color_& color) {
      return nMeet(p, q, this->depth(), color);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    const Obj<typename Sieve<Point_,Marker_,Color_>::coneSet>& Sieve<Point_,Marker_,Color_>::meet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1) {
      return nMeet(chain0, chain1, this->depth());
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    const Obj<typename Sieve<Point_,Marker_,Color_>::coneSet>& Sieve<Point_,Marker_,Color_>::meet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, const Color_& color) {
        return nMeet(chain0, chain1, this->depth(), color);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    const Obj<typename Sieve<Point_,Marker_,Color_>::coneSet>& Sieve<Point_,Marker_,Color_>::nMeet(const Point_& p, const Point_& q, int n) {
      if (n == 1) {
        std::vector<point_type> vecA, vecB;
        const Obj<typename traits::coneSequence>&     coneA  = this->cone(p);
        const typename traits::coneSequence::iterator beginA = coneA->begin();
        const typename traits::coneSequence::iterator endA   = coneA->end();
        const Obj<typename traits::coneSequence>&     coneB  = this->cone(q);
        const typename traits::coneSequence::iterator beginB = coneB->begin();
        const typename traits::coneSequence::iterator endB   = coneB->end();

        vecA.insert(vecA.begin(), beginA, endA);
        std::sort(vecA.begin(), vecA.end());
        vecB.insert(vecB.begin(), beginB, endB);
        std::sort(vecB.begin(), vecB.end());
        this->_meetSet->clear();
        std::set_intersection(vecA.begin(), vecA.end(), vecB.begin(), vecB.end(), std::insert_iterator<typename Sieve<Point_,Marker_,Color_>::coneSet>(*this->_meetSet, this->_meetSet->begin()));
        return this->_meetSet;
      }
      return nMeet(p, q, n, Color_(), false);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    const Obj<typename Sieve<Point_,Marker_,Color_>::coneSet>& Sieve<Point_,Marker_,Color_>::nMeet(const Point_& p, const Point_& q, int n, const Color_& color, bool useColor ) {
      Obj<coneSet> chain0 = new coneSet();
      Obj<coneSet> chain1 = new coneSet();
      chain0->insert(p);
      chain1->insert(q);
      return this->nMeet(chain0, chain1, n, color, useColor);
    };

    template <typename Point_, typename Marker_, typename Color_>     
    template<class InputSequence> 
    const Obj<typename Sieve<Point_,Marker_,Color_>::coneSet>& Sieve<Point_,Marker_,Color_>::nMeet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n) {
      return this->nMeet(chain0, chain1, n, Color_(), false);
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    const Obj<typename Sieve<Point_,Marker_,Color_>::coneSet>& Sieve<Point_,Marker_,Color_>::nMeet(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1,int n,const Color_& color, bool useColor){
      // The strategy is to compute the intersection of cones over the chains, remove the intersection 
      // and use the remaining two parts -- two disjoined components of the symmetric difference of cones -- as the new chains.
      // The intersections at each stage are accumulated and their union is the meet.
      // The iteration stops after n steps in addition to the meet of the initial chains or sooner if at least one of the chains is empty.
      Obj<coneSet> cone;

      this->_meetSet->clear();
      if((chain0->size() != 0) && (chain1->size() != 0)) {
        for(int i = 0; i <= n; ++i) {
          // Compute the intersection of chains and put it in meet at the same time removing it from c and cc
          std::set<point_type> intersect;
          //std::set_intersection(chain0->begin(), chain0->end(), chain1->begin(), chain1->end(), std::insert_iterator<coneSet>(meet, meet->begin()));
          std::set_intersection(chain0->begin(), chain0->end(), chain1->begin(), chain1->end(), std::insert_iterator<std::set<point_type> >(intersect, intersect.begin()));
          this->_meetSet->insert(intersect.begin(), intersect.end());
          for(typename std::set<point_type>::iterator i_iter = intersect.begin(); i_iter != intersect.end(); ++i_iter) {
            chain0->erase(chain0->find(*i_iter));
            chain1->erase(chain1->find(*i_iter));
          }
          // Replace each of the cones with a cone over it, and check if either is empty; if so, return what's in meet at the moment.
          cone = this->cone(chain0);
          chain0->insert(cone->begin(), cone->end());
          if(chain0->size() == 0) {
            break;
          }
          cone = this->cone(chain1);
          chain1->insert(cone->begin(), cone->end());
          if(chain1->size() == 0) {
            break;
          }
          // If both cones are empty, we should quit
        }
      }
      return this->_meetSet;
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::join(const Point_& p, const Point_& q) {
      return this->nJoin(p, q, this->depth());
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::join(const Point_& p, const Point_& q, const Color_& color) {
      return this->nJoin(p, q, this->depth(), color);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::join(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1) {
      return this->nJoin(chain0, chain1, this->depth());
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::join(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, const Color_& color) {
      return this->nJoin(chain0, chain1, this->depth(), color);
    };

    // Warning: I think this can be much more efficient by eliminating copies
    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nJoin1(const Obj<InputSequence>& chain) {
      Obj<supportSet> join = new supportSet(); 
      std::set<point_type> intersectA;
      std::set<point_type> intersectB;
      int p = 0;

      //std::cout << "Doing nJoin1:" << std::endl;
      for(typename InputSequence::iterator p_iter = chain->begin(); p_iter != chain->end(); ++p_iter) {
        //std::cout << "  point " << *p_iter << std::endl;
        const Obj<typename traits::supportSequence>& support = this->support(*p_iter);

        join->insert(support->begin(), support->end());
        if (p == 0) {
          intersectB.insert(support->begin(), support->end());
          p++;
        } else {
          std::set_intersection(intersectA.begin(), intersectA.end(), join->begin(), join->end(), std::insert_iterator<std::set<point_type> >(intersectB, intersectB.begin()));
        }
        intersectA.clear();
        intersectA.insert(intersectB.begin(), intersectB.end());
        intersectB.clear();
        join->clear();
        //std::cout << "  intersection:" << std::endl;
        //for(typename std::set<point_type>::iterator i_iter = intersectA.begin(); i_iter != intersectA.end(); ++i_iter) {
        //  std::cout << "    " << *i_iter << std::endl;
        //}
      }
      join->insert(intersectA.begin(), intersectA.end());
      return join;
    };

    // Warning: I think this can be much more efficient by eliminating copies
    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nJoin(const Obj<InputSequence>& chain, const int depth) {
      Obj<supportSet> join = new supportSet(); 
      std::set<point_type> intersectA;
      std::set<point_type> intersectB;
      int p = 0;

      //std::cout << "Doing nJoin1:" << std::endl;
      for(typename InputSequence::iterator p_iter = chain->begin(); p_iter != chain->end(); ++p_iter) {
        //std::cout << "  point " << *p_iter << std::endl;
        const Obj<supportArray> support = this->nSupport(*p_iter, depth);

        join->insert(support->begin(), support->end());
        if (p == 0) {
          intersectB.insert(support->begin(), support->end());
          p++;
        } else {
          std::set_intersection(intersectA.begin(), intersectA.end(), join->begin(), join->end(), std::insert_iterator<std::set<point_type> >(intersectB, intersectB.begin()));
        }
        intersectA.clear();
        intersectA.insert(intersectB.begin(), intersectB.end());
        intersectB.clear();
        join->clear();
        //std::cout << "  intersection:" << std::endl;
        //for(typename std::set<point_type>::iterator i_iter = intersectA.begin(); i_iter != intersectA.end(); ++i_iter) {
        //  std::cout << "    " << *i_iter << std::endl;
        //}
      }
      join->insert(intersectA.begin(), intersectA.end());
      return join;
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nJoin(const Point_& p, const Point_& q, int n) {
      return this->nJoin(p, q, n, Color_(), false);
    };
    
    template <typename Point_, typename Marker_, typename Color_> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nJoin(const Point_& p, const Point_& q, int n, const Color_& color, bool useColor) {
      Obj<supportSet> chain0 = supportSet();
      Obj<supportSet> chain1 = supportSet();
      chain0->insert(p);
      chain1->insert(q);
      return this->nJoin(chain0, chain1, n, color, useColor);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nJoin(const Obj<InputSequence>& chain0, const Obj<InputSequence>& chain1, int n) {
      return this->nJoin(chain0, chain1, n, Color_(), false);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    Obj<typename Sieve<Point_,Marker_,Color_>::supportSet> Sieve<Point_,Marker_,Color_>::nJoin(const Obj<InputSequence>& chain0,const Obj<InputSequence>& chain1,int n,const Color_& color,bool useColor){
      // The strategy is to compute the intersection of supports over the chains, remove the intersection 
      // and use the remaining two parts -- two disjoined components of the symmetric difference of supports -- as the new chains.
      // The intersections at each stage are accumulated and their union is the join.
      // The iteration stops after n steps in addition to the join of the initial chains or sooner if at least one of the chains is empty.
      Obj<supportSet> join = supportSet(); 
      Obj<supportSet> support;
//       std::cout << "Computing nJoin" << std::endl;
//       std::cout << "  chain 0:" << std::endl;
//       for(typename InputSequence::iterator i_iter = chain0->begin(); i_iter != chain0->end(); ++i_iter) {
//         std::cout << "    " << *i_iter << std::endl;
//       }
//       std::cout << "  chain 1:" << std::endl;
//       for(typename InputSequence::iterator i_iter = chain1->begin(); i_iter != chain1->end(); ++i_iter) {
//         std::cout << "    " << *i_iter << std::endl;
//       }
      if((chain0->size() != 0) && (chain1->size() != 0)) {
        for(int i = 0; i <= n; ++i) {
//           std::cout << "Level " << i << std::endl;
          // Compute the intersection of chains and put it in meet at the same time removing it from c and cc
          std::set<point_type> intersect;
          //std::set_intersection(chain0->begin(), chain0->end(), chain1->begin(), chain1->end(), std::insert_iterator<supportSet>(join.obj(), join->begin()));
          std::set_intersection(chain0->begin(), chain0->end(), chain1->begin(), chain1->end(), std::insert_iterator<std::set<point_type> >(intersect, intersect.begin()));
          join->insert(intersect.begin(), intersect.end());
//           std::cout << "  Join set:" << std::endl;
//           for(typename supportSet::iterator i_iter = join->begin(); i_iter != join->end(); ++i_iter) {
//             std::cout << "    " << *i_iter << std::endl;
//           }
          for(typename std::set<point_type>::iterator i_iter = intersect.begin(); i_iter != intersect.end(); ++i_iter) {
            chain0->erase(chain0->find(*i_iter));
            chain1->erase(chain1->find(*i_iter));
          }
          // Replace each of the supports with the support over it, and check if either is empty; if so, return what's in join at the moment.
          support = this->support(chain0);
          chain0->insert(support->begin(), support->end());
          if(chain0->size() == 0) {
            break;
          }
//           std::cout << "  chain 0:" << std::endl;
//           for(typename InputSequence::iterator i_iter = chain0->begin(); i_iter != chain0->end(); ++i_iter) {
//             std::cout << "    " << *i_iter << std::endl;
//           }
          support = this->support(chain1);
          chain1->insert(support->begin(), support->end());
          if(chain1->size() == 0) {
            break;
          }
//           std::cout << "  chain 1:" << std::endl;
//           for(typename InputSequence::iterator i_iter = chain1->begin(); i_iter != chain1->end(); ++i_iter) {
//             std::cout << "    " << *i_iter << std::endl;
//           }
          // If both supports are empty, we should quit
        }
      }
      return join;
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<typename ostream_type>
    void Sieve<Point_,Marker_,Color_>::view(ostream_type& os, const char* label, bool rawData){
      if(label != NULL) {
        os << "Viewing Sieve '" << label << "':" << std::endl;
      } 
      else {
        os << "Viewing a Sieve:" << std::endl;
      }
      if(!rawData) {
        os << "cap --> base:" << std::endl;
        Obj<typename traits::capSequence> cap = this->cap();
        for(typename traits::capSequence::iterator capi = cap->begin(); capi != cap->end(); ++capi) {
          Obj<typename traits::supportSequence> supp = this->support(*capi);
          for(typename traits::supportSequence::iterator suppi = supp->begin(); suppi != supp->end(); ++suppi) {
            os << *capi << "--(" << suppi.color() << ")-->" << *suppi << std::endl;
          }
        }
        os << "base <-- cap:" << std::endl;
        Obj<typename traits::baseSequence> base = this->base();
        for(typename traits::baseSequence::iterator basei = base->begin(); basei != base->end(); ++basei) {
          Obj<typename traits::coneSequence> cone = this->cone(*basei);
          for(typename traits::coneSequence::iterator conei = cone->begin(); conei != cone->end(); ++conei) {
            os << *basei <<  "<--(" << conei.color() << ")--" << *conei << std::endl;
          }
        }
#if 0
        os << "cap --> (outdegree, marker, depth, height):" << std::endl;
        for(typename traits::capSequence::iterator capi = cap->begin(); capi != cap->end(); ++capi) {
          os << *capi <<  "-->" << capi.degree() << ", " << capi.marker() << ", " << capi.depth() << ", " << capi.height() << std::endl;
        }
        os << "base --> (indegree, marker, depth, height):" << std::endl;
        for(typename traits::baseSequence::iterator basei = base->begin(); basei != base->end(); ++basei) {
          os << *basei <<  "-->" << basei.degree() << ", " << basei.marker() << ", " << basei.depth() << ", " << basei.height() << std::endl;
        }
#endif
      }
      else {
        os << "'raw' arrow set:" << std::endl;
        for(typename traits::arrow_container_type::set_type::iterator ai = this->_arrows.set.begin(); ai != this->_arrows.set.end(); ai++)
        {
          typename traits::arrow_type arr = *ai;
          os << arr << std::endl;
        }
        os << "'raw' base set:" << std::endl;
        for(typename traits::base_container_type::set_type::iterator bi = this->_base.set.begin(); bi != this->_base.set.end(); bi++) 
        {
          typename traits::base_container_type::traits::rec_type bp = *bi;
          os << bp << std::endl;
        }
        os << "'raw' cap set:" << std::endl;
        for(typename traits::cap_container_type::set_type::iterator ci = this->_cap.set.begin(); ci != this->_cap.set.end(); ci++) 
        {
          typename traits::cap_container_type::traits::rec_type cp = *ci;
          os << cp << std::endl;
        }
      }
    };
    template <typename Point_, typename Marker_, typename Color_> 
    void Sieve<Point_,Marker_,Color_>::view(const char* label, MPI_Comm comm) {
        ostringstream txt;

        if (this->debug()) {
          std::cout << "viewing a Sieve, comm = " << this->comm() << ", commRank = " << this->commRank() << std::endl;
        }
        if(label != NULL) {
          if(this->commRank() == 0) {
            txt << "viewing Sieve :'" << label << "'" << std::endl;
          }
        } 
        else {
          if(this->commRank() == 0) {
            txt << "viewing a Sieve" << std::endl;
          }
        }
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
          const typename traits::supportSequence::iterator suppEnd = supp->end();

          for(typename traits::supportSequence::iterator suppi = supp->begin(); suppi != suppEnd; suppi++) {
            txt << "[" << this->commRank() << "]: " << *capi << "--(" << suppi.color() << ")-->" << *suppi << std::endl;
          }
        }
        PetscSynchronizedPrintf(this->comm(), txt.str().c_str());
        PetscSynchronizedFlush(this->comm());
        //
        ostringstream txt1;
        if(this->commRank() == 0) {
          txt1 << "base --> cap:\n";
        }
        if(base.empty()) {
          txt1 << "[" << this->commRank() << "]: empty" << std::endl; 
        }
        for(typename traits::baseSequence::iterator basei = base.begin(); basei != base.end(); basei++) {
          const Obj<typename traits::coneSequence>& cone = this->cone(*basei);
          const typename traits::coneSequence::iterator coneEnd = cone->end();

          for(typename traits::coneSequence::iterator conei = cone->begin(); conei != coneEnd; conei++) {
            txt1 << "[" << this->commRank() << "]: " << *basei << "<--(" << conei.color() << ")--" << *conei << std::endl;
          }
        }
        //
        PetscSynchronizedPrintf(this->comm(), txt1.str().c_str());
        PetscSynchronizedFlush(this->comm());
#if 0
        //
        ostringstream txt2;
        if(this->commRank() == 0) {
          txt2 << "cap <point, outdegree, marker, depth, height>:\n";
        }
        txt2 << "[" << this->commRank() << "]:  [";
        for(typename traits::capSequence::iterator capi = cap.begin(); capi != cap.end(); capi++) {
          txt2 << " <" << *capi << ", " << capi.degree() << ", " << capi.marker() << ", " << capi.depth() << ", " << capi.height() << ">";
        }
        txt2 << " ]" << std::endl;
        //
        PetscSynchronizedPrintf(this->comm(), txt2.str().c_str());
        PetscSynchronizedFlush(this->comm());
        //
        ostringstream txt3;
        if(this->commRank() == 0) {
          txt3 << "base <point, indegree, marker, depth, height>:\n";
        }
        txt3 << "[" << this->commRank() << "]:  [";
        for(typename traits::baseSequence::iterator basei = base.begin(); basei != base.end(); basei++) {
          txt3 << " <" << *basei << "," << basei.degree() << ", " << basei.marker() << ", " << basei.depth() << ", " << basei.height() << ">";
        }
        txt3 << " ]" << std::endl;
        //
        PetscSynchronizedPrintf(this->comm(), txt3.str().c_str());
        PetscSynchronizedFlush(this->comm());
#endif
    };
    //
    // Structural queries
    //
    template <typename Point_, typename Marker_, typename Color_> 
    void Sieve<Point_,Marker_,Color_>::setMarker(const point_type& p, const marker_type& marker) {
      typename ::boost::multi_index::index<typename traits::base_container_type::set_type,typename traits::base_container_type::traits::pointTag>::type& bIndex = ::boost::multi_index::get<typename traits::base_container_type::traits::pointTag>(this->_base.set);
      typename ::boost::multi_index::index<typename traits::cap_container_type::set_type,typename traits::cap_container_type::traits::pointTag>::type& cIndex = ::boost::multi_index::get<typename traits::cap_container_type::traits::pointTag>(this->_cap.set);

      if (bIndex.find(p) != bIndex.end()) {
        bIndex.modify(bIndex.find(p), changeMarker(marker));
      }
      if (cIndex.find(p) != cIndex.end()) {
        cIndex.modify(cIndex.find(p), changeMarker(marker));
      }
      this->_markers->insert(marker);
    };

    template <typename Point_, typename Marker_, typename Color_>
    template <typename Sequence>
    void Sieve<Point_,Marker_,Color_>::setMarker(const Obj<Sequence>& points, const marker_type& marker) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setMarker(*p_iter, marker);
      }
    };

    template <typename Point_, typename Marker_, typename Color_> 
    int Sieve<Point_,Marker_,Color_>::depth() {
      return this->maxDepth;
    }; 
    template <typename Point_, typename Marker_, typename Color_> 
    int Sieve<Point_,Marker_,Color_>::depth(const point_type& p) {
      const typename ::boost::multi_index::index<typename traits::base_container_type::set_type,typename traits::cap_container_type::traits::pointTag>::type& i = ::boost::multi_index::get<typename traits::base_container_type::traits::pointTag>(this->_base.set);
      if (i.find(p) != i.end()) {
        return i.find(p)->depth;
      }
      return 0;
    };
    template <typename Point_, typename Marker_, typename Color_> 
    template<typename InputSequence>
    int Sieve<Point_,Marker_,Color_>::depth(const Obj<InputSequence>& points) {
      const typename ::boost::multi_index::index<typename traits::base_container_type::set_type,typename traits::cap_container_type::traits::pointTag>::type& i = ::boost::multi_index::get<typename traits::base_container_type::traits::pointTag>(this->_base.set);
      int maxDepth = 0;
      
      for(typename InputSequence::iterator iter = points->begin(); iter != points->end(); ++iter) {
        if (i.find(*iter) != i.end()) {
          maxDepth = std::max(maxDepth, i.find(*iter)->depth);
        }
      }
      return maxDepth;
    };
    template <typename Point_, typename Marker_, typename Color_> 
    int Sieve<Point_,Marker_,Color_>::height() {
      return this->maxHeight;
    }; 
    template <typename Point_, typename Marker_, typename Color_> 
    int Sieve<Point_,Marker_,Color_>::height(const point_type& p) {
      const typename ::boost::multi_index::index<typename traits::cap_container_type::set_type,typename traits::cap_container_type::traits::pointTag>::type& i = ::boost::multi_index::get<typename traits::cap_container_type::traits::pointTag>(this->_cap.set);
      if (i.find(p) != i.end()) {
        return i.find(p)->height;
      }
      return 0;
    };
    template <typename Point_, typename Marker_, typename Color_> 
    template<typename InputSequence>
    int Sieve<Point_,Marker_,Color_>::height(const Obj<InputSequence>& points) {
      const typename ::boost::multi_index::index<typename traits::cap_container_type::set_type,typename traits::cap_container_type::traits::pointTag>::type& i = ::boost::multi_index::get<typename traits::cap_container_type::traits::pointTag>(this->_cap.set);
      int maxHeight = 0;
      
      for(typename InputSequence::iterator iter = points->begin(); iter != points->end(); ++iter) {
        if (i.find(*iter) != i.end()) {
          maxHeight = std::max(maxHeight, i.find(*iter)->height);
        }
      }
      return maxHeight;
    };

    template <typename Point_, typename Marker_, typename Color_> 
    int Sieve<Point_,Marker_,Color_>::diameter() {
      int globalDiameter;
      int ierr = MPI_Allreduce(&this->graphDiameter, &globalDiameter, 1, MPI_INT, MPI_MAX, this->comm());
      CHKMPIERROR(ierr, ERRORMSG("Error in MPI_Allreduce"));
      return globalDiameter;
    };
    template <typename Point_, typename Marker_, typename Color_> 
    int Sieve<Point_,Marker_,Color_>::diameter(const point_type& p) {
      return this->depth(p) + this->height(p);
    };

    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    void Sieve<Point_,Marker_,Color_>::__computeClosureHeights(const Obj<InputSequence>& points) {
      typename ::boost::multi_index::index<typename traits::cap_container_type::set_type,typename traits::cap_container_type::traits::pointTag>::type& index = ::boost::multi_index::get<typename traits::cap_container_type::traits::pointTag>(this->_cap.set);
      typename ::boost::multi_index::index<typename traits::base_container_type::set_type,typename traits::base_container_type::traits::pointTag>::type& bIndex = ::boost::multi_index::get<typename traits::base_container_type::traits::pointTag>(this->_base.set);
      Obj<coneSet> modifiedPoints = coneSet();
      
      for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
        // Compute the max height of the points in the support of p, and add 1
        int h0 = this->height(*p_itor);
        int h1 = this->height(this->support(*p_itor)) + 1;
        if(h1 != h0) {
          typename ::boost::multi_index::index<typename traits::base_container_type::set_type,typename traits::base_container_type::traits::pointTag>::type::iterator bIter = bIndex.find(*p_itor);

          index.modify(index.find(*p_itor), changeHeight(h1));
          if (bIter != bIndex.end()) {
            bIndex.modify(bIter, changeHeight(h1));
          }
          if (h1 > this->maxHeight) this->maxHeight = h1;
          modifiedPoints->insert(*p_itor);
        }
      }
      // FIX: We would like to avoid the copy here with cone()
      if(modifiedPoints->size() > 0) {
        this->__computeClosureHeights(this->cone(modifiedPoints));
      }
    };
    template <typename Point_, typename Marker_, typename Color_> 
    template<class InputSequence> 
    void Sieve<Point_,Marker_,Color_>::__computeStarDepths(const Obj<InputSequence>& points) {
      typename ::boost::multi_index::index<typename traits::base_container_type::set_type,typename traits::base_container_type::traits::pointTag>::type& index = ::boost::multi_index::get<typename traits::base_container_type::traits::pointTag>(this->_base.set);
      typename ::boost::multi_index::index<typename traits::cap_container_type::set_type,typename traits::cap_container_type::traits::pointTag>::type& cIndex = ::boost::multi_index::get<typename traits::cap_container_type::traits::pointTag>(this->_cap.set);
      Obj<supportSet> modifiedPoints = supportSet();
      for(typename InputSequence::iterator p_itor = points->begin(); p_itor != points->end(); ++p_itor) {
        // Compute the max depth of the points in the support of p, and add 1
        int d0 = this->depth(*p_itor);
        int d1 = this->depth(this->cone(*p_itor)) + 1;
        if(d1 != d0) {
          typename ::boost::multi_index::index<typename traits::cap_container_type::set_type,typename traits::cap_container_type::traits::pointTag>::type::iterator cIter = cIndex.find(*p_itor);

          index.modify(index.find(*p_itor), changeDepth(d1));
          if (cIter != cIndex.end()) {
            cIndex.modify(cIter, changeDepth(d1));
          }
          if (d1 > this->maxDepth) this->maxDepth = d1;
          modifiedPoints->insert(*p_itor);
        }
      }
      // FIX: We would like to avoid the copy here with cone()
      if(modifiedPoints->size() > 0) {
        this->__computeStarDepths(this->support(modifiedPoints));
      }
    };
    #undef __FUNCT__
    #define __FUNCT__ "Sieve::stratify"
    template <typename Point_, typename Marker_, typename Color_> 
    void Sieve<Point_,Marker_,Color_>::stratify(bool show) {
      ALE_LOG_EVENT_BEGIN;
      // FIX: We would like to avoid the copy here with cone() and support()
      this->__computeClosureHeights(this->cone(this->leaves()));
      this->__computeStarDepths(this->support(this->roots()));
      
      Obj<typename traits::capSequence> base = this->base();

      for(typename traits::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        maxDepth = std::max(maxDepth, b_iter.depth());
        //b_iter.setDegree(this->cone(*b_iter)->size());
        this->_base.adjustDegree(*b_iter, this->cone(*b_iter)->size());
      }
      Obj<typename traits::capSequence> cap = this->cap();

      for(typename traits::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
        maxHeight = std::max(maxHeight, c_iter.height());
        //c_iter.setDegree(this->support(*c_iter)->size());
        this->_cap.adjustDegree(*c_iter, this->support(*c_iter)->size());
      }
      if (this->debug() || show) {
//         const typename ::boost::multi_index::index<StratumSet,point>::type& points = ::boost::multi_index::get<point>(this->strata);
//         for(typename ::boost::multi_index::index<StratumSet,point>::type::iterator i = points.begin(); i != points.end(); i++) {
//           std::cout << *i << std::endl;
//         }
      }
      ALE_LOG_EVENT_END;
    };
    //
    // Structural manipulation
    //
  
} // namespace ALE

#endif
