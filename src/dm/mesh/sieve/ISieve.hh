#ifndef included_ALE_ISieve_hh
#define included_ALE_ISieve_hh

#ifndef  included_ALE_hh
#include <ALE.hh>
#endif

namespace ALE {
  template<typename Point_, typename Alloc_ = malloc_allocator<Point_> >
  class Interval {
  public:
    typedef Point_            point_type;
    typedef Alloc_            alloc_type;
  public:
    class const_iterator {
    protected:
      point_type _p;
    public:
      const_iterator(const point_type p): _p(p) {};
      ~const_iterator() {};
    public:
      const_iterator& operator=(const const_iterator& iter) {this->_p = iter._p;};
      bool operator==(const const_iterator& iter) const {return this->_p == iter._p;};
      bool operator!=(const const_iterator& iter) const {return this->_p != iter._p;};
      const_iterator& operator++() {++this->_p; return *this;}
      const_iterator& operator++(int) {
        const_iterator tmp(*this);
        ++(*this);
        return tmp;
      };
      const_iterator& operator--() {--this->_p; return *this;}
      const_iterator& operator--(int) {
        const_iterator tmp(*this);
        --(*this);
        return tmp;
      };
      point_type operator*() const {return this->_p;};
    };
  protected:
    point_type _min, _max;
  public:
    Interval(): _min(point_type()), _max(point_type()) {};
    Interval(const point_type& min, const point_type& max): _min(min), _max(max) {};
    Interval(const Interval& interval): _min(interval.min()), _max(interval.max()) {};
  public:
    Interval& operator=(const Interval& interval) {_min = interval.min(); _max = interval.max(); return *this;};
    friend std::ostream& operator<<(std::ostream& stream, const Interval& interval) {
      stream << "(" << interval.min() << ", " << interval.max() << ")";
      return stream;
    };
  public:
    const_iterator begin() const {return const_iterator(this->_min);};
    const_iterator end() const {return const_iterator(this->_max);};
    size_t size() const {return this->_max - this->_min;};
    size_t count(const point_type& p) const {return ((p >= _min) && (p < _max)) ? 1 : 0;};
    point_type min() const {return this->_min;};
    point_type max() const {return this->_max;};
    bool hasPoint(const point_type& point) const {
      if (point < this->_min || point >= this->_max) return false;
      return true;
    };
    void checkPoint(const point_type& point) const {
      if (point < this->_min || point >= this->_max) {
        ostringstream msg;
        msg << "Invalid point " << point << " not in " << *this << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
  };

  template<typename Source_, typename Target_>
  struct SimpleArrow {
    typedef Source_ source_type;
    typedef Target_ target_type;
    const source_type source;
    const target_type target;
    SimpleArrow(const source_type& s, const target_type& t) : source(s), target(t) {};
    template<typename OtherSource_, typename OtherTarget_>
    struct rebind {
      typedef SimpleArrow<OtherSource_, OtherTarget_> other;
    };
    struct flip {
      typedef SimpleArrow<target_type, source_type> other;
      other arrow(const SimpleArrow& a) {return type(a.target, a.source);};
    };
    friend bool operator<(const SimpleArrow& x, const SimpleArrow& y) {
      return ((x.source < y.source) || ((x.source == y.source) && (x.target < y.target)));
    };
    friend std::ostream& operator<<(std::ostream& os, const SimpleArrow& a) {
      os << a.source << " ----> " << a.target;
      return os;
    }
  };

  namespace ISieveVisitor {
    template<typename Sieve>
    class NullVisitor {
    public:
      void visitArrow(const typename Sieve::arrow_type&) {};
      void visitPoint(const typename Sieve::point_type&) {};
      void visitArrow(const typename Sieve::arrow_type&, const int orientation) {};
      void visitPoint(const typename Sieve::point_type&, const int orientation) {};
    };
    class PrintVisitor {
    protected:
      ostringstream& os;
      const int      rank;
    public:
      PrintVisitor(ostringstream& s, const int rank = 0) : os(s), rank(rank) {};
      template<typename Arrow>
      void visitArrow(const Arrow& arrow) const {
        this->os << "["<<this->rank<<"]: " << arrow << std::endl;
      };
      template<typename Point>
      void visitPoint(const Point&) const {};
    };
    class ReversePrintVisitor : public PrintVisitor {
    public:
      ReversePrintVisitor(ostringstream& s, const int rank) : PrintVisitor(s, rank) {};
      template<typename Arrow>
      void visitArrow(const Arrow& arrow) const {
        this->os << "["<<this->rank<<"]: " << arrow.target << "<----" << arrow.source << std::endl;
      };
      template<typename Arrow>
      void visitArrow(const Arrow& arrow, const int orientation) const {
        this->os << "["<<this->rank<<"]: " << arrow.target << "<----" << arrow.source << ": " << orientation << std::endl;
      };
      template<typename Point>
      void visitPoint(const Point&) const {};
      template<typename Point>
      void visitPoint(const Point&, const int) const {};
    };
    template<typename Sieve, typename Visitor = NullVisitor<Sieve> >
    class PointRetriever {
    public:
      typedef typename Sieve::point_type point_type;
      typedef typename Sieve::arrow_type arrow_type;
      typedef std::pair<point_type,int>  oriented_point_type;
    protected:
      const bool           unique;
      size_t               i, o;
      Visitor             *visitor;
      size_t               size;
      point_type          *points;
      oriented_point_type *oPoints;
    protected:
      inline virtual bool accept(const point_type& point) {return true;};
    public:
      PointRetriever() : unique(false), i(0), o(0) {
        this->size    = -1;
        this->points  = NULL;
        this->oPoints = NULL;
      };
      PointRetriever(const size_t size, const bool unique = false) : unique(unique), i(0), o(0) {
        static Visitor nV;
        this->visitor = &nV;
        this->points  = NULL;
        this->oPoints = NULL;
        this->setSize(size);
      };
      PointRetriever(const size_t size, Visitor& v, const bool unique = false) : unique(unique), i(0), o(0), visitor(&v) {
        this->points  = NULL;
        this->oPoints = NULL;
        this->setSize(size);
      };
      virtual ~PointRetriever() {
        delete [] this->points;
        delete [] this->oPoints;
        this->points  = NULL;
        this->oPoints = NULL;
      };
      void visitArrow(const arrow_type& arrow) {
        this->visitor->visitArrow(arrow);
      };
      void visitArrow(const arrow_type& arrow, const int orientation) {
        this->visitor->visitArrow(arrow, orientation);
      };
      void visitPoint(const point_type& point) {
        if (i >= size) {
          ostringstream msg;
          msg << "Too many points (>" << size << ")for PointRetriever visitor";
          throw ALE::Exception(msg.str().c_str());
        }
        if (this->accept(point)) {
          if (this->unique) {
            size_t p;
            for(p = 0; p < i; ++p) {if (points[p] == point) break;}
            if (p != i) return;
          }
          points[i++] = point;
          this->visitor->visitPoint(point);
        }
      };
      void visitPoint(const point_type& point, const int orientation) {
        if (o >= size) {
          ostringstream msg;
          msg << "Too many ordered points (>" << size << ")for PointRetriever visitor";
          throw ALE::Exception(msg.str().c_str());
        }
        if (this->accept(point)) {
          if (this->unique) {
            size_t p;
            for(p = 0; p < i; ++p) {if (points[p] == point) break;}
            if (p != i) return;
          }
          points[i++]  = point;
          oPoints[o++] = oriented_point_type(point, orientation);
          this->visitor->visitPoint(point, orientation);
        }
      };
    public:
      const size_t               getSize() const {return this->i;};
      const point_type          *getPoints() const {return this->points;};
      const size_t               getOrientedSize() const {return this->o;};
      const oriented_point_type *getOrientedPoints() const {return this->oPoints;};
      void clear() {this->i = this->o = 0;};
      void setSize(const size_t s) {
        if (this->points) {
          delete [] this->points;
          delete [] this->oPoints;
        }
        this->size    = s;
        this->points  = new point_type[this->size];
        this->oPoints = new oriented_point_type[this->size];
      };
    };
    template<typename Sieve, typename Visitor = NullVisitor<Sieve> >
    class NConeRetriever : public PointRetriever<Sieve,Visitor> {
    public:
      typedef PointRetriever<Sieve,Visitor>           base_type;
      typedef typename Sieve::point_type              point_type;
      typedef typename Sieve::arrow_type              arrow_type;
      typedef typename base_type::oriented_point_type oriented_point_type;
    protected:
      const Sieve& sieve;
    protected:
      inline virtual bool accept(const point_type& point) {
        if (!this->sieve.getConeSize(point))
          return true;
        return false;
      };
    public:
      NConeRetriever(const Sieve& s, const size_t size) : PointRetriever<Sieve,Visitor>(size, true), sieve(s) {};
      NConeRetriever(const Sieve& s, const size_t size, Visitor& v) : PointRetriever<Sieve,Visitor>(size, v, true), sieve(s) {};
      virtual ~NConeRetriever() {};
    };
    template<typename Sieve, typename Set, typename Renumbering>
    class FilteredPointRetriever {
    public:
      typedef typename Sieve::point_type point_type;
      typedef typename Sieve::arrow_type arrow_type;
      typedef std::pair<point_type,int>  oriented_point_type;
    protected:
      const Set&   pointSet;
      Renumbering& renumbering;
      const size_t size;
      size_t       i;
      point_type  *points;
    public:
      FilteredPointRetriever(const Set& s, Renumbering& r, const size_t size) : pointSet(s), renumbering(r), size(size), i(0) {
        this->points = new point_type[this->size];
      };
      ~FilteredPointRetriever() {delete [] this->points;};
      void visitArrow(const arrow_type& arrow) {};
      void visitPoint(const point_type& point) {
        if (i >= size) throw ALE::Exception("Too many points for FilteredPointRetriever visitor");
        if (this->pointSet.find(point) == this->pointSet.end()) return;
        points[i++] = this->renumbering[point];
      };
    public:
      const size_t      getSize() const {return this->i;};
      const point_type *getPoints() const {return this->points;};
      void clear() {this->i = 0;};
    };
    template<typename Sieve, int size, typename Visitor = NullVisitor<Sieve> >
    class ArrowRetriever {
    public:
      typedef typename Sieve::point_type point_type;
      typedef typename Sieve::arrow_type arrow_type;
      typedef std::pair<arrow_type,int>  oriented_arrow_type;
    protected:
      arrow_type          arrows[size];
      oriented_arrow_type oArrows[size];
      size_t              i, o;
      Visitor            *visitor;
    public:
      ArrowRetriever() : i(0), o(0) {
        static Visitor nV;
        this->visitor = &nV;
      };
      ArrowRetriever(Visitor& v) : i(0), o(0), visitor(&v) {};
      void visitArrow(const typename Sieve::arrow_type& arrow) {
        if (i >= size) throw ALE::Exception("Too many arrows for visitor");
        arrows[i++] = arrow;
        this->visitor->visitArrow(arrow);
      };
      void visitArrow(const typename Sieve::arrow_type& arrow, const int orientation) {
        if (o >= size) throw ALE::Exception("Too many arrows for visitor");
        oArrows[o++] = oriented_arrow_type(arrow, orientation);
        this->visitor->visitArrow(arrow, orientation);
      };
      void visitPoint(const point_type& point) {
        this->visitor->visitPoint(point);
      };
      void visitPoint(const point_type& point, const int orientation) {
        this->visitor->visitPoint(point, orientation);
      };
    public:
      const size_t               getSize() const {return this->i;};
      const point_type          *getArrows() const {return this->arrows;};
      const size_t               getOrientedSize() const {return this->o;};
      const oriented_arrow_type *getOrientedArrows() const {return this->oArrows;};
      void clear() {this->i = this->o = 0;};
    };
    template<typename Sieve, typename Visitor>
    class ConeVisitor {
    protected:
      const Sieve& sieve;
      Visitor&     visitor;
      bool         useSource;
    public:
      ConeVisitor(const Sieve& s, Visitor& v, bool useSource = false) : sieve(s), visitor(v), useSource(useSource) {};
      void visitPoint(const typename Sieve::point_type& point) {
        this->sieve.cone(point, visitor);
      };
      void visitArrow(const typename Sieve::arrow_type& arrow) {};
    };
    template<typename Sieve, typename Visitor>
    class OrientedConeVisitor {
    protected:
      const Sieve& sieve;
      Visitor&     visitor;
      bool         useSource;
    public:
      OrientedConeVisitor(const Sieve& s, Visitor& v, bool useSource = false) : sieve(s), visitor(v), useSource(useSource) {};
      void visitPoint(const typename Sieve::point_type& point) {
        this->sieve.orientedCone(point, visitor);
      };
      void visitArrow(const typename Sieve::arrow_type& arrow) {};
    };
    template<typename Sieve, typename Visitor>
    class SupportVisitor {
    protected:
      const Sieve& sieve;
      Visitor&     visitor;
      bool         useSource;
    public:
      SupportVisitor(const Sieve& s, Visitor& v, bool useSource = true) : sieve(s), visitor(v), useSource(useSource) {};
      void visitPoint(const typename Sieve::point_type& point) {
        this->sieve.support(point, visitor);
      };
      void visitArrow(const typename Sieve::arrow_type& arrow) {};
    };
    template<typename Sieve, typename Visitor = NullVisitor<Sieve> >
    class TransitiveClosureVisitor {
    public:
      typedef Visitor visitor_type;
    protected:
      const Sieve& sieve;
      Visitor&     visitor;
      bool         isCone;
      std::set<typename Sieve::point_type> seen;
    public:
      TransitiveClosureVisitor(const Sieve& s, Visitor& v) : sieve(s), visitor(v), isCone(true) {};
      void visitPoint(const typename Sieve::point_type& point) const {};
      void visitArrow(const typename Sieve::arrow_type& arrow) {
        if (this->isCone) {
          if (this->seen.find(arrow.target) == this->seen.end()) {
            this->seen.insert(arrow.target);
            this->visitor.visitPoint(arrow.target);
          }
          this->visitor.visitArrow(arrow);
          if (this->seen.find(arrow.source) == this->seen.end()) {
            if (this->sieve.getConeSize(arrow.source)) {
              this->sieve.cone(arrow.source, *this);
            } else {
              this->seen.insert(arrow.source);
              this->visitor.visitPoint(arrow.source);
            }
          }
        } else {
          if (this->seen.find(arrow.source) == this->seen.end()) {
            this->seen.insert(arrow.source);
            this->visitor.visitPoint(arrow.source);
          }
          this->visitor.visitArrow(arrow);
          if (this->seen.find(arrow.target) == this->seen.end()) {
            if (this->sieve.getSupportSize(arrow.target)) {
              this->sieve.support(arrow.target, *this);
            } else {
              this->seen.insert(arrow.target);
              this->visitor.visitPoint(arrow.target);
            }
          }
        }
      };
    public:
      bool getIsCone() const {return this->isCone;};
      void setIsCone(const bool isCone) {this->isCone = isCone;};
      const std::set<typename Sieve::point_type>& getPoints() const {return this->seen;};
      void clear() {this->seen.clear();};
    };
    template<typename Sieve, typename Section>
    class SizeVisitor {
    protected:
      const Section& section;
      int            size;
    public:
      SizeVisitor(const Section& s) : section(s), size(0) {};
      void visitPoint(const typename Sieve::point_type& point) {
        this->size += section.getConstrainedFiberDimension(point);
      };
      void visitArrow(const typename Sieve::arrow_type&) {};
    public:
      int getSize() {return this->size;};
    };
    template<typename Sieve, typename Section>
    class SizeWithBCVisitor {
    protected:
      const Section& section;
      int            size;
    public:
      SizeWithBCVisitor(const Section& s) : section(s), size(0) {};
      void visitPoint(const typename Sieve::point_type& point) {
        this->size += section.getFiberDimension(point);
      };
      void visitArrow(const typename Sieve::arrow_type&) {};
    public:
      int getSize() {return this->size;};
    };
    template<typename Section>
    class RestrictVisitor {
    public:
      typedef typename Section::value_type value_type;
    protected:
      const Section& section;
      int            size;
      int            i;
      value_type    *values;
      bool           allocated;
    public:
      RestrictVisitor(const Section& s, const int size) : section(s), size(size), i(0) {
        this->values    = new value_type[this->size];
        this->allocated = true;
      };
      RestrictVisitor(const Section& s, const int size, value_type *values) : section(s), size(size), i(0) {
        this->values    = values;
        this->allocated = false;
      };
      ~RestrictVisitor() {if (this->allocated) {delete [] this->values;}};
      template<typename Point>
      void visitPoint(const Point& point, const int orientation) {
        const int         dim = section.getFiberDimension(point);
        if (i+dim > size) {throw ALE::Exception("Too many values for RestrictVisitor.");}
        const value_type *v   = section.restrictPoint(point);

        if (orientation >= 0) {
          for(int d = 0; d < dim; ++d, ++i) {
            this->values[i] = v[d];
          }
        } else {
          for(int d = dim-1; d >= 0; --d, ++i) {
            this->values[i] = v[d];
          }
        }
      };
      template<typename Arrow>
      void visitArrow(const Arrow& arrow, const int orientation) {};
    public:
      const value_type *getValues() const {return this->values;};
      int  getSize() const {return this->i;};
      int  getMaxSize() const {return this->size;};
      void ensureSize(const int size) {
        this->clear();
        if (size > this->size) {
          this->size = size;
          if (this->allocated) {delete [] this->values;}
          this->values = new value_type[this->size];
          this->allocated = true;
        }
      };
      void clear() {this->i = 0;};
    };
    template<typename Section>
    class UpdateVisitor {
    public:
      typedef typename Section::value_type value_type;
    protected:
      Section&          section;
      const value_type *values;
      int               i;
    public:
      UpdateVisitor(Section& s, const value_type *v) : section(s), values(v), i(0) {};
      template<typename Point>
      void visitPoint(const Point& point, const int orientation) {
        const int dim = section.getFiberDimension(point);
        this->section.updatePoint(point, &this->values[this->i], orientation);
        this->i += dim;
      };
      template<typename Arrow>
      void visitArrow(const Arrow& arrow, const int orientation) {};
    };
    template<typename Section>
    class UpdateAddVisitor {
    public:
      typedef typename Section::value_type value_type;
    protected:
      Section&          section;
      const value_type *values;
      int               i;
    public:
      UpdateAddVisitor(Section& s, const value_type *v) : section(s), values(v), i(0) {};
      template<typename Point>
      void visitPoint(const Point& point, const int orientation) {
        const int dim = section.getFiberDimension(point);
        this->section.updateAddPoint(point, &this->values[this->i], orientation);
        this->i += dim;
      };
      template<typename Arrow>
      void visitArrow(const Arrow& arrow, const int orientation) {};
    };
    template<typename Section, typename Order, typename Value>
    class IndicesVisitor {
    public:
      typedef Value                        value_type;
      typedef typename Section::point_type point_type;
    protected:
      const Section& section;
      // This can't be const because UniformSection can't have a const restrict(), because of stupid map semantics
      Order&         order;
      int            size;
      int            i;
      // If false, constrained indices are returned as negative values. Otherwise, they are omitted
      bool           freeOnly;
      // If true, it allows space for constrained variables (even if the indices are not returned) Wierd
      bool           skipConstraints;
      value_type    *values;
      bool           allocated;
    protected:
      void getUnconstrainedIndices(const point_type& p, const int orientation) {
        if (i+section.getFiberDimension(p) > size) {throw ALE::Exception("Too many values for IndicesVisitor.");}
        if (orientation >= 0) {
          const int start = this->order.getIndex(p);
          const int end   = start + section.getFiberDimension(p);

          for(int j = start; j < end; ++j, ++i) {
            this->values[i] = j;
          }
        } else if (!section.getNumSpaces()) {
          const int start = this->order.getIndex(p);
          const int end   = start + section.getFiberDimension(p);

          for(int j = end-1; j >= start; --j, ++i) {
            this->values[i] = j;
          }
        } else {
          const int numSpaces = section.getNumSpaces();
          int       start     = this->order.getIndex(p);

          for(int space = 0; space < numSpaces; ++space) {
            const int end = start + section.getFiberDimension(p, space);

            for(int j = end-1; j >= start; --j, ++i) {
              this->values[i] = j;
            }
            start = end;
          }
        }
      };
      void getConstrainedIndices(const point_type& p, const int orientation) {
        const int cDim = this->section.getConstraintDimension(p);
        if (i+cDim > size) {throw ALE::Exception("Too many values for IndicesVisitor.");}
        typedef typename Section::bc_type::value_type index_type;
        const index_type *cDof  = this->section.getConstraintDof(p);
        const int         start = this->order.getIndex(p);

        if (orientation >= 0) {
          const int dim = this->section.getFiberDimension(p);

          for(int j = start, cInd = 0, k = 0; k < dim; ++k) {
            if ((cInd < cDim) && (k == cDof[cInd])) {
              if (!freeOnly) values[i++] = -(k+1);
              if (skipConstraints) ++j;
              ++cInd;
            } else {
              values[i++] = j++;
            }
          }
        } else {
          int offset  = 0;
          int cOffset = 0;
          int k       = -1;

          for(int space = 0; space < section.getNumSpaces(); ++space) {
            const int  dim = this->section.getFiberDimension(p, space);
            const int tDim = this->section.getConstrainedFiberDimension(p, space);
            int       cInd = (dim - tDim)-1;

            k += dim;
            for(int l = 0, j = start+tDim+offset; l < dim; ++l, --k) {
              if ((cInd >= 0) && (k == cDof[cInd+cOffset])) {
                if (!freeOnly) values[i++] = -(offset+l+1);
                if (skipConstraints) --j;
                --cInd;
              } else {
                values[i++] = --j;
              }
            }
            k       += dim;
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      };
    public:
      IndicesVisitor(const Section& s, Order& o, const int size) : section(s), order(o), size(size), i(0), freeOnly(false), skipConstraints(false) {
        this->values    = new value_type[this->size];
        this->allocated = true;
      };
      IndicesVisitor(const Section& s, Order& o, const int size, value_type *values) : section(s), order(o), size(size), i(0), freeOnly(false), skipConstraints(false) {
        this->values    = values;
        this->allocated = false;
      };
      ~IndicesVisitor() {if (this->allocated) {delete [] this->values;}};
      void visitPoint(const point_type& point, const int orientation) {
        const int cDim = this->section.getConstraintDimension(point);

        if (!cDim) {
          this->getUnconstrainedIndices(point, orientation);
        } else {
          this->getConstrainedIndices(point, orientation);
        }
      };
      template<typename Arrow>
      void visitArrow(const Arrow& arrow, const int orientation) {};
    public:
      const value_type *getValues() const {return this->values;};
      int  getSize() const {return this->i;};
      int  getMaxSize() const {return this->size;};
      void ensureSize(const int size) {
        this->clear();
        if (size > this->size) {
          this->size = size;
          if (this->allocated) {delete [] this->values;}
          this->values = new value_type[this->size];
          this->allocated = true;
        }
      };
      void clear() {this->i = 0;};
    };
  };

  template<typename Sieve>
  class ISieveTraversal {
  public:
    typedef typename Sieve::point_type point_type;
  public:
    template<typename Visitor>
    static void orientedClosure(const Sieve& sieve, const point_type& p, Visitor& v) {
      typedef ISieveVisitor::PointRetriever<Sieve,Visitor> Retriever;
      Retriever cV[2] = {Retriever(200,v), Retriever(200,v)};
      int       c     = 0;

      v.visitPoint(p, 0);
      // Cone is guarateed to be ordered correctly
      ///std::cout << "Getting cone from " << p << std::endl;
      sieve.orientedCone(p, cV[c]);

      while(cV[c].getOrientedSize()) {
        const typename Retriever::oriented_point_type *cone     = cV[c].getOrientedPoints();
        const int                                      coneSize = cV[c].getOrientedSize();
        c = 1 - c;

        for(int p = 0; p < coneSize; ++p) {
          const typename Retriever::point_type& point = cone[p].first;
          const int&                            pO    = cone[p].second;

#if 0
          const Obj<typename sieve_type::traits::coneSequence>& pCone = sieve->cone(point);
          typename arrow_section_type::value_type               o     = orientation->restrictPoint(arrow)[0];
#endif
          int o = 1;

          if (pO < 0) {
            o = -(o+1);
          }
          if (o < 0) {
            if (o == -sieve.getConeSize(point)) {
              ///std::cout << "Getting reverse cone from " << point << std::endl;
              sieve.orientedReverseCone(point, cV[c]);
            } else {
              throw ALE::Exception("Not yet implemented");
            }
#if 0
            const int size = pCone->size();

            if (o == -size) {
              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter) {
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, point);
                  int              pointO = orientation->restrictPoint(newArrow)[0];

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, pointO ? -(pointO+1): pointO));
                }
              }
            } else {
              const int numSkip = size + o;
              int       count   = 0;

              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter, ++count) {
                if (count < numSkip) continue;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, point);
                  int              pointO = orientation->restrictPoint(newArrow)[0];

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, pointO ? -(pointO+1): pointO));
                }
              }
              count = 0;
              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter, ++count) {
                if (count >= numSkip) break;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, point);
                  int              pointO = orientation->restrictPoint(newArrow)[0];

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, pointO ? -(pointO+1): pointO));
                }
              }
            }
#endif
          } else {
            if (o == 1) {
              ///std::cout << "Getting cone from " << point << std::endl;
              sieve.orientedCone(point, cV[c]);
#if 0
            } else {
              const int numSkip = o-1;
              int       count   = 0;

              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter, ++count) {
                if (count < numSkip) continue;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, point);

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(newArrow)[0]));
                }
              }
              count = 0;
              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter, ++count) {
                if (count >= numSkip) break;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, point);

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(newArrow)[0]));
                }
              }
#endif
            } else {
              throw ALE::Exception("Not yet implemented");
            }
          }
        }
        cV[1-c].clear();
      }
#if 0
      Obj<orientedArrowArray> cone    = new orientedArrowArray();
      Obj<orientedArrowArray> base    = new orientedArrowArray();
      coneSet                 seen;

      for(typename sieve_type::traits::coneSequence::iterator c_iter = initCone->begin(); c_iter != initCone->end(); ++c_iter) {
        const arrow_type arrow(*c_iter, p);

        cone->push_back(oriented_arrow_type(arrow, 1));
        closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(arrow)[0]));
      }
      for(int i = 1; i < depth; ++i) {
        Obj<orientedArrowArray> tmp = cone; cone = base; base = tmp;

        cone->clear();
        for(typename orientedArrowArray::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          const arrow_type&                                     arrow = b_iter->first;
          const Obj<typename sieve_type::traits::coneSequence>& pCone = sieve->cone(arrow.source);
          typename arrow_section_type::value_type               o     = orientation->restrictPoint(arrow)[0];

          if (b_iter->second < 0) {
            o = -(o+1);
          }
          if (o < 0) {
            const int size = pCone->size();

            if (o == -size) {
              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter) {
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);
                  int              pointO = orientation->restrictPoint(newArrow)[0];

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, pointO ? -(pointO+1): pointO));
                }
              }
            } else {
              const int numSkip = size + o;
              int       count   = 0;

              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter, ++count) {
                if (count < numSkip) continue;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);
                  int              pointO = orientation->restrictPoint(newArrow)[0];

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, pointO ? -(pointO+1): pointO));
                }
              }
              count = 0;
              for(typename sieve_type::traits::coneSequence::reverse_iterator c_iter = pCone->rbegin(); c_iter != pCone->rend(); ++c_iter, ++count) {
                if (count >= numSkip) break;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);
                  int              pointO = orientation->restrictPoint(newArrow)[0];

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, pointO ? -(pointO+1): pointO));
                }
              }
            }
          } else {
            if (o == 1) {
              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter) {
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(newArrow)[0]));
                }
              }
            } else {
              const int numSkip = o-1;
              int       count   = 0;

              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter, ++count) {
                if (count < numSkip) continue;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(newArrow)[0]));
                }
              }
              count = 0;
              for(typename sieve_type::traits::coneSequence::iterator c_iter = pCone->begin(); c_iter != pCone->end(); ++c_iter, ++count) {
                if (count >= numSkip) break;
                if (seen.find(*c_iter) == seen.end()) {
                  const arrow_type newArrow(*c_iter, arrow.source);

                  seen.insert(*c_iter);
                  cone->push_back(oriented_arrow_type(newArrow, o));
                  closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(newArrow)[0]));
                }
              }
            }
          }
        }
      }
#endif
    };
  };

  // Interval Final Sieve
  // This is just two CSR matrices that give cones and supports
  //   It is completely static and cannot be resized
  //   It will operator on visitors, rather than sequences (which are messy)
  template<typename Point_, typename Allocator_ = malloc_allocator<Point_> >
  class IFSieve : public ParallelObject {
  public:
    // Types
    typedef IFSieve<Point_,Allocator_>         this_type;
    typedef Point_                             point_type;
    typedef SimpleArrow<point_type,point_type> arrow_type;
    typedef typename arrow_type::source_type   source_type;
    typedef typename arrow_type::target_type   target_type;
    typedef int                                index_type;
    // Allocators
    typedef Allocator_                                                        point_allocator_type;
    typedef typename point_allocator_type::template rebind<index_type>::other index_allocator_type;
    typedef typename point_allocator_type::template rebind<int>::other        int_allocator_type;
    // Interval
    typedef Interval<point_type, point_allocator_type> chart_type;
    // Dynamic structure
    typedef std::map<point_type, std::vector<point_type> > newpoints_type;
    // Compatibility types for SieveAlgorithms (until we rewrite for visitors)
    typedef std::set<point_type>   pointSet;
    typedef ALE::array<point_type> pointArray;
    typedef pointSet               coneSet;
    typedef pointSet               supportSet;
    typedef pointArray             coneArray;
    typedef pointArray             supportArray;
  protected:
    // Arrow Containers
    typedef index_type *offsets_type;
    typedef point_type *cones_type;
    typedef point_type *supports_type;
    // Decorators
    typedef int        *orientations_type;
  protected:
    // Data
    bool                 indexAllocated;
    offsets_type         coneOffsets;
    offsets_type         supportOffsets;
    bool                 pointAllocated;
    index_type           maxConeSize;
    index_type           maxSupportSize;
    index_type           baseSize;
    index_type           capSize;
    cones_type           cones;
    supports_type        supports;
    bool                 orientCones;
    orientations_type    coneOrientations;
    chart_type           chart;
    int_allocator_type   intAlloc;
    index_allocator_type indexAlloc;
    point_allocator_type pointAlloc;
    newpoints_type       newCones;
    newpoints_type       newSupports;
  protected: // Memory Management
    void createIndices() {
      this->coneOffsets = indexAlloc.allocate(this->chart.size()+1);
      this->coneOffsets -= this->chart.min();
      for(index_type i = this->chart.min(); i <= this->chart.max(); ++i) {indexAlloc.construct(this->coneOffsets+i, index_type(0));}
      this->supportOffsets = indexAlloc.allocate(this->chart.size()+1);
      this->supportOffsets -= this->chart.min();
      for(index_type i = this->chart.min(); i <= this->chart.max(); ++i) {indexAlloc.construct(this->supportOffsets+i, index_type(0));}
      this->indexAllocated = true;
    };
    void destroyIndices(const chart_type& chart, offsets_type *coneOffsets, offsets_type *supportOffsets) {
      if (*coneOffsets) {
        for(index_type i = chart.min(); i <= chart.max(); ++i) {indexAlloc.destroy((*coneOffsets)+i);}
        *coneOffsets += chart.min();
        indexAlloc.deallocate(*coneOffsets, chart.size()+1);
        *coneOffsets = NULL;
      }
      if (*supportOffsets) {
        for(index_type i = chart.min(); i <= chart.max(); ++i) {indexAlloc.destroy((*supportOffsets)+i);}
        *supportOffsets += chart.min();
        indexAlloc.deallocate(*supportOffsets, chart.size()+1);
        *supportOffsets = NULL;
      }
    };
    void destroyIndices() {
      this->destroyIndices(this->chart, &this->coneOffsets, &this->supportOffsets);
      this->indexAllocated = false;
      this->maxConeSize    = -1;
      this->maxSupportSize = -1;
      this->baseSize       = -1;
      this->capSize        = -1;
    };
    void createPoints() {
      this->cones = pointAlloc.allocate(this->coneOffsets[this->chart.max()]-this->coneOffsets[this->chart.min()]);
      for(index_type i = this->coneOffsets[this->chart.min()]; i < this->coneOffsets[this->chart.max()]; ++i) {pointAlloc.construct(this->cones+i, point_type(0));}
      this->supports = pointAlloc.allocate(this->supportOffsets[this->chart.max()]-this->supportOffsets[this->chart.min()]);
      for(index_type i = this->supportOffsets[this->chart.min()]; i < this->supportOffsets[this->chart.max()]; ++i) {pointAlloc.construct(this->supports+i, point_type(0));}
      if (orientCones) {
        this->coneOrientations = intAlloc.allocate(this->coneOffsets[this->chart.max()]-this->coneOffsets[this->chart.min()]);
        for(index_type i = this->coneOffsets[this->chart.min()]; i < this->coneOffsets[this->chart.max()]; ++i) {intAlloc.construct(this->coneOrientations+i, 0);}
      }
      this->pointAllocated = true;
    };
    void destroyPoints(const chart_type& chart, const offsets_type coneOffsets, cones_type *cones, const offsets_type supportOffsets, supports_type *supports, orientations_type *coneOrientations) {
      if (*cones) {
        for(index_type i = coneOffsets[chart.min()]; i < coneOffsets[chart.max()]; ++i) {pointAlloc.destroy((*cones)+i);}
        pointAlloc.deallocate(*cones, coneOffsets[chart.max()]-coneOffsets[chart.min()]);
        *cones = NULL;
      }
      if (*supports) {
        for(index_type i = supportOffsets[chart.min()]; i < supportOffsets[chart.max()]; ++i) {pointAlloc.destroy((*supports)+i);}
        pointAlloc.deallocate(*supports, supportOffsets[chart.max()]-supportOffsets[chart.min()]);
        *supports = NULL;
      }
      if (*coneOrientations) {
        for(index_type i = coneOffsets[chart.min()]; i < coneOffsets[chart.max()]; ++i) {pointAlloc.destroy((*coneOrientations)+i);}
        intAlloc.deallocate(*coneOrientations, coneOffsets[chart.max()]-coneOffsets[chart.min()]);
        *coneOrientations = NULL;
      }
    };
    void destroyPoints() {
      this->destroyPoints(this->chart, this->coneOffsets, &this->cones, this->supportOffsets, &this->supports, &this->coneOrientations);
      this->pointAllocated = false;
    };
    void prefixSum(const offsets_type array) {
      for(index_type p = this->chart.min()+1; p <= this->chart.max(); ++p) {
        array[p] = array[p] + array[p-1];
      }
    };
    void calculateBaseAndCapSize() {
      this->baseSize = 0;
      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->coneOffsets[p+1]-this->coneOffsets[p] > 0) {
          ++this->baseSize;
        }
      }
      this->capSize = 0;
      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->supportOffsets[p+1]-this->supportOffsets[p] > 0) {
          ++this->capSize;
        }
      }
    };
  public:
    IFSieve(const MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), indexAllocated(false), coneOffsets(NULL), supportOffsets(NULL), pointAllocated(false), maxConeSize(-1), maxSupportSize(-1), baseSize(-1), capSize(-1), cones(NULL), supports(NULL), orientCones(true), coneOrientations(NULL) {};
    IFSieve(const MPI_Comm comm, const point_type& min, const point_type& max, const int debug = 0) : ParallelObject(comm, debug), indexAllocated(false), coneOffsets(NULL), supportOffsets(NULL), pointAllocated(false), maxConeSize(-1), maxSupportSize(-1), baseSize(-1), capSize(-1), cones(NULL), supports(NULL), orientCones(true), coneOrientations(NULL) {
      this->setChart(chart_type(min, max));
    };
    ~IFSieve() {
      this->destroyPoints();
      this->destroyIndices();
    };
  public: // Accessors
    const chart_type& getChart() const {return this->chart;};
    void setChart(const chart_type& chart) {
      this->destroyPoints();
      this->destroyIndices();
      this->chart = chart;
      this->createIndices();
    };
    index_type getMaxConeSize() const {
      return this->maxConeSize;
    };
    index_type getMaxSupportSize() const {
      return this->maxSupportSize;
    };
    bool baseContains(const point_type& p) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);

      if (this->coneOffsets[p+1]-this->coneOffsets[p] > 0) {
        return true;
      }
      return false;
    };
  public: // Construction
    index_type getConeSize(const point_type& p) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      return this->coneOffsets[p+1]-this->coneOffsets[p];
    };
    void setConeSize(const point_type& p, const index_type c) {
      if (this->pointAllocated) {throw ALE::Exception("IFSieve points have already been allocated.");}
      this->chart.checkPoint(p);
      this->coneOffsets[p+1] = c;
      this->maxConeSize = std::max(this->maxConeSize, c);
    };
    void addConeSize(const point_type& p, const index_type c) {
      if (this->pointAllocated) {throw ALE::Exception("IFSieve points have already been allocated.");}
      this->chart.checkPoint(p);
      this->coneOffsets[p+1] += c;
      this->maxConeSize = std::max(this->maxConeSize, this->coneOffsets[p+1]);
    };
    index_type getSupportSize(const point_type& p) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      return this->supportOffsets[p+1]-this->supportOffsets[p];
    };
    void setSupportSize(const point_type& p, const index_type s) {
      if (this->pointAllocated) {throw ALE::Exception("IFSieve points have already been allocated.");}
      this->chart.checkPoint(p);
      this->supportOffsets[p+1] = s;
      this->maxSupportSize = std::max(this->maxSupportSize, s);
    };
    void addSupportSize(const point_type& p, const index_type s) {
      if (this->pointAllocated) {throw ALE::Exception("IFSieve points have already been allocated.");}
      this->chart.checkPoint(p);
      this->supportOffsets[p+1] += s;
      this->maxSupportSize = std::max(this->maxSupportSize, this->supportOffsets[p+1]);
    };
    void allocate() {
      if (this->pointAllocated) {throw ALE::Exception("IFSieve points have already been allocated.");}
      this->prefixSum(this->coneOffsets);
      this->prefixSum(this->supportOffsets);
      this->createPoints();
      this->calculateBaseAndCapSize();
    };
    void addArrow(const point_type& p, const point_type& q) {
      if (!this->chart.hasPoint(q)) {
        if (!this->newCones[q].size() && this->chart.hasPoint(q)) {
          const index_type start = this->coneOffsets[q];
          const index_type end   = this->coneOffsets[q+1];

          for(int c = start; c < end; ++c) {
            this->newCones[q].push_back(this->cones[c]);
          }
        }
        this->newCones[q].push_back(p);
      }
      if (!this->chart.hasPoint(p)) {
        if (!this->newSupports[p].size() && this->chart.hasPoint(p)) {
          const index_type start = this->supportOffsets[p];
          const index_type end   = this->supportOffsets[p+1];

          for(int s = start; s < end; ++s) {
            this->newSupports[p].push_back(this->supports[s]);
          }
        }
        this->newSupports[p].push_back(q);
      }
    };
    void reallocate() {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      if (!this->newCones.size() && !this->newSupports.size()) return;
      const chart_type     oldChart            = this->chart;
      offsets_type         oldConeOffsets      = this->coneOffsets;
      offsets_type         oldSupportOffsets   = this->supportOffsets;
      cones_type           oldCones            = this->cones;
      supports_type        oldSupports         = this->supports;
      orientations_type    oldConeOrientations = this->coneOrientations;
      point_type           min                 = this->chart.min();
      point_type           max                 = this->chart.max()-1;

      for(typename newpoints_type::const_iterator c_iter = this->newCones.begin(); c_iter != this->newCones.end(); ++c_iter) {
        min = std::min(min, c_iter->first);
        max = std::max(max, c_iter->first);
      }
      for(typename newpoints_type::const_iterator s_iter = this->newSupports.begin(); s_iter != this->newSupports.end(); ++s_iter) {
        min = std::min(min, s_iter->first);
        max = std::max(max, s_iter->first);
      }
      this->chart = chart_type(min, max+1);
      this->createIndices();
      // Copy sizes (converted from offsets)
      for(point_type p = oldChart.min(); p < oldChart.max(); ++p) {
        this->coneOffsets[p+1]    = oldConeOffsets[p+1]-oldConeOffsets[p];
        this->supportOffsets[p+1] = oldSupportOffsets[p+1]-oldSupportOffsets[p];
      }
      // Inject new sizes
      for(typename newpoints_type::const_iterator c_iter = this->newCones.begin(); c_iter != this->newCones.end(); ++c_iter) {
        this->coneOffsets[c_iter->first+1]    = c_iter->second.size();
        this->maxConeSize                     = std::max(this->maxConeSize,    (int) c_iter->second.size());
      }
      for(typename newpoints_type::const_iterator s_iter = this->newSupports.begin(); s_iter != this->newSupports.end(); ++s_iter) {
        this->supportOffsets[s_iter->first+1] = s_iter->second.size();
        this->maxSupportSize                  = std::max(this->maxSupportSize, (int) s_iter->second.size());
      }
      this->prefixSum(this->coneOffsets);
      this->prefixSum(this->supportOffsets);
      this->createPoints();
      this->calculateBaseAndCapSize();
      // Copy cones and supports
      for(point_type p = oldChart.min(); p < oldChart.max(); ++p) {
        const index_type cStart  = this->coneOffsets[p];
        const index_type cEnd    = this->coneOffsets[p+1];
        const index_type cOStart = oldConeOffsets[p];
        const index_type cOEnd   = oldConeOffsets[p+1];
        const index_type sStart  = this->supportOffsets[p];
        const index_type sEnd    = this->supportOffsets[p+1];
        const index_type sOStart = oldSupportOffsets[p];
        const index_type sOEnd   = oldSupportOffsets[p+1];

        for(int cO = cOStart, c = cStart; cO < cOEnd; ++cO, ++c) {
          this->cones[c] = oldCones[cO];
        }
        for(int sO = sOStart, s = sStart; sO < sOEnd; ++sO, ++s) {
          this->supports[s] = oldSupports[sO];
        }
        if (this->orientCones) {
          for(int cO = cOStart, c = cStart; cO < cOEnd; ++cO, ++c) {
            this->coneOrientations[c] = oldConeOrientations[cO];
          }
        }
      }
      // Inject new cones and supports
      for(typename newpoints_type::const_iterator c_iter = this->newCones.begin(); c_iter != this->newCones.end(); ++c_iter) {
        index_type start = this->coneOffsets[c_iter->first];

        for(typename std::vector<point_type>::const_iterator p_iter = c_iter->second.begin(); p_iter != c_iter->second.end(); ++p_iter) {
          this->cones[start++] = *p_iter;
        }
      }
      for(typename newpoints_type::const_iterator s_iter = this->newSupports.begin(); s_iter != this->newSupports.end(); ++s_iter) {
        index_type start = this->supportOffsets[s_iter->first];

        for(typename std::vector<point_type>::const_iterator p_iter = s_iter->second.begin(); p_iter != s_iter->second.end(); ++p_iter) {
          this->supports[start++] = *p_iter;
        }
      }
      this->newCones.clear();
      this->newSupports.clear();
      this->destroyPoints(oldChart, oldConeOffsets, &oldCones, oldSupportOffsets, &oldSupports, &oldConeOrientations);
      this->destroyIndices(oldChart, &oldConeOffsets, &oldSupportOffsets);
    };
    void setCone(const point_type cone[], const point_type& p) {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->coneOffsets[p];
      const index_type end   = this->coneOffsets[p+1];

      for(index_type c = start, i = 0; c < end; ++c, ++i) {
        this->cones[c] = cone[i];
      }
    };
#if 0
    template<typename PointSequence>
    void setCone(const PointSequence& cone, const point_type& p) {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->coneOffsets[p];
      const index_type end   = this->coneOffsets[p+1];
      if (cone.size() != end - start) {throw ALE::Exception("Invalid size for IFSieve cone.");}
      typename PointSequence::iterator c_iter = cone.begin();

      for(index_type c = start; c < end; ++c, ++c_iter) {
        this->cones[c] = *c_iter;
      }
    };
#endif
    void setSupport(const point_type& p, const point_type support[]) {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->supportOffsets[p];
      const index_type end   = this->supportOffsets[p+1];

      for(index_type s = start, i = 0; s < end; ++s, ++i) {
        this->supports[s] = support[i];
      }
    };
#if 0
    template<typename PointSequence>
    void setSupport(const point_type& p, const PointSequence& support) {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->supportOffsets[p];
      const index_type end   = this->supportOffsets[p+1];
      if (support.size() != end - start) {throw ALE::Exception("Invalid size for IFSieve support.");}
      typename PointSequence::iterator s_iter = support.begin();

      for(index_type s = start; s < end; ++s, ++s_iter) {
        this->supports[s] = *s_iter;
      }
    };
#endif
    void setConeOrientation(const int coneOrientation[], const point_type& p) {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->coneOffsets[p];
      const index_type end   = this->coneOffsets[p+1];

      for(index_type c = start, i = 0; c < end; ++c, ++i) {
        this->coneOrientations[c] = coneOrientation[i];
      }
    };
    void symmetrize() {
      index_type *offsets = indexAlloc.allocate(this->chart.size()+1);
      offsets -= this->chart.min();
      for(index_type i = this->chart.min(); i <= this->chart.max(); ++i) {indexAlloc.construct(offsets+i, index_type(0));}
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        const index_type start = this->coneOffsets[p];
        const index_type end   = this->coneOffsets[p+1];

        for(index_type c = start; c < end; ++c) {
          const point_type q = this->cones[c];

          this->supports[this->supportOffsets[q]+offsets[q]] = p;
          ++offsets[q];
        }
      }      
    };
    index_type getBaseSize() const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      return this->baseSize;
    };
    index_type getCapSize() const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      return this->capSize;
    };
  public: // Traversals
    template<typename Visitor>
    void roots(const Visitor& v) const {
      this->roots(const_cast<Visitor&>(v));
    };
    template<typename Visitor>
    void roots(Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->coneOffsets[p+1] == this->coneOffsets[p]) {
          if (this->supportOffsets[p+1]-this->supportOffsets[p] > 0) {
            v.visitPoint(p);
          }
        }
      }
    };
    template<typename Visitor>
    void leaves(const Visitor& v) const {
      this->leaves(const_cast<Visitor&>(v));
    };
    template<typename Visitor>
    void leaves(Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->supportOffsets[p+1] == this->supportOffsets[p]) {
          if (this->coneOffsets[p+1]-this->coneOffsets[p] > 0) {
            v.visitPoint(p);
          }
        }
      }
    };
    template<typename Visitor>
    void base(const Visitor& v) const {
      this->base(const_cast<Visitor&>(v));
    };
    template<typename Visitor>
    void base(Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->coneOffsets[p+1]-this->coneOffsets[p] > 0) {
          v.visitPoint(p);
        }
      }
    };
    template<typename Visitor>
    void cap(const Visitor& v) const {
      this->cap(const_cast<Visitor&>(v));
    };
    template<typename Visitor>
    void cap(Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->supportOffsets[p+1]-this->supportOffsets[p] > 0) {
          v.visitPoint(p);
        }
      }
    };
    template<typename PointSequence, typename Visitor>
    void cone(const PointSequence& points, Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      for(typename PointSequence::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        const point_type p = *p_iter;
        this->chart.checkPoint(p);
        const index_type start = this->coneOffsets[p];
        const index_type end   = this->coneOffsets[p+1];

        for(index_type c = start; c < end; ++c) {
          v.visitArrow(arrow_type(this->cones[c], p));
          v.visitPoint(this->cones[c]);
        }
      }
    };
    template<typename Visitor>
    void cone(const point_type& p, Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->coneOffsets[p];
      const index_type end   = this->coneOffsets[p+1];

      for(index_type c = start; c < end; ++c) {
        v.visitArrow(arrow_type(this->cones[c], p));
        v.visitPoint(this->cones[c]);
      }
    };
    template<typename PointSequence, typename Visitor>
    void support(const PointSequence& points, Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      for(typename PointSequence::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        const point_type p = *p_iter;
        this->chart.checkPoint(p);
        const index_type start = this->supportOffsets[p];
        const index_type end   = this->supportOffsets[p+1];

        for(index_type s = start; s < end; ++s) {
          v.visitArrow(arrow_type(p, this->supports[s]));
          v.visitPoint(this->supports[s]);
        }
      }
    };
    template<typename Visitor>
    void support(const point_type& p, Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->supportOffsets[p];
      const index_type end   = this->supportOffsets[p+1];

      for(index_type s = start; s < end; ++s) {
        v.visitArrow(arrow_type(p, this->supports[s]));
        v.visitPoint(this->supports[s]);
      }
    };
    template<typename Visitor>
    void orientedCone(const point_type& p, Visitor& v) const {
      if (!this->orientCones) {throw ALE::Exception("IFSieve cones have not been oriented.");}
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->coneOffsets[p];
      const index_type end   = this->coneOffsets[p+1];

      for(index_type c = start; c < end; ++c) {
        v.visitArrow(arrow_type(this->cones[c], p), this->coneOrientations[c]);
        v.visitPoint(this->cones[c], this->coneOrientations[c]);
      }
    };
    template<typename Visitor>
    void orientedReverseCone(const point_type& p, Visitor& v) const {
      if (!this->orientCones) {throw ALE::Exception("IFSieve cones have not been oriented.");}
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->coneOffsets[p];
      const index_type end   = this->coneOffsets[p+1];

      for(index_type c = end-1; c >= start; --c) {
        v.visitArrow(arrow_type(this->cones[c], p), this->coneOrientations[c]);
        v.visitPoint(this->cones[c], this->coneOrientations[c]);
      }
    };
    // Currently does only 1 level
    //   Does not check for uniqueness
    template<typename Visitor>
    void meet(const point_type& p, const point_type& q, Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      this->chart.checkPoint(q);
      const index_type startP = this->coneOffsets[p];
      const index_type endP   = this->coneOffsets[p+1];
      const index_type startQ = this->coneOffsets[q];
      const index_type endQ   = this->coneOffsets[q+1];

      for(index_type cP = startP; cP < endP; ++cP) {
        const point_type& c1 = this->cones[cP];

        for(index_type cQ = startQ; cQ < endQ; ++cQ) {
          if (c1 == this->cones[cQ]) v.visitPoint(c1);
        }
        if (this->coneOffsets[c1+1] > this->coneOffsets[c1]) {throw ALE::Exception("Cannot handle multiple level meet()");}
      }
    };
    // Currently does only 1 level
    template<typename Sequence, typename Visitor>
    void join(const Sequence& points, Visitor& v) const {
      typedef std::set<point_type> pointSet;
      pointSet intersect[2] = {pointSet(), pointSet()};
      pointSet tmp;
      int      p = 0;
      int      c = 0;

      for(typename Sequence::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        this->chart.checkPoint(*p_iter);
        tmp.insert(&this->supports[this->supportOffsets[*p_iter]], &this->supports[this->supportOffsets[(*p_iter)+1]]);
        if (p == 0) {
          intersect[1-c].insert(tmp.begin(), tmp.end());
          p++;
        } else {
          std::set_intersection(intersect[c].begin(), intersect[c].end(), tmp.begin(), tmp.end(),
                                std::insert_iterator<pointSet>(intersect[1-c], intersect[1-c].begin()));
          intersect[c].clear();
        }
        c = 1 - c;
        tmp.clear();
      }
      for(typename pointSet::const_iterator p_iter = intersect[c].begin(); p_iter != intersect[c].end(); ++p_iter) {
        v.visitPoint(*p_iter);
      }
    };
  public: // Viewing
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      ostringstream txt;
      int rank;

      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
        rank = this->commRank();
      } else {
        MPI_Comm_rank(comm, &rank);
      }
      if (name == "") {
        if(rank == 0) {
          txt << "viewing an IFSieve" << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing IFSieve '" << name << "'" << std::endl;
        }
      }
      if(rank == 0) {
        txt << "cap --> base:" << std::endl;
      }
      ISieveVisitor::PrintVisitor pV(txt, rank);
      this->cap(ISieveVisitor::SupportVisitor<IFSieve,ISieveVisitor::PrintVisitor>(*this, pV));
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
      ostringstream txt2;

      if(rank == 0) {
        txt2 << "base <-- cap:" << std::endl;
      }
      ISieveVisitor::ReversePrintVisitor rV(txt2, rank);
      this->base(ISieveVisitor::ConeVisitor<IFSieve,ISieveVisitor::ReversePrintVisitor>(*this, rV));
      PetscSynchronizedPrintf(comm, txt2.str().c_str());
      PetscSynchronizedFlush(comm);
      if (orientCones) {
        ostringstream txt3;

        if(rank == 0) {
          txt3 << "Orientation:" << std::endl;
        }
        ISieveVisitor::ReversePrintVisitor rV2(txt3, rank);
        this->base(ISieveVisitor::OrientedConeVisitor<IFSieve,ISieveVisitor::ReversePrintVisitor>(*this, rV2));
        PetscSynchronizedPrintf(comm, txt3.str().c_str());
        PetscSynchronizedFlush(comm);
      }
    };
  };

  class ISieveConverter {
  public:
    template<typename Sieve, typename ISieve, typename Renumbering>
    static void convertSieve(Sieve& sieve, ISieve& isieve, Renumbering& renumbering, bool renumber = true) {
      // First construct a renumbering of the sieve points
      const Obj<typename Sieve::baseSequence>& base = sieve.base();
      const Obj<typename Sieve::capSequence>&  cap  = sieve.cap();
      typename ISieve::point_type              min  = 0;
      typename ISieve::point_type              max  = 0;

      if (renumber) {
        for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          renumbering[*b_iter] = max++;
        }
        for(typename Sieve::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
          if (renumbering.find(*c_iter) == renumbering.end()) {
            renumbering[*c_iter] = max++;
          }
        }
      } else {
        if (base->size()) {
          min = *base->begin();
          max = *base->begin();
        } else if (cap->size()) {
          min = *cap->begin();
          max = *cap->begin();
        }
        for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          min = std::min(min, *b_iter);
          max = std::max(max, *b_iter);
        }
        for(typename Sieve::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
          min = std::min(min, *c_iter);
          max = std::max(max, *c_iter);
        }
        ++max;
        for(typename ISieve::point_type p = min; p < max; ++p) {
          renumbering[p] = p;
        }
      }
      // Create the ISieve
      isieve.setChart(typename ISieve::chart_type(min, max));
      // Set cone and support sizes
      size_t maxSize = 0;

      for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        const Obj<typename Sieve::coneSequence>& cone = sieve.cone(*b_iter);

        isieve.setConeSize(renumbering[*b_iter], cone->size());
        maxSize = std::max(maxSize, cone->size());
      }
      for(typename Sieve::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
        const Obj<typename Sieve::supportSequence>& support = sieve.support(*c_iter);

        isieve.setSupportSize(renumbering[*c_iter], support->size());
        maxSize = std::max(maxSize, support->size());
      }
      isieve.allocate();
      // Fill up cones and supports
      typename Sieve::point_type *points = new typename Sieve::point_type[maxSize];

      for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        const Obj<typename Sieve::coneSequence>& cone = sieve.cone(*b_iter);
        int i = 0;

        for(typename Sieve::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++i) {
          points[i] = renumbering[*c_iter];
        }
        isieve.setCone(points, renumbering[*b_iter]);
      }
      for(typename Sieve::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
        const Obj<typename Sieve::supportSequence>& support = sieve.support(*c_iter);
        int i = 0;

        for(typename Sieve::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter, ++i) {
          points[i] = renumbering[*s_iter];
        }
        isieve.setSupport(renumbering[*c_iter], points);
      }
      delete [] points;
    };
    template<typename Sieve, typename ISieve, typename Renumbering, typename ArrowSection>
    static void convertOrientation(Sieve& sieve, ISieve& isieve, Renumbering& renumbering, ArrowSection *orientation) {
      if (isieve.getMaxConeSize() < 0) return;
      const Obj<typename Sieve::baseSequence>& base = sieve.base();
      int *orientations = new int[isieve.getMaxConeSize()];

      for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        const Obj<typename Sieve::coneSequence>& cone = sieve.cone(*b_iter);
        int i = 0;

        for(typename Sieve::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++i) {
          typename ArrowSection::point_type arrow(*c_iter, *b_iter);

          orientations[i] = orientation->restrictPoint(arrow)[0];
        }
        isieve.setConeOrientation(orientations, renumbering[*b_iter]);
      }
      delete [] orientations;
    };
    template<typename Section, typename ISection, typename Renumbering>
    static void convertCoordinates(Section& coordinates, ISection& icoordinates, Renumbering& renumbering) {
      const typename Section::chart_type& chart = coordinates.getChart();
      typename ISection::point_type       min   = *chart.begin();
      typename ISection::point_type       max   = *chart.begin();

      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        min = std::min(min, *p_iter);
        max = std::max(max, *p_iter);
      }
      icoordinates.setChart(typename ISection::chart_type(min, max+1));
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        icoordinates.setFiberDimension(*p_iter, coordinates.getFiberDimension(*p_iter));
      }
      icoordinates.allocatePoint();
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        icoordinates.updatePoint(*p_iter, coordinates.restrictPoint(*p_iter));
      }
    };
    template<typename Mesh, typename IMesh, typename Renumbering>
    static void convertMesh(Mesh& mesh, IMesh& imesh, Renumbering& renumbering, bool renumber = true) {
      convertSieve(*mesh.getSieve(), *imesh.getSieve(), renumbering, renumber);
      imesh.stratify();
      convertOrientation(*mesh.getSieve(), *imesh.getSieve(), renumbering, mesh.getArrowSection("orientation").ptr());
      convertCoordinates(*mesh.getRealSection("coordinates"), *imesh.getRealSection("coordinates"), renumbering);
    };
  };
}

#endif
