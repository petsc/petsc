#ifndef included_ALE_ISieve_hh
#define included_ALE_ISieve_hh

#ifndef  included_ALE_hh
#include <sieve/ALE.hh>
#endif

#include <petscdmcomplex.h>

#include <fstream>

//#define IMESH_NEW_LABELS

namespace ALE {
  template<typename Point>
  class OrientedPoint : public std::pair<Point, int> {
  public:
    OrientedPoint(const int o) : std::pair<Point, int>(o, o) {};
    ~OrientedPoint() {};
    friend std::ostream& operator<<(std::ostream& stream, const OrientedPoint& point) {
      stream << "(" << point.first << ", " << point.second << ")";
      return stream;
    };
  };

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
      const_iterator& operator=(const const_iterator& iter) {this->_p = iter._p; return *this;};
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
    template<typename Iterator>
    Interval(Iterator& iterator) {
      this->_min = *std::min_element(iterator.begin(), iterator.end());
      this->_max = (*std::max_element(iterator.begin(), iterator.end()))+1;
    }
  public:
    Interval& operator=(const Interval& interval) {_min = interval.min(); _max = interval.max(); return *this;}
    friend std::ostream& operator<<(std::ostream& stream, const Interval& interval) {
      stream << "(" << interval.min() << ", " << interval.max() << ")";
      return stream;
    }
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
      inline void visitArrow(const typename Sieve::arrow_type&) {};
      inline void visitPoint(const typename Sieve::point_type&) {};
      inline void visitArrow(const typename Sieve::arrow_type&, const int orientation) {};
      inline void visitPoint(const typename Sieve::point_type&, const int orientation) {};
    };
    class PrintVisitor {
    protected:
      ostringstream& os;
      const int      rank;
    public:
      PrintVisitor(ostringstream& s, const int rank = 0) : os(s), rank(rank) {};
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow) const {
        this->os << "["<<this->rank<<"]: " << arrow << std::endl;
      }
      template<typename Point>
      inline void visitPoint(const Point&) const {}
    };
    class ReversePrintVisitor : public PrintVisitor {
    public:
      ReversePrintVisitor(ostringstream& s, const int rank) : PrintVisitor(s, rank) {};
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow) const {
        this->os << "["<<this->rank<<"]: " << arrow.target << "<----" << arrow.source << std::endl;
      }
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow, const int orientation) const {
        this->os << "["<<this->rank<<"]: " << arrow.target << "<----" << arrow.source << ": " << orientation << std::endl;
      }
      template<typename Point>
      inline void visitPoint(const Point&) const {}
      template<typename Point>
      inline void visitPoint(const Point&, const int) const {}
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
      size_t               skip, limit;
      Visitor             *visitor;
      size_t               size;
      point_type          *points;
      oriented_point_type *oPoints;
    protected:
      inline virtual bool accept(const point_type& point) {return true;};
    public:
      PointRetriever() : unique(false), i(0), o(0), skip(0), limit(0) {
        this->size    = 0;
        this->points  = NULL;
        this->oPoints = NULL;
      };
      PointRetriever(const size_t size, const bool unique = false) : unique(unique), i(0), o(0), skip(0), limit(0) {
        static Visitor nV;
        this->visitor = &nV;
        this->points  = NULL;
        this->oPoints = NULL;
        this->setSize(size);
      };
      PointRetriever(const size_t size, Visitor& v, const bool unique = false) : unique(unique), i(0), o(0), skip(0), limit(0), visitor(&v) {
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
      inline void visitArrow(const arrow_type& arrow) {
        this->visitor->visitArrow(arrow);
      };
      inline void visitArrow(const arrow_type& arrow, const int orientation) {
        this->visitor->visitArrow(arrow, orientation);
      };
      inline void visitPoint(const point_type& point) {
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
          if ((i < this->skip) || ((this->limit) && (i >= this->limit))) {--this->skip; return;}
          points[i++] = point;
          this->visitor->visitPoint(point);
        }
      };
      inline void visitPoint(const point_type& point, const int orientation) {
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
          if ((i < this->skip) || ((this->limit) && (i >= this->limit))) {--this->skip; return;}
          points[i++]  = point;
          oPoints[o++] = oriented_point_type(point, orientation);
          this->visitor->visitPoint(point, orientation);
        }
      };
    public:
      size_t                     getSize() const {return this->i;}
      const point_type          *getPoints() const {return this->points;}
      size_t                     getOrientedSize() const {return this->o;}
      const oriented_point_type *getOrientedPoints() const {return this->oPoints;}
      void clear() {this->i = this->o = 0;}
      void setSize(const size_t s) {
        if (this->points) {
          delete [] this->points;
          delete [] this->oPoints;
        }
        this->size    = s;
        this->points  = new point_type[this->size];
        this->oPoints = new oriented_point_type[this->size];
      }
      void setSkip(size_t s) {this->skip = s;};
      void setLimit(size_t l) {this->limit = l;};
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
    template<typename Mesh, typename Visitor = NullVisitor<typename Mesh::sieve_type> >
    class MeshNConeRetriever : public PointRetriever<typename Mesh::sieve_type,Visitor> {
    public:
      typedef typename Mesh::Sieve                    Sieve;
      typedef PointRetriever<Sieve,Visitor>           base_type;
      typedef typename Sieve::point_type              point_type;
      typedef typename Sieve::arrow_type              arrow_type;
      typedef typename base_type::oriented_point_type oriented_point_type;
    protected:
      const Mesh& mesh;
      const int   depth;
    protected:
      inline virtual bool accept(const point_type& point) {
        if (this->mesh.depth(point) == this->depth)
          return true;
        return false;
      };
    public:
      MeshNConeRetriever(const Mesh& m, const int depth, const size_t size) : PointRetriever<typename Mesh::Sieve,Visitor>(size, true), mesh(m), depth(depth) {};
      MeshNConeRetriever(const Mesh& m, const int depth, const size_t size, Visitor& v) : PointRetriever<typename Mesh::Sieve,Visitor>(size, v, true), mesh(m), depth(depth) {};
      virtual ~MeshNConeRetriever() {};
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
      size_t       i, o;
      bool         renumber;
      point_type  *points;
      oriented_point_type *oPoints;
    public:
      FilteredPointRetriever(const Set& s, Renumbering& r, const size_t size) : pointSet(s), renumbering(r), size(size), i(0), o(0), renumber(true) {
        this->points  = new point_type[this->size];
        this->oPoints = new oriented_point_type[this->size];
      };
      ~FilteredPointRetriever() {
        delete [] this->points;
        delete [] this->oPoints;
      };
      inline void visitArrow(const arrow_type& arrow) {};
      inline void visitPoint(const point_type& point) {
        if (i >= size) throw ALE::Exception("Too many points for FilteredPointRetriever visitor");
        if (this->pointSet.find(point) == this->pointSet.end()) return;
        if (renumber) {
          points[i++] = this->renumbering[point];
        } else {
          points[i++] = point;
        }
      };
      inline void visitArrow(const arrow_type& arrow, const int orientation) {};
      inline void visitPoint(const point_type& point, const int orientation) {
        if (o >= size) throw ALE::Exception("Too many points for FilteredPointRetriever visitor");
        if (this->pointSet.find(point) == this->pointSet.end()) return;
        if (renumber) {
          points[i++]  = this->renumbering[point];
          oPoints[o++] = oriented_point_type(this->renumbering[point], orientation);
        } else {
          points[i++]  = point;
          oPoints[o++] = oriented_point_type(point, orientation);
        }
      };
    public:
      size_t            getSize() const {return this->i;}
      const point_type *getPoints() const {return this->points;}
      size_t            getOrientedSize() const {return this->o;}
      const oriented_point_type *getOrientedPoints() const {return this->oPoints;}
      void clear() {this->i = 0; this->o = 0;}
      void useRenumbering(const bool renumber) {this->renumber = renumber;}
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
      inline void visitArrow(const typename Sieve::arrow_type& arrow) {
        if (i >= size) throw ALE::Exception("Too many arrows for visitor");
        arrows[i++] = arrow;
        this->visitor->visitArrow(arrow);
      };
      inline void visitArrow(const typename Sieve::arrow_type& arrow, const int orientation) {
        if (o >= size) throw ALE::Exception("Too many arrows for visitor");
        oArrows[o++] = oriented_arrow_type(arrow, orientation);
        this->visitor->visitArrow(arrow, orientation);
      };
      inline void visitPoint(const point_type& point) {
        this->visitor->visitPoint(point);
      };
      inline void visitPoint(const point_type& point, const int orientation) {
        this->visitor->visitPoint(point, orientation);
      };
    public:
      size_t                     getSize() const {return this->i;}
      const point_type          *getArrows() const {return this->arrows;}
      size_t                     getOrientedSize() const {return this->o;}
      const oriented_arrow_type *getOrientedArrows() const {return this->oArrows;}
      void clear() {this->i = this->o = 0;}
    };
    template<typename Sieve, typename Visitor>
    class ConeVisitor {
    protected:
      const Sieve& sieve;
      Visitor&     visitor;
      bool         useSource;
    public:
      ConeVisitor(const Sieve& s, Visitor& v, bool useSource = false) : sieve(s), visitor(v), useSource(useSource) {};
      inline void visitPoint(const typename Sieve::point_type& point) {
        this->sieve.cone(point, visitor);
      };
      inline void visitArrow(const typename Sieve::arrow_type& arrow) {};
    };
    template<typename Sieve, typename Visitor>
    class OrientedConeVisitor {
    protected:
      const Sieve& sieve;
      Visitor&     visitor;
      bool         useSource;
    public:
      OrientedConeVisitor(const Sieve& s, Visitor& v, bool useSource = false) : sieve(s), visitor(v), useSource(useSource) {};
      inline void visitPoint(const typename Sieve::point_type& point) {
        this->sieve.orientedCone(point, visitor);
      };
      inline void visitArrow(const typename Sieve::arrow_type& arrow) {};
    };
    template<typename Sieve, typename Visitor>
    class SupportVisitor {
    protected:
      const Sieve& sieve;
      Visitor&     visitor;
      bool         useSource;
    public:
      SupportVisitor(const Sieve& s, Visitor& v, bool useSource = true) : sieve(s), visitor(v), useSource(useSource) {};
      inline void visitPoint(const typename Sieve::point_type& point) {
        this->sieve.support(point, visitor);
      };
      inline void visitArrow(const typename Sieve::arrow_type& arrow) {};
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
      inline void visitPoint(const typename Sieve::point_type& point) const {};
      inline void visitArrow(const typename Sieve::arrow_type& arrow) {
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
      inline void visitPoint(const typename Sieve::point_type& point) {
        this->size += section.getConstrainedFiberDimension(point);
      };
      inline void visitArrow(const typename Sieve::arrow_type&) {};
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
      inline void visitPoint(const typename Sieve::point_type& point) {
        this->size += section.getFiberDimension(point);
      };
      inline void visitArrow(const typename Sieve::arrow_type&) {};
    public:
      int getSize() {return this->size;};
    };
    template<typename Sieve>
    class SizeWithBCVisitor<Sieve,PetscSection> {
    protected:
      PetscSection section;
      int          size;
      PetscInt    *fieldSize;
      PetscInt     numFields;
    public:
      SizeWithBCVisitor(PetscSection s) : section(s), size(0), fieldSize(PETSC_NULL), numFields(0) {};
      SizeWithBCVisitor(PetscSection s, PetscInt *fieldSize) : section(s), size(0), fieldSize(fieldSize) {
        PetscErrorCode ierr = PetscSectionGetNumFields(section, &numFields);CHKERRXX(ierr);
        for(PetscInt f = 0; f < numFields; ++f) {this->fieldSize[f] = 0;}
      };
      inline void visitPoint(const typename Sieve::point_type& point) {
        PetscInt dim;
        PetscErrorCode ierr;
        ierr = PetscSectionGetDof(section, point, &dim);CHKERRXX(ierr);
        this->size += dim;
        for(PetscInt f = 0; f < numFields; ++f) {
          ierr = PetscSectionGetFieldDof(section, point, f, &dim);CHKERRXX(ierr);
          this->fieldSize[f] += dim;
        }
      };
      inline void visitArrow(const typename Sieve::arrow_type&) {};
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
      inline void visitPoint(const Point& point, const int orientation) {
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
      }
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow, const int orientation) {}
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
    template<typename ValueType>
    class RestrictVecVisitor {
    public:
      typedef ValueType value_type;
    protected:
      const Vec          v;
      const PetscSection section;
      int                size;
      int                i;
      int                nF;
      int               *offsets;
      int               *indices;
      bool               processed;
      value_type        *values;
      bool               allocated;
      value_type        *array;
    protected:
      inline void swap(value_type& v, value_type& w) {
        value_type tmp = v;
        v = w;
        w = tmp;
      };
      void processArray() {
        for(PetscInt f = 1; f < nF; ++f) {
          offsets[f+1] += offsets[f];
        }
        for(PetscInt i = 0; i < offsets[nF]; ++i) {
          indices[i] = offsets[indices[i]]++;
        }
        assert(offsets[nF-1] == offsets[nF]);
        for(PetscInt i = 0; i < offsets[nF]; ++i) {
          if (indices[i] == -1) continue;
          PetscInt   startPos = indices[i];
          PetscInt   j        = startPos, k;
          value_type val      = values[i];

          do {
            swap(val, values[j]);
            k = indices[j];
            indices[j] = -1;
            j = k;
          } while(j != startPos);
        }
      };
    public:
      RestrictVecVisitor(const Vec v, const PetscSection s, const int size) : v(v), section(s), size(size), i(0), nF(0), processed(true) {
        this->values    = new value_type[this->size];
        this->allocated = true;
        PetscErrorCode ierr = VecGetArray(this->v, &this->array);CHKERRXX(ierr);
      };
      RestrictVecVisitor(const Vec v, const PetscSection s, const int size, value_type *values) : v(v), section(s), size(size), i(0), nF(0), processed(true) {
        this->values    = values;
        this->allocated = false;
        PetscErrorCode ierr = VecGetArray(this->v, &this->array);CHKERRXX(ierr);
      };
      RestrictVecVisitor(const Vec v, const PetscSection s, const int size, value_type *values, int *offsets, int *indices) : v(v), section(s), size(size), i(0), nF(0), offsets(offsets), indices(indices), processed(false) {
        PetscErrorCode ierr;

        this->values    = values;
        this->allocated = false;
        ierr = VecGetArray(this->v, &this->array);CHKERRXX(ierr);
        ierr = PetscSectionGetNumFields(section, &nF);CHKERRXX(ierr);
        for(PetscInt f = 0; f <= nF; ++f) {offsets[f] = 0;}
      };
      ~RestrictVecVisitor() {
        if (this->allocated) {delete [] this->values;}
        PetscErrorCode ierr = VecRestoreArray(this->v, &this->array);CHKERRXX(ierr);
      };
      template<typename Point>
      inline void visitPoint(const Point& point, const int orientation) {
        // Known:
        //   Nf: number of fields
        // Unknown:
        //   Np: number of points
        //   P:  points
        // Algorithm:
        //   Pass 1: Stack up values as before, but also
        //           count size of each field
        //   Comp 1: Sum up sizes to get field offsets
        //   Pass 2: Number each entry with its intended position
        //   Pass 3: Reorder entries
        // Algorithm if field sizes are known:
        //   Comp 1: Partition array into field components
        //   Pass 1: Stack up values at field offsets
        PetscInt       dim, off;
        PetscErrorCode ierr;

        ierr = PetscSectionGetDof(section, point, &dim);CHKERRXX(ierr);
        if (i+dim > size) {
          ostringstream msg;
          msg << "Too many values for RestrictVisitor "<<i+dim<<" > "<<size<< std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
        ierr = PetscSectionGetOffset(section, point, &off);CHKERRXX(ierr);
        const value_type *v = &array[off];

        if (nF) {
          for(PetscInt f = 0, fOff = 0; f < nF; ++f) {
            PetscInt comp, fDim;

            ierr = PetscSectionGetFieldDof(section, point, f, &fDim);CHKERRXX(ierr);
            offsets[f+1] += fDim;
            for(PetscInt d = 0; d < fDim; ++d) {
              indices[i+d] = f;
            }
            if (orientation >= 0) {
              for(PetscInt d = 0; d < fDim; ++d, ++i) {
                this->values[i] = v[fOff+d];
              }
            } else {
              ierr = PetscSectionGetFieldComponents(section, f, &comp);CHKERRXX(ierr);
              for(PetscInt d = fDim/comp-1; d >= 0; --d) {
                for(PetscInt c = 0; c < comp; ++c, ++i) {
                  this->values[i] = v[fOff+d*comp+c];
                }
              }
            }
            fOff += fDim;
          }
        } else {
          if (orientation >= 0) {
            for(PetscInt d = 0; d < dim; ++d, ++i) {
              this->values[i] = v[d];
            }
          } else {
            for(PetscInt d = dim-1; d >= 0; --d, ++i) {
              this->values[i] = v[d];
            }
          }
        }
      }
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow, const int orientation) {}
    public:
      const value_type *getValues() {
        if (!processed) {processArray(); processed = true;}
        return this->values;
      };
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
      void clear() {this->i = 0; if (processed) {processed = false;}};
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
      inline void visitPoint(const Point& point, const int orientation) {
        const int dim = section.getFiberDimension(point);
        this->section.updatePoint(point, &this->values[this->i], orientation);
        this->i += dim;
      }
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow, const int orientation) {}
      void clear() {this->i = 0;};
    };
    template<typename Section>
    class UpdateAllVisitor {
    public:
      typedef typename Section::value_type value_type;
    protected:
      Section&          section;
      const value_type *values;
      int               i;
    public:
      UpdateAllVisitor(Section& s, const value_type *v) : section(s), values(v), i(0) {};
      template<typename Point>
      inline void visitPoint(const Point& point, const int orientation) {
        const int dim = section.getFiberDimension(point);
        this->section.updatePointAll(point, &this->values[this->i], orientation);
        this->i += dim;
      }
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow, const int orientation) {}
      void clear() {this->i = 0;};
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
      inline void visitPoint(const Point& point, const int orientation) {
        const int dim = section.getFiberDimension(point);
        this->section.updateAddPoint(point, &this->values[this->i], orientation);
        this->i += dim;
      }
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow, const int orientation) {}
      void clear() {this->i = 0;};
    };
    template<typename ValueType>
    class UpdateVecVisitor {
    public:
      typedef ValueType value_type;
    protected:
      const Vec          v;
      const PetscSection section;
      const value_type  *values;
      const InsertMode   mode;
      PetscInt           nF;
      PetscInt           i;
      value_type        *array;
      PetscInt          *fieldSize;
      PetscInt          *j;
    protected:
      inline static void add   (value_type& x, value_type y) {x += y;}
      inline static void insert(value_type& x, value_type y) {x  = y;}
      template<typename Point>
      void updatePoint(const Point& point, void (*fuse)(value_type&, value_type), const bool setBC, const int orientation = 1) {
        PetscInt        dim;  // The number of dof on this point
        PetscInt        cDim; // The nubmer of constraints on this point
        const PetscInt *cDof; // The indices of the constrained dofs on this point
        value_type     *a;    // The values on this point
        PetscInt        offset, cInd = 0;
        PetscErrorCode  ierr;

        ierr = PetscSectionGetDof(section, point, &dim);CHKERRXX(ierr);
        ierr = PetscSectionGetConstraintDof(section, point, &cDim);CHKERRXX(ierr);
        ierr = PetscSectionGetOffset(section, point, &offset);CHKERRXX(ierr);
        a    = &array[offset];
        if (!cDim || setBC) {
          if (orientation >= 0) {
            for(PetscInt k = 0; k < dim; ++k) {
              fuse(a[k], values[i+k]);
            }
          } else {
            for(PetscInt k = 0; k < dim; ++k) {
              fuse(a[k], values[i+dim-k-1]);
            }
          }
        } else {
          ierr = PetscSectionGetConstraintIndices(section, point, &cDof);CHKERRXX(ierr);
          if (orientation >= 0) {
            for(PetscInt k = 0; k < dim; ++k) {
              if ((cInd < cDim) && (k == cDof[cInd])) {++cInd; continue;}
              fuse(a[k], values[i+k]);
            }
          } else {
            for(PetscInt k = 0; k < dim; ++k) {
              if ((cInd < cDim) && (k == cDof[cInd])) {++cInd; continue;}
              fuse(a[k], values[i+dim-k-1]);
            }
          }
        }
        i += dim;
      }
      template<typename Point>
      void updatePointFields(const Point& point, void (*fuse)(value_type&, value_type), const bool setBC, const int orientation = 1) {
        value_type    *a;
        PetscInt       offset;
        PetscInt       fOff = 0;
        PetscErrorCode ierr;

        ierr = PetscSectionGetOffset(section, point, &offset);CHKERRXX(ierr);
        a    = &array[offset];
        for(PetscInt f = 0; f < nF; ++f) {
          PetscInt    dim;  // The number of dof for field f on this point
          PetscInt    comp; // The number of components for field f on this point
          PetscInt    cDim; // The nubmer of constraints for field f on this point
          const PetscInt *cDof; // The indices of the constrained dofs for field f on this point
          PetscInt    cInd = 0;

          ierr = PetscSectionGetFieldComponents(section, f, &comp);CHKERRXX(ierr);
          ierr = PetscSectionGetFieldDof(section, point, f, &dim);CHKERRXX(ierr);
          ierr = PetscSectionGetFieldConstraintDof(section, point, f, &cDim);CHKERRXX(ierr);
          if (!cDim || setBC) {
            if (orientation >= 0) {
              for(PetscInt k = 0; k < dim; ++k) {
                fuse(a[fOff+k], values[j[f]+k]);
              }
            } else {
              for(PetscInt k = dim/comp-1; k >= 0; --k) {
                for(PetscInt c = 0; c < comp; ++c) {
                  fuse(a[fOff+(dim/comp-1-k)*comp+c], values[j[f]+k*comp+c]);
                }
              }
            }
          } else {
            ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &cDof);CHKERRXX(ierr);
            if (orientation >= 0) {
              for(PetscInt k = 0; k < dim; ++k) {
                if ((cInd < cDim) && (k == cDof[cInd])) {++cInd; continue;}
                fuse(a[fOff+k], values[j[f]+k]);
              }
            } else {
              for(PetscInt k = dim/comp-1; k >= 0; --k) {
                for(PetscInt c = 0; c < comp; ++c) {
                  PetscInt ind = k*comp+c;
                  if ((cInd < cDim) && (ind == cDof[cInd])) {++cInd; continue;}
                  fuse(a[fOff+(dim/comp-1-k)*comp+c], values[j[f]+ind]);
                }
              }
            }
          }
          fOff += dim;
          j[f] += dim;
        }
      }
    public:
      UpdateVecVisitor(const Vec v, const PetscSection s, const value_type *values, InsertMode mode) : v(v), section(s), values(values), mode(mode), nF(0), i(0) {};
      UpdateVecVisitor(const Vec v, const PetscSection s, const value_type *values, InsertMode mode, PetscInt numFields, PetscInt fieldSize[]) : v(v), section(s), values(values), mode(mode), nF(numFields), i(0) {
        PetscErrorCode ierr;

        ierr = VecGetArray(this->v, &this->array);CHKERRXX(ierr);
        ierr = PetscMalloc2(numFields,PetscInt,&this->fieldSize,numFields,PetscInt,&j);CHKERRXX(ierr);
        for(PetscInt f = 0; f < nF; ++f) {
          this->fieldSize[f] = fieldSize[f];
        }
        this->clear();
      };
      ~UpdateVecVisitor() {
        PetscErrorCode ierr;
        ierr = VecRestoreArray(this->v, &this->array);CHKERRXX(ierr);
        ierr = PetscFree2(fieldSize,j);CHKERRXX(ierr);
      };
      template<typename Point>
      inline void visitPoint(const Point& point, const int orientation) {
        if (nF) {
          switch(mode) {
          case INSERT_VALUES:
            updatePointFields(point, this->insert, false, orientation);break;
          case INSERT_ALL_VALUES:
            updatePointFields(point, this->insert, true,  orientation);break;
          case ADD_VALUES:
            updatePointFields(point, this->add, false, orientation);break;
          case ADD_ALL_VALUES:
            updatePointFields(point, this->add, true,  orientation);break;
          default:
            throw PETSc::Exception("Invalid mode");
          }
        } else {
          switch(mode) {
          case INSERT_VALUES:
            updatePoint(point, this->insert, false, orientation);break;
          case INSERT_ALL_VALUES:
            updatePoint(point, this->insert, true,  orientation);break;
          case ADD_VALUES:
            updatePoint(point, this->add, false, orientation);break;
          case ADD_ALL_VALUES:
            updatePoint(point, this->add, true,  orientation);break;
          default:
            throw PETSc::Exception("Invalid mode");
          }
        }
      }
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow, const int orientation) {}
    public:
      void clear() {
        this->i = 0;
        if (nF) {
          j[0] = 0;
          for(PetscInt f = 1; f < nF; ++f) {
            j[f] = j[f-1] + fieldSize[f-1];
          }
        }
      };
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
      int            i, p;
      // If false, constrained indices are returned as negative values. Otherwise, they are omitted
      bool           freeOnly;
      // If true, it allows space for constrained variables (even if the indices are not returned) Wierd
      bool           skipConstraints;
      value_type    *values;
      bool           allocated;
      point_type    *points;
      bool           allocatedPoints;
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
      IndicesVisitor(const Section& s, Order& o, const int size, const bool unique = false) : section(s), order(o), size(size), i(0), p(0), freeOnly(false), skipConstraints(false) {
        this->values    = new value_type[this->size];
        this->allocated = true;
        if (unique) {
          this->points          = new point_type[this->size];
          this->allocatedPoints = true;
        } else {
          this->points          = NULL;
          this->allocatedPoints = false;
        }
      };
      IndicesVisitor(const Section& s, Order& o, const int size, value_type *values, const bool unique = false) : section(s), order(o), size(size), i(0), p(0), freeOnly(false), skipConstraints(false) {
        this->values    = values;
        this->allocated = false;
        if (unique) {
          this->points          = new point_type[this->size];
          this->allocatedPoints = true;
        } else {
          this->points          = NULL;
          this->allocatedPoints = false;
        }
      };
      ~IndicesVisitor() {
        if (this->allocated) {delete [] this->values;}
        if (this->allocatedPoints) {delete [] this->points;}
      };
      inline void visitPoint(const point_type& point, const int orientation) {
        if (p >= size) {
          ostringstream msg;
          msg << "Too many points (>" << size << ")for IndicesVisitor visitor";
          throw ALE::Exception(msg.str().c_str());
        }
        if (points) {
          int pp;
          for(pp = 0; pp < p; ++pp) {if (points[pp] == point) break;}
          if (pp != p) return;
          points[p++] = point;
        }
        const int cDim = this->section.getConstraintDimension(point);

        if (!cDim) {
          this->getUnconstrainedIndices(point, orientation);
        } else {
          this->getConstrainedIndices(point, orientation);
        }
      }
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow, const int orientation) {}
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
          if (this->allocatedPoints) {delete [] this->points;}
          this->points = new point_type[this->size];
          this->allocatedPoints = true;
        }
      };
      void clear() {this->i = 0; this->p = 0;};
    };
    template<typename Order, typename Value>
    class IndicesVisitor<PetscSection, Order, Value> {
    public:
      typedef Value                      value_type;
      typedef typename Order::point_type point_type;
    protected:
      const PetscSection& section;
      // This can't be const because UniformSection can't have a const restrict(), because of stupid map semantics
      Order&              order;
      int                 size;
      int                 i, p;
      bool                setBC;           // If true, returns indices for constrained dofs, otherwise negative values are returned
      //bool                skipConstraints; // If true, do not return constrained indices at all
      value_type         *values;
      bool                allocated;
      point_type         *points;
      PetscInt            nF;
      PetscInt           *fieldSize;
      PetscInt           *j;
    protected:
      void updatePoint(const point_type& point, const bool setBC, const int orientation = 1) {
        PetscInt        dim;  // The number of dof on this point
        PetscInt        cDim; // The nubmer of constraints on this point
        const PetscInt *cDof; // The indices of the constrained dofs on this point
        PetscInt        offset = this->order.getIndex(point);
        PetscInt        cInd   = 0;
        PetscErrorCode  ierr;

        ierr = PetscSectionGetDof(section, point, &dim);CHKERRXX(ierr);
        ierr = PetscSectionGetConstraintDof(section, point, &cDim);CHKERRXX(ierr);
        if (!cDim || setBC) {
          if (orientation >= 0) {
            for(PetscInt k = 0; k < dim; ++k) {
              values[i+k] = offset+k;
            }
          } else {
            for(PetscInt k = 0; k < dim; ++k) {
              values[i+dim-k-1] = offset+k;
            }
          }
        } else {
          ierr = PetscSectionGetConstraintIndices(section, point, &cDof);CHKERRXX(ierr);
          if (orientation >= 0) {
            for(PetscInt k = 0; k < dim; ++k) {
              if ((cInd < cDim) && (k == cDof[cInd])) {
                // Insert check for returning constrained indices
                values[i+k] = -(offset+k+1);
                ++cInd;
              } else {
                values[i+k] = offset+k;
              }
            }
          } else {
            for(PetscInt k = 0; k < dim; ++k) {
              if ((cInd < cDim) && (k == cDof[cInd])) {
                // Insert check for returning constrained indices
                values[i+dim-k-1] = -(offset+k+1);
                ++cInd;
              } else {
                values[i+dim-k-1] = offset+k;
              }
            }
          }
        }
        i += dim;
      }
      void updatePointFields(const point_type& point, const bool setBC, const int orientation = 1) {
        PetscInt       offset = this->order.getIndex(point);
        PetscInt       fOff   = 0;
        PetscErrorCode ierr;

        for(PetscInt f = 0; f < nF; ++f) {
          PetscInt  dim;  // The number of dof for field f on this point
          PetscInt  comp; // The number of components for field f on this point
          PetscInt  cDim; // The nubmer of constraints for field f on this point
          const PetscInt *cDof; // The indices of the constrained dofs for field f on this point
          PetscInt  cInd = 0;

          ierr = PetscSectionGetFieldComponents(section, f, &comp);CHKERRXX(ierr);
          ierr = PetscSectionGetFieldDof(section, point, f, &dim);CHKERRXX(ierr);
          ierr = PetscSectionGetFieldConstraintDof(section, point, f, &cDim);CHKERRXX(ierr);
          if (!cDim || setBC) {
            if (orientation >= 0) {
              for(PetscInt k = 0; k < dim; ++k) {
                values[j[f]+k] = offset+fOff+k;
              }
            } else {
              for(PetscInt k = dim/comp-1; k >= 0; --k) {
                for(PetscInt c = 0; c < comp; ++c) {
                  values[j[f]+(dim/comp-1-k)*comp+c] = offset+fOff+k*comp+c;
                }
              }
            }
          } else {
            ierr = PetscSectionGetFieldConstraintIndices(section, point, f, &cDof);CHKERRXX(ierr);
            if (orientation >= 0) {
              for(PetscInt k = 0; k < dim; ++k) {
                if ((cInd < cDim) && (k == cDof[cInd])) {
                  values[j[f]+k] = -(offset+fOff+k+1);
                  ++cInd;
                } else {
                  values[j[f]+k] = offset+fOff+k;
                }
              }
            } else {
              for(PetscInt k = dim/comp-1; k >= 0; --k) {
                for(PetscInt c = 0; c < comp; ++c) {
                  PetscInt ind = k*comp+c;
                  if ((cInd < cDim) && (ind == cDof[cInd])) {
                    values[j[f]+(dim/comp-1-k)*comp+c] = -(offset+fOff+ind+1);
                    ++cInd;
                  } else {
                    values[j[f]+(dim/comp-1-k)*comp+c] = offset+fOff+ind;
                  }
                }
              }
            }
          }
          fOff += dim - cDim;
          j[f] += dim;
          i    += dim;
        }
      }
    public:
      IndicesVisitor(const PetscSection& s, Order& o, const int size, const bool unique = false, const PetscInt fieldSize[] = PETSC_NULL) : section(s), order(o), size(size), i(0), p(0), setBC(false) {
        PetscErrorCode ierr;

        ierr = PetscMalloc(this->size * sizeof(value_type), &this->values);CHKERRXX(ierr);
        this->allocated = true;
        this->points    = PETSC_NULL;
        if (unique) {
          ierr = PetscMalloc(this->size * sizeof(point_type), &this->points);CHKERRXX(ierr);
        }
        nF = 0;
        this->fieldSize = this->j = PETSC_NULL;
        if (fieldSize) {
          ierr = PetscSectionGetNumFields(section, &nF);CHKERRXX(ierr);
          ierr = PetscMalloc2(nF,PetscInt,&this->fieldSize,nF,PetscInt,&j);CHKERRXX(ierr);
          for(PetscInt f = 0; f < nF; ++f) {
            this->fieldSize[f] = fieldSize[f];
          }
        }
        this->clear();
      };
      IndicesVisitor(const PetscSection& s, Order& o, const int size, value_type *values, const bool unique = false, const PetscInt fieldSize[] = PETSC_NULL) : section(s), order(o), size(size), i(0), p(0), setBC(false) {
        PetscErrorCode ierr;

        this->values    = values;
        this->allocated = false;
        this->points    = PETSC_NULL;
        if (unique) {
          ierr = PetscMalloc(this->size * sizeof(point_type), &this->points);CHKERRXX(ierr);
        }
        nF = 0;
        this->fieldSize = this->j = PETSC_NULL;
        if (fieldSize) {
          ierr = PetscSectionGetNumFields(section, &nF);CHKERRXX(ierr);
          ierr = PetscMalloc2(nF,PetscInt,&fieldSize,nF,PetscInt,&j);CHKERRXX(ierr);
          for(PetscInt f = 0; f < nF; ++f) {
            this->fieldSize[f] = fieldSize[f];
          }
        }
        this->clear();
      };
      ~IndicesVisitor() {
        PetscErrorCode ierr;
        if (this->allocated) {ierr = PetscFree(values);CHKERRXX(ierr);}
        ierr = PetscFree(points);CHKERRXX(ierr);
        ierr = PetscFree2(fieldSize,j);CHKERRXX(ierr);
      };
    public:
      inline void visitPoint(const point_type& point, const int orientation) {
        if (p >= size) {
          ostringstream msg;
          msg << "Too many points (>" << size << ")for IndicesVisitor visitor";
          throw ALE::Exception(msg.str().c_str());
        }
        if (points) {
          PetscInt pp;
          for(pp = 0; pp < p; ++pp) {if (points[pp] == point) break;}
          if (pp != p) return;
          points[p++] = point;
        }
        if (nF) {
          updatePointFields(point, setBC, orientation);
        } else {
          updatePoint(point, setBC, orientation);
        }
      }
      template<typename Arrow>
      inline void visitArrow(const Arrow& arrow, const int orientation) {}
    public:
      const value_type *getValues() const {return this->values;};
      int  getSize() const {return this->i;};
      int  getMaxSize() const {return this->size;};
      void ensureSize(const int size) {
        this->clear();
        if (size > this->size) {
          PetscErrorCode ierr;

          this->size = size;
          if (this->allocated) {ierr = PetscFree(this->values);CHKERRXX(ierr);}
          ierr = PetscMalloc(this->size * sizeof(value_type), &this->values);CHKERRXX(ierr);
          this->allocated = true;
          ierr = PetscFree(this->points);CHKERRXX(ierr);
          ierr = PetscMalloc(this->size * sizeof(point_type), &this->points);CHKERRXX(ierr);
        }
      };
      void clear() {
        this->p = 0;
        this->i = 0;
        if (nF) {
          j[0] = 0;
          for(PetscInt f = 1; f < nF; ++f) {
            j[f] = j[f-1] + fieldSize[f-1];
          }
        }
      };
    };
    template<typename Sieve, typename Label>
    class MarkVisitor {
    protected:
      Label& label;
      int    marker;
    public:
      MarkVisitor(Label& l, const int marker) : label(l), marker(marker) {};
      inline void visitPoint(const typename Sieve::point_type& point) {
        this->label.setCone(this->marker, point);
      };
      inline void visitArrow(const typename Sieve::arrow_type&) {};
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
      typedef ISieveVisitor::PointRetriever<Sieve,Retriever> TmpRetriever;
      Retriever    pV(200, v, true); // Correct estimate is pow(std::max(1, sieve->getMaxConeSize()), mesh->depth())
      TmpRetriever cV[2] = {TmpRetriever(200,pV), TmpRetriever(200,pV)};
      int          c     = 0;

      v.visitPoint(p, 0);
      // Cone is guarateed to be ordered correctly
      sieve.orientedCone(p, cV[c]);

      while(cV[c].getOrientedSize()) {
        const typename Retriever::oriented_point_type *cone     = cV[c].getOrientedPoints();
        const int                                      coneSize = cV[c].getOrientedSize();
        c = 1 - c;

        for(int p = 0; p < coneSize; ++p) {
          const typename Retriever::point_type& point     = cone[p].first;
          int                                   pO        = cone[p].second == 0 ? 1 : cone[p].second;
          const int                             pConeSize = sieve.getConeSize(point);

          if (pO < 0) {
            if (pO == -pConeSize) {
              sieve.orientedReverseCone(point, cV[c]);
            } else {
              const int numSkip = sieve.getConeSize(point) + pO;

              cV[c].setSkip(cV[c].getSize()+numSkip);
              cV[c].setLimit(cV[c].getSize()+pConeSize);
              sieve.orientedReverseCone(point, cV[c]);
              sieve.orientedReverseCone(point, cV[c]);
              cV[c].setSkip(0);
              cV[c].setLimit(0);
            }
          } else {
            if (pO == 1) {
              sieve.orientedCone(point, cV[c]);
            } else {
              const int numSkip = pO-1;

              cV[c].setSkip(cV[c].getSize()+numSkip);
              cV[c].setLimit(cV[c].getSize()+pConeSize);
              sieve.orientedCone(point, cV[c]);
              sieve.orientedCone(point, cV[c]);
              cV[c].setSkip(0);
              cV[c].setLimit(0);
            }
          }
        }
        cV[1-c].clear();
      }
#if 0
      // These contain arrows paired with orientations from the \emph{previous} arrow
      Obj<orientedArrowArray> cone    = new orientedArrowArray();
      Obj<orientedArrowArray> base    = new orientedArrowArray();
      coneSet                 seen;

      for(typename sieve_type::traits::coneSequence::iterator c_iter = initCone->begin(); c_iter != initCone->end(); ++c_iter) {
        const arrow_type arrow(*c_iter, p);

        cone->push_back(oriented_arrow_type(arrow, 1)); // Notice the orientation of leaf cells is always 1
        closure->push_back(oriented_point_type(*c_iter, orientation->restrictPoint(arrow)[0])); // However, we return the actual arrow orientation
      }
      for(int i = 1; i < depth; ++i) {
        Obj<orientedArrowArray> tmp = cone; cone = base; base = tmp;

        cone->clear();
        for(typename orientedArrowArray::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          const arrow_type&                                     arrow = b_iter->first; // This is an arrow considered in the previous round
          const Obj<typename sieve_type::traits::coneSequence>& pCone = sieve->cone(arrow.source); // We are going to get the cone of this guy
          typename arrow_section_type::value_type               o     = orientation->restrictPoint(arrow)[0]; // The orientation of arrow, which is our pO

          if (b_iter->second < 0) { // This is the problem. The second orientation is carried up, being from the previous round
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
    }
    template<typename Visitor>
    static void orientedStar(const Sieve& sieve, const point_type& p, Visitor& v) {
      typedef ISieveVisitor::PointRetriever<Sieve,Visitor> Retriever;
      Retriever sV[2] = {Retriever(200,v), Retriever(200,v)};
      int       s     = 0;

      v.visitPoint(p, 0);
      // Support is guarateed to be ordered correctly
      sieve.orientedSupport(p, sV[s]);

      while(sV[s].getOrientedSize()) {
        const typename Retriever::oriented_point_type *support     = sV[s].getOrientedPoints();
        const int                                      supportSize = sV[s].getOrientedSize();
        s = 1 - s;

        for(int p = 0; p < supportSize; ++p) {
          const typename Retriever::point_type& point        = support[p].first;
          int                                   pO           = support[p].second == 0 ? 1 : support[p].second;
          const int                             pSupportSize = sieve.getSupportSize(point);

          if (pO < 0) {
            if (pO == -pSupportSize) {
              sieve.orientedReverseSupport(point, sV[s]);
            } else {
              const int numSkip = sieve.getSupportSize(point) + pO;

              sV[s].setSkip(sV[s].getSize()+numSkip);
              sV[s].setLimit(sV[s].getSize()+pSupportSize);
              sieve.orientedReverseSupport(point, sV[s]);
              sieve.orientedReverseSupport(point, sV[s]);
              sV[s].setSkip(0);
              sV[s].setLimit(0);
            }
          } else {
            if (pO == 1) {
              sieve.orientedSupport(point, sV[s]);
            } else {
              const int numSkip = pO-1;

              sV[s].setSkip(sV[s].getSize()+numSkip);
              sV[s].setLimit(sV[s].getSize()+pSupportSize);
              sieve.orientedSupport(point, sV[s]);
              sieve.orientedSupport(point, sV[s]);
              sV[s].setSkip(0);
              sV[s].setLimit(0);
            }
          }
        }
        sV[1-s].clear();
      }
    }
  };

  namespace IFSieveDef {
    template<typename PointType_>
    class Sequence {
    public:
      typedef PointType_ point_type;
      class const_iterator {
      public:
        // Standard iterator typedefs
        typedef std::input_iterator_tag iterator_category;
        typedef PointType_              value_type;
        typedef int                     difference_type;
        typedef value_type*             pointer;
        typedef value_type&             reference;
      protected:
        const point_type *_data;
        int               _pos;
      public:
        const_iterator(const point_type data[], const int pos) : _data(data), _pos(pos) {};
        virtual ~const_iterator() {};
      public:
        virtual bool              operator==(const const_iterator& iter) const {return this->_pos == iter._pos;};
        virtual bool              operator!=(const const_iterator& iter) const {return this->_pos != iter._pos;};
        virtual const value_type  operator*() const {return this->_data[this->_pos];};
        virtual const_iterator&   operator++() {++this->_pos; return *this;};
        virtual const_iterator    operator++(int n) {
          const_iterator tmp(*this);
          ++this->_pos;
          return tmp;
        };
      };
      typedef const_iterator iterator;
    protected:
      const point_type *_data;
      int               _begin;
      int               _end;
    public:
      Sequence(const point_type data[], const int begin, const int end) : _data(data), _begin(begin), _end(end) {};
      virtual ~Sequence() {};
    public:
      virtual iterator begin() const {return iterator(_data, _begin);};
      virtual iterator end()   const {return iterator(_data, _end);};
      size_t size() const {return(_end - _begin);}
      void reset(const point_type data[], const int begin, const int end) {_data = data; _begin = begin; _end = end;};
    };
  }

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
    // Iterator interface
    typedef typename IFSieveDef::Sequence<point_type> coneSequence;
    typedef typename IFSieveDef::Sequence<point_type> supportSequence;
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
    // Sequences
    Obj<coneSequence>    coneSeq;
    Obj<supportSequence> supportSeq;
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
      this->maxConeSize    = 0;
      this->maxSupportSize = 0;
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
    IFSieve(const MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), indexAllocated(false), coneOffsets(NULL), supportOffsets(NULL), pointAllocated(false), maxConeSize(-1), maxSupportSize(-1), baseSize(-1), capSize(-1), cones(NULL), supports(NULL), orientCones(true), coneOrientations(NULL) {
      this->coneSeq    = new coneSequence(NULL, 0, 0);
      this->supportSeq = new supportSequence(NULL, 0, 0);
    };
    IFSieve(const MPI_Comm comm, const point_type& min, const point_type& max, const int debug = 0) : ParallelObject(comm, debug), indexAllocated(false), coneOffsets(NULL), supportOffsets(NULL), pointAllocated(false), maxConeSize(-1), maxSupportSize(-1), baseSize(-1), capSize(-1), cones(NULL), supports(NULL), orientCones(true), coneOrientations(NULL) {
      this->setChart(chart_type(min, max));
      this->coneSeq    = new coneSequence(NULL, 0, 0);
      this->supportSeq = new supportSequence(NULL, 0, 0);
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
    bool orientedCones() const {return this->orientCones;};
    // Raw array access
    offsets_type      getConeOffsets() {return this->coneOffsets;};
    offsets_type      getSupportOffsets() {return this->supportOffsets;};
    cones_type        getCones() {return this->cones;};
    supports_type     getSupports() {return this->supports;};
    orientations_type getConeOrientations() {return this->coneOrientations;};
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
    }
    // Purely for backwards compatibility
    template<typename Color>
    void addArrow(const point_type& p, const point_type& q, const Color c, const bool forceSupport = false) {
      this->addArrow(p, q, forceSupport);
    }
    void addArrow(const point_type& p, const point_type& q, const bool forceSupport = false) {
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
      if (!this->chart.hasPoint(p) || forceSupport) {
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
        const index_type cOStart = oldConeOffsets[p];
        const index_type cOEnd   = oldConeOffsets[p+1];
        const index_type sStart  = this->supportOffsets[p];
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
        if (start != this->coneOffsets[c_iter->first+1]) throw ALE::Exception("Invalid size for new cone array");
      }
      for(typename newpoints_type::const_iterator s_iter = this->newSupports.begin(); s_iter != this->newSupports.end(); ++s_iter) {
        index_type start = this->supportOffsets[s_iter->first];

        for(typename std::vector<point_type>::const_iterator p_iter = s_iter->second.begin(); p_iter != s_iter->second.end(); ++p_iter) {
          this->supports[start++] = *p_iter;
        }
        if (start != this->supportOffsets[s_iter->first+1]) throw ALE::Exception("Invalid size for new support array");
      }
      this->newCones.clear();
      this->newSupports.clear();
      this->destroyPoints(oldChart, oldConeOffsets, &oldCones, oldSupportOffsets, &oldSupports, &oldConeOrientations);
      this->destroyIndices(oldChart, &oldConeOffsets, &oldSupportOffsets);
    };
    // Recalculate the support offsets and fill the supports
    //   This is used when an IFSieve is being used as a label
    void recalculateLabel() {
      ISieveVisitor::PointRetriever<IFSieve> v(1);

      for(point_type p = this->getChart().min(); p < this->getChart().max(); ++p) {
        this->supportOffsets[p+1] = 0;
      }
      this->maxSupportSize = 0;
      for(point_type p = this->getChart().min(); p < this->getChart().max(); ++p) {
        this->cone(p, v);
        if (v.getSize()) {
          this->supportOffsets[v.getPoints()[0]+1]++;
          this->maxSupportSize = std::max(this->maxSupportSize, this->supportOffsets[v.getPoints()[0]+1]);
        }
        v.clear();
      }
      this->prefixSum(this->supportOffsets);
      this->calculateBaseAndCapSize();
      this->symmetrize();
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
    void setCone(const point_type cone, const point_type& p) {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->coneOffsets[p];
      const index_type end   = this->coneOffsets[p+1];

      if (end - start != 1) {throw ALE::Exception("IFSieve setCone() called with too few points.");}
      this->cones[start] = cone;
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
    void symmetrizeSizes(const int numCells, const int numCorners, const int cones[], const int offset = 0) {
      for(point_type p = 0; p < numCells; ++p) {
        const index_type start = p*numCorners;
        const index_type end   = (p+1)*numCorners;

        for(index_type c = start; c < end; ++c) {
          const point_type q = cones[c]+offset;

          this->supportOffsets[q+1]++;
        }
      }
      for(point_type p = numCells; p < this->chart.max(); ++p) {
        this->maxSupportSize = std::max(this->maxSupportSize, this->supportOffsets[p+1]);
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

          this->chart.checkPoint(q);
          this->supports[this->supportOffsets[q]+offsets[q]] = p;
          ++offsets[q];
        }
      }
      for(index_type i = this->chart.min(); i <= this->chart.max(); ++i) {indexAlloc.destroy(offsets+i);}
      indexAlloc.deallocate(offsets, this->chart.size()+1);
    }
    index_type getBaseSize() const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      return this->baseSize;
    }
    index_type getCapSize() const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      return this->capSize;
    }
    template<typename _Section>
    void relabel(_Section& labeling) {
      PetscErrorCode ierr;

      index_type *offsets = indexAlloc.allocate(this->chart.size()+1);
      offsets -= this->chart.min();
      for(index_type i = this->chart.min(); i <= this->chart.max(); ++i) {indexAlloc.construct(offsets+i, index_type(0));}
      // Recalculate coneOffsets
      for(index_type p = this->chart.min(); p < this->chart.max(); ++p) {
        const point_type newP = labeling.restrictPoint(p)[0];

        offsets[newP+1] = this->getConeSize(p);
      }
      this->prefixSum(offsets);
      ierr = PetscMemcpy(this->coneOffsets, offsets, (this->chart.size()+1)*sizeof(index_type));CHKERRXX(ierr);
      // Recalculate supportOffsets
      for(index_type p = this->chart.min(); p < this->chart.max(); ++p) {
        const point_type newP = labeling.restrictPoint(p)[0];

        offsets[newP+1] = this->getSupportSize(p);
      }
      this->prefixSum(offsets);
      ierr = PetscMemcpy(this->supportOffsets, offsets, (this->chart.size()+1)*sizeof(index_type));CHKERRXX(ierr);
      for(index_type i = this->chart.min(); i <= this->chart.max(); ++i) {indexAlloc.destroy(offsets+i);}
      indexAlloc.deallocate(offsets, this->chart.size()+1);
      index_type  size = std::max(this->coneOffsets[this->chart.max()] - this->coneOffsets[this->chart.min()],
				   this->supportOffsets[this->chart.max()] - this->supportOffsets[this->chart.min()]);
      index_type *orientations = offsets = indexAlloc.allocate(size);
      for(index_type i = 0; i < size; ++i) {indexAlloc.construct(orientations+i, index_type(0));}
      // Recalculate coneOrientations
      for(index_type p = this->chart.min(), offset = 0; p < this->chart.max(); ++p) {
        const point_type newP  = labeling.restrictPoint(p)[0];
        const index_type start = this->coneOffsets[newP];
        const index_type end   = this->coneOffsets[newP+1];

        for(index_type c = start; c < end; ++c, ++offset) {
          orientations[c] = this->coneOrientations[offset];
        }
      }
      ierr = PetscMemcpy(this->coneOrientations, orientations, (this->coneOffsets[this->chart.max()] - this->coneOffsets[this->chart.min()])*sizeof(index_type));CHKERRXX(ierr);
      for(index_type i = 0; i < size; ++i) {indexAlloc.destroy(orientations+i);}
      indexAlloc.deallocate(orientations, size);
      // Recalculate cones
      point_type *array = pointAlloc.allocate(size);

      for(index_type i = 0; i < size; ++i) {pointAlloc.construct(array+i, point_type(0));}
      for(index_type p = this->chart.min(), offset = 0; p < this->chart.max(); ++p) {
        const point_type newP  = labeling.restrictPoint(p)[0];
        const index_type start = this->coneOffsets[newP];
        const index_type end   = this->coneOffsets[newP+1];

        for(index_type c = start; c < end; ++c, ++offset) {
          const point_type newQ = labeling.restrictPoint(this->cones[offset])[0];

          array[c] = newQ;
        }
      }
      ierr = PetscMemcpy(this->cones, array, size*sizeof(point_type));CHKERRXX(ierr);
      // Recalculate supports
      for(index_type p = this->chart.min(), offset = 0; p < this->chart.max(); ++p) {
        const point_type newP  = labeling.restrictPoint(p)[0];
        const index_type start = this->supportOffsets[newP];
        const index_type end   = this->supportOffsets[newP+1];

        for(index_type c = start; c < end; ++c, ++offset) {
          const point_type newQ = labeling.restrictPoint(this->supports[offset])[0];

          array[c] = newQ;
        }
      }
      ierr = PetscMemcpy(this->supports, array, size*sizeof(point_type));CHKERRXX(ierr);
      for(index_type i = 0; i < size; ++i) {pointAlloc.destroy(array+i);}
      pointAlloc.deallocate(array, size);
    }
  public: // Traversals
    template<typename Visitor>
    void roots(const Visitor& v) const {
      this->roots(const_cast<Visitor&>(v));
    }
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
    }
    int numRoots() {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      int n = 0;

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->coneOffsets[p+1] == this->coneOffsets[p]) {
          if (this->supportOffsets[p+1]-this->supportOffsets[p] > 0) {
            ++n;
          }
        }
      }
      return n;
    }
    template<typename Visitor>
    void leaves(const Visitor& v) const {
      this->leaves(const_cast<Visitor&>(v));
    }
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
    }
    int numLeaves() {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      int n = 0;

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->supportOffsets[p+1] == this->supportOffsets[p]) {
          if (this->coneOffsets[p+1]-this->coneOffsets[p] > 0) {
            ++n;
          }
        }
      }
      return n;
    }
    template<typename Visitor>
    void base(const Visitor& v) const {
      this->base(const_cast<Visitor&>(v));
    }
    template<typename Visitor>
    void base(Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->coneOffsets[p+1]-this->coneOffsets[p] > 0) {
          v.visitPoint(p);
        }
      }
    }
    template<typename Visitor>
    void cap(const Visitor& v) const {
      this->cap(const_cast<Visitor&>(v));
    }
    template<typename Visitor>
    void cap(Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->supportOffsets[p+1]-this->supportOffsets[p] > 0) {
          v.visitPoint(p);
        }
      }
    }
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
    }
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
    }
    const Obj<coneSequence>& cone(const point_type& p) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      if (!this->chart.hasPoint(p)) {
        this->coneSeq->reset(this->cones, 0, 0);
      } else {
        this->coneSeq->reset(this->cones, this->coneOffsets[p], this->coneOffsets[p+1]);
      }
      return this->coneSeq;
    }
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
    }
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
    }
    const Obj<supportSequence>& support(const point_type& p) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      if (!this->chart.hasPoint(p)) {
        this->supportSeq->reset(this->supports, 0, 0);
      } else {
        this->supportSeq->reset(this->supports, this->supportOffsets[p], this->supportOffsets[p+1]);
      }
      return this->supportSeq;
    }
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
    }
    template<typename Visitor>
    void orientedReverseCone(const point_type& p, Visitor& v) const {
      if (!this->orientCones) {throw ALE::Exception("IFSieve cones have not been oriented.");}
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->coneOffsets[p];
      const index_type end   = this->coneOffsets[p+1];

      for(index_type c = end-1; c >= start; --c) {
        v.visitArrow(arrow_type(this->cones[c], p), this->coneOrientations[c]);
        v.visitPoint(this->cones[c], this->coneOrientations[c] ? -(this->coneOrientations[c]+1): 0);
      }
    }
    template<typename Visitor>
    void orientedSupport(const point_type& p, Visitor& v) const {
      //if (!this->orientCones) {throw ALE::Exception("IFSieve cones have not been oriented.");}
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->supportOffsets[p];
      const index_type end   = this->supportOffsets[p+1];

      for(index_type s = start; s < end; ++s) {
        //v.visitArrow(arrow_type(this->supports[s], p), this->supportOrientations[s]);
        //v.visitPoint(this->supports[s], this->supportOrientations[s]);
        v.visitArrow(arrow_type(this->supports[s], p), 0);
        v.visitPoint(this->supports[s], 0);
      }
    }
    template<typename Visitor>
    void orientedReverseSupport(const point_type& p, Visitor& v) const {
      //if (!this->orientCones) {throw ALE::Exception("IFSieve cones have not been oriented.");}
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->supportOffsets[p];
      const index_type end   = this->supportOffsets[p+1];

      for(index_type s = end-1; s >= start; --s) {
        //v.visitArrow(arrow_type(this->supports[s], p), this->supportOrientations[s]);
        //v.visitPoint(this->supports[s], this->supportOrientations[s] ? -(this->supportOrientations[s]+1): 0);
        v.visitArrow(arrow_type(this->supports[s], p), 0);
        v.visitPoint(this->supports[s], 0);
      }
    }
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
    }
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
    }
    // Helper function
    void insertNSupport(point_type p, pointSet& set, const int depth) {
      const index_type start = this->supportOffsets[p];
      const index_type end   = this->supportOffsets[p+1];

      if (depth == 1) {
        set.insert(&this->supports[start], &this->supports[end]);
      } else {
        for(index_type s = start; s < end; ++s) {
          this->insertNSupport(this->supports[s], set, depth-1);
        }
      }
    }
    // Gives only the join of depth n
    template<typename SequenceIterator, typename Visitor>
    void nJoin(const SequenceIterator& pointsBegin, const SequenceIterator& pointsEnd, const int depth, Visitor& v) {
      typedef std::set<point_type> pointSet;
      pointSet intersect[2] = {pointSet(), pointSet()};
      pointSet tmp;
      int      p = 0;
      int      c = 0;

      for(SequenceIterator p_iter = pointsBegin; p_iter != pointsEnd; ++p_iter) {
        this->chart.checkPoint(*p_iter);
        // Put points in the nSupport into tmp (duplicates are fine since it is a set)
        this->insertNSupport(*p_iter, tmp, depth);
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
    }
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
      PetscSynchronizedPrintf(comm, "Max sizes cone: %d support: %d\n", this->getMaxConeSize(), this->getMaxSupportSize());
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
        /* Roots/Leaves from Sieve do not seem to work */

        for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          if (sieve.support(*b_iter)->size() == 0) {
            renumbering[*b_iter] = max++;
          }
        }
        for(typename Sieve::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
          if (sieve.cone(*c_iter)->size() == 0) {
            renumbering[*c_iter] = max++;
          }
        }
        for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          if (sieve.support(*b_iter)->size() == 0) {
            const typename Sieve::coneSequence::iterator coneBegin = sieve.coneBegin(*b_iter);
            const typename Sieve::coneSequence::iterator coneEnd   = sieve.coneEnd(*b_iter);

            for(typename Sieve::coneSequence::iterator c_iter = coneBegin; c_iter != coneEnd; ++c_iter) {
              if (renumbering.find(*c_iter) == renumbering.end()) {
                renumbering[*c_iter] = max++;
              }
            }
          }
        }
        for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          if (renumbering.find(*b_iter) == renumbering.end()) {
            renumbering[*b_iter] = max++;
          }
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
        if (base->size() || cap->size()) {
          ++max;
        }
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
      for(typename Sieve::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
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
      for(typename Sieve::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
        const Obj<typename Sieve::supportSequence>& support = sieve.support(*c_iter);
        int i = 0;

        for(typename Sieve::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter, ++i) {
          points[i] = renumbering[*s_iter];
        }
        isieve.setSupport(renumbering[*c_iter], points);
      }
      delete [] points;
    }
    template<typename Sieve, typename Renumbering>
    static void convertSieve(Sieve& sieve, DM dm, Renumbering& renumbering, bool renumber = true) {
      // First construct a renumbering of the sieve points
      const Obj<typename Sieve::baseSequence>& base = sieve.base();
      const Obj<typename Sieve::capSequence>&  cap  = sieve.cap();
      PetscInt                                 min  = 0;
      PetscInt                                 max  = 0;
      PetscErrorCode                           ierr;

      if (renumber) {
        /* Roots/Leaves from Sieve do not seem to work */

        for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          if (sieve.support(*b_iter)->size() == 0) {
            renumbering[*b_iter] = max++;
          }
        }
        for(typename Sieve::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
          if (sieve.cone(*c_iter)->size() == 0) {
            renumbering[*c_iter] = max++;
          }
        }
#if 0
        for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          if (sieve.support(*b_iter)->size() == 0) {
            const typename Sieve::coneSequence::iterator coneBegin = sieve.coneBegin(*b_iter);
            const typename Sieve::coneSequence::iterator coneEnd   = sieve.coneEnd(*b_iter);

            for(typename Sieve::coneSequence::iterator c_iter = coneBegin; c_iter != coneEnd; ++c_iter) {
              if (renumbering.find(*c_iter) == renumbering.end()) {
                renumbering[*c_iter] = max++;
              }
            }
          }
        }
#else
        std::vector<typename Sieve::point_type> faces;
        for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          if (sieve.support(*b_iter)->size() == 0) {
            const typename Sieve::coneSequence::iterator coneBegin = sieve.coneBegin(*b_iter);
            const typename Sieve::coneSequence::iterator coneEnd   = sieve.coneEnd(*b_iter);

            for(typename Sieve::coneSequence::iterator c_iter = coneBegin; c_iter != coneEnd; ++c_iter) {
              if (renumbering.find(*c_iter) == renumbering.end()) {
                faces.push_back(*c_iter);
              }
            }
          }
        }
        std::sort(faces.begin(), faces.end());
        typename std::vector<typename Sieve::point_type>::const_iterator fEnd = std::unique(faces.begin(), faces.end());
        for(typename std::vector<typename Sieve::point_type>::const_iterator c_iter = faces.begin(); c_iter != fEnd; ++c_iter) {
          renumbering[*c_iter] = max++;
        }
        faces.clear();
#endif
        for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          if (renumbering.find(*b_iter) == renumbering.end()) {
            renumbering[*b_iter] = max++;
          }
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
        if (base->size() || cap->size()) {
          ++max;
        }
        for(PetscInt p = min; p < max; ++p) {
          renumbering[p] = p;
        }
      }
      // Create the ISieve
      ierr = DMComplexSetChart(dm, min, max);CHKERRXX(ierr);
      // Set cone and support sizes
      size_t maxSize = 0;

      for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        const Obj<typename Sieve::coneSequence>& cone = sieve.cone(*b_iter);

        ierr = DMComplexSetConeSize(dm, renumbering[*b_iter], cone->size());CHKERRXX(ierr);
        maxSize = std::max(maxSize, cone->size());
      }
      for(typename Sieve::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
        const Obj<typename Sieve::supportSequence>& support = sieve.support(*c_iter);

        ierr = DMComplexSetSupportSize(dm, renumbering[*c_iter], support->size());CHKERRXX(ierr);
        maxSize = std::max(maxSize, support->size());
      }
      ierr = DMSetUp(dm);CHKERRXX(ierr);
      // Fill up cones and supports
      typename Sieve::point_type *points = new typename Sieve::point_type[maxSize];

      for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        const Obj<typename Sieve::coneSequence>& cone = sieve.cone(*b_iter);
        int i = 0;

        for(typename Sieve::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++i) {
          points[i] = renumbering[*c_iter];
        }
        ierr = DMComplexSetCone(dm, renumbering[*b_iter], points);CHKERRXX(ierr);
      }
      for(typename Sieve::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
        const Obj<typename Sieve::supportSequence>& support = sieve.support(*c_iter);
        int i = 0;

        for(typename Sieve::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter, ++i) {
          points[i] = renumbering[*s_iter];
        }
        ierr = DMComplexSetSupport(dm, renumbering[*c_iter], points);CHKERRXX(ierr);
      }
      delete [] points;
    }
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
    }
    template<typename Sieve, typename Renumbering, typename ArrowSection>
    static void convertOrientation(Sieve& sieve, DM dm, Renumbering& renumbering, ArrowSection *orientation) {
      PetscInt       maxConeSize;
      PetscErrorCode ierr;

      ierr = DMComplexGetMaxSizes(dm, &maxConeSize, PETSC_NULL);CHKERRXX(ierr);
      if (maxConeSize < 0) return;
      const Obj<typename Sieve::baseSequence>& base = sieve.base();
      int *orientations = new int[maxConeSize];

      for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        const Obj<typename Sieve::coneSequence>& cone = sieve.cone(*b_iter);
        int i = 0;

        for(typename Sieve::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++i) {
          typename ArrowSection::point_type arrow(*c_iter, *b_iter);

          orientations[i] = orientation->restrictPoint(arrow)[0];
        }
        ierr = DMComplexSetConeOrientation(dm, renumbering[*b_iter], orientations);
      }
      delete [] orientations;
    }
    template<typename Section, typename ISection, typename Renumbering>
    static void convertCoordinates(Section& coordinates, ISection& icoordinates, Renumbering& renumbering) {
      const typename Section::chart_type& chart = coordinates.getChart();
      typename ISection::point_type       min   = *chart.begin();
      typename ISection::point_type       max   = *chart.begin();

      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        min = std::min(min, renumbering[*p_iter]);
        max = std::max(max, renumbering[*p_iter]);
      }
      icoordinates.setChart(typename ISection::chart_type(min, max+1));
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        icoordinates.setFiberDimension(renumbering[*p_iter], coordinates.getFiberDimension(*p_iter));
      }
      icoordinates.allocatePoint();
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        icoordinates.updatePoint(renumbering[*p_iter], coordinates.restrictPoint(*p_iter));
      }
    }
    template<typename Section, typename Renumbering>
    static void convertCoordinates(Section& coordinates, PetscSection coordSection, Vec coords, Renumbering& renumbering) {
      const typename Section::chart_type& chart = coordinates.getChart();
      PetscInt                            min   = *chart.begin();
      PetscInt                            max   = *chart.begin();
      PetscScalar                        *a;
      PetscInt                            n;
      PetscErrorCode                      ierr;

      if (!chart.size()) return;
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        min = std::min(min, renumbering[*p_iter]);
        max = std::max(max, renumbering[*p_iter]);
      }
      ierr = PetscSectionSetChart(coordSection, min, max+1);CHKERRXX(ierr);
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        ierr = PetscSectionSetDof(coordSection, renumbering[*p_iter], coordinates.getFiberDimension(*p_iter));CHKERRXX(ierr);
      }
      ierr = PetscSectionSetUp(coordSection);CHKERRXX(ierr);
      ierr = PetscSectionGetStorageSize(coordSection, &n);CHKERRXX(ierr);
      ierr = VecSetSizes(coords, n, PETSC_DETERMINE);CHKERRXX(ierr);
      ierr = VecSetFromOptions(coords);CHKERRXX(ierr);
      ierr = VecGetArray(coords, &a);CHKERRXX(ierr);
      for(typename Section::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const typename Section::value_type *values = coordinates.restrictPoint(*p_iter);
        PetscInt dof, off;

        ierr = PetscSectionGetDof(coordSection, renumbering[*p_iter], &dof);CHKERRXX(ierr);
        ierr = PetscSectionGetOffset(coordSection, renumbering[*p_iter], &off);CHKERRXX(ierr);
        for(int d = 0; d < dof; ++d) {
          a[off+d] = values[d];
        }
      }
      ierr = VecRestoreArray(coords, &a);CHKERRXX(ierr);
    }
    template<typename Label, typename Renumbering>
    static void convertLabel(const Obj<Label>& newLabel, const Obj<Label>& oldLabel, Renumbering& renumbering) {
      for(typename Renumbering::const_iterator p = renumbering.begin(); p != renumbering.end(); ++p) {
        if (oldLabel->getConeSize(p->first)) {
          newLabel->setCone(*oldLabel->cone(p->first)->begin(), p->second);
        }
      }
    }
    template<typename Label, typename Renumbering>
    static void convertLabel(DM dm, const char name[], const Obj<Label>& label, Renumbering& renumbering) {
      PetscErrorCode ierr;

      for(typename Renumbering::const_iterator p = renumbering.begin(); p != renumbering.end(); ++p) {
        if (label->getConeSize(p->first)) {
          ierr = DMComplexSetLabelValue(dm, name, p->second, *label->cone(p->first)->begin());CHKERRXX(ierr);
        }
      }
    }
    template<typename Mesh, typename IMesh, typename Renumbering>
    static void convertMesh(Mesh& mesh, IMesh& imesh, Renumbering& renumbering, bool renumber = true) {
      convertSieve(*mesh.getSieve(), *imesh.getSieve(), renumbering, renumber);
      imesh.stratify();
      convertOrientation(*mesh.getSieve(), *imesh.getSieve(), renumbering, mesh.getArrowSection("orientation").ptr());
      convertCoordinates(*mesh.getRealSection("coordinates"), *imesh.getRealSection("coordinates"), renumbering);
      if (mesh.hasRealSection("normals")) {
        convertCoordinates(*mesh.getRealSection("normals"), *imesh.getRealSection("normals"), renumbering);
      }
      const typename Mesh::labels_type& labels = mesh.getLabels();
      std::string heightName("height");
      std::string depthName("depth");

      for(typename Mesh::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
#ifdef IMESH_NEW_LABELS
        if ((l_iter->first != heightName) && (l_iter->first != depthName)) {
          convertLabel(imesh, l_iter->first, l_iter->second);
        }
#else
        if (renumber) {
          convertLabel(imesh.createLabel(l_iter->first), l_iter->second, renumbering);
        } else {
          imesh.setLabel(l_iter->first, l_iter->second);
        }
#endif
      }
    }
    template<typename Mesh, typename Renumbering>
    static void convertMesh(Mesh& mesh, DM *dm, Renumbering& renumbering, bool renumber = true) {
      PetscSection   coordSection;
      Vec            coordinates;
      PetscErrorCode ierr;

      ierr = DMCreate(mesh.comm(), dm);CHKERRXX(ierr);
      ierr = DMSetType(*dm, DMCOMPLEX);CHKERRXX(ierr);
      ierr = DMComplexSetDimension(*dm, mesh.getDimension());CHKERRXX(ierr);
      convertSieve(*mesh.getSieve(), *dm, renumbering, renumber);
      ierr = DMComplexStratify(*dm);CHKERRXX(ierr);
      convertOrientation(*mesh.getSieve(), *dm, renumbering, mesh.getArrowSection("orientation").ptr());
      ierr = DMComplexGetCoordinateSection(*dm, &coordSection);CHKERRXX(ierr);
      ierr = VecCreate(mesh.comm(), &coordinates);CHKERRXX(ierr);
      convertCoordinates(*mesh.getRealSection("coordinates"), coordSection, coordinates, renumbering);
      ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRXX(ierr);
      const typename Mesh::labels_type& labels = mesh.getLabels();

      for(typename Mesh::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
        convertLabel(*dm, l_iter->first.c_str(), l_iter->second, renumbering);
      }
    }
  };

  class ISieveSerializer {
  public:
    template<typename ISieve>
    static void writeSieve(const std::string& filename, ISieve& sieve) {
      std::ofstream fs;

      if (sieve.commRank() == 0) {
        fs.open(filename.c_str());
      }
      writeSieve(fs, sieve);
      if (sieve.commRank() == 0) {
        fs.close();
      }
    }
    template<typename ISieve>
    static void writeSieve(std::ofstream& fs, ISieve& sieve) {
      typedef ISieveVisitor::PointRetriever<ISieve> Visitor;
      const Obj<typename ISieve::chart_type>& chart = sieve.getChart();
      typename ISieve::point_type             min   = chart->min();
      typename ISieve::point_type             max   = chart->max();
      PetscInt                               *mins  = new PetscInt[sieve.commSize()];
      PetscInt                               *maxs  = new PetscInt[sieve.commSize()];

      // Write sizes
      if (sieve.commRank() == 0) {
        // Write local
        fs << min <<" "<< max << std::endl;
        for(typename ISieve::point_type p = min; p < max; ++p) {
          fs << sieve.getConeSize(p) << " " << sieve.getSupportSize(p) << std::endl;
        }
        // Receive and write remote
        for(int pr = 1; pr < sieve.commSize(); ++pr) {
          PetscInt       minp, maxp;
          PetscInt      *sizes;
          MPI_Status     status;
          PetscErrorCode ierr;

          ierr = MPI_Recv(&minp, 1, MPIU_INT, pr, 1, sieve.comm(), &status);CHKERRXX(ierr);
          ierr = MPI_Recv(&maxp, 1, MPIU_INT, pr, 1, sieve.comm(), &status);CHKERRXX(ierr);
          ierr = PetscMalloc(2*(maxp - minp) * sizeof(PetscInt), &sizes);CHKERRXX(ierr);
          ierr = MPI_Recv(sizes, (maxp - minp)*2, MPIU_INT, pr, 1, sieve.comm(), &status);CHKERRXX(ierr);
          fs << minp <<" "<< maxp << std::endl;
          for(PetscInt s = 0; s < maxp - minp; ++s) {
            fs << sizes[s*2+0] << " " << sizes[s*2+1] << std::endl;
          }
          ierr = PetscFree(sizes);CHKERRXX(ierr);
          mins[pr] = minp;
          maxs[pr] = maxp;
        }
      } else {
        // Send remote
        //PetscInt       min = chart->min();
        //PetscInt       max = chart->max();
        PetscInt       s   = 0;
        PetscInt      *sizes;
        PetscErrorCode ierr;

        ierr = MPI_Send(&min, 1, MPIU_INT, 0, 1, sieve.comm());CHKERRXX(ierr);
        ierr = MPI_Send(&max, 1, MPIU_INT, 0, 1, sieve.comm());CHKERRXX(ierr);
        ierr = PetscMalloc((max - min)*2 * sizeof(PetscInt) + 1, &sizes);CHKERRXX(ierr);
        for(typename ISieve::point_type p = min; p < max; ++p, ++s) {
          sizes[s*2+0] = sieve.getConeSize(p);
          sizes[s*2+1] = sieve.getSupportSize(p);
        }
        ierr = MPI_Send(sizes, (max - min)*2, MPIU_INT, 0, 1, sieve.comm());CHKERRXX(ierr);
        ierr = PetscFree(sizes);CHKERRXX(ierr);
      }
      // Write covering
      if (sieve.commRank() == 0) {
        // Write local
        Visitor pV(std::max(sieve.getMaxConeSize(), sieve.getMaxSupportSize())+1);

        for(typename ISieve::point_type p = min; p < max; ++p) {
          sieve.cone(p, pV);
          const typename Visitor::point_type *cone  = pV.getPoints();
          const int                           cSize = pV.getSize();

          if (cSize > 0) {
            for(int c = 0; c < cSize; ++c) {
              if (c) {fs << " ";}
              fs << cone[c];
            }
            fs << std::endl;
          }
          pV.clear();

          sieve.orientedCone(p, pV);
          const typename Visitor::oriented_point_type *oCone = pV.getOrientedPoints();
          const int                                    oSize = pV.getOrientedSize();

          if (oSize > 0) {
            for(int c = 0; c < oSize; ++c) {
              if (c) {fs << " ";}
              fs << oCone[c].second;
            }
            fs << std::endl;
          }
          pV.clear();

          sieve.support(p, pV);
          const typename Visitor::point_type *support = pV.getPoints();
          const int                           sSize   = pV.getSize();

          if (sSize > 0) {
            for(int s = 0; s < sSize; ++s) {
              if (s) {fs << " ";}
              fs << support[s];
            }
            fs << std::endl;
          }
          pV.clear();
        }
        // Receive and write remote
        for(int pr = 1; pr < sieve.commSize(); ++pr) {
          PetscInt       size;
          PetscInt      *data;
          PetscInt       off = 0;
          MPI_Status     status;
          PetscErrorCode ierr;

          ierr = MPI_Recv(&size, 1, MPIU_INT, pr, 1, sieve.comm(), &status);CHKERRXX(ierr);
          ierr = PetscMalloc(size*sizeof(PetscInt), &data);CHKERRXX(ierr);
          ierr = MPI_Recv(data, size, MPIU_INT, pr, 1, sieve.comm(), &status);CHKERRXX(ierr);
          for(typename ISieve::point_type p = mins[pr]; p < maxs[pr]; ++p) {
            PetscInt cSize = data[off++];

            fs << cSize << std::endl;
            if (cSize > 0) {
              for(int c = 0; c < cSize; ++c) {
                if (c) {fs << " ";}
                fs << data[off++];
              }
              fs << std::endl;
            }
            PetscInt oSize = data[off++];

            if (oSize > 0) {
              for(int c = 0; c < oSize; ++c) {
                if (c) {fs << " ";}
                fs << data[off++];
              }
              fs << std::endl;
            }
            PetscInt sSize = data[off++];

            fs << sSize << std::endl;
            if (sSize > 0) {
              for(int s = 0; s < sSize; ++s) {
                if (s) {fs << " ";}
                fs << data[off++];
              }
              fs << std::endl;
            }
          }
          assert(off == size);
          ierr = PetscFree(data);CHKERRXX(ierr);
        }
      } else {
        // Send remote
        Visitor pV(std::max(sieve.getMaxConeSize(), sieve.getMaxSupportSize())+1);
        PetscInt totalConeSize    = 0;
        PetscInt totalSupportSize = 0;

        for(typename ISieve::point_type p = min; p < max; ++p) {
          totalConeSize    += sieve.getConeSize(p);
          totalSupportSize += sieve.getSupportSize(p);
        }
        PetscInt       size = (sieve.getChart().size()+totalConeSize)*2 + sieve.getChart().size()+totalSupportSize;
        PetscInt       off  = 0;
        PetscInt      *data;
        PetscErrorCode ierr;

        ierr = MPI_Send(&size, 1, MPIU_INT, 0, 1, sieve.comm());CHKERRXX(ierr);
        // There is no nice way to make a generic MPI type here. Really sucky
        ierr = PetscMalloc(size * sizeof(PetscInt), &data);CHKERRXX(ierr);
        for(typename ISieve::point_type p = min; p < max; ++p) {
          sieve.cone(p, pV);
          const typename Visitor::point_type *cone  = pV.getPoints();
          const int                           cSize = pV.getSize();

          data[off++] = cSize;
          for(int c = 0; c < cSize; ++c) {
            data[off++] = cone[c];
          }
          pV.clear();

          sieve.orientedCone(p, pV);
          const typename Visitor::oriented_point_type *oCone = pV.getOrientedPoints();
          const int                                    oSize = pV.getOrientedSize();

          data[off++] = oSize;
          for(int c = 0; c < oSize; ++c) {
            data[off++] = oCone[c].second;
          }
          pV.clear();

          sieve.support(p, pV);
          const typename Visitor::point_type *support = pV.getPoints();
          const int                           sSize   = pV.getSize();

          data[off++] = sSize;
          for(int s = 0; s < sSize; ++s) {
            data[off++] = support[s];
          }
          pV.clear();
        }
        ierr = MPI_Send(data, size, MPIU_INT, 0, 1, sieve.comm());CHKERRXX(ierr);
        ierr = PetscFree(data);CHKERRXX(ierr);
      }
      delete [] mins;
      delete [] maxs;
      // Output renumbering
    }
    template<typename ISieve>
    static void loadSieve(const std::string& filename, ISieve& sieve) {
      std::ifstream fs;

      if (sieve.commRank() == 0) {
        fs.open(filename.c_str());
      }
      loadSieve(fs, sieve);
      if (sieve.commRank() == 0) {
        fs.close();
      }
    }
    template<typename ISieve>
    static void loadSieve(std::ifstream& fs, ISieve& sieve) {
      typename ISieve::point_type min, max;
      PetscInt                   *mins = new PetscInt[sieve.commSize()];
      PetscInt                   *maxs = new PetscInt[sieve.commSize()];
      PetscInt                   *totalConeSizes    = new PetscInt[sieve.commSize()];
      PetscInt                   *totalSupportSizes = new PetscInt[sieve.commSize()];

      // Load sizes
      if (sieve.commRank() == 0) {
        // Load local
        fs >> min;
        fs >> max;
        sieve.setChart(typename ISieve::chart_type(min, max));
        for(typename ISieve::point_type p = min; p < max; ++p) {
          typename ISieve::index_type coneSize, supportSize;

          fs >> coneSize;
          fs >> supportSize;
          sieve.setConeSize(p, coneSize);
          sieve.setSupportSize(p, supportSize);
        }
        // Load and send remote
        for(int pr = 1; pr < sieve.commSize(); ++pr) {
          PetscInt       minp, maxp;
          PetscInt      *sizes;
          PetscErrorCode ierr;

          fs >> minp;
          fs >> maxp;
          ierr = MPI_Send(&minp, 1, MPIU_INT, pr, 1, sieve.comm());CHKERRXX(ierr);
          ierr = MPI_Send(&maxp, 1, MPIU_INT, pr, 1, sieve.comm());CHKERRXX(ierr);
          ierr = PetscMalloc((maxp - minp)*2 * sizeof(PetscInt), &sizes);CHKERRXX(ierr);
          totalConeSizes[pr]    = 0;
          totalSupportSizes[pr] = 0;
          for(PetscInt s = 0; s < maxp - minp; ++s) {
            fs >> sizes[s*2+0];
            fs >> sizes[s*2+1];
            totalConeSizes[pr]    += sizes[s*2+0];
            totalSupportSizes[pr] += sizes[s*2+1];
          }
          ierr = MPI_Send(sizes, (maxp - minp)*2, MPIU_INT, pr, 1, sieve.comm());CHKERRXX(ierr);
          ierr = PetscFree(sizes);CHKERRXX(ierr);
          mins[pr] = minp;
          maxs[pr] = maxp;
        }
      } else {
        // Load remote
        PetscInt       s   = 0;
        PetscInt      *sizes;
        MPI_Status     status;
        PetscErrorCode ierr;

        ierr = MPI_Recv(&min, 1, MPIU_INT, 0, 1, sieve.comm(), &status);CHKERRXX(ierr);
        ierr = MPI_Recv(&max, 1, MPIU_INT, 0, 1, sieve.comm(), &status);CHKERRXX(ierr);
        sieve.setChart(typename ISieve::chart_type(min, max));
        ierr = PetscMalloc((max - min)*2 * sizeof(PetscInt), &sizes);CHKERRXX(ierr);
        ierr = MPI_Recv(sizes, (max - min)*2, MPIU_INT, 0, 1, sieve.comm(), &status);CHKERRXX(ierr);
        for(typename ISieve::point_type p = min; p < max; ++p, ++s) {
          sieve.setConeSize(p, sizes[s*2+0]);
          sieve.setSupportSize(p, sizes[s*2+1]);
        }
        ierr = PetscFree(sizes);CHKERRXX(ierr);
      }
      sieve.allocate();
      // Load covering
      if (sieve.commRank() == 0) {
        // Load local
        typename ISieve::index_type  maxSize = std::max(sieve.getMaxConeSize(), sieve.getMaxSupportSize());
        typename ISieve::point_type *points  = new typename ISieve::point_type[maxSize];

        for(typename ISieve::point_type p = min; p < max; ++p) {
          typename ISieve::index_type coneSize    = sieve.getConeSize(p);
          typename ISieve::index_type supportSize = sieve.getSupportSize(p);

          if (coneSize > 0) {
            for(int c = 0; c < coneSize; ++c) {
              fs >> points[c];
            }
            sieve.setCone(points, p);
            if (sieve.orientedCones()) {
              for(int c = 0; c < coneSize; ++c) {
                fs >> points[c];
              }
              sieve.setConeOrientation(points, p);
            }
          }
          if (supportSize > 0) {
            for(int s = 0; s < supportSize; ++s) {
              fs >> points[s];
            }
            sieve.setSupport(p, points);
          }
        }
        delete [] points;
        // Load and send remote
        for(int pr = 1; pr < sieve.commSize(); ++pr) {
          PetscInt       size = (maxs[pr] - mins[pr])+totalConeSizes[pr]*2 + (maxs[pr] - mins[pr])+totalSupportSizes[pr];
          PetscInt       off  = 0;
          PetscInt      *data;
          PetscErrorCode ierr;

          ierr = MPI_Send(&size, 1, MPIU_INT, pr, 1, sieve.comm());CHKERRXX(ierr);
          // There is no nice way to make a generic MPI type here. Really sucky
          ierr = PetscMalloc(size * sizeof(PetscInt), &data);CHKERRXX(ierr);
          for(typename ISieve::point_type p = mins[pr]; p < maxs[pr]; ++p) {
            PetscInt coneSize, supportSize;

            fs >> coneSize;
            data[off++] = coneSize;
            if (coneSize > 0) {
              for(int c = 0; c < coneSize; ++c) {
                fs >> data[off++];
              }
              for(int c = 0; c < coneSize; ++c) {
                fs >> data[off++];
              }
            }
            fs >> supportSize;
            data[off++] = supportSize;
            if (supportSize > 0) {
              for(int s = 0; s < supportSize; ++s) {
                fs >> data[off++];
              }
            }
          }
          assert(off == size);
          ierr = MPI_Send(data, size, MPIU_INT, pr, 1, sieve.comm());CHKERRXX(ierr);
          ierr = PetscFree(data);CHKERRXX(ierr);
        }
        delete [] mins;
        delete [] maxs;
      } else {
        // Load remote
        PetscInt       size;
        PetscInt      *data;
        PetscInt       off = 0;
        MPI_Status     status;
        PetscErrorCode ierr;

        ierr = MPI_Recv(&size, 1, MPIU_INT, 0, 1, sieve.comm(), &status);CHKERRXX(ierr);
        ierr = PetscMalloc(size*sizeof(PetscInt), &data);CHKERRXX(ierr);
        ierr = MPI_Recv(data, size, MPIU_INT, 0, 1, sieve.comm(), &status);CHKERRXX(ierr);
        for(typename ISieve::point_type p = min; p < max; ++p) {
          typename ISieve::index_type coneSize    = sieve.getConeSize(p);
          typename ISieve::index_type supportSize = sieve.getSupportSize(p);
          PetscInt cs = data[off++];

          assert(cs == coneSize);
          if (coneSize > 0) {
            sieve.setCone(&data[off], p);
            off += coneSize;
            if (sieve.orientedCones()) {
              sieve.setConeOrientation(&data[off], p);
              off += coneSize;
            }
          }
          PetscInt ss = data[off++];

          assert(ss == supportSize);
          if (supportSize > 0) {
            sieve.setSupport(p, &data[off]);
            off += supportSize;
          }
        }
        assert(off == size);
      }
      // Load renumbering
    }
  };
}

#endif
