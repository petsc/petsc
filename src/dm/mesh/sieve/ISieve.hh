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
    void checkPoint(const point_type& point) const {
      if (point < this->_min || point >= this->_max) {
        ostringstream msg;
        msg << "Invalid point " << point << " not in " << this << std::endl;
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
    template<typename Sieve, int size, typename Visitor = NullVisitor<Sieve> >
    class PointRetriever {
    public:
      typedef typename Sieve::point_type point_type;
      typedef typename Sieve::arrow_type arrow_type;
      typedef std::pair<point_type,int>  oriented_point_type;
    protected:
      point_type          points[size];
      oriented_point_type oPoints[size];
      size_t              i, o;
      Visitor            *visitor;
    public:
      PointRetriever() : i(0), o(0) {
        static Visitor nV;
        this->visitor = &nV;
      };
      PointRetriever(Visitor& v) : i(0), o(0), visitor(&v) {};
      void visitArrow(const arrow_type& arrow) {
        this->visitor->visitArrow(arrow);
      };
      void visitArrow(const arrow_type& arrow, const int orientation) {
        this->visitor->visitArrow(arrow, orientation);
      };
      void visitPoint(const point_type& point) {
        if (i >= size) throw ALE::Exception("Too many points for visitor");
        points[i++] = point;
        this->visitor->visitPoint(point);
      };
      void visitPoint(const point_type& point, const int orientation) {
        if (o >= size) throw ALE::Exception("Too many points for visitor");
        oPoints[o++] = oriented_point_type(point, orientation);
        this->visitor->visitPoint(point, orientation);
      };
    public:
      const size_t               getSize() const {return this->i;};
      const point_type          *getPoints() const {return this->points;};
      const size_t               getOrientedSize() const {return this->o;};
      const oriented_point_type *getOrientedPoints() const {return this->oPoints;};
      void clear() {this->i = this->o = 0;};
    };
    template<typename Sieve, typename Visitor = NullVisitor<Sieve> >
    class PointSetRetriever {
    public:
      typedef typename Sieve::point_type       point_type;
      typedef std::vector<point_type>          points_type;
      typedef typename Sieve::arrow_type       arrow_type;
      typedef std::pair<point_type,int>        oriented_point_type;
      typedef std::vector<oriented_point_type> oriented_points_type;
    protected:
      points_type          points;
      oriented_points_type oPoints;
      Visitor             *visitor;
    public:
      PointSetRetriever() {
        static Visitor nV;
        this->visitor = &nV;
      };
      PointSetRetriever(Visitor& v) : visitor(&v) {};
      void visitArrow(const arrow_type& arrow) {
        this->visitor->visitArrow(arrow);
      };
      void visitArrow(const arrow_type& arrow, const int orientation) {
        this->visitor->visitArrow(arrow, orientation);
      };
      void visitPoint(const point_type& point) {
        const size_t size = points.size();
        size_t       p;

        for(p = 0; p < size; ++p) {if (points[p] == point) break;}
        if (p == size) {points.push_back(point);}
        this->visitor->visitPoint(point);
      };
      void visitPoint(const point_type& point, const int orientation) {
        const oriented_point_type oP(point, orientation);
        const size_t size = oPoints.size();
        size_t       p;

        for(p = 0; p < size; ++p) {if (oPoints[p] == oP) break;}
        if (p == size) {oPoints.push_back(oP);}
        this->visitor->visitPoint(point, orientation);
      };
    public:
      const size_t                getSize() const {return this->points.size();};
      const points_type           getPoints() const {return this->points;};
      const size_t                getOrientedSize() const {return this->oPoints.size();};
      const oriented_points_type& getOrientedPoints() const {return this->oPoints;};
      void clear() {this->points.clear(); this->oPoints.clear();};
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
    class SupportVisitor {
    protected:
      const Sieve& sieve;
      Visitor&     visitor;
      bool         useSource;
    public:
      SupportVisitor(const Sieve& s, Visitor& v, bool useSource = true) : sieve(s), visitor(v), useSource(useSource) {};
      void visitArrow(const typename Sieve::arrow_type& arrow) {
        if (this->useSource) {
          this->sieve.support(arrow.source, visitor);
        } else {
          this->sieve.support(arrow.target, visitor);
        }
      };
    };
    class PrintVisitor {
    protected:
      ostringstream& os;
    public:
      PrintVisitor(ostringstream& s) : os(s) {};
      template<typename Arrow>
      void visitArrow(const Arrow& arrow) const {
        this->os << arrow << std::endl;
      };
    };
    class ReversePrintVisitor : public PrintVisitor {
    public:
      ReversePrintVisitor(ostringstream& s) : PrintVisitor(s) {};
      template<typename Arrow>
      void visitArrow(const Arrow& arrow) const {
        this->os << arrow.target << "<----" << arrow.source << std::endl;
      };
    };
    template<typename Sieve>
    class ConePrintVisitor : public ReversePrintVisitor {
    protected:
      const Sieve& s;
    public:
      ConePrintVisitor(const Sieve& sieve, ostringstream& s) : ReversePrintVisitor(s), s(sieve) {};
      void visitPoint(const typename Sieve::point_type& p) const {
        this->s.cone(p, *this);
      };
    };
    template<typename Sieve>
    class SupportPrintVisitor : public PrintVisitor {
    protected:
      const Sieve& s;
    public:
      SupportPrintVisitor(const Sieve& sieve, ostringstream& s) : PrintVisitor(s), s(sieve) {};
      void visitPoint(const typename Sieve::point_type& p) const {
        this->s.support(p, *this);
      };
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
      void visitArrow(const typename Sieve::arrow_type& arrow) const {
        if (this->isCone) {
          if (this->seen.find(arrow.source) != this->seen.end()) return;
          this->seen.insert(arrow.source);
          this->visitor.visitPoint(arrow.source);
          this->visitor.visitArrow(arrow);
          if (this->sieve.getConeSize(arrow.source)) {
            this->sieve.cone(arrow.source, *this);
          } else {
            this->visitor.visitPoint(arrow.target);
          }
        } else {
          if (this->seen.find(arrow.target) != this->seen.end()) return;
          this->seen.insert(arrow.target);
          this->visitor.visitPoint(arrow.target);
          this->visitor.visitArrow(arrow);
          if (this->sieve.getSupportSize(arrow.source)) {
            this->sieve.support(arrow.target, *this);
          } else {
            this->visitor.visitPoint(arrow.source);
          }
        }
      };
    public:
      bool getIsCone() {return this->isCone;};
      void setIsCone(const bool isCone) {this->isCone = isCone;};
      const std::set<typename Sieve::point_type>& getPoints() {return this->seen;};
    };
    template<typename Sieve, typename Section>
    class SizeVisitor {
    protected:
      const Section& section;
      int            size;
    public:
      SizeVisitor(const Section& s) : section(s), size(0) {};
      void visitPoint(const typename Sieve::point_type& point) const {
        this->size += section.getConstrainedFiberDimension(point);
      };
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
      void visitPoint(const typename Sieve::point_type& point) const {
        this->size += section.getFiberDimension(point);
      };
    public:
      int getSize() {return this->size;};
    };
  };

  template<typename Sieve>
  class ISieveTraversal {
  public:
    typedef typename Sieve::point_type point_type;
  public:
    template<typename Visitor>
    static void orientedClosure(const Sieve& sieve, const point_type& p, Visitor& v) {
      typedef ISieveVisitor::PointRetriever<Sieve,200,Visitor> Retriever;
      Retriever cV[2] = {Retriever(v), Retriever(v)};
      int       c     = 0;

      v.visitPoint(p, 0);
      // Cone is guarateed to be ordered correctly
      std::cout << "Getting cone from " << p << std::endl;
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
              std::cout << "Getting cone from " << point << std::endl;
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
    cones_type           cones;
    supports_type        supports;
    bool                 orientCones;
    orientations_type    coneOrientations;
    chart_type           chart;
    int_allocator_type   intAlloc;
    index_allocator_type indexAlloc;
    point_allocator_type pointAlloc;
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
    void destroyIndices() {
      if (this->coneOffsets) {
        for(index_type i = this->chart.min(); i <= this->chart.max(); ++i) {indexAlloc.destroy(this->coneOffsets+i);}
        this->coneOffsets += this->chart.min();
        indexAlloc.deallocate(this->coneOffsets, this->chart.size()+1);
        this->coneOffsets = NULL;
      }
      if (this->supportOffsets) {
        for(index_type i = this->chart.min(); i <= this->chart.max(); ++i) {indexAlloc.destroy(this->supportOffsets+i);}
        this->supportOffsets += this->chart.min();
        indexAlloc.deallocate(this->supportOffsets, this->chart.size()+1);
        this->supportOffsets = NULL;
      }
      this->indexAllocated = false;
      this->maxConeSize    = -1;
      this->maxSupportSize = -1;
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
    void destroyPoints() {
      if (this->cones) {
        for(index_type i = this->coneOffsets[this->chart.min()]; i < this->coneOffsets[this->chart.max()]; ++i) {pointAlloc.destroy(this->cones+i);}
        pointAlloc.deallocate(this->cones, this->coneOffsets[this->chart.max()]-this->coneOffsets[this->chart.min()]);
        this->cones = NULL;
      }
      if (this->supports) {
        for(index_type i = this->supportOffsets[this->chart.min()]; i < this->supportOffsets[this->chart.max()]; ++i) {pointAlloc.destroy(this->supports+i);}
        pointAlloc.deallocate(this->supports, this->supportOffsets[this->chart.max()]-this->supportOffsets[this->chart.min()]);
        this->supports = NULL;
      }
      if (this->coneOrientations) {
        for(index_type i = this->coneOffsets[this->chart.min()]; i < this->coneOffsets[this->chart.max()]; ++i) {pointAlloc.destroy(this->coneOrientations+i);}
        intAlloc.deallocate(this->coneOrientations, this->coneOffsets[this->chart.max()]-this->coneOffsets[this->chart.min()]);
        this->coneOrientations = NULL;
      }
      this->pointAllocated = false;
    };
    void prefixSum(const offsets_type array) {
      for(index_type p = this->chart.min()+1; p <= this->chart.max(); ++p) {
        array[p] = array[p] + array[p-1];
      }
    };
  public:
    IFSieve(const MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), indexAllocated(false), coneOffsets(NULL), supportOffsets(NULL), pointAllocated(false), maxConeSize(-1), maxSupportSize(-1), cones(NULL), supports(NULL), orientCones(true), coneOrientations(NULL) {};
    IFSieve(const MPI_Comm comm, const point_type& min, const point_type& max, const int debug = 0) : ParallelObject(comm, debug), indexAllocated(false), coneOffsets(NULL), supportOffsets(NULL), pointAllocated(false), maxConeSize(-1), maxSupportSize(-1), cones(NULL), supports(NULL), orientCones(true), coneOrientations(NULL) {
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
  public: // Construction
    index_type getConeSize(const point_type& p) const {
      if (!this->indexAllocated) {throw ALE::Exception("IFSieve indices have not been allocated.");}
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      return this->coneOffsets[p+1]-this->coneOffsets[p];
    };
    void setConeSize(const point_type& p, const index_type c) {
      if (!this->indexAllocated) {throw ALE::Exception("IFSieve indices have not been allocated.");}
      if (this->pointAllocated) {throw ALE::Exception("IFSieve points have already been allocated.");}
      this->chart.checkPoint(p);
      this->coneOffsets[p+1] = c;
      this->maxConeSize = std::max(this->maxConeSize, c);
    };
    index_type getSupportSize(const point_type& p) const {
      if (!this->indexAllocated) {throw ALE::Exception("IFSieve indices have not been allocated.");}
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      return this->supportOffsets[p+1]-this->supportOffsets[p];
    };
    void setSupportSize(const point_type& p, const index_type s) {
      if (!this->indexAllocated) {throw ALE::Exception("IFSieve indices have not been allocated.");}
      if (this->pointAllocated) {throw ALE::Exception("IFSieve points have already been allocated.");}
      this->chart.checkPoint(p);
      this->supportOffsets[p+1] = s;
      this->maxSupportSize = std::max(this->maxSupportSize, s);
    };
    void allocate() {
      if (this->pointAllocated) {throw ALE::Exception("IFSieve points have already been allocated.");}
      this->prefixSum(this->coneOffsets);
      this->prefixSum(this->supportOffsets);
      this->createPoints();
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

        for(index_type c = start, i = 0; c < end; ++c, ++i) {
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

      for(index_type c = start, i = 0; c < end; ++c, ++i) {
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

        for(index_type s = start, i = 0; s < end; ++s, ++i) {
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

      for(index_type s = start, i = 0; s < end; ++s, ++i) {
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

      for(index_type c = start, i = 0; c < end; ++c, ++i) {
        v.visitArrow(arrow_type(this->cones[c], p), this->coneOrientations[c]);
        v.visitPoint(this->cones[c], this->coneOrientations[c]);
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
      this->cap(ISieveVisitor::SupportPrintVisitor<this_type>(*this, txt));
      if(rank == 0) {
        txt << "base <-- cap:" << std::endl;
      }
      this->base(ISieveVisitor::ConePrintVisitor<this_type>(*this, txt));
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };

  class ISieveConverter {
  public:
    template<typename Sieve, typename ISieve, typename Renumbering>
    static void convertSieve(Sieve& sieve, ISieve& isieve, Renumbering& renumbering) {
      // First construct a renumbering of the sieve points
      const Obj<typename Sieve::baseSequence>& base     = sieve.base();
      typename ISieve::point_type              newPoint = 0;

      for(typename Sieve::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
        renumbering[*b_iter] = newPoint++;
      }
      const Obj<typename Sieve::capSequence>& cap = sieve.cap();

      for(typename Sieve::baseSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
        if (renumbering.find(*c_iter) == renumbering.end()) {
          renumbering[*c_iter] = newPoint++;
        }
      }
      // Create the ISieve
      isieve.setChart(typename ISieve::chart_type(0, newPoint));
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
  };
}

#endif
