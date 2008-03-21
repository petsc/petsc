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

  // Interval Final Sieve
  // This is just two CSR matrices that give cones and supports
  //   It is completely static and cannot be resized
  //   It will operator on visitors, rather than sequences (which are messy)
  template<typename Point_, typename Allocator_ = malloc_allocator<Point_> >
  class IFSieve : public ParallelObject {
  public:
    // Types
    typedef Point_                             point_type;
    typedef SimpleArrow<point_type,point_type> arrow_type;
    typedef typename arrow_type::source_type   source_type;
    typedef typename arrow_type::target_type   target_type;
    typedef int                                index_type;
    // Allocators
    typedef Allocator_                                                        point_allocator_type;
    typedef typename point_allocator_type::template rebind<index_type>::other index_allocator_type;
    // Interval
    typedef Interval<point_type, point_allocator_type> chart_type;
  protected:
    // Arrow Containers
    typedef index_type *offsets_type;
    typedef point_type *cones_type;
    typedef point_type *supports_type;
  public:
    // Visitor
    class Visitor {
    public:
      virtual ~Visitor() {};
      virtual void visitArrow(const arrow_type& arrow) const {};
      virtual void visitPoint(const point_type& point) const {};
    };
  protected:
    // Data
    bool                 indexAllocated;
    offsets_type         coneOffsets;
    offsets_type         supportOffsets;
    bool                 pointAllocated;
    cones_type           cones;
    supports_type        supports;
    chart_type           chart;
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
    };
    void createPoints() {
      this->cones = pointAlloc.allocate(this->coneOffsets[this->chart.max()]-this->coneOffsets[this->chart.min()]);
      for(index_type i = this->coneOffsets[this->chart.min()]; i < this->coneOffsets[this->chart.max()]; ++i) {pointAlloc.construct(this->cones+i, point_type(0));}
      this->supports = pointAlloc.allocate(this->supportOffsets[this->chart.max()]-this->supportOffsets[this->chart.min()]);
      for(index_type i = this->supportOffsets[this->chart.min()]; i < this->supportOffsets[this->chart.max()]; ++i) {pointAlloc.construct(this->supports+i, point_type(0));}
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
      this->pointAllocated = false;
    };
    void prefixSum(const offsets_type array) {
      for(index_type p = this->chart.min()+1; p <= this->chart.max(); ++p) {
        array[p] = array[p] + array[p-1];
      }
    };
  public:
    IFSieve(const MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), indexAllocated(false), coneOffsets(NULL), supportOffsets(NULL), pointAllocated(false), cones(NULL), supports(NULL) {};
    IFSieve(const MPI_Comm comm, const point_type& min, const point_type& max, const int debug = 0) : ParallelObject(comm, debug), indexAllocated(false), coneOffsets(NULL), supportOffsets(NULL), pointAllocated(false), cones(NULL), supports(NULL) {
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
  public: // Construction
    void setConeSize(const point_type& p, const index_type c) {
      if (!this->indexAllocated) {throw ALE::Exception("IFSieve indices have not been allocated.");}
      if (this->pointAllocated) {throw ALE::Exception("IFSieve points have already been allocated.");}
      this->chart.checkPoint(p);
      this->coneOffsets[p+1] = c;
    };
    void setSupportSize(const point_type& p, const index_type s) {
      if (!this->indexAllocated) {throw ALE::Exception("IFSieve indices have not been allocated.");}
      if (this->pointAllocated) {throw ALE::Exception("IFSieve points have already been allocated.");}
      this->chart.checkPoint(p);
      this->supportOffsets[p+1] = s;
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
    void setSupport(const point_type& p, const point_type support[]) {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->supportOffsets[p];
      const index_type end   = this->supportOffsets[p+1];

      for(index_type c = start, i = 0; c < end; ++c, ++i) {
        this->supports[c] = support[i];
      }
    };
  public: // Queries
    void base(const Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->coneOffsets[p+1]-this->coneOffsets[p] > 0) {
          v.visitPoint(p);
        }
      }
    };
    void cap(const Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}

      for(point_type p = this->chart.min(); p < this->chart.max(); ++p) {
        if (this->supportOffsets[p+1]-this->supportOffsets[p] > 0) {
          v.visitPoint(p);
        }
      }
    };
    void cone(const point_type& p, const Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->coneOffsets[p];
      const index_type end   = this->coneOffsets[p+1];

      for(index_type c = start, i = 0; c < end; ++c, ++i) {
        v.visitArrow(arrow_type(this->cones[c], p));
      }
    };
    void support(const point_type& p, const Visitor& v) const {
      if (!this->pointAllocated) {throw ALE::Exception("IFSieve points have not been allocated.");}
      this->chart.checkPoint(p);
      const index_type start = this->supportOffsets[p];
      const index_type end   = this->supportOffsets[p+1];

      for(index_type c = start, i = 0; c < end; ++c, ++i) {
        v.visitArrow(arrow_type(p, this->supports[c]));
      }
    };
  public: // Viewing
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      class PrintVisitor : public Visitor {
      protected:
        ostringstream& os;
      public:
        PrintVisitor(ostringstream& s) : os(s) {};
        virtual void visitArrow(const arrow_type& arrow) const {
          this->os << arrow << std::endl;
        };
      };
      class ReversePrintVisitor : public PrintVisitor {
      public:
        ReversePrintVisitor(ostringstream& s) : PrintVisitor(s) {};
        virtual void visitArrow(const arrow_type& arrow) const {
          this->os << arrow.target << "<----" << arrow.source << std::endl;
        };
      };
      class ConeVisitor : public PrintVisitor {
      protected:
        const IFSieve& s;
      public:
        ConeVisitor(const IFSieve& sieve, ostringstream& s) : PrintVisitor(s), s(sieve) {};
        virtual void visitPoint(const point_type& p) const {
          this->s.cone(p, *this);
        };
      };
      class SupportVisitor : public ReversePrintVisitor {
      protected:
        const IFSieve& s;
      public:
        SupportVisitor(const IFSieve& sieve, ostringstream& s) : ReversePrintVisitor(s), s(sieve) {};
        virtual void visitPoint(const point_type& p) const {
          this->s.support(p, *this);
        };
      };
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
      this->base(ConeVisitor(*this, txt));
      if(rank == 0) {
        txt << "base <-- cap:" << std::endl;
      }
      this->cap(SupportVisitor(*this, txt));
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };
}

#endif
