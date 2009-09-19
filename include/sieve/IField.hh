#ifndef included_ALE_IField_hh
#define included_ALE_IField_hh

#ifndef  included_ALE_Field_hh
#include <Field.hh>
#endif

#ifndef  included_ALE_ISieve_hh
#include <ISieve.hh>
#endif

// An ISection (or IntervalSection) is a section over a simple interval domain
namespace ALE {
  // An IConstantSection is the simplest ISection
  //   All fibers are dimension 1
  //   All values are equal to a constant
  //     We need no value storage and no communication for completion
  //     The default value is returned for any point not in the domain
  template<typename Point_, typename Value_, typename Alloc_ = malloc_allocator<Point_> >
  class IConstantSection : public ALE::ParallelObject {
  public:
    typedef Point_ point_type;
    typedef Value_ value_type;
    typedef Alloc_ alloc_type;
    typedef Interval<point_type, alloc_type> chart_type;
    typedef point_type                       index_type;
  protected:
    chart_type _chart;
    value_type _value[2]; // Value and default value
  public:
    IConstantSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
      _value[1] = 0;
    };
    IConstantSection(MPI_Comm comm, const point_type& min, const point_type& max, const value_type& value, const int debug) : ParallelObject(comm, debug), _chart(min, max) {
      _value[0] = value;
      _value[1] = value;
    };
    IConstantSection(MPI_Comm comm, const point_type& min, const point_type& max, const value_type& value, const value_type& defaultValue, const int debug) : ParallelObject(comm, debug), _chart(min, max) {
      _value[0] = value;
      _value[1] = defaultValue;
    };
  public: // Verifiers
    void checkPoint(const point_type& point) const {
      this->_chart.checkPoint(point);
    };
    void checkDimension(const int& dim) {
      if (dim != 1) {
        ostringstream msg;
        msg << "Invalid fiber dimension " << dim << " must be 1" << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
    bool hasPoint(const point_type& point) const {
      return this->_chart.hasPoint(point);
    };
  public: // Accessors
    const chart_type& getChart() const {return this->_chart;};
    void setChart(const chart_type& chart) {this->_chart = chart;};
    void addPoint(const point_type& point) {
      this->checkPoint(point);
    };
    template<typename Points>
    void addPoint(const Obj<Points>& points) {
      for(typename Points::const_iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->checkPoint(*p_iter);
      }
    }
    template<typename Points>
    void addPoint(const Points& points) {
      for(typename Points::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        this->checkPoint(*p_iter);
      }
    }
    value_type getDefaultValue() {return this->_value[1];};
    void setDefaultValue(const value_type value) {this->_value[1] = value;};
    void copy(const Obj<IConstantSection>& section) {
      const chart_type& chart = section->getChart();

      this->_chart = chart;
      this->_value[0] = section->restrictPoint(*chart.begin())[0];
      this->_value[1] = section->restrictPoint(*chart.begin())[1];
    }
  public: // Sizes
    ///void clear() {};
    int getFiberDimension(const point_type& p) const {
      if (this->hasPoint(p)) return 1;
      return 0;
    }
    void setFiberDimension(const point_type& p, int dim) {
      this->checkDimension(dim);
      this->addPoint(p);
    }
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    }
    void addFiberDimension(const point_type& p, int dim) {
      if (this->hasPoint(p)) {
        ostringstream msg;
        msg << "Invalid addition to fiber dimension " << dim << " cannot exceed 1" << std::endl;
        throw ALE::Exception(msg.str().c_str());
      } else {
        this->setFiberDimension(p, dim);
      }
    }
    int size(const point_type& p) {return this->getFiberDimension(p);}
  public: // Restriction
    void clear() {};
    const value_type *restrictSpace() const {
      return this->_value;
    };
    const value_type *restrictPoint(const point_type& p) const {
      if (this->hasPoint(p)) {
        return this->_value;
      }
      return &this->_value[1];
    };
    void updatePoint(const point_type&, const value_type v[]) {
      this->_value[0] = v[0];
    };
    void updateAddPoint(const point_type&, const value_type v[]) {
      this->_value[0] += v[0];
    };
  public:
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
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
          txt << "viewing an IConstantSection" << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing IConstantSection '" << name << "'" << std::endl;
        }
      }
      txt <<"["<<this->commRank()<<"]: chart " << this->_chart << std::endl;
      txt <<"["<<this->commRank()<<"]: Value " << this->_value[0] << " Default Value " << this->_value[1] << std::endl;
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };

  // An IUniformSection often acts as an Atlas
  //   All fibers are the same dimension
  //     Note we can use a IConstantSection for this Atlas
  //   Each point may have a different vector
  //     Thus we need storage for values, and hence must implement completion
  template<typename Point_, typename Value_, int fiberDim = 1, typename Alloc_ = malloc_allocator<Value_> >
  class IUniformSection : public ALE::ParallelObject {
  public:
    typedef Point_                                                  point_type;
    typedef Value_                                                  value_type;
    typedef Alloc_                                                  alloc_type;
    typedef typename alloc_type::template rebind<point_type>::other point_alloc_type;
    typedef IConstantSection<point_type, int, point_alloc_type>     atlas_type;
    typedef typename atlas_type::chart_type                         chart_type;
    typedef point_type                                              index_type;
    typedef struct {value_type v[fiberDim];}                        fiber_type;
    typedef value_type*                                             values_type;
    typedef typename alloc_type::template rebind<atlas_type>::other atlas_alloc_type;
    typedef typename atlas_alloc_type::pointer                      atlas_ptr;
  protected:
    Obj<atlas_type> _atlas;
    values_type     _array;
    fiber_type      _emptyValue;
    alloc_type      _allocator;
  public:
    IUniformSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
      atlas_ptr pAtlas = atlas_alloc_type(this->_allocator).allocate(1);
      atlas_alloc_type(this->_allocator).construct(pAtlas, atlas_type(comm, debug));
      this->_atlas = Obj<atlas_type>(pAtlas, sizeof(atlas_type));
      this->_array = NULL;
      for(int i = 0; i < fiberDim; ++i) this->_emptyValue.v[i] = value_type();
    };
    IUniformSection(MPI_Comm comm, const point_type& min, const point_type& max, const int debug = 0) : ParallelObject(comm, debug) {
      atlas_ptr pAtlas = atlas_alloc_type(this->_allocator).allocate(1);
      atlas_alloc_type(this->_allocator).construct(pAtlas, atlas_type(comm, min, max, fiberDim, debug));
      this->_atlas = Obj<atlas_type>(pAtlas, sizeof(atlas_type));
      this->_array = NULL;
      for(int i = 0; i < fiberDim; ++i) this->_emptyValue.v[i] = value_type();
    };
    IUniformSection(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas) {
      int dim = fiberDim;
      this->_atlas->update(*this->_atlas->getChart().begin(), &dim);
      this->_array = NULL;
      for(int i = 0; i < fiberDim; ++i) this->_emptyValue.v[i] = value_type();
    };
    virtual ~IUniformSection() {
      if (this->_array) {
        for(int i = this->getChart().min()*fiberDim; i < this->getChart().max()*fiberDim; ++i) {this->_allocator.destroy(this->_array+i);}
        this->_array += this->getChart().min()*fiberDim;
        this->_allocator.deallocate(this->_array, this->sizeWithBC());
        this->_array = NULL;
        this->_atlas = NULL;
      }
    };
  public:
    value_type *getRawArray(const int size) {
      static value_type *array   = NULL;
      static int         maxSize = 0;

      if (size > maxSize) {
        const value_type dummy(0);

        if (array) {
          for(int i = 0; i < maxSize; ++i) {this->_allocator.destroy(array+i);}
          this->_allocator.deallocate(array, maxSize);
        }
        maxSize = size;
        array   = this->_allocator.allocate(maxSize);
        for(int i = 0; i < maxSize; ++i) {this->_allocator.construct(array+i, dummy);}
      };
      return array;
    };
  public: // Verifiers
    bool hasPoint(const point_type& point) const {
      return this->_atlas->hasPoint(point);
    };
    void checkDimension(const int& dim) {
      if (dim != fiberDim) {
        ostringstream msg;
        msg << "Invalid fiber dimension " << dim << " must be " << fiberDim << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
  public: // Accessors
    const chart_type& getChart() const {return this->_atlas->getChart();};
    void setChart(const chart_type& chart) {
      this->_atlas->setChart(chart);
      int dim = fiberDim;
      this->_atlas->updatePoint(*this->getChart().begin(), &dim);
    };
    bool resizeChart(const chart_type& chart) {
      if ((chart.min() >= this->getChart().min()) && (chart.max() <= this->getChart().max())) return false;
      this->setChart(chart);
      return true;
    };
    const Obj<atlas_type>& getAtlas() const {return this->_atlas;};
    void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;};
    void addPoint(const point_type& point) {
      this->setFiberDimension(point, fiberDim);
    };
    template<typename Points>
    void addPoint(const Obj<Points>& points) {
      for(typename Points::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, fiberDim);
      }
    }
    void copy(const Obj<IUniformSection>& section) {
      this->getAtlas()->copy(section->getAtlas());
      const chart_type& chart = section->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        this->updatePoint(*c_iter, section->restrictPoint(*c_iter));
      }
    }
    const value_type *getDefault() const {return this->_emptyValue.v;}
    void setDefault(const value_type v[]) {for(int i = 0; i < fiberDim; ++i) {this->_emptyValue.v[i] = v[i];}}
  public: // Sizes
    void clear() {
      this->zero();
      this->_atlas->clear();
    }
    int getFiberDimension(const point_type& p) const {
      return this->_atlas->restrictPoint(p)[0];
    }
    void setFiberDimension(const point_type& p, int dim) {
      this->checkDimension(dim);
      this->_atlas->addPoint(p);
      this->_atlas->updatePoint(p, &dim);
    }
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    }
    void setFiberDimension(const std::set<point_type>& points, int dim) {
      for(typename std::set<point_type>::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    };
    void addFiberDimension(const point_type& p, int dim) {
      if (this->hasPoint(p)) {
        ostringstream msg;
        msg << "Invalid addition to fiber dimension " << dim << " cannot exceed " << fiberDim << std::endl;
        throw ALE::Exception(msg.str().c_str());
      } else {
        this->setFiberDimension(p, dim);
      }
    };
    int size() const {return this->_atlas->getChart().size()*fiberDim;};
    int sizeWithBC() const {return this->size();};
    void allocatePoint() {
      this->_array = this->_allocator.allocate(this->sizeWithBC());
      this->_array -= this->getChart().min()*fiberDim;
      for(index_type i = this->getChart().min()*fiberDim; i < this->getChart().max()*fiberDim; ++i) {this->_allocator.construct(this->_array+i, this->_emptyValue.v[0]);}
    };
    bool reallocatePoint(const chart_type& chart, values_type *oldData = NULL) {
      const chart_type  oldChart = this->getChart();
      const int         oldSize  = this->sizeWithBC();
      values_type       oldArray = this->_array;
      if (!this->resizeChart(chart)) return false;
      const int         size     = this->sizeWithBC();

      this->_array = this->_allocator.allocate(size);
      this->_array -= this->getChart().min()*fiberDim;
      for(index_type i = this->getChart().min()*fiberDim; i < this->getChart().max()*fiberDim; ++i) {this->_allocator.construct(this->_array+i, this->_emptyValue.v[0]);}
      for(int i = oldChart.min()*fiberDim; i < oldChart.max()*fiberDim; ++i) {
        this->_array[i] = oldArray[i];
      }
      if (!oldData) {
        for(index_type i = oldChart.min()*fiberDim; i < oldChart.max()*fiberDim; ++i) {this->_allocator.destroy(oldArray+i);}
        oldArray += this->getChart().min()*fiberDim;
        this->_allocator.deallocate(oldArray, oldSize);
        ///std::cout << "Freed IUniformSection data" << std::endl;
      } else {
        ///std::cout << "Did not free IUniformSection data" << std::endl;
        *oldData = oldArray;
      }
      return true;
    };
    template<typename Iterator, typename Extractor>
    bool reallocatePoint(const Iterator& begin, const Iterator& end, const Extractor& extractor) {
      point_type min = this->getChart().min();
      point_type max = this->getChart().max()-1;

      for(Iterator p_iter = begin; p_iter != end; ++p_iter) {
        min = std::min(extractor(*p_iter), min);
        max = std::max(extractor(*p_iter), max);
      }
      return reallocatePoint(chart_type(min, max+1));
    }
  public: // Restriction
    // Zero entries
    void zero() {
      memset(this->_array+(this->getChart().min()*fiberDim), 0, this->sizeWithBC()* sizeof(value_type));
    };
    // Return a pointer to the entire contiguous storage array
    const values_type& restrictSpace() const {
      return this->_array;
    };
    // Return only the values associated to this point, not its closure
    const value_type *restrictPoint(const point_type& p) const {
      if (!this->hasPoint(p)) return this->_emptyValue.v;
      return &this->_array[p*fiberDim];
    };
    // Update only the values associated to this point, not its closure
    void updatePoint(const point_type& p, const value_type v[]) {
      for(int i = 0, idx = p*fiberDim; i < fiberDim; ++i, ++idx) {
        this->_array[idx] = v[i];
      }
    };
    // Update only the values associated to this point, not its closure
    void updateAddPoint(const point_type& p, const value_type v[]) {
      for(int i = 0, idx = p*fiberDim; i < fiberDim; ++i, ++idx) {
        this->_array[idx] += v[i];
      }
    };
    void updatePointAll(const point_type& p, const value_type v[]) {
      this->updatePoint(p, v);
    };
  public:
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
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
          txt << "viewing an IUniformSection" << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing IUniformSection '" << name << "'" << std::endl;
        }
      }
      const typename atlas_type::chart_type& chart = this->_atlas->getChart();
      values_type                            array = this->_array;

      for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const int idx = (*p_iter)*fiberDim;

        if (fiberDim != 0) {
          txt << "[" << this->commRank() << "]:   " << *p_iter << " dim " << fiberDim << "  ";
          for(int i = 0; i < fiberDim; i++) {
            txt << " " << array[idx+i];
          }
          txt << std::endl;
        }
      }
      if (chart.size() == 0) {
        txt << "[" << this->commRank() << "]: empty" << std::endl;
      }
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };
  // An ISection allows variable fiber sizes per point
  //   The Atlas is a UniformSection of dimension 1 and value type Point
  //     to hold each fiber dimension and offsets into a contiguous array
  template<typename Point_, typename Value_, typename Alloc_ = malloc_allocator<Value_> >
  class ISection : public Section<Point_, Value_, Alloc_, IUniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other> > {
  public:
    typedef Section<Point_, Value_, Alloc_, IUniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other> > base;
    typedef typename base::point_type       point_type;
    typedef typename base::value_type       value_type;
    typedef typename base::alloc_type       alloc_type;
    typedef typename base::index_type       index_type;
    typedef typename base::atlas_type       atlas_type;
    typedef typename base::chart_type       chart_type;
    typedef typename base::values_type      values_type;
    typedef typename base::atlas_alloc_type atlas_alloc_type;
    typedef typename base::atlas_ptr        atlas_ptr;
  public:
    ISection(MPI_Comm comm, const int debug = 0) : Section<Point_, Value_, Alloc_, IUniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other> >(comm, debug) {
    };
    ISection(MPI_Comm comm, const point_type& min, const point_type& max, const int debug = 0) : Section<Point_, Value_, Alloc_, IUniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other> >(comm, debug) {
      this->_atlas->setChart(chart_type(min, max));
      this->_atlas->allocatePoint();
    };
    ISection(const Obj<atlas_type>& atlas) : Section<Point_, Value_, Alloc_, IUniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other> >(atlas) {};
    virtual ~ISection() {};
  public:
    void setChart(const chart_type& chart) {
      this->_atlas->setChart(chart);
      this->_atlas->allocatePoint();
    };
    bool resizeChart(const chart_type& chart) {
      if (!this->_atlas->reallocatePoint(chart)) return false;
      return true;
    };
    bool reallocatePoint(const chart_type& chart) {
      typedef typename atlas_type::alloc_type atlas_alloc_type;
      const chart_type        oldChart = this->getChart();
      const int               oldSize  = this->sizeWithBC();
      const values_type       oldArray = this->_array;
      const int               oldAtlasSize = this->_atlas->sizeWithBC();
      typename atlas_type::values_type oldAtlasArray;
      if (!this->_atlas->reallocatePoint(chart, &oldAtlasArray)) return false;

      this->orderPoints(this->_atlas);
      this->allocateStorage();
      for(int i = oldChart.min(); i < oldChart.max(); ++i) {
        const typename atlas_type::value_type& idx = this->_atlas->restrictPoint(i)[0];
        const int                              dim = idx.prefix;
        const int                              off = idx.index;

        for(int d = 0; d < dim; ++d) {
          this->_array[off+d] = oldArray[oldAtlasArray[i].index+d];
        }
      }
      for(int i = 0; i < oldSize; ++i) {this->_allocator.destroy(oldArray+i);}
      this->_allocator.deallocate(oldArray, oldSize);
      for(int i = oldChart.min(); i < oldChart.max(); ++i) {atlas_alloc_type(this->_allocator).destroy(oldAtlasArray+i);}
      oldAtlasArray += oldChart.min();
      atlas_alloc_type(this->_allocator).deallocate(oldAtlasArray, oldAtlasSize);
      ///std::cout << "In ISection, Freed IUniformSection data" << std::endl;
      return true;
    };
  public:
    // Return the free values on a point
    //   This is overridden, because the one in Section cannot be const due to problem in the interface with UniformSection
    const value_type *restrictPoint(const point_type& p) const {
      return &(this->_array[this->_atlas->restrictPoint(p)[0].index]);
    };
  };
  // IGeneralSection will support BC on a subset of unknowns on a point
  //   We use a separate constraint Atlas to mark constrained dofs on a point
  template<typename Point_, typename Value_, typename Alloc_ = malloc_allocator<Value_> >
  class IGeneralSection : public GeneralSection<Point_, Value_, Alloc_, IUniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other>, ISection<Point_, int, typename Alloc_::template rebind<int>::other> > {
  public:
    typedef GeneralSection<Point_, Value_, Alloc_, IUniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other>, ISection<Point_, int, typename Alloc_::template rebind<int>::other> > base;
    typedef typename base::point_type       point_type;
    typedef typename base::value_type       value_type;
    typedef typename base::alloc_type       alloc_type;
    typedef typename base::index_type       index_type;
    typedef typename base::atlas_type       atlas_type;
    typedef typename base::bc_type          bc_type;
    typedef typename base::chart_type       chart_type;
    typedef typename base::values_type      values_type;
    typedef typename base::atlas_alloc_type atlas_alloc_type;
    typedef typename base::atlas_ptr        atlas_ptr;
    typedef typename base::bc_alloc_type    bc_alloc_type;
    typedef typename base::bc_ptr           bc_ptr;
    typedef std::pair<point_type,int>       newpoint_type;
  protected:
    std::set<newpoint_type> newPoints;
  public:
    IGeneralSection(MPI_Comm comm, const int debug = 0) : GeneralSection<Point_, Value_, Alloc_, IUniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other>, ISection<Point_, int, typename Alloc_::template rebind<int>::other> >(comm, debug) {};
    IGeneralSection(MPI_Comm comm, const point_type& min, const point_type& max, const int debug = 0) : GeneralSection<Point_, Value_, Alloc_, IUniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other>, ISection<Point_, int, typename Alloc_::template rebind<int>::other> >(comm, debug) {
      this->_atlas->setChart(chart_type(min, max));
      this->_atlas->allocatePoint();
      this->_bc->setChart(chart_type(min, max));
    };
    IGeneralSection(const Obj<atlas_type>& atlas) : GeneralSection<Point_, Value_, Alloc_, IUniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other>, ISection<Point_, int, typename Alloc_::template rebind<int>::other> >(atlas) {
      this->_bc->setChart(atlas->getChart());
    };
    ~IGeneralSection() {};
  public:
    void setChart(const chart_type& chart) {
      this->_atlas->setChart(chart);
      this->_atlas->allocatePoint();
      this->_bc->setChart(chart);
      ///this->_bc->getAtlas()->allocatePoint();
      for(int s = 0; s < (int) this->_spaces.size(); ++s) {
        this->_spaces[s]->setChart(chart);
        this->_spaces[s]->allocatePoint();
        this->_bcs[s]->setChart(chart);
        ///this->_bcs[s]->getAtlas()->allocatePoint();
      }
    };
  public:
    bool hasNewPoints() {return this->newPoints.size() > 0;};
    const std::set<newpoint_type>& getNewPoints() {return this->newPoints;};
    void addPoint(const point_type& point, const int dim) {
      if (dim == 0) return;
      this->newPoints.insert(newpoint_type(point, dim));
    };
    // Returns true if the chart was changed
    bool resizeChart(const chart_type& chart) {
      if (!this->_atlas->reallocatePoint(chart)) return false;
      this->_bc->reallocatePoint(chart);
      for(int s = 0; s < (int) this->_spaces.size(); ++s) {
        this->_spaces[s]->reallocatePoint(chart);
        this->_bcs[s]->reallocatePoint(chart);
      }
      return true;
    };
    // Returns true if the chart was changed
    bool reallocatePoint(const chart_type& chart) {
      typedef typename alloc_type::template rebind<typename atlas_type::value_type>::other atlas_alloc_type;
      const chart_type        oldChart = this->getChart();
      const int               oldSize  = this->sizeWithBC();
      const values_type       oldArray = this->_array;
      const int               oldAtlasSize = this->_atlas->sizeWithBC();
      typename atlas_type::values_type oldAtlasArray;
      if (!this->_atlas->reallocatePoint(chart, &oldAtlasArray)) return false;
      this->_bc->reallocatePoint(chart);
      for(int s = 0; s < (int) this->_spaces.size(); ++s) {
        this->_spaces[s]->reallocatePoint(chart);
        this->_bcs[s]->reallocatePoint(chart);
      }
      for(typename std::set<newpoint_type>::const_iterator p_iter = this->newPoints.begin(); p_iter != this->newPoints.end(); ++p_iter) {
        this->setFiberDimension(p_iter->first, p_iter->second);
      }
      this->orderPoints(this->_atlas);
      this->allocateStorage();
      for(int i = oldChart.min(); i < oldChart.max(); ++i) {
        const typename atlas_type::value_type& idx = this->_atlas->restrictPoint(i)[0];
        const int                              dim = idx.prefix;
        const int                              off = idx.index;

        for(int d = 0; d < dim; ++d) {
          this->_array[off+d] = oldArray[oldAtlasArray[i].index+d];
        }
      }
      for(int i = 0; i < oldSize; ++i) {this->_allocator.destroy(oldArray+i);}
      this->_allocator.deallocate(oldArray, oldSize);
      for(int i = oldChart.min(); i < oldChart.max(); ++i) {atlas_alloc_type(this->_allocator).destroy(oldAtlasArray+i);}
      oldAtlasArray += oldChart.min();
      atlas_alloc_type(this->_allocator).deallocate(oldAtlasArray, oldAtlasSize);
      this->newPoints.clear();
      return true;
    };
  public:
    void addSpace() {
      Obj<atlas_type> space = new atlas_type(this->comm(), this->debug());
      Obj<bc_type>    bc    = new bc_type(this->comm(), this->debug());
      space->setChart(this->_atlas->getChart());
      space->allocatePoint();
      bc->setChart(this->_bc->getChart());
      this->_spaces.push_back(space);
      this->_bcs.push_back(bc);
    };
    Obj<IGeneralSection> getFibration(const int space) const {
      Obj<IGeneralSection> field = new IGeneralSection(this->comm(), this->debug());
//     Obj<atlas_type> _atlas;
//     std::vector<Obj<atlas_type> > _spaces;
//     Obj<bc_type>    _bc;
//     std::vector<Obj<bc_type> >    _bcs;
      field->setChart(this->getChart());
      field->addSpace();
      const chart_type& chart = this->getChart();

      // Copy sizes
      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        const int fDim = this->getFiberDimension(*c_iter, space);
        const int cDim = this->getConstraintDimension(*c_iter, space);

        if (fDim) {
          field->setFiberDimension(*c_iter, fDim);
          field->setFiberDimension(*c_iter, fDim, 0);
        }
        if (cDim) {
          field->setConstraintDimension(*c_iter, cDim);
          field->setConstraintDimension(*c_iter, cDim, 0);
        }
      }
      field->allocateStorage();
      Obj<atlas_type>   newAtlas = new atlas_type(this->comm(), this->debug());
      const chart_type& newChart = field->getChart();

      for(typename chart_type::const_iterator c_iter = newChart.begin(); c_iter != newChart.end(); ++c_iter) {
        const int cDim   = field->getConstraintDimension(*c_iter);
        const int dof[1] = {0};

        if (cDim) {
          field->setConstraintDof(*c_iter, dof);
        }
      }
      // Copy offsets
      newAtlas->setChart(newChart);
      newAtlas->allocatePoint();
      for(typename chart_type::const_iterator c_iter = newChart.begin(); c_iter != newChart.end(); ++c_iter) {
        index_type idx;

        idx.prefix = field->getFiberDimension(*c_iter);
        idx.index  = this->_atlas->restrictPoint(*c_iter)[0].index;
        for(int s = 0; s < space; ++s) {
          idx.index += this->getFiberDimension(*c_iter, s);
        }
        newAtlas->addPoint(*c_iter);
        newAtlas->updatePoint(*c_iter, &idx);
      }
      field->replaceStorage(this->_array, true, this->getStorageSize());
      field->setAtlas(newAtlas);
      return field;
    };
  };

  class SectionSerializer {
  public:
    template<typename Point_, typename Value_>
    static void writeSection(std::ofstream& fs, IConstantSection<Point_, Value_>& section) {
      if (section.commRank() == 0) {
        // Write local
        fs << section.getChart().min() << " " << section.getChart().max() << std::endl;
        fs.precision(15);
        fs << section.restrictPoint(section.getChart().min())[0] << " ";
        fs << section.getDefaultValue() << std::endl;
        // Receive and write remote
        for(int p = 1; p < section.commSize(); ++p) {
          PetscInt       sizes[2];
          PetscScalar    values[2];
          MPI_Status     status;
          PetscErrorCode ierr;

          ierr = MPI_Recv(sizes,  2, MPIU_INT,    p, 1, section.comm(), &status);CHKERRXX(ierr);
          ierr = MPI_Recv(values, 2, MPIU_SCALAR, p, 1, section.comm(), &status);CHKERRXX(ierr);
          fs << sizes[0] << " " << sizes[1] << std::endl;
          fs.precision(15);
          fs << values[0] << " " << values[1] << std::endl;
        }
      } else {
        // Send remote
        PetscInt       sizes[2];
        PetscScalar    values[2];
        PetscErrorCode ierr;

        sizes[0]  = section.getChart().min();
        sizes[1]  = section.getChart().max();
        values[0] = section.restrictPoint(section.getChart().min())[0];
        values[1] = section.getDefaultValue();
        ierr = MPI_Send(sizes,  2, MPIU_INT,    0, 1, section.comm());CHKERRXX(ierr);
        ierr = MPI_Send(values, 2, MPIU_SCALAR, 0, 1, section.comm());CHKERRXX(ierr);
      }
    };
    template<typename Point_, typename Value_, int fiberDim>
    static void writeSection(std::ofstream& fs, IUniformSection<Point_, Value_, fiberDim>& section) {
      typedef typename IUniformSection<Point_, Value_, fiberDim>::index_type index_type;
      typedef typename IUniformSection<Point_, Value_, fiberDim>::value_type value_type;
      index_type min = section.getChart().min();
      index_type max = section.getChart().max();

      // Write atlas
      writeSection(fs, *section.getAtlas());
      if (section.commRank() == 0) {
        // Write local values
        fs.precision(15);
        for(index_type p = min; p < max; ++p) {
          const value_type *values = section.restrictPoint(p);

          for(int i = 0; i < fiberDim; ++i) {
            fs << values[i] << std::endl;
          }
        }
        // Write empty value
        const value_type *defValue = section.getDefault();

        for(int i = 0; i < fiberDim; ++i) {
          if (i > 0) fs << " ";
          fs << defValue[i];
        }
        fs << std::endl;
        // Receive and write remote
        for(int p = 1; p < section.commSize(); ++p) {
          PetscInt       size;
          PetscScalar   *values;
          PetscScalar    emptyValues[fiberDim];
          MPI_Status     status;
          PetscErrorCode ierr;

          ierr = MPI_Recv(&size, 1, MPIU_INT, p, 1, section.comm(), &status);CHKERRXX(ierr);
          ierr = PetscMalloc(size*fiberDim * sizeof(PetscScalar), &values);CHKERRXX(ierr);
          ierr = MPI_Recv(values, size*fiberDim, MPIU_SCALAR, p, 1, section.comm(), &status);CHKERRXX(ierr);
          for(PetscInt v = 0; v < size; ++v) {
            fs << values[v] << std::endl;
          }
          ierr = PetscFree(values);CHKERRXX(ierr);
          ierr = MPI_Recv(emptyValues, fiberDim, MPIU_SCALAR, p, 1, section.comm(), &status);CHKERRXX(ierr);
          for(int i = 0; i < fiberDim; ++i) {
            if (i > 0) fs << " ";
            fs << emptyValues[i];
          }
          fs << std::endl;
        }
      } else {
        // Send remote
        PetscInt          size = section.getChart().size();
        PetscInt          v    = 0;
        const value_type *defValue = section.getDefault();
        PetscScalar      *values;
        PetscScalar       emptyValues[fiberDim];
        PetscErrorCode    ierr;

        assert(sizeof(value_type) <= sizeof(PetscScalar));
        ierr = MPI_Send(&size, 1, MPIU_INT, 0, 1, section.comm());CHKERRXX(ierr);
        ierr = PetscMalloc(size*fiberDim * sizeof(PetscScalar), &values);CHKERRXX(ierr);
        for(index_type p = min; p < max; ++p) {
          const value_type *val = section.restrictPoint(p);

          for(int i = 0; i < fiberDim; ++i, ++v) {
            values[v] = ((PetscScalar *) &val[i])[0];
          }
        }
        ierr = MPI_Send(values, size*fiberDim, MPIU_SCALAR, 0, 1, section.comm());CHKERRXX(ierr);
        for(int i = 0; i < fiberDim; ++i) {emptyValues[i] = ((PetscScalar *) &defValue[i])[0];}
        ierr = MPI_Send(emptyValues, fiberDim, MPIU_SCALAR, 0, 1, section.comm());CHKERRXX(ierr);
      }
    };
    template<typename Point_, typename Value_>
    static void writeSection(std::ofstream& fs, ISection<Point_, Value_>& section) {
      typedef typename ISection<Point_, Value_>::point_type point_type;
      typedef typename ISection<Point_, Value_>::value_type value_type;
      point_type min = section.getChart().min();
      point_type max = section.getChart().max();

      // Write atlas
      writeSection(fs, *section.getAtlas());
      if (section.commRank() == 0) {
      // Write local values
        fs.precision(15);
        for(point_type p = min; p < max; ++p) {
          const int         fiberDim = section.getFiberDimension(p);
          const value_type *values   = section.restrictPoint(p);

          for(int i = 0; i < fiberDim; ++i) {
            fs << values[i] << std::endl;
          }
        }
        // Receive and write remote
        for(int p = 1; p < section.commSize(); ++p) {
          PetscInt       size;
          PetscScalar   *values;
          MPI_Status     status;
          PetscErrorCode ierr;

          ierr = MPI_Recv(&size, 1, MPIU_INT, p, 1, section.comm(), &status);CHKERRXX(ierr);
          ierr = PetscMalloc(size * sizeof(PetscScalar), &values);CHKERRXX(ierr);
          ierr = MPI_Recv(values, size, MPIU_SCALAR, p, 1, section.comm(), &status);CHKERRXX(ierr);
          for(PetscInt v = 0; v < size; ++v) {
            fs << values[v] << std::endl;
          }
          ierr = PetscFree(values);CHKERRXX(ierr);
        }
      } else {
        // Send remote
        PetscInt       size = section.size();
        PetscInt       v    = 0;
        PetscScalar   *values;
        PetscErrorCode ierr;

        ierr = MPI_Send(&size, 1, MPIU_INT, 0, 1, section.comm());CHKERRXX(ierr);
        ierr = PetscMalloc(size * sizeof(PetscScalar), &values);CHKERRXX(ierr);
        for(point_type p = min; p < max; ++p) {
          const int         fiberDim = section.getFiberDimension(p);
          const value_type *val      = section.restrictPoint(p);

          for(int i = 0; i < fiberDim; ++i, ++v) {
            values[v] = val[i];
          }
        }
        ierr = MPI_Send(values, size, MPIU_SCALAR, 0, 1, section.comm());CHKERRXX(ierr);
      }
    };
    template<typename Point_, typename Value_>
    static void writeSection(std::ofstream& fs, IGeneralSection<Point_, Value_>& section) {
      typedef typename IGeneralSection<Point_, Value_>::point_type point_type;
      typedef typename IGeneralSection<Point_, Value_>::value_type value_type;
      point_type min = section.getChart().min();
      point_type max = section.getChart().max();

      // Write atlas
      writeSection(fs, *section.getAtlas());
      if (section.commRank() == 0) {
        // Write local values
        fs.precision(15);
        for(point_type p = min; p < max; ++p) {
          const int         fiberDim = section.getFiberDimension(p);
          const value_type *values   = section.restrictPoint(p);

          for(int i = 0; i < fiberDim; ++i) {
            fs << values[i] << std::endl;
          }
        }
        // Receive and write remote
        for(int p = 1; p < section.commSize(); ++p) {
          PetscInt       size;
          PetscScalar   *values;
          MPI_Status     status;
          PetscErrorCode ierr;

          ierr = MPI_Recv(&size, 1, MPIU_INT, p, 1, section.comm(), &status);CHKERRXX(ierr);
          ierr = PetscMalloc(size * sizeof(PetscScalar), &values);CHKERRXX(ierr);
          ierr = MPI_Recv(values, size, MPIU_SCALAR, p, 1, section.comm(), &status);CHKERRXX(ierr);
          for(PetscInt v = 0; v < size; ++v) {
            fs << values[v] << std::endl;
          }
          ierr = PetscFree(values);CHKERRXX(ierr);
        }
      } else {
        // Send remote
        PetscInt       size = section.sizeWithBC();
        PetscInt       v    = 0;
        PetscScalar   *values;
        PetscErrorCode ierr;

        ierr = MPI_Send(&size, 1, MPIU_INT, 0, 1, section.comm());CHKERRXX(ierr);
        ierr = PetscMalloc(size * sizeof(PetscScalar), &values);CHKERRXX(ierr);
        for(point_type p = min; p < max; ++p) {
          const int         fiberDim = section.getFiberDimension(p);
          const value_type *val      = section.restrictPoint(p);

          for(int i = 0; i < fiberDim; ++i, ++v) {
            values[v] = val[i];
          }
        }
        ierr = MPI_Send(values, size, MPIU_SCALAR, 0, 1, section.comm());CHKERRXX(ierr);
      }
      // Write BC
      writeSection(fs, *section.getBC());
      // Write spaces
      //   std::vector<Obj<atlas_type> > _spaces;
      //   std::vector<Obj<bc_type> >    _bcs;
    };
    template<typename Point_, typename Value_>
    static void loadSection(std::ifstream& fs, IConstantSection<Point_, Value_>& section) {
      typedef typename IConstantSection<Point_, Value_>::index_type index_type;
      typedef typename IConstantSection<Point_, Value_>::value_type value_type;
      index_type min, max;
      value_type val;

      if (section.commRank() == 0) {
        // Load local
        fs >> min;
        fs >> max;
        section.setChart(typename IConstantSection<Point_, Value_>::chart_type(min, max));
        fs >> val;
        section.updatePoint(min, &val);
        fs >> val;
        section.setDefaultValue(val);
        // Load and send remote
        for(int p = 1; p < section.commSize(); ++p) {
          PetscInt       sizes[2];
          PetscScalar    values[2];
          PetscErrorCode ierr;

          fs >> sizes[0];
          fs >> sizes[1];
          fs >> values[0];
          fs >> values[1];
          ierr = MPI_Send(sizes,  2, MPIU_INT,    p, 1, section.comm());CHKERRXX(ierr);
          ierr = MPI_Send(values, 2, MPIU_SCALAR, p, 1, section.comm());CHKERRXX(ierr);
        }
      } else {
        // Load remote
        PetscInt       sizes[2];
        PetscScalar    values[2];
        value_type     value;
        MPI_Status     status;
        PetscErrorCode ierr;

        assert(sizeof(value_type) <= sizeof(PetscScalar));
        ierr = MPI_Recv(sizes,  2, MPIU_INT,    0, 1, section.comm(), &status);CHKERRXX(ierr);
        section.setChart(typename IConstantSection<Point_, Value_>::chart_type(sizes[0], sizes[1]));
        ierr = MPI_Recv(values, 2, MPIU_SCALAR, 0, 1, section.comm(), &status);CHKERRXX(ierr);
        value = values[0];
        section.updatePoint(min, &value);
        section.setDefaultValue(values[1]);
      }
    };
    template<typename Point_, typename Value_, int fiberDim>
    static void loadSection(std::ifstream& fs, IUniformSection<Point_, Value_, fiberDim>& section) {
      typedef typename IUniformSection<Point_, Value_, fiberDim>::index_type index_type;
      typedef typename IUniformSection<Point_, Value_, fiberDim>::value_type value_type;
      // Load atlas
      loadSection(fs, *section.getAtlas());
      section.allocatePoint();
      index_type min = section.getChart().min();
      index_type max = section.getChart().max();

      if (section.commRank() == 0) {
        // Load local values
        for(index_type p = min; p < max; ++p) {
          value_type values[fiberDim];

          for(int i = 0; i < fiberDim; ++i) {
            typename IUniformSection<Point_, Value_, fiberDim>::value_type value;

            fs >> value;
            values[i] = value;
          }
          section.updatePoint(p, values);
        }
        // Load empty value
        value_type defValue[fiberDim];

        for(int i = 0; i < fiberDim; ++i) {
          fs >> defValue[i];
        }
        section.setDefault(defValue);
        // Load and send remote
        for(int pr = 1; pr < section.commSize(); ++pr) {
          PetscInt          size = section.getChart().size();
          PetscInt          v    = 0;
          PetscScalar      *values;
          PetscScalar       emptyValues[fiberDim];
          PetscErrorCode    ierr;

          ierr = MPI_Send(&size, 1, MPIU_INT, pr, 1, section.comm());CHKERRXX(ierr);
          ierr = PetscMalloc(size*fiberDim * sizeof(PetscScalar), &values);CHKERRXX(ierr);
          for(index_type p = min; p < max; ++p) {
            for(int i = 0; i < fiberDim; ++i, ++v) {
              fs >> values[v];
            }
          }
          ierr = MPI_Send(values, size*fiberDim, MPIU_SCALAR, pr, 1, section.comm());CHKERRXX(ierr);
          for(int i = 0; i < fiberDim; ++i) {
            fs >> emptyValues[i];
          }
          ierr = MPI_Send(emptyValues, fiberDim, MPIU_SCALAR, pr, 1, section.comm());CHKERRXX(ierr);
        }
      } else {
        // Load remote
        PetscInt       size;
        PetscScalar   *values;
        PetscScalar    emptyValues[fiberDim];
        value_type     pvalues[fiberDim];
        MPI_Status     status;
        PetscInt       v = 0;
        PetscErrorCode ierr;

        assert(sizeof(value_type) <= sizeof(PetscScalar));
        ierr = MPI_Recv(&size, 1, MPIU_INT, 0, 1, section.comm(), &status);CHKERRXX(ierr);
        ierr = PetscMalloc(size*fiberDim * sizeof(PetscScalar), &values);CHKERRXX(ierr);
        ierr = MPI_Recv(values, size*fiberDim, MPIU_SCALAR, 0, 1, section.comm(), &status);CHKERRXX(ierr);
        for(index_type p = min; p < max; ++p) {
          for(int i = 0; i < fiberDim; ++i, ++v) {
            pvalues[i] = ((value_type *) &values[v])[0];
          }
          section.updatePoint(p, pvalues);
        }
        ierr = PetscFree(values);CHKERRXX(ierr);
        ierr = MPI_Recv(emptyValues, fiberDim, MPIU_SCALAR, 0, 1, section.comm(), &status);CHKERRXX(ierr);
        for(int i = 0; i < fiberDim; ++i) {pvalues[i] = ((value_type *) &emptyValues[i])[0];}
        section.setDefault(pvalues);
      }
    };
    template<typename Point_, typename Value_>
    static void loadSection(std::ifstream& fs, ISection<Point_, Value_>& section) {
      typedef typename ISection<Point_, Value_>::point_type point_type;
      typedef typename ISection<Point_, Value_>::value_type value_type;
      // Load atlas
      loadSection(fs, *section.getAtlas());
      section.allocatePoint();
      point_type min    = section.getChart().min();
      point_type max    = section.getChart().max();
      int        maxDim = -1;

      if (section.commRank() == 0) {
        // Load local values
        for(point_type p = min; p < max; ++p) {
          maxDim = std::max(maxDim, section.getFiberDimension(p));
        }
        value_type *values = new value_type[maxDim];
        for(point_type p = min; p < max; ++p) {
          const int fiberDim = section.getFiberDimension(p);

          for(int i = 0; i < fiberDim; ++i) {
            fs >> values[i];
          }
          section.updatePoint(p, values);
        }
        delete [] values;
        // Load and send remote
        for(int p = 1; p < section.commSize(); ++p) {
          PetscInt       size = section.size();
          PetscScalar   *values;
          PetscErrorCode ierr;

          ierr = MPI_Send(&size, 1, MPIU_INT, p, 1, section.comm());CHKERRXX(ierr);
          ierr = PetscMalloc(size * sizeof(PetscScalar), &values);CHKERRXX(ierr);
          for(PetscInt v = 0; v < size; ++v) {
            fs >> values[v];
          }
          ierr = MPI_Send(values, size, MPIU_SCALAR, p, 1, section.comm());CHKERRXX(ierr);
        }
      } else {
        // Load remote
        PetscInt       size;
        PetscScalar   *values;
        MPI_Status     status;
        PetscInt       maxDim = 0;
        PetscInt       off    = 0;
        PetscErrorCode ierr;

        assert(sizeof(value_type) <= sizeof(PetscScalar));
        ierr = MPI_Recv(&size, 1, MPIU_INT, 0, 1, section.comm(), &status);CHKERRXX(ierr);
        ierr = PetscMalloc(size * sizeof(PetscScalar), &values);CHKERRXX(ierr);
        ierr = MPI_Recv(values, size, MPIU_SCALAR, 0, 1, section.comm(), &status);CHKERRXX(ierr);
        for(point_type p = min; p < max; ++p) {
          maxDim = std::max(maxDim, section.getFiberDimension(p));
        }
        value_type *pvalues = new value_type[maxDim];
        for(point_type p = min; p < max; ++p) {
          const int fiberDim = section.getFiberDimension(p);

          for(int i = 0; i < fiberDim; ++i, ++off) {
            pvalues[i] = values[off];
          }
          section.updatePoint(p, pvalues);
        }
        delete [] pvalues;
        ierr = PetscFree(values);CHKERRXX(ierr);
      }
    };
    template<typename Point_, typename Value_>
    static void loadSection(std::ifstream& fs, IGeneralSection<Point_, Value_>& section) {
      typedef typename IGeneralSection<Point_, Value_>::point_type point_type;
      typedef typename IGeneralSection<Point_, Value_>::value_type value_type;
      // Load atlas
      loadSection(fs, *section.getAtlas());
      section.allocatePoint();
      point_type min    = section.getChart().min();
      point_type max    = section.getChart().max();
      int        maxDim = -1;

      if (section.commRank() == 0) {
        // Load local values
        for(point_type p = min; p < max; ++p) {
          maxDim = std::max(maxDim, section.getFiberDimension(p));
        }
        value_type *values = new value_type[maxDim];
        for(point_type p = min; p < max; ++p) {
          const int fiberDim = section.getFiberDimension(p);

          for(int i = 0; i < fiberDim; ++i) {
            fs >> values[i];
          }
          section.updatePoint(p, values);
        }
        delete [] values;
        // Load and send remote
        for(int p = 1; p < section.commSize(); ++p) {
          PetscInt       size = section.sizeWithBC();
          PetscScalar   *values;
          PetscErrorCode ierr;

          ierr = MPI_Send(&size, 1, MPIU_INT, p, 1, section.comm());CHKERRXX(ierr);
          ierr = PetscMalloc(size * sizeof(PetscScalar), &values);CHKERRXX(ierr);
          for(PetscInt v = 0; v < size; ++v) {
            fs >> values[v];
          }
          ierr = MPI_Send(values, size, MPIU_SCALAR, p, 1, section.comm());CHKERRXX(ierr);
        }
      } else {
        // Load remote
        PetscInt       size;
        PetscScalar   *values;
        MPI_Status     status;
        PetscInt       maxDim = 0;
        PetscInt       off    = 0;
        PetscErrorCode ierr;

        assert(sizeof(value_type) <= sizeof(PetscScalar));
        ierr = MPI_Recv(&size, 1, MPIU_INT, 0, 1, section.comm(), &status);CHKERRXX(ierr);
        ierr = PetscMalloc(size * sizeof(PetscScalar), &values);CHKERRXX(ierr);
        ierr = MPI_Recv(values, size, MPIU_SCALAR, 0, 1, section.comm(), &status);CHKERRXX(ierr);
        for(point_type p = min; p < max; ++p) {
          maxDim = std::max(maxDim, section.getFiberDimension(p));
        }
        value_type *pvalues = new value_type[maxDim];
        for(point_type p = min; p < max; ++p) {
          const int fiberDim = section.getFiberDimension(p);

          for(int i = 0; i < fiberDim; ++i, ++off) {
            pvalues[i] = values[off];
          }
          section.updatePoint(p, pvalues);
        }
        delete [] pvalues;
        ierr = PetscFree(values);CHKERRXX(ierr);
      }
      // Load BC
      loadSection(fs, *section.getBC());
      // Load spaces
      //   std::vector<Obj<atlas_type> > _spaces;
      //   std::vector<Obj<bc_type> >    _bcs;
    };
  };
}

#endif
