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
    };
    template<typename Points>
    void addPoint(const Points& points) {
      for(typename Points::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        this->checkPoint(*p_iter);
      }
    };
    value_type getDefaultValue() {return this->_value[1];};
    void setDefaultValue(const value_type value) {this->_value[1] = value;};
    void copy(const Obj<IConstantSection>& section) {
      const chart_type& chart = section->getChart();

      this->_chart = chart;
      this->_value[0] = section->restrict(*chart.begin())[0];
      this->_value[1] = section->restrict(*chart.begin())[1];
    };
  public: // Sizes
    ///void clear() {};
    int getFiberDimension(const point_type& p) const {
      if (this->hasPoint(p)) return 1;
      return 0;
    };
    void setFiberDimension(const point_type& p, int dim) {
      this->checkDimension(dim);
      this->addPoint(p);
    };
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    };
    void addFiberDimension(const point_type& p, int dim) {
      if (this->hasPoint(p)) {
        ostringstream msg;
        msg << "Invalid addition to fiber dimension " << dim << " cannot exceed 1" << std::endl;
        throw ALE::Exception(msg.str().c_str());
      } else {
        this->setFiberDimension(p, dim);
      }
    };
    int size(const point_type& p) {return this->getFiberDimension(p);};
  public: // Restriction
    void clear() {};
    const value_type *restrict() const {
      return this->_value;
    };
    const value_type *restrict(const point_type& p) const {
      if (this->hasPoint(p)) {
        return this->_value;
      }
      return &this->_value[1];
    };
    const value_type *restrictPoint(const point_type& p) const {return this->restrict(p);};
    void update(const point_type& p, const value_type v[]) {
      this->_value[0] = v[0];
    };
    void updatePoint(const point_type& p, const value_type v[]) {return this->update(p, v);};
    void updateAdd(const point_type& p, const value_type v[]) {
      this->_value[0] += v[0];
    };
    void updateAddPoint(const point_type& p, const value_type v[]) {return this->updateAdd(p, v);};
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
      this->_atlas->update(*this->getChart().begin(), &dim);
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
    };
    void copy(const Obj<IUniformSection>& section) {
      this->getAtlas()->copy(section->getAtlas());
      const chart_type& chart = section->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        this->updatePoint(*c_iter, section->restrictPoint(*c_iter));
      }
    };
  public: // Sizes
    void clear() {
      this->_atlas->clear(); 
    };
    int getFiberDimension(const point_type& p) const {
      return this->_atlas->restrictPoint(p)[0];
    };
    void setFiberDimension(const point_type& p, int dim) {
      this->checkDimension(dim);
      this->_atlas->addPoint(p);
      this->_atlas->updatePoint(p, &dim);
    };
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    };
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
      const value_type dummy = 0;
      this->_array = this->_allocator.allocate(this->sizeWithBC());
      this->_array -= this->getChart().min()*fiberDim;
      for(index_type i = this->getChart().min()*fiberDim; i < this->getChart().max()*fiberDim; ++i) {this->_allocator.construct(this->_array+i, dummy);}
    };
    bool reallocatePoint(const chart_type& chart, values_type *oldData = NULL) {
      const chart_type  oldChart = this->getChart();
      const int         oldSize  = this->sizeWithBC();
      const values_type oldArray = this->_array;
      if (!this->resizeChart(chart)) return false;
      const int         size     = this->sizeWithBC();
      const value_type  dummy    = 0;

      this->_array = this->_allocator.allocate(size);
      this->_array -= this->getChart().min()*fiberDim;
      for(index_type i = this->getChart().min()*fiberDim; i < this->getChart().max()*fiberDim; ++i) {this->_allocator.construct(this->_array+i, dummy);}
      for(int i = oldChart.min()*fiberDim; i < oldChart.max()*fiberDim; ++i) {
        this->_array[i] = oldArray[i];
      }
      if (!oldData) {
        for(index_type i = oldChart.min()*fiberDim; i < oldChart.max()*fiberDim; ++i) {this->_allocator.destroy(oldArray+i);}
        this->_array += this->getChart().min()*fiberDim;
        this->_allocator.deallocate(oldArray, oldSize);
        std::cout << "Freed IUniformSection data" << std::endl;
      } else {
        std::cout << "Did not free IUniformSection data" << std::endl;
        *oldData = oldArray;
      }
      return true;
    };
  public: // Restriction
    // Return a pointer to the entire contiguous storage array
    const values_type& restrict() const {
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
      std::cout << "In ISection, Freed IUniformSection data" << std::endl;
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
      this->_bc->getAtlas()->allocatePoint();
      for(int s = 0; s < (int) this->_spaces.size(); ++s) {
        this->_spaces[s]->setChart(chart);
        this->_spaces[s]->allocatePoint();
        this->_bcs[s]->setChart(chart);
        this->_bcs[s]->getAtlas()->allocatePoint();
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
}

#endif
