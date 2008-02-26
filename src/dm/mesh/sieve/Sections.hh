#ifndef included_ALE_Sections_hh
#define included_ALE_Sections_hh

#ifndef  included_ALE_Numbering_hh
#include <Numbering.hh>
#endif

namespace ALE {
  template<typename Sieve_, typename Alloc_ = malloc_allocator<typename Sieve_::target_type> >
  class BaseSection : public ALE::ParallelObject {
  public:
    typedef Sieve_                                    sieve_type;
    typedef Alloc_                                    alloc_type;
    typedef int                                       value_type;
    typedef typename sieve_type::target_type          point_type;
    typedef typename sieve_type::traits::baseSequence chart_type;
  protected:
    Obj<sieve_type> _sieve;
    chart_type      _chart;
    int             _size;
    int             _empty;
  public:
    BaseSection(const Obj<sieve_type>& sieve) : ParallelObject(sieve->comm(), sieve->debug()), _sieve(sieve), _chart(*sieve->base()), _size(1), _empty(0) {};
    ~BaseSection() {};
  public: // Verifiers
    bool hasPoint(const point_type& point) const {
      return this->_sieve->baseContains(point);
    };
  public:
    const chart_type& getChart() const {
      return this->_chart;
    };
    const int getFiberDimension(const point_type& p) const {
      return this->hasPoint(p) ? 1 : 0;
    };
    const value_type *restrictPoint(const point_type& p) const {
      if (this->hasPoint(p)) return &this->_size;
      return &this->_empty;
    };
  };

  template<typename Sieve_, typename Alloc_ = malloc_allocator<int> >
  class ConeSizeSection : public ALE::ParallelObject {
  public:
    typedef Sieve_                              sieve_type;
    typedef Alloc_                              alloc_type;
    typedef int                                 value_type;
    typedef typename sieve_type::target_type    point_type;
    typedef BaseSection<sieve_type, alloc_type> atlas_type;
    typedef typename atlas_type::chart_type     chart_type;
    typedef typename alloc_type::template rebind<atlas_type>::other atlas_alloc_type;
    typedef typename atlas_alloc_type::pointer                      atlas_ptr;
  protected:
    Obj<sieve_type> _sieve;
    Obj<atlas_type> _atlas;
    int             _size;
  public:
    ConeSizeSection(const Obj<sieve_type>& sieve) : ParallelObject(sieve->comm(), sieve->debug()), _sieve(sieve) {
      atlas_ptr pAtlas = atlas_alloc_type().allocate(1);
      atlas_alloc_type().construct(pAtlas, atlas_type(sieve));
      this->_atlas     = Obj<atlas_type>(pAtlas, sizeof(atlas_type));
    };
    ~ConeSizeSection() {};
  public: // Verifiers
    bool hasPoint(const point_type& point) {
      return this->_atlas->hasPoint(point);
    };
  public: // Accessors
    const Obj<atlas_type>& getAtlas() {return this->_atlas;};
    void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;};
  public:
    const int getFiberDimension(const point_type& p) {
      return this->hasPoint(p) ? 1 : 0;
    };
    const value_type *restrictPoint(const point_type& p) {
      this->_size = this->_sieve->cone(p)->size();
      return &this->_size;
    };
  };

  template<typename Sieve_, typename Alloc_ = malloc_allocator<typename Sieve_::source_type> >
  class ConeSection : public ALE::ParallelObject {
  public:
    typedef Sieve_                                  sieve_type;
    typedef Alloc_                                  alloc_type;
    typedef typename sieve_type::target_type        point_type;
    typedef typename sieve_type::source_type        value_type;
    typedef ConeSizeSection<sieve_type, alloc_type> atlas_type;
    typedef typename atlas_type::chart_type         chart_type;
    typedef typename alloc_type::template rebind<atlas_type>::other atlas_alloc_type;
    typedef typename atlas_alloc_type::pointer                      atlas_ptr;
  protected:
    Obj<sieve_type> _sieve;
    Obj<atlas_type> _atlas;
    alloc_type      _allocator;
  public:
    ConeSection(const Obj<sieve_type>& sieve) : ParallelObject(sieve->comm(), sieve->debug()), _sieve(sieve) {
      atlas_ptr pAtlas = atlas_alloc_type(this->_allocator).allocate(1);
      atlas_alloc_type(this->_allocator).construct(pAtlas, atlas_type(sieve));
      this->_atlas     = Obj<atlas_type>(pAtlas, sizeof(atlas_type));
    };
    ~ConeSection() {};
  protected:
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
    bool hasPoint(const point_type& point) {
      return this->_atlas->hasPoint(point);
    };
  public: // Accessors
    const Obj<atlas_type>& getAtlas() {return this->_atlas;};
    void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;};
  public: // Sizes and storage
    int getFiberDimension(const point_type& p) {
      return this->_atlas->restrictPoint(p)[0];
    };
  public: // Restriction and update
    const value_type *restrictPoint(const point_type& p) {
      const Obj<typename sieve_type::traits::coneSequence>& cone = this->_sieve->cone(p);
      value_type *array = this->getRawArray(cone->size());
      int         c     = 0;

      for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
        array[c++] = *c_iter;
      }
      return array;
    };
  };

  namespace New {
    // This section takes an existing section, and reports instead the fiber dimensions as values
    template<typename Section_>
    class SizeSection : public ALE::ParallelObject {
    public:
      typedef Section_                          section_type;
      typedef typename section_type::point_type point_type;
      typedef int                               value_type;
    protected:
      Obj<section_type> _section;
      value_type        _size;
    public:
      SizeSection(const Obj<section_type>& section) : ParallelObject(MPI_COMM_SELF, section->debug()), _section(section) {};
      virtual ~SizeSection() {};
    public:
      bool hasPoint(const point_type& point) {
        return this->_section->hasPoint(point);
      };
      const value_type *restrict(const point_type& p) {
        this->_size = this->_section->getFiberDimension(p); // Could be size()
        return &this->_size;
      };
      const value_type *restrictPoint(const point_type& p) {
        this->_size = this->_section->getFiberDimension(p);
        return &this->_size;
      };
    public:
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        this->_section->view(name, comm);
      };
    };

    // This section reports as values the size of the partition associated with the partition point
    template<typename Bundle_, typename Marker_>
    class PartitionSizeSection : public ALE::ParallelObject {
    public:
      typedef Bundle_                          bundle_type;
      typedef typename bundle_type::sieve_type sieve_type;
      typedef typename bundle_type::point_type point_type;
      typedef Marker_                          marker_type;
      typedef int                              value_type;
      typedef std::map<marker_type, int>       sizes_type;
    protected:
      sizes_type _sizes;
      int        _height;
      void _init(const Obj<bundle_type>& bundle, const int numElements, const marker_type partition[]) {
        const Obj<typename bundle_type::label_sequence>& cells      = bundle->heightStratum(this->_height);
        const Obj<typename bundle_type::numbering_type>& cNumbering = bundle->getFactory()->getLocalNumbering(bundle, bundle->depth() - this->_height);
        std::map<marker_type, std::set<point_type> >     points;

        if (numElements != (int) cells->size()) {
          throw ALE::Exception("Partition size does not match the number of elements");
        }
        for(typename bundle_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          typedef ALE::SieveAlg<bundle_type> sieve_alg_type;
          const Obj<typename sieve_alg_type::coneArray>& closure = sieve_alg_type::closure(bundle, *e_iter);
          const int idx = cNumbering->getIndex(*e_iter);

          points[partition[idx]].insert(closure->begin(), closure->end());
          if (this->_height > 0) {
            const Obj<typename sieve_alg_type::supportArray>& star = sieve_alg_type::star(bundle, *e_iter);

            points[partition[idx]].insert(star->begin(), star->end());
          }
        }
        for(typename std::map<marker_type, std::set<point_type> >::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          this->_sizes[p_iter->first] = p_iter->second.size();
        }
      };
    public:
      PartitionSizeSection(const Obj<bundle_type>& bundle, const int elementHeight, const int numElements, const marker_type *partition) : ParallelObject(MPI_COMM_SELF, bundle->debug()), _height(elementHeight) {
        this->_init(bundle, numElements, partition);
      };
      virtual ~PartitionSizeSection() {};
    public:
      bool hasPoint(const point_type& point) {return true;};
      const value_type *restrictPoint(const point_type& p) {
        return &this->_sizes[p];
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
            txt << "viewing a PartitionSizeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing PartitionSizeSection '" << name << "'" << std::endl;
          }
        }
        for(typename sizes_type::const_iterator s_iter = this->_sizes.begin(); s_iter != this->_sizes.end(); ++s_iter) {
          const marker_type& partition = s_iter->first;
          const value_type   size      = s_iter->second;

          txt << "[" << this->commRank() << "]: Partition " << partition << " size " << size << std::endl;
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Point_>
    class PartitionDomain {
    public:
      typedef Point_ point_type;
    public:
      PartitionDomain() {};
      ~PartitionDomain() {};
    public:
      int count(const point_type& point) const {return 1;};
    };

    // This section returns the points in each partition
    template<typename Bundle_, typename Marker_>
    class PartitionSection : public ALE::ParallelObject {
    public:
      typedef Bundle_                            bundle_type;
      typedef typename bundle_type::sieve_type   sieve_type;
      typedef typename bundle_type::point_type   point_type;
      typedef Marker_                            marker_type;
      typedef int                                value_type;
      typedef std::map<marker_type, point_type*> points_type;
      typedef PartitionDomain<point_type>        chart_type;
    protected:
      points_type _points;
      chart_type  _domain;
      int         _height;
      void _init(const Obj<bundle_type>& bundle, const int numElements, const marker_type partition[]) {
        // Should check for patch 0
        const Obj<typename bundle_type::label_sequence>& cells      = bundle->heightStratum(this->_height);
        const Obj<typename bundle_type::numbering_type>& cNumbering = bundle->getFactory()->getLocalNumbering(bundle, bundle->depth() - this->_height);
        std::map<marker_type, std::set<point_type> >     points;
        std::map<marker_type, int>                       offsets;

        if (numElements != (int) cells->size()) {
          throw ALE::Exception("Partition size does not match the number of elements");
        }
        for(typename bundle_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          typedef ALE::SieveAlg<bundle_type> sieve_alg_type;
          const Obj<typename sieve_alg_type::coneArray>& closure = sieve_alg_type::closure(bundle, *e_iter);
          const int idx = cNumbering->getIndex(*e_iter);

          points[partition[idx]].insert(closure->begin(), closure->end());
          if (this->_height > 0) {
            const Obj<typename sieve_alg_type::supportArray>& star = sieve_alg_type::star(bundle, *e_iter);

            points[partition[idx]].insert(star->begin(), star->end());
          }
        }
        for(typename std::map<marker_type, std::set<point_type> >::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          this->_points[p_iter->first] = new point_type[p_iter->second.size()];
          offsets[p_iter->first] = 0;
          for(typename std::set<point_type>::const_iterator s_iter = p_iter->second.begin(); s_iter != p_iter->second.end(); ++s_iter) {
            this->_points[p_iter->first][offsets[p_iter->first]++] = *s_iter;
          }
        }
        for(typename std::map<marker_type, std::set<point_type> >::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          if (offsets[p_iter->first] != (int) p_iter->second.size()) {
            ostringstream txt;
            txt << "Invalid offset for partition " << p_iter->first << ": " << offsets[p_iter->first] << " should be " << p_iter->second.size();
            throw ALE::Exception(txt.str().c_str());
          }
        }
      };
    public:
      PartitionSection(const Obj<bundle_type>& bundle, const int elementHeight, const int numElements, const marker_type *partition) : ParallelObject(MPI_COMM_SELF, bundle->debug()), _height(elementHeight) {
        this->_init(bundle, numElements, partition);
      };
      virtual ~PartitionSection() {
        for(typename points_type::iterator p_iter = this->_points.begin(); p_iter != this->_points.end(); ++p_iter) {
          delete [] p_iter->second;
        }
      };
    public:
      const chart_type& getChart() {return this->_domain;};
      bool hasPoint(const point_type& point) {return true;};
      const value_type *restrictPoint(const point_type& p) {
        return this->_points[p];
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
            txt << "viewing a PartitionSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing PartitionSection '" << name << "'" << std::endl;
          }
        }
        for(typename points_type::const_iterator p_iter = this->_points.begin(); p_iter != this->_points.end(); ++p_iter) {
          const marker_type& partition  = p_iter->first;
          //const point_type *points = p_iter->second;

          txt << "[" << this->commRank() << "]: Partition " << partition << std::endl;
        }
        if (this->_points.size() == 0) {
          txt << "[" << this->commRank() << "]: empty" << std::endl;
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Bundle_, typename Sieve_>
    class ConeSizeSection : public ALE::ParallelObject {
    public:
      typedef ConeSizeSection<Bundle_, Sieve_> section_type;
      typedef int                              patch_type;
      typedef Bundle_                          bundle_type;
      typedef Sieve_                           sieve_type;
      typedef typename bundle_type::point_type point_type;
      typedef int                              value_type;
    protected:
      Obj<bundle_type> _bundle;
      Obj<sieve_type>  _sieve;
      value_type       _size;
      int              _minHeight;
      Obj<section_type> _section;
    public:
      ConeSizeSection(const Obj<bundle_type>& bundle, const Obj<sieve_type>& sieve, int minimumHeight = 0) : ParallelObject(MPI_COMM_SELF, sieve->debug()), _bundle(bundle), _sieve(sieve), _minHeight(minimumHeight) {
        this->_section = this;
        this->_section.addRef();
      };
      virtual ~ConeSizeSection() {};
    public: // Verifiers
      bool hasPoint(const point_type& point) {return true;};
    public: // Restriction
      const value_type *restrictPoint(const point_type& p) {
        if ((this->_minHeight == 0) || (this->_bundle->height(p) >= this->_minHeight)) {
          this->_size = this->_sieve->cone(p)->size();
        } else {
          this->_size = 0;
        }
        return &this->_size;
      };
    public: // Adapter
      const Obj<section_type>& getSection(const patch_type& patch) {
        return this->_section;
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
            txt << "viewing a ConeSizeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing ConeSizeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Sieve_>
    class ConeSection : public ALE::ParallelObject {
    public:
      typedef Sieve_                           sieve_type;
      typedef typename sieve_type::target_type point_type;
      typedef typename sieve_type::source_type value_type;
      typedef PartitionDomain<sieve_type>      chart_type;
    protected:
      Obj<sieve_type> _sieve;
      int             _coneSize;
      value_type     *_cone;
      chart_type      _domain;
      void ensureCone(const int size) {
        if (size > this->_coneSize) {
          if (this->_cone) delete [] this->_cone;
          this->_coneSize = size;
          this->_cone     = new value_type[this->_coneSize];
        }
      };
    public:
      ConeSection(const Obj<sieve_type>& sieve) : ParallelObject(MPI_COMM_SELF, sieve->debug()), _sieve(sieve), _coneSize(-1), _cone(NULL) {};
      virtual ~ConeSection() {if (this->_cone) delete [] this->_cone;};
    public:
      const chart_type& getChart() {return this->_domain;};
      bool hasPoint(const point_type& point) {return true;};
      const value_type *restrictPoint(const point_type& p) {
        const Obj<typename sieve_type::traits::coneSequence>& cone = this->_sieve->cone(p);
        int c = 0;

        this->ensureCone(cone->size());
        for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
          this->_cone[c++] = *c_iter;
        }
        return this->_cone;
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
            txt << "viewing a ConeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing ConeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Bundle_, typename Sieve_>
    class SupportSizeSection : public ALE::ParallelObject {
    public:
      typedef Bundle_                          bundle_type;
      typedef Sieve_                           sieve_type;
      typedef typename sieve_type::source_type point_type;
      typedef typename sieve_type::target_type value_type;
    protected:
      Obj<bundle_type> _bundle;
      Obj<sieve_type>  _sieve;
      value_type       _size;
      int              _minDepth;
    public:
      SupportSizeSection(const Obj<bundle_type>& bundle, const Obj<sieve_type>& sieve, int minimumDepth = 0) : ParallelObject(MPI_COMM_SELF, bundle->debug()), _bundle(bundle), _sieve(sieve), _minDepth(minimumDepth) {};
      virtual ~SupportSizeSection() {};
    public:
      bool hasPoint(const point_type& point) {return true;};
      const value_type *restrictPoint(const point_type& p) {
        if ((this->_minDepth == 0) || (this->_bundle->depth(p) >= this->_minDepth)) {
          this->_size = this->_sieve->support(p)->size();
        } else {
          this->_size = 0;
        }
        return &this->_size;
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
            txt << "viewing a SupportSizeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing SupportSizeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Sieve_>
    class SupportSection : public ALE::ParallelObject {
    public:
      typedef Sieve_                           sieve_type;
      typedef typename sieve_type::source_type point_type;
      typedef typename sieve_type::target_type value_type;
      typedef PartitionDomain<sieve_type>      chart_type;
    protected:
      Obj<sieve_type> _sieve;
      int             _supportSize;
      value_type     *_support;
      chart_type      _domain;
      void ensureSupport(const int size) {
        if (size > this->_supportSize) {
          if (this->_support) delete [] this->_support;
          this->_supportSize = size;
          this->_support     = new value_type[this->_supportSize];
        }
      };
    public:
      SupportSection(const Obj<sieve_type>& sieve) : ParallelObject(MPI_COMM_SELF, sieve->debug()), _sieve(sieve), _supportSize(-1), _support(NULL) {};
      virtual ~SupportSection() {if (this->_support) delete [] this->_support;};
    public:
      const chart_type& getChart() {return this->_domain;};
      bool hasPoint(const point_type& point) {return true;};
      const value_type *restrictPoint(const point_type& p) {
        const Obj<typename sieve_type::traits::supportSequence>& support = this->_sieve->support(p);
        int s = 0;

        this->ensureSupport(support->size());
        for(typename sieve_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
          this->_support[s++] = *s_iter;
        }
        return this->_support;
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
            txt << "viewing a SupportSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing SupportSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Sieve_, typename Section_>
    class ArrowSection : public ALE::ParallelObject {
    public:
      typedef Sieve_                            sieve_type;
      typedef Section_                          section_type;
      typedef typename sieve_type::target_type  point_type;
      typedef typename section_type::point_type arrow_type;
      typedef typename section_type::value_type value_type;
    protected:
      Obj<sieve_type>   _sieve;
      Obj<section_type> _section;
      int               _coneSize;
      value_type       *_cone;
      void ensureCone(const int size) {
        if (size > this->_coneSize) {
          if (this->_cone) delete [] this->_cone;
          this->_coneSize = size;
          this->_cone     = new value_type[this->_coneSize];
        }
      };
    public:
      ArrowSection(const Obj<sieve_type>& sieve, const Obj<section_type>& section) : ParallelObject(MPI_COMM_SELF, sieve->debug()), _sieve(sieve), _section(section), _coneSize(-1), _cone(NULL) {};
      virtual ~ArrowSection() {if (this->_cone) delete [] this->_cone;};
    public:
      bool hasPoint(const point_type& point) {return this->_sieve->baseContains(point);};
      const value_type *restrictPoint(const point_type& p) {
        const Obj<typename sieve_type::traits::coneSequence>& cone = this->_sieve->cone(p);
        int c = -1;

        this->ensureCone(cone->size());
        for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
          this->_cone[++c] = this->_section->restrictPoint(arrow_type(*c_iter, p))[0];
        }
        return this->_cone;
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
            txt << "viewing a ConeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing ConeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };
  }
}

namespace ALECompat {
  namespace New {
    // This section takes an existing section, and reports instead the fiber dimensions as values
    //   Should not need the _patch variable
    template<typename Section_>
    class SizeSection : public ALE::ParallelObject {
    public:
      typedef Section_                          section_type;
      typedef typename section_type::patch_type patch_type;
      typedef typename section_type::point_type point_type;
      typedef int                               value_type;
    protected:
      Obj<section_type> _section;
      const patch_type  _patch;
      value_type        _size;
    public:
      SizeSection(const Obj<section_type>& section, const patch_type& patch) : ParallelObject(MPI_COMM_SELF, section->debug()), _section(section), _patch(patch) {};
      virtual ~SizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {
        return this->_section->hasPoint(patch, point);
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {
        this->_size = this->_section->getFiberDimension(this->_patch, p); // Could be size()
        return &this->_size;
      };
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        this->_size = this->_section->getFiberDimension(this->_patch, p);
        return &this->_size;
      };
    public:
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        this->_section->view(name, comm);
      };
    };

    // This section reports as values the size of the partition associated with the partition point
    template<typename Topology_, typename MeshTopology_, typename Marker_>
    class PartitionSizeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef MeshTopology_                      mesh_topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef typename ALECompat::New::NumberingFactory<mesh_topology_type> NumberingFactory;
      typedef Marker_                            marker_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type> _topology;
      sizes_type         _sizes;
      int                _height;
      void _init(const Obj<mesh_topology_type>& topology, const int numElements, const marker_type partition[]) {
        if (topology->depth() > 1) throw ALE::Exception("Compatibility layer is not for interpolated meshes");
        // Should check for patch 0
        const typename mesh_topology_type::patch_type           patch = 0;
        const Obj<typename mesh_topology_type::sieve_type>&     sieve = topology->getPatch(patch);
        const Obj<typename mesh_topology_type::label_sequence>& cells = topology->heightStratum(patch, this->_height);
        const Obj<typename NumberingFactory::numbering_type>&   cNumbering = NumberingFactory::singleton(topology->debug())->getLocalNumbering(topology, patch, topology->depth(patch) - this->_height);
        std::map<patch_type, std::set<point_type> >             points;

        if (numElements != (int) cells->size()) {
          throw ALE::Exception("Partition size does not match the number of elements");
        }
        for(typename mesh_topology_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          const Obj<typename mesh_topology_type::sieve_type::coneSequence>& closure = sieve->cone(*e_iter);
          const int idx = cNumbering->getIndex(*e_iter);

          points[partition[idx]].insert(*e_iter);
          points[partition[idx]].insert(closure->begin(), closure->end());
          if (this->_height > 0) {
            const Obj<typename mesh_topology_type::sieve_type::supportSequence>& star = sieve->support(*e_iter);

            points[partition[idx]].insert(star->begin(), star->end());
          }
        }
        for(typename std::map<patch_type, std::set<point_type> >::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          this->_sizes[p_iter->first] = p_iter->second.size();
        }
      };
    public:
      PartitionSizeSection(const Obj<topology_type>& topology, const Obj<mesh_topology_type>& meshTopology, const int elementHeight, const int numElements, const marker_type *partition) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _height(elementHeight) {
        this->_init(meshTopology, numElements, partition);
      };
      virtual ~PartitionSizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        return &this->_sizes[p];
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
            txt << "viewing a PartitionSizeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing PartitionSizeSection '" << name << "'" << std::endl;
          }
        }
        for(typename sizes_type::const_iterator s_iter = this->_sizes.begin(); s_iter != this->_sizes.end(); ++s_iter) {
          const patch_type& patch = s_iter->first;
          const value_type  size  = s_iter->second;

          txt << "[" << this->commRank() << "]: Patch " << patch << " size " << size << std::endl;
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_>
    class PartitionDomain {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::point_type point_type;
    public:
      PartitionDomain() {};
      ~PartitionDomain() {};
    public:
      int count(const point_type& point) const {return 1;};
    };

    // This section returns the points in each partition
    template<typename Topology_, typename MeshTopology_, typename Marker_>
    class PartitionSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef MeshTopology_                      mesh_topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef ALECompat::New::NumberingFactory<mesh_topology_type> NumberingFactory;
      typedef Marker_                            marker_type;
      typedef int                                value_type;
      typedef std::map<patch_type,point_type*>   points_type;
      typedef PartitionDomain<topology_type>     chart_type;
    protected:
      Obj<topology_type> _topology;
      points_type        _points;
      chart_type         _domain;
      int                _height;
      void _init(const Obj<mesh_topology_type>& topology, const int numElements, const marker_type partition[]) {
        if (topology->depth() > 1) throw ALE::Exception("Compatibility layer is not for interpolated meshes");
        // Should check for patch 0
        const typename mesh_topology_type::patch_type           patch = 0;
        const Obj<typename mesh_topology_type::sieve_type>&     sieve = topology->getPatch(patch);
        const Obj<typename mesh_topology_type::label_sequence>& cells = topology->heightStratum(patch, this->_height);
        const Obj<typename NumberingFactory::numbering_type>&   cNumbering = NumberingFactory::singleton(topology->debug())->getLocalNumbering(topology, patch, topology->depth(patch) - this->_height);
        std::map<patch_type, std::set<point_type> >             points;
        std::map<patch_type, int>                               offsets;

        if (numElements != (int) cells->size()) {
          throw ALE::Exception("Partition size does not match the number of elements");
        }
        for(typename mesh_topology_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          const Obj<typename mesh_topology_type::sieve_type::coneSequence>& closure = sieve->cone(*e_iter);
          const int idx = cNumbering->getIndex(*e_iter);

          points[partition[idx]].insert(*e_iter);
          points[partition[idx]].insert(closure->begin(), closure->end());
          if (this->_height > 0) {
            const Obj<typename mesh_topology_type::sieve_type::supportSequence>& star = sieve->support(*e_iter);

            points[partition[idx]].insert(star->begin(), star->end());
          }
        }
        for(typename std::map<patch_type, std::set<point_type> >::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          this->_points[p_iter->first] = new point_type[p_iter->second.size()];
          offsets[p_iter->first] = 0;
          for(typename std::set<point_type>::const_iterator s_iter = p_iter->second.begin(); s_iter != p_iter->second.end(); ++s_iter) {
            this->_points[p_iter->first][offsets[p_iter->first]++] = *s_iter;
          }
        }
        for(typename std::map<patch_type, std::set<point_type> >::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          if (offsets[p_iter->first] != (int) p_iter->second.size()) {
            ostringstream txt;
            txt << "Invalid offset for partition " << p_iter->first << ": " << offsets[p_iter->first] << " should be " << p_iter->second.size();
            throw ALE::Exception(txt.str().c_str());
          }
        }
      };
    public:
      PartitionSection(const Obj<topology_type>& topology, const Obj<mesh_topology_type>& meshTopology, const int elementHeight, const int numElements, const marker_type *partition) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _height(elementHeight) {
        this->_init(meshTopology, numElements, partition);
      };
      virtual ~PartitionSection() {
        for(typename points_type::iterator p_iter = this->_points.begin(); p_iter != this->_points.end(); ++p_iter) {
          delete [] p_iter->second;
        }
      };
    public:
      const chart_type& getPatch(const patch_type& patch) {return this->_domain;};
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        return this->_points[p];
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
            txt << "viewing a PartitionSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing PartitionSection '" << name << "'" << std::endl;
          }
        }
        for(typename points_type::const_iterator p_iter = this->_points.begin(); p_iter != this->_points.end(); ++p_iter) {
          const patch_type& patch  = p_iter->first;
          //const point_type *points = p_iter->second;

          txt << "[" << this->commRank() << "]: Patch " << patch << std::endl;
        }
        if (this->_points.size() == 0) {
          txt << "[" << this->commRank() << "]: empty" << std::endl;
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_, typename MeshTopology_, typename Sieve_>
    class ConeSizeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef MeshTopology_                      mesh_topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Sieve_                             cone_sieve_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type>      _topology;
      Obj<mesh_topology_type> _meshTopology;
      Obj<cone_sieve_type>    _sieve;
      value_type              _size;
      int                     _minHeight;
    public:
      ConeSizeSection(const Obj<topology_type>& topology, const Obj<mesh_topology_type>& meshTopology, const Obj<cone_sieve_type>& sieve, int minimumHeight = 0) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _meshTopology(meshTopology), _sieve(sieve), _minHeight(minimumHeight) {};
      virtual ~ConeSizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        if ((this->_minHeight == 0) || (this->_meshTopology->height(patch, p) >= this->_minHeight)) {
          this->_size = this->_sieve->cone(p)->size();
        } else {
          this->_size = 0;
        }
        return &this->_size;
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
            txt << "viewing a ConeSizeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing ConeSizeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_, typename Sieve_>
    class ConeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Sieve_                             cone_sieve_type;
      typedef point_type                         value_type;
      typedef std::map<patch_type,int>           sizes_type;
      typedef PartitionDomain<topology_type>     chart_type;
    protected:
      Obj<topology_type>      _topology;
      Obj<cone_sieve_type>    _sieve;
      int                     _coneSize;
      value_type             *_cone;
      chart_type              _domain;
      void ensureCone(const int size) {
        if (size > this->_coneSize) {
          if (this->_cone) delete [] this->_cone;
          this->_coneSize = size;
          this->_cone     = new value_type[this->_coneSize];
        }
      };
    public:
      ConeSection(MPI_Comm comm, const Obj<cone_sieve_type>& sieve, const int debug = 0) : ParallelObject(comm, debug), _sieve(sieve), _coneSize(-1), _cone(NULL) {
        this->_topology = new topology_type(comm, debug);
      };
      ConeSection(const Obj<topology_type>& topology, const Obj<cone_sieve_type>& sieve) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _sieve(sieve), _coneSize(-1), _cone(NULL) {};
      virtual ~ConeSection() {if (this->_cone) delete [] this->_cone;};
    public:
      const chart_type& getPatch(const patch_type& patch) {return this->_domain;};
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        const Obj<typename cone_sieve_type::traits::coneSequence>& cone = this->_sieve->cone(p);
        int c = 0;

        this->ensureCone(cone->size());
        for(typename cone_sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
          this->_cone[c++] = *c_iter;
        }
        return this->_cone;
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
            txt << "viewing a ConeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing ConeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_, typename MeshTopology_, typename Sieve_>
    class SupportSizeSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef MeshTopology_                      mesh_topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Sieve_                             support_sieve_type;
      typedef int                                value_type;
      typedef std::map<patch_type,int>           sizes_type;
    protected:
      Obj<topology_type>      _topology;
      Obj<mesh_topology_type> _meshTopology;
      Obj<support_sieve_type> _sieve;
      value_type              _size;
      int                     _minDepth;
    public:
      SupportSizeSection(const Obj<topology_type>& topology, const Obj<mesh_topology_type>& meshTopology, const Obj<support_sieve_type>& sieve, int minimumDepth = 0) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _meshTopology(meshTopology), _sieve(sieve), _minDepth(minimumDepth) {};
      virtual ~SupportSizeSection() {};
    public:
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        if ((this->_minDepth == 0) || (this->_meshTopology->depth(patch, p) >= this->_minDepth)) {
          this->_size = this->_sieve->support(p)->size();
        } else {
          this->_size = 0;
        }
        return &this->_size;
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
            txt << "viewing a SupportSizeSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing SupportSizeSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Topology_, typename Sieve_>
    class SupportSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Sieve_                             support_sieve_type;
      typedef point_type                         value_type;
      typedef std::map<patch_type,int>           sizes_type;
      typedef PartitionDomain<topology_type>     chart_type;
    protected:
      Obj<topology_type>      _topology;
      Obj<support_sieve_type> _sieve;
      int                     _supportSize;
      value_type             *_support;
      chart_type              _domain;
      void ensureSupport(const int size) {
        if (size > this->_supportSize) {
          if (this->_support) delete [] this->_support;
          this->_supportSize = size;
          this->_support     = new value_type[this->_supportSize];
        }
      };
    public:
      SupportSection(MPI_Comm comm, const Obj<support_sieve_type>& sieve, const int debug = 0) : ParallelObject(comm, debug), _sieve(sieve), _supportSize(-1), _support(NULL) {
        this->_topology = new topology_type(comm, debug);
      };
      SupportSection(const Obj<topology_type>& topology, const Obj<support_sieve_type>& sieve) : ParallelObject(MPI_COMM_SELF, topology->debug()), _topology(topology), _sieve(sieve), _supportSize(-1), _support(NULL) {};
      virtual ~SupportSection() {if (this->_support) delete [] this->_support;};
    public:
      const chart_type& getPatch(const patch_type& patch) {return this->_domain;};
      bool hasPoint(const patch_type& patch, const point_type& point) {return true;};
      const value_type *restrict(const patch_type& patch, const point_type& p) {return this->restrictPoint(patch, p);};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        const Obj<typename support_sieve_type::traits::supportSequence>& support = this->_sieve->support(p);
        int s = 0;

        this->ensureSupport(support->size());
        for(typename support_sieve_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
          this->_support[s++] = *s_iter;
        }
        return this->_support;
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
            txt << "viewing a SupportSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing SupportSection '" << name << "'" << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };
  }
}
#endif
