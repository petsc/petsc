#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#include <list>
#include <valarray>

#ifndef  included_ALE_Numbering_hh
#include <sieve/Numbering.hh>
#endif

#ifndef  included_ALE_INumbering_hh
#include <sieve/INumbering.hh>
#endif

#ifndef  included_ALE_Field_hh
#include <sieve/Field.hh>
#endif

#ifndef  included_ALE_IField_hh
#include <sieve/IField.hh>
#endif

#ifndef  included_ALE_SieveBuilder_hh
#include <sieve/SieveBuilder.hh>
#endif

#ifndef  included_ALE_LabelSifter_hh
#include <sieve/LabelSifter.hh>
#endif

#ifndef  included_ALE_Partitioner_hh
#include <sieve/Partitioner.hh>
#endif

#ifndef  included_ALE_Ordering_hh
#include <sieve/Ordering.hh>
#endif

#ifndef  included_PETSc_Overlap_hh
#include <sieve/Overlap.hh>
#endif

namespace ALE {
  class indexSet : public std::valarray<int> {
  public:
    inline bool
    operator<(const indexSet& __x) {
      if (__x.size() != this->size()) return __x.size() < this->size();
      for(unsigned int i = 0; i < __x.size(); ++i) {
        if (__x[i] == (*this)[i]) continue;
          return __x[i] < (*this)[i];
      }
      return false;
    }
  };
  inline bool
  operator<(const indexSet& __x, const indexSet& __y) {
    if (__x.size() != __y.size()) return __x.size() < __y.size();
    for(unsigned int i = 0; i < __x.size(); ++i) {
      if (__x[i] == __y[i]) continue;
      return __x[i] < __y[i];
    }
    return false;
  };
  inline bool
  operator<=(const indexSet& __x, const indexSet& __y) {
    if (__x.size() != __y.size()) return __x.size() < __y.size();
    for(unsigned int i = 0; i < __x.size(); ++i) {
      if (__x[i] == __y[i]) continue;
      return __x[i] < __y[i];
    }
    return true;
  };
  inline bool
  operator==(const indexSet& __x, const indexSet& __y) {
    if (__x.size() != __y.size()) return false;
    for(unsigned int i = 0; i < __x.size(); ++i) {
      if (__x[i] != __y[i]) return false;
    }
    return true;
  };
  inline bool
  operator!=(const indexSet& __x, const indexSet& __y) {
    if (__x.size() != __y.size()) return true;
    for(unsigned int i = 0; i < __x.size(); ++i) {
      if (__x[i] != __y[i]) return true;
    }
    return false;
  };

  template<typename Sieve_,
           typename RealSection_  = Section<typename Sieve_::point_type, double>,
           typename IntSection_   = Section<typename Sieve_::point_type, int>,
           typename ArrowSection_ = UniformSection<MinimalArrow<typename Sieve_::point_type, typename Sieve_::point_type>, int> >
  class Bundle : public ALE::ParallelObject {
  public:
    typedef Sieve_                                                    sieve_type;
    typedef RealSection_                                              real_section_type;
    typedef IntSection_                                               int_section_type;
    typedef ArrowSection_                                             arrow_section_type;
    typedef Bundle<Sieve_,RealSection_,IntSection_,ArrowSection_>     this_type;
    typedef typename sieve_type::point_type                           point_type;
    typedef malloc_allocator<point_type>                              alloc_type;
    typedef typename ALE::LabelSifter<int, point_type>                label_type;
    typedef typename std::map<const std::string, Obj<label_type> >    labels_type;
    typedef typename label_type::supportSequence                      label_sequence;
    typedef std::map<std::string, Obj<arrow_section_type> >           arrow_sections_type;
    typedef std::map<std::string, Obj<real_section_type> >            real_sections_type;
    typedef std::map<std::string, Obj<int_section_type> >             int_sections_type;
    typedef ALE::Point                                                index_type;
    typedef std::pair<index_type, int>                                oIndex_type;
    typedef std::vector<oIndex_type>                                  oIndexArray;
    typedef std::pair<int *, int>                                     indices_type;
    typedef NumberingFactory<this_type>                               MeshNumberingFactory;
    typedef typename ALE::Partitioner<>::part_type                    rank_type;
    typedef typename ALE::Sifter<point_type,rank_type,point_type>     send_overlap_type;
    typedef typename ALE::Sifter<rank_type,point_type,point_type>     recv_overlap_type;
    typedef typename MeshNumberingFactory::numbering_type             numbering_type;
    typedef typename MeshNumberingFactory::order_type                 order_type;
    typedef std::map<point_type, point_type>                          renumbering_type;
    typedef typename ALE::SieveAlg<this_type>                         sieve_alg_type;
    typedef typename sieve_alg_type::coneArray                        coneArray;
    typedef typename sieve_alg_type::orientedConeArray                oConeArray;
    typedef typename sieve_alg_type::supportArray                     supportArray;
  protected:
    Obj<sieve_type>       _sieve;
    labels_type           _labels;
    int                   _maxHeight;
    int                   _maxDepth;
    arrow_sections_type   _arrowSections;
    real_sections_type    _realSections;
    int_sections_type     _intSections;
    Obj<oIndexArray>      _indexArray;
    Obj<MeshNumberingFactory> _factory;
    bool                   _calculatedOverlap;
    Obj<send_overlap_type> _sendOverlap;
    Obj<recv_overlap_type> _recvOverlap;
    Obj<send_overlap_type> _distSendOverlap;
    Obj<recv_overlap_type> _distRecvOverlap;
    renumbering_type       _renumbering; // Maps global points to local points
    // Work space
    Obj<std::set<point_type> > _modifiedPoints;
  public:
    Bundle(MPI_Comm comm, int debug = 0) : ALE::ParallelObject(comm, debug), _maxHeight(-1), _maxDepth(-1) {
      this->_indexArray        = new oIndexArray();
      this->_modifiedPoints    = new std::set<point_type>();
      this->_factory           = MeshNumberingFactory::singleton(this->comm(), this->debug());
      this->_calculatedOverlap = false;
      this->_sendOverlap       = new send_overlap_type(comm, debug);
      this->_recvOverlap       = new recv_overlap_type(comm, debug);
    };
    Bundle(const Obj<sieve_type>& sieve) : ALE::ParallelObject(sieve->comm(), sieve->debug()), _sieve(sieve), _maxHeight(-1), _maxDepth(-1) {
      this->_indexArray        = new oIndexArray();
      this->_modifiedPoints    = new std::set<point_type>();
      this->_factory           = MeshNumberingFactory::singleton(this->comm(), this->debug());
      this->_calculatedOverlap = false;
      this->_sendOverlap       = new send_overlap_type(this->comm(), this->debug());
      this->_recvOverlap       = new recv_overlap_type(this->comm(), this->debug());
    };
    virtual ~Bundle() {};
  public: // Verifiers
    bool hasLabel(const std::string& name) {
      if (this->_labels.find(name) != this->_labels.end()) {
        return true;
      }
      return false;
    };
    void checkLabel(const std::string& name) {
      if (!this->hasLabel(name)) {
        ostringstream msg;
        msg << "Invalid label name: " << name << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
  public: // Accessors
    const Obj<sieve_type>& getSieve() const {return this->_sieve;};
    void setSieve(const Obj<sieve_type>& sieve) {this->_sieve = sieve;};
    bool hasArrowSection(const std::string& name) const {
      return this->_arrowSections.find(name) != this->_arrowSections.end();
    };
    const Obj<arrow_section_type>& getArrowSection(const std::string& name) {
      if (!this->hasArrowSection(name)) {
        Obj<arrow_section_type> section = new arrow_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new arrow section: " << name << std::endl;}
        this->_arrowSections[name] = section;
      }
      return this->_arrowSections[name];
    };
    void setArrowSection(const std::string& name, const Obj<arrow_section_type>& section) {
      this->_arrowSections[name] = section;
    };
    Obj<std::set<std::string> > getArrowSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename arrow_sections_type::const_iterator s_iter = this->_arrowSections.begin(); s_iter != this->_arrowSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasRealSection(const std::string& name) const {
      return this->_realSections.find(name) != this->_realSections.end();
    };
    const Obj<real_section_type>& getRealSection(const std::string& name) {
      if (!this->hasRealSection(name)) {
        Obj<real_section_type> section = new real_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new real section: " << name << std::endl;}
        this->_realSections[name] = section;
      }
      return this->_realSections[name];
    };
    void setRealSection(const std::string& name, const Obj<real_section_type>& section) {
      this->_realSections[name] = section;
    };
    Obj<std::set<std::string> > getRealSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename real_sections_type::const_iterator s_iter = this->_realSections.begin(); s_iter != this->_realSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasIntSection(const std::string& name) const {
      return this->_intSections.find(name) != this->_intSections.end();
    };
    const Obj<int_section_type>& getIntSection(const std::string& name) {
      if (!this->hasIntSection(name)) {
        Obj<int_section_type> section = new int_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new int section: " << name << std::endl;}
        this->_intSections[name] = section;
      }
      return this->_intSections[name];
    };
    void setIntSection(const std::string& name, const Obj<int_section_type>& section) {
      this->_intSections[name] = section;
    };
    Obj<std::set<std::string> > getIntSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename int_sections_type::const_iterator s_iter = this->_intSections.begin(); s_iter != this->_intSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    const Obj<MeshNumberingFactory>& getFactory() const {return this->_factory;};
    bool getCalculatedOverlap() const {return this->_calculatedOverlap;};
    void setCalculatedOverlap(const bool calc) {this->_calculatedOverlap = calc;};
    const Obj<send_overlap_type>& getSendOverlap() const {return this->_sendOverlap;};
    void setSendOverlap(const Obj<send_overlap_type>& overlap) {this->_sendOverlap = overlap;};
    const Obj<recv_overlap_type>& getRecvOverlap() const {return this->_recvOverlap;};
    void setRecvOverlap(const Obj<recv_overlap_type>& overlap) {this->_recvOverlap = overlap;};
    const Obj<send_overlap_type>& getDistSendOverlap() const {return this->_distSendOverlap;};
    void setDistSendOverlap(const Obj<send_overlap_type>& overlap) {this->_distSendOverlap = overlap;};
    const Obj<recv_overlap_type>& getDistRecvOverlap() const {return this->_distRecvOverlap;};
    void setDistRecvOverlap(const Obj<recv_overlap_type>& overlap) {this->_distRecvOverlap = overlap;};
    renumbering_type& getRenumbering() {return this->_renumbering;};
  public: // Labels
    int getValue (const Obj<label_type>& label, const point_type& point, const int defValue = 0) {
      const Obj<typename label_type::coneSequence>& cone = label->cone(point);

      if (cone->size() == 0) return defValue;
      return *cone->begin();
    };
    void setValue(const Obj<label_type>& label, const point_type& point, const int value) {
      label->setCone(value, point);
    };
    template<typename InputPoints>
    int getMaxValue (const Obj<label_type>& label, const Obj<InputPoints>& points, const int defValue = 0) {
      int maxValue = defValue;

      for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        maxValue = std::max(maxValue, this->getValue(label, *p_iter, defValue));
      }
      return maxValue;
    }
    const Obj<label_type>& createLabel(const std::string& name) {
      this->_labels[name] = new label_type(this->comm(), this->debug());
      return this->_labels[name];
    };
    const Obj<label_type>& getLabel(const std::string& name) {
      this->checkLabel(name);
      return this->_labels[name];
    };
    void setLabel(const std::string& name, const Obj<label_type>& label) {
      this->_labels[name] = label;
    };
    const labels_type& getLabels() {
      return this->_labels;
    };
    virtual const Obj<label_sequence>& getLabelStratum(const std::string& name, int value) {
      this->checkLabel(name);
      return this->_labels[name]->support(value);
    };
  public: // Stratification
    template<class InputPoints>
    void computeHeight(const Obj<label_type>& height, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points, int& maxHeight) {
      this->_modifiedPoints->clear();

      for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        // Compute the max height of the points in the support of p, and add 1
        int h0 = this->getValue(height, *p_iter, -1);
        int h1 = this->getMaxValue(height, sieve->support(*p_iter), -1) + 1;

        if(h1 != h0) {
          this->setValue(height, *p_iter, h1);
          if (h1 > maxHeight) maxHeight = h1;
          this->_modifiedPoints->insert(*p_iter);
        }
      }
      // FIX: We would like to avoid the copy here with cone()
      if(this->_modifiedPoints->size() > 0) {
        this->computeHeight(height, sieve, sieve->cone(this->_modifiedPoints), maxHeight);
      }
    }
    void computeHeights() {
      const Obj<label_type>& label = this->createLabel(std::string("height"));

      this->_maxHeight = -1;
      this->computeHeight(label, this->_sieve, this->_sieve->leaves(), this->_maxHeight);
    };
    virtual int height() const {return this->_maxHeight;};
    virtual int height(const point_type& point) {
      return this->getValue(this->_labels["height"], point, -1);
    };
    virtual const Obj<label_sequence>& heightStratum(int height) {
      return this->getLabelStratum("height", height);
    };
    void setHeight(const Obj<label_type>& label) {
      this->_labels["height"] = label;
      const Obj<typename label_type::traits::capSequence> cap = label->cap();

      for(typename label_type::traits::capSequence::iterator c_iter = cap->begin(); c_iter != cap->end(); ++c_iter) {
        this->_maxHeight = std::max(this->_maxHeight, *c_iter);
      }
    };
    template<class InputPoints>
    void computeDepth(const Obj<label_type>& depth, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points, int& maxDepth) {
      this->_modifiedPoints->clear();

      for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        // Compute the max depth of the points in the cone of p, and add 1
        int d0 = this->getValue(depth, *p_iter, -1);
        int d1 = this->getMaxValue(depth, sieve->cone(*p_iter), -1) + 1;

        if(d1 != d0) {
          this->setValue(depth, *p_iter, d1);
          if (d1 > maxDepth) maxDepth = d1;
          this->_modifiedPoints->insert(*p_iter);
        }
      }
      // FIX: We would like to avoid the copy here with support()
      if(this->_modifiedPoints->size() > 0) {
        this->computeDepth(depth, sieve, sieve->support(this->_modifiedPoints), maxDepth);
      }
    }
    void computeDepths() {
      const Obj<label_type>& label = this->createLabel(std::string("depth"));

      this->_maxDepth = -1;
      this->computeDepth(label, this->_sieve, this->_sieve->roots(), this->_maxDepth);
    };
    virtual int depth() const {return this->_maxDepth;};
    virtual int depth(const point_type& point) {
      return this->getValue(this->_labels["depth"], point, -1);
    };
    virtual const Obj<label_sequence>& depthStratum(int depth) {
      return this->getLabelStratum("depth", depth);
    };
    #undef __FUNCT__
    #define __FUNCT__ "stratify"
    virtual void stratify() {
      ALE_LOG_EVENT_BEGIN;
      this->computeHeights();
      this->computeDepths();
      ALE_LOG_EVENT_END;
    };
  public: // Size traversal
    template<typename Section_>
    int size(const Obj<Section_>& section, const point_type& p) {
      const typename Section_::chart_type& chart = section->getChart();
      int                                  size  = 0;

      if (this->height() < 2) {
        const Obj<typename sieve_type::coneSequence>& cone = this->_sieve->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        if (chart.count(p)) {
          size += section->getConstrainedFiberDimension(p);
        }
        for(typename sieve_type::coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter) {
          if (chart.count(*c_iter)) {
            size += section->getConstrainedFiberDimension(*c_iter);
          }
        }
      } else {
        const Obj<coneArray>         closure = sieve_alg_type::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename coneArray::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
          if (chart.count(*c_iter)) {
            size += section->getConstrainedFiberDimension(*c_iter);
          }
        }
      }
      return size;
    }
    template<typename Section_>
    int sizeWithBC(const Obj<Section_>& section, const point_type& p) {
      const typename Section_::chart_type& chart = section->getChart();
      int                                  size  = 0;

      if (this->height() < 2) {
        const Obj<typename sieve_type::coneSequence>& cone = this->_sieve->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        if (chart.count(p)) {
          size += section->getFiberDimension(p);
        }
        for(typename sieve_type::coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter) {
          if (chart.count(*c_iter)) {
            size += section->getFiberDimension(*c_iter);
          }
        }
      } else {
        const Obj<coneArray>         closure = sieve_alg_type::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename coneArray::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
          if (chart.count(*c_iter)) {
            size += section->getFiberDimension(*c_iter);
          }
        }
      }
      return size;
    }
  protected:
    int *getIndexArray(const int size) {
      static int *array   = NULL;
      static int  maxSize = 0;

      if (size > maxSize) {
        maxSize = size;
        if (array) delete [] array;
        array = new int[maxSize];
      };
      return array;
    };
  public: // Index traversal
    void expandInterval(const index_type& interval, PetscInt indices[], PetscInt *indx) {
      const int end = interval.prefix + interval.index;

      for(int i = interval.index; i < end; ++i) {
        indices[(*indx)++] = i;
      }
    };
    void expandInterval(const index_type& interval, const int orientation, PetscInt indices[], PetscInt *indx) {
      if (orientation >= 0) {
        for(int i = 0; i < interval.prefix; ++i) {
          indices[(*indx)++] = interval.index + i;
        }
      } else {
        for(int i = interval.prefix-1; i >= 0; --i) {
          indices[(*indx)++] = interval.index + i;
        }
      }
      for(int i = 0; i < -interval.prefix; ++i) {
        indices[(*indx)++] = -1;
      }
    };
    void expandIntervals(Obj<oIndexArray> intervals, PetscInt *indices) {
      int k = 0;

      for(typename oIndexArray::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
        this->expandInterval(i_iter->first, i_iter->second, indices, &k);
      }
    }
    template<typename Section_>
    const indices_type getIndicesRaw(const Obj<Section_>& section, const point_type& p) {
      int *indexArray = NULL;
      int  size       = 0;

      const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
      typename oConeArray::iterator begin   = closure->begin();
      typename oConeArray::iterator end     = closure->end();

      for(typename oConeArray::iterator p_iter = begin; p_iter != end; ++p_iter) {
        size    += section->getFiberDimension(p_iter->first);
      }
      indexArray = this->getIndexArray(size);
      int  k     = 0;
      for(typename oConeArray::iterator p_iter = begin; p_iter != end; ++p_iter) {
        section->getIndicesRaw(p_iter->first, section->getIndex(p_iter->first), indexArray, &k, p_iter->second);
      }
      return indices_type(indexArray, size);
    }
    template<typename Section_>
    const indices_type getIndices(const Obj<Section_>& section, const point_type& p, const int level = -1) {
      int *indexArray = NULL;
      int  size       = 0;

      if (level == 0) {
        size      += section->getFiberDimension(p);
        indexArray = this->getIndexArray(size);
        int  k     = 0;

        section->getIndices(p, indexArray, &k);
      } else if ((level == 1) || (this->height() == 1)) {
        const Obj<typename sieve_type::coneSequence>& cone = this->_sieve->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        size      += section->getFiberDimension(p);
        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          size    += section->getFiberDimension(*p_iter);
        }
        indexArray = this->getIndexArray(size);
        int  k     = 0;

        section->getIndices(p, indexArray, &k);
        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          section->getIndices(*p_iter, indexArray, &k);
        }
      } else if (level == -1) {
        const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
        typename oConeArray::iterator begin   = closure->begin();
        typename oConeArray::iterator end     = closure->end();

        for(typename oConeArray::iterator p_iter = begin; p_iter != end; ++p_iter) {
          size    += section->getFiberDimension(p_iter->first);
        }
        indexArray = this->getIndexArray(size);
        int  k     = 0;
        for(typename oConeArray::iterator p_iter = begin; p_iter != end; ++p_iter) {
          section->getIndices(p_iter->first, indexArray, &k, p_iter->second);
        }
      } else {
        throw ALE::Exception("Bundle has not yet implemented getIndices() for an arbitrary level");
      }
      if (this->debug()) {
        for(int i = 0; i < size; ++i) {
          printf("[%d]index %d: %d\n", this->commRank(), i, indexArray[i]);
        }
      }
      return indices_type(indexArray, size);
    }
    template<typename Section_, typename Numbering>
    const indices_type getIndices(const Obj<Section_>& section, const point_type& p, const Obj<Numbering>& numbering, const int level = -1) {
      int *indexArray = NULL;
      int  size       = 0;

      if (level == 0) {
        size      += section->getFiberDimension(p);
        indexArray = this->getIndexArray(size);
        int  k     = 0;

        section->getIndices(p, numbering, indexArray, &k);
      } else if ((level == 1) || (this->height() == 1)) {
        const Obj<typename sieve_type::coneSequence>& cone = this->_sieve->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        size      += section->getFiberDimension(p);
        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          size    += section->getFiberDimension(*p_iter);
        }
        indexArray = this->getIndexArray(size);
        int  k     = 0;

        section->getIndices(p, numbering, indexArray, &k);
        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          section->getIndices(*p_iter, numbering, indexArray, &k);
        }
      } else if (level == -1) {
        const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
        typename oConeArray::iterator end     = closure->end();

        for(typename oConeArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          size    += section->getFiberDimension(p_iter->first);
        }
        indexArray = this->getIndexArray(size);
        int  k     = 0;
        for(typename oConeArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->getIndices(p_iter->first, numbering, indexArray, &k, p_iter->second);
        }
      } else {
        throw ALE::Exception("Bundle has not yet implemented getIndices() for an arbitrary level");
      }
      return indices_type(indexArray, size);
    }
  public: // Retrieval traversal
    // Return the values for the closure of this point
    //   use a smart pointer?
    template<typename Section_>
    const typename Section_::value_type *restrictClosure(const Obj<Section_>& section, const point_type& p) {
      const int size = this->sizeWithBC(section, p);
      return this->restrictClosure(section, p, section->getRawArray(size), size);
    }
    template<typename Section_>
    const typename Section_::value_type *restrictClosure(const Obj<Section_>& section, const point_type& p, typename Section_::value_type  *values, const int valuesSize) {
      const int size = this->sizeWithBC(section, p);
      int       j    = -1;
      if (valuesSize < size) throw ALE::Exception("Input array too small");

      // We could actually ask for the height of the individual point
      if (this->height() < 2) {
        const int& dim = section->getFiberDimension(p);
        const typename Section_::value_type *array = section->restrictPoint(p);

        for(int i = 0; i < dim; ++i) {
          values[++j] = array[i];
        }
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          const int& dim = section->getFiberDimension(*p_iter);

          array = section->restrictPoint(*p_iter);
          for(int i = 0; i < dim; ++i) {
            values[++j] = array[i];
          }
        }
      } else {
        const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
        typename oConeArray::iterator end     = closure->end();

        for(typename oConeArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          const typename Section_::value_type *array = section->restrictPoint(p_iter->first);
          const int& dim = section->getFiberDimension(p_iter->first);

          if (p_iter->second >= 0) {
            for(int i = 0; i < dim; ++i) {
              values[++j] = array[i];
            }
          } else {
            for(int i = dim-1; i >= 0; --i) {
              values[++j] = array[i];
            }
          }
        }
      }
      if (j != size-1) {
        ostringstream txt;

        txt << "Invalid restrict to point " << p << std::endl;
        txt << "  j " << j << " should be " << (size-1) << std::endl;
        std::cout << txt.str();
        throw ALE::Exception(txt.str().c_str());
      }
      return values;
    }
    template<typename Section_>
    const typename Section_::value_type *restrictNew(const Obj<Section_>& section, const point_type& p) {
      const int size = this->sizeWithBC(section, p);
      return this->restrictNew(section, p, section->getRawArray(size), size);
    }
    template<typename Section_>
    const typename Section_::value_type *restrictNew(const Obj<Section_>& section, const point_type& p, typename Section_::value_type  *values, const int valuesSize) {
      const int                     size    = this->sizeWithBC(section, p);
      const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
      typename oConeArray::iterator end     = closure->end();
      int                           j       = -1;
      if (valuesSize < size) throw ALE::Exception("Input array too small");

      for(typename oConeArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
        const typename Section_::value_type *array = section->restrictPoint(p_iter->first);

        if (p_iter->second >= 0) {
          const int& dim = section->getFiberDimension(p_iter->first);

          for(int i = 0; i < dim; ++i) {
            values[++j] = array[i];
          }
        } else {
          int offset = 0;

          for(int space = 0; space < section->getNumSpaces(); ++space) {
            const int& dim = section->getFiberDimension(p_iter->first, space);

            for(int i = dim-1; i >= 0; --i) {
              values[++j] = array[i+offset];
            }
            offset += dim;
          }
        }
      }
      if (j != size-1) {
        ostringstream txt;

        txt << "Invalid restrict to point " << p << std::endl;
        txt << "  j " << j << " should be " << (size-1) << std::endl;
        std::cout << txt.str();
        throw ALE::Exception(txt.str().c_str());
      }
      return values;
    }
    template<typename Section_>
    void update(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      if (this->height() < 2) {
        section->updatePoint(p, &v[j]);
        j += section->getFiberDimension(p);
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updatePoint(*p_iter, &v[j]);
          j += section->getFiberDimension(*p_iter);
        }
      } else {
        const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
        typename oConeArray::iterator end     = closure->end();

        for(typename oConeArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updatePoint(p_iter->first, &v[j], p_iter->second);
          j += section->getFiberDimension(p_iter->first);
        }
      }
    }
    template<typename Section_>
    void updateAdd(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      if (this->height() < 2) {
        section->updateAddPoint(p, &v[j]);
        j += section->getFiberDimension(p);
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updateAddPoint(*p_iter, &v[j]);
          j += section->getFiberDimension(*p_iter);
        }
      } else {
        const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
        typename oConeArray::iterator end     = closure->end();

        for(typename oConeArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updateAddPoint(p_iter->first, &v[j], p_iter->second);
          j += section->getFiberDimension(p_iter->first);
        }
      }
    }
    template<typename Section_>
    void updateBC(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      if (this->height() < 2) {
        section->updatePointBC(p, &v[j]);
        j += section->getFiberDimension(p);
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updatePointBC(*p_iter, &v[j]);
          j += section->getFiberDimension(*p_iter);
        }
      } else {
        const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
        typename oConeArray::iterator end     = closure->end();

        for(typename oConeArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updatePointBC(p_iter->first, &v[j], p_iter->second);
          j += section->getFiberDimension(p_iter->first);
        }
      }
    }
    template<typename Section_>
    void updateAll(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      if (this->height() < 2) {
        section->updatePointAll(p, &v[j]);
        j += section->getFiberDimension(p);
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updatePointAll(*p_iter, &v[j]);
          j += section->getFiberDimension(*p_iter);
        }
      } else {
        const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
        typename oConeArray::iterator end     = closure->end();

        for(typename oConeArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updatePointAll(p_iter->first, &v[j], p_iter->second);
          j += section->getFiberDimension(p_iter->first);
        }
      }
    }
    template<typename Section_>
    void updateAllAdd(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      if (this->height() < 2) {
        section->updatePointAllAdd(p, &v[j]);
        j += section->getFiberDimension(p);
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updatePointAllAdd(*p_iter, &v[j]);
          j += section->getFiberDimension(*p_iter);
        }
      } else {
        const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
        typename oConeArray::iterator end     = closure->end();

        for(typename oConeArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updatePointAllAdd(p_iter->first, &v[j], p_iter->second);
          j += section->getFiberDimension(p_iter->first);
        }
      }
    }
  public: // Optimization
    // Calculate a custom atlas for the given traversal
    //   This returns the tag value assigned to the traversal
    template<typename Section_, typename Sequence_>
    int calculateCustomAtlas(const Obj<Section_>& section, const Obj<Sequence_>& points) {
      const typename Sequence_::iterator begin    = points->begin();
      const typename Sequence_::iterator end      = points->end();
      const int                          num      = points->size();
      int                               *rOffsets = new int[num+1];
      int                               *rIndices;
      int                               *uOffsets = new int[num+1];
      int                               *uIndices;
      int                                p;

      p = 0;
      rOffsets[p] = 0;
      uOffsets[p] = 0;
      for(typename Sequence_::iterator p_iter = begin; p_iter != end; ++p_iter, ++p) {
        rOffsets[p+1] = rOffsets[p] + this->sizeWithBC(section, *p_iter);
        uOffsets[p+1] = rOffsets[p+1];
        //uOffsets[p+1] = uOffsets[p] + this->size(section, *p_iter);
      }
      rIndices = new int[rOffsets[p]];
      uIndices = new int[uOffsets[p]];
      p = 0;
      for(typename Sequence_::iterator p_iter = begin; p_iter != end; ++p_iter, ++p) {
        const indices_type rIdx = this->getIndicesRaw(section, *p_iter);
        for(int i = 0, k = rOffsets[p]; k < rOffsets[p+1]; ++i, ++k) rIndices[k] = rIdx.first[i];

        const indices_type uIdx = this->getIndices(section, *p_iter);
        for(int i = 0, k = uOffsets[p]; k < uOffsets[p+1]; ++i, ++k) uIndices[k] = uIdx.first[i];
      }
      return section->setCustomAtlas(rOffsets, rIndices, uOffsets, uIndices);
    }
    template<typename Section_>
    const typename Section_::value_type *restrictClosure(const Obj<Section_>& section, const int tag, const int p) {
      const int *offsets, *indices;

      section->getCustomRestrictAtlas(tag, &offsets, &indices);
      const int size = offsets[p+1] - offsets[p];
      return this->restrictClosure(section, tag, p, section->getRawArray(size), offsets, indices);
    }
    template<typename Section_>
    const typename Section_::value_type *restrictClosure(const Obj<Section_>& section, const int tag, const int p, typename Section_::value_type  *values, const int valuesSize) {
      const int *offsets, *indices;

      section->getCustomRestrictAtlas(tag, &offsets, &indices);
      const int size = offsets[p+1] - offsets[p];
      if (valuesSize < size) {throw ALE::Exception("Input array too small");}
      return this->restrictClosure(section, tag, p, values, offsets, indices);
    }
    template<typename Section_>
    const typename Section_::value_type *restrictClosure(const Obj<Section_>& section, const int tag, const int p, typename Section_::value_type  *values, const int offsets[], const int indices[]) {
      const typename Section_::value_type *array = section->restrictSpace();

      const int size = offsets[p+1] - offsets[p];
      for(int j = 0, k = offsets[p]; j < size; ++j, ++k) {
        values[j] = array[indices[k]];
      }
      return values;
    }
    template<typename Section_>
    void updateAdd(const Obj<Section_>& section, const int tag, const int p, const typename Section_::value_type values[]) {
      typename Section_::value_type *array = (typename Section_::value_type *) section->restrictSpace();
      const int *offsets, *indices;

      section->getCustomUpdateAtlas(tag, &offsets, &indices);
      const int size = offsets[p+1] - offsets[p];
      for(int j = 0, k = offsets[p]; j < size; ++j, ++k) {
        if (indices[k] < 0) continue;
        array[indices[k]] += values[j];
      }
    }
  public: // Allocation
    template<typename Section_>
    void allocate(const Obj<Section_>& section, const Obj<send_overlap_type>& sendOverlap = NULL) {
      bool doGhosts = !sendOverlap.isNull();

      this->_factory->orderPatch(section, this->getSieve(), sendOverlap);
      if (doGhosts) {
        if (this->_debug > 1) {std::cout << "Ordering patch for ghosts" << std::endl;}
        const typename Section_::chart_type& points = section->getChart();
        typename Section_::index_type::index_type offset = 0;

        for(typename Section_::chart_type::const_iterator point = points.begin(); point != points.end(); ++point) {
          const typename Section_::index_type& idx = section->getIndex(*point);

          offset = std::max(offset, idx.index + std::abs(idx.prefix));
        }
        this->_factory->orderPatch(section, this->getSieve(), NULL, offset);
        if (offset != section->sizeWithBC()) throw ALE::Exception("Inconsistent array sizes in section");
      }
      section->allocateStorage();
    }
    template<typename Section_>
    void reallocate(const Obj<Section_>& section) {
      if (section->getNewAtlas().isNull()) return;
      // Since copy() preserves offsets, we must reinitialize them before ordering
      const Obj<typename Section_::atlas_type>         atlas    = section->getAtlas();
      const Obj<typename Section_::atlas_type>&        newAtlas = section->getNewAtlas();
      const typename Section_::atlas_type::chart_type& chart    = newAtlas->getChart();
      const typename Section_::atlas_type::chart_type& oldChart = atlas->getChart();
      int                                              newSize  = 0;
      typename Section_::index_type                    defaultIdx(0, -1);

      for(typename Section_::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        defaultIdx.prefix = newAtlas->restrictPoint(*c_iter)[0].prefix;
        newAtlas->updatePoint(*c_iter, &defaultIdx);
        newSize += defaultIdx.prefix;
      }
      section->setAtlas(newAtlas);
      this->_factory->orderPatch(section, this->getSieve());
      // Copy over existing values
      typedef typename alloc_type::template rebind<typename Section_::value_type>::other value_alloc_type;
      value_alloc_type value_allocator;
      typename Section_::value_type                   *newArray = value_allocator.allocate(newSize);
      for(int i = 0; i < newSize; ++i) {value_allocator.construct(newArray+i, typename Section_::value_type());}
      ///typename Section_::value_type                   *newArray = new typename Section_::value_type[newSize];
      const typename Section_::value_type             *array    = section->restrictSpace();

      for(typename Section_::atlas_type::chart_type::const_iterator c_iter = oldChart.begin(); c_iter != oldChart.end(); ++c_iter) {
        const int& dim       = section->getFiberDimension(*c_iter);
        const int& offset    = atlas->restrictPoint(*c_iter)->index;
        const int& newOffset = newAtlas->restrictPoint(*c_iter)->index;

        for(int i = 0; i < dim; ++i) {
          newArray[newOffset+i] = array[offset+i];
        }
      }
      section->replaceStorage(newArray);
    }
  public: // Overlap
    template<typename Sequence>
    void constructOverlap(const Obj<Sequence>& points, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
      point_type *sendBuf = new point_type[points->size()];
      int         size    = 0;
      for(typename Sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
        sendBuf[size++] = *l_iter;
      }
      int *sizes   = new int[this->commSize()];   // The number of points coming from each process
      int *offsets = new int[this->commSize()+1]; // Prefix sums for sizes
      int *oldOffs = new int[this->commSize()+1]; // Temporary storage
      point_type *remotePoints = NULL;            // The points from each process
      int        *remoteRanks  = NULL;            // The rank and number of overlap points of each process that overlaps another

      // Change to Allgather() for the correct binning algorithm
      MPI_Gather(&size, 1, MPI_INT, sizes, 1, MPI_INT, 0, this->comm());
      if (this->commRank() == 0) {
        offsets[0] = 0;
        for(int p = 1; p <= this->commSize(); p++) {
          offsets[p] = offsets[p-1] + sizes[p-1];
        }
        remotePoints = new point_type[offsets[this->commSize()]];
      }
      MPI_Gatherv(sendBuf, size, MPI_INT, remotePoints, sizes, offsets, MPI_INT, 0, this->comm());
      delete [] sendBuf;
      std::map<int, std::map<int, std::set<point_type> > > overlapInfo; // Maps (p,q) to their set of overlap points

      if (this->commRank() == 0) {
        for(int p = 0; p < this->commSize(); p++) {
          std::sort(&remotePoints[offsets[p]], &remotePoints[offsets[p+1]]);
        }
        for(int p = 0; p <= this->commSize(); p++) {
          oldOffs[p] = offsets[p];
        }
        for(int p = 0; p < this->commSize(); p++) {
          for(int q = p+1; q < this->commSize(); q++) {
            std::set_intersection(&remotePoints[oldOffs[p]], &remotePoints[oldOffs[p+1]],
                                  &remotePoints[oldOffs[q]], &remotePoints[oldOffs[q+1]],
                                  std::insert_iterator<std::set<point_type> >(overlapInfo[p][q], overlapInfo[p][q].begin()));
            overlapInfo[q][p] = overlapInfo[p][q];
          }
          sizes[p]     = overlapInfo[p].size()*2;
          offsets[p+1] = offsets[p] + sizes[p];
        }
        remoteRanks = new int[offsets[this->commSize()]];
        int       k = 0;
        for(int p = 0; p < this->commSize(); p++) {
          for(typename std::map<int, std::set<point_type> >::iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
            remoteRanks[k*2]   = r_iter->first;
            remoteRanks[k*2+1] = r_iter->second.size();
            k++;
          }
        }
      }
      int numOverlaps;                          // The number of processes overlapping this process
      MPI_Scatter(sizes, 1, MPI_INT, &numOverlaps, 1, MPI_INT, 0, this->comm());
      int *overlapRanks = new int[numOverlaps]; // The rank and overlap size for each overlapping process
      MPI_Scatterv(remoteRanks, sizes, offsets, MPI_INT, overlapRanks, numOverlaps, MPI_INT, 0, this->comm());
      point_type *sendPoints = NULL;            // The points to send to each process
      if (this->commRank() == 0) {
        for(int p = 0, k = 0; p < this->commSize(); p++) {
          sizes[p] = 0;
          for(int r = 0; r < (int) overlapInfo[p].size(); r++) {
            sizes[p] += remoteRanks[k*2+1];
            k++;
          }
          offsets[p+1] = offsets[p] + sizes[p];
        }
        sendPoints = new point_type[offsets[this->commSize()]];
        for(int p = 0, k = 0; p < this->commSize(); p++) {
          for(typename std::map<int, std::set<point_type> >::iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
            int rank = r_iter->first;
            for(typename std::set<point_type>::iterator p_iter = (overlapInfo[p][rank]).begin(); p_iter != (overlapInfo[p][rank]).end(); ++p_iter) {
              sendPoints[k++] = *p_iter;
            }
          }
        }
      }
      int numOverlapPoints = 0;
      for(int r = 0; r < numOverlaps/2; r++) {
        numOverlapPoints += overlapRanks[r*2+1];
      }
      point_type *overlapPoints = new point_type[numOverlapPoints];
      MPI_Scatterv(sendPoints, sizes, offsets, MPI_INT, overlapPoints, numOverlapPoints, MPI_INT, 0, this->comm());

      for(int r = 0, k = 0; r < numOverlaps/2; r++) {
        int rank = overlapRanks[r*2];

        for(int p = 0; p < overlapRanks[r*2+1]; p++) {
          point_type point = overlapPoints[k++];

          sendOverlap->addArrow(point, rank, point);
          recvOverlap->addArrow(rank, point, point);
        }
      }

      delete [] overlapPoints;
      delete [] overlapRanks;
      delete [] sizes;
      delete [] offsets;
      delete [] oldOffs;
      if (this->commRank() == 0) {
        delete [] remoteRanks;
        delete [] remotePoints;
        delete [] sendPoints;
      }
    }
    void constructOverlap() {
      if (this->_calculatedOverlap) return;
      this->constructOverlap(this->getSieve()->base(), this->getSendOverlap(), this->getRecvOverlap());
      this->constructOverlap(this->getSieve()->cap(),  this->getSendOverlap(), this->getRecvOverlap());
      if (this->debug()) {
        this->_sendOverlap->view("Send overlap");
        this->_recvOverlap->view("Receive overlap");
      }
      this->_calculatedOverlap = true;
    }
  };
  class BoundaryCondition : public ALE::ParallelObject {
  public:
    typedef double (*function_type)(const PetscReal []);
    typedef double (*integrator_type)(const PetscReal [], const PetscReal [], const int, function_type);
  protected:
    std::string     _labelName;
    int             _marker;
    function_type   _func;
    integrator_type _integrator;
  public:
    BoundaryCondition(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {};
    ~BoundaryCondition() {};
  public:
    const std::string& getLabelName() const {return this->_labelName;};
    void setLabelName(const std::string& name) {this->_labelName = name;};
    int getMarker() const {return this->_marker;};
    void setMarker(const int marker) {this->_marker = marker;};
    function_type getFunction() const {return this->_func;};
    void setFunction(function_type func) {this->_func = func;};
    integrator_type getDualIntegrator() const {return this->_integrator;};
    void setDualIntegrator(integrator_type integrator) {this->_integrator = integrator;};
  public:
    PetscReal evaluate(const PetscReal coords[]) const {return this->_func(coords);};
    PetscReal integrateDual(const PetscReal v0[], const PetscReal J[], const int dualIndex) const {return this->_integrator(v0, J, dualIndex, this->_func);};
  };
  class Discretization : public ALE::ParallelObject {
    typedef std::map<std::string, Obj<BoundaryCondition> > boundaryConditions_type;
  protected:
    boundaryConditions_type _boundaryConditions;
    Obj<BoundaryCondition> _exactSolution;
    std::map<int,int> _dim2dof;
    std::map<int,int> _dim2class;
    int           _quadSize;
    const PetscReal *_points;
    const PetscReal *_weights;
    int           _basisSize;
    const PetscReal *_basis;
    const PetscReal *_basisDer;
    const int    *_indices;
    std::map<int, const int *> _exclusionIndices;
  public:
    Discretization(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _quadSize(0), _points(NULL), _weights(NULL), _basisSize(0), _basis(NULL), _basisDer(NULL), _indices(NULL) {};
    virtual ~Discretization() {
      if (this->_indices) {delete [] this->_indices;}
      for(std::map<int, const int *>::iterator i_iter = _exclusionIndices.begin(); i_iter != _exclusionIndices.end(); ++i_iter) {
        delete [] i_iter->second;
      }
    };
  public:
    bool hasBoundaryCondition() {return (this->_boundaryConditions.find("default") != this->_boundaryConditions.end());};
    const Obj<BoundaryCondition>& getBoundaryCondition() {return this->getBoundaryCondition("default");};
    void setBoundaryCondition(const Obj<BoundaryCondition>& boundaryCondition) {this->setBoundaryCondition("default", boundaryCondition);};
    const Obj<BoundaryCondition>& getBoundaryCondition(const std::string& name) {return this->_boundaryConditions[name];};
    void setBoundaryCondition(const std::string& name, const Obj<BoundaryCondition>& boundaryCondition) {this->_boundaryConditions[name] = boundaryCondition;};
    Obj<std::set<std::string> > getBoundaryConditions() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(boundaryConditions_type::const_iterator d_iter = this->_boundaryConditions.begin(); d_iter != this->_boundaryConditions.end(); ++d_iter) {
        names->insert(d_iter->first);
      }
      return names;
    };
    const Obj<BoundaryCondition>& getExactSolution() {return this->_exactSolution;};
    void setExactSolution(const Obj<BoundaryCondition>& exactSolution) {this->_exactSolution = exactSolution;};
    int           getQuadratureSize() {return this->_quadSize;};
    void          setQuadratureSize(const int size) {this->_quadSize = size;};
    const PetscReal *getQuadraturePoints() {return this->_points;};
    void          setQuadraturePoints(const PetscReal *points) {this->_points = points;};
    const PetscReal *getQuadratureWeights() {return this->_weights;};
    void          setQuadratureWeights(const PetscReal *weights) {this->_weights = weights;};
    int           getBasisSize() {return this->_basisSize;};
    void          setBasisSize(const int size) {this->_basisSize = size;};
    const PetscReal *getBasis() {return this->_basis;};
    void          setBasis(const PetscReal *basis) {this->_basis = basis;};
    const PetscReal *getBasisDerivatives() {return this->_basisDer;};
    void          setBasisDerivatives(const PetscReal *basisDer) {this->_basisDer = basisDer;};
    int  getNumDof(const int dim) {return this->_dim2dof[dim];};
    void setNumDof(const int dim, const int numDof) {this->_dim2dof[dim] = numDof;};
    int  getDofClass(const int dim) {return this->_dim2class[dim];};
    void setDofClass(const int dim, const int dofClass) {this->_dim2class[dim] = dofClass;};
  public:
    const int *getIndices() {return this->_indices;};
    const int *getIndices(const int marker) {
      if (!marker) return this->getIndices();
      return this->_exclusionIndices[marker];
    };
    void       setIndices(const int *indices) {this->_indices = indices;};
    void       setIndices(const int *indices, const int marker) {
      if (!marker) this->_indices = indices;
      this->_exclusionIndices[marker] = indices;
    };
    template<typename Bundle>
    int sizeV(Bundle& mesh) {
      typedef typename ISieveVisitor::PointRetriever<typename Bundle::sieve_type> Visitor;
      Visitor pV((int) pow((double) mesh.getSieve()->getMaxConeSize(), mesh.depth())+1, true);
      ISieveTraversal<typename Bundle::sieve_type>::orientedClosure(*mesh.getSieve(), *mesh.heightStratum(0)->begin(), pV);
      const typename Visitor::point_type *oPoints = pV.getPoints();
      const int                           oSize   = pV.getSize();
      int                                 size    = 0;

      for(int cl = 0; cl < oSize; ++cl) {
        size += this->_dim2dof[mesh.depth(oPoints[cl])];
      }
      return size;
    }
    template<typename Bundle>
    int size(const Obj<Bundle>& mesh) {
      const Obj<typename Bundle::label_sequence>& cells   = mesh->heightStratum(0);
      const Obj<typename Bundle::coneArray>       closure = ALE::SieveAlg<Bundle>::closure(mesh, *cells->begin());
      const typename Bundle::coneArray::iterator  end     = closure->end();
      int                                         size    = 0;

      for(typename Bundle::coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
        size += this->_dim2dof[mesh->depth(*cl_iter)];
      }
      return size;
    }
  };
}

namespace ALE {
  template<typename Sieve_,
           typename RealSection_  = Section<typename Sieve_::point_type, double>,
           typename IntSection_   = Section<typename Sieve_::point_type, int>,
           typename Label_        = LabelSifter<int, typename Sieve_::point_type>,
           typename ArrowSection_ = UniformSection<MinimalArrow<typename Sieve_::point_type, typename Sieve_::point_type>, int> >
  class IBundle : public ALE::ParallelObject {
  public:
    typedef Sieve_                                                    sieve_type;
    typedef RealSection_                                              real_section_type;
    typedef IntSection_                                               int_section_type;
    typedef ArrowSection_                                             arrow_section_type;
    typedef IBundle<Sieve_,RealSection_,IntSection_,Label_,ArrowSection_> this_type;
    typedef typename sieve_type::point_type                           point_type;
    typedef malloc_allocator<point_type>                              alloc_type;
    typedef Label_                                                    label_type;
    typedef typename std::map<const std::string, Obj<label_type> >    labels_type;
    typedef typename label_type::supportSequence                      label_sequence;
    typedef std::map<std::string, Obj<arrow_section_type> >           arrow_sections_type;
    typedef std::map<std::string, Obj<real_section_type> >            real_sections_type;
    typedef std::map<std::string, Obj<int_section_type> >             int_sections_type;
    typedef ALE::Point                                                index_type;
    typedef std::pair<index_type, int>                                oIndex_type;
    typedef std::vector<oIndex_type>                                  oIndexArray;
    typedef std::pair<int *, int>                                     indices_type;
    typedef NumberingFactory<this_type>                               MeshNumberingFactory;
    typedef typename ALE::Partitioner<>::part_type                    rank_type;
#define USE_NEW_OVERLAP
#ifdef USE_NEW_OVERLAP
    typedef typename PETSc::SendOverlap<point_type,rank_type>         send_overlap_type;
    typedef typename PETSc::RecvOverlap<point_type,rank_type>         recv_overlap_type;
#else
    typedef typename ALE::Sifter<point_type,rank_type,point_type>     send_overlap_type;
    typedef typename ALE::Sifter<rank_type,point_type,point_type>     recv_overlap_type;
#endif
    typedef typename MeshNumberingFactory::numbering_type             numbering_type;
    typedef typename MeshNumberingFactory::order_type                 order_type;
    typedef std::map<point_type, point_type>                          renumbering_type;
    // These should go away
    typedef typename ALE::SieveAlg<this_type>                         sieve_alg_type;
    typedef typename sieve_alg_type::coneArray                        coneArray;
    typedef typename sieve_alg_type::orientedConeArray                oConeArray;
    typedef typename sieve_alg_type::supportArray                     supportArray;
  public:
    class LabelVisitor {
    protected:
      label_type& label;
      int         defaultValue;
      int         value;
    public:
      LabelVisitor(label_type& l, const int defValue) : label(l), defaultValue(defValue), value(defValue) {};
      int getLabelValue(const point_type& point) const {
        const Obj<typename label_type::coneSequence>& cone = this->label.cone(point);

        if (cone->size() == 0) return this->defaultValue;
        return *cone->begin();
      };
      void setLabelValue(const point_type& point, const int value) {
        this->label.setCone(value, point);
      };
      int getValue() const {return this->value;};
    };
    class MaxConeVisitor : public LabelVisitor {
    public:
      MaxConeVisitor(label_type& l, const int defValue) : LabelVisitor(l, defValue) {};
      void visitPoint(const typename sieve_type::point_type& point) {};
      void visitArrow(const typename sieve_type::arrow_type& arrow) {
        this->value = std::max(this->value, this->getLabelValue(arrow.source));
      };
    };
    class MaxSupportVisitor : public LabelVisitor {
    public:
      MaxSupportVisitor(label_type& l, const int defValue) : LabelVisitor(l, defValue) {};
      void visitPoint(const typename sieve_type::point_type& point) {};
      void visitArrow(const typename sieve_type::arrow_type& arrow) {
        this->value = std::max(this->value, this->getLabelValue(arrow.target));
      };
    };
    class BinaryStratifyVisitor {
    protected:
      label_type& height;
      label_type& depth;
      bool        isLeaf;
    public:
      BinaryStratifyVisitor(label_type& h, label_type& d, bool isLeaf) : height(h), depth(d), isLeaf(isLeaf) {};
      void visitPoint(const typename sieve_type::point_type& point) {
        if (isLeaf) {
          height.setCone(0, point);
          depth.setCone(1, point);
        } else {
          height.setCone(1, point);
          depth.setCone(0, point);
        }
      };
    };
    class HeightVisitor {
    protected:
      const sieve_type& sieve;
      label_type&       height;
      int               maxHeight;
      std::set<typename sieve_type::point_type> modifiedPoints;
    public:
      HeightVisitor(const sieve_type& s, label_type& h) : sieve(s), height(h), maxHeight(0) {};
      void visitPoint(const typename sieve_type::point_type& point) {
        MaxSupportVisitor v(height, -1);

        // Compute the max height of the points in the support of p, and add 1
        this->sieve.support(point, v);
        const int h0 = v.getLabelValue(point);
        const int h1 = v.getValue() + 1;

        if(h1 != h0) {
          v.setLabelValue(point, h1);
          if (h1 > this->maxHeight) this->maxHeight = h1;
          this->modifiedPoints.insert(point);
        }
      };
      void visitArrow(const typename sieve_type::arrow_type& arrow) {
        this->visitPoint(arrow.source);
      };
      int getMaxHeight() const {return this->maxHeight;};
      bool isModified() const {return this->modifiedPoints.size() > 0;};
      const std::set<typename sieve_type::point_type>& getModifiedPoints() const {return this->modifiedPoints;};
      void clear() {this->modifiedPoints.clear();};
    };
    class DepthVisitor {
    public:
      typedef typename sieve_type::point_type point_type;
    protected:
      const sieve_type& sieve;
      label_type&       depth;
      int               maxDepth;
      const point_type  limitPoint;
      std::set<point_type> modifiedPoints;
    public:
      DepthVisitor(const sieve_type& s, label_type& d) : sieve(s), depth(d), maxDepth(0), limitPoint(sieve.getChart().max()+1) {};
      DepthVisitor(const sieve_type& s, const point_type& limit, label_type& d) : sieve(s), depth(d), maxDepth(-1), limitPoint(limit) {};
      void visitPoint(const point_type& point) {
        if (point >= this->limitPoint) return;
        MaxConeVisitor v(depth, -1);

        // Compute the max height of the points in the support of p, and add 1
        this->sieve.cone(point, v);
        const int d0 = v.getLabelValue(point);
        const int d1 = v.getValue() + 1;

        if(d1 != d0) {
          v.setLabelValue(point, d1);
          if (d1 > this->maxDepth) this->maxDepth = d1;
          this->modifiedPoints.insert(point);
        }
      };
      void visitArrow(const typename sieve_type::arrow_type& arrow) {
        this->visitPoint(arrow.target);
      };
      int getMaxDepth() const {return this->maxDepth;};
      bool isModified() const {return this->modifiedPoints.size() > 0;};
      const std::set<typename sieve_type::point_type>& getModifiedPoints() const {return this->modifiedPoints;};
      void clear() {this->modifiedPoints.clear();};
    };
  protected:
    Obj<sieve_type>       _sieve;
    labels_type           _labels;
    int                   _maxHeight;
    int                   _maxDepth;
    arrow_sections_type   _arrowSections;
    real_sections_type    _realSections;
    int_sections_type     _intSections;
    Obj<oIndexArray>      _indexArray;
    Obj<MeshNumberingFactory> _factory;
    bool                   _calculatedOverlap;
    Obj<send_overlap_type> _sendOverlap;
    Obj<recv_overlap_type> _recvOverlap;
    Obj<send_overlap_type> _distSendOverlap;
    Obj<recv_overlap_type> _distRecvOverlap;
    renumbering_type       _renumbering;
    // Work space
    Obj<std::set<point_type> > _modifiedPoints;
  public:
    IBundle(MPI_Comm comm, int debug = 0) : ALE::ParallelObject(comm, debug), _maxHeight(-1), _maxDepth(-1) {
      this->_indexArray        = new oIndexArray();
      this->_modifiedPoints    = new std::set<point_type>();
      this->_factory           = MeshNumberingFactory::singleton(this->comm(), this->debug());
      this->_calculatedOverlap = false;
      this->_sendOverlap       = new send_overlap_type(this->comm(), this->debug());
      this->_recvOverlap       = new recv_overlap_type(this->comm(), this->debug());
    };
    IBundle(const Obj<sieve_type>& sieve) : ALE::ParallelObject(sieve->comm(), sieve->debug()), _sieve(sieve), _maxHeight(-1), _maxDepth(-1) {
      this->_indexArray        = new oIndexArray();
      this->_modifiedPoints    = new std::set<point_type>();
      this->_factory           = MeshNumberingFactory::singleton(this->comm(), this->debug());
      this->_calculatedOverlap = false;
      this->_sendOverlap       = new send_overlap_type(this->comm(), this->debug());
      this->_recvOverlap       = new recv_overlap_type(this->comm(), this->debug());
    };
    virtual ~IBundle() {};
  public: // Verifiers
    bool hasLabel(const std::string& name) {
      if (this->_labels.find(name) != this->_labels.end()) {
        return true;
      }
      return false;
    };
    void checkLabel(const std::string& name) {
      if (!this->hasLabel(name)) {
        ostringstream msg;
        msg << "Invalid label name: " << name << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
  public: // Accessors
    const Obj<sieve_type>& getSieve() const {return this->_sieve;};
    void setSieve(const Obj<sieve_type>& sieve) {this->_sieve = sieve;};
    bool hasArrowSection(const std::string& name) const {
      return this->_arrowSections.find(name) != this->_arrowSections.end();
    };
    const Obj<arrow_section_type>& getArrowSection(const std::string& name) {
      if (!this->hasArrowSection(name)) {
        Obj<arrow_section_type> section = new arrow_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new arrow section: " << name << std::endl;}
        this->_arrowSections[name] = section;
      }
      return this->_arrowSections[name];
    };
    void setArrowSection(const std::string& name, const Obj<arrow_section_type>& section) {
      this->_arrowSections[name] = section;
    };
    Obj<std::set<std::string> > getArrowSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename arrow_sections_type::const_iterator s_iter = this->_arrowSections.begin(); s_iter != this->_arrowSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasRealSection(const std::string& name) const {
      return this->_realSections.find(name) != this->_realSections.end();
    };
    const Obj<real_section_type>& getRealSection(const std::string& name) {
      if (!this->hasRealSection(name)) {
        Obj<real_section_type> section = new real_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new real section: " << name << std::endl;}
        this->_realSections[name] = section;
      }
      return this->_realSections[name];
    };
    void setRealSection(const std::string& name, const Obj<real_section_type>& section) {
      this->_realSections[name] = section;
    };
    Obj<std::set<std::string> > getRealSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename real_sections_type::const_iterator s_iter = this->_realSections.begin(); s_iter != this->_realSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasIntSection(const std::string& name) const {
      return this->_intSections.find(name) != this->_intSections.end();
    };
    const Obj<int_section_type>& getIntSection(const std::string& name) {
      if (!this->hasIntSection(name)) {
        Obj<int_section_type> section = new int_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new int section: " << name << std::endl;}
        this->_intSections[name] = section;
      }
      return this->_intSections[name];
    };
    void setIntSection(const std::string& name, const Obj<int_section_type>& section) {
      this->_intSections[name] = section;
    };
    Obj<std::set<std::string> > getIntSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename int_sections_type::const_iterator s_iter = this->_intSections.begin(); s_iter != this->_intSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    const Obj<MeshNumberingFactory>& getFactory() const {return this->_factory;};
    renumbering_type& getRenumbering() {return this->_renumbering;};
  public: // Labels
    int getValue (const Obj<label_type>& label, const point_type& point, const int defValue = 0) {
      const Obj<typename label_type::coneSequence>& cone = label->cone(point);

      if (cone->size() == 0) return defValue;
      return *cone->begin();
    };
    void setValue(const Obj<label_type>& label, const point_type& point, const int value) {
      label->setCone(value, point);
    };
    void addValue(const Obj<label_type>& label, const point_type& point, const int value) {
      label->addCone(value, point);
    };
    template<typename InputPoints>
    int getMaxValue (const Obj<label_type>& label, const Obj<InputPoints>& points, const int defValue = 0) {
      int maxValue = defValue;

      for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        maxValue = std::max(maxValue, this->getValue(label, *p_iter, defValue));
      }
      return maxValue;
    }
    const Obj<label_type>& createLabel(const std::string& name) {
      this->_labels[name] = new label_type(this->comm(), this->debug());
      return this->_labels[name];
    };
    const Obj<label_type>& getLabel(const std::string& name) {
      this->checkLabel(name);
      return this->_labels[name];
    };
    void setLabel(const std::string& name, const Obj<label_type>& label) {
      this->_labels[name] = label;
    };
    const labels_type& getLabels() {
      return this->_labels;
    };
    virtual const Obj<label_sequence>& getLabelStratum(const std::string& name, int value) {
      this->checkLabel(name);
      return this->_labels[name]->support(value);
    };
  public: // Stratification
    void computeBinaryStratification() {
      const Obj<label_type>& height = this->createLabel("height");
      const Obj<label_type>& depth  = this->createLabel("depth");
      BinaryStratifyVisitor  l(*height, *depth, true);
      BinaryStratifyVisitor  r(*height, *depth, false);

      this->_sieve->leaves(l);
      this->_sieve->roots(r);
      if (this->_sieve->numRoots()) {
        this->setHeight(1);
        this->setDepth(1);
      } else {
        this->setHeight(0);
        this->setDepth(0);
      }
    };
    void computeHeights() {
      const Obj<label_type>& label = this->createLabel(std::string("height"));
      HeightVisitor          h(*this->_sieve, *label);

#ifdef IMESH_NEW_LABELS
      label->setChart(this->getSieve()->getChart());
      for(point_type p = label->getChart().min(); p < label->getChart().max(); ++p) {label->setConeSize(p, 1);}
      if (label->getChart().size()) {label->setSupportSize(0, label->getChart().size());}
      label->allocate();
      for(point_type p = label->getChart().min(); p < label->getChart().max(); ++p) {label->setCone(-1, p);}
#endif
      this->_sieve->leaves(h);
      while(h.isModified()) {
        // FIX: Avoid the copy here somehow by fixing the traversal
        std::vector<point_type> modifiedPoints(h.getModifiedPoints().begin(), h.getModifiedPoints().end());

        h.clear();
        this->_sieve->cone(modifiedPoints, h);
      }
#ifdef IMESH_NEW_LABELS
      // Recalculate supportOffsets and populate support
      label->recalculateLabel();
#endif
      this->_maxHeight = h.getMaxHeight();
    };
    virtual int height() const {return this->_maxHeight;};
    virtual void setHeight(const int height) {this->_maxHeight = height;};
    virtual int height(const point_type& point) {
      return this->getValue(this->_labels["height"], point, -1);
    };
    virtual void setHeight(const point_type& point, const int height) {
      return this->setValue(this->_labels["height"], point, height);
    };
    virtual const Obj<label_sequence>& heightStratum(int height) {
      return this->getLabelStratum("height", height);
    };
    void computeDepths() {
      const Obj<label_type>& label = this->createLabel(std::string("depth"));
      DepthVisitor           d(*this->_sieve, *label);

#ifdef IMESH_NEW_LABELS
      label->setChart(this->getSieve()->getChart());
      for(point_type p = label->getChart().min(); p < label->getChart().max(); ++p) {label->setConeSize(p, 1);}
      if (label->getChart().size()) {label->setSupportSize(0, label->getChart().size());}
      label->allocate();
      for(point_type p = label->getChart().min(); p < label->getChart().max(); ++p) {label->setCone(-1, p);}
#endif
      this->_sieve->roots(d);
      while(d.isModified()) {
        // FIX: Avoid the copy here somehow by fixing the traversal
        std::vector<point_type> modifiedPoints(d.getModifiedPoints().begin(), d.getModifiedPoints().end());

        d.clear();
        this->_sieve->support(modifiedPoints, d);
      }
#ifdef IMESH_NEW_LABELS
      // Recalculate supportOffsets and populate support
      label->recalculateLabel();
#endif
      this->_maxDepth = d.getMaxDepth();
    };
    virtual int depth() const {return this->_maxDepth;};
    virtual void setDepth(const int depth) {this->_maxDepth = depth;};
    virtual int depth(const point_type& point) {
      return this->getValue(this->_labels["depth"], point, -1);
    };
    virtual void setDepth(const point_type& point, const int depth) {
      return this->setValue(this->_labels["depth"], point, depth);
    };
    virtual const Obj<label_sequence>& depthStratum(int depth) {
      return this->getLabelStratum("depth", depth);
    };
    #undef __FUNCT__
    #define __FUNCT__ "stratify"
    void stratify() {
      ALE::LogEvent event = ALE::LogEventRegister(__FUNCT__);
      ALE::LogEventBegin(event);
      if (this->_sieve->numRoots() + this->_sieve->numLeaves() == (int) this->_sieve->getChart().size()) {
        this->computeBinaryStratification();
      } else {
        this->computeHeights();
        this->computeDepths();
      }
      ALE::LogEventEnd(event);
    };
  protected:
    template<typename Value>
    static bool lt1(const Value& a, const Value& b) {
      return a.first < b.first;
    }
  public: // Allocation
    template<typename Section_>
    void reallocate(const Obj<Section_>& section) {
      if (!section->hasNewPoints()) return;
      typename Section_::chart_type newChart(std::min(std::min_element(section->getNewPoints().begin(), section->getNewPoints().end(), lt1<typename Section_::newpoint_type>)->first, section->getChart().min()),
                                             std::max(std::max_element(section->getNewPoints().begin(), section->getNewPoints().end(), lt1<typename Section_::newpoint_type>)->first, section->getChart().max()-1)+1);
      section->reallocatePoint(newChart);
    }
  };
#ifdef IMESH_NEW_LABELS
  template<typename Label_ = IFSieve<int> >
#else
  template<typename IndexType, typename ScalarType, typename Label_ = LabelSifter<IndexType, IndexType> >
#endif
  class IMesh : public IBundle<IFSieve<IndexType>, IGeneralSection<IndexType, ScalarType>, IGeneralSection<IndexType, IndexType>,  Label_> {
  public:
    typedef IBundle<IFSieve<IndexType>, IGeneralSection<IndexType, ScalarType>, IGeneralSection<IndexType, IndexType>, Label_> base_type;
    typedef typename base_type::sieve_type            sieve_type;
    typedef typename sieve_type::point_type           point_type;
    typedef typename base_type::alloc_type            alloc_type;
    typedef typename base_type::label_type            label_type;
    typedef typename base_type::labels_type           labels_type;
    typedef typename base_type::label_sequence        label_sequence;
    typedef typename base_type::real_section_type     real_section_type;
    typedef typename base_type::int_section_type      int_section_type;
    typedef typename base_type::numbering_type        numbering_type;
    typedef typename base_type::order_type            order_type;
    typedef typename base_type::send_overlap_type     send_overlap_type;
    typedef typename base_type::recv_overlap_type     recv_overlap_type;
    typedef std::set<std::string>                     names_type;
    typedef std::vector<typename PETSc::Point<3> >    holes_type;
    typedef std::map<std::string, Obj<Discretization> > discretizations_type;
  protected:
    int _dim;
    bool                   _calculatedOverlap;
    Obj<send_overlap_type> _sendOverlap;
    Obj<recv_overlap_type> _recvOverlap;
    std::map<int,double>   _periodicity;
    holes_type             _holes;
    discretizations_type   _discretizations;
    int                    _maxDof;
  public:
    IMesh(MPI_Comm comm, int dim, int debug = 0) : base_type(comm, debug), _dim(dim) {
      this->_calculatedOverlap = false;
      this->_sendOverlap       = new send_overlap_type(comm, debug);
      this->_recvOverlap       = new recv_overlap_type(comm, debug);
      this->_maxDof            = -1;
    };
  public: // Accessors
    int getDimension() const {return this->_dim;};
    void setDimension(const int dim) {this->_dim = dim;};
    bool getCalculatedOverlap() const {return this->_calculatedOverlap;};
    void setCalculatedOverlap(const bool calc) {this->_calculatedOverlap = calc;};
    const Obj<send_overlap_type>& getSendOverlap() const {return this->_sendOverlap;};
    void setSendOverlap(const Obj<send_overlap_type>& overlap) {this->_sendOverlap = overlap;};
    const Obj<recv_overlap_type>& getRecvOverlap() const {return this->_recvOverlap;};
    void setRecvOverlap(const Obj<recv_overlap_type>& overlap) {this->_recvOverlap = overlap;};
    bool getPeriodicity(const int d) {return this->_periodicity[d];};
    void setPeriodicity(const int d, const double length) {this->_periodicity[d] = length;};
    const holes_type& getHoles() const {return this->_holes;};
    void addHole(const double hole[]) {
      this->_holes.push_back(hole);
    };
    void copyHoles(const Obj<IMesh>& m) {
      const holes_type& holes = m->getHoles();

      for(holes_type::const_iterator h_iter = holes.begin(); h_iter != holes.end(); ++h_iter) {
        this->_holes.push_back(*h_iter);
      }
    };
    const Obj<Discretization>& getDiscretization() {return this->getDiscretization("default");};
    const Obj<Discretization>& getDiscretization(const std::string& name) {return this->_discretizations[name];};
    void setDiscretization(const Obj<Discretization>& disc) {this->setDiscretization("default", disc);};
    void setDiscretization(const std::string& name, const Obj<Discretization>& disc) {this->_discretizations[name] = disc;};
    Obj<names_type> getDiscretizations() const {
      Obj<names_type> names = names_type();

      for(discretizations_type::const_iterator d_iter = this->_discretizations.begin(); d_iter != this->_discretizations.end(); ++d_iter) {
        names->insert(d_iter->first);
      }
      return names;
    };
    int getMaxDof() const {return this->_maxDof;};
    void setMaxDof(const int maxDof) {this->_maxDof = maxDof;};
  public: // Sizes
    template<typename Section>
    int size(const Obj<Section>& section, const point_type& p) {
      typedef ISieveVisitor::SizeVisitor<sieve_type,Section>                        size_visitor_type;
      typedef ISieveVisitor::TransitiveClosureVisitor<sieve_type,size_visitor_type> closure_visitor_type;
      size_visitor_type    sV(*section);
      closure_visitor_type cV(*this->getSieve(), sV);

      this->getSieve()->cone(p, cV);
      if (!sV.getSize()) sV.visitPoint(p);
      return sV.getSize();
    }
    template<typename Section>
    int sizeWithBC(const Obj<Section>& section, const point_type& p) {
      typedef ISieveVisitor::SizeWithBCVisitor<sieve_type,Section>                  size_visitor_type;
      typedef ISieveVisitor::TransitiveClosureVisitor<sieve_type,size_visitor_type> closure_visitor_type;
      size_visitor_type    sV(*section);
      closure_visitor_type cV(*this->getSieve(), sV);

      this->getSieve()->cone(p, cV);
      if (!sV.getSize()) sV.visitPoint(p);
      return sV.getSize();
    }
    int sizeWithBC(PetscSection section, const point_type& p) {
      typedef ISieveVisitor::SizeWithBCVisitor<sieve_type,PetscSection>             size_visitor_type;
      typedef ISieveVisitor::TransitiveClosureVisitor<sieve_type,size_visitor_type> closure_visitor_type;
      size_visitor_type    sV(section);
      closure_visitor_type cV(*this->getSieve(), sV);

      this->getSieve()->cone(p, cV);
      if (!sV.getSize()) sV.visitPoint(p);
      return sV.getSize();
    }
    void sizeWithBC(PetscSection section, const point_type& p, PetscInt fieldSize[]) {
      typedef ISieveVisitor::SizeWithBCVisitor<sieve_type,PetscSection>             size_visitor_type;
      typedef ISieveVisitor::TransitiveClosureVisitor<sieve_type,size_visitor_type> closure_visitor_type;
      size_visitor_type    sV(section, fieldSize);
      closure_visitor_type cV(*this->getSieve(), sV);

      this->getSieve()->cone(p, cV);
      if (!sV.getSize()) sV.visitPoint(p);
    }
    template<typename Section>
    void allocate(const Obj<Section>& section) {
      section->allocatePoint();
    }
  public: // Restrict/Update closures
    template<typename Sieve, typename Visitor>
    void closure1(const Sieve& sieve, const point_type& p, Visitor& v)
    {
      v.visitPoint(p, 0);
      // Cone is guarateed to be ordered correctly
      sieve.orientedCone(p, v);
    }
    // Return the values for the closure of this point
    template<typename Section>
    const typename Section::value_type *restrictClosure(const Obj<Section>& section, const point_type& p) {
      const int size = this->sizeWithBC(section, p);
      ISieveVisitor::RestrictVisitor<Section> rV(*section, size, section->getRawArray(size));

      if (this->depth() == 1) {
        closure1(*this->getSieve(), p, rV);
      } else {
        ISieveVisitor::PointRetriever<sieve_type,ISieveVisitor::RestrictVisitor<Section> > pV((int) pow((double) this->getSieve()->getMaxConeSize(), this->depth())+1, rV, true);

        ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), p, pV);
      }
      return rV.getValues();
    }
    template<typename Section>
    const typename Section::value_type *restrictClosure(const Obj<Section>& section, const point_type& p, typename Section::value_type *values, const int valuesSize) {
      const int size = this->sizeWithBC(section, p);
      if (valuesSize < size) {throw ALE::Exception("Input array to small for restrictClosure()");}
      ISieveVisitor::RestrictVisitor<Section> rV(*section, size, values);

      if (this->depth() == 1) {
        closure1(*this->getSieve(), p, rV);
      } else {
        ISieveVisitor::PointRetriever<sieve_type,ISieveVisitor::RestrictVisitor<Section> > pV((int) pow((double) this->getSieve()->getMaxConeSize(), this->depth())+1, rV, true);

        ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), p, pV);
      }
      return rV.getValues();
    }
    template<typename Visitor>
    void restrictClosure(const point_type& p, Visitor& v) {
      if (this->depth() == 1) {
        closure1(*this->getSieve(), p, v);
      } else {
        ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), p, v);
      }
    }
    // Replace the values for the closure of this point
    template<typename Section>
    void update(const Obj<Section>& section, const point_type& p, const typename Section::value_type *v) {
      ISieveVisitor::UpdateVisitor<Section> uV(*section, v);

      if (this->depth() == 1) {
        closure1(*this->getSieve(), p, uV);
      } else {
        ISieveVisitor::PointRetriever<sieve_type,ISieveVisitor::UpdateVisitor<Section> > pV((int) pow((double) this->getSieve()->getMaxConeSize(), this->depth())+1, uV, true);

        ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), p, pV);
      }
    }
    // Replace the values for the closure of this point, including points constrained by BC
    template<typename Section>
    void updateAll(const Obj<Section>& section, const point_type& p, const typename Section::value_type *v) {
      ISieveVisitor::UpdateAllVisitor<Section> uV(*section, v);

      if (this->depth() == 1) {
        closure1(*this->getSieve(), p, uV);
      } else {
        ISieveVisitor::PointRetriever<sieve_type,ISieveVisitor::UpdateAllVisitor<Section> > pV((int) pow((double) this->getSieve()->getMaxConeSize(), this->depth())+1, uV, true);

        ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), p, pV);
      }
    }
    // Augment the values for the closure of this point
    template<typename Section>
    void updateAdd(const Obj<Section>& section, const point_type& p, const typename Section::value_type *v) {
      ISieveVisitor::UpdateAddVisitor<Section> uV(*section, v);

      if (this->depth() == 1) {
        closure1(*this->getSieve(), p, uV);
      } else {
        ISieveVisitor::PointRetriever<sieve_type,ISieveVisitor::UpdateAddVisitor<Section> > pV((int) pow((double) this->getSieve()->getMaxConeSize(), this->depth())+1, uV, true);

        ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), p, pV);
      }
    }
    // Augment the values for the closure of this point
    template<typename Visitor>
    void updateClosure(const point_type& p, Visitor& v) {
      if (this->depth() == 1) {
        closure1(*this->getSieve(), p, v);
      } else {
        ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), p, v);
      }
    }
  public: // Overlap
    void constructOverlap() {
      if (!this->_calculatedOverlap && (this->commSize() > 1)) {throw ALE::Exception("Must calculate overlap during distribution");}
    };
  public: // Cell topology and geometry
    int getNumCellCorners(const point_type& p, const int depth = -1) const {
      const int d = depth < 0 ? this->depth() : depth;

      if (d == 1) {
        return this->_sieve->getConeSize(p);
      } else if (d <= 0) {
        return 0;
      }
      // Warning: this is slow
      ISieveVisitor::NConeRetriever<sieve_type> ncV(*this->_sieve, (int) pow((double) this->_sieve->getMaxConeSize(), this->depth()));
      ALE::ISieveTraversal<sieve_type>::orientedClosure(*this->_sieve, p, ncV);
      return ncV.getOrientedSize();
    };
    int getNumCellCorners() {
      return getNumCellCorners(*this->heightStratum(0)->begin());
    };
    void setupCoordinates(const Obj<real_section_type>& coordinates) {
      const Obj<label_sequence>& vertices = this->depthStratum(0);

      if (vertices->size() > 0) {
        coordinates->setChart(typename real_section_type::chart_type(*std::min_element(vertices->begin(), vertices->end()),
                                                                     *std::max_element(vertices->begin(), vertices->end())+1));
      } else {
        coordinates->setChart(typename real_section_type::chart_type(0, 0));
      }
    };
    // Find the cell in which this point lies (stupid algorithm)
    point_type locatePoint_Simplex_2D(const typename real_section_type::value_type point[]) {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const int                     embedDim    = 2;
      typename real_section_type::value_type v0[2], J[4], invJ[4], detJ;

      for(typename label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        //std::cout << "Checking cell " << *c_iter << std::endl;
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        double xi   = invJ[0*embedDim+0]*(point[0] - v0[0]) + invJ[0*embedDim+1]*(point[1] - v0[1]);
        double eta  = invJ[1*embedDim+0]*(point[0] - v0[0]) + invJ[1*embedDim+1]*(point[1] - v0[1]);

        if ((xi >= 0.0) && (eta >= 0.0) && (xi + eta <= 2.0)) {
          return *c_iter;
        }
      }
      {
        ostringstream msg;
        msg << "Could not locate point: (" << point[0] <<","<< point[1] << ")" << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
    point_type locatePoint_General_2D(const typename real_section_type::value_type p[]) {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const PetscInt                faces[8]    = {0, 1, 1, 2, 2, 3, 3, 0};

      for(typename label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        const PetscReal *coords    = this->restrictClosure(coordinates, *c_iter);
        PetscInt         crossings = 0;

        //std::cout << "Checking cell " << *c_iter << std::endl;
        for(PetscInt f = 0; f < 4; f++) {
          PetscReal x_i   = coords[faces[2*f+0]*2+0];
          PetscReal y_i   = coords[faces[2*f+0]*2+1];
          PetscReal x_j   = coords[faces[2*f+1]*2+0];
          PetscReal y_j   = coords[faces[2*f+1]*2+1];
          PetscReal slope = (y_j - y_i) / (x_j - x_i);
          bool      cond1 = (x_i <= p[0]) && (p[0] < x_j);
          bool      cond2 = (x_j <= p[0]) && (p[0] < x_i);
          bool      above = (p[1] < slope * (p[0] - x_i) + y_i);
          if ((cond1 || cond2)  && above) ++crossings;
        }
        if (crossings % 2) {return *c_iter;}
      }
      {
        ostringstream msg;
        msg << "Could not locate point: (" << p[0] <<","<< p[1] << ")" << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
    //   Assume a simplex and 3D
    point_type locatePoint_Simplex_3D(const typename real_section_type::value_type point[]) {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const int                     embedDim    = 3;
      typename real_section_type::value_type v0[3], J[9], invJ[9], detJ;

      for(typename label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        double xi   = invJ[0*embedDim+0]*(point[0] - v0[0]) + invJ[0*embedDim+1]*(point[1] - v0[1]) + invJ[0*embedDim+2]*(point[2] - v0[2]);
        double eta  = invJ[1*embedDim+0]*(point[0] - v0[0]) + invJ[1*embedDim+1]*(point[1] - v0[1]) + invJ[1*embedDim+2]*(point[2] - v0[2]);
        double zeta = invJ[2*embedDim+0]*(point[0] - v0[0]) + invJ[2*embedDim+1]*(point[1] - v0[1]) + invJ[2*embedDim+2]*(point[2] - v0[2]);

        if ((xi >= 0.0) && (eta >= 0.0) && (zeta >= 0.0) && (xi + eta + zeta <= 2.0)) {
          return *c_iter;
        }
      }
#if 0
      {
        ostringstream msg;
        msg << "Could not locate point: (" << point[0] <<","<< point[1] <<","<< point[2] << ")" << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
#else
      return -1;
#endif
    };
    point_type locatePoint_General_3D(const typename real_section_type::value_type p[]) {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const PetscInt                faces[24]   = {0, 1, 2, 3,  5, 4, 7, 6,  1, 0, 4, 5,
                                                   3, 2, 6, 7,  1, 5, 6, 2,  0, 3, 7, 4};

      for(typename label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        const PetscReal *coords = this->restrictClosure(coordinates, *c_iter);
        PetscBool        found  = PETSC_TRUE;

        //std::cout << "Checking cell " << *c_iter << std::endl;
        for(PetscInt f = 0; f < 6; f++) {
          /* Check the point is under plane */
          /*   Get face normal */
          PetscReal v_i[3]    = {coords[faces[f*4+3]*3+0]-coords[faces[f*4+0]*3+0],coords[faces[f*4+3]*3+1]-coords[faces[f*4+0]*3+1],coords[faces[f*4+3]*3+2]-coords[faces[f*4+0]*3+2]};
          PetscReal v_j[3]    = {coords[faces[f*4+1]*3+0]-coords[faces[f*4+0]*3+0],coords[faces[f*4+1]*3+1]-coords[faces[f*4+0]*3+1],coords[faces[f*4+1]*3+2]-coords[faces[f*4+0]*3+2]};
          PetscReal normal[3] = {v_i[1]*v_j[2] - v_i[2]*v_j[1], v_i[2]*v_j[0] - v_i[0]*v_j[2], v_i[0]*v_j[1] - v_i[1]*v_j[0]};
          PetscReal pp[3]     = {coords[faces[f*4+0]*3+0] - p[0],coords[faces[f*4+0]*3+1] - p[1],coords[faces[f*4+0]*3+2] - p[2]};
          PetscReal dot       = normal[0]*pp[0] + normal[1]*pp[1] + normal[2]*pp[2];
          /* Check that projected point is in face (2D location problem) */
          if (dot < 0.0) {
            found = PETSC_FALSE;
            break;
          }
        }
        if (found) {return *c_iter;}
      }
#if 0
      {
        ostringstream msg;
        msg << "Could not locate point: (" << p[0] <<","<< p[1] <<","<< p[2] << ")" << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
#else
      return -1;
#endif
    };
    point_type locatePoint(const typename real_section_type::value_type point[], point_type guess = -1) {
      //guess overrides this by saying that we already know the relation of this point to this mesh.  We will need to make it a more robust "guess" later for more than P1
      if (guess != -1) {
        return guess;
      } else if (this->_dim == 2) {
        const int e = *this->heightStratum(0)->begin();
        switch(this->getSieve()->getConeSize(e)) {
        case 3:
          return locatePoint_Simplex_2D(point);
        case 4:
          return locatePoint_General_2D(point);
        default:
          throw ALE::Exception("No point location for cone size");
        }
      } else if (this->_dim == 3) {
        const int e = *this->heightStratum(0)->begin();
        switch(this->getSieve()->getConeSize(e)) {
        case 4:
          return locatePoint_Simplex_3D(point);
        case 8:
          return locatePoint_General_3D(point);
        default:
          throw ALE::Exception("No point location for cone size");
        }
      } else {
        throw ALE::Exception("No point location for mesh dimension");
      }
    };
    void computeTriangleGeometry(const Obj<real_section_type>& coordinates, const point_type& e, typename real_section_type::value_type v0[], typename real_section_type::value_type J[], typename real_section_type::value_type invJ[], typename real_section_type::value_type& detJ) {
      const PetscReal *coords = this->restrictClosure(coordinates, e);
      const int        dim    = 2;
      typename real_section_type::value_type           invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        for(int d = 0; d < dim; d++) {
          for(int f = 0; f < dim; f++) {
            J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
          }
        }
        detJ = J[0]*J[3] - J[1]*J[2];
        if (detJ < 0.0) {
          const typename real_section_type::value_type  xLength = this->_periodicity[0];

          if (xLength != 0.0) {
            typename real_section_type::value_type v0x = coords[0*dim+0];

            if (v0x == 0.0) {
              v0x = v0[0] = xLength;
            }
            for(int f = 0; f < dim; f++) {
              const typename real_section_type::value_type px = coords[(f+1)*dim+0] == 0.0 ? xLength : coords[(f+1)*dim+0];

              J[0*dim+f] = 0.5*(px - v0x);
            }
          }
          detJ = J[0]*J[3] - J[1]*J[2];
        }
        PetscLogFlopsNoError(8.0 + 3.0);
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[1];
        invJ[2] = -invDet*J[2];
        invJ[3] =  invDet*J[0];
        PetscLogFlopsNoError(5.0);
      }
    };
    void computeRectangleGeometry(const Obj<real_section_type>& coordinates, const point_type& e, typename real_section_type::value_type v0[], typename real_section_type::value_type J[], typename real_section_type::value_type invJ[], typename real_section_type::value_type& detJ) {
      const PetscReal *coords = this->restrictClosure(coordinates, e);
      const int        dim    = 2;
      typename real_section_type::value_type invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        for(int d = 0; d < dim; d++) {
          for(int f = 0; f < dim; f++) {
            J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
          }
        }
        detJ = J[0]*J[3] - J[1]*J[2];
        PetscLogFlopsNoError(8.0 + 3.0);
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[1];
        invJ[2] = -invDet*J[2];
        invJ[3] =  invDet*J[0];
        PetscLogFlopsNoError(5.0);
      }
      detJ *= 2.0;
    };
    void computeTetrahedronGeometry(const Obj<real_section_type>& coordinates, const point_type& e, typename real_section_type::value_type v0[], typename real_section_type::value_type J[], typename real_section_type::value_type invJ[], typename real_section_type::value_type& detJ) {
      const PetscReal *coords = this->restrictClosure(coordinates, e);
      const int        dim    = 3;
      typename real_section_type::value_type           invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        for(int d = 0; d < dim; d++) {
          for(int f = 0; f < dim; f++) {
            J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
          }
        }
        // The minus sign is here since I orient the first face to get the outward normal
        detJ = -(J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
                 J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
                 J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
        PetscLogFlopsNoError(18.0 + 12.0);
      }
      if (invJ) {
        invDet  = -1.0/detJ;
        invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
        invJ[0*3+1] = invDet*(J[0*3+2]*J[2*3+1] - J[0*3+1]*J[2*3+2]);
        invJ[0*3+2] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
        invJ[1*3+0] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
        invJ[1*3+1] = invDet*(J[0*3+0]*J[2*3+2] - J[0*3+2]*J[2*3+0]);
        invJ[1*3+2] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
        invJ[2*3+0] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        invJ[2*3+1] = invDet*(J[0*3+1]*J[2*3+0] - J[0*3+0]*J[2*3+1]);
        invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
        PetscLogFlopsNoError(37.0);
      }
    };
    void computeHexahedronGeometry(const Obj<real_section_type>& coordinates, const point_type& e, typename real_section_type::value_type v0[], typename real_section_type::value_type J[], typename real_section_type::value_type invJ[], typename real_section_type::value_type& detJ) {
      const PetscReal *coords = this->restrictClosure(coordinates, e);
      const int        dim    = 3;
      typename real_section_type::value_type invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        for(int d = 0; d < dim; d++) {
          J[d*dim+0] = 0.5*(coords[(0+1)*dim+d] - coords[0*dim+d]);
          J[d*dim+1] = 0.5*(coords[(1+1)*dim+d] - coords[0*dim+d]);
          J[d*dim+2] = 0.5*(coords[(3+1)*dim+d] - coords[0*dim+d]);
        }
        detJ = (J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
                J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
                J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
        PetscLogFlopsNoError(18.0 + 12.0);
      }
      if (invJ) {
        invDet  = -1.0/detJ;
        invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
        invJ[0*3+1] = invDet*(J[0*3+2]*J[2*3+1] - J[0*3+1]*J[2*3+2]);
        invJ[0*3+2] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
        invJ[1*3+0] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
        invJ[1*3+1] = invDet*(J[0*3+0]*J[2*3+2] - J[0*3+2]*J[2*3+0]);
        invJ[1*3+2] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
        invJ[2*3+0] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        invJ[2*3+1] = invDet*(J[0*3+1]*J[2*3+0] - J[0*3+0]*J[2*3+1]);
        invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
        PetscLogFlopsNoError(37.0);
      }
      detJ *= 8.0;
    };
    void computeElementGeometry(const Obj<real_section_type>& coordinates, const point_type& e, typename real_section_type::value_type v0[], typename real_section_type::value_type J[], typename real_section_type::value_type invJ[], typename real_section_type::value_type& detJ) {
      const int coneSize = this->getSieve()->getConeSize(e);

      if (this->_dim == 2) {
        if (coneSize == 3) {
          computeTriangleGeometry(coordinates, e, v0, J, invJ, detJ);
        } else if (coneSize == 4) {
          computeRectangleGeometry(coordinates, e, v0, J, invJ, detJ);
        } else {
          throw ALE::Exception("Unsupported coneSize for element geometry computation");
        }
      } else if (this->_dim == 3) {
        if (coneSize == 4) {
          computeTetrahedronGeometry(coordinates, e, v0, J, invJ, detJ);
        } else if (coneSize == 8) {
          computeHexahedronGeometry(coordinates, e, v0, J, invJ, detJ);
        } else {
          throw ALE::Exception("Unsupported coneSize for element geometry computation");
        }
      } else {
        throw ALE::Exception("Unsupported dimension for element geometry computation");
      }
    };
    void computeBdSegmentGeometry(const Obj<real_section_type>& coordinates, const point_type& e, typename real_section_type::value_type v0[], typename real_section_type::value_type J[], typename real_section_type::value_type invJ[], typename real_section_type::value_type& detJ) {
      const typename real_section_type::value_type *coords = this->restrictClosure(coordinates, e);
      const int     dim    = 2;
      typename real_section_type::value_type        invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        //r2   = coords[1*dim+0]*coords[1*dim+0] + coords[1*dim+1]*coords[1*dim+1];
        J[0] =  (coords[1*dim+0] - coords[0*dim+0])*0.5; J[1] = (-coords[1*dim+1] + coords[0*dim+1])*0.5;
        J[2] =  (coords[1*dim+1] - coords[0*dim+1])*0.5; J[3] = ( coords[1*dim+0] - coords[0*dim+0])*0.5;
        detJ = J[0]*J[3] - J[1]*J[2];
        if (detJ < 0.0) {
          const typename real_section_type::value_type  xLength = this->_periodicity[0];

          if (xLength != 0.0) {
            typename real_section_type::value_type v0x = coords[0*dim+0];

            if (v0x == 0.0) {
              v0x = v0[0] = xLength;
            }
            for(int f = 0; f < dim; f++) {
              const typename real_section_type::value_type px = coords[(f+1)*dim+0] == 0.0 ? xLength : coords[(f+1)*dim+0];

              J[0*dim+f] = 0.5*(px - v0x);
            }
          }
          detJ = J[0]*J[3] - J[1]*J[2];
        }
        PetscLogFlopsNoError(8.0 + 3.0);
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[1];
        invJ[2] = -invDet*J[2];
        invJ[3] =  invDet*J[0];
        PetscLogFlopsNoError(5.0);
      }
    };
    void computeBdElementGeometry(const Obj<real_section_type>& coordinates, const point_type& e, typename real_section_type::value_type v0[], typename real_section_type::value_type J[], typename real_section_type::value_type invJ[], typename real_section_type::value_type& detJ) {
      if (this->_dim == 1) {
        computeBdSegmentGeometry(coordinates, e, v0, J, invJ, detJ);
        //      } else if (this->_dim == 2) {
        //        computeBdTriangleGeometry(coordinates, e, v0, J, invJ, detJ);
      } else {
        throw ALE::Exception("Unsupported dimension for element geometry computation");
      }
    };
    typename real_section_type::value_type getMaxVolume() {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const int                     dim         = this->getDimension();
      typename real_section_type::value_type v0[3], J[9], invJ[9], detJ, refVolume = 0.0, maxVolume = 0.0;

      if (dim == 1) refVolume = 2.0;
      if (dim == 2) refVolume = 2.0;
      if (dim == 3) refVolume = 4.0/3.0;
      for(typename label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        maxVolume = std::max(maxVolume, detJ*refVolume);
      }
      return maxVolume;
    };
  public:
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
      }
      if (name == "") {
        PetscPrintf(comm, "viewing a Mesh\n");
      } else {
        PetscPrintf(comm, "viewing Mesh '%s'\n", name.c_str());
      }
      this->getSieve()->view("mesh sieve", comm);
      Obj<names_type> sections = this->getRealSections();

      for(names_type::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getRealSection(*name)->view(*name);
      }
      sections = this->getIntSections();
      for(names_type::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getIntSection(*name)->view(*name);
      }
      sections = this->getArrowSections();
      for(names_type::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getArrowSection(*name)->view(*name);
      }
    };
  public: // Discretization
    void markBoundaryCells(const std::string& name, const int marker = 1, const int newMarker = 2, const bool onlyVertices = false) {
      const Obj<label_type>&                  label    = this->getLabel(name);
      const Obj<label_sequence>&              boundary = this->getLabelStratum(name, marker);
      const typename label_sequence::iterator end      = boundary->end();
      const Obj<sieve_type>&                  sieve    = this->getSieve();

      if (!onlyVertices) {
        typename ISieveVisitor::MarkVisitor<sieve_type,label_type> mV(*label, newMarker);

        for(typename label_sequence::iterator e_iter = boundary->begin(); e_iter != end; ++e_iter) {
          if (this->height(*e_iter) == 1) {
            sieve->support(*e_iter, mV);
          }
        }
      } else {
#if 1
        throw ALE::Exception("Rewrite this to first mark boundary edges/faces.");
#else
        const int depth = this->depth();

        for(typename label_sequence::iterator v_iter = boundary->begin(); v_iter != end; ++v_iter) {
          const Obj<supportArray>               support = sieve->nSupport(*v_iter, depth);
          const typename supportArray::iterator sEnd    = support->end();

          for(typename supportArray::iterator c_iter = support->begin(); c_iter != sEnd; ++c_iter) {
            const Obj<typename sieve_type::traits::coneSequence>&     cone = sieve->cone(*c_iter);
            const typename sieve_type::traits::coneSequence::iterator cEnd = cone->end();

            for(typename sieve_type::traits::coneSequence::iterator e_iter = cone->begin(); e_iter != cEnd; ++e_iter) {
              if (sieve->support(*e_iter)->size() == 1) {
                this->setValue(label, *c_iter, newMarker);
                break;
              }
            }
          }
        }
#endif
      }
    };
    int setFiberDimensions(const Obj<real_section_type>& s, const Obj<names_type>& discs, names_type& bcLabels) {
      const int debug  = this->debug();
      int       maxDof = 0;

      for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
        s->addSpace();
      }
      for(int d = 0; d <= this->_dim; ++d) {
        int numDof = 0;
        int f      = 0;

        for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
          const Obj<ALE::Discretization>& disc = this->getDiscretization(*f_iter);
          const int                       sDof = disc->getNumDof(d);

          numDof += sDof;
          if (sDof) s->setFiberDimension(this->depthStratum(d), sDof, f);
        }
        if (numDof) s->setFiberDimension(this->depthStratum(d), numDof);
        maxDof = std::max(maxDof, numDof);
      }
      // Process exclusions
      typedef ISieveVisitor::PointRetriever<sieve_type> Visitor;
      int f = 0;

      for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
        const Obj<ALE::Discretization>& disc      = this->getDiscretization(*f_iter);
        std::string                     labelName = "exclude-"+*f_iter;
        std::set<point_type>            seen;
        Visitor pV((int) pow((double) this->getSieve()->getMaxConeSize(), this->depth()), true);

        if (this->hasLabel(labelName)) {
          const Obj<label_type>&                  label     = this->getLabel(labelName);
          const Obj<label_sequence>&              exclusion = this->getLabelStratum(labelName, 1);
          const typename label_sequence::iterator end       = exclusion->end();
          if (debug > 1) {label->view(labelName.c_str());}

          for(typename label_sequence::iterator e_iter = exclusion->begin(); e_iter != end; ++e_iter) {
            ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), *e_iter, pV);
            const typename Visitor::point_type *oPoints = pV.getPoints();
            const int                           oSize   = pV.getSize();

            for(int cl = 0; cl < oSize; ++cl) {
              if (seen.find(oPoints[cl]) != seen.end()) continue;
              if (this->getValue(label, oPoints[cl]) == 1) {
                seen.insert(oPoints[cl]);
                s->setFiberDimension(oPoints[cl], 0, f);
                s->addFiberDimension(oPoints[cl], -disc->getNumDof(this->depth(oPoints[cl])));
                if (debug > 1) {std::cout << "  point: " << oPoints[cl] << " dim: " << disc->getNumDof(this->depth(oPoints[cl])) << std::endl;}
              }
            }
            pV.clear();
          }
        }
      }
      // Process constraints
      f = 0;
      for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
        const Obj<ALE::Discretization>&   disc        = this->getDiscretization(*f_iter);
        const Obj<std::set<std::string> > bcs         = disc->getBoundaryConditions();
        std::string                       excludeName = "exclude-"+*f_iter;

        for(std::set<std::string>::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter) {
          const Obj<ALE::BoundaryCondition>& bc       = disc->getBoundaryCondition(*bc_iter);
          const Obj<label_sequence>&         boundary = this->getLabelStratum(bc->getLabelName(), bc->getMarker());

          bcLabels.insert(bc->getLabelName());
          if (this->hasLabel(excludeName)) {
            const Obj<label_type>& label = this->getLabel(excludeName);

            for(typename label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
              if (!this->getValue(label, *e_iter)) {
                const int numDof = disc->getNumDof(this->depth(*e_iter));

                if (numDof) s->addConstraintDimension(*e_iter, numDof);
                if (numDof) s->setConstraintDimension(*e_iter, numDof, f);
              }
            }
          } else {
            for(typename label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
              const int numDof = disc->getNumDof(this->depth(*e_iter));

              if (numDof) s->addConstraintDimension(*e_iter, numDof);
              if (numDof) s->setConstraintDimension(*e_iter, numDof, f);
            }
          }
        }
      }
      return maxDof;
    };
    void calculateIndices() {
      typedef ISieveVisitor::PointRetriever<sieve_type> Visitor;
      // Should have an iterator over the whole tree
      Obj<names_type> discs = this->getDiscretizations();
      const int       debug = this->debug();
      std::map<std::string, std::pair<int, int*> > indices;

      for(typename names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
        const Obj<Discretization>& disc = this->getDiscretization(*d_iter);

        indices[*d_iter] = std::pair<int, int*>(0, new int[disc->sizeV(*this)]);
        disc->setIndices(indices[*d_iter].second);
      }
      const Obj<label_sequence>& cells   = this->heightStratum(0);
      Visitor pV((int) pow((double) this->getSieve()->getMaxConeSize(), this->depth())+1, true);
      ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), *cells->begin(), pV);
      const typename Visitor::point_type *oPoints = pV.getPoints();
      const int                           oSize   = pV.getSize();
      int                                 offset  = 0;

      if (debug > 1) {std::cout << "Closure for first element" << std::endl;}
      for(int cl = 0; cl < oSize; ++cl) {
        const int dim = this->depth(oPoints[cl]);

        if (debug > 1) {std::cout << "  point " << oPoints[cl] << " depth " << dim << std::endl;}
        for(typename names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
          const Obj<Discretization>& disc = this->getDiscretization(*d_iter);
          const int                  num  = disc->getNumDof(dim);

          if (debug > 1) {std::cout << "    disc " << disc->getName() << " numDof " << num << std::endl;}
          for(int o = 0; o < num; ++o) {
            indices[*d_iter].second[indices[*d_iter].first++] = offset++;
          }
        }
      }
      pV.clear();
      if (debug > 1) {
        for(typename names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
          const Obj<Discretization>& disc = this->getDiscretization(*d_iter);

          std::cout << "Discretization " << disc->getName() << " indices:";
          for(int i = 0; i < indices[*d_iter].first; ++i) {
            std::cout << " " << indices[*d_iter].second[i];
          }
          std::cout << std::endl;
        }
      }
    };
    void calculateIndicesExcluded(const Obj<real_section_type>& s, const Obj<names_type>& discs) {
      typedef ISieveVisitor::PointRetriever<sieve_type> Visitor;
      typedef std::map<std::string, std::pair<int, indexSet> > indices_type;
      const Obj<label_type>& indexLabel = this->createLabel("cellExclusion");
      const int debug  = this->debug();
      int       marker = 0;
      std::map<indices_type, int> indexMap;
      indices_type                indices;
      Visitor pV((int) pow((double) this->getSieve()->getMaxConeSize(), this->depth())+1, true);

      for(typename names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
        const Obj<Discretization>& disc = this->getDiscretization(*d_iter);
        const int                  size = disc->sizeV(*this);

        indices[*d_iter].second.resize(size);
      }
      const typename names_type::const_iterator dBegin = discs->begin();
      const typename names_type::const_iterator dEnd   = discs->end();
      std::set<point_type> seen;
      int f = 0;

      for(typename names_type::const_iterator f_iter = dBegin; f_iter != dEnd; ++f_iter, ++f) {
        std::string labelName = "exclude-"+*f_iter;

        if (this->hasLabel(labelName)) {
          const Obj<label_sequence>&              exclusion = this->getLabelStratum(labelName, 1);
          const typename label_sequence::iterator end       = exclusion->end();

          if (debug > 1) {std::cout << "Processing exclusion " << labelName << std::endl;}
          for(typename label_sequence::iterator e_iter = exclusion->begin(); e_iter != end; ++e_iter) {
            if (this->height(*e_iter)) continue;
            ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), *e_iter, pV);
            const typename Visitor::point_type *oPoints = pV.getPoints();
            const int                           oSize   = pV.getSize();
            int                                 offset  = 0;

            if (debug > 1) {std::cout << "  Closure for cell " << *e_iter << std::endl;}
            for(int cl = 0; cl < oSize; ++cl) {
              int g = 0;

              if (debug > 1) {std::cout << "    point " << oPoints[cl] << std::endl;}
              for(typename names_type::const_iterator g_iter = dBegin; g_iter != dEnd; ++g_iter, ++g) {
                const int fDim = s->getFiberDimension(oPoints[cl], g);

                if (debug > 1) {std::cout << "      disc " << *g_iter << " numDof " << fDim << std::endl;}
                for(int d = 0; d < fDim; ++d) {
                  indices[*g_iter].second[indices[*g_iter].first++] = offset++;
                }
              }
            }
            pV.clear();
            const std::map<indices_type, int>::iterator entry = indexMap.find(indices);

            if (debug > 1) {
              for(std::map<indices_type, int>::iterator i_iter = indexMap.begin(); i_iter != indexMap.end(); ++i_iter) {
                for(typename names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
                  std::cout << "Discretization (" << i_iter->second << ") " << *g_iter << " indices:";
                  for(int i = 0; i < ((indices_type) i_iter->first)[*g_iter].first; ++i) {
                    std::cout << " " << ((indices_type) i_iter->first)[*g_iter].second[i];
                  }
                  std::cout << std::endl;
                }
                std::cout << "Comparison: " << (indices == i_iter->first) << std::endl;
              }
              for(typename names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
                std::cout << "Discretization " << *g_iter << " indices:";
                for(int i = 0; i < indices[*g_iter].first; ++i) {
                  std::cout << " " << indices[*g_iter].second[i];
                }
                std::cout << std::endl;
              }
            }
            if (entry != indexMap.end()) {
              this->setValue(indexLabel, *e_iter, entry->second);
              if (debug > 1) {std::cout << "  Found existing indices with marker " << entry->second << std::endl;}
            } else {
              indexMap[indices] = ++marker;
              this->setValue(indexLabel, *e_iter, marker);
              if (debug > 1) {std::cout << "  Created new indices with marker " << marker << std::endl;}
            }
            for(typename names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
              indices[*g_iter].first  = 0;
              for(unsigned int i = 0; i < indices[*g_iter].second.size(); ++i) indices[*g_iter].second[i] = 0;
            }
          }
        }
      }
      if (debug > 1) {indexLabel->view("cellExclusion");}
      for(std::map<indices_type, int>::iterator i_iter = indexMap.begin(); i_iter != indexMap.end(); ++i_iter) {
        if (debug > 1) {std::cout << "Setting indices for marker " << i_iter->second << std::endl;}
        for(typename names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
          const Obj<Discretization>& disc = this->getDiscretization(*g_iter);
          const indexSet  indSet   = ((indices_type) i_iter->first)[*g_iter].second;
          const int       size     = indSet.size();
          int            *_indices = new int[size];

          if (debug > 1) {std::cout << "  field " << *g_iter << std::endl;}
          for(int i = 0; i < size; ++i) {
            _indices[i] = indSet[i];
            if (debug > 1) {std::cout << "    indices["<<i<<"] = " << _indices[i] << std::endl;}
          }
          disc->setIndices(_indices, i_iter->second);
        }
      }
    };
    void setupField(const Obj<real_section_type>& s, const int cellMarker = 2, const bool noUpdate = false) {
      typedef ISieveVisitor::PointRetriever<sieve_type> Visitor;
      const Obj<names_type>& discs  = this->getDiscretizations();
      const int              debug  = s->debug();
      names_type             bcLabels;

      s->setChart(this->getSieve()->getChart());
      this->_maxDof = this->setFiberDimensions(s, discs, bcLabels);
      this->calculateIndices();
      this->calculateIndicesExcluded(s, discs);
      this->allocate(s);
      s->defaultConstraintDof();
      const Obj<label_type>& cellExclusion = this->getLabel("cellExclusion");

      if (debug > 1) {std::cout << "Setting boundary values" << std::endl;}
      for(typename names_type::const_iterator n_iter = bcLabels.begin(); n_iter != bcLabels.end(); ++n_iter) {
        const Obj<label_sequence>&              boundaryCells = this->getLabelStratum(*n_iter, cellMarker);
        const Obj<real_section_type>&           coordinates   = this->getRealSection("coordinates");
        const Obj<names_type>&                  discs         = this->getDiscretizations();
        const point_type                        firstCell     = *boundaryCells->begin();
        const int                               numFields     = discs->size();
        typename real_section_type::value_type *values        = new typename real_section_type::value_type[this->sizeWithBC(s, firstCell)];
        int                                    *dofs          = new int[this->_maxDof];
        int                                    *v             = new int[numFields];
        typename real_section_type::value_type *v0            = new typename real_section_type::value_type[this->getDimension()];
        typename real_section_type::value_type *J             = new typename real_section_type::value_type[this->getDimension()*this->getDimension()];
        typename real_section_type::value_type  detJ;
        Visitor pV((int) pow((double) this->getSieve()->getMaxConeSize(), this->depth())+1, true);

        for(typename label_sequence::iterator c_iter = boundaryCells->begin(); c_iter != boundaryCells->end(); ++c_iter) {
          ISieveTraversal<sieve_type>::orientedClosure(*this->getSieve(), *c_iter, pV);
          const typename Visitor::point_type *oPoints = pV.getPoints();
          const int                           oSize   = pV.getSize();

          if (debug > 1) {std::cout << "  Boundary cell " << *c_iter << std::endl;}
          this->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
          for(int f = 0; f < numFields; ++f) v[f] = 0;
          for(int cl = 0; cl < oSize; ++cl) {
            const int cDim = s->getConstraintDimension(oPoints[cl]);
            int       off  = 0;
            int       f    = 0;
            int       i    = -1;

            if (debug > 1) {std::cout << "    point " << oPoints[cl] << std::endl;}
            if (cDim) {
              if (debug > 1) {std::cout << "      constrained excMarker: " << this->getValue(cellExclusion, *c_iter) << std::endl;}
              for(typename names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
                const Obj<ALE::Discretization>& disc    = this->getDiscretization(*f_iter);
                const Obj<names_type>           bcs     = disc->getBoundaryConditions();
                const int                       fDim    = s->getFiberDimension(oPoints[cl], f);//disc->getNumDof(this->depth(oPoints[cl]));
                const int                      *indices = disc->getIndices(this->getValue(cellExclusion, *c_iter));
                int                             b       = 0;

                for(typename names_type::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter, ++b) {
                  const Obj<ALE::BoundaryCondition>& bc    = disc->getBoundaryCondition(*bc_iter);
                  const int                          value = this->getValue(this->getLabel(bc->getLabelName()), oPoints[cl]);

                  if (b > 0) v[f] -= fDim;
                  if (value == bc->getMarker()) {
                    if (debug > 1) {std::cout << "      field " << *f_iter << " marker " << value << std::endl;}
                    for(int d = 0; d < fDim; ++d, ++v[f]) {
                      dofs[++i] = off+d;
                      if (!noUpdate) values[indices[v[f]]] = (*bc->getDualIntegrator())(v0, J, v[f], bc->getFunction());
                      if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                    }
                    // Allow only one condition per point
                    ++b;
                    break;
                  } else {
                    if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
                    for(int d = 0; d < fDim; ++d, ++v[f]) {
                      values[indices[v[f]]] = 0.0;
                      if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                    }
                  }
                }
                if (b == 0) {
                  if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
                  for(int d = 0; d < fDim; ++d, ++v[f]) {
                    values[indices[v[f]]] = 0.0;
                    if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                  }
                }
                off += fDim;
              }
              if (i != cDim-1) {throw ALE::Exception("Invalid constraint initialization");}
              s->setConstraintDof(oPoints[cl], dofs);
            } else {
              if (debug > 1) {std::cout << "      unconstrained" << std::endl;}
              for(typename names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
                const Obj<ALE::Discretization>& disc    = this->getDiscretization(*f_iter);
                const int                       fDim    = s->getFiberDimension(oPoints[cl], f);//disc->getNumDof(this->depth(oPoints[cl]));
                const int                      *indices = disc->getIndices(this->getValue(cellExclusion, *c_iter));

                if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
                for(int d = 0; d < fDim; ++d, ++v[f]) {
                  values[indices[v[f]]] = 0.0;
                  if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                }
              }
            }
          }
          if (debug > 1) {
            for(int f = 0; f < numFields; ++f) v[f] = 0;
            for(int cl = 0; cl < oSize; ++cl) {
              int f = 0;
              for(typename names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
                const Obj<ALE::Discretization>& disc    = this->getDiscretization(*f_iter);
                const int                       fDim    = s->getFiberDimension(oPoints[cl], f);
                const int                      *indices = disc->getIndices(this->getValue(cellExclusion, *c_iter));

                for(int d = 0; d < fDim; ++d, ++v[f]) {
                  std::cout << "    "<<*f_iter<<"-value["<<indices[v[f]]<<"] " << values[indices[v[f]]] << std::endl;
                }
              }
            }
          }
          if (!noUpdate) {
            this->updateAll(s, *c_iter, values);
          }
          pV.clear();
        }
        delete [] dofs;
        delete [] values;
        delete [] v0;
        delete [] J;
      }
      if (debug > 1) {s->view("");}
    };
  public:
    // Take in a map for the cells labels
    template<typename Section_>
    void relabel(Section_& labeling) {
      this->getSieve()->relabel(labeling);
      // Relabel sections
      Obj<std::set<std::string> > realNames = this->getRealSections();

      for(std::set<std::string>::const_iterator n_iter = realNames->begin(); n_iter != realNames->end(); ++n_iter) {
        Obj<real_section_type> section = new real_section_type(this->comm(), this->debug());

        section->setName(*n_iter);
        ALE::Ordering<>::relabelSection(*this->getRealSection(*n_iter), labeling, *section);
        this->setRealSection(*n_iter, section);
      }
      Obj<std::set<std::string> > intNames = this->getIntSections();

      for(std::set<std::string>::const_iterator n_iter = intNames->begin(); n_iter != intNames->end(); ++n_iter) {
        Obj<int_section_type> section = new int_section_type(this->comm(), this->debug());

        section->setName(*n_iter);
        ALE::Ordering<>::relabelSection(*this->getIntSection(*n_iter), labeling, *section);
        this->setIntSection(*n_iter, section);
      }
      // Relabel labels
      const labels_type& labels = this->getLabels();

      for(typename labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
        Obj<label_type> label = new label_type(this->comm(), this->debug());

        l_iter->second->relabel(labeling, *label);
        this->setLabel(l_iter->first, label);
      }
      // Relabel overlap
      Obj<send_overlap_type> sendOverlap = new send_overlap_type(this->comm(), this->debug());
      Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(this->comm(), this->debug());

      this->getSendOverlap()->relabel(labeling, *sendOverlap);
      this->setSendOverlap(sendOverlap);
      this->getRecvOverlap()->relabel(labeling, *recvOverlap);
      this->setRecvOverlap(recvOverlap);
      // Relabel distribution overlap ???
      // Relabel renumbering
    }
  };
}

namespace ALE {
  template<typename IndexType, typename ScalarType>
  class Mesh : public Bundle<ALE::Sieve<IndexType,IndexType,IndexType>, GeneralSection<IndexType,ScalarType> > {
  public:
    typedef Bundle<ALE::Sieve<IndexType,IndexType,IndexType>, GeneralSection<IndexType,ScalarType> > base_type;
    typedef typename base_type::sieve_type            sieve_type;
    typedef typename sieve_type::point_type           point_type;
    typedef typename base_type::alloc_type            alloc_type;
    typedef typename base_type::label_type            label_type;
    typedef typename base_type::label_sequence        label_sequence;
    typedef typename base_type::coneArray             coneArray;
    typedef typename base_type::supportArray          supportArray;
    typedef typename base_type::arrow_section_type    arrow_section_type;
    typedef typename base_type::real_section_type     real_section_type;
    typedef typename base_type::numbering_type        numbering_type;
    typedef typename base_type::order_type            order_type;
    typedef typename base_type::send_overlap_type     send_overlap_type;
    typedef typename base_type::recv_overlap_type     recv_overlap_type;
    typedef typename base_type::sieve_alg_type        sieve_alg_type;
    typedef std::set<std::string>            names_type;
    typedef std::map<std::string, Obj<Discretization> > discretizations_type;
    typedef std::vector<PETSc::Point<3> >    holes_type;
  protected:
    int                  _dim;
    discretizations_type _discretizations;
    std::map<int,double> _periodicity;
    holes_type           _holes;
  public:
    Mesh(MPI_Comm comm, int dim, int debug = 0) : base_type(comm, debug), _dim(dim) {
      ///this->_factory = MeshNumberingFactory::singleton(debug);
      //std::cout << "["<<this->commRank()<<"]: Creating an ALE::Mesh" << std::endl;
    };
    ~Mesh() {
      //std::cout << "["<<this->commRank()<<"]: Destroying an ALE::Mesh" << std::endl;
    };
  public: // Accessors
    int getDimension() const {return this->_dim;};
    void setDimension(const int dim) {this->_dim = dim;};
    const Obj<Discretization>& getDiscretization() {return this->getDiscretization("default");};
    const Obj<Discretization>& getDiscretization(const std::string& name) {return this->_discretizations[name];};
    void setDiscretization(const Obj<Discretization>& disc) {this->setDiscretization("default", disc);};
    void setDiscretization(const std::string& name, const Obj<Discretization>& disc) {this->_discretizations[name] = disc;};
    Obj<names_type> getDiscretizations() const {
      Obj<names_type> names = names_type();

      for(discretizations_type::const_iterator d_iter = this->_discretizations.begin(); d_iter != this->_discretizations.end(); ++d_iter) {
        names->insert(d_iter->first);
      }
      return names;
    };
    bool getPeriodicity(const int d) {return this->_periodicity[d];};
    void setPeriodicity(const int d, const double length) {this->_periodicity[d] = length;};
    const holes_type& getHoles() const {return this->_holes;};
    void addHole(const double hole[]) {
      this->_holes.push_back(hole);
    };
    void copyHoles(const Obj<Mesh>& m) {
      const holes_type& holes = m->getHoles();

      for(holes_type::const_iterator h_iter = holes.begin(); h_iter != holes.end(); ++h_iter) {
        this->_holes.push_back(*h_iter);
      }
    };
    void copy(const Obj<Mesh>& m) {
      this->setSieve(m->getSieve());
      this->setLabel("height", m->getLabel("height"));
      this->_maxHeight = m->height();
      this->setLabel("depth", m->getLabel("depth"));
      this->_maxDepth  = m->depth();
      this->setLabel("marker", m->getLabel("marker"));
      this->setRealSection("coordinates", m->getRealSection("coordinates"));
      this->setArrowSection("orientation", m->getArrowSection("orientation"));
    };
  public: // Cell topology and geometry
    int getNumCellCorners(const point_type& p, const int depth = -1) const {
      return (this->getDimension() > 0) ? this->_sieve->nCone(p, depth < 0 ? this->depth() : depth)->size() : 1;
    };
    int getNumCellCorners() {
      return getNumCellCorners(*(this->heightStratum(0)->begin()));
    };
    void setupCoordinates(const Obj<real_section_type>& coordinates) {};
    void computeTriangleGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrictClosure(coordinates, e);
      const int     dim    = 2;
      double        invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        for(int d = 0; d < dim; d++) {
          for(int f = 0; f < dim; f++) {
            J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
          }
        }
        detJ = J[0]*J[3] - J[1]*J[2];
        if (detJ < 0.0) {
          const double  xLength = this->_periodicity[0];

          if (xLength != 0.0) {
            double v0x = coords[0*dim+0];

            if (v0x == 0.0) {
              v0x = v0[0] = xLength;
            }
            for(int f = 0; f < dim; f++) {
              const double px = coords[(f+1)*dim+0] == 0.0 ? xLength : coords[(f+1)*dim+0];

              J[0*dim+f] = 0.5*(px - v0x);
            }
          }
          detJ = J[0]*J[3] - J[1]*J[2];
        }
        PetscLogFlopsNoError(8.0 + 3.0);
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[1];
        invJ[2] = -invDet*J[2];
        invJ[3] =  invDet*J[0];
        PetscLogFlopsNoError(5.0);
      }
    };
    void computeQuadrilateralGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double point[], double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrictClosure(coordinates, e);
      const int     dim    = 2;
      double        invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        double x_1 = coords[2] - coords[0];
        double y_1 = coords[3] - coords[1];
        double x_2 = coords[6] - coords[0];
        double y_2 = coords[7] - coords[1];
        double x_3 = coords[4] - coords[0];
        double y_3 = coords[5] - coords[1];

        J[0] = x_1 + (x_3 - x_1 - x_2)*point[1];
        J[1] = x_2 + (x_3 - x_1 - x_2)*point[0];
        J[2] = y_1 + (y_3 - y_1 - y_2)*point[1];
        J[3] = y_1 + (y_3 - y_1 - y_2)*point[0];
        detJ = J[0]*J[3] - J[1]*J[2];
        PetscLogFlopsNoError(6.0 + 16.0 + 3.0);
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[1];
        invJ[2] = -invDet*J[2];
        invJ[3] =  invDet*J[0];
        PetscLogFlopsNoError(5.0);
      }
    };
    void computeTetrahedronGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrictClosure(coordinates, e);
      const int     dim    = 3;
      double        invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        for(int d = 0; d < dim; d++) {
          for(int f = 0; f < dim; f++) {
            J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
          }
        }
        // The minus sign is here since I orient the first face to get the outward normal
        detJ = -(J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
                 J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
                 J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
        PetscLogFlopsNoError(18.0 + 12.0);
      }
      if (invJ) {
        invDet  = -1.0/detJ;
        invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
        invJ[0*3+1] = invDet*(J[0*3+2]*J[2*3+1] - J[0*3+1]*J[2*3+2]);
        invJ[0*3+2] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
        invJ[1*3+0] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
        invJ[1*3+1] = invDet*(J[0*3+0]*J[2*3+2] - J[0*3+2]*J[2*3+0]);
        invJ[1*3+2] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
        invJ[2*3+0] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        invJ[2*3+1] = invDet*(J[0*3+1]*J[2*3+0] - J[0*3+0]*J[2*3+1]);
        invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
        PetscLogFlopsNoError(37.0);
      }
    };
    void computeHexahedralGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double point[], double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrictClosure(coordinates, e);
      const int     dim    = 3;
      double        invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        double x_1 = coords[3]  - coords[0];
        double y_1 = coords[4]  - coords[1];
        double z_1 = coords[5]  - coords[2];
        double x_2 = coords[6]  - coords[0];
        double y_2 = coords[7]  - coords[1];
        double z_2 = coords[8]  - coords[2];
        double x_3 = coords[9]  - coords[0];
        double y_3 = coords[10] - coords[1];
        double z_3 = coords[11] - coords[2];
        double x_4 = coords[12] - coords[0];
        double y_4 = coords[13] - coords[1];
        double z_4 = coords[14] - coords[2];
        double x_5 = coords[15] - coords[0];
        double y_5 = coords[16] - coords[1];
        double z_5 = coords[17] - coords[2];
        double x_6 = coords[18] - coords[0];
        double y_6 = coords[19] - coords[1];
        double z_6 = coords[20] - coords[2];
        double x_7 = coords[21] - coords[0];
        double y_7 = coords[22] - coords[1];
        double z_7 = coords[23] - coords[2];
        double g_x = x_1 - x_2 + x_3 + x_4 - x_5 + x_6 - x_7;
        double g_y = y_1 - y_2 + y_3 + y_4 - y_5 + y_6 - y_7;
        double g_z = z_1 - z_2 + z_3 + z_4 - z_5 + z_6 - z_7;

        J[0] = x_1 + (x_2 - x_1 - x_3)*point[1] + (x_5 - x_1 - x_4)*point[2] + g_x*point[1]*point[2];
        J[1] = x_3 + (x_2 - x_1 - x_3)*point[0] + (x_7 - x_3 - x_4)*point[2] + g_x*point[2]*point[0];
        J[2] = x_4 + (x_7 - x_3 - x_4)*point[1] + (x_5 - x_1 - x_4)*point[0] + g_x*point[0]*point[1];
        J[3] = y_1 + (y_2 - y_1 - y_3)*point[1] + (y_5 - y_1 - y_4)*point[2] + g_y*point[1]*point[2];
        J[4] = y_3 + (y_2 - y_1 - y_3)*point[0] + (y_7 - y_3 - y_4)*point[2] + g_y*point[2]*point[0];
        J[5] = y_4 + (y_7 - y_3 - y_4)*point[1] + (y_5 - y_1 - y_4)*point[0] + g_y*point[0]*point[1];
        J[6] = z_1 + (z_2 - z_1 - z_3)*point[1] + (z_5 - z_1 - z_4)*point[2] + g_z*point[1]*point[2];
        J[7] = z_3 + (z_2 - z_1 - z_3)*point[0] + (z_7 - z_3 - z_4)*point[2] + g_z*point[2]*point[0];
        J[8] = z_4 + (z_7 - z_3 - z_4)*point[1] + (z_5 - z_1 - z_4)*point[0] + g_z*point[0]*point[1];
        detJ = (J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
                J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
                J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]));
        PetscLogFlopsNoError(39.0 + 81.0 + 12.0);
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
        invJ[0*3+1] = invDet*(J[0*3+2]*J[2*3+1] - J[0*3+1]*J[2*3+2]);
        invJ[0*3+2] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
        invJ[1*3+0] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
        invJ[1*3+1] = invDet*(J[0*3+0]*J[2*3+2] - J[0*3+2]*J[2*3+0]);
        invJ[1*3+2] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
        invJ[2*3+0] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        invJ[2*3+1] = invDet*(J[0*3+1]*J[2*3+0] - J[0*3+0]*J[2*3+1]);
        invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
        PetscLogFlopsNoError(37.0);
      }
    };
    void computeElementGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      if (this->_dim == 2) {
        computeTriangleGeometry(coordinates, e, v0, J, invJ, detJ);
      } else if (this->_dim == 3) {
        computeTetrahedronGeometry(coordinates, e, v0, J, invJ, detJ);
      } else {
        throw ALE::Exception("Unsupported dimension for element geometry computation");
      }
    };
    void computeLineFaceGeometry(const point_type& cell, const point_type& face, const int f, const double cellInvJ[], double invJ[], double& detJ, double normal[], double tangent[]) {
      const typename arrow_section_type::point_type arrow(cell, face);
      const bool reversed = (this->getArrowSection("orientation")->restrictPoint(arrow)[0] == -2);
      const int  dim      = this->getDimension();
      double     norm     = 0.0;
      double    *vec      = tangent;

      if (f == 0) {
        vec[0] = 0.0;        vec[1] = -1.0;
      } else if (f == 1) {
        vec[0] = 0.70710678; vec[1] = 0.70710678;
      } else if (f == 2) {
        vec[0] = -1.0;       vec[1] = 0.0;
      }
      for(int d = 0; d < dim; ++d) {
        normal[d] = 0.0;
        for(int e = 0; e < dim; ++e) normal[d] += cellInvJ[e*dim+d]*vec[e];
        if (reversed) normal[d] = -normal[d];
        norm += normal[d]*normal[d];
      }
      norm = std::sqrt(norm);
      for(int d = 0; d < dim; ++d) {
        normal[d] /= norm;
      }
      tangent[0] =  normal[1];
      tangent[1] = -normal[0];
      if (this->debug()) {
        std::cout << "Cell: " << cell << " Face: " << face << "("<<f<<")" << std::endl;
        for(int d = 0; d < dim; ++d) {
          std::cout << "Normal["<<d<<"]: " << normal[d] << " Tangent["<<d<<"]: " << tangent[d] << std::endl;
        }
      }
      // Now get 1D Jacobian info
      //   Should be a way to get this directly
      const double *coords = this->restrictClosure(this->getRealSection("coordinates"), face);
      detJ    = std::sqrt(PetscSqr(coords[1*2+0] - coords[0*2+0]) + PetscSqr(coords[1*2+1] - coords[0*2+1]))/2.0;
      invJ[0] = 1.0/detJ;
    };
    void computeTriangleFaceGeometry(const point_type& cell, const point_type& face, const int f, const double cellInvJ[], double invJ[], double& detJ, double normal[], double tangent[]) {
      const typename arrow_section_type::point_type arrow(cell, face);
      const bool reversed = this->getArrowSection("orientation")->restrictPoint(arrow)[0] < 0;
      const int  dim      = this->getDimension();
      const int  faceDim  = dim-1;
      double     norm     = 0.0;
      double    *vec      = tangent;

      if (f == 0) {
        vec[0] = 0.0;        vec[1] = 0.0;        vec[2] = -1.0;
      } else if (f == 1) {
        vec[0] = 0.0;        vec[1] = -1.0;       vec[2] = 0.0;
      } else if (f == 2) {
        vec[0] = 0.57735027; vec[1] = 0.57735027; vec[2] = 0.57735027;
      } else if (f == 3) {
        vec[0] = -1.0;       vec[1] = 0.0;        vec[2] = 0.0;
      }
      for(int d = 0; d < dim; ++d) {
        normal[d] = 0.0;
        for(int e = 0; e < dim; ++e) normal[d] += cellInvJ[e*dim+d]*vec[e];
        if (reversed) normal[d] = -normal[d];
        norm += normal[d]*normal[d];
      }
      norm = std::sqrt(norm);
      for(int d = 0; d < dim; ++d) {
        normal[d] /= norm;
      }
      // Get tangents
      tangent[0] = normal[1] - normal[2];
      tangent[1] = normal[2] - normal[0];
      tangent[2] = normal[0] - normal[1];
      norm = 0.0;
      for(int d = 0; d < dim; ++d) {
        norm += tangent[d]*tangent[d];
      }
      norm = std::sqrt(norm);
      for(int d = 0; d < dim; ++d) {
        tangent[d] /= norm;
      }
      tangent[3] = normal[1]*tangent[2] - normal[2]*tangent[1];
      tangent[4] = normal[2]*tangent[0] - normal[0]*tangent[2];
      tangent[5] = normal[0]*tangent[1] - normal[1]*tangent[0];
      if (this->debug()) {
        std::cout << "Cell: " << cell << " Face: " << face << "("<<f<<")" << std::endl;
        for(int d = 0; d < dim; ++d) {
          std::cout << "Normal["<<d<<"]: " << normal[d] << " TangentA["<<d<<"]: " << tangent[d] << " TangentB["<<d<<"]: " << tangent[dim+d] << std::endl;
        }
      }
      // Now get 2D Jacobian info
      //   Should be a way to get this directly
      const double *coords = this->restrictClosure(this->getRealSection("coordinates"), face);
      // Rotate so that normal in z
      double invR[9], R[9];
      double detR, invDetR;
      for(int d = 0; d < dim; d++) {
        invR[d*dim+0] = tangent[d];
        invR[d*dim+1] = tangent[dim+d];
        invR[d*dim+2] = normal[d];
      }
      invDetR = (invR[0*3+0]*(invR[1*3+1]*invR[2*3+2] - invR[1*3+2]*invR[2*3+1]) +
                 invR[0*3+1]*(invR[1*3+2]*invR[2*3+0] - invR[1*3+0]*invR[2*3+2]) +
                 invR[0*3+2]*(invR[1*3+0]*invR[2*3+1] - invR[1*3+1]*invR[2*3+0]));
      detR  = 1.0/invDetR;
      R[0*3+0] = detR*(invR[1*3+1]*invR[2*3+2] - invR[1*3+2]*invR[2*3+1]);
      R[0*3+1] = detR*(invR[0*3+2]*invR[2*3+1] - invR[0*3+1]*invR[2*3+2]);
      R[0*3+2] = detR*(invR[0*3+1]*invR[1*3+2] - invR[0*3+2]*invR[1*3+1]);
      R[1*3+0] = detR*(invR[1*3+2]*invR[2*3+0] - invR[1*3+0]*invR[2*3+2]);
      R[1*3+1] = detR*(invR[0*3+0]*invR[2*3+2] - invR[0*3+2]*invR[2*3+0]);
      R[1*3+2] = detR*(invR[0*3+2]*invR[1*3+0] - invR[0*3+0]*invR[1*3+2]);
      R[2*3+0] = detR*(invR[1*3+0]*invR[2*3+1] - invR[1*3+1]*invR[2*3+0]);
      R[2*3+1] = detR*(invR[0*3+1]*invR[2*3+0] - invR[0*3+0]*invR[2*3+1]);
      R[2*3+2] = detR*(invR[0*3+0]*invR[1*3+1] - invR[0*3+1]*invR[1*3+0]);
      for(int d = 0; d < dim; d++) {
        for(int e = 0; e < dim; e++) {
          invR[d*dim+e] = 0.0;
          for(int g = 0; g < dim; g++) {
            invR[d*dim+e] += R[e*dim+g]*coords[d*dim+g];
          }
        }
      }
      for(int d = dim-1; d >= 0; --d) {
        invR[d*dim+2] -= invR[0*dim+2];
        if (this->debug() && (d == dim-1)) {
          double ref[9];
          for(int q = 0; q < dim; q++) {
            for(int e = 0; e < dim; e++) {
              ref[q*dim+e] = 0.0;
              for(int g = 0; g < dim; g++) {
                ref[q*dim+e] += cellInvJ[e*dim+g]*coords[q*dim+g];
              }
            }
          }
          std::cout << "f: " << f << std::endl;
          std::cout << this->printMatrix(std::string("coords"), dim, dim, coords) << std::endl;
          std::cout << this->printMatrix(std::string("ref coords"), dim, dim, ref) << std::endl;
          std::cout << this->printMatrix(std::string("R"), dim, dim, R) << std::endl;
          std::cout << this->printMatrix(std::string("invR"), dim, dim, invR) << std::endl;
        }
        if (fabs(invR[d*dim+2]) > 1.0e-8) {
          throw ALE::Exception("Invalid rotation");
        }
      }
      double J[4];
      for(int d = 0; d < faceDim; d++) {
        for(int e = 0; e < faceDim; e++) {
          J[d*faceDim+e] = 0.5*(invR[(e+1)*dim+d] - invR[0*dim+d]);
        }
      }
      detJ = fabs(J[0]*J[3] - J[1]*J[2]);
      // Probably need something here if detJ < 0
      const double invDet = 1.0/detJ;
      invJ[0] =  invDet*J[3];
      invJ[1] = -invDet*J[1];
      invJ[2] = -invDet*J[2];
      invJ[3] =  invDet*J[0];
    };
    void computeFaceGeometry(const point_type& cell, const point_type& face, const int f, const double cellInvJ[], double invJ[], double& detJ, double normal[], double tangent[]) {
      if (this->_dim == 2) {
        computeLineFaceGeometry(cell, face, f, cellInvJ, invJ, detJ, normal, tangent);
      } else if (this->_dim == 3) {
        computeTriangleFaceGeometry(cell, face, f, cellInvJ, invJ, detJ, normal, tangent);
      } else {
        throw ALE::Exception("Unsupported dimension for element geometry computation");
      }
    };
    double getMaxVolume() {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const int                     dim         = this->getDimension();
      double v0[3], J[9], invJ[9], detJ, refVolume = 0.0, maxVolume = 0.0;

      if (dim == 1) refVolume = 2.0;
      if (dim == 2) refVolume = 2.0;
      if (dim == 3) refVolume = 4.0/3.0;
      for(typename label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        maxVolume = std::max(maxVolume, detJ*refVolume);
      }
      return maxVolume;
    };
    // Find the cell in which this point lies (stupid algorithm)
    point_type locatePoint_2D(const typename real_section_type::value_type point[]) {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const int                     embedDim    = 2;
      double v0[2], J[4], invJ[4], detJ;

      for(typename label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        double xi   = invJ[0*embedDim+0]*(point[0] - v0[0]) + invJ[0*embedDim+1]*(point[1] - v0[1]);
        double eta  = invJ[1*embedDim+0]*(point[0] - v0[0]) + invJ[1*embedDim+1]*(point[1] - v0[1]);

        if ((xi >= 0.0) && (eta >= 0.0) && (xi + eta <= 2.0)) {
          return *c_iter;
        }
      }
      throw ALE::Exception("Could not locate point");
    };
    //   Assume a simplex and 3D
    point_type locatePoint_3D(const typename real_section_type::value_type point[]) {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const int                     embedDim    = 3;
      double v0[3], J[9], invJ[9], detJ;

      for(typename label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        double xi   = invJ[0*embedDim+0]*(point[0] - v0[0]) + invJ[0*embedDim+1]*(point[1] - v0[1]) + invJ[0*embedDim+2]*(point[2] - v0[2]);
        double eta  = invJ[1*embedDim+0]*(point[0] - v0[0]) + invJ[1*embedDim+1]*(point[1] - v0[1]) + invJ[1*embedDim+2]*(point[2] - v0[2]);
        double zeta = invJ[2*embedDim+0]*(point[0] - v0[0]) + invJ[2*embedDim+1]*(point[1] - v0[1]) + invJ[2*embedDim+2]*(point[2] - v0[2]);

        if ((xi >= 0.0) && (eta >= 0.0) && (zeta >= 0.0) && (xi + eta + zeta <= 2.0)) {
          return *c_iter;
        }
      }
      throw ALE::Exception("Could not locate point");
    };
    point_type locatePoint(const typename real_section_type::value_type point[], point_type guess = -1) {
      //guess overrides this by saying that we already know the relation of this point to this mesh.  We will need to make it a more robust "guess" later for more than P1
      if (guess != -1) {
        return guess;
      }else if (this->_dim == 2) {
        return locatePoint_2D(point);
      } else if (this->_dim == 3) {
        return locatePoint_3D(point);
      } else {
        throw ALE::Exception("No point location for mesh dimension");
      }
    };
  public: // Discretization
    void markBoundaryCells(const std::string& name, const int marker = 1, const int newMarker = 2, const bool onlyVertices = false) {
      const Obj<label_type>&     label    = this->getLabel(name);
      const Obj<label_sequence>& boundary = this->getLabelStratum(name, marker);
      const Obj<sieve_type>&     sieve    = this->getSieve();

      if (!onlyVertices) {
        const typename label_sequence::iterator end = boundary->end();

        for(typename label_sequence::iterator e_iter = boundary->begin(); e_iter != end; ++e_iter) {
          if (this->height(*e_iter) == 1) {
            const point_type cell = *sieve->support(*e_iter)->begin();

            this->setValue(label, cell, newMarker);
          }
        }
      } else {
        const typename label_sequence::iterator end   = boundary->end();
        const int                      depth = this->depth();

        for(typename label_sequence::iterator v_iter = boundary->begin(); v_iter != end; ++v_iter) {
          const Obj<supportArray>      support = sieve->nSupport(*v_iter, depth);
          const typename supportArray::iterator sEnd    = support->end();

          for(typename supportArray::iterator c_iter = support->begin(); c_iter != sEnd; ++c_iter) {
            const Obj<typename sieve_type::traits::coneSequence>&     cone = sieve->cone(*c_iter);
            const typename sieve_type::traits::coneSequence::iterator cEnd = cone->end();

            for(typename sieve_type::traits::coneSequence::iterator e_iter = cone->begin(); e_iter != cEnd; ++e_iter) {
              if (sieve->support(*e_iter)->size() == 1) {
                this->setValue(label, *c_iter, newMarker);
                break;
              }
            }
          }
        }
      }
    };
    int setFiberDimensions(const Obj<real_section_type>& s, const Obj<names_type>& discs, names_type& bcLabels) {
      const int debug  = this->debug();
      int       maxDof = 0;

      for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
        s->addSpace();
      }
      for(int d = 0; d <= this->_dim; ++d) {
        int numDof = 0;
        int f      = 0;

        for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
          const Obj<ALE::Discretization>& disc = this->getDiscretization(*f_iter);
          const int                       sDof = disc->getNumDof(d);

          numDof += sDof;
          if (sDof) s->setFiberDimension(this->depthStratum(d), sDof, f);
        }
        if (numDof) s->setFiberDimension(this->depthStratum(d), numDof);
        maxDof = std::max(maxDof, numDof);
      }
      // Process exclusions
      int f = 0;

      for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
        const Obj<ALE::Discretization>& disc      = this->getDiscretization(*f_iter);
        std::string                     labelName = "exclude-"+*f_iter;
        std::set<point_type>            seen;

        if (this->hasLabel(labelName)) {
          const Obj<label_type>&         label     = this->getLabel(labelName);
          const Obj<label_sequence>&     exclusion = this->getLabelStratum(labelName, 1);
          const typename label_sequence::iterator end       = exclusion->end();
          if (debug > 1) {label->view(labelName.c_str());}

          for(typename label_sequence::iterator e_iter = exclusion->begin(); e_iter != end; ++e_iter) {
            const Obj<coneArray>      closure = ALE::SieveAlg<ALE::Mesh<IndexType,ScalarType> >::closure(this, this->getArrowSection("orientation"), *e_iter);
            const typename coneArray::iterator cEnd    = closure->end();

            for(typename coneArray::iterator c_iter = closure->begin(); c_iter != cEnd; ++c_iter) {
              if (seen.find(*c_iter) != seen.end()) continue;
              if (this->getValue(label, *c_iter) == 1) {
                seen.insert(*c_iter);
                s->setFiberDimension(*c_iter, 0, f);
                s->addFiberDimension(*c_iter, -disc->getNumDof(this->depth(*c_iter)));
                if (debug > 1) {std::cout << "  cell: " << *c_iter << " dim: " << disc->getNumDof(this->depth(*c_iter)) << std::endl;}
              }
            }
          }
        }
      }
      // Process constraints
      f = 0;
      for(typename std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
        const Obj<ALE::Discretization>&    disc        = this->getDiscretization(*f_iter);
        const Obj<std::set<std::string> >  bcs         = disc->getBoundaryConditions();
        std::string                        excludeName = "exclude-"+*f_iter;

        for(typename std::set<std::string>::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter) {
          const Obj<ALE::BoundaryCondition>& bc       = disc->getBoundaryCondition(*bc_iter);
          const Obj<label_sequence>&         boundary = this->getLabelStratum(bc->getLabelName(), bc->getMarker());

          bcLabels.insert(bc->getLabelName());
          if (this->hasLabel(excludeName)) {
            const Obj<label_type>& label = this->getLabel(excludeName);

            for(typename label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
              if (!this->getValue(label, *e_iter)) {
                const int numDof = disc->getNumDof(this->depth(*e_iter));

                if (numDof) s->addConstraintDimension(*e_iter, numDof);
                if (numDof) s->setConstraintDimension(*e_iter, numDof, f);
              }
            }
          } else {
            for(typename label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
              const int numDof = disc->getNumDof(this->depth(*e_iter));

              if (numDof) s->addConstraintDimension(*e_iter, numDof);
              if (numDof) s->setConstraintDimension(*e_iter, numDof, f);
            }
          }
        }
      }
      return maxDof;
    };
    void calculateIndices() {
      // Should have an iterator over the whole tree
      Obj<names_type> discs = this->getDiscretizations();
      Obj<Mesh>       mesh  = this;
      const int       debug = this->debug();
      std::map<std::string, std::pair<int, int*> > indices;

      mesh.addRef();
      for(names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
        const Obj<Discretization>& disc = this->getDiscretization(*d_iter);

        indices[*d_iter] = std::pair<int, int*>(0, new int[disc->size(mesh)]);
        disc->setIndices(indices[*d_iter].second);
      }
      const Obj<label_sequence>& cells   = this->heightStratum(0);
      const Obj<coneArray>       closure = sieve_alg_type::closure(this, this->getArrowSection("orientation"), *cells->begin());
      const typename coneArray::iterator  end     = closure->end();
      int                        offset  = 0;

      if (debug > 1) {std::cout << "Closure for first element" << std::endl;}
      for(typename coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
        const int dim = this->depth(*cl_iter);

        if (debug > 1) {std::cout << "  point " << *cl_iter << " depth " << dim << std::endl;}
        for(typename names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
          const Obj<Discretization>& disc = this->getDiscretization(*d_iter);
          const int                  num  = disc->getNumDof(dim);

          if (debug > 1) {std::cout << "    disc " << disc->getName() << " numDof " << num << std::endl;}
          for(int o = 0; o < num; ++o) {
            indices[*d_iter].second[indices[*d_iter].first++] = offset++;
          }
        }
      }
      if (debug > 1) {
        for(typename names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
          const Obj<Discretization>& disc = this->getDiscretization(*d_iter);

          std::cout << "Discretization " << disc->getName() << " indices:";
          for(int i = 0; i < indices[*d_iter].first; ++i) {
            std::cout << " " << indices[*d_iter].second[i];
          }
          std::cout << std::endl;
        }
      }
    };
    void calculateIndicesExcluded(const Obj<real_section_type>& s, const Obj<names_type>& discs) {
      typedef std::map<std::string, std::pair<int, indexSet> > indices_type;
      const Obj<label_type>& indexLabel = this->createLabel("cellExclusion");
      const int debug  = this->debug();
      int       marker = 0;
      Obj<Mesh> mesh   = this;
      std::map<indices_type, int> indexMap;
      indices_type                indices;

      mesh.addRef();
      for(names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
        const Obj<Discretization>& disc = this->getDiscretization(*d_iter);
        const int                  size = disc->size(mesh);

        indices[*d_iter].second.resize(size);
      }
      const names_type::const_iterator dBegin = discs->begin();
      const names_type::const_iterator dEnd   = discs->end();
      std::set<point_type> seen;
      int f = 0;

      for(names_type::const_iterator f_iter = dBegin; f_iter != dEnd; ++f_iter, ++f) {
        std::string labelName = "exclude-"+*f_iter;

        if (this->hasLabel(labelName)) {
          const Obj<label_sequence>&     exclusion = this->getLabelStratum(labelName, 1);
          const typename label_sequence::iterator end       = exclusion->end();

          if (debug > 1) {std::cout << "Processing exclusion " << labelName << std::endl;}
          for(typename label_sequence::iterator e_iter = exclusion->begin(); e_iter != end; ++e_iter) {
            if (this->height(*e_iter)) continue;
            const Obj<coneArray>      closure = ALE::SieveAlg<ALE::Mesh<IndexType,ScalarType> >::closure(this, this->getArrowSection("orientation"), *e_iter);
            const typename coneArray::iterator clEnd   = closure->end();
            int                       offset  = 0;

            if (debug > 1) {std::cout << "  Closure for cell " << *e_iter << std::endl;}
            for(typename coneArray::iterator cl_iter = closure->begin(); cl_iter != clEnd; ++cl_iter) {
              int g = 0;

              if (debug > 1) {std::cout << "    point " << *cl_iter << std::endl;}
              for(typename names_type::const_iterator g_iter = dBegin; g_iter != dEnd; ++g_iter, ++g) {
                const int fDim = s->getFiberDimension(*cl_iter, g);

                if (debug > 1) {std::cout << "      disc " << *g_iter << " numDof " << fDim << std::endl;}
                for(int d = 0; d < fDim; ++d) {
                  indices[*g_iter].second[indices[*g_iter].first++] = offset++;
                }
              }
            }
            const typename std::map<indices_type, int>::iterator entry = indexMap.find(indices);

            if (debug > 1) {
              for(typename std::map<indices_type, int>::iterator i_iter = indexMap.begin(); i_iter != indexMap.end(); ++i_iter) {
                for(typename names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
                  std::cout << "Discretization (" << i_iter->second << ") " << *g_iter << " indices:";
                  for(int i = 0; i < ((indices_type) i_iter->first)[*g_iter].first; ++i) {
                    std::cout << " " << ((indices_type) i_iter->first)[*g_iter].second[i];
                  }
                  std::cout << std::endl;
                }
                std::cout << "Comparison: " << (indices == i_iter->first) << std::endl;
              }
              for(typename names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
                std::cout << "Discretization " << *g_iter << " indices:";
                for(int i = 0; i < indices[*g_iter].first; ++i) {
                  std::cout << " " << indices[*g_iter].second[i];
                }
                std::cout << std::endl;
              }
            }
            if (entry != indexMap.end()) {
              this->setValue(indexLabel, *e_iter, entry->second);
              if (debug > 1) {std::cout << "  Found existing indices with marker " << entry->second << std::endl;}
            } else {
              indexMap[indices] = ++marker;
              this->setValue(indexLabel, *e_iter, marker);
              if (debug > 1) {std::cout << "  Created new indices with marker " << marker << std::endl;}
            }
            for(names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
              indices[*g_iter].first  = 0;
              for(unsigned int i = 0; i < indices[*g_iter].second.size(); ++i) indices[*g_iter].second[i] = 0;
            }
          }
        }
      }
      if (debug > 1) {indexLabel->view("cellExclusion");}
      for(std::map<indices_type, int>::iterator i_iter = indexMap.begin(); i_iter != indexMap.end(); ++i_iter) {
        if (debug > 1) {std::cout << "Setting indices for marker " << i_iter->second << std::endl;}
        for(names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
          const Obj<Discretization>& disc = this->getDiscretization(*g_iter);
          const indexSet  indSet   = ((indices_type) i_iter->first)[*g_iter].second;
          const int       size     = indSet.size();
          int            *_indices = new int[size];

          if (debug > 1) {std::cout << "  field " << *g_iter << std::endl;}
          for(int i = 0; i < size; ++i) {
            _indices[i] = indSet[i];
            if (debug > 1) {std::cout << "    indices["<<i<<"] = " << _indices[i] << std::endl;}
          }
          disc->setIndices(_indices, i_iter->second);
        }
      }
    };
    void setupField(const Obj<real_section_type>& s, const int cellMarker = 2, const bool noUpdate = false) {
      const Obj<names_type>& discs  = this->getDiscretizations();
      const int              debug  = s->debug();
      names_type             bcLabels;
      int                    maxDof;

      maxDof = this->setFiberDimensions(s, discs, bcLabels);
      this->calculateIndices();
      this->calculateIndicesExcluded(s, discs);
      this->allocate(s);
      s->defaultConstraintDof();
      const Obj<label_type>& cellExclusion = this->getLabel("cellExclusion");

      if (debug > 1) {std::cout << "Setting boundary values" << std::endl;}
      for(names_type::const_iterator n_iter = bcLabels.begin(); n_iter != bcLabels.end(); ++n_iter) {
        const Obj<label_sequence>&     boundaryCells = this->getLabelStratum(*n_iter, cellMarker);
        const Obj<real_section_type>&  coordinates   = this->getRealSection("coordinates");
        const Obj<names_type>&         discs         = this->getDiscretizations();
        const point_type               firstCell     = *boundaryCells->begin();
        const int                      numFields     = discs->size();
        typename real_section_type::value_type *values = new typename real_section_type::value_type[this->sizeWithBC(s, firstCell)];
        int                           *dofs          = new int[maxDof];
        int                           *v             = new int[numFields];
        typename real_section_type::value_type *v0   = new typename real_section_type::value_type[this->getDimension()];
        typename real_section_type::value_type *J    = new typename real_section_type::value_type[this->getDimension()*this->getDimension()];
        typename real_section_type::value_type  detJ;

        for(typename label_sequence::iterator c_iter = boundaryCells->begin(); c_iter != boundaryCells->end(); ++c_iter) {
          const Obj<coneArray>      closure = sieve_alg_type::closure(this, this->getArrowSection("orientation"), *c_iter);
          const typename coneArray::iterator end     = closure->end();

          if (debug > 1) {std::cout << "  Boundary cell " << *c_iter << std::endl;}
          this->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
          for(int f = 0; f < numFields; ++f) v[f] = 0;
          for(typename coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
            const int cDim = s->getConstraintDimension(*cl_iter);
            int       off  = 0;
            int       f    = 0;
            int       i    = -1;

            if (debug > 1) {std::cout << "    point " << *cl_iter << std::endl;}
            if (cDim) {
              if (debug > 1) {std::cout << "      constrained excMarker: " << this->getValue(cellExclusion, *c_iter) << std::endl;}
              for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
                const Obj<ALE::Discretization>& disc    = this->getDiscretization(*f_iter);
                const Obj<names_type>           bcs     = disc->getBoundaryConditions();
                const int                       fDim    = s->getFiberDimension(*cl_iter, f);//disc->getNumDof(this->depth(*cl_iter));
                const int                      *indices = disc->getIndices(this->getValue(cellExclusion, *c_iter));
                int                             b       = 0;

                for(names_type::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter, ++b) {
                  const Obj<ALE::BoundaryCondition>& bc    = disc->getBoundaryCondition(*bc_iter);
                  const int                          value = this->getValue(this->getLabel(bc->getLabelName()), *cl_iter);

                  if (b > 0) v[f] -= fDim;
                  if (value == bc->getMarker()) {
                    if (debug > 1) {std::cout << "      field " << *f_iter << " marker " << value << std::endl;}
                    for(int d = 0; d < fDim; ++d, ++v[f]) {
                      dofs[++i] = off+d;
                      if (!noUpdate) values[indices[v[f]]] = (*bc->getDualIntegrator())(v0, J, v[f], bc->getFunction());
                      if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                    }
                    // Allow only one condition per point
                    ++b;
                    break;
                  } else {
                    if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
                    for(int d = 0; d < fDim; ++d, ++v[f]) {
                      values[indices[v[f]]] = 0.0;
                      if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                    }
                  }
                }
                if (b == 0) {
                  if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
                  for(int d = 0; d < fDim; ++d, ++v[f]) {
                    values[indices[v[f]]] = 0.0;
                    if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                  }
                }
                off += fDim;
              }
              if (i != cDim-1) {throw ALE::Exception("Invalid constraint initialization");}
              s->setConstraintDof(*cl_iter, dofs);
            } else {
              if (debug > 1) {std::cout << "      unconstrained" << std::endl;}
              for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
                const Obj<ALE::Discretization>& disc    = this->getDiscretization(*f_iter);
                const int                       fDim    = s->getFiberDimension(*cl_iter, f);//disc->getNumDof(this->depth(*cl_iter));
                const int                      *indices = disc->getIndices(this->getValue(cellExclusion, *c_iter));

                if (debug > 1) {std::cout << "      field " << *f_iter << std::endl;}
                for(int d = 0; d < fDim; ++d, ++v[f]) {
                  values[indices[v[f]]] = 0.0;
                  if (debug > 1) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                }
              }
            }
          }
          if (debug > 1) {
            const Obj<coneArray>      closure = sieve_alg_type::closure(this, this->getArrowSection("orientation"), *c_iter);
            const typename coneArray::iterator end     = closure->end();

            for(int f = 0; f < numFields; ++f) v[f] = 0;
            for(typename coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
              int f = 0;
              for(typename names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
                const Obj<ALE::Discretization>& disc    = this->getDiscretization(*f_iter);
                const int                       fDim    = s->getFiberDimension(*cl_iter, f);
                const int                      *indices = disc->getIndices(this->getValue(cellExclusion, *c_iter));

                for(int d = 0; d < fDim; ++d, ++v[f]) {
                  std::cout << "    "<<*f_iter<<"-value["<<indices[v[f]]<<"] " << values[indices[v[f]]] << std::endl;
                }
              }
            }
          }
          if (!noUpdate) {
            this->updateAll(s, *c_iter, values);
          }
        }
        delete [] dofs;
        delete [] values;
        delete [] v0;
        delete [] J;
      }
      if (debug > 1) {s->view("");}
    };
  public:
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
      }
      if (name == "") {
        PetscPrintf(comm, "viewing a Mesh\n");
      } else {
        PetscPrintf(comm, "viewing Mesh '%s'\n", name.c_str());
      }
      this->getSieve()->view("mesh sieve", comm);
      Obj<names_type> sections = this->getRealSections();

      for(names_type::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getRealSection(*name)->view(*name);
      }
      sections = this->getIntSections();
      for(names_type::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getIntSection(*name)->view(*name);
      }
      sections = this->getArrowSections();
      for(names_type::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getArrowSection(*name)->view(*name);
      }
    };
    template<typename value_type>
    static std::string printMatrix(const std::string& name, const int rows, const int cols, const value_type matrix[], const int rank = -1)
    {
      ostringstream output;
      ostringstream rankStr;

      if (rank >= 0) {
        rankStr << "[" << rank << "]";
      }
      output << rankStr.str() << name << " = " << std::endl;
      for(int r = 0; r < rows; r++) {
        if (r == 0) {
          output << rankStr.str() << " /";
        } else if (r == rows-1) {
          output << rankStr.str() << " \\";
        } else {
          output << rankStr.str() << " |";
        }
        for(int c = 0; c < cols; c++) {
          output << " " << matrix[r*cols+c];
        }
        if (r == 0) {
          output << " \\" << std::endl;
        } else if (r == rows-1) {
          output << " /" << std::endl;
        } else {
          output << " |" << std::endl;
        }
      }
      return output.str();
    }
  };
  template<typename Mesh>
  class MeshBuilder {
  public:
    typedef typename Mesh::real_section_type::value_type real;
  public:
    #undef __FUNCT__
    #define __FUNCT__ "createSquareBoundary"
    /*
      Simple square boundary:

     18--5-17--4--16
      |     |     |
      6    10     3
      |     |     |
     19-11-20--9--15
      |     |     |
      7     8     2
      |     |     |
     12--0-13--1--14
    */
    static Obj<Mesh> createSquareBoundary(const MPI_Comm comm, const real lower[], const real upper[], const int edges[], const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 1, debug);
      int       numVertices = (edges[0]+1)*(edges[1]+1);
      int       numEdges    = edges[0]*(edges[1]+1) + (edges[0]+1)*edges[1];
      real     *coords      = new real[numVertices*2];
      const Obj<typename Mesh::sieve_type> sieve    = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      typename Mesh::point_type           *vertices = new typename Mesh::point_type[numVertices];
      int                         order    = 0;

      mesh->setSieve(sieve);
      const Obj<typename Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        /* Create sieve and ordering */
        for(int v = numEdges; v < numEdges+numVertices; v++) {
          vertices[v-numEdges] = typename Mesh::point_type(v);
        }
        for(int vy = 0; vy <= edges[1]; vy++) {
          for(int ex = 0; ex < edges[0]; ex++) {
            typename Mesh::point_type edge(vy*edges[0] + ex);
            int vertex = vy*(edges[0]+1) + ex;

            sieve->addArrow(vertices[vertex+0], edge, order++);
            sieve->addArrow(vertices[vertex+1], edge, order++);
            if ((vy == 0) || (vy == edges[1])) {
              mesh->setValue(markers, edge, 1);
              mesh->setValue(markers, vertices[vertex], 1);
              if (ex == edges[0]-1) {
                mesh->setValue(markers, vertices[vertex+1], 1);
              }
            }
          }
        }
        for(int vx = 0; vx <= edges[0]; vx++) {
          for(int ey = 0; ey < edges[1]; ey++) {
            typename Mesh::point_type edge(vx*edges[1] + ey + edges[0]*(edges[1]+1));
            int vertex = ey*(edges[0]+1) + vx;

            sieve->addArrow(vertices[vertex],            edge, order++);
            sieve->addArrow(vertices[vertex+edges[0]+1], edge, order++);
            if ((vx == 0) || (vx == edges[0])) {
              mesh->setValue(markers, edge, 1);
              mesh->setValue(markers, vertices[vertex], 1);
              if (ey == edges[1]-1) {
                mesh->setValue(markers, vertices[vertex+edges[0]+1], 1);
              }
            }
          }
        }
      }
      mesh->stratify();
      for(int vy = 0; vy <= edges[1]; ++vy) {
        for(int vx = 0; vx <= edges[0]; ++vx) {
          coords[(vy*(edges[0]+1)+vx)*2+0] = lower[0] + ((upper[0] - lower[0])/edges[0])*vx;
          coords[(vy*(edges[0]+1)+vx)*2+1] = lower[1] + ((upper[1] - lower[1])/edges[1])*vy;
        }
      }
      delete [] vertices;
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      return mesh;
    };
    #undef __FUNCT__
    #define __FUNCT__ "createSquareBoundary"
    /*
      Simple square boundary:

     14--5-13--4--12
      |           |
      6           3
      |           |
     15           11
      |           |
      7           2
      |           |
      8--0--9--1--10
    */
    static Obj<Mesh> createSquareBoundary(const MPI_Comm comm, const real lower[], const real upper[], const int edges, const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 1, debug);
      int       numVertices = edges*4;
      int       numEdges    = edges*4;
      real     *coords      = new real[numVertices*2];
      const Obj<typename Mesh::sieve_type> sieve    = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      typename Mesh::point_type           *vertices = new typename Mesh::point_type[numVertices];

      mesh->setSieve(sieve);
      const Obj<typename Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        /* Create sieve and ordering */
        for(int v = numEdges; v < numEdges+numVertices; v++) {
          vertices[v-numEdges] = typename Mesh::point_type(v);
        }
        for(int e = 0; e < numEdges; ++e) {
          typename Mesh::point_type edge(e);
          int order = 0;

          sieve->addArrow(vertices[e],                 edge, order++);
          sieve->addArrow(vertices[(e+1)%numVertices], edge, order++);
          mesh->setValue(markers, edge, 2);
          mesh->setValue(markers, vertices[e], 1);
          mesh->setValue(markers, vertices[(e+1)%numVertices], 1);
        }
      }
      mesh->stratify();
      for(int v = 0; v < edges; ++v) {
        coords[(v+edges*0)*2+0] = lower[0] + ((upper[0] - lower[0])/edges)*v;
        coords[(v+edges*0)*2+1] = lower[1];
      }
      for(int v = 0; v < edges; ++v) {
        coords[(v+edges*1)*2+0] = upper[0];
        coords[(v+edges*1)*2+1] = lower[1] + ((upper[1] - lower[1])/edges)*v;
      }
      for(int v = 0; v < edges; ++v) {
        coords[(v+edges*2)*2+0] = upper[0] - ((upper[0] - lower[0])/edges)*v;
        coords[(v+edges*2)*2+1] = upper[1];
      }
      for(int v = 0; v < edges; ++v) {
        coords[(v+edges*3)*2+0] = lower[0];
        coords[(v+edges*3)*2+1] = upper[1] - ((upper[1] - lower[1])/edges)*v;
      }
      delete [] vertices;
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      // Build normals for cells
      const Obj<typename Mesh::real_section_type>& normals = mesh->getRealSection("normals");
      const Obj<typename Mesh::label_sequence>&    cells   = mesh->heightStratum(0);

      //normals->setChart(typename Mesh::real_section_type::chart_type(*std::min_element(cells->begin(), cells->end()),
      //                                                               *std::max_element(cells->begin(), cells->end())+1));
      normals->setFiberDimension(cells, mesh->getDimension()+1);
      mesh->allocate(normals);
      for(int e = 0; e < edges; ++e) {
        real normal[2] = {0.0, -1.0};
        normals->updatePoint(e+edges*0, normal);
      }
      for(int e = 0; e < edges; ++e) {
        real normal[2] = {1.0, 0.0};
        normals->updatePoint(e+edges*1, normal);
      }
      for(int e = 0; e < edges; ++e) {
        real normal[2] = {0.0, 1.0};
        normals->updatePoint(e+edges*2, normal);
      }
      for(int e = 0; e < edges; ++e) {
        real normal[2] = {-1.0, 0.0};
        normals->updatePoint(e+edges*3, normal);
      }
      return mesh;
    };
    #undef __FUNCT__
    #define __FUNCT__ "createParticleInSquareBoundary"
    /*
      Simple square boundary:

     18--5-17--4--16
      |     |     |
      6    10     3
      |     |     |
     19-11-20--9--15
      |     |     |
      7     8     2
      |     |     |
     12--0-13--1--14
    */
    static Obj<Mesh> createParticleInSquareBoundary(const MPI_Comm comm, const real lower[], const real upper[], const int edges[], const real radius, const int partEdges, const int debug = 0) {
      Obj<Mesh> mesh              = new Mesh(comm, 1, debug);
      const int numSquareVertices = (edges[0]+1)*(edges[1]+1);
      const int numVertices       = numSquareVertices + partEdges;
      const int numSquareEdges    = edges[0]*(edges[1]+1) + (edges[0]+1)*edges[1];
      const int numEdges          = numSquareEdges + partEdges;
      real   *coords            = new real[numVertices*2];
      const Obj<typename Mesh::sieve_type> sieve    = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      typename Mesh::point_type           *vertices = new typename Mesh::point_type[numVertices];
      int                         order    = 0;

      mesh->setSieve(sieve);
      const Obj<typename Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        /* Create sieve and ordering */
        for(int v = numEdges; v < numEdges+numVertices; v++) {
          vertices[v-numEdges] = typename Mesh::point_type(v);
        }
        // Make square
        for(int vy = 0; vy <= edges[1]; vy++) {
          for(int ex = 0; ex < edges[0]; ex++) {
            typename Mesh::point_type edge(vy*edges[0] + ex);
            int vertex = vy*(edges[0]+1) + ex;

            sieve->addArrow(vertices[vertex+0], edge, order++);
            sieve->addArrow(vertices[vertex+1], edge, order++);
            if ((vy == 0) || (vy == edges[1])) {
              mesh->setValue(markers, edge, 1);
              mesh->setValue(markers, vertices[vertex], 1);
              if (ex == edges[0]-1) {
                mesh->setValue(markers, vertices[vertex+1], 1);
              }
            }
          }
        }
        for(int vx = 0; vx <= edges[0]; vx++) {
          for(int ey = 0; ey < edges[1]; ey++) {
            typename Mesh::point_type edge(vx*edges[1] + ey + edges[0]*(edges[1]+1));
            int vertex = ey*(edges[0]+1) + vx;

            sieve->addArrow(vertices[vertex],            edge, order++);
            sieve->addArrow(vertices[vertex+edges[0]+1], edge, order++);
            if ((vx == 0) || (vx == edges[0])) {
              mesh->setValue(markers, edge, 1);
              mesh->setValue(markers, vertices[vertex], 1);
              if (ey == edges[1]-1) {
                mesh->setValue(markers, vertices[vertex+edges[0]+1], 1);
              }
            }
          }
        }
        // Make particle
        for(int ep = 0; ep < partEdges; ++ep) {
          typename Mesh::point_type edge(numSquareEdges + ep);
          const int vertexA = numSquareVertices + ep;
          const int vertexB = numSquareVertices + (ep+1)%partEdges;

          sieve->addArrow(vertices[vertexA], edge, order++);
          sieve->addArrow(vertices[vertexB], edge, order++);
          mesh->setValue(markers, edge, 2);
          mesh->setValue(markers, vertices[vertexA], 2);
          mesh->setValue(markers, vertices[vertexB], 2);
        }
      }
      mesh->stratify();
      for(int vy = 0; vy <= edges[1]; ++vy) {
        for(int vx = 0; vx <= edges[0]; ++vx) {
          coords[(vy*(edges[0]+1)+vx)*2+0] = lower[0] + ((upper[0] - lower[0])/edges[0])*vx;
          coords[(vy*(edges[0]+1)+vx)*2+1] = lower[1] + ((upper[1] - lower[1])/edges[1])*vy;
        }
      }
      const real centroidX = 0.5*(upper[0] + lower[0]);
      const real centroidY = 0.5*(upper[1] + lower[1]);
      for(int vp = 0; vp < partEdges; ++vp) {
        const real rad = 2.0*PETSC_PI*vp/partEdges;
        coords[(numSquareVertices+vp)*2+0] = centroidX + radius*cos(rad);
        coords[(numSquareVertices+vp)*2+1] = centroidY + radius*sin(rad);
      }
      delete [] vertices;
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      return mesh;
    };
    #undef __FUNCT__
    #define __FUNCT__ "createReentrantBoundary"
    /*
      Simple boundary with reentrant singularity:

     12--5-11
      |     |
      |     4
      |     |
      6    10--3--9
      |           |
      |           2
      |           |
      7-----1-----8
    */
    static Obj<Mesh> createReentrantBoundary(const MPI_Comm comm, const real lower[], const real upper[], real notchpercent[], const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 1, debug);
      int       numVertices = 6;
      int       numEdges    = numVertices;
      real   *coords      = new real[numVertices*2];
      const Obj<typename Mesh::sieve_type> sieve    = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      typename Mesh::point_type           *vertices = new typename Mesh::point_type[numVertices];

      mesh->setSieve(sieve);
      const Obj<typename Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        /* Create sieve and ordering */
        for (int b = 0; b < numVertices; b++) {
          sieve->addArrow(numEdges+b, b);
          sieve->addArrow(numEdges+b, (b+1)%numVertices);
          mesh->setValue(markers, b, 1);
          mesh->setValue(markers, b+numVertices, 1);
        }
        coords[0] = upper[0];
        coords[1] = lower[1];

        coords[2] = lower[0];
        coords[3] = lower[1];
        
        coords[4] = lower[0];
        coords[5] = notchpercent[1]*lower[1] + (1 - notchpercent[1])*upper[1];
        
        coords[6] = notchpercent[0]*upper[0] + (1 - notchpercent[0])*lower[0];
        coords[7] = notchpercent[1]*lower[1] + (1 - notchpercent[1])*upper[1];
        
        
        coords[8] = notchpercent[0]*upper[0] + (1 - notchpercent[0])*lower[0];
        coords[9] = upper[1];

        coords[10] = upper[0];
        coords[11] = upper[1];
        mesh->stratify();
      }
      delete [] vertices;
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      return mesh;
    }

    #undef __FUNCT__
    #define __FUNCT__ "createCircularReentrantBoundary"
    /*
      Circular boundary with reentrant singularity:

         ---1
      --    |
     -      |
     |      |
     |      0-----n
     |            |
     -           -
      --       --
        -------
    */
    static Obj<Mesh> createCircularReentrantBoundary(const MPI_Comm comm, const int segments, const real radius, const real arc_percent, const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 1, debug);
      int       numVertices = segments+2;
      int       numEdges    = numVertices;
      real   *coords      = new real[numVertices*2];
      const Obj<typename Mesh::sieve_type> sieve    = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      typename Mesh::point_type           *vertices = new typename Mesh::point_type[numVertices];

      mesh->setSieve(sieve);
      const Obj<typename Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        /* Create sieve and ordering */

        int startvertex = 1;
        if (arc_percent < 1.) {
          coords[0] = 0.;
          coords[1] = 0.;
        } else {
          numVertices = segments;
          numEdges = numVertices;
          startvertex = 0;
        }

        for (int b = 0; b < numVertices; b++) {
          sieve->addArrow(numEdges+b, b);
          sieve->addArrow(numEdges+b, (b+1)%numVertices);
          mesh->setValue(markers, b, 1);
          mesh->setValue(markers, b+numVertices, 1);
        }

        real anglestep = arc_percent*2.*3.14159265/((float)segments);

        for (int i = startvertex; i < numVertices; i++) {
          coords[2*i] = radius * sin(anglestep*(i-startvertex));
          coords[2*i+1] = radius*cos(anglestep*(i-startvertex));
        }
        mesh->stratify();
      }
      delete [] vertices;
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      return mesh;
    };
    #undef __FUNCT__
    #define __FUNCT__ "createAnnularBoundary"
    static Obj<Mesh> createAnnularBoundary(const MPI_Comm comm, const int segments, const real centers[4], const real radii[2], const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 1, debug);
      int       numVertices = segments*2;
      int       numEdges    = numVertices;
      real   *coords      = new real[numVertices*2];
      const Obj<typename Mesh::sieve_type> sieve    = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      typename Mesh::point_type           *vertices = new typename Mesh::point_type[numVertices];

      mesh->setSieve(sieve);
      const Obj<typename Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        for (int e = 0; e < segments; ++e) {
          sieve->addArrow(numEdges+e,              e);
          sieve->addArrow(numEdges+(e+1)%segments, e);
          sieve->addArrow(numEdges+segments+e,              e+segments);
          sieve->addArrow(numEdges+segments+(e+1)%segments, e+segments);
          mesh->setValue(markers, e,          1);
          mesh->setValue(markers, e+segments, 1);
          mesh->setValue(markers, e+numEdges,          1);
          mesh->setValue(markers, e+numEdges+segments, 1);
        }
        const real anglestep = 2.0*M_PI/segments;

        for (int v = 0; v < segments; ++v) {
          coords[v*2]              = centers[0] + radii[0]*cos(anglestep*v);
          coords[v*2+1]            = centers[1] + radii[0]*sin(anglestep*v);
          coords[(v+segments)*2]   = centers[2] + radii[1]*cos(anglestep*v);
          coords[(v+segments)*2+1] = centers[3] + radii[1]*sin(anglestep*v);
        }
        mesh->addHole(&centers[2]);
      }
      mesh->stratify();
      delete [] vertices;
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      return mesh;
    };
    #undef __FUNCT__
    #define __FUNCT__ "createCubeBoundary"
    /*
      Simple cubic boundary:

     30----31-----32
      |     |     |
      |  3  |  2  |
      |     |     |
     27----28-----29
      |     |     |
      |  0  |  1  |
      |     |     |
     24----25-----26
    */
    static Obj<Mesh> createCubeBoundary(const MPI_Comm comm, const real lower[], const real upper[], const int faces[], const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 2, debug);
      int       numVertices = (faces[0]+1)*(faces[1]+1)*(faces[2]+1);
      int       numFaces    = 6;
      real   *coords      = new real[numVertices*3];
      const Obj<typename Mesh::sieve_type> sieve    = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      typename Mesh::point_type           *vertices = new typename Mesh::point_type[numVertices];
      int                         order    = 0;

      mesh->setSieve(sieve);
      const Obj<typename Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        /* Create sieve and ordering */
        for(int v = numFaces; v < numFaces+numVertices; v++) {
          vertices[v-numFaces] = typename Mesh::point_type(v);
          mesh->setValue(markers, vertices[v-numFaces], 1);
        }
        {
          // Side 0 (Front)
          typename Mesh::point_type face(0);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 1 (Back)
          typename Mesh::point_type face(1);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 2 (Bottom)
          typename Mesh::point_type face(2);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[0], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 3 (Top)
          typename Mesh::point_type face(3);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 4 (Left)
          typename Mesh::point_type face(4);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 5 (Right)
          typename Mesh::point_type face(5);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          mesh->setValue(markers, face, 1);
        }
      }
      mesh->stratify();
#if 0
      for(int vz = 0; vz <= edges[2]; ++vz) {
        for(int vy = 0; vy <= edges[1]; ++vy) {
          for(int vx = 0; vx <= edges[0]; ++vx) {
            coords[((vz*(edges[1]+1)+vy)*(edges[0]+1)+vx)*2+0] = lower[0] + ((upper[0] - lower[0])/faces[0])*vx;
            coords[((vz*(edges[1]+1)+vy)*(edges[0]+1)+vx)*2+1] = lower[1] + ((upper[1] - lower[1])/faces[1])*vy;
            coords[((vz*(edges[1]+1)+vy)*(edges[0]+1)+vx)*2+2] = lower[2] + ((upper[2] - lower[2])/faces[2])*vz;
          }
        }
      }
#else
      coords[0*3+0] = lower[0];
      coords[0*3+1] = lower[1];
      coords[0*3+2] = upper[2];
      coords[1*3+0] = upper[0];
      coords[1*3+1] = lower[1];
      coords[1*3+2] = upper[2];
      coords[2*3+0] = upper[0];
      coords[2*3+1] = upper[1];
      coords[2*3+2] = upper[2];
      coords[3*3+0] = lower[0];
      coords[3*3+1] = upper[1];
      coords[3*3+2] = upper[2];
      coords[4*3+0] = lower[0];
      coords[4*3+1] = lower[1];
      coords[4*3+2] = lower[2];
      coords[5*3+0] = upper[0];
      coords[5*3+1] = lower[1];
      coords[5*3+2] = lower[2];
      coords[6*3+0] = upper[0];
      coords[6*3+1] = upper[1];
      coords[6*3+2] = lower[2];
      coords[7*3+0] = lower[0];
      coords[7*3+1] = upper[1];
      coords[7*3+2] = lower[2];
#endif
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      return mesh;
    };

    // Creates a triangular prism boundary
    static Obj<Mesh> createPrismBoundary(const MPI_Comm comm, const real lower[], const real upper[], const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 2, debug);
      int       numVertices = 6;
      int       numFaces    = 5;
      real   *coords      = new real[numVertices*3];
      const Obj<typename Mesh::sieve_type> sieve    = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      typename Mesh::point_type           *vertices = new typename Mesh::point_type[numVertices];
      int                         order    = 0;

      mesh->setSieve(sieve);
      const Obj<typename Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        /* Create sieve and ordering */
        for(int v = numFaces; v < numFaces+numVertices; v++) {
          vertices[v-numFaces] = typename Mesh::point_type(v);
          mesh->setValue(markers, vertices[v-numFaces], 1);
        }
        {
          // Side 0 (Top)
          typename Mesh::point_type face(0);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 1 (Bottom)
          typename Mesh::point_type face(1);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[4], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 2 (Front)
          typename Mesh::point_type face(2);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[1], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 3 (Left)
          typename Mesh::point_type face(3);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 4 (Right)
          typename Mesh::point_type face(4);
          sieve->addArrow(vertices[2], face, order++);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[0], face, order++);
          mesh->setValue(markers, face, 1);
        }
      }
      mesh->stratify();
      coords[0*3+0] = lower[0];
      coords[0*3+1] = lower[1];
      coords[0*3+2] = upper[2];
      coords[1*3+0] = upper[0];
      coords[1*3+1] = lower[1];
      coords[1*3+2] = upper[2];
      coords[2*3+0] = upper[0];
      coords[2*3+1] = upper[1];
      coords[2*3+2] = upper[2];
      coords[3*3+0] = lower[0];
      coords[3*3+1] = upper[1];
      coords[3*3+2] = upper[2];
      coords[4*3+0] = lower[0];
      coords[4*3+1] = lower[1];
      coords[4*3+2] = lower[2];
      coords[5*3+0] = upper[0];
      coords[5*3+1] = lower[1];
      coords[5*3+2] = lower[2];
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      return mesh;
    };

    #undef __FUNCT__
    #define __FUNCT__ "createFicheraCornerBoundary"
    /*    v0
         / \
        /   \
    2  /  4  \  1
      /       \
     /   v12   \
  v6|\   /|\   /|v5
    | \v8 | v7/ |          z  
    |  |7 |8 |  |          | 
    |  |v13\ |  |  <-v4   / \
    | v9/ 9 \v10|        x   y
 v1 | 5 \   / 6 |v2
     \   \ /   /
      \  v11  /
       \  |  /
     3  \ | /
         \|/
          v3
    */
    static Obj<Mesh> createFicheraCornerBoundary(const MPI_Comm comm, const real lower[], const real upper[], const real offset[], const int debug = 0) {
      Obj<Mesh> mesh            = new Mesh(comm, 2, debug);
      const int nVertices = 14;
      const int nFaces = 12;
      real ilower[3];
      ilower[0] = lower[0]*(1. - offset[0]) + upper[0]*offset[0];
      ilower[1] = lower[1]*(1. - offset[1]) + upper[1]*offset[1];
      ilower[2] = lower[2]*(1. - offset[2]) + upper[2]*offset[2];
      real coords[nVertices*3];
      //outer square-triplet
      coords[0*3+0] = lower[0];
      coords[0*3+1] = lower[1];
      coords[0*3+2] = upper[2];
      coords[1*3+0] = upper[0];
      coords[1*3+1] = lower[1];
      coords[1*3+2] = lower[2];
      coords[2*3+0] = lower[0];
      coords[2*3+1] = upper[1];
      coords[2*3+2] = lower[2];
      coords[3*3+0] = upper[0];
      coords[3*3+1] = upper[1];
      coords[3*3+2] = lower[2];
      coords[4*3+0] = lower[0];
      coords[4*3+1] = lower[1];
      coords[4*3+2] = lower[2];
      coords[5*3+0] = lower[0];
      coords[5*3+1] = upper[1];
      coords[5*3+2] = upper[2];
      coords[6*3+0] = upper[0];
      coords[6*3+1] = lower[1];
      coords[6*3+2] = upper[2];

      //inner square-triplet
      coords[7*3+0] = ilower[0];
      coords[7*3+1] = upper[1];
      coords[7*3+2] = upper[2];
      coords[8*3+0] = upper[0];
      coords[8*3+1] = ilower[1];
      coords[8*3+2] = upper[2];
      coords[9*3+0] = upper[0];
      coords[9*3+1] = ilower[1];
      coords[9*3+2] = ilower[2];
      coords[10*3+0] = ilower[0];
      coords[10*3+1] = upper[1];
      coords[10*3+2] = ilower[2];
      coords[11*3+0] = upper[0];
      coords[11*3+1] = upper[1];
      coords[11*3+2] = ilower[2];
      coords[12*3+0] = ilower[0];
      coords[12*3+1] = ilower[1];
      coords[12*3+2] = upper[2];
      coords[13*3+0] = ilower[0];
      coords[13*3+1] = ilower[1];
      coords[13*3+2] = ilower[2];

 
      const Obj<typename Mesh::sieve_type> sieve    = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      mesh->setSieve(sieve);
      typename Mesh::point_type p[nVertices];
      typename Mesh::point_type f[nFaces];
      const Obj<typename Mesh::label_type>& markers = mesh->createLabel("marker");
      for (int i = 0; i < nVertices; i++) {
        p[i] = typename Mesh::point_type(i+nFaces);
        mesh->setValue(markers, p[i], 1);
      }
      for (int i = 0; i < nFaces; i++) {
        f[i] = typename Mesh::point_type(i);
      }
      int order = 0; 
     //assemble the larger square sides
      sieve->addArrow(p[0], f[0], order++);
      sieve->addArrow(p[5], f[0], order++);
      sieve->addArrow(p[2], f[0], order++);
      sieve->addArrow(p[4], f[0], order++);
      mesh->setValue(markers, f[0], 1);      

      sieve->addArrow(p[0], f[1], order++);
      sieve->addArrow(p[4], f[1], order++);
      sieve->addArrow(p[1], f[1], order++);
      sieve->addArrow(p[6], f[1], order++);
      mesh->setValue(markers, f[1], 1);      

      sieve->addArrow(p[4], f[2], order++);
      sieve->addArrow(p[1], f[2], order++);
      sieve->addArrow(p[3], f[2], order++);
      sieve->addArrow(p[2], f[2], order++);
      mesh->setValue(markers, f[2], 1);
     
      //assemble the L-shaped sides

      sieve->addArrow(p[0], f[3], order++);
      sieve->addArrow(p[12], f[3], order++);
      sieve->addArrow(p[7], f[3], order++);
      sieve->addArrow(p[5], f[3], order++);
      mesh->setValue(markers, f[3], 1);

      sieve->addArrow(p[0], f[4], order++);
      sieve->addArrow(p[12],f[4], order++);
      sieve->addArrow(p[8], f[4], order++);
      sieve->addArrow(p[6], f[4], order++);
      mesh->setValue(markers, f[4], 1);

      sieve->addArrow(p[9], f[5], order++);
      sieve->addArrow(p[1], f[5], order++);
      sieve->addArrow(p[3], f[5], order++);
      sieve->addArrow(p[11], f[5], order++);
      mesh->setValue(markers, f[5], 1);

      sieve->addArrow(p[9], f[6], order++);
      sieve->addArrow(p[1], f[6], order++);
      sieve->addArrow(p[6], f[6], order++);
      sieve->addArrow(p[8], f[6], order++);
      mesh->setValue(markers, f[6], 1);

      sieve->addArrow(p[10], f[7], order++);
      sieve->addArrow(p[2], f[7], order++);
      sieve->addArrow(p[5], f[7], order++);
      sieve->addArrow(p[7], f[7], order++);
      mesh->setValue(markers, f[7], 1);

      sieve->addArrow(p[10], f[8], order++);
      sieve->addArrow(p[2], f[8], order++);
      sieve->addArrow(p[3], f[8], order++);
      sieve->addArrow(p[11], f[8], order++);
      mesh->setValue(markers, f[8], 1);

      //assemble the smaller square sides

      sieve->addArrow(p[13], f[9], order++);
      sieve->addArrow(p[10], f[9], order++);
      sieve->addArrow(p[11], f[9], order++);
      sieve->addArrow(p[9], f[9], order++);
      mesh->setValue(markers, f[9], 1);

      sieve->addArrow(p[12], f[10], order++);
      sieve->addArrow(p[7], f[10], order++);
      sieve->addArrow(p[10], f[10], order++);
      sieve->addArrow(p[13], f[10], order++);
      mesh->setValue(markers, f[10], 1);

      sieve->addArrow(p[8], f[11], order++);
      sieve->addArrow(p[12], f[11], order++);
      sieve->addArrow(p[13], f[11], order++);
      sieve->addArrow(p[9], f[11], order++);
      mesh->setValue(markers, f[11], 1);

      mesh->stratify();
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      Obj<typename Mesh::real_section_type> coordinates = mesh->getRealSection("coordinates");
      //coordinates->view("coordinates");
      return mesh;

    }

    #undef __FUNCT__
    #define __FUNCT__ "createSphereBoundary"
    /*
      //"sphere" out a cube 

    */
#if 0
    static Obj<Mesh> createSphereBoundary(const MPI_Comm comm, const real radius, const int refinement, const int debug = 0) {
      Obj<Mesh> m = new Mesh(comm, 2, debug);
      Obj<Mesh::sieve_type> s = new Mesh::sieve_type(comm, debug);
      m->setSieve(s);
      Mesh::point_type p = 0;
      int nVertices = 8+12*(refinement)+6*(refinement)*(refinement);
      Mesh::point_type vertices[nVertices];
      real coords[3*nVertices];
      int nCells = 6*2*(refinement+1)*(refinement+1);
      real delta = 2./((real)(refinement+1));
      Mesh::point_type cells[nCells];
      for (int i = 0; i < nCells; i++) {
        cells[i] = p;
        p++;
      }
      for (int i = 0; i < nVertices; i++) {
        vertices[i] = p;
        p++;
      }
      //set up the corners;
      //lll
      coords[0*3+0] = -1.;
      coords[0*3+1] = -1.;
      coords[0*3+2] = -1.;
      //llh
      coords[1*3+0] = -1.;
      coords[1*3+1] = -1.;
      coords[1*3+2] = 1.;
      //lhh
      coords[2*3+0] = -1.;
      coords[2*3+1] = 1.;
      coords[2*3+2] = 1.;
      //lhl
      coords[3*3+0] = -1.;
      coords[3*3+1] = 1.;
      coords[3*3+2] = -1.;
      //hhl
      coords[4*3+0] = 1.;
      coords[4*3+1] = 1.;
      coords[4*3+2] = -1.;
      //hhh
      coords[5*3+0] = 1.;
      coords[5*3+1] = 1.;
      coords[5*3+2] = 1.;
      //hlh
      coords[6*3+0] = 1.;
      coords[6*3+1] = -1.;
      coords[6*3+2] = 1.;
      //hll
      coords[7*3+0] = 1.;
      coords[7*3+1] = -1.;
      coords[7*3+2] = -1.;
      //set up the edges (always go low to high)
      //xll
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+0*refinement+i)+0] = -1. + delta*i;
	coords[3*(8+0*refinement+i)+1] = -1.;
        coords[3*(8+0*refinement+i)+2] = -1.;
      }
      //xlh
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+1*refinement+i)+0] = -1. + delta*i;
	coords[3*(8+1*refinement+i)+1] = -1.;
        coords[3*(8+1*refinement+i)+2] = 1.;
      }
      //xhh
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+2*refinement+i)+0] = -1. + delta*i;
	coords[3*(8+2*refinement+i)+1] = 1.;
        coords[3*(8+2*refinement+i)+2] = 1.;
      }
      //xhl
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+3*refinement+i)+0] = -1. + delta*i;
	coords[3*(8+3*refinement+i)+1] = 1.;
        coords[3*(8+3*refinement+i)+2] = -1.;
      }
      //lxl
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+4*refinement+i)+0] = -1.;
	coords[3*(8+4*refinement+i)+1] = -1. + delta*i;
        coords[3*(8+4*refinement+i)+2] = -1.;
      }
      //lxh
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+5*refinement+i)+0] = -1.;
	coords[3*(8+5*refinement+i)+1] = -1. + delta*i;
        coords[3*(8+5*refinement+i)+2] = 1.;
      }
      //hxh
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+6*refinement+i)+0] = 1.;
	coords[3*(8+6*refinement+i)+1] = -1. + delta*i;
        coords[3*(8+6*refinement+i)+2] = 1.;
      }
      //hxl
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+7*refinement+i)+0] = 1.;
	coords[3*(8+7*refinement+i)+1] = -1. + delta*i;
        coords[3*(8+7*refinement+i)+2] = -1.;
      }
      //llx
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+8*refinement+i)+0] = -1.;
	coords[3*(8+8*refinement+i)+1] = -1.;
        coords[3*(8+8*refinement+i)+2] = -1. + delta*i;
      }
      //lhx
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+9*refinement+i)+0] = -1.;
	coords[3*(8+9*refinement+i)+1] = 1.;
        coords[3*(8+9*refinement+i)+2] = -1. + delta*i;
      }
      //hhx
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+10*refinement+i)+0] = 1.;
	coords[3*(8+10*refinement+i)+1] = 1.;
        coords[3*(8+10*refinement+i)+2] = -1. + delta*i;
      }
      //hlx
      for (int i = 0; i < refinement; i++) {
        coords[3*(8+11*refinement+i)+0] = 1.;
	coords[3*(8+11*refinement+i)+1] = -1.;
        coords[3*(8+11*refinement+i)+2] = -1. + delta*i;
      }
      //set up the faces
      //lxx
      for (int i = 0; i < refinement; i++) for (int j = 0; j < refinement; j++) {
        coords[3*(8+12*refinement+0*refinement*refinement+i*refinement+j)+0] = -1.;
	coords[3*(8+12*refinement+0*refinement*refinement+i*refinement+j)+1] = -1. + delta*j;
        coords[3*(8+12*refinement+0*refinement*refinement+i*refinement+j)+2] = -1. + delta*i;
      }
      //hxx 
      for (int i = 0; i < refinement; i++) for (int j = 0; j < refinement; j++) {
        coords[3*(8+12*refinement+1*refinement*refinement+i*refinement+j)+0] = 1.;
	coords[3*(8+12*refinement+1*refinement*refinement+i*refinement+j)+1] = -1. + delta*j;
        coords[3*(8+12*refinement+1*refinement*refinement+i*refinement+j)+2] = -1. + delta*i;
      }
      //xlx
      for (int i = 0; i < refinement; i++) for (int j = 0; j < refinement; j++) {
        coords[3*(8+12*refinement+2*refinement*refinement+i*refinement+j)+0] = -1. + delta*j;
	coords[3*(8+12*refinement+2*refinement*refinement+i*refinement+j)+1] = -1.;
        coords[3*(8+12*refinement+2*refinement*refinement+i*refinement+j)+2] = -1. + delta*i;
      }
      //xhx
      for (int i = 0; i < refinement; i++) for (int j = 0; j < refinement; j++) {
        coords[3*(8+12*refinement+3*refinement*refinement+i*refinement+j)+0] = -1. + delta*j;
	coords[3*(8+12*refinement+3*refinement*refinement+i*refinement+j)+1] = 1.;
        coords[3*(8+12*refinement+3*refinement*refinement+i*refinement+j)+2] = -1. + delta*i;
      }
      //xxl
      for (int i = 0; i < refinement; i++) for (int j = 0; j < refinement; j++) {
        coords[3*(8+12*refinement+4*refinement*refinement+i*refinement+j)+0] = -1.;
	coords[3*(8+12*refinement+4*refinement*refinement+i*refinement+j)+1] = -1. + delta*j;
        coords[3*(8+12*refinement+4*refinement*refinement+i*refinement+j)+2] = -1. + delta*i;
      }
      //xxh
      for (int i = 0; i < refinement; i++) for (int j = 0; j < refinement; j++) {
        coords[3*(8+12*refinement+5*refinement*refinement+i*refinement+j)+0] = 1.;
	coords[3*(8+12*refinement+5*refinement*refinement+i*refinement+j)+1] = -1. + delta*j;
        coords[3*(8+12*refinement+5*refinement*refinement+i*refinement+j)+2] = -1. + delta*i;
      }
      //stitch the corners up with the edges and the faces
      
      //stitch the edges to the faces
      //fill in the faces
      int face_offset = 8 + 12*refinement;
      for (int i = 0; i < 6; i++) for (int j = 0; j < refinement; j++) for (int k = 0; k < refinement; k++) {
        //build each square doublet
      }
    }

#endif

    #undef __FUNCT__
    #define __FUNCT__ "createParticleInCubeBoundary"
    /*
      Simple cubic boundary:

     30----31-----32
      |     |     |
      |  3  |  2  |
      |     |     |
     27----28-----29
      |     |     |
      |  0  |  1  |
      |     |     |
     24----25-----26
    */
    static Obj<Mesh> createParticleInCubeBoundary(const MPI_Comm comm, const real lower[], const real upper[], const int faces[], const real radius, const int thetaEdges, const int phiSlices, const int debug = 0) {
      Obj<Mesh> mesh            = new Mesh(comm, 2, debug);
      const int numCubeVertices = (faces[0]+1)*(faces[1]+1)*(faces[2]+1);
      const int numPartVertices = (thetaEdges - 1)*phiSlices + 2;
      const int numVertices     = numCubeVertices + numPartVertices;
      const int numCubeFaces    = 6;
      const int numFaces        = numCubeFaces + thetaEdges*phiSlices;
      real   *coords          = new real[numVertices*3];
      const Obj<typename Mesh::sieve_type> sieve    = new typename Mesh::sieve_type(mesh->comm(), mesh->debug());
      typename Mesh::point_type           *vertices = new typename Mesh::point_type[numVertices];
      int                         order    = 0;

      mesh->setSieve(sieve);
      const Obj<typename Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        // Make cube
        for(int v = numFaces; v < numFaces+numVertices; v++) {
          vertices[v-numFaces] = typename Mesh::point_type(v);
          mesh->setValue(markers, vertices[v-numFaces], 1);
        }
        {
          // Side 0 (Front)
          typename Mesh::point_type face(0);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 1 (Back)
          typename Mesh::point_type face(1);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 2 (Bottom)
          typename Mesh::point_type face(2);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[0], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 3 (Top)
          typename Mesh::point_type face(3);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 4 (Left)
          typename Mesh::point_type face(4);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 5 (Right)
          typename Mesh::point_type face(5);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          mesh->setValue(markers, face, 1);
        }
        // Make particle
        for(int s = 0; s < phiSlices; ++s) {
          for(int ep = 0; ep < thetaEdges; ++ep) {
            // Vertices on each slice are 0..thetaEdges
            typename Mesh::point_type face(numCubeFaces + s*thetaEdges + ep);
            int vertexA = numCubeVertices + ep + 0 +     s*(thetaEdges+1);
            int vertexB = numCubeVertices + ep + 1 +     s*(thetaEdges+1);
            int vertexC = numCubeVertices + (ep + 1 + (s+1)*(thetaEdges+1))%((thetaEdges+1)*phiSlices);
            int vertexD = numCubeVertices + (ep + 0 + (s+1)*(thetaEdges+1))%((thetaEdges+1)*phiSlices);
            const int correction1 = (s > 0)*((s-1)*2 + 1);
            const int correction2 = (s < phiSlices-1)*(s*2 + 1);

            if ((vertexA - numCubeVertices)%(thetaEdges+1) == 0) {
              vertexA = vertexD = numCubeVertices;
              vertexB -= correction1;
              vertexC -= correction2;
            } else if ((vertexB - numCubeVertices)%(thetaEdges+1) == thetaEdges) {
              vertexA -= correction1;
              vertexD -= correction2;
              vertexB = vertexC = numCubeVertices + thetaEdges;
            } else {
              vertexA -= correction1;
              vertexB -= correction1;
              vertexC -= correction2;
              vertexD -= correction2;
            }
            if ((vertexA >= numVertices) || (vertexB >= numVertices) || (vertexC >= numVertices) || (vertexD >= numVertices)) {
              throw ALE::Exception("Bad vertex");
            }
            sieve->addArrow(vertices[vertexA], face, order++);
            sieve->addArrow(vertices[vertexB], face, order++);
            if (vertexB != vertexC) sieve->addArrow(vertices[vertexC], face, order++);
            if (vertexA != vertexD) sieve->addArrow(vertices[vertexD], face, order++);
            mesh->setValue(markers, face, 2);
            mesh->setValue(markers, vertices[vertexA], 2);
            mesh->setValue(markers, vertices[vertexB], 2);
            if (vertexB != vertexC) mesh->setValue(markers, vertices[vertexC], 2);
            if (vertexA != vertexD) mesh->setValue(markers, vertices[vertexD], 2);
          }
        }
      }
      mesh->stratify();
#if 0
      for(int vz = 0; vz <= edges[2]; ++vz) {
        for(int vy = 0; vy <= edges[1]; ++vy) {
          for(int vx = 0; vx <= edges[0]; ++vx) {
            coords[((vz*(edges[1]+1)+vy)*(edges[0]+1)+vx)*2+0] = lower[0] + ((upper[0] - lower[0])/faces[0])*vx;
            coords[((vz*(edges[1]+1)+vy)*(edges[0]+1)+vx)*2+1] = lower[1] + ((upper[1] - lower[1])/faces[1])*vy;
            coords[((vz*(edges[1]+1)+vy)*(edges[0]+1)+vx)*2+2] = lower[2] + ((upper[2] - lower[2])/faces[2])*vz;
          }
        }
      }
#else
      coords[0*3+0] = lower[0];
      coords[0*3+1] = lower[1];
      coords[0*3+2] = upper[2];
      coords[1*3+0] = upper[0];
      coords[1*3+1] = lower[1];
      coords[1*3+2] = upper[2];
      coords[2*3+0] = upper[0];
      coords[2*3+1] = upper[1];
      coords[2*3+2] = upper[2];
      coords[3*3+0] = lower[0];
      coords[3*3+1] = upper[1];
      coords[3*3+2] = upper[2];
      coords[4*3+0] = lower[0];
      coords[4*3+1] = lower[1];
      coords[4*3+2] = lower[2];
      coords[5*3+0] = upper[0];
      coords[5*3+1] = lower[1];
      coords[5*3+2] = lower[2];
      coords[6*3+0] = upper[0];
      coords[6*3+1] = upper[1];
      coords[6*3+2] = lower[2];
      coords[7*3+0] = lower[0];
      coords[7*3+1] = upper[1];
      coords[7*3+2] = lower[2];
#endif
      const real centroidX = 0.5*(upper[0] + lower[0]);
      const real centroidY = 0.5*(upper[1] + lower[1]);
      const real centroidZ = 0.5*(upper[2] + lower[2]);
      for(int s = 0; s < phiSlices; ++s) {
        for(int v = 0; v <= thetaEdges; ++v) {
          int          vertex  = numCubeVertices + v + s*(thetaEdges+1);
          const real theta   = v*(PETSC_PI/thetaEdges);
          const real phi     = s*(2.0*PETSC_PI/phiSlices);
          const int correction = (s > 0)*((s-1)*2 + 1);

          if ((vertex- numCubeVertices)%(thetaEdges+1) == 0) {
            vertex = numCubeVertices;
          } else if ((vertex - numCubeVertices)%(thetaEdges+1) == thetaEdges) {
            vertex = numCubeVertices + thetaEdges;
          } else {
            vertex -= correction;
          }
          coords[vertex*3+0] = centroidX + radius*sin(theta)*cos(phi);
          coords[vertex*3+1] = centroidY + radius*sin(theta)*sin(phi);
          coords[vertex*3+2] = centroidZ + radius*cos(theta);
        }
      }
      delete [] vertices;
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      return mesh;
    };

#if 0 // WORKING ON REDESIGN WITHIN PYLITH SOURCE TREE
    template<typename MeshType, typename EdgeType>
    class CellRefiner {
    public:
      typedef typename MeshType::point_type point_type;
      typedef EdgeType                      edge_type;
      typedef typename std::map<edge_type, point_type> edge_map_type;
      typedef enum {LINE, LINE_LAGRANGE, TRIANGLE, QUADRILATERAL, TETRAHEDRON, HEXAHEDRON, TRIANGULAR_PRISM, TRIANGULAR_PRISM_LAGRANGE, HEXAHEDRON_LAGRANGE} CellType;
    protected:
      const MeshType&     _mesh;
      int           _dim;
      point_type    _vertexOffset;
      edge_map_type _edge2vertex;
    public:
      CellRefiner(MeshType& mesh) : _mesh(mesh) {
        _dim = _mesh.getDimension();
      };
      ~CellRefiner() {};
    protected:
      CellType getCellType(const point_type cell) {
        const int corners = _mesh.getSieve()->getConeSize(cell);
	switch (_dim) {
	  return LINE;
	case 2:
	  switch (corners) {
	  case 3:
	    return TRIANGLE;
	  case 4:
	    throw ALE::Exception("Not implemented.");
	    return QUADRILATERAL;
	  case 6:
	    return LINE_LAGRANGE;
	  case 0: {
	    std::ostringstream msg;
	    std::cerr << "Internal error. Cone size for mesh point " << cell << " is zero. May be a vertex.";
	    assert(0);
	    throw ALE::Exception("Could not determine 2-D cell type.");
	  } // case 0
	  default : {
	    std::ostringstream msg;
	    std::cerr << "Internal error. Unknown cone size for mesh point " << cell << ". Unknown cell type.";
	    assert(0);
	    throw ALE::Exception("Could not determine 2-D cell type.");
	  } // default
	  } // switch
	case 3:
	  switch (corners) {
	  case 4:
	    return TETRAHEDRON;
	  case 6:
            return TRIANGULAR_PRISM;
	  case 9:
            return TRIANGULAR_PRISM_LAGRANGE;
	  case 12:
	    throw ALE::Exception("Not implemented.");
            return HEXAHEDRON_LAGRANGE;
	  case 0: {
	    std::ostringstream msg;
	    std::cerr << "Internal error. Cone size for mesh point " << cell << " is zero. May be a vertex.";
	    assert(0);
	    throw ALE::Exception("Could not determine 3-D cell type.");
	  } // case 0
	  default : {
	    std::ostringstream msg;
	    std::cerr << "Internal error. Unknown cone size for mesh point " << cell << ". Unknown cell type.";
	    assert(0);
	    throw ALE::Exception("Could not determine 3-D cell type.");
	  } // default
	  } // switch
	} // switch
      };
      void getEdges_TRIANGLE(const int coneSize, const point_type cone[],  int *numEdges, const edge_type **edges) {
        static edge_type triEdges[3];

        assert(coneSize == 3);
        triEdges[0] = edge_type(std::min(cone[0], cone[1]), std::max(cone[0], cone[1]));
        triEdges[1] = edge_type(std::min(cone[1], cone[2]), std::max(cone[1], cone[2]));
        triEdges[2] = edge_type(std::min(cone[2], cone[0]), std::max(cone[2], cone[0]));
        *numEdges = 3;
        *edges    = triEdges;
      };
      void getEdges_LINE_LAGRANGE(const int coneSize, const point_type cone[],  int *numEdges, const edge_type **edges) {
        static edge_type lineEdges[6];

        assert(coneSize == 6);
        lineEdges[0] = edge_type(std::min(cone[0], cone[1]), std::max(cone[0], cone[1]));
        lineEdges[1] = edge_type(std::min(cone[2], cone[3]), std::max(cone[2], cone[3]));
        lineEdges[2] = edge_type(std::min(cone[4], cone[5]), std::max(cone[4], cone[5]));
        *numEdges = 3;
        *edges    = lineEdges;
      };
      void getEdges_TETRAHEDRON(const int coneSize, const point_type cone[],  int *numEdges, const edge_type **edges) {
        static edge_type tetEdges[6];

        assert(coneSize == 4);
        // As per Brad's diagram
        tetEdges[0] = edge_type(std::min(cone[0], cone[1]), std::max(cone[0], cone[1]));
        tetEdges[1] = edge_type(std::min(cone[1], cone[2]), std::max(cone[1], cone[2]));
        tetEdges[2] = edge_type(std::min(cone[2], cone[0]), std::max(cone[2], cone[0]));
        tetEdges[3] = edge_type(std::min(cone[0], cone[3]), std::max(cone[0], cone[3]));
        tetEdges[4] = edge_type(std::min(cone[1], cone[3]), std::max(cone[1], cone[3]));
        tetEdges[5] = edge_type(std::min(cone[2], cone[3]), std::max(cone[2], cone[3]));
        *numEdges = 6;
        *edges    = tetEdges;
      };
      void getEdges_TRIANGULAR_PRISM(const int coneSize, const point_type cone[],  int *numEdges, const edge_type **edges) {
        static edge_type triPrismEdges[6];

        assert(coneSize == 6);
        triPrismEdges[0] = edge_type(std::min(cone[0], cone[1]), std::max(cone[0], cone[1]));
        triPrismEdges[1] = edge_type(std::min(cone[1], cone[2]), std::max(cone[1], cone[2]));
        triPrismEdges[2] = edge_type(std::min(cone[2], cone[0]), std::max(cone[2], cone[0]));
        triPrismEdges[3] = edge_type(std::min(cone[3], cone[4]), std::max(cone[3], cone[4]));
        triPrismEdges[4] = edge_type(std::min(cone[4], cone[5]), std::max(cone[4], cone[5]));
        triPrismEdges[5] = edge_type(std::min(cone[5], cone[3]), std::max(cone[5], cone[3]));
        *numEdges = 6;
        *edges    = triPrismEdges;
      };
      void getEdges_TRIANGULAR_PRISM_LAGRANGE(const int coneSize, const point_type cone[],  int *numEdges, const edge_type **edges) {
        static edge_type triPrismLEdges[9];

        assert(coneSize == 9);
        triPrismLEdges[0] = edge_type(std::min(cone[0], cone[1]), std::max(cone[0], cone[1]));
        triPrismLEdges[1] = edge_type(std::min(cone[1], cone[2]), std::max(cone[1], cone[2]));
        triPrismLEdges[2] = edge_type(std::min(cone[2], cone[0]), std::max(cone[2], cone[0]));
        triPrismLEdges[3] = edge_type(std::min(cone[3], cone[4]), std::max(cone[3], cone[4]));
        triPrismLEdges[4] = edge_type(std::min(cone[4], cone[5]), std::max(cone[4], cone[5]));
        triPrismLEdges[5] = edge_type(std::min(cone[5], cone[3]), std::max(cone[5], cone[3]));
        triPrismLEdges[6] = edge_type(std::min(cone[6], cone[7]), std::max(cone[6], cone[7]));
        triPrismLEdges[7] = edge_type(std::min(cone[7], cone[8]), std::max(cone[7], cone[8]));
        triPrismLEdges[8] = edge_type(std::min(cone[8], cone[6]), std::max(cone[8], cone[6]));
        *numEdges = 9;
        *edges    = triPrismLEdges;
      };
      void getNewCells_TRIANGLE(const int coneSize, const point_type cone[],  int *numCells, const point_type **cells) {
        int               numEdges;
        const edge_type  *edges;
        static point_type triCells[4*3];
        point_type        newVertices[3];

        getEdges_TRIANGLE(coneSize, cone, &numEdges, &edges);
        assert(numEdges == 3);
        for(int e = 0; e < numEdges; ++e) {
          if (_edge2vertex.find(edges[e]) == _edge2vertex.end()) {
            throw ALE::Exception("Missing edge in refined mesh");
          }
          newVertices[e] = _edge2vertex[edges[e]];
        }
        triCells[0*3+0] = cone[0]+_vertexOffset; triCells[0*3+1] = newVertices[0]; triCells[0*3+2] = newVertices[2];
        triCells[1*3+0] = newVertices[0];       triCells[1*3+1] = newVertices[1]; triCells[1*3+2] = newVertices[2];
        triCells[2*3+0] = cone[1]+_vertexOffset; triCells[2*3+1] = newVertices[1]; triCells[2*3+2] = newVertices[0];
        triCells[3*3+0] = cone[2]+_vertexOffset; triCells[3*3+1] = newVertices[2]; triCells[3*3+2] = newVertices[1];
        *numCells = 4;
        *cells    = triCells;
      };
      void getNewCells_LINE_LAGRANGE(const int coneSize, const point_type cone[],  int *numCells, const point_type **cells) {
        int               numEdges;
        const edge_type  *edges;
        static point_type lineCells[2*6];
        point_type        newVertices[3];

        getEdges_LINE_LAGRANGE(coneSize, cone, &numEdges, &edges);
        assert(numEdges == 3);
        for(int e = 0; e < numEdges; ++e) {
          if (_edge2vertex.find(edges[e]) == _edge2vertex.end()) {
            throw ALE::Exception("Missing edge in refined mesh");
          }
          newVertices[e] = _edge2vertex[edges[e]];
        }
	lineCells[0*6+0] = cone[0]+_vertexOffset; // new cell 0
        lineCells[0*6+1] = newVertices[0];
	lineCells[0*6+2] = cone[2]+_vertexOffset;
        lineCells[0*6+3] = newVertices[1];
	lineCells[0*6+4] = cone[4]+_vertexOffset;
        lineCells[0*6+5] = newVertices[2];

        lineCells[1*6+0] = newVertices[0]; // new cell 1
	lineCells[1*6+1] = cone[1]+_vertexOffset;
        lineCells[1*6+2] = newVertices[1];
	lineCells[1*6+3] = cone[3]+_vertexOffset;
        lineCells[1*6+4] = newVertices[2];
	lineCells[1*6+5] = cone[5]+_vertexOffset;

        *numCells = 2;
        *cells    = lineCells;
      };
      void getNewCells_TETRAHEDRON(const int coneSize, const point_type cone[],  int *numCells, const point_type **cells) {
        int               numEdges;
        const edge_type  *edges;
        static point_type tetCells[8*4];
        point_type        newVertices[6];

        getEdges_TETRAHEDRON(coneSize, cone, &numEdges, &edges);
        assert(numEdges == 6);
        for(int e = 0; e < numEdges; ++e) {
          if (_edge2vertex.find(edges[e]) == _edge2vertex.end()) {
            throw ALE::Exception("Missing edge in refined mesh");
          }
          newVertices[e] = _edge2vertex[edges[e]];
        }
        tetCells[0*4+0] = cone[0]+_vertexOffset; tetCells[0*4+1] = newVertices[3]; tetCells[0*4+2] = newVertices[0]; tetCells[0*4+3] = newVertices[2];
        tetCells[1*4+0] = newVertices[0];       tetCells[1*4+1] = newVertices[1]; tetCells[1*4+2] = newVertices[2]; tetCells[1*4+3] = newVertices[3];
        tetCells[2*4+0] = newVertices[0];       tetCells[2*4+1] = newVertices[3]; tetCells[2*4+2] = newVertices[4]; tetCells[2*4+3] = newVertices[1];
        tetCells[3*4+0] = cone[1]+_vertexOffset; tetCells[3*4+1] = newVertices[4]; tetCells[3*4+2] = newVertices[1]; tetCells[3*4+3] = newVertices[0];
        tetCells[4*4+0] = newVertices[2];       tetCells[4*4+1] = newVertices[5]; tetCells[4*4+2] = newVertices[3]; tetCells[4*4+3] = newVertices[1];
        tetCells[5*4+0] = cone[2]+_vertexOffset; tetCells[5*4+1] = newVertices[5]; tetCells[5*4+2] = newVertices[2]; tetCells[5*4+3] = newVertices[1];
        tetCells[6*4+0] = newVertices[1];       tetCells[6*4+1] = newVertices[4]; tetCells[6*4+2] = newVertices[5]; tetCells[6*4+3] = newVertices[3];
        tetCells[7*4+0] = cone[3]+_vertexOffset; tetCells[7*4+1] = newVertices[3]; tetCells[7*4+2] = newVertices[5]; tetCells[7*4+3] = newVertices[4];
        *numCells = 8;
        *cells    = tetCells;
      };
      void getNewCells_TRIANGULAR_PRISM_LAGRANGE(const int coneSize, const point_type cone[],  int *numCells, const point_type **cells) {
        int               numEdges;
        const edge_type  *edges;
        static point_type tcells[4*9];
        point_type        newVertices[9];

        getEdges_TRIANGULAR_PRISM_LAGRANGE(coneSize, cone, &numEdges, &edges);
        assert(numEdges == 9);
        for(int e = 0; e < numEdges; ++e) {
          if (_edge2vertex.find(edges[e]) == _edge2vertex.end()) {
            throw ALE::Exception("Missing edge in refined mesh");
          }
          newVertices[e] = _edge2vertex[edges[e]];
        }
        tcells[0*9+0] = cone[0]+_vertexOffset; // New cell 0
	tcells[0*9+1] = newVertices[0];
	tcells[0*9+2] = newVertices[2];
        tcells[0*9+3] = cone[3]+_vertexOffset;
	tcells[0*9+4] = newVertices[3];
	tcells[0*9+5] = newVertices[5];
        tcells[0*9+6] = cone[6]+_vertexOffset;
	tcells[0*9+7] = newVertices[6];
	tcells[0*9+8] = newVertices[8];

        tcells[1*9+0] = newVertices[0]; // New cell 1
	tcells[1*9+1] = newVertices[1];
	tcells[1*9+2] = newVertices[2];
        tcells[1*9+3] = newVertices[3];
	tcells[1*9+4] = newVertices[4];
	tcells[1*9+5] = newVertices[5];
        tcells[1*9+6] = newVertices[6];
	tcells[1*9+7] = newVertices[7];
	tcells[1*9+8] = newVertices[8];

        tcells[2*9+0] = cone[1]+_vertexOffset; // New cell 2
	tcells[2*9+1] = newVertices[1];
	tcells[2*9+2] = newVertices[0];
        tcells[2*9+3] = cone[4]+_vertexOffset;
	tcells[2*9+4] = newVertices[4];
	tcells[2*9+5] = newVertices[3];
        tcells[2*9+6] = cone[7]+_vertexOffset;
	tcells[2*9+7] = newVertices[7];
	tcells[2*9+8] = newVertices[6];

        tcells[3*9+0] = cone[2]+_vertexOffset; // New cell 3
	tcells[3*9+1] = newVertices[2];
	tcells[3*9+2] = newVertices[1];
        tcells[3*9+3] = cone[5]+_vertexOffset;
	tcells[3*9+4] = newVertices[5];
	tcells[3*9+5] = newVertices[4];
        tcells[3*9+6] = cone[8]+_vertexOffset;
	tcells[3*9+7] = newVertices[8];
	tcells[3*9+8] = newVertices[7];

        *numCells = 4;
        *cells    = tcells;
      };
    public:
      point_type getVertexRelativeOffset()                        {return _vertexOffset;};
      void       setVertexRelativeOffset(const point_type offset) {_vertexOffset = offset;};
      edge_map_type& getEdgeToVertex() {return _edge2vertex;};
      int numNewCells(const point_type cell) {
        switch(this->getCellType(cell)) {
	case TRIANGLE:
	  return 4;
	case LINE_LAGRANGE:
	  return 2;
        case TETRAHEDRON:
          return 8;
        case TRIANGULAR_PRISM:
        case TRIANGULAR_PRISM_LAGRANGE:
          return 4;
        }
        throw ALE::Exception("Could not determine number of new cells for this cell type");
      };
      void splitEdge(const point_type cell, const int coneSize, const point_type cone[], point_type& curNewVertex) {
        const CellType   t = this->getCellType(cell);
        int              numEdges;
        const edge_type *edges;

        switch(t) {
	case TRIANGLE:
          getEdges_TRIANGLE(coneSize, cone, &numEdges, &edges);
          break;
	case LINE_LAGRANGE:
          getEdges_LINE_LAGRANGE(coneSize, cone, &numEdges, &edges);
          break;	  
        case TETRAHEDRON:
          getEdges_TETRAHEDRON(coneSize, cone, &numEdges, &edges);
          break;
        case TRIANGULAR_PRISM:
          getEdges_TRIANGULAR_PRISM(coneSize, cone, &numEdges, &edges);
          break;
        case TRIANGULAR_PRISM_LAGRANGE:
          getEdges_TRIANGULAR_PRISM_LAGRANGE(coneSize, cone, &numEdges, &edges);
          break;
        default:
          throw ALE::Exception("Could not determine number of new cells for this cell type");
        }
        // Check that vertex does not yet exist
        for(int v = 0; v < numEdges; ++v) {
          if (_edge2vertex.find(edges[v]) == _edge2vertex.end()) {
	    std::cout << "Edge: " << edges[v] << ", new vertex: " << curNewVertex << std::endl;
            _edge2vertex[edges[v]] = curNewVertex++;
          }
        }
      };
      void getNewCell(const point_type cell, const int coneSize, const point_type cone[], int newCellNumber, int *newConeSize, const point_type **newCone) {
        const CellType    t = this->getCellType(cell);
        int               numCells;
        const point_type *cells;

        switch(t) {
        case TRIANGLE:
          getNewCells_TRIANGLE(coneSize, cone,  &numCells, &cells);
          *newConeSize = 3;
          *newCone     = &cells[newCellNumber*3];
          break;
        case LINE_LAGRANGE:
          getNewCells_LINE_LAGRANGE(coneSize, cone,  &numCells, &cells);
          *newConeSize = 6;
          *newCone     = &cells[newCellNumber*6];
          break;
        case TETRAHEDRON:
          getNewCells_TETRAHEDRON(coneSize, cone,  &numCells, &cells);
          *newConeSize = 4;
          *newCone     = &cells[newCellNumber*4];
          break;
        case TRIANGULAR_PRISM_LAGRANGE:
          getNewCells_TRIANGULAR_PRISM_LAGRANGE(coneSize, cone,  &numCells, &cells);
          *newConeSize = 9;
          *newCone     = &cells[newCellNumber*9];
          break;
        default:
          throw ALE::Exception("Could not create new cell for this cell type");
        }
      };
      void getNeighboringVertices(const point_type cell, const int coneSize, const point_type cone[], const point_type firstNewVertex, point_type vertex2edge[]) {
        const CellType   t = this->getCellType(cell);
        int              numEdges;
        const edge_type *edges;

        switch(t) {
        case TRIANGLE:
          getEdges_TRIANGLE(coneSize, cone, &numEdges, &edges);
          break;
        case LINE_LAGRANGE:
          getEdges_LINE_LAGRANGE(coneSize, cone, &numEdges, &edges);
          break;
        case TETRAHEDRON:
          getEdges_TETRAHEDRON(coneSize, cone, &numEdges, &edges);
          break;
        case TRIANGULAR_PRISM:
          getEdges_TRIANGULAR_PRISM(coneSize, cone, &numEdges, &edges);
          break;
        case TRIANGULAR_PRISM_LAGRANGE:
          getEdges_TRIANGULAR_PRISM_LAGRANGE(coneSize, cone, &numEdges, &edges);
          break;
        default:
          throw ALE::Exception("Could not determine number of new cells for this cell type");
        }
        for(int v = 0; v < numEdges; ++v) {
          point_type newVertex = _edge2vertex[edges[v]];

	  std::cout << "VERTEX2EDGE index: " << newVertex-firstNewVertex << ", first: " << edges[v].first << ", second: " << edges[v].second << std::endl;
          vertex2edge[(newVertex-firstNewVertex)*2+0] = edges[v].first;
          vertex2edge[(newVertex-firstNewVertex)*2+1] = edges[v].second;
        }
      };
    };
    // This method takes a mesh and performs a refinement of each cell
    //
    //   triangle: 1 --> 4 refinement, adding a new vertex at the midpoint of each edge
    //   tetrahedra:        1 --> 8 refinement,  adding a new vertex at the midpoint of each edge
    //
    // :WARNING: This method currently only works for uninterpolated meshes with tri and tet cells.
    template<typename MeshType, typename Refiner>
    static void refineGeneral(const Obj<MeshType>& mesh, const Obj<MeshType>& newMesh, Refiner& refiner) {
      typedef typename MeshType::sieve_type sieve_type;
      typedef typename MeshType::point_type point_type;
      typedef typename Refiner::edge_type   edge_type;

      // :WARNING: Assumed order of mesh points (cells and vertices):
      //
      // normal cells (in censored depth)
      // normal vertices (in censored depth)
      // other vertices
      // other cells
      //
      // This permits omitting in output the other vertices (e.g.,
      // Lagrange multipliers) and other cells (e.g., cohesive cells)
      // which have a custom reference cell that is not recognized.

      assert(!mesh.isNull());
      assert(!newMesh.isNull());

      // Get original mesh stuff.
      const Obj<typename MeshType::label_sequence>& cells = mesh->heightStratum(0);
      assert(!cells.isNull());
      const typename MeshType::label_sequence::iterator cellsEnd = cells->end();

      const Obj<typename MeshType::label_sequence>& vertices = mesh->depthStratum(0);
      assert(!vertices.isNull());

      const Obj<sieve_type>& sieve = mesh->getSieve();
      assert(!sieve.isNull());
      ALE::ISieveVisitor::PointRetriever<sieve_type> cV(std::max(1, sieve->getMaxConeSize()));

      if (mesh->hasLabel("censored depth")) {
	// :WARNING: Assume all cells in the censored depth come before
	// any other cells. This guarantees that we add vertices in the
	// censored depth before adding other vertices.

	int counterBegin = 0;

	int oldNumCellsNormal = 0;
	int oldNumCellsOther = 0;
	int oldNumVerticesNormal = 0;
	int oldNumVerticesOther = 0;

	int newNumCellsNormal = 0;
	int newNumCellsOther = 0;
	int newNumVerticesNormal = 0;
	int newNumVerticesOther = 0;

	// Count number of cells in censored depth (normal cells).
	const Obj<typename MeshType::label_sequence>& cellsNormal = mesh->getLabelStratum("censored depth", mesh->depth());
	assert(!cellsNormal.isNull());
	const typename MeshType::label_sequence::iterator cellsNormalEnd = cellsNormal->end();
	oldNumCellsNormal = cellsNormal->size();
	for(typename MeshType::label_sequence::iterator c_iter = cellsNormal->begin(); c_iter != cellsNormalEnd; ++c_iter)
	  newNumCellsNormal += refiner.numNewCells(*c_iter);

	// Count number of remaining cells (other cells).
	const int numSkip = oldNumCellsNormal;
	oldNumCellsOther = cells->size() - oldNumCellsNormal;
	typename MeshType::label_sequence::iterator c_iter = cells->begin();
	for (int i=0; i < numSkip; ++i)
	  ++c_iter;
	for (; c_iter != cellsEnd; ++c_iter)
	  newNumCellsOther += refiner.numNewCells(*c_iter);

	// Get number of old normal vertices.
	assert(!mesh->getFactory.isNull());
	Obj<typename Mesh::numbering_type> vNumbering = mesh->getFactory()->getNumbering(mesh, "censored depth", 0);
	assert(!vNumbering.isNull());
	oldNumVerticesNormal = vNumbering->size();

	// Count number of new normal vertices.
	int counterBegin = newNumCellsNormal + vertices->size();
	const point_type curNewVertex = counterBegin;
	for(typename MeshType::label_sequence::iterator c_iter = cellsNormal->begin(); c_iter != cellsNormalEnd; ++c_iter) {
	  cV.clear();
	  sieve->cone(*c_iter, cV);
	  refiner.splitEdge(*c_iter, cV.getSize(), cV.getPoints(), curNewVertex);
	} // for
	newNumVerticesNormal = curNewVertex - counterBegin;

	// Count number of remaining vertices (other vertices).
	oldNumVerticesOther = vertices->size() - oldNumVerticessNormal;
	counterBegin = curNewVertex + oldNumVerticesOther;
	curNewVertex = counterBegin;
	c_iter = cells->begin();
	for (int i=0; i < numSkip; ++i)
	  ++c_iter;
	for (; c_iter != cellsEnd; ++c_iter) {
	  cV.clear();
	  sieve->cone(*c_iter, cV);
	  refiner.splitEdge(*c_iter, cV.getSize(), cV.getPoints(), curNewVertex);
	} // for
	newNumVerticesOther = curNewVertex - counterBegin;

	Interval<point_type> oldCellsNormalRange(0, oldNumCellsNormal);
	Interval<point_type> newCellsNormalRange(0, newNumCellsNormal);

	Interval<point_type> oldVerticesNormalRange(oldNumCellsNormal, oldNumCellsNormal+oldNumVerticesNormal);
	Interval<point_type> newVerticesNormalRange(newNumCellsNormal, newNumCellsNormal+newNumVerticesNormal);

	Interval<point_type> oldVerticesOtherRange(oldNumCellsNormal+oldNumVerticesNormal ,
						   oldNumCellsNormal+oldNumVerticesNormal+oldNumVerticesOther);
	Interval<point_type> newVerticesOtherRange(newNumCellsNormal+newNumVerticesNormal ,
						   newNumCellsNormal+newNumVerticesNormal+newNumVerticesOther);

	Interval<point_type> oldCellssOtherRange(oldNumCellsNormal+oldNumVerticesNormal+oldNumVerticesOther,
						 oldNumCellsNormal+oldNumVerticesNormal+oldNumVerticesOther+oldNumCellsOther);
	Interval<point_type> newCellssOtherRange(newNumCellsNormal+newNumVerticesNormal+newNumVerticesOther,
						 newNumCellsNormal+newNumVerticesNormal+newNumVerticesOther+newNumCellsOther);


	// Allocate chart for new sieve.
	const Obj<sieve_type>& newSieve = newMesh->getSieve();
	assert(!newSieve.isNull());
	newSieve->setChart(typename sieve_type::chart_type(0, newCellsOtherRange.end()));
	refiner.setVertexRelativeOffset(newNumCellsNormal-oldNumCellsNormal); // THIS DOES NOT WORK FOR COHESIVE CELLS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	// Create new sieve with correct sizes for refined cells

	// Start with normal cells.
	point_type curNewCell = newCellsNormalRange.begin();
	const typename Interval<point_type>::const_iterator oldCellsNormalRangeEnd = oldCellsNormalRange.end();
	for (typename Interval<point_type>::const_iterator c_iter=oldCellsNormalRange.begin();
	     c_iter != oldCellsNormalRangeEnd;
	     ++c_iter) {
	  // Set new cone and support sizes
	  cV.clear();
	  sieve->cone(*c_iter, cV);
	  const point_type *cone = cV.getPoints();
	  const int coneSize = cV.getSize();
	  const int newCells = refiner.numNewCells(*c_iter);

	  for(int iCell=0; iCell < newCells; ++iCell, ++curNewCell) {
	    const point_type *newCone;
	    int newConeSize;

	    newSieve->setConeSize(curNewCell, sieve->getConeSize(*c_iter));
	    // OPTIMIZE THIS
	    refiner.getNewCell(*c_iter, coneSize, cone, iCell, &newConeSize, &newCone);
	    for(int v = 0; v < newConeSize; ++v)
	      newSieve->addSupportSize(newCone[v], 1);
	  } // for
	} // for

	// Continue with other cells.
	point_type curNewCell = newCellsOtherRange.begin();
	const typename Interval<point_type>::const_iterator oldCellsOtherRangeEnd = oldCellsOtherRange.end();
	for (typename Interval<point_type>::const_iterator c_iter=oldCellOtherRange.begin();
	     c_iter != oldCellsOtherRangeEnd;
	     ++c_iter) {
	  // Set new cone and support sizes
	  cV.clear();
	  sieve->cone(*c_iter, cV);
	  const point_type *cone = cV.getPoints();
	  const int coneSize = cV.getSize();
	  const int newCells = refiner.numNewCells(*c_iter);

	  for(int iCell=0; iCell < newCells; ++iCell, ++curNewCell) {
	    const point_type *newCone;
	    int newConeSize;

	    newSieve->setConeSize(curNewCell, sieve->getConeSize(*c_iter));
	    // OPTIMIZE THIS
	    refiner.getNewCell(*c_iter, coneSize, cone, iCell, &newConeSize, &newCone);
	    for(int v = 0; v < newConeSize; ++v)
	      newSieve->addSupportSize(newCone[v], 1);
	  } // for
	} // for
	newSieve->allocate();
	point_type* vertex2edge = new point_type[(newNumVerticesNormal+newNumVerticesOther)*2];
	typename Refiner::edge_map_type& edge2vertex = refiner.getEdgeToVertex();


	// Create refined normal cells in new sieve.
	curNewCell = newCellsNormalRange.begin();
	for (typename Interval<point_type>::const_iterator c_iter=oldCellsNormalRange.begin();
	     c_iter != oldCellsNormalRangeEnd;
	     ++c_iter) {
	  cV.clear();
	  sieve->cone(*c_iter, cV);
	  const point_type *cone = cV.getPoints();
	  const int coneSize = cV.getSize();
	  const int newCells = refiner.numNewCells(*c_iter);

	  for (int iCell=0; iCell < newCells; ++iCell, ++curNewCell) {
	    const point_type *newCone;
	    int newConeSize;

	    refiner.getNewCell(*c_iter, coneSize, cone, iCell, &newConeSize, &newCone);
	    newSieve->setCone(newCone, curNewCell);
	  } // for

	  refiner.getNeighboringVertices(*c_iter, coneSize, cone, firstNewVertex, vertex2edge);
	} // for

	// Create refined other cells in new sieve.
	curNewCell = newCellsOtherRange.begin();
	for (typename Interval<point_type>::const_iterator c_iter=oldCellsOtherRange.begin();
	     c_iter != oldCellsOtherRangeEnd;
	     ++c_iter) {
	  cV.clear();
	  sieve->cone(*c_iter, cV);
	  const point_type *cone = cV.getPoints();
	  const int coneSize = cV.getSize();
	  const int newCells = refiner.numNewCells(*c_iter);

	  for (int iCell=0; iCell < newCells; ++iCell, ++curNewCell) {
	    const point_type *newCone;
	    int newConeSize;

	    refiner.getNewCell(*c_iter, coneSize, cone, iCell, &newConeSize, &newCone);
	    newSieve->setCone(newCone, curNewCell);
	  } // for

	  // FIX THIS!!! LAGRANGE VERTICES MUST BE AFTER ALL OTHER VERTICES (INCLUDING OLD LAGRANGE VERTICES)
	  refiner.getNeighboringVertices(*c_iter, coneSize, cone, firstNewVertex, vertex2edge);
	} // for
	newSieve->symmetrize();

	// Set coordinates in refined mesh.
	const Obj<typename MeshType::real_section_type>& coordinates = mesh->getRealSection("coordinates");
	assert(!coordinates.isNull());
	const Obj<typename MeshType::real_section_type>& newCoordinates = newMesh->getRealSection("coordinates");
	assert(!newCoordinates.isNull());

	const typename MeshType::label_sequence::const_iterator verticesEnd = vertices->end();
	assert(vertices->size() > 0);
	const int spaceDim = coordinates->getFiberDimension(*vertices->begin());
	assert(spaceDim > 0);
	newCoordinates->setChart(typename sieve_type::chart_type(newNumCellsNormal, newNumCellsNormal+newNumVertices));

	for (int iVertex=0, offset=newNumCellsNormal; iVertex < newNumVertices; ++iVertex) {
	  const point_type vNew = iVertex + offset;
	  newCoordinates->setFiberDimension(vNew, spaceDim);
	} // for
	newCoordinates->allocatePoint();

	for (int iVertex=0, oldOffset=oldNumCellsNormal, newOffset=newNumCellsNormal; iVertex < oldNumVertices; ++iVertex) {
	  const point_type vOld = iVertex + oldOffset;
	  const point_type vNew = iVertex + newOffset;
	  newCoordinates->updatePoint(vNew, coordinates->restrictPoint(vOld));
	} // for
	for(int v=0, iVertex=oldNumVertices, newOffset=newNumCellsNormal; iVertex < newNumVertices; ++v, ++iVertex) {
	  const point_type vNew = iVertex + newOffset;
	  const point_type endpointA = vertex2edge[v*2+0];
	  const point_type endpointB = vertex2edge[v*2+1];
	  std::cout << "Setting coordinates of vertex " << vNew << " between vertices "
		    << endpointA << " and " << endpointB << std::endl;
	  const real *coordsA   = coordinates->restrictPoint(endpointA);
	  real coords[3];

	  for(int d = 0; d < 3; ++d)
	    coords[d]  = coordsA[d];
	  const real *coordsB = coordinates->restrictPoint(endpointB);
	  for(int d = 0; d < 3; ++d) {
	    coords[d] += coordsB[d];
	    coords[d] *= 0.5;
	  } // for
	  newCoordinates->updatePoint(vNew, coords);
	} // for
	delete [] vertex2edge;
	// Fast stratification
	const Obj<typename MeshType::label_type>& height = newMesh->createLabel("height");
	const Obj<typename MeshType::label_type>& depth  = newMesh->createLabel("depth");

	for (int iCell=0; iCell < newNumCellsNormal; ++iCell) {
	  const point_type cNew = iCell;
	  height->setCone(0, cNew);
	  depth->setCone(1, cNew);
	} // for
	for (int iCell=newNumCellsNormal, offset=newNumVertices; iCell < newNumCells; ++iCell) {
	  const point_type cNew = iCell + offset;
	  height->setCone(0, cNew);
	  depth->setCone(1, cNew);
	} // for
	for (int iVertex=0, newOffset=newNumCellsNormal; iVertex < newNumVertices; ++iVertex) {
	  const point_type vNew = iVertex + newOffset;
	  height->setCone(1, vNew);
	  depth->setCone(0, vNew);
	} // for
	newMesh->setHeight(1);
	newMesh->setDepth(1);

      } else {
	int counterBegin = 0;

	int oldNumCells = 0;
	int oldNumVertices = 0;

	int newNumCells = 0;
	int newNumVertices = 0;

	// Count number of cells.
	oldNumCells = cells->size();
	for (typename MeshType::label_sequence::iterator c_iter = cells->begin(); c_iter != cellsEnd; ++c_iter)
	  newNumCells += refiner.numNewCells(*c_iter);

	// Count number of vertices (normal vertices).
	oldNumVertices = vertices->size();
	int counterBegin = newNumCells + oldNumVertices;
	const point_type curNewVertex = counterBegin;
	for(typename MeshType::label_sequence::iterator c_iter = cells->begin(); c_iter != cellsEnd; ++c_iter) {
	  cV.clear();
	  sieve->cone(*c_iter, cV);
	  refiner.splitEdge(*c_iter, cV.getSize(), cV.getPoints(), curNewVertex);
	} // for
	newNumVertices = curNewVertex - counterBegin;

	Interval<point_type> oldCellsRange(0, oldNumCells);
	Interval<point_type> newCellsRange(0, newNumCells);

	Interval<point_type> oldVerticesRange(oldNumCells, oldNumCells+oldNumVertices);
	Interval<point_type> newVerticesRange(newNumCells, newNumCells+newNumVertices);

	// Allocate chart for new sieve.
	const Obj<sieve_type>& newSieve = newMesh->getSieve();
	assert(!newSieve.isNull());
	newSieve->setChart(typename sieve_type::chart_type(0, newCellsOtherRange.end()));
	refiner.setVertexRelativeOffset(newNumCells-oldNumCells);

	// Create new sieve with correct sizes for refined cells

	// Start with normal cells.
	point_type curNewCell = newCellsRange.begin();
	const typename Interval<point_type>::const_iterator oldCellsRangeEnd = oldCellsRange.end();
	for (typename Interval<point_type>::const_iterator c_iter=oldCellsRange.begin();
	     c_iter != oldCellsRangeEnd;
	     ++c_iter) {
	  // Set new cone and support sizes
	  cV.clear();
	  sieve->cone(*c_iter, cV);
	  const point_type *cone = cV.getPoints();
	  const int coneSize = cV.getSize();
	  const int newCells = refiner.numNewCells(*c_iter);

	  for(int iCell=0; iCell < newCells; ++iCell, ++curNewCell) {
	    const point_type *newCone;
	    int newConeSize;

	    newSieve->setConeSize(curNewCell, sieve->getConeSize(*c_iter));
	    // OPTIMIZE THIS
	    refiner.getNewCell(*c_iter, coneSize, cone, iCell, &newConeSize, &newCone);
	    for(int v = 0; v < newConeSize; ++v)
	      newSieve->addSupportSize(newCone[v], 1);
	  } // for
	} // for

	// Create refined normal cells in new sieve.
	curNewCell = newCellsRange.begin();
	for (typename Interval<point_type>::const_iterator c_iter=oldCellsRange.begin();
	     c_iter != oldCellsRangeEnd;
	     ++c_iter) {
	  cV.clear();
	  sieve->cone(*c_iter, cV);
	  const point_type *cone = cV.getPoints();
	  const int coneSize = cV.getSize();
	  const int newCells = refiner.numNewCells(*c_iter);

	  for (int iCell=0; iCell < newCells; ++iCell, ++curNewCell) {
	    const point_type *newCone;
	    int newConeSize;

	    refiner.getNewCell(*c_iter, coneSize, cone, iCell, &newConeSize, &newCone);
	    newSieve->setCone(newCone, curNewCell);
	  } // for

	  refiner.getNeighboringVertices(*c_iter, coneSize, cone, firstNewVertex, vertex2edge);
	} // for
	newSieve->symmetrize();

	// Set coordinates in refined mesh.
	const Obj<typename MeshType::real_section_type>& coordinates = mesh->getRealSection("coordinates");
	assert(!coordinates.isNull());
	const Obj<typename MeshType::real_section_type>& newCoordinates = newMesh->getRealSection("coordinates");
	assert(!newCoordinates.isNull());

	const typename MeshType::label_sequence::const_iterator verticesEnd = vertices->end();
	assert(vertices->size() > 0);
	const int spaceDim = coordinates->getFiberDimension(*vertices->begin());
	assert(spaceDim > 0);
	newCoordinates->setChart(typename sieve_type::chart_type(newVerticesRange.begin(), newVerticesRange.end()));

	const typename Interval<point_type>::const_iterator newVerticesRangeEnd = newVerticesRange.end();
	for (typename Interval<point_type>::const_iterator v_iter=newVerticesRange.begin(); v_iter != newVerticesRangeEnd; ++v_iter)
	  newCoordinates->setFiberDimension(v_iter, spaceDim);
	newCoordinates->allocatePoint();

	const typename Interval<point_type>::const_iterator oldVerticesRangeEnd = oldVerticesRange.end();
	for (typename Interval<point_type>::const_iterator vOld_iter=oldVerticesRange.begin(), vNew_iter=newVerticesRange.begin(); vOld_iter != oldVerticesRangeEnd; ++vOld_iter)
	  newCoordinates->updatePoint(*vNew_iter, coordinates->restrictPoint(*vOld_iter));
	for(int v=0, iVertex=oldNumVertices; iVertex < newNumVertices; ++v, ++iVertex) {
	  const point_type vNew = newVerticesRange.begin() + iVertex;
	  const point_type endpointA = vertex2edge[v*2+0];
	  const point_type endpointB = vertex2edge[v*2+1];
	  std::cout << "Setting coordinates of vertex " << vNew << " between vertices "
		    << endpointA << " and " << endpointB << std::endl;
	  const real *coordsA   = coordinates->restrictPoint(endpointA);
	  real coords[3];

	  for(int d = 0; d < 3; ++d)
	    coords[d]  = coordsA[d];
	  const real *coordsB = coordinates->restrictPoint(endpointB);
	  for(int d = 0; d < 3; ++d) {
	    coords[d] += coordsB[d];
	    coords[d] *= 0.5;
	  } // for
	  newCoordinates->updatePoint(vNew, coords);
	} // for
	delete [] vertex2edge;

	// Fast stratification
	const Obj<typename MeshType::label_type>& height = newMesh->createLabel("height");
	const Obj<typename MeshType::label_type>& depth  = newMesh->createLabel("depth");
	for (int iCell=0; iCell < newNumCells; ++iCell) {
	  const point_type cNew = iCell;
	  height->setCone(0, cNew);
	  depth->setCone(1, cNew);
	} // for
	for (int iVertex=0, newOffset=newNumCellsNormal; iVertex < newNumVertices; ++iVertex) {
	  const point_type vNew = iVertex + newOffset;
	  height->setCone(1, vNew);
	  depth->setCone(0, vNew);
	} // for
	newMesh->setHeight(1);
	newMesh->setDepth(1);
      } // if/else

      // Exchange new boundary vertices
      //   We can convert endpoints, and then just match to new vertex on this side
      //   1) Create the overlap of edges which are vertex pairs (do not need for interpolated meshes)
      //   2) Create a section of overlap edge --> new vertex (this will generalize to other split points in interpolated meshes)
      //   3) Copy across new overlap
      //   4) Fuse matches new vertex pairs and inserts them into the old overlap


      // Create the parallel overlap
      int *oldNumCellsP    = new int[mesh->commSize()];
      int *newNumCellsP = new int[newMesh->commSize()];
      int  ierr;

      ierr = MPI_Allgather((void *) &oldNumCells, 1, MPI_INT, oldNumCellsP, 1, MPI_INT, mesh->comm());CHKERRXX(ierr);
      ierr = MPI_Allgather((void *) &newNumCells, 1, MPI_INT, newNumCellsP, 1, MPI_INT, newMesh->comm());CHKERRXX(ierr);
      Obj<typename MeshType::send_overlap_type> newSendOverlap = newMesh->getSendOverlap();
      Obj<typename MeshType::recv_overlap_type> newRecvOverlap = newMesh->getRecvOverlap();
      const Obj<typename MeshType::send_overlap_type>& sendOverlap = mesh->getSendOverlap();
      const Obj<typename MeshType::recv_overlap_type>& recvOverlap = mesh->getRecvOverlap();
      Obj<typename MeshType::send_overlap_type::traits::capSequence> sendPoints  = sendOverlap->cap();
      const typename MeshType::send_overlap_type::source_type        localOffset = newNumCellsP[newMesh->commRank()] - oldNumCellsP[mesh->commRank()];

      for(typename MeshType::send_overlap_type::traits::capSequence::iterator p_iter = sendPoints->begin(); p_iter != sendPoints->end(); ++p_iter) {
        const Obj<typename MeshType::send_overlap_type::traits::supportSequence>& ranks      = sendOverlap->support(*p_iter);
        const typename MeshType::send_overlap_type::source_type&                  localPoint = *p_iter;

        for(typename MeshType::send_overlap_type::traits::supportSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          const int                                   rank         = *r_iter;
          const typename MeshType::send_overlap_type::source_type& remotePoint  = r_iter.color();
          const typename MeshType::send_overlap_type::source_type  remoteOffset = newNumCellsP[rank] - oldNumCellsP[rank];

          newSendOverlap->addArrow(localPoint+localOffset, rank, remotePoint+remoteOffset);
        }
      }
      Obj<typename MeshType::recv_overlap_type::traits::baseSequence> recvPoints = recvOverlap->base();

      for(typename MeshType::recv_overlap_type::traits::baseSequence::iterator p_iter = recvPoints->begin(); p_iter != recvPoints->end(); ++p_iter) {
        const Obj<typename MeshType::recv_overlap_type::traits::coneSequence>& ranks      = recvOverlap->cone(*p_iter);
        const typename MeshType::recv_overlap_type::target_type&               localPoint = *p_iter;

        for(typename MeshType::recv_overlap_type::traits::coneSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          const int                                        rank         = *r_iter;
          const typename MeshType::recv_overlap_type::target_type& remotePoint  = r_iter.color();
          const typename MeshType::recv_overlap_type::target_type  remoteOffset = newNumCellsP[rank] - oldNumCellsP[rank];

          newRecvOverlap->addArrow(rank, localPoint+localOffset, remotePoint+remoteOffset);
        }
      }
      newMesh->setCalculatedOverlap(true);
      delete [] oldNumCellsP;
      delete [] newNumCellsP;
      // Check edges in edge2vertex for both endpoints sent to same process
      //   Put it in section with point being the lowest numbered vertex and value (other endpoint, new vertex)
      Obj<ALE::Section<point_type, edge_type> > newVerticesSection = new ALE::Section<point_type, edge_type>(mesh->comm());
      std::map<edge_type, std::vector<int> > bdedge2rank;

      for(typename std::map<edge_type, point_type>::const_iterator e_iter = edge2vertex.begin(); e_iter != edge2vertex.end(); ++e_iter) {
        const point_type left  = e_iter->first.first;
        const point_type right = e_iter->first.second;

        if (sendOverlap->capContains(left) && sendOverlap->capContains(right)) {
          const Obj<typename MeshType::send_overlap_type::traits::supportSequence>& leftRanksSeq  = sendOverlap->support(left);
          std::set<int> leftRanks(leftRanksSeq->begin(), leftRanksSeq->end());
          const Obj<typename MeshType::send_overlap_type::traits::supportSequence>& rightRanksSeq = sendOverlap->support(right);
          std::set<int> rightRanks(rightRanksSeq->begin(), rightRanksSeq->end());
          std::set<int> ranks;
          std::set_intersection(leftRanks.begin(), leftRanks.end(), rightRanks->begin(), rightRanks->end(),
                                std::insert_iterator<std::list<int> >(ranks, ranks.begin()));

          if(ranks.size()) {
            newVerticesSection->addFiberDimension(std::min(e_iter->first.first, e_iter->first.second)+localOffset, 1);
            for(typename std::list<int>::const_iterator r_iter = ranks.begin(); r_iter != ranks.end(); ++r_iter) {
              bdedge2rank[e_iter->first].push_back(*r_iter);
            }
          }
        }
      }
      newVerticesSection->allocatePoint();
      const typename ALE::Section<point_type, edge_type>::chart_type& chart = newVerticesSection->getChart();

      for(typename ALE::Section<point_type, edge_type>::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        typedef typename ALE::Section<point_type, edge_type>::value_type value_type;
        const point_type p      = *c_iter;
        const int        dim    = newVerticesSection->getFiberDimension(p);
        int              v      = 0;
        value_type      *values = new value_type[dim];

        for(typename std::map<edge_type, std::vector<int> >::const_iterator e_iter = bdedge2rank.begin(); e_iter != bdedge2rank.end() && v < dim; ++e_iter) {
          if (std::min(e_iter->first.first, e_iter->first.second)+localOffset == p) {
            values[v++] = edge_type(std::max(e_iter->first.first, e_iter->first.second)+localOffset, edge2vertex[e_iter->first]);
          }
        }
        newVerticesSection->updatePoint(p, values);
        delete [] values;
      }
      // Copy across overlap
      typedef ALE::Pair<int, point_type> overlap_point_type;
      Obj<ALE::Section<overlap_point_type, edge_type> > overlapVertices = new ALE::Section<overlap_point_type, edge_type>(mesh->comm());

      ALE::Pullback::SimpleCopy::copy(newSendOverlap, newRecvOverlap, newVerticesSection, overlapVertices);
      // Merge by translating edge to local points, finding edge in edge2vertex, and adding (local new vetex, remote new vertex) to overlap
      for(typename std::map<edge_type, std::vector<int> >::const_iterator e_iter = bdedge2rank.begin(); e_iter != bdedge2rank.end(); ++e_iter) {
        const point_type localPoint = edge2vertex[e_iter->first];

        for(typename std::vector<int>::const_iterator r_iter = e_iter->second.begin(); r_iter != e_iter->second.end(); ++r_iter) {
          point_type remoteLeft = -1, remoteRight = -1;
          const int  rank       = *r_iter;

          const Obj<typename MeshType::send_overlap_type::traits::supportSequence>& leftRanks = newSendOverlap->support(e_iter->first.first+localOffset);
          for(typename MeshType::send_overlap_type::traits::supportSequence::iterator lr_iter = leftRanks->begin(); lr_iter != leftRanks->end(); ++lr_iter) {
            if (rank == *lr_iter) {
              remoteLeft = lr_iter.color();
              break;
            }
          }
          const Obj<typename MeshType::send_overlap_type::traits::supportSequence>& rightRanks = newSendOverlap->support(e_iter->first.second+localOffset);
          for(typename MeshType::send_overlap_type::traits::supportSequence::iterator rr_iter = rightRanks->begin(); rr_iter != rightRanks->end(); ++rr_iter) {
            if (rank == *rr_iter) {
              remoteRight = rr_iter.color();
              break;
            }
          }
          const point_type remoteMin   = std::min(remoteLeft, remoteRight);
          const point_type remoteMax   = std::max(remoteLeft, remoteRight);
          const int        remoteSize  = overlapVertices->getFiberDimension(overlap_point_type(rank, remoteMin));
          const edge_type *remoteVals  = overlapVertices->restrictPoint(overlap_point_type(rank, remoteMin));
          point_type       remotePoint = -1;

          for(int d = 0; d < remoteSize; ++d) {
            if (remoteVals[d].first == remoteMax) {
              remotePoint = remoteVals[d].second;
              break;
            }
          }
          newSendOverlap->addArrow(localPoint, rank, remotePoint);
          newRecvOverlap->addArrow(rank, localPoint, remotePoint);
        }
      }
    };
#endif
  };

  class MeshSerializer {
  public:
    template<typename Mesh>
    static void writeMesh(const std::string& filename, Mesh& mesh) {
      std::ofstream fs;

      if (mesh.commRank() == 0) {
        fs.open(filename.c_str());
      }
      writeMesh(fs, mesh);
      if (mesh.commRank() == 0) {
        fs.close();
      }
    }
    template<typename Mesh>
    static void writeMesh(std::ofstream& fs, Mesh& mesh) {
      ISieveSerializer::writeSieve(fs, *mesh.getSieve());
      // Write labels
      const typename Mesh::labels_type& labels = mesh.getLabels();

      if (!mesh.commRank()) {fs << labels.size() << std::endl;}
      for(typename Mesh::labels_type::const_iterator l_iter = labels.begin(); l_iter != labels.end(); ++l_iter) {
        if (!mesh.commRank()) {fs << l_iter->first << std::endl;}
        LabelSifterSerializer::writeLabel(fs, *l_iter->second);
      }
      // Write sections
      Obj<std::set<std::string> > realNames = mesh.getRealSections();

      if (!mesh.commRank()) {fs << realNames->size() << std::endl;}
      for(std::set<std::string>::const_iterator n_iter = realNames->begin(); n_iter != realNames->end(); ++n_iter) {
        if (!mesh.commRank()) {fs << *n_iter << std::endl;}
        SectionSerializer::writeSection(fs, *mesh.getRealSection(*n_iter));
      }
      Obj<std::set<std::string> > intNames = mesh.getIntSections();

      if (!mesh.commRank()) {fs << intNames->size() << std::endl;}
      for(std::set<std::string>::const_iterator n_iter = intNames->begin(); n_iter != intNames->end(); ++n_iter) {
        if (!mesh.commRank()) {fs << *n_iter << std::endl;}
        SectionSerializer::writeSection(fs, *mesh.getIntSection(*n_iter));
      }
      // Write overlap
#ifdef USE_NEW_OVERLAP
      PETSc::OverlapSerializer::writeOverlap(fs, *mesh.getSendOverlap());
      PETSc::OverlapSerializer::writeOverlap(fs, *mesh.getRecvOverlap());
#else
      SifterSerializer::writeSifter(fs, *mesh.getSendOverlap());
      SifterSerializer::writeSifter(fs, *mesh.getRecvOverlap());
#endif
      // Write distribution overlap
      // Write renumbering
    }
    template<typename Mesh>
    static void loadMesh(const std::string& filename, Mesh& mesh) {
      std::ifstream fs;

      if (mesh.commRank() == 0) {
        fs.open(filename.c_str());
      }
      loadMesh(fs, mesh);
      if (mesh.commRank() == 0) {
        fs.close();
      }
    }
    template<typename Mesh>
    static void loadMesh(std::ifstream& fs, Mesh& mesh) {
      ALE::Obj<typename Mesh::sieve_type> sieve = new typename Mesh::sieve_type(mesh.comm(), mesh.debug());
      PetscErrorCode                      ierr;

      ISieveSerializer::loadSieve(fs, *sieve);
      mesh.setSieve(sieve);
      // Load labels
      int numLabels;

      if (!mesh.commRank()) {fs >> numLabels;}
      ierr = MPI_Bcast(&numLabels, 1, MPI_INT, 0, mesh.comm());CHKERRXX(ierr);
      for(int l = 0; l < numLabels; ++l) {
        ALE::Obj<typename Mesh::label_type> label = new typename Mesh::label_type(mesh.comm(), mesh.debug());
        std::string                         name;
        int                                 len;

        if (!mesh.commRank()) {
          fs >> name;
          len = name.size();
          ierr = MPI_Bcast(&len, 1, MPI_INT, 0, mesh.comm());CHKERRXX(ierr);
          ierr = MPI_Bcast((void *) name.c_str(), len+1, MPI_CHAR, 0, mesh.comm());CHKERRXX(ierr);
        } else {
          ierr = MPI_Bcast(&len, 1, MPI_INT, 0, mesh.comm());CHKERRXX(ierr);
          char *n = new char[len+1];
          ierr = MPI_Bcast(n, len+1, MPI_CHAR, 0, mesh.comm());CHKERRXX(ierr);
          name = n;
          delete [] n;
        }
        LabelSifterSerializer::loadLabel(fs, *label);
        mesh.setLabel(name, label);
      }
      // Load sections
      int numRealSections;

      if (!mesh.commRank()) {fs >> numRealSections;}
      ierr = MPI_Bcast(&numRealSections, 1, MPI_INT, 0, mesh.comm());CHKERRXX(ierr);
      for(int s = 0; s < numRealSections; ++s) {
        ALE::Obj<typename Mesh::real_section_type> section = new typename Mesh::real_section_type(mesh.comm(), mesh.debug());
        std::string                                name;
        int                                        len;

        if (!mesh.commRank()) {
          fs >> name;
          len = name.size();
          ierr = MPI_Bcast(&len, 1, MPI_INT, 0, mesh.comm());CHKERRXX(ierr);
          ierr = MPI_Bcast((void *) name.c_str(), len+1, MPI_CHAR, 0, mesh.comm());CHKERRXX(ierr);
        } else {
          ierr = MPI_Bcast(&len, 1, MPI_INT, 0, mesh.comm());CHKERRXX(ierr);
          char *n = new char[len+1];
          ierr = MPI_Bcast(n, len+1, MPI_CHAR, 0, mesh.comm());CHKERRXX(ierr);
          name = n;
          delete [] n;
        }
        SectionSerializer::loadSection(fs, *section);
        mesh.setRealSection(name, section);
      }
      int numIntSections;

      if (!mesh.commRank()) {fs >> numIntSections;}
      ierr = MPI_Bcast(&numIntSections, 1, MPI_INT, 0, mesh.comm());CHKERRXX(ierr);
      for(int s = 0; s < numIntSections; ++s) {
        ALE::Obj<typename Mesh::int_section_type> section = new typename Mesh::int_section_type(mesh.comm(), mesh.debug());
        std::string                               name;
        int                                       len;

        if (!mesh.commRank()) {
          fs >> name;
          len = name.size();
          ierr = MPI_Bcast(&len, 1, MPI_INT, 0, mesh.comm());CHKERRXX(ierr);
          ierr = MPI_Bcast((void *) name.c_str(), len+1, MPI_CHAR, 0, mesh.comm());CHKERRXX(ierr);
        } else {
          ierr = MPI_Bcast(&len, 1, MPI_INT, 0, mesh.comm());CHKERRXX(ierr);
          char *n = new char[len+1];
          ierr = MPI_Bcast(n, len+1, MPI_CHAR, 0, mesh.comm());CHKERRXX(ierr);
          name = n;
          delete [] n;
        }
        SectionSerializer::loadSection(fs, *section);
        mesh.setIntSection(name, section);
      }
      // Load overlap
#ifdef USE_NEW_OVERLAP
      PETSc::OverlapSerializer::loadOverlap(fs, *mesh.getSendOverlap());
      PETSc::OverlapSerializer::loadOverlap(fs, *mesh.getRecvOverlap());
#else
      SifterSerializer::loadSifter(fs, *mesh.getSendOverlap());
      SifterSerializer::loadSifter(fs, *mesh.getRecvOverlap());
#endif
      // Load distribution overlap
      // Load renumbering
    }
  };
} // namespace ALE
#endif
