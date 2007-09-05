#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#include <valarray>

#ifndef  included_ALE_Numbering_hh
#include <Numbering.hh>
#endif

#ifndef  included_ALE_Field_hh
#include <Field.hh>
#endif

#ifndef  included_ALE_SieveBuilder_hh
#include <SieveBuilder.hh>
#endif

#ifndef  included_ALE_LabelSifter_hh
#include <LabelSifter.hh>
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
#define NEW_LABEL
#ifdef NEW_LABEL
    typedef typename ALE::LabelSifter<int, point_type>                label_type;
#else
    typedef typename ALE::Sifter<int, point_type, int>                label_type;
#endif
    typedef typename std::map<const std::string, Obj<label_type> >    labels_type;
    typedef typename label_type::supportSequence                      label_sequence;
    typedef std::map<std::string, Obj<arrow_section_type> >           arrow_sections_type;
    typedef std::map<std::string, Obj<real_section_type> >            real_sections_type;
    typedef std::map<std::string, Obj<int_section_type> >             int_sections_type;
    typedef ALE::Point                                                index_type;
    typedef std::pair<index_type, int>                                oIndex_type;
    typedef std::vector<oIndex_type>                                  oIndexArray;
    typedef std::pair<int *, int>                                     indices_type;
    typedef NumberingFactory<this_type>                               NumberingFactory;
    typedef typename NumberingFactory::numbering_type                 numbering_type;
    typedef typename NumberingFactory::order_type                     order_type;
    typedef typename ALE::Sifter<int,point_type,point_type>           send_overlap_type;
    typedef typename ALE::Sifter<point_type,int,point_type>           recv_overlap_type;
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
    Obj<NumberingFactory> _factory;
    bool                   _calculatedOverlap;
    Obj<send_overlap_type> _sendOverlap;
    Obj<recv_overlap_type> _recvOverlap;
    Obj<send_overlap_type> _distSendOverlap;
    Obj<recv_overlap_type> _distRecvOverlap;
    // Work space
    Obj<std::set<point_type> > _modifiedPoints;
  public:
    Bundle(MPI_Comm comm, int debug = 0) : ALE::ParallelObject(comm, debug), _maxHeight(-1), _maxDepth(-1) {
      this->_indexArray        = new oIndexArray();
      this->_modifiedPoints    = new std::set<point_type>();
      this->_factory           = NumberingFactory::singleton(this->comm(), this->debug());
      this->_calculatedOverlap = false;
      this->_sendOverlap       = new send_overlap_type(comm, debug);
      this->_recvOverlap       = new recv_overlap_type(comm, debug);
    };
    Bundle(const Obj<sieve_type>& sieve) : ALE::ParallelObject(sieve->comm(), sieve->debug()), _sieve(sieve), _maxHeight(-1), _maxDepth(-1) {
      this->_indexArray        = new oIndexArray();
      this->_modifiedPoints    = new std::set<point_type>();
      this->_factory           = NumberingFactory::singleton(this->comm(), this->debug());
      this->_calculatedOverlap = false;
      this->_sendOverlap       = new send_overlap_type(comm, debug);
      this->_recvOverlap       = new recv_overlap_type(comm, debug);
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
    const Obj<NumberingFactory>& getFactory() const {return this->_factory;};
    const Obj<send_overlap_type>& getSendOverlap() const {return this->_sendOverlap;};
    void setSendOverlap(const Obj<send_overlap_type>& overlap) {this->_sendOverlap = overlap;};
    const Obj<recv_overlap_type>& getRecvOverlap() const {return this->_recvOverlap;};
    void setRecvOverlap(const Obj<recv_overlap_type>& overlap) {this->_recvOverlap = overlap;};
    const Obj<send_overlap_type>& getDistSendOverlap() const {return this->_distSendOverlap;};
    void setDistSendOverlap(const Obj<send_overlap_type>& overlap) {this->_distSendOverlap = overlap;};
    const Obj<recv_overlap_type>& getDistRecvOverlap() const {return this->_distRecvOverlap;};
    void setDistRecvOverlap(const Obj<recv_overlap_type>& overlap) {this->_distRecvOverlap = overlap;};
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
    };
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
    };
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
    };
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
    #define __FUNCT__ "Bundle::stratify"
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
    };
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
    };
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
    const indices_type getIndices(const Obj<Section_>& section, const point_type& p, const int level = -1) {
      this->_indexArray->clear();
      int size = 0;

      if (level == 0) {
        const index_type& idx = section->getIndex(p);

        this->_indexArray->push_back(oIndex_type(idx, 0));
        size += std::abs(idx.prefix);
      } else if ((level == 1) || (this->height() == 1)) {
        const Obj<typename sieve_type::coneSequence>& cone = this->_sieve->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();
        const index_type& idx = section->getIndex(p);

        this->_indexArray->push_back(idx);
        size += idx.prefix;
        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          const index_type& pIdx = section->getIndex(*p_iter);

          this->_indexArray->push_back(oIndex_type(pIdx, 0));
          size += std::abs(pIdx.prefix);
        }
      } else if (level == -1) {
        const Obj<oConeArray>         closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
        typename oConeArray::iterator end     = closure->end();

        for(typename oConeArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          const index_type& pIdx = section->getIndex(p_iter->first);

          this->_indexArray->push_back(oIndex_type(pIdx, p_iter->second));
          size += std::abs(pIdx.prefix);
        }
      } else {
        throw ALE::Exception("Bundle has not yet implemented nCone");
      }
      if (this->debug()) {
        for(typename oIndexArray::iterator i_iter = this->_indexArray->begin(); i_iter != this->_indexArray->end(); ++i_iter) {
          printf("[%d]index interval (%d, %d)\n", this->commRank(), i_iter->first.prefix, i_iter->first.index);
        }
      }
      int *indexArray = this->getIndexArray(size);

      this->expandIntervals(this->_indexArray, indexArray);
      return indices_type(indexArray, size);
    };
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
    };
  public: // Retrieval traversal
    // Return the values for the closure of this point
    //   use a smart pointer?
    template<typename Section_>
    const typename Section_::value_type *restrict(const Obj<Section_>& section, const point_type& p) {
      const int size = this->sizeWithBC(section, p);
      return this->restrict(section, p, section->getRawArray(size), size);
    };
    template<typename Section_>
    const typename Section_::value_type *restrict(const Obj<Section_>& section, const point_type& p, typename Section_::value_type  *values, const int valuesSize) {
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
    };
    template<typename Section_>
    const typename Section_::value_type *restrictNew(const Obj<Section_>& section, const point_type& p) {
      const int                       size    = this->sizeWithBC(section, p);
      typename Section_::value_type  *values  = section->getRawArray(size);
      const Obj<oConeArray>           closure = sieve_alg_type::orientedClosure(this, this->getArrowSection("orientation"), p);
      typename oConeArray::iterator   end     = closure->end();
      int                             j       = -1;

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
    };
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
    };
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
    };
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
    };
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
    };
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
    };
  public: // Allocation
    template<typename Section_>
    void allocate(const Obj<Section_>& section, const Obj<send_overlap_type>& sendOverlap = NULL) {
      bool doGhosts = !sendOverlap.isNull();

      this->_factory->orderPatch(section, this->getSieve(), sendOverlap);
      if (doGhosts) {
        if (this->_debug > 1) {std::cout << "Ordering patch for ghosts" << std::endl;}
        const typename Section_::chart_type& points = section->getChart();
        typename Section_::index_type::index_type offset = 0;

        for(typename Section_::chart_type::iterator point = points.begin(); point != points.end(); ++point) {
          const typename Section_::index_type& idx = section->getIndex(*point);

          offset = std::max(offset, idx.index + std::abs(idx.prefix));
        }
        this->_factory->orderPatch(section, this->getSieve(), NULL, offset);
        if (offset != section->sizeWithBC()) throw ALE::Exception("Inconsistent array sizes in section");
      }
      section->allocateStorage();
    };
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
      typename Section_::value_type                   *newArray = new typename Section_::value_type[newSize];
      const typename Section_::value_type             *array    = section->restrict();

      for(typename Section_::atlas_type::chart_type::const_iterator c_iter = oldChart.begin(); c_iter != oldChart.end(); ++c_iter) {
        const int& dim       = section->getFiberDimension(*c_iter);
        const int& offset    = atlas->restrictPoint(*c_iter)->index;
        const int& newOffset = newAtlas->restrictPoint(*c_iter)->index;

        for(int i = 0; i < dim; ++i) {
          newArray[newOffset+i] = array[offset+i];
        }
      }
      section->replaceStorage(newArray);
    };
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
    };
    void constructOverlap() {
      if (this->_calculatedOverlap) return;
      this->constructOverlap(this->getSieve()->base(), this->getSendOverlap(), this->getRecvOverlap());
      this->constructOverlap(this->getSieve()->cap(),  this->getSendOverlap(), this->getRecvOverlap());
      if (this->debug()) {
        this->_sendOverlap->view("Send overlap");
        this->_recvOverlap->view("Receive overlap");
      }
      this->_calculatedOverlap = true;
    };
  };
  class BoundaryCondition : public ALE::ParallelObject {
  public:
    typedef double (*function_type)(const double []);
    typedef double (*integrator_type)(const double [], const double [], const int, function_type);
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
    double evaluate(const double coords[]) const {return this->_func(coords);};
    double integrateDual(const double v0[], const double J[], const int dualIndex) const {return this->_integrator(v0, J, dualIndex, this->_func);};
  };
  class Discretization : public ALE::ParallelObject {
    typedef std::map<std::string, Obj<BoundaryCondition> > boundaryConditions_type;
  protected:
    boundaryConditions_type _boundaryConditions;
    Obj<BoundaryCondition> _exactSolution;
    std::map<int,int> _dim2dof;
    std::map<int,int> _dim2class;
    int           _quadSize;
    const double *_points;
    const double *_weights;
    int           _basisSize;
    const double *_basis;
    const double *_basisDer;
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
    const int     getQuadratureSize() {return this->_quadSize;};
    void          setQuadratureSize(const int size) {this->_quadSize = size;};
    const double *getQuadraturePoints() {return this->_points;};
    void          setQuadraturePoints(const double *points) {this->_points = points;};
    const double *getQuadratureWeights() {return this->_weights;};
    void          setQuadratureWeights(const double *weights) {this->_weights = weights;};
    const int     getBasisSize() {return this->_basisSize;};
    void          setBasisSize(const int size) {this->_basisSize = size;};
    const double *getBasis() {return this->_basis;};
    void          setBasis(const double *basis) {this->_basis = basis;};
    const double *getBasisDerivatives() {return this->_basisDer;};
    void          setBasisDerivatives(const double *basisDer) {this->_basisDer = basisDer;};
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
    int size(const Obj<Bundle>& mesh) {
      const Obj<typename Bundle::label_sequence>& cells   = mesh->heightStratum(0);
      const Obj<typename Bundle::coneArray>       closure = ALE::SieveAlg<Bundle>::closure(mesh, *cells->begin());
      const typename Bundle::coneArray::iterator  end     = closure->end();
      int                                         size    = 0;

      for(typename Bundle::coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
        size += this->_dim2dof[mesh->depth(*cl_iter)];
      }
      return size;
    };
  };
}

namespace ALE {
#ifdef NEW_SECTION
  class Mesh : public Bundle<ALE::Sieve<int,int,int>, GeneralSection<int, double> > {
#else
  class Mesh : public Bundle<ALE::Sieve<int,int,int> > {
#endif
  public:
#ifdef NEW_SECTION
    typedef Bundle<ALE::Sieve<int,int,int>, GeneralSection<int, double> > base_type;
#else
    typedef Bundle<ALE::Sieve<int,int,int> > base_type;
#endif
    typedef base_type::sieve_type            sieve_type;
    typedef sieve_type::point_type           point_type;
    typedef base_type::label_sequence        label_sequence;
    typedef base_type::real_section_type     real_section_type;
    typedef base_type::numbering_type        numbering_type;
    typedef base_type::order_type            order_type;
    typedef base_type::send_overlap_type     send_overlap_type;
    typedef base_type::recv_overlap_type     recv_overlap_type;
    typedef base_type::sieve_alg_type        sieve_alg_type;
    typedef std::set<std::string>            names_type;
    typedef std::map<std::string, Obj<Discretization> > discretizations_type;
  protected:
    int                  _dim;
    discretizations_type _discretizations;
    std::map<int,double> _periodicity;
  public:
    Mesh(MPI_Comm comm, int dim, int debug = 0) : base_type(comm, debug), _dim(dim) {
      ///this->_factory = NumberingFactory::singleton(debug);
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
  public: // Mesh geometry
    void computeTriangleGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrict(coordinates, e);
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
        PetscLogFlops(8 + 3);
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[1];
        invJ[2] = -invDet*J[2];
        invJ[3] =  invDet*J[0];
        PetscLogFlops(5);
      }
    };
    void computeQuadrilateralGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double point[], double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrict(coordinates, e);
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
        PetscLogFlops(6 + 16 + 3);
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[1];
        invJ[2] = -invDet*J[2];
        invJ[3] =  invDet*J[0];
        PetscLogFlops(5);
      }
    };
    void computeTetrahedronGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrict(coordinates, e);
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
        PetscLogFlops(18 + 12);
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
        PetscLogFlops(37);
      }
    };
    void computeHexahedralGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double point[], double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrict(coordinates, e);
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
        PetscLogFlops(39 + 81 + 12);
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
        PetscLogFlops(37);
      }
    };
    void computeElementGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      if (this->_dim == 2) {
        computeTriangleGeometry(coordinates, e, v0, J, invJ, detJ);
      } else if (this->_dim == 3) {
        computeTetrahedronGeometry(coordinates, e, v0, J, invJ, detJ);
      } else {
        throw ALE::Exception("Unsupport dimension for element geometry computation");
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
      for(label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        maxVolume = std::max(maxVolume, detJ*refVolume);
      }
      return maxVolume;
    };
    // Find the cell in which this point lies (stupid algorithm)
    point_type locatePoint_2D(const real_section_type::value_type point[]) {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const int                     embedDim    = 2;
      double v0[2], J[4], invJ[4], detJ;

      for(label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
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
    point_type locatePoint_3D(const real_section_type::value_type point[]) {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const int                     embedDim    = 3;
      double v0[3], J[9], invJ[9], detJ;

      for(label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
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
    point_type locatePoint(const real_section_type::value_type point[], point_type guess = -1) {
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
    void markBoundaryCells(const std::string& name, const int marker = 1, const int newMarker = 2) {
      const Obj<label_type>&     label    = this->getLabel(name);
      const Obj<label_sequence>& boundary = this->getLabelStratum(name, marker);
      const Obj<sieve_type>&     sieve    = this->getSieve();

      for(label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
        if (this->height(*e_iter) == 1) {
          const point_type cell = *sieve->support(*e_iter)->begin();

          this->setValue(label, cell, newMarker);
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

          numDof += disc->getNumDof(d);
          s->setFiberDimension(this->depthStratum(d), disc->getNumDof(d), f);
        }
        s->setFiberDimension(this->depthStratum(d), numDof);
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
          const label_sequence::iterator end       = exclusion->end();
          if (debug) {label->view(labelName.c_str());}

          for(label_sequence::iterator e_iter = exclusion->begin(); e_iter != end; ++e_iter) {
            const Obj<coneArray>      closure = ALE::SieveAlg<ALE::Mesh>::closure(this, this->getArrowSection("orientation"), *e_iter);
            const coneArray::iterator cEnd    = closure->end();

            for(coneArray::iterator c_iter = closure->begin(); c_iter != cEnd; ++c_iter) {
              if (seen.find(*c_iter) != seen.end()) continue;
              if (this->getValue(label, *c_iter) == 1) {
                seen.insert(*c_iter);
                s->setFiberDimension(*c_iter, 0, f);
                s->addFiberDimension(*c_iter, -disc->getNumDof(this->depth(*c_iter)));
                if (debug) {std::cout << "  cell: " << *c_iter << " dim: " << disc->getNumDof(this->depth(*c_iter)) << std::endl;}
              }
            }
          }
        }
      }
      // Process constraints
      f = 0;
      for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
        const Obj<ALE::Discretization>&    disc        = this->getDiscretization(*f_iter);
        const Obj<std::set<std::string> >  bcs         = disc->getBoundaryConditions();
        std::string                        excludeName = "exclude-"+*f_iter;

        for(std::set<std::string>::const_iterator bc_iter = bcs->begin(); bc_iter != bcs->end(); ++bc_iter) {
          const Obj<ALE::BoundaryCondition>& bc       = disc->getBoundaryCondition(*bc_iter);
          const Obj<label_sequence>&         boundary = this->getLabelStratum(bc->getLabelName(), bc->getMarker());

          bcLabels.insert(bc->getLabelName());
          if (this->hasLabel(excludeName)) {
            const Obj<label_type>& label = this->getLabel(excludeName);

            for(label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
              if (!this->getValue(label, *e_iter)) {
                s->addConstraintDimension(*e_iter, disc->getNumDof(this->depth(*e_iter)));
                s->setConstraintDimension(*e_iter, disc->getNumDof(this->depth(*e_iter)), f);
              }
            }
          } else {
            for(label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
              s->addConstraintDimension(*e_iter, disc->getNumDof(this->depth(*e_iter)));
              s->setConstraintDimension(*e_iter, disc->getNumDof(this->depth(*e_iter)), f);
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
      const coneArray::iterator  end     = closure->end();
      int                        offset  = 0;

      if (debug) {std::cout << "Closure for first element" << std::endl;}
      for(coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
        const int dim = this->depth(*cl_iter);

        if (debug) {std::cout << "  point " << *cl_iter << " depth " << dim << std::endl;}
        for(names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
          const Obj<Discretization>& disc = this->getDiscretization(*d_iter);
          const int                  num  = disc->getNumDof(dim);

          if (debug) {std::cout << "    disc " << disc->getName() << " numDof " << num << std::endl;}
          for(int o = 0; o < num; ++o) {
            indices[*d_iter].second[indices[*d_iter].first++] = offset++;
          }
        }
      }
      if (debug) {
        for(names_type::const_iterator d_iter = discs->begin(); d_iter != discs->end(); ++d_iter) {
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
      Obj<Mesh> mesh   = this;
      const int debug  = this->debug();
      int       marker = 0;
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
          const label_sequence::iterator end       = exclusion->end();

          if (debug) {std::cout << "Processing exclusion " << labelName << std::endl;}
          for(label_sequence::iterator e_iter = exclusion->begin(); e_iter != end; ++e_iter) {
            if (this->height(*e_iter)) continue;
            const Obj<coneArray>      closure = ALE::SieveAlg<ALE::Mesh>::closure(this, this->getArrowSection("orientation"), *e_iter);
            const coneArray::iterator clEnd   = closure->end();
            int                       offset  = 0;

            if (debug) {std::cout << "  Closure for cell " << *e_iter << std::endl;}
            for(coneArray::iterator cl_iter = closure->begin(); cl_iter != clEnd; ++cl_iter) {
              int g = 0;

              if (debug) {std::cout << "    point " << *cl_iter << std::endl;}
              for(names_type::const_iterator g_iter = dBegin; g_iter != dEnd; ++g_iter, ++g) {
                const int fDim = s->getFiberDimension(*cl_iter, g);

                if (debug) {std::cout << "      disc " << *g_iter << " numDof " << fDim << std::endl;}
                for(int d = 0; d < fDim; ++d) {
                  indices[*g_iter].second[indices[*g_iter].first++] = offset++;
                }
              }
            }
            const std::map<indices_type, int>::iterator entry = indexMap.find(indices);

            if (debug) {
              for(std::map<indices_type, int>::iterator i_iter = indexMap.begin(); i_iter != indexMap.end(); ++i_iter) {
                for(names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
                  std::cout << "Discretization (" << i_iter->second << ") " << *g_iter << " indices:";
                  for(int i = 0; i < ((indices_type) i_iter->first)[*g_iter].first; ++i) {
                    std::cout << " " << ((indices_type) i_iter->first)[*g_iter].second[i];
                  }
                  std::cout << std::endl;
                }
                std::cout << "Comparison: " << (indices == i_iter->first) << std::endl;
              }
              for(names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
                std::cout << "Discretization " << *g_iter << " indices:";
                for(int i = 0; i < indices[*g_iter].first; ++i) {
                  std::cout << " " << indices[*g_iter].second[i];
                }
                std::cout << std::endl;
              }
            }
            if (entry != indexMap.end()) {
              this->setValue(indexLabel, *e_iter, entry->second);
              if (debug) {std::cout << "  Found existing indices with marker " << entry->second << std::endl;}
            } else {
              indexMap[indices] = ++marker;
              this->setValue(indexLabel, *e_iter, marker);
              if (debug) {std::cout << "  Created new indices with marker " << marker << std::endl;}
            }
            for(names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
              indices[*g_iter].first  = 0;
              for(unsigned int i = 0; i < indices[*g_iter].second.size(); ++i) indices[*g_iter].second[i] = 0;
            }
          }
        }
      }
      if (debug) {indexLabel->view("cellExclusion");}
      for(std::map<indices_type, int>::iterator i_iter = indexMap.begin(); i_iter != indexMap.end(); ++i_iter) {
        if (debug) {std::cout << "Setting indices for marker " << i_iter->second << std::endl;}
        for(names_type::const_iterator g_iter = discs->begin(); g_iter != discs->end(); ++g_iter) {
          const Obj<Discretization>& disc = this->getDiscretization(*g_iter);
          const indexSet  indSet   = ((indices_type) i_iter->first)[*g_iter].second;
          const int       size     = indSet.size();
          int            *_indices = new int[size];

          if (debug) {std::cout << "  field " << *g_iter << std::endl;}
          for(int i = 0; i < size; ++i) {
            _indices[i] = indSet[i];
            if (debug) {std::cout << "    indices["<<i<<"] = " << _indices[i] << std::endl;}
          }
          disc->setIndices(_indices, i_iter->second);
        }
      }
    };
    void setupField(const Obj<real_section_type>& s, const int cellMarker = 2) {
      const Obj<names_type>& discs  = this->getDiscretizations();
      const int              debug  = s->debug();
      names_type             bcLabels;
      int                    maxDof;

      maxDof = setFiberDimensions(s, discs, bcLabels);
      this->calculateIndices();
      this->calculateIndicesExcluded(s, discs);
      this->allocate(s);
      s->defaultConstraintDof();
      const Obj<label_type>& cellExclusion = this->getLabel("cellExclusion");

      if (debug) {std::cout << "Setting boundary values" << std::endl;}
      for(names_type::const_iterator n_iter = bcLabels.begin(); n_iter != bcLabels.end(); ++n_iter) {
        const Obj<label_sequence>&     boundaryCells = this->getLabelStratum(*n_iter, cellMarker);
        const Obj<real_section_type>&  coordinates   = this->getRealSection("coordinates");
        const Obj<names_type>&         discs         = this->getDiscretizations();
        const point_type               firstCell     = *boundaryCells->begin();
        const int                      numFields     = discs->size();
        real_section_type::value_type *values        = new real_section_type::value_type[this->sizeWithBC(s, firstCell)];
        int                           *dofs          = new int[maxDof];
        int                           *v             = new int[numFields];
        double                        *v0            = new double[this->getDimension()];
        double                        *J             = new double[this->getDimension()*this->getDimension()];
        double                         detJ;

        for(label_sequence::iterator c_iter = boundaryCells->begin(); c_iter != boundaryCells->end(); ++c_iter) {
          const Obj<coneArray>      closure = sieve_alg_type::closure(this, this->getArrowSection("orientation"), *c_iter);
          const coneArray::iterator end     = closure->end();

          if (debug) {std::cout << "  Boundary cell " << *c_iter << std::endl;}
          this->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
          for(int f = 0; f < numFields; ++f) v[f] = 0;
          for(coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
            const int cDim = s->getConstraintDimension(*cl_iter);
            int       off  = 0;
            int       f    = 0;
            int       i    = -1;

            if (debug) {std::cout << "    point " << *cl_iter << std::endl;}
            if (cDim) {
              if (debug) {std::cout << "      constrained excMarker: " << this->getValue(cellExclusion, *c_iter) << std::endl;}
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
                    if (debug) {std::cout << "      field " << *f_iter << " marker " << value << std::endl;}
                    for(int d = 0; d < fDim; ++d, ++v[f]) {
                      dofs[++i] = off+d;
                      values[indices[v[f]]] = (*bc->getDualIntegrator())(v0, J, v[f], bc->getFunction());
                      if (debug) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                    }
                    // Allow only one condition per point
                    ++b;
                    break;
                  } else {
                    if (debug) {std::cout << "      field " << *f_iter << std::endl;}
                    for(int d = 0; d < fDim; ++d, ++v[f]) {
                      values[indices[v[f]]] = 0.0;
                      if (debug) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                    }
                  }
                }
                if (b == 0) {
                  if (debug) {std::cout << "      field " << *f_iter << std::endl;}
                  for(int d = 0; d < fDim; ++d, ++v[f]) {
                    values[indices[v[f]]] = 0.0;
                    if (debug) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                  }
                }
                off += fDim;
              }
              if (i != cDim-1) {throw ALE::Exception("Invalid constraint initialization");}
              s->setConstraintDof(*cl_iter, dofs);
            } else {
              if (debug) {std::cout << "      unconstrained" << std::endl;}
              for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
                const Obj<ALE::Discretization>& disc    = this->getDiscretization(*f_iter);
                const int                       fDim    = s->getFiberDimension(*cl_iter, f);//disc->getNumDof(this->depth(*cl_iter));
                const int                      *indices = disc->getIndices(this->getValue(cellExclusion, *c_iter));

                if (debug) {std::cout << "      field " << *f_iter << std::endl;}
                for(int d = 0; d < fDim; ++d, ++v[f]) {
                  values[indices[v[f]]] = 0.0;
                  if (debug) {std::cout << "      setting values["<<indices[v[f]]<<"] = " << values[indices[v[f]]] << std::endl;}
                }
              }
            }
          }
          if (debug) {
            const Obj<coneArray>      closure = sieve_alg_type::closure(this, this->getArrowSection("orientation"), *c_iter);
            const coneArray::iterator end     = closure->end();

            for(int f = 0; f < numFields; ++f) v[f] = 0;
            for(coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
              int f = 0;
              for(names_type::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
                const Obj<ALE::Discretization>& disc    = this->getDiscretization(*f_iter);
                const int                       fDim    = s->getFiberDimension(*cl_iter, f);
                const int                      *indices = disc->getIndices(this->getValue(cellExclusion, *c_iter));

                for(int d = 0; d < fDim; ++d, ++v[f]) {
                  std::cout << "    "<<*f_iter<<"-value["<<indices[v[f]]<<"] " << values[indices[v[f]]] << std::endl;
                }
              }
            }
          }
          this->updateAll(s, *c_iter, values);
        }
        delete [] dofs;
        delete [] values;
        delete [] v0;
        delete [] J;
      }
      if (debug) {s->view("");}
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
    };
  };
  class MeshBuilder {
    typedef ALE::Mesh Mesh;
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
    static Obj<ALE::Mesh> createSquareBoundary(const MPI_Comm comm, const double lower[], const double upper[], const int edges[], const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 1, debug);
      int       numVertices = (edges[0]+1)*(edges[1]+1);
      int       numEdges    = edges[0]*(edges[1]+1) + (edges[0]+1)*edges[1];
      double   *coords      = new double[numVertices*2];
      const Obj<Mesh::sieve_type> sieve    = new Mesh::sieve_type(mesh->comm(), mesh->debug());
      Mesh::point_type           *vertices = new Mesh::point_type[numVertices];
      int                         order    = 0;

      mesh->setSieve(sieve);
      const Obj<Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        /* Create sieve and ordering */
        for(int v = numEdges; v < numEdges+numVertices; v++) {
          vertices[v-numEdges] = Mesh::point_type(v);
        }
        for(int vy = 0; vy <= edges[1]; vy++) {
          for(int ex = 0; ex < edges[0]; ex++) {
            Mesh::point_type edge(vy*edges[0] + ex);
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
            Mesh::point_type edge(vx*edges[1] + ey + edges[0]*(edges[1]+1));
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
    static Obj<ALE::Mesh> createParticleInSquareBoundary(const MPI_Comm comm, const double lower[], const double upper[], const int edges[], const double radius, const int partEdges, const int debug = 0) {
      Obj<Mesh> mesh              = new Mesh(comm, 1, debug);
      const int numSquareVertices = (edges[0]+1)*(edges[1]+1);
      const int numVertices       = numSquareVertices + partEdges;
      const int numSquareEdges    = edges[0]*(edges[1]+1) + (edges[0]+1)*edges[1];
      const int numEdges          = numSquareEdges + partEdges;
      double   *coords            = new double[numVertices*2];
      const Obj<Mesh::sieve_type> sieve    = new Mesh::sieve_type(mesh->comm(), mesh->debug());
      Mesh::point_type           *vertices = new Mesh::point_type[numVertices];
      int                         order    = 0;

      mesh->setSieve(sieve);
      const Obj<Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        /* Create sieve and ordering */
        for(int v = numEdges; v < numEdges+numVertices; v++) {
          vertices[v-numEdges] = Mesh::point_type(v);
        }
        // Make square
        for(int vy = 0; vy <= edges[1]; vy++) {
          for(int ex = 0; ex < edges[0]; ex++) {
            Mesh::point_type edge(vy*edges[0] + ex);
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
            Mesh::point_type edge(vx*edges[1] + ey + edges[0]*(edges[1]+1));
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
          Mesh::point_type edge(numSquareEdges + ep);
          const int vertexA = numSquareVertices + ep;
          const int vertexB = numSquareVertices + (ep+1)%partEdges;

          sieve->addArrow(vertices[vertexA], edge, order++);
          sieve->addArrow(vertices[vertexB], edge, order++);
          mesh->setValue(markers, edge, 1);
          mesh->setValue(markers, vertices[vertexA], 1);
          mesh->setValue(markers, vertices[vertexB], 1);
        }
      }
      mesh->stratify();
      for(int vy = 0; vy <= edges[1]; ++vy) {
        for(int vx = 0; vx <= edges[0]; ++vx) {
          coords[(vy*(edges[0]+1)+vx)*2+0] = lower[0] + ((upper[0] - lower[0])/edges[0])*vx;
          coords[(vy*(edges[0]+1)+vx)*2+1] = lower[1] + ((upper[1] - lower[1])/edges[1])*vy;
        }
      }
      const double centroidX = 0.5*(upper[0] + lower[0]);
      const double centroidY = 0.5*(upper[1] + lower[1]);
      for(int vp = 0; vp < partEdges; ++vp) {
        const double rad = 2.0*PETSC_PI*vp/partEdges;
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
    static Obj<ALE::Mesh> createReentrantBoundary(const MPI_Comm comm, const double lower[], const double upper[], double notchpercent[], const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 1, debug);
      int       numVertices = 6;
      int       numEdges    = numVertices;
      double   *coords      = new double[numVertices*2];
      const Obj<Mesh::sieve_type> sieve    = new Mesh::sieve_type(mesh->comm(), mesh->debug());
      Mesh::point_type           *vertices = new Mesh::point_type[numVertices];

      mesh->setSieve(sieve);
      const Obj<Mesh::label_type>& markers = mesh->createLabel("marker");
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

       ___--1
      |     |
      |     |
      |     |
      -     0-----n
       -          |
        -         |
         -___  _|
             --
    */
    static Obj<ALE::Mesh> createCircularReentrantBoundary(const MPI_Comm comm, const int segments, const double radius, const double arc_percent, const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 1, debug);
      int       numVertices = segments+2;
      int       numEdges    = numVertices;
      double   *coords      = new double[numVertices*2];
      const Obj<Mesh::sieve_type> sieve    = new Mesh::sieve_type(mesh->comm(), mesh->debug());
      Mesh::point_type           *vertices = new Mesh::point_type[numVertices];

      mesh->setSieve(sieve);
      const Obj<Mesh::label_type>& markers = mesh->createLabel("marker");
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

        double anglestep = arc_percent*2.*3.14159265/((float)segments);

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
    static Obj<Mesh> createCubeBoundary(const MPI_Comm comm, const double lower[], const double upper[], const int faces[], const int debug = 0) {
      Obj<Mesh> mesh        = new Mesh(comm, 2, debug);
      int       numVertices = (faces[0]+1)*(faces[1]+1)*(faces[2]+1);
      int       numFaces    = 6;
      double   *coords      = new double[numVertices*3];
      const Obj<Mesh::sieve_type> sieve    = new Mesh::sieve_type(mesh->comm(), mesh->debug());
      Mesh::point_type           *vertices = new Mesh::point_type[numVertices];
      int                         order    = 0;

      mesh->setSieve(sieve);
      const Obj<Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        /* Create sieve and ordering */
        for(int v = numFaces; v < numFaces+numVertices; v++) {
          vertices[v-numFaces] = Mesh::point_type(v);
          mesh->setValue(markers, vertices[v-numFaces], 1);
        }
        {
          // Side 0 (Front)
          Mesh::point_type face(0);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 1 (Back)
          Mesh::point_type face(1);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 2 (Bottom)
          Mesh::point_type face(2);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[0], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 3 (Top)
          Mesh::point_type face(3);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 4 (Left)
          Mesh::point_type face(4);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 5 (Right)
          Mesh::point_type face(5);
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
    static Obj<Mesh> createParticleInCubeBoundary(const MPI_Comm comm, const double lower[], const double upper[], const int faces[], const double radius, const int thetaEdges, const int phiSlices, const int debug = 0) {
      Obj<Mesh> mesh            = new Mesh(comm, 2, debug);
      const int numCubeVertices = (faces[0]+1)*(faces[1]+1)*(faces[2]+1);
      const int numPartVertices = (thetaEdges - 1)*phiSlices + 2;
      const int numVertices     = numCubeVertices + numPartVertices;
      const int numCubeFaces    = 6;
      const int numFaces        = numCubeFaces + thetaEdges*phiSlices;
      double   *coords          = new double[numVertices*3];
      const Obj<Mesh::sieve_type> sieve    = new Mesh::sieve_type(mesh->comm(), mesh->debug());
      Mesh::point_type           *vertices = new Mesh::point_type[numVertices];
      int                         order    = 0;

      mesh->setSieve(sieve);
      const Obj<Mesh::label_type>& markers = mesh->createLabel("marker");
      if (mesh->commRank() == 0) {
        // Make cube
        for(int v = numFaces; v < numFaces+numVertices; v++) {
          vertices[v-numFaces] = Mesh::point_type(v);
          mesh->setValue(markers, vertices[v-numFaces], 1);
        }
        {
          // Side 0 (Front)
          Mesh::point_type face(0);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 1 (Back)
          Mesh::point_type face(1);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 2 (Bottom)
          Mesh::point_type face(2);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[0], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 3 (Top)
          Mesh::point_type face(3);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 4 (Left)
          Mesh::point_type face(4);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 5 (Right)
          Mesh::point_type face(5);
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
            Mesh::point_type face(numCubeFaces + s*thetaEdges + ep);
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
            mesh->setValue(markers, face, 1);
            mesh->setValue(markers, vertices[vertexA], 1);
            mesh->setValue(markers, vertices[vertexB], 1);
            if (vertexB != vertexC) mesh->setValue(markers, vertices[vertexC], 1);
            if (vertexA != vertexD) mesh->setValue(markers, vertices[vertexD], 1);
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
      const double centroidX = 0.5*(upper[0] + lower[0]);
      const double centroidY = 0.5*(upper[1] + lower[1]);
      const double centroidZ = 0.5*(upper[2] + lower[2]);
      for(int s = 0; s < phiSlices; ++s) {
        for(int v = 0; v <= thetaEdges; ++v) {
          int          vertex  = numCubeVertices + v + s*(thetaEdges+1);
          const double theta   = v*(PETSC_PI/thetaEdges);
          const double phi     = s*(2.0*PETSC_PI/phiSlices);
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
  };
} // namespace ALE

namespace ALECompat {
  using ALE::Obj;
  template<typename Topology_>
  class Bundle : public ALE::ParallelObject {
  public:
    typedef Topology_                                      topology_type;
    typedef typename topology_type::patch_type             patch_type;
    typedef typename topology_type::point_type             point_type;
    typedef typename topology_type::sieve_type             sieve_type;
    typedef typename sieve_type::coneArray                 coneArray;
    typedef ALECompat::New::Section<topology_type, double> real_section_type;
    typedef ALECompat::New::Section<topology_type, int>    int_section_type;
    typedef ALE::MinimalArrow<point_type, point_type>      arrow_type;
    typedef ALE::UniformSection<arrow_type, int>           arrow_section_type;
    typedef struct {double x, y, z;}                       split_value;
    typedef ALE::pair<point_type, split_value>             pair_type;
    typedef ALECompat::New::Section<topology_type, pair_type>    pair_section_type;
    typedef std::map<std::string, Obj<real_section_type> > real_sections_type;
    typedef std::map<std::string, Obj<int_section_type> >  int_sections_type;
    typedef std::map<std::string, Obj<pair_section_type> > pair_sections_type;
    typedef std::map<std::string, Obj<arrow_section_type> > arrow_sections_type;
    typedef typename topology_type::send_overlap_type      send_overlap_type;
    typedef typename topology_type::recv_overlap_type      recv_overlap_type;
    typedef typename ALECompat::New::SectionCompletion<topology_type, point_type>::topology_type      comp_topology_type;
    typedef typename ALECompat::New::OverlapValues<send_overlap_type, comp_topology_type, point_type> send_section_type;
    typedef typename ALECompat::New::OverlapValues<recv_overlap_type, comp_topology_type, point_type> recv_section_type;
  protected:
    Obj<topology_type> _topology;
    bool               _distributed;
    real_sections_type _realSections;
    int_sections_type  _intSections;
    pair_sections_type _pairSections;
    arrow_sections_type _arrowSections;
  public:
    Bundle(MPI_Comm comm, int debug = 0) : ALE::ParallelObject(comm, debug), _distributed(false) {
      this->_topology = new topology_type(comm, debug);
    };
    Bundle(const Obj<topology_type>& topology) : ALE::ParallelObject(topology->comm(), topology->debug()), _topology(topology), _distributed(false) {};
  public: // Accessors
    bool getDistributed() const {return this->_distributed;};
    void setDistributed(const bool distributed) {this->_distributed = distributed;};
    const Obj<topology_type>& getTopology() const {return this->_topology;};
    void setTopology(const Obj<topology_type>& topology) {this->_topology = topology;};
  public:
    bool hasRealSection(const std::string& name) {
      return this->_realSections.find(name) != this->_realSections.end();
    };
    const Obj<real_section_type>& getRealSection(const std::string& name) {
      if (this->_realSections.find(name) == this->_realSections.end()) {
        Obj<real_section_type> section = new real_section_type(this->_topology);

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
    bool hasIntSection(const std::string& name) {
      return this->_intSections.find(name) != this->_intSections.end();
    };
    const Obj<int_section_type>& getIntSection(const std::string& name) {
      if (this->_intSections.find(name) == this->_intSections.end()) {
        Obj<int_section_type> section = new int_section_type(this->_topology);

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
    bool hasPairSection(const std::string& name) {
      return this->_pairSections.find(name) != this->_pairSections.end();
    };
    const Obj<pair_section_type>& getPairSection(const std::string& name) {
      if (this->_pairSections.find(name) == this->_pairSections.end()) {
        Obj<pair_section_type> section = new pair_section_type(this->_topology);

        if (this->_debug) {std::cout << "Creating new pair section: " << name << std::endl;}
        this->_pairSections[name] = section;
      }
      return this->_pairSections[name];
    };
    void setPairSection(const std::string& name, const Obj<pair_section_type>& section) {
      this->_pairSections[name] = section;
    };
    Obj<std::set<std::string> > getPairSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename pair_sections_type::const_iterator s_iter = this->_pairSections.begin(); s_iter != this->_pairSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasArrowSection(const std::string& name) {
      return this->_arrowSections.find(name) != this->_arrowSections.end();
    };
    const Obj<arrow_section_type>& getArrowSection(const std::string& name) {
      if (this->_arrowSections.find(name) == this->_arrowSections.end()) {
        Obj<arrow_section_type> section = new arrow_section_type(this->comm(), this->debug());

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
  public: // Printing
    friend std::ostream& operator<<(std::ostream& os, const split_value& s) {
      os << "(" << s.x << ", "<< s.y << ", "<< s.z << ")";
      return os;
    };
  public: // Adapter
    const Obj<sieve_type>& getSieve() {return this->_topology->getPatch(0);};
    int height() {return 2;};
    int depth() {return 2;};
  };

  class Discretization : public ALE::ParallelObject {
  protected:
    std::map<int,int> _dim2dof;
    std::map<int,int> _dim2class;
  public:
    Discretization(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {};
    ~Discretization() {};
  public:
    const double *getQuadraturePoints() {return NULL;};
    const double *getQuadratureWeights() {return NULL;};
    const double *getBasis() {return NULL;};
    const double *getBasisDerivatives() {return NULL;};
    int  getNumDof(const int dim) {return this->_dim2dof[dim];};
    void setNumDof(const int dim, const int numDof) {this->_dim2dof[dim] = numDof;};
    int  getDofClass(const int dim) {return this->_dim2class[dim];};
    void setDofClass(const int dim, const int dofClass) {this->_dim2class[dim] = dofClass;};
  };

  class BoundaryCondition : public ALE::ParallelObject {
  protected:
    std::string _labelName;
    double    (*_func)(const double []);
  public:
    BoundaryCondition(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {};
    ~BoundaryCondition() {};
  public:
    const std::string& getLabelName() {return this->_labelName;};
    void setLabelName(const std::string& name) {this->_labelName = name;};
    void setFunction(double (*func)(const double [])) {this->_func = func;};
  public:
    double evaluate(const double coords[]) {return this->_func(coords);};
  };

  class Mesh : public Bundle<ALE::Topology<int, ALE::Sieve<int,int,int> > > {
  public:
    typedef int                                       point_type;
    typedef ALE::Sieve<point_type,int,int>            sieve_type;
    typedef ALE::Topology<int, sieve_type>            topology_type;
    typedef topology_type::patch_type                 patch_type;
    typedef Bundle<topology_type>                     base_type;
    typedef ALECompat::New::NumberingFactory<topology_type> NumberingFactory;
    typedef NumberingFactory::numbering_type          numbering_type;
    typedef NumberingFactory::order_type              order_type;
    typedef base_type::send_overlap_type              send_overlap_type;
    typedef base_type::recv_overlap_type              recv_overlap_type;
    typedef base_type::send_section_type              send_section_type;
    typedef base_type::recv_section_type              recv_section_type;
    typedef base_type::real_sections_type             real_sections_type;
    // PCICE BC
    typedef struct {double rho,u,v,p;}                bc_value_type;
    typedef std::map<int, bc_value_type>              bc_values_type;
    // PyLith BC
    typedef ALECompat::New::Section<topology_type, ALE::pair<int,double> > foliated_section_type;
  protected:
    int                   _dim;
    Obj<NumberingFactory> _factory;
    // PCICE BC
    bc_values_type        _bcValues;
    // PyLith BC
    Obj<foliated_section_type> _boundaries;
    // Discretization
    Obj<Discretization>    _discretization;
    Obj<BoundaryCondition> _boundaryCondition;
  public:
    Mesh(MPI_Comm comm, int dim, int debug = 0) : Bundle<ALE::Topology<int, ALE::Sieve<int,int,int> > >(comm, debug), _dim(dim) {
      this->_factory = NumberingFactory::singleton(debug);
      this->_boundaries = NULL;
      this->_discretization = new Discretization(comm, debug);
      this->_boundaryCondition = new BoundaryCondition(comm, debug);
    };
    Mesh(const Obj<topology_type>& topology, int dim) : Bundle<ALE::Topology<int, ALE::Sieve<int,int,int> > >(topology), _dim(dim) {
      this->_factory = NumberingFactory::singleton(topology->debug());
      this->_boundaries = NULL;
    };
  public: // Accessors
    int getDimension() const {return this->_dim;};
    void setDimension(const int dim) {this->_dim = dim;};
    const Obj<NumberingFactory>& getFactory() {return this->_factory;};
    const Obj<Discretization>&    getDiscretization() {return this->_discretization;};
    void setDiscretization(const Obj<Discretization>& discretization) {this->_discretization = discretization;};
    const Obj<BoundaryCondition>& getBoundaryCondition() {return this->_boundaryCondition;};
    void setBoundaryCondition(const Obj<BoundaryCondition>& boundaryCondition) {this->_boundaryCondition = boundaryCondition;};
  public: // Mesh geometry
#if 0
    void computeTriangleGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrict(coordinates, e);
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
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[1];
        invJ[2] = -invDet*J[2];
        invJ[3] =  invDet*J[0];
      }
    };
    static void computeTetrahedronGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      const patch_type patch  = 0;
      const double    *coords = coordinates->restrict(patch, e);
      const int        dim    = 3;
      double           invDet;

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
        detJ = J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
          J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
          J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
        invJ[0*3+1] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
        invJ[0*3+2] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        invJ[1*3+0] = invDet*(J[0*3+1]*J[2*3+2] - J[0*3+2]*J[2*3+1]);
        invJ[1*3+1] = invDet*(J[0*3+2]*J[2*3+0] - J[0*3+0]*J[2*3+2]);
        invJ[1*3+2] = invDet*(J[0*3+0]*J[2*3+1] - J[0*3+1]*J[2*3+0]);
        invJ[2*3+0] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
        invJ[2*3+1] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
        invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
      }
    };
    void computeElementGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      if (this->_dim == 2) {
        computeTriangleGeometry(coordinates, e, v0, J, invJ, detJ);
      } else if (this->_dim == 3) {
        computeTetrahedronGeometry(coordinates, e, v0, J, invJ, detJ);
      } else {
        throw ALE::Exception("Unsupport dimension for element geometry computation");
      }
    };
    double getMaxVolume() {
      const topology_type::sheaf_type& patches = this->getTopology()->getPatches();
      double v0[3], J[9], invJ[9], detJ, maxVolume = 0.0;

      for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        const Obj<real_section_type>&             coordinates = this->getRealSection("coordinates");
        const Obj<topology_type::label_sequence>& cells       = this->getTopology()->heightStratum(p_iter->first, 0);

        for(topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
          this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
          maxVolume = std::max(maxVolume, detJ);
        }
      }
      return maxVolume;
    };
    // Find the cell in which this point lies (stupid algorithm)
    point_type locatePoint_2D(const patch_type& patch, const real_section_type::value_type point[]) {
      const Obj<real_section_type>&             coordinates = this->getRealSection("coordinates");
      const Obj<topology_type::label_sequence>& cells       = this->getTopology()->heightStratum(patch, 0);
      const int                                 embedDim    = 2;
      double v0[2], J[4], invJ[4], detJ;

      for(topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        double xi   = invJ[0*embedDim+0]*(point[0] - v0[0]) + invJ[0*embedDim+1]*(point[1] - v0[1]);
        double eta  = invJ[1*embedDim+0]*(point[0] - v0[0]) + invJ[1*embedDim+1]*(point[1] - v0[1]);

        if ((xi >= 0.0) && (eta >= 0.0) && (xi + eta <= 1.0)) {
          return *c_iter;
        }
      }
      throw ALE::Exception("Could not locate point");
    };
    //   Assume a simplex and 3D
    point_type locatePoint_3D(const patch_type& patch, const real_section_type::value_type point[]) {
      const Obj<real_section_type>&             coordinates = this->getRealSection("coordinates");
      const Obj<topology_type::label_sequence>& cells       = this->getTopology()->heightStratum(patch, 0);
      const int                                 embedDim    = 3;
      double v0[3], J[9], invJ[9], detJ;

      for(topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        double xi   = invJ[0*embedDim+0]*(point[0] - v0[0]) + invJ[0*embedDim+1]*(point[1] - v0[1]) + invJ[0*embedDim+2]*(point[2] - v0[2]);
        double eta  = invJ[1*embedDim+0]*(point[0] - v0[0]) + invJ[1*embedDim+1]*(point[1] - v0[1]) + invJ[1*embedDim+2]*(point[2] - v0[2]);
        double zeta = invJ[2*embedDim+0]*(point[0] - v0[0]) + invJ[2*embedDim+1]*(point[1] - v0[1]) + invJ[2*embedDim+2]*(point[2] - v0[2]);

        if ((xi >= 0.0) && (eta >= 0.0) && (zeta >= 0.0) && (xi + eta + zeta <= 1.0)) {
          return *c_iter;
        }
      }
      throw ALE::Exception("Could not locate point");
    };
    point_type locatePoint(const patch_type& patch, const real_section_type::value_type point[]) {
      if (this->_dim == 2) {
        return locatePoint_2D(patch, point);
      } else if (this->_dim == 3) {
        return locatePoint_3D(patch, point);
      } else {
        throw ALE::Exception("No point location for mesh dimension");
      }
    };
#endif
    // Only works in 2D
    int orientation(const patch_type& patch, const point_type& cell) {
      const Obj<real_section_type>&     coordinates = this->getRealSection("coordinates");
      const Obj<topology_type>&         topology    = this->getTopology();
      const Obj<sieve_type>&            sieve       = topology->getPatch(patch);
      const Obj<sieve_type::coneArray>& cone        = sieve->nCone(cell, topology->depth());
      sieve_type::coneArray::iterator   cBegin      = cone->begin();
      real_section_type::value_type     root[2];
      real_section_type::value_type     vA[2];
      real_section_type::value_type     vB[2];

      const real_section_type::value_type *coords = coordinates->restrictPoint(patch, *cBegin);
      root[0] = coords[0];
      root[1] = coords[1];
      ++cBegin;
      coords = coordinates->restrictPoint(patch, *cBegin);
      vA[0] = coords[0] - root[0];
      vA[1] = coords[1] - root[1];
      ++cBegin;
      coords = coordinates->restrictPoint(patch, *cBegin);
      vB[0] = coords[0] - root[0];
      vB[1] = coords[1] - root[1];
      double det = vA[0]*vB[1] - vA[1]*vB[0];
      if (det > 0.0) return  1;
      if (det < 0.0) return -1;
      return 0;
    };
  public: // BC values for PCICE
    const bc_value_type& getBCValue(const int bcFunc) {
      return this->_bcValues[bcFunc];
    };
    void setBCValue(const int bcFunc, const bc_value_type& value) {
      this->_bcValues[bcFunc] = value;
    };
    bc_values_type& getBCValues() {
      return this->_bcValues;
    };
    void distributeBCValues() {
      int size = this->_bcValues.size();

      MPI_Bcast(&size, 1, MPI_INT, 0, this->comm()); 
      if (this->commRank()) {
        for(int bc = 0; bc < size; ++bc) {
          int           funcNum;
          bc_value_type funcVal;

          MPI_Bcast((void *) &funcNum, 1, MPI_INT,    0, this->comm());
          MPI_Bcast((void *) &funcVal, 4, MPI_DOUBLE, 0, this->comm());
          this->_bcValues[funcNum] = funcVal;
        }
      } else {
        for(bc_values_type::iterator bc_iter = this->_bcValues.begin(); bc_iter != this->_bcValues.end(); ++bc_iter) {
          const int&           funcNum = bc_iter->first;
          const bc_value_type& funcVal = bc_iter->second;
          MPI_Bcast((void *) &funcNum, 1, MPI_INT,    0, this->comm());
          MPI_Bcast((void *) &funcVal, 4, MPI_DOUBLE, 0, this->comm());
        }
      }
    };
  public: // BC values for PyLith
    const Obj<foliated_section_type>& getBoundariesNew() {
      if (this->_boundaries.isNull()) {
        this->_boundaries = new foliated_section_type(this->getTopology());
      }
      return this->_boundaries;
    };
  public: // Discretization
    void setupField(const Obj<real_section_type>& s, const bool postponeGhosts = false) {
      const std::string& name  = this->_boundaryCondition->getLabelName();
      const patch_type   patch = 0;

      for(int d = 0; d <= this->_dim; ++d) {
        s->setFiberDimensionByDepth(patch, d, this->_discretization->getNumDof(d));
      }
      if (!name.empty()) {
        const Obj<topology_type::label_sequence>& boundary = this->_topology->getLabelStratum(patch, name, 1);

        for(topology_type::label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
          s->setFiberDimension(patch, *e_iter, -this->_discretization->getNumDof(this->_topology->depth(patch, *e_iter)));
        }
      }
      s->allocate(postponeGhosts);
      if (!name.empty()) {
        const Obj<real_section_type>&             coordinates = this->getRealSection("coordinates");
        const Obj<topology_type::label_sequence>& boundary    = this->_topology->getLabelStratum(patch, name, 1);

        for(topology_type::label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
          const real_section_type::value_type *coords = coordinates->restrictPoint(patch, *e_iter);
          const PetscScalar                    value  = this->_boundaryCondition->evaluate(coords);

          s->updatePointBC(patch, *e_iter, &value);
        }
      }
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
      this->getTopology()->view("mesh topology", comm);
      Obj<std::set<std::string> > sections = this->getRealSections();

      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getRealSection(*name)->view(*name);
      }
      sections = this->getIntSections();
      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getIntSection(*name)->view(*name);
      }
      sections = this->getPairSections();
      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getPairSection(*name)->view(*name);
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
    };
  };
} // namespace ALECompat
#endif
