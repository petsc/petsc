#ifndef included_ALE_Mesh_hh
#define included_ALE_Mesh_hh

#ifndef  included_ALE_Numbering_hh
#include <Numbering.hh>
#endif

#ifndef  included_ALE_Field_hh
#include <Field.hh>
#endif

namespace ALE {
  template<typename Sieve_>
  class Bundle : public ALE::ParallelObject {
  public:
    typedef Sieve_                                                    sieve_type;
    typedef typename sieve_type::point_type                           point_type;
    typedef typename ALE::Sifter<int, point_type, int>                label_type;
    typedef typename std::map<const std::string, Obj<label_type> >    labels_type;
    typedef typename label_type::supportSequence                      label_sequence;
    typedef UniformSection<MinimalArrow<point_type, point_type>, int> arrow_section_type;
    typedef std::map<std::string, Obj<arrow_section_type> >           arrow_sections_type;
    typedef Section<point_type, double>                               real_section_type;
    typedef std::map<std::string, Obj<real_section_type> >            real_sections_type;
    typedef Section<point_type, int>                                  int_section_type;
    typedef std::map<std::string, Obj<int_section_type> >             int_sections_type;
    typedef ALE::Point                                                index_type;
    typedef typename sieve_type::coneArray                            coneArray;
    typedef typename sieve_type::supportArray                         supportArray;
    typedef std::vector<index_type>                                   indexArray;
    typedef NumberingFactory<Bundle<sieve_type> >                     NumberingFactory;
    typedef typename NumberingFactory::numbering_type                 numbering_type;
    typedef typename NumberingFactory::order_type                     order_type;
    typedef typename ALE::Sifter<int,point_type,point_type>           send_overlap_type;
    typedef typename ALE::Sifter<point_type,int,point_type>           recv_overlap_type;
  protected:
    Obj<sieve_type>       _sieve;
    labels_type           _labels;
    int                   _maxHeight;
    int                   _maxDepth;
    arrow_sections_type   _arrowSections;
    real_sections_type    _realSections;
    int_sections_type     _intSections;
    Obj<indexArray>       _indexArray;
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
      this->_indexArray        = new indexArray();
      this->_modifiedPoints    = new std::set<point_type>();
      this->_factory           = NumberingFactory::singleton(this->comm(), this->debug());
      this->_calculatedOverlap = false;
      this->_sendOverlap       = new send_overlap_type(comm, debug);
      this->_recvOverlap       = new recv_overlap_type(comm, debug);
    };
    Bundle(const Obj<sieve_type>& sieve) : ALE::ParallelObject(sieve->comm(), sieve->debug()), _sieve(sieve), _maxHeight(-1), _maxDepth(-1) {
      this->_indexArray        = new indexArray();
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
    const labels_type& getLabels() {
      return this->_labels;
    };
    const Obj<label_sequence>& getLabelStratum(const std::string& name, int value) {
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
    int height() const {return this->_maxHeight;};
    int height(const point_type& point) {
      return this->getValue(this->_labels["height"], point, -1);
    };
    const Obj<label_sequence>& heightStratum(int height) {
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
    int depth() const {return this->_maxDepth;};
    int depth(const point_type& point) {
      return this->getValue(this->_labels["depth"], point, -1);
    };
    const Obj<label_sequence>& depthStratum(int depth) {
      return this->getLabelStratum("depth", depth);
    };
    #undef __FUNCT__
    #define __FUNCT__ "Bundle::stratify"
    void stratify() {
      ALE_LOG_EVENT_BEGIN;
      this->computeHeights();
      this->computeDepths();
      ALE_LOG_EVENT_END;
    };
  public: // Size traversal
    template<typename Section_>
    int size(const Obj<Section_>& section, const point_type& p) {
      const typename Section_::chart_type&  chart   = section->getAtlas()->getChart();
      const Obj<coneArray>                  closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
      typename coneArray::iterator          end     = closure->end();
      int                                   size    = 0;

      for(typename coneArray::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
        if (chart.count(*c_iter)) {
          size += section->getConstrainedFiberDimension(*c_iter);
        }
      }
      return size;
    };
    template<typename Section_>
    int sizeWithBC(const Obj<Section_>& section, const point_type& p) {
      const typename Section_::chart_type&  chart   = section->getAtlas()->getChart();
      const Obj<coneArray>                  closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
      typename coneArray::iterator          end     = closure->end();
      int                                   size    = 0;

      for(typename coneArray::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
        if (chart.count(*c_iter)) {
          size += std::abs(section->getFiberDimension(*c_iter));
        }
      }
      return size;
    };
  public: // Index traversal
    template<typename Section_>
    const Obj<indexArray>& getIndices(const Obj<Section_>& section, const point_type& p, const int level = -1) {
      this->_indexArray->clear();

      if (level == 0) {
        this->_indexArray->push_back(section->getIndex(p));
      } else if ((level == 1) || (this->height() == 1)) {
        const Obj<typename sieve_type::coneSequence>& cone = this->_sieve->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        this->_indexArray->push_back(section->getIndex(p));
        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          this->_indexArray->push_back(section->getIndex(*p_iter));
        }
      } else if (level == -1) {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          this->_indexArray->push_back(section->getIndex(*p_iter));
        }
      } else {
        throw ALE::Exception("Bundle has not yet implemented nCone");
      }
      return this->_indexArray;
    };
    template<typename Section_, typename Numbering>
    const Obj<indexArray>& getIndices(const Obj<Section_>& section, const point_type& p, const Obj<Numbering>& numbering, const int level = -1) {
      this->_indexArray->clear();

      if (level == 0) {
        this->_indexArray->push_back(section->getIndex(p, numbering));
      } else if ((level == 1) || (this->height() == 1)) {
        const Obj<typename sieve_type::coneSequence>& cone = this->_sieve->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        this->_indexArray->push_back(section->getIndex(p, numbering));
        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          this->_indexArray->push_back(section->getIndex(*p_iter, numbering));
        }
      } else if (level == -1) {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename sieve_type::coneArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          this->_indexArray->push_back(section->getIndex(*p_iter, numbering));
        }
      } else {
        throw ALE::Exception("Bundle has not yet implemented nCone");
      }
      return this->_indexArray;
    };
  public: // Retrieval traversal
    // Return the values for the closure of this point
    //   use a smart pointer?
    template<typename Section_>
    const typename Section_::value_type *restrict(const Obj<Section_>& section, const point_type& p) {
      const int                             size   = this->sizeWithBC(section, p);
      typename Section_::value_type        *values = section->getRawArray(size);
      int                                   j      = -1;

      // We could actually ask for the height of the individual point
      if (this->height() < 2) {
        const int& dim = std::abs(section->getFiberDimension(p));
        const typename Section_::value_type *array = section->restrictPoint(p);

        for(int i = 0; i < dim; ++i) {
          values[++j] = array[i];
        }
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          const int& dim = std::abs(section->getFiberDimension(*p_iter));

          array = section->restrictPoint(*p_iter);
          for(int i = 0; i < dim; ++i) {
            values[++j] = array[i];
          }
        }
      } else {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename coneArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          const int& dim = std::abs(section->getFiberDimension(*p_iter));
          const typename Section_::value_type *array = section->restrictPoint(*p_iter);

          for(int i = 0; i < dim; ++i) {
            values[++j] = array[i];
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
        j += std::abs(section->getFiberDimension(p));
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updatePoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      } else {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename coneArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updatePoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      }
    };
    template<typename Section_>
    void updateAdd(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      if (this->height() < 2) {
        section->updateAddPoint(p, &v[j]);
        j += std::abs(section->getFiberDimension(p));
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updateAddPoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      } else {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename coneArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updateAddPoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      }
    };
    template<typename Section_>
    void updateBC(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      if (this->height() < 2) {
        section->updateBCPoint(p, &v[j]);
        j += std::abs(section->getFiberDimension(p));
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updateBCPoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      } else {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename coneArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updateBCPoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      }
    };
  public: // Allocation
    template<typename Section_>
    void allocate(const Obj<Section_>& section, const Obj<send_overlap_type>& sendOverlap = NULL) {
      bool doGhosts = !sendOverlap.isNull();

      this->_factory->orderPatch(section->getAtlas(), this->getSieve(), sendOverlap);
      if (doGhosts) {
        if (this->_debug > 1) {std::cout << "Ordering patch for ghosts" << std::endl;}
        const typename Section_::atlas_type::chart_type& points = section->getAtlas()->getChart();
        int offset = 0;

        for(typename Section_::atlas_type::chart_type::iterator point = points.begin(); point != points.end(); ++point) {
          const typename Section_::index_type& idx = section->getIndex(*point);

          offset = std::max(offset, idx.index + std::abs(idx.prefix));
        }
        this->_factory->orderPatch(section->getAtlas(), this->getSieve(), NULL, offset);
        if (offset != section->sizeWithBC()) throw ALE::Exception("Inconsistent array sizes in section");
      }
      section->allocateStorage();
    };
    template<typename Section_>
    void reallocate(const Obj<Section_>& section) {
      if (section->getNewAtlas().isNull()) return;
      // Since copy() preserves offsets, we must reinitialize them before ordering
      const Obj<typename Section_::atlas_type>&        newAtlas = section->getNewAtlas();
      const typename Section_::atlas_type::chart_type& chart    = newAtlas->getChart();
      int                                              newSize  = 0;
      typename Section_::index_type                    defaultIdx(0, -1);

      for(typename Section_::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        defaultIdx.prefix = newAtlas->restrictPoint(*c_iter)[0].prefix;
        newAtlas->updatePoint(*c_iter, &defaultIdx);
        newSize += defaultIdx.prefix;
      }
      this->_factory->orderPatch(newAtlas, this->getSieve());
      // Copy over existing values
      typename Section_::value_type                   *newArray = new typename Section_::value_type[newSize];
      const typename Section_::atlas_type::chart_type& oldChart = newAtlas->getChart();

      for(typename Section_::atlas_type::chart_type::const_iterator c_iter = oldChart.begin(); c_iter != oldChart.end(); ++c_iter) {
        const typename Section_::value_type *array  = section->restrictPoint(*c_iter);
        const int&                           dim    = std::abs(section->getFiberDimension(*c_iter));
        const int&                           offset = newAtlas->restrictPoint(*c_iter)[0].index;

        for(int i = 0; i < dim; ++i) {
          newArray[offset+i] = array[i];
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
  class Mesh : public Bundle<ALE::Sieve<int,int,int> > {
  public:
    typedef Bundle<ALE::Sieve<int,int,int> > base_type;
    typedef base_type::sieve_type            sieve_type;
    typedef sieve_type::point_type           point_type;
    typedef base_type::label_sequence        label_sequence;
    typedef base_type::real_section_type     real_section_type;
    typedef base_type::numbering_type        numbering_type;
    typedef base_type::order_type            order_type;
    typedef base_type::send_overlap_type     send_overlap_type;
    typedef base_type::recv_overlap_type     recv_overlap_type;
  protected:
    int                   _dim;
    // Discretization
    Obj<Discretization>    _discretization;
    Obj<BoundaryCondition> _boundaryCondition;
  public:
    Mesh(MPI_Comm comm, int dim, int debug = 0) : Bundle<ALE::Sieve<int,int,int> >(comm, debug), _dim(dim) {
      ///this->_factory = NumberingFactory::singleton(debug);
      this->_discretization = new Discretization(comm, debug);
      this->_boundaryCondition = new BoundaryCondition(comm, debug);
    };
  public: // Accessors
    int getDimension() const {return this->_dim;};
    void setDimension(const int dim) {this->_dim = dim;};
    const Obj<Discretization>&    getDiscretization() {return this->_discretization;};
    void setDiscretization(const Obj<Discretization>& discretization) {this->_discretization = discretization;};
    const Obj<BoundaryCondition>& getBoundaryCondition() {return this->_boundaryCondition;};
    void setBoundaryCondition(const Obj<BoundaryCondition>& boundaryCondition) {this->_boundaryCondition = boundaryCondition;};
  public: // Mesh geometry
    void computeTriangleGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      const double    *coords = this->restrict(coordinates, e);
      const int        dim    = 2;
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
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      double v0[3], J[9], invJ[9], detJ, maxVolume = 0.0;

      for(label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        maxVolume = std::max(maxVolume, detJ);
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

        if ((xi >= 0.0) && (eta >= 0.0) && (xi + eta <= 1.0)) {
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

        if ((xi >= 0.0) && (eta >= 0.0) && (zeta >= 0.0) && (xi + eta + zeta <= 1.0)) {
          return *c_iter;
        }
      }
      throw ALE::Exception("Could not locate point");
    };
    point_type locatePoint(const real_section_type::value_type point[]) {
      if (this->_dim == 2) {
        return locatePoint_2D(point);
      } else if (this->_dim == 3) {
        return locatePoint_3D(point);
      } else {
        throw ALE::Exception("No point location for mesh dimension");
      }
    };
  public: // Discretization
    void setupField(const Obj<real_section_type>& s, const bool postponeGhosts = false) {
      const std::string& name = this->_boundaryCondition->getLabelName();

      for(int d = 0; d <= this->_dim; ++d) {
        s->setFiberDimension(this->depthStratum(d), this->_discretization->getNumDof(d));
      }
      if (!name.empty()) {
        const Obj<label_sequence>& boundary = this->getLabelStratum(name, 1);

        for(label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
          s->setFiberDimension(*e_iter, -this->_discretization->getNumDof(this->depth(*e_iter)));
        }
      }
      if (postponeGhosts) throw ALE::Exception("Not implemented yet");
      this->allocate(s);
      if (!name.empty()) {
        const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
        const Obj<label_sequence>&    boundary    = this->getLabelStratum(name, 1);

        for(label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
          const real_section_type::value_type *coords = coordinates->restrictPoint(*e_iter);
          const PetscScalar                    value  = this->_boundaryCondition->evaluate(coords);

          s->updatePointBC(*e_iter, &value);
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
      this->getSieve()->view("mesh sieve", comm);
      Obj<std::set<std::string> > sections = this->getRealSections();

      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getRealSection(*name)->view(*name);
      }
      sections = this->getIntSections();
      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getIntSection(*name)->view(*name);
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
  class MeshOld : public ALE::ParallelObject {
  public:
    // PCICE BC
    typedef struct {double rho,u,v,p;}   bc_value_type;
    typedef std::map<int, bc_value_type> bc_values_type;
  protected:
    int            _dim;
    // PCICE BC
    bc_values_type _bcValues;
  public:
    MeshOld(MPI_Comm comm, int dim, int debug = 0) : ALE::ParallelObject(comm, debug), _dim(dim) {};
  public: // Accessors
#if 0
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
#endif
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
    }
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
        /* Create topology and ordering */
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
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[6], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 2 (Bottom)
          Mesh::point_type face(2);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[4], face, order++);
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
          sieve->addArrow(vertices[1], face, order++);
          sieve->addArrow(vertices[4], face, order++);
          sieve->addArrow(vertices[7], face, order++);
          sieve->addArrow(vertices[2], face, order++);
          mesh->setValue(markers, face, 1);
        }
        {
          // Side 5 (Right)
          Mesh::point_type face(5);
          sieve->addArrow(vertices[5], face, order++);
          sieve->addArrow(vertices[0], face, order++);
          sieve->addArrow(vertices[3], face, order++);
          sieve->addArrow(vertices[6], face, order++);
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
      coords[0*2+0] = lower[0];
      coords[0*2+1] = lower[1];
      coords[0*2+2] = upper[2];
      coords[1*2+0] = upper[0];
      coords[1*2+1] = lower[1];
      coords[1*2+2] = upper[2];
      coords[2*2+0] = upper[0];
      coords[2*2+1] = upper[1];
      coords[2*2+2] = upper[2];
      coords[3*2+0] = lower[0];
      coords[3*2+1] = upper[1];
      coords[3*2+2] = upper[2];
      coords[4*2+0] = upper[0];
      coords[4*2+1] = lower[1];
      coords[4*2+2] = lower[2];
      coords[5*2+0] = lower[0];
      coords[5*2+1] = lower[1];
      coords[5*2+2] = lower[2];
      coords[6*2+0] = lower[0];
      coords[6*2+1] = upper[1];
      coords[6*2+2] = lower[2];
      coords[7*2+0] = upper[0];
      coords[7*2+1] = upper[1];
      coords[7*2+2] = lower[2];
#endif
      ALE::SieveBuilder<Mesh>::buildCoordinates(mesh, mesh->getDimension()+1, coords);
      return mesh;
    }
  };
} // namespace ALE

#endif
