#ifndef included_ALE_INumbering_hh
#define included_ALE_INumbering_hh

#ifndef  included_ALE_IField_hh
#include <sieve/IField.hh>
#endif

#ifndef  included_ALE_Completion_hh
#include <sieve/Completion.hh>
#endif

namespace ALE {
  template<typename Point_, typename Value_ = int, typename Alloc_ = malloc_allocator<Point_> >
  class INumbering : public IUniformSection<Point_, Value_, 1, Alloc_> {
  public:
    typedef IUniformSection<Point_, Value_, 1, Alloc_> base_type;
    typedef typename base_type::point_type             point_type;
    typedef typename base_type::value_type             value_type;
    typedef typename base_type::atlas_type             atlas_type;
  protected:
    int                       _localSize;
    int                      *_offsets;
    std::map<int, point_type> _invOrder;
  public:
    INumbering(MPI_Comm comm, const int debug = 0) : IUniformSection<Point_, Value_, 1, Alloc_>(comm, debug), _localSize(0) {
      this->_offsets    = new int[this->commSize()+1];
      this->_offsets[0] = 0;
    };
    ~INumbering() {
      delete [] this->_offsets;
    };
  public: // Sizes
    int        getLocalSize() const {return this->_localSize;};
    void       setLocalSize(const int size) {this->_localSize = size;};
    int        getGlobalSize() const {return this->_offsets[this->commSize()];};
    int        getGlobalOffset(const int p) const {return this->_offsets[p];};
    const int *getGlobalOffsets() const {return this->_offsets;};
    void       setGlobalOffsets(const int offsets[]) {
      for(int p = 0; p <= this->commSize(); ++p) {
        this->_offsets[p] = offsets[p];
      }
    };
  public: // Indices
    virtual int getIndex(const point_type& point) {
      const value_type& idx = this->restrictPoint(point)[0];
      if (idx >= 0) {
        return idx;
      }
      return -(idx+1);
    };
    virtual void setIndex(const point_type& point, const int index) {this->updatePoint(point, &index);};
    virtual bool isLocal(const point_type& point) {return this->restrictPoint(point)[0] >= 0;};
    virtual bool isRemote(const point_type& point) {return this->restrictPoint(point)[0] < 0;};
    point_type getPoint(const int& index) {return this->_invOrder[index];};
    void setPoint(const int& index, const point_type& point) {this->_invOrder[index] = point;};
  };
  template<typename Point_, typename Value_ = ALE::Point, typename Alloc_ = malloc_allocator<Point_> >
  class IGlobalOrder : public IUniformSection<Point_, Value_, 1, Alloc_> {
  public:
    typedef IUniformSection<Point_, Value_, 1, Alloc_> base_type;
    typedef typename base_type::point_type             point_type;
    typedef typename base_type::value_type             value_type;
    typedef typename base_type::atlas_type             atlas_type;
  protected:
    int  _localSize;
    int *_offsets;
  public:
    IGlobalOrder(MPI_Comm comm, const int debug = 0) : IUniformSection<Point_, Value_, 1, Alloc_>(comm, debug), _localSize(0) {
      this->_offsets    = new int[this->commSize()+1];
      this->_offsets[0] = 0;
    };
    ~IGlobalOrder() {
      delete [] this->_offsets;
    };
  public: // Sizes
    int        getLocalSize() const {return this->_localSize;};
    void       setLocalSize(const int size) {this->_localSize = size;};
    int        getGlobalSize() const {return this->_offsets[this->commSize()];};
    int        getGlobalOffset(const int p) const {return this->_offsets[p];};
    const int *getGlobalOffsets() const {return this->_offsets;};
    void       setGlobalOffsets(const int offsets[]) {
      for(int p = 0; p <= this->commSize(); ++p) {
        this->_offsets[p] = offsets[p];
      }
    };
  public: // Indices
    virtual int getIndex(const point_type& p) {
      const int idx = this->restrictPoint(p)[0].first;
      if (idx >= 0) {
        return idx;
      }
      return -(idx+1);
    };
    virtual void setIndex(const point_type& p, const int index) {
      const value_type idx(index, this->restrictPoint(p)[0].second);
      this->updatePoint(p, &idx);
    };
    virtual bool isLocal(const point_type& p) {return this->restrictPoint(p)[0].first >= 0;};
    virtual bool isRemote(const point_type& p) {return this->restrictPoint(p)[0].first < 0;};
  };
  template<typename Bundle_, typename Value_ = int, typename Alloc_ = typename Bundle_::alloc_type>
  class INumberingFactory : public ALE::ParallelObject {
  public:
    typedef Bundle_                                                                              bundle_type;
    typedef typename bundle_type::point_type                                                     point_type;
    typedef Value_                                                                               value_type;
    typedef Alloc_                                                                               alloc_type;
    typedef INumbering<point_type, value_type, alloc_type>                                       numbering_type;
    typedef std::map<bundle_type*, std::map<std::string, std::map<int, Obj<numbering_type> > > > numberings_type;
    typedef typename ALE::Pair<value_type, value_type>                                           oValue_type;
    typedef typename alloc_type::template rebind<oValue_type>::other                             oAlloc_type;
    typedef IGlobalOrder<point_type, oValue_type, oAlloc_type>                                   order_type;
    typedef std::map<bundle_type*, std::map<std::string, Obj<order_type> > >                     orders_type;
    typedef typename bundle_type::send_overlap_type                                              send_overlap_type;
    typedef typename bundle_type::recv_overlap_type                                              recv_overlap_type;
  protected:
    numberings_type   _localNumberings;
    numberings_type   _numberings;
    orders_type       _orders;
    const value_type  _unknownNumber;
    const oValue_type _unknownOrder;
  protected:
    INumberingFactory(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug), _unknownNumber(-1), _unknownOrder(-1, 0) {};
  public:
    ~INumberingFactory() {};
  public:
    static const INumberingFactory& singleton(MPI_Comm comm, const int debug, bool cleanup = false) {
      static INumberingFactory *_singleton = NULL;

      if (cleanup) {
        if (debug) {std::cout << "Destroying NumberingFactory" << std::endl;}
        if (_singleton) {delete _singleton;}
        _singleton = NULL;
      } else if (_singleton == NULL) {
        if (debug) {std::cout << "Creating new NumberingFactory" << std::endl;}
        _singleton = new INumberingFactory(comm, debug);
      }
      return *_singleton;
    };
    void clear() {
      this->_localNumberings.clear();
      this->_numberings.clear();
      this->_orders.clear();
    };
  protected: // Local numberings
    // Number all local points
    //   points in the overlap are only numbered by the owner with the lowest rank
    template<typename Sequence_>
    void constructLocalNumbering(const Obj<numbering_type>& numbering, const Obj<send_overlap_type>& sendOverlap, const Obj<Sequence_>& points) {
      const int debug = sendOverlap->debug();
      int localSize = 0;

      if (debug) {std::cout << "["<<numbering->commRank()<<"] Constructing local numbering" << std::endl;}
      numbering->setChart(typename order_type::chart_type(*std::min_element(points->begin(), points->end()), *std::max_element(points->begin(), points->end())+1));
      numbering->setFiberDimension(points, 1);
      numbering->allocatePoint();
      for(typename Sequence_::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
        value_type val;

        if (debug) {std::cout << "["<<numbering->commRank()<<"]   Checking point " << *l_iter << std::endl;}
        if (sendOverlap->capContains(*l_iter)) {
          const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = sendOverlap->support(*l_iter);
          int minRank = sendOverlap->commSize();

          for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
            if (*p_iter < minRank) minRank = *p_iter;
          }
          if (minRank < sendOverlap->commRank()) {
            if (debug) {std::cout << "["<<numbering->commRank()<<"]     remote point, on proc " << minRank << std::endl;}
            val = this->_unknownNumber;
          } else {
            if (debug) {std::cout << "["<<numbering->commRank()<<"]     local point" << std::endl;}
            val = localSize++;
          }
        } else {
          if (debug) {std::cout << "["<<numbering->commRank()<<"]     local point" << std::endl;}
          val = localSize++;
        }
        if (debug) {std::cout << "["<<numbering->commRank()<<"]     has number " << val << std::endl;}
        numbering->updatePoint(*l_iter, &val);
      }
      if (debug) {std::cout << "["<<numbering->commRank()<<"]   local points" << std::endl;}
      numbering->setLocalSize(localSize);
    }
    // Order all local points
    //   points in the overlap are only ordered by the owner with the lowest rank
    template<typename Sequence_, typename Section_>
    void constructLocalOrder(const Obj<order_type>& order, const Obj<send_overlap_type>& sendOverlap, const Sequence_& points, const Obj<Section_>& section) {
      int localSize = 0;

      ///std::cout << "["<<order->commRank()<<"] Constructing local ordering" << std::endl;
      order->setChart(typename order_type::chart_type(*std::min_element(points.begin(), points.end()), *std::max_element(points.begin(), points.end())+1));
      for(typename Sequence_::const_iterator l_iter = points.begin(); l_iter != points.end(); ++l_iter) {
        order->setFiberDimension(*l_iter, 1);
      }
      order->allocatePoint();
      for(typename Sequence_::const_iterator l_iter = points.begin(); l_iter != points.end(); ++l_iter) {
        oValue_type val;

        ///std::cout << "["<<order->commRank()<<"]   Checking point " << *l_iter << std::endl;
        if (sendOverlap->capContains(*l_iter)) {
          const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = sendOverlap->support(*l_iter);
          int minRank = sendOverlap->commSize();

          for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
            if (*p_iter < minRank) minRank = *p_iter;
          }
          if (minRank < sendOverlap->commRank()) {
            ///std::cout << "["<<order->commRank()<<"]     remote point, on proc " << minRank << std::endl;
            val = this->_unknownOrder;
          } else {
            ///std::cout << "["<<order->commRank()<<"]     local point" << std::endl;
            val.first  = localSize;
            val.second = section->getConstrainedFiberDimension(*l_iter);
          }
        } else {
          ///std::cout << "["<<order->commRank()<<"]     local point" << std::endl;
          val.first  = localSize;
          val.second = section->getConstrainedFiberDimension(*l_iter);
        }
        ///std::cout << "["<<order->commRank()<<"]     has offset " << val.prefix << " and size " << val.index << std::endl;
        localSize += val.second;
        order->updatePoint(*l_iter, &val);
      }
      ///std::cout << "["<<order->commRank()<<"]   local size" << std::endl;
      order->setLocalSize(localSize);
    }
  protected: // Global offsets
    // Calculate process offsets
    template<typename Numbering>
    void calculateOffsets(const Obj<Numbering>& numbering) {
      int  localSize = numbering->getLocalSize();
      int *offsets   = new int[numbering->commSize()+1];

      offsets[0] = 0;
      MPI_Allgather(&localSize, 1, MPI_INT, &(offsets[1]), 1, MPI_INT, numbering->comm());
      for(int p = 2; p <= numbering->commSize(); p++) {
        offsets[p] += offsets[p-1];
      }
      numbering->setGlobalOffsets(offsets);
      delete [] offsets;
    }
    // Update local offsets based upon process offsets
    template<typename Numbering, typename Sequence>
    void updateOrder(const Obj<Numbering>& numbering, Sequence& points) {
      const typename Numbering::value_type val = numbering->getGlobalOffset(numbering->commRank());

      for(typename Sequence::const_iterator l_iter = points.begin(); l_iter != points.end(); ++l_iter) {
        if (numbering->isLocal(*l_iter)) {
          numbering->updateAddPoint(*l_iter, &val);
        }
      }
    }
    template<typename OverlapSection, typename RecvOverlap>
    static void fuseNumbering(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<numbering_type>& numbering) {
      typedef typename OverlapSection::point_type overlap_point_type;
      const Obj<typename RecvOverlap::traits::baseSequence>      rPoints = recvOverlap->base();
      const typename RecvOverlap::traits::baseSequence::iterator rEnd    = rPoints->end();
      const int                                                  debug   = numbering->debug();
      const bool                                                 allowDuplicates = false;

      numbering->reallocatePoint(rPoints->begin(), rEnd, Identity<typename recv_overlap_type::target_type>());
      for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
        const Obj<typename recv_overlap_type::traits::coneSequence>& ranks      = recvOverlap->cone(*p_iter);
        const typename recv_overlap_type::target_type&               localPoint = *p_iter;

        for(typename recv_overlap_type::traits::coneSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          const typename recv_overlap_type::target_type&       remotePoint = r_iter.color();
          const int                                            rank        = *r_iter;
          const int                                            size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
          const typename OverlapSection::value_type           *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));

          if (size == 0)             continue;
          if (debug) {std::cout << "["<<numbering->commRank()<<"]     local point " << localPoint << " remote point " << remotePoint << " number " << values[0] << std::endl;}
          if (values[0] >= 0) {
            if (debug) {std::cout << "["<<numbering->commRank()<<"] local point " << localPoint << " dim " << numbering->getAtlas()->getFiberDimension(localPoint) << std::endl;}
            if (numbering->isLocal(localPoint) && !allowDuplicates) {
              ostringstream msg;
              msg << "["<<numbering->commRank()<<"]Multiple indices for local point " << localPoint << " remote point " << remotePoint << " from " << rank << " with index " << values[0];
              throw ALE::Exception(msg.str().c_str());
            }
            if (numbering->getAtlas()->getFiberDimension(localPoint) == 0) {
              ostringstream msg;
              msg << "["<<numbering->commRank()<<"]Unexpected local point " << localPoint << " remote point " << remotePoint << " from " << rank << " with index " << values[0];
              throw ALE::Exception(msg.str().c_str());
            }
            const typename numbering_type::value_type val = -(values[0]+1);
            numbering->updatePoint(localPoint, &val);
          }
        }
      }
    }
    template<typename OverlapSection, typename RecvOverlap>
    static void fuse(const Obj<OverlapSection>& overlapSection, const Obj<RecvOverlap>& recvOverlap, const Obj<order_type>& order) {
      typedef typename OverlapSection::point_type overlap_point_type;
      const Obj<typename RecvOverlap::traits::baseSequence>      rPoints = recvOverlap->base();
      const typename RecvOverlap::traits::baseSequence::iterator rEnd    = rPoints->end();
      const bool                                                 allowDuplicates = false;

      order->reallocatePoint(rPoints->begin(), rEnd, Identity<typename recv_overlap_type::target_type>());
      for(typename RecvOverlap::traits::baseSequence::iterator p_iter = rPoints->begin(); p_iter != rEnd; ++p_iter) {
        const Obj<typename recv_overlap_type::traits::coneSequence>& ranks      = recvOverlap->cone(*p_iter);
        const typename recv_overlap_type::target_type&               localPoint = *p_iter;

        for(typename recv_overlap_type::traits::coneSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          const typename recv_overlap_type::target_type&       remotePoint = r_iter.color();
          const int                                            rank        = *r_iter;
          const int                                            size        = overlapSection->getFiberDimension(overlap_point_type(rank, remotePoint));
          const typename OverlapSection::value_type           *values      = overlapSection->restrictPoint(overlap_point_type(rank, remotePoint));

          if (size == 0)             continue;
          if (values[0].second == 0) continue;
          if (values[0].first >= 0) {
            if (order->isLocal(localPoint)) {
              if (!allowDuplicates) {
                ostringstream msg;
                msg << "["<<order->commRank()<<"]Multiple indices for local point " << localPoint << " remote point " << remotePoint << " from " << rank << " with index " << values[0];
                throw ALE::Exception(msg.str().c_str());
              }
              continue;
            }
            const typename order_type::value_type val(-(values[0].first+1), values[0].second);
            order->updatePoint(localPoint, &val);
          } else {
            if (order->isLocal(localPoint)) continue;
            order->updatePoint(localPoint, values);
          }
        }
      }
    }
  public: // Completion
    void completeNumbering(const Obj<numbering_type>& numbering, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, bool allowDuplicates = false) {
#if 0
      ALE::Completion::completeSection(sendOverlap, recvOverlap, numbering, numbering);
#else
      typedef ALE::UniformSection<ALE::Pair<int, typename send_overlap_type::source_type>, typename numbering_type::value_type> OverlapSection;
      Obj<OverlapSection> overlapSection = new OverlapSection(numbering->comm(), numbering->debug());

      if (numbering->debug()) {numbering->view("Local Numbering");}
      ALE::Pullback::SimpleCopy::copy(sendOverlap, recvOverlap, numbering, overlapSection);
      if (overlapSection->debug()) {overlapSection->view("Overlap Section");}
      fuseNumbering(overlapSection, recvOverlap, numbering);
      if (numbering->debug()) {numbering->view("Global Numbering");}
#endif
    }
    void completeOrder(const Obj<order_type>& order, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, bool allowDuplicates = false) {
#if 0
      ALE::Completion::completeSection(sendOverlap, recvOverlap, order, order);
#else
      typedef ALE::UniformSection<ALE::Pair<int, typename send_overlap_type::source_type>, typename order_type::value_type> OverlapSection;
      Obj<OverlapSection> overlapSection = new OverlapSection(order->comm(), order->debug());

      if (order->debug()) {order->view("Local Order");}
      ALE::Pullback::SimpleCopy::copy(sendOverlap, recvOverlap, order, overlapSection);
      if (overlapSection->debug()) {overlapSection->view("Overlap Section");}
      fuse(overlapSection, recvOverlap, order);
      if (order->debug()) {order->view("Global Order");}
#endif
    }
  public: // Construct a full global numberings
    template<typename Sequence>
    void constructNumbering(const Obj<numbering_type>& numbering, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<Sequence>& points) {
      this->constructLocalNumbering(numbering, sendOverlap, points);
      this->calculateOffsets(numbering);
      this->updateOrder(numbering, *points.ptr());
      this->completeNumbering(numbering, sendOverlap, recvOverlap);
    }
    template<typename Sequence, typename Section>
    void constructOrder(const Obj<order_type>& order, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Sequence& points, const Obj<Section>& section) {
      this->constructLocalOrder(order, sendOverlap, points, section);
      this->calculateOffsets(order);
      this->updateOrder(order, points);
      this->completeOrder(order, sendOverlap, recvOverlap);
    }
    template<typename Sequence, typename Section>
    void constructOrder(const Obj<order_type>& order, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<Sequence>& points, const Obj<Section>& section) {
      this->constructLocalOrder(order, sendOverlap, *points.ptr(), section);
      this->calculateOffsets(order);
      this->updateOrder(order, *points.ptr());
      this->completeOrder(order, sendOverlap, recvOverlap);
    }
  public: // Real interface
    template<typename ABundle_>
    const Obj<numbering_type>& getLocalNumbering(const Obj<ABundle_>& bundle, const int depth) {
      if ((this->_localNumberings.find(bundle.ptr()) == this->_localNumberings.end()) ||
          (this->_localNumberings[bundle.ptr()].find("depth") == this->_localNumberings[bundle.ptr()].end()) ||
          (this->_localNumberings[bundle.ptr()]["depth"].find(depth) == this->_localNumberings[bundle.ptr()]["depth"].end())) {
        Obj<numbering_type>    numbering   = new numbering_type(bundle->comm(), bundle->debug());
        Obj<send_overlap_type> sendOverlap = new send_overlap_type(bundle->comm(), bundle->debug());

        this->constructLocalNumbering(numbering, sendOverlap, bundle->depthStratum(depth));
        if (this->_debug) {std::cout << "Creating new local numbering: ptr " << bundle.ptr() << " depth " << depth << std::endl;}
        this->_localNumberings[bundle.ptr()]["depth"][depth] = numbering;
      } else {
        if (this->_debug) {std::cout << "Using old local numbering: ptr " << bundle.ptr() << " depth " << depth << std::endl;}
      }
      return this->_localNumberings[bundle.ptr()]["depth"][depth];
    }
    template<typename ABundle_>
    const Obj<numbering_type>& getNumbering(const Obj<ABundle_>& bundle, const int depth) {
      if ((this->_numberings.find(bundle.ptr()) == this->_numberings.end()) ||
          (this->_numberings[bundle.ptr()].find("depth") == this->_numberings[bundle.ptr()].end()) ||
          (this->_numberings[bundle.ptr()]["depth"].find(depth) == this->_numberings[bundle.ptr()]["depth"].end())) {
        bundle->constructOverlap();
        Obj<numbering_type>    numbering   = new numbering_type(bundle->comm(), bundle->debug());
        Obj<send_overlap_type> sendOverlap = bundle->getSendOverlap();
        Obj<recv_overlap_type> recvOverlap = bundle->getRecvOverlap();

        this->constructNumbering(numbering, sendOverlap, recvOverlap, bundle->depthStratum(depth));
        if (this->_debug) {std::cout << "Creating new numbering: depth " << depth << std::endl;}
        this->_numberings[bundle.ptr()]["depth"][depth] = numbering;
      } else {
        if (this->_debug) {std::cout << "["<<bundle->commRank()<<"]Using old numbering: depth " << depth << std::endl;}
      }
      return this->_numberings[bundle.ptr()]["depth"][depth];
    }
    template<typename ABundle_>
    const Obj<numbering_type>& getNumbering(const Obj<ABundle_>& bundle, const std::string& labelname, const int value) {
      if ((this->_numberings.find(bundle.ptr()) == this->_numberings.end()) ||
          (this->_numberings[bundle.ptr()].find(labelname) == this->_numberings[bundle.ptr()].end()) ||
          (this->_numberings[bundle.ptr()][labelname].find(value) == this->_numberings[bundle.ptr()][labelname].end())) {
        bundle->constructOverlap();
        Obj<numbering_type>    numbering   = new numbering_type(bundle->comm(), bundle->debug());
        Obj<send_overlap_type> sendOverlap = bundle->getSendOverlap();
        Obj<recv_overlap_type> recvOverlap = bundle->getRecvOverlap();

        numbering->setDefault(&_unknownNumber);
        if (this->_debug) {std::cout << "["<<bundle->commRank()<<"]Creating new numbering: " << labelname << " value " << value << std::endl;}
        this->constructNumbering(numbering, sendOverlap, recvOverlap, bundle->getLabelStratum(labelname, value));
        this->_numberings[bundle.ptr()][labelname][value] = numbering;
      } else {
        if (this->_debug) {std::cout << "["<<bundle->commRank()<<"]Using old numbering: " << labelname << " value " << value << std::endl;}
      }
      return this->_numberings[bundle.ptr()][labelname][value];
    }
    template<typename ABundle_, typename Section_>
    const Obj<order_type>& getGlobalOrder(const Obj<ABundle_>& bundle, const std::string& name, const Obj<Section_>& section) {
      if ((this->_orders.find(bundle.ptr()) == this->_orders.end()) ||
          (this->_orders[bundle.ptr()].find(name) == this->_orders[bundle.ptr()].end())) {
        bundle->constructOverlap();
        Obj<order_type>        order       = new order_type(bundle->comm(), bundle->debug());
        Obj<send_overlap_type> sendOverlap = bundle->getSendOverlap();
        Obj<recv_overlap_type> recvOverlap = bundle->getRecvOverlap();

        order->setDefault(&_unknownOrder);
        if (this->_debug) {std::cout << "["<<bundle->commRank()<<"]Creating new global order: " << name << std::endl;}
        this->constructOrder(order, sendOverlap, recvOverlap, section->getChart(), section);
        this->_orders[bundle.ptr()][name] = order;
      } else {
        if (this->_debug) {std::cout << "["<<bundle->commRank()<<"]Using old global order: " << name << std::endl;}
      }
      return this->_orders[bundle.ptr()][name];
    }
#if 0
    template<typename ABundle_, typename Section_>
    const Obj<order_type>& getGlobalOrderWithBC(const Obj<ABundle_>& bundle, const std::string& name, const Obj<Section_>& section) {
      if ((this->_orders.find(bundle.ptr()) == this->_orders.end()) ||
          (this->_orders[bundle.ptr()].find(name) == this->_orders[bundle.ptr()].end())) {
        bundle->constructOverlap();
        Obj<order_type>        order       = new order_type(bundle->comm(), bundle->debug());
        Obj<send_overlap_type> sendOverlap = bundle->getSendOverlap();
        Obj<recv_overlap_type> recvOverlap = bundle->getRecvOverlap();

        order->setDefault(&_unknownOrder);
        if (this->_debug) {std::cout << "["<<bundle->commRank()<<"]Creating new global order: " << name << std::endl;}
        this->constructOrderWithBC(order, sendOverlap, recvOverlap, section->getChart(), section);
        this->_orders[bundle.ptr()][name] = order;
      } else {
        if (this->_debug) {std::cout << "["<<bundle->commRank()<<"]Using old global order: " << name << std::endl;}
      }
      return this->_orders[bundle.ptr()][name];
    }
#endif
    template<typename ABundle_>
    void setGlobalOrder(const Obj<ABundle_>& bundle, const std::string& name, const Obj<order_type>& order) {
      this->_orders[bundle.ptr()][name] = order;
    }
  };
}
#endif
