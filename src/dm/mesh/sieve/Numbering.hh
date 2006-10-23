#ifndef included_ALE_Numbering_hh
#define included_ALE_Numbering_hh

#ifndef  included_ALE_Completion_hh
#include <Completion.hh>
#endif

extern PetscErrorCode PetscCommSynchronizeTags(MPI_Comm);

namespace ALE {
  namespace New {
    template<typename Topology_, typename Value_ = int>
    class NewNumbering : public UniformSection<Topology_, Value_> {
    public:
      typedef UniformSection<Topology_, Value_>  base_type;
      typedef Topology_                          topology_type;
      typedef Value_                             value_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::point_type point_type;
      typedef typename base_type::atlas_type     atlas_type;
    protected:
      int                       _localSize;
      int                      *_offsets;
      std::map<int, point_type> _invOrder;
    public:
      NewNumbering(const Obj<topology_type>& topology) : UniformSection<Topology_, Value_>(topology), _localSize(0) {
        this->_offsets    = new int[this->commSize()+1];
        this->_offsets[0] = 0;
      };
      ~NewNumbering() {
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
        return getIndex(0, point);
      };
      virtual int getIndex(const patch_type& patch, const point_type& point) {
        const value_type& idx = this->restrictPoint(patch, point)[0];
        if (idx >= 0) {
          return idx;
        }
        return -(idx+1);
      };
      virtual void setIndex(const point_type& point, const int index) {this->updatePoint(0, point, &index);};
      virtual bool isLocal(const point_type& point) {return this->restrictPoint(0, point)[0] >= 0;};
      virtual bool isRemote(const point_type& point) {return this->restrictPoint(0, point)[0] < 0;};
      point_type getPoint(const int& index) {return this->_invOrder[index];};
      void setPoint(const int& index, const point_type& point) {this->_invOrder[index] = point;};
    };

    template<typename Topology_>
    class NewGlobalOrder : public UniformSection<Topology_, ALE::Point> {
    public:
      typedef UniformSection<Topology_, ALE::Point> base_type;
      typedef Topology_                          topology_type;
      typedef ALE::Point                         value_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::point_type point_type;
      typedef typename base_type::atlas_type     atlas_type;
    protected:
      int                       _localSize;
      int                      *_offsets;
      //std::map<int, point_type> _invOrder;
    public:
      NewGlobalOrder(const Obj<topology_type>& topology) : UniformSection<Topology_, ALE::Point>(topology), _localSize(0) {
        this->_offsets    = new int[this->commSize()+1];
        this->_offsets[0] = 0;
      };
      ~NewGlobalOrder() {
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
        return getIndex(0, point);
      };
      virtual int getIndex(const patch_type& patch, const point_type& point) {
        if (this->restrictPoint(0, point)[0].prefix >= 0) {
          return this->restrictPoint(0, point)[0].prefix;
        }
        return -(this->restrictPoint(0, point)[0].prefix+1);
      };
      virtual void setIndex(const point_type& point, const int index) {
        const value_type idx(index, this->restrictPoint(0, point)[0].index);
        this->updatePoint(0, point, &idx);
      };
      virtual bool isLocal(const point_type& point) {return this->restrictPoint(0, point)[0].prefix >= 0;};
      virtual bool isRemote(const point_type& point) {return this->restrictPoint(0, point)[0].prefix < 0;};
    };

    // We have a dichotomy between \emph{types}, describing the structure of objects,
    //   and \emph{concepts}, describing the role these objects play in the algorithm.
    //   Below we identify concepts with potential implementing types.
    //
    //   Concept           Type
    //   -------           ----
    //   Overlap           Sifter
    //   Atlas             ConstantSection, UniformSection
    //   Numbering         UniformSection
    //   GlobalOrder       UniformSection
    //
    // We will use factory types to create objects which satisfy a given concept.
    template<typename Topology_, typename Value_ = int>
    class NumberingFactory {
    public:
      typedef Topology_                                                           topology_type;
      typedef Value_                                                              value_type;
      typedef typename topology_type::point_type                                  point_type;
      typedef typename ALE::New::DiscreteSieve<point_type>                        dsieve_type;
      typedef typename ALE::New::Topology<int, dsieve_type>                       dtopology_type;
      typedef typename topology_type::patch_type                                  patch_type;
      typedef typename topology_type::send_overlap_type                           send_overlap_type;
      typedef typename topology_type::recv_overlap_type                           recv_overlap_type;
      typedef typename ALE::New::NewNumbering<topology_type, value_type>          numbering_type;
      typedef std::map<int, Obj<numbering_type> >                                 depthMap_type;
      typedef std::map<patch_type, depthMap_type>                                 patchMap_type;
      typedef std::map<topology_type*, patchMap_type>                             numberings_type;
      typedef typename ALE::New::NewGlobalOrder<topology_type>                    order_type;
      typedef std::map<std::string, Obj<order_type> >                             sectionMap_type;
      typedef std::map<patch_type, sectionMap_type>                               oPatchMap_type;
      typedef std::map<topology_type*, oPatchMap_type>                            orders_type;
      typedef typename order_type::value_type                                     oValue_type;
    protected:
      int             _debug;
      numberings_type _localNumberings;
      numberings_type _numberings;
      orders_type     _orders;
      value_type      _unknownNumber;
      ALE::Point      _unknownOrder;
    protected:
      NumberingFactory(const int debug) : _debug(debug), _unknownNumber(-1), _unknownOrder(-1, 0) {};
    public:
      ~NumberingFactory() {};
    public:
      static const Obj<NumberingFactory>& singleton(const int debug, bool cleanup = false) {
        static Obj<NumberingFactory> *_singleton = NULL;

        if (cleanup) {
          if (debug) {std::cout << "Destroying NumberingFactory" << std::endl;}
          if (_singleton) {delete _singleton;}
          _singleton = NULL;
        } else if (_singleton == NULL) {
          if (debug) {std::cout << "Creating new NumberingFactory" << std::endl;}
          _singleton  = new Obj<NumberingFactory>();
          *_singleton = new NumberingFactory(debug);
        }
        return *_singleton;
      };
    public:
      const int debug() {return this->_debug;};
    public:
      // Number all local points
      //   points in the overlap are only numbered by the owner with the lowest rank
      template<typename Sequence>
      void constructLocalNumbering(const Obj<numbering_type>& numbering, const Obj<send_overlap_type>& sendOverlap, const patch_type& patch, const Obj<Sequence>& points) {
        int localSize = 0;

        numbering->setFiberDimension(patch, points, 1);
        for(typename Sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          value_type val;

          if (sendOverlap->capContains(*l_iter)) {
            const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = sendOverlap->support(*l_iter);
            int minRank = sendOverlap->commSize();

            for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
              if (*p_iter < minRank) minRank = *p_iter;
            }
            if (minRank < sendOverlap->commRank()) {
              val = this->_unknownNumber;
            } else {
              val = localSize++;
            }
          } else {
            val = localSize++;
          }
          numbering->updatePoint(patch, *l_iter, &val);
        }
        numbering->setLocalSize(localSize);
      };
      // Order all local points
      //   points in the overlap are only ordered by the owner with the lowest rank
      template<typename Sequence, typename Atlas>
      void constructLocalOrder(const Obj<order_type>& order, const Obj<send_overlap_type>& sendOverlap, const patch_type& patch, const Obj<Sequence>& points, const Obj<Atlas>& atlas) {
        int localSize = 0;

        order->setFiberDimension(patch, points, 1);
        for(typename Sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          oValue_type val;

          if (sendOverlap->capContains(*l_iter)) {
            const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = sendOverlap->support(*l_iter);
            int minRank = sendOverlap->commSize();

            for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
              if (*p_iter < minRank) minRank = *p_iter;
            }
            if (minRank < sendOverlap->commRank()) {
              val = this->_unknownOrder;
            } else {
              val.prefix = localSize;
              val.index  = atlas->restrict(patch, *l_iter)[0].prefix;
            }
          } else {
            val.prefix = localSize;
            val.index  = atlas->restrict(patch, *l_iter)[0].prefix;
          }
          localSize += val.index;
          order->updatePoint(patch, *l_iter, &val);
        }
        order->setLocalSize(localSize);
      };
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
      };
      // Update local offsets based upon process offsets
      template<typename Numbering, typename Sequence>
      void updateOrder(const Obj<Numbering>& numbering, const patch_type& patch, const Obj<Sequence>& points) {
        const typename Numbering::value_type val = numbering->getGlobalOffset(numbering->commRank());

        for(typename Sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          if (numbering->isLocal(*l_iter)) {
            numbering->updateAddPoint(patch, *l_iter, &val);
          }
        }
      };
      // Communicate numbers in the overlap
      void complete(const Obj<numbering_type>& numbering, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const patch_type& patch, bool allowDuplicates = false) {
        typedef dtopology_type topo_type;
        typedef typename ALE::New::OverlapValues<send_overlap_type, topo_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topo_type, value_type> recv_section_type;
        typedef typename ALE::New::ConstantSection<topology_type, int> constant_sizer;
        const Obj<send_section_type> sendSection = new send_section_type(numbering->comm(), this->debug());
        const Obj<recv_section_type> recvSection = new recv_section_type(numbering->comm(), sendSection->getTag(), this->debug());
        //const Obj<constant_sizer>    sizer       = new constant_sizer(numbering->comm(), 1, this->debug());

        Completion<topology_type, int>::completeSection(sendOverlap, recvOverlap, numbering->getAtlas(), numbering, sendSection, recvSection);
        const typename recv_section_type::topology_type::sheaf_type& patches = recvSection->getTopology()->getPatches();

        for(typename recv_section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename recv_section_type::patch_type& rPatch = p_iter->first;
          const typename recv_section_type::chart_type& points = recvSection->getPatch(rPatch);

          for(typename recv_section_type::chart_type::iterator r_iter = points.begin(); r_iter != points.end(); ++r_iter) {
            const typename recv_section_type::point_type& point  = *r_iter;
            const typename recv_section_type::value_type *values = recvSection->restrict(rPatch, point);

            if (recvSection->getFiberDimension(rPatch, point) == 0) continue;
            if (values[0] >= 0) {
              if (numbering->isLocal(point) && !allowDuplicates) {
                ostringstream msg;
                msg << "["<<numbering->commRank()<<"]Multiple indices for point " << point << " from " << rPatch << " with index " << values[0];
                throw ALE::Exception(msg.str().c_str());
              }
              if (numbering->getAtlas()->getFiberDimension(0, point) == 0) {
                ostringstream msg;
                msg << "["<<numbering->commRank()<<"]Unexpected point " << point << " from " << rPatch << " with index " << values[0];
                throw ALE::Exception(msg.str().c_str());
              }
              int val = -(values[0]+1);
              numbering->updatePoint(patch, point, &val);
            }
          }
        }
#if 0
        Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = recvOverlap->base();

        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          numbering->setFiberDimension(0, *r_iter, 1);
        }
        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>& recvPatches = recvOverlap->cone(*r_iter);
    
          for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != recvPatches->end(); ++p_iter) {
            const typename recv_section_type::value_type *values = recvSection->restrict(*p_iter, *r_iter);

            if (values[0] >= 0) {
              if (numbering->isLocal(*r_iter) && !allowDuplicates) {
                ostringstream msg;
                msg << "["<<numbering->commRank()<<"]Multiple indices for point " << *r_iter << " from " << *p_iter << " with index " << values[0];
                throw ALE::Exception(msg.str().c_str());
              }
              if (numbering->getAtlas()->getFiberDimension(0, *r_iter) == 0) {
                ostringstream msg;
                msg << "["<<numbering->commRank()<<"]Unexpected point " << *r_iter << " from " << *p_iter << " with index " << values[0];
                throw ALE::Exception(msg.str().c_str());
              }
              int val = -(values[0]+1);
              numbering->updatePoint(patch, *r_iter, &val);
            }
          }
        }
#endif
        PetscErrorCode ierr;
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
      };
      // Communicate (size,offset)s in the overlap
      void completeOrder(const Obj<order_type>& order, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const patch_type& patch, bool allowDuplicates = false) {
        typedef dtopology_type topo_type;
        typedef typename ALE::New::OverlapValues<send_overlap_type, topo_type, oValue_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topo_type, oValue_type> recv_section_type;
        typedef typename ALE::New::ConstantSection<topology_type, int> constant_sizer;
        const Obj<send_section_type> sendSection = new send_section_type(order->comm(), this->debug());
        const Obj<recv_section_type> recvSection = new recv_section_type(order->comm(), sendSection->getTag(), this->debug());
        const Obj<constant_sizer>    sizer       = new constant_sizer(order->comm(), 1, this->debug());

        Completion<topology_type, int>::completeSection(sendOverlap, recvOverlap, sizer, order, sendSection, recvSection);
        Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = recvOverlap->base();

        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          if (!order->hasPoint(patch, *r_iter)) {
            order->setFiberDimension(patch, *r_iter, 1);
            order->update(patch, *r_iter, &this->_unknownOrder);
          }
        }
        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>& recvPatches = recvOverlap->cone(*r_iter);
    
          for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != recvPatches->end(); ++p_iter) {
            const typename recv_section_type::value_type *values = recvSection->restrict(*p_iter, *r_iter);

            if (values[0].index == 0) continue;
            if (values[0].prefix >= 0) {
              if (order->isLocal(*r_iter)) {
                if (!allowDuplicates) {
                  ostringstream msg;
                  msg << "["<<order->commRank()<<"]Multiple indices for point " << *r_iter << " from " << *p_iter << " with index " << values[0];
                  throw ALE::Exception(msg.str().c_str());
                }
                continue;
              }
              const oValue_type val(-(values[0].prefix+1), values[0].index);
              order->updatePoint(patch, *r_iter, &val);
            } else {
              if (order->isLocal(*r_iter)) continue;
              order->updatePoint(patch, *r_iter, values);
            }
          }
        }
      };
      // Construct a full global numbering
      template<typename Sequence>
      void construct(const Obj<numbering_type>& numbering, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const patch_type& patch, const Obj<Sequence>& points) {
        this->constructLocalNumbering(numbering, sendOverlap, patch, points);
        this->calculateOffsets(numbering);
        this->updateOrder(numbering, patch, points);
        this->complete(numbering, sendOverlap, recvOverlap, patch);
      };
      // Construct a full global order
      template<typename Sequence, typename Atlas>
      void constructOrder(const Obj<order_type>& order, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const patch_type& patch, const Obj<Sequence>& points, const Obj<Atlas>& atlas) {
        this->constructLocalOrder(order, sendOverlap, patch, points, atlas);
        this->calculateOffsets(order);
        this->updateOrder(order, patch, points);
        this->completeOrder(order, sendOverlap, recvOverlap, patch);
      };
      // Construct the inverse map from numbers to points
      //   If we really need this, then we should consider using a label
      void constructInverseOrder(const Obj<numbering_type>& numbering) {
        const typename numbering_type::chart_type& patch = numbering->getAtlas()->getPatch(0);

        for(typename numbering_type::chart_type::iterator p_iter = patch.begin(); p_iter != patch.end(); ++p_iter) {
          numbering->setPoint(numbering->getIndex(*p_iter), *p_iter);
        }
      };
    public:
      const Obj<numbering_type>& getLocalNumbering(const Obj<topology_type>& topology, const patch_type& patch, const int depth) {
        if ((this->_localNumberings.find(topology.ptr()) == this->_localNumberings.end()) ||
            (this->_localNumberings[topology.ptr()].find(patch) == this->_localNumberings[topology.ptr()].end()) ||
            (this->_localNumberings[topology.ptr()][patch].find(depth) == this->_localNumberings[topology.ptr()][patch].end())) {
          Obj<numbering_type>    numbering   = new numbering_type(topology);
          // These go in the Topology soon
          Obj<send_overlap_type> sendOverlap = new send_overlap_type(topology->comm(), topology->debug());
          //Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(this->comm(), this->debug());

          this->constructLocalNumbering(numbering, sendOverlap, patch, topology->getLabelStratum(patch, "depth", depth));
          if (this->_debug) {std::cout << "Creating new local numbering: patch " << patch << " depth " << depth << std::endl;}
          this->_localNumberings[topology.ptr()][patch][depth] = numbering;
        }
        return this->_localNumberings[topology.ptr()][patch][depth];
      };
      const Obj<numbering_type>& getNumbering(const Obj<topology_type>& topology, const patch_type& patch, const int depth) {
        if ((this->_numberings.find(topology.ptr()) == this->_numberings.end()) ||
            (this->_numberings[topology.ptr()].find(patch) == this->_numberings[topology.ptr()].end()) ||
            (this->_numberings[topology.ptr()][patch].find(depth) == this->_numberings[topology.ptr()][patch].end())) {
          topology->constructOverlap(patch);
          Obj<numbering_type>    numbering   = new numbering_type(topology);
          Obj<send_overlap_type> sendOverlap = topology->getSendOverlap();
          Obj<recv_overlap_type> recvOverlap = topology->getRecvOverlap();

          this->construct(numbering, sendOverlap, recvOverlap, patch, topology->getLabelStratum(patch, "depth", depth));
          if (this->_debug) {std::cout << "Creating new numbering: patch " << patch << " depth " << depth << std::endl;}
          this->_numberings[topology.ptr()][patch][depth] = numbering;
        }
        return this->_numberings[topology.ptr()][patch][depth];
      };
      template<typename Atlas>
      const Obj<order_type>& getGlobalOrder(const Obj<topology_type>& topology, const patch_type& patch, const std::string& name, const Obj<Atlas>& atlas) {
        if ((this->_orders.find(topology.ptr()) == this->_orders.end()) ||
            (this->_orders[topology.ptr()].find(patch) == this->_orders[topology.ptr()].end()) ||
            (this->_orders[topology.ptr()][patch].find(name) == this->_orders[topology.ptr()][patch].end())) {
          topology->constructOverlap(patch);
          Obj<order_type>        order       = new order_type(topology);
          Obj<send_overlap_type> sendOverlap = topology->getSendOverlap();
          Obj<recv_overlap_type> recvOverlap = topology->getRecvOverlap();

          // FIX:
          //this->constructOrder(order, sendOverlap, recvOverlap, patch, atlas->getPatch(patch), atlas);
          this->constructOrder(order, sendOverlap, recvOverlap, patch, topology->depthStratum(patch, 0), atlas);
          if (this->_debug) {std::cout << "Creating new global order: patch " << patch << " name " << name << std::endl;}
          this->_orders[topology.ptr()][patch][name] = order;
        }
        return this->_orders[topology.ptr()][patch][name];
      };
    };
  }
}

#endif
