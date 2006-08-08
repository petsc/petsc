#ifndef included_ALE_Distribution_hh
#define included_ALE_Distribution_hh

#ifndef  included_ALE_Mesh_hh
#include <Mesh.hh>
#endif

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

#ifndef  included_ALE_Completion_hh
#include <Completion.hh>
#endif

extern PetscErrorCode PetscCommSynchronizeTags(MPI_Comm);

namespace ALE {
  namespace New {
    template<typename Topology_>
    class Distribution {
    public:
      typedef Topology_                                                       topology_type;
      typedef ALE::New::Completion<Topology_, Mesh::sieve_type::point_type>   sieveCompletion;
      typedef ALE::New::Completion<Topology_, Mesh::section_type::value_type> sectionCompletion;
      typedef typename sectionCompletion::send_overlap_type                   send_overlap_type;
      typedef typename sectionCompletion::recv_overlap_type                   recv_overlap_type;
      typedef typename sectionCompletion::send_section_type                   send_section_type;
      typedef typename sectionCompletion::recv_section_type                   recv_section_type;
    public:
      static void sendMesh(const Obj<Mesh>& serialMesh, const Obj<Mesh>& parallelMesh) {
        typedef ALE::New::PatchlessSection<Mesh::section_type> CoordFiller;
        const Obj<Mesh::topology_type> topology         = serialMesh->getTopologyNew();
        const Obj<Mesh::topology_type> parallelTopology = parallelMesh->getTopologyNew();
        const int dim   = serialMesh->getDimension();
        const int debug = serialMesh->debug;

        Obj<send_overlap_type> cellOverlap   = sieveCompletion::sendDistribution(topology, dim, parallelTopology);
        Obj<send_overlap_type> vertexOverlap = new send_overlap_type(serialMesh->comm(), debug);
        Obj<Mesh::sieve_type>  sieve         = topology->getPatch(0);
        const Obj<typename send_overlap_type::traits::capSequence> cap = cellOverlap->cap();

        for(typename send_overlap_type::traits::baseSequence::iterator p_iter = cap->begin(); p_iter != cap->end(); ++p_iter) {
          const Obj<typename send_overlap_type::traits::supportSequence>& ranks = cellOverlap->support(*p_iter);

          for(typename send_overlap_type::traits::supportSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
            const Obj<typename Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*p_iter);

            for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              vertexOverlap->addArrow(*c_iter, *r_iter, *c_iter);
            }
          }
        }
        const Mesh::section_type::patch_type patch = 0;
        const Obj<Mesh::section_type> coordinates         = serialMesh->getSection("coordinates");
        const Obj<Mesh::section_type> parallelCoordinates = parallelMesh->getSection("coordinates");
        const Obj<send_section_type>  sendCoords          = new send_section_type(serialMesh->comm(), debug);
        const Obj<CoordFiller>        coordFiller         = new CoordFiller(coordinates, patch);
        const int embedDim = coordinates->getAtlas()->getFiberDimension(patch, *topology->depthStratum(patch, 0)->begin());
        const Obj<typename sectionCompletion::constant_sizer> constantSizer = new typename sectionCompletion::constant_sizer(MPI_COMM_SELF, embedDim, debug);

        sectionCompletion::sendSection(vertexOverlap, constantSizer, coordFiller, sendCoords);
        parallelCoordinates->getAtlas()->setFiberDimensionByDepth(patch, 0, embedDim);
        parallelCoordinates->getAtlas()->orderPatches();
        parallelCoordinates->allocate();
        const Obj<Mesh::topology_type::label_sequence>& vertices = topology->depthStratum(patch, 0);

        for(Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          parallelCoordinates->update(patch, *v_iter, coordinates->restrict(patch, *v_iter));
        }
      };
      static void receiveMesh(const Obj<Mesh>& serialMesh, const Obj<Mesh>& parallelMesh) {
        const Obj<Mesh::topology_type> topology         = serialMesh->getTopologyNew();
        const Obj<Mesh::topology_type> parallelTopology = parallelMesh->getTopologyNew();
        Obj<recv_overlap_type> cellOverlap   = sieveCompletion::receiveDistribution(topology, parallelTopology);
        Obj<recv_overlap_type> vertexOverlap = new recv_overlap_type(serialMesh->comm(), serialMesh->debug);
        Obj<Mesh::sieve_type>  parallelSieve = parallelTopology->getPatch(0);
        const Obj<typename send_overlap_type::traits::baseSequence> base = cellOverlap->base();

        for(typename send_overlap_type::traits::baseSequence::iterator p_iter = base->begin(); p_iter != base->end(); ++p_iter) {
          const Obj<typename send_overlap_type::traits::coneSequence>& ranks = cellOverlap->cone(*p_iter);

          for(typename send_overlap_type::traits::coneSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
            const Obj<typename ALE::Mesh::sieve_type::traits::coneSequence>& cone = parallelSieve->cone(*p_iter);

            for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              vertexOverlap->addArrow(*r_iter, *c_iter, *c_iter);
            }
          }
        }
        const Obj<Mesh::section_type> coordinates         = serialMesh->getSection("coordinates");
        const Obj<Mesh::section_type> parallelCoordinates = parallelMesh->getSection("coordinates");
        const Obj<recv_section_type>  recvCoords          = new recv_section_type(serialMesh->comm(), serialMesh->debug);
        const Mesh::section_type::patch_type patch        = 0;

        sectionCompletion::recvSection(vertexOverlap, recvCoords);
        const typename sectionCompletion::topology_type::sheaf_type& patches = recvCoords->getAtlas()->getTopology()->getPatches();
        const int embedDim = recvCoords->getAtlas()->getFiberDimension(patch, *recvCoords->getAtlas()->getTopology()->depthStratum(patches.begin()->first, 0)->begin());
        parallelCoordinates->getAtlas()->setFiberDimensionByDepth(patch, 0, embedDim);
        parallelCoordinates->getAtlas()->orderPatches();
        parallelCoordinates->allocate();

        for(typename sectionCompletion::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename sectionCompletion::topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(typename sectionCompletion::topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            parallelCoordinates->update(patch, *b_iter, recvCoords->restrict(p_iter->first, *b_iter));
          }
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "distributeMesh"
      static Obj<Mesh> distributeMesh(const Obj<Mesh>& serialMesh) {
        ALE_LOG_EVENT_BEGIN;
        Obj<Mesh> parallelMesh = Mesh(serialMesh->comm(), serialMesh->getDimension(), serialMesh->debug);
        const Obj<Mesh::topology_type>& topology = new Mesh::topology_type(serialMesh->comm(), serialMesh->debug);
        const Obj<Mesh::sieve_type>&    sieve    = new Mesh::sieve_type(serialMesh->comm(), serialMesh->debug);
        PetscErrorCode                  ierr;

        topology->setPatch(0, sieve);
        parallelMesh->setTopologyNew(topology);
        if (serialMesh->commRank() == 0) {
          Distribution<topology_type>::sendMesh(serialMesh, parallelMesh);
        } else {
          Distribution<topology_type>::receiveMesh(serialMesh, parallelMesh);
        }
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (serialMesh->debug) {
          serialMesh->getTopologyNew()->view("Serial topology");
          parallelMesh->getTopologyNew()->view("Parallel topology");
          //parallelMesh->getBoundary()->view("Parallel boundary");
        }
        parallelMesh->distributed = true;
        ALE_LOG_EVENT_END;
        return parallelMesh;
      };
    };

    template<typename Topology_>
    class Numbering : public ParallelObject {
    public:
      typedef Topology_                                                                           topology_type;
      typedef typename topology_type::point_type                                                  point_type;
      typedef typename topology_type::sieve_type                                                  sieve_type;
      typedef typename ALE::New::DiscreteSieve<point_type>                                        dsieve_type;
      typedef typename ALE::New::Topology<int, dsieve_type>                                       overlap_topology_type;
      typedef typename ALE::New::Atlas<overlap_topology_type, ALE::Point>                         overlap_atlas_type;
      typedef typename ALE::Sifter<int,point_type,point_type>                                     send_overlap_type;
      typedef typename ALE::New::OverlapValues<send_overlap_type, overlap_atlas_type, point_type> send_section_type;
      typedef typename ALE::Sifter<point_type,int,point_type>                                     recv_overlap_type;
      typedef typename ALE::New::OverlapValues<recv_overlap_type, overlap_atlas_type, point_type> recv_section_type;
    protected:
      Obj<topology_type>        _topology;
      std::string               _label;
      int                       _value;
      std::map<point_type, int> _order;
      Obj<send_overlap_type>    _sendOverlap;
      Obj<recv_overlap_type>    _recvOverlap;
      Obj<send_section_type>    _sendSection;
      Obj<recv_section_type>    _recvSection;
      int                       _localSize;
      int                      *_offsets;
    public:
      Numbering(const Obj<topology_type>& topology, const std::string& label, int value) : ParallelObject(topology->comm(), topology->debug()), _topology(topology), _label(label), _value(value) {
        this->_sendOverlap = new send_overlap_type(this->comm(), this->debug());
        this->_recvOverlap = new recv_overlap_type(this->comm(), this->debug());
        this->_sendSection = new send_section_type(this->comm(), this->debug());
        this->_recvSection = new recv_section_type(this->comm(), this->_sendSection->getTag(), this->debug());
        this->_offsets     = new int[this->commSize()+1];
        this->_offsets[0]  = 0;
      };
      ~Numbering() {
        delete [] this->_offsets;
      };
    public: // Accessors
      int getLocalSize() const {return this->_localSize;};
      int getGlobalSize() const {return this->_offsets[this->commSize()];};
      int getIndex(const point_type& point) {return std::abs(this->_order[point]);};
      bool isLocal(const point_type& point) {return this->_order[point] >= 0;};
      bool isRemote(const point_type& point) {return this->_order[point] < 0;};
    public:
      void constructOverlap() {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->getLabelStratum(0, this->_label, this->_value);

        point_type *sendBuf = new point_type[points->size()];
        int         size    = 0;
        for(typename topology_type::label_sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          sendBuf[size++] = *l_iter;
        }
        int *sizes   = new int[this->commSize()];
        int *offsets = new int[this->commSize()+1];
        point_type *remotePoints = NULL;
        int        *remoteRanks  = NULL;

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
        std::map<int, std::map<int, std::set<point_type> > > overlapInfo;

        if (this->commRank() == 0) {
          for(int p = 0; p < this->commSize(); p++) {
            std::sort(&remotePoints[offsets[p]], &remotePoints[offsets[p+1]]);
          }
          for(int p = 0; p < this->commSize(); p++) {
            for(int q = p+1; q < this->commSize(); q++) {
              std::set_intersection(&remotePoints[offsets[p]], &remotePoints[offsets[p+1]],
                                    &remotePoints[offsets[q]], &remotePoints[offsets[q+1]],
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
        int numOverlaps;
        MPI_Scatter(sizes, 1, MPI_INT, &numOverlaps, 1, MPI_INT, 0, this->comm());
        int *overlapRanks = new int[numOverlaps];
        MPI_Scatterv(remoteRanks, sizes, offsets, MPI_INT, overlapRanks, numOverlaps, MPI_INT, 0, this->comm());
        if (this->commRank() == 0) {
          for(int p = 0, k = 0; p < this->commSize(); p++) {
            sizes[p] = 0;
            for(int r = 0; r < (int) overlapInfo[p].size(); r++) {
              sizes[p] += remoteRanks[k*2+1];
              k++;
            }
            offsets[p+1] = offsets[p] + sizes[p];
          }
          for(int p = 0, k = 0; p < this->commSize(); p++) {
            for(typename std::map<int, std::set<point_type> >::iterator r_iter = overlapInfo[p].begin(); r_iter != overlapInfo[p].end(); ++r_iter) {
              int rank = r_iter->first;
              for(typename std::set<point_type>::iterator p_iter = (overlapInfo[p][rank]).begin(); p_iter != (overlapInfo[p][rank]).end(); ++p_iter) {
                remotePoints[k++] = *p_iter;
              }
            }
          }
        }
        int numOverlapPoints = 0;
        for(int r = 0; r < numOverlaps/2; r++) {
          numOverlapPoints += overlapRanks[r*2+1];
        }
        point_type *overlapPoints = new point_type[numOverlapPoints];
        MPI_Scatterv(remotePoints, sizes, offsets, MPI_INT, overlapPoints, numOverlapPoints, MPI_INT, 0, this->comm());

        for(int r = 0, k = 0; r < numOverlaps/2; r++) {
          int rank = overlapRanks[r*2];

          for(int p = 0; p < overlapRanks[r*2+1]; p++) {
            point_type point = overlapPoints[k++];

            this->_sendOverlap->addArrow(point, rank, point);
            this->_recvOverlap->addArrow(rank, point, point);
          }
        }

        delete [] overlapPoints;
        delete [] overlapRanks;
        delete [] sizes;
        delete [] offsets;
        if (this->commRank() == 0) {
          delete [] remoteRanks;
          delete [] remotePoints;
        }
        ostringstream sendMsg;
        sendMsg << "Send overlap for rank " << this->commRank();
        this->_sendOverlap->view(std::cout, sendMsg.str().c_str());
        ostringstream recvMsg;
        recvMsg << "Receive overlap for rank " << this->commRank();
        this->_recvOverlap->view(std::cout, recvMsg.str().c_str());
      };
      void constructLocalOrder() {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->getLabelStratum(0, this->_label, this->_value);

        this->_order.clear();
        this->_localSize = 0;
        for(typename topology_type::label_sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          if (this->_sendOverlap->capContains(*l_iter)) {
            const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = this->_sendOverlap->support(*l_iter);
            int minRank = this->_sendOverlap->commSize();

            for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
              if (*p_iter < minRank) minRank = *p_iter;
            }
            if (minRank < this->_sendOverlap->commRank()) {
              this->_order[*l_iter] = -1;
            } else {
              this->_order[*l_iter] = this->_localSize++;
            }
          } else {
            this->_order[*l_iter] = this->_localSize++;
          }
        }
        MPI_Allgather(&this->_localSize, 1, MPI_INT, &(this->_offsets[1]), 1, MPI_INT, this->comm());
        for(int p = 2; p <= this->commSize(); p++) {
          this->_offsets[p] += this->_offsets[p-1];
        }
        for(typename topology_type::label_sequence::iterator l_iter = points->begin(); l_iter != points->end(); ++l_iter) {
          if (this->_order[*l_iter] >= 0) {
            this->_order[*l_iter] += this->_offsets[this->commRank()];
          }
        }
      };
      void constructCommunication() {
        Obj<typename send_overlap_type::baseSequence> sendRanks = this->_sendOverlap->base();

        for(typename send_overlap_type::baseSequence::iterator r_iter = sendRanks->begin(); r_iter != sendRanks->end(); ++r_iter) {
          const Obj<typename send_overlap_type::coneSequence>& cone = this->_sendOverlap->cone(*r_iter);
          Obj<dsieve_type> sieve = new dsieve_type();

          for(typename send_overlap_type::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
            sieve->addPoint(*c_iter);
          }
          this->_sendSection->getAtlas()->getTopology()->setPatch(*r_iter, sieve);
        }
        this->_sendSection->getAtlas()->getTopology()->stratify();
        Obj<typename recv_overlap_type::capSequence> recvRanks = this->_recvOverlap->cap();

        for(typename recv_overlap_type::capSequence::iterator r_iter = recvRanks->begin(); r_iter != recvRanks->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::supportSequence>& support = this->_recvOverlap->support(*r_iter);
          Obj<dsieve_type> sieve = new dsieve_type();

          for(typename recv_overlap_type::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
            sieve->addPoint(*s_iter);
          }
          this->_recvSection->getAtlas()->getTopology()->setPatch(*r_iter, sieve);
        }
        this->_recvSection->getAtlas()->getTopology()->stratify();
        // Setup sections
        this->_sendSection->construct(1);
        this->_recvSection->construct(1);
        this->_sendSection->getAtlas()->orderPatches();
        this->_recvSection->getAtlas()->orderPatches();
        this->_sendSection->allocate();
        this->_recvSection->allocate();
        this->_sendSection->constructCommunication(send_section_type::SEND);
        this->_recvSection->constructCommunication(recv_section_type::RECEIVE);
      };
      void fillSection() {
        Obj<typename send_overlap_type::traits::capSequence> sendPoints = this->_sendOverlap->cap();

        for(typename send_overlap_type::traits::capSequence::iterator s_iter = sendPoints->begin(); s_iter != sendPoints->end(); ++s_iter) {
          const Obj<typename send_overlap_type::traits::supportSequence>& sendPatches = this->_sendOverlap->support(*s_iter);

          for(typename send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
            this->_sendSection->update(*p_iter, *s_iter, &(this->_order[*s_iter]));
          }
        }
      };
      void communicate() {
        this->_sendSection->startCommunication();
        this->_recvSection->startCommunication();
        this->_sendSection->endCommunication();
        this->_recvSection->endCommunication();
      };
      void fillOrder() {
        Obj<typename recv_overlap_type::traits::baseSequence> recvPoints = this->_recvOverlap->base();

        for(typename recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
          const Obj<typename recv_overlap_type::traits::coneSequence>& recvPatches = this->_recvOverlap->cone(*r_iter);
    
          for(typename recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != recvPatches->end(); ++p_iter) {
            const typename recv_section_type::value_type *values = this->_recvSection->restrict(*p_iter, *r_iter);

            if (values[0] >= 0) {
              if (this->_order[*r_iter] >= 0) {
                ostringstream msg;
                msg << "Multiple indices for point " << *r_iter;
                throw ALE::Exception(msg.str().c_str());
              }
              this->_order[*r_iter] = -values[0];
            }
          }
        }
      };
      void construct() {
        //const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap
        this->constructOverlap();
        this->constructLocalOrder();
        this->constructCommunication();
        this->fillSection();
        this->communicate();
        this->fillOrder();
      };
      void view(const std::string& name) {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->getLabelStratum(0, this->_label, this->_value);
        ostringstream txt;

        if (name == "") {
          if(this->commRank() == 0) {
            txt << "viewing a Numbering" << std::endl;
          }
        } else {
          if(this->commRank() == 0) {
            txt << "viewing Numbering '" << name << "'" << std::endl;
          }
        }
        for(typename topology_type::label_sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          txt << "[" << this->commRank() << "] " << *p_iter << " --> " << this->_order[*p_iter] << std::endl;
        }
        PetscSynchronizedPrintf(this->comm(), txt.str().c_str());
        PetscSynchronizedFlush(this->comm());
      };
    };
  }
}

#endif
