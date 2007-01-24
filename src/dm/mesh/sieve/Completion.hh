#ifndef included_ALE_Completion_hh
#define included_ALE_Completion_hh

#ifndef  included_ALE_Sections_hh
#include <Sections.hh>
#endif

extern PetscErrorCode PetscCommSynchronizeTags(MPI_Comm);

namespace ALE {
  namespace New {
    template<typename Section_>
    class PatchlessSection : public ALE::ParallelObject {
    public:
      typedef Section_                          section_type;
      typedef typename section_type::patch_type patch_type;
      typedef typename section_type::sieve_type sieve_type;
      typedef typename section_type::point_type point_type;
      typedef typename section_type::value_type value_type;
      typedef typename section_type::chart_type chart_type;
    protected:
      Obj<section_type> _section;
      const patch_type  _patch;
    public:
      PatchlessSection(const Obj<section_type>& section, const patch_type& patch) : ParallelObject(MPI_COMM_SELF, section->debug()), _section(section), _patch(patch) {};
      virtual ~PatchlessSection() {};
    public:
      const chart_type& getPatch(const patch_type& patch) {
        return this->_section->getAtlas()->getPatch(this->_patch);
      };
      bool hasPoint(const patch_type& patch, const point_type& point) {
        return this->_section->hasPoint(patch, point);
      };
      const value_type *restrict(const patch_type& patch) {
        return this->_section->restrict(this->_patch);
      };
      const value_type *restrict(const patch_type& patch, const point_type& p) {
        return this->_section->restrict(this->_patch, p);
      };
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        return this->_section->restrictPoint(this->_patch, p);
      };
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->_section->update(this->_patch, p, v);
      };
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->_section->updateAdd(this->_patch, p, v);
      };
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->_section->updatePoint(this->_patch, p, v);
      };
      template<typename Input>
      void update(const patch_type& patch, const point_type& p, const Obj<Input>& v) {
        this->_section->update(this->_patch, p, v);
      };
    public:
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        this->_section->view(name, comm);
      };
    };

    template<typename Topology_, typename Value_>
    class Completion {
    public:
      typedef int                                                                         point_type;
      typedef Value_                                                                      value_type;
      typedef Topology_                                                                   mesh_topology_type;
      typedef typename mesh_topology_type::sieve_type                                     sieve_type;
      typedef typename ALE::New::DiscreteSieve<point_type>                                dsieve_type;
      typedef typename ALE::New::Topology<int, dsieve_type>                               topology_type;
      typedef typename ALE::Sifter<int, point_type, point_type>                           send_overlap_type;
      typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, int>     send_sizer_type;
      typedef typename ALE::Sifter<point_type, int, point_type>                           recv_overlap_type;
      typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, int>     recv_sizer_type;
      typedef typename ALE::New::ConstantSection<topology_type, int>                      constant_sizer;
      typedef typename ALE::New::ConstantSection<topology_type, value_type>               constant_section;
      typedef typename ALE::New::ConeSizeSection<topology_type, mesh_topology_type, sieve_type> cone_size_section;
      typedef typename ALE::New::ConeSection<topology_type, sieve_type>                   cone_section;
      typedef typename ALE::New::SectionCompletion<mesh_topology_type,value_type>         completion;
    public:
      template<typename PartitionType>
      static void scatterSieve(const Obj<mesh_topology_type>& topology, const Obj<sieve_type>& sieve, const int dim, const Obj<sieve_type>& sieveNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const int numCells, const PartitionType assignment[]) {
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type> recv_section_type;
        int rank  = sieve->commRank();
        int debug = sieve->debug();

        // Create local sieve
        const Obj<topology_type::label_sequence>& cells = topology->heightStratum(0, 0);
        int e = 0;

        for(topology_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
          if (assignment[e] == rank) {
            Obj<typename sieve_type::coneSet> current = new typename sieve_type::coneSet();
            Obj<typename sieve_type::coneSet> next    = new typename sieve_type::coneSet();
            Obj<typename sieve_type::coneSet> tmp;

            current->insert(*e_iter);
            while(current->size()) {
              for(typename sieve_type::coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
                const Obj<typename sieve_type::traits::coneSequence>& cone = sieve->cone(*p_iter);
            
                for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
                  sieveNew->addArrow(*c_iter, *p_iter, c_iter.color());
                  next->insert(*c_iter);
                }
              }
              tmp = current; current = next; next = tmp;
              next->clear();
            }
          }
          e++;
        }
        sieveNew->stratify();
        // Complete sizer section
        typedef typename ALE::New::PartitionSizeSection<topology_type, mesh_topology_type, PartitionType> partition_size_section;
        typedef typename ALE::New::PartitionSection<topology_type, mesh_topology_type, PartitionType>     partition_section;
        Obj<topology_type>          secTopology          = completion::createSendTopology(sendOverlap);
        Obj<partition_size_section> partitionSizeSection = new partition_size_section(secTopology, topology, 0, numCells, assignment);
        Obj<partition_section>      partitionSection     = new partition_section(secTopology, topology, 0, numCells, assignment);
        Obj<send_section_type>      sendSection          = new send_section_type(sieve->comm(), sieve->debug());
        Obj<recv_section_type>      recvSection          = new recv_section_type(sieve->comm(), sendSection->getTag(), sieve->debug());

        completion::completeSection(sendOverlap, recvOverlap, partitionSizeSection, partitionSection, sendSection, recvSection);
        // Unpack the section into the overlap
        sendOverlap->clear();
        recvOverlap->clear();
        const topology_type::sheaf_type& sendPatches = sendSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = sendPatches.begin(); p_iter != sendPatches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename send_section_type::value_type *points = sendSection->restrict(p_iter->first, *b_iter);
            int size = sendSection->size(p_iter->first, *b_iter);

            for(int p = 0; p < size; p++) {
              sendOverlap->addArrow(points[p], p_iter->first, points[p]);
            }
          }
        }
        const topology_type::sheaf_type& recvPatches = recvSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = recvPatches.begin(); p_iter != recvPatches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                 rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getFiberDimension(rank, *b_iter);

            for(int p = 0; p < size; p++) {
              recvOverlap->addArrow(rank, points[p], points[p]);
            }
          }
        }
        if (debug) {
          sendOverlap->view(std::cout, "Send overlap for points");
          recvOverlap->view(std::cout, "Receive overlap for points");
        }
        // Receive the point section
        ALE::New::Completion<mesh_topology_type,value_type>::scatterCones(sieve, sieveNew, sendOverlap, recvOverlap);
        sieveNew->stratify();
      };
      template<typename PartitionType>
      static void scatterSieveByFace(const Obj<mesh_topology_type>& topology, const Obj<sieve_type>& sieve, const int dim, const Obj<sieve_type>& sieveNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const int numFaces, const PartitionType assignment[]) {
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type> send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type> recv_section_type;
        const typename topology_type::patch_type patch = 0;
        int rank  = sieve->commRank();
        int debug = sieve->debug();

        // Create local sieve
        const Obj<topology_type::label_sequence>& faces = topology->heightStratum(patch, 1);
        int f = 0;

        for(topology_type::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
          if (assignment[f] == rank) {
            Obj<typename sieve_type::coneSet> current = new typename sieve_type::coneSet();
            Obj<typename sieve_type::coneSet> next    = new typename sieve_type::coneSet();
            Obj<typename sieve_type::coneSet> tmp;

            current->insert(*f_iter);
            while(current->size()) {
              for(typename sieve_type::coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
                const Obj<typename sieve_type::traits::coneSequence>& cone = sieve->cone(*p_iter);
            
                for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
                  sieveNew->addArrow(*c_iter, *p_iter, c_iter.color());
                  next->insert(*c_iter);
                }
              }
              tmp = current; current = next; next = tmp;
              next->clear();
            }
            const Obj<typename sieve_type::traits::supportSequence>& support = sieve->support(*f_iter);

            for(typename sieve_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
              sieveNew->addArrow(*f_iter, *s_iter, s_iter.color());
            }
          }
          f++;
        }
        sieveNew->stratify();
        // Complete sizer section
        typedef typename ALE::New::PartitionSizeSection<topology_type, mesh_topology_type, PartitionType> partition_size_section;
        typedef typename ALE::New::PartitionSection<topology_type, mesh_topology_type, PartitionType>     partition_section;
        Obj<topology_type>          secTopology          = completion::createSendTopology(sendOverlap);
        Obj<partition_size_section> partitionSizeSection = new partition_size_section(secTopology, topology, 1, numFaces, assignment);
        Obj<partition_section>      partitionSection     = new partition_section(secTopology, topology, 1, numFaces, assignment);
        Obj<send_section_type>      sendSection          = new send_section_type(sieve->comm(), sieve->debug());
        Obj<recv_section_type>      recvSection          = new recv_section_type(sieve->comm(), sendSection->getTag(), sieve->debug());

        completion::completeSection(sendOverlap, recvOverlap, partitionSizeSection, partitionSection, sendSection, recvSection);
        // Unpack the section into the overlap
        sendOverlap->clear();
        recvOverlap->clear();
        const topology_type::sheaf_type& sendPatches = sendSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = sendPatches.begin(); p_iter != sendPatches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename send_section_type::value_type *points = sendSection->restrict(p_iter->first, *b_iter);
            int size = sendSection->size(p_iter->first, *b_iter);

            for(int p = 0; p < size; p++) {
              sendOverlap->addArrow(points[p], p_iter->first, points[p]);
            }
          }
        }
        const topology_type::sheaf_type& recvPatches = recvSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = recvPatches.begin(); p_iter != recvPatches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                 rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getFiberDimension(rank, *b_iter);

            for(int p = 0; p < size; p++) {
              recvOverlap->addArrow(rank, points[p], points[p]);
            }
          }
        }
        if (debug) {
          sendOverlap->view(std::cout, "Send overlap for points");
          recvOverlap->view(std::cout, "Receive overlap for points");
        }
        // Receive the point section
        ALE::New::Completion<mesh_topology_type,value_type>::scatterCones(sieve, sieveNew, sendOverlap, recvOverlap, topology, 1);
        ALE::New::Completion<mesh_topology_type,value_type>::scatterSupports(sieve, sieveNew, sendOverlap, recvOverlap, topology, topology->depth()-1);
        sieveNew->stratify();
      };
      template<typename SifterType>
      static void scatterCones(const Obj<SifterType>& sifter, const Obj<SifterType>& sifterNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<mesh_topology_type>& topology = NULL, const int minimumHeight = 0) {
        typedef typename ALE::New::ConeSizeSection<topology_type, mesh_topology_type, SifterType> cone_size_section;
        typedef typename ALE::New::ConeSection<topology_type, SifterType>                         cone_section;
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type>    send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type>    recv_section_type;
        Obj<topology_type>     secTopology     = completion::createSendTopology(sendOverlap);
        Obj<cone_size_section> coneSizeSection = new cone_size_section(secTopology, topology, sifter, minimumHeight);
        Obj<cone_section>      coneSection     = new cone_section(secTopology, sifter);
        Obj<send_section_type> sendSection     = new send_section_type(sifter->comm(), sifter->debug());
        Obj<recv_section_type> recvSection     = new recv_section_type(sifter->comm(), sendSection->getTag(), sifter->debug());

        completion::completeSection(sendOverlap, recvOverlap, coneSizeSection, coneSection, sendSection, recvSection);
        // Unpack the section into the sieve
        const topology_type::sheaf_type& patches = recvSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                 rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getFiberDimension(rank, *b_iter);
            int c    = 0;

            for(int p = 0; p < size; p++) {
              sifterNew->addArrow(points[p], *b_iter, c++);
            }
          }
        }
      };
      template<typename SifterType>
      static void scatterSupports(const Obj<SifterType>& sifter, const Obj<SifterType>& sifterNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<mesh_topology_type>& topology = NULL, const int minimumDepth = 0) {
        typedef typename ALE::New::SupportSizeSection<topology_type, mesh_topology_type, SifterType> support_size_section;
        typedef typename ALE::New::SupportSection<topology_type, SifterType>                         support_section;
        typedef typename ALE::New::OverlapValues<send_overlap_type, topology_type, value_type>       send_section_type;
        typedef typename ALE::New::OverlapValues<recv_overlap_type, topology_type, value_type>       recv_section_type;
        Obj<topology_type>        secTopology        = completion::createSendTopology(sendOverlap);
        Obj<support_size_section> supportSizeSection = new support_size_section(secTopology, topology, sifter, minimumDepth);
        Obj<support_section>      supportSection     = new support_section(secTopology, sifter);
        Obj<send_section_type>    sendSection        = new send_section_type(sifter->comm(), sifter->debug());
        Obj<recv_section_type>    recvSection        = new recv_section_type(sifter->comm(), sendSection->getTag(), sifter->debug());

        completion::completeSection(sendOverlap, recvOverlap, supportSizeSection, supportSection, sendSection, recvSection);
        // Unpack the section into the sieve
        const topology_type::sheaf_type& patches = recvSection->getTopology()->getPatches();

        for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                                 rank = p_iter->first;

          for(topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            const typename recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
            int size = recvSection->getFiberDimension(rank, *b_iter);
            int c    = 0;

            for(int p = 0; p < size; p++) {
              sifterNew->addArrow(*b_iter, points[p], c++);
            }
          }
        }
      };
    };

    template<typename Value_>
    class ParallelFactory {
    public:
      typedef Value_ value_type;
    protected:
      int          _debug;
      MPI_Datatype _mpiType;
    protected:
      MPI_Datatype constructMPIType() {
        if (sizeof(value_type) == 4) {
          return MPI_INT;
        } else if (sizeof(value_type) == 8) {
          return MPI_DOUBLE;
        } else if (sizeof(value_type) == 28) {
          int          blen[2];
          MPI_Aint     indices[2];
          MPI_Datatype oldtypes[2], newtype;
          blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_INT;
          blen[1] = 3; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
          MPI_Type_struct(2, blen, indices, oldtypes, &newtype);
          MPI_Type_commit(&newtype);
          return newtype;
        } else if (sizeof(value_type) == 32) {
          int          blen[2];
          MPI_Aint     indices[2];
          MPI_Datatype oldtypes[2], newtype;
          blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_DOUBLE;
          blen[1] = 3; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
          MPI_Type_struct(2, blen, indices, oldtypes, &newtype);
          MPI_Type_commit(&newtype);
          return newtype;
        }
        throw ALE::Exception("Cannot determine MPI type for value type");
      };
      ParallelFactory(const int debug) : _debug(debug) {
        this->_mpiType = this->constructMPIType();
      };
    public:
      ~ParallelFactory() {};
    public:
      static const Obj<ParallelFactory>& singleton(const int debug, bool cleanup = false) {
        static Obj<ParallelFactory> *_singleton = NULL;

        if (cleanup) {
          if (debug) {std::cout << "Destroying ParallelFactory" << std::endl;}
          if (_singleton) {delete _singleton;}
          _singleton = NULL;
        } else if (_singleton == NULL) {
          if (debug) {std::cout << "Creating new ParallelFactory" << std::endl;}
          _singleton  = new Obj<ParallelFactory>();
          *_singleton = new ParallelFactory(debug);
        }
        return *_singleton;
      };
    public: // Accessors
      int debug() const {return this->_debug;};
      MPI_Datatype getMPIType() const {return this->_mpiType;};
    };
  }
}
#endif
