#ifndef included_ALE_Completion_hh
#define included_ALE_Completion_hh

#ifndef  included_ALE_Sections_hh
#include <Sections.hh>
#endif

#ifndef  included_ALE_ParallelMapping_hh
#include <ParallelMapping.hh>
#endif

#include <iostream>
#include <fstream>

namespace ALE {
  class Completion {
  public:
    template<typename SendOverlap, typename RecvOverlap, typename SendSection, typename RecvSection>
    static void completeSection(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection) {
      typedef ALE::Section<ALE::Pair<int, typename SendOverlap::source_type>, typename SendSection::value_type> OverlapSection;
      //typedef ALE::Section<typename SendSection::point_type, typename SendSection::value_type> OverlapSection;
      Obj<OverlapSection> overlapSection = new OverlapSection(sendSection->comm(), sendSection->debug());

      if (sendSection->debug()) {sendSection->view("Send Section");}
      ALE::Pullback::SimpleCopy::copy(sendOverlap, recvOverlap, sendSection, overlapSection);
      if (overlapSection->debug()) {overlapSection->view("Overlap Section");}
      ALE::Pullback::InsertionBinaryFusion::fuse(overlapSection, recvOverlap, recvSection);
      if (recvSection->debug()) {recvSection->view("Receieve Section");}
    };
    template<typename SendOverlap, typename RecvOverlap, typename SendSection, typename RecvSection>
    static void completeSectionAdd(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const Obj<SendSection>& sendSection, const Obj<RecvSection>& recvSection) {
      typedef ALE::Section<ALE::Pair<int, typename SendOverlap::source_type>, typename SendSection::value_type> OverlapSection;
      Obj<OverlapSection> overlapSection = new OverlapSection(sendSection->comm(), sendSection->debug());

      if (sendSection->debug()) {sendSection->view("Send Section");}
      ALE::Pullback::SimpleCopy::copy(sendOverlap, recvOverlap, sendSection, overlapSection);
      if (overlapSection->debug()) {overlapSection->view("Overlap Section");}
      ALE::Pullback::AdditiveBinaryFusion::fuse(overlapSection, recvOverlap, recvSection);
      if (recvSection->debug()) {recvSection->view("Receieve Section");}
    };
  };
  namespace New {
    template<typename Bundle_, typename Value_, typename Alloc_ = malloc_allocator<typename Bundle_::point_type> >
    class Completion {
    public:
      typedef int                                                                         point_type;
      typedef Value_                                                                      value_type;
      typedef Bundle_                                                                     bundle_type;
      typedef Alloc_                                                                      alloc_type;
      typedef typename alloc_type::template rebind<int>::other                            int_alloc_type;
      typedef typename alloc_type::template rebind<value_type>::other                     value_alloc_type;
      typedef typename bundle_type::sieve_type                                            sieve_type;
      typedef typename ALE::DiscreteSieve<point_type, alloc_type>                         dsieve_type;
      typedef typename ALE::Topology<int, dsieve_type, alloc_type>                        topology_type;
      typedef typename bundle_type::rank_type                                             rank_type;
      typedef typename ALE::Sifter<point_type, rank_type, point_type>                     send_overlap_type;
      typedef typename ALE::Sifter<rank_type, point_type, point_type>                     recv_overlap_type;
      typedef typename ALE::Field<send_overlap_type, int, ALE::Section<point_type, int, int_alloc_type> > send_sizer_type;
      typedef typename ALE::Field<recv_overlap_type, int, ALE::Section<point_type, int, int_alloc_type> > recv_sizer_type;
      typedef typename ALE::New::ConeSizeSection<bundle_type, sieve_type>                 cone_size_section;
      typedef typename ALE::New::ConeSection<sieve_type>                                  cone_section;
      typedef typename ALE::New::SectionCompletion<bundle_type, value_type, alloc_type>   completion;
    public:
      template<typename PartitionType>
      static void scatterSieve(const Obj<bundle_type>& bundle, const Obj<sieve_type>& sieve, const int dim, const Obj<sieve_type>& sieveNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const int height, const int numCells, const PartitionType assignment[]) {
        typedef typename ALE::Field<send_overlap_type, int, ALE::Section<point_type, value_type, value_alloc_type> > send_section_type;
        typedef typename ALE::Field<recv_overlap_type, int, ALE::Section<point_type, value_type, value_alloc_type> > recv_section_type;
        int rank  = sieve->commRank();
        int debug = sieve->debug();

        // Create local sieve
        const Obj<typename bundle_type::label_sequence>& cells = bundle->heightStratum(height);
        int e = 0;

        if (sieve->debug()) {
          int e2 = 0;
          for(typename bundle_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
            std::cout << "assignment["<<*e_iter<<"]" << assignment[e2++] << std::endl;
          }
        }
        PetscTruth flg;
        PetscOptionsHasName(PETSC_NULL, "-output_partition", &flg);
        if (flg) {
          ostringstream fname;
          fname << "part." << sieve->commSize() << ".dat";
          std::ofstream f(fname.str().c_str());
          int e2 = 0;
          f << sieve->commSize() << std::endl;
          for(typename bundle_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
            f << assignment[e2++] << std::endl;
          }
        }
        for(typename bundle_type::label_sequence::iterator e_iter = cells->begin(); e_iter != cells->end(); ++e_iter) {
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
            if (height) {
              current->insert(*e_iter);
              while(current->size()) {
                for(typename sieve_type::coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
                  const Obj<typename sieve_type::traits::supportSequence>& support = sieve->support(*p_iter);
            
                  for(typename sieve_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
                    sieveNew->addArrow(*p_iter, *s_iter, s_iter.color());
                    next->insert(*s_iter);
                  }
                }
                tmp = current; current = next; next = tmp;
                next->clear();
              }
            }
          }
          e++;
        }
        // Complete sizer section
        typedef typename ALE::New::PartitionSizeSection<bundle_type, PartitionType> partition_size_section;
        typedef typename ALE::New::PartitionSection<bundle_type, PartitionType>     partition_section;
        Obj<topology_type>          secTopology          = completion::createSendTopology(sendOverlap);
        Obj<partition_size_section> partitionSizeSection = new partition_size_section(bundle, height, numCells, assignment);
        Obj<partition_section>      partitionSection     = new partition_section(bundle, height, numCells, assignment);
        Obj<send_section_type>      sendSection          = new send_section_type(sieve->comm(), sieve->debug());
        Obj<recv_section_type>      recvSection          = new recv_section_type(sieve->comm(), sendSection->getTag(), sieve->debug());

        completion::completeSection(sendOverlap, recvOverlap, partitionSizeSection, partitionSection, sendSection, recvSection);
        // Unpack the section into the overlap
        sendOverlap->clear();
        recvOverlap->clear();
        const typename send_section_type::sheaf_type& sendPatches = sendSection->getPatches();

        for(typename send_section_type::sheaf_type::const_iterator p_iter = sendPatches.begin(); p_iter != sendPatches.end(); ++p_iter) {
          const typename send_section_type::patch_type               rank    = p_iter->first;
          const Obj<typename send_section_type::section_type>&       section = p_iter->second;
          const typename send_section_type::section_type::chart_type chart   = section->getChart();

          for(typename send_section_type::section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename send_section_type::value_type *points = section->restrictPoint(*c_iter);
            int                                           size   = section->getFiberDimension(*c_iter);

            for(int p = 0; p < size; p++) {
              sendOverlap->addArrow(points[p], rank, points[p]);
            }
          }
        }
        const typename recv_section_type::sheaf_type& recvPatches = recvSection->getPatches();

        for(typename recv_section_type::sheaf_type::const_iterator p_iter = recvPatches.begin(); p_iter != recvPatches.end(); ++p_iter) {
          const typename send_section_type::patch_type               rank    = p_iter->first;
          const Obj<typename send_section_type::section_type>&       section = p_iter->second;
          const typename send_section_type::section_type::chart_type chart   = section->getChart();

          for(typename recv_section_type::section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename recv_section_type::value_type *points = section->restrictPoint(*c_iter);
            int                                           size   = section->getFiberDimension(*c_iter);

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
        ALE::New::Completion<bundle_type, value_type>::scatterCones(sieve, sieveNew, sendOverlap, recvOverlap, bundle, height);
        if (height) {
          ALE::New::Completion<bundle_type, value_type>::scatterSupports(sieve, sieveNew, sendOverlap, recvOverlap, bundle, bundle->depth()-height);
        }
      };
      template<typename SifterType>
      static void scatterCones(const Obj<SifterType>& sifter, const Obj<SifterType>& sifterNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<bundle_type>& bundle = NULL, const int minimumHeight = 0) {
        typedef typename ALE::New::ConeSizeSection<bundle_type, SifterType> cone_size_section;
        typedef typename ALE::New::ConeSection<SifterType>                  cone_section;
        typedef typename ALE::Field<send_overlap_type, int, ALE::Section<point_type, value_type, value_alloc_type> > send_section_type;
        typedef typename ALE::Field<recv_overlap_type, int, ALE::Section<point_type, value_type, value_alloc_type> > recv_section_type;
        Obj<topology_type>     secTopology     = completion::createSendTopology(sendOverlap);
        Obj<cone_size_section> coneSizeSection = new cone_size_section(bundle, sifter, minimumHeight);
        Obj<cone_section>      coneSection     = new cone_section(sifter);
        Obj<send_section_type> sendSection     = new send_section_type(sifter->comm(), sifter->debug());
        Obj<recv_section_type> recvSection     = new recv_section_type(sifter->comm(), sendSection->getTag(), sifter->debug());

        completion::completeSection(sendOverlap, recvOverlap, coneSizeSection, coneSection, sendSection, recvSection);
        // Unpack the section into the sieve
        const typename recv_section_type::sheaf_type& patches = recvSection->getPatches();

        for(typename recv_section_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename recv_section_type::section_type>&        section = p_iter->second;
          const typename recv_section_type::section_type::chart_type& chart   = section->getChart();

          for(typename recv_section_type::section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename recv_section_type::value_type *points = section->restrictPoint(*c_iter);
            int size = section->getFiberDimension(*c_iter);
            int c    = 0;

            for(int p = 0; p < size; p++) {
              sifterNew->addArrow(points[p], *c_iter, c++);
            }
          }
        }
      };
      template<typename SifterType, typename Renumbering>
      static void scatterCones(const Obj<SifterType>& sifter, const Obj<SifterType>& sifterNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, Renumbering& renumbering, const Obj<bundle_type>& bundle = NULL) {
	PETSc::Log::Event("ScatterCones").begin();
        typedef typename ALE::New::ConeSizeSection<bundle_type, SifterType> cone_size_section;
        typedef typename ALE::New::ConeSection<SifterType>                  cone_section;
        typedef typename ALE::Field<send_overlap_type, int, ALE::Section<point_type, value_type, value_alloc_type> > send_section_type;
        typedef typename ALE::Field<recv_overlap_type, int, ALE::Section<point_type, value_type, value_alloc_type> > recv_section_type;
        Obj<topology_type>     secTopology     = completion::createSendTopology(sendOverlap);
        Obj<cone_size_section> coneSizeSection = new cone_size_section(bundle, sifter);
        Obj<cone_section>      coneSection     = new cone_section(sifter);
        Obj<send_section_type> sendSection     = new send_section_type(sifter->comm(), sifter->debug());
        Obj<recv_section_type> recvSection     = new recv_section_type(sifter->comm(), sendSection->getTag(), sifter->debug());

	PETSc::Log::Event("ScatterConesComplete").begin();
        completion::completeSection(sendOverlap, recvOverlap, coneSizeSection, coneSection, sendSection, recvSection);
	PETSc::Log::Event("ScatterConesComplete").end();
	PETSc::Log::Event("ScatterConesUpdate").begin();
        // Unpack the section into the sieve
        const typename recv_section_type::sheaf_type& patches = recvSection->getPatches();

        for(typename recv_section_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename recv_section_type::section_type>&        section = p_iter->second;
          const typename recv_section_type::section_type::chart_type& chart   = section->getChart();

          for(typename recv_section_type::section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename recv_section_type::value_type *points = section->restrictPoint(*c_iter);
            int size = section->getFiberDimension(*c_iter);
            int c    = 0;

            for(int p = 0; p < size; p++) {
              sifterNew->addArrow(points[p], renumbering[*c_iter], c++);
            }
          }
        }
	PETSc::Log::Event("ScatterConesUpdate").end();
	PETSc::Log::Event("ScatterCones").end();
      };
      template<typename SifterType>
      static void scatterSupports(const Obj<SifterType>& sifter, const Obj<SifterType>& sifterNew, const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap, const Obj<bundle_type>& bundle = NULL, const int minimumDepth = 0) {
        typedef typename ALE::New::SupportSizeSection<bundle_type, SifterType> support_size_section;
        typedef typename ALE::New::SupportSection<SifterType>                  support_section;
        typedef typename ALE::Field<send_overlap_type, int, ALE::Section<point_type, value_type, value_alloc_type> > send_section_type;
        typedef typename ALE::Field<recv_overlap_type, int, ALE::Section<point_type, value_type, value_alloc_type> > recv_section_type;
        Obj<topology_type>        secTopology        = completion::createSendTopology(sendOverlap);
        Obj<support_size_section> supportSizeSection = new support_size_section(bundle, sifter, minimumDepth);
        Obj<support_section>      supportSection     = new support_section(sifter);
        Obj<send_section_type>    sendSection        = new send_section_type(sifter->comm(), sifter->debug());
        Obj<recv_section_type>    recvSection        = new recv_section_type(sifter->comm(), sendSection->getTag(), sifter->debug());

        completion::completeSection(sendOverlap, recvOverlap, supportSizeSection, supportSection, sendSection, recvSection);
        // Unpack the section into the sieve
        const typename recv_section_type::sheaf_type& recvPatches = recvSection->getPatches();

        for(typename recv_section_type::sheaf_type::const_iterator p_iter = recvPatches.begin(); p_iter != recvPatches.end(); ++p_iter) {
          const Obj<typename send_section_type::section_type>&       section = p_iter->second;
          const typename send_section_type::section_type::chart_type chart   = section->getChart();

          for(typename recv_section_type::section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename recv_section_type::value_type *points = section->restrictPoint(*c_iter);
            int                                           size   = section->getFiberDimension(*c_iter);
            int                                           c      = 0;

            for(int p = 0; p < size; p++) {
              sifterNew->addArrow(*c_iter, points[p], c++);
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
        if (sizeof(value_type) == 1) {
          return MPI_BYTE;
        } else if (sizeof(value_type) == 2) {
          return MPI_SHORT;
        } else if (sizeof(value_type) == 4) {
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
