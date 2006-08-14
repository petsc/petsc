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
      template<typename Sizer, typename Filler>
      static void sendSection(const Obj<send_overlap_type>& overlap, const Obj<Sizer>& sizer, const Obj<Filler>& filler, const Obj<Mesh::section_type>& serialSection, const Obj<Mesh::section_type>& parallelSection) {
        const Obj<send_section_type>         sendSec = new send_section_type(serialSection->comm(), serialSection->debug());

        parallelSection->getAtlas()->copyByDepth(serialSection->getAtlas());
        parallelSection->allocate();
        sectionCompletion::sendSection(overlap, sizer, filler, sendSec);
        const typename Mesh::section_type::topology_type::sheaf_type& patches = parallelSection->getAtlas()->getTopology()->getPatches();

        for(typename Mesh::section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename send_section_type::patch_type&             patch = p_iter->first;
          const typename send_section_type::atlas_type::chart_type& chart = parallelSection->getAtlas()->getChart(patch);

          for(typename send_section_type::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename send_section_type::point_type& point = c_iter->first;

            parallelSection->update(patch, point, serialSection->restrict(patch, point));
          }
        }
      };
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
        Obj<std::set<std::string> > sections = serialMesh->getSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          const Mesh::section_type::patch_type                  patch         = 0;
          const Obj<Mesh::section_type>&                        serialSection = serialMesh->getSection(*name);
          const Obj<CoordFiller>                                filler        = new CoordFiller(serialSection, patch);
          int size = serialSection->getAtlas()->getChart(patch).begin()->second.index;
          const Obj<typename sectionCompletion::constant_sizer> sizer  = new typename sectionCompletion::constant_sizer(MPI_COMM_SELF, size, debug);
          // Need to associate overlaps with sections somehow (through the atlas?)
          if (*name == "material") {
            sendSection(cellOverlap, sizer, filler, serialSection, parallelMesh->getSection(*name));
          } else {
            sendSection(vertexOverlap, sizer, filler, serialSection, parallelMesh->getSection(*name));
          }
        }
      };
      static void receiveSection(const Obj<recv_overlap_type>& overlap, const Obj<Mesh::section_type>& serialSection, const Obj<Mesh::section_type>& parallelSection) {
        const Obj<recv_section_type>         recvSec = new recv_section_type(serialSection->comm(), serialSection->debug());
        const Mesh::section_type::patch_type patch   = 0;

        sectionCompletion::recvSection(overlap, recvSec);
        parallelSection->getAtlas()->copyByDepth(recvSec->getAtlas(), parallelSection->getAtlas()->getTopology());
        parallelSection->allocate();
        const typename sectionCompletion::topology_type::sheaf_type& patches = recvSec->getAtlas()->getTopology()->getPatches();

        for(typename sectionCompletion::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename sectionCompletion::topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

          for(typename sectionCompletion::topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            parallelSection->update(patch, *b_iter, recvSec->restrict(p_iter->first, *b_iter));
          }
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
        Obj<std::set<std::string> > sections = serialMesh->getSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          // Need to associate overlaps with sections somehow (through the atlas?)
          if (*name == "material") {
            receiveSection(cellOverlap, serialMesh->getSection(*name), parallelMesh->getSection(*name));
          } else {
            receiveSection(vertexOverlap, serialMesh->getSection(*name), parallelMesh->getSection(*name));
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
        Obj<std::set<std::string> > sections = serialMesh->getSections();
        if (serialMesh->debug) {
          serialMesh->getTopologyNew()->view("Serial topology");
          for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
            serialMesh->getSection(*name)->view(*name);
          }
        }
        if (serialMesh->commRank() == 0) {
          Distribution<topology_type>::sendMesh(serialMesh, parallelMesh);
        } else {
          Distribution<topology_type>::receiveMesh(serialMesh, parallelMesh);
        }
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        if (parallelMesh->debug) {
          parallelMesh->getTopologyNew()->view("Parallel topology");
          for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
            parallelMesh->getSection(*name)->view(*name);
          }
        }
        parallelMesh->distributed = true;
        ALE_LOG_EVENT_END;
        return parallelMesh;
      };
    };
  }
}

#endif
