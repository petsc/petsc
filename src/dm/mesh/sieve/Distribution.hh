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
    public:
      #undef __FUNCT__
      #define __FUNCT__ "sendSection"
      template<typename Section, typename Sizer, typename Filler>
      static void sendSection(const Obj<send_overlap_type>& overlap, const Obj<Sizer>& sizer, const Obj<Filler>& filler, const Obj<Section>& serialSection, const Obj<Section>& parallelSection) {
        ALE_LOG_EVENT_BEGIN;
        typedef typename ALE::New::OverlapValues<send_overlap_type, typename sectionCompletion::atlas_type, typename Section::value_type> send_section_type;
        const Obj<send_section_type> sendSec = new send_section_type(serialSection->comm(), serialSection->debug());

        sectionCompletion::sendSection(overlap, sizer, filler, sendSec);
        const Obj<typename Section::atlas_type>&                      serialAtlas = serialSection->getAtlas();
        const Obj<typename Section::atlas_type>&                      atlas       = parallelSection->getAtlas();
        const Obj<typename Section::topology_type>&                   topology    = atlas->getTopology();
        const typename Mesh::section_type::topology_type::sheaf_type& patches     = topology->getPatches();

        for(typename Mesh::section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename Section::patch_type&                     patch = p_iter->first;
          const Obj<typename Section::topology_type::sieve_type>& sieve = p_iter->second;
          const typename Section::atlas_type::chart_type&         chart = serialAtlas->getChart(patch);

          for(typename Section::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename Section::point_type& point = c_iter->first;

            if (sieve->hasPoint(point)) {
              atlas->setFiberDimension(patch, point, serialAtlas->getFiberDimension(patch, point));
            }
          }
        }
        atlas->orderPatches();
        //parallelSection->getAtlas()->copyByDepth(serialSection->getAtlas());
        parallelSection->allocate();
        for(typename Mesh::section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const typename Section::patch_type&             patch = p_iter->first;
          const typename Section::atlas_type::chart_type& chart = parallelSection->getAtlas()->getChart(patch);

          for(typename Section::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename Section::point_type& point = c_iter->first;

            parallelSection->update(patch, point, serialSection->restrict(patch, point));
          }
        }
        ALE_LOG_EVENT_END;
      };
      static void sendMesh(const Obj<Mesh>& serialMesh, const Obj<Mesh>& parallelMesh) {
        typedef ALE::New::SizeSection<Mesh::section_type>      SectionSizer;
        typedef ALE::New::PatchlessSection<Mesh::section_type> SectionFiller;
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
          const Mesh::section_type::patch_type patch         = 0;
          const Obj<Mesh::section_type>&       serialSection = serialMesh->getSection(*name);
          const Obj<SectionSizer>              sizer         = new SectionSizer(serialSection, patch);
          const Obj<SectionFiller>             filler        = new SectionFiller(serialSection, patch);
          // Need to associate overlaps with sections somehow (through the atlas?)
          if (*name == "material") {
            sendSection(cellOverlap, sizer, filler, serialSection, parallelMesh->getSection(*name));
          } else {
            sendSection(vertexOverlap, sizer, filler, serialSection, parallelMesh->getSection(*name));
          }
        }
        if (!serialMesh->getSplitSection().isNull()) {
          typedef ALE::New::SizeSection<Mesh::split_section_type>      SplitSizer;
          typedef ALE::New::PatchlessSection<Mesh::split_section_type> SplitFiller;
          const Mesh::section_type::patch_type patch              = 0;
          const Obj<Mesh::split_section_type>& serialSplitField   = serialMesh->getSplitSection();
          const Obj<SplitSizer>                sizer              = new SplitSizer(serialSplitField, patch);
          const Obj<SplitFiller>               filler             = new SplitFiller(serialSplitField, patch);
          Obj<Mesh::split_section_type>        parallelSplitField = new Mesh::split_section_type(serialMesh->comm(), debug);

          parallelSplitField->getAtlas()->setTopology(parallelMesh->getTopologyNew());
          sendSection(cellOverlap, sizer, filler, serialSplitField, parallelSplitField);
          parallelMesh->setSplitSection(parallelSplitField);
        }
      };
      template<typename Section>
      static void receiveSection(const Obj<recv_overlap_type>& overlap, const Obj<Section>& serialSection, const Obj<Section>& parallelSection) {
        typedef typename ALE::New::OverlapValues<recv_overlap_type, typename sectionCompletion::atlas_type, typename Section::value_type> recv_section_type;
        const Obj<recv_section_type>         recvSec = new recv_section_type(serialSection->comm(), serialSection->debug());
        const Mesh::section_type::patch_type patch   = 0;

        sectionCompletion::recvSection(overlap, recvSec);
        const Obj<typename recv_section_type::atlas_type>&           serialAtlas = recvSec->getAtlas();
        const Obj<typename Section::atlas_type>&                     atlas       = parallelSection->getAtlas();
        const typename sectionCompletion::topology_type::sheaf_type& patches     = serialAtlas->getTopology()->getPatches();

        for(typename sectionCompletion::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename sectionCompletion::topology_type::sieve_type::baseSequence>& base  = p_iter->second->base();

          for(typename sectionCompletion::topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            atlas->setFiberDimension(patch, *b_iter, serialAtlas->getFiberDimension(p_iter->first, *b_iter));
          }
        }
        atlas->orderPatches();
        //parallelSection->getAtlas()->copyByDepth(recvSec->getAtlas(), parallelSection->getAtlas()->getTopology());
        parallelSection->allocate();

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
        if (!serialMesh->getSplitSection().isNull()) {
          Obj<Mesh::split_section_type> parallelSplitField = new Mesh::split_section_type(serialMesh->comm(), serialMesh->debug);

          parallelSplitField->getAtlas()->setTopology(parallelMesh->getTopologyNew());
          receiveSection(cellOverlap, serialMesh->getSplitSection(), parallelSplitField);
          parallelMesh->setSplitSection(parallelSplitField);
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "distributeMesh"
      static Obj<Mesh> distributeMesh(const Obj<Mesh>& serialMesh) {
        Obj<Mesh> parallelMesh = Mesh(serialMesh->comm(), serialMesh->getDimension(), serialMesh->debug);
        const Obj<Mesh::topology_type>& topology = new Mesh::topology_type(serialMesh->comm(), serialMesh->debug);
        const Obj<Mesh::sieve_type>&    sieve    = new Mesh::sieve_type(serialMesh->comm(), serialMesh->debug);
        PetscErrorCode                  ierr;

        if (serialMesh->distributed) return serialMesh;
        ALE_LOG_EVENT_BEGIN;
        topology->setPatch(0, sieve);
        parallelMesh->setTopologyNew(topology);
        if (serialMesh->debug) {
          Obj<std::set<std::string> > sections = serialMesh->getSections();

          serialMesh->getTopologyNew()->view("Serial topology");
          for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
            serialMesh->getSection(*name)->view(*name);
          }
          if (!serialMesh->getSplitSection().isNull()) {
            serialMesh->getSplitSection()->view("Serial split field");
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
          Obj<std::set<std::string> > sections = serialMesh->getSections();

          parallelMesh->getTopologyNew()->view("Parallel topology");
          for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
            parallelMesh->getSection(*name)->view(*name);
          }
          if (!parallelMesh->getSplitSection().isNull()) {
            parallelMesh->getSplitSection()->view("Parallel split field");
          }
        }
        parallelMesh->distributed = true;
        ALE_LOG_EVENT_END;
        return parallelMesh;
      };
      static void receiveMesh2(const Obj<Mesh>& parallelMesh, const Obj<Mesh>& serialMesh) {
        const Obj<Mesh::topology_type> serialTopology   = serialMesh->getTopologyNew();
        const Obj<Mesh::topology_type> parallelTopology = parallelMesh->getTopologyNew();
        Obj<recv_overlap_type> cellOverlap   = sieveCompletion::receiveDistribution2(parallelTopology, serialTopology);
        Obj<recv_overlap_type> vertexOverlap = new recv_overlap_type(serialMesh->comm(), serialMesh->debug);
        Obj<Mesh::sieve_type>  serialSieve   = serialTopology->getPatch(0);
        const Obj<typename send_overlap_type::traits::baseSequence> base = cellOverlap->base();

        for(typename send_overlap_type::traits::baseSequence::iterator p_iter = base->begin(); p_iter != base->end(); ++p_iter) {
          const Obj<typename send_overlap_type::traits::coneSequence>& ranks = cellOverlap->cone(*p_iter);

          for(typename send_overlap_type::traits::coneSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
            const Obj<typename ALE::Mesh::sieve_type::traits::coneSequence>& cone = serialSieve->cone(*p_iter);

            for(typename Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
              vertexOverlap->addArrow(*r_iter, *c_iter, *c_iter);
            }
          }
        }
        Obj<std::set<std::string> > sections = parallelMesh->getSections();

        for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
          // Need to associate overlaps with sections somehow (through the atlas?)
          if (*name == "material") {
            receiveSection2(cellOverlap, parallelMesh->getSection(*name), serialMesh->getSection(*name));
          } else {
            receiveSection2(vertexOverlap, parallelMesh->getSection(*name), serialMesh->getSection(*name));
          }
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "unifyMesh"
      static Obj<Mesh> unifyMesh(const Obj<Mesh>& parallelMesh) {
        Obj<Mesh> serialMesh = Mesh(parallelMesh->comm(), parallelMesh->getDimension(), parallelMesh->debug);
        const Obj<Mesh::topology_type>& topology = new Mesh::topology_type(parallelMesh->comm(), parallelMesh->debug);
        const Obj<Mesh::sieve_type>&    sieve    = new Mesh::sieve_type(parallelMesh->comm(), parallelMesh->debug);
        PetscErrorCode                  ierr;

        if (!parallelMesh->distributed) return parallelMesh;
        ALE_LOG_EVENT_BEGIN;
        topology->setPatch(0, sieve);
        serialMesh->setTopologyNew(topology);
        if (serialMesh->commRank() != 0) {
          Distribution<topology_type>::sendMesh2(parallelMesh, serialMesh);
        } else {
          Distribution<topology_type>::receiveMesh2(parallelMesh, serialMesh);
        }
        // This is necessary since we create types (like PartitionSection) on a subset of processors
        ierr = PetscCommSynchronizeTags(PETSC_COMM_WORLD);
        serialMesh->distributed = false;
        ALE_LOG_EVENT_END;
        return serialMesh;
      };
    };
  }
}

#endif
