#ifndef included_ALE_Partitioner_hh
#define included_ALE_Partitioner_hh

#include <petscvec.h>

/* Chaco does not have an include file */
extern "C" {
  extern int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                       float *ewgts, float *x, float *y, float *z, char *outassignname,
                       char *outfilename, short *assignment, int architecture, int ndims_tot,
                       int mesh_dims[3], double *goal, int global_method, int local_method,
                       int rqi_flag, int vmax, int ndims, double eigtol, long seed);

  extern int FREE_GRAPH;
}

namespace ALE {
  class Distributer {
  public:
    #undef __FUNCT__
    #define __FUNCT__ "Part::distribute"
    template<typename Sifter_>
    static void distribute(Obj<Sifter_> oldSifter, Obj<Sifter_> newSifter, bool restrict = true) {
      typedef Sifter_ sifter_type;
      typedef RightSequenceDuplicator<ConeArraySequence<typename sifter_type::traits::arrow_type> > fuser;
      typedef ParConeDelta<sifter_type, fuser,
                           typename sifter_type::template rebind<typename fuser::fusion_source_type,
                                                                 typename fuser::fusion_target_type,
                                                                 typename fuser::fusion_color_type,
                                                                 typename sifter_type::traits::cap_container_type::template rebind<typename fuser::fusion_source_type, typename sifter_type::traits::sourceRec_type::template rebind<typename fuser::fusion_source_type>::type>::type,
                                                                 typename sifter_type::traits::base_container_type::template rebind<typename fuser::fusion_target_type, typename sifter_type::traits::targetRec_type::template rebind<typename fuser::fusion_target_type>::type>::type
      >::type> coneDelta_type;
      typedef ParSupportDelta<sifter_type, fuser,
                              typename sifter_type::template rebind<typename fuser::fusion_source_type,
                                                                    typename fuser::fusion_target_type,
                                                                    typename fuser::fusion_color_type,
                                                                    typename sifter_type::traits::cap_container_type::template rebind<typename fuser::fusion_source_type, typename sifter_type::traits::sourceRec_type::template rebind<typename fuser::fusion_source_type>::type>::type,
                                                                    typename sifter_type::traits::base_container_type::template rebind<typename fuser::fusion_target_type, typename sifter_type::traits::targetRec_type::template rebind<typename fuser::fusion_target_type>::type>::type
      >::type> supportDelta_type;
      ALE_LOG_EVENT_BEGIN;
      // Construct a Delta object and a base overlap object
      coneDelta_type::setDebug(oldSifter->debug);
      Obj<typename coneDelta_type::bioverlap_type> overlap = coneDelta_type::overlap(oldSifter, newSifter);
      // Cone complete to move the partitions to the other processors
      Obj<typename coneDelta_type::fusion_type>    fusion  = coneDelta_type::fusion(oldSifter, newSifter, overlap);
      // Merge in the completion
      newSifter->add(fusion);
      if (oldSifter->debug) {
        overlap->view("Initial overlap");
        fusion->view("Initial fusion");
        newSifter->view("After merging inital fusion");
      }
      // Remove partition points
      for(int p = 0; p < oldSifter->commSize(); ++p) {
        oldSifter->removeBasePoint(typename sifter_type::traits::target_type(-1, p));
        newSifter->removeBasePoint(typename sifter_type::traits::target_type(-1, p));
      }
      // Support complete to build the local topology
      supportDelta_type::setDebug(oldSifter->debug);
      Obj<typename supportDelta_type::bioverlap_type> overlap2 = supportDelta_type::overlap(oldSifter, newSifter);
      Obj<typename supportDelta_type::fusion_type>    fusion2  = supportDelta_type::fusion(oldSifter, newSifter, overlap2);
      newSifter->add(fusion2, true && restrict);
      if (oldSifter->debug) {
        overlap2->view("Second overlap");
        fusion2->view("Second fusion");
        newSifter->view("After merging second fusion");
      }
      ALE_LOG_EVENT_END;
    };
    #undef __FUNCT__
    #define __FUNCT__ "createMappingStoP"
    template<typename FieldType, typename OverlapType>
    static VecScatter createMappingStoP(Obj<FieldType> serialSifter, Obj<FieldType> parallelSifter, Obj<OverlapType> overlap, bool doExchange = false) {
      VecScatter scatter;
      Obj<typename OverlapType::traits::baseSequence> neighbors = overlap->base();
      MPI_Comm comm = serialSifter->comm();
      int      rank = serialSifter->commRank();
      int      debug = serialSifter->debug;
      typename FieldType::patch_type patch;
      Vec        serialVec, parallelVec;
      PetscErrorCode ierr;

      if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "Creating mapping\n");}
      // Use an MPI vector for the serial data since it has no overlap
      if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "  Creating serial indices\n");}
      if (serialSifter->debug) {
        serialSifter->view("SerialSifter");
        overlap->view("Partition Overlap");
      }
      ierr = VecCreateMPIWithArray(serialSifter->comm(), serialSifter->getSize(patch), PETSC_DETERMINE, serialSifter->restrict(patch), &serialVec);CHKERROR(ierr, "Error in VecCreate");
      // Use individual serial vectors for each of the parallel domains
      if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "  Creating parallel indices\n");}
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, parallelSifter->getSize(patch), parallelSifter->restrict(patch), &parallelVec);CHKERROR(ierr, "Error in VecCreate");

      int NeighborCountA = 0, NeighborCountB = 0;
      for(typename OverlapType::traits::baseSequence::iterator neighbor = neighbors->begin(); neighbor != neighbors->end(); ++neighbor) {
        Obj<typename OverlapType::traits::coneSequence> cone = overlap->cone(*neighbor);

        for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          if ((*p_iter).first == 0) {
            NeighborCountA++;
            break;
          }
        }
        for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          if ((*p_iter).first == 1) {
            NeighborCountB++;
            break;
          }
        } 
      }

      int *NeighborsA, *NeighborsB; // Neighbor processes
      int *SellSizesA, *BuySizesA;  // Sizes of the A cones to transmit and B cones to receive
      int *SellSizesB, *BuySizesB;  // Sizes of the B cones to transmit and A cones to receive
      int *SellConesA = PETSC_NULL;
      int *SellConesB = PETSC_NULL;
      int nA, nB, offsetA, offsetB;
      ierr = PetscMalloc2(NeighborCountA,int,&NeighborsA,NeighborCountB,int,&NeighborsB);CHKERROR(ierr, "Error in PetscMalloc");
      ierr = PetscMalloc2(NeighborCountA,int,&SellSizesA,NeighborCountA,int,&BuySizesA);CHKERROR(ierr, "Error in PetscMalloc");
      ierr = PetscMalloc2(NeighborCountB,int,&SellSizesB,NeighborCountB,int,&BuySizesB);CHKERROR(ierr, "Error in PetscMalloc");

      nA = 0;
      nB = 0;
      for(typename OverlapType::traits::baseSequence::iterator neighbor = neighbors->begin(); neighbor != neighbors->end(); ++neighbor) {
        Obj<typename OverlapType::traits::coneSequence> cone = overlap->cone(*neighbor);

        for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          if ((*p_iter).first == 0) {
            NeighborsA[nA] = *neighbor;
            BuySizesA[nA] = 0;
            SellSizesA[nA] = 0;
            nA++;
            break;
          }
        }
        for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          if ((*p_iter).first == 1) {
            NeighborsB[nB] = *neighbor;
            BuySizesB[nB] = 0;
            SellSizesB[nB] = 0;
            nB++;
            break;
          }
        } 
      }
      if ((nA != NeighborCountA) || (nB != NeighborCountB)) {
        throw ALE::Exception("Invalid neighbor count");
      }

      nA = 0;
      offsetA = 0;
      nB = 0;
      offsetB = 0;
      for(typename OverlapType::traits::baseSequence::iterator neighbor = neighbors->begin(); neighbor != neighbors->end(); ++neighbor) {
        Obj<typename OverlapType::traits::coneSequence> cone = overlap->cone(*neighbor);
        int foundA = 0, foundB = 0;

        for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          if ((*p_iter).first == 0) {
            // Assume the same index sizes
            int idxSize = serialSifter->getIndex(patch, (*p_iter).second).index;

            BuySizesA[nA] += idxSize;
            SellSizesA[nA] += idxSize;
            offsetA += idxSize;
            foundA = 1;
          } else {
            // Assume the same index sizes
            int idxSize = parallelSifter->getIndex(patch, (*p_iter).second).index;

            BuySizesB[nB] += idxSize;
            SellSizesB[nB] += idxSize;
            offsetB += idxSize;
            foundB = 1;
          }
        }
        if (foundA) nA++;
        if (foundB) nB++;
      }

      ierr = PetscMalloc2(offsetA,int,&SellConesA,offsetB,int,&SellConesB);CHKERROR(ierr, "Error in PetscMalloc");
      offsetA = 0;
      offsetB = 0;
      for(typename OverlapType::traits::baseSequence::iterator neighbor = neighbors->begin(); neighbor != neighbors->end(); ++neighbor) {
        Obj<typename OverlapType::traits::coneSequence> cone = overlap->cone(*neighbor);

        for(typename OverlapType::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          const Point& p = (*p_iter).second;

          if ((*p_iter).first == 0) {
            const typename FieldType::index_type& idx = serialSifter->getIndex(patch, p);

            if (debug) {
              ostringstream txt;

              txt << "["<<rank<<"]Packing A index " << idx << " for " << *neighbor << std::endl;
              ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
            }
            for(int i = idx.prefix; i < idx.prefix+idx.index; ++i) {
              SellConesA[offsetA++] = i;
            }
          } else {
            const typename FieldType::index_type& idx = parallelSifter->getIndex(patch, p);

            if (debug) {
              ostringstream txt;

              txt << "["<<rank<<"]Packing B index " << idx << " for " << *neighbor << std::endl;
              ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
            }
            for(int i = idx.prefix; i < idx.prefix+idx.index; ++i) {
              SellConesB[offsetB++] = i;
            }
          }
        }
      }
      if (debug) {
        ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
      }

      ierr = VecScatterCreateEmpty(comm, &scatter);CHKERROR(ierr, "Error in VecScatterCreate");
      scatter->from_n = serialSifter->getSize(patch);
      scatter->to_n = parallelSifter->getSize(patch);
      ierr = VecScatterCreateLocal_PtoS(NeighborCountA, SellSizesA, NeighborsA, SellConesA, NeighborCountB, SellSizesB, NeighborsB, SellConesB, 1, scatter);CHKERROR(ierr, "Error in VecScatterCreate");

      if (doExchange) {
        if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "  Exchanging data\n");}
        ierr = VecScatterBegin(serialVec, parallelVec, INSERT_VALUES, SCATTER_FORWARD, scatter);CHKERROR(ierr, "Error in VecScatter");
        ierr = VecScatterEnd(serialVec, parallelVec, INSERT_VALUES, SCATTER_FORWARD, scatter);CHKERROR(ierr, "Error in VecScatter");
      }

      ierr = VecDestroy(serialVec);CHKERROR(ierr, "Error in VecDestroy");
      ierr = VecDestroy(parallelVec);CHKERROR(ierr, "Error in VecDestroy");
      return scatter;
    };
  };
  template<typename Mesh_> class MeshPartitioner {
  public:
    typedef Mesh_                                      mesh_type;
    typedef typename mesh_type::sieve_type             sieve_type;
    typedef typename mesh_type::field_type::order_type sifter_type;
  private:
    #undef __FUNCT__
    #define __FUNCT__ "partition_Simple"
    static void partition_Simple(Obj<sieve_type> oldSieve, Obj<sieve_type> newSieve) {
      typedef typename sieve_type::traits::target_type point_type;
      int numLeaves = oldSieve->leaves()->size();

      ALE_LOG_EVENT_BEGIN;
      if (oldSieve->commRank() == 0) {
        int size = oldSieve->commSize();

        for(int p = 0; p < size; p++) {
          point_type partitionPoint(-1, p);

          for(int l = (numLeaves/size)*p + PetscMin(numLeaves%size, p); l < (numLeaves/size)*(p+1) + PetscMin(numLeaves%size, p+1); l++) {
            oldSieve->addCone(oldSieve->closure(point_type(0, l)), partitionPoint);
          }
        }
      }
      point_type partitionPoint(-1, newSieve->commRank());

      newSieve->addBasePoint(partitionPoint);
      if (oldSieve->debug) {
        oldSieve->view("Partition of old sieve");
        newSieve->view("Partition of new sieve");
      }
      ALE_LOG_EVENT_END;
    };
    #undef __FUNCT__
    #define __FUNCT__ "partition_Simple"
    static void partition_Simple(Obj<sieve_type> oldSieve, Obj<sifter_type> oldSifter, Obj<sifter_type> newSifter) {
      typedef typename sifter_type::traits::target_type point_type;
      Obj<typename sifter_type::traits::capSequence> cap = oldSifter->cap();
      int numLeaves = oldSieve->leaves()->size();

      ALE_LOG_EVENT_BEGIN;
      if (oldSifter->commRank() == 0) {
        int size = oldSifter->commSize();

        for(int p = 0; p < size; p++) {
          point_type partitionPoint(-1, p);

          for(int l = (numLeaves/size)*p + PetscMin(numLeaves%size, p); l < (numLeaves/size)*(p+1) + PetscMin(numLeaves%size, p+1); l++) {
            Obj<typename sieve_type::coneSet> closure = oldSieve->closure(point_type(0, l));

            for(typename sieve_type::coneSet::iterator c_iter = closure->begin(); c_iter != closure->end(); ++c_iter) {
              if (cap->contains(*c_iter)) {
                oldSifter->addCone(*c_iter, partitionPoint);
              }
            }
          }
        }
      }
      point_type partitionPoint(-1, newSifter->commRank());

      newSifter->addBasePoint(partitionPoint);
      if (oldSifter->debug) {
        oldSifter->view("Partition of old sifter");
        newSifter->view("Partition of new sifter");
      }
      ALE_LOG_EVENT_END;
    };
#ifdef PETSC_HAVE_CHACO
    static void partition_Chaco(Obj<mesh_type> oldMesh, Obj<sieve_type> oldSieve, const Obj<sieve_type> newSieve) {
      ALE_LOG_EVENT_BEGIN;
      typename mesh_type::patch_type patch;
      PetscErrorCode ierr;
      int size = oldSieve->commSize();
      int *offsets;


      Obj<typename sieve_type::traits::heightSequence> faces = oldSieve->heightStratum(1);
      Obj<typename sieve_type::traits::heightSequence> elements = oldSieve->heightStratum(0);
      Obj<typename mesh_type::bundle_type> vertexBundle = oldMesh->getBundle(0);
      Obj<typename mesh_type::bundle_type> elementBundle = oldMesh->getBundle(oldSieve->depth());
      if (oldSieve->commRank() == 0) {
        /* arguments for Chaco library */
        FREE_GRAPH = 0;                         /* Do not let Chaco free my memory */
        int nvtxs;                              /* number of vertices in full graph */
        int *start;                             /* start of edge list for each vertex */
        int *adjacency;                         /* = adj -> j; edge list data  */
        int *vwgts = NULL;                      /* weights for all vertices */
        float *ewgts = NULL;                    /* weights for all edges */
        float *x = NULL, *y = NULL, *z = NULL;  /* coordinates for inertial method */
        char *outassignname = NULL;             /*  name of assignment output file */
        char *outfilename = NULL;               /* output file name */
        short *assignment;                      /* set number of each vtx (length n) */
        int architecture = 1;                   /* 0 => hypercube, d => d-dimensional mesh */
        int ndims_tot = 0;                      /* total number of cube dimensions to divide */
        int mesh_dims[3];                       /* dimensions of mesh of processors */
        double *goal = NULL;                    /* desired set sizes for each set */
        int global_method = 1;                  /* global partitioning algorithm */
        int local_method = 1;                   /* local partitioning algorithm */
        int rqi_flag = 0;                       /* should I use RQI/Symmlq eigensolver? */
        int vmax = 200;                         /* how many vertices to coarsen down to? */
        int ndims = 1;                          /* number of eigenvectors (2^d sets) */
        double eigtol = 0.001;                  /* tolerance on eigenvectors */
        long seed = 123636512;                  /* for random graph mutations */

        nvtxs = oldSieve->heightStratum(0)->size();
        start = new int[nvtxs+1];
        offsets = new int[nvtxs];
        mesh_dims[0] = size; mesh_dims[1] = 1; mesh_dims[2] = 1;
        ierr = PetscMemzero(start, (nvtxs+1) * sizeof(int));CHKERROR(ierr, "Error in PetscMemzero");
        for(typename sieve_type::traits::heightSequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
          Obj<typename sieve_type::supportSequence> cells = oldSieve->support(*f_iter);

          if (cells->size() == 2) {
            start[elementBundle->getIndex(patch, *cells->begin()).prefix+1]++;
            start[elementBundle->getIndex(patch, *(++cells->begin())).prefix+1]++;
          }
        }
        for(int v = 1; v <= nvtxs; v++) {
          offsets[v-1] = start[v-1];
          start[v]    += start[v-1];
        }
        adjacency = new int[start[nvtxs]];
        for(typename sieve_type::traits::heightSequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
          Obj<typename sieve_type::supportSequence> cells = oldSieve->support(*f_iter);

          if (cells->size() == 2) {
            int cellA = elementBundle->getIndex(patch, *cells->begin()).prefix;
            int cellB = elementBundle->getIndex(patch, *(++cells->begin())).prefix;

            adjacency[offsets[cellA]++] = cellB+1;
            adjacency[offsets[cellB]++] = cellA+1;
          }
        }

        assignment = new short int[nvtxs];
        ierr = PetscMemzero(assignment, nvtxs * sizeof(short));CHKERROR(ierr, "Error in PetscMemzero");

        /* redirect output to buffer: chaco -> msgLog */
#ifdef PETSC_HAVE_UNISTD_H
        char *msgLog;
        int fd_stdout, fd_pipe[2], count;

        fd_stdout = dup(1);
        pipe(fd_pipe);
        close(1);
        dup2(fd_pipe[1], 1);
        msgLog = new char[16284];
#endif

        ierr = interface(nvtxs, start, adjacency, vwgts, ewgts, x, y, z,
                         outassignname, outfilename, assignment, architecture, ndims_tot,
                         mesh_dims, goal, global_method, local_method, rqi_flag, vmax, ndims,
                         eigtol, seed);

#ifdef PETSC_HAVE_UNISTD_H
        int SIZE_LOG  = 10000;

        fflush(stdout);
        count = read(fd_pipe[0], msgLog, (SIZE_LOG - 1) * sizeof(char));
        if (count < 0) count = 0;
        msgLog[count] = 0;
        close(1);
        dup2(fd_stdout, 1);
        close(fd_stdout);
        close(fd_pipe[0]);
        close(fd_pipe[1]);
        std::cout << msgLog << std::endl;
        delete [] msgLog;
#endif
        for(typename sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          if ((*e_iter).prefix >= 0) {
            oldSieve->addCone(oldSieve->closure(*e_iter), typename mesh_type::point_type(-1, assignment[elementBundle->getIndex(patch, *e_iter).prefix]));
          }
        }

        delete [] assignment;
        delete [] adjacency;
        delete [] start;
        delete [] offsets;
      }
      typename mesh_type::point_type partitionPoint(-1, newSieve->commRank());

      newSieve->addBasePoint(partitionPoint);
      if (oldSieve->debug) {
        oldSieve->view("Partition of old sieve");
        newSieve->view("Partition of new sieve");
      }
      ALE_LOG_EVENT_END;
    };
#endif
  public:
    static void partition(const Obj<mesh_type> serialMesh, const Obj<mesh_type> parallelMesh) {
      Obj<sieve_type> serialTopology = serialMesh->getTopology();
      Obj<sieve_type> parallelTopology = parallelMesh->getTopology();
      Obj<typename mesh_type::field_type> serialBoundary = serialMesh->getBoundary();
      Obj<typename mesh_type::field_type> parallelBoundary = parallelMesh->getBoundary();
      bool hasBd = (serialBoundary->getPatches()->size() > 0);

#ifdef PETSC_HAVE_CHACO
      int dim = serialMesh->getDimension();

      if (dim == 2) {
        partition_Chaco(serialMesh, serialTopology, parallelTopology);
      } else {
        partition_Simple(serialTopology, parallelTopology);
      }
#else
      partition_Simple(serialTopology, parallelTopology);
#endif
      if (hasBd) {
        partition_Simple(serialTopology, serialBoundary->__getOrder(), parallelBoundary->__getOrder());
      }
      Distributer::distribute(serialTopology, parallelTopology);
      if (hasBd) {
        Distributer::distribute(serialBoundary->__getOrder(), parallelBoundary->__getOrder(), false);
      }
    };
    static void unify(const Obj<mesh_type> parallelMesh, const Obj<mesh_type> serialMesh) {
      Obj<sieve_type>                parallelTopology = parallelMesh->getTopology();
      Obj<sieve_type>                serialTopology = serialMesh->getTopology();
      typename mesh_type::point_type partitionPoint(-1, 0);

      parallelTopology->addCone(parallelTopology->space(), partitionPoint);
      if (serialTopology->commRank == 0) {
        serialTopology->addBasePoint(partitionPoint);
      }
      Distributer::distribute(parallelTopology, serialTopology);
    };
  };
}
#endif
