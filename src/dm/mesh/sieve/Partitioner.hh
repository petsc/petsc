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
  template<typename Sifter_>
  class Distributer {
  public:
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
    typedef typename supportDelta_type::bioverlap_type overlap_type;
  public:
    #undef __FUNCT__
    #define __FUNCT__ "Part::distribute"
    static Obj<overlap_type> distribute(Obj<sifter_type> oldSifter, Obj<sifter_type> newSifter, bool restrict = true) {
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
      return overlap2;
    };
    #undef __FUNCT__
    #define __FUNCT__ "createMappingStoP"
    // ERROR: This crap only works for a single patch
    template<typename FieldType, typename OverlapType>
    static VecScatter createMappingStoP(Obj<FieldType> serialSifter, Obj<FieldType> parallelSifter, Obj<OverlapType> overlap, bool doExchange = false) {
      VecScatter scatter;
      Obj<typename OverlapType::traits::baseSequence> neighbors = overlap->base();
      MPI_Comm comm = serialSifter->comm();
      int      rank = serialSifter->commRank();
      int      debug = serialSifter->debug;
      Vec      serialVec, parallelVec;
      PetscErrorCode ierr;

      if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "Creating mapping\n");}
      // Use an MPI vector for the serial data since it has no overlap
      if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "  Creating serial indices\n");}
      if (serialSifter->debug) {
        serialSifter->view("SerialSifter");
        overlap->view("Partition Overlap");
      }
      // We may restrict to the first patch, since they are allocated in order
      Obj<typename FieldType::order_type::baseSequence> serialPatches = serialSifter->getPatches();
      Obj<typename FieldType::order_type::baseSequence> parallelPatches = parallelSifter->getPatches();
      
      int *serialOffsets = new int[serialPatches->size()+1];
      int serialSize = 0;
      int k = 0;
      serialOffsets[0] = 0;
      for(typename FieldType::order_type::baseSequence::iterator p_iter = serialPatches->begin(); p_iter != serialPatches->end(); ++p_iter) {
        serialSize += serialSifter->getSize(*p_iter);
        serialOffsets[++k] = serialSize;
      }
      ierr = VecCreateMPIWithArray(serialSifter->comm(), serialSize, PETSC_DETERMINE, serialSifter->restrict(*serialPatches->begin()), &serialVec);CHKERROR(ierr, "Error in VecCreate");
      // Use individual serial vectors for each of the parallel domains
      if (serialSifter->debug && !serialSifter->commRank()) {PetscSynchronizedPrintf(serialSifter->comm(), "  Creating parallel indices\n");}
      int *parallelOffsets = new int[parallelPatches->size()+1];
      int parallelSize = 0;
      k = 0;
      parallelOffsets[0] = 0;
      for(typename FieldType::order_type::baseSequence::iterator p_iter = parallelPatches->begin(); p_iter != parallelPatches->end(); ++p_iter) {
        parallelSize += parallelSifter->getSize(*p_iter);
        parallelOffsets[++k] = parallelSize;
      }
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, parallelSize, parallelSifter->restrict(*parallelPatches->begin()), &parallelVec);CHKERROR(ierr, "Error in VecCreate");

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
            for(typename FieldType::order_type::baseSequence::iterator sp_iter = serialPatches->begin(); sp_iter != serialPatches->end(); ++sp_iter) {
              // Assume the same index sizes
              int idxSize = serialSifter->getIndex(*sp_iter, (*p_iter).second).index;

              BuySizesA[nA] += idxSize;
              SellSizesA[nA] += idxSize;
              offsetA += idxSize;
              foundA = 1;
            }
          } else {
            for(typename FieldType::order_type::baseSequence::iterator pp_iter = parallelPatches->begin(); pp_iter != parallelPatches->end(); ++pp_iter) {
              // Assume the same index sizes
              int idxSize = parallelSifter->getIndex(*pp_iter, (*p_iter).second).index;

              BuySizesB[nB] += idxSize;
              SellSizesB[nB] += idxSize;
              offsetB += idxSize;
              foundB = 1;
            }
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
            int patchNum = 0;
            for(typename FieldType::order_type::baseSequence::iterator sp_iter = serialPatches->begin(); sp_iter != serialPatches->end(); ++sp_iter) {
              const typename FieldType::index_type& idx = serialSifter->getIndex(*sp_iter, p);

              if (debug) {
                ostringstream txt;

                txt << "["<<rank<<"]Packing A patch " << *sp_iter << " index " << idx << "(" << serialOffsets[patchNum] << ") for " << *neighbor << std::endl;
                ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
              }
              for(int i = serialOffsets[patchNum]+idx.prefix; i < serialOffsets[patchNum]+idx.prefix+idx.index; ++i) {
                SellConesA[offsetA++] = i;
              }
              patchNum++;
            }
          } else {
            int patchNum = 0;
            for(typename FieldType::order_type::baseSequence::iterator pp_iter = parallelPatches->begin(); pp_iter != parallelPatches->end(); ++pp_iter) {
              const typename FieldType::index_type& idx = parallelSifter->getIndex(*pp_iter, p);

              if (debug) {
                ostringstream txt;

                txt << "["<<rank<<"]Packing B patch " << *pp_iter << " index " << idx << "(" << parallelOffsets[patchNum] << ") for " << *neighbor << std::endl;
                ierr = PetscSynchronizedPrintf(comm, txt.str().c_str()); CHKERROR(ierr, "Error in PetscSynchronizedPrintf");
              }
              for(int i = parallelOffsets[patchNum]+idx.prefix; i < parallelOffsets[patchNum]+idx.prefix+idx.index; ++i) {
                SellConesB[offsetB++] = i;
              }
              patchNum++;
            }
          }
        }
      }
      if (debug) {
        ierr = PetscSynchronizedFlush(comm);CHKERROR(ierr,"Error in PetscSynchronizedFlush");
      }

      ierr = VecScatterCreateEmpty(comm, &scatter);CHKERROR(ierr, "Error in VecScatterCreate");
      scatter->from_n = serialSize;
      scatter->to_n = parallelSize;
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
    #define __FUNCT__ "partition_Induced"
    static void partition_Induced(short assignment[], Obj<mesh_type> oldMesh, Obj<sieve_type> oldSieve, Obj<sieve_type> newSieve) {
      Obj<typename sieve_type::traits::heightSequence> elements = oldSieve->heightStratum(0);
      Obj<typename mesh_type::bundle_type> elementBundle = oldMesh->getBundle(oldSieve->depth());
      typename mesh_type::patch_type patch;

      for(typename sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        if ((*e_iter).prefix >= 0) {
          oldSieve->addCone(oldSieve->closure(*e_iter), typename mesh_type::point_type(-1, assignment[elementBundle->getIndex(patch, *e_iter).prefix]));
        }
      }
      typename mesh_type::point_type partitionPoint(-1, newSieve->commRank());

      newSieve->addBasePoint(partitionPoint);
      if (oldSieve->debug) {
        oldSieve->view("Partition of old sieve");
        newSieve->view("Partition of new sieve");
      }
    };
    #undef __FUNCT__
    #define __FUNCT__ "partition_Induced"
    static void partition_Induced(short assignment[], Obj<mesh_type> oldMesh, Obj<sifter_type> oldSifter, Obj<sifter_type> newSifter) {
      Obj<typename mesh_type::sieve_type> oldSieve = oldMesh->getTopology();
      Obj<typename sieve_type::traits::heightSequence> elements = oldSieve->heightStratum(0);
      Obj<typename mesh_type::bundle_type> elementBundle = oldMesh->getBundle(oldSieve->depth());
      Obj<typename sifter_type::traits::capSequence> cap = oldSifter->cap();
      typename mesh_type::patch_type patch;

      for(typename sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        if ((*e_iter).prefix >= 0) {
          Obj<typename sieve_type::coneSet> closure = oldSieve->closure(*e_iter);
          typename mesh_type::point_type partitionPoint(-1, assignment[elementBundle->getIndex(patch, *e_iter).prefix]);

          for(typename sieve_type::coneSet::iterator c_iter = closure->begin(); c_iter != closure->end(); ++c_iter) {
            if (cap->contains(*c_iter)) {
              oldSifter->addCone(*c_iter, partitionPoint);
            }
          }
        }
      }
      typename mesh_type::point_type partitionPoint(-1, newSifter->commRank());

      newSifter->addBasePoint(partitionPoint);
      if (oldSifter->debug) {
        oldSifter->view("Partition of old sifter");
        newSifter->view("Partition of new sifter");
      }
    };
    #undef __FUNCT__
    #define __FUNCT__ "partition_Simple"
    static short *partition_Simple(Obj<mesh_type> oldMesh, Obj<sieve_type> oldSieve, Obj<sieve_type> newSieve) {
      typedef typename sieve_type::traits::target_type point_type;
      Obj<typename mesh_type::bundle_type> elementBundle = oldMesh->getBundle(oldSieve->depth());
      typename mesh_type::patch_type patch;
      short *assignment = NULL;

      ALE_LOG_EVENT_BEGIN;
      if (oldSieve->commRank() == 0) {
        int numLeaves = oldSieve->leaves()->size();
        int size = oldSieve->commSize();

        assignment = new short[numLeaves];
        for(int p = 0; p < size; p++) {
          for(int l = (numLeaves/size)*p + PetscMin(numLeaves%size, p); l < (numLeaves/size)*(p+1) + PetscMin(numLeaves%size, p+1); l++) {
            assignment[elementBundle->getIndex(patch, point_type(0, l)).prefix] = p;
          }
        }
      }
      partition_Induced(assignment, oldMesh, oldSieve, newSieve);
      ALE_LOG_EVENT_END;
      return assignment;
    };
#ifdef PETSC_HAVE_CHACO
    static short *partition_Chaco(Obj<mesh_type> oldMesh, Obj<sieve_type> oldSieve, const Obj<sieve_type> newSieve) {
      ALE_LOG_EVENT_BEGIN;
      typename mesh_type::patch_type patch;
      PetscErrorCode ierr;
      int size = oldSieve->commSize();
      short *assignment = NULL; /* set number of each vtx (length n) */
      int *offsets = NULL;


      Obj<typename sieve_type::traits::heightSequence> faces = oldSieve->heightStratum(1);
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
        delete [] adjacency;
        delete [] start;
        delete [] offsets;
      }

      partition_Induced(assignment, oldMesh, oldSieve, newSieve);
      ALE_LOG_EVENT_END;
      return assignment;
    };
#endif
  public:
    static void partition(const Obj<mesh_type> serialMesh, const Obj<mesh_type> parallelMesh) {
      typedef typename mesh_type::field_type::order_type order_type;
      Obj<sieve_type> serialTopology = serialMesh->getTopology();
      Obj<sieve_type> parallelTopology = parallelMesh->getTopology();
      Obj<typename mesh_type::field_type> serialBoundary = serialMesh->getBoundary();
      Obj<typename mesh_type::field_type> parallelBoundary = parallelMesh->getBoundary();
      short *assignment = NULL;
      bool useSimple = true;
      //bool hasBd = (serialBoundary->getPatches()->size() > 0);
      bool hasBd = false;

      parallelTopology->setStratification(false);
#ifdef PETSC_HAVE_CHACO
      if (serialMesh->getDimension() > 1) {
        assignment = partition_Chaco(serialMesh, serialTopology, parallelTopology);
        useSimple = false;
      }
#endif
      if (useSimple) {
        assignment = partition_Simple(serialMesh, serialTopology, parallelTopology);
      }
      if (hasBd) {
        partition_Induced(assignment, serialMesh, serialBoundary->__getOrder(), parallelBoundary->__getOrder());
      }
      Obj<std::set<std::string> > fieldNames = serialMesh->getFields();

      for(typename std::set<std::string>::iterator f_iter = fieldNames->begin(); f_iter != fieldNames->end(); ++f_iter) {
        partition_Induced(assignment, serialMesh, serialMesh->getField(*f_iter)->__getOrder(), parallelMesh->getField(*f_iter)->__getOrder());
      }
      delete [] assignment;

      Obj<typename Distributer<sieve_type>::overlap_type> partitionOverlap = Distributer<sieve_type>::distribute(serialTopology, parallelTopology);
      parallelTopology->stratify();
      parallelTopology->setStratification(true);

      if (hasBd) {
        Distributer<order_type>::distribute(serialBoundary->__getOrder(), parallelBoundary->__getOrder(), false);
        parallelBoundary->reorderPatches();
        parallelBoundary->allocatePatches();
        parallelBoundary->createGlobalOrder();

        VecScatter scatter = Distributer<order_type>::createMappingStoP(serialBoundary, parallelBoundary, partitionOverlap, true);
        PetscErrorCode ierr = VecScatterDestroy(scatter);CHKERROR(ierr, "Error in VecScatterDestroy");
      }
      for(typename std::set<std::string>::iterator f_iter = fieldNames->begin(); f_iter != fieldNames->end(); ++f_iter) {
        Obj<typename mesh_type::field_type> serialField   = serialMesh->getField(*f_iter);
        Obj<typename mesh_type::field_type> parallelField = parallelMesh->getField(*f_iter);

        Distributer<order_type>::distribute(serialField->__getOrder(), parallelField->__getOrder(), false);
        std::string msg = "Distributed A field ";
        msg += *f_iter;
        parallelField->view(msg.c_str());
        parallelField->reorderPatches();
        msg = "Distributed B field ";
        msg += *f_iter;
        parallelField->view(msg.c_str());
        parallelField->allocatePatches();
        msg = "Distributed C field ";
        msg += *f_iter;
        parallelField->view(msg.c_str());
        parallelField->createGlobalOrder();

        serialField->debug = 1;
        VecScatter scatter = Distributer<order_type>::createMappingStoP(serialField, parallelField, partitionOverlap, true);
        PetscErrorCode ierr = VecScatterDestroy(scatter);CHKERROR(ierr, "Error in VecScatterDestroy");
        serialField->debug = 0;
        msg = "Parallel field ";
        msg += *f_iter;
        parallelField->view(msg.c_str());
      }
    };
    static void unify(const Obj<mesh_type> parallelMesh, const Obj<mesh_type> serialMesh) {
      typedef typename mesh_type::field_type::order_type order_type;
      Obj<sieve_type>                     parallelTopology = parallelMesh->getTopology();
      Obj<sieve_type>                     serialTopology = serialMesh->getTopology();
      Obj<typename mesh_type::field_type> parallelBoundary = parallelMesh->getBoundary();
      Obj<typename mesh_type::field_type> serialBoundary = serialMesh->getBoundary();
      bool                                hasBd = (parallelBoundary->getPatches()->size() > 0);
      typename mesh_type::point_type      partitionPoint(-1, 0);

      parallelTopology->addCone(parallelTopology->cap(), partitionPoint);
      parallelTopology->addCone(parallelTopology->base(), partitionPoint);
      parallelTopology->removeCapPoint(partitionPoint);
      parallelBoundary->__getOrder()->addCone(parallelBoundary->__getOrder()->cap(), partitionPoint);
      if (serialTopology->commRank() == 0) {
        serialTopology->addBasePoint(partitionPoint);
        serialBoundary->__getOrder()->addBasePoint(partitionPoint);
      }
      Distributer<sieve_type>::distribute(parallelTopology, serialTopology);
      if (hasBd) {
        Distributer<order_type>::distribute(parallelBoundary->__getOrder(), serialBoundary->__getOrder(), false);
        serialBoundary->allocatePatches();
      }
    };
  };
}
#endif
