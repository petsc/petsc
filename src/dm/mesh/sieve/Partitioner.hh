#ifndef included_ALE_Partitioner_hh
#define included_ALE_Partitioner_hh

#ifndef  included_ALE_Numbering_hh
#include <Numbering.hh>
#endif

#ifdef PETSC_HAVE_CHACO
/* Chaco does not have an include file */
extern "C" {
  extern int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                       float *ewgts, float *x, float *y, float *z, char *outassignname,
                       char *outfilename, short *assignment, int architecture, int ndims_tot,
                       int mesh_dims[3], double *goal, int global_method, int local_method,
                       int rqi_flag, int vmax, int ndims, double eigtol, long seed);

  extern int FREE_GRAPH;
}
#endif
#ifdef PETSC_HAVE_CHACO
extern "C" {
  extern void METIS_PartGraphKway(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
}
#endif

namespace ALE {
  namespace New {
    template<typename Topology_>
    class Partitioner {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::point_type point_type;
    public:
      #undef __FUNCT__
      #define __FUNCT__ "buildDualCSR"
      // This creates a CSR representation of the adjacency matrix for cells
      static void buildDualCSR(const Obj<topology_type>& topology, const int dim, const patch_type& patch, int **offsets, int **adjacency) {
        ALE_LOG_EVENT_BEGIN;
        const Obj<sieve_type>&                             sieve    = topology->getPatch(patch);
        const Obj<typename topology_type::label_sequence>& elements = topology->heightStratum(patch, 0);
        int numElements = elements->size();
        int corners     = sieve->cone(*elements->begin())->size();
        int *off        = new int[numElements+1];

        std::set<point_type> *neighborCells = new std::set<point_type>[numElements];
        int faceVertices = -1;

        if (topology->depth(patch) != 1) {
          throw ALE::Exception("Not yet implemented for interpolated meshes");
        }
        if (corners == dim+1) {
          faceVertices = dim;
        } else if ((dim == 2) && (corners == 4)) {
          faceVertices = 2;
        } else if ((dim == 3) && (corners == 8)) {
          faceVertices = 4;
        } else {
          throw ALE::Exception("Could not determine number of face vertices");
        }
        for(typename topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          const Obj<typename sieve_type::traits::coneSequence>& vertices  = sieve->cone(*e_iter);
          typename sieve_type::traits::coneSequence::iterator vEnd = vertices->end();

          for(typename sieve_type::traits::coneSequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
            const Obj<typename sieve_type::traits::supportSequence>& neighbors = sieve->support(*v_iter);
            typename sieve_type::traits::supportSequence::iterator nEnd = neighbors->end();

            for(typename sieve_type::traits::supportSequence::iterator n_iter = neighbors->begin(); n_iter != nEnd; ++n_iter) {
              if (*e_iter == *n_iter) continue;
              if ((int) sieve->meet(*e_iter, *n_iter)->size() == faceVertices) {
                neighborCells[*e_iter].insert(*n_iter);
              }
            }
          }
        }
        off[0] = 0;
        for(int e = 1; e <= numElements; e++) {
          off[e] = neighborCells[e-1].size() + off[e-1];
        }
        int *adj    = new int[off[numElements]];
        int  offset = 0;
        for(int e = 0; e < numElements; e++) {
          for(typename std::set<point_type>::iterator n_iter = neighborCells[e].begin(); n_iter != neighborCells[e].end(); ++n_iter) {
            adj[offset++] = *n_iter;
          }
        }
        delete [] neighborCells;
        if (offset != off[numElements]) {
          ostringstream msg;
          msg << "ERROR: Total number of neighbors " << offset << " does not match the offset array " << off[numElements];
          throw ALE::Exception(msg.str().c_str());
        }
        *offsets   = off;
        *adjacency = adj;
        ALE_LOG_EVENT_END;
      };
      template<typename PartitionType>
      static PartitionType *subordinatePartition(const Obj<topology_type>& topology, const Obj<topology_type>& subTopology, const PartitionType assignment[]) {
        typedef ALE::New::NumberingFactory<topology_type> NumberingFactory;
        const patch_type patch = 0;
        const Obj<typename topology_type::label_sequence>&    cells      = subTopology->heightStratum(patch, 0);
        const Obj<typename NumberingFactory::numbering_type>& cNumbering = NumberingFactory::singleton(topology->debug())->getLocalNumbering(topology, patch, topology->depth(patch));
        const int        numCells      = cells->size();
        PartitionType   *subAssignment = new PartitionType[numCells];
        int              c = 0;

        if (topology->depth(patch) == subTopology->depth(patch)) {
          for(typename topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
            subAssignment[c++] = assignment[cNumbering->getIndex(*c_iter)];
          }
        } else {
          const Obj<typename topology_type::sieve_type>& sieve = topology->getPatch(patch);

          for(typename topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
            const Obj<typename topology_type::sieve_type::supportSequence>& support = sieve->cone();

            if (support->size != 1) {
              throw ALE::Exception("Indeterminate subordinate partition");
            }
            subAssignment[c++] = assignment[cNumbering->getIndex(*support->begin())];
          }
        }
        return subAssignment;
      };
    };
#ifdef PETSC_HAVE_CHACO
    namespace Chaco {
      template<typename Topology_>
      class Partitioner {
      public:
        typedef Topology_                          topology_type;
        typedef typename topology_type::sieve_type sieve_type;
        typedef typename topology_type::patch_type patch_type;
        typedef typename topology_type::point_type point_type;
        typedef short int                          part_type;
      public:
        #undef __FUNCT__
        #define __FUNCT__ "ChacoPartitionSieve"
        static part_type *partitionSieve(const Obj<topology_type>& topology, const int dim) {
          part_type *assignment = NULL; /* set number of each vtx (length n) */

          ALE_LOG_EVENT_BEGIN;
          if (topology->commRank() == 0) {
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
            int patch = 0;
            PetscErrorCode ierr;

            nvtxs = topology->heightStratum(patch, 0)->size();
            mesh_dims[0] = topology->commSize(); mesh_dims[1] = 1; mesh_dims[2] = 1;
            ALE::New::Partitioner<topology_type>::buildDualCSR(topology, dim, patch, &start, &adjacency);
            for(int e = 0; e < start[nvtxs]; e++) {
              adjacency[e]++;
            }
            assignment = new part_type[nvtxs];
            ierr = PetscMemzero(assignment, nvtxs * sizeof(part_type));CHKERROR(ierr, "Error in PetscMemzero");

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
            if (topology->debug()) {
              std::cout << msgLog << std::endl;
            }
            delete [] msgLog;
#endif
            delete [] adjacency;
            delete [] start;
          }
          ALE_LOG_EVENT_END;
          return assignment;
        };
      };
    };
#endif
#ifdef PETSC_HAVE_PARMETIS
    namespace ParMetis {
      template<typename Topology_>
      class Partitioner {
      public:
        typedef Topology_                          topology_type;
        typedef typename topology_type::sieve_type sieve_type;
        typedef typename topology_type::patch_type patch_type;
        typedef typename topology_type::point_type point_type;
        typedef int                                part_type;
      public:
        #undef __FUNCT__
        #define __FUNCT__ "ParMetisPartitionSieve"
        static part_type *partitionSieve(const Obj<topology_type>& topology, const int dim) {
          int    nvtxs;      // The number of vertices in full graph
          int   *xadj;       // Start of edge list for each vertex
          int   *adjncy;     // Edge lists for all vertices
          int   *vwgt;       // Vertex weights
          int   *adjwgt;     // Edge weights
          int    wgtflag;    // Indicates which weights are present
          int    numflag;    // Indicates initial offset (0 or 1)
          int    nparts;     // The number of partitions
          int    options[5]; // Options
          // Outputs
          int    edgeCut;    // The number of edges cut by the partition
          int   *assignment; // The vertex partition
          const typename topology_type::patch_type patch = 0;

          if (topology->commRank() == 0) {
            nvtxs = topology->heightStratum(patch, 0)->size();
            vwgt       = NULL;
            adjwgt     = NULL;
            wgtflag    = 0;
            numflag    = 0;
            nparts     = topology->commSize();
            options[0] = 0; // Use all defaults
            assignment = new part_type[nvtxs];
            if (topology->commSize() == 1) {
              PetscMemzero(assignment, nvtxs * sizeof(part_type));
            } else {
              ALE::New::Partitioner<topology_type>::buildDualCSR(topology, dim, patch, &xadj, &adjncy);
              METIS_PartGraphKway(&nvtxs, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &nparts, options, &edgeCut, assignment);
              delete [] xadj;
              delete [] adjncy;
            }
          } else {
            assignment = NULL;
          }
          return assignment;
        };
      };
    };
#endif
  }
}

#endif
