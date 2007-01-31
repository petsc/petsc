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
#ifdef PETSC_HAVE_PARMETIS
extern "C" {
  extern void METIS_PartGraphKway(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
}
#endif
#ifdef PETSC_HAVE_HMETIS
extern "C" {
  extern void HMETIS_PartKway(int nvtxs, int nhedges, int *vwgts, int *eptr, int *eind, int *hewgts, int nparts, int ubfactor, int *options, int *part, int *edgeCut);
}
#endif

#ifdef PETSC_HAVE_ZOLTAN
#include <zoltan.h>
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
        int  numElements = elements->size();
        int *off         = new int[numElements+1];
        int  offset      = 0;
        int *adj;

        if (topology->depth(patch) == dim) {
          int e = 1;

          off[0] = 0;
          for(typename topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
            const Obj<typename sieve_type::traits::coneSequence>& faces  = sieve->cone(*e_iter);
            typename sieve_type::traits::coneSequence::iterator   fBegin = faces->begin();
            typename sieve_type::traits::coneSequence::iterator   fEnd   = faces->end();

            off[e] = off[e-1];
            for(typename sieve_type::traits::coneSequence::iterator f_iter = fBegin; f_iter != fEnd; ++f_iter) {
              if (sieve->support(*f_iter)->size() == 2) off[e]++;
            }
            e++;
          }
          adj = new int[off[numElements]];
          for(typename topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
            const Obj<typename sieve_type::traits::coneSequence>& faces  = sieve->cone(*e_iter);
            typename sieve_type::traits::coneSequence::iterator   fBegin = faces->begin();
            typename sieve_type::traits::coneSequence::iterator   fEnd   = faces->end();

            for(typename sieve_type::traits::coneSequence::iterator f_iter = fBegin; f_iter != fEnd; ++f_iter) {
              const Obj<typename sieve_type::traits::supportSequence>& neighbors = sieve->support(*f_iter);
              typename sieve_type::traits::supportSequence::iterator   nBegin    = neighbors->begin();
              typename sieve_type::traits::supportSequence::iterator   nEnd      = neighbors->end();

              for(typename sieve_type::traits::supportSequence::iterator n_iter = nBegin; n_iter != nEnd; ++n_iter) {
                if (*n_iter != *e_iter) adj[offset++] = *n_iter;
              }
            }
          }
        } else if (topology->depth(patch) == 1) {
          std::set<point_type> *neighborCells = new std::set<point_type>[numElements];
          int corners      = sieve->cone(*elements->begin())->size();
          int faceVertices = -1;

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
          adj = new int[off[numElements]];
          for(int e = 0; e < numElements; e++) {
            for(typename std::set<point_type>::iterator n_iter = neighborCells[e].begin(); n_iter != neighborCells[e].end(); ++n_iter) {
              adj[offset++] = *n_iter;
            }
          }
          delete [] neighborCells;
        } else {
          throw ALE::Exception("Dual creation not defined for partially interpolated meshes");
        }
        if (offset != off[numElements]) {
          ostringstream msg;
          msg << "ERROR: Total number of neighbors " << offset << " does not match the offset array " << off[numElements];
          throw ALE::Exception(msg.str().c_str());
        }
        *offsets   = off;
        *adjacency = adj;
        ALE_LOG_EVENT_END;
      };
      #undef __FUNCT__
      #define __FUNCT__ "buildFaceCSR"
      // This creates a CSR representation of the adjacency hypergraph for faces
      static void buildFaceCSR(const Obj<topology_type>& topology, const int dim, const patch_type& patch, const Obj<ALE::Mesh::numbering_type>& fNumbering, int *numEdges, int **offsets, int **adjacency) {
        ALE_LOG_EVENT_BEGIN;
        const Obj<sieve_type>&                             sieve      = topology->getPatch(patch);
        const Obj<typename topology_type::label_sequence>& elements   = topology->heightStratum(patch, 0);
        int  numElements = elements->size();
        int *off         = new int[numElements+1];
        int  e;

        if (topology->depth(patch) != dim) {
          throw ALE::Exception("Not yet implemented for non-interpolated meshes");
        }
        off[0] = 0;
        e      = 1;
        for(typename topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          off[e] = sieve->cone(*e_iter)->size() + off[e-1];
          e++;
        }
        int *adj    = new int[off[numElements]];
        int  offset = 0;
        for(typename topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          const Obj<typename sieve_type::traits::coneSequence>& faces = sieve->cone(*e_iter);
          typename sieve_type::traits::coneSequence::iterator   fEnd  = faces->end();

          for(typename sieve_type::traits::coneSequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter) {
            adj[offset++] = fNumbering->getIndex(*f_iter);
          }
        }
        if (offset != off[numElements]) {
          ostringstream msg;
          msg << "ERROR: Total number of neighbors " << offset << " does not match the offset array " << off[numElements];
          throw ALE::Exception(msg.str().c_str());
        }
        *numEdges  = numElements;
        *offsets   = off;
        *adjacency = adj;
        ALE_LOG_EVENT_END;
      };
      template<typename PartitionType>
      static PartitionType *subordinatePartition(const Obj<topology_type>& topology, int levels, const Obj<topology_type>& subTopology, const PartitionType assignment[]) {
        typedef ALE::New::NumberingFactory<topology_type> NumberingFactory;
        const patch_type patch = 0;
        const Obj<typename NumberingFactory::numbering_type>& cNumbering = NumberingFactory::singleton(topology->debug())->getLocalNumbering(topology, patch, topology->depth(patch));
        const Obj<typename topology_type::label_sequence>&    cells      = subTopology->heightStratum(patch, 0);
        const Obj<typename NumberingFactory::numbering_type>& sNumbering = NumberingFactory::singleton(subTopology->debug())->getLocalNumbering(subTopology, patch, subTopology->depth(patch));
        const int        numCells      = cells->size();
        PartitionType   *subAssignment = new PartitionType[numCells];

        if (levels != 1) {
          throw ALE::Exception("Cannot calculate subordinate partition for any level separation other than 1");
        } else {
          const Obj<typename topology_type::sieve_type>& sieve = topology->getPatch(patch);

          for(typename topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
            const Obj<typename topology_type::sieve_type::supportSequence>& support = sieve->support(*c_iter);

            if (support->size() != 1) {
              // Could relax this to choosing the first one
              throw ALE::Exception("Indeterminate subordinate partition");
            }
            subAssignment[sNumbering->getIndex(*c_iter)] = assignment[cNumbering->getIndex(*support->begin())];
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
        #undef __FUNCT__
        #define __FUNCT__ "ParMetisPartitionSieveByFace"
        static part_type *partitionSieveByFace(const Obj<topology_type>& topology, const int dim) {
          int    nvtxs;      // The number of vertices
          int    nhedges;    // The number of hyperedges
          int   *vwgts;      // The vertex weights
          int   *eptr;       // The offsets of each hyperedge
          int   *eind;       // The vertices in each hyperedge, indexed by eptr
          int   *hewgts;     // The hyperedge weights
          int    nparts;     // The number of partitions
          int    ubfactor;   // The allowed load imbalance (1-50)
          int    options[9]; // Options
          // Outputs
          int    edgeCut;    // The number of edges cut by the partition
          int   *assignment; // The vertex partition
          const typename topology_type::patch_type patch = 0;
          const Obj<ALE::Mesh::numbering_type>& fNumbering = ALE::New::NumberingFactory<topology_type>::singleton(topology->debug())->getNumbering(topology, patch, topology->depth()-1);

          if (topology->commRank() == 0) {
            nvtxs      = topology->heightStratum(patch, 1)->size();
            vwgts      = NULL;
            hewgts     = NULL;
            nparts     = topology->commSize();
            ubfactor   = 5;
            options[0] = 0;  // Use all defaults
            options[1] = 10; // Number of bisections tested
            options[2] = 1;  // Vertex grouping scheme
            options[3] = 1;  // Objective function
            options[4] = 1;  // V-cycle refinement
            options[7] = -1; // Random seed
            options[8] = 24; // Debugging level
            assignment = new part_type[nvtxs];

            if (topology->commSize() == 1) {
              PetscMemzero(assignment, nvtxs * sizeof(part_type));
            } else {
              ALE::New::Partitioner<topology_type>::buildFaceCSR(topology, dim, patch, fNumbering, &nhedges, &eptr, &eind);
#ifdef PETSC_HAVE_HMETIS
              HMETIS_PartKway(nvtxs, nhedges, vwgts, eptr, eind, hewgts, nparts, ubfactor, options, assignment, &edgeCut);
#endif

              delete [] eptr;
              delete [] eind;
            }
          } else {
            assignment = NULL;
          }
          return assignment;
        };
      };
    };
#endif
#ifdef PETSC_HAVE_ZOLTAN
    // Inputs
    static int  nvtxs;   // The number of vertices
    static int  nhedges; // The number of hyperedges
    static int *eptr;    // The offsets of each hyperedge
    static int *eind;    // The vertices in each hyperedge, indexed by eptr

    extern "C" {
      int getNumVertices(void *data, int *ierr) {
        *ierr = 0;
        return nvtxs;
      };

      void getLocalElements(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts, int *ierr) {
        if ((wgt_dim != 0) || (num_gid_entries != 1) || (num_lid_entries != 1)) {
          *ierr = 1;
          return;
        }
        *ierr = 0;
        for(int v = 0; v < nvtxs; ++v) {
          global_ids[v]= v;
          local_ids[v] = v;
        }
        return;
      };

      void getHgSizes(void *data, int *num_lists, int *num_pins, int *format, int *ierr) {
        *ierr = 0;
        *num_lists = nhedges;
        *num_pins  = eptr[nhedges];
        *format    = ZOLTAN_COMPRESSED_EDGE;
      };

      void getHg(void *data, int num_gid_entries, int num_row_or_col, int num_pins, int format, ZOLTAN_ID_PTR vtxedge_GID, int *vtxedge_ptr, ZOLTAN_ID_PTR pin_GID, int *ierr) {
        if ((num_gid_entries != 1) || (num_row_or_col != nhedges) || (num_pins != eptr[nhedges]) || (format != ZOLTAN_COMPRESSED_EDGE)) {
          *ierr = 1;
          return;
        }
        *ierr = 0;
        for(int e = 0; e < num_row_or_col; ++e) {
          vtxedge_GID[e] = e;
        }
        for(int e = 0; e < num_row_or_col; ++e) {
          vtxedge_ptr[e] = eptr[e];
        }
        for(int p = 0; p < num_pins; ++p) {
          pin_GID[p] = eind[p];
        }
      };
    }

    namespace Zoltan {
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
        #define __FUNCT__ "ZoltanPartitionSieveByFace"
        static part_type *partitionSieveByFace(const Obj<topology_type>& topology, const int dim) {
          // Outputs
          float         version;           // The library version
          int           changed;           // Did the partition change?
          int           numGidEntries;     // Number of array entries for a single global ID (1)
          int           numLidEntries;     // Number of array entries for a single local ID (1)
          int           numImport;         // The number of imported points
          ZOLTAN_ID_PTR import_global_ids; // The imported points
          ZOLTAN_ID_PTR import_local_ids;  // The imported points
          int          *import_procs;      // The proc each point was imported from
          int          *import_to_part;    // The partition of each imported point
          int           numExport;         // The number of exported points
          ZOLTAN_ID_PTR export_global_ids; // The exported points
          ZOLTAN_ID_PTR export_local_ids;  // The exported points
          int          *export_procs;      // The proc each point was exported to
          int          *export_to_part;    // The partition assignment of all local points
          int          *assignment;        // The partition assignment of all local points

          const typename topology_type::patch_type patch = 0;
          const Obj<ALE::Mesh::numbering_type>& fNumbering = ALE::New::NumberingFactory<topology_type>::singleton(topology->debug())->getNumbering(topology, patch, topology->depth()-1);

          if (topology->commSize() == 1) {
            PetscMemzero(assignment, nvtxs * sizeof(part_type));
          } else {
            if (topology->commRank() == 0) {
              nvtxs      = topology->heightStratum(patch, 1)->size();
              ALE::New::Partitioner<topology_type>::buildFaceCSR(topology, dim, patch, fNumbering, &nhedges, &eptr, &eind);
              assignment = new int[nvtxs];
            } else {
              nvtxs      = topology->heightStratum(patch, 1)->size();
              nhedges    = 0;
              eptr       = new int[1];
              eind       = new int[1];
              eptr[0]    = 0;
              assignment = NULL;
            }

            int ierr = Zoltan_Initialize(0, NULL, &version);
            struct Zoltan_Struct *zz = Zoltan_Create(topology->comm());
            // General parameters
            Zoltan_Set_Param(zz, "DEBUG_LEVEL", "2");
            Zoltan_Set_Param(zz, "LB_METHOD", "PHG");
            Zoltan_Set_Param(zz, "RETURN_LISTS", "PARTITION");
            // PHG parameters
            Zoltan_Set_Param(zz, "PHG_OUTPUT_LEVEL", "2");
            // Call backs
            Zoltan_Set_Num_Obj_Fn(zz, getNumVertices, NULL);
            Zoltan_Set_Obj_List_Fn(zz, getLocalElements, NULL);
            Zoltan_Set_HG_Size_CS_Fn(zz, getHgSizes, NULL);
            Zoltan_Set_HG_CS_Fn(zz, getHg, NULL);

            ierr = Zoltan_LB_Partition(zz, &changed, &numGidEntries, &numLidEntries,
                                       &numImport, &import_global_ids, &import_local_ids, &import_procs, &import_to_part,
                                       &numExport, &export_global_ids, &export_local_ids, &export_procs, &export_to_part);
            for(int v = 0; v < nvtxs; ++v) {
              assignment[v] = export_to_part[v];
            }
            Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
            Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
            Zoltan_Destroy(&zz);

            delete [] eptr;
            delete [] eind;
          }
          return assignment;
        };
      };
    };
#endif
  }
}

#endif
