#ifndef included_ALE_Partitioner_hh
#define included_ALE_Partitioner_hh

#ifndef  included_ALE_Completion_hh
#include <Completion.hh>
#endif

#ifdef PETSC_HAVE_ZOLTAN
#include <zoltan.h>

extern "C" {
  // Inputs
  extern int  nvtxs_Zoltan;   // The number of vertices
  extern int  nhedges_Zoltan; // The number of hyperedges
  extern int *eptr_Zoltan;    // The offsets of each hyperedge
  extern int *eind_Zoltan;    // The vertices in each hyperedge, indexed by eptr

  int getNumVertices_Zoltan(void *, int *);

  void getLocalElements_Zoltan(void *, int, int, ZOLTAN_ID_PTR, ZOLTAN_ID_PTR, int, float *, int *);

  void getHgSizes_Zoltan(void *, int *, int *, int *, int *);

  void getHg_Zoltan(void *, int, int, int, int, ZOLTAN_ID_PTR, int *, ZOLTAN_ID_PTR, int *);
}

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
  #include <parmetis.h>
  extern void METIS_PartGraphKway(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
}
#endif
#ifdef PETSC_HAVE_HMETIS
extern "C" {
  extern void HMETIS_PartKway(int nvtxs, int nhedges, int *vwgts, int *eptr, int *eind, int *hewgts, int nparts, int ubfactor, int *options, int *part, int *edgeCut);
}
#endif

namespace ALE {
  namespace New {
    template<typename Bundle_, typename Alloc_ = typename Bundle_::alloc_type>
    class Partitioner {
    public:
      typedef Bundle_                          bundle_type;
      typedef Alloc_                           alloc_type;
      typedef typename bundle_type::sieve_type sieve_type;
      typedef typename bundle_type::point_type point_type;
    public:
      #undef __FUNCT__
      #define __FUNCT__ "buildDualCSR"
      // This creates a CSR representation of the adjacency matrix for cells
      // - We allow an exception to contiguous numbering.
      //   If the cell id > numElements, we assign a new number starting at
      //     the top and going downward. I know these might not match up with
      //     the iterator order, but we can fix it later.
      static void buildDualCSR(const Obj<bundle_type>& bundle, const int dim, int **offsets, int **adjacency) {
        ALE_LOG_EVENT_BEGIN;
        typedef typename ALE::New::Completion<bundle_type, point_type, alloc_type> completion;
        const Obj<sieve_type>&                           sieve        = bundle->getSieve();
        const Obj<typename bundle_type::label_sequence>& elements     = bundle->heightStratum(0);
        Obj<sieve_type>                                  overlapSieve = new sieve_type(bundle->comm(), bundle->debug());
        std::map<point_type, point_type>                 newCells;
        int  numElements = elements->size();
        int  newCell     = numElements;
        int *off         = new int[numElements+1];
        int  offset      = 0;
        int *adj;

        completion::scatterSupports(sieve, overlapSieve, bundle->getSendOverlap(), bundle->getRecvOverlap(), bundle);
        if (numElements == 0) {
          *offsets   = NULL;
          *adjacency = NULL;
          ALE_LOG_EVENT_END;
          return;
        }
        if (bundle->depth() == dim) {
          int e = 1;

          off[0] = 0;
          for(typename bundle_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
            const Obj<typename sieve_type::traits::coneSequence>& faces  = sieve->cone(*e_iter);
            typename sieve_type::traits::coneSequence::iterator   fBegin = faces->begin();
            typename sieve_type::traits::coneSequence::iterator   fEnd   = faces->end();

            off[e] = off[e-1];
            for(typename sieve_type::traits::coneSequence::iterator f_iter = fBegin; f_iter != fEnd; ++f_iter) {
              if (sieve->support(*f_iter)->size() == 2) {
                off[e]++;
              } else if ((sieve->support(*f_iter)->size() == 1) && (overlapSieve->support(*f_iter)->size() == 1)) {
                off[e]++;
              }
            }
            e++;
          }
          adj = new int[off[numElements]];
          for(typename bundle_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
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
              const Obj<typename sieve_type::traits::supportSequence>& oNeighbors = overlapSieve->support(*f_iter);
              typename sieve_type::traits::supportSequence::iterator   onBegin    = oNeighbors->begin();
              typename sieve_type::traits::supportSequence::iterator   onEnd      = oNeighbors->end();

              for(typename sieve_type::traits::supportSequence::iterator n_iter = onBegin; n_iter != onEnd; ++n_iter) {
                adj[offset++] = *n_iter;
              }
            }
          }
        } else if (bundle->depth() == 1) {
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
          for(typename bundle_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
            const Obj<typename sieve_type::traits::coneSequence>& vertices  = sieve->cone(*e_iter);
            typename sieve_type::traits::coneSequence::iterator vEnd = vertices->end();

            for(typename sieve_type::traits::coneSequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
              const Obj<typename sieve_type::traits::supportSequence>& neighbors = sieve->support(*v_iter);
              typename sieve_type::traits::supportSequence::iterator nEnd = neighbors->end();

              for(typename sieve_type::traits::supportSequence::iterator n_iter = neighbors->begin(); n_iter != nEnd; ++n_iter) {
                if (*e_iter == *n_iter) continue;
                if ((int) sieve->nMeet(*e_iter, *n_iter, 1)->size() == faceVertices) {
                  if ((*e_iter < numElements) && (*n_iter < numElements)) {
                    neighborCells[*e_iter].insert(*n_iter);
                  } else {
                    point_type e = *e_iter, n = *n_iter;

                    if (*e_iter >= numElements) {
                      if (newCells.find(*e_iter) == newCells.end()) newCells[*e_iter] = --newCell;
                      e = newCells[*e_iter];
                    }
                    if (*n_iter >= numElements) {
                      if (newCells.find(*n_iter) == newCells.end()) newCells[*n_iter] = --newCell;
                      n = newCells[*n_iter];
                    }
                    neighborCells[e].insert(n);
                  }
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
        //std::cout << "numElements: " << numElements << " newCell: " << newCell << std::endl;
        *offsets   = off;
        *adjacency = adj;
        ALE_LOG_EVENT_END;
      };
      #undef __FUNCT__
      #define __FUNCT__ "buildFaceCSR"
      // This creates a CSR representation of the adjacency hypergraph for faces
      static void buildFaceCSR(const Obj<bundle_type>& bundle, const int dim, const Obj<typename bundle_type::numbering_type>& fNumbering, int *numEdges, int **offsets, int **adjacency) {
        ALE_LOG_EVENT_BEGIN;
        const Obj<sieve_type>&                           sieve    = bundle->getSieve();
        const Obj<typename bundle_type::label_sequence>& elements = bundle->heightStratum(0);
        int  numElements = elements->size();
        int *off         = new int[numElements+1];
        int  e;

        if (bundle->depth() != dim) {
          throw ALE::Exception("Not yet implemented for non-interpolated meshes");
        }
        off[0] = 0;
        e      = 1;
        for(typename bundle_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          off[e] = sieve->cone(*e_iter)->size() + off[e-1];
          e++;
        }
        int *adj    = new int[off[numElements]];
        int  offset = 0;
        for(typename bundle_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
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
      static PartitionType *subordinatePartition(const Obj<bundle_type>& bundle, int levels, const Obj<bundle_type>& subBundle, const PartitionType assignment[]) {
        const Obj<typename bundle_type::numbering_type>& cNumbering = bundle->getFactory()->getLocalNumbering(bundle, bundle->depth());
        const Obj<typename bundle_type::label_sequence>& cells      = subBundle->heightStratum(0);
        const Obj<typename bundle_type::numbering_type>& sNumbering = bundle->getFactory()->getLocalNumbering(subBundle, subBundle->depth());
        const int        numCells      = cells->size();
        PartitionType   *subAssignment = new PartitionType[numCells];

        if (levels != 1) {
          throw ALE::Exception("Cannot calculate subordinate partition for any level separation other than 1");
        } else {
          const Obj<typename bundle_type::sieve_type>&   sieve    = bundle->getSieve();
          const Obj<typename bundle_type::sieve_type>&   subSieve = subBundle->getSieve();
          Obj<typename bundle_type::sieve_type::coneSet> tmpSet   = new typename bundle_type::sieve_type::coneSet();
          Obj<typename bundle_type::sieve_type::coneSet> tmpSet2  = new typename bundle_type::sieve_type::coneSet();

          for(typename bundle_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
            const Obj<typename bundle_type::sieve_type::coneSequence>& cone = subSieve->cone(*c_iter);

            Obj<typename bundle_type::sieve_type::supportSet> cell = sieve->nJoin1(cone);
            if (cell->size() != 1) {
              std::cout << "Indeterminate subordinate partition for face " << *c_iter << std::endl;
              for(typename bundle_type::sieve_type::supportSet::iterator s_iter = cell->begin(); s_iter != cell->end(); ++s_iter) {
                std::cout << "  cell " << *s_iter << std::endl;
              }
              // Could relax this to choosing the first one
              throw ALE::Exception("Indeterminate subordinate partition");
            }
            subAssignment[sNumbering->getIndex(*c_iter)] = assignment[cNumbering->getIndex(*cell->begin())];
            tmpSet->clear();
            tmpSet2->clear();
          }
        }
        return subAssignment;
      };
    };
#ifdef PETSC_HAVE_CHACO
    namespace Chaco {
      template<typename Bundle_>
      class Partitioner {
      public:
        typedef Bundle_                          bundle_type;
        typedef typename bundle_type::sieve_type sieve_type;
        typedef typename bundle_type::point_type point_type;
        typedef short int                        part_type;
      public:
        #undef __FUNCT__
        #define __FUNCT__ "ChacoPartitionSieve"
        static part_type *partitionSieve(const Obj<bundle_type>& bundle, const int dim) {
          part_type *assignment = NULL; /* set number of each vtx (length n) */
          int       *start;             /* start of edge list for each vertex */
          int       *adjacency;         /* = adj -> j; edge list data  */

          ALE_LOG_EVENT_BEGIN;
          ALE::New::Partitioner<bundle_type>::buildDualCSR(bundle, dim, &start, &adjacency);
          if (bundle->commRank() == 0) {
            /* arguments for Chaco library */
            FREE_GRAPH = 0;                         /* Do not let Chaco free my memory */
            int nvtxs;                              /* number of vertices in full graph */
            int *vwgts = NULL;                      /* weights for all vertices */
            float *ewgts = NULL;                    /* weights for all edges */
            float *x = NULL, *y = NULL, *z = NULL;  /* coordinates for inertial method */
            char *outassignname = NULL;             /*  name of assignment output file */
            char *outfilename = NULL;               /* output file name */
            int architecture = dim;                 /* 0 => hypercube, d => d-dimensional mesh */
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
	    float *vCoords[3];
            PetscErrorCode ierr;

	    ierr = PetscOptionsGetInt(PETSC_NULL, "-partitioner_chaco_global_method", &global_method, PETSC_NULL);CHKERROR(ierr, "Error in PetscOptionsGetInt");
	    ierr = PetscOptionsGetInt(PETSC_NULL, "-partitioner_chaco_local_method",  &local_method,  PETSC_NULL);CHKERROR(ierr, "Error in PetscOptionsGetInt");
	    if (global_method == 3) {
	      // Inertial Partitioning
	      ierr = PetscMalloc3(nvtxs,float,&x,nvtxs,float,&y,nvtxs,float,&z);CHKERROR(ierr, "Error in PetscMalloc");
	      vCoords[0] = x; vCoords[1] = y; vCoords[2] = z;
	      const Obj<typename bundle_type::label_sequence>&    cells       = bundle->heightStratum(0);
	      const Obj<typename bundle_type::real_section_type>& coordinates = bundle->getRealSection("coordinates");
	      const int corners = bundle->size(coordinates, *(cells->begin()))/dim;
	      int       c       = 0;

	      for(typename bundle_type::label_sequence::iterator c_iter = cells->begin(); c_iter !=cells->end(); ++c_iter, ++c) {
		const double *coords = bundle->restrict(coordinates, *c_iter);

		for(int d = 0; d < dim; ++d) {
		  vCoords[d][c] = 0.0;
		}
		for(int v = 0; v < corners; ++v) {
		  for(int d = 0; d < dim; ++d) {
		    vCoords[d][c] += coords[v*dim+d];
		  }
		}
		for(int d = 0; d < dim; ++d) {
		  vCoords[d][c] /= corners;
		}
	      }
	    }

            nvtxs = bundle->heightStratum(0)->size();
            mesh_dims[0] = bundle->commSize(); mesh_dims[1] = 1; mesh_dims[2] = 1;
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
            if (bundle->debug()) {
              std::cout << msgLog << std::endl;
            }
            delete [] msgLog;
#endif
	    if (global_method == 3) {
	      // Inertial Partitioning
	      ierr = PetscFree3(x, y, z);CHKERROR(ierr, "Error in PetscFree");
	    }
          }
          if (adjacency) delete [] adjacency;
          if (start)     delete [] start;
          ALE_LOG_EVENT_END;
          return assignment;
        };
        static part_type *partitionSieveByFace(const Obj<bundle_type>& bundle, const int dim) {
          throw ALE::Exception("Chaco cannot partition a mesh by faces");
        };
      };
    };
#endif
#ifdef PETSC_HAVE_PARMETIS
    namespace ParMetis {
      template<typename Bundle_>
      class Partitioner {
      public:
        typedef Bundle_                          bundle_type;
        typedef typename bundle_type::sieve_type sieve_type;
        typedef typename bundle_type::point_type point_type;
        typedef int                              part_type;
      public:
        #undef __FUNCT__
        #define __FUNCT__ "ParMetisPartitionSieve"
        static part_type *partitionSieve(const Obj<bundle_type>& bundle, const int dim) {
          int    nvtxs      = 0;    // The number of vertices in full graph
          int   *vtxdist;           // Distribution of vertices across processes
          int   *xadj;              // Start of edge list for each vertex
          int   *adjncy;            // Edge lists for all vertices
          int   *vwgt       = NULL; // Vertex weights
          int   *adjwgt     = NULL; // Edge weights
          int    wgtflag    = 0;    // Indicates which weights are present
          int    numflag    = 0;    // Indicates initial offset (0 or 1)
          int    ncon       = 1;    // The number of weights per vertex
          int    nparts     = bundle->commSize(); // The number of partitions
          float *tpwgts;            // The fraction of vertex weights assigned to each partition
          float *ubvec;             // The balance intolerance for vertex weights
          int    options[5];        // Options
          // Outputs
          int    edgeCut;           // The number of edges cut by the partition
          int   *assignment = NULL; // The vertex partition

          options[0] = 0; // Use all defaults
          vtxdist    = new int[nparts+1];
          vtxdist[0] = 0;
          tpwgts     = new float[ncon*nparts];
          for(int p = 0; p < nparts; ++p) {
            tpwgts[p] = 1.0/nparts;
          }
          ubvec      = new float[ncon];
          ubvec[0]   = 1.05;
          nvtxs      = bundle->heightStratum(0)->size();
          assignment = new part_type[nvtxs];
          MPI_Allgather(&nvtxs, 1, MPI_INT, &vtxdist[1], 1, MPI_INT, bundle->comm());
          for(int p = 2; p <= nparts; ++p) {
            vtxdist[p] += vtxdist[p-1];
          }
          if (bundle->commSize() == 1) {
            PetscMemzero(assignment, nvtxs * sizeof(part_type));
          } else {
            ALE::New::Partitioner<bundle_type>::buildDualCSR(bundle, dim, &xadj, &adjncy);

            if (bundle->debug() && nvtxs) {
              for(int p = 0; p <= nvtxs; ++p) {
                std::cout << "["<<bundle->commRank()<<"]xadj["<<p<<"] = " << xadj[p] << std::endl;
              }
              for(int i = 0; i < xadj[nvtxs]; ++i) {
                std::cout << "["<<bundle->commRank()<<"]adjncy["<<i<<"] = " << adjncy[i] << std::endl;
              }
            }
            if (vtxdist[1] == vtxdist[nparts]) {
              if (bundle->commRank() == 0) {
                METIS_PartGraphKway(&nvtxs, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &nparts, options, &edgeCut, assignment);
                if (bundle->debug()) {std::cout << "Metis: edgecut is " << edgeCut << std::endl;}
              }
            } else {
              MPI_Comm comm = bundle->comm();

              ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgeCut, assignment, &comm);
              if (bundle->debug()) {std::cout << "ParMetis: edgecut is " << edgeCut << std::endl;}
            }
            if (xadj   != NULL) delete [] xadj;
            if (adjncy != NULL) delete [] adjncy;
          }
          delete [] vtxdist;
          delete [] tpwgts;
          delete [] ubvec;
          return assignment;
        };
        #undef __FUNCT__
        #define __FUNCT__ "ParMetisPartitionSieveByFace"
        static part_type *partitionSieveByFace(const Obj<bundle_type>& bundle, const int dim) {
#ifdef PETSC_HAVE_HMETIS
          int   *assignment = NULL; // The vertex partition
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
          const Obj<ALE::Mesh::numbering_type>& fNumbering = bundle->getFactory()->getNumbering(bundle, bundle->depth()-1);

          if (topology->commRank() == 0) {
            nvtxs      = bundle->heightStratum(1)->size();
            vwgts      = NULL;
            hewgts     = NULL;
            nparts     = bundle->commSize();
            ubfactor   = 5;
            options[0] = 1;  // Use all defaults
            options[1] = 10; // Number of bisections tested
            options[2] = 1;  // Vertex grouping scheme
            options[3] = 1;  // Objective function
            options[4] = 1;  // V-cycle refinement
            options[5] = 0;
            options[6] = 0;
            options[7] = 1; // Random seed
            options[8] = 24; // Debugging level
            assignment = new part_type[nvtxs];

            if (bundle->commSize() == 1) {
              PetscMemzero(assignment, nvtxs * sizeof(part_type));
            } else {
              ALE::New::Partitioner<bundle_type>::buildFaceCSR(bundle, dim, fNumbering, &nhedges, &eptr, &eind);
              HMETIS_PartKway(nvtxs, nhedges, vwgts, eptr, eind, hewgts, nparts, ubfactor, options, assignment, &edgeCut);

              delete [] eptr;
              delete [] eind;
            }
            if (bundle->debug()) {for (int i = 0; i<nvtxs; i++) printf("[%d] %d\n", PetscGlobalRank, assignment[i]);}
          } else {
            assignment = NULL;
          }
          return assignment;
#else
          throw ALE::Exception("hmetis partitioner is not available.");
#endif
        };
      };
    };
#endif
#ifdef PETSC_HAVE_ZOLTAN
    namespace Zoltan {
      template<typename Bundle_>
      class Partitioner {
      public:
        typedef Bundle_                          bundle_type;
        typedef typename bundle_type::sieve_type sieve_type;
        typedef typename bundle_type::point_type point_type;
        typedef int                              part_type;
      public:
        static part_type *partitionSieve(const Obj<bundle_type>& bundle, const int dim) {
          throw ALE::Exception("Zoltan partition by cells not implemented");
        };
        #undef __FUNCT__
        #define __FUNCT__ "ZoltanPartitionSieveByFace"
        static part_type *partitionSieveByFace(const Obj<bundle_type>& bundle, const int dim) {
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
          const Obj<typename bundle_type::numbering_type>& fNumbering = bundle->getFactory()->getNumbering(bundle, bundle->depth()-1);

          if (bundle->commSize() == 1) {
            PetscMemzero(assignment, bundle->heightStratum(1)->size() * sizeof(part_type));
          } else {
            if (bundle->commRank() == 0) {
              nvtxs_Zoltan = bundle->heightStratum(1)->size();
              ALE::New::Partitioner<bundle_type>::buildFaceCSR(bundle, dim, fNumbering, &nhedges_Zoltan, &eptr_Zoltan, &eind_Zoltan);
              assignment = new int[nvtxs_Zoltan];
            } else {
              nvtxs_Zoltan   = bundle->heightStratum(1)->size();
              nhedges_Zoltan = 0;
              eptr_Zoltan    = new int[1];
              eind_Zoltan    = new int[1];
              eptr_Zoltan[0] = 0;
              assignment     = NULL;
            }

            int ierr = Zoltan_Initialize(0, NULL, &version);
            struct Zoltan_Struct *zz = Zoltan_Create(bundle->comm());
            // General parameters
            Zoltan_Set_Param(zz, "DEBUG_LEVEL", "2");
            Zoltan_Set_Param(zz, "LB_METHOD", "PHG");
            Zoltan_Set_Param(zz, "RETURN_LISTS", "PARTITION");
            // PHG parameters
            Zoltan_Set_Param(zz, "PHG_OUTPUT_LEVEL", "2");
            Zoltan_Set_Param(zz, "PHG_EDGE_SIZE_THRESHOLD", "1.0"); // Do not throw out dense edges
            // Call backs
            Zoltan_Set_Num_Obj_Fn(zz, getNumVertices_Zoltan, NULL);
            Zoltan_Set_Obj_List_Fn(zz, getLocalElements_Zoltan, NULL);
            Zoltan_Set_HG_Size_CS_Fn(zz, getHgSizes_Zoltan, NULL);
            Zoltan_Set_HG_CS_Fn(zz, getHg_Zoltan, NULL);
            // Debugging
            //Zoltan_Generate_Files(zz, "zoltan.debug", 1, 0, 0, 1); // if using hypergraph callbacks

            ierr = Zoltan_LB_Partition(zz, &changed, &numGidEntries, &numLidEntries,
                                       &numImport, &import_global_ids, &import_local_ids, &import_procs, &import_to_part,
                                       &numExport, &export_global_ids, &export_local_ids, &export_procs, &export_to_part);
            for(int v = 0; v < nvtxs_Zoltan; ++v) {
              assignment[v] = export_to_part[v];
            }
            Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
            Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
            Zoltan_Destroy(&zz);

            delete [] eptr_Zoltan;
            delete [] eind_Zoltan;
          }
          if (assignment) {for (int i=0; i<nvtxs_Zoltan; i++) printf("[%d] %d\n",PetscGlobalRank,assignment[i]);}
          return assignment;
        };
      };
    };
#endif
  }
}

namespace ALECompat {
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
        typedef typename ALECompat::New::Completion<topology_type, typename Mesh::sieve_type::point_type> completion;
        const Obj<sieve_type>&                             sieve        = topology->getPatch(patch);
        const Obj<typename topology_type::label_sequence>& elements     = topology->heightStratum(patch, 0);
        Obj<sieve_type>                                    overlapSieve = new sieve_type(topology->comm(), topology->debug());
        int  numElements = elements->size();
        int *off         = new int[numElements+1];
        int  offset      = 0;
        int *adj;

        completion::scatterSupports(sieve, overlapSieve, topology->getSendOverlap(), topology->getRecvOverlap(), topology);
        if (numElements == 0) {
          *offsets   = NULL;
          *adjacency = NULL;
          ALE_LOG_EVENT_END;
          return;
        }
        if (topology->depth(patch) == dim) {
          int e = 1;

          off[0] = 0;
          for(typename topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
            const Obj<typename sieve_type::traits::coneSequence>& faces  = sieve->cone(*e_iter);
            typename sieve_type::traits::coneSequence::iterator   fBegin = faces->begin();
            typename sieve_type::traits::coneSequence::iterator   fEnd   = faces->end();

            off[e] = off[e-1];
            for(typename sieve_type::traits::coneSequence::iterator f_iter = fBegin; f_iter != fEnd; ++f_iter) {
              if (sieve->support(*f_iter)->size() == 2) {
                off[e]++;
              } else if ((sieve->support(*f_iter)->size() == 1) && (overlapSieve->support(*f_iter)->size() == 1)) {
                off[e]++;
              }
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
              const Obj<typename sieve_type::traits::supportSequence>& oNeighbors = overlapSieve->support(*f_iter);
              typename sieve_type::traits::supportSequence::iterator   onBegin    = oNeighbors->begin();
              typename sieve_type::traits::supportSequence::iterator   onEnd      = oNeighbors->end();

              for(typename sieve_type::traits::supportSequence::iterator n_iter = onBegin; n_iter != onEnd; ++n_iter) {
                adj[offset++] = *n_iter;
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
      template<typename PartitionType>
      static PartitionType *subordinatePartition(const Obj<topology_type>& topology, int levels, const Obj<topology_type>& subTopology, const PartitionType assignment[]) {
        typedef ALECompat::New::NumberingFactory<topology_type> NumberingFactory;
        const patch_type patch = 0;
        const Obj<typename NumberingFactory::numbering_type>& cNumbering = NumberingFactory::singleton(topology->debug())->getLocalNumbering(topology, patch, topology->depth(patch));
        const Obj<typename topology_type::label_sequence>&    cells      = subTopology->heightStratum(patch, 0);
        const Obj<typename NumberingFactory::numbering_type>& sNumbering = NumberingFactory::singleton(subTopology->debug())->getLocalNumbering(subTopology, patch, subTopology->depth(patch));
        const int        numCells      = cells->size();
        PartitionType   *subAssignment = new PartitionType[numCells];

        if (levels != 1) {
          throw ALE::Exception("Cannot calculate subordinate partition for any level separation other than 1");
        } else {
          const Obj<typename topology_type::sieve_type>&   sieve    = topology->getPatch(patch);
          const Obj<typename topology_type::sieve_type>&   subSieve = subTopology->getPatch(patch);
          Obj<typename topology_type::sieve_type::coneSet> tmpSet   = new typename topology_type::sieve_type::coneSet();
          Obj<typename topology_type::sieve_type::coneSet> tmpSet2  = new typename topology_type::sieve_type::coneSet();

          for(typename topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
            const Obj<typename topology_type::sieve_type::coneSequence>& cone = subSieve->cone(*c_iter);

            Obj<typename topology_type::sieve_type::supportSet> cell = sieve->nJoin1(cone);
            if (cell->size() != 1) {
              std::cout << "Indeterminate subordinate partition for face " << *c_iter << std::endl;
              for(typename topology_type::sieve_type::supportSet::iterator s_iter = cell->begin(); s_iter != cell->end(); ++s_iter) {
                std::cout << "  cell " << *s_iter << std::endl;
              }
              // Could relax this to choosing the first one
              throw ALE::Exception("Indeterminate subordinate partition");
            }
            subAssignment[sNumbering->getIndex(*c_iter)] = assignment[cNumbering->getIndex(*cell->begin())];
            tmpSet->clear();
            tmpSet2->clear();
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
          int       *start;             /* start of edge list for each vertex */
          int       *adjacency;         /* = adj -> j; edge list data  */
          typename topology_type::patch_type patch = 0;

          ALE_LOG_EVENT_BEGIN;
          ALECompat::New::Partitioner<topology_type>::buildDualCSR(topology, dim, patch, &start, &adjacency);
          if (topology->commRank() == 0) {
            /* arguments for Chaco library */
            FREE_GRAPH = 0;                         /* Do not let Chaco free my memory */
            int nvtxs;                              /* number of vertices in full graph */
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
            PetscErrorCode ierr;

            nvtxs = topology->heightStratum(patch, 0)->size();
            mesh_dims[0] = topology->commSize(); mesh_dims[1] = 1; mesh_dims[2] = 1;
            for(int e = 0; e < start[nvtxs]; e++) {
              adjacency[e]++;
            }
            assignment = new part_type[nvtxs];
            ierr = PetscMemzero(assignment, nvtxs * sizeof(part_type));

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
          }
          if (adjacency) delete [] adjacency;
          if (start)     delete [] start;
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
          int    nvtxs      = 0;    // The number of vertices in full graph
          int   *vtxdist;           // Distribution of vertices across processes
          int   *xadj;              // Start of edge list for each vertex
          int   *adjncy;            // Edge lists for all vertices
          int   *vwgt       = NULL; // Vertex weights
          int   *adjwgt     = NULL; // Edge weights
          int    wgtflag    = 0;    // Indicates which weights are present
          int    numflag    = 0;    // Indicates initial offset (0 or 1)
          int    ncon       = 1;    // The number of weights per vertex
          int    nparts     = topology->commSize(); // The number of partitions
          float *tpwgts;            // The fraction of vertex weights assigned to each partition
          float *ubvec;             // The balance intolerance for vertex weights
          int    options[5];        // Options
          // Outputs
          int    edgeCut;           // The number of edges cut by the partition
          int   *assignment = NULL; // The vertex partition
          const typename topology_type::patch_type patch = 0;

          options[0] = 0; // Use all defaults
          vtxdist    = new int[nparts+1];
          vtxdist[0] = 0;
          tpwgts     = new float[ncon*nparts];
          for(int p = 0; p < nparts; ++p) {
            tpwgts[p] = 1.0/nparts;
          }
          ubvec      = new float[ncon];
          ubvec[0]   = 1.05;
          if (topology->hasPatch(patch)) {
            nvtxs      = topology->heightStratum(patch, 0)->size();
            assignment = new part_type[nvtxs];
          }
          MPI_Allgather(&nvtxs, 1, MPI_INT, &vtxdist[1], 1, MPI_INT, topology->comm());
          for(int p = 2; p <= nparts; ++p) {
            vtxdist[p] += vtxdist[p-1];
          }
          if (topology->commSize() == 1) {
            PetscMemzero(assignment, nvtxs * sizeof(part_type));
          } else {
            ALECompat::New::Partitioner<topology_type>::buildDualCSR(topology, dim, patch, &xadj, &adjncy);

            if (topology->debug() && nvtxs) {
              for(int p = 0; p <= nvtxs; ++p) {
                std::cout << "["<<topology->commRank()<<"]xadj["<<p<<"] = " << xadj[p] << std::endl;
              }
              for(int i = 0; i < xadj[nvtxs]; ++i) {
                std::cout << "["<<topology->commRank()<<"]adjncy["<<i<<"] = " << adjncy[i] << std::endl;
              }
            }
            if (vtxdist[1] == vtxdist[nparts]) {
              if (topology->commRank() == 0) {
                METIS_PartGraphKway(&nvtxs, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &nparts, options, &edgeCut, assignment);
                if (topology->debug()) {std::cout << "Metis: edgecut is " << edgeCut << std::endl;}
              }
            } else {
              MPI_Comm comm = topology->comm();

              ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgeCut, assignment, &comm);
              if (topology->debug()) {std::cout << "ParMetis: edgecut is " << edgeCut << std::endl;}
            }
            if (xadj   != NULL) delete [] xadj;
            if (adjncy != NULL) delete [] adjncy;
          }
          delete [] vtxdist;
          delete [] tpwgts;
          delete [] ubvec;
          return assignment;
        };
      };
    };
#endif
  }
}
#endif
