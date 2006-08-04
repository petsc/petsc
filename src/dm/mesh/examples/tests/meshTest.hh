#include <Mesh.hh>
#include <src/dm/mesh/meshpcice.h>
#include "sectionTest.hh"

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

using ALE::Obj;

namespace ALE {
  namespace Test {
    class MeshProcessor {
    public:
      static std::string printMatrix(const std::string& name, const int rows, const int cols, const section_type::value_type matrix[], const int rank = -1)
      {
        ostringstream output;
        ostringstream rankStr;

        if (rank >= 0) {
          rankStr << "[" << rank << "]";
        }
        output << rankStr.str() << name << " = " << std::endl;
        for(int r = 0; r < rows; r++) {
          if (r == 0) {
            output << rankStr.str() << " /";
          } else if (r == rows-1) {
            output << rankStr.str() << " \\";
          } else {
            output << rankStr.str() << " |";
          }
          for(int c = 0; c < cols; c++) {
            output << " " << matrix[r*cols+c];
          }
          if (r == 0) {
            output << " \\" << std::endl;
          } else if (r == rows-1) {
            output << " /" << std::endl;
          } else {
            output << " |" << std::endl;
          }
        }
        return output.str();
      }
      static std::string printElement(const section_type::point_type& e, const int dim, const section_type::value_type coords[], const int rank = -1) {
        ostringstream output;
        ostringstream r;

        if (rank >= 0) {
          r << "[" << rank << "]";
        }
        output << r.str() << "Element " << e << std::endl;
        output << r.str() << "Coordinates: " << e << std::endl << r.str() << "  ";
        for(int f = 0; f <= dim; f++) {
          output << " (";
          for(int d = 0; d < dim; d++) {
            if (d > 0) output << ", ";
            output << coords[f*dim+d];
          }
          output << ")";
        }
        output << std::endl;
        return output.str();
      };
      static void computeElementGeometry(const Obj<section_type>& coordinates, int dim, const sieve_type::point_type& e, section_type::value_type v0[], section_type::value_type J[], section_type::value_type invJ[], section_type::value_type& detJ)
      {
        const section_type::patch_type  patch  = 0;
        const section_type::value_type *coords = coordinates->restrict(patch, e);
        section_type::value_type        invDet;

        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
        for(int d = 0; d < dim; d++) {
          for(int f = 0; f < dim; f++) {
            J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
          }
        }
        if (dim == 1) {
          detJ = J[0];
        } else if (dim == 2) {
          detJ = J[0]*J[3] - J[1]*J[2];
        } else if (dim == 3) {
          detJ = J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
            J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
            J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        }
        invDet = 1.0/detJ;
        if (dim == 2) {
          invJ[0] =  invDet*J[3];
          invJ[1] = -invDet*J[1];
          invJ[2] = -invDet*J[2];
          invJ[3] =  invDet*J[0];
        } else if (dim == 3) {
          // FIX: This may be wrong
          invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
          invJ[0*3+1] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
          invJ[0*3+2] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
          invJ[1*3+0] = invDet*(J[0*3+1]*J[2*3+2] - J[0*3+2]*J[2*3+1]);
          invJ[1*3+1] = invDet*(J[0*3+2]*J[2*3+0] - J[0*3+0]*J[2*3+2]);
          invJ[1*3+2] = invDet*(J[0*3+0]*J[2*3+1] - J[0*3+1]*J[2*3+0]);
          invJ[2*3+0] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
          invJ[2*3+1] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
          invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
        }
      };
      #undef __FUNCT__
      #define __FUNCT__ "buildDualCSR"
      // This creates a CSR representation of the adjacency matrix for cells
      static void buildDualCSR(const Obj<ALE::Mesh>& mesh, const topology_type::patch_type& patch, int **offsets, int **adjacency) {
        const Obj<topology_type>& topology = mesh->getTopologyNew();
        const Obj<sieve_type>&    sieve    = topology->getPatch(patch);
        const Obj<topology_type::label_sequence> elements = topology->heightStratum(patch, 0);
        int dim         = mesh->getDimension();
        int numElements = elements->size();
        int corners     = sieve->cone(*elements->begin())->size();
        int *off        = new int[numElements+1];

        std::set<topology_type::point_type> *neighborCells = new std::set<topology_type::point_type>[numElements];
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
        for(topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          const Obj<sieve_type::traits::coneSequence>& vertices  = sieve->cone(*e_iter);
          sieve_type::traits::coneSequence::iterator vEnd = vertices->end();

          for(sieve_type::traits::coneSequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
            const Obj<sieve_type::traits::supportSequence>& neighbors = sieve->support(*v_iter);
            sieve_type::traits::supportSequence::iterator nEnd = neighbors->end();

            for(sieve_type::traits::supportSequence::iterator n_iter = neighbors->begin(); n_iter != nEnd; ++n_iter) {
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
          for(std::set<topology_type::point_type>::iterator n_iter = neighborCells[e].begin(); n_iter != neighborCells[e].end(); ++n_iter) {
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
      };
#ifdef PETSC_HAVE_CHACO
      #undef __FUNCT__
      #define __FUNCT__ "partitionMesh_Chaco"
      static short *partitionMesh_Chaco(Obj<ALE::Mesh> mesh) {
        short *assignment = NULL; /* set number of each vtx (length n) */

        if (mesh->commRank() == 0) {
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

          nvtxs = mesh->getTopologyNew()->heightStratum(0, 0)->size();
          mesh_dims[0] = mesh->commSize(); mesh_dims[1] = 1; mesh_dims[2] = 1;
          ALE::Test::MeshProcessor::buildDualCSR(mesh, patch, &start, &adjacency);
          for(int e = 0; e < start[nvtxs+1]; e++) {
            adjacency[e]++;
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
          if (mesh->debug) {
            std::cout << msgLog << std::endl;
          }
          delete [] msgLog;
#endif
          delete [] adjacency;
          delete [] start;
        }
        return assignment;
      };
#endif
    };
  };
};
