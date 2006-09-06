#include "src/dm/mesh/meshpcice.h"   /*I      "petscmesh.h"   I*/

namespace ALE {
  namespace PCICE {
    //
    // Builder methods
    //
    void Builder::readConnectivity(MPI_Comm comm, const std::string& filename, int& corners, const bool useZeroBase, int& numElements, int *vertices[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       numCells, cellCount = 0;
      PetscInt      *verts;
      char           buf[2048];
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);

      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      if (fgets(buf, 2048, f) == NULL) {
        throw ALE::Exception("Invalid connectivity file: Missing number of elements");
      }
      const char *sizes = strtok(buf, " ");
      numCells = atoi(sizes);
      sizes = strtok(NULL, " ");
      if (sizes != NULL) {
        corners = atoi(sizes);
        std::cout << "Reset corners to " << corners << std::endl;
      }
      ierr = PetscMalloc(numCells*corners * sizeof(PetscInt), &verts);
      while(fgets(buf, 2048, f) != NULL) {
        const char *v = strtok(buf, " ");
      
        /* Ignore cell number */
        v = strtok(NULL, " ");
        for(c = 0; c < corners; c++) {
          int vertex = atoi(v);
        
          if (!useZeroBase) vertex -= 1;
          verts[cellCount*corners+c] = vertex;
          v = strtok(NULL, " ");
        }
        cellCount++;
      }
      ierr = PetscViewerDestroy(viewer);
      numElements = numCells;
      *vertices = verts;
    };
    void Builder::readCoordinates(MPI_Comm comm, const std::string& filename, const int dim, int& numVertices, double *coordinates[]) {
      PetscViewer    viewer;
      FILE          *f;
      PetscInt       numVerts, vertexCount = 0;
      PetscScalar   *coords;
      char           buf[2048];
      PetscInt       c;
      PetscInt       commRank;
      PetscErrorCode ierr;

      ierr = MPI_Comm_rank(comm, &commRank);

      if (commRank != 0) return;
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);
      ierr = PetscViewerFileSetName(viewer, filename.c_str());
      ierr = PetscViewerASCIIGetPointer(viewer, &f);
      numVerts = atoi(fgets(buf, 2048, f));
      ierr = PetscMalloc(numVerts*dim * sizeof(PetscScalar), &coords);
      while(fgets(buf, 2048, f) != NULL) {
        const char *x = strtok(buf, " ");
      
        /* Ignore vertex number */
        x = strtok(NULL, " ");
        for(c = 0; c < dim; c++) {
          coords[vertexCount*dim+c] = atof(x);
          x = strtok(NULL, " ");
        }
        vertexCount++;
      }
      ierr = PetscViewerDestroy(viewer);
      numVertices = numVerts;
      *coordinates = coords;
    };
    void Builder::buildCoordinates(const Obj<section_type>& coords, const int embedDim, const double coordinates[]) {
      const section_type::patch_type            patch    = 0;
      const Obj<topology_type::label_sequence>& vertices = coords->getTopology()->depthStratum(patch, 0);
      const int numCells = coords->getTopology()->heightStratum(patch, 0)->size();

      coords->setFiberDimensionByDepth(patch, 0, embedDim);
      coords->allocate();
      for(topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        coords->update(patch, *v_iter, &(coordinates[(*v_iter - numCells)*embedDim]));
      }
    };
    Obj<Mesh> Builder::readMesh(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase = true, const bool interpolate = true, const int debug = 0) {
      Obj<Mesh>          mesh     = Mesh(comm, dim, debug);
      Obj<sieve_type>    sieve    = new sieve_type(comm, debug);
      Obj<topology_type> topology = new topology_type(comm, debug);
      int    *cells;
      double *coordinates;
      int     numCells = 0, numVertices = 0, numCorners = dim+1;

      ALE::PCICE::Builder::readConnectivity(comm, basename+".lcon", numCorners, useZeroBase, numCells, &cells);
      ALE::PCICE::Builder::readCoordinates(comm, basename+".nodes", dim, numVertices, &coordinates);
      ALE::New::SieveBuilder<sieve_type>::buildTopology(sieve, dim, numCells, cells, numVertices, interpolate, numCorners);
      sieve->stratify();
      topology->setPatch(0, sieve);
      topology->stratify();
      mesh->setTopologyNew(topology);
      buildCoordinates(mesh->getSection("coordinates"), dim, coordinates);
      return mesh;
    };
    Obj<Mesh> Builder::readMeshBoundary(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase = true, const bool interpolate = true, const int debug = 0) {
      Obj<Mesh>          mesh     = Mesh(comm, dim-1, debug);
      Obj<sieve_type>    sieve    = new sieve_type(comm, debug);
      Obj<topology_type> topology = new topology_type(comm, debug);
      int    *cells;
      double *coordinates;
      int     numCells = 0, numVertices = 0, numCorners = dim;

      ALE::PCICE::Builder::readConnectivity(comm, basename+".lcon", numCorners, useZeroBase, numCells, &cells);
      ALE::PCICE::Builder::readCoordinates(comm, basename+".nodes", dim, numVertices, &coordinates);
      ALE::New::SieveBuilder<sieve_type>::buildTopology(sieve, dim-1, numCells, cells, numVertices, interpolate, numCorners);
      sieve->stratify();
      topology->setPatch(0, sieve);
      topology->stratify();
      mesh->setTopologyNew(topology);
      buildCoordinates(mesh->getSection("coordinates"), dim, coordinates);
      return mesh;
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PCICEWriteVertices"
    PetscErrorCode Viewer::writeVertices(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer) {
      ALE::Obj<ALE::Mesh::section_type> coordinates = mesh->getSection("coordinates");
#if 0
      ALE::Mesh::field_type::patch_type patch;
      const double  *array = coordinates->restrict(patch);
      int            numVertices;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      //FIX:
      if (vertexBundle->getGlobalOffsets()) {
        numVertices = vertexBundle->getGlobalOffsets()[mesh->commSize()];
      } else {
        numVertices = mesh->getTopology()->depthStratum(0)->size();
      }
      ierr = PetscViewerASCIIPrintf(viewer, "%D\n", numVertices);CHKERRQ(ierr);
      if (mesh->commRank() == 0) {
        int numLocalVertices = mesh->getTopology()->depthStratum(0)->size();
        int embedDim = coordinates->getFiberDimension(patch, *mesh->getTopology()->depthStratum(0)->begin());
        int vertexCount = 1;

        for(int v = 0; v < numLocalVertices; v++) {
          ierr = PetscViewerASCIIPrintf(viewer, "%7D   ", vertexCount++);CHKERRQ(ierr);
          for(int d = 0; d < embedDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "% 12.5E", array[v*embedDim+d]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        for(int p = 1; p < mesh->commSize(); p++) {
          double    *remoteCoords;
          MPI_Status status;

          ierr = MPI_Recv(&numLocalVertices, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          ierr = PetscMalloc(numLocalVertices*embedDim * sizeof(double), &remoteCoords);CHKERRQ(ierr);
          ierr = MPI_Recv(remoteCoords, numLocalVertices*embedDim, MPI_DOUBLE, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          for(int v = 0; v < numLocalVertices; v++) {
            ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", vertexCount++);CHKERRQ(ierr);
            for(int d = 0; d < embedDim; d++) {
              if (d > 0) {
                ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
              }
              ierr = PetscViewerASCIIPrintf(viewer, "% 12.5E", remoteCoords[v*embedDim+d]);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
          }
        }
      } else {
        ALE::Obj<ALE::Mesh::bundle_type>                           globalOrder = coordinates->getGlobalOrder();
        ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone        = globalOrder->getPatch(patch);
        const int *offsets = coordinates->getGlobalOffsets();
        int        embedDim = coordinates->getFiberDimension(patch, *mesh->getTopology()->depthStratum(0)->begin());
        int        numLocalVertices = (offsets[mesh->commRank()+1] - offsets[mesh->commRank()])/embedDim;
        double    *localCoords;
        int        k = 0;

        ierr = PetscMalloc(numLocalVertices*embedDim * sizeof(double), &localCoords);CHKERRQ(ierr);
        for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          int dim = globalOrder->getFiberDimension(patch, *p_iter);

          if (dim > 0) {
            int offset = coordinates->getFiberOffset(patch, *p_iter);

            for(int i = offset; i < offset+dim; ++i) {
              localCoords[k++] = array[i];
            }
          }
        }
        if (k != numLocalVertices*embedDim) {
          SETERRQ2(PETSC_ERR_PLIB, "Invalid number of coordinates to send %d should be %d", k, numLocalVertices*embedDim);
        }
        ierr = MPI_Send(&numLocalVertices, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = MPI_Send(localCoords, numLocalVertices*embedDim, MPI_DOUBLE, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = PetscFree(localCoords);CHKERRQ(ierr);
      }
#endif
      PetscFunctionReturn(0);
    };
    #undef __FUNCT__  
    #define __FUNCT__ "PCICEWriteElements"
    PetscErrorCode Viewer::writeElements(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer) {
      ALE::Obj<ALE::Mesh::topology_type> topology = mesh->getTopologyNew();
#if 0
      ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
      ALE::Obj<ALE::Mesh::bundle_type> elementBundle = mesh->getBundle(topology->depth());
      ALE::Obj<ALE::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
      ALE::Obj<ALE::Mesh::bundle_type> globalVertex = vertexBundle->getGlobalOrder();
      ALE::Obj<ALE::Mesh::bundle_type> globalElement = elementBundle->getGlobalOrder();
      ALE::Mesh::bundle_type::patch_type patch;
      std::string    orderName("element");
      int            dim  = mesh->getDimension();
      int            corners = topology->nCone(*elements->begin(), topology->depth())->size();
      int            numElements;
      PetscErrorCode ierr;

      PetscFunctionBegin;
      if (corners != dim+1) {
        SETERRQ(PETSC_ERR_SUP, "PCICE only supports simplicies");
      }
      if (!globalVertex) {
        globalVertex = vertexBundle;
      }
      if (elementBundle->getGlobalOffsets()) {
        numElements = elementBundle->getGlobalOffsets()[mesh->commSize()];
      } else {
        numElements = mesh->getTopology()->heightStratum(0)->size();
      }
      if (mesh->commRank() == 0) {
        int elementCount = 1;

        ierr = PetscViewerASCIIPrintf(viewer, "%d\n", numElements);CHKERRQ(ierr);
        for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

          ierr = PetscViewerASCIIPrintf(viewer, "%7d", elementCount++);CHKERRQ(ierr);
          for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
            ierr = PetscViewerASCIIPrintf(viewer, " %7d", globalVertex->getIndex(patch, *c_itor).prefix);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
        for(int p = 1; p < mesh->commSize(); p++) {
          int        numLocalElements;
          int       *remoteVertices;
          MPI_Status status;

          ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &remoteVertices);CHKERRQ(ierr);
          ierr = MPI_Recv(remoteVertices, numLocalElements*corners, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
          for(int e = 0; e < numLocalElements; e++) {
            ierr = PetscViewerASCIIPrintf(viewer, "%7d", elementCount++);CHKERRQ(ierr);
            for(int c = 0; c < corners; c++) {
              ierr = PetscViewerASCIIPrintf(viewer, " %7d", remoteVertices[e*corners+c]);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
          }
          ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
        }
      } else {
        const int *offsets = elementBundle->getGlobalOffsets();
        int        numLocalElements = offsets[mesh->commRank()+1] - offsets[mesh->commRank()];
        int       *localVertices;
        int        k = 0;

        ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &localVertices);CHKERRQ(ierr);
        for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
          ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

          if (globalElement->getFiberDimension(patch, *e_itor) > 0) {
            for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
              localVertices[k++] = globalVertex->getIndex(patch, *c_itor).prefix;
            }
          }
        }
        if (k != numLocalElements*corners) {
          SETERRQ2(PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", k, numLocalElements*corners);
        }
        ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = MPI_Send(localVertices, numLocalElements*corners, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
        ierr = PetscFree(localVertices);CHKERRQ(ierr);
      }
#endif
      PetscFunctionReturn(0);
    };
  };
};
