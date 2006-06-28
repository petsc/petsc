#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

class PCICEViewer {
 public:
  PCICEViewer() {};
  virtual ~PCICEViewer() {};

  #undef __FUNCT__  
  #define __FUNCT__ "PCICEWriteVertices"
  static PetscErrorCode writeVertices(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer) {
    ALE::Obj<ALE::Mesh::field_type>   coordinates  = mesh->getCoordinates();
    ALE::Obj<ALE::Mesh::bundle_type>  vertexBundle = mesh->getBundle(0);
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
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "PCICEWriteElements"
  static PetscErrorCode writeElements(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer) {
    ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
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
    PetscFunctionReturn(0);
  };
};
