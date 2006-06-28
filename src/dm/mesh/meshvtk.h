#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/

class VTKViewer {
 public:
  VTKViewer() {};
  virtual ~VTKViewer() {};

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteHeader"
  static PetscErrorCode writeHeader(PetscViewer viewer) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer,"# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Simplicial Mesh Example\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"ASCII\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"DATASET UNSTRUCTURED_GRID\n");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteVertices"
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
    ierr = PetscViewerASCIIPrintf(viewer, "POINTS %d double\n", numVertices);CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      int numLocalVertices = mesh->getTopology()->depthStratum(0)->size();
      int embedDim = coordinates->getFiberDimension(patch, *mesh->getTopology()->depthStratum(0)->begin());

      for(int v = 0; v < numLocalVertices; v++) {
        for(int d = 0; d < embedDim; d++) {
          if (d > 0) {
            ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "%G", array[v*embedDim+d]);CHKERRQ(ierr);
        }
        for(int d = embedDim; d < 3; d++) {
          ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
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
          for(int d = 0; d < embedDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "%G", remoteCoords[v*embedDim+d]);CHKERRQ(ierr);
          }
          for(int d = embedDim; d < 3; d++) {
            ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
      }
    } else {
      ALE::Obj<ALE::Mesh::bundle_type> globalOrder = coordinates->getGlobalOrder();
      ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone = globalOrder->getPatch(patch);
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
  #define __FUNCT__ "VTKWriteField"
  static PetscErrorCode writeField(ALE::Obj<ALE::Mesh> mesh, ALE::Obj<ALE::Mesh::field_type> field, const std::string& name, const int fiberDim, ALE::Obj<ALE::Mesh::bundle_type> globalOrder, PetscViewer viewer, int enforceDim = -1) {
    ALE::Mesh::field_type::patch_type patch;
    const double  *array = field->restrict(patch);
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (enforceDim < 0) enforceDim = fiberDim;
    if (enforceDim == 3) {
      ierr = PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name.c_str());CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "SCALARS %s double %d\n", name.c_str(), fiberDim);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
    }
    if (mesh->commRank() == 0) {
      ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> elements = globalOrder->getPatch(patch);

      for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        const ALE::Mesh::bundle_type::index_type& idx = field->getIndex(patch, *e_iter);

        if (idx.index > 0) {
          for(int d = 0; d < fiberDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "%G", array[idx.prefix+d]);CHKERRQ(ierr);
          }
          for(int d = fiberDim; d < enforceDim; d++) {
            ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
      }
      for(int p = 1; p < mesh->commSize(); p++) {
        double    *remoteValues;
        int        numLocalElements;
        MPI_Status status;

        ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        ierr = PetscMalloc(numLocalElements*fiberDim * sizeof(double), &remoteValues);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteValues, numLocalElements*fiberDim, MPI_DOUBLE, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        for(int e = 0; e < numLocalElements; e++) {
          for(int d = 0; d < fiberDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "%G", remoteValues[e*fiberDim+d]);CHKERRQ(ierr);
          }
          for(int d = fiberDim; d < enforceDim; d++) {
            ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
      }
    } else {
      ALE::Obj<ALE::Mesh::bundle_type>                           fieldGlobalOrder = field->getGlobalOrder();
      ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> elements = globalOrder->getPatch(patch);
      const int *offsets          = field->getGlobalOffsets();
      int        numLocalElements = (offsets[mesh->commRank()+1] - offsets[mesh->commRank()])/fiberDim;
      double    *localValues;
      int        k = 0;

      ierr = PetscMalloc(numLocalElements*fiberDim * sizeof(double), &localValues);CHKERRQ(ierr);
      for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
        int dim = fieldGlobalOrder->getFiberDimension(patch, *e_iter);

        if (dim > 0) {
          int offset = field->getFiberOffset(patch, *e_iter);

          for(int i = offset; i < offset+dim; ++i) {
            localValues[k++] = array[i];
          }
        }
      }
      if (k != numLocalElements*fiberDim) {
        SETERRQ3(PETSC_ERR_PLIB, "Invalid number of values to send for field %s, %d should be %d", name.c_str(), k, numLocalElements*fiberDim);
      }
      ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localValues, numLocalElements*fiberDim, MPI_DOUBLE, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = PetscFree(localValues);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteField2"
  static PetscErrorCode writeField2(ALE::Obj<ALE::Mesh> mesh, ALE::Obj<ALE::Mesh::field_type> field, const std::string& name, const int fiberDim, PetscViewer viewer, int enforceDim = -1) {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (enforceDim < 0) enforceDim = fiberDim;
    if (enforceDim == 3) {
      ierr = PetscViewerASCIIPrintf(viewer, "VECTORS %s double\n", name.c_str());CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "SCALARS %s double %d\n", name.c_str(), fiberDim);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
    }
    if (mesh->commRank() == 0) {
      ALE::Obj<ALE::Mesh::field_type::order_type::baseSequence> patches = field->getPatches();

      ALE::Obj<ALE::Mesh::bundle_type> globalVertex = mesh->getBundle(0)->getGlobalOrder();

      for(ALE::Mesh::field_type::order_type::baseSequence::iterator p_iter = patches->begin(); p_iter != patches->end(); ++p_iter) {
        ALE::Obj<ALE::Mesh::field_type::order_type::coneSequence> elements = field->getPatch(*p_iter);
        const double *array = field->restrict(*p_iter);

        for(ALE::Mesh::field_type::order_type::coneSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          const ALE::Mesh::field_type::index_type& idx = field->getIndex(*p_iter, *e_iter);

          if (idx.index > 0) {
            if (enforceDim == 3) {
              ierr = PetscViewerASCIIPrintf(viewer, "Vertex %d (%d)", globalVertex->getIndex(*p_iter, *e_iter).prefix, idx.prefix);CHKERRQ(ierr);
            }
            for(int d = 0; d < idx.index; d++) {
              if (d > 0) {
                ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
              }
              ierr = PetscViewerASCIIPrintf(viewer, "%G", array[idx.prefix+d]);CHKERRQ(ierr);
            }
            for(int d = idx.index; d < enforceDim; d++) {
              ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
          }
        }
      }
      for(int p = 1; p < mesh->commSize(); p++) {
        double    *remoteCoords;
        MPI_Status status;
        int        numLocalElements;

        ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        ierr = PetscMalloc(numLocalElements*fiberDim * sizeof(double), &remoteCoords);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteCoords, numLocalElements*fiberDim, MPI_DOUBLE, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        for(int v = 0; v < numLocalElements; v++) {
          for(int d = 0; d < fiberDim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "%G", remoteCoords[v*fiberDim+d]);CHKERRQ(ierr);
          }
          for(int d = fiberDim; d < enforceDim; d++) {
            ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
      }
    } else {
      ALE::Obj<ALE::Mesh::bundle_type>                           globalOrder = field->getGlobalOrder();
      ALE::Obj<ALE::Mesh::bundle_type::order_type::baseSequence> patches     = globalOrder->getPatches();
      const int                                                 *offsets     = field->getGlobalOffsets();
      int                                                        numLocalElements = 0;
      int                                                        k           = 0;
      double                                                    *localVals;

      for(ALE::Mesh::bundle_type::order_type::baseSequence::iterator p_iter = patches->begin(); p_iter != patches->end(); ++p_iter) {
        numLocalElements += (offsets[mesh->commRank()+1] - offsets[mesh->commRank()])/fiberDim;
      }
      ierr = PetscMalloc(numLocalElements*fiberDim * sizeof(double), &localVals);CHKERRQ(ierr);
      for(ALE::Mesh::bundle_type::order_type::baseSequence::iterator p_iter = patches->begin(); p_iter != patches->end(); ++p_iter) {
        ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> elements = globalOrder->getPatch(*p_iter);
        const double *array = field->restrict(*p_iter);

        for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          int dim = globalOrder->getFiberDimension(*p_iter, *e_iter);

          if (dim > 0) {
            int offset = field->getFiberOffset(*p_iter, *e_iter);

            for(int i = offset; i < offset+dim; ++i) {
              localVals[k++] = array[i];
            }
          }
        }
      }
      if (k != numLocalElements*fiberDim) {
        SETERRQ2(PETSC_ERR_PLIB, "Invalid number of coordinates to send %d should be %d", k, numLocalElements*fiberDim);
      }
      ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localVals, numLocalElements*fiberDim, MPI_DOUBLE, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = PetscFree(localVals);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "VTKWriteElements"
  static PetscErrorCode writeElements(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer)
  {
    ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
    ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
    ALE::Obj<ALE::Mesh::bundle_type> elementBundle = mesh->getBundle(topology->depth());
    ALE::Obj<ALE::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
    ALE::Obj<ALE::Mesh::bundle_type> globalVertex = vertexBundle->getGlobalOrder();
    ALE::Obj<ALE::Mesh::bundle_type> globalElement = elementBundle->getGlobalOrder();
    ALE::Mesh::bundle_type::patch_type patch;
    std::string    orderName("element");
    int            corners = topology->nCone(*elements->begin(), topology->depth())->size();
    int            numElements;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (!globalVertex) {
      globalVertex = vertexBundle;
    }
    if (elementBundle->getGlobalOffsets()) {
      numElements = elementBundle->getGlobalOffsets()[mesh->commSize()];
    } else {
      numElements = mesh->getTopology()->heightStratum(0)->size();
    }
    ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
        ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

        ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
        for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
          ierr = PetscViewerASCIIPrintf(viewer, " %d", globalVertex->getIndex(patch, *c_itor).prefix);CHKERRQ(ierr);
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
          ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
          for(int c = 0; c < corners; c++) {
            ierr = PetscViewerASCIIPrintf(viewer, " %d", remoteVertices[e*corners+c]);CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer, "CELL_TYPES %d\n", numElements);CHKERRQ(ierr);
    if (corners == 2) {
      // VTK_LINE
      for(int e = 0; e < numElements; e++) {
        ierr = PetscViewerASCIIPrintf(viewer, "3\n");CHKERRQ(ierr);
      }
    } else if (corners == 3) {
      // VTK_TRIANGLE
      for(int e = 0; e < numElements; e++) {
        ierr = PetscViewerASCIIPrintf(viewer, "5\n");CHKERRQ(ierr);
      }
    } else if (corners == 4) {
      if (mesh->getDimension() == 3) {
        // VTK_TETRA
        for(int e = 0; e < numElements; e++) {
          ierr = PetscViewerASCIIPrintf(viewer, "10\n");CHKERRQ(ierr);
        }
      } else if (mesh->getDimension() == 2) {
        // VTK_QUAD
        for(int e = 0; e < numElements; e++) {
          ierr = PetscViewerASCIIPrintf(viewer, "9\n");CHKERRQ(ierr);
        }
      }
    } else if (corners == 8) {
      // VTK_HEXAHEDRON
      for(int e = 0; e < numElements; e++) {
        ierr = PetscViewerASCIIPrintf(viewer, "12\n");CHKERRQ(ierr);
      }
    }
    PetscFunctionReturn(0);
  };
};
