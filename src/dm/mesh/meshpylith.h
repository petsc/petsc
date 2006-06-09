#include "src/dm/mesh/mesh.h"   /*I      "petscmesh.h"   I*/

static int vertexCount = 0;

class PyLithViewer {
  struct vertexOutput {
    ALE::Obj<ALE::Mesh::field_type> coordinates;
    PetscViewer viewer;
    int dim;
  public:
  vertexOutput(PetscViewer viewer, ALE::Obj<ALE::Mesh::field_type> coordinates, int dim) : coordinates(coordinates), viewer(viewer), dim(dim) {vertexCount = 0;};

    bool operator()(const ALE::Mesh::point_type& p) const {
      const double *array = coordinates->restrict(ALE::Mesh::field_type::patch_type(), p);

      PetscViewerASCIIPrintf(this->viewer, "%7D ", 1 + vertexCount++);
      for(int d = 0; d < dim; d++) {
        if (d > 0) {
          PetscViewerASCIIPrintf(this->viewer, " ");
        }
        PetscViewerASCIIPrintf(this->viewer, "% 16.8E", array[d]);
      }
      PetscViewerASCIIPrintf(this->viewer, "\n");
      return true;
    };
  };

 public:
  PyLithViewer() {};
  virtual ~PyLithViewer() {};

  #undef __FUNCT__  
  #define __FUNCT__ "PyLithWriteVertices"
  static PetscErrorCode writeVertices(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer) {
    ALE::Obj<ALE::Mesh::field_type>   coordinates  = mesh->getCoordinates();
    ALE::Obj<ALE::Mesh::bundle_type>  vertexBundle = mesh->getBundle(0);
    ALE::Mesh::field_type::patch_type patch;
    const double  *array = coordinates->restrict(ALE::Mesh::field_type::patch_type());
    int            dim = mesh->getDimension();
    int            numVertices;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    //FIX:
    if (vertexBundle->getGlobalOffsets()) {
      numVertices = vertexBundle->getGlobalOffsets()[mesh->commSize()];
    } else {
      numVertices = mesh->getTopology()->depthStratum(0)->size();
    }
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"coord_units = km\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#  Node      X-coord           Y-coord           Z-coord\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      int numLocalVertices = mesh->getTopology()->depthStratum(0)->size();
      int vertexCount = 1;

      for(int v = 0; v < numLocalVertices; v++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%7D ", vertexCount++);CHKERRQ(ierr);
        for(int d = 0; d < dim; d++) {
          if (d > 0) {
            ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer,"% 16.8E", array[v*dim+d]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      for(int p = 1; p < mesh->commSize(); p++) {
        double    *remoteCoords;
        MPI_Status status;

        ierr = MPI_Recv(&numLocalVertices, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        ierr = PetscMalloc(numLocalVertices*dim * sizeof(double), &remoteCoords);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteCoords, numLocalVertices*dim, MPI_DOUBLE, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        for(int v = 0; v < numLocalVertices; v++) {
          ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", vertexCount++);CHKERRQ(ierr);
          for(int d = 0; d < dim; d++) {
            if (d > 0) {
              ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer, "% 16.8E", remoteCoords[v*dim+d]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
        }
      }
    } else {
      ALE::Obj<ALE::Mesh::bundle_type> globalOrder = coordinates->getGlobalOrder();
      ALE::Obj<ALE::Mesh::field_type::order_type::coneSequence> cone = globalOrder->getPatch(patch);
      const int *offsets = coordinates->getGlobalOffsets();
      int        numLocalVertices = (offsets[mesh->commRank()+1] - offsets[mesh->commRank()])/dim;
      double    *localCoords;
      int        k = 0;

      ierr = PetscMalloc(numLocalVertices*dim * sizeof(double), &localCoords);CHKERRQ(ierr);
      for(ALE::Mesh::field_type::order_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
        int dim = globalOrder->getFiberDimension(patch, *p_iter);

        if (dim > 0) {
          int offset = coordinates->getFiberOffset(patch, *p_iter);

          for(int i = offset; i < offset+dim; ++i) {
            localCoords[k++] = array[i];
          }
        }
      }
      if (k != numLocalVertices*dim) {
        SETERRQ2(PETSC_ERR_PLIB, "Invalid number of coordinates to send %d should be %d", k, numLocalVertices*dim);
      }
      ierr = MPI_Send(&numLocalVertices, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localCoords, numLocalVertices*dim, MPI_DOUBLE, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = PetscFree(localCoords);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "PyLithWriteElements"
  static PetscErrorCode writeElements(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer) {
    ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
    ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
    ALE::Obj<ALE::Mesh::field_type> material = mesh->getField("material");
    ALE::Obj<ALE::Mesh::bundle_type> elementBundle = mesh->getBundle(topology->depth());
    ALE::Obj<ALE::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
    ALE::Obj<ALE::Mesh::bundle_type> globalVertex = vertexBundle->getGlobalOrder();
    ALE::Obj<ALE::Mesh::bundle_type> globalElement = elementBundle->getGlobalOrder();
    ALE::Mesh::bundle_type::patch_type patch;
    std::string    orderName("element");
    int            dim  = mesh->getDimension();
    int            corners = topology->nCone(*elements->begin(), topology->depth())->size();
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (dim != 3) {
      SETERRQ(PETSC_ERR_SUP, "PyLith only supports 3D meshes.");
    }
    if (corners != 4) {
      SETERRQ(PETSC_ERR_SUP, "We only support linear tetrahedra for PyLith.");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#     N ETP MAT INF     N1     N2     N3     N4     N5     N6     N7     N8\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    if (mesh->commRank() == 0) {
      int elementCount = 1;

      for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
        ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

        // Only linear tetrahedra, material, no infinite elements
        ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, 5, (int) material->restrict(patch, *e_itor)[0], 0);CHKERRQ(ierr);
        for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
          ierr = PetscViewerASCIIPrintf(viewer, " %6d", globalVertex->getIndex(patch, *c_itor).prefix+1);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
      for(int p = 1; p < mesh->commSize(); p++) {
        int         numLocalElements;
        int        *remoteVertices;
        MPI_Status  status;

        ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        ierr = PetscMalloc(numLocalElements*(corners+1) * sizeof(int), &remoteVertices);CHKERRQ(ierr);
        ierr = MPI_Recv(remoteVertices, numLocalElements*(corners+1), MPI_INT, p, 1, mesh->comm(), &status);CHKERRQ(ierr);
        for(int e = 0; e < numLocalElements; e++) {
          // Only linear tetrahedra, material, no infinite elements
          int mat = remoteVertices[e*(corners+1)+corners];

          ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, 5, mat, 0);CHKERRQ(ierr);
          for(int c = 0; c < corners; c++) {
            ierr = PetscViewerASCIIPrintf(viewer, " %6d", remoteVertices[e*(corners+1)+c]);CHKERRQ(ierr);
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

      ierr = PetscMalloc(numLocalElements*(corners+1) * sizeof(int), &localVertices);CHKERRQ(ierr);
      for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
        ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

        if (globalElement->getFiberDimension(patch, *e_itor) > 0) {
          for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
            localVertices[k++] = globalVertex->getIndex(patch, *c_itor).prefix;
          }
          localVertices[k++] = (int) material->restrict(patch, *e_itor)[0];
        }
      }
      if (k != numLocalElements*corners) {
        SETERRQ2(PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", k, numLocalElements*corners);
      }
      ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = MPI_Send(localVertices, numLocalElements*(corners+1), MPI_INT, 0, 1, mesh->comm());CHKERRQ(ierr);
      ierr = PetscFree(localVertices);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "PyLithWriteVerticesLocal"
  static PetscErrorCode writeVerticesLocal(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer) {
    ALE::Obj<ALE::Mesh::field_type> coordinates = mesh->getCoordinates();
    ALE::Obj<ALE::Mesh::bundle_type> vertexBundle = ALE::Mesh::bundle_type(mesh->comm());
    ALE::Mesh::bundle_type::patch_type patch;
    int            dim = mesh->getDimension();
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"coord_units = km\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#  Node      X-coord           Y-coord           Z-coord\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);

    vertexBundle->setTopology(mesh->getTopology());
    vertexBundle->setPatch(mesh->getTopology()->leaves(), patch);
    vertexBundle->setFiberDimensionByDepth(patch, 0, 1);
    vertexBundle->orderPatches(vertexOutput(viewer, coordinates, dim));
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "PyLithWriteElementsLocal"
  static PetscErrorCode writeElementsLocal(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer) {
    ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
    ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
    ALE::Obj<ALE::Mesh::field_type> material = mesh->getField("material");
    ALE::Obj<ALE::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
    ALE::Mesh::bundle_type::patch_type patch;
    std::string    orderName("element");
    int            dim  = mesh->getDimension();
    int            corners = topology->nCone(*elements->begin(), topology->depth())->size();
    int            elementCount = 1;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (dim != 3) {
      SETERRQ(PETSC_ERR_SUP, "PyLith only supports 3D meshes.");
    }
    if (corners != 4) {
      SETERRQ(PETSC_ERR_SUP, "We only support linear tetrahedra for PyLith.");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#     N ETP MAT INF     N1     N2     N3     N4     N5     N6     N7     N8\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
    for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
      ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_itor);

      // Only linear tetrahedra, material, no infinite elements
      ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, 5, (int) material->restrict(patch, *e_itor)[0], 0);CHKERRQ(ierr);
      for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
        ierr = PetscViewerASCIIPrintf(viewer, " %6d", vertexBundle->getIndex(patch, *c_itor).prefix+1);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  };

  #undef __FUNCT__  
  #define __FUNCT__ "PyLithWriteSplitLocal"
  // The elements seem to be implicitly numbered by appearance, which makes it impossible to
  //   number here by bundle, but we can fix it by traversing the elements like the vertices
  static PetscErrorCode writeSplitLocal(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer) {
    ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
    ALE::Obj<ALE::Mesh::field_type> splitField = mesh->getField("split");
    ALE::Obj<ALE::Mesh::field_type::order_type::baseSequence> splitElements = splitField->getPatches();
    ALE::Obj<ALE::Mesh::bundle_type> elementBundle = mesh->getBundle(topology->depth());
    ALE::Obj<ALE::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
    ALE::Mesh::bundle_type::patch_type patch;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (mesh->getDimension() != 3) {
      SETERRQ(PETSC_ERR_SUP, "PyLith only supports 3D meshes.");
    }
    for(ALE::Mesh::field_type::order_type::baseSequence::iterator e_itor = splitElements->begin(); e_itor != splitElements->end(); ++e_itor) {
      const ALE::Mesh::bundle_type::index_type& idx = elementBundle->getIndex(patch, *e_itor);

      if (idx.index > 0) {
        ALE::Obj<ALE::Mesh::field_type::order_type::coneSequence> cone = splitField->getPatch(*e_itor);
        int e = idx.prefix+1;

        for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
          const double *values = splitField->restrict(*e_itor, *c_itor);
          int v = vertexBundle->getIndex(patch, *c_itor).prefix+1;

          // No time history
          ierr = PetscViewerASCIIPrintf(viewer, "%6d %6d 0 %15.9g %15.9g %15.9g\n", e, v, values[0], values[1], values[2]);CHKERRQ(ierr);
        }
      }
    }
    PetscFunctionReturn(0);
  };
};

