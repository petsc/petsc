
#include "src/dm/mesh/mesh.h"   /*I      "petscmesh.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVertices"
PetscErrorCode WritePyLithVertices(ALE::Obj<ALE::Two::Mesh> mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Two::Mesh::field_type> coordinates = mesh->getCoordinates();
  const double  *array = coordinates->restrict(ALE::Two::Mesh::field_type::patch_type());
  int            dim = mesh->getDimension();
  int            numVertices = mesh->getTopology()->depthStratum(0)->size();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"coord_units = km\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#  Node      X-coord           Y-coord           Z-coord\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  for(int v = 0; v < numVertices; v++) {
    ierr = PetscViewerASCIIPrintf(viewer,"%7D ", v+1);CHKERRQ(ierr);
    for(int d = 0; d < dim; d++) {
      if (d > 0) {
        ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"% 16.8E",array[v*dim+d]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElements"
PetscErrorCode WritePyLithElements(ALE::Obj<ALE::Two::Mesh> mesh, PetscViewer viewer)
{
  PetscMPIInt       rank = mesh->commRank();
  int               dim  = mesh->getDimension();
  ALE::Obj<ALE::Two::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::Two::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
  ALE::Obj<ALE::Two::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
  // FIX: Needs to be global
  int               corners = topology->nCone(*elements->begin(), topology->depth())->size();
  int               elementCount = 1;
  PetscErrorCode    ierr;

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
  if (rank == 0) {
    for(ALE::Two::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
      ALE::Obj<ALE::Two::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch("element", *e_itor);
      ALE::Two::Mesh::bundle_type::patch_type patch;

      // Only linear tetrahedra, 1 material, no infinite elements
      ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, 5, 1, 0);CHKERRQ(ierr);
      for(ALE::Two::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
        ierr = PetscViewerASCIIPrintf(viewer, " %6d", vertexBundle->getIndex(patch, *c_itor).prefix+1);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    }
#ifdef PARALLEL
    for(int p = 1; p < size; p++) {
      MPI_Status  status;
      int        *remoteVertices;
      int         numLocalElements;

      ierr = MPI_Recv(&numLocalElements, 1, MPI_INT, p, 1, comm, &status);CHKERRQ(ierr);
      ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &remoteVertices);CHKERRQ(ierr);
      ierr = MPI_Recv(remoteVertices, numLocalElements*corners, MPI_INT, p, 1, comm, &status);CHKERRQ(ierr);
      for(int e = 0; e < numLocalElements; e++) {
        // Only linear tetrahedra, 1 material, no infinite elements
        ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, 5, 1, 0);CHKERRQ(ierr);
        for(int c = 0; c < corners; c++) {
          ierr = PetscViewerASCIIPrintf(viewer, " %6d", remoteVertices[e*corners+c]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      }
      ierr = PetscFree(remoteVertices);CHKERRQ(ierr);
    }
#endif
  } else {
#ifdef PARALLEL
    int  numLocalElements = elements.size(), offset = 0;
    int *array;

    ierr = PetscMalloc(numLocalElements*corners * sizeof(int), &array);CHKERRQ(ierr);
    for(ALE::Point_set::iterator e_itor = elements.begin(); e_itor != elements.end(); e_itor++) {
      ALE::Obj<ALE::Point_array> intervals = vertexBundle.getGlobalOrderedClosureIndices(orientation->cone(*e_itor));

      for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
        array[offset++] = (*i_itor).prefix+1;
      }
    }
    if (offset != numLocalElements*corners) {
      SETERRQ2(PETSC_ERR_PLIB, "Invalid number of vertices to send %d should be %d", offset, numLocalElements*corners);
    }
    ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = MPI_Send(array, numLocalElements*corners, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = PetscFree(array);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

static int testerCount = 0;

struct outputTester {
  PetscViewer viewer;
  ALE::Obj<ALE::Two::Mesh::field_type> coordinates;
  int dim;
  int rank;
  public:
  outputTester(PetscViewer viewer, ALE::Obj<ALE::Two::Mesh::field_type> coordinates, int dim, int rank) : viewer(viewer), coordinates(coordinates), dim(dim), rank(rank) {testerCount = 0;};
  bool operator()(const ALE::Two::Mesh::point_type& p) const {
    ALE::Two::Mesh::field_type::patch_type patch;
    const double  *array = coordinates->restrict(patch, p);

    std::cout << "[" <<rank<< "]: Writing " << testerCount << " vertex " << p << std::endl;
    PetscViewerASCIIPrintf(this->viewer,"%7D ", 1 + testerCount++);
    for(int d = 0; d < dim; d++) {
      if (d > 0) {
        PetscViewerASCIIPrintf(this->viewer," ");
      }
      PetscViewerASCIIPrintf(this->viewer,"% 16.8E",array[d]);
    }
    PetscViewerASCIIPrintf(this->viewer,"\n");
    return true;
  };
};

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVerticesLocal"
PetscErrorCode WritePyLithVerticesLocal(ALE::Obj<ALE::Two::Mesh> mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Two::Mesh::field_type> coordinates = mesh->getCoordinates();
  int            dim = mesh->getDimension();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"coord_units = km\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#  Node      X-coord           Y-coord           Z-coord\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
#if 0
  ALE::Obj<ALE::Two::Mesh::sieve_type::traits::depthSequence> vertices = mesh->getTopology()->depthStratum(0);
  const double  *array = coordinates->restrict(ALE::Two::Mesh::field_type::patch_type());
  int            numVertices = mesh->getTopology()->depthStratum(0)->size();
  int            count = 0;
  for(ALE::Two::Mesh::sieve_type::traits::depthSequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    std::cout << "[" <<mesh->commRank()<< "]: Writing " << count++ << " vertex " << *v_iter << std::endl;
  }
  for(int v = 0; v < numVertices; v++) {
    ierr = PetscViewerASCIIPrintf(viewer,"%7D ", v+1);CHKERRQ(ierr);
    for(int d = 0; d < dim; d++) {
      if (d > 0) {
        ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"% 16.8E",array[v*dim+d]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }
#else
  ALE::Obj<ALE::Two::Mesh::bundle_type> vertexBundle = ALE::Two::Mesh::bundle_type(mesh->comm());
  ALE::Two::Mesh::bundle_type::patch_type patch;

  vertexBundle->setTopology(mesh->getTopology());
  vertexBundle->setPatch(mesh->getTopology()->leaves(), patch);
  vertexBundle->setFiberDimensionByDepth(patch, 0, 1);
  vertexBundle->orderPatches(outputTester(viewer, coordinates, dim, mesh->commRank()));
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElementsLocal"
PetscErrorCode WritePyLithElementsLocal(ALE::Obj<ALE::Two::Mesh> mesh, PetscViewer viewer)
{
  int            dim  = mesh->getDimension();
  ALE::Obj<ALE::Two::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::Two::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
  ALE::Obj<ALE::Two::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
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
  for(ALE::Two::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
    ALE::Obj<ALE::Two::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch("element", *e_itor);
    ALE::Two::Mesh::bundle_type::patch_type patch;

    // Only linear tetrahedra, 1 material, no infinite elements
    std::cout << "[" <<mesh->commRank()<< "]: Writing " << elementCount << " element " << *e_itor << std::endl;
    std::cout << "[" <<mesh->commRank()<< "]:   ";
    ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, 5, 1, 0);CHKERRQ(ierr);
    for(ALE::Two::Mesh::bundle_type::order_type::coneSequence::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
      std::cout << " " << *c_itor << "(" << vertexBundle->getIndex(patch, *c_itor).prefix+1 << ")";
      ierr = PetscViewerASCIIPrintf(viewer, " %6d", vertexBundle->getIndex(patch, *c_itor).prefix+1);CHKERRQ(ierr);
    }
    std::cout << std::endl;
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
