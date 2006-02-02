#define PETSCDM_DLL
 
#include "petscmesh.h"   /*I      "petscmesh.h"   I*/
#include "petscmat.h"    /*I      "petscmat.h"    I*/


typedef struct _MeshOps *MeshOps;
struct _MeshOps {
  PetscErrorCode (*view)(Mesh,PetscViewer);
  PetscErrorCode (*createglobalvector)(Mesh,Vec*);
  PetscErrorCode (*getcoloring)(Mesh,ISColoringType,ISColoring*);
  PetscErrorCode (*getmatrix)(Mesh,MatType,Mat*);
  PetscErrorCode (*getinterpolation)(Mesh,Mesh,Mat*,Vec*);
  PetscErrorCode (*refine)(Mesh,MPI_Comm,Mesh*);
};

struct _p_Mesh {
  PETSCHEADER(struct _MeshOps);
  ALE::Obj<ALE::Two::Mesh> m;
  Vec      globalvector;
  PetscInt bs,n,N,Nghosts,*ghosts;
  PetscInt d_nz,o_nz,*d_nnz,*o_nnz;
};

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKHeader"
PetscErrorCode WriteVTKHeader(PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Simplicial Mesh Example\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"ASCII\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"DATASET UNSTRUCTURED_GRID\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKVertices"
PetscErrorCode WriteVTKVertices(ALE::Obj<ALE::def::Mesh> mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::def::Mesh::coordinate_type> coordinates = mesh->getCoordinates();
  const double  *array = coordinates->restrict(0);
  int            dim = mesh->getDimension();
  int            numVertices = mesh->getTopology()->depthStratum(0)->size();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"POINTS %d double\n", numVertices);CHKERRQ(ierr);
  for(int v = 0; v < numVertices; v++) {
    for(int d = 0; d < dim; d++) {
      if (d > 0) {
        ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"%G",array[v*dim+d]);CHKERRQ(ierr);
    }
    for(int d = dim; d < 3; d++) {
      ierr = PetscViewerASCIIPrintf(viewer," 0.0");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKVertices_New"
PetscErrorCode WriteVTKVertices_New(ALE::Obj<ALE::Two::Mesh> mesh, PetscViewer viewer)
{
  const double  *array = mesh->getCoordinates()->restrict(ALE::Two::Mesh::field_type::patch_type());
  int            dim = mesh->getDimension();
  int            numVertices = mesh->getTopology()->depthStratum(0)->size();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer, "POINTS %d double\n", numVertices);CHKERRQ(ierr);
  for(int v = 0; v < numVertices; v++) {
    for(int d = 0; d < dim; d++) {
      if (d > 0) {
        ierr = PetscViewerASCIIPrintf(viewer, " ");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "%G", array[v*dim+d]);CHKERRQ(ierr);
    }
    for(int d = dim; d < 3; d++) {
      ierr = PetscViewerASCIIPrintf(viewer, " 0.0");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKElements"
PetscErrorCode WriteVTKElements(ALE::Obj<ALE::def::Mesh> mesh, PetscViewer viewer)
{
  MPI_Comm          comm = mesh->getComm();
  ALE::Obj<ALE::def::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::def::Mesh::sieve_type::heightSequence> elements = topology->heightStratum(0);
  // FIX: Needs to be global
  int               numElements = elements->size();
  int               corners = topology->nCone(*elements->begin(), topology->depth())->size();
  ALE::Obj<ALE::def::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
  PetscMPIInt       rank, size;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);
  ierr = MPI_Comm_size(comm, &size);
  ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
  if (rank == 0) {
    for(ALE::def::Mesh::sieve_type::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
      ALE::Obj<ALE::def::PointArray> cone = topology->nCone(*e_itor, topology->depth());

      ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
      for(ALE::def::PointArray::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
        ierr = PetscViewerASCIIPrintf(viewer, " %d", vertexBundle->getIndex(0, *c_itor).prefix);CHKERRQ(ierr);
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
        ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
        for(int c = 0; c < corners; c++) {
          ierr = PetscViewerASCIIPrintf(viewer, " %d", remoteVertices[e*corners+c]);CHKERRQ(ierr);
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
      ALE::Point_set cone = topology->nCone(*e_itor, dim);
      for(ALE::Point_set::iterator c_itor = cone.begin(); c_itor != cone.end(); c_itor++) {
        ALE::Point index = vertexBundle->getGlobalFiberInterval(*c_itor);

        array[offset++] = index.prefix;
      }
    }
    if (offset != numLocalElements*corners) {
      SETERRQ2(PETSC_ERR_PLIB, "Invalid number of vertices to send %d shuold be %d", offset, numLocalElements*corners);
    }
    ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = MPI_Send(array, numLocalElements*corners, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = PetscFree(array);CHKERRQ(ierr);
#endif
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
    // VTK_TETRA
    for(int e = 0; e < numElements; e++) {
      ierr = PetscViewerASCIIPrintf(viewer, "10\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKElements_New"
PetscErrorCode WriteVTKElements_New(ALE::Obj<ALE::Two::Mesh> mesh, PetscViewer viewer)
{
  MPI_Comm          comm = mesh->getComm();
  ALE::Obj<ALE::Two::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::Two::Mesh::sieve_type::heightSequence> elements = topology->heightStratum(0);
  // FIX: Needs to be global
  int               numElements = elements->size();
  int               corners = topology->nCone(*elements->begin(), topology->depth())->size();
  ALE::Obj<ALE::Two::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
  PetscMPIInt       rank, size;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);
  ierr = MPI_Comm_size(comm, &size);
  ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
  if (rank == 0) {
    for(ALE::Two::Mesh::sieve_type::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
      ALE::Obj<ALE::def::PointArray> cone = topology->nCone(*e_itor, topology->depth());
      ALE::Two::Mesh::bundle_type::patch_type patch;

      ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
      for(ALE::def::PointArray::iterator c_itor = cone->begin(); c_itor != cone->end(); ++c_itor) {
        ierr = PetscViewerASCIIPrintf(viewer, " %d", vertexBundle->getIndex(patch, *c_itor).prefix);CHKERRQ(ierr);
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
        ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
        for(int c = 0; c < corners; c++) {
          ierr = PetscViewerASCIIPrintf(viewer, " %d", remoteVertices[e*corners+c]);CHKERRQ(ierr);
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
      ALE::Point_set cone = topology->nCone(*e_itor, dim);
      for(ALE::Point_set::iterator c_itor = cone.begin(); c_itor != cone.end(); c_itor++) {
        ALE::Point index = vertexBundle->getGlobalFiberInterval(*c_itor);

        array[offset++] = index.prefix;
      }
    }
    if (offset != numLocalElements*corners) {
      SETERRQ2(PETSC_ERR_PLIB, "Invalid number of vertices to send %d shuold be %d", offset, numLocalElements*corners);
    }
    ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = MPI_Send(array, numLocalElements*corners, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = PetscFree(array);CHKERRQ(ierr);
#endif
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
    // VTK_TETRA
    for(int e = 0; e < numElements; e++) {
      ierr = PetscViewerASCIIPrintf(viewer, "10\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEVertices"
PetscErrorCode WritePCICEVertices(ALE::Obj<ALE::def::Mesh> mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::def::Mesh::coordinate_type> coordinates = mesh->getCoordinates();
  const double  *array = coordinates->restrict(0);
  int            dim = mesh->getDimension();
  int            numVertices = mesh->getTopology()->depthStratum(0)->size();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"%D\n", numVertices);CHKERRQ(ierr);
  for(int v = 0; v < numVertices; v++) {
    ierr = PetscViewerASCIIPrintf(viewer,"%7D   ", v+1);CHKERRQ(ierr);
    for(int d = 0; d < dim; d++) {
      if (d > 0) {
        ierr = PetscViewerASCIIPrintf(viewer," ");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"% 12.5E",array[v*dim+d]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEElements"
PetscErrorCode WritePCICEElements(ALE::Obj<ALE::def::Mesh> mesh, PetscViewer viewer)
{
  MPI_Comm          comm = mesh->getComm();
  int               dim  = mesh->getDimension();
  ALE::Obj<ALE::def::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::def::Mesh::sieve_type> orientation = mesh->getOrientation();
  ALE::Obj<ALE::def::Mesh::sieve_type::heightSequence> elements = topology->heightStratum(0);
  ALE::Obj<ALE::def::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
  // FIX: Needs to be global
  int               numElements = elements->size();
  int               corners = topology->nCone(*elements->begin(), dim)->size();
  PetscMPIInt       rank, size;
  int               elementCount = 1;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);
  ierr = MPI_Comm_size(comm, &size);
  if (corners != dim+1) {
    SETERRQ(PETSC_ERR_SUP, "PCICE only supports simplicies");
  }
  if (rank == 0) {
    ierr = PetscViewerASCIIPrintf(viewer, "%d\n", numElements);CHKERRQ(ierr);
    for(ALE::def::Mesh::sieve_type::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
      // FIX: Need to globalize
      ALE::Obj<ALE::def::Mesh::bundle_type::IndexArray> intervals = vertexBundle->getOrderedIndices(0, orientation->cone(*e_itor));

      ierr = PetscViewerASCIIPrintf(viewer, "%7d", elementCount++);CHKERRQ(ierr);
      for(ALE::def::Mesh::bundle_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
        ierr = PetscViewerASCIIPrintf(viewer, " %7d", i_itor->prefix+1);CHKERRQ(ierr);
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
        ierr = PetscViewerASCIIPrintf(viewer, "%7d", elementCount++);CHKERRQ(ierr);
        for(int c = 0; c < corners; c++) {
          ierr = PetscViewerASCIIPrintf(viewer, " %7d", remoteVertices[e*corners+c]);CHKERRQ(ierr);
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
      SETERRQ2(PETSC_ERR_PLIB, "Invalid number of vertices to send %d shuold be %d", offset, numLocalElements*corners);
    }
    ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = MPI_Send(array, numLocalElements*corners, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = PetscFree(array);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVertices"
PetscErrorCode WritePyLithVertices(ALE::Obj<ALE::def::Mesh> mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::def::Mesh::coordinate_type> coordinates = mesh->getCoordinates();
  const double  *array = coordinates->restrict(0);
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
PetscErrorCode WritePyLithElements(ALE::Obj<ALE::def::Mesh> mesh, PetscViewer viewer)
{
  MPI_Comm          comm = mesh->getComm();
  int               dim  = mesh->getDimension();
  ALE::Obj<ALE::def::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::def::Mesh::sieve_type> orientation = mesh->getOrientation();
  ALE::Obj<ALE::def::Mesh::sieve_type::heightSequence> elements = topology->heightStratum(0);
  ALE::Obj<ALE::def::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
  // FIX: Needs to be global
  int               corners = topology->nCone(*elements->begin(), topology->depth())->size();
  PetscMPIInt       rank, size;
  int               elementCount = 1;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);
  ierr = MPI_Comm_size(comm, &size);
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
    for(ALE::def::Mesh::sieve_type::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
      // FIX: Need to globalize
      ALE::Obj<ALE::def::Mesh::bundle_type::IndexArray> intervals = vertexBundle->getOrderedIndices(0, orientation->cone(*e_itor));

      // Only linear tetrahedra, 1 material, no infinite elements
      ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, 5, 1, 0);CHKERRQ(ierr);
      for(ALE::def::Mesh::bundle_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
        ierr = PetscViewerASCIIPrintf(viewer, " %6d", (*i_itor).prefix+1);CHKERRQ(ierr);
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
      SETERRQ2(PETSC_ERR_PLIB, "Invalid number of vertices to send %d shuold be %d", offset, numLocalElements*corners);
    }
    ierr = MPI_Send(&numLocalElements, 1, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = MPI_Send(array, numLocalElements*corners, MPI_INT, 0, 1, comm);CHKERRQ(ierr);
    ierr = PetscFree(array);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVerticesLocal"
PetscErrorCode WritePyLithVerticesLocal(ALE::Obj<ALE::def::Mesh> mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::def::Mesh::coordinate_type> coordinates = mesh->getCoordinates();
  const double  *array = coordinates->restrict(0);
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
#define __FUNCT__ "WritePyLithElementsLocal"
PetscErrorCode WritePyLithElementsLocal(ALE::Obj<ALE::def::Mesh> mesh, PetscViewer viewer)
{
  int            dim  = mesh->getDimension();
  ALE::Obj<ALE::def::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::def::Mesh::sieve_type> orientation = mesh->getOrientation();
  ALE::Obj<ALE::def::Mesh::sieve_type::heightSequence> elements = topology->heightStratum(0);
  ALE::Obj<ALE::def::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
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
  for(ALE::def::Mesh::sieve_type::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
    ALE::Obj<ALE::def::Mesh::bundle_type::IndexArray> intervals = vertexBundle->getOrderedIndices(0, orientation->cone(*e_itor));

    // Only linear tetrahedra, 1 material, no infinite elements
    ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, 5, 1, 0);CHKERRQ(ierr);
    for(ALE::def::Mesh::bundle_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
      ierr = PetscViewerASCIIPrintf(viewer, " %6d", (*i_itor).prefix+1);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve_Ascii"
PetscErrorCode MeshView_Sieve_Ascii(ALE::Obj<ALE::def::Mesh> mesh, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK) {
    ierr = WriteVTKHeader(viewer);CHKERRQ(ierr);
    ierr = WriteVTKVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = WriteVTKElements(mesh, viewer);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_PCICE) {
    char      *filename;
    char       coordFilename[2048];
    PetscTruth isConnect;
    size_t     len;

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);
    ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
    ierr = PetscStrcmp(&(filename[len-5]), ".lcon", &isConnect);CHKERRQ(ierr);
    if (!isConnect) {
      SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid element connectivity filename: %s", filename);
    }
    ierr = WritePCICEElements(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscStrncpy(coordFilename, filename, len-5);CHKERRQ(ierr);
    coordFilename[len-5] = '\0';
    ierr = PetscStrcat(coordFilename, ".nodes");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, coordFilename);CHKERRQ(ierr);
    ierr = WritePCICEVertices(mesh, viewer);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_PYLITH) {
    char *filename;
    char  connectFilename[2048];
    char  coordFilename[2048];

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscStrcpy(connectFilename, filename);CHKERRQ(ierr);
    ierr = PetscStrcat(connectFilename, ".connect");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, connectFilename);CHKERRQ(ierr);
    ierr = WritePyLithElements(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscStrcpy(coordFilename, filename);CHKERRQ(ierr);
    ierr = PetscStrcat(coordFilename, ".coord");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, coordFilename);CHKERRQ(ierr);
    ierr = WritePyLithVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscExceptionTry1(PetscViewerFileSetName(viewer, filename), PETSC_ERR_FILE_OPEN);
    if (PetscExceptionValue(ierr)) {
      /* this means that a caller above me has also tryed this exception so I don't handle it here, pass it up */
    } else if (PetscExceptionCaught(ierr, PETSC_ERR_FILE_OPEN)) {
      ierr = 0;
    } 
    CHKERRQ(ierr);
#if 0
  } else if (format == PETSC_VIEWER_ASCII_PYLITH_LOCAL) {
    PetscViewer connectViewer, coordViewer;
    char       *filename;
    char        localFilename[2048];
    MPI_Comm    comm;
    PetscMPIInt rank;

    ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);

    sprintf(localFilename, "%s.%d.connect", filename, rank);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &connectViewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(connectViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(connectViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(connectViewer, localFilename);CHKERRQ(ierr);
    ierr = WritePyLithElementsLocal(mesh, connectViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(connectViewer);CHKERRQ(ierr);

    sprintf(localFilename, "%s.%d.coord", filename, rank);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &coordViewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(coordViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(coordViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(coordViewer, localFilename);CHKERRQ(ierr);
    ierr = WritePyLithVerticesLocal(mesh, coordViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(coordViewer);CHKERRQ(ierr);
#endif
  } else {
    int dim = mesh->getDimension();

    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    for(int d = 0; d <= dim; d++) {
      // FIX: Need to globalize
      ierr = PetscViewerASCIIPrintf(viewer, "  %d %d-cells\n", mesh->getTopology()->depthStratum(d)->size(), d);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve_Ascii"
PetscErrorCode MeshView_Sieve_Ascii(ALE::Obj<ALE::Two::Mesh> mesh, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK) {
    ierr = WriteVTKHeader(viewer);CHKERRQ(ierr);
    ierr = WriteVTKVertices_New(mesh, viewer);CHKERRQ(ierr);
    ierr = WriteVTKElements_New(mesh, viewer);CHKERRQ(ierr);
  } else {
    int dim = mesh->getDimension();

    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    for(int d = 0; d <= dim; d++) {
      // FIX: Need to globalize
      ierr = PetscViewerASCIIPrintf(viewer, "  %d %d-cells\n", mesh->getTopology()->depthStratum(d)->size(), d);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve_Newer"
PetscErrorCode MeshView_Sieve_Newer(ALE::Obj<ALE::Two::Mesh> mesh, PetscViewer viewer)
{
  PetscTruth     iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_DRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ierr = MeshView_Sieve_Ascii(mesh, viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    SETERRQ(PETSC_ERR_SUP, "Binary viewer not implemented for Mesh");
  } else if (isdraw){ 
    SETERRQ(PETSC_ERR_SUP, "Draw viewer not implemented for Mesh");
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve_New"
PetscErrorCode MeshView_Sieve_New(ALE::Obj<ALE::def::Mesh> mesh, PetscViewer viewer)
{
  PetscTruth     iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_DRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ierr = MeshView_Sieve_Ascii(mesh, viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    SETERRQ(PETSC_ERR_SUP, "Binary viewer not implemented for Mesh");
  } else if (isdraw){ 
    SETERRQ(PETSC_ERR_SUP, "Draw viewer not implemented for Mesh");
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve"
PetscErrorCode MeshView_Sieve(Mesh mesh, PetscViewer viewer)
{
  PetscTruth     iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_DRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    SETERRQ(PETSC_ERR_SUP, "Ascii viewer not implemented for Mesh");
  } else if (isbinary) {
    SETERRQ(PETSC_ERR_SUP, "Binary viewer not implemented for Mesh");
  } else if (isdraw){ 
    SETERRQ(PETSC_ERR_SUP, "Draw viewer not implemented for Mesh");
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView"
/*@C
   MeshView - Views a Mesh object. 

   Collective on Mesh

   Input Parameters:
+  mesh - the mesh
-  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   You can change the format the mesh is printed using the 
   option PetscViewerSetFormat().

   The user can open alternative visualization contexts with
+    PetscViewerASCIIOpen() - Outputs mesh to a specified file
.    PetscViewerBinaryOpen() - Outputs mesh in binary to a
         specified file; corresponding input uses MeshLoad()
.    PetscViewerDrawOpen() - Outputs mesh to an X window display

   The user can call PetscViewerSetFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and PetscViewerASCIIOpen).  Available formats include
+    PETSC_VIEWER_ASCII_DEFAULT - default, prints mesh information
-    PETSC_VIEWER_ASCII_VTK - outputs a VTK file describing the mesh

   Level: beginner

   Concepts: mesh^printing
   Concepts: mesh^saving to disk

.seealso: PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerBinaryOpen(),
          MeshLoad(), PetscViewerCreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshView(Mesh mesh, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, DA_COOKIE, 1);
  PetscValidType(mesh, 1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(mesh->comm);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE, 2);
  PetscCheckSameComm(mesh, 1, viewer, 2);

  ierr = (*mesh->ops->view)(mesh, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshLoad" 
/*@C
    MeshLoad - Create a mesh topology from the saved data in a viewer.

    Collective on Viewer

    Input Parameter:
.   viewer - The viewer containing the data

    Output Parameters:
.   mesh - the mesh object

    Level: advanced

.seealso MeshView()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshLoad(PetscViewer viewer, Mesh *mesh)
{
  SETERRQ(PETSC_ERR_SUP, "");
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetMesh"
/*@C
    MeshGetMesh - Gets the internal mesh object

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    m - the internal mesh object
 
    Level: advanced

.seealso MeshCreate(), MeshSetMesh()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetMesh(Mesh mesh, ALE::Obj<ALE::Two::Mesh> *m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, DA_COOKIE, 1);
  if (m) {
    PetscValidPointer(m,2);
    *m = mesh->m;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetMesh"
/*@C
    MeshSetMesh - Sets the internal mesh object

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    boundary - the internal mesh object
 
    Level: advanced

.seealso MeshCreate(), MeshGetMesh()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetMesh(Mesh mesh, ALE::Obj<ALE::Two::Mesh> m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, DA_COOKIE, 1);
  mesh->m = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetMatrix" 
/*@C
    MeshGetMatrix - Creates a matrix with the correct parallel layout required for 
      computing the Jacobian on a function defined using the informatin in Mesh.

    Collective on Mesh

    Input Parameter:
+   mesh - the mesh object
-   mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ,
            or any type which inherits from one of these (such as MATAIJ, MATLUSOL, etc.).

    Output Parameters:
.   J  - matrix with the correct nonzero preallocation
        (obviously without the correct Jacobian values)

    Level: advanced

    Notes: This properly preallocates the number of nonzeros in the sparse matrix so you 
       do not need to do it yourself.

.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate(), DASetBlockFills()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetMatrix(Mesh mesh, MatType mtype,Mat *J)
{
  ALE::Obj<ALE::Two::Mesh> m;
#if 0
  ISLocalToGlobalMapping lmap;
  PetscInt              *globals,rstart,i;
#endif
  PetscInt               localSize, globalSize;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  localSize = m->getField("u")->getSize(ALE::Two::Mesh::field_type::patch_type());
  globalSize = localSize;

  ierr = MatCreate(mesh->comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,localSize,localSize,globalSize,globalSize);CHKERRQ(ierr);
  ierr = MatSetType(*J,mtype);CHKERRQ(ierr);
  ierr = MatSetBlockSize(*J,1);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*J,mesh->d_nz,mesh->d_nnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*J,mesh->d_nz,mesh->d_nnz,mesh->o_nz,mesh->o_nnz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*J,mesh->bs,mesh->d_nz,mesh->d_nnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(*J,mesh->bs,mesh->d_nz,mesh->d_nnz,mesh->o_nz,mesh->o_nnz);CHKERRQ(ierr);

#if 0
  ierr = PetscMalloc((mesh->n+mesh->Nghosts+1)*sizeof(PetscInt),&globals);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*J,&rstart,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<mesh->n; i++) {
    globals[i] = rstart + i;
  }
  ierr = PetscMemcpy(globals+mesh->n,mesh->ghosts,mesh->Nghosts*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,mesh->n+mesh->Nghosts,globals,&lmap);CHKERRQ(ierr);
  ierr = PetscFree(globals);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,lmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(lmap);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MeshSetGhosts"
/*@C
    MeshSetGhosts - Sets the global indices of other processes elements that will
      be ghosts on this process

    Not Collective

    Input Parameters:
+    mesh - the Mesh object
.    bs - block size
.    nlocal - number of local (non-ghost) entries
.    Nghosts - number of ghosts on this process
-    ghosts - indices of all the ghost points

    Level: advanced

.seealso MeshDestroy(), MeshCreateGlobalVector(), MeshGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetGhosts(Mesh mesh,PetscInt bs,PetscInt nlocal,PetscInt Nghosts,const PetscInt ghosts[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(mesh,1);
  ierr = PetscFree(mesh->ghosts);CHKERRQ(ierr);
  ierr = PetscMalloc((1+Nghosts)*sizeof(PetscInt),&mesh->ghosts);CHKERRQ(ierr);
  ierr = PetscMemcpy(mesh->ghosts,ghosts,Nghosts*sizeof(PetscInt));CHKERRQ(ierr);
  mesh->bs      = bs;
  mesh->n       = nlocal;
  mesh->Nghosts = Nghosts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetPreallocation"
/*@C
    MeshSetPreallocation - sets the matrix memory preallocation for matrices computed by Mesh

    Not Collective

    Input Parameters:
+    mesh - the Mesh object
.    d_nz - maximum number of nonzeros in any row of diagonal block
.    d_nnz - number of nonzeros in each row of diagonal block
.    o_nz - maximum number of nonzeros in any row of off-diagonal block
.    o_nnz - number of nonzeros in each row of off-diagonal block


    Level: advanced

.seealso MeshDestroy(), MeshCreateGlobalVector(), MeshGetGlobalIndices(), MatMPIAIJSetPreallocation(),
         MatMPIBAIJSetPreallocation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetPreallocation(Mesh mesh,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  PetscFunctionBegin;
  PetscValidPointer(mesh,1);
  mesh->d_nz  = d_nz;
  mesh->d_nnz = (PetscInt*)d_nnz;
  mesh->o_nz  = o_nz;
  mesh->o_nnz = (PetscInt*)o_nnz;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreate"
/*@C
    MeshCreate - Creates a DM object, used to manage data for an unstructured problem
    described by a Sieve.

    Collective on MPI_Comm

    Input Parameter:
.   comm - the processors that will share the global vector

    Output Parameters:
.   mesh - the mesh object

    Level: advanced

.seealso MeshDestroy(), MeshCreateGlobalVector(), MeshGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreate(MPI_Comm comm,Mesh *mesh)
{
  PetscErrorCode ierr;
  Mesh         p;

  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  *mesh = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(p,_p_Mesh,struct _MeshOps,DA_COOKIE,0,"Mesh",comm,MeshDestroy,0);CHKERRQ(ierr);
  p->ops->view               = MeshView_Sieve;
  p->ops->createglobalvector = MeshCreateGlobalVector;
  p->ops->getmatrix          = MeshGetMatrix;

  ierr = PetscObjectChangeTypeName((PetscObject) p, "sieve");CHKERRQ(ierr);

  p->globalvector = PETSC_NULL;
  *mesh = p;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshDestroy"
/*@C
    MeshDestroy - Destroys a mesh.

    Collective on Mesh

    Input Parameter:
.   mesh - the mesh object

    Level: advanced

.seealso MeshCreate(), MeshCreateGlobalVector(), MeshGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshDestroy(Mesh mesh)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (--mesh->refct > 0) PetscFunctionReturn(0);
  if (mesh->globalvector) {ierr = VecDestroy(mesh->globalvector);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExpandInterval"
inline void ExpandInterval(ALE::Point interval, PetscInt indices[], PetscInt *indx)
{
  for(int i = 0; i < interval.index; i++) {
    indices[(*indx)++] = interval.prefix + i;
  }
}

#undef __FUNCT__
#define __FUNCT__ "ExpandInterval_New"
inline void ExpandInterval_New(ALE::def::Point interval, PetscInt indices[], PetscInt *indx)
{
  for(int i = 0; i < interval.index; i++) {
    indices[(*indx)++] = interval.prefix + i;
  }
}

#undef __FUNCT__
#define __FUNCT__ "ExpandIntervals"
PetscErrorCode ExpandIntervals(ALE::Obj<ALE::def::Mesh::bundle_type::IndexArray> intervals, PetscInt *indices)
{
  int k = 0;

  PetscFunctionBegin;
  for(ALE::def::Mesh::bundle_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    ExpandInterval_New(*i_itor, indices, &k);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateVector"
/*
  Creates a ghosted vector based upon the global ordering in the bundle.
*/
PetscErrorCode MeshCreateVector(Mesh mesh, ALE::Obj<ALE::Two::Mesh> m, Vec *v)
{
  ALE::Obj<ALE::Two::Mesh::field_type> field = m->getField("u");
  ALE::Two::Mesh::field_type::patch_type patch;
  // FIX: Must not include ghosts
  PetscInt       localSize = field->getSize(ALE::Two::Mesh::field_type::patch_type());
  MPI_Comm       comm = m->getComm();
  PetscMPIInt    rank = m->getRank();
  PetscInt      *ghostIndices = NULL;
  PetscInt       ghostSize = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef PARALLEL
  ALE::Obj<ALE::PreSieve> globalIndices = bundle->getGlobalIndices();
  ALE::Obj<ALE::PreSieve> pointTypes = bundle->getPointTypes();
  ALE::Obj<ALE::Point_set> rentedPoints = pointTypes->cone(ALE::Point(rank, ALE::rentedPoint));

  for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
    ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);

    if (cone->size()) {
      ALE::Point interval = *cone->begin();

      ghostSize += interval.index;
    }
  }
#endif
  if (ghostSize) {
    ierr = PetscMalloc(ghostSize * sizeof(PetscInt), &ghostIndices);CHKERRQ(ierr);
  }
#ifdef PARALLEL
  PetscInt ghostIdx = 0;

  for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
    ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);

    if (cone->size()) {
      ALE::Point interval = *cone->begin();

      // Must insert into ghostIndices at the index given by localIndices
      //   However, I think right now its correct because rentedPoints iterates in the same way in both methods
      ExpandInterval(interval, ghostIndices, &ghostIdx);
    }
  }
#endif
  ierr = VecCreateGhost(comm, localSize, PETSC_DETERMINE, ghostSize, ghostIndices, v);CHKERRQ(ierr);
  if (m->debug) {
    PetscInt globalSize, g;

    ierr = VecGetSize(*v, &globalSize);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Making an ordering over the vertices\n===============================\n");
    ierr = PetscSynchronizedPrintf(comm, "[%d]  global size: %d localSize: %d ghostSize: %d\n", rank, globalSize, localSize, ghostSize);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(comm, "[%d]  ghostIndices:", rank);CHKERRQ(ierr);
    for(g = 0; g < ghostSize; g++) {
      ierr = PetscSynchronizedPrintf(comm, "[%d] %d\n", rank, ghostIndices[g]);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedPrintf(comm, "\n");CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(ghostIndices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateGlobalVector"
/*@C
    MeshCreateGlobalVector - Creates a vector of the correct size to be gathered into 
        by the mesh.

    Collective on Mesh

    Input Parameter:
.    mesh - the mesh object

    Output Parameters:
.   gvec - the global vector

    Level: advanced

    Notes: Once this has been created you cannot add additional arrays or vectors to be packed.

.seealso MeshDestroy(), MeshCreate(), MeshGetGlobalIndices()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalVector(Mesh mesh,Vec *gvec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Turned off caching for this method so that bundle can be reset to make different vectors */
#if 0
  if (mesh->globalvector) {
    ierr = VecDuplicate(mesh->globalvector, gvec);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif
#ifdef __cplusplus
  ALE::Obj<ALE::Two::Mesh> m;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  ierr = MeshCreateVector(mesh, m, gvec);CHKERRQ(ierr);
#endif
#if 0
  mesh->globalvector = *gvec;
  ierr = PetscObjectReference((PetscObject) mesh->globalvector);CHKERRQ(ierr); 
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetGlobalIndices"
/*@C
    MeshGetGlobalIndices - Gets the global indices for all the local entries

    Collective on Mesh

    Input Parameter:
.    mesh - the mesh object

    Output Parameters:
.    idx - the individual indices for each packed vector/array
 
    Level: advanced

    Notes:
       The idx parameters should be freed by the calling routine with PetscFree()

.seealso MeshDestroy(), MeshCreateGlobalVector(), MeshCreate()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalIndices(Mesh mesh,PetscInt *idx[])
{
  SETERRQ(PETSC_ERR_SUP, "");
}

EXTERN PetscErrorCode assembleFullField(VecScatter, Vec, Vec, InsertMode);

#undef __FUNCT__
#define __FUNCT__ "restrictVector"
/*@
  restrictVector - Insert values from a global vector into a local ghosted vector

  Collective on g

  Input Parameters:
+ g - The global vector
. l - The local vector
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: beginner

.seealso: MatSetOption()
@*/
PetscErrorCode restrictVector(Vec g, Vec l, InsertMode mode)
{
  //VecScatter     injection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //ierr = PetscObjectQuery((PetscObject) g, "injection", (PetscObject *) &injection);CHKERRQ(ierr);
  //ierr = VecScatterBegin(g, l, mode, SCATTER_REVERSE, injection);
  //ierr = VecScatterEnd(g, l, mode, SCATTER_REVERSE, injection);
  if (mode == INSERT_VALUES) {
    ierr = VecCopy(g, l);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY(l, 1.0, g);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleVectorComplete"
/*@
  assembleVectorComplete - Insert values from a local ghosted vector into a global vector

  Collective on g

  Input Parameters:
+ g - The global vector
. l - The local vector
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: beginner

.seealso: MatSetOption()
@*/
PetscErrorCode assembleVectorComplete(Vec g, Vec l, InsertMode mode)
{
  //VecScatter     injection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //ierr = PetscObjectQuery((PetscObject) g, "injection", (PetscObject *) &injection);CHKERRQ(ierr);
  //ierr = VecScatterBegin(l, g, mode, SCATTER_FORWARD, injection);CHKERRQ(ierr);
  //ierr = VecScatterEnd(l, g, mode, SCATTER_FORWARD, injection);CHKERRQ(ierr);
  if (mode == INSERT_VALUES) {
    ierr = VecCopy(l, g);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY(g, 1.0, l);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleVector"
/*@
  assembleVector - Insert values into a vector

  Collective on A

  Input Parameters:
+ b - the vector
. e - The element number
. v - The values
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: beginner

.seealso: VecSetOption()
@*/
PetscErrorCode assembleVector(Vec b, PetscInt e, PetscScalar v[], InsertMode mode)
{
  Mesh                     mesh;
  ALE::Obj<ALE::Two::Mesh> m;
  ALE::Two::Mesh::field_type::patch_type patch;
  PetscInt                 firstElement;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) b, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  //firstElement = elementBundle->getLocalSizes()[bundle->getCommRank()];
  firstElement = 0;
  // Must relate b to field
  if (mode == INSERT_VALUES) {
    m->getField(std::string("x"))->update(patch, ALE::Two::Mesh::point_type(0, e + firstElement), v);
  } else {
    m->getField(std::string("x"))->updateAdd(patch, ALE::Two::Mesh::point_type(0, e + firstElement), v);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperator"
PetscErrorCode updateOperator(Mat A, ALE::Obj<ALE::Two::Mesh::field_type> field, const ALE::Two::Mesh::point_type& e, PetscScalar array[], InsertMode mode)
{
  ALE::Obj<ALE::Two::Mesh::field_type::IndexArray> intervals = field->getIndices("element", e);
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for(ALE::def::Mesh::bundle_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    numIndices += (*i_itor).index;
    if (0) {
      //printf("[%d]interval (%d, %d)\n", mesh->getCommRank(), (*i_itor).prefix, (*i_itor).index);
      printf("[%d]interval (%d, %d)\n", 0, (*i_itor).prefix, (*i_itor).index);
    }
  }
  if (indicesSize && (indicesSize != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
  }
  if (!indices) {
    indicesSize = numIndices;
    ierr = PetscMalloc(indicesSize * sizeof(PetscInt), &indices); CHKERRQ(ierr);
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  if (0) {
    for(int i = 0; i < numIndices; i++) {
      //printf("[%d]indices[%d] = %d\n", mesh->getCommRank(), i, indices[i]);
      printf("[%d]indices[%d] = %d\n", 0, i, indices[i]);
    }
  }
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode assembleOperator_New(Mat A, ALE::Obj<ALE::def::Mesh::coordinate_type> field, ALE::Obj<ALE::def::Mesh::sieve_type> orientation, ALE::def::Mesh::sieve_type::point_type e, PetscScalar array[], InsertMode mode)
{
  ALE::Obj<ALE::def::Mesh::bundle_type::IndexArray> intervals = field->getOrderedIndices(0, orientation->cone(e));
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  for(ALE::def::Mesh::bundle_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    numIndices += (*i_itor).index;
    if (0) {
      //printf("[%d]interval (%d, %d)\n", mesh->getCommRank(), (*i_itor).prefix, (*i_itor).index);
      printf("[%d]interval (%d, %d)\n", 0, (*i_itor).prefix, (*i_itor).index);
    }
  }
  if (indicesSize && (indicesSize != numIndices)) {
    ierr = PetscFree(indices); CHKERRQ(ierr);
    indices = NULL;
  }
  if (!indices) {
    indicesSize = numIndices;
    ierr = PetscMalloc(indicesSize * sizeof(PetscInt), &indices); CHKERRQ(ierr);
  }
  ierr = ExpandIntervals(intervals, indices); CHKERRQ(ierr);
  if (0) {
    for(int i = 0; i < numIndices; i++) {
      //printf("[%d]indices[%d] = %d\n", mesh->getCommRank(), i, indices[i]);
      printf("[%d]indices[%d] = %d\n", 0, i, indices[i]);
    }
  }
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, array, mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleMatrix"
/*@
  assembleMatrix - Insert values into a matrix

  Collective on A

  Input Parameters:
+ A - the matrix
. e - The element number
. v - The values
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: beginner

.seealso: MatSetOption()
@*/
PetscErrorCode assembleMatrix(Mat A, PetscInt e, PetscScalar v[], InsertMode mode)
{
  PetscObjectContainer c;
  ALE::def::Mesh      *mesh;
  int                  firstElement = 0;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) A, "mesh", (PetscObject *) &c);CHKERRQ(ierr);
  ierr = PetscObjectContainerGetPointer(c, (void **) &mesh);CHKERRQ(ierr);
  // Need to globalize
  // firstElement = elementBundle->getLocalSizes()[bundle->getCommRank()];
  try {
    ierr = assembleOperator_New(A, mesh->getField(), mesh->getOrientation(), ALE::def::Mesh::sieve_type::point_type(0, e + firstElement), v, mode);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  PetscFunctionReturn(0);
}
