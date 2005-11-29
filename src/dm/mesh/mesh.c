#define PETSCDM_DLL
 
#include "petscda.h"     /*I      "petscda.h"     I*/
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
  void    *topology;
  void    *boundary;
  void    *boundaryBundle;
  void    *orientation;
  void    *spaceFootprint;
  void    *bundle;
  void    *vertexBundle;
  void    *elementBundle;
  void    *coordBundle;
  Vec      coordinates;
  Vec      globalvector;
  PetscInt bs,n,N,Nghosts,*ghosts;
  PetscInt d_nz,o_nz,*d_nnz,*o_nnz;
};

#ifdef __cplusplus
#include <IndexBundle.hh>

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKHeader"
PetscErrorCode WriteVTKHeader(Mesh mesh, PetscViewer viewer)
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
PetscErrorCode WriteVTKVertices(Mesh mesh, PetscViewer viewer)
{
  Vec            coordinates;
  PetscInt       dim, numVertices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetDimension(mesh, &dim);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = VecGetSize(coordinates, &numVertices);CHKERRQ(ierr);
  numVertices /= dim;
  ierr = PetscViewerASCIIPrintf(viewer,"POINTS %d double\n", numVertices);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_COORDS);CHKERRQ(ierr);
  ierr = VecView(coordinates, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKElements"
PetscErrorCode WriteVTKElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Sieve       *topology;
  ALE::IndexBundle *elementBundle;
  ALE::Point_set    elements;
  int               dim, numElements, corners;
  MPI_Comm          comm;
  PetscMPIInt       rank, size;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  ierr = MPI_Comm_size(comm, &size);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetElementBundle(mesh, (void **) &elementBundle);CHKERRQ(ierr);
  ALE::IndexBundle *vertexBundle = new ALE::IndexBundle(topology);
  vertexBundle->setFiberDimensionByDepth(0, 1);
  vertexBundle->computeOverlapIndices();
  vertexBundle->computeGlobalIndices();
  elements = topology->heightStratum(0);
  dim = topology->depth(*elements.begin());
  numElements = elementBundle->getGlobalSize();
  corners = topology->nCone(*elements.begin(), dim).size();
  ierr = PetscViewerASCIIPrintf(viewer,"CELLS %d %d\n", numElements, numElements*(corners+1));CHKERRQ(ierr);
  if (rank == 0) {
    for(ALE::Point_set::iterator e_itor = elements.begin(); e_itor != elements.end(); e_itor++) {
      ierr = PetscViewerASCIIPrintf(viewer, "%d ", corners);CHKERRQ(ierr);
      ALE::Point_set cone = topology->nCone(*e_itor, dim);
      for(ALE::Point_set::iterator c_itor = cone.begin(); c_itor != cone.end(); c_itor++) {
        ALE::Point index = vertexBundle->getGlobalFiberInterval(*c_itor);

        ierr = PetscViewerASCIIPrintf(viewer, " %d", index.prefix);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    }
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
  } else {
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
PetscErrorCode WritePCICEVertices(Mesh mesh, PetscViewer viewer)
{
  Vec            coordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_PCICE);CHKERRQ(ierr);
  ierr = VecView(coordinates, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEElements"
PetscErrorCode WritePCICEElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Sieve       *topology;
  ALE::PreSieve    *orientation;
  ALE::IndexBundle *elementBundle;
  ALE::Point_set    elements;
  MPI_Comm          comm;
  PetscMPIInt       rank, size;
  int               dim, numElements, corners, elementCount = 1;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  ierr = MPI_Comm_size(comm, &size);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetElementBundle(mesh, (void **) &elementBundle);CHKERRQ(ierr);
  ALE::IndexBundle vertexBundle(topology);
  vertexBundle.setFiberDimensionByDepth(0, 1);
  vertexBundle.computeOverlapIndices();
  vertexBundle.computeGlobalIndices();
  elements = topology->heightStratum(0);
  numElements = elementBundle->getGlobalSize();
  dim = topology->depth(*elements.begin());
  corners = topology->nCone(*elements.begin(), dim).size();
  if (corners != dim+1) {
    SETERRQ(PETSC_ERR_SUP, "PCICE only supports simplicies");
  }
  if (rank == 0) {
    ierr = PetscViewerASCIIPrintf(viewer, "%d\n", numElements);CHKERRQ(ierr);
    for(ALE::Point_set::iterator e_itor = elements.begin(); e_itor != elements.end(); e_itor++) {
      ALE::Obj<ALE::Point_array> intervals = vertexBundle.getGlobalOrderedClosureIndices(orientation->cone(*e_itor));

      ierr = PetscViewerASCIIPrintf(viewer, "%7d", elementCount++);CHKERRQ(ierr);
      for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
        ierr = PetscViewerASCIIPrintf(viewer, " %7d", (*i_itor).prefix+1);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    }
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
  } else {
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
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVertices"
PetscErrorCode WritePyLithVertices(Mesh mesh, PetscViewer viewer)
{
  Vec            coordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"coord_units = km\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
  ierr = VecView(coordinates, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElements"
PetscErrorCode WritePyLithElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Sieve    *topology;
  ALE::PreSieve *orientation;
  ALE::Point_set elements;
  MPI_Comm       comm;
  PetscMPIInt    rank, size;
  int            dim, corners, elementCount = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  ierr = MPI_Comm_size(comm, &size);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ALE::IndexBundle vertexBundle(topology);
  vertexBundle.setFiberDimensionByDepth(0, 1);
  vertexBundle.computeOverlapIndices();
  vertexBundle.computeGlobalIndices();
  elements = topology->heightStratum(0);
  dim = topology->depth(*elements.begin());
  corners = topology->nCone(*elements.begin(), dim).size();
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
    for(ALE::Point_set::iterator e_itor = elements.begin(); e_itor != elements.end(); e_itor++) {
      ALE::Obj<ALE::Point_array> intervals = vertexBundle.getLocalOrderedClosureIndices(orientation->cone(*e_itor));

      // Only linear tetrahedra, 1 material, no infinite elements
      ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, 5, 1, 0);CHKERRQ(ierr);
      for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
        ierr = PetscViewerASCIIPrintf(viewer, " %6d", (*i_itor).prefix+1);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    }
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
  } else {
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
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVerticesLocal"
PetscErrorCode WritePyLithVerticesLocal(Mesh mesh, PetscViewer viewer)
{
  Vec            coordinates, locCoordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"coord_units = km\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = MeshGetCoordinates(mesh, &coordinates);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(coordinates, &locCoordinates);CHKERRQ(ierr);
  ierr = VecView(locCoordinates, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElementsLocal"
PetscErrorCode WritePyLithElementsLocal(Mesh mesh, PetscViewer viewer)
{
  ALE::Sieve    *topology;
  ALE::PreSieve *orientation;
  ALE::Point_set elements;
  MPI_Comm       comm;
  PetscMPIInt    rank, size;
  int            dim, corners, elementCount = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);
  ierr = MPI_Comm_size(comm, &size);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ALE::IndexBundle vertexBundle(topology);
  vertexBundle.setFiberDimensionByDepth(0, 1);
  vertexBundle.computeOverlapIndices();
  vertexBundle.computeGlobalIndices();
  elements = topology->heightStratum(0);
  dim = topology->depth(*elements.begin());
  corners = topology->nCone(*elements.begin(), dim).size();
  if (dim != 3) {
    SETERRQ(PETSC_ERR_SUP, "PyLith only supports 3D meshes.");
  }
  if (corners != 4) {
    SETERRQ(PETSC_ERR_SUP, "We only support linear tetrahedra for PyLith.");
  }
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#     N ETP MAT INF     N1     N2     N3     N4     N5     N6     N7     N8\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"#\n");CHKERRQ(ierr);
  for(ALE::Point_set::iterator e_itor = elements.begin(); e_itor != elements.end(); e_itor++) {
    ALE::Obj<ALE::Point_array> intervals = vertexBundle.getLocalOrderedClosureIndices(orientation->cone(*e_itor));

    // Only linear tetrahedra, 1 material, no infinite elements
    ierr = PetscViewerASCIIPrintf(viewer, "%7d %3d %3d %3d", elementCount++, 5, 1, 0);CHKERRQ(ierr);
    for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
      ierr = PetscViewerASCIIPrintf(viewer, " %6d", (*i_itor).prefix+1);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve_Ascii"
PetscErrorCode MeshView_Sieve_Ascii(Mesh mesh, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK) {
    ierr = WriteVTKHeader(mesh, viewer);CHKERRQ(ierr);
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
    char      *filename;
    char       coordFilename[2048];
    PetscTruth isConnect;
    size_t     len;

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);
    ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
    ierr = PetscStrcmp(&(filename[len-8]), ".connect", &isConnect);CHKERRQ(ierr);
    if (!isConnect) {
      SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid element connectivity filename: %s", filename);
    }
    ierr = WritePyLithElements(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscStrncpy(coordFilename, filename, len-8);CHKERRQ(ierr);
    coordFilename[len-8] = '\0';
    ierr = PetscStrcat(coordFilename, ".coord");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, coordFilename);CHKERRQ(ierr);
    ierr = WritePyLithVertices(mesh, viewer);CHKERRQ(ierr);
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
  } else {
    ALE::Sieve *topology;
    PetscInt dim, d;

    ierr = MeshGetDimension(mesh, &dim);CHKERRQ(ierr);
    ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    for(d = 0; d <= dim; d++) {
      ALE::IndexBundle dBundle(topology);

      dBundle.setFiberDimensionByDepth(d, 1);
      ierr = PetscViewerASCIIPrintf(viewer, "  %d %d-cells\n", dBundle.getGlobalSize(), d);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

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
#ifdef __cplusplus
    ierr = MeshView_Sieve_Ascii(mesh, viewer);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_ERR_SUP, "Ascii viewer not implemented for Mesh");
#endif
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
#if 0
  ISLocalToGlobalMapping lmap;
  PetscInt              *globals,rstart,i;
#endif
  PetscInt               localSize = 0, globalSize = 0;
  PetscErrorCode         ierr;

  PetscFunctionBegin;

#ifdef __cplusplus
  localSize = ((ALE::IndexBundle *) mesh->bundle)->getLocalSize();
  globalSize = ((ALE::IndexBundle *) mesh->bundle)->getGlobalSize();
#endif
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
  if (mesh->ghosts) {ierr = PetscFree(mesh->ghosts);CHKERRQ(ierr);}
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

  p->topology       = PETSC_NULL;
  p->boundary       = PETSC_NULL;
  p->boundaryBundle = PETSC_NULL;
  p->orientation    = PETSC_NULL;
  p->spaceFootprint = PETSC_NULL;
  p->bundle         = PETSC_NULL;
  p->elementBundle  = PETSC_NULL;
  p->coordBundle    = PETSC_NULL;
  p->coordinates    = PETSC_NULL;
  p->globalvector   = PETSC_NULL;
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
  if (mesh->spaceFootprint) {
    /* delete (ALE::Stack *) p->spaceFootprint; */
  }
  ierr = PetscHeaderDestroy(mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef __cplusplus

#undef __FUNCT__
#define __FUNCT__ "ExpandInterval"
/* This is currently duplicated in ex33mesh.c */
inline void ExpandInterval(ALE::Point interval, PetscInt indices[], PetscInt *indx)
{
  for(int i = 0; i < interval.index; i++) {
    indices[(*indx)++] = interval.prefix + i;
  }
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateVector"
/*
  Creates a ghosted vector based upon the global ordering in the bundle.
*/
PetscErrorCode MeshCreateVector(Mesh mesh, ALE::IndexBundle *bundle, int debug, Vec *v)
{
  MPI_Comm       comm;
  PetscMPIInt    rank = bundle->getCommRank();
  ALE::Obj<ALE::PreSieve> pointTypes = bundle->getPointTypes();
  ALE::Obj<ALE::PreSieve> globalIndices = bundle->getGlobalIndices();
  ALE::Obj<ALE::Point_set> rentedPoints = pointTypes->cone(ALE::Point(rank, ALE::rentedPoint));
  PetscInt      *ghostIndices, ghostSize = 0, ghostIdx = 0;
  PetscInt       localSize = bundle->getLocalSize();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
    ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);

    if (cone->size()) {
      ALE::Point interval = *cone->begin();

      ghostSize += interval.index;
    }
  }
  ierr = PetscMalloc(ghostSize * sizeof(PetscInt), &ghostIndices);CHKERRQ(ierr);
  for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
    ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);

    if (cone->size()) {
      ALE::Point interval = *cone->begin();

      // Must insert into ghostIndices at the index given by localIndices
      //   However, I think right now its correct because rentedPoints iterates in the same way in both methods
      ExpandInterval(interval, ghostIndices, &ghostIdx);
    }
  }
  ierr = VecCreateGhost(comm, localSize, PETSC_DETERMINE, ghostSize, ghostIndices, v);CHKERRQ(ierr);
  if (debug) {
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

#endif

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
  ierr = MeshCreateVector(mesh, (ALE::IndexBundle *) mesh->bundle, 0, gvec);CHKERRQ(ierr);
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

#undef __FUNCT__  
#define __FUNCT__ "MeshGetTopology"
/*@C
    MeshGetTopology - Gets the topology Sieve

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    topology - the topology Sieve
 
    Level: advanced

.seealso MeshCreate(), MeshSetTopology()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetTopology(Mesh mesh,void **topology)
{
  if (topology) {
    PetscValidPointer(topology,2);
    *topology = mesh->topology;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetTopology"
/*@C
    MeshSetTopology - Sets the topology Sieve

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    topology - the topology Sieve
 
    Level: advanced

.seealso MeshCreate(), MeshGetTopology()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetTopology(Mesh mesh,void *topology)
{
  PetscValidPointer(topology,2);
  mesh->topology = topology;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetBoundary"
/*@C
    MeshGetBoundary - Gets the boundary Sieve

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    boundary - the boundary Sieve
 
    Level: advanced

.seealso MeshCreate(), MeshSetBoundary()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetBoundary(Mesh mesh,void **boundary)
{
  if (boundary) {
    PetscValidPointer(boundary,2);
    *boundary = mesh->boundary;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetBoundary"
/*@C
    MeshSetBoundary - Sets the boundary Sieve

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    boundary - the boundary Sieve
 
    Level: advanced

.seealso MeshCreate(), MeshGetBoundary()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetBoundary(Mesh mesh,void *boundary)
{
  PetscValidPointer(boundary,2);
  mesh->boundary = boundary;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetBoundaryBundle"
/*@C
    MeshGetBoundaryBundle - Gets the Sieve boundary bundle

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    bundle - the Sieve boundary bundle
 
    Level: advanced

.seealso MeshCreate(), MeshSetBoundaryBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetBoundaryBundle(Mesh mesh,void **bundle)
{
  if (bundle) {
    PetscValidPointer(bundle,2);
    *bundle = mesh->boundaryBundle;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetBoundaryBundle"
/*@C
    MeshSetBoundaryBundle - Sets the Sieve boundary bundle

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    bundle - the Sieve boundary bundle
 
    Level: advanced

.seealso MeshCreate(), MeshGetBoundaryBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetBoundaryBundle(Mesh mesh,void *bundle)
{
  PetscValidPointer(bundle,2);
  mesh->boundaryBundle = bundle;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetBundle"
/*@C
    MeshGetBundle - Gets the Sieve bundle

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    bundle - the Sieve bundle
 
    Level: advanced

.seealso MeshCreate(), MeshSetBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetBundle(Mesh mesh,void **bundle)
{
  if (bundle) {
    PetscValidPointer(bundle,2);
    *bundle = mesh->bundle;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetBundle"
/*@C
    MeshSetBundle - Sets the Sieve bundle

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    bundle - the Sieve bundle
 
    Level: advanced

.seealso MeshCreate(), MeshGetBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetBundle(Mesh mesh,void *bundle)
{
  PetscValidPointer(bundle,2);
  mesh->bundle = bundle;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetVertexBundle"
/*@C
    MeshGetVertexBundle - Gets the vertex bundle

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    bundle - the vertex bundle
 
    Level: advanced

.seealso MeshCreate(), MeshSetVertexBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexBundle(Mesh mesh,void **bundle)
{
  if (bundle) {
    PetscValidPointer(bundle,2);
    *bundle = mesh->vertexBundle;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetVertexBundle"
/*@C
    MeshSetVertexBundle - Sets the vertex bundle

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    bundle - the vertex bundle
 
    Level: advanced

.seealso MeshCreate(), MeshGetVertexBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetVertexBundle(Mesh mesh,void *bundle)
{
  PetscValidPointer(bundle,2);
  mesh->vertexBundle = bundle;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetElementBundle"
/*@C
    MeshGetElementBundle - Gets the element bundle

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    bundle - the element bundle
 
    Level: advanced

.seealso MeshCreate(), MeshSetElementBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetElementBundle(Mesh mesh,void **bundle)
{
  if (bundle) {
    PetscValidPointer(bundle,2);
    *bundle = mesh->elementBundle;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetElementBundle"
/*@C
    MeshSetElementBundle - Sets the element bundle

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    bundle - the element bundle
 
    Level: advanced

.seealso MeshCreate(), MeshGetElementBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetElementBundle(Mesh mesh,void *bundle)
{
  PetscValidPointer(bundle,2);
  mesh->elementBundle = bundle;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetCoordinateBundle"
/*@C
    MeshGetCoordinateBundle - Gets the coordinate bundle

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    bundle - the coordinate bundle
 
    Level: advanced

.seealso MeshCreate(), MeshSetCoordinateBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetCoordinateBundle(Mesh mesh,void **bundle)
{
  if (bundle) {
    PetscValidPointer(bundle,2);
    *bundle = mesh->coordBundle;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetCoordinateBundle"
/*@C
    MeshSetCoordinateBundle - Sets the coordinate bundle

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    bundle - the coordinate bundle
 
    Level: advanced

.seealso MeshCreate(), MeshGetCoordinateBundle()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetCoordinateBundle(Mesh mesh,void *bundle)
{
  PetscValidPointer(bundle,2);
  mesh->coordBundle = bundle;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetOrientation"
/*@C
    MeshGetOrientation - Gets the orientation sieve

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    orientation - the orientation sieve
 
    Level: advanced

.seealso MeshCreate(), MeshSetOrientation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetOrientation(Mesh mesh,void **orientation)
{
  if (orientation) {
    PetscValidPointer(orientation,2);
    *orientation = mesh->orientation;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetOrientation"
/*@C
    MeshOrientation - Sets the orientation sieve

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    orientation - the orientation sieve
 
    Level: advanced

.seealso MeshCreate(), MeshGetOrientation()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetOrientation(Mesh mesh,void *orientation)
{
  PetscValidPointer(orientation,2);
  mesh->orientation = orientation;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetCoordinates"
/*@C
    MeshGetCoordinates - Gets the coordinate vector

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    coordinates - the coordinate vector
 
    Level: advanced

.seealso MeshCreate(), MeshSetCoordinates()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetCoordinates(Mesh mesh, Vec *coordinates)
{
  if (coordinates) {
    PetscValidPointer(coordinates,2);
    *coordinates = mesh->coordinates;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetCoordinates"
/*@C
    MeshCoordinates - Sets the coordinate vector

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    coordinates - the coordinate vector
 
    Level: advanced

.seealso MeshCreate(), MeshGetCoordinates()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetCoordinates(Mesh mesh, Vec coordinates)
{
  PetscValidPointer(coordinates,2);
  mesh->coordinates = coordinates;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetSpaceFootprint"
/*@C
    MeshGetSpaceFootprint - Gets the stack endcoding element overlap

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    spaceFootprint - the overlap stack
 
    Level: advanced

.seealso MeshCreate(), MeshSetSpaceFootprint()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetSpaceFootprint(Mesh mesh, void **spaceFootprint)
{
  if (spaceFootprint) {
    PetscValidPointer(spaceFootprint,2);
    *spaceFootprint = mesh->spaceFootprint;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetSpaceFootprint"
/*@C
    MeshSpaceFootprint - Sets the stack endcoding element overlap

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    spaceFootprint - the overlap stack
 
    Level: advanced

.seealso MeshCreate(), MeshGetSpaceFootprint()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetSpaceFootprint(Mesh mesh, void *spaceFootprint)
{
  PetscValidPointer(spaceFootprint,2);
  mesh->spaceFootprint = spaceFootprint;
  PetscFunctionReturn(0);
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
  VecScatter     injection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) g, "injection", (PetscObject *) &injection);CHKERRQ(ierr);
  ierr = VecScatterBegin(g, l, mode, SCATTER_REVERSE, injection);
  ierr = VecScatterEnd(g, l, mode, SCATTER_REVERSE, injection);
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
  VecScatter     injection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) g, "injection", (PetscObject *) &injection);CHKERRQ(ierr);
  ierr = VecScatterBegin(l, g, mode, SCATTER_FORWARD, injection);
  ierr = VecScatterEnd(l, g, mode, SCATTER_FORWARD, injection);
  PetscFunctionReturn(0);
}

extern int debug;
PetscErrorCode assembleField(ALE::IndexBundle *, ALE::PreSieve *, Vec, ALE::Point, PetscScalar[], InsertMode);

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
  Mesh              mesh;
  ALE::PreSieve    *orientation;
  ALE::IndexBundle *elementBundle;
  ALE::IndexBundle *bundle;
  PetscInt          firstElement;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) b, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetElementBundle(mesh, (void **) &elementBundle);CHKERRQ(ierr);
  ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
  firstElement = elementBundle->getLocalSizes()[bundle->getCommRank()];
  debug = 1;
  ierr = assembleField(bundle, orientation, b, ALE::Point(0, e + firstElement), v, mode);CHKERRQ(ierr);
  debug = 0;
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode assembleOperator(ALE::IndexBundle *, ALE::PreSieve *, Mat, ALE::Point, PetscScalar[], InsertMode);

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
  Mesh              mesh;
  ALE::PreSieve    *orientation;
  ALE::IndexBundle *elementBundle;
  ALE::IndexBundle *bundle;
  PetscInt          firstElement;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) A, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  ierr = MeshGetOrientation(mesh, (void **) &orientation);CHKERRQ(ierr);
  ierr = MeshGetElementBundle(mesh, (void **) &elementBundle);CHKERRQ(ierr);
  ierr = MeshGetBundle(mesh, (void **) &bundle);CHKERRQ(ierr);
  firstElement = elementBundle->getLocalSizes()[bundle->getCommRank()];
  ierr = assembleOperator(bundle, orientation, A, ALE::Point(0, e + firstElement), v, mode);CHKERRQ(ierr);
  if (e == 0) {
    bundle->getGlobalIndices()->view("Global indices");
    bundle->getLocalIndices()->view("Local indices");
  }
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_TRIANGLE

#include <triangle.h>

PetscErrorCode initGeneratorInput(struct triangulateio *inputCtx)
{
  PetscFunctionBegin;
  inputCtx->numberofpoints = 0;
  inputCtx->numberofpointattributes = 0;
  inputCtx->pointlist = NULL;
  inputCtx->pointattributelist = NULL;
  inputCtx->pointmarkerlist = NULL;
  inputCtx->numberofsegments = 0;
  inputCtx->segmentlist = NULL;
  inputCtx->segmentmarkerlist = NULL;
  inputCtx->numberoftriangleattributes = 0;
  inputCtx->numberofholes = 0;
  inputCtx->holelist = NULL;
  inputCtx->numberofregions = 0;
  inputCtx->regionlist = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode initGeneratorOutput(struct triangulateio *outputCtx)
{
  PetscFunctionBegin;
  outputCtx->pointlist = NULL;
  outputCtx->pointattributelist = NULL;
  outputCtx->pointmarkerlist = NULL;
  outputCtx->trianglelist = NULL;
  outputCtx->triangleattributelist = NULL;
  outputCtx->neighborlist = NULL;
  outputCtx->segmentlist = NULL;
  outputCtx->segmentmarkerlist = NULL;
  outputCtx->edgelist = NULL;
  outputCtx->edgemarkerlist = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode destroyGeneratorOutput(struct triangulateio *outputCtx)
{
  PetscFunctionBegin;
  free(outputCtx->pointmarkerlist);
  free(outputCtx->edgelist);
  free(outputCtx->edgemarkerlist);
  free(outputCtx->trianglelist);
  free(outputCtx->neighborlist);
  PetscFunctionReturn(0);
}

extern PetscErrorCode restrictField(ALE::IndexBundle *, ALE::PreSieve *, PetscScalar *, ALE::Point, PetscScalar *[]);

#undef __FUNCT__
#define __FUNCT__ "MeshGenerate_Triangle"
PetscErrorCode MeshGenerate_Triangle(Mesh boundary, Mesh *mesh)
{
  Mesh                 m;
  ALE::Sieve          *bdTopology;
  ALE::PreSieve       *bdOrientation;
  struct triangulateio in;
  struct triangulateio out;
  PetscInt             dim = 2;
  PetscMPIInt          rank;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(boundary->comm, &m);CHKERRQ(ierr);
  ierr = MeshGetTopology(boundary, (void **) &bdTopology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(boundary, (void **) &bdOrientation);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(boundary->comm, &rank);CHKERRQ(ierr);
  ierr = initGeneratorInput(&in);CHKERRQ(ierr);
  ierr = initGeneratorOutput(&out);CHKERRQ(ierr);
  ALE::IndexBundle vertexBundle(bdTopology);
  vertexBundle.setFiberDimensionByDepth(0, 1);
  if (rank == 0) {
    ALE::Sieve              *bdSieve;
    ALE::Obj<ALE::Point_set> vertices = bdTopology->depthStratum(0);
    ALE::Obj<ALE::Point_set> edges = bdTopology->depthStratum(1);
    char                    *args = (char *) "pqenzQ";
    PetscTruth               createConvexHull = PETSC_FALSE;

    ierr = MeshGetBoundary(boundary, (void **) &bdSieve);CHKERRQ(ierr);
    in.numberofpoints = vertices->size();
    if (in.numberofpoints > 0) {
      ALE::IndexBundle *coordBundle;
      Vec               coordinates;
      PetscScalar      *coords;

      ierr = MeshGetCoordinateBundle(boundary, (void **) &coordBundle);CHKERRQ(ierr);
      ierr = MeshGetCoordinates(boundary, &coordinates);CHKERRQ(ierr);
      ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
      ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);CHKERRQ(ierr);
      ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRQ(ierr);
      for(ALE::Point_set::iterator v_itor = vertices->begin(); v_itor != vertices->end(); v_itor++) {
        ALE::Point     vertex(*v_itor);
        ALE::Point     interval = coordBundle->getFiberInterval(vertex);
        ALE::Point_set support;
        PetscScalar   *array;

        ierr = restrictField(coordBundle, bdOrientation, coords, vertex, &array); CHKERRQ(ierr);
        for(int d = 0; d < interval.index; d++) {
          in.pointlist[interval.prefix + d] = array[d];
        }

        interval = vertexBundle.getFiberInterval(vertex);
        support  = bdSieve->support(vertex);
        if (support.size()) {
          in.pointmarkerlist[interval.prefix] = support.begin()->index;
        } else {
          in.pointmarkerlist[interval.prefix] = 0;
        }
      }
      ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    }

    in.numberofsegments = edges->size();
    if (in.numberofsegments > 0) {
      ALE::IndexBundle *bdElementBundle;

      ierr = MeshGetElementBundle(boundary, (void **) &bdElementBundle);CHKERRQ(ierr);
      ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);CHKERRQ(ierr);
      ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);CHKERRQ(ierr);
      for(ALE::Point_set::iterator e_itor = edges->begin(); e_itor != edges->end(); e_itor++) {
        ALE::Point               edge = *e_itor;
        ALE::Point               interval = bdElementBundle->getFiberInterval(edge);
        ALE::Obj<ALE::Point_set> cone = bdTopology->cone(edge);
        ALE::Point_set           support;
        PetscInt                 p = 0;
        
        for(ALE::Point_set::iterator c_itor = cone->begin(); c_itor != cone->end(); c_itor++) {
          in.segmentlist[interval.prefix * 2 + (p++)] = (*c_itor).index;
        }

        support = bdSieve->support(edge);
        if (support.size()) {
          in.segmentmarkerlist[interval.prefix] = support.begin()->index;
        } else {
          in.segmentmarkerlist[interval.prefix] = 0;
        }
      }
    }
    in.numberofholes = 0;
    if (in.numberofholes > 0) {
      ierr = PetscMalloc(in.numberofholes * dim * sizeof(int), &in.holelist);CHKERRQ(ierr);
    }
    if (createConvexHull) {
      char  *newArgs;
      size_t len;

      ierr = PetscStrlen(args, &len);CHKERRQ(ierr);
      ierr = PetscMalloc((strlen(args) + 2) * sizeof(char), &newArgs);CHKERRQ(ierr);
      ierr = PetscStrcpy(newArgs, args);CHKERRQ(ierr);
      ierr = PetscStrcat(newArgs, "c");CHKERRQ(ierr);
      args = newArgs;
    }
    triangulate((char *) args, &in, &out, NULL);
    if (createConvexHull) {
      ierr = PetscFree(args);CHKERRQ(ierr);
    }
    ierr = PetscFree(in.pointlist);CHKERRQ(ierr);
    ierr = PetscFree(in.pointmarkerlist);CHKERRQ(ierr);
    ierr = PetscFree(in.segmentlist);CHKERRQ(ierr);
    ierr = PetscFree(in.segmentmarkerlist);CHKERRQ(ierr);
  }
  ierr = MeshPopulate(m, dim, out.numberofpoints, out.numberoftriangles, out.trianglelist, out.pointlist);CHKERRQ(ierr);
  if (rank == 0) {
    ALE::Sieve *boundary = new ALE::Sieve(m->comm);
    ALE::Sieve *topology;

    ierr = MeshGetTopology(m, (void **) &topology);CHKERRQ(ierr);
    for(int v = 0; v < out.numberofpoints; v++) {
      if (out.pointmarkerlist[v]) {
        ALE::Point boundaryPoint(-1, out.pointmarkerlist[v]);
        ALE::Point point(0, v + out.numberoftriangles);

        boundary->addCone(point, boundaryPoint);
      }
    }
    for(int e = 0; e < out.numberofedges; e++) {
      if (out.edgemarkerlist[e]) {
        ALE::Point     boundaryPoint(-1, out.edgemarkerlist[e]);
        ALE::Point     endpointA(0, out.edgelist[e*2+0] + out.numberoftriangles);
        ALE::Point     endpointB(0, out.edgelist[e*2+1] + out.numberoftriangles);
        ALE::Point_set supportA = topology->support(endpointA);
        ALE::Point_set supportB = topology->support(endpointB);
        supportA.meet(supportB);
        ALE::Point     edge = *(supportA.begin());

        boundary->addCone(edge, boundaryPoint);
      }
    }
    ierr = MeshSetBoundary(m, (void *) boundary);CHKERRQ(ierr);
  }
  ierr = destroyGeneratorOutput(&out);CHKERRQ(ierr);
  *mesh = m;
  PetscFunctionReturn(0);
}

extern PetscErrorCode PartitionPreSieve(ALE::Obj<ALE::PreSieve>, const char *, bool localize, ALE::PreSieve **);
extern PetscErrorCode MeshCreateMapping(Mesh, ALE::IndexBundle *, ALE::PreSieve *, ALE::IndexBundle *, VecScatter *);

#undef __FUNCT__
#define __FUNCT__ "MeshRefine_Triangle"
PetscErrorCode MeshRefine_Triangle(Mesh oldMesh, PetscReal maxArea, /*CoSieve*/ Vec maxAreas, Mesh *mesh)
{
  Mesh                 m, serialMesh;
  ALE::Sieve          *serialTopology;
  ALE::PreSieve       *serialOrientation;
  ALE::Sieve          *serialBoundary;
  ALE::IndexBundle    *elementBundle, *serialElementBundle, *serialVertexBundle;
  ALE::IndexBundle    *serialCoordBundle;
  ALE::PreSieve       *partitionTypes;
  Vec                  serialCoordinates;
  PetscScalar         *coords;
  struct triangulateio in;
  struct triangulateio out;
  PetscReal            maxElementArea, *areas;
  PetscInt             numElements;
  PetscInt             dim = 2;
  PetscMPIInt          rank;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(oldMesh->comm, &m);CHKERRQ(ierr);
  ierr = MeshGetElementBundle(oldMesh, (void **) &elementBundle);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(oldMesh->comm, &rank);CHKERRQ(ierr);
  ierr = initGeneratorInput(&in);CHKERRQ(ierr);
  ierr = initGeneratorOutput(&out);CHKERRQ(ierr);
  ierr = MeshUnify(oldMesh, &serialMesh);CHKERRQ(ierr);
  ierr = MeshGetTopology(serialMesh, (void **) &serialTopology);CHKERRQ(ierr);
  ierr = MeshGetOrientation(serialMesh, (void **) &serialOrientation);CHKERRQ(ierr);
  ierr = MeshGetBoundary(serialMesh, (void **) &serialBoundary);CHKERRQ(ierr);
  ierr = MeshGetVertexBundle(serialMesh, (void **) &serialVertexBundle);CHKERRQ(ierr);
  ierr = MeshGetElementBundle(serialMesh, (void **) &serialElementBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinateBundle(serialMesh, (void **) &serialCoordBundle);CHKERRQ(ierr);
  ierr = MeshGetCoordinates(serialMesh, &serialCoordinates);CHKERRQ(ierr);

  numElements = elementBundle->getGlobalSize();
  if (maxArea > 0.0) {
    maxElementArea = maxArea;
  } else {
    /* TODO: Should be the volume of the bounding box */
    maxElementArea = 1.0;
  }
  if (rank == 0) {
    ierr = PetscMalloc(numElements * sizeof(double), &areas);CHKERRQ(ierr);
    in.trianglearealist = areas;
  }
  if (maxAreas) {
    Vec        locAreas;
    VecScatter areaScatter;

    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, numElements, areas, &locAreas);CHKERRQ(ierr);
    ierr = MeshCreateMapping(oldMesh, elementBundle, partitionTypes, serialElementBundle, &areaScatter);CHKERRQ(ierr);
    ierr = VecScatterBegin(maxAreas, locAreas, INSERT_VALUES, SCATTER_FORWARD, areaScatter);CHKERRQ(ierr);
    ierr = VecScatterEnd(maxAreas, locAreas, INSERT_VALUES, SCATTER_FORWARD, areaScatter);CHKERRQ(ierr);
    ierr = VecDestroy(locAreas);CHKERRQ(ierr);
    ierr = VecScatterDestroy(areaScatter);CHKERRQ(ierr);
  } else {
    if (rank == 0) {
      for(int e = 0; e < numElements; e++) {
        areas[e] = maxElementArea;
      }
    }
  }

  if (rank == 0) {
    ALE::Obj<ALE::Point_set> vertices = serialTopology->depthStratum(0);
    ALE::Obj<ALE::Point_set> edges = serialTopology->depthStratum(1);
    ALE::Obj<ALE::Point_set> faces = serialTopology->heightStratum(0);
    ALE::Obj<ALE::Point_set> boundaries = serialBoundary->base();
    char                    *args = (char *) "pqenzQ";
    char                    *newArgs;
    size_t                   len;
    int                      f = 0;

    in.numberofpoints = vertices->size();
    ierr = PetscMalloc(in.numberofpoints * dim * sizeof(double), &in.pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc(in.numberofpoints * sizeof(int), &in.pointmarkerlist);CHKERRQ(ierr);
    ierr = VecGetArray(serialCoordinates, &coords);CHKERRQ(ierr);
    for(ALE::Point_set::iterator v_itor = vertices->begin(); v_itor != vertices->end(); v_itor++) {
      ALE::Point     vertex(*v_itor);
      ALE::Point     interval = serialCoordBundle->getFiberInterval(vertex);
      ALE::Point_set support;
      PetscScalar   *array;

      ierr = restrictField(serialCoordBundle, serialOrientation, coords, vertex, &array); CHKERRQ(ierr);
      for(int d = 0; d < interval.index; d++) {
        in.pointlist[interval.prefix + d] = array[d];
      }

      interval = serialVertexBundle->getFiberInterval(vertex);
      support  = serialBoundary->support(vertex);
      if (support.size()) {
        in.pointmarkerlist[interval.prefix] = support.begin()->index;
      } else {
        in.pointmarkerlist[interval.prefix] = 0;
      }
    }
    ierr = VecRestoreArray(serialCoordinates, &coords);CHKERRQ(ierr);

    in.numberofcorners = 3;
    in.numberoftriangles = faces->size();
    ierr = PetscMalloc(in.numberoftriangles * in.numberofcorners * sizeof(int), &in.trianglelist);CHKERRQ(ierr);
    for(ALE::Point_set::iterator f_itor = faces->begin(); f_itor != faces->end(); f_itor++) {
      ALE::Obj<ALE::Point_array> intervals = serialVertexBundle->getGlobalOrderedClosureIndices(serialOrientation->cone(*f_itor));
      int                        v = 0;

      for(ALE::Point_array::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
        in.trianglelist[f * in.numberofcorners + v++] = i_itor->prefix;
      }
      f++;
    }

    in.numberofsegments = 0;
    for(ALE::Point_set::iterator b_itor = boundaries->begin(); b_itor != boundaries->end(); b_itor++) {
      ALE::Point_set segments = serialBoundary->cone(*b_itor);

      /* Should be done with a fibration instead */
      segments.meet(edges);
      in.numberofsegments += segments.size();
    }
    if (in.numberofsegments > 0) {
      ierr = PetscMalloc(in.numberofsegments * 2 * sizeof(int), &in.segmentlist);CHKERRQ(ierr);
      ierr = PetscMalloc(in.numberofsegments * sizeof(int), &in.segmentmarkerlist);CHKERRQ(ierr);
      for(ALE::Point_set::iterator b_itor = boundaries->begin(); b_itor != boundaries->end(); b_itor++) {
        ALE::Point_set segments = serialBoundary->cone(*b_itor);

        /* Should be done with a fibration instead */
        segments.meet(edges);
        for(ALE::Point_set::iterator s_itor = segments.begin(); s_itor != segments.end(); s_itor++) {
          ALE::Point               segment = *s_itor;
          ALE::Point               interval = serialElementBundle->getFiberInterval(segment);
          ALE::Obj<ALE::Point_set> cone = serialTopology->cone(segment);
          PetscInt                 p = 0;
        
          for(ALE::Point_set::iterator c_itor = cone->begin(); c_itor != cone->end(); c_itor++) {
            in.segmentlist[interval.prefix * 2 + (p++)] = (*c_itor).index;
          }
          in.segmentmarkerlist[interval.prefix] = b_itor->index;
        }
      }
    }

    in.numberofholes = 0;
    if (in.numberofholes > 0) {
      ierr = PetscMalloc(in.numberofholes * dim * sizeof(int), &in.holelist);CHKERRQ(ierr);
    }
    ierr = PetscStrlen(args, &len);CHKERRQ(ierr);
    ierr = PetscMalloc((strlen(args) + 4) * sizeof(char), &newArgs);CHKERRQ(ierr);
    ierr = PetscStrcpy(newArgs, args);CHKERRQ(ierr);
    ierr = PetscStrcat(newArgs, "pra");CHKERRQ(ierr);
    triangulate((char *) newArgs, &in, &out, NULL);
    ierr = PetscFree(newArgs);CHKERRQ(ierr);
    ierr = PetscFree(in.pointlist);CHKERRQ(ierr);
    ierr = PetscFree(in.pointmarkerlist);CHKERRQ(ierr);
    ierr = PetscFree(in.segmentlist);CHKERRQ(ierr);
    ierr = PetscFree(in.segmentmarkerlist);CHKERRQ(ierr);
  }
  ierr = MeshPopulate(m, dim, out.numberofpoints, out.numberoftriangles, out.trianglelist, out.pointlist);CHKERRQ(ierr);
  ierr = MeshDistribute(m);CHKERRQ(ierr);
  /* Need to make boundary */
  ierr = destroyGeneratorOutput(&out);CHKERRQ(ierr);
  *mesh = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCoarsen_Triangle"
PetscErrorCode MeshCoarsen_Triangle(Mesh mesh, PetscReal minArea, /*CoSieve*/ Vec minAreas, Mesh *coarseMesh)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "MeshGenerate"
PetscErrorCode MeshGenerate(Mesh boundary, Mesh *mesh)
{
#ifdef PETSC_HAVE_TRIANGLE
  PetscErrorCode ierr;
#endif
  PetscFunctionBegin;
#ifdef PETSC_HAVE_TRIANGLE
  ierr = MeshGenerate_Triangle(boundary, mesh);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshRefine"
PetscErrorCode MeshRefine(Mesh mesh, PetscReal maxArea, /*CoSieve*/ Vec maxAreas, Mesh *refinedMesh)
{
#ifdef PETSC_HAVE_TRIANGLE
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
#ifdef PETSC_HAVE_TRIANGLE
  ierr = MeshRefine_Triangle(mesh, maxArea, maxAreas, refinedMesh);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCoarsen"
PetscErrorCode MeshCoarsen(Mesh mesh, PetscReal minArea, /*CoSieve*/ Vec minAreas, Mesh *coarseMesh)
{
#ifdef PETSC_HAVE_TRIANGLE
  PetscErrorCode ierr;
#endif
  PetscFunctionBegin;
#ifdef PETSC_HAVE_TRIANGLE
  ierr = MeshCoarsen_Triangle(mesh, minArea, minAreas, coarseMesh);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadConnectivity_PCICE"
PetscErrorCode ReadConnectivity_PCICE(MPI_Comm comm, const char *filename, PetscInt dim, PetscTruth useZeroBase, PetscInt *numElements, PetscInt **vertices)
{
  PetscViewer    viewer;
  FILE          *f;
  PetscInt       numCells, cellCount = 0;
  PetscInt      *verts;
  char           buf[2048];
  PetscInt       c;
  PetscInt       commSize, commRank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &commSize); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &commRank); CHKERRQ(ierr);

  if(commRank == 0) {
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRQ(ierr);
    numCells = atoi(fgets(buf, 2048, f));
    ierr = PetscMalloc(numCells*(dim+1) * sizeof(PetscInt), &verts);CHKERRQ(ierr);
    while(fgets(buf, 2048, f) != NULL) {
      const char *v = strtok(buf, " ");
      
      /* Ignore cell number */
      v = strtok(NULL, " ");
      for(c = 0; c <= dim; c++) {
        int vertex = atoi(v);
        
        if (!useZeroBase) vertex -= 1;
        verts[cellCount*(dim+1)+c] = vertex;
        v = strtok(NULL, " ");
      }
      cellCount++;
    }
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    *numElements = numCells;
    *vertices = verts;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadCoordinates_PCICE"
PetscErrorCode ReadCoordinates_PCICE(MPI_Comm comm, const char *filename, PetscInt dim, PetscInt *numVertices, PetscScalar **coordinates)
{
  PetscViewer    viewer;
  FILE          *f;
  PetscInt       numVerts, vertexCount = 0;
  PetscScalar   *coords;
  char           buf[2048];
  PetscInt       c;
  PetscInt       commSize, commRank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &commSize); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &commRank); CHKERRQ(ierr);

  if (commRank == 0) {
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRQ(ierr);
    numVerts = atoi(fgets(buf, 2048, f));
    ierr = PetscMalloc(numVerts*dim * sizeof(PetscScalar), &coords);CHKERRQ(ierr);
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
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    *numVertices = numVerts;
    *coordinates = coords;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IgnoreComments_PyLith"
PetscErrorCode IgnoreComments_PyLith(char *buf, PetscInt bufSize, FILE *f)
{
  PetscFunctionBegin;
  while((fgets(buf, bufSize, f) != NULL) && (buf[0] == '#')) {}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadConnectivity_PyLith"
PetscErrorCode ReadConnectivity_PyLith(MPI_Comm comm, const char *filename, PetscInt dim, PetscTruth useZeroBase, PetscInt *numElements, PetscInt **vertices)
{
  PetscViewer    viewer;
  FILE          *f;
  PetscInt       maxCells = 1024, cellCount = 0;
  PetscInt      *verts;
  char           buf[2048];
  PetscInt       c;
  PetscInt       commSize, commRank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &commSize); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &commRank); CHKERRQ(ierr);
  if (dim != 3) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "PyLith only works in 3D");
  }
  if(commRank == 0) {
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRQ(ierr);
    /* Ignore comments */
    IgnoreComments_PyLith(buf, 2048, f);
    ierr = PetscMalloc(maxCells*(dim+1) * sizeof(PetscInt), &verts);CHKERRQ(ierr);
    do {
      const char *v = strtok(buf, " ");
      int         elementType;

      if (cellCount == maxCells) {
        PetscInt *vtmp;

        vtmp = verts;
        ierr = PetscMalloc(maxCells*2*(dim+1) * sizeof(PetscInt), &verts);CHKERRQ(ierr);
        ierr = PetscMemcpy(verts, vtmp, maxCells*(dim+1) * sizeof(PetscInt));CHKERRQ(ierr);
        ierr = PetscFree(vtmp);CHKERRQ(ierr);
        maxCells *= 2;
      }
      /* Ignore cell number */
      v = strtok(NULL, " ");
      /* Verify element type is linear tetrahedron */
      elementType = atoi(v);
      if (elementType != 5) {
        SETERRQ(PETSC_ERR_ARG_WRONG, "We only accept linear tetrahedra right now");
      }
      v = strtok(NULL, " ");
      /* Ignore material type */
      v = strtok(NULL, " ");
      /* Ignore infinite domain element code */
      v = strtok(NULL, " ");
      for(c = 0; c <= dim; c++) {
        int vertex = atoi(v);
        
        if (!useZeroBase) vertex -= 1;
        verts[cellCount*(dim+1)+c] = vertex;
        v = strtok(NULL, " ");
      }
      cellCount++;
    } while(fgets(buf, 2048, f) != NULL);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    *numElements = cellCount;
    *vertices = verts;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadCoordinates_PyLith"
PetscErrorCode ReadCoordinates_PyLith(MPI_Comm comm, const char *filename, PetscInt dim, PetscInt *numVertices, PetscScalar **coordinates)
{
  PetscViewer    viewer;
  FILE          *f;
  PetscInt       maxVerts = 1024, vertexCount = 0;
  PetscScalar   *coords;
  char           buf[2048];
  PetscInt       c;
  PetscInt       commSize, commRank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &commSize); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &commRank); CHKERRQ(ierr);
  if (commRank == 0) {
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRQ(ierr);
    /* Ignore comments and units line */
    IgnoreComments_PyLith(buf, 2048, f);
    ierr = PetscMalloc(maxVerts*dim * sizeof(PetscScalar), &coords);CHKERRQ(ierr);
    while(fgets(buf, 2048, f) != NULL) {
      const char *x = strtok(buf, " ");

      if (vertexCount == maxVerts) {
        PetscScalar *ctmp;

        ctmp = coords;
        ierr = PetscMalloc(maxVerts*2*dim * sizeof(PetscScalar), &coords);CHKERRQ(ierr);
        ierr = PetscMemcpy(coords, ctmp, maxVerts*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = PetscFree(ctmp);CHKERRQ(ierr);
        maxVerts *= 2;
      }
      /* Ignore vertex number */
      x = strtok(NULL, " ");
      for(c = 0; c < dim; c++) {
        coords[vertexCount*dim+c] = atof(x);
        x = strtok(NULL, " ");
      }
      vertexCount++;
    }
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    *numVertices = vertexCount;
    *coordinates = coords;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreatePyLith"
/*@C
  MeshCreatePyLith - Create a mesh from a set of PyLith mesh files.

  Collective on Mesh

  Input Parameters:
+ comm - The communicator
- baseFilename - The base name for all mesh files

  Output Parameter:
. mesh - the mesh object

  Level: intermediate

.seealso MeshCreate(), MeshPopulate(), MeshCreatePCICE()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreatePyLith(MPI_Comm comm, const char baseFilename[], Mesh *mesh)
{
  PetscInt       dim = 3;
  PetscTruth     useZeroBase = PETSC_FALSE;
  char           filename[2048];
  PetscInt      *vertices;
  PetscScalar   *coordinates;
  PetscInt       numElements, numVertices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscStrcpy(filename, baseFilename);CHKERRQ(ierr);
  ierr = PetscStrcat(filename, ".connect");CHKERRQ(ierr);
  ierr = ReadConnectivity_PyLith(comm, filename, dim, useZeroBase, &numElements, &vertices);CHKERRQ(ierr);
  ierr = PetscStrcpy(filename, baseFilename);CHKERRQ(ierr);
  ierr = PetscStrcat(filename, ".coord");CHKERRQ(ierr);
  ierr = ReadCoordinates_PyLith(comm, filename, dim, &numVertices, &coordinates);CHKERRQ(ierr);
  ierr = MeshPopulate(*mesh, dim, numVertices, numElements, vertices, coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreatePCICE"
/*@C
  MeshCreatePCICE - Create a mesh from a set of PCICE mesh files.

  Collective on Mesh

  Input Parameters:
+ comm - The communicator
. baseFilename - The base name for all mesh files
. dim - The mesh dimension
- useZeroBase - Start numbering from 0

  Output Parameter:
. mesh - the mesh object

  Level: intermediate

.seealso MeshCreate(), MeshPopulate(), MeshCreatePyLith()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreatePCICE(MPI_Comm comm, const char baseFilename[], PetscInt dim, PetscTruth useZeroBase, Mesh *mesh)
{
  char           filename[2048];
  PetscInt      *vertices;
  PetscScalar   *coordinates;
  PetscInt       numElements, numVertices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscStrcpy(filename, baseFilename);CHKERRQ(ierr);
  ierr = PetscStrcat(filename, ".lcon");CHKERRQ(ierr);
  ierr = ReadConnectivity_PCICE(comm, filename, dim, useZeroBase, &numElements, &vertices);CHKERRQ(ierr);
  ierr = PetscStrcpy(filename, baseFilename);CHKERRQ(ierr);
  ierr = PetscStrcat(filename, ".nodes");CHKERRQ(ierr);
  ierr = ReadCoordinates_PCICE(comm, filename, dim, &numVertices, &coordinates);CHKERRQ(ierr);
  ierr = MeshPopulate(*mesh, dim, numVertices, numElements, vertices, coordinates);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
