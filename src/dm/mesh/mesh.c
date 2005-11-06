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
  void    *orientation;
  void    *spaceFootprint;
  void    *bundle;
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
  ALE::IndexBundle *vertexBundle = new ALE::IndexBundle(topology);
  vertexBundle->setFiberDimensionByDepth(0, 1);
  vertexBundle->computeOverlapIndices();
  vertexBundle->computeGlobalIndices();
  ALE::IndexBundle *elementBundle = new ALE::IndexBundle(topology);
  elementBundle->setFiberDimensionByHeight(0, 1);
  elementBundle->computeOverlapIndices();
  elementBundle->computeGlobalIndices();
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
      ALE::Obj<ALE::Point_array> intervals = vertexBundle.getGlobalOrderedClosureIndices(orientation->cone(*e_itor));

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
    SETERRQ(PETSC_ERR_SUP, "");
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
  PetscInt               localSize, globalSize;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  localSize = ((ALE::IndexBundle *) mesh->bundle)->getLocalSize();
  globalSize = ((ALE::IndexBundle *) mesh->bundle)->getGlobalSize();
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
  p->orientation    = PETSC_NULL;
  p->spaceFootprint = PETSC_NULL;
  p->bundle         = PETSC_NULL;
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
  if (mesh->globalvector) {
    ierr = VecDuplicate(mesh->globalvector, gvec);CHKERRQ(ierr);
  } else {
    int localSize = ((ALE::IndexBundle *) mesh->bundle)->getLocalSize();
    int globalSize = ((ALE::IndexBundle *) mesh->bundle)->getGlobalSize();
    ALE::Obj<ALE::PreSieve> globalIndices = ((ALE::IndexBundle *) mesh->bundle)->getGlobalIndices();
    ALE::Obj<ALE::PreSieve> pointTypes = ((ALE::IndexBundle *) mesh->bundle)->getPointTypes();
    ALE::Obj<ALE::Point_set> rentedPoints = pointTypes->cone(ALE::Point(((ALE::IndexBundle *) mesh->bundle)->getCommRank(), ALE::rentedPoint));
    int ghostSize = 0;
    for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
      ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);

      if (cone->size()) {
        ALE::Point interval = *cone->begin();

        ghostSize += interval.index;
      }
    }
    int *ghostIndices = new int[ghostSize];
    int ghostIdx = 0;
    for(ALE::Point_set::iterator e_itor = rentedPoints->begin(); e_itor != rentedPoints->end(); e_itor++) {
      ALE::Obj<ALE::Point_set> cone = globalIndices->cone(*e_itor);

      if (cone->size()) {
        ALE::Point interval = *cone->begin();

        ExpandInterval(interval, ghostIndices, &ghostIdx);
      }
    }

    ierr = VecCreateGhostBlock(mesh->comm, 1, localSize, globalSize, ghostSize, ghostIndices, &mesh->globalvector);CHKERRQ(ierr);
    *gvec = mesh->globalvector;
    ierr = PetscObjectReference((PetscObject) *gvec);CHKERRQ(ierr); 
  }
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
