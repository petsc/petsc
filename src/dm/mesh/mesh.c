 
#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/
#include "src/dm/mesh/meshvtk.h"
#include "src/dm/mesh/meshpcice.h"
#include "src/dm/mesh/meshpylith.h"

/* Logging support */
PetscCookie PETSCDM_DLLEXPORT MESH_COOKIE = 0;
PetscEvent  Mesh_View = 0, Mesh_GetGlobalScatter = 0, Mesh_restrictVector = 0, Mesh_assembleVector = 0,
            Mesh_assembleVectorComplete = 0, Mesh_assembleMatrix = 0, Mesh_updateOperator = 0;

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve_Ascii"
PetscErrorCode MeshView_Sieve_Ascii(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK) {
    ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeElements(mesh, viewer);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_PYLITH) {
    char *filename;
    char  connectFilename[2048];
    char  coordFilename[2048];

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscStrcpy(connectFilename, filename);CHKERRQ(ierr);
    ierr = PetscStrcat(connectFilename, ".connect");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, connectFilename);CHKERRQ(ierr);
    ierr = ALE::PyLith::Viewer::writeElements(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscStrcpy(coordFilename, filename);CHKERRQ(ierr);
    ierr = PetscStrcat(coordFilename, ".coord");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, coordFilename);CHKERRQ(ierr);
    ierr = ALE::PyLith::Viewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscExceptionTry1(PetscViewerFileSetName(viewer, filename), PETSC_ERR_FILE_OPEN);
    if (PetscExceptionValue(ierr)) {
      /* this means that a caller above me has also tryed this exception so I don't handle it here, pass it up */
    } else if (PetscExceptionCaught(ierr, PETSC_ERR_FILE_OPEN)) {
      ierr = 0;
    } 
    CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_PYLITH_LOCAL) {
    PetscViewer connectViewer, coordViewer, splitViewer;
    char       *filename;
    char        localFilename[2048];
    int         rank = mesh->commRank();

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);

    sprintf(localFilename, "%s.%d.connect", filename, rank);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &connectViewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(connectViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(connectViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(connectViewer, localFilename);CHKERRQ(ierr);
    ierr = ALE::PyLith::Viewer::writeElementsLocal(mesh, connectViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(connectViewer);CHKERRQ(ierr);

    sprintf(localFilename, "%s.%d.coord", filename, rank);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &coordViewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(coordViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(coordViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(coordViewer, localFilename);CHKERRQ(ierr);
    ierr = ALE::PyLith::Viewer::writeVerticesLocal(mesh, coordViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(coordViewer);CHKERRQ(ierr);

    if (mesh->hasField("split")) {
      sprintf(localFilename, "%s.%d.split", filename, rank);
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &splitViewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(splitViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(splitViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(splitViewer, localFilename);CHKERRQ(ierr);
      ierr = ALE::PyLith::Viewer::writeSplitLocal(mesh, splitViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(splitViewer);CHKERRQ(ierr);
    }
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
    ierr = ALE::PCICE::Viewer::writeElements(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscStrncpy(coordFilename, filename, len-5);CHKERRQ(ierr);
    coordFilename[len-5] = '\0';
    ierr = PetscStrcat(coordFilename, ".nodes");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, coordFilename);CHKERRQ(ierr);
    ierr = ALE::PCICE::Viewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
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
#define __FUNCT__ "MeshView_Sieve"
PetscErrorCode MeshView_Sieve(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer)
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
#define __FUNCT__ "FieldView_Sieve_Ascii"
PetscErrorCode FieldView_Sieve_Ascii(ALE::Obj<ALE::Mesh> mesh, const std::string& name, PetscViewer viewer)
{
  // state 0: No header has been output
  // state 1: Only POINT_DATA has been output
  // state 2: Only CELL_DATA has been output
  // state 3: Output both, POINT_DATA last
  // state 4: Output both, CELL_DATA last
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK || format == PETSC_VIEWER_ASCII_VTK_CELL) {
    static PetscInt   stateId     = -1;
    PetscInt          doOutput    = 0;
    PetscInt          outputState = 0;
    PetscInt          fiberDim    = 0;
    PetscTruth        hasState;

    if (stateId < 0) {
      ierr = PetscObjectComposedDataRegister(&stateId);CHKERRQ(ierr);
      ierr = PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, 0);CHKERRQ(ierr);
    }
    ierr = PetscObjectComposedDataGetInt((PetscObject) viewer, stateId, outputState, hasState);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_VTK) {
      if (outputState == 0) {
        outputState = 1;
        doOutput = 1;
      } else if (outputState == 1) {
        doOutput = 0;
      } else if (outputState == 2) {
        outputState = 3;
        doOutput = 1;
      } else if (outputState == 3) {
        doOutput = 0;
      } else if (outputState == 4) {
        SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");
      }
      ALE::Obj<ALE::Mesh::section_type> field = mesh->getSection(name);
      if (doOutput) {
        ALE::Mesh::section_type::patch_type patch = mesh->getTopologyNew()->getPatches().begin()->first;
        int                                 N     = 0;

        fiberDim = field->getAtlas()->size(patch, *mesh->getTopologyNew()->depthStratum(patch, 0)->begin());
#if 0
        if (field->getGlobalOffsets()) {
          N = field->getGlobalOffsets()[mesh->commSize()]/fiberDim;
        } else {
          N = field->getSize(*field->getPatches()->begin())/fiberDim;
        }
#else
        N = mesh->getTopologyNew()->depthStratum(patch, 0)->size();
#endif
        ierr = PetscViewerASCIIPrintf(viewer, "POINT_DATA %d\n", N);CHKERRQ(ierr);
      }
      VTKViewer::writeField(mesh, field, name, fiberDim, viewer);
    } else {
      if (outputState == 0) {
        outputState = 2;
        doOutput = 1;
      } else if (outputState == 1) {
        outputState = 4;
        doOutput = 1;
      } else if (outputState == 2) {
        doOutput = 0;
      } else if (outputState == 3) {
        SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
      } else if (outputState == 4) {
        doOutput = 0;
      }
      ALE::Obj<ALE::Mesh::section_type> field = mesh->getSection(name);
      if (doOutput) {
        ALE::Mesh::section_type::patch_type patch = mesh->getTopologyNew()->getPatches().begin()->first;
        int                                 N     = 0;

        fiberDim = field->getAtlas()->size(patch, *mesh->getTopologyNew()->heightStratum(patch, 0)->begin());
#if 0
        if (field->getGlobalOffsets()) {
          N = field->getGlobalOffsets()[mesh->commSize()]/fiberDim;
        } else {
          N = field->getSize(*field->getPatches()->begin())/fiberDim;
        }
#else
        N = mesh->getTopologyNew()->heightStratum(patch, 0)->size();
#endif
        ierr = PetscViewerASCIIPrintf(viewer, "CELL_DATA %d\n", N);CHKERRQ(ierr);
      }
      VTKViewer::writeField(mesh, field, name, fiberDim, viewer);
    }
    ierr = PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, outputState);CHKERRQ(ierr);
  } else {
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "FieldView_Sieve"
PetscErrorCode FieldView_Sieve(ALE::Obj<ALE::Mesh> mesh, const std::string& name, PetscViewer viewer)
{
  PetscTruth     iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_DRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ierr = FieldView_Sieve_Ascii(mesh, name, viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    SETERRQ(PETSC_ERR_SUP, "Binary viewer not implemented for Field");
  } else if (isdraw){ 
    SETERRQ(PETSC_ERR_SUP, "Draw viewer not implemented for Field");
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by this field object", ((PetscObject)viewer)->type_name);
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
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  PetscValidType(mesh, 1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(mesh->comm);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE, 2);
  PetscCheckSameComm(mesh, 1, viewer, 2);

  ierr = PetscLogEventBegin(Mesh_View,0,0,0,0);CHKERRQ(ierr);
  ierr = (*mesh->ops->view)(mesh->m, viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mesh_View,0,0,0,0);CHKERRQ(ierr);
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
PetscErrorCode PETSCDM_DLLEXPORT MeshGetMesh(Mesh mesh, ALE::Obj<ALE::Mesh> *m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
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
PetscErrorCode PETSCDM_DLLEXPORT MeshSetMesh(Mesh mesh, ALE::Obj<ALE::Mesh> m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
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
  ALE::Obj<ALE::Mesh> m;
#if 0
  ISLocalToGlobalMapping lmap;
  PetscInt              *globals,rstart,i;
#endif
  PetscInt               localSize;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  localSize = m->getField("u")->getGlobalOrder()->getSize(ALE::Mesh::field_type::patch_type());

  ierr = MatCreate(mesh->comm,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,localSize,localSize,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
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

  ierr = PetscHeaderCreate(p,_p_Mesh,struct _MeshOps,MESH_COOKIE,0,"Mesh",comm,MeshDestroy,0);CHKERRQ(ierr);
  p->ops->view               = MeshView_Sieve;
  p->ops->createglobalvector = MeshCreateGlobalVector;
  p->ops->getmatrix          = MeshGetMatrix;

  ierr = PetscObjectChangeTypeName((PetscObject) p, "sieve");CHKERRQ(ierr);

  p->m            = PETSC_NULL;
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
inline void ExpandInterval_New(ALE::Point interval, PetscInt indices[], PetscInt *indx)
{
  for(int i = 0; i < interval.index; i++) {
    indices[(*indx)++] = interval.prefix + i;
  }
}

#undef __FUNCT__
#define __FUNCT__ "ExpandIntervals"
PetscErrorCode ExpandIntervals(ALE::Obj<ALE::Mesh::bundle_type::IndexArray> intervals, PetscInt *indices)
{
  int k = 0;

  PetscFunctionBegin;
  for(ALE::Mesh::bundle_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    ExpandInterval_New(*i_itor, indices, &k);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateVector"
/*
  Creates a ghosted vector based upon the global ordering in the bundle.
*/
PetscErrorCode MeshCreateVector(ALE::Obj<ALE::Mesh> m, ALE::Obj<ALE::Mesh::bundle_type> bundle, Vec *v)
{
  // FIX: Must not include ghosts
  PetscInt       localSize = bundle->getGlobalOrder()->getSize(ALE::Mesh::bundle_type::patch_type());
  MPI_Comm       comm = m->comm();
  PetscMPIInt    rank = m->commRank();
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
  ALE::Obj<ALE::Mesh> m;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  ierr = MeshCreateVector(m, m->getBundle(0), gvec);CHKERRQ(ierr);
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

template<typename IntervalSequence>
PetscErrorCode __expandIntervals(ALE::Obj<IntervalSequence> intervals, PetscInt *indices[]) {
  PetscInt      *ind;
  PetscInt       k = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
    int dim = i_iter.color().index;

    k += std::abs(dim);
  }
  //std::cout << "Allocated indices of size " << k << std::endl;
  ierr = PetscMalloc(k * sizeof(PetscInt), &ind);CHKERRQ(ierr);
  k = 0;
  for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
    //std::cout << "  indices for " << *i_iter << std::endl;
    for(int i = i_iter.color().prefix; i < i_iter.color().prefix + std::abs(i_iter.color().index); i++) {
      //std::cout << "  indices[" << k << "] = " << i << std::endl;
      ind[k++] = i;
    }
  }
  *indices = ind;
  PetscFunctionReturn(0);
}

template<typename IntervalSequence>
PetscErrorCode __expandIntervals(ALE::Obj<IntervalSequence> intervals, ALE::Obj<ALE::Mesh::bundle_type::order_type> order, PetscInt *indices[]) {
  typename ALE::Mesh::bundle_type::patch_type patch;
  PetscInt      *ind;
  PetscInt       k = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
    int dim = order->getColor(*i_iter, patch, false).index;

    k += std::abs(dim);
  }
  //std::cout << "Allocated indices of size " << k << std::endl;
  ierr = PetscMalloc(k * sizeof(PetscInt), &ind);CHKERRQ(ierr);
  k = 0;
  for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
    const ALE::Mesh::bundle_type::index_type& color = order->getColor(*i_iter, patch, false);

    //std::cout << "  indices for " << *i_iter << std::endl;
    for(int i = color.prefix; i < color.prefix + std::abs(color.index); i++) {
      //std::cout << "  indices[" << k << "] = " << i << std::endl;
      ind[k++] = i;
    }
  }
  *indices = ind;
  PetscFunctionReturn(0);
}

template<typename IntervalSequence,typename Field>
  PetscErrorCode __expandCanonicalIntervals(ALE::Obj<IntervalSequence> intervals, ALE::Obj<Field> field,PetscInt *indices[]) {
  typename Field::patch_type patch;
  PetscInt      *ind;
  PetscInt       k = 0;
  PetscErrorCode ierr;

  for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
    k += std::abs(field->getFiberDimension(patch, *i_iter));
  }
  //std::cout << "Allocated indices of size " << k << std::endl;
  ierr = PetscMalloc(k * sizeof(PetscInt), &ind);CHKERRQ(ierr);
  k = 0;
  for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
    int dim = field->getFiberDimension(patch, *i_iter);
    int offset = field->getFiberOffset(patch, *i_iter);

    //std::cout << "  indices for " << *i_iter << std::endl;
    for(int i = offset; i < offset + std::abs(dim); i++) {
      //std::cout << "  indices[" << k << "] = " << i << std::endl;
      ind[k++] = i;
    }
  }
  *indices = ind;
  PetscFunctionReturn(0);
};

template<typename IntervalSequence,typename Field>
  PetscErrorCode __expandCanonicalIntervals(ALE::Obj<IntervalSequence> intervals, ALE::Obj<Field> field,PetscInt indices[]) {
  typename Field::patch_type patch;
  PetscInt k = 0;

  for(typename IntervalSequence::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
    int dim = field->getFiberDimension(patch, *i_iter);
    int offset = field->getFiberOffset(patch, *i_iter);

    //std::cout << "  indices for " << *i_iter << std::endl;
    for(int i = offset; i < offset + std::abs(dim); i++) {
      //std::cout << "  indices[" << k << "] = " << i << std::endl;
      indices[k++] = i;
    }
  }
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "MeshGetGlobalScatter"
PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalScatter(ALE::Mesh *mesh,const char fieldName[],Vec g,VecScatter *scatter)
{
  ALE::Mesh::patch_type patch;
  Vec            localVec;
  IS             globalIS, localIS;
  PetscInt      *indices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_GetGlobalScatter,0,0,0,0);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh::field_type>  field       = mesh->getField(std::string(fieldName));
  ALE::Obj<ALE::Mesh::bundle_type> globalOrder = field->getGlobalOrder();
  ALE::Obj<ALE::Mesh::bundle_type> localOrder  = field->getLocalOrder();

  ierr = __expandIntervals(globalOrder->getPatch(patch), &indices);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, field->getSize(patch), indices, &globalIS);CHKERRQ(ierr);
  ierr = PetscFree(indices);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, field->getSize(patch), field->restrict(patch), &localVec);CHKERRQ(ierr);
  //ierr = ISCreateStride(PETSC_COMM_SELF, localOrder->getSize(patch), 0, 1, &localIS);CHKERRQ(ierr);
  ierr = __expandCanonicalIntervals(globalOrder->getPatch(patch), field, &indices);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, field->getSize(patch), indices, &localIS);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = VecScatterCreate(localVec, localIS, g, globalIS, scatter);CHKERRQ(ierr);
  ierr = ISDestroy(globalIS);CHKERRQ(ierr);
  ierr = ISDestroy(localIS);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mesh_GetGlobalScatter,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode assembleFullField(VecScatter, Vec, Vec, InsertMode);

#undef __FUNCT__
#define __FUNCT__ "restrictVector"
/*@C
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
  ierr = PetscLogEventBegin(Mesh_restrictVector,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) g, "injection", (PetscObject *) &injection);CHKERRQ(ierr);
  if (injection) {
    ierr = VecScatterBegin(g, l, mode, SCATTER_REVERSE, injection);
    ierr = VecScatterEnd(g, l, mode, SCATTER_REVERSE, injection);
  } else {
    if (mode == INSERT_VALUES) {
      ierr = VecCopy(g, l);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(l, 1.0, g);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(Mesh_restrictVector,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleVectorComplete"
/*@C
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
  ierr = PetscLogEventBegin(Mesh_assembleVectorComplete,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) g, "injection", (PetscObject *) &injection);CHKERRQ(ierr);
  if (injection) {
    ierr = VecScatterBegin(l, g, mode, SCATTER_FORWARD, injection);CHKERRQ(ierr);
    ierr = VecScatterEnd(l, g, mode, SCATTER_FORWARD, injection);CHKERRQ(ierr);
  } else {
    if (mode == INSERT_VALUES) {
      ierr = VecCopy(l, g);CHKERRQ(ierr);
    } else {
      ierr = VecAXPY(g, 1.0, l);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(Mesh_assembleVectorComplete,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleVector"
/*@C
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
  ALE::Obj<ALE::Mesh> m;
  ALE::Mesh::field_type::patch_type patch;
  PetscInt                 firstElement;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_assembleVector,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) b, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  //firstElement = elementBundle->getLocalSizes()[bundle->getCommRank()];
  firstElement = 0;
  // Must relate b to field
  if (mode == INSERT_VALUES) {
    m->getField(std::string("x"))->update(patch, ALE::Mesh::point_type(0, e + firstElement), v);
  } else {
    m->getField(std::string("x"))->updateAdd(patch, ALE::Mesh::point_type(0, e + firstElement), v);
  }
  ierr = PetscLogEventEnd(Mesh_assembleVector,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperator"
PetscErrorCode updateOperator(Mat A, ALE::Obj<ALE::Mesh::section_type> field, const ALE::Mesh::point_type& e, PetscScalar array[], InsertMode mode)
{
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ALE::Mesh::section_type::patch_type patch = 0;
  const ALE::Obj<ALE::Mesh::atlas_type::IndexArray> intervals = field->getAtlas()->getIndices(patch, e);

  ierr = PetscLogEventBegin(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  if (field->debug()) {printf("[%d]mat for element (%d, %d)\n", field->commRank(), e.prefix, e.index);}
  for(ALE::Mesh::atlas_type::IndexArray::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
    //numIndices += std::abs(globalOrder->getFiberDimension(patch, *i_iter));
    numIndices += i_iter->index;
    if (field->debug()) {
      printf("[%d]mat interval (%d, %d)\n", field->commRank(), i_iter->prefix, i_iter->index);
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
  //ierr = __expandCanonicalIntervals(intervals, globalOrder, indices); CHKERRQ(ierr);
  if (field->debug()) {
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]mat indices[%d] = %d\n", field->commRank(), i, indices[i]);
    }
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]", field->commRank());
      for(int j = 0; j < numIndices; j++) {
        printf(" %g", array[i*numIndices+j]);
      }
      printf("\n");
    }
  }
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, array, mode);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleMatrix"
/*@C
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
  static ALE::Mesh::field_type::patch_type patch;
  PetscObjectContainer  c;
  ALE::Mesh            *mesh;
  //ALE::Mesh::point_type order(e, 0);
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_assembleMatrix,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) A, "mesh", (PetscObject *) &c);CHKERRQ(ierr);
  ierr = PetscObjectContainerGetPointer(c, (void **) &mesh);CHKERRQ(ierr);
  try {
    // Notice that we map the global element number to the point
    //ierr = updateOperator(A, mesh->getField("displacement"), *mesh->getBundle(mesh->getTopology()->depth())->__getOrder()->cone(patch, order)->begin(), v, mode);CHKERRQ(ierr);
    ierr = updateOperator(A, mesh->getSection("displacement"), ALE::Mesh::point_type(0, e), v, mode);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e.msg() << std::endl;
  }
  ierr = PetscLogEventEnd(Mesh_assembleMatrix,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "preallocateMatrix"
PetscErrorCode preallocateMatrix(Mat A, ALE::Mesh *mesh, ALE::Obj<ALE::Mesh::field_type> field)
{
  PetscInt numLocalRows, firstRow;
  PetscInt *rnz, *dnz, *onz;
  std::string orderName("element");
  int *indices = NULL;
  int indicesMaxSize = 12;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A, &numLocalRows, PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A, &firstRow, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMalloc3(numLocalRows,PetscInt,&rnz,numLocalRows,PetscInt,&dnz,numLocalRows,PetscInt,&onz);CHKERRQ(ierr);
  ierr = PetscMemzero(rnz, numLocalRows * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(dnz, numLocalRows * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(onz, numLocalRows * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc(indicesMaxSize * sizeof(PetscInt), &indices);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements    = mesh->getTopology()->heightStratum(0);
  ALE::Obj<ALE::Mesh::bundle_type>                        globalOrder = field->getGlobalOrder();
  std::set<int>                                          *localCols   = new std::set<int>[numLocalRows];

  for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); e_iter++) {
    ALE::Obj<ALE::Mesh::field_type::order_type::coneSequence> intervals = field->getPatch(orderName, *e_iter);
    ALE::Mesh::bundle_type::patch_type patch;
    int numIndices = 0;

    for(ALE::Mesh::field_type::order_type::coneSequence::iterator i_itor = intervals->begin(); i_itor != intervals->end(); ++i_itor) {
      numIndices += std::abs(globalOrder->getFiberDimension(patch, *i_itor));
      if (field->debug) {
        printf("[%d]Allocation mat interval (%d, %d)\n", field->commRank(), (*i_itor).prefix, (*i_itor).index);
      }
    }
    if (numIndices > indicesMaxSize) {
      int *tmpIndices = indices;

      ierr = PetscMalloc(indicesMaxSize*2 * sizeof(PetscInt), &indices);CHKERRQ(ierr);
      ierr = PetscMemcpy(indices, tmpIndices, indicesMaxSize * sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscFree(tmpIndices);CHKERRQ(ierr);
      indicesMaxSize *= 2;
    }
    ierr = __expandCanonicalIntervals(intervals, globalOrder, indices); CHKERRQ(ierr);
    for(int i = 0; i < numIndices; i++) {
      int localRow = indices[i] - firstRow;

      if ((localRow >= 0) && (localRow < numLocalRows)) {
        for(int j = 0; j < numIndices; j++) {
          localCols[localRow].insert(indices[j] - firstRow);
        }
      }
    }
  }
  for(int r = 0; r < numLocalRows; r++) {
    rnz[r] = localCols[r].size();

    for(std::set<int>::iterator c_iter = localCols[r].begin(); c_iter != localCols[r].end(); ++c_iter) {
      if ((*c_iter >= 0) && (*c_iter < numLocalRows)) {
        dnz[r]++;
      } else {
        onz[r]++;
      }
    }
  }
  delete [] localCols;
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A, 0, rnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, dnz, 0, onz);CHKERRQ(ierr);
  ierr = PetscFree3(rnz, dnz, onz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/******************************** C Wrappers **********************************/

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKHeader"
PetscErrorCode WriteVTKHeader(PetscViewer viewer)
{
  return VTKViewer::writeHeader(viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKVertices"
PetscErrorCode WriteVTKVertices(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  return VTKViewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKElements"
PetscErrorCode WriteVTKElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  return VTKViewer::writeElements(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEVertices"
PetscErrorCode WritePCICEVertices(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEElements"
PetscErrorCode WritePCICEElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeElements(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVertices"
PetscErrorCode WritePyLithVertices(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElements"
PetscErrorCode WritePyLithElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeElements(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVerticesLocal"
PetscErrorCode WritePyLithVerticesLocal(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeVerticesLocal(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElementsLocal"
PetscErrorCode WritePyLithElementsLocal(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeElementsLocal(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithSplitLocal"
PetscErrorCode WritePyLithSplitLocal(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeSplitLocal(m, viewer);
}
