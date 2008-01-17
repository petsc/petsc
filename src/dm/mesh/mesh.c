#include "private/meshimpl.h"   /*I      "petscmesh.h"   I*/
#include <petscmesh_viewers.hh>
#include <petscmesh_formats.hh>

/* Logging support */
PetscCookie PETSCDM_DLLEXPORT MESH_COOKIE = 0;
PetscEvent  Mesh_View = 0, Mesh_GetGlobalScatter = 0, Mesh_restrictVector = 0, Mesh_assembleVector = 0,
            Mesh_assembleVectorComplete = 0, Mesh_assembleMatrix = 0, Mesh_updateOperator = 0;

PetscTruth MeshRegisterAllCalled = PETSC_FALSE;
PetscFList MeshList;

EXTERN PetscErrorCode MeshView_Mesh(Mesh, PetscViewer);
EXTERN PetscErrorCode MeshRefine_Mesh(Mesh, MPI_Comm, Mesh *);
EXTERN PetscErrorCode MeshCoarsenHierarchy_Mesh(Mesh, int, Mesh **);
EXTERN PetscErrorCode MeshGetInterpolation_Mesh(Mesh, Mesh, Mat *, Vec *);
EXTERN PetscErrorCode MeshGetInterpolation_Mesh_New(Mesh, Mesh, Mat *, Vec *);

EXTERN PetscErrorCode updateOperatorCompat(Mat, const ALE::Obj<ALECompat::Mesh::real_section_type>&, const ALE::Obj<ALECompat::Mesh::order_type>&, const ALECompat::Mesh::point_type&, PetscScalar[], InsertMode);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "Mesh_DelTag" 
/*
   Private routine to delete internal tag storage when a communicator is freed.

   This is called by MPI, not by users.

   Note: this is declared extern "C" because it is passed to MPI_Keyval_create

         we do not use PetscFree() since it is unsafe after PetscFinalize()
*/
PetscMPIInt PETSC_DLLEXPORT Mesh_DelTag(MPI_Comm comm,PetscMPIInt keyval,void* attr_val,void* extra_state)
{
  free(attr_val);
  return(MPI_SUCCESS);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MeshFinalize"
PetscErrorCode MeshFinalize()
{
  PetscFunctionBegin;
  ALE::Mesh::NumberingFactory::singleton(0, 0, true);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve_Ascii"
PetscErrorCode MeshView_Sieve_Ascii(const ALE::Obj<ALE::Mesh>& mesh, PetscViewer viewer)
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
    ierr = ALE::PyLith::Viewer::writeElements(mesh, mesh->getIntSection("material"), viewer);CHKERRQ(ierr);
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
    PetscViewer connectViewer, coordViewer;
    char       *filename;
    char        localFilename[2048];
    int         rank = mesh->commRank();

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);

    sprintf(localFilename, "%s.%d.connect", filename, rank);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &connectViewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(connectViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(connectViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(connectViewer, localFilename);CHKERRQ(ierr);
    ierr = ALE::PyLith::Viewer::writeElementsLocal(mesh, mesh->getIntSection("material"), connectViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(connectViewer);CHKERRQ(ierr);

    sprintf(localFilename, "%s.%d.coord", filename, rank);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &coordViewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(coordViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(coordViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(coordViewer, localFilename);CHKERRQ(ierr);
    ierr = ALE::PyLith::Viewer::writeVerticesLocal(mesh, coordViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(coordViewer);CHKERRQ(ierr);
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
      ierr = PetscViewerASCIIPrintf(viewer, "  %d %d-cells\n", mesh->depthStratum(d)->size(), d);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCompatView_Sieve_Ascii"
PetscErrorCode MeshCompatView_Sieve_Ascii(const ALE::Obj<ALECompat::Mesh>& mesh, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_PYLITH) {
    char *filename;
    char  connectFilename[2048];
    char  coordFilename[2048];

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscStrcpy(connectFilename, filename);CHKERRQ(ierr);
    ierr = PetscStrcat(connectFilename, ".connect");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, connectFilename);CHKERRQ(ierr);
    ierr = ALECompat::PyLith::Viewer::writeElements(mesh, mesh->getIntSection("material"), viewer);CHKERRQ(ierr);
    ierr = PetscStrcpy(coordFilename, filename);CHKERRQ(ierr);
    ierr = PetscStrcat(coordFilename, ".coord");CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, coordFilename);CHKERRQ(ierr);
    ierr = ALECompat::PyLith::Viewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscExceptionTry1(PetscViewerFileSetName(viewer, filename), PETSC_ERR_FILE_OPEN);
    if (PetscExceptionValue(ierr)) {
      /* this means that a caller above me has also tryed this exception so I don't handle it here, pass it up */
    } else if (PetscExceptionCaught(ierr, PETSC_ERR_FILE_OPEN)) {
      ierr = 0;
    } 
    CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_PYLITH_LOCAL) {
    PetscViewer connectViewer, coordViewer;
    char       *filename;
    char        localFilename[2048];
    int         rank = mesh->commRank();

    ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);

    sprintf(localFilename, "%s.%d.connect", filename, rank);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &connectViewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(connectViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(connectViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(connectViewer, localFilename);CHKERRQ(ierr);
    ierr = ALECompat::PyLith::Viewer::writeElementsLocal(mesh, mesh->getIntSection("material"), connectViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(connectViewer);CHKERRQ(ierr);

    sprintf(localFilename, "%s.%d.coord", filename, rank);
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &coordViewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(coordViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(coordViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(coordViewer, localFilename);CHKERRQ(ierr);
    ierr = ALECompat::PyLith::Viewer::writeVerticesLocal(mesh, coordViewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(coordViewer);CHKERRQ(ierr);

    if (mesh->hasPairSection("split")) {
      PetscViewer splitViewer;

      sprintf(localFilename, "%s.%d.split", filename, rank);
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &splitViewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(splitViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(splitViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(splitViewer, localFilename);CHKERRQ(ierr);
      ierr = ALECompat::PyLith::Viewer::writeSplitLocal(mesh, mesh->getPairSection("split"), splitViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(splitViewer);CHKERRQ(ierr);
    }

    if (mesh->hasRealSection("traction")) {
      PetscViewer tractionViewer;

      sprintf(localFilename, "%s.%d.traction", filename, rank);
      ierr = PetscViewerCreate(PETSC_COMM_SELF, &tractionViewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(tractionViewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(tractionViewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(tractionViewer, localFilename);CHKERRQ(ierr);
      ierr = ALECompat::PyLith::Viewer::writeTractionsLocal(mesh, mesh->getRealSection("traction"), tractionViewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(tractionViewer);CHKERRQ(ierr);
    }
  } else {
    int dim = mesh->getDimension();

    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    for(int d = 0; d <= dim; d++) {
      // FIX: Need to globalize
      ierr = PetscViewerASCIIPrintf(viewer, "  %d %d-cells\n", mesh->getTopology()->depthStratum(0, d)->size(), d);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve"
PetscErrorCode MeshView_Sieve(const ALE::Obj<ALE::Mesh>& mesh, PetscViewer viewer)
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
#define __FUNCT__ "MeshView_Mesh"
PetscErrorCode PETSCDM_DLLEXPORT MeshView_Mesh(Mesh mesh, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!mesh->mcompat.isNull()) {
    ierr = MeshCompatView_Sieve_Ascii(mesh->mcompat, viewer);CHKERRQ(ierr);
  } else {
    ierr = MeshView_Sieve(mesh->m, viewer);CHKERRQ(ierr);
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
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)mesh)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE, 2);
  PetscCheckSameComm(mesh, 1, viewer, 2);

  ierr = PetscLogEventBegin(Mesh_View,0,0,0,0);CHKERRQ(ierr);
  ierr = (*mesh->ops->view)(mesh, viewer);CHKERRQ(ierr);
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
PetscErrorCode PETSCDM_DLLEXPORT MeshGetMesh(Mesh mesh, ALE::Obj<ALE::Mesh>& m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  m = mesh->m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetMesh"
/*@C
    MeshSetMesh - Sets the internal mesh object

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    m - the internal mesh object
 
    Level: advanced

.seealso MeshCreate(), MeshGetMesh()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetMesh(Mesh mesh, const ALE::Obj<ALE::Mesh>& m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  mesh->m = m;
  if (mesh->globalScatter) {
    PetscErrorCode ierr;

    ierr = VecScatterDestroy(mesh->globalScatter);CHKERRQ(ierr);
    mesh->globalScatter = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateMatrix" 
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateMatrix(Mesh mesh, SectionReal section, MatType mtype, Mat *J)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ierr = MeshCreateMatrix(m, s, mtype, J);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) *J, "mesh", (PetscObject) mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetVertexMatrix" 
PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexMatrix(Mesh mesh, MatType mtype, Mat *J)
{
  SectionReal    section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetVertexSectionReal(mesh, 1, &section);CHKERRQ(ierr);
  ierr = MeshCreateMatrix(mesh, section, mtype, J);CHKERRQ(ierr);
  ierr = SectionRealDestroy(section);CHKERRQ(ierr);
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
PetscErrorCode PETSCDM_DLLEXPORT MeshGetMatrix(Mesh mesh, MatType mtype, Mat *J)
{
  ALE::Obj<ALE::Mesh> m;
  PetscTruth          flag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshHasSectionReal(mesh, "default", &flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshCreateMatrix(m, m->getRealSection("default"), mtype, J);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) *J, "mesh", (PetscObject) mesh);CHKERRQ(ierr);
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

  ierr = PetscHeaderCreate(p,_p_Mesh,struct _MeshOps,MESH_COOKIE,0,"Mesh",comm,MeshDestroy,MeshView);CHKERRQ(ierr);
  p->ops->view               = MeshView_Mesh;
  p->ops->destroy            = PETSC_NULL;
  p->ops->createglobalvector = MeshCreateGlobalVector;
  p->ops->createlocalvector  = MeshCreateLocalVector;
  p->ops->getcoloring        = PETSC_NULL;
  p->ops->getmatrix          = MeshGetMatrix;
  p->ops->getinterpolation   = MeshGetInterpolation_Mesh_New;
  p->ops->getinjection       = PETSC_NULL;
  p->ops->refine             = MeshRefine_Mesh;
  p->ops->coarsen            = PETSC_NULL;
  p->ops->refinehierarchy    = PETSC_NULL;
  p->ops->coarsenhierarchy   = MeshCoarsenHierarchy_Mesh;

  ierr = PetscObjectChangeTypeName((PetscObject) p, "sieve");CHKERRQ(ierr);

  new(&p->m) ALE::Obj<ALE::Mesh>(PETSC_NULL);
  p->globalScatter = PETSC_NULL;
  p->lf            = PETSC_NULL;
  p->lj            = PETSC_NULL;
  new(&p->mcompat) ALE::Obj<ALECompat::Mesh>(PETSC_NULL);
  p->data          = PETSC_NULL;
  *mesh = p;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshDestroy"
/*@
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
  if (--((PetscObject)mesh)->refct > 0) PetscFunctionReturn(0);
  if (mesh->globalScatter) {ierr = VecScatterDestroy(mesh->globalScatter);CHKERRQ(ierr);}
  mesh->m = PETSC_NULL;
  ierr = PetscHeaderDestroy(mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetType"
/*@C
  MeshSetType - Sets the Mesh type

  Collective on Mesh

  Input Parameters:
+ mesh - the Mesh context
- type - the type

  Options Database Key:
. -mesh_type  <method> - Sets the type; use -help for a list 
    of available types (for instance, cartesian or sieve)

  Notes:
  See "petsc/include/petscmesh.h" for available types (for instance,
  MESHCARTESIAN or MESHSIEVE).

  Level: intermediate

.keywords: Mesh, set, typr
.seealso: MeshGetType(), MeshType
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshSetType(Mesh mesh, MeshType type)
{
  PetscErrorCode ierr,(*r)(Mesh);
  PetscTruth     match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh,MESH_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)mesh,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFListFind(MeshList,((PetscObject)mesh)->comm,type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested Mesh type %s",type);
  /* Destroy the previous private Mesh context */
  if (mesh->ops->destroy) { ierr = (*mesh->ops->destroy)(mesh);CHKERRQ(ierr); }
  /* Reinitialize function pointers in MeshOps structure */
  ierr = PetscMemzero(mesh->ops, sizeof(struct _MeshOps));CHKERRQ(ierr);
  /* Call the MeshCreate_XXX routine for this particular mesh */
  ierr = (*r)(mesh);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) mesh, type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetType"
/*@C
  MeshGetType - Gets the Mesh type as a string from the Mesh object.

  Not Collective

  Input Parameter:
. mesh - Mesh context 

  Output Parameter:
. name - name of Mesh type 

  Level: intermediate

.keywords: Mesh, get, type
.seealso: MeshSetType()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetType(Mesh mesh,MeshType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh,MESH_COOKIE,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)mesh)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshRegister"
/*@C
  MeshRegister - See MeshRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode PETSCKSP_DLLEXPORT MeshRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(Mesh))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&MeshList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MeshCreate_Cartesian(Mesh);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MeshRegisterAll"
/*@C
  MeshRegisterAll - Registers all of the Mesh types in the Mesh package.

  Not Collective

  Level: advanced

.keywords: Mesh, register, all
.seealso:  MeshRegisterDestroy()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MeshRegisterAllCalled = PETSC_TRUE;

  ierr = MeshRegisterDynamic(MESHCARTESIAN, path, "MeshCreate_Cartesian", MeshCreate_Cartesian);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshRegisterDestroy"
/*@
  MeshRegisterDestroy - Frees the list of Mesh types that were
  registered by MeshRegister().

  Not Collective

  Level: advanced

.keywords: Mesh, register, destroy
.seealso: MeshRegister(), MeshRegisterAll()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&MeshList);CHKERRQ(ierr);
  MeshRegisterAllCalled = PETSC_FALSE;
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
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalVector(Mesh mesh, Vec *gvec)
{
  ALE::Obj<ALE::Mesh> m;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshHasSectionReal(mesh, "default", &flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::order_type>& order = m->getFactory()->getGlobalOrder(m, "default", m->getRealSection("default"));

  ierr = VecCreate(m->comm(), gvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*gvec, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(*gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateLocalVector"
/*@C
    MeshCreateLocalVector - Creates a vector of the correct size for local computation.

    Collective on Mesh

    Input Parameter:
.    mesh - the mesh object

    Output Parameters:
.   lvec - the local vector

    Level: advanced

    Notes: Once this has been created you cannot add additional arrays or vectors to be packed.

.seealso MeshDestroy(), MeshCreate(), MeshCreateGlobalVector()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateLocalVector(Mesh mesh, Vec *lvec)
{
  ALE::Obj<ALE::Mesh> m;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshHasSectionReal(mesh, "default", &flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const int size = m->getRealSection("default")->getStorageSize();

  ierr = VecCreate(PETSC_COMM_SELF, lvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*lvec, size, size);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*lvec);CHKERRQ(ierr);
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
#define __FUNCT__ "MeshCreateGlobalScatter"
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalScatter(Mesh mesh, SectionReal section, VecScatter *scatter)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ierr = MeshCreateGlobalScatter(m, s, scatter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshGetGlobalScatter"
PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalScatter(Mesh mesh, VecScatter *scatter)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  PetscValidPointer(scatter, 2);
  if (!mesh->globalScatter) {
    SectionReal section;

    ierr = MeshGetSectionReal(mesh, "default", &section);CHKERRQ(ierr);
    ierr = MeshCreateGlobalScatter(mesh, section, &mesh->globalScatter);CHKERRQ(ierr);
    ierr = SectionRealDestroy(section);CHKERRQ(ierr);
  }
  *scatter = mesh->globalScatter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshGetLocalFunction"
PetscErrorCode PETSCDM_DLLEXPORT MeshGetLocalFunction(Mesh mesh, PetscErrorCode (**lf)(Mesh, SectionReal, SectionReal, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  if (lf) *lf = mesh->lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshSetLocalFunction"
PetscErrorCode PETSCDM_DLLEXPORT MeshSetLocalFunction(Mesh mesh, PetscErrorCode (*lf)(Mesh, SectionReal, SectionReal, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  mesh->lf = lf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshSetLocalJacobian"
PetscErrorCode PETSCDM_DLLEXPORT MeshGetLocalJacobian(Mesh mesh, PetscErrorCode (**lj)(Mesh, SectionReal, Mat, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  if (lj) *lj = mesh->lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshSetLocalJacobian"
PetscErrorCode PETSCDM_DLLEXPORT MeshSetLocalJacobian(Mesh mesh, PetscErrorCode (*lj)(Mesh, SectionReal, Mat, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  mesh->lj = lj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshFormFunction"
PetscErrorCode PETSCDM_DLLEXPORT MeshFormFunction(Mesh mesh, SectionReal X, SectionReal F, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  PetscValidHeaderSpecific(X, SECTIONREAL_COOKIE, 2);
  PetscValidHeaderSpecific(F, SECTIONREAL_COOKIE, 3);
  if (mesh->lf) {
    ierr = (*mesh->lf)(mesh, X, F, ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshFormJacobian"
PetscErrorCode PETSCDM_DLLEXPORT MeshFormJacobian(Mesh mesh, SectionReal X, Mat J, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  PetscValidHeaderSpecific(X, SECTIONREAL_COOKIE, 2);
  PetscValidHeaderSpecific(J, MAT_COOKIE, 3);
  if (mesh->lj) {
    ierr = (*mesh->lj)(mesh, X, J, ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshInterpolatePoints"
// Here we assume:
//  - Assumes 3D and tetrahedron
//  - The section takes values on vertices and is P1
//  - Points have the same dimension as the mesh
//  - All values have the same dimension
PetscErrorCode PETSCDM_DLLEXPORT MeshInterpolatePoints(Mesh mesh, SectionReal section, int numPoints, double *points, double **values)
{
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> s;
  double        *v0, *J, *invJ, detJ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  int embedDim = coordinates->getFiberDimension(*m->depthStratum(0)->begin());
  int dim      = s->getFiberDimension(*m->depthStratum(0)->begin());

  ierr = PetscMalloc3(embedDim,double,&v0,embedDim*embedDim,double,&J,embedDim*embedDim,double,&invJ);CHKERRQ(ierr);
  ierr = PetscMalloc(numPoints*dim * sizeof(double), &values);CHKERRQ(ierr);
  for(int p = 0; p < numPoints; p++) {
    double *point = &points[p*embedDim];
    
    ALE::Mesh::point_type e = m->locatePoint(point);
    const ALE::Mesh::real_section_type::value_type *coeff = s->restrictPoint(e);

    m->computeElementGeometry(coordinates, e, v0, J, invJ, detJ);
    double xi   = (invJ[0*embedDim+0]*(point[0] - v0[0]) + invJ[0*embedDim+1]*(point[1] - v0[1]) + invJ[0*embedDim+2]*(point[2] - v0[2]))*0.5;
    double eta  = (invJ[1*embedDim+0]*(point[0] - v0[0]) + invJ[1*embedDim+1]*(point[1] - v0[1]) + invJ[1*embedDim+2]*(point[2] - v0[2]))*0.5;
    double zeta = (invJ[2*embedDim+0]*(point[0] - v0[0]) + invJ[2*embedDim+1]*(point[1] - v0[1]) + invJ[2*embedDim+2]*(point[2] - v0[2]))*0.5;

    for(int d = 0; d < dim; d++) {
      (*values)[p*dim+d] = coeff[0*dim+d]*(1 - xi - eta - zeta) + coeff[1*dim+d]*xi + coeff[2*dim+d]*eta + coeff[3*dim+d]*zeta;
    }
  }
  ierr = PetscFree3(v0, J, invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshGetMaximumDegree"
/*@
  MeshGetMaximumDegree - Return the maximum degree of any mesh vertex

  Collective on mesh

  Input Parameter:
. mesh - The Mesh

  Output Parameter:
. maxDegree - The maximum number of edges at any vertex

   Level: beginner

.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetMaximumDegree(Mesh mesh, PetscInt *maxDegree)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::label_sequence>& vertices = m->depthStratum(0);
  const ALE::Obj<ALE::Mesh::sieve_type>&     sieve    = m->getSieve();
  PetscInt                                          maxDeg   = -1;

  for(ALE::Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    maxDeg = PetscMax(maxDeg, (PetscInt) sieve->support(*v_iter)->size());
  }
  *maxDegree = maxDeg;
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
    ierr = VecScatterBegin(injection, g, l, mode, SCATTER_REVERSE);
    ierr = VecScatterEnd(injection, g, l, mode, SCATTER_REVERSE);
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
    ierr = VecScatterBegin(injection, l, g, mode, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(injection, l, g, mode, SCATTER_FORWARD);CHKERRQ(ierr);
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
  Mesh                       mesh;
  ALE::Obj<ALE::Mesh> m;
  PetscInt                   firstElement;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_assembleVector,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) b, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  //firstElement = elementBundle->getLocalSizes()[bundle->getCommRank()];
  firstElement = 0;
  // Must relate b to field
  if (mode == INSERT_VALUES) {
    m->update(m->getRealSection(std::string("x")), ALE::Mesh::point_type(e + firstElement), v);
  } else {
    m->updateAdd(m->getRealSection(std::string("x")), ALE::Mesh::point_type(e + firstElement), v);
  }
  ierr = PetscLogEventEnd(Mesh_assembleVector,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperator"
PetscErrorCode updateOperator(Mat A, const ALE::Obj<ALE::Mesh>& m, const ALE::Obj<ALE::Mesh::real_section_type>& section, const ALE::Obj<ALE::Mesh::order_type>& globalOrder, const ALE::Mesh::point_type& e, PetscScalar array[], InsertMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  const ALE::Mesh::indices_type indicesBlock = m->getIndices(section, e, globalOrder);
  const PetscInt *indices    = indicesBlock.first;
  const int&      numIndices = indicesBlock.second;

  ierr = PetscLogEventBegin(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  if (section->debug()) {
    printf("[%d]mat for element %d\n", section->commRank(), e);
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]mat indices[%d] = %d\n", section->commRank(), i, indices[i]);
    }
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]", section->commRank());
      for(int j = 0; j < numIndices; j++) {
        printf(" %g", array[i*numIndices+j]);
      }
      printf("\n");
    }
  }
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, array, mode);
  if (ierr) {
    printf("[%d]ERROR in updateOperator: point %d\n", section->commRank(), e);
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]mat indices[%d] = %d\n", section->commRank(), i, indices[i]);
    }
    CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperator"
PetscErrorCode updateOperator(Mat A, const ALE::Obj<ALE::Mesh>& m, const ALE::Obj<ALE::Mesh::real_section_type>& section, const ALE::Obj<ALE::Mesh::order_type>& globalOrder, int tag, int p, PetscScalar array[], InsertMode mode)
{
  const int *offsets, *indices;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  section->getCustomRestrictAtlas(tag, &offsets, &indices);
  const int& numIndices = offsets[p+1] - offsets[p];

  ierr = PetscLogEventBegin(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, array, mode);
  if (ierr) {
    printf("[%d]ERROR in updateOperator: tag %d point num %d\n", section->commRank(), tag, p);
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]mat indices[%d] = %d\n", section->commRank(), i, indices[i]);
    }
    CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperatorGeneral"
PetscErrorCode updateOperatorGeneral(Mat A, const ALE::Obj<ALE::Mesh>& rowM, const ALE::Obj<ALE::Mesh::real_section_type>& rowSection, const ALE::Obj<ALE::Mesh::order_type>& rowGlobalOrder, const ALE::Mesh::point_type& rowE, const ALE::Obj<ALE::Mesh>& colM, const ALE::Obj<ALE::Mesh::real_section_type>& colSection, const ALE::Obj<ALE::Mesh::order_type>& colGlobalOrder, const ALE::Mesh::point_type& colE, PetscScalar array[], InsertMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  const ALE::Mesh::indices_type rowIndicesBlock = rowM->getIndices(rowSection, rowE, rowGlobalOrder);

  const PetscInt *tmpIndices    = rowIndicesBlock.first;
  const int      numRowIndices = rowIndicesBlock.second;
  PetscInt rowIndices[numRowIndices];
  PetscMemcpy(rowIndices, tmpIndices, numRowIndices*sizeof(PetscInt));

  const ALE::Mesh::indices_type colIndicesBlock = colM->getIndices(colSection, colE, colGlobalOrder);

  const PetscInt *colIndices    = colIndicesBlock.first;
  const int      numColIndices = colIndicesBlock.second;

  ierr = PetscLogEventBegin(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  if (rowSection->debug()) {
    printf("[%d]mat for elements %d %d\n", rowSection->commRank(), rowE, colE);
    for(int i = 0; i < numRowIndices; i++) {
      printf("[%d]mat row indices[%d] = %d\n", rowSection->commRank(), i, rowIndices[i]);
    }
  }
  if (colSection->debug()) {
    for(int i = 0; i < numColIndices; i++) {
      printf("[%d]mat col indices[%d] = %d\n", colSection->commRank(), i, colIndices[i]);
    }
    for(int i = 0; i < numRowIndices; i++) {
      printf("[%d]", rowSection->commRank());
      for(int j = 0; j < numColIndices; j++) {
        printf(" %g", array[i*numColIndices+j]);
      }
      printf("\n");
    }
  }
  ierr = MatSetValues(A, numRowIndices, rowIndices, numColIndices, colIndices, array, mode);
  if (ierr) {
    printf("[%d]ERROR in updateOperator: points %d %d\n", colSection->commRank(), rowE, colE);
    for(int i = 0; i < numRowIndices; i++) {
      printf("[%d]mat row indices[%d] = %d\n", rowSection->commRank(), i, rowIndices[i]);
    }
    for(int i = 0; i < numColIndices; i++) {
      printf("[%d]mat col indices[%d] = %d\n", colSection->commRank(), i, colIndices[i]);
    }
    CHKERRQ(ierr);
  }
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
  Mesh           mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_assembleMatrix,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) A, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  try {
    if (!mesh->mcompat.isNull()) {
      Obj<ALECompat::Mesh> m;

      ierr = MeshCompatGetMesh(mesh, m);CHKERRQ(ierr);
      const ALE::Obj<ALECompat::Mesh::topology_type>&     topology    = m->getTopology();
      const ALE::Obj<ALECompat::Mesh::numbering_type>&    cNumbering  = m->getFactory()->getLocalNumbering(topology, 0, topology->depth());
      const ALE::Obj<ALECompat::Mesh::real_section_type>& s           = m->getRealSection("default");
      const ALE::Obj<ALECompat::Mesh::order_type>&        globalOrder = m->getFactory()->getGlobalOrder(topology, 0, "default", s->getAtlas());

      if (m->debug()) {
        std::cout << "Assembling matrix for element number " << e << " --> point " << cNumbering->getPoint(e) << std::endl;
      }
      ierr = updateOperatorCompat(A, s, globalOrder, cNumbering->getPoint(e), v, mode);CHKERRQ(ierr);
    } else {
      Obj<ALE::Mesh> m;

      ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
      const ALE::Obj<ALE::Mesh::numbering_type>&    cNumbering  = m->getFactory()->getLocalNumbering(m, m->depth());
      const ALE::Obj<ALE::Mesh::real_section_type>& s           = m->getRealSection("default");
      const ALE::Obj<ALE::Mesh::order_type>&        globalOrder = m->getFactory()->getGlobalOrder(m, "default", s);

      if (m->debug()) {
        std::cout << "Assembling matrix for element number " << e << " --> point " << cNumbering->getPoint(e) << std::endl;
      }
      ierr = updateOperator(A, m, s, globalOrder, cNumbering->getPoint(e), v, mode);CHKERRQ(ierr);
    }
  } catch (ALE::Exception e) {
    std::cout << e.msg() << std::endl;
  }
  ierr = PetscLogEventEnd(Mesh_assembleMatrix,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "preallocateMatrix"
PetscErrorCode preallocateMatrix(const ALE::Obj<ALE::Mesh>& mesh, const int bs, const ALE::Obj<ALE::Mesh::real_section_type::atlas_type>& atlas, const ALE::Obj<ALE::Mesh::order_type>& globalOrder, Mat A)
{
  return preallocateOperator(mesh, bs, atlas, globalOrder, A);
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

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return VTKViewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKElements"
PetscErrorCode WriteVTKElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return VTKViewer::writeElements(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEVertices"
PetscErrorCode WritePCICEVertices(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEElements"
PetscErrorCode WritePCICEElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeElements(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICERestart"
PetscErrorCode WritePCICERestart(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeRestart(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreatePFLOTRAN"
/*@C
  MeshCreatePFLOTRAN - Create a Mesh from PFLOTRAN HDF5 files.

  Not Collective

  Input Parameters:
+ dim - The topological mesh dimension
. hdf5Filename - The HDF5 file containing the vertices for each element and vertex coordinates
. interpolate - The flag for construction of intermediate elements

  Output Parameter:
. mesh - The Mesh object

  Level: beginner

.keywords: mesh, PFLOTRAN
.seealso: MeshCreate()
@*/
PetscErrorCode MeshCreatePFLOTRAN(MPI_Comm comm, const int dim, const char hdf5Filename[], PetscTruth interpolate, Mesh *mesh)
{
  ALE::Obj<ALE::Mesh> m;
  PetscInt            debug = 0;
  PetscTruth          flag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  try {
    m  = ALE::PFLOTRAN::Builder::readMesh(comm, dim, std::string(hdf5Filename), true, interpolate, debug);
    if (debug) {m->view("Mesh");}
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_ERR_FILE_OPEN, e.message());
  }
#if 0
  if (bcFilename) {
    ALE::PFLOTRAN::Builder::readBoundary(m, std::string(bcFilename));
  }
#endif
  ierr = MeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreatePCICE"
/*@C
  MeshCreatePCICE - Create a Mesh from PCICE files.

  Not Collective

  Input Parameters:
+ dim - The topological mesh dimension
. coordFilename - The file containing vertex coordinates
. adjFilename - The file containing the vertices for each element
. interpolate - The flag for construction of intermediate elements
. bcFilename - The file containing the boundary topology and conditions
. numBdFaces - The number of boundary faces (or edges)
- numBdVertices - The number of boundary vertices

  Output Parameter:
. mesh - The Mesh object

  Level: beginner

.keywords: mesh, PCICE
.seealso: MeshCreate()
@*/
PetscErrorCode MeshCreatePCICE(MPI_Comm comm, const int dim, const char coordFilename[], const char adjFilename[], PetscTruth interpolate, const char bcFilename[], Mesh *mesh)
{
  ALE::Obj<ALE::Mesh> m;
  PetscInt            debug = 0;
  PetscTruth          flag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  try {
    m  = ALE::PCICE::Builder::readMesh(comm, dim, std::string(coordFilename), std::string(adjFilename), false, interpolate, debug);
    if (debug) {m->view("Mesh");}
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_ERR_FILE_OPEN, e.message());
  }
  if (bcFilename) {
    ALE::PCICE::Builder::readBoundary(m, std::string(bcFilename));
  }
  ierr = MeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreatePyLith"
/*@C
  MeshCreatePyLith - Create a Mesh from PyLith files.

  Not Collective

  Input Parameters:
+ dim - The topological mesh dimension
. baseFilename - The basename for mesh files
. zeroBase - Use 0 to start numbering
- interpolate - The flag for mesh interpolation

  Output Parameter:
. mesh - The Mesh object

  Level: beginner

.keywords: mesh, PCICE
.seealso: MeshCreate()
@*/
PetscErrorCode MeshCreatePyLith(MPI_Comm comm, const int dim, const char baseFilename[], PetscTruth zeroBase, PetscTruth interpolate, Mesh *mesh)
{
  ALE::Obj<ALE::Mesh> m;
  PetscInt            debug = 0;
  PetscTruth          flag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  try {
    m  = ALE::PyLith::Builder::readMesh(comm, dim, std::string(baseFilename), zeroBase, interpolate, debug);
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_ERR_FILE_OPEN, e.message());
  }
  ierr = MeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetCoordinates"
/*@C
  MeshGetCoordinates - Creates an array holding the coordinates.

  Not Collective

  Input Parameter:
+ mesh - The Mesh object
- columnMajor - Flag for column major order

  Output Parameter:
+ numVertices - The number of vertices
. dim - The embedding dimension
- coords - The array holding local coordinates

  Level: intermediate

.keywords: mesh, coordinates
.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetCoordinates(Mesh mesh, PetscTruth columnMajor, PetscInt *numVertices, PetscInt *dim, PetscReal *coords[])
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ALE::PCICE::Builder::outputVerticesLocal(m, numVertices, dim, coords, columnMajor);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetElements"
/*@C
  MeshGetElements - Creates an array holding the vertices on each element.

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- columnMajor - Flag for column major order

  Output Parameters:
+ numElements - The number of elements
. numCorners - The number of vertices per element
- vertices - The array holding vertices on each local element

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetElements(Mesh mesh, PetscTruth columnMajor, PetscInt *numElements, PetscInt *numCorners, PetscInt *vertices[])
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ALE::PCICE::Builder::outputElementsLocal(m, numElements, numCorners, vertices, columnMajor);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshDistribute"
/*@C
  MeshDistribute - Distributes the mesh and any associated sections.

  Not Collective

  Input Parameter:
+ serialMesh  - The original Mesh object
- partitioner - The partitioning package, or NULL for the default

  Output Parameter:
. parallelMesh - The distributed Mesh object

  Level: intermediate

.keywords: mesh, elements

.seealso: MeshCreate(), MeshDistributeByFace()
@*/
PetscErrorCode MeshDistribute(Mesh serialMesh, const char partitioner[], Mesh *parallelMesh)
{
  ALE::Obj<ALE::Mesh> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(serialMesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(oldMesh->comm(), parallelMesh);CHKERRQ(ierr);
  if (partitioner == NULL) {
    ALE::Obj<ALE::Mesh> newMesh = ALE::Distribution<ALE::Mesh>::distributeMesh(oldMesh);
    ierr = MeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  } else {
    ALE::Obj<ALE::Mesh> newMesh = ALE::Distribution<ALE::Mesh>::distributeMesh(oldMesh, 0, partitioner);
    ierr = MeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshDistributeByFace"
/*@C
  MeshDistribute - Distributes the mesh and any associated sections.

  Not Collective

  Input Parameter:
+ serialMesh  - The original Mesh object
- partitioner - The partitioning package, or NULL for the default

  Output Parameter:
. parallelMesh - The distributed Mesh object

  Level: intermediate

.keywords: mesh, elements

.seealso: MeshCreate(), MeshDistribute()
@*/
PetscErrorCode MeshDistributeByFace(Mesh serialMesh, const char partitioner[], Mesh *parallelMesh)
{
  ALE::Obj<ALE::Mesh> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(serialMesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(oldMesh->comm(), parallelMesh);CHKERRQ(ierr);
  if (partitioner == NULL) {
    ALE::Obj<ALE::Mesh> newMesh = ALE::Distribution<ALE::Mesh>::distributeMesh(oldMesh, 1);
    ierr = MeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  } else {
    ALE::Obj<ALE::Mesh> newMesh = ALE::Distribution<ALE::Mesh>::distributeMesh(oldMesh, 1, partitioner);
    ierr = MeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGenerate"
/*@C
  MeshGenerate - Generates a mesh.

  Not Collective

  Input Parameters:
+ boundary - The Mesh boundary object
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. mesh - The Mesh object

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate(), MeshRefine()
@*/
PetscErrorCode MeshGenerate(Mesh boundary, PetscTruth interpolate, Mesh *mesh)
{
  ALE::Obj<ALE::Mesh> mB;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(boundary, mB);CHKERRQ(ierr);
  ierr = MeshCreate(mB->comm(), mesh);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh> m = ALE::Generator::generateMesh(mB, interpolate);
  ierr = MeshSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshRefine"
/*@C
  MeshRefine - Refines the mesh.

  Not Collective

  Input Parameters:
+ mesh - The original Mesh object
. refinementLimit - The maximum size of any cell
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. refinedMesh - The refined Mesh object

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate(), MeshGenerate()
@*/
PetscErrorCode MeshRefine(Mesh mesh, double refinementLimit, PetscTruth interpolate, Mesh *refinedMesh)
{
  ALE::Obj<ALE::Mesh> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(oldMesh->comm(), refinedMesh);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh> newMesh = ALE::Generator::refineMesh(oldMesh, refinementLimit, interpolate);
  ierr = MeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshRefine_Mesh"
PetscErrorCode MeshRefine_Mesh(Mesh mesh, MPI_Comm comm, Mesh *refinedMesh)
{
  ALE::Obj<ALE::Mesh> oldMesh;
  double              refinementLimit;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(comm, refinedMesh);CHKERRQ(ierr);
  refinementLimit = oldMesh->getMaxVolume()/2.0;
  ALE::Obj<ALE::Mesh> newMesh = ALE::Generator::refineMesh(oldMesh, refinementLimit, true);
  ierr = MeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::real_section_type>& s = newMesh->getRealSection("default");
  const Obj<std::set<std::string> >& discs = oldMesh->getDiscretizations();

  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    newMesh->setDiscretization(*f_iter, oldMesh->getDiscretization(*f_iter));
  }
  newMesh->setupField(s);
  PetscFunctionReturn(0);
}


#include "Hierarchy_New.hh"

#undef __FUNCT__
#define __FUNCT__ "MeshCoarsenHierarchy_New"


#include "Hierarchy.hh"

#undef __FUNCT__  
#define __FUNCT__ "MeshCoarsenHierarchy"
/*@C
  MeshCoarsenHierarchy - Coarsens the mesh into a hierarchy.

  Not Collective

  Input Parameters:
+ mesh - The original Mesh object
. numLevels - The number of 
. coarseningFactor - The expansion factor for coarse meshes
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. coarseHierarchy - The coarse Mesh objects

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate(), MeshGenerate()
@*/
PetscErrorCode MeshCoarsenHierarchy(Mesh mesh, int numLevels, double coarseningFactor, PetscTruth interpolate, Mesh **coarseHierarchy)
{
  ALE::Obj<ALE::Mesh> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (numLevels < 1) {
    *coarseHierarchy = PETSC_NULL;
    PetscFunctionReturn(0);
  }
  ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = PetscMalloc((numLevels+1) * sizeof(Mesh), coarseHierarchy);CHKERRQ(ierr);
  for (int i = 0; i < numLevels+1; i++) {
    ierr = MeshCreate(oldMesh->comm(), &(*coarseHierarchy)[i]);CHKERRQ(ierr);
  }
  ierr = MeshSpacingFunction(mesh);CHKERRQ(ierr);
  ierr = MeshCreateHierarchyLabel_Link(mesh, coarseningFactor, numLevels+1, *coarseHierarchy);
  
#if 0
  if (oldMesh->getDimension() != 2) SETERRQ(PETSC_ERR_SUP, "Coarsening only works in two dimensions right now");
  ierr = ALE::Coarsener::IdentifyBoundary(oldMesh, 2);CHKERRQ(ierr);
  ierr = ALE::Coarsener::make_coarsest_boundary(oldMesh, 2, numLevels+1);CHKERRQ(ierr);
  ierr = ALE::Coarsener::CreateSpacingFunction(oldMesh, 2);CHKERRQ(ierr);
  ierr = ALE::Coarsener::CreateCoarsenedHierarchyNew(oldMesh, 2, numLevels, coarseningFactor);CHKERRQ(ierr);
  ierr = PetscMalloc(numLevels * sizeof(Mesh),coarseHierarchy);CHKERRQ(ierr);
  for(int l = 0; l < numLevels; l++) {
    ALE::Obj<ALE::Mesh> newMesh = new ALE::Mesh(oldMesh->comm(), oldMesh->debug());
    const ALE::Obj<ALE::Mesh::real_section_type>& s = newMesh->getRealSection("default");

    ierr = MeshCreate(oldMesh->comm(), &(*coarseHierarchy)[l]);CHKERRQ(ierr);
    newMesh->getTopology()->setPatch(0, oldMesh->getTopology()->getPatch(l+1));
    newMesh->setDiscretization(oldMesh->getDiscretization());
    newMesh->setBoundaryCondition(oldMesh->getBoundaryCondition());
    newMesh->setupField(s);
    ierr = MeshSetMesh((*coarseHierarchy)[l], newMesh);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MeshCoarsenHierarchy_Mesh(Mesh mesh, int numLevels, Mesh **coarseHierarchy)
{
  PetscErrorCode ierr;
  double cfactor = 1.5;
  PetscFunctionBegin;
  ierr = PetscOptionsReal("-dmmg_coarsen_factor", "The coarsening factor", PETSC_NULL, cfactor, &cfactor, PETSC_NULL);CHKERRQ(ierr);
  ierr = MeshCoarsenHierarchy(mesh, numLevels, cfactor, PETSC_FALSE, coarseHierarchy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0

#undef __FUNCT__
#define __FUNCT__ "MeshGetInterpolation_Mesh_General"

//Interpolate between two meshes whenever the unknowns can be evaluated at points.

PetscErrorCode MeshGetInterpolation_Mesh_General(Mesh coarse_mesh, Mesh fine_mesh, Mat *interpolation, Vec *scaling) {
  ALE::Obj<ALE::Mesh> fm, cm;
  Mat                 P;
  PetscErrorCode      ierr;
  
  PetscFunctionBegin;
  //Stages: 
  //  1. Create a section on the fine mesh describing the location in the fine mesh of the assorted unknowns.
  //  2. Fill in this section by traversing across the mesh via cones and supports, transforming the coordinates of the assorted functional points
  //  3. Preallocate the matrix rows/columns
  //  4. Assemble the matrix by writing evaluating each unknown as the point 
  ierr = MeshGetMesh(dmFine, fm);CHKERRQ(ierr);
  ierr = MeshGetMesh(dmCoarse, cm);CHKERRQ(ierr);
  //  ALE::Obj<ALE::Mesh::label_type> coarsetraversal = cm->createLabel("traversal");
  //  ALE::Obj<ALE::Mesh::label_type> finetraversal   = fm->createLabel ("traversal");
  const int                       debug           = fm->debug();
  if (debug) {ierr = PetscPrintf(fm->comm(), "Fine: %d vertices, Coarse: %d vertices\n", fm->depthStratum(0)->size(), cm->depthStratum(0)->size());CHKERRQ(ierr);}
  const ALE::Obj<ALE::Mesh::real_section_type>& finecoordinates   = fm->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::real_section_type>& coarsecoordinates = cm->getRealSection("coordinates");

  const ALE::Obj<ALE::Mesh::real_section_type>& sCoarse           = cm->getRealSection("default");
  const ALE::Obj<ALE::Mesh::real_section_type>& sFine             = fm->getRealSection("default");

  const ALE::Obj<ALE::Mesh::order_type>&        coarseOrder       = cm->getFactory()->getGlobalOrder(cm, "default", sCoarse);
  const ALE::Obj<ALE::Mesh::order_type>&        fineOrder         = fm->getFactory()->getGlobalOrder(fm, "default", sFine);

  std::list<ALE::Mesh::point_type> travlist;        // store point
  std::list<ALE::Mesh::point_type> travguesslist;   // store guess
  std::list<ALE::Mesh::point_type> eguesslist;      // store the next guesses for the location of the current point.

  const ALE::Obj<ALE::Mesh::sieve_type::supportSet> coarse_traversal = ALE::Mesh::sieve_type::supportSet();
  const ALE::Obj<ALE::Mesh::sieve_type::supportSet> fine_traversal = ALE::Mesh::sieve_type::supportSet();
  const ALE::Obj<ALE::Mesh::sieve_type::supportSet> covering_points = ALE::Mesh::sieve_type::supportSet();

  const ALE::Obj<ALE::Mesh::sieve_type::supportSet> uncorrected_points = ALE::Mesh::sieve_type::supportSet();
  static double loc[4], v0[3], J[9], invJ[9], detJ; // first point, jacobian, inverse jacobian, and jacobian determinant of a cell
  if (debug) {ierr = PetscPrintf(fm->comm(), "Starting Interpolation Matrix Build\n");CHKERRQ(ierr);}

  //set up the new section holding the names of the contained points.  
  
  

  const ALE::Obj<ALE::Mesh::int_section_type> & node_locations = fm->getIntSection("node_locations");
  for (int i = 0; i < dim; i++) {
    const ALE::Obj<ALE::Mesh::label_sequence> & present_level = fm->depthStratum(i);
    int current_dimension = 
    node_locations->setFiberDimension(present_level);
  }
  node_locations->allocate();

  

}

#endif


#undef __FUNCT__
#define __FUNCT__ "MeshGetInterpolation_Mesh_New"

PetscErrorCode MeshGetInterpolation_Mesh_New(Mesh dmCoarse, Mesh dmFine, Mat *interpolation, Vec *scaling) {

  ALE::Obj<ALE::Mesh> fm, cm;
  Mat                 P;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(dmFine, fm);CHKERRQ(ierr);
  ierr = MeshGetMesh(dmCoarse, cm);CHKERRQ(ierr);
  //  ALE::Obj<ALE::Mesh::label_type> coarsetraversal = cm->createLabel("traversal");
  //  ALE::Obj<ALE::Mesh::label_type> finetraversal   = fm->createLabel ("traversal");
  const int                       debug           = fm->debug();
  if (debug) {ierr = PetscPrintf(fm->comm(), "Fine: %d vertices, Coarse: %d vertices\n", fm->depthStratum(0)->size(), cm->depthStratum(0)->size());CHKERRQ(ierr);}
  const ALE::Obj<ALE::Mesh::real_section_type>& finecoordinates   = fm->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::real_section_type>& coarsecoordinates = cm->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::real_section_type>& sCoarse           = cm->getRealSection("default");
  const ALE::Obj<ALE::Mesh::real_section_type>& sFine             = fm->getRealSection("default");
  const ALE::Obj<ALE::Mesh::order_type>&        coarseOrder       = cm->getFactory()->getGlobalOrder(cm, "default", sCoarse);
  const ALE::Obj<ALE::Mesh::order_type>&        fineOrder         = fm->getFactory()->getGlobalOrder(fm, "default", sFine);
  std::list<ALE::Mesh::point_type> travlist;        // store point
  std::list<ALE::Mesh::point_type> travguesslist;   // store guess
  std::list<ALE::Mesh::point_type> eguesslist;      // store the next guesses for the location of the current point.
  const ALE::Obj<ALE::Mesh::sieve_type::supportSet> coarse_traversal = ALE::Mesh::sieve_type::supportSet();
  const ALE::Obj<ALE::Mesh::sieve_type::supportSet> fine_traversal = ALE::Mesh::sieve_type::supportSet();
  const ALE::Obj<ALE::Mesh::sieve_type::supportSet> uncorrected_points = ALE::Mesh::sieve_type::supportSet();
  static double loc[4], v0[3], J[9], invJ[9], detJ; // first point, jacobian, inverse jacobian, and jacobian determinant of a cell
  if (debug) {ierr = PetscPrintf(fm->comm(), "Starting Interpolation Matrix Build\n");CHKERRQ(ierr);}

  ierr = MatCreate(fm->comm(), &P);CHKERRQ(ierr);
  ierr = MatSetSizes(P, sFine->size(), sCoarse->size(), PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(P,10,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(P);CHKERRQ(ierr);

  const int dim = fm->getDimension();
  int maxComparisons = 60; //point is considered a lost cause beyond this many comparisons with volumes
  if (dim == 3) maxComparisons = 1000; //3D is odd
  if (dim != cm->getDimension()) throw ALE::Exception("Dimensions of the fine and coarse meshes do not match"); 

  //traversal labels on both layers
  const ALE::Obj<ALE::Mesh::label_sequence>& finevertices = fm->depthStratum(0);
  const ALE::Mesh::label_sequence::iterator  fv_iter_end  = finevertices->end();
  ALE::Mesh::label_sequence::iterator        fv_iter      = finevertices->begin();

  //  while (fv_iter != fv_iter_end) {
  //    fm->setValue(finetraversal, *fv_iter, 0);
  //    fv_iter++;
  //  }

  const ALE::Obj<ALE::Mesh::label_sequence>& coarseelements = cm->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator  ce_iter_end    = coarseelements->end();
  ALE::Mesh::label_sequence::iterator        ce_iter        = coarseelements->begin();
  
  //  while (ce_iter != ce_iter_end) {
  //    cm->setValue(coarsetraversal, *ce_iter, 0);
  //    ce_iter++;
  //  }

  double fvCoords[dim], nvCoords[dim];
  bool pointIsInElement;

  if (debug) {ierr = PetscPrintf(fm->comm(), "starting iterations\n");CHKERRQ(ierr);}
  fv_iter = finevertices->begin();
  while (fv_iter != fv_iter_end) {
    // locate an initial point.
    //    if (fm->getValue(finetraversal, *fv_iter) == 0) {
    if ((fine_traversal->find(*fv_iter) == fine_traversal->end()) && (uncorrected_points->find(*fv_iter) == uncorrected_points->end())) {
      bool isLocated = false;

      ce_iter = coarseelements->begin();
      ierr = PetscMemcpy(fvCoords, finecoordinates->restrictPoint(*fv_iter), dim*sizeof(double));
      while ((ce_iter != ce_iter_end) && (!isLocated)) {
        cm->computeElementGeometry(coarsecoordinates, *ce_iter, v0, J, invJ, detJ);
        // generalized simplicial location for 2D, 3D:
        loc[0] = 1.0;
        pointIsInElement = true;
        for(int i = 0; i < dim; i++) {
          loc[i+1] = 0.0;
          for(int j = 0; j < dim; j++) {
            loc[i+1] += 0.5*invJ[i*dim+j]*(fvCoords[j] - v0[j]);
          }
          loc[0] -= loc[i+1];
          //PetscPrintf(fm->comm(), "%f, ", loc[i+1]);
          if (loc[i+1] < -0.000000000001) pointIsInElement = false;
        }
        //PetscPrintf(fm->comm(), "%f\n", loc[0]);
        if (loc[0] < -0.000000000001) pointIsInElement = false;
        if (pointIsInElement) {
          //PetscPrintf(fm->comm(), "%f, %f, %f\n", loc[0], loc[1], loc[2]);
          //PetscPrintf(fm->comm(), "located by guess.\n");
          isLocated = true;
          ierr = updateOperatorGeneral(P, fm, sFine, fineOrder, *fv_iter, cm, sCoarse, coarseOrder, *ce_iter, loc, INSERT_VALUES);CHKERRQ(ierr);
          //fm->setValue(finetraversal, *fv_iter, 1);
          fine_traversal->insert(*fv_iter);
          const ALE::Obj<ALE::Mesh::sieve_type::coneSet> & neighbors  = fm->getSieve()->cone(fm->getSieve()->support(*fv_iter));
          const ALE::Mesh::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
          ALE::Mesh::sieve_type::coneSet::iterator       n_iter     = neighbors->begin();
          while (n_iter != n_iter_end) {
	    //            if (fm->getValue(finetraversal, *n_iter) == 0) {
	    if (fine_traversal->find(*n_iter) != fine_traversal->end()) {
              travlist.push_back(*n_iter);
	      //              fm->setValue(finetraversal, *n_iter, 1);
              fine_traversal->insert(*n_iter);
              travguesslist.push_back(*ce_iter);
            }
            n_iter++;
          }
          //do a DFS across the finemesh with BFSes on the coarse mesh for each point using assumed regularity of edgelength as a justification for guessing neighboring point's locations.
          while (!travlist.empty()) {
            ALE::Mesh::point_type curVert = *travlist.begin();
            PetscMemcpy(nvCoords, finecoordinates->restrictPoint(curVert), dim*sizeof(double));
            ALE::Mesh::point_type curEle =  *travguesslist.begin();
            travlist.pop_front();
            travguesslist.pop_front();
            eguesslist.push_front(curEle);
            //cm->setValue(coarsetraversal, curEle, 1);
            coarse_traversal->insert(curEle);
            bool locationDiscovered  = false;
            //int traversalcomparisons = 0;
            while ((!eguesslist.empty()) && (!locationDiscovered) && (int)coarse_traversal->size() < maxComparisons) {
              //traversalcomparisons = 0;
              ALE::Mesh::point_type curguess = *eguesslist.begin();
              eguesslist.pop_front();
              pointIsInElement = true;
              cm->computeElementGeometry(coarsecoordinates, curguess, v0, J, invJ, detJ);
              loc[0] = 1.0;
              for(int i = 0; i < dim; i++) {
                loc[i+1] = 0.0;
                for(int j = 0; j < dim; j++) {
                  loc[i+1] += 0.5*invJ[i*dim+j]*(nvCoords[j] - v0[j]);
                }
                loc[0] -= loc[i+1];
                if (loc[i+1] < -0.00000000001) pointIsInElement = false;
              }
              if (loc[0] < -0.00000000001) pointIsInElement = false;

              if (pointIsInElement) {
                //PetscPrintf(fm->comm(), "%f, %f, %f\n", loc[0], loc[1], loc[2]);
                locationDiscovered = true;
                //PetscPrintf(fm->comm(), "located by traversal.\n");
                //set the label.
                //fm->setValue(prolongation, curVert, curguess);
                ierr = updateOperatorGeneral(P, fm, sFine, fineOrder, curVert, cm, sCoarse, coarseOrder, curguess, loc, INSERT_VALUES);CHKERRQ(ierr);
                //PetscPrintf(fm->comm(), "Point %d located in %d.\n",  curVert, curguess);
                //stick its neighbors in the queue along with its location as a good guess of the location of its neighbors
                const ALE::Obj<ALE::Mesh::sieve_type::coneSet> newNeighbors = fm->getSieve()->cone(fm->getSieve()->support(curVert));
                const ALE::Mesh::sieve_type::coneSet::iterator nn_iter_end  = newNeighbors->end();
                ALE::Mesh::sieve_type::coneSet::iterator       nn_iter      = newNeighbors->begin();
                while (nn_iter != nn_iter_end) {
		  //if (fm->getValue(finetraversal, *nn_iter) == 0) { //unlocated neighbor
                  if (fine_traversal->find(*nn_iter) == fine_traversal->end()) {
                    travlist.push_back(*nn_iter);
                    travguesslist.push_back(curguess);
                    //fm->setValue(finetraversal, *nn_iter, 1);
                    fine_traversal->insert(*nn_iter);
                  }
                  nn_iter++;
                }
              } else {
              //add the current guesses neighbors to the comparison queue and start over.
                const ALE::Obj<ALE::Mesh::sieve_type::supportSet> & curguessneighbors = cm->getSieve()->support(cm->getSieve()->cone(curguess));
                const ALE::Mesh::sieve_type::supportSet::iterator cgn_iter_end      = curguessneighbors->end();
                ALE::Mesh::sieve_type::supportSet::iterator       cgn_iter          = curguessneighbors->begin();
                while (cgn_iter != cgn_iter_end) {
                  //if (cm->getValue(coarsetraversal, *cgn_iter) == 0) {
                  if (coarse_traversal->find(*cgn_iter) == coarse_traversal->end()) {
                    eguesslist.push_back(*cgn_iter);
                    //cm->setValue(coarsetraversal, *cgn_iter, 1);
                    coarse_traversal->insert(*cgn_iter);
                  }
                  cgn_iter++;
                }
              }
            }
            coarse_traversal->clear();
            if (!locationDiscovered) {  //if a position for it is not discovered, it doesn't get corrected; complain
              if (fm->debug())PetscPrintf(fm->comm(), "Point %d (%f, %f) not located.\n",  curVert, nvCoords[0], nvCoords[1]);
              //fm->setValue(finetraversal, curVert, 2); //don't try again.
              uncorrected_points->insert(curVert);
            }
            eguesslist.clear(); //we've discovered the location of the point or exhausted our possibilities on this contiguous block of elements.
            //unset the traversed element list
            //const ALE::Obj<ALE::Mesh::label_sequence>& traved_elements = cm->getLabelStratum("traversal", 1);
            //const ALE::Mesh::label_sequence::iterator  tp_iter_end     = traved_elements->end();
            //ALE::Mesh::label_sequence::iterator        tp_iter         = traved_elements->begin();
            //PetscPrintf(cm->comm(), "%d\n", traved_elements->size());
            //while (tp_iter != tp_iter_end) {
            //  eguesslist.push_back(*tp_iter);
            //  tp_iter++;
            //}
            //while (!eguesslist.empty()) {
            //  cm->setValue(coarsetraversal, *eguesslist.begin(), 0);
            //  eguesslist.pop_front();
            //}
            
          }
        }
        ce_iter++;
      }
      if (!isLocated) {
       if (fm->debug())ierr = PetscPrintf(fm->comm(), "NOT located\n");CHKERRQ(ierr);
       //fm->setValue(finetraversal, *fv_iter, 2); //don't try again.
       uncorrected_points->insert(*fv_iter);
      }
    }
    // printf("-");
    fv_iter++;
  }
  ierr = MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //MatView(P, PETSC_VIEWER_STDOUT_SELF);
  *interpolation = P;
  if (debug) {ierr = PetscPrintf(fm->comm(), "Ending Interpolation Matrix Build\n");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MeshGetInterpolation_Mesh"
/*
  This method only handle P_1 discretizations at present.
*/
PetscErrorCode MeshGetInterpolation_Mesh(Mesh dmCoarse, Mesh dmFine, Mat *interpolation, Vec *scaling)
{
  ALE::Obj<ALE::Mesh> coarse;
  ALE::Obj<ALE::Mesh> fine;
  Mat                 P;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(dmFine,   fine);CHKERRQ(ierr);
  ierr = MeshGetMesh(dmCoarse, coarse);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::real_section_type>& coarseCoordinates = coarse->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::real_section_type>& fineCoordinates   = fine->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::label_sequence>&    vertices          = fine->depthStratum(0);
  const ALE::Obj<ALE::Mesh::real_section_type>& sCoarse           = coarse->getRealSection("default");
  const ALE::Obj<ALE::Mesh::real_section_type>& sFine             = fine->getRealSection("default");
  const ALE::Obj<ALE::Mesh::order_type>&        coarseOrder = coarse->getFactory()->getGlobalOrder(coarse, "default", sCoarse);
  const ALE::Obj<ALE::Mesh::order_type>&        fineOrder   = fine->getFactory()->getGlobalOrder(fine, "default", sFine);

  const int dim    = coarse->getDimension();
  const int numDof = fine->getDiscretization()->getNumDof(fine->getDimension());
  double *v0, *J, *invJ, detJ, *refCoords, *values;

  ierr = MatCreate(fine->comm(), &P);CHKERRQ(ierr);
  ierr = MatSetSizes(P, sFine->size(), sCoarse->size(), PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(P);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(P, numDof, PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(P, numDof, PETSC_NULL, numDof, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ,dim,double,&refCoords,dim+1,double,&values);CHKERRQ(ierr);
  bool hasprolong;
  if (fine->hasLabel("prolongation")) { 
    hasprolong = true;
  } else {
    hasprolong = false;
    PetscPrintf(fine->comm(), "WARNING: Point Location Label Does Not Exist");
  }
  ALE::Mesh::label_sequence::iterator v_iter_end = vertices->end();
  ALE::Mesh::real_section_type::value_type coords[dim];

  for(ALE::Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != v_iter_end; ++v_iter) {
    //const ALE::Mesh::real_section_type::value_type *coords     = fineCoordinates->restrictPoint(*v_iter);
    ierr = PetscMemcpy(coords, fineCoordinates->restrictPoint(*v_iter), dim*sizeof(double));CHKERRQ(ierr);
    ALE::Mesh::point_type coarseCell;
    ALE::Mesh::point_type cellguess = -1;
    if (hasprolong) {
      cellguess = fine->getValue(fine->getLabel("prolongation"), *v_iter);
      coarseCell = coarse->locatePoint(coords, cellguess);
    } else {
      coarseCell = coarse->locatePoint(coords);
    }
//      coarseCell = coarse->locatePoint(coords);
    if (coarseCell == -1) {
     // do NO CORRECTION!
    } else {
      coarse->computeElementGeometry(coarseCoordinates, coarseCell, v0, J, invJ, detJ);
      for(int d = 0; d < dim; ++d) {
        refCoords[d] = 0.0;
        for(int e = 0; e < dim; ++e) {
          refCoords[d] += invJ[d*dim+e]*(coords[e] - v0[e]);
        }
        refCoords[d] -= 1.0;
      }
      values[0] = -(refCoords[0] + refCoords[1])/2.0;
      values[1] = 0.5*(refCoords[0] + 1.0);
      values[2] = 0.5*(refCoords[1] + 1.0);
  //    PetscPrintf(fine->comm(), "%f, %f, %f\n", values[0], values[1], values[2]);
      ierr = updateOperatorGeneral(P, fine, sFine, fineOrder, *v_iter, coarse, sCoarse, coarseOrder, coarseCell, values, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree5(v0,J,invJ,refCoords,values);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *interpolation = P;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshHasSectionReal"
/*@C
  MeshHasSectionReal - Determines whether this mesh has a SectionReal with the given name.

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameter:
. flag - True if the SectionReal is present in the Mesh

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshHasSectionReal(Mesh mesh, const char name[], PetscTruth *flag)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  *flag = (PetscTruth) m->hasRealSection(std::string(name));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetSectionReal"
/*@C
  MeshGetSectionReal - Returns a SectionReal of the given name from the Mesh.

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameter:
. section - The SectionReal

  Note: The section is a new object, and must be destroyed by the user

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetSectionReal(Mesh mesh, const char name[], SectionReal *section)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  ierr = SectionRealSetSection(*section, m->getRealSection(std::string(name)));CHKERRQ(ierr);
  ierr = SectionRealSetBundle(*section, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetSectionReal"
/*@C
  MeshSetSectionReal - Puts a SectionReal of the given name into the Mesh.

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- section - The SectionReal

  Note: This takes the section name from the PETSc object

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshSetSectionReal(Mesh mesh, SectionReal section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::real_section_type> s;
  const char         *name;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  m->setRealSection(std::string(name), s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshHasSectionInt"
/*@C
  MeshHasSectionInt - Determines whether this mesh has a SectionInt with the given name.

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameter:
. flag - True if the SectionInt is present in the Mesh

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshHasSectionInt(Mesh mesh, const char name[], PetscTruth *flag)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  *flag = (PetscTruth) m->hasIntSection(std::string(name));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetSectionInt"
/*@C
  MeshGetSectionInt - Returns a SectionInt of the given name from the Mesh.

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameter:
. section - The SectionInt

  Note: The section is a new object, and must be destroyed by the user

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetSectionInt(Mesh mesh, const char name[], SectionInt *section)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionIntCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  ierr = SectionIntSetSection(*section, m->getIntSection(std::string(name)));CHKERRQ(ierr);
  ierr = SectionIntSetBundle(*section, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshSetSectionInt"
/*@C
  MeshSetSectionInt - Puts a SectionInt of the given name into the Mesh.

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- section - The SectionInt

  Note: This takes the section name from the PETSc object

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode MeshSetSectionInt(Mesh mesh, SectionInt section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::int_section_type> s;
  const char         *name;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  m->setIntSection(std::string(name), s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionGetArray"
/*@C
  SectionGetArray - Returns the array underlying the Section.

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- name - The section name

  Output Parameters:
+ numElements - The number of mesh element with values
. fiberDim - The number of values per element
- array - The array

  Level: intermediate

.keywords: mesh, elements
.seealso: MeshCreate()
@*/
PetscErrorCode SectionGetArray(Mesh mesh, const char name[], PetscInt *numElements, PetscInt *fiberDim, PetscScalar *array[])
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& section = m->getRealSection(std::string(name));
  if (section->size() == 0) {
    *numElements = 0;
    *fiberDim    = 0;
    *array       = NULL;
    PetscFunctionReturn(0);
  }
  const ALE::Mesh::real_section_type::chart_type& chart = section->getChart();
/*   const int                                  depth   = m->depth(*chart.begin()); */
/*   *numElements = m->depthStratum(depth)->size(); */
/*   *fiberDim    = section->getFiberDimension(*chart.begin()); */
/*   *array       = (PetscScalar *) m->restrict(section); */
  int fiberDimMin = section->getFiberDimension(*chart.begin());
  int numElem     = 0;

  for(ALE::Mesh::real_section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(*c_iter);

    if (fiberDim < fiberDimMin) fiberDimMin = fiberDim;
  }
  for(ALE::Mesh::real_section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(*c_iter);

    numElem += fiberDim/fiberDimMin;
  }
  *numElements = numElem;
  *fiberDim    = fiberDimMin;
  *array       = (PetscScalar *) section->restrict();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVertices"
PetscErrorCode WritePyLithVertices(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElements"
PetscErrorCode WritePyLithElements(Mesh mesh, SectionInt material, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::int_section_type> s;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionIntGetSection(material, s);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeElements(m, s, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVerticesLocal"
PetscErrorCode WritePyLithVerticesLocal(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeVerticesLocal(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElementsLocal"
PetscErrorCode WritePyLithElementsLocal(Mesh mesh, SectionInt material, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::int_section_type> s;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionIntGetSection(material, s);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeElementsLocal(m, s, viewer);
}

#if 0
#undef __FUNCT__  
#define __FUNCT__ "WritePyLithTractionsLocal"
PetscErrorCode WritePyLithTractionsLocal(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeTractionsLocal(m, m->getRealSection("tractions"), viewer);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "MeshCompatGetMesh"
/*@C
    MeshCompatGetMesh - Gets the internal mesh object

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    m - the internal mesh object

    Notes: This is part of the PyLith 0.8 compatibility layer. DO NOT USE unless you are
    developing for that tool.
 
    Level: developer

.seealso MeshCreate(), MeshSetMesh()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCompatGetMesh(Mesh mesh, ALE::Obj<ALECompat::Mesh>& m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  m = mesh->mcompat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCompatSetMesh"
/*@C
    MeshCompatSetMesh - Sets the internal mesh object

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    m - the internal mesh object

    Notes: This is part of the PyLith 0.8 compatibility layer. DO NOT USE unless you are
    developing for that tool.
 
    Level: developer

.seealso MeshCreate(), MeshGetMesh()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCompatSetMesh(Mesh mesh, const ALE::Obj<ALECompat::Mesh>& m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  mesh->mcompat = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExpandInterval"
inline void ExpandInterval(const ALE::Point& interval, int indices[], int& indx)
{
  const int end = interval.prefix + interval.index;
  for(int i = interval.index; i < end; i++) {
    indices[indx++] = i;
  }
}

#undef __FUNCT__
#define __FUNCT__ "ExpandInterval_New"
inline void ExpandInterval_New(ALE::Point interval, PetscInt indices[], PetscInt *indx)
{
  for(int i = 0; i < interval.prefix; i++) {
    indices[(*indx)++] = interval.index + i;
  }
  for(int i = 0; i < -interval.prefix; i++) {
    indices[(*indx)++] = -1;
  }
}

#undef __FUNCT__
#define __FUNCT__ "ExpandIntervals"
PetscErrorCode ExpandIntervals(ALE::Obj<ALECompat::Mesh::real_section_type::IndexArray> intervals, PetscInt *indices)
{
  int k = 0;

  PetscFunctionBegin;
  for(ALECompat::Mesh::real_section_type::IndexArray::iterator i_itor = intervals->begin(); i_itor != intervals->end(); i_itor++) {
    ExpandInterval_New(*i_itor, indices, &k);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCompatCreateGlobalScatter"
template<typename Section>
PetscErrorCode PETSCDM_DLLEXPORT MeshCompatCreateGlobalScatter(const ALE::Obj<ALECompat::Mesh>& m, const ALE::Obj<Section>& s, VecScatter *scatter)
{
  typedef ALECompat::Mesh::real_section_type::index_type index_type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_GetGlobalScatter,0,0,0,0);CHKERRQ(ierr);
  const ALE::Obj<ALECompat::Mesh::topology_type>&                   topology = m->getTopology();
  const ALE::Obj<ALECompat::Mesh::real_section_type::atlas_type>&   atlas    = s->getAtlas();
  const ALECompat::Mesh::real_section_type::patch_type              patch    = 0;
  const ALECompat::Mesh::real_section_type::atlas_type::chart_type& chart    = atlas->getPatch(patch);
  const ALE::Obj<ALECompat::Mesh::order_type>& globalOrder = m->getFactory()->getGlobalOrder(topology, patch, s->getName(), atlas);
  int *localIndices, *globalIndices;
  int  localSize = s->size(patch);
  int  localIndx = 0, globalIndx = 0;
  Vec  globalVec, localVec;
  IS   localIS, globalIS;

  ierr = VecCreate(m->comm(), &globalVec);CHKERRQ(ierr);
  ierr = VecSetSizes(globalVec, globalOrder->getLocalSize(), PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(globalVec);CHKERRQ(ierr);
  // Loop over all local points
  ierr = PetscMalloc(localSize*sizeof(int), &localIndices); CHKERRQ(ierr);
  ierr = PetscMalloc(localSize*sizeof(int), &globalIndices); CHKERRQ(ierr);
  for(ALECompat::Mesh::real_section_type::atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
    const ALECompat::Mesh::real_section_type::index_type& idx = atlas->restrictPoint(patch, *p_iter)[0];

    // Map local indices to global indices
    ExpandInterval(idx, localIndices, localIndx);
    ExpandInterval(index_type(idx.prefix, globalOrder->getIndex(*p_iter)), globalIndices, globalIndx);
  }
  if (localIndx  != localSize) SETERRQ2(PETSC_ERR_ARG_SIZ, "Invalid number of local indices %d, should be %d", localIndx, localSize);
  if (globalIndx != localSize) SETERRQ2(PETSC_ERR_ARG_SIZ, "Invalid number of global indices %d, should be %d", globalIndx, localSize);
  if (m->debug()) {
    globalOrder->view("Global Order");
    for(int i = 0; i < localSize; ++i) {
      printf("[%d] localIndex[%d]: %d globalIndex[%d]: %d\n", m->commRank(), i, localIndices[i], i, globalIndices[i]);
    }
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, localSize, localIndices,  &localIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, localSize, globalIndices, &globalIS);CHKERRQ(ierr);
  ierr = PetscFree(localIndices);CHKERRQ(ierr);
  ierr = PetscFree(globalIndices);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, localSize, s->restrict(patch), &localVec);CHKERRQ(ierr);
  ierr = VecScatterCreate(localVec, localIS, globalVec, globalIS, scatter);CHKERRQ(ierr);
  ierr = ISDestroy(globalIS);CHKERRQ(ierr);
  ierr = ISDestroy(localIS);CHKERRQ(ierr);
  ierr = VecDestroy(localVec);CHKERRQ(ierr);
  ierr = VecDestroy(globalVec);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Mesh_GetGlobalScatter,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshCompatGetGlobalScatter"
PetscErrorCode PETSCDM_DLLEXPORT MeshCompatGetGlobalScatter(Mesh mesh, VecScatter *scatter)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  PetscValidPointer(scatter, 2);
  if (!mesh->globalScatter) {
    ALE::Obj<ALECompat::Mesh> m;

    ierr = MeshCompatGetMesh(mesh, m);CHKERRQ(ierr);
    ierr = MeshCompatCreateGlobalScatter(m, m->getRealSection("default"), &mesh->globalScatter);CHKERRQ(ierr);
  }
  *scatter = mesh->globalScatter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "preallocateOperatorCompat"
template<typename Atlas>
PetscErrorCode preallocateOperatorCompat(const ALE::Obj<ALECompat::Mesh::topology_type>& topology, const ALE::Obj<Atlas>& atlas, const ALE::Obj<ALECompat::Mesh::order_type>& globalOrder, Mat A)
{
  typedef ALECompat::New::NumberingFactory<ALECompat::Mesh::topology_type> NumberingFactory;
  const ALE::Obj<ALECompat::Mesh::sieve_type>     adjGraph    = new ALECompat::Mesh::sieve_type(topology->comm(), topology->debug());
  const ALE::Obj<ALECompat::Mesh::topology_type>  adjTopology = new ALECompat::Mesh::topology_type(topology->comm(), topology->debug());
  const ALECompat::Mesh::real_section_type::patch_type patch  = 0;
  const ALE::Obj<ALECompat::Mesh::sieve_type>&    sieve       = topology->getPatch(patch);
  PetscInt       numLocalRows, firstRow;
  PetscInt      *dnz, *onz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  adjTopology->setPatch(patch, adjGraph);
  numLocalRows = globalOrder->getLocalSize();
  firstRow     = globalOrder->getGlobalOffsets()[topology->commRank()];
  ierr = PetscMalloc2(numLocalRows, PetscInt, &dnz, numLocalRows, PetscInt, &onz);CHKERRQ(ierr);
  /* Create local adjacency graph */
  /*   In general, we need to get FIAT info that attaches dual basis vectors to sieve points */
  const ALECompat::Mesh::real_section_type::atlas_type::chart_type& chart = atlas->getPatch(patch);

  for(ALECompat::Mesh::real_section_type::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const ALECompat::Mesh::real_section_type::atlas_type::point_type& point = *c_iter;

    adjGraph->addCone(sieve->cone(sieve->support(point)), point);
  }
  /* Distribute adjacency graph */
  topology->constructOverlap(patch);
  const Obj<ALECompat::Mesh::send_overlap_type>& vertexSendOverlap = topology->getSendOverlap();
  const Obj<ALECompat::Mesh::recv_overlap_type>& vertexRecvOverlap = topology->getRecvOverlap();
  const Obj<ALECompat::Mesh::send_overlap_type>  nbrSendOverlap    = new ALECompat::Mesh::send_overlap_type(topology->comm(), topology->debug());
  const Obj<ALECompat::Mesh::recv_overlap_type>  nbrRecvOverlap    = new ALECompat::Mesh::recv_overlap_type(topology->comm(), topology->debug());
  const Obj<ALECompat::Mesh::send_section_type>  sendSection       = new ALECompat::Mesh::send_section_type(topology->comm(), topology->debug());
  const Obj<ALECompat::Mesh::recv_section_type>  recvSection       = new ALECompat::Mesh::recv_section_type(topology->comm(), sendSection->getTag(), topology->debug());

  ALECompat::New::Distribution<ALECompat::Mesh::topology_type>::coneCompletion(vertexSendOverlap, vertexRecvOverlap, adjTopology, sendSection, recvSection);
  /* Distribute indices for new points */
  ALECompat::New::Distribution<ALECompat::Mesh::topology_type>::updateOverlap(sendSection, recvSection, nbrSendOverlap, nbrRecvOverlap);
  NumberingFactory::singleton(topology->debug())->completeOrder(globalOrder, nbrSendOverlap, nbrRecvOverlap, patch, true);
  /* Read out adjacency graph */
  const ALE::Obj<ALECompat::Mesh::sieve_type> graph = adjTopology->getPatch(patch);

  ierr = PetscMemzero(dnz, numLocalRows * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(onz, numLocalRows * sizeof(PetscInt));CHKERRQ(ierr);
  for(ALECompat::Mesh::real_section_type::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const ALECompat::Mesh::real_section_type::atlas_type::point_type& point = *c_iter;

    if (globalOrder->isLocal(point)) {
      const ALE::Obj<ALECompat::Mesh::sieve_type::traits::coneSequence>& adj   = graph->cone(point);
      const ALECompat::Mesh::order_type::value_type&          rIdx  = globalOrder->restrictPoint(patch, point)[0];
      const int                                               row   = rIdx.prefix;
      const int                                               rSize = rIdx.index;

      for(ALECompat::Mesh::sieve_type::traits::coneSequence::iterator v_iter = adj->begin(); v_iter != adj->end(); ++v_iter) {
        const ALECompat::Mesh::real_section_type::atlas_type::point_type& neighbor = *v_iter;
        const ALECompat::Mesh::order_type::value_type& cIdx     = globalOrder->restrictPoint(patch, neighbor)[0];
        const int&                                     cSize    = cIdx.index;

        if (cSize > 0) {
          if (globalOrder->isLocal(neighbor)) {
            for(int r = 0; r < rSize; ++r) {dnz[row - firstRow + r] += cSize;}
          } else {
            for(int r = 0; r < rSize; ++r) {onz[row - firstRow + r] += cSize;}
          }
        }
      }
    }
  }
  if (topology->debug()) {
    int rank = topology->commRank();
    for(int r = 0; r < numLocalRows; r++) {
      std::cout << "["<<rank<<"]: dnz["<<r<<"]: " << dnz[r] << " onz["<<r<<"]: " << onz[r] << std::endl;
    }
  }
  ierr = MatSeqAIJSetPreallocation(A, 0, dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, dnz, 0, onz);CHKERRQ(ierr);
  ierr = PetscFree2(dnz, onz);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "preallocateMatrixCompat"
PetscErrorCode preallocateMatrixCompat(const ALE::Obj<ALECompat::Mesh::topology_type>& topology, const ALE::Obj<ALECompat::Mesh::real_section_type::atlas_type>& atlas, const ALE::Obj<ALECompat::Mesh::order_type>& globalOrder, Mat A)
{
  return preallocateOperatorCompat(topology, atlas, globalOrder, A);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperatorCompat"
PetscErrorCode updateOperatorCompat(Mat A, const ALE::Obj<ALECompat::Mesh::real_section_type>& section, const ALE::Obj<ALECompat::Mesh::order_type>& globalOrder, const ALECompat::Mesh::point_type& e, PetscScalar array[], InsertMode mode)
{
  ALECompat::Mesh::real_section_type::patch_type patch = 0;
  static PetscInt  indicesSize = 0;
  static PetscInt *indices = NULL;
  PetscInt         numIndices = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  const ALE::Obj<ALECompat::Mesh::real_section_type::IndexArray> intervals = section->getIndices(patch, e, globalOrder);

  ierr = PetscLogEventBegin(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  if (section->debug()) {printf("[%d]mat for element %d\n", section->commRank(), e);}
  for(ALECompat::Mesh::real_section_type::IndexArray::iterator i_iter = intervals->begin(); i_iter != intervals->end(); ++i_iter) {
    numIndices += std::abs(i_iter->prefix);
    if (section->debug()) {
      printf("[%d]mat interval (%d, %d)\n", section->commRank(), i_iter->prefix, i_iter->index);
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
  if (section->debug()) {
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]mat indices[%d] = %d\n", section->commRank(), i, indices[i]);
    }
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]", section->commRank());
      for(int j = 0; j < numIndices; j++) {
        printf(" %g", array[i*numIndices+j]);
      }
      printf("\n");
    }
  }
  ierr = MatSetValues(A, numIndices, indices, numIndices, indices, array, mode);
  if (ierr) {
    printf("[%d]ERROR in updateOperator: point %d\n", section->commRank(), e);
    for(int i = 0; i < numIndices; i++) {
      printf("[%d]mat indices[%d] = %d\n", section->commRank(), i, indices[i]);
    }
    CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(Mesh_updateOperator,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCompatCreatePyLith"
/*@C
  MeshCompatCreatePyLith - Create a Mesh from PyLith files.

  Not Collective

  Input Parameters:
+ dim - The topological mesh dimension
. baseFilename - The basename for mesh files
. zeroBase - Use 0 to start numbering
- interpolate - The flag for mesh interpolation

  Output Parameter:
. mesh - The Mesh object

  Notes: This is part of the PyLith 0.8 compatibility layer. DO NOT USE unless you are
  developing for that tool.

  Level: developer

.keywords: mesh, PCICE
.seealso: MeshCreate()
@*/
PetscErrorCode MeshCompatCreatePyLith(MPI_Comm comm, const int dim, const char baseFilename[], PetscTruth zeroBase, PetscTruth interpolate, Mesh *mesh)
{
  ALE::Obj<ALECompat::Mesh> m;
  PetscInt            debug = 0;
  PetscTruth          flag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  try {
    m  = ALECompat::PyLith::Builder::readMesh(comm, dim, std::string(baseFilename), zeroBase, interpolate, debug);
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_ERR_FILE_OPEN, e.message());
  }
  ierr = MeshCompatSetMesh(*mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
