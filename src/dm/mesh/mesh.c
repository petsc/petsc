#include "private/meshimpl.h"   /*I      "petscmesh.h"   I*/
#include <petscmesh_viewers.hh>
#include <petscmesh_formats.hh>

/* Logging support */
PetscCookie PETSCDM_DLLEXPORT MESH_COOKIE;
PetscLogEvent  Mesh_View, Mesh_GetGlobalScatter, Mesh_restrictVector, Mesh_assembleVector,
            Mesh_assembleVectorComplete, Mesh_assembleMatrix, Mesh_updateOperator;

PetscTruth MeshRegisterAllCalled = PETSC_FALSE;
PetscFList MeshList;

#if PETSC_HAVE_SIEVE
ALE::MemoryLogger Petsc_MemoryLogger;
#endif

EXTERN PetscErrorCode MeshView_Mesh(Mesh, PetscViewer);
EXTERN PetscErrorCode MeshRefine_Mesh(Mesh, MPI_Comm, Mesh *);
EXTERN PetscErrorCode MeshCoarsenHierarchy_Mesh(Mesh, int, Mesh *);
EXTERN PetscErrorCode MeshGetInterpolation_Mesh(Mesh, Mesh, Mat *, Vec *);
EXTERN PetscErrorCode MeshGetInterpolation_Mesh_New(Mesh, Mesh, Mat *, Vec *);

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
  PETSC_MESH_TYPE::MeshNumberingFactory::singleton(0, 0, true);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve_Ascii"
PetscErrorCode MeshView_Sieve_Ascii(const ALE::Obj<PETSC_MESH_TYPE>& mesh, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK) {
    ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeElements(mesh, viewer);CHKERRQ(ierr);
    const ALE::Obj<PETSC_MESH_TYPE::int_section_type>& p     = mesh->getIntSection("Partition");
    const ALE::Obj<PETSC_MESH_TYPE::label_sequence>&   cells = mesh->heightStratum(0);
    const PETSC_MESH_TYPE::label_sequence::iterator    end   = cells->end();
    const int                                          rank  = mesh->commRank();

#ifdef PETSC_OPT_SIEVE
    p->setChart(PETSC_MESH_TYPE::int_section_type::chart_type(*cells));
#endif
    p->setFiberDimension(cells, 1);
    p->allocatePoint();
    for(PETSC_MESH_TYPE::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
      p->updatePoint(*c_iter, &rank);
    }
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
    ierr = SectionView_Sieve_Ascii(mesh, p, "Partition", viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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
  } else if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    mesh->view("");
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
#define __FUNCT__ "MeshView_Sieve_Binary"
PetscErrorCode MeshView_Sieve_Binary(const ALE::Obj<PETSC_MESH_TYPE>& mesh, PetscViewer viewer)
{
  char           *filename;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);
  ALE::MeshSerializer::writeMesh(filename, *mesh);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Sieve"
PetscErrorCode MeshView_Sieve(const ALE::Obj<PETSC_MESH_TYPE>& mesh, PetscViewer viewer)
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
    ierr = MeshView_Sieve_Binary(mesh, viewer);CHKERRQ(ierr);
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
  ierr = MeshView_Sieve(mesh->m, viewer);CHKERRQ(ierr);
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
+    PETSC_VIEWER_DEFAULT - default, prints mesh information
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
PetscErrorCode PETSCDM_DLLEXPORT MeshLoad(PetscViewer viewer, Mesh mesh)
{
  char           *filename;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!mesh->m) {
    MPI_Comm comm;

    ierr = PetscObjectGetComm((PetscObject) viewer, &comm);CHKERRQ(ierr);
    ALE::Obj<PETSC_MESH_TYPE> m = new PETSC_MESH_TYPE(comm, 1);

    ierr = MeshSetMesh(mesh, m);CHKERRQ(ierr);
  }
  ierr = PetscViewerFileGetName(viewer, &filename);CHKERRQ(ierr);
  ALE::MeshSerializer::loadMesh(filename, *mesh->m);
  PetscFunctionReturn(0);
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
PetscErrorCode PETSCDM_DLLEXPORT MeshGetMesh(Mesh mesh, ALE::Obj<PETSC_MESH_TYPE>& m)
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
PetscErrorCode PETSCDM_DLLEXPORT MeshSetMesh(Mesh mesh, const ALE::Obj<PETSC_MESH_TYPE>& m)
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
/*@C
  MeshCreateMatrix - Creates a matrix with the correct parallel layout required for 
    computing the Jacobian on a function defined using the information in the Section.

  Collective on Mesh

  Input Parameters:
+ mesh    - the mesh object
. section - the section which determines data layout
- mtype   - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ,
            or any type which inherits from one of these (such as MATAIJ, MATLUSOL, etc.).

  Output Parameter:
. J  - matrix with the correct nonzero preallocation
       (obviously without the correct Jacobian values)

  Level: advanced

  Notes: This properly preallocates the number of nonzeros in the sparse matrix so you 
       do not need to do it yourself.

.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate(), DASetBlockFills()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateMatrix(Mesh mesh, SectionReal section, MatType mtype, Mat *J)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  try {
    ierr = MeshCreateMatrix(m, s, mtype, J);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    SETERRQ(PETSC_ERR_LIB, e.message());
  }
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
  ierr = MeshGetVertexSectionReal(mesh, "default", 1, &section);CHKERRQ(ierr);
  ierr = MeshCreateMatrix(mesh, section, mtype, J);CHKERRQ(ierr);
  ierr = SectionRealDestroy(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetMatrix" 
/*@C
    MeshGetMatrix - Creates a matrix with the correct parallel layout required for 
      computing the Jacobian on a function defined using the information in Mesh.

    Collective on Mesh

    Input Parameters:
+   mesh - the mesh object
-   mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, MATSEQSBAIJ, MATMPISBAIJ,
            or any type which inherits from one of these (such as MATAIJ, MATLUSOL, etc.).

    Output Parameter:
.   J  - matrix with the correct nonzero preallocation
        (obviously without the correct Jacobian values)

    Level: advanced

    Notes: This properly preallocates the number of nonzeros in the sparse matrix so you 
       do not need to do it yourself.

.seealso ISColoringView(), ISColoringGetIS(), MatFDColoringCreate(), DASetBlockFills()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshGetMatrix(Mesh mesh, const MatType mtype, Mat *J)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
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

  new(&p->m) ALE::Obj<PETSC_MESH_TYPE>(PETSC_NULL);
  p->globalScatter = PETSC_NULL;
  p->lf            = PETSC_NULL;
  p->lj            = PETSC_NULL;
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
  PETSc::Log::Event("MeshDestroy").begin();
  if (mesh->globalScatter) {ierr = VecScatterDestroy(mesh->globalScatter);CHKERRQ(ierr);}
  mesh->m = PETSC_NULL;
  ierr = PetscHeaderDestroy(mesh);CHKERRQ(ierr);
  PETSc::Log::Event("MeshDestroy").end();
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
PetscErrorCode PETSCDM_DLLEXPORT MeshSetType(Mesh mesh, const MeshType type)
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
PetscErrorCode PETSCDM_DLLEXPORT MeshGetType(Mesh mesh,const MeshType *type)
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
/*@C
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
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshHasSectionReal(mesh, "default", &flag);CHKERRQ(ierr);
  if (!flag) {SETERRQ1(PETSC_ERR_ARG_WRONGSTATE, "Must set default section for mesh %p", m.objPtr);}
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>& order = m->getFactory()->getGlobalOrder(m, "default", m->getRealSection("default"));

  ierr = VecCreate(m->comm(), gvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*gvec, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(*gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCreateVector"
/*@
  MeshCreateVector - Creates a global vector matching the input section

  Collective on Mesh

  Input Parameters:
+ mesh - the Mesh
- section - the Section

  Output Parameter:
. vec - the global vector

  Level: advanced

  Notes: The vector can safely be destroyed using VecDestroy().
.seealso MeshDestroy(), MeshCreate(), MeshGetGlobalIndices()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateVector(Mesh mesh, SectionReal section, Vec *vec)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>& order = m->getFactory()->getGlobalOrder(m, s->getName(), s);

  ierr = VecCreate(m->comm(), vec);CHKERRQ(ierr);
  ierr = VecSetSizes(*vec, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(*vec);CHKERRQ(ierr);
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
  ALE::Obj<PETSC_MESH_TYPE> m;
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
/*@
  MeshCreateGlobalScatter - Create a VecScatter which maps from local, overlapping
  storage in the Section to a global Vec

  Collective on Mesh

  Input Parameters:
+ mesh - the mesh object
- section - The Scetion which determines data layout

  Output Parameter:
. scatter - the VecScatter
 
  Level: advanced

.seealso MeshDestroy(), MeshCreateGlobalVector(), MeshCreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalScatter(Mesh mesh, SectionReal section, VecScatter *scatter)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
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
  Obj<PETSC_MESH_TYPE> m;
  Obj<PETSC_MESH_TYPE::real_section_type> s;
  double        *v0, *J, *invJ, detJ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");
  int embedDim = coordinates->getFiberDimension(*m->depthStratum(0)->begin());
  int dim      = s->getFiberDimension(*m->depthStratum(0)->begin());

  ierr = PetscMalloc3(embedDim,double,&v0,embedDim*embedDim,double,&J,embedDim*embedDim,double,&invJ);CHKERRQ(ierr);
  ierr = PetscMalloc(numPoints*dim * sizeof(double), &values);CHKERRQ(ierr);
  for(int p = 0; p < numPoints; p++) {
    double *point = &points[p*embedDim];
    
    PETSC_MESH_TYPE::point_type e = m->locatePoint(point);
    const PETSC_MESH_TYPE::real_section_type::value_type *coeff = s->restrictPoint(e);

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
/*@C
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
  Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& vertices = m->depthStratum(0);
  const ALE::Obj<PETSC_MESH_TYPE::sieve_type>&     sieve    = m->getSieve();
  PetscInt                                          maxDeg   = -1;

  for(PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    maxDeg = PetscMax(maxDeg, (PetscInt) sieve->getSupportSize(*v_iter));
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
  Mesh           mesh;
  SectionReal    section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) b, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "x", &section);CHKERRQ(ierr);
  ierr = assembleVector(b, mesh, section, e, v, mode);CHKERRQ(ierr);
  ierr = SectionRealDestroy(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode assembleVector(Vec b, Mesh mesh, SectionReal section, PetscInt e, PetscScalar v[], InsertMode mode)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscInt                  firstElement;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_assembleVector,0,0,0,0);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  //firstElement = elementBundle->getLocalSizes()[bundle->getCommRank()];
  firstElement = 0;
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_ERR_SUP, "SectionReal does not support complex update");
#else
  if (mode == INSERT_VALUES) {
    m->update(s, PETSC_MESH_TYPE::point_type(e + firstElement), v);
  } else {
    m->updateAdd(s, PETSC_MESH_TYPE::point_type(e + firstElement), v);
  }
#endif
  ierr = PetscLogEventEnd(Mesh_assembleVector,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperator"
PetscErrorCode updateOperator(Mat A, const ALE::Obj<PETSC_MESH_TYPE>& m, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& section, const ALE::Obj<PETSC_MESH_TYPE::order_type>& globalOrder, const PETSC_MESH_TYPE::point_type& e, PetscScalar array[], InsertMode mode)
{
  PetscFunctionBegin;
#ifdef PETSC_OPT_SIEVE
  typedef ALE::ISieveVisitor::IndicesVisitor<PETSC_MESH_TYPE::real_section_type,PETSC_MESH_TYPE::order_type,PetscInt> visitor_type;
  visitor_type iV(*section, *globalOrder, (int) pow((double) m->getSieve()->getMaxConeSize(), m->depth())*m->getMaxDof(), m->depth() > 1);

  PetscErrorCode ierr = updateOperator(A, *m->getSieve(), iV, e, array, mode);CHKERRQ(ierr);
#else
  const PETSC_MESH_TYPE::indices_type indicesBlock = m->getIndices(section, e, globalOrder);
  const PetscInt *indices    = indicesBlock.first;
  const int&      numIndices = indicesBlock.second;
  PetscErrorCode  ierr;

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
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperator"
PetscErrorCode updateOperator(Mat A, const ALE::Obj<PETSC_MESH_TYPE>& m, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& section, const ALE::Obj<PETSC_MESH_TYPE::order_type>& globalOrder, int tag, int p, PetscScalar array[], InsertMode mode)
{
#ifdef PETSC_OPT_SIEVE
  SETERRQ(PETSC_ERR_SUP, "This is not applicable for optimized sieves");
#else
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
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateOperatorGeneral"
PetscErrorCode updateOperatorGeneral(Mat A, const ALE::Obj<PETSC_MESH_TYPE>& rowM, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& rowSection, const ALE::Obj<PETSC_MESH_TYPE::order_type>& rowGlobalOrder, const PETSC_MESH_TYPE::point_type& rowE, const ALE::Obj<PETSC_MESH_TYPE>& colM, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& colSection, const ALE::Obj<PETSC_MESH_TYPE::order_type>& colGlobalOrder, const PETSC_MESH_TYPE::point_type& colE, PetscScalar array[], InsertMode mode)
{
#ifdef PETSC_OPT_SIEVE
  typedef ALE::ISieveVisitor::IndicesVisitor<PETSC_MESH_TYPE::real_section_type,PETSC_MESH_TYPE::order_type,PetscInt> visitor_type;
  visitor_type iVr(*rowSection, *rowGlobalOrder, (int) pow((double) rowM->getSieve()->getMaxConeSize(), rowM->depth())*rowM->getMaxDof(), rowM->depth() > 1);
  visitor_type iVc(*colSection, *colGlobalOrder, (int) pow((double) colM->getSieve()->getMaxConeSize(), colM->depth())*colM->getMaxDof(), colM->depth() > 1);

  PetscErrorCode ierr = updateOperator(A, *rowM->getSieve(), iVr, rowE, *colM->getSieve(), iVc, colE, array, mode);CHKERRQ(ierr);
#else
  PetscErrorCode ierr;

  PetscFunctionBegin;
  const PETSC_MESH_TYPE::indices_type rowIndicesBlock = rowM->getIndices(rowSection, rowE, rowGlobalOrder);

  const PetscInt *tmpIndices    = rowIndicesBlock.first;
  const int       numRowIndices = rowIndicesBlock.second;
  PetscInt       *rowIndices    = new PetscInt[numRowIndices];
  PetscMemcpy(rowIndices, tmpIndices, numRowIndices*sizeof(PetscInt));

  const PETSC_MESH_TYPE::indices_type colIndicesBlock = colM->getIndices(colSection, colE, colGlobalOrder);

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
  delete [] rowIndices;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshSetMaxDof"
/*@
  MeshSetMaxDof - Sets the maximum number of degrees of freedom on any sieve point

  Collective on A

  Input Parameters:
+ A - the matrix
. mesh - Mesh needed for orderings
. section - A Section which describes the layout
. e - The element number
. v - The values
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes: This is used by routines like updateOperator() to bound buffer sizes

   Level: developer

.seealso: updateOperator(), assembleMatrix()
@*/
PetscErrorCode MeshSetMaxDof(Mesh mesh, PetscInt maxDof)
{
  Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  m->setMaxDof(maxDof);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "assembleMatrix"
/*@
  assembleMatrix - Insert values into a matrix

  Collective on A

  Input Parameters:
+ A - the matrix
. mesh - Mesh needed for orderings
. section - A Section which describes the layout
. e - The element
. v - The values
- mode - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Level: beginner

.seealso: MatSetOption()
@*/
PetscErrorCode assembleMatrix(Mat A, Mesh mesh, SectionReal section, PetscInt e, PetscScalar v[], InsertMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(Mesh_assembleMatrix,0,0,0,0);CHKERRQ(ierr);
  try {
    Obj<PETSC_MESH_TYPE> m;
    Obj<PETSC_MESH_TYPE::real_section_type> s;

    ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
    ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
    const ALE::Obj<PETSC_MESH_TYPE::order_type>& globalOrder = m->getFactory()->getGlobalOrder(m, s->getName(), s);

    if (m->debug()) {
      std::cout << "Assembling matrix for element number " << e << " --> point " << e << std::endl;
    }
    ierr = updateOperator(A, m, s, globalOrder, e, v, mode);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e.msg() << std::endl;
  }
  ierr = PetscLogEventEnd(Mesh_assembleMatrix,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "preallocateMatrix"
PetscErrorCode preallocateMatrix(const ALE::Obj<PETSC_MESH_TYPE>& mesh, const int bs, const ALE::Obj<PETSC_MESH_TYPE::real_section_type::atlas_type>& atlas, const ALE::Obj<PETSC_MESH_TYPE::order_type>& globalOrder, Mat A)
{
#ifdef PETSC_OPT_SIEVE
  SETERRQ(PETSC_ERR_SUP, "This is not applicable for optimized sieves");
#else
  PetscInt       localSize = globalOrder->getLocalSize();
  PetscInt      *dnz, *onz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(localSize, PetscInt, &dnz, localSize, PetscInt, &onz);CHKERRQ(ierr);
  ierr = preallocateOperator(mesh, bs, atlas, globalOrder, dnz, onz, A);CHKERRQ(ierr);
  ierr = PetscFree2(dnz, onz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
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
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return VTKViewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WriteVTKElements"
PetscErrorCode WriteVTKElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return VTKViewer::writeElements(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEVertices"
PetscErrorCode WritePCICEVertices(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICEElements"
PetscErrorCode WritePCICEElements(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeElements(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePCICERestart"
PetscErrorCode WritePCICERestart(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PCICE::Viewer::writeRestart(m, viewer);
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
  ALE::Obj<PETSC_MESH_TYPE> m;
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
  ALE::Obj<PETSC_MESH_TYPE> m;
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
  ALE::Obj<PETSC_MESH_TYPE> m;
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
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ALE::PCICE::Builder::outputElementsLocal(m, numElements, numCorners, vertices, columnMajor);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetCone"
/*@C
  MeshGetCone - Creates an array holding the cone of a given point

  Not Collective

  Input Parameters:
+ mesh - The Mesh object
- p - The mesh point

  Output Parameters:
+ numPoints - The number of points in the cone
- points - The array holding the cone points

  Level: intermediate

.keywords: mesh, cone
.seealso: MeshCreate()
@*/
PetscErrorCode MeshGetCone(Mesh mesh, PetscInt p, PetscInt *numPoints, PetscInt *points[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  *numPoints = m->getSieve()->getConeSize(p);
  ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> v(*numPoints);

  m->getSieve()->cone(p, v);
  *points = const_cast<PetscInt*>(v.getPoints());
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
  ALE::Obj<PETSC_MESH_TYPE> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(serialMesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(oldMesh->comm(), parallelMesh);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  const Obj<PETSC_MESH_TYPE>             newMesh  = new PETSC_MESH_TYPE(oldMesh->comm(), oldMesh->getDimension(), oldMesh->debug());
  const Obj<PETSC_MESH_TYPE::sieve_type> newSieve = new PETSC_MESH_TYPE::sieve_type(oldMesh->comm(), oldMesh->debug());

  newMesh->setSieve(newSieve);
  ierr = PetscPrintf(oldMesh->comm(), "Starting Mesh Distribution");CHKERRQ(ierr);
  ALE::DistributionNew<PETSC_MESH_TYPE>::distributeMeshAndSectionsV(oldMesh, newMesh);
  ierr = MeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  ierr = PetscPrintf(oldMesh->comm(), "Ending Mesh Distribution");CHKERRQ(ierr);
#else
  if (partitioner == NULL) {
    ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Distribution<PETSC_MESH_TYPE>::distributeMesh(oldMesh);
    ierr = MeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  } else {
    ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Distribution<PETSC_MESH_TYPE>::distributeMesh(oldMesh, 0, partitioner);
    ierr = MeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  }
#endif
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
  ALE::Obj<PETSC_MESH_TYPE> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(serialMesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(oldMesh->comm(), parallelMesh);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  SETERRQ(PETSC_ERR_SUP, "I am being lazy, bug me.");
#else
  if (partitioner == NULL) {
    ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Distribution<PETSC_MESH_TYPE>::distributeMesh(oldMesh, 1);
    ierr = MeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  } else {
    ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Distribution<PETSC_MESH_TYPE>::distributeMesh(oldMesh, 1, partitioner);
    ierr = MeshSetMesh(*parallelMesh, newMesh);CHKERRQ(ierr);
  }
#endif
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
  ALE::Obj<PETSC_MESH_TYPE> mB;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(boundary, mB);CHKERRQ(ierr);
  ierr = MeshCreate(mB->comm(), mesh);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  ALE::Obj<PETSC_MESH_TYPE> m = ALE::Generator<PETSC_MESH_TYPE>::generateMeshV(mB, interpolate);
#else
  ALE::Obj<PETSC_MESH_TYPE> m = ALE::Generator<PETSC_MESH_TYPE>::generateMesh(mB, interpolate);
#endif
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
  ALE::Obj<PETSC_MESH_TYPE> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(oldMesh->comm(), refinedMesh);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMeshV(oldMesh, refinementLimit, interpolate);
#else
  ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMesh(oldMesh, refinementLimit, interpolate);
#endif
  ierr = MeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshRefine_Mesh"
PetscErrorCode MeshRefine_Mesh(Mesh mesh, MPI_Comm comm, Mesh *refinedMesh)
{
  ALE::Obj<PETSC_MESH_TYPE> oldMesh;
  double              refinementLimit;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(comm, refinedMesh);CHKERRQ(ierr);
  refinementLimit = oldMesh->getMaxVolume()/2.0;
#ifdef PETSC_OPT_SIEVE
  ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMeshV(oldMesh, refinementLimit, true);
#else
  ALE::Obj<PETSC_MESH_TYPE> newMesh = ALE::Generator<PETSC_MESH_TYPE>::refineMesh(oldMesh, refinementLimit, true);
#endif
  ierr = MeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
#ifndef PETSC_OPT_SIEVE
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s = newMesh->getRealSection("default");
  const Obj<std::set<std::string> >& discs = oldMesh->getDiscretizations();

  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    newMesh->setDiscretization(*f_iter, oldMesh->getDiscretization(*f_iter));
  }
  newMesh->setupField(s);
#endif
  PetscFunctionReturn(0);
}

#ifndef PETSC_OPT_SIEVE

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
PetscErrorCode MeshCoarsenHierarchy(Mesh mesh, int numLevels, double coarseningFactor, PetscTruth interpolate, Mesh *coarseHierarchy)
{
  ALE::Obj<PETSC_MESH_TYPE> oldMesh;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (numLevels < 1) {
    *coarseHierarchy = PETSC_NULL;
    PetscFunctionReturn(0);
  }
  ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  for (int i = 0; i < numLevels; i++) {
    ierr = MeshCreate(oldMesh->comm(), &coarseHierarchy[i]);CHKERRQ(ierr);
  }
  ierr = MeshSpacingFunction(mesh);CHKERRQ(ierr);
  ierr = MeshCreateHierarchyLabel_Link(mesh, coarseningFactor, numLevels+1, coarseHierarchy);
  
#if 0
  if (oldMesh->getDimension() != 2) SETERRQ(PETSC_ERR_SUP, "Coarsening only works in two dimensions right now");
  ierr = ALE::Coarsener::IdentifyBoundary(oldMesh, 2);CHKERRQ(ierr);
  ierr = ALE::Coarsener::make_coarsest_boundary(oldMesh, 2, numLevels+1);CHKERRQ(ierr);
  ierr = ALE::Coarsener::CreateSpacingFunction(oldMesh, 2);CHKERRQ(ierr);
  ierr = ALE::Coarsener::CreateCoarsenedHierarchyNew(oldMesh, 2, numLevels, coarseningFactor);CHKERRQ(ierr);
  ierr = PetscMalloc(numLevels * sizeof(Mesh),coarseHierarchy);CHKERRQ(ierr);
  for(int l = 0; l < numLevels; l++) {
    ALE::Obj<PETSC_MESH_TYPE> newMesh = new PETSC_MESH_TYPE(oldMesh->comm(), oldMesh->debug());
    const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s = newMesh->getRealSection("default");

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

#endif

PetscErrorCode MeshCoarsenHierarchy_Mesh(Mesh mesh, int numLevels, Mesh *coarseHierarchy)
{
  PetscErrorCode ierr;
  double cfactor = 1.5;
  PetscFunctionBegin;
  ierr = PetscOptionsReal("-dmmg_coarsen_factor", "The coarsening factor", PETSC_NULL, cfactor, &cfactor, PETSC_NULL);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  SETERRQ(PETSC_ERR_SUP, "This needs to be rewritten for optimized meshes.");
#else
  ierr = MeshCoarsenHierarchy(mesh, numLevels, cfactor, PETSC_FALSE, coarseHierarchy);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#if 0

#undef __FUNCT__
#define __FUNCT__ "MeshGetInterpolation_Mesh_General"

//Interpolate between two meshes whenever the unknowns can be evaluated at points.

PetscErrorCode MeshGetInterpolation_Mesh_General(Mesh coarse_mesh, Mesh fine_mesh, Mat *interpolation, Vec *scaling) {
  ALE::Obj<PETSC_MESH_TYPE> fm, cm;
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
  //  ALE::Obj<PETSC_MESH_TYPE::label_type> coarsetraversal = cm->createLabel("traversal");
  //  ALE::Obj<PETSC_MESH_TYPE::label_type> finetraversal   = fm->createLabel ("traversal");
  const int                       debug           = fm->debug();
  if (debug) {ierr = PetscPrintf(fm->comm(), "Fine: %d vertices, Coarse: %d vertices\n", fm->depthStratum(0)->size(), cm->depthStratum(0)->size());CHKERRQ(ierr);}
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& finecoordinates   = fm->getRealSection("coordinates");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coarsecoordinates = cm->getRealSection("coordinates");

  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& sCoarse           = cm->getRealSection("default");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& sFine             = fm->getRealSection("default");

  const ALE::Obj<PETSC_MESH_TYPE::order_type>&        coarseOrder       = cm->getFactory()->getGlobalOrder(cm, "default", sCoarse);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>&        fineOrder         = fm->getFactory()->getGlobalOrder(fm, "default", sFine);

  std::list<PETSC_MESH_TYPE::point_type> travlist;        // store point
  std::list<PETSC_MESH_TYPE::point_type> travguesslist;   // store guess
  std::list<PETSC_MESH_TYPE::point_type> eguesslist;      // store the next guesses for the location of the current point.

  const ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> coarse_traversal = PETSC_MESH_TYPE::sieve_type::supportSet();
  const ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> fine_traversal = PETSC_MESH_TYPE::sieve_type::supportSet();
  const ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> covering_points = PETSC_MESH_TYPE::sieve_type::supportSet();

  const ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> uncorrected_points = PETSC_MESH_TYPE::sieve_type::supportSet();
  static double loc[4], v0[3], J[9], invJ[9], detJ; // first point, jacobian, inverse jacobian, and jacobian determinant of a cell
  if (debug) {ierr = PetscPrintf(fm->comm(), "Starting Interpolation Matrix Build\n");CHKERRQ(ierr);}

  //set up the new section holding the names of the contained points.  

  const ALE::Obj<PETSC_MESH_TYPE::int_section_type> & node_locations = fm->getIntSection("node_locations");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type> & fine_default = fm->getRealSection("default");
  int total_dimension
  for (int i = 0; i < dim; i++) {
    const ALE::Obj<PETSC_MESH_TYPE::label_sequence> & present_level = fm->depthStratum(i);
    int current_dimension = fine_default->getFiberDimension(*present_level->begin());
    node_locations->setFiberDimension(present_level, current_dimension);
  }
  node_locations->allocate();
  //traverse!

  

  ALE::Obj<PETSC_MESH_TYPE::label_sequence> fine_cells = fm->heightStratum(0);
  ALE::Obj<PETSC_MESH_TYPE::label_sequence> coarse_cells = cm->heightStratum(0);

  PETSC_MESH_TYPE::label_sequence::iterator fc_iter = fine_cells->begin();
  PETSC_MESH_TYPE::label_sequence::iterator fc_iter_end = fine_cells->end();
  while (fc_iter != fc_iter_end) {
    //locate an initial coarse cell that overlaps with this fine cell in terms of their bounding boxes;
    PETSC_MESH_TYPE::label_sequence::iterator cc_iter = coarse_cells->begin();
    PETSC_MESH_TYPE::label_sequence::iterator cc_iter_end = coarse_cells->end();
    while (cc_iter != cc_iter_end) {
      
      cc_iter++;
    }
    fc_iter++;
  }
}

#endif

#undef __FUNCT__
#define __FUNCT__ "MeshGetInterpolation_Mesh_New"

PetscErrorCode MeshGetInterpolation_Mesh_New(Mesh dmCoarse, Mesh dmFine, Mat *interpolation, Vec *scaling) {

#ifdef PETSC_OPT_SIEVE
  SETERRQ(PETSC_ERR_SUP, "This needs to be rewritten for optimized meshes.");
#else
  ALE::Obj<PETSC_MESH_TYPE> fm, cm;
  Mat                 P;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(dmFine, fm);CHKERRQ(ierr);
  ierr = MeshGetMesh(dmCoarse, cm);CHKERRQ(ierr);
  //  ALE::Obj<PETSC_MESH_TYPE::label_type> coarsetraversal = cm->createLabel("traversal");
  //  ALE::Obj<PETSC_MESH_TYPE::label_type> finetraversal   = fm->createLabel ("traversal");
  const int                       debug           = fm->debug();
  if (debug) {ierr = PetscPrintf(fm->comm(), "Fine: %d vertices, Coarse: %d vertices\n", fm->depthStratum(0)->size(), cm->depthStratum(0)->size());CHKERRQ(ierr);}
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& finecoordinates   = fm->getRealSection("coordinates");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coarsecoordinates = cm->getRealSection("coordinates");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& sCoarse           = cm->getRealSection("default");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& sFine             = fm->getRealSection("default");
  const ALE::Obj<PETSC_MESH_TYPE::order_type>&        coarseOrder       = cm->getFactory()->getGlobalOrder(cm, "default", sCoarse);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>&        fineOrder         = fm->getFactory()->getGlobalOrder(fm, "default", sFine);
  std::list<PETSC_MESH_TYPE::point_type> travlist;        // store point
  std::list<PETSC_MESH_TYPE::point_type> travguesslist;   // store guess
  std::list<PETSC_MESH_TYPE::point_type> eguesslist;      // store the next guesses for the location of the current point.
  const ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> coarse_traversal = PETSC_MESH_TYPE::sieve_type::supportSet();
  const ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> fine_traversal = PETSC_MESH_TYPE::sieve_type::supportSet();
  const ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> uncorrected_points = PETSC_MESH_TYPE::sieve_type::supportSet();
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
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& finevertices = fm->depthStratum(0);
  const PETSC_MESH_TYPE::label_sequence::iterator  fv_iter_end  = finevertices->end();
  PETSC_MESH_TYPE::label_sequence::iterator        fv_iter      = finevertices->begin();

  //  while (fv_iter != fv_iter_end) {
  //    fm->setValue(finetraversal, *fv_iter, 0);
  //    fv_iter++;
  //  }

  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& coarseelements = cm->heightStratum(0);
  const PETSC_MESH_TYPE::label_sequence::iterator  ce_iter_end    = coarseelements->end();
  PETSC_MESH_TYPE::label_sequence::iterator        ce_iter        = coarseelements->begin();
  
  //  while (ce_iter != ce_iter_end) {
  //    cm->setValue(coarsetraversal, *ce_iter, 0);
  //    ce_iter++;
  //  }

  double *fvCoords = new double[dim], *nvCoords = new double[dim];
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
          const ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSet> & neighbors  = fm->getSieve()->cone(fm->getSieve()->support(*fv_iter));
          const PETSC_MESH_TYPE::sieve_type::coneSet::iterator n_iter_end = neighbors->end();
          PETSC_MESH_TYPE::sieve_type::coneSet::iterator       n_iter     = neighbors->begin();
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
            PETSC_MESH_TYPE::point_type curVert = *travlist.begin();
            PetscMemcpy(nvCoords, finecoordinates->restrictPoint(curVert), dim*sizeof(double));
            PETSC_MESH_TYPE::point_type curEle =  *travguesslist.begin();
            travlist.pop_front();
            travguesslist.pop_front();
            eguesslist.push_front(curEle);
            //cm->setValue(coarsetraversal, curEle, 1);
            coarse_traversal->insert(curEle);
            bool locationDiscovered  = false;
            //int traversalcomparisons = 0;
            while ((!eguesslist.empty()) && (!locationDiscovered) && (int)coarse_traversal->size() < maxComparisons) {
              //traversalcomparisons = 0;
              PETSC_MESH_TYPE::point_type curguess = *eguesslist.begin();
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
                const ALE::Obj<PETSC_MESH_TYPE::sieve_type::coneSet> newNeighbors = fm->getSieve()->cone(fm->getSieve()->support(curVert));
                const PETSC_MESH_TYPE::sieve_type::coneSet::iterator nn_iter_end  = newNeighbors->end();
                PETSC_MESH_TYPE::sieve_type::coneSet::iterator       nn_iter      = newNeighbors->begin();
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
                const ALE::Obj<PETSC_MESH_TYPE::sieve_type::supportSet> & curguessneighbors = cm->getSieve()->support(cm->getSieve()->cone(curguess));
                const PETSC_MESH_TYPE::sieve_type::supportSet::iterator cgn_iter_end      = curguessneighbors->end();
                PETSC_MESH_TYPE::sieve_type::supportSet::iterator       cgn_iter          = curguessneighbors->begin();
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
            //const ALE::Obj<PETSC_MESH_TYPE::label_sequence>& traved_elements = cm->getLabelStratum("traversal", 1);
            //const PETSC_MESH_TYPE::label_sequence::iterator  tp_iter_end     = traved_elements->end();
            //PETSC_MESH_TYPE::label_sequence::iterator        tp_iter         = traved_elements->begin();
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
  delete [] fvCoords; delete [] nvCoords;
  *interpolation = P;
  if (debug) {ierr = PetscPrintf(fm->comm(), "Ending Interpolation Matrix Build\n");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
#endif
}


#undef __FUNCT__  
#define __FUNCT__ "MeshGetInterpolation_Mesh"
/*
  This method only handle P_1 discretizations at present.
*/
PetscErrorCode MeshGetInterpolation_Mesh(Mesh dmCoarse, Mesh dmFine, Mat *interpolation, Vec *scaling)
{
#ifdef PETSC_OPT_SIEVE
  SETERRQ(PETSC_ERR_SUP, "This has been superceded.");
#else
  ALE::Obj<PETSC_MESH_TYPE> coarse;
  ALE::Obj<PETSC_MESH_TYPE> fine;
  Mat                 P;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(dmFine,   fine);CHKERRQ(ierr);
  ierr = MeshGetMesh(dmCoarse, coarse);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& coarseCoordinates = coarse->getRealSection("coordinates");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& fineCoordinates   = fine->getRealSection("coordinates");
  const ALE::Obj<PETSC_MESH_TYPE::label_sequence>&    vertices          = fine->depthStratum(0);
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& sCoarse           = coarse->getRealSection("default");
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& sFine             = fine->getRealSection("default");
  const ALE::Obj<PETSC_MESH_TYPE::order_type>&        coarseOrder = coarse->getFactory()->getGlobalOrder(coarse, "default", sCoarse);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>&        fineOrder   = fine->getFactory()->getGlobalOrder(fine, "default", sFine);

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
  PETSC_MESH_TYPE::label_sequence::iterator v_iter_end = vertices->end();
  PETSC_MESH_TYPE::real_section_type::value_type *coords = new PETSC_MESH_TYPE::real_section_type::value_type[dim];

  for(PETSC_MESH_TYPE::label_sequence::iterator v_iter = vertices->begin(); v_iter != v_iter_end; ++v_iter) {
    //const PETSC_MESH_TYPE::real_section_type::value_type *coords     = fineCoordinates->restrictPoint(*v_iter);
    ierr = PetscMemcpy(coords, fineCoordinates->restrictPoint(*v_iter), dim*sizeof(double));CHKERRQ(ierr);
    PETSC_MESH_TYPE::point_type coarseCell;
    PETSC_MESH_TYPE::point_type cellguess = -1;
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
  delete [] coords;
  *interpolation = P;
  PetscFunctionReturn(0);
#endif
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
  ALE::Obj<PETSC_MESH_TYPE> m;
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
  ALE::Obj<PETSC_MESH_TYPE> m;
  bool                      has;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  has  = m->hasRealSection(std::string(name));
  ierr = SectionRealSetSection(*section, m->getRealSection(std::string(name)));CHKERRQ(ierr);
  ierr = SectionRealSetBundle(*section, m);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  if (!has) {
    m->getRealSection(std::string(name))->setChart(m->getSieve()->getChart());
  }
#endif
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
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
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
  ALE::Obj<PETSC_MESH_TYPE> m;
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
  ALE::Obj<PETSC_MESH_TYPE> m;
  bool                      has;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionIntCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  has  = m->hasIntSection(std::string(name));
  ierr = SectionIntSetSection(*section, m->getIntSection(std::string(name)));CHKERRQ(ierr);
  ierr = SectionIntSetBundle(*section, m);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  if (!has) {
    m->getIntSection(std::string(name))->setChart(m->getSieve()->getChart());
  }
#endif
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
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
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
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<PETSC_MESH_TYPE::real_section_type>& section = m->getRealSection(std::string(name));
  if (section->size() == 0) {
    *numElements = 0;
    *fiberDim    = 0;
    *array       = NULL;
    PetscFunctionReturn(0);
  }
  const PETSC_MESH_TYPE::real_section_type::chart_type& chart = section->getChart();
/*   const int                                  depth   = m->depth(*chart.begin()); */
/*   *numElements = m->depthStratum(depth)->size(); */
/*   *fiberDim    = section->getFiberDimension(*chart.begin()); */
/*   *array       = (PetscScalar *) m->restrict(section); */
  int fiberDimMin = section->getFiberDimension(*chart.begin());
  int numElem     = 0;

  for(PETSC_MESH_TYPE::real_section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(*c_iter);

    if (fiberDim < fiberDimMin) fiberDimMin = fiberDim;
  }
  for(PETSC_MESH_TYPE::real_section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const int fiberDim = section->getFiberDimension(*c_iter);

    numElem += fiberDim/fiberDimMin;
  }
  *numElements = numElem;
  *fiberDim    = fiberDimMin;
  *array       = (PetscScalar *) section->restrictSpace();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVertices"
PetscErrorCode WritePyLithVertices(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeVertices(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElements"
PetscErrorCode WritePyLithElements(Mesh mesh, SectionInt material, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionIntGetSection(material, s);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeElements(m, s, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithVerticesLocal"
PetscErrorCode WritePyLithVerticesLocal(Mesh mesh, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeVerticesLocal(m, viewer);
}

#undef __FUNCT__  
#define __FUNCT__ "WritePyLithElementsLocal"
PetscErrorCode WritePyLithElementsLocal(Mesh mesh, SectionInt material, PetscViewer viewer)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
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
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode ierr;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  return ALE::PyLith::Viewer::writeTractionsLocal(m, m->getRealSection("tractions"), viewer);
}
#endif

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
