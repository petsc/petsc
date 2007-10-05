#define PETSC_DLL
/* 
        Written by Matt Dorbin, mrdorbin@cs.purdue.edu 3/1/99
        For database format and API from LLNL
        Updated by Matt Knepley, knepley@cs.purdue.edu 11/16/99
*/
#include "vsilo.h"

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_Silo"
static PetscErrorCode PetscViewerDestroy_Silo(PetscViewer viewer)
{
  Viewer_Silo    *silo = (Viewer_Silo *) viewer->data;
  int            rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)viewer)->comm, &rank);CHKERRQ(ierr);
  if(!rank) {
    DBClose(silo->file_pointer);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFlush_Silo"
PetscErrorCode PetscViewerFlush_Silo(PetscViewer viewer)
{
  int            rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)viewer)->comm, &rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);
  PetscFunctionReturn(0);
}

/*-----------------------------------------Public Functions-----------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSiloGetFilePointer"
/*@C
  PetscViewerSiloGetFilePointer - Extracts the file pointer from a Silo viewer.

  Input Parameter:
. viewer - viewer context, obtained from PetscViewerSiloOpen()

  Output Parameter:
. fd     - file pointer

  Level: advanced

.keywords: PetscViewer, file, get, pointer
.seealso: PetscViewerSiloOpen()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloGetFilePointer(PetscViewer viewer, DBfile **fd)
{
  Viewer_Silo *silo = (Viewer_Silo *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  PetscValidPointer(fd,2);
  *fd = silo->file_pointer;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSiloOpen"
/*@C
  PetscViewerSiloOpen - This routine writes the mesh and the partition in the 
  SILO format used by MeshTv, which can be used to create plots and
  MPEG movies.

  Collectiveon MPI_Comm

  Input Parameters:
+ comm - The MPI communicator
- name - The name for the Silo files

  Output Parameter:
. viewer  - A viewer

  Notes:
  This viewer is intended to work with the SILO portable database format.
  Details on SILO, MeshTv, and companion software can be obtained by sending
  mail to meshtv@viper.llnl.gov

  Options Database Keys:

  Level: beginner

.keywords: PetscViewer, Silo, open
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloOpen(MPI_Comm comm, const char name[], PetscViewer *viewer)
{
  PetscViewer    v;
  Viewer_Silo    *silo;
  char           filename[PETSC_MAX_PATH_LEN], filetemp[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHeaderCreate(v, _p_PetscViewer, struct _PetscViewerOps, PETSC_VIEWER_COOKIE, -1, PETSC_VIEWER_SILO, comm, PetscViewerDestroy, 0);CHKERRQ(ierr);
  ierr            = PetscNewLog(v,Viewer_Silo,&silo); CHKPTRQ(silo);
  v->data         = (void*)silo;
  v->ops->destroy = PetscViewerDestroy_Silo;
  v->ops->flush   = PetscViewerFlush_Silo;
  ierr            = PetscStrallocpy(PETSC_VIEWER_SILO, &((PetscObject)v)->type_name);CHKERRQ(ierr);

  ierr = PetscStrncpy(filetemp, name, 251);CHKERRQ(ierr);
  ierr = PetscStrcat(filetemp, ".pdb");CHKERRQ(ierr);
  ierr = PetscFixFilename(filetemp, filename);CHKERRQ(ierr);

  silo->file_pointer = DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, DB_PDB);
  if (!silo->file_pointer) SETERRQ(PETSC_ERR_FILE_OPEN,"Cannot open Silo viewer file");
#if defined(PETSC_USE_LOG)
  PLogObjectState((PetscObject) v, "Silo File: %s", name);
#endif
  silo->meshName = PETSC_NULL;
  silo->objName  = PETSC_NULL;

  *viewer = v;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__  "PetscViewerSiloCheckMesh"
/*@C
  PetscViewerSiloCheckMesh - This routine checks a Silo viewer to determine whether the
                        mesh has already been put in the .silo file. It also checks for type,
                        and at the moment accepts only UCD_MESH meshes.

  Not collective

  Input Parameters:
+ mesh - The mesh that should be in place
. viewer - The viewer that should contain the mesh
- fp - The pointer to the DBFile that should contain the mesh 

  Level: intermediate

.keywords: viewer, Silo, mesh
.seealso: PetscViewerSiloOpen()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloCheckMesh(PetscViewer viewer, Mesh mesh)
{
  Viewer_Silo    *vsilo = (Viewer_Silo *) viewer->data;
  DBfile         *fp;
  int            mesh_type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerSiloGetFilePointer(viewer, &fp);CHKERRQ(ierr);
  if (!vsilo->meshName) {
    mesh_type = DBInqMeshtype(fp, "PetscMesh");
  } else {
    mesh_type = DBInqMeshtype(fp, vsilo->meshName);
  }
  if(mesh_type != DB_UCDMESH) { 
    /* DBInqMeshType returns -1 if the mesh is not found*/
    ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSiloGetName"
/*@C
  PetscViewerSiloGetName - Retrieve the default name for objects communicated to Silo

  Input Parameter:
. viewer - The Silo viewer

  Output Parameter:
. name   - The name for new objects created in Silo

  Level: intermediate

.keywords PetscViewer, Silo, name
.seealso PetscViewerSiloSetName(), PetscViewerSiloClearName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloGetName(PetscViewer viewer, char **name)
{
  PetscViewer_Silo *vsilo = (PetscViewer_Silo *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  PetscValidPointer(name,2);
  *name = vsilo->objName;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSiloSetName"
/*@C
  PetscViewerSiloSetName - Override the default name for objects communicated to Silo

  Input Parameters:
. viewer - The Silo viewer
. name   - The name for new objects created in Silo

  Level: intermediate

.keywords PetscViewer, Silo, name
.seealso PetscViewerSiloSetName(), PetscViewerSiloClearName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloSetName(PetscViewer viewer, char *name)
{
  Viewer_Silo *vsilo = (Viewer_Silo *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  PetscValidPointer(name,2);
  vsilo->objName = name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSiloClearName"
/*@C
  PetscViewerSiloClearName - Use the default name for objects communicated to Silo

  Input Parameter:
. viewer - The Silo viewer

  Level: intermediate

.keywords PetscViewer, Silo, name
.seealso PetscViewerSiloGetName(), PetscViewerSiloSetName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloClearName(PetscViewer viewer)
{
  Viewer_Silo *vsilo = (Viewer_Silo *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  vsilo->objName = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSiloGetMeshName"
/*@C
  PetscViewerSiloGetMeshName - Retrieve the default name for the mesh in Silo

  Input Parameter:
. viewer - The Silo viewer

  Output Parameter:
. name   - The name for new objects created in Silo

  Level: intermediate

.keywords PetscViewer, Silo, name, mesh
.seealso PetscViewerSiloSetMeshName(), PetscViewerSiloClearMeshName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloGetMeshName(PetscViewer viewer, char **name)
{
  Viewer_Silo *vsilo = (Viewer_Silo *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  PetscValidPointer(name,2);
  *name = vsilo->meshName;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSiloSetMeshName"
/*@C
  PetscViewerSiloSetMeshName - Override the default name for the mesh in Silo

  Input Parameters:
+ viewer - The Silo viewer
- name   - The name for new objects created in Silo

  Level: intermediate

.keywords PetscViewer, Silo, name, mesh
.seealso PetscViewerSiloSetMeshName(), PetscViewerSiloClearMeshName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloSetMeshName(PetscViewer viewer, char *name)
{
  Viewer_Silo *vsilo = (Viewer_Silo *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  PetscValidCharPointer(name,2);
  vsilo->meshName = name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSiloClearMeshName"
/*@C
  PetscViewerSiloClearMeshName - Use the default name for the mesh in Silo

  Input Parameter:
. viewer - The Silo viewer

  Level: intermediate

.keywords PetscViewer, Silo, name, mesh
.seealso PetscViewerSiloGetMeshName(), PetscViewerSiloSetMeshName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerSiloClearMeshName(PetscViewer viewer)
{
  Viewer_Silo *vsilo = (Viewer_Silo *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  vsilo->meshName = PETSC_NULL;
  PetscFunctionReturn(0);
}

  
