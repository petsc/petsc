#define PETSC_DLL
/*
     Code for the parallel NetCDF viewer.
*/
#include "src/sys/src/viewer/viewerimpl.h"    /*I   "petsc.h"   I*/
#include "petscsys.h"
EXTERN_C_BEGIN
#include "pnetcdf.h"
EXTERN_C_END
typedef struct  {
  int                 ncid;            /* NetCDF dataset id */
  char                *filename;        /* NetCDF dataset name */
  PetscViewerFileType nctype;          /* read or write? */
} PetscViewer_Netcdf;


#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_Netcdf" 
PetscErrorCode PetscViewerDestroy_Netcdf(PetscViewer v)
{
  PetscViewer_Netcdf *vnetcdf = (PetscViewer_Netcdf*)v->data;
  PetscErrorCode     ierr;
  int                rank;

  PetscFunctionBegin;
  if (vnetcdf->ncid) {
    ierr = ncmpi_close(vnetcdf->ncid);CHKERRQ(ierr);
  }
  ierr = PetscStrfree(vnetcdf->filename);CHKERRQ(ierr);
  ierr = PetscFree(vnetcdf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_Netcdf" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerCreate_Netcdf(PetscViewer v)
{  
  PetscErrorCode     ierr;
  PetscViewer_Netcdf *vnetcdf;

  PetscFunctionBegin;
  ierr               = PetscNew(PetscViewer_Netcdf,&vnetcdf);CHKERRQ(ierr);
  v->data            = (void*)vnetcdf;
  v->ops->destroy    = PetscViewerDestroy_Netcdf;
  v->ops->flush      = 0;
  v->iformat         = 0;
  vnetcdf->ncid      = -1;
  vnetcdf->nctype    = (PetscViewerFileType) -1; 
  vnetcdf->filename  = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerSetFilename_C",
                                    "PetscViewerSetFilename_Netcdf",
                                     PetscViewerSetFilename_Netcdf);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerSetFileType_C",
                                    "PetscViewerSetFileType_Netcdf",
                                     PetscViewerSetFileType_Netcdf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PetscViewerNetcdfGetID" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerNetcdfGetID(PetscViewer viewer,int *ncid)
{
  PetscViewer_Netcdf *vnetcdf = (PetscViewer_Netcdf*)viewer->data;

  PetscFunctionBegin;
  *ncid = vnetcdf->ncid;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSetFileType_Netcdf" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerSetFileType_Netcdf(PetscViewer viewer,PetscViewerFileType type)
{
  PetscViewer_Netcdf *vnetcdf = (PetscViewer_Netcdf*)viewer->data;

  PetscFunctionBegin;
  vnetcdf->nctype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PetscViewerNetcdfOpen"
PetscErrorCode PETSC_DLLEXPORT PetscViewerNetcdfOpen(MPI_Comm comm,const char name[],PetscViewerFileType type,PetscViewer* viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer,PETSC_VIEWER_NETCDF);CHKERRQ(ierr);
  ierr = PetscViewerSetFileType(*viewer,type);CHKERRQ(ierr);
  ierr = PetscViewerSetFilename(*viewer,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSetFilename_Netcdf" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerSetFilename_Netcdf(PetscViewer viewer,const char name[])
{
  int                 rank;
  PetscErrorCode      ierr;
  PetscViewer_Netcdf  *vnetcdf = (PetscViewer_Netcdf*)viewer->data;
  PetscViewerFileType type = vnetcdf->nctype;
  MPI_Comm            comm = viewer->comm;
  PetscTruth          flg;
  char                fname[PETSC_MAX_PATH_LEN],*gz;
  
  PetscFunctionBegin;
  ierr = PetscOptionsGetString(PETSC_NULL,"-netcdf_viewer_name",fname,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {  
    ierr = PetscStrallocpy(fname,&vnetcdf->filename);CHKERRQ(ierr);
  } else {
    ierr = PetscStrallocpy(name,&vnetcdf->filename);CHKERRQ(ierr);
  }
  if (type == (PetscViewerFileType) -1) {
    SETERRQ(PETSC_ERR_ORDER,"Must call PetscViewerSetFileType() before PetscViewerSetFilename()");
  } else if (type == PETSC_FILE_RDONLY) {
    ierr = ncmpi_open(comm,vnetcdf->filename,0,MPI_INFO_NULL,&vnetcdf->ncid);CHKERRQ(ierr);
  } else if (type == PETSC_FILE_RDWR) {
    ierr = ncmpi_open(comm,vnetcdf->filename,NC_WRITE,MPI_INFO_NULL,&vnetcdf->ncid);CHKERRQ(ierr);
  } else if (type == PETSC_FILE_CREATE) {
    ierr = ncmpi_create(comm,vnetcdf->filename,NC_CLOBBER,MPI_INFO_NULL,&vnetcdf->ncid);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_END
