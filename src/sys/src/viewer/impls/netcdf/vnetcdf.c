/*
     Code for the parallel NetCDF viewer.
*/
#include "src/sys/src/viewer/viewerimpl.h"    /*I   "petsc.h"   I*/
#include "petscsys.h"
#include "pnetcdf.h"

typedef struct  {
  int                   ncid;            /* NetCDF dataset id */
  char                  *filename;        /* NetCDF dataset name */
  PetscViewerNetcdfType nctype;          /* read or write? */
} PetscViewer_Netcdf;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_Netcdf" 
int PetscViewerDestroy_Netcdf(PetscViewer v)
{
  PetscViewer_Netcdf *vnetcdf = (PetscViewer_Netcdf*)v->data;
  int                ierr,rank;

  PetscFunctionBegin;
  if (vnetcdf->ncid) {
    ierr = ncmpi_close(vnetcdf->ncid); CHKERRQ(ierr);
  }
  ierr = PetscStrfree(vnetcdf->filename);CHKERRQ(ierr);
  ierr = PetscFree(vnetcdf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_Netcdf" 
int PetscViewerCreate_Netcdf(PetscViewer v)
{  
  int                ierr;
  PetscViewer_Netcdf *vnetcdf;

  PetscFunctionBegin;
  ierr               = PetscNew(PetscViewer_Netcdf,&vnetcdf);CHKERRQ(ierr);
  v->data            = (void*)vnetcdf;
  v->ops->destroy    = PetscViewerDestroy_Netcdf;
  v->ops->flush      = 0;
  v->iformat         = 0;
  vnetcdf->ncid      = -1;
  vnetcdf->nctype           = (PetscViewerNetcdfType) -1; 
  vnetcdf->filename        = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerSetFilename_C",
                                    "PetscViewerSetFilename_Netcdf",
                                     PetscViewerSetFilename_Netcdf);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerNetcdfSetType_C",
                                    "PetscViewerNetcdfSetType_Netcdf",
                                     PetscViewerNetcdfSetType_Netcdf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PetscViewerNetcdfGetID" 
int PetscViewerNetcdfGetID(PetscViewer viewer,int *ncid)
{
  PetscViewer_Netcdf *vnetcdf = (PetscViewer_Netcdf*)viewer->data;

  PetscFunctionBegin;
  *ncid = vnetcdf->ncid;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerNetcdfSetType_Netcdf" 
int PetscViewerNetcdfSetType_Netcdf(PetscViewer viewer,PetscViewerNetcdfType type)
{
  PetscViewer_Netcdf *vnetcdf = (PetscViewer_Netcdf*)viewer->data;

  PetscFunctionBegin;
  vnetcdf->nctype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerNetcdfOpen"
int PetscViewerNetcdfOpen(MPI_Comm comm,const char name[],PetscViewerNetcdfType type,PetscViewer* viewer)
{
  int ierr;
  PetscFunctionBegin;

  ierr = PetscViewerCreate(comm,viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer,PETSC_VIEWER_NETCDF);CHKERRQ(ierr);
  ierr = PetscViewerNetcdfSetType(*viewer,type);CHKERRQ(ierr);
  ierr = PetscViewerSetFilename(*viewer,name);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int PetscViewerNetcdfSetType(PetscViewer viewer,PetscViewerNetcdfType type)
{
  int ierr,(*f)(PetscViewer,PetscViewerNetcdfType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)viewer,"PetscViewerNetcdfSetType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(viewer,type);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSetFilename_Netcdf" 
int PetscViewerSetFilename_Netcdf(PetscViewer viewer,const char name[])
{
  int   rank,ierr,flg;
  PetscViewer_Netcdf    *vnetcdf = (PetscViewer_Netcdf*)viewer->data;
  PetscViewerNetcdfType type = vnetcdf->nctype;
  MPI_Comm              comm = viewer->comm;
  PetscFunctionBegin;
  if (type == (PetscViewerNetcdfType) -1) {
    SETERRQ(1,"Must call PetscViewerNetcdfSetType() before PetscViewerSetFilename()");
  }
  if (type == PETSC_NETCDF_RDONLY) {
    ierr = ncmpi_open(comm,name,0,MPI_INFO_NULL,&vnetcdf->ncid); CHKERRQ(ierr);
  }
  else if (type == PETSC_NETCDF_RDWR) {
    ierr = ncmpi_open(comm,name,NC_WRITE,MPI_INFO_NULL,&vnetcdf->ncid); CHKERRQ(ierr);
  }
  else if (type == PETSC_NETCDF_CREATE) {
    PetscTruth  flg;
    char        fname[PETSC_MAX_PATH_LEN];
    ierr = PetscOptionsGetString(PETSC_NULL,"-netcdf_viewer_name",fname,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = ncmpi_create(comm,fname,NC_CLOBBER,MPI_INFO_NULL,&vnetcdf->ncid); CHKERRQ(ierr);
    } else {
      ierr = ncmpi_create(comm,name,NC_CLOBBER,MPI_INFO_NULL,&vnetcdf->ncid); CHKERRQ(ierr);
    }

  }
  /*vnetcdf->filename = name; */
  PetscFunctionReturn(0);
}
