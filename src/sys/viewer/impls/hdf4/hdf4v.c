#define PETSC_DLL
#define HERE do { PetscErrorCode ierr, rank; ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] LINE %d (%s)\n", rank, __LINE__, __FUNCTION__);CHKERRQ(ierr); } while (0)

#define CE do { CHKERRQ(ierr); } while (0)

#include "src/sys/viewer/viewerimpl.h"    /*I   "petsc.h"   I*/
#if defined(PETSC_HAVE_HDF4)
#include <mfhdf.h>
#endif

typedef struct {
 int                 sd_id;
 char                *filename;
 PetscFileMode btype;
} PetscViewer_HDF4;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_HDF4" 
PetscErrorCode PetscViewerDestroy_HDF4(PetscViewer v)
{
 PetscErrorCode   ierr;
 PetscViewer_HDF4 *vhdf4 = (PetscViewer_HDF4 *)v->data;

 PetscFunctionBegin;
 if (vhdf4->sd_id >= 0) {
 SDend(vhdf4->sd_id);
 vhdf4->sd_id = -1;
 }
 if (vhdf4->filename) {
   ierr = PetscFree(vhdf4->filename);CE;
 }
 ierr = PetscFree(vhdf4); CE;
 PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetMode_HDF4" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetMode_HDF4(PetscViewer viewer, PetscFileMode type)
{
 PetscViewer_HDF4 *vhdf4 = (PetscViewer_HDF4 *)viewer->data;
 
 PetscFunctionBegin;
 PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
 vhdf4->btype = type;
 PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetName_HDF4" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetName_HDF4(PetscViewer viewer,const char name[])
{
 PetscErrorCode        ierr;
 int                   rank;
 PetscViewer_HDF4      *vhdf4 = (PetscViewer_HDF4*)viewer->data;
 int32                 acc;

 PetscFunctionBegin;

 switch (vhdf4->btype) {
 case FILE_MODE_READ:
   acc = DFACC_READ;
   break;
 case FILE_MODE_WRITE:
   acc = DFACC_WRITE;
 break;
 default:
   SETERRQ(PETSC_ERR_ORDER,"Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
 }

 ierr = MPI_Comm_rank(viewer->comm,&rank);CE;
 ierr = PetscStrallocpy(name,&vhdf4->filename);CE;
 if (!rank) {
   vhdf4->sd_id = SDstart(name, acc);
   if (vhdf4->sd_id < 0) {
     SETERRQ1(PETSC_ERR_LIB, "SDstart failed for %s", name);
   }
 }
 viewer->format = PETSC_VIEWER_NOFORMAT;
 PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_HDF4" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerCreate_HDF4(PetscViewer v)
{  
 PetscErrorCode   ierr;
 PetscViewer_HDF4 *vhdf4;
 
 PetscFunctionBegin;
 ierr = PetscNew(PetscViewer_HDF4,&vhdf4); CE;
 v->data            = (void*)vhdf4;
 v->ops->destroy    = PetscViewerDestroy_HDF4;
 v->ops->flush      = 0;
 v->iformat         = 0;
 vhdf4->btype       = (PetscFileMode) -1; 
 vhdf4->filename    = 0;
 vhdf4->sd_id       = -1;
 
 ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerFileSetName_C","PetscViewerFileSetName_HDF4",
                                           PetscViewerFileSetName_HDF4);CE;
 ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerFileSetMode_C","PetscViewerFileSetMode_HDF4",
                                           PetscViewerFileSetMode_HDF4);CE;
 PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PetscViewerHDF4Open" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerHDF4Open(MPI_Comm comm, const char *name, PetscFileMode type, PetscViewer *hdf4v)
{
 PetscErrorCode ierr;
 
 PetscFunctionBegin;
 ierr = PetscViewerCreate(comm,hdf4v);CE;
 ierr = PetscViewerSetType(*hdf4v,PETSC_VIEWER_HDF4);CE;
 ierr = PetscViewerFileSetMode(*hdf4v, type);CE;
 ierr = PetscViewerFileSetName(*hdf4v, name);CE;
 PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscViewerHDF4WriteSDS" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerHDF4WriteSDS(PetscViewer viewer, float *xf, int d, int *dims,int bs)
{
 int                   i;
 PetscViewer_HDF4      *vhdf4 = (PetscViewer_HDF4*)viewer->data;
 int32                 sds_id,zero32[3],dims32[3];

 PetscFunctionBegin;

 for (i = 0; i < d; i++) {
   zero32[i] = 0;
   dims32[i] = dims[i];
 }
 sds_id = SDcreate(vhdf4->sd_id, "Vec", DFNT_FLOAT32, d, dims32);
 if (sds_id < 0) {
   SETERRQ(PETSC_ERR_LIB,"SDcreate failed");
 }
 SDwritedata(sds_id, zero32, 0, dims32, xf);
 SDendaccess(sds_id);
 PetscFunctionReturn(0);
}

