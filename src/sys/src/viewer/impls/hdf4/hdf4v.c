#define HERE do { int ierr, rank; ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] LINE %d (%s)\n", rank, __LINE__, __FUNCTION__);CHKERRQ(ierr); } while (0)

#define CE do { CHKERRQ(ierr); } while (0)

#include "src/sys/src/viewer/viewerimpl.h"    /*I   "petsc.h"   I*/
#if defined(PETSC_HAVE_HDF4) && !defined(PETSC_USE_COMPLEX)
#include <mfhdf.h>
#endif

typedef struct {
 int sd_id;
 char *filename;
 PetscViewerBinaryType btype;
} PetscViewer_HDF4;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_HDF4" 
int PetscViewerDestroy_HDF4(PetscViewer v)
{
 int ierr;
 PetscViewer_HDF4 *vhdf4 = (PetscViewer_HDF4 *)v->data;

 PetscFunctionBegin;
 if (vhdf4->sd_id >= 0) {
 SDend(vhdf4->sd_id);
 vhdf4->sd_id = -1;
 }
 if (vhdf4->filename) {
 PetscFree(vhdf4->filename);
 }
 ierr = PetscFree(vhdf4); CE;
 PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
EXTERN int PetscViewerSetFilename_HDF4(PetscViewer,const char[]);

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_HDF4" 
int PetscViewerCreate_HDF4(PetscViewer v)
{  
 int ierr;
 PetscViewer_HDF4 *vhdf4;
 
 PetscFunctionBegin;
 ierr = PetscNew(PetscViewer_HDF4,&vhdf4); CE;
 v->data            = (void*)vhdf4;
 v->ops->destroy    = PetscViewerDestroy_HDF4;
 v->ops->flush      = 0;
 v->iformat         = 0;
 vhdf4->btype       = (PetscViewerBinaryType) -1; 
 vhdf4->filename    = 0;
 vhdf4->sd_id       = -1;
#if 0
 v->ops->getsingleton     = PetscViewerGetSingleton_Binary;
 v->ops->restoresingleton = PetscViewerRestoreSingleton_Binary;
#endif 
 
 ierr = PetscObjectComposeFunctionDynamic(
 (PetscObject)v,
 "PetscViewerSetFilename_C",
 "PetscViewerSetFilename_HDF4",
 PetscViewerSetFilename_HDF4);CE;
 PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerHDF4SetType" 
int PetscViewerHDF4SetType(PetscViewer viewer, PetscViewerBinaryType type)
{
 PetscViewer_HDF4 *vhdf4 = (PetscViewer_HDF4 *)viewer->data;
 
 PetscFunctionBegin;
 PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
 vhdf4->btype = type;
 PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerHDF4Open" 
int PetscViewerHDF4Open(MPI_Comm comm, const char *name,
 PetscViewerBinaryType type, PetscViewer *hdf4v)
{
 int ierr;
 
 PetscFunctionBegin;
 ierr = PetscViewerCreate(comm,hdf4v);CE;
 ierr = PetscViewerSetType(*hdf4v,PETSC_VIEWER_HDF4);CE;
 ierr = PetscViewerHDF4SetType(*hdf4v, type);CE;
 ierr = PetscViewerSetFilename(*hdf4v, name);CE;
 PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerSetFilename_HDF4" 
int PetscViewerSetFilename_HDF4(PetscViewer viewer,const char name[])
{
 int                   ierr, rank;
 PetscViewer_HDF4      *vhdf4 = (PetscViewer_HDF4*)viewer->data;
 int32                 acc;

 PetscFunctionBegin;

 switch (vhdf4->btype) {
 case PETSC_BINARY_RDONLY:
   acc = DFACC_READ;
   break;
 case PETSC_BINARY_WRONLY:
   acc = DFACC_WRITE;
 break;
 case PETSC_BINARY_CREATE:
   acc = DFACC_CREATE;
   break;
 default:
   SETERRQ(1,"Must call PetscViewerHDF4SetType() before "
	   "PetscViewerSetFilename()");
 }

 ierr = MPI_Comm_rank(viewer->comm,&rank);CE;

 ierr = PetscStrallocpy(name,&vhdf4->filename);CE;

 if (rank == 0) {
   vhdf4->sd_id = SDstart(name, acc);
   if (vhdf4->sd_id < 0) {
     SETERRQ1(1, "SDstart failed for %s", name);
   }
 }

 viewer->format = PETSC_VIEWER_NOFORMAT;
 PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerHDF4WriteSDS" 
int PetscViewerHDF4WriteSDS(PetscViewer viewer, float *xf, int d, int *dims,int bs)
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
   SETERRQ(1, "SDcreate failed");
 }
 SDwritedata(sds_id, zero32, 0, dims32, xf);
 SDendaccess(sds_id);
 PetscFunctionReturn(0);
}

