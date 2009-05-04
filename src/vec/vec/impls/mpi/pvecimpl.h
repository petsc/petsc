
#ifndef __PVECIMPL
#define __PVECIMPL

#include "private/vecimpl.h"
#include "../src/vec/vec/impls/dvecimpl.h"

typedef struct {
  VECHEADER
  MPI_Request *send_waits,*recv_waits;  /* for communication during VecAssembly() */
  PetscInt    nsends,nrecvs;
  PetscScalar *svalues,*rvalues;
  PetscInt    rmax;
  
  PetscInt    nghost;                   /* length of local portion including ghost padding */
  
  Vec         localrep;                 /* local representation of vector */
  VecScatter  localupdate;              /* scatter to update ghost values */
} Vec_MPI;

EXTERN PetscErrorCode VecMDot_MPI(Vec,PetscInt,const Vec[],PetscScalar *);
EXTERN PetscErrorCode VecMTDot_MPI(Vec,PetscInt,const Vec[],PetscScalar *);
EXTERN PetscErrorCode VecNorm_MPI(Vec,NormType,PetscReal *);
EXTERN PetscErrorCode VecMax_MPI(Vec,PetscInt *,PetscReal *);
EXTERN PetscErrorCode VecMin_MPI(Vec,PetscInt *,PetscReal *);
EXTERN PetscErrorCode VecDestroy_MPI(Vec);
EXTERN PetscErrorCode VecView_MPI_File(Vec,PetscViewer);
EXTERN PetscErrorCode VecView_MPI_Files(Vec,PetscViewer);
EXTERN PetscErrorCode VecView_MPI_Binary(Vec,PetscViewer);
EXTERN PetscErrorCode VecView_MPI_Netcdf(Vec,PetscViewer);
EXTERN PetscErrorCode VecView_MPI_Draw_LG(Vec,PetscViewer);
EXTERN PetscErrorCode VecView_MPI_Socket(Vec,PetscViewer);
EXTERN PetscErrorCode VecView_MPI(Vec,PetscViewer);
EXTERN PetscErrorCode VecGetSize_MPI(Vec,PetscInt *);
EXTERN PetscErrorCode VecSetValues_MPI(Vec,PetscInt,const PetscInt [],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode VecSetValuesBlocked_MPI(Vec,PetscInt,const PetscInt [],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode VecAssemblyBegin_MPI(Vec);
EXTERN PetscErrorCode VecAssemblyEnd_MPI(Vec);

EXTERN PetscErrorCode VecCreate_MPI_Private(Vec,PetscTruth,PetscInt,const PetscScalar[]);

#endif



