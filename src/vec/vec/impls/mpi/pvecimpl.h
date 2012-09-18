
#ifndef __PVECIMPL
#define __PVECIMPL

#include <petsc-private/vecimpl.h>
#include <../src/vec/vec/impls/dvecimpl.h>

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

extern PetscErrorCode VecMDot_MPI(Vec,PetscInt,const Vec[],PetscScalar *);
extern PetscErrorCode VecTDot_MPI(Vec,Vec,PetscScalar *);
extern PetscErrorCode VecMTDot_MPI(Vec,PetscInt,const Vec[],PetscScalar *);
extern PetscErrorCode VecNorm_MPI(Vec,NormType,PetscReal *);
extern PetscErrorCode VecMax_MPI(Vec,PetscInt *,PetscReal *);
extern PetscErrorCode VecMin_MPI(Vec,PetscInt *,PetscReal *);
extern PetscErrorCode VecDestroy_MPI(Vec);
extern PetscErrorCode VecView_MPI_Binary(Vec,PetscViewer);
extern PetscErrorCode VecView_MPI_Netcdf(Vec,PetscViewer);
extern PetscErrorCode VecView_MPI_Draw_LG(Vec,PetscViewer);
extern PetscErrorCode VecView_MPI_Socket(Vec,PetscViewer);
extern PetscErrorCode VecView_MPI_HDF5(Vec,PetscViewer);
extern PetscErrorCode VecView_MPI(Vec,PetscViewer);
extern PetscErrorCode VecGetSize_MPI(Vec,PetscInt *);
extern PetscErrorCode VecPlaceArray_MPI(Vec,const PetscScalar []);
extern PetscErrorCode VecGetValues_MPI(Vec,PetscInt,const PetscInt [], PetscScalar []);
extern PetscErrorCode VecSetValues_MPI(Vec,PetscInt,const PetscInt [],const PetscScalar[],InsertMode);
extern PetscErrorCode VecSetValuesBlocked_MPI(Vec,PetscInt,const PetscInt [],const PetscScalar[],InsertMode);
extern PetscErrorCode VecAssemblyBegin_MPI(Vec);
extern PetscErrorCode VecAssemblyEnd_MPI(Vec);

extern PetscErrorCode VecCreate_MPI_Private(Vec,PetscBool ,PetscInt,const PetscScalar[]);

#endif



