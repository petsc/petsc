
#if !defined(__PVECIMPL)
#define __PVECIMPL

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

PETSC_INTERN PetscErrorCode VecMDot_MPI(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecMTDot_MPI(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecNorm_MPI(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecMax_MPI(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecMin_MPI(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecDestroy_MPI(Vec);
PETSC_INTERN PetscErrorCode VecView_MPI_Binary(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_Netcdf(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_Draw_LG(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_Socket(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_HDF5(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_MPI(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecGetSize_MPI(Vec,PetscInt*);
PETSC_INTERN PetscErrorCode VecGetValues_MPI(Vec,PetscInt,const PetscInt [], PetscScalar []);
PETSC_INTERN PetscErrorCode VecSetValues_MPI(Vec,PetscInt,const PetscInt [],const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode VecSetValuesBlocked_MPI(Vec,PetscInt,const PetscInt [],const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode VecAssemblyBegin_MPI(Vec);
PETSC_INTERN PetscErrorCode VecAssemblyEnd_MPI(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPI_Private(Vec,PetscBool,PetscInt,const PetscScalar[]);



#endif



