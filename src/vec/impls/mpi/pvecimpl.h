/* $Id: pvecimpl.h,v 1.37 2001/09/07 20:09:01 bsmith Exp $ */
/* 
 */

#ifndef __PVECIMPL
#define __PVECIMPL

#include "vecimpl.h"
#include "src/vec/impls/dvecimpl.h"

typedef struct {
  VECHEADER
  MPI_Request *send_waits,*recv_waits;  /* for communication during VecAssembly() */
  int         nsends,nrecvs;
  PetscScalar *svalues,*rvalues;
  int         rmax;
  
  int         nghost;                   /* length of local portion including ghost padding */
  
  Vec         localrep;                 /* local representation of vector */
  VecScatter  localupdate;              /* scatter to update ghost values */
} Vec_MPI;

EXTERN int VecNorm_Seq(Vec,NormType,PetscReal *work);
EXTERN int VecMDot_MPI(int,Vec,const Vec[],PetscScalar *);
EXTERN int VecMTDot_MPI(int,Vec,const Vec[],PetscScalar *);
EXTERN int VecNorm_MPI(Vec,NormType,PetscReal *);
EXTERN int VecMax_MPI(Vec,int *,PetscReal *);
EXTERN int VecMin_MPI(Vec,int *,PetscReal *);
EXTERN int VecDestroy_MPI(Vec);
EXTERN int VecView_MPI_File(Vec,PetscViewer);
EXTERN int VecView_MPI_Files(Vec,PetscViewer);
EXTERN int VecView_MPI_Binary(Vec,PetscViewer);
EXTERN int VecView_MPI_Netcdf(Vec,PetscViewer);
EXTERN int VecView_MPI_Draw_LG(Vec,PetscViewer);
EXTERN int VecView_MPI_Socket(Vec,PetscViewer);
EXTERN int VecView_MPI(Vec,PetscViewer);
EXTERN int VecGetSize_MPI(Vec,int *);
EXTERN int VecSetValues_MPI(Vec,int,const int [],const PetscScalar[],InsertMode);
EXTERN int VecSetValuesBlocked_MPI(Vec,int,const int [],const PetscScalar[],InsertMode);
EXTERN int VecAssemblyBegin_MPI(Vec);
EXTERN int VecAssemblyEnd_MPI(Vec);

EXTERN int VecCreate_MPI_Private(Vec,int,const PetscScalar[],PetscMap);

#endif



