/* $Id: pvecimpl.h,v 1.33 2001/01/15 21:45:04 bsmith Exp bsmith $ */
/* 
 */

#ifndef __PVECIMPL
#define __PVECIMPL

#include "src/vec/vecimpl.h"
#include "src/vec/impls/dvecimpl.h"

typedef struct {
  VECHEADER
  int         size,rank;
  InsertMode  insertmode;
  PetscTruth  donotstash;               /* Flag indicates stash values should be ignored */
  int         *browners;                /* block-row-ownership,used for assembly */
  MPI_Request *send_waits,*recv_waits;  /* for communication during VecAssembly() */
  int         nsends,nrecvs;
  Scalar      *svalues,*rvalues;
  int         rmax;
  
  int         nghost;                   /* length of local portion including ghost padding */
  
  Vec         localrep;                 /* local representation of vector */
  VecScatter  localupdate;              /* scatter to update ghost values */
} Vec_MPI;

EXTERN int VecNorm_Seq(Vec,NormType,double *work);
EXTERN int VecMDot_MPI(int,Vec,const Vec[],Scalar *);
EXTERN int VecMTDot_MPI(int,Vec,const Vec[],Scalar *);
EXTERN int VecNorm_MPI(Vec,NormType,double *);
EXTERN int VecMax_MPI(Vec,int *,double *);
EXTERN int VecMin_MPI(Vec,int *,double *);
EXTERN int VecGetOwnershipRange_MPI(Vec,int *,int*); 
EXTERN int VecDestroy_MPI(Vec);
EXTERN int VecView_MPI_File(Vec,PetscViewer);
EXTERN int VecView_MPI_Files(Vec,PetscViewer);
EXTERN int VecView_MPI_Binary(Vec,PetscViewer);
EXTERN int VecView_MPI_Draw_LG(Vec,PetscViewer);
EXTERN int VecView_MPI_Socket(Vec,PetscViewer);
EXTERN int VecView_MPI(Vec,PetscViewer);
EXTERN int VecGetSize_MPI(Vec,int *);
EXTERN int VecSetValues_MPI(Vec,int,const int [],const Scalar[],InsertMode);
EXTERN int VecSetValuesBlocked_MPI(Vec,int,const int [],const Scalar[],InsertMode);
EXTERN int VecAssemblyBegin_MPI(Vec);
EXTERN int VecAssemblyEnd_MPI(Vec);

EXTERN int VecCreate_MPI_Private(Vec,int,const Scalar[],PetscMap);

#endif



