/* $Id: pvecimpl.h,v 1.29 1999/10/13 20:37:07 bsmith Exp bsmith $ */
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
  int         donotstash;               /* Flag indicates stash values should be ignored */
  int         *browners;                /* block-row-ownership,used for assembly */
  MPI_Request *send_waits,*recv_waits;  /* for communication during VecAssembly() */
  int         nsends,nrecvs;
  Scalar      *svalues,*rvalues;
  int         rmax;
  
  int         nghost;                   /* length of local portion including ghost padding */
  
  Vec         localrep;                 /* local representation of vector */
  VecScatter  localupdate;              /* scatter to update ghost values */
} Vec_MPI;

extern int VecNorm_Seq(Vec,NormType,double *work);
extern int VecMDot_MPI(int,Vec,const Vec[],Scalar *);
extern int VecMTDot_MPI(int,Vec,const Vec[],Scalar *);
extern int VecNorm_MPI(Vec,NormType,double *);
extern int VecMax_MPI(Vec,int *,double *);
extern int VecMin_MPI(Vec,int *,double *);
extern int VecGetOwnershipRange_MPI(Vec,int *,int*); 
extern int VecDestroy_MPI(Vec);
extern int VecView_MPI_File(Vec,Viewer);
extern int VecView_MPI_Files(Vec,Viewer);
extern int VecView_MPI_Binary(Vec,Viewer);
extern int VecView_MPI_Draw_LG(Vec,Viewer);
extern int VecView_MPI_Socket(Vec,Viewer);
extern int VecView_MPI(Vec,Viewer);
extern int VecGetSize_MPI(Vec,int *);
extern int VecSetValues_MPI(Vec,int,const int [],const Scalar[],InsertMode);
extern int VecSetValuesBlocked_MPI(Vec,int,const int [],const Scalar[],InsertMode);
extern int VecAssemblyBegin_MPI(Vec);
extern int VecAssemblyEnd_MPI(Vec);

extern int VecCreate_MPI_Private(Vec,int,const Scalar[],Map);

#endif



