/* $Id: pvecimpl.h,v 1.23 1999/02/04 22:31:12 bsmith Exp balay $ */
/* 
 */

#ifndef __PVECIMPL
#define __PVECIMPL

#include "src/vec/vecimpl.h"
#include "src/vec/impls/dvecimpl.h"

typedef  struct {
  int    donotstash;                   /* Flag indicates stash values should be ignored */
  int    nmax;                         /* Current max-size of the array. */
  int    oldnmax;                      /* previous max size. Used with new mallocs */
  int    n;                            /* no of entries currently stashed */
  int    *idx;                         /* list of indices */
  Scalar *array;                       /* values corresponding to these indices */
  } VecStash;

typedef struct {
  VECHEADER
  int         N;                         /* length of total vector */
  int         size,rank;
  InsertMode  insertmode;
  VecStash    stash;
  MPI_Request *send_waits,*recv_waits;  /* for communication during VecAssembly() */
  int         nsends,nrecvs;
  Scalar      *svalues,*rvalues;
  int         rmax;
  
  int         nghost;                   /* length of local portion including ghost padding */
  
  Vec         localrep;                 /* local representation of vector */
  VecScatter  localupdate;              /* scatter to update ghost values */
} Vec_MPI;

extern int VecNorm_Seq(Vec, NormType, double *work );
extern int VecMDot_MPI(int, Vec,const Vec[], Scalar *);
extern int VecMTDot_MPI(int, Vec,const Vec[], Scalar *);
extern int VecNorm_MPI(Vec,NormType, double *);
extern int VecMax_MPI(Vec, int *, double *);
extern int VecMin_MPI(Vec, int *, double *);
extern int VecGetOwnershipRange_MPI(Vec,int *,int*); 
extern int VecDestroy_MPI(Vec);
extern int VecView_MPI_File(Vec, Viewer);
extern int VecView_MPI_Files(Vec, Viewer);
extern int VecView_MPI_Binary(Vec , Viewer);
extern int VecView_MPI_Draw_LG(Vec ,Viewer);
extern int VecView_MPI_Socket(Vec , Viewer);
extern int VecView_MPI(Vec,Viewer);
extern int VecGetSize_MPI(Vec,int *);
extern int VecSetValues_MPI(Vec, int, const int [], const Scalar[],InsertMode);
extern int VecSetValuesBlocked_MPI(Vec, int, const int [], const Scalar[],InsertMode);
extern int VecAssemblyBegin_MPI(Vec);
extern int VecAssemblyEnd_MPI(Vec);

extern int VecCreate_MPI_Private(Vec,int,const Scalar[],Map);

#endif



