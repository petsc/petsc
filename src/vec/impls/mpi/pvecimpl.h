/* $Id: pvecimpl.h,v 1.16 1997/11/23 16:49:07 bsmith Exp bsmith $ */
/* 
 */

#ifndef __PVECIMPL
#define __PVECIMPL

#include "src/vec/vecimpl.h"
#include "src/vec/impls/dvecimpl.h"

typedef struct {
    VECHEADER
    int         N;                         /* length of total vector */
    int         size,rank,*ownership;
    InsertMode  insertmode;

    struct      {int donotstash, nmax, n, *idx; Scalar *array;} stash;

    MPI_Request *send_waits,*recv_waits;  /* for communication during VecAssembly() */
    int         nsends,nrecvs;
    Scalar      *svalues,*rvalues;
    int         rmax;

    int         nghost;                   /* length of local portion including ghost padding */

    Vec         localrep;                 /* local representation of vector */
    VecScatter  localupdate;              /* scatter to update ghost values */
} Vec_MPI;

extern int VecNorm_Seq(Vec, NormType, double *work );
extern int VecMDot_MPI(int, Vec, Vec *, Scalar *);
extern int VecMTDot_MPI(int, Vec, Vec *, Scalar *);
extern int VecNorm_MPI(Vec,NormType, double *);
extern int VecMax_MPI(Vec, int *, double *);
extern int VecMin_MPI(Vec, int *, double *);
extern int VecGetOwnershipRange_MPI(Vec,int *,int*); 
extern int VecDestroy_MPI(PetscObject);
extern int VecView_MPI_File(Vec, Viewer);
extern int VecView_MPI_Files(Vec, Viewer);
extern int VecView_MPI_Binary(Vec , Viewer);
extern int VecView_MPI_Draw_LG(Vec ,Viewer);
extern int VecView_MPI_Draw(Vec , Viewer);
extern int VecView_MPI_Matlab(Vec , Viewer);
extern int VecView_MPI(PetscObject,Viewer);
extern int VecGetSize_MPI(Vec,int *);
extern int VecSetValues_MPI(Vec, int, int *, Scalar*,InsertMode);
extern int VecSetValuesBlocked_MPI(Vec, int, int *, Scalar*,InsertMode);
extern int VecAssemblyBegin_MPI(Vec);
extern int VecAssemblyEnd_MPI(Vec);

extern int VecCreateMPI_Private(MPI_Comm,int,int,int,int,int,int *,Scalar *,Vec *);

#endif



