/* $Id: pvecimpl.h,v 1.10 1996/11/19 16:29:42 bsmith Exp bsmith $ */
/* 
 */

#ifndef __PVECIMPL
#define __PVECIMPL

#include "src/vec/vecimpl.h"
#include "src/vec/impls/dvecimpl.h"

/* The first two elements of this structure should remain the same */
typedef struct {
    VECHEADER
    int         N;           /* length of total vector */
    int         size,rank,*ownership;
    InsertMode  insertmode;
    struct      {int donotstash, nmax, n, *idx; Scalar *array;} stash;
    MPI_Request *send_waits,*recv_waits;
    int         nsends,nrecvs;
    Scalar      *svalues,*rvalues;
    int         rmax;
} Vec_MPI;

extern int VecNorm_Seq(Vec, NormType, double *work );
extern int VecMDot_MPI(int, Vec, Vec *, Scalar *);
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
extern int VecAssemblyBegin_MPI(Vec);
extern int VecAssemblyEnd_MPI(Vec);

#endif



