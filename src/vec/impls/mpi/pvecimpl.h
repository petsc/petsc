/* $Id: pvecimpl.h,v 1.8 1996/08/04 21:16:15 bsmith Exp bsmith $ */
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
    struct      {int nmax, n, *idx; Scalar *array;} stash;
    MPI_Request *send_waits,*recv_waits;
    int         nsends,nrecvs;
    Scalar      *svalues,*rvalues;
    int         rmax;
} Vec_MPI;

extern int VecNorm_Seq(Vec, NormType, double *work );

#endif



