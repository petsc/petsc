/* $Id: pvecimpl.h,v 1.6 1995/10/19 22:16:25 curfman Exp bsmith $ */
/* 
 */

#ifndef __PVECIMPL
#define __PVECIMPL

#include "vecimpl.h"
#include "dvecimpl.h"

/* The first two elements of this structure should remain the same */
typedef struct {
    int         n;           /* Length of LOCAL vector */
    Scalar      *array;
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



