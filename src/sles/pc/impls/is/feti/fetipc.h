/* Implementation of the FETI-DP-preconditioner */
/* /src/sles/pc/impls/is/feti/fetipc.h */
#if !defined(__fetipc_h)
#define __fetipc_h

/* Naming convention: Member-Functions start with the name of the struct they belong to */
/* the pointer to the struct (pointer to this) is always first */
/* although functions may get a PC object (pointer to _P_PC)   */

#include "src/sles/pc/pcimpl.h" /* also includes petscksp.h and petscpc.h */
#include "petscsles.h"

#include "src/mat/impls/feti/feti.h" /* */

/* one FetiPartition is a collection of FetiDomains executed on one processor */

/* The PC holds a pointer to the matrix (?) and thereby can access e.g. the scatters */

typedef struct {

    Mat Kbb; /* for unassembled Schur complement */
    Mat Kib;
    Mat Kii;
    SLES Kii_inv;

    Vec D;    /* diagonal scaling matrix */

} PC_FetiDomain;


typedef struct {

    PC_FetiDomain * pcdomains;

} PC_Feti;
/* The scatters Br (and probably also the scaling rho) will be taken from Mat_Feti */

int PC_FetiApplySchurComplement(Vec xb, Vec yb)
{

}

#endif
