/* Implementation of the Feti-DP-preconditioner */
/* /src/sles/pc/impls/is/feti/fetipc.h */
#if !defined(__fetipc_h)
#define __fetipc_h

/* Naming convention: 
-  Member-Functions start with the name of the struct they belong to 
-  if they overwrite functions from base class (i.e. they will go to  ops) 
   they start with the name of the base class and have a suffix for the derived class
-  the pointer to the struct (this-pointer) is always the first argument              */

#include "../src/sles/pc/pcimpl.h"      /* also includes petscksp.h and petscpc.h */
#include "petscsles.h"

#include "../src/mat/impls/feti/feti.h" /* */

#define FETIDP_KBB     "Kbb"
#define FETIDP_KII     "Kii"
#define FETIDP_KIB     "Kib"   /* although Kbi may be more efficient */
#define FETIDP_BBR     "Bbr"

typedef enum { FETI_SCALING_MULTIPLICITY, FETI_SCALING_RHO, FETI_SCALING_NONE } FetiScaleType;

typedef struct {

    int domain_num;

    /* unassembled Schur complement */
    Mat  Kbb;           /* these are boundary without corners; => br */
    Mat  Kib;          
    Mat  Kii;           /* interior */
    SLES Kii_inv;

    Mat Bbr;            /* translate from index set r to index set br */

    Vec  D;             /* diagonal scaling matrix */

    /* ------   Solver internal data structures  ------ */
    Vec ubr_tmp1;       /* can have different sizes in different domains */
    Vec ubr_tmp2;  
    Vec uii_tmp1;
    Vec uii_tmp2;

} PC_FetiDomain;

typedef struct {

    PC_FetiDomain * pcdomains;
    int n_dom;

    PetscTruth lumped;  /* do not use Dirichlet PC; economizes the schur complement */

#if 0
    static const FetiScaleType scaling=FETI_SCALING_MULTIPLICITY;
#else
    static const FetiScaleType scaling=FETI_SCALING_RHO;  /* for the moment fixed to rho */

#endif

} PC_Feti;
/* The scatters Br (and also the scaling rho) will be taken from Mat_Feti */

/* forward declarations */
int PCApply_Feti(PC pc, Vec src_lambda, Vec dst_lambda);
int PCCreate_Feti(PC pc);
int PCFetiDomainLoad(PC_FetiDomain *pcdomain,const char * prefix);
int PCFetiSetUpTemporarySpace(PC pc);
int PCFetiDestroyTemporarySpace(PC pc);
int PCFetiSetUp(PC pc);
int PCDestroy_Feti(PC pc);
int PCFetiDomainDestroy(PC_FetiDomain *pcdomain);

/* user functions */
int PCCreateFeti(PC *pc);
int PCFetiSetMatFeti(PC pc, Mat A); /* set pointer to MatFeti for access to B aso. */
int PCLoad_Feti(PC pc);

#endif

