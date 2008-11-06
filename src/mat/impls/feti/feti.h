/* feti.h defines a matrix type that stores the Feti-DP system matrix */ 
/* /src/mat/impls/feti/feti.h */

#if !defined(__feti_h)
#define __feti_h

#include <stdlib.h>
#include "private/matimpl.h"       /* includes also petscmat.h */
#include "petscsles.h"
#include "../src/vec/vecimpl.h"       /* otherwise complains at first use of vec->N aso. (typedef, but declaration) */ 
#include <string.h>

#define MATFETI         "MatFeti"
#define FETIDP_PREFIX   "data/FetiDP_"
#define FETIDP_KCC      "Kcc"      /* only globally assembled needed; all processors read this */
#define FETIDP_KRC      "Krc"
#define FETIDP_KRR      "Krr"
#define FETIDP_BRT      "BrT"
#define FETIDP_FR       "fr"
#define FETIDP_FC       "fc"       /* globally assembled needed */
#define FETIDP_RHO      "rho"      /* all processors read this */
#define FETIDP_BC       "Bc"
#define FETIDP_Q        "Q"

#define FETIDP_VIEWER_XDISPLAY "localhost:0.0"

#define ASSERT(A,S) if(!(A)) SETERRQ1(4711,"Assertion Failed. %s\n",(S));
#define ASSERT2(A,S,x,y) if(!(A)) SETERRQ3(4711,"Assertion Failed. %s  %d  %d\n",(S),(x),(y));
#define WARN_IF(A,S) if((A)) { PetscSynchronizedPrintf(PETSC_COMM_SELF,"Warning. %s\n",(S)); PetscSynchronizedFlush(PETSC_COMM_SELF); }
#define WAIT(A) {int d; PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s Press key.\n",(A)); PetscSleep(1); }

typedef enum {
               FETI_SCATTER_FORWARD_ADD,          /* scales only with the sign of scaling i.e. (+1)/(-1) */
	       FETI_SCATTER_REVERSE_INSERT, 
	       FETI_SCATTER_FORWARD_ADD_SCALE,    /* scales with the double value of scaling, for the pc */
	       FETI_SCATTER_REVERSE_INSERT_SCALE
             } FetiScatterMode;

/* -------------------------------------------  FetiDomain  -------------------------------------------- */

typedef struct { 

    /* ------         User provided input        ------ */
    int domain_num;      /* global number of the domain */

          /*      all matrices here are sequential aij      */

    Mat Krr;             /* local stiffness matrix */
    Mat Krc;             

    Vec fr;

    PetscScalar my_rho;  /* a representative for the local coefficient
                            that will be used for scaling                     */
                         /* for now, we have only one per domain; even though 
			    the coefficient may vary within the domain        */

    Mat BrT;             /* transpose if the connectivity matrices; transposed so that
		            columns are more easily accessible using MatGetRow            */

    Mat Bc;              /* the corner connectivity matrix */

    Mat Q;               /* matrix for optional (e.g. edge) constraints */
                         /* storing Q not QT (mu is small but grows with domains)=> ur=Q*mu */

    /* ------   Solver internal data structures  ------ */
    SLES Krrinv;         /* inverse of Krr  */

    Vec ur;              /* 1. temporary space for solver; 
			    2. space for storing the primal solution */

    PetscScalar * ur_array;       /* temporary for VecGetArray */

    Vec ur_multiplicity;          /* will store the multiplicity */

    Vec ur_tmp1;         /* space for storing temporary results, same dimensions as ur */
    Vec ur_tmp2;         /* space for storing temporary results, same dimensions as ur */

    Vec ucg_tmp1;        /* space for storing temporary results, global c = c_ass */
    Vec ucg_tmp2;        /* space for storing temporary results, global c = c_ass */

    Vec uc;              /* 1. space for storing temporary results, 
			    2. space for storing the primal solution on the corners */

    PetscScalar * uc_array;       /* temporary for VecGetArray */

    Vec mu_tmp;

    const static bool use_Q=0;

} FetiDomain;            

/* -------------------------------------------  Mat_Feti  -------------------------------------------- */
typedef struct {
    /* Note that PETSCHEADER(struct_ MatOps); in struct _p_Mat introduces the virtual function table Mat->ops */

    /* ------            MPI-Matrices            ------ */
    Mat Kcc_ass;      /*   corner matrix global/assembled                                          */
    Vec fc_ass;       /*   corner rhs                                                              */

    Mat Scc_ass;      /*   corner Schur-complement Scc before consolidation to all processors      */
    Vec fScc_ass;     /*   rhs fScc before consolidation to all processors                         */
                      /*   both could be made local variables to the respective functions          */
    Vec uc_ass_tmp;   /*   used by BcT-Scatter; MPI-Vec like fc_ass                                */
    Vec uc_tmp1;      /*   used by BcT-Scatter; sequential like Scc                                */
    Vec uc_tmp2;      /*   used by BcT-Scatter; sequential like Scc                                */

    /* ------           Seq-Matrices             ------ */
    Mat Scc;          /*   Scc after consolidation                                                 */
    SLES Sccinv;      /*   inverse of Scc                                                          */
    Vec fScc;         /*   fScc after consolidation                                                */

    int n_dom;        /* number of domains on this processor; equals length of domains[]           */
    FetiDomain * domains;

    Vec contrib;      /* temporary space where all ur's from the domains will be stored        */
                      /* local result of Br*ur                                                 */

    /*   First Stage Scatter u ---> contrib        */
    int * scatter_src;                 /*  first stage local scatter to Mat_Feti               */
    int * scatter_src_domain;          /*  first stage local scatter to Mat_Feti               */
    PetscScalar * scatter_scale;       /*  use:  dest[i]=scatter_scale[i]*src[scatter_src[i]]  */

    int scatter_len;                   /* length of scatter_* and contrib                      */

    /*   Second Stage Scatter contrib ---> lambda  */
    VecScatter Br_scatter;             /* one single scatter per processor */

    Vec BcT_contrib;                   

    /* First Stage Scatter uc ---> BcT_contrib     */
    int * BcT_scatter_src;             /* first stage local scatter to Mat_Feti                */
    int * BcT_scatter_src_domain;      /* first stage local scatter to Mat_Feti                */
    PetscScalar * BcT_scatter_scale;   /* is always always one                                 */

    int BcT_scatter_len;               /* length of BcT_scatter_* and BcT_contrib              */

    /* Second Stage Scatter BcT_contrib ---> uc_ass / ucg */
    VecScatter BcT_scatter;           

    Vec mu; 
    Vec ucmu_tmp1; /* now needed for Scctilde */
    Vec ucmu_tmp2;
    Vec mu_seq;

    /* ---------- */

    Vec lambda_copy; /* this stores a copy of the user contributed lambda for access
			to the partitioning information needed by the VecScatter */
                     /* should be a const object, or a rather a pointer to const in this case
			but this is not feasible in C (only way: typedef _p_Vec const* ConstVec; */

} Mat_Feti;          /* a collection of several domains */

/* Some forward declarations */
int MatCreate_Feti(Mat A); /* internal, if this is called, some setup still has to be made manually afterwards */
int MatSetUp_Feti(Mat A);  /* setup coarse, solve local, setup scatter; */
int MatMult_Feti(Mat A, Vec src_lambda, Vec dst_lambda);
int MatDestroy_Feti(Mat a);
int MatView_Feti(Mat A,PetscViewer);
int MatFetiCreateScatter(Mat A);
int MatFetiScatter(Mat A, Vec lambda, FetiScatterMode mode);
int MatFetiScatterBc(Mat A, Vec ucg, FetiScatterMode mode);
int MatFetiSetUpTemporarySpace(Mat A);
int MatFetiSetUpScc(Mat A);
int MatFetiSetUpSccTilde(Mat A);
int MatFetiConvertScc2Seq(Mat A); 
int MatlabInspect(Mat, char const * const, int const);
int MatlabInspectVec(Vec);
int MatlabInspectVecs(Vec v, int dom);
int MatFetiCalculateMultiplicity(Mat A);
int MatFetiCalculateRhoNeighbor(Mat A);
int MatFetiCalculateRhoScaling(Mat A);
int FetiDomainLoad(FetiDomain *, char const * const);
int FetiLoadMatSeq(char const * const prefix, char const * const name, char const * const postfix, Mat* A);
int FetiLoadMatMPI(char const * const prefix, char const * const name, char const * const postfix, Mat* A);
int FetiLoadVecSeq(char const * const prefix, char const * const name, char const * const postfix, Vec* v);
int FetiLoadVecMPI(char const * const prefix, char const * const name, char const * const postfix, Vec* v);
int MatFetiBalance(MPI_Comm, int, int *, int *);              

/* for user use */
int MatCreateFeti(MPI_Comm comm, const int dom_per_proc, Mat* A);
int MatLoad_Feti(Mat A, Vec* lambda); 
int MatFetiCalculateRHS(Mat A, Vec * dr_Scc);
int MatlabWrite_lambda(Mat A);
int MatFetiMatlabInspect(Mat A); 

#endif

