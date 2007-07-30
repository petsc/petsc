/*
      Data structure used for adaptive smoothed aggregation preconditioner.
*/
#if !defined(__ASA_IMPL)
#define __ASA_IMPL
#include "private/pcimpl.h"
#include "petscasa.h"

/*
     Structure for adaptive smoothed aggregation solver. 
*/

/* 
   Private context (data structure) for the ASA preconditioner.

   This is one level in the multigrid hierarchy
*/
struct __PC_ASA_level {
  PetscInt   level;                        /* current level: 1 original level, 2 coarser, ... */
  PetscInt   size;                         /* The size of the matrices and vectors for the
					      current level */

  Mat        A;                            /* The coarsened matrix for the current level */
  Mat        B;                            /* Near kernel components. We will 
					      allocate sufficient space for max_cand_vecs */
  PetscInt   cand_vecs;                    /* How many candidate vectors are stored in B */

  /* Temp working vectors */
  Vec        x;                            /* Current candidate vector to be processed */
  Vec        b;                            /* Current right hand side */
  Vec        r;                            /* Current residual */
  
  DM         dm;                           /* Grid information about the layout.
					      Undecided on data structure for algebraic case
					      combine with DM in some way */

  PetscInt   aggnum;                       /* The number of aggregates */
  Mat        agg;                          /* The aggregates. Row i has an entry in the columns j
					      corresponding to the fine-level points that map to 
					      the i-th node on the coarse level */
  PetscInt   *loc_agg_dofs;                /* The dofs for each aggregate, each processor only stores
					      the dof for the local aggregates */
  Mat        agg_corr, bridge_corr;        /* Correction terms for the aggregates that are introduced
					      by adding candidate vectors. The final structure of the
					      aggregate matrix agg is given by agg*agg_corr if we do not
					      construct a bridge operator and agg*bridge_corr, if we do.*/

  Mat        P;                            /* tentative prolongator */
  Mat        Pt;                           /* tentative restriction operator (P^t) */
/*   Mat        S;                            /\* prolongation smoother *\/ */
  PetscReal  spec_rad;                     /* estimate of the spectral radius of A */
  Mat        smP;                            /* smoothed prolongator */
  Mat        smPt;                           /* smoothed restriction operator */
  
  MPI_Comm   comm;                         /* communicator object for this level */

  KSP        smoothd;                      /* pre smoother */
  KSP        smoothu;                      /* post smoother */

  struct __PC_ASA_level *prev, *next;      /* next and previous levels */
};

typedef struct __PC_ASA_level  PC_ASA_level;

/* 
   Private context (data structure) for the ASA preconditioner.

   This is the abstract object that contains the global data for the ASA object
*/
typedef struct {
  /* parameters for the algorithm */
  PetscInt   nu;                           /* Number of cycles to run smoother */
  PetscInt   gamma;                        /* Number of cycles to run coarse grid correction */
  PetscReal  epsilon;                      /* Tolerance for the relaxation method */
  PetscInt   mu;                           /* Number of cycles to relax in setup stages */
  PetscInt   mu_initial;                   /* Number of cycles to relax for generating first candidate vector */
  PetscInt   direct_solver;                /* For which matrix size should we use the direct solver? */
  PetscTruth scale_diag;                   /* Should we scale the matrix with the inverse of its diagonal? */
  /* parameters for relaxation */
  char *     ksptype_smooth;               /* The relaxation method used on each level (KSP) */
  char *     pctype_smooth;                /* The relaxation method used on each level (PC) */
  PetscReal  smoother_rtol;                /* Relative convergence tolerance for smoothers */
  PetscReal  smoother_abstol;              /* Absolute convergence tolerance for smoothers */
  PetscReal  smoother_dtol;                /* Divergence tolerance for smoothers */
  char *     ksptype_direct;               /* The solving method used on the coarsest level (KSP) */
  char *     pctype_direct;                /* The solving method used on the coarsest level (PC) */
  PetscReal  direct_rtol;                  /* Relative convergence tolerance for direct solver */
  PetscReal  direct_abstol;                /* Absolute convergence tolerance for direct solver */
  PetscReal  direct_dtol;                  /* Divergence tolerance for direct solver */
  /* parameters for various relaxation methods */
  PetscReal  richardson_scale;             /* Scaling parameter to use if relaxation KSP is Richardson.
					      In this case each step is
					      x^{(k+1)} = x^{(k)} + richardson_scale * M^{-1} (b - A x^{(k)})
					      where M is the preconditioning matrix. */
  PetscReal  sor_omega;                    /* omega parameter for SOR PC */
  /* parameters for direct solver */
  char *     coarse_mat_type;              /* matrix type to use for distributed LU factorization
					      (e.g. superlu_dist, mumps, etc.)*/
  /* parameters that limit the allocation */
  PetscInt   max_cand_vecs;                /* Maximum number of candidate vectors */
  PetscInt   max_dof_lev_2;                /* The maximum number of degrees of freedom per
					      node on level 2 (K in paper) */

  PetscTruth multigrid_constructed;        /* Flag that checks whether we have constructed an
					      applicable multigrid yet */

  /* parameters that rule the behaviour of the iteration */
  PetscReal  rtol;                         /* relative error convergence criteria */
  PetscReal  abstol;                       /* absolute error convergence criteria */
  PetscReal  divtol;                       /* divergence threshold */
  PetscInt   max_it;                       /* maximum number of iterations */
  PetscReal  rq_improve;                   /* determines when we should add another candidate during
					      the construction of the multigrid. candidate is added when
					      the new Rayleigh quotient rq_new does not beat
					      rq_improve*rq_old */

  /* data regarding the problem */
  Mat        A;                            /* matrix used in forming residual, scaled by insqrtdiag */
  Vec        invsqrtdiag;                  /* 1/sqrt(diagonal A) */
  Vec        b;                            /* Right hand side */ 
  Vec        x;                            /* Solution */
  Vec        r;                            /* Residual */

  /* grid information, not sure how to treat unstructured grids yet */
  DM         dm;                           /* Grid information about the layout */

  /* the actual levels */
  PetscInt   levels;                       /* number of active levels used */
  PC_ASA_level *levellist;                 /* linked list of all levels */

  MPI_Comm   comm;                         /* communicator object for ASA */

} PC_ASA;

#endif

