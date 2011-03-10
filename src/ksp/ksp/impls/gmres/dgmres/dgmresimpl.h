/*
   Private data structure used by the GMRES method. This data structure
  must be identical to the beginning of the KSP_FGMRES data structure
  so if you CHANGE anything here you must also change it there.
*/
#if !defined(__DGMRES)
#define __DGMRES

#include "private/kspimpl.h"        /*I "petscksp.h" I*/
#include "petscblaslapack.h"
#if defined(PETSC_BLASLAPACK_UNDERSCORE)
#include "petscdgmresblaslapack_uscore.h"
#elif defined(PETSC_BLASLAPACK_CAPS)
#include "petscdgmresblaslapack_caps.h"
#else
#include "petscdgmresblaslapack_c.h"
#endif

EXTERN_C_BEGIN
#if !defined(PETSC_USE_COMPLEX)
 extern void LAPACKhseqr_(const char *, const char *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *,PetscBLASInt * );
 extern void LAPACKhgeqz_(const char *, const char *, const char *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt * );
 extern void LAPACKgerfs_(const char *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscScalar *, PetscInt *, PetscBLASInt *);
 extern void LAPACKgges_( const char *, const char *, const char *, void **, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *); 
 extern void LAPACKtrsen_(const char *, const char *, PetscBLASInt *, PetscBLASInt *, PetscReal *, PetscBLASInt *, PetscReal *, PetscBLASInt *, PetscReal *, PetscReal *, PetscBLASInt *, PetscReal *, PetscReal *, PetscReal *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
 extern void LAPACKtgsen_(PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscReal *, PetscBLASInt *, PetscReal *, PetscBLASInt *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscBLASInt *, PetscReal *, PetscBLASInt *, PetscBLASInt *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);

#else
 extern void LAPACKhseqr_(const char *, const char *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *,PetscBLASInt * );
 extern void LAPACKhgeqz_(const char *, const char *, const char *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt * );
 extern void LAPACKgerfs_(const char *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscScalar *, PetscReal *, PetscBLASInt *);
 extern void LAPACKgges_( const char *, const char *, const char *, void **, PetscBLASInt *, PetscReal *, PetscBLASInt *, PetscReal *, PetscBLASInt *, PetscBLASInt *, PetscReal *, PetscReal *, PetscReal *, PetscBLASInt *, PetscReal *, PetscBLASInt *, PetscReal *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
 extern void LAPACKtrsen_(const char *, const char *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscReal *, PetscReal *, PetscScalar *, PetscBLASInt *, PetscBLASInt *);
 extern void LAPACKtgsen_(PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscScalar *, PetscScalar *, PetscBLASInt *, PetscScalar *, PetscBLASInt *, PetscBLASInt *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *, PetscBLASInt *);
#endif 
EXTERN_C_END

typedef struct {
  /* Hessenberg matrix and orthogonalization information.  Hes holds
       the original (unmodified) hessenberg matrix which may be used
       to estimate the Singular Values of the matrix */
  PetscScalar *hh_origin,*hes_origin,*cc_origin,*ss_origin,*rs_origin;

  PetscScalar *orthogwork; /* holds dot products computed in orthogonalization */

  /* Work space for computing eigenvalues/singular values */
  PetscReal   *Dsvd;
  PetscScalar *Rsvd;
      
  /* parameters */
  PetscReal   haptol;            /* tolerance used for the "HAPPY BREAK DOWN"  */
  PetscInt    max_k;             /* maximum size of the approximation space  
  before restarting */

  PetscErrorCode (*orthog)(KSP,PetscInt); /* Functions to use (special to gmres) */
  KSPGMRESCGSRefinementType cgstype;
    
  Vec   *vecs;  /* holds the work vectors */
  /* vv_allocated is the number of allocated gmres direction vectors */
  PetscInt    q_preallocate,delta_allocate;
  PetscInt    vv_allocated;
  /* vecs_allocated is the total number of vecs available (used to 
       simplify the dynamic allocation of vectors */
  PetscInt    vecs_allocated;
  /* Since we may call the user "obtain_work_vectors" several times, 
       we have to keep track of the pointers that it has returned 
      (so that we may free the storage) */
  Vec         **user_work;
  PetscInt    *mwork_alloc;    /* Number of work vectors allocated as part of
                               a work-vector chunck */
  PetscInt    nwork_alloc;     /* Number of work vectors allocated */

  /* In order to allow the solution to be constructed during the solution
     process, we need some additional information: */

  PetscInt    it;              /* Current itethreshn: inside restart */
  PetscScalar *nrs;            /* temp that holds the coefficients of the 
                               Krylov vectors that form the minimum residual
                               solution */
  Vec         sol_temp;        /* used to hold temporary solution */
  
  /* Data specific to DGMRES */
  Vec			*U;	/* Vectors that form the basis of the invariant subspace */
  PetscScalar	*T;	/* T=U^T*M^{-1}*A*U */
  PetscScalar	*TF;	/* The factors L and U from T = P*L*U */
  PetscInt 		*InvP;	/* Permutation Vector from the LU factorization of T */
  PetscInt		neig;	/* number of eigenvalues to extract at each restart */
  PetscInt		r;		/* current number of deflated eigenvalues */
  PetscInt 		max_neig;	/* Maximum number of eigenvalues to deflate */
  PetscReal 	lambdaN;	/* modulus of the largest eigenvalue of A */
  PetscReal		smv; 	/* smaller multiple of the remaining allowed number of steps -- used for the adaptive strategy */
  PetscInt		force; /* Force the use of the deflation at the restart */
  PetscInt		matvecs; /* Total number of matrix-vectors */
  PetscInt		GreatestEig; /* Extract the greatest eigenvalues instead */
  
  /* Work spaces */
  Vec			*mu;	/* Save the product M^{-1}AU */
  PetscScalar	*Sr; 	/* Schur vectors to extract */
  Vec			*X; 	/* Schurs Vectors Sr projected to the entire space */
  Vec			*mx; 	/* store the product M^{-1}*A*X */
  PetscScalar	*umx; 	/* store the product U^T*M^{-1}*A*X */
  PetscScalar	*xmu; 	/* store the product X^T*M^{-1}*A*U */
  PetscScalar	*xmx;	/* store the product X^T*M^{-1}*A*X */
  PetscScalar	*x1; 	/* store the product U^T*x */
  PetscScalar	*x2; 	/* store the product U^T*x */
  PetscScalar 	*Sr2; 	/* Schur vectors at the improvement step */
  PetscScalar	*auau; 	/* product of (M*A*U)^T*M*A*U */
  PetscScalar	*auu; 	/* product of (M*A*U)^T*U */
  
  PetscScalar 	*work; 	/* work space for LAPACK functions */
  PetscInt		*iwork;	/* work space for LAPACK functions */
  PetscReal		*orth; 	/* Coefficients for the orthogonalization */
  
  PetscInt		improve; /* 0 = do not improve the eigenvalues; This is an experimental option */
  
} KSP_DGMRES;

PetscLogEvent KSP_DGMRESComputeDeflationData, KSP_DGMRESApplyDeflation;
#define HH(a,b)  (dgmres->hh_origin + (b)*(dgmres->max_k+2)+(a))
#define HES(a,b) (dgmres->hes_origin + (b)*(dgmres->max_k+1)+(a))
#define CC(a)    (dgmres->cc_origin + (a))
#define SS(a)    (dgmres->ss_origin + (a))
#define GRS(a)   (dgmres->rs_origin + (a))

/* vector names */
#define VEC_OFFSET     2
#define VEC_TEMP       dgmres->vecs[0]
#define VEC_TEMP_MATOP dgmres->vecs[1]
#define VEC_VV(i)      dgmres->vecs[VEC_OFFSET+i]

#define EIG_OFFSET			2
#define DGMRES_DEFAULT_EIG	1
#define DGMRES_DEFAULT_MAXEIG 100

#define	UU		dgmres->U
#define	TT		dgmres->T
#define	TTF		dgmres->TF
#define	XX		dgmres->X
#define	INVP	dgmres->InvP
#define	MU		dgmres->mu
#define	MX		dgmres->mx
#define	UMX		dgmres->umx
#define	XMU		dgmres->xmu
#define	XMX		dgmres->xmx
#define	X1		dgmres->x1
#define	X2		dgmres->x2
#define	SR		dgmres->Sr
#define	SR2		dgmres->Sr2
#define AUAU	dgmres->auau
#define AUU		dgmres->auu
#define	MAX_K	dgmres->max_k
#define	MAX_NEIG dgmres->max_neig
#define WORK	dgmres->work
#define	IWORK	dgmres->iwork
#define	ORTH	dgmres->orth
#define SMV 1
#endif
