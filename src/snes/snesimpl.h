/* $Id: nlctx.h,v 1.1 1995/03/20 00:11:30 bsmith Exp $ */

/* -------------------------------------------------------------- */
/*        Context variable used by the SNES solvers.              */
/*        See nlcreate.c for details.                             */
/*        SNES - Scalable Nonlinear Equations Solvers             */
/*        SUMS - Scalable Unconstrained Minimization Solvers      */
/*            (component of SNES for unconstrained minimization)  */
/* -------------------------------------------------------------- */

#ifndef __NLCONTEXT
#define __NLCONTEXT
#include "tools.h"
#include "vectors/vector.h"
#include "solvers/svctx.h"
#include <stdio.h>

/*
   Known SNES methods.  Note that methods for solving systems of
   nonlinear equations have the the form NLE_XXX, while those for
   solving unconstrained minimization problems have the form NLM_XXX.  
 */
typedef enum { NLE_NLS1,
               NLE_NTR1,
               NLE_NTR2_DOG,
               NLE_NTR2_LIN,
               NLE_NBASIC,
               NLM_NLS1,
               NLM_NTR1 }
  NLMETHOD;

typedef enum { NLINITIAL_GUESS,
	       NLRESIDUAL,
	       NLFUNCTION,
	       NLGRADIENT,
	       NLMATRIX,
	       NLCORE,
	       NLSTEP_SETUP,
	       NLSTEP_COMPUTE,
	       NLSTEP_DESTROY,
	       NLTOTAL }
   NLPHASE;

typedef enum { NLE, NLM } NLTYPE;

/* 
   Define the NLSpMat context, which provides an interface to the 
   SpMat matrix context.  This context replaces the outdated user
   context NLSpMatCtx. 
 */
   typedef struct {
   /* ------------- Data used for all SpMat storage types ------------ */
      int    nrows;         /* Number of rows of the matrix (global) */
      int    ncols;         /* Number of columns of the matrix (global) */
      SPTYPE type;          /* Type of SpMat matrix storage.  At present, 
                               the known matrix types are given in 
                               "sparse/sppriv.h" as:
					#define MATROW      1
					#define MATSPLIT    2 (obselete)
					#define MATAIJ      3
					#define MATBLOCK    4
					#define MATROWDIST  5
					#define MATDIAG     6
					#define MATDENSE    7 
					#define MATPDERIVED 8
					#define MATBDIAG    9 */
      int    save_spmat;    /* If (save_spmat), then save SpMat context */

   /* --- Data used only for MATAIJ, MATDENSE, and MATBDIAG storage --- */
      int    user_alloc;    /* If (user_alloc), then the user allocates
			       memory for Jacobian (or Hessian) matrix */

   /* --------------- Data used only for MATROW storage -------------- */
      int    est_row_elem;  /* Estimated number of nonzero elements in each 
			       row of the matrix */

   /* --------------- Data used only for MATAIJ storage -------------- */
      int    max_nonzeros;  /* Maximum number of nonzeros in the matrix,
			       or allocated size of ja and a when using
			       SpAIJCreateFromData() */
      /* Needed only when user controls memory allocation with MATAIJ */
      int    *ia;	    /* Row pointers */
      int    *ja;	    /* Column indices */
      double *a;	    /* matrix entries */

   /* ------------- Data used only for MATDENSE storage -------------- */
      int    nrowsd;        /* Number of declared rows of the matrix */
      double *dense_data;   /* Location of dense matrix data; needed only
			       when the user controls memory allocation */

   /* -------- Data used only for MATDIAG and MATBDIAG storage ------- */
      int    ndiags;	    /* Number of diagonal blocks */

   /* -------------- Data used only for MATBDIAG storage ------------- */
      int    nb;	    /* Size of diagonal blocks */
      int    *diag;	    /* Block diagonal number, (row-col)/nb */
      double **diagv;	    /* Pointer to the actual diagonals */
      void   *pc;	    /* optional parallel context */
      int    print_mat;     /* flag to turn on/off printing */
   } NLSpMat;

/*
   Define the NLSles context, which provides an interface to the
   SVCtx solver context.
 */
   typedef struct {

  /* --- NOTE:  This structure is under construction; it will change --- */

      SVMETHOD svmethod;   /* SLES method (preconditioner or direct solver) */
      ITMETHOD accel;      /* accelerator */
      int      save_svctx; /* If (save_svctx), then save SVctx context */
      void     *pc;	   /* optional parallel context */
      int      damping;	   /* flag to turn on/off damping */
   } NLSles;

/* Basic step context */
typedef struct {

  /* --- NOTE:  This structure is under construction; it will change --- */

  /* ----------- Primary phases of the step computation ---------- */

  void   *stepctx;            /* Local context for the step routines */
  int    step_setup_called;   /* Equals 1 if step_setup has been called */
  int    step_compute_called; /* Equals 1 if step_compute has been called */
  void   (*step_setup)();     /* Routine for setting up a step */
  int    (*step_compute)();   /* Routine for computing a step */
  void   (*step_destroy)();   /* Routine for destroying a step */ 

  /* ----------- Data structures and related information ---------- */

  void       *Mctx;           /* Jacobian (or Hessian) matrix context */
  NLSpMat    nlsp;	      /* SMEIT (SpMat) interface */
  int        is_spmat;        /* Equals 1 if Mctx is the SpMat context; 
				 equals 0 otherwise */
  void       *svctx;          /* Linear solver context */
  NLSles     nlsv;	      /* SLES (SVCtx) interface */
  int        is_sles;         /* Equals 1 if svctx is the SVCtx context; 
				 equals 0 otherwise */
  void       *D;              /* Scaling matrix context - allocated by default 
			         in NLENewtonTR2SetUp to be double precision 
				 diagonal matrix if not set by user */
  int        D_alloc;         /* Equals 1 if space for D is allocated
				 internally; equals 0 otherwise */

  /* --------- Routines used within certain step computations -------- */

  void   (*FormMatrix)();     /* Evaluate matrix (Jacobian or Hessian) */
  void   (*NormM)();          /* Compute matrix norm */
  void   (*Mv)();             /* Compute matrix-vector product */
  void   (*MTv)();            /* Compute matrix transpose-vector product */
  void   (*ScaleM)();         /* Form scaling matrix */
  void   (*Fsolve)();	      /* Perform forward triangular solve */
  void   (*Bsolve)();	      /* Perform backward triangular solve */

} NLBasicStepCtx;

/* NLMonBase - Basic performance statistics that are used for cumulative 
   info on various phases of the solution process */
typedef struct {
  int	  ncalls;		/* number of calls */
  int	  nflops;		/* number of flops */
  double  tcpu;			/* CPU time */
  double  telp;			/* elapsed time */
} NLMonBase;

/* NLMonCore - Performance statistics that are used for cumulative
   information on the core SNES routines of the solution process.
   The data exclude all non-core phases, namely the following, which
   are monitored separately:
      o step-set-up, step-compute, step-destroy
      o initial guess, matrix evaluation
      o residual evaluation (NLE only) 
      o function and gradient evaluation (NLM only)
 */
typedef struct {
  int	    ncalls;		/* number of calls */
  int       nflops;		/* number of flops */
  double    tcpu;      		/* CPU time */
  double    telp;      		/* elapsed time */
  int       nmatops;		/* number of applications of matrix-vector 
			  	   product AND preconditioner */
  int	    namults;		/* number of applications of matrix-vector 
                                   product not counted in nmatops */
  int	    natmults;		/* number of applications of matrix^T-vector 
                                   product */
  int       nbinvs;		/* number of applications of preconditioner
                                   not counted in nmatops */
  int       nvectors;		/* number of vector operations */
  int       nscalars;		/* number of scalar operations */
  int       nflops_1prec;	/* flops for 1 precond application */
  int       nflops_1mv;		/* flops for 1 mat-vec mult */
} NLMonCore;

/* Performance monitoring context */
typedef struct {
  int       niter_inner;	/* cumulative number of inner iterations */
  int       nunsuc_inner;	/* cumulative number of unsuccessful inner 
				   iterations */
  int       nunsuc;		/* number of unsuccessful steps */
  int	    memory;		/* amount of memory used (bytes) */
  double    start_tcpu;		/* starting time for NLSolve call */
  double    start_telp;
  double    wktotal_tcpu;	/* working storage for total time */
  double    wktotal_telp; 
  NLMonCore core;		/* core of SNES solvers */
  NLMonBase init;		/* initial guess routine */
  NLMonBase resid;		/* residual evaluations (NLE only) */
  NLMonBase func;		/* function evaluations (NLM only) */
  NLMonBase grad;		/* gradient evaluations (NLM only) */
  NLMonBase matrix;		/* matrix evals */
  NLMonBase step_setup;		/* step-set-up phase */
  NLMonBase step_compute;	/* step-compute phase */
  NLMonBase step_destroy;	/* step-destroy phase */
  NLMonBase total;		/* total solution process */
} NLMonCtx;

/*
   Nonlinear solver context
 */
typedef struct {
  int cookie;			/* cookie to test for context validity */

  /* ------------------------- Basic Contexts -------------------------- */

  VECntx         *vc;		/* Vector information */
  NLBasicStepCtx sc;		/* Information regarding the inner step
				   (matrix formation, linear solvers, etc.) */
  NLMonCtx       mon;		/* performance monitoring information */
  void           *user;		/* User context */
  void	         *pc;		/* Parallel context (used only in parallel 
				   version ... See pnlctx.h )*/
  int            is_par;	/* is_par = 1 if version is parallel;
				   is_par = 0 otherwise . */

  /* ---------------- Contexts for user-defined routines --------------- */
  void           *monP;		/* optional context for user-defined 
				   monitoring routine */
  void           *cnvP;		/* optional context for user-defined 
				   convergence tester */

  /* --- Routines and data that are unique to each particular solver --- */

  NLMETHOD method;             /* Name of solver */
  NLTYPE   method_type;	       /* Type of solver, where
				   NLE = system of nonlinear equations
				   NLM = unconstrained minimization, also
                                         known as the SUMS component */
  int      setup_called;       /* setup_called = 1 if setup has been called;
				  setup_called = 0 otherwise. */
  int      solver_called;      /* solver_called = 1 if solver has been called;
				  solver_called = 0 otherwise. */
  void     (*setup)();         /* Sets up the nonlinear solver */
  int      (*solver)();        /* Actual nonlinear solver */
  void     (*destroy)();       /* Destroys the nonlinear context */
  void     *MethodPrivate;     /* Holder for misc. data associated 
                                  with a particular method */

  /* ------------------ User (or default) Parameters ------------------ */

  int      max_its;            /* Max number of iterations */
  int      max_resids;         /* Max number of residual evals (NLE only) */
  int      max_funcs;          /* Max number of function evals (NLM only) */
  int      iter;               /* Global iteration number */
  int      conv_info;          /* Integer indicating type of termination as
				  set by convergence monitoring routine */
  double   norm;               /* Residual norm of current iterate (NLE)
				  or gradient norm of current iterate (NLM) */
  double   rtol_lin;           /* Relative tolerance parameter for evaluating
                                  convergence of iterative linear solvers */
  double   rtol;               /* Relative tolerance */
  double   atol;               /* Absolute tolerance */
  double   xtol;               /* Relative tolerance in solution */
  double   trunctol;           /* Minimum tolerance for linear solvers */
  double   fmin;               /* Minimum tolerance for function (NLM only) */
  double   deltatol;           /* Trust region convergence tolerance */

  /* ---------------------- User (or default) Routines ------------------ */

  void     (*initial_guess)(); /* Calculates an initial guess */
  void     (*resid)();         /* Evaluates residual (NLE only) */
  int      resid_neg;          /* resid_neg = 1 if resid evaluates -f;
				  resid_neg = 0 otherwise */
  void     (*func)();          /* Evaluates function (NLM only) */
  void     (*grad)();          /* Evaluates gradient (NLM only) */
  void     (*usr_monitor)();   /* Returns control to user after residual
                                  calculation and allows user to, for 
                                  instance, print residual norm, etc. */
  int      (*converged)();     /* Monitors convergence */
  char     *(*term_type)();    /* Determines type of termination */
  void     (*set_param)();     /* Sets solver parameters */
  double   (*get_param)();     /* Gets solver parameters */

  /* ------------------- Default work-area management ------------------ */

  int      nwork;              
  void     **work;

  /* -------------------- Miscellaneous Information --------------------- */

  void     *vec_sol;           /* Pointer to solution */
  void     *vec_rg;            /* Pointer to residual (NLE) or 
				  gradient (NLM) */
  void     *resid_work;	       /* Work space for forming negative of
				  residual if necessary (NLE only) */
  double   fc;		       /* Function value (NLM only) */
  int      s_alloc;            /* Equals 1 if space for solution is allocated
				  internally; equals 0 otherwise */
  int      rg_alloc;           /* Equals 1 if space for residual (NLE)
				  (NLE) or gradient (NLM) is allocated 
				  internally; equals 0 otherwise */
  double   *conv_hist;         /* If !0, stores residual norm (NLE) or
				  gradient norm (NLM) at each iteration */
  int      conv_hist_len;      /* Amount of convergence history space */
  FILE     *fp;                /* File for trace information */

} NLCtx;

/* -------------------------------------------------------------------- */
/* Below are routines that are used throughout all of the SNES package. */
/* See snes/nlefunc.h for routines used to solve systems of nonlinear   */
/* only (NLE).  Likewise, see sums/nlmfunc.h for routines used to solve */
/* unconstrained minimization problems only (NLM).                      */
/* -------------------------------------------------------------------- */

#ifdef ANSI_ARG
#undef ANSI_ARG
#endif
#ifdef __STDC__
#define ANSI_ARGS(a) a
#else
#define ANSI_ARGS(a) ()
#endif

/* ----- Routines to initialize, run, and destroy the solver ----- */
extern NLCtx    *NLCreate 		ANSI_ARGS((NLMETHOD, void *));
extern void     NLSetMatrixRoutine 	ANSI_ARGS((NLCtx*, 
					void (*)(NLCtx *, void *) ));
extern void     NLCreateDVectors 	ANSI_ARGS((NLCtx*, int ));
extern void     NLCreateDBVectors 	ANSI_ARGS((NLCtx*, int ));
extern void     NLSetUp 		ANSI_ARGS((NLCtx *));
extern int      NLSolve 		ANSI_ARGS((NLCtx *));
extern void     NLDestroy 		ANSI_ARGS((NLCtx *));
extern void	NLFormMatrix 		ANSI_ARGS((NLCtx*, void *));

/* ----- Routines to set solver parameters ----- */
extern void     NLSetMaxIterations 	ANSI_ARGS((NLCtx*, int ));
extern void     NLSetRelConvergenceTol 	ANSI_ARGS((NLCtx*, double ));
extern void     NLSetAbsConvergenceTol 	ANSI_ARGS((NLCtx*, double ));
extern void     NLSetTruncationTol 	ANSI_ARGS((NLCtx*, double ));
extern void     NLSetSolutionTol 	ANSI_ARGS((NLCtx*, double ));
extern void     NLSetTrustRegionTol 	ANSI_ARGS((NLCtx*, double ));
extern void     NLSetRelativeLinearTol 	ANSI_ARGS((NLCtx*, double ));
extern void     NLSetParameter 		ANSI_ARGS((NLCtx*, char*, double *));

/* ----- Routines to set various aspects of solver ----- */
extern void     NLSetSolution 		ANSI_ARGS((NLCtx*, void *));
extern void     NLSetUserCtx 		ANSI_ARGS((NLCtx*, void *));
extern void     NLSetMatrixCtx 		ANSI_ARGS((NLCtx*, void *, int));
extern void     NLSetLinearSolverCtx 	ANSI_ARGS((NLCtx*, void *, int));
extern void     NLSetVectorCtx 		ANSI_ARGS((NLCtx*, VECntx *));
extern void     NLSetDVectors 		ANSI_ARGS((NLCtx*, int ));
extern void     NLSetDBVectors 		ANSI_ARGS((NLCtx*, int ));
extern void     NLSetStepSetUp 		ANSI_ARGS((NLCtx*, 
					void (*)(NLCtx*, void *) ));
extern void     NLSetStepCompute 	ANSI_ARGS((NLCtx*, 
					int (*)(NLCtx*, void*, void*, void*, 
					double*, double*, double*, double*, 
					double*, void *) ));
extern void     NLSetStepDestroy 	ANSI_ARGS((NLCtx*, 
					void (*)(NLCtx *) ));
extern void     NLSetInitialGuessRoutine ANSI_ARGS((NLCtx*, 
					void (*)(NLCtx *, void*) ));

/* ----- Routines to set performance monitoring options ----- */
extern void     NLCommandLineInterface	ANSI_ARGS((NLCtx*, int*, 
					char**, FILE *));
extern void     NLSetLog 		ANSI_ARGS((NLCtx*, FILE *));
extern void     NLSetFlopsZero 		ANSI_ARGS((NLCtx *));
extern void     NLSetFlops 		ANSI_ARGS((NLCtx*, NLPHASE, int ));
extern void     NLSaveConvergenceHistory ANSI_ARGS((NLCtx *));

/* Note:  The monitoring and convergence testing routines have different 
          calling sequences for the NLE and NLM components */
extern void     NLSetConvergenceTest();
extern void     NLSetMonitor(); 

/* ----- Routines to extract operation counts ----- */
extern void     NLClearCoreWorkCounts	ANSI_ARGS((NLCtx *));
extern void     NLGetCoreWorkCounts	ANSI_ARGS((NLCtx*, int*, int*, int*, 
					int*, int*, int *));
extern int      NLGetNumberOfCalls 	ANSI_ARGS((NLCtx*, NLPHASE));
extern int      NLGetFlops 		ANSI_ARGS((NLCtx*, NLPHASE ));
extern int      NLGetNumberUnsuccessfulSteps ANSI_ARGS((NLCtx *));

/* ----- Routines to extract miscellaneous information ----- */
extern void     NLGetTime 		ANSI_ARGS((NLCtx*, NLPHASE, 
					double*, double *));
extern void     *NLGetSolution 		ANSI_ARGS((NLCtx *));
extern void     *NLGetUserCtx		ANSI_ARGS((NLCtx *));
extern double   NLGetParameter 		ANSI_ARGS((NLCtx*, char *));
extern int      NLGetIterationNumber 	ANSI_ARGS((NLCtx *));
extern int      NLGetMemory		ANSI_ARGS((NLCtx *));
extern int      NLGetMethodFromCtx	ANSI_ARGS((NLCtx *));
extern int      NLGetMethodType		ANSI_ARGS((NLCtx *));
extern int      NLGetVectorDimension	ANSI_ARGS((NLCtx *));
extern char     *NLGetMethodName	ANSI_ARGS((NLMETHOD ));
extern char     *NLGetTerminationType	ANSI_ARGS((NLCtx *));
extern void     NLGetMethod		ANSI_ARGS((int*, char**, int, 
					char*, NLMETHOD* ));
extern void     *NLGetLinearSolverCtx	ANSI_ARGS((NLCtx *));
extern void     *NLGetMatrixCtx		ANSI_ARGS((NLCtx *));
extern VECntx   *NLGetVectorCtx		ANSI_ARGS((NLCtx *));
extern void     NLOutput		ANSI_ARGS((NLCtx*, int*, 
					char**, FILE *));
extern void     NLSimpleOutput		ANSI_ARGS((NLCtx*, FILE *));
extern void     NLGetStatistics		ANSI_ARGS((NLCtx*, FILE *));
extern double   *NLGetConvergenceHistory ANSI_ARGS((NLCtx *));
extern int      NLGetNumberInnerIterations ANSI_ARGS((NLCtx *));
extern int      NLGetNumberUnsuccessfulInnerIterations ANSI_ARGS((NLCtx *));

/* ----- Miscellaneous routines ----- */
extern NLMETHOD NLIsMethodAvailable	ANSI_ARGS((char *));
extern int      NLIsContextValid	ANSI_ARGS((NLCtx *));
extern void     NLSetDefaults		ANSI_ARGS((NLCtx*, void *));
extern void     NLRegister		ANSI_ARGS((int, char*, void (*)()));
extern void     NLRegisterAll		ANSI_ARGS((void));
extern void     NLRegisterDestroy	ANSI_ARGS((void));
extern int      NLHelpMessage		ANSI_ARGS((int *, char **));
extern void     NLScaleStep		ANSI_ARGS((NLCtx*, double*, double*, 
					double*, double*, double *));

/* ---- Old routine names for SNES ---- */
int   NLGetFlopsStepSetUps();
int   NLGetFlopsStepComputations();
int   NLGetFlopsStepDestructions();
int   NLGetFlopsTotalSolution();
int   NLGetFlopsCoreSolution();

void  NLGetTimeMatrixEvaluations();
void  NLGetTimeInitialGuess();
void  NLGetTimeStepSetUps();
void  NLGetTimeStepComputations();
void  NLGetTimeStepDestructions();
void  NLGetTimeTotalSolution();

int   NLGetNumberStepSetUps();
int   NLGetNumberStepComputations();
int   NLGetNumberStepDestructions();
int   NLGetNumberMatrixEvaluations();

#define NLGetFlopsStepSetUps( ctx ) \
	NLGetFlops( ctx, NLSTEP_SETUP )
#define NLGetFlopsStepComputations( ctx ) \
	NLGetFlops( ctx, NLSTEP_COMPUTE )
#define NLGetFlopsStepDestructions( ctx ) \
	NLGetFlops( ctx, NLSTEP_DESTROY )
#define NLGetFlopsTotalSolution( ctx ) \
	NLGetFlops( ctx, NLTOTAL )
#define NLGetFlopsCoreSolution( ctx ) \
	NLGetFlops( ctx, NLCORE )

#define NLGetTimeMatrixEvaluations( ctx, te, tc ) \
	NLGetTime( ctx, NLMATRIX, te, tc )
#define NLGetTimeInitialGuess( ctx, te, tc ) \
	NLGetTime( ctx, NLINITIAL_GUESS, te, tc )
#define NLGetTimeStepSetUps( ctx, te, tc ) \
	NLGetTime( ctx, NLSTEP_SETUP, te, tc )
#define NLGetTimeStepComputations( ctx, te, tc ) \
	NLGetTime( ctx, NLSTEP_COMPUTE, te, tc )
#define NLGetTimeStepDestructions( ctx, te, tc ) \
	NLGetTime( ctx, NLSTEP_DESTROY, te, tc )
#define NLGetTimeTotalSolution( ctx, te, tc ) \
	NLGetTime( ctx, NLTOTAL, te, tc )

#define NLGetNumberStepSetUps( ctx ) \
	NLGetNumberOfCalls( ctx, NLSTEP_SETUP )
#define NLGetNumberStepComputations( ctx ) \
	NLGetNumberOfCalls( ctx, NLSTEP_COMPUTE )
#define NLGetNumberStepDestructions( ctx ) \
	NLGetNumberOfCalls( ctx, NLSTEP_DESTROY )
#define NLGetNumberMatrixEvaluations( ctx ) \
	NLGetNumberOfCalls( ctx, NLMATRIX )

#endif
