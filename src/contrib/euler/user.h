/* $Id: mat.h,v 1.143 1997/09/26 02:22:17 bsmith Exp $ */

/* Include file for 3D Euler application code */

#if !defined(__APPCTX)
#define __APPCTX

/* PETSc and standard include files */
#include "snes.h"
#include "da.h"
#include "dfvec.h"
#include "draw.h"
#include <math.h>

/* Other application include files */
#include "mm.h"

/*
   IMPLICIT ==> fully implicit treatment of boundary conditions
   EXPLICIT ==> explicit treatment of boundary conditions
   This version of code supports only IMPLICIT BCs.
 */
typedef enum {EXPLICIT=0, IMPLICIT_SIZE=1, IMPLICIT=2} BCType;

/* Are we advancing CFL? */
typedef enum {CONSTANT=0, ADVANCE=1} CFLAdvanceType;

/* Options for scaling the system for pseudo-transient 
   continuation; we use DT_DIV. */
typedef enum {DT_MULT=0, DT_DIV=1} ScaleType;

/* Are we using global or local timestepping? Global is the default. */
typedef enum {LOCAL_TS=0, GLOBAL_TS=1} TimeStepType;

/* Application data structure for 3D Euler code */
typedef struct {
  /* ----------------- Basic data structures ------------------- */

    DA       da;                 /* distributed array for X, F */ 
    DA       da1;                /* distributed array for pressure */ 
    Mat      J;                  /* Jacobian (preconditioner) matrix */
    MatFDColoring fdcoloring;    /* coloring context for FD Jacobian approx (optional) */
    Mat      Jmf;                /* matrix-free Jacobian context */
    KSP      ksp;                /* Krylov context */
    Vec      X, Xbc, F;          /* global solution, residual vectors */
    Vec      P, Pbc, localP;     /* pressure work vectors */
    Vec      localX, localDX;    /* local solution, dx vectors */
    Vec      localXBC;           /* local BC vector */
    Scalar   *xx, *dxx, *xx_bc;  /* corresponding local arrays */
    int      fort_ao;            /* Fortran pointer to AO context */

  /* ----------------- General parameters and flags ------------------- */

    TimeStepType ts_type;        /* type of timestepping */
    int    first_time_resid;     /* flag: 1: first time computing residual */
    int    bc_test;              /* flag: 1: test boundary condition scatters */
    int    no_output;            /* flag: 1: no runtime output (use when timing code) */
    int    no_wake;              /* flag: 1: no wake BCs in Jacobian */
    int    use_vecsetvalues;     /* flag: 1: use of VecSetValues() */
    int    post_process;         /* flag: 1: do post-processing */
    int    global_grid;          /* flag: 1: store global grid instead of just local part */
    int    iter;                 /* nonlinear iteration number */
    int    sles_tot;             /* total linear solve iterations */
    int    fct_tot;              /* total function evaluations */
    Scalar fstagnate_ratio;      /* stagnation detection */
    Scalar ksp_rtol_max;         /* maximum KSP relative tolerance */
    Scalar ksp_rtol_min;         /* minimum KSP relative tolerance */
    int    ksp_max_it;           /* maximum KSP iterations per linear solve */
    int    bcswitch;             /* iteration for activating BC switch for impermeability */

  /* ------------- Control of computing Jacobian (preconditioner matrix) ------------ */

    int    mat_assemble_direct;  /* flag - indicates assembling matrix directly */
    int    jfreq;		 /* frequency of computing Jacobian */
    Scalar jratio;		 /* ratio indicating when to form a new preconditioning Jacobian */
    int    use_jratio;           /* flag - are we using the jratio test? */
    Scalar fnorm_last_jac;       /* || F || - last iteration Jac precond was formed */
    int    iter_last_jac;        /* iteration at which last Jac precond was formed */
    Scalar eps_jac;              /* differencing parameter for FD Jacobian approx */
    Scalar eps_jac_inv;          /* 1.0/eps_jac_inv */

  /* ------------- Control of computing Jacobian (matrix-free approx) ------------ */

    int    matrix_free;          /* flag - using matrix-free method */
    int    matrix_free_mult;     /* flag - currently in the midst of a matrix-free mult */
    Scalar eps_mf_default;       /* default differencing parameter for FD mat-vec product approx */
    Scalar fnorm_init, fnorm_last;  /* || F || - initial and last iterations */

  /* ----------------- CFL advancement control ------------------- */

    CFLAdvanceType cfl_advance;             /* flag - indicates type of CFL advancement */
    Scalar   cfl, cfl_init, cfl_max;        /* CFL parameters */
    Scalar   cfl_switch;                    /* CFL at which to dump binary linear system */
    Scalar   cfl_begin_advancement;         /* flag - 1 indicates CFL advancement has begun */
    Scalar   f_reduction;                   /* reduce fnorm by this much before advancing CFL */
    Scalar   cfl_max_incr, cfl_max_decr;    /* maximum increase/decrease for CFL number */
    int      cfl_snes_it;                   /* number of SNES iterations at each CFL step */

  /* ----------------- Output: visualization and debugging  ------------------- */

    Vec    Fvrml;                /* work vector form dumping residual to VRML */
    int    print_vecs;           /* flag: 1: print vectors */
    int    print_grid;           /* flag: 1: print grid info */
    int    print_debug;          /* flag: 1: print debug info */
    int    dump_general;         /* flag: 1: dump fields for later viewing */
    int    dump_vrml;            /* flag: 1: dump fields directly into VRML format */
    int    dump_freq;            /* flag: 1: dump fields every X iterations */
    int    dump_vrml_pressure;   /* flag: 1: dump pressure field directly into VRML format */
    int    dump_vrml_residual;   /* flag: 1: dump residual directly into VRML format */
    int    check_solution;       /* flag: 1: check solution components size */


  /* ----------------- Parallel information ------------------- */

    MPI_Comm   comm;               /* communicator for entire problem */
    int        wing;               /* flag: 1: this processor owns part of the wing */
    int        fort_wing_comm;     /* communicator of procs that own the wing (Fortran pointer) */
    VecScatter Xbcscatter;         /* scatter context for vector BCs */
    VecScatter Pbcscatter;         /* scatter context for pressure BCs */
    int        rank;               /* my processor number */
    int        size;               /* number of processors */
    int        ldim;               /* local dimension */
    int        lbkdim;             /* block local dimension */
    int        gdim;               /* global dimension */
    int        Nx, Ny, Nz;         /* number of procs in x-, y-, and z-directions */
    int        mx, my, mz;         /* global vector dimensions */
    int        nloc;               /* number of ghosted local grid points */
    int        *ltog;              /* local-to-global mapping */

  /* ----------------- Problem-specific parameters and flags ------------------- */

    int       problem;             /* test problem number */
    int       ni, nj, nk;          /* sizes of the grid */
    int       ni1, nj1, nk1;	   /* ni-1, nj-1, nk-1 */
    int       nim, njm, nkm;	   /* ni+1, nj+1, nk+1 */
    int       itl, itu, ile, ktip; /* wing parameters (i:lower, upper, leading edge, k:tip) */
    int       ndof;		   /* DOF per node */
    int       nd;		   /* number of diagonals for interior of grid
                                      (are more for C-grid j=0 boundary condition) */
    char      **label;             /* labels for components */
    Scalar    angle;               /* flow parameter - angle of attack */
    BCType    bctype;              /* flag - boundary condition type */
    ScaleType sctype;              /* flag - type of scaling */

  /* ----------------- Local grid information ------------------- */
    int    xs, ys, zs, xe, ye, ze;        /* local starting/ending grid points */
    int    xsi, ysi, zsi, xei, yei, zei;  /* local starting/ending grid points (interior) */
    int    gxs, gys, gzs, gxe, gye, gze;  /* local starting/ending ghost points */
    int    gxsi, gysi, gzsi, gxei, gyei, gzei; /* local starting/ending ghost points (interior) */
    int    xm, ym, zm, gxm, gym, gzm;     /* local grid/ghost widths */
    int    xsf, ysf, zsf;                 /* Fortran starting grid points */
    int    xefm1, yefm1, zefm1;           /* Fortran ending grid points */
    int    gxsf, gysf, gzsf;              /* Fortran starting ghost points */
    int    xef, yef, zef;                 /* Fortran ending points */
    int    xef01, yef01, zef01;           /* Fortran ending points ni,nj,nk */
    int    gxef, gyef, gzef;              /* ending ghost points */
    int    gxef01, gyef01, gzef01;        /* ending ghost points ni,nj,nk */
    int    xefp1, yefp1, zefp1;           /* Fortran ending points ni1,nj1,nk1 */
    int    gxefp1, gyefp1, gzefp1;        /* ending ghost points ni1,nj1,nk1 */
    int    xsf1, ysf1, zsf1;              /* Fortran starting points 1,1,1 */
    int    gxsf1, gysf1, gzsf1;           /* Fortran starting points 1,1,1 */
    int    xsf2, ysf2, zsf2;              /* Fortran starting points 2,2,2 */
    int    gxsf2, gysf2, gzsf2;           /* Fortran starting points 2,2,2 */
    int    gxsfw, gysfw, gzsfw;           /* Fortran starting ghost points + 1 */
    int    gxefw, gyefw, gzefw;           /* Fortran ending ghost points - 1 */
    int    gxmfp1, gymfp1, gzmfp1;        /* Julianne ghost width */
    int    xmfp1, ymfp1, zmfp1;           /* Julianne width */
    int    *is1;                          /* mapping from application to PETSc ordering */


   /* --------------------------------- Output data -------------------------- */

    Scalar time_init;                       /* initial time */
    Scalar *farray;                         /* array for use with SNESSetConvergenceHistory() */
    Scalar *favg;                           /* array of average fnorm for the past 10 iterations */
    FILE   *fp;                             /* file for stashing convergence info */
    Scalar *flog, *ftime, *fcfl, *lin_rtol; /* the convergence info */
    Scalar *c_lift, *c_drag;
    int    *lin_its, *nsup, last_its;
    int    event_pack, event_unpack;        /* events for performance monitoring */
    int    event_localf, event_monitor;


  /* ------------------- Fortran work arrays ------------------- */
    Scalar *dt;                              /* timestepping */
    Scalar *p;                               /* pressure */
    Scalar *r_bc, *ru_bc, *rv_bc;            /* boundary conditions */
    Scalar *rw_bc, *e_bc, *p_bc;
    Scalar *br, *bl, *be;                    /* eigen[vectors,values] */
    Scalar *sadai, *sadaj, *sadak;           /* mesh metrics */
    Scalar *aix, *ajx, *akx;
    Scalar *aiy, *ajy, *aky;
    Scalar *aiz, *ajz, *akz;
    Scalar *b1, *b2, *b3, *b4, *b5, *b6;     /* matrix elements (off-diagonal blocks) */
    Scalar *diag;                            /* diagonal block */
    int    diag_len;                         /* length of work array to store diagonal block */
    Scalar *b1bc, *b2bc, *b3bc, *b2bc_tmp;   /* matrix elements for implicit bcs */
    Vec    vcoord;
    Scalar *xc, *yc, *zc;                    /* mesh geometry */
    Scalar *work_p;                          /* misc work space for Fortran */
    Scalar *f1, *g1, *h1;
    Scalar *sp, *sm, *sp1, *sm1, *sp2, *sm2;
    int    refine;
    int    dim2, nktot;                      /* 2-dimensional problem variant */

   /* --------------------------------------------------------
       multi-model data
      -------------------------------------------------------- */

    MM     multimodel;
    MMType mmtype;
    Vec    den, xvel, yvel, zvel;             /* full potential work space */
    Scalar *den_a, *xvel_a, *yvel_a, *zvel_a; /* full potential work space */
    Scalar phi_te[2];                         /* 2-component circulation */
    VecScatter phi_te_scatter;
    } Euler;

/* Fortran routine declarations, needed for portablilty */
#ifdef HAVE_FORTRAN_CAPS
#define eigenv_              EIGENV
#define localfortfct_euler_  LOCALFORTFCT_EULER
#define localfortfct_fp_     LOCALFORTFCT_FP
#define resid_               RESID
#define residbc_             RESIDBC
#define bc_uni_              BC_UNI
#define rbuild_              RBUILD
#define rbuild_direct_euler_ RBUILD_DIRECT_EULER
#define rbuild_direct_fp_    RBUILD_DIRECT_FP
#define jmonitor_            JMONITOR
#define jform_               JFORM
#define jform2_              JFORM2
#define jformdt_             JFORMDT
#define jformdt2_            JFORMDT2
#define buildmat_            BUILDMAT
#define nzmat_               NZMAT
#define pvar_                PVAR
#define printvec_            PRINTVEC
#define julianne_            JULIANNE
#define printjul_            PRINTJUL
#define printgjul_           PRINTGJUL
#define printbjul_           PRINTBJUL
#define parsetup_            PARSETUP
#define jpressure_           JPRESSURE
#define bc_                  BC
#define bcpart_j1_           BCPART_J1
#define readmesh_            READMESH
#define wingsurface_         WINGSURFACE

#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define eigenv_              eigenv
#define localfortfct_euler_ localfortfct_euler
#define localfortfct_fp_    localfortfct_fp
#define resid_               resid
#define bc_uni_              bc_uni
#define residbc_             residbc
#define rbuild_              rbuild
#define rbuild_direct_euler_ rbuild_direct_euler
#define rbuild_direct_fp_    rbuild_direct_fp
#define jmonitor_            jmonitor
#define jform_               jform
#define jform2_              jform2
#define jformdt_             jformdt
#define jformdt2_            jformdt2
#define buildmat_            buildmat
#define nzmat_               nzmat
#define pvar_                pvar
#define julianne_            julianne
#define printjul_            printjul
#define printgjul_           printgjul
#define printbjul_           printbjul
#define parsetup_            parsetup
#define jpressure_           jpressure
#define bc_                  bc
#define bcpart_j1_           bcpart_j1
#define readmesh_            readmesh
#define wingsurface_         wingsurface
#endif

/* Basic routines */
int UserCreateEuler(MPI_Comm,int,int,Euler**);
int UserDestroyEuler(Euler*);
int InitialGuess(SNES,Euler*,Vec);
int ComputeFunctionBasic(SNES,Vec,Vec,void*);
int ComputeFunctionNoWake(SNES,Vec,Vec,void*);
int ComputeFunctionCore(int,SNES,Vec,Vec,void*);
int ComputeJacobianFDColoring(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int ComputeJacobianFDBasic(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int UserSetJacobian(SNES,Euler*);
int UserMatrixFreeMatCreate(SNES,Euler*,Vec,Mat*);
int UserMatrixFreeMatDestroy(Mat);
int UserSetMatrixFreeParameters(SNES,double,double);
int UserSetGridParameters(Euler*);
int UserSetGrid(Euler*);
int GetWingCommunicator(Euler*,int*,int*);
int BoundaryConditionsImplicit(Euler*,Vec);
int BoundaryConditionsExplicit(Euler*,Vec);
int BCScatterSetUp(Euler*);

/* Utility routines */
int UnpackWork(Euler*,DA,Scalar*,Vec,Vec);
int UnpackWorkComponent(Euler*,Scalar*,Vec);
int PackWork(Euler*,DA,Vec,Vec,Scalar**);
int MatViewDFVec_MPIAIJ(Mat,DFVec,Viewer);
int MatViewDFVec_MPIBAIJ(Mat,DFVec,Viewer);
int GridTest(Euler*);

/* Monitoring routines */
int CheckSolution(Euler*,Vec);
int MonitorEuler(SNES,int,double,void*);
int ConvergenceTestEuler(SNES,double,double,double,void*);
int MonitorDumpGeneral(SNES,Vec,Euler*);
int MonitorDumpVRML(SNES,Vec,Vec,Euler*);
int ComputeNodalResiduals(Euler*,Vec,Vec);
int DumpField(Euler*,Draw,Scalar*);
void dump_angle_vrml(float);
void MonitorDumpIter(int);

/* Fortran routines */
extern int readmesh_(int*,int*,int*,int*,Scalar*,Scalar*,Scalar*);
extern int printvec_(double*,int*,FILE*);
extern int printjul_(double*,double*,int*);
extern int printgjul_(double*,double*,int*);
extern int printbjul_(double*,double*,int*);
extern int wingsurface_(int*);
extern int jmonitor_(Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,int*);
extern int jpressure_(Scalar*,Scalar*);
extern void eigenv_(Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,TimeStepType*);
extern int  localfortfct_euler_(int*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*);
extern int  residbc_(Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*);
extern int julianne_(Scalar*,int*,int*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,int*);
extern int jformdt2_(Scalar*,Scalar*,int*,int*,int*,int*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,int*);
extern int jformdt_(Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,int*);
extern int jform_(Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*);
extern int jform2_(Scalar*,Scalar*,int*,int*,int*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*);
extern int resid_(Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*);
extern int rbuild_direct_fp_(Scalar*,ScaleType*,Scalar*,Scalar*);
extern int rbuild_direct_euler_(Scalar*,ScaleType*,Scalar*,Scalar*);
extern int rbuild_(int*,ScaleType*,Scalar*,Scalar*,Scalar*,int*,int*);
extern int bc_(Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*);
extern int bcpart_j1_(Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*);
extern int parsetup_(int*,int*,int*,int*,BCType*,int*,int*,int*,int*,int*,int*,int*,
                      int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,
                      int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,
                      int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,
                      int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,MMType*);
extern int buildmat_(int*,ScaleType*,int*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,int*,int*,
                      Scalar*,Scalar*,Scalar*,Scalar*,int*);
extern int nzmat_(MatType*,MMType*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*);
extern int  pvar_(Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,int*,
                      Scalar*,Scalar*,int*,int*);

extern int  localfortfct_fp_(int*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,
                      Scalar*,Scalar*,Scalar*,Scalar*,Scalar*,Scalar*);

/* Fortran interface definitions */

#ifdef HAVE_64BITS
extern void *MPIR_ToPointer();
extern int MPIR_FromPointer();
extern void MPIR_RmPointer();
#else
#define MPIR_ToPointer(a) (a)
#define MPIR_FromPointer(a) (int)(a)
#define MPIR_RmPointer(a)
#endif

#endif
