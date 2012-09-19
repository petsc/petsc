#include <petscsys.h>
#include <petsc-private/fortranimpl.h>

#define max_colors  200
#define max_nbtran 20

#define REAL double

typedef struct gxy{                            /* GRID STRUCTURE             */
   int nnodes;                                 /* Number of nodes            */
   int ncell;                                  /* Number of cells            */
   int nedge;                                  /* Number of edges            */
   int ncolor;                                 /* Number of colors           */
   int nccolor;                                /* Number of colors for cells */
   int nncolor;                                /* Number of colors for nodes */
   int ncount[max_colors];                     /* No. of faces in color      */
   int nccount[max_colors];                    /* No. of cells in color      */
   int nncount[max_colors];                    /* No. of nodes in color      */
   int iup;                                    /* if 1, upward int coef reqd */
   int idown;                                  /* if 1, dwnwrd int coef reqd */

   int nsface;                                 /* Total # of solid faces     */
   int nvface;                                 /* Total # of viscous faces   */
   int nfface;                                 /* Total # of far field faces */
   int nsnode;                                 /* Total # of solid nodes     */
   int nvnode;                                 /* Total # of viscous nodes   */
   int nfnode;                                 /* Total # of far field nodes */
   int jvisc;                                  /* 0 = Euler                  */
                                               /* 1 = laminar no visc LHS    */
                                               /* 2 = laminar w/ visc LHS    */
                                               /* 3 = turb B-B no visc LHS   */
                                               /* 4 = turb B-B w/ visc LHS   */
                                               /* 5 = turb Splrt no visc LHS */
                                               /* 6 = turb Splrt w/ visc LHS */
   int ileast;                                 /* 1 = Lst square gradient    */
   int nsets;                                  /* No of levels for scheduling*/
   int *eptr;                                  /* edge pointers              */
   int *isface;                                /* Face # of solid faces      */
   int *ifface;                                /* Face # of far field faces  */
   int *ivface;                                /* Face # of viscous faces    */
   int *isford;                                /* Copies of isface, ifface,  */
   int *ifford;                                /*  and ivface used for       */
   int *ivford;                                /*  ordering                  */
   int *isnode;                                /* Node # of solid nodes      */
   int *ivnode;                                /* Node # of viscous nodes    */
   int *ifnode;                                /* Node # of far field nodes  */
   int *nflag;                                 /* Node flag                  */
   int *nnext;                                 /* Next node                  */
   int *nneigh;                                /* Neighbor of a node         */
   int *c2n;                                   /* Cell-to-node pointers      */
   int *c2e;                                   /* Cell-to-edge pointers      */
   int *c2c;                                   /* Cell-to-cell pointers      */
   int *ctag;                                  /* Cell tags                  */
   int *csearch;                               /* Cell search list           */
   int *cenc;                                  /* Enclosing cell for node    */
   int *clist;                                 /* Colored list of cells      */
   int *iupdate;                               /* Tells whether to update    */
   int *sface;                                 /* Nodes for solid faces      */
   int *vface;                                 /* Nodes for viscous faces    */
   int *fface;                                 /* Nodes for far field faces  */
   int *icount;                                /* # of surrounding nodes     */
   int *isetc;                                /* Nodes in each level        */
   int *iset;                                 /* Actual nodes for levels    */
/* Forward substitution for ILU */
   int *nlcol;                                /* No of edge colors for sets */
   int *nlcount;                              /* How many edges in each colr*/
   int *lvface;                               /* Edges that influence a set */
/* Back substitution for ILU */
   int *nbcol;                                /* No of edge colors for sets */
   int *nbcount;                              /* How many edges in each colr*/
   int *lbface;                               /* Edges that influence a set */
   REAL *x, *y, *z;                           /* Node Coordinates           */
   REAL *area;                                /* Area of control volumes    */
   /*REAL *gradx, *grady, *gradz;*/           /* Gradients                  */
   REAL *cdt;                                 /* Local time step            */
   REAL *qcp, *rcp;                           /* Two work arrays            */
   REAL *ff;                                  /* MG forcing function        */
   REAL *dfp, *dfm;                           /* Flux Jacobians             */
   REAL *dft1, *dft2;                         /* Turb mod linearization     */
   REAL *slen;                                /* Generalized distance       */
   REAL *turbre;                              /* nu x turb Reynolds #       */
   REAL *amut;                                /* Turbulent mu (viscosity)   */
   REAL *turbres;                             /* Turbulent residual         */
   REAL *turbff;                              /* Turbulent forcing function */
   REAL *turbold;                             /* Turbulent unknown (for MG) */
   REAL *sxn, *syn, *szn, *sa;                /* Normals at solid nodes     */
   REAL *vxn, *vyn, *vzn, *va;                /* Normals at viscous nodes   */
   REAL *fxn, *fyn, *fzn, *fa;                /* Normals at far field nodes */
   REAL *xn, *yn, *zn, *rl;                   /* Normal to faces and length */
   REAL *us, *vs, *ws, *as;                   /* For linearizing viscous    */
   REAL *phi;                                 /* Flux limiter               */
   REAL *rxy;                                 /*                            */

   int  *icoefup;                             /* Surrounding nodes          */
   REAL *rcoefup;                             /* Coefficients               */
   int  *icoefdn;                             /* Surrounding nodes          */
   REAL *rcoefdn;                             /* Coefficients               */
   REAL *AP;                                  /* Array for GMRES            */
   REAL *Fgm;                                 /* Big array for GMRES        */
   REAL *Xgm;                                 /* Another GMRES array        */
   REAL *temr;                                /* Temporary array            */
   REAL *ALU;                                 /* Big array for ILU(0)       */
   int  *ia, *iau, *ja, *fhelp;               /* Stuff for ILU(0)           */

/*
 * stuff to read in daves grid file
 */
   int nnbound,nvbound,nfbound,nnfacet,nvfacet,nffacet,ntte;
   int *ncolorn,*countn,*ncolorv,*countv,*ncolorf,*countf;
   int *nntet,*nnpts,*nvtet,*nvpts,*nftet,*nfpts;
   int *f2ntn,*f2ntv,*f2ntf;

/* PETSc datastructures and other related info */
   Vec qnode;                                 /* Global distributed solution
						 vector*/
   Vec qnodeLoc;                              /* Local sequential solution
						 vector*/
   Vec dq;                                    /* Delta Q                    */
   Vec qold;                                  /* Global distributed solution
						 vector*/
   Vec  res;                                  /* Residual                   */
   Vec  grad;              		      /* Gradient Vector	    */
   Vec  gradLoc;                  	      /* Local Gradient Vector	    */
   Vec B;                                     /* Right hand side            */
   Mat A;                                     /* Left hand side             */
   VecScatter scatter, gradScatter;           /* Scatter between local
                                                 and global vectors         */
   int *loc2pet;                              /* local to PETSc mapping     */
   int *loc2glo;                              /* local to global mapping     */
   int *v2p;				      /* Vertex to processor mapping */
   AO   ao;
   int *sface_bit, *vface_bit;
   int nnodesLoc, nedgeLoc, nvertices;
   int nsnodeLoc, nvnodeLoc, nfnodeLoc;
   int nnfacetLoc, nvfacetLoc, nffacetLoc;

   /* global arrays needed for post processing */
   /*int *indGlo, *isnodeGlo, *ivnodeGlo, *f2ntnGlo, *f2ntvGlo;
   REAL *xGlo, *yGlo, *zGlo;
   Vec qGlo;
   VecScatter scatterGlo;*/




}GRID;                                         /* Grids                      */
                                               /*============================*/

                                               /*============================*/
typedef struct{                               /* GENERAL INFORMATION        */
   REAL title[20];                            /* Title line                 */
   REAL xmach;                                /* Mach # in X-direction      */
   REAL alpha;                                /* Angle of attack            */
   REAL yaw;                                  /* Yaw Angle                  */
   REAL Re;                                   /* Reynolds number            */
   REAL dt;                                   /* Input cfl                  */
   REAL tot;                                  /* total computer time        */
   REAL res0;                                 /* Begining residual          */
   REAL resc;                                 /* Current residual           */
   int ntt;                                   /* A counter                  */
   int mseq;                                  /* Mesh sequencing            */
   int ivisc;                                 /* 0 = Euler                  */
                                              /* 1 = laminar no visc LHS    */
                                              /* 2 = laminar w/ visc LHS    */
                                              /* 3 = turb BB no visc LHS    */
                                              /* 4 = turb BB w/ visc LHS    */
                                              /* 5 = turb SA w/ visc LHS    */
                                              /* 6 = turb SA w/ visc LHS    */
   int irest;                                 /* for restarts irest = 1     */
   int icyc;                                  /* iterations completed       */
   int ihane;                                 /* ihane = 0 for van leer fds */
                                              /*       = 1 for hanel flux   */
                                              /*       = 2 for Roe's fds    */
   int ntturb;                                /* Counter for turbulence     */
}CINFO;                                       /* COMMON INFO                */
                                              /*============================*/

                                              /*============================*/
typedef struct {                              /* FLOW SOLVER VARIABLES      */
   REAL cfl1;                                 /* Starting CFL number        */
   REAL cfl2;                                 /* Ending   CFL number        */
   int nsmoth;                                /* How many its for Res smooth*/
   int iflim;                                 /* 1=use limiter 0=no limiter */
   int itran;                                 /* 1=transition (spalart only)*/
   int nbtran;                                /* No. of transition points   */
   int jupdate;                               /* For freezing Jacobians */
   int nstage;                                /* Number of subiterations    */
   int ncyct;                                 /* Subiterations for turb mod */
   int iramp;                                 /* Ramp CFL over iramp iters  */
   int nitfo;                                 /* Iterations first order     */
   int ncyc;                                  /* Number of iterations to run*/
}CRUNGE;                                      /* COMMON RUNGE               */
                                               /*============================*/

typedef struct {                               /*============================*/
   REAL sref;                                  /* Reference area             */
   REAL cref;                                  /* Reference chord            */
   REAL bref;                                  /* Reference span (semi-span?)*/
   REAL xmc;                                   /* x-location for moments     */
   REAL ymc;                                   /* y-location for moments     */
   REAL zmc;                                   /* z-location for moments     */
}CREFGEOM;                                     /*============================*/
                                              /*============================*/

typedef struct{                               /*============================*/
   REAL  gtol;                                /* linear system tolerence    */
   int   icycle;                              /* Number of GMRES iterations */
   int   nsrch;                               /* Dimension of Krylov        */
   int   ilu0;                                /* 1 for ILU(0)               */
   int   ifcn;                                /* 0=fcn2 1=fcneval(nwt Krlv) */
}CGMCOM;                                      /* COMMON GMCOM               */
                                              /*============================*/
int set_up_grid(GRID *);
int write_fine_grid(GRID *);

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#       define f77name(ucase,lcase,lcbar) lcbar
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#       define f77name(ucase,lcase,lcbar) ucase
#else
#       define f77name(ucase,lcase,lcbar) lcase
#endif
#define f77INFO     f77name(INFO,info,info_)
#define f77REFGEOM  f77name(REFGEOM,refgeom,refgeom_)
#define f77RUNGE    f77name(RUNGE,runge,runge_)
#define f77GMCOM    f77name(GMCOM,gmcom,gmcom_)
#define f77FORLINK  f77name(FORLINK,forlink,forlink_)
#define f77OPENM    f77name(OPENM,openm,openm_)
#define f77READR1   f77name(READR1,readr1,readr1_)
#define f77READR2   f77name(READR2,readr2,readr2_)
#define f77READR3   f77name(READR3,readr3,readr3_)
#define f77RDGPAR   f77name(RDGPAR,rdgpar,rdgpar_)
#define f77README   f77name(README,readme,readme_)
#define f77COLORCJ  f77name(COLORCJ,colorcj,colorcj_)
#define f77COLORCGS f77name(COLORCGS,colorcgs,colorcgs_)
#define f77BNDORD   f77name(BNDORD,bndord,bndord_)
#define f77FINDIN   f77name(FINDIN,findin,findin_)
#define f77ELMORD   f77name(ELMORD,elmord,elmord_)
#define f77BNSHFT   f77name(BNSHFT,bnshft,bnshft_)
#define f77VNSHFT   f77name(VNSHFT,vnshft,vnshft_)
#define f77NSHIFT   f77name(NSHIFT,nshift,nshift_)
#define f77NEIGHBR  f77name(NEIGHBR,neighbr,neighbr_)
#define f77NSTACK   f77name(NSTACK,nstack,nstack_)
#define f77GTCPTR   f77name(GTCPTR,gtcptr,gtcptr_)
#define f77GTENCC   f77name(GTENCC,gtencc,gtencc_)
#define f77INCOEF   f77name(INCOEF,incoef,incoef_)
#define f77INTERP1  f77name(INTERP1,interp1,interp1_)
#define f77INTERP4  f77name(INTERP4,interp4,interp4_)
#define f77RCOLL1   f77name(RCOLL1,rcoll1,rcoll1_)
#define f77RCOLL    f77name(RCOLL,rcoll,rcoll_)
#define f77INIT     f77name(INIT,init,init_)
#define f77SUMGS    f77name(SUMGS,sumgs,sumgs_)
#define f77GETAREA  f77name(GETAREA,getarea,getarea_)
#define f77INFOTRN  f77name(INFOTRN,infotrn,infotrn_)
#define f77SCLOCK   f77name(SCLOCK,sclock,sclock_)
#define f77GETRES   f77name(GETRES,getres,getres_)
#define f77L2NORM   f77name(L2NORM,l2norm,l2norm_)
#define f77FORCE    f77name(FORCE,force,force_)
#define f77UPDATE   f77name(UPDATE,update,update_)
#define f77WREST    f77name(WREST,wrest,wrest_)
#define f77RREST    f77name(RREST,rrest,rrest_)
#define f77PLLAN    f77name(PLLAN,pllan,pllan_)
#define f77FLLAN    f77name(FLLAN,fllan,fllan_)
#define f77TECFLO   f77name(TECFLO,tecflo,tecflo_)
#define f77FASFLO   f77name(FASFLO,fasflo,fasflo_)
#define f77BC       f77name(BC,bc,bc_)
#define f77CLINK    f77name(CLINK,clink,clink_)
#define f77SLENGTH  f77name(SLENGTH,slength,slength_)
#define f77GETNDEX  f77name(GETNDEX,getndex,getndex_)
#define f77CHANGEV  f77name(CHANGEV,changev,changev_)
#define f77CHANGEP  f77name(CHANGEP,changep,changep_)
#define f77TURBER   f77name(TURBER,turber,turber_)
#define f77TURBRES  f77name(TURBRES,turbres,turbres_)
#define f77SPALART  f77name(SPALART,spalart,spalart_)
#define f77SPALRES  f77name(SPALRES,spalres,spalres_)
#define f77PLOTURB  f77name(PLOTURB,ploturb,ploturb_)
#define f77GETSKIN  f77name(GETSKIN,getskin,getskin_)
#define f77GETC2N   f77name(GETC2N,getc2n,getc2n_)
#define f77VWEIGHT  f77name(VWEIGHT,vweight,vweight_)
#define f77PLOTCP   f77name(PLOTCP,plotcp,plotcp_)
#define f77CORRSM   f77name(CORRSM,corrsm,corrsm_)
#define f77CORRSM1  f77name(CORRSM1,corrsm1,corrsm1_)

#define f77GETIA    f77name(GETIA,getia,getia_)
#define f77GETJA    f77name(GETJA,getja,getja_)
#define f77SORTER   f77name(SORTER,sorter,sorter_)
#define f77BLKILU   f77name(BLKILU,blkilu,blkilu_)
#define f77BLKSOL   f77name(BLKSOL,blksol,blksol_)
#define f77GETLEVEL f77name(GETLEVEL,getlevel,getlevel_)
#define f77LVCOLOR  f77name(LVCOLOR,lvcolor,lvcolor_)
#define f77LBCOLOR  f77name(LBCOLOR,lbcolor,lbcolor_)

/* Added by D. K. Kaushik 1/10/97) */
#define f77FILLA    f77name(FILLA,filla,filla_)
#define f77LSTGS    f77name(LSTGS,lstgs,lstgs_)
#define f77IREAD    f77name(IREAD,iread,iread_)
#define f77RREAD    f77name(RREAD,rread,rread_)

EXTERN_C_BEGIN
extern void PETSC_STDCALL f77FORLINK(void);
extern void PETSC_STDCALL f77OPENM(int*);
extern void PETSC_STDCALL f77FILLA(int* nnodesLoc,int* nedgeLoc,int* eptr,int*nsface,
				   int* isface,PetscScalar* fxn,PetscScalar* fyn,PetscScalar* fzn,
				   PetscScalar* sxn,PetscScalar* syn,PetscScalar* szn,int* nsnodeLoc,int* nvnodeLoc,
				   int* nfnodeLoc,int* isnode,int* ivnode,int* ifnode,
				   PetscScalar* qnode,Mat* jac,PetscScalar* cdt,PetscScalar* rl,PetscScalar* area,PetscScalar* xn,PetscScalar*,PetscScalar* zn,
             PetscScalar* cfl,int* rank,int* nvertices);
extern void PETSC_STDCALL f77READR1(int*,int*);
extern void PETSC_STDCALL f77SUMGS(int*,int*,int*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,int*,int*);
extern void PETSC_STDCALL f77INIT(int*,PetscScalar*,PetscScalar*,PetscScalar*,int*,int*,int*);
extern void PETSC_STDCALL f77LSTGS(int*,int*,int*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,int*,int*);
extern void PETSC_STDCALL f77GETRES(int* nnodesLoc,int* ncell,int *nedgeLoc,int* nsface,
                                    int* nvface,int* nfface,int* nbface,
                                    int* nsnodeLoc,int* nvnodeLoc,int* nfnodeLoc,
				    int* isface,int* ivface,int* ifface,int* ileast,
				    int* isnode,int* ivnode,int* ifnode,
				    int* nnfacetLoc,int* f2ntn,int* nnbound,
				    int* nvfacetLoc,int* f2ntv,int* nvbound,
				    int* nffacetLoc,int* f2ntf,int* nfbound,int* eptr,
				    PetscScalar* sxn,PetscScalar* syn,PetscScalar* szn,
				    PetscScalar* vxn,PetscScalar* vyn,PetscScalar* vzn,
				    PetscScalar* fxn,PetscScalar* fyn,PetscScalar* fzn,
				    PetscScalar* xn,PetscScalar*,PetscScalar* zn,PetscScalar* rl,
				    PetscScalar* qnode,PetscScalar* cdt,PetscScalar* x,PetscScalar* y,PetscScalar* z,PetscScalar* area,
				    PetscScalar* grad,PetscScalar* res,PetscScalar* turbre,PetscScalar* slen,int* c2n,int* c2e,
				    PetscScalar* us,PetscScalar* vs,PetscScalar* as,PetscScalar* phi,PetscScalar* amut,
				    int* ires,int* rank,int* nvertices);
extern void PETSC_STDCALL f77FORCE(int* nnodesLoc,int* nedgeLoc,
            int* isnode,int* ivnode,
            int* nnfacetLoc,int* f2ntn,int* nnbound,
            int* nvfacetLoc,int* f2ntv,int* nvbound,
            int* eptr,PetscScalar* qnode,PetscScalar* x,PetscScalar* y,PetscScalar* z,
            int* nvnodeLoc,int* c2n,int* ncell,
	    PetscScalar* amut,int* sface_bit,int* vface_bit,
            PetscScalar* clift,PetscScalar* cdrag,PetscScalar* cmom,int* rank,int* nvertices);
extern void PETSC_STDCALL f77GETIA(int*,int*,int*,int*,int*,int*);
extern void PETSC_STDCALL f77GETJA(int*,int*,int*,int*,int*,int*,int*);
extern void PETSC_STDCALL f77TECFLO(int* nnodes,int* nvbound,int* nvfacet,int* nvnode,
             PetscScalar* x,PetscScalar* y,PetscScalar* z,
             PetscScalar* qnode, int* nvpts, int* nvtet,
             int* f2ntv, int* ivnode,
             int* timeStep, int* rank, int* openFile, int* closeFile,
             int* boundaryType,PetscScalar* title);
EXTERN_C_END
