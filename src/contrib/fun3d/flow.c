/* "$Id: flow.c,v 1.14 2000/01/23 18:37:05 bsmith Exp kaushik $";*/

static char help[] = "FUN3D - 3-D, Unstructured Incompressible Euler Solver\n\
originally written by W. K. Anderson of NASA Langley, \n\
and ported into PETSc framework by D. K. Kaushik, ODU and ICASE.\n\n";

#include <assert.h>
#include "snes.h"
#include "draw.h"
#include "ao.h"
#include "is.h"
#include "user.h"

/* These are hacks to get Fun3d to compile with version 2.0.24 and the master copy of PETSc */

#if PETSC_VERSION_SUBMINOR >= 26  
#define PETSCTRUTH PetscTruth
#else
#define PETSCTRUTH int
#define MatSetLocalToGlobalMappingBlock MatSetLocalToGlobalMappingBlocked
#endif
 
typedef struct {
 Vec     qnew, qold, func;
 double  fnorm_ini, dt_ini, cfl_ini;
 double  ptime;
 double  cfl_max, max_time;
 double  fnorm, dt, cfl;
 double  fnorm_ratio;
 int     ires, iramp;
 int     max_steps, print_freq;
} TstepCtx;
 
typedef struct {
                                               /*============================*/
 GRID     *grid;                               /* Pointer to Grid info       */
 TstepCtx *tsCtx;                              /* Pointer to Time Stepping 
                                                * Context                    */
                                               /*============================*/
} AppCtx;

int  FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction(SNES,Vec,Vec,void*),
     FormInitialGuess(SNES, GRID *),
     Update(SNES, void*),
     ComputeTimeStep(SNES, int, void*),
     GetLocalOrdering(GRID *),
     SetPetscDS(GRID *, TstepCtx *);

/* Global Variables */ 

                                               /*============================*/
CINFO  *c_info;                                /* Pointer to COMMON INFO     */
CRUNGE *c_runge;                               /* Pointer to COMMON RUNGE    */
CGMCOM *c_gmcom;                               /* Pointer to COMMON GMCOM    */
                                               /*============================*/
int rank, CommSize, rstart;
REAL memSize = 0.0, grad_time = 0.0;

#if defined(PARCH_IRIX64) && defined(USE_HW_COUNTERS)
int event0, event1;
Scalar time_counters;
long long counter0, counter1;
#endif
int  ntran[max_nbtran];        /* transition stuff put here to make global */
REAL dxtran[max_nbtran];

 
/* ======================== MAIN ROUTINE =================================== */
/*                                                                           */
/* Finite volume flux split solver for general polygons                      */
/*                                                                           */
/*===========================================================================*/

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  AppCtx 	user;
  GRID 		f_pntr;
  TstepCtx      tsCtx ;
  SNES          snes;                  /* SNES context */
  Mat           Jpc;                   /* Jacobian and Preconditioner matrices */
  Scalar        *qnode;
  int 		ierr,solIt;
  PETSCTRUTH    flg;
  
  ierr = PetscInitialize(&argc,&args,"testgrid/petsc.opt",help);CHKERRA(ierr);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&CommSize);

  /*PetscPrintf(MPI_COMM_WORLD, " Program name is %s\n",OptionsGetProgramName());*/
  /*ierr =*/ PetscInitializeFortran();CHKERRA(ierr);
  
  /*======================================================================*/
  /* Initilize stuff related to time stepping */
  /*======================================================================*/
  tsCtx.fnorm_ini = 0.0; tsCtx.cfl_ini = 50.0; tsCtx.cfl_max = 1.0e+05;
  tsCtx.max_steps = 50;  tsCtx.max_time = 1.0e+12; tsCtx.iramp = -50;
  tsCtx.dt = -5.0; tsCtx.fnorm_ratio = 1.0e+10;
  ierr = OptionsGetInt(PETSC_NULL,"-max_st",&tsCtx.max_steps,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-ts_rtol",&tsCtx.fnorm_ratio,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-cfl_ini",&tsCtx.cfl_ini,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-cfl_max",&tsCtx.cfl_max,PETSC_NULL);CHKERRA(ierr);
  /*======================================================================*/

  f77FORLINK();                               /* Link FORTRAN and C COMMONS */
 
  f77OPENM(&rank);                            /* Open files for I/O         */

  /*  Read input */

  f77READR1(&ileast, &rank);
  f_pntr.jvisc   = c_info->ivisc;
  f_pntr.ileast  = ileast;
  c_gmcom->ilu0  = 1;
  c_gmcom->nsrch = 10;   
 
  c_runge->nitfo = 0;

/* Read & process the grid */
  /*  f77RDGPAR(&f_pntr.nnodes,  &f_pntr.ncell,   &f_pntr.nedge,
             &f_pntr.nccolor, &f_pntr.ncolor,
             &f_pntr.nnbound, &f_pntr.nvbound, &f_pntr.nfbound,
             &f_pntr.nnfacet, &f_pntr.nvfacet, &f_pntr.nffacet,
             &f_pntr.nsnode,  &f_pntr.nvnode,  &f_pntr.nfnode,
             &f_pntr.ntte,
             &f_pntr.nsface,  &f_pntr.nvface,  &f_pntr.nfface,
             &rank);*/

 
/* Read the grid information */

   /*f77README(&f_pntr.nnodes,  &f_pntr.ncell,   &f_pntr.nedge,
             &f_pntr.ncolor,  &f_pntr.nccolor,
             &f_pntr.nnbound, &f_pntr.nvbound, &f_pntr.nfbound,
             &f_pntr.nnfacet, &f_pntr.nvfacet, &f_pntr.nffacet,
             &f_pntr.nsnode,  &f_pntr.nvnode,  &f_pntr.nfnode,
             &f_pntr.ntte,
              f_pntr.eptr,
              f_pntr.x,        f_pntr.y,        f_pntr.z,
              f_pntr.area,     f_pntr.c2n,      f_pntr.c2e,
              f_pntr.xn,       f_pntr.yn,       f_pntr.zn,
              f_pntr.rl,
              f_pntr.nntet,    f_pntr.nnpts,    f_pntr.nvtet,
              f_pntr.nvpts,    f_pntr.nftet,    f_pntr.nfpts,
              f_pntr.f2ntn,    f_pntr.f2ntv,    f_pntr.f2ntf,
              f_pntr.isnode,   f_pntr.sxn,      f_pntr.syn,
              f_pntr.szn,      f_pntr.ivnode,   f_pntr.vxn,
              f_pntr.vyn,      f_pntr.vzn,      f_pntr.ifnode,
              f_pntr.fxn,      f_pntr.fyn,      f_pntr.fzn,
              f_pntr.slen,
             &rank);*/

/* Get the grid information into local ordering */
 
   ierr = GetLocalOrdering(&f_pntr);CHKERRA(ierr);

/* Allocate Memory for Some Other Grid Arrays */ 
   ierr = set_up_grid(&f_pntr);CHKERRA(ierr);
 
/* Now set up PETSc datastructure */
   /* ierr = SetPetscDS(&f_pntr, &tsCtx);CHKERRA(ierr);*/
 
/* If using least squares for the gradients, calculate the r's */
   if (f_pntr.ileast == 4) {
         f77SUMGS(&f_pntr.nnodesLoc, &f_pntr.nedgeLoc, f_pntr.eptr,
                  f_pntr.xyz,
                  f_pntr.rxy,
                  &rank, &f_pntr.nvertices);
      }
 
   /*write_fine_grid(&f_pntr);*/
 
   user.grid  = &f_pntr;
   user.tsCtx = &tsCtx;


   /* 
     Perform the nonlinear solver twice to eliminate the
   effects of paging in the executable on the first fun of 
   the nonlinear solver 
   */

 for (solIt = 0; solIt < 2; solIt++) {
  PLogStagePush(solIt);
  /* Create nonlinear solver */
  ierr = SetPetscDS(&f_pntr, &tsCtx);CHKERRA(ierr);
  ierr = SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);
  ierr = SNESSetType(snes,"ls");CHKERRA(ierr);
 
  /* Set various routines and options */
  ierr = SNESSetFunction(snes,user.grid->res,FormFunction,&user);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-matrix_free",&flg);CHKERRA(ierr);
  if (flg) {
    /* Use matrix-free Jacobian to define Newton system; use explicit (approx)
       Jacobian for preconditioner */
     /*ierr = SNESDefaultMatrixFreeMatCreate(snes,user.grid->qnode,&Jpc);*/
     ierr = MatCreateSNESMF(snes,user.grid->qnode,&Jpc);CHKERRA(ierr);
     ierr = SNESSetJacobian(snes,Jpc,user.grid->A,FormJacobian,&user);CHKERRA(ierr);
     /*ierr = SNESSetJacobian(snes,Jpc,user.grid->A,0,&user);*/
  } else {
    /* Use explicit (approx) Jacobian to define Newton system and
       preconditioner */
    ierr = SNESSetJacobian(snes,user.grid->A,user.grid->A,FormJacobian,&user);CHKERRA(ierr);
  }
 
  ierr = SNESSetFromOptions(snes);CHKERRA(ierr);

 
  /* Force ILU(0) preconditioning to be used.  Note that this overrides 
     whatever choices may have been specified in the options database. 
     Also factorization would be done in place */
  /*ierr = SNESGetSLES(snes,&sles);CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCILU);CHKERRA(ierr);
  ierr = PCILUSetUseInPlace(pc);CHKERRA(ierr);*/
 
/* Initialize the flowfield */
  ierr = FormInitialGuess(snes,user.grid);CHKERRA(ierr);

  /* Solve nonlinear system */

  ierr = Update(snes,&user);CHKERRA(ierr);


/* Write restart file */
   ierr = VecGetArray(user.grid->qnode, &qnode);CHKERRA(ierr);
   /*f77WREST(&user.grid->nnodes,qnode,user.grid->turbre,
             user.grid->amut);*/

/* Write Tecplot solution file */
/*
   if (!rank) 
   f77TECFLO(&user.grid->nnodes,  
             &user.grid->nnbound, &user.grid->nvbound, &user.grid->nfbound,
             &user.grid->nnfacet, &user.grid->nvfacet, &user.grid->nffacet, 
             &user.grid->nsnode,  &user.grid->nvnode,  &user.grid->nfnode,   
              c_info->title,      
              user.grid->x,        user.grid->y,        user.grid->z,
              qnode,
              user.grid->nnpts,    user.grid->nntet,    user.grid->nvpts, 
              user.grid->nvtet,    user.grid->nfpts,    user.grid->nftet,    
              user.grid->f2ntn,    user.grid->f2ntv,    user.grid->f2ntf, 
              user.grid->isnode,   user.grid->ivnode,   user.grid->ifnode,
              &rank); 
*/

   /*f77FASFLO(&user.grid->nnodes, &user.grid->nsnode, &user.grid->nnfacet,
              user.grid->isnode,  user.grid->f2ntn,
              user.grid->x,       user.grid->y,       user.grid->z,
              qnode);*/

/* Write residual, lift, drag, and moment history file */
/*
   if (!rank) 
      f77PLLAN(&user.grid->nnodes, &rank);
*/

   ierr = VecRestoreArray(user.grid->qnode, &qnode);CHKERRA(ierr);
   ierr = OptionsHasName(PETSC_NULL,"-mem_use",&flg);CHKERRA(ierr);
   if (flg) {
    PLogDouble space, maxSpace;
    ierr = PetscTrSpace(&space,0,&maxSpace);
    PetscPrintf(PETSC_COMM_WORLD,"Space allocated before destroying is %g\n",space);
     PetscPrintf(PETSC_COMM_WORLD,"Max space allocated so far is %g\n",maxSpace);
   }

   ierr = VecDestroy(user.grid->qnode);CHKERRA(ierr);
   ierr = VecDestroy(user.grid->qnodeLoc);CHKERRA(ierr);
   ierr = VecDestroy(user.tsCtx->qold);CHKERRA(ierr);
   ierr = VecDestroy(user.tsCtx->func);CHKERRA(ierr);
   ierr = VecDestroy(user.grid->res);CHKERRA(ierr);
   ierr = VecDestroy(user.grid->grad);CHKERRA(ierr);
   ierr = VecDestroy(user.grid->gradLoc);CHKERRA(ierr);
   ierr = MatDestroy(user.grid->A);CHKERRA(ierr);
   ierr = OptionsHasName(PETSC_NULL,"-matrix_free",&flg);CHKERRA(ierr);
   if (flg) { ierr = MatDestroy(Jpc);CHKERRA(ierr);}
   ierr = SNESDestroy(snes);CHKERRA(ierr);
   ierr = VecScatterDestroy(user.grid->scatter);CHKERRA(ierr);
   ierr = VecScatterDestroy(user.grid->gradScatter);CHKERRA(ierr);
   PLogStagePop();
   ierr = OptionsHasName(PETSC_NULL,"-mem_use",&flg);CHKERRA(ierr);
   if (flg) {
    PLogDouble space, maxSpace;
    ierr = PetscTrSpace(&space,0,&maxSpace);CHKERRA(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Space allocated after destroying is %g\n",space);
     PetscPrintf(PETSC_COMM_WORLD,"Max space allocated so far is %g\n",maxSpace);
   }

  }
  /*ierr = PetscGetResidentSetSize((PLogDouble *)&totMem); 
  PetscPrintf(MPI_COMM_WORLD, "Memory used by the process is %g bytes\n", totMem);*/
  PetscPrintf(MPI_COMM_WORLD, "Time taken in gradient calculation is %g sec.\n",grad_time);

  PetscFinalize();
  return 0;
}

/*---------------------------------------------------------------------*/
/* ---------------------  Form initial approximation ----------------- */
#undef __FUNC__
#define __FUNC__ "FormInitialGuess"
int FormInitialGuess(SNES snes, GRID *grid)
/*---------------------------------------------------------------------*/
{
   int    ierr;
   Scalar *qnode;

   PetscFunctionBegin;
   ierr = VecGetArray(grid->qnode,&qnode);CHKERRQ(ierr);

   f77INIT(&grid->nnodesLoc, qnode, grid->turbre,
            grid->amut, &grid->nvnodeLoc, grid->ivnode, &rank);

   ierr = VecRestoreArray(grid->qnode,&qnode);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
 
/*---------------------------------------------------------------------*/
/* ---------------------  Evaluate Function F(x) --------------------- */
#undef __FUNC__
#define __FUNC__ "FormFunction"
int FormFunction(SNES snes,Vec x,Vec f,void *dummy)
/*---------------------------------------------------------------------*/
{
   AppCtx       *user = (AppCtx *) dummy;
   GRID         *grid = user->grid;
   TstepCtx     *tsCtx = user->tsCtx;
   Scalar       *qnode, *res, *qold;
   Scalar       *grad;
   Scalar       temp;
   VecScatter   scatter = grid->scatter;
   VecScatter   gradScatter = grid->gradScatter;
   Vec          localX = grid->qnodeLoc;
   Vec          localGrad = grid->gradLoc;
   int          i, j,in, ierr;
   int          nbface, ires;
   Scalar	time_ini, time_fin;
 
   PetscFunctionBegin;
   ierr = VecScatterBegin(x,localX,INSERT_VALUES,SCATTER_FORWARD,scatter);CHKERRQ(ierr);
   ierr = VecScatterEnd(x,localX,INSERT_VALUES,SCATTER_FORWARD,scatter);CHKERRQ(ierr);
   ires = tsCtx->ires;
   ierr = VecGetArray(f,&res);CHKERRQ(ierr);
   ierr = VecGetArray(grid->grad,&grad);CHKERRQ(ierr);
   ierr = VecGetArray(localX,&qnode);CHKERRQ(ierr);

   ierr = PetscGetTime(&time_ini);CHKERRQ(ierr);
   f77LSTGS(&grid->nnodesLoc,&grid->nedgeLoc,grid->eptr,
             qnode,grad,grid->xyz,
             grid->rxy,
            &rank,&grid->nvertices);
   ierr = PetscGetTime(&time_fin);CHKERRQ(ierr);
   grad_time += time_fin - time_ini;
   ierr = VecRestoreArray(grid->grad,&grad);CHKERRQ(ierr);

   ierr = VecScatterBegin(grid->grad,localGrad,INSERT_VALUES,
                          SCATTER_FORWARD,gradScatter);CHKERRQ(ierr);
   ierr = VecScatterEnd(grid->grad,localGrad,INSERT_VALUES,
                          SCATTER_FORWARD,gradScatter);CHKERRQ(ierr);

   ierr = VecGetArray(localGrad,&grad);CHKERRQ(ierr);
   nbface = grid->nsface + grid->nvface + grid->nfface;
   f77GETRES(&grid->nnodesLoc, &grid->ncell,   &grid->nedgeLoc,   &grid->nsface,
             &grid->nvface, &grid->nfface,  &nbface,
             &grid->nsnodeLoc, &grid->nvnodeLoc,  &grid->nfnodeLoc,
              grid->isface,  grid->ivface,   grid->ifface,  &grid->ileast,
              grid->isnode,  grid->ivnode,   grid->ifnode,
             &grid->nnfacetLoc, grid->f2ntn,   &grid->nnbound,
             &grid->nvfacetLoc, grid->f2ntv,   &grid->nvbound,
             &grid->nffacetLoc, grid->f2ntf,   &grid->nfbound,
              grid->eptr,
              grid->sxn,     grid->syn,      grid->szn,
              grid->vxn,     grid->vyn,      grid->vzn,
              grid->fxn,     grid->fyn,      grid->fzn,
              grid->xyzn,
              qnode,         grid->cdt,
              grid->xyz,     grid->area,
              grad,
              res,           
              grid->turbre,
              grid->slen,    grid->c2n,
              grid->c2e,
              grid->us,      grid->vs,       grid->as,
              grid->phi,
              grid->amut,    &ires, &rank, &grid->nvertices);
/* Add the contribution due to time stepping */
   if (ires == 1) {
    /*ierr = VecGetLocalSize(tsCtx->qold, &vecSize);CHKERRQ(ierr);
    printf("Size of local vector tsCtx->qold is %d\n", vecSize);*/
    ierr = VecGetArray(tsCtx->qold,&qold);CHKERRQ(ierr);
#if defined(INTERLACING)
    for (i = 0; i < grid->nnodesLoc; i++) {
     temp = grid->area[i]/(tsCtx->cfl*grid->cdt[i]);
     for (j = 0; j < 4; j++) {
      in = 4*i + j;
      res[in] += temp*(qnode[in] - qold[in]);
     }
    }
#else
    for (j = 0; j < 4; j++) {
     for (i = 0; i < grid->nnodesLoc; i++) {
      temp = grid->area[i]/(tsCtx->cfl*grid->cdt[i]);
      in = grid->nnodesLoc*j + i;
      res[in] += temp*(qnode[in] - qold[in]);
     }
    }
#endif
    ierr = VecRestoreArray(tsCtx->qold,&qold);CHKERRQ(ierr);
   }

   ierr = VecRestoreArray(localX,&qnode);CHKERRQ(ierr);
   /*ierr = VecRestoreArray(grid->dq,&dq);CHKERRQ(ierr);*/
   ierr = VecRestoreArray(f,&res);CHKERRQ(ierr);
   ierr = VecRestoreArray(localGrad,&grad);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
/* --------------------  Evaluate Jacobian F'(x) -------------------- */
 
#undef __FUNC__
#define __FUNC__ "FormJacobian"
int FormJacobian(SNES snes, Vec x, Mat *Jac, Mat *B,MatStructure *flag, void *dummy)
/*---------------------------------------------------------------------*/
{
   AppCtx       *user = (AppCtx *) dummy;
   GRID         *grid = user->grid;
   TstepCtx     *tsCtx = user->tsCtx;
   Mat          jac = *B;
   VecScatter   scatter = grid->scatter;
   Vec          localX = grid->qnodeLoc;
   Scalar       *xx, *ff, *qnode, *qold;
   int          i, j, in, in1, ierr, n, k, jstart, jend, nbrs;
   int          nnz, nnodes;
   int          mat_dump_flag = 0, mat_dump_freq = tsCtx->max_steps; 
   FILE         *fptr;
 
   PetscFunctionBegin;
   ierr = VecScatterBegin(x,localX,INSERT_VALUES,SCATTER_FORWARD,scatter);CHKERRQ(ierr);
 
   ierr = MatSetUnfactored(jac);CHKERRQ(ierr);
   nnodes = grid->nnodes;
 
   ierr = VecScatterEnd(x,localX,INSERT_VALUES,SCATTER_FORWARD,scatter);CHKERRQ(ierr);
 
   ierr = VecGetArray(localX,&qnode);CHKERRQ(ierr);
   /*ierr = MatZeroEntries(jac);CHKERRQ(ierr);*/

   f77FILLA(&grid->nnodesLoc, &grid->nedgeLoc, grid->eptr,
            &grid->nsface, 
             grid->isface, grid->fxn, grid->fyn, grid->fzn, 
             grid->sxn, grid->syn, grid->szn,
            &grid->nsnodeLoc,&grid->nvnodeLoc,&grid->nfnodeLoc, grid->isnode, 
             grid->ivnode, grid->ifnode, qnode, &jac, grid->cdt, 
             grid->area, grid->xyzn, &tsCtx->cfl, 
            &rank, &grid->nvertices);

   /*ierr = MatView(jac,VIEWER_STDOUT_SELF);CHKERRQ(ierr);*/
   ierr = VecRestoreArray(localX,&qnode);CHKERRQ(ierr);
   *flag = SAME_NONZERO_PATTERN;
   /*{
       Viewer viewer;
       ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,"mat.bin", BINARY_CREATE, &viewer);
       ierr = MatView(jac,viewer);CHKERRQ(ierr);
       ierr = VecView(x,viewer);CHKERRQ(ierr);
       ierr = ViewerDestroy(viewer);
       ierr = MPI_Abort(MPI_COMM_WORLD,1);
     }*/
#if defined (MAT_DUMP)
#if defined(INTERLACING) 
#if defined(BLOCKING)
   /* Write the matrix in compressed row format */
   ierr = OptionsHasName(PETSC_NULL,"-mat_dump",&mat_dump_flag);CHKERRQ(ierr);
   if (mat_dump_flag) {
    if (((c_info->ntt - 1)%mat_dump_freq) == 0) {
     int fd, nbytes,nwrt, itstep = c_info->ntt - 1;
     int *rows, *cols;
     int tmp[10];
     Scalar *aij;
     char str[256];
     PetscPrintf(PETSC_COMM_WORLD,"Writing the matrix at time step %d\n",
                 itstep);
     sprintf(str,"matAij%d.bin", itstep);
     fptr = fopen("matAij.dat","w");
     /*fd = open("matAij.bin", O_WRONLY, 0);*/
     if ((fd = creat(str, 0755)) < 0) 
        SETERRQ(1,1,"Error in opening the file %s");
     printf("fd is %d\n", fd);
     nnz = grid->ia[nnodes] - 1;
     fprintf(fptr,"%d %d %d\n",4*nnodes,4*nnodes,16*nnz);
     tmp[0] = 4*nnodes;
     tmp[1] = 4*nnodes;
     tmp[2] = 16*nnz;
     nwrt = 3*sizeof(int);
     if ((nbytes = write(fd, (char *) tmp, nwrt)) != nwrt)
         SETERRQ(1,1,"Write error on file");
     in = 0;
     for (i = 0; i < nnodes; i++) {
      jend = 4*(grid->ia[i+1]-grid->ia[i]); 
      fprintf(fptr,"%d %d %d %d ", in,in+jend,in+2*jend,in+3*jend);
      if ((i%3) == 0)
       fprintf(fptr,"\n");
      tmp[0] = in;
      tmp[1] = in+jend;
      tmp[2] = in+2*jend;
      tmp[3] = in+3*jend;
      in += 4*jend;
      nwrt = 4*sizeof(int);
      if ((nbytes = write(fd, (char *)tmp, nwrt)) != nwrt)
          SETERRQ(1,1,"Write error on file -- row begin array");
     }
     tmp[0] = in;
     nwrt = sizeof(int);
     if ((nbytes = write(fd, (char *) tmp, nwrt)) != nwrt)
          SETERRQ(1,1,"Write error on file");
     for (i = 0; i < nnodes; i++) {
      jstart = grid->ia[i] - 1;
      jend = grid->ia[i+1] - 1;
      for (j = jstart; j < jend; j++) {
       in = 4*grid->ja[j]; 
       fprintf(fptr,"%d %d %d %d ", in,in+1,in+2,in+3);
       tmp[0] = in;
       tmp[1] = in+1;
       tmp[2] = in+2;
       tmp[3] = in+3;
       nwrt = 4*sizeof(int);
       if ((nbytes = write(fd, (char *)tmp, nwrt)) != nwrt)
           SETERRQ(1,1,"Write error on file -- col index array");
      }
      for (j = jstart; j < jend; j++) {
       in = 4*grid->ja[j]; 
       fprintf(fptr,"%d %d %d %d ", in,in+1,in+2,in+3);
       tmp[0] = in;
       tmp[1] = in+1;
       tmp[2] = in+2;
       tmp[3] = in+3;
       nwrt = 4*sizeof(int);
       if ((nbytes = write(fd, (char *)tmp, nwrt)) != nwrt)
           SETERRQ(1,1,"Write error on file -- col index array");
      }
      for (j = jstart; j < jend; j++) {
       in = 4*grid->ja[j]; 
       fprintf(fptr,"%d %d %d %d ", in,in+1,in+2,in+3);
       tmp[0] = in;
       tmp[1] = in+1;
       tmp[2] = in+2;
       tmp[3] = in+3;
       nwrt = 4*sizeof(int);
       if ((nbytes = write(fd, (char *)tmp, nwrt)) != nwrt)
           SETERRQ(1,1,"Write error on file %s -- col index array");
      }
      for (j = jstart; j < jend; j++) {
       in = 4*grid->ja[j]; 
       fprintf(fptr,"%d %d %d %d ", in,in+1,in+2,in+3);
       tmp[0] = in;
       tmp[1] = in+1;
       tmp[2] = in+2;
       tmp[3] = in+3;
       nwrt = 4*sizeof(int);
       if ((nbytes = write(fd, (char *) tmp, nwrt)) != nwrt)
           SETERRQ(1,1,"Write error on file -- col index array");
      }
      fprintf(fptr,"\n");
     }
     icalloc(4*nnodes,&rows);
     icalloc(4*nnodes,&cols);
     fcalloc(16*nnodes,&aij);
     for (i = 0; i < nnodes; i++) { 
      in = 4*i;
      rows[0] = in; rows[1] = in+1; 
      rows[2] = in+2; rows[3] = in+3; 
      jstart = grid->ia[i] - 1;
      jend = grid->ia[i+1] - 1;
      k = 0;
      for (j = jstart; j < jend; j++) {
       in = 4*grid->ja[j]; 
       cols[k++]=in; cols[k++]=in+1;
       cols[k++]=in+2; cols[k++]=in+3;
      }
      ierr = MatGetValues(jac,4,rows,k,cols,aij);
      nwrt = 4*k*sizeof(REAL);
      if ((nbytes = write(fd, (char *) aij, nwrt)) != nwrt)
          SETERRQ(1,1,"Write error on file -- data array");
      for (in = 0; in < 4; in++) {
       fprintf(fptr,"row %d ....\n",rows[in]);
       for (j = 0; j < k; j++) 
         fprintf(fptr,"%d %g ", cols[j],aij[j+in*k]); 
       fprintf(fptr,"\n");
      }
     }
     fclose(fptr);
     close(fd);
    }
   }
#endif
#endif
#endif
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "Update"
int Update(SNES snes, void *ctx)
/*---------------------------------------------------------------------*/
{
 
 AppCtx 	*user = (AppCtx *) ctx;
 GRID 		*grid = user->grid;
 TstepCtx 	*tsCtx = user->tsCtx;
 VecScatter   	scatter = grid->scatter;
 Vec          	localX = grid->qnodeLoc;
 Scalar 	*qnode, *res, *xx;
 Scalar 	clift, cdrag, cmom;
 int 		i, ierr, its, iter;
 Scalar 	fnorm, dt, dt_ini, fnorm_ini, fratio;
 Scalar 	time1, time2, cpuloc, cpuglo;
 Scalar         cpu_ini, cpu_fin, cpu_time;
 int 		max_steps;
 Scalar	 	max_time;
 int		vecSize;
 PETSCTRUTH     print_flag = PETSC_FALSE, flg;
 FILE 		*fptr;
 static int     PreLoadFlag = 1;
 int		nfailsCum = 0, nfails = 0;
/*int 		event0 = 14, event1 = 25, gen_start, gen_read;
 Scalar		time_start_counters, time_read_counters;
 long long      counter0, counter1;*/

  PetscFunctionBegin;

  ierr = OptionsHasName(PETSC_NULL,"-print", &print_flag);CHKERRQ(ierr);
/*
 if (PreLoadFlag) {
  ierr = VecCopy(grid->qnode, tsCtx->qold);CHKERRQ(ierr);
  ierr = ComputeTimeStep(snes,i,user);CHKERRQ(ierr);
  ierr = SNESSolve(snes,grid->qnode,&its);CHKERRQ(ierr);
  tsCtx->fnorm_ini = 0.0;
  PetscPrintf(MPI_COMM_WORLD, "Preloading done ...\n");
  PreLoadFlag = 0;
 }
*/
 if ((!rank) && (print_flag)) {
    fptr = fopen("history.out", "w");
    fprintf(fptr, "VARIABLES = iter,cfl,fnorm,clift,cdrag,cmom,cpu\n");
 }
 if (PreLoadFlag) 
  max_steps = 1;
 else
  max_steps = tsCtx->max_steps;
 max_time = tsCtx->max_time;
 fratio = 1.0;
 tsCtx->ptime = 0.0;
 /*ierr = VecDuplicate(grid->qnode,&tsCtx->qold);CHKERRQ(ierr);
 ierr = VecGetLocalSize(tsCtx->qold, &vecSize);CHKERRQ(ierr);
 printf("Size of local vector tsCtx->qold is %d\n", vecSize);*/
 ierr = VecCopy(grid->qnode, tsCtx->qold);CHKERRQ(ierr);
 ierr = PetscGetTime(&time1);CHKERRQ(ierr);
#if defined (PARCH_IRIX64) && defined(USE_HW_COUNTERS)
 /*if (!PreLoadFlag) {
  ierr = OptionsGetInt(PETSC_NULL,"-e0",&event0,&flg);CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-e1",&event1,&flg);CHKERRQ(ierr);
  ierr = PetscGetTime(&time_start_counters);CHKERRQ(ierr);
  if ((gen_start = start_counters(event0,event1)) < 0)
   SETERRQ(1,1,"Error in start_counters\n"); 
 }*/
#endif
 /*cpu_ini = PetscGetCPUTime();*/
 for (i = 0; i < max_steps && fratio <= tsCtx->fnorm_ratio; i++) {
  ierr = ComputeTimeStep(snes,i,user);CHKERRQ(ierr);
  /*tsCtx->ptime +=  tsCtx->dt;*/
  ierr = PetscTrValid(0,0,0,0);CHKERRQ(ierr);
  ierr = SNESSolve(snes,grid->qnode,&its);CHKERRQ(ierr);
  ierr=PetscTrValid(0,0,0,0);CHKERRQ(ierr); 
  ierr = SNESGetNumberUnsuccessfulSteps(snes, &nfails);CHKERRQ(ierr);
  nfailsCum += nfails; nfails = 0;
  if (nfailsCum >= 2) 
    SETERRQ(1,1,"Unable to find a Newton Step");
  if (print_flag)
   PetscPrintf(MPI_COMM_WORLD,"At Time Step %d cfl = %g and fnorm = %g\n",i,tsCtx->cfl,tsCtx->fnorm); 
  ierr = VecCopy(grid->qnode,tsCtx->qold);CHKERRQ(ierr);

/* For history file */
  c_info->ntt = i+1;
/* Timing Info */
  ierr = PetscGetTime(&time2);CHKERRQ(ierr);
  /*cpu_fin = PetscGetCPUTime();*/
  cpuloc = time2-time1;            
  cpuglo = 0.0;
  ierr = MPI_Allreduce(&cpuloc,&cpuglo,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);CHKERRQ(ierr);

  /*cpu_fin = PetscGetCPUTime();
  cpuloc = cpu_fin - cpu_ini;            
  cpu_time = 0.0;
  ierr = MPI_Allreduce(&cpuloc,&cpu_time,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);  */
  c_info->tot = cpuglo;    /* Total CPU time used upto this time step */
  

  ierr = VecScatterBegin(grid->qnode,localX,INSERT_VALUES,SCATTER_FORWARD,scatter);CHKERRQ(ierr);
  ierr = VecScatterEnd(grid->qnode,localX,INSERT_VALUES,SCATTER_FORWARD,scatter);CHKERRQ(ierr);

  ierr = VecGetArray(grid->res, &res);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&qnode);CHKERRQ(ierr);
  /*f77L2NORM(res, &grid->nnodesLoc, &grid->nnodes, grid->x,
            grid->y,    grid->z,
            grid->area, &rank);*/
  f77FORCE(&grid->nnodesLoc, &grid->nedgeLoc,   
            grid->isnode,  grid->ivnode,  
           &grid->nnfacetLoc, grid->f2ntn, &grid->nnbound,
           &grid->nvfacetLoc, grid->f2ntv, &grid->nvbound,
            grid->eptr,    qnode,    
            grid->xyz,
            grid->sface_bit, grid->vface_bit,
            &clift, &cdrag, &cmom, &rank, &grid->nvertices);
  /*
  if (!rank)
   fprintf(fptr, "%d\t%15.8e\t%15.8e\t%15.8e\t%15.8e\t%15.8e\t%15.8e\n", 
           i, tsCtx->cfl, tsCtx->fnorm, clift, cdrag, cmom, cpuglo);
  */
  if (print_flag) {
   PetscPrintf(MPI_COMM_WORLD, "%d\t%g\t%g\t%g\t%g\t%g\n", i, 
               tsCtx->cfl, tsCtx->fnorm, clift, cdrag, cmom); 
   PetscPrintf(MPI_COMM_WORLD,"Wall clock time needed %g seconds for %d time steps\n", 
               cpuglo, i);
   if (!rank)
    fprintf(fptr, "%d\t%g\t%g\t%g\t%g\t%g\t%g\n", 
            i, tsCtx->cfl, tsCtx->fnorm, clift, cdrag, cmom, cpuglo);
  }
  ierr = VecRestoreArray(localX, &qnode);CHKERRQ(ierr);
  ierr = VecRestoreArray(grid->res, &res);CHKERRQ(ierr);
  fratio = tsCtx->fnorm_ini/tsCtx->fnorm;
  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
 } /* End of time step loop */

#if defined (PARCH_IRIX64) && defined(USE_HW_COUNTERS)
 if (!PreLoadFlag) {
  int eve0, eve1;
  FILE *cfp0, *cfp1;
  char str[256];
  /*if ((gen_read = read_counters(event0,&counter0,event1,&counter1)) < 0)
   SETERRQ(1,1,"Error in read_counter\n");
  ierr = PetscGetTime(&time_read_counters);CHKERRQ(ierr);
  if (gen_read != gen_start) {
   SETERRQ(1,1,"Lost Counters!! Aborting ...\n");
  }*/
  /*sprintf(str,"counters%d_and_%d",event0,event1);
  cfp0 = fopen(str,"a");*/
  /*ierr = print_counters(event0,counter0,event1,counter1);*/
  /*fprintf(cfp0,"%lld %lld %g\n",counter0,counter1,
                time_counters);
  fclose(cfp0);*/
 }
#endif
 PetscPrintf(MPI_COMM_WORLD,"Total wall clock time needed %g seconds for %d time steps\n", 
 cpuglo, i);
 PetscPrintf(MPI_COMM_WORLD, "cfl = %g fnorm = %g\n",
             tsCtx->cfl, tsCtx->fnorm);
 PetscPrintf(MPI_COMM_WORLD, "clift = %g cdrag = %g cmom = %g\n",
             clift, cdrag, cmom);

 /*PetscPrintf(MPI_COMM_WORLD, "Total cpu time needed %g seconds\n", cpu_time);*/
 if ((!rank) && (print_flag))
   fclose(fptr);
 if (PreLoadFlag) {
  tsCtx->fnorm_ini = 0.0;
  PetscPrintf(MPI_COMM_WORLD, "Preloading done ...\n");
  PreLoadFlag = 0;
 }

 PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "ComputeTimeStep"
int ComputeTimeStep(SNES snes, int iter, void *ctx)
/*---------------------------------------------------------------------*/
{
  AppCtx    *user = (AppCtx *) ctx;
  GRID      *grid = user->grid;
  TstepCtx  *tsCtx = user->tsCtx;
  Vec	    func = tsCtx->func;
  double    inc = 1.1;
  double    newcfl, fnorm;
  double    *res;
  int       ierr;
  int	    iramp = tsCtx->iramp;
 
  PetscFunctionBegin;
 
  tsCtx->ires = 0;
  ierr = FormFunction(snes,tsCtx->qold,func,user);CHKERRQ(ierr);
  tsCtx->ires = 1;
  ierr = VecNorm(func,NORM_2,&tsCtx->fnorm);CHKERRQ(ierr);
  /* first time through so compute initial function norm */
  if (tsCtx->fnorm_ini == 0.0) {
    tsCtx->fnorm_ini = tsCtx->fnorm;
    tsCtx->cfl = tsCtx->cfl_ini;
  } else {
     newcfl = inc*tsCtx->cfl_ini*tsCtx->fnorm_ini/tsCtx->fnorm;
     tsCtx->cfl = PetscMin(newcfl, tsCtx->cfl_max);
  }
 
  /*if (iramp < 0) {
   newcfl = inc*tsCtx->cfl_ini*tsCtx->fnorm_ini/tsCtx->fnorm;
  } else {
   if (tsCtx->dt < 0 && iramp > 0)
    if (iter > iramp) newcfl = tsCtx->cfl_max;
    else newcfl = tsCtx->cfl_ini + (tsCtx->cfl_max - tsCtx->cfl_ini)*
                                (double) iter/(double) iramp;
  }
  tsCtx->cfl = MIN(newcfl, tsCtx->cfl_max);*/
  /*printf("In ComputeTime Step - fnorm is %f\n", tsCtx->fnorm);*/
  /*ierr = VecDestroy(func);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "GetLocalOrdering"
int GetLocalOrdering(GRID *grid)
/*---------------------------------------------------------------------*/
{

  int        ierr, i, j, k, inode, isurf, nte, nb, node1, node2, node3, node4,totTr;
  int 	     nnodes, nedge, nnz, nnzLoc, jstart, jend, nbrs;
  int	     nnodesLoc, nvertices, nedgeLoc, nLoc,nnodesLocEst;
  int        nedgeLocEst, remEdges, readEdges, remNodes, readNodes;
  int 	     nnfacet, nvfacet, nffacet;
  int 	     nnfacetLoc, nvfacetLoc, nffacetLoc;
  int	     nsnode, nvnode, nfnode;
  int	     nsnodeLoc, nvnodeLoc, nfnodeLoc;
  int        nnbound, nvbound, nfbound;
  int        fdes, currentPos = 0, newPos = 0;
  int        one = 1, two = 2, three = 3, four = 4, unit = 20, grid_param = 13;
  int        *edge_bit, *pordering, *vertices, *verticesmask, *svertices;
  int	     *l2p, *l2a, *p2a, *p2l, *a2l, *v2p, *eperm;
  int        *ialoc, *jaloc, *ia, *ja, *ideg, *rowInd;
  int	     *tmp, *tmp1, *tmp2;
  Scalar     time_ini, time_fin;
  Scalar     *ftmp, *ftmp1, *ftmp2;
  char       str[256], form[256], part_name[256];
  AO         ao;
  IS         isglobal,islocal, isrow, iscol;
  Mat        Adj;
  FILE       *fptr, *fptr1;
  PETSCTRUTH flg;

  PetscFunctionBegin;
  /* Read the integer grid parameters */ 
  icalloc(grid_param, &tmp);
  if (!rank) {
   ierr = PetscBinaryOpen("testgrid/uns3d.msh",BINARY_RDONLY,&fdes);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, tmp, grid_param, PETSC_INT);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(tmp, grid_param, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  grid->ncell   = tmp[0];
  grid->nnodes  = tmp[1];
  grid->nedge   = tmp[2];
  grid->nnbound = tmp[3];
  grid->nvbound = tmp[4];
  grid->nfbound = tmp[5];
  grid->nnfacet = tmp[6];
  grid->nvfacet = tmp[7];
  grid->nffacet = tmp[8];
  grid->nsnode = tmp[9];
  grid->nvnode = tmp[10];
  grid->nfnode = tmp[11];
  grid->ntte   = tmp[12];
  grid->nsface = 0;
  grid->nvface = 0;
  grid->nfface = 0;
  ierr = PetscFree(tmp);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "nnodes = %d, nedge = %d, nnfacet = %d, nsnode = %d, nfnode = %d\n",
              grid->nnodes,grid->nedge,grid->nnfacet,grid->nsnode,grid->nfnode);
/* Until the above line, the equivalent I/O and other initialization
  work of RDGPAR has been done */
  nnodes  = grid->nnodes;
  nedge   = grid->nedge;
  nnfacet = grid->nnfacet;
  nvfacet = grid->nvfacet;
  nffacet = grid->nffacet;
  nnbound = grid->nnbound;
  nvbound = grid->nvbound;
  nfbound = grid->nfbound;
  nsnode  = grid->nsnode;
  nvnode  = grid->nvnode;
  nfnode  = grid->nfnode;

  /* Read the partitioning vector generated by MeTiS */
  icalloc(nnodes, &l2a);
  icalloc(nnodes, &v2p);
  icalloc(nnodes, &a2l);
  nnodesLoc = 0;

  for (i = 0; i < nnodes; i++)  
    a2l[i] = -1;
  ierr = PetscGetTime(&time_ini);CHKERRQ(ierr);

  if (!rank) {
   sprintf(part_name,"testgrid/part_vec.part.%d",CommSize);
   fptr = fopen(part_name,"r");
   assert(fptr != 0);
   for (inode = 0; inode < nnodes; inode++) {
    fscanf(fptr, "%d\n", &node1); 
    v2p[inode] = node1;
   }
   fclose(fptr);
  }
  ierr = MPI_Bcast(v2p, nnodes, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  for (inode = 0; inode < nnodes; inode++) {
    if (v2p[inode] == rank) {
      l2a[nnodesLoc] = inode ; 
      a2l[inode] = nnodesLoc ; 
      nnodesLoc++;
    } 
  }
  ierr = PetscGetTime(&time_fin);CHKERRQ(ierr);
  time_fin -= time_ini;
  PetscPrintf(PETSC_COMM_WORLD, "Partition Vector read successfully\n");
  PetscPrintf(PETSC_COMM_WORLD, "Time taken in this phase was %g\n", time_fin);

  MPI_Scan(&nnodesLoc,&rstart,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  rstart -= nnodesLoc;
  icalloc(nnodesLoc, &pordering);
  for (i=0; i < nnodesLoc; i++) {
    pordering[i] = rstart + i;
  }
  ierr = AOCreateBasic(MPI_COMM_WORLD,nnodesLoc,l2a,pordering,&ao);CHKERRQ(ierr);
  ierr = PetscFree(pordering);CHKERRQ(ierr);

  /* Now count the local number of edges - including edges with 
   ghost nodes but edges between ghost nodes are NOT counted */
  nedgeLoc = 0;
  nvertices = nnodesLoc;
  /* Choose an estimated number of local edges. The choice
   nedgeLocEst = 1000000 looks reasonable as it will read
   the edge and edge normal arrays in 8 MB chunks */ 
  /*nedgeLocEst = nedge/CommSize;*/
  nedgeLocEst = PetscMin(nedge,1000000); 
  remEdges = nedge;
  icalloc(2*nedgeLocEst, &tmp);
  if (!rank) {ierr = PetscBinarySeek(fdes, 0, BINARY_SEEK_CUR,&currentPos);CHKERRQ(ierr);}
  ierr = PetscGetTime(&time_ini);CHKERRQ(ierr);
  while (remEdges > 0) {
   readEdges = PetscMin(remEdges,nedgeLocEst); 
   /*time_ini = PetscGetTime();*/
   if (!rank) {
    ierr = PetscBinaryRead(fdes, tmp, readEdges, PETSC_INT);CHKERRQ(ierr);
    ierr = PetscBinarySeek(fdes,(nedge-readEdges)*BINARY_INT_SIZE,BINARY_SEEK_CUR,&newPos);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fdes, tmp+readEdges, readEdges, PETSC_INT);CHKERRQ(ierr);
    ierr = PetscBinarySeek(fdes,-nedge*BINARY_INT_SIZE,BINARY_SEEK_CUR,&newPos);CHKERRQ(ierr);
   }
   ierr = MPI_Bcast(tmp, 2*readEdges, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
   /*time_fin += PetscGetTime()-time_ini;*/
   for (j = 0; j < readEdges; j++) {
     node1 = tmp[j]-1;
     node2 = tmp[j+readEdges]-1;
     if ((v2p[node1] == rank) || (v2p[node2] == rank)) {
       nedgeLoc++;
       if (a2l[node1] == -1) {
        l2a[nvertices] = node1;
        a2l[node1] = nvertices;
        nvertices++;
       }
       if (a2l[node2] == -1) {
        l2a[nvertices] = node2;
        a2l[node2] = nvertices;
        nvertices++;
       }
     }
   }
   remEdges = remEdges - readEdges; 
   ierr = MPI_Barrier(MPI_COMM_WORLD);
  }
  ierr = PetscGetTime(&time_fin);CHKERRQ(ierr);
  time_fin -= time_ini;
  PetscPrintf(PETSC_COMM_WORLD, "Local edges counted with MPI_Bcast %d\n",nedgeLoc);
  PetscPrintf(PETSC_COMM_WORLD, "Local vertices counted %d\n",nvertices);
  PetscPrintf(PETSC_COMM_WORLD, "Time taken in this phase was %g\n", time_fin);

  /* Now store the local edges */
  icalloc(2*nedgeLoc, &grid->eptr);
  icalloc(nedgeLoc, &edge_bit);
  icalloc(nedgeLoc, &eperm);
  i = 0; j = 0; k = 0;
  remEdges = nedge;
  if (!rank) {
   ierr = PetscBinarySeek(fdes, currentPos, BINARY_SEEK_SET,&newPos);CHKERRQ(ierr);
   currentPos = newPos;
  }
  ierr = PetscGetTime(&time_ini);CHKERRQ(ierr);
  while (remEdges > 0) {
   readEdges = PetscMin(remEdges,nedgeLocEst); 
   if (!rank) {
    ierr = PetscBinaryRead(fdes, tmp, readEdges, PETSC_INT);CHKERRQ(ierr);
    ierr = PetscBinarySeek(fdes,(nedge-readEdges)*BINARY_INT_SIZE,BINARY_SEEK_CUR,&newPos);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fdes, tmp+readEdges, readEdges, PETSC_INT);CHKERRQ(ierr);
    ierr = PetscBinarySeek(fdes,-nedge*BINARY_INT_SIZE,BINARY_SEEK_CUR,&newPos);CHKERRQ(ierr);
   }
   ierr = MPI_Bcast(tmp, 2*readEdges, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
   for (j = 0; j < readEdges; j++) {
     node1 = tmp[j]-1;
     node2 = tmp[j+readEdges]-1;
     if ((v2p[node1] == rank) || (v2p[node2] == rank)) {
       grid->eptr[k] = a2l[node1];
       grid->eptr[k+nedgeLoc] = a2l[node2];
       edge_bit[k] = i;
       eperm[k] = k;
       k++;
     }
     i++;
   }
   remEdges = remEdges - readEdges; 
   ierr = MPI_Barrier(MPI_COMM_WORLD);
  }
  if (!rank) {ierr = PetscBinarySeek(fdes,currentPos+2*nedge*BINARY_INT_SIZE,BINARY_SEEK_SET,&newPos);CHKERRQ(ierr);}
  ierr = PetscGetTime(&time_fin);CHKERRQ(ierr);
  time_fin -= time_ini;
  PetscPrintf(PETSC_COMM_WORLD, "Local edges stored\n");
  PetscPrintf(PETSC_COMM_WORLD, "Time taken in this phase was %g\n", time_fin);

  ierr = PetscFree(tmp);CHKERRQ(ierr);
  icalloc(2*nedgeLoc, &tmp);
  /* Now reorder the edges for better cache locality */     
  /*
  tmp[0]=7;tmp[1]=6;tmp[2]=3;tmp[3]=9;tmp[4]=2;tmp[5]=0;
  ierr = PetscSortIntWithPermutation(6, tmp, eperm); 
  for (i=0; i<6; i++)
   printf("%d %d %d\n", i, tmp[i], eperm[i]);
  */
  ierr = PetscMemcpy(tmp,grid->eptr,2*nedgeLoc*sizeof(int));CHKERRQ(ierr);
  ierr = OptionsHasName(0,"-no_edge_reordering",&flg);CHKERRQ(ierr);
  if (!flg) {
   ierr = PetscSortIntWithPermutation(nedgeLoc,tmp,eperm);CHKERRQ(ierr);
  }
  k = 0;
  for (i = 0; i < nedgeLoc; i++) {
#if defined(INTERLACING) 
   grid->eptr[k++] = tmp[eperm[i]] + 1;
   grid->eptr[k++] = tmp[nedgeLoc+eperm[i]] + 1;
#else
   grid->eptr[i] = tmp[eperm[i]] + 1;
   grid->eptr[nedgeLoc+i] = tmp[nedgeLoc+eperm[i]] + 1;
#endif
  }
  ierr = PetscFree(tmp);CHKERRQ(ierr);

  /* Now make the local 'ia' and 'ja' arrays */
   icalloc(nnodesLoc+1, &grid->ia);
  /* Use tmp for a work array */
   icalloc(nnodesLoc, &tmp);
   f77GETIA(&nnodesLoc, &nedgeLoc, grid->eptr, 
             grid->ia, tmp, &rank);
   nnz = grid->ia[nnodesLoc] - 1;
   icalloc(nnz, &grid->ja);
   f77GETJA(&nnodesLoc, &nedgeLoc, grid->eptr, 
             grid->ia,      grid->ja,    tmp, &rank);

   ierr = PetscFree(tmp);CHKERRQ(ierr);

  icalloc(nvertices, &grid->loc2glo);
  ierr = PetscMemcpy(grid->loc2glo,l2a,nvertices*sizeof(int));CHKERRQ(ierr);
  ierr = PetscFree(l2a);CHKERRQ(ierr);
  l2a = grid->loc2glo;
  icalloc(nvertices, &grid->loc2pet);
  l2p = grid->loc2pet;
  ierr = PetscMemcpy(l2p,l2a,nvertices*sizeof(int));CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,nvertices,l2p);CHKERRQ(ierr);

/* Map the 'ja' array in petsc ordering */
  nnz = grid->ia[nnodesLoc] - 1;
  for (i = 0; i < nnz; i++)
   grid->ja[i] = l2a[grid->ja[i] - 1];
  ierr = AOApplicationToPetsc(ao,nnz,grid->ja);CHKERRQ(ierr);
  ierr = AODestroy(ao);CHKERRQ(ierr);

 /* Renumber unit normals of dual face (from node1 to node2)
     and the area of the dual mesh face */
  fcalloc(nedgeLocEst, &ftmp);
  fcalloc(nedgeLoc, &ftmp1);
  fcalloc(4*nedgeLoc, &grid->xyzn);
  /* Do the x-component */
  i = 0; k = 0;
  remEdges = nedge;
  ierr = PetscGetTime(&time_ini);CHKERRQ(ierr);
  while (remEdges > 0) {
   readEdges = PetscMin(remEdges,nedgeLocEst); 
   if (!rank) {
    ierr = PetscBinaryRead(fdes, ftmp, readEdges, PETSC_SCALAR);CHKERRQ(ierr);
   }
   ierr = MPI_Bcast(ftmp, readEdges, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
   for (j = 0; j < readEdges; j++) {
     if (edge_bit[k] == (i+j)) {
      ftmp1[k] = ftmp[j];
      k++;
     }
   }
   i+= readEdges;
   remEdges = remEdges - readEdges; 
   ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  }
  for (i = 0; i < nedgeLoc; i++) 
#if defined(INTERLACING) 
   grid->xyzn[4*i] = ftmp1[eperm[i]];
#else
   grid->xyzn[i] = ftmp1[eperm[i]];
#endif
  /* Do the y-component */
  i = 0; k = 0;
  remEdges = nedge;
  while (remEdges > 0) {
   readEdges = PetscMin(remEdges,nedgeLocEst); 
   if (!rank) 
    ierr = PetscBinaryRead(fdes, ftmp, readEdges, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = MPI_Bcast(ftmp, readEdges, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
   for (j = 0; j < readEdges; j++) {
     if (edge_bit[k] == (i+j)) {
      ftmp1[k] = ftmp[j];
      k++;
     }
   }
   i+= readEdges;
   remEdges = remEdges - readEdges; 
   ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  }
  for (i = 0; i < nedgeLoc; i++) 
#if defined(INTERLACING) 
   grid->xyzn[4*i+1] = ftmp1[eperm[i]];
#else
   grid->xyzn[nedgeLoc+i] = ftmp1[eperm[i]];
#endif
  /* Do the z-component */
  i = 0; k = 0;
  remEdges = nedge;
  while (remEdges > 0) {
   readEdges = PetscMin(remEdges,nedgeLocEst); 
   if (!rank) 
    ierr = PetscBinaryRead(fdes, ftmp, readEdges, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = MPI_Bcast(ftmp, readEdges, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
   for (j = 0; j < readEdges; j++) {
     if (edge_bit[k] == (i+j)) {
      ftmp1[k] = ftmp[j];
      k++;
     }
   }
   i+= readEdges;
   remEdges = remEdges - readEdges; 
   ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  }
  for (i = 0; i < nedgeLoc; i++) 
#if defined(INTERLACING) 
   grid->xyzn[4*i+2] = ftmp1[eperm[i]];
#else
   grid->xyzn[2*nedgeLoc+i] = ftmp1[eperm[i]];
#endif
  /* Do the length */
  i = 0; k = 0;
  remEdges = nedge;
  while (remEdges > 0) {
   readEdges = PetscMin(remEdges,nedgeLocEst); 
   if (!rank) 
   ierr = PetscBinaryRead(fdes, ftmp, readEdges, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = MPI_Bcast(ftmp, readEdges, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
   for (j = 0; j < readEdges; j++) {
     if (edge_bit[k] == (i+j)) {
      ftmp1[k] = ftmp[j];
      k++;
     }
   }
   i+= readEdges;
   remEdges = remEdges - readEdges; 
   ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  }
  for (i = 0; i < nedgeLoc; i++) 
#if defined(INTERLACING) 
   grid->xyzn[4*i+3] = ftmp1[eperm[i]];
#else
   grid->xyzn[3*nedgeLoc+i] = ftmp1[eperm[i]];
#endif

  ierr = PetscFree(edge_bit);CHKERRQ(ierr);
  ierr = PetscFree(eperm);CHKERRQ(ierr);
  ierr = PetscFree(ftmp);CHKERRQ(ierr);
  ierr = PetscFree(ftmp1);CHKERRQ(ierr);
  ierr = PetscGetTime(&time_fin);CHKERRQ(ierr);
  time_fin -= time_ini;
  PetscPrintf(PETSC_COMM_WORLD, "Edge normals partitioned\n");
  PetscPrintf(PETSC_COMM_WORLD, "Time taken in this phase was %g\n", time_fin);

  /* Remap coordinates */
  /*nnodesLocEst = nnodes/CommSize;*/
  nnodesLocEst = PetscMin(nnodes,500000);
  fcalloc(nnodesLocEst, &ftmp);
  fcalloc(3*nvertices, &grid->xyz);
  remNodes = nnodes;
  i = 0;
  ierr = PetscGetTime(&time_ini);CHKERRQ(ierr);
  while (remNodes > 0) {
   readNodes = PetscMin(remNodes, nnodesLocEst); 
   if (!rank) 
    ierr = PetscBinaryRead(fdes, ftmp, readNodes, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = MPI_Bcast(ftmp, readNodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
   for (j = 0; j < readNodes; j++) {
     if (a2l[i+j] >= 0) {
#if defined(INTERLACING) 
      grid->xyz[3*a2l[i+j]] = ftmp[j];
#else
      grid->xyz[a2l[i+j]] = ftmp[j];
#endif
     }
   }
   i+= nnodesLocEst;
   remNodes -= nnodesLocEst; 
   ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  }

  remNodes = nnodes;
  i = 0;
  while (remNodes > 0) {
   readNodes = PetscMin(remNodes,nnodesLocEst); 
   if (!rank) 
    ierr = PetscBinaryRead(fdes, ftmp, readNodes, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = MPI_Bcast(ftmp, readNodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
   for (j = 0; j < readNodes; j++) {
     if (a2l[i+j] >= 0) {
#if defined(INTERLACING) 
      grid->xyz[3*a2l[i+j]+1] = ftmp[j];
#else
      grid->xyz[nnodesLoc+a2l[i+j]] = ftmp[j];
#endif
     }
   }
   i+= nnodesLocEst;
   remNodes -= nnodesLocEst; 
   ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  }

  remNodes = nnodes;
  i = 0;
  while (remNodes > 0) {
   readNodes = PetscMin(remNodes,nnodesLocEst); 
   if (!rank) 
    ierr = PetscBinaryRead(fdes, ftmp, readNodes, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = MPI_Bcast(ftmp, readNodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
   for (j = 0; j < readNodes; j++) {
     if (a2l[i+j] >= 0) {
#if defined(INTERLACING) 
      grid->xyz[3*a2l[i+j]+2] = ftmp[j];
#else
      grid->xyz[2*nnodesLoc+a2l[i+j]] = ftmp[j];
#endif
     }
   }
   i+= nnodesLocEst;
   remNodes -= nnodesLocEst; 
   ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  }


  /* Renumber dual volume */
  fcalloc(nvertices, &grid->area);
  remNodes = nnodes;
  i = 0;
  while (remNodes > 0) {
   readNodes = PetscMin(remNodes,nnodesLocEst); 
   if (!rank) 
    ierr = PetscBinaryRead(fdes, ftmp, readNodes, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = MPI_Bcast(ftmp, readNodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
   for (j = 0; j < readNodes; j++) {
     if (a2l[i+j] >= 0) {
      grid->area[a2l[i+j]] = ftmp[j];
     }
   }
   i+= nnodesLocEst;
   remNodes -= nnodesLocEst; 
   ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  }

  ierr = PetscFree(ftmp);CHKERRQ(ierr);
  ierr = PetscGetTime(&time_fin);CHKERRQ(ierr);
  time_fin -= time_ini;
  PetscPrintf(PETSC_COMM_WORLD, "Coordinates remapped\n");
  PetscPrintf(PETSC_COMM_WORLD, "Time taken in this phase was %g\n", time_fin);

/* Now, handle all the solid boundaries - things to be done :
 * 1. Identify the nodes belonging to the solid  
 *    boundaries and count them.
 * 2. Put proper indices into f2ntn array, after making it
 *    of suitable size.
 * 3. Remap the normals and areas of solid faces (sxn, syn, szn, 
 *    and sa arrays). 
 */
  icalloc(nnbound,   &grid->nntet);
  icalloc(nnbound,   &grid->nnpts);
  icalloc(4*nnfacet, &grid->f2ntn);
  icalloc(nsnode, &grid->isnode);
  fcalloc(nsnode, &grid->sxn);
  fcalloc(nsnode, &grid->syn);
  fcalloc(nsnode, &grid->szn);
  fcalloc(nsnode, &grid->sa);
  if (!rank) {
   ierr = PetscBinaryRead(fdes, (void *) grid->nntet, nnbound, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->nnpts, nnbound, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->f2ntn, 4*nnfacet, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->isnode, nsnode, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->sxn, nsnode, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->syn, nsnode, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->szn, nsnode, PETSC_SCALAR);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(grid->nntet, nnbound, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->nnpts, nnbound, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->f2ntn, 4*nnfacet, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->isnode, nsnode, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->sxn, nsnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->syn, nsnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->szn, nsnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);

  isurf = 0;
  nsnodeLoc = 0;
  nnfacetLoc = 0;
  nb = 0;
  nte = 0;
  icalloc(3*nnfacet, &tmp);
  icalloc(nsnode, &tmp1);
  icalloc(nnodes, &tmp2);
  fcalloc(4*nsnode, &ftmp);
  ierr = PetscMemzero(tmp,3*nnfacet*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemzero(tmp1,nsnode*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemzero(tmp2,nnodes*sizeof(int));CHKERRQ(ierr);

  j = 0;
  for (i = 0; i < nsnode; i++) {
    node1 = a2l[grid->isnode[i] - 1];
    if (node1 >= 0) {
     tmp1[nsnodeLoc] = node1;
     tmp2[node1] = nsnodeLoc;
     ftmp[j++] = grid->sxn[i];
     ftmp[j++] = grid->syn[i];
     ftmp[j++] = grid->szn[i];
     ftmp[j++] = grid->sa[i];
     nsnodeLoc++;
    }
  }
  for (i = 0; i < nnbound; i++) {
   for (j = isurf; j < isurf + grid->nntet[i]; j++) {
    node1 = a2l[grid->isnode[grid->f2ntn[j] - 1] - 1];
    node2 = a2l[grid->isnode[grid->f2ntn[nnfacet + j] - 1] - 1];
    node3 = a2l[grid->isnode[grid->f2ntn[2*nnfacet + j] - 1] - 1];
    
    if ((node1 >= 0) && (node2 >= 0) && (node3 >= 0)) {
       nnfacetLoc++;
       nte++;
       tmp[nb++] = tmp2[node1];
       tmp[nb++] = tmp2[node2];
       tmp[nb++] = tmp2[node3];
    }
   }
   isurf += grid->nntet[i];
   /*printf("grid->nntet[%d] before reordering is %d\n",i,grid->nntet[i]);*/
   grid->nntet[i] = nte;
   /*printf("grid->nntet[%d] after reordering is %d\n",i,grid->nntet[i]);*/
   nte = 0;
  }
  ierr = PetscFree(grid->f2ntn);CHKERRQ(ierr);
  ierr = PetscFree(grid->isnode);CHKERRQ(ierr);
  ierr = PetscFree(grid->sxn);CHKERRQ(ierr);
  ierr = PetscFree(grid->syn);CHKERRQ(ierr);
  ierr = PetscFree(grid->szn);CHKERRQ(ierr);
  ierr = PetscFree(grid->sa);CHKERRQ(ierr);
  icalloc(4*nnfacetLoc, &grid->f2ntn);
  icalloc(nsnodeLoc, &grid->isnode);
  fcalloc(nsnodeLoc, &grid->sxn);
  fcalloc(nsnodeLoc, &grid->syn);
  fcalloc(nsnodeLoc, &grid->szn);
  fcalloc(nsnodeLoc, &grid->sa);
  j = 0;
  for (i = 0; i < nsnodeLoc; i++) {
   grid->isnode[i] = tmp1[i] + 1;
   grid->sxn[i] = ftmp[j++];
   grid->syn[i] = ftmp[j++];
   grid->szn[i] = ftmp[j++];
   grid->sa[i] = ftmp[j++];
  }
  j = 0;
  for (i = 0; i < nnfacetLoc; i++) {
   grid->f2ntn[i] = tmp[j++] + 1; 
   grid->f2ntn[nnfacetLoc+i] = tmp[j++] + 1; 
   grid->f2ntn[2*nnfacetLoc+i] = tmp[j++] + 1; 
  }
 ierr = PetscFree(tmp);CHKERRQ(ierr);
 ierr = PetscFree(tmp1);CHKERRQ(ierr);
 ierr = PetscFree(tmp2);CHKERRQ(ierr);
 ierr = PetscFree(ftmp);CHKERRQ(ierr);

/* Now identify the triangles on which the current proceesor
   would perform force calculation */
  icalloc(nnfacetLoc, &grid->sface_bit);
  PetscMemzero(grid->sface_bit,nnfacetLoc*sizeof(int));
  for (i = 0; i < nnfacetLoc; i++) {
    node1 = l2a[grid->isnode[grid->f2ntn[i] - 1] - 1];
    node2 = l2a[grid->isnode[grid->f2ntn[nnfacetLoc + i] - 1] - 1];
    node3 = l2a[grid->isnode[grid->f2ntn[2*nnfacetLoc + i] - 1] - 1];
    if (((v2p[node1] >= rank) && (v2p[node2] >= rank) 
        && (v2p[node3] >= rank)) &&
        ((v2p[node1] == rank) || (v2p[node2] == rank)
        || (v2p[node3] == rank))) {
         grid->sface_bit[i] = 1;
    }
  }
  /*printf("On processor %d total solid triangles = %d, locally owned = %d alpha = %d\n", rank, totTr, myTr,alpha);*/
 PetscPrintf(PETSC_COMM_WORLD, "Solid boundaries partitioned\n");

/* Now, handle all the viscous boundaries - things to be done :
 * 1. Identify the nodes belonging to the viscous
 *    boundaries and count them.
 * 2. Put proper indices into f2ntv array, after making it
 *    of suitable size
 * 3. Remap the normals and areas of viscous faces (vxn, vyn, vzn, 
 *    and va arrays). 
 */
  icalloc(nvbound,   &grid->nvtet);
  icalloc(nvbound,   &grid->nvpts);
  icalloc(4*nvfacet, &grid->f2ntv);
  icalloc(nvnode, &grid->ivnode);
  fcalloc(nvnode, &grid->vxn);
  fcalloc(nvnode, &grid->vyn);
  fcalloc(nvnode, &grid->vzn);
  fcalloc(nvnode, &grid->va);
  if (!rank) {
   ierr = PetscBinaryRead(fdes, (void *) grid->nvtet, nvbound, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->nvpts, nvbound, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->f2ntv, 4*nvfacet, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->ivnode, nvnode, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->vxn, nvnode, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->vyn, nvnode, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->vzn, nvnode, PETSC_SCALAR);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(grid->nvtet, nvbound, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->nvpts, nvbound, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->f2ntv, 4*nvfacet, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->ivnode, nvnode, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->vxn, nvnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->vyn, nvnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->vzn, nvnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);


  isurf = 0;
  nvnodeLoc = 0;
  nvfacetLoc = 0;
  nb = 0;
  nte = 0;
  icalloc(3*nvfacet, &tmp);
  icalloc(nvnode, &tmp1);
  icalloc(nnodes, &tmp2);
  fcalloc(4*nvnode, &ftmp);
  ierr = PetscMemzero(tmp,3*nvfacet*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemzero(tmp1,nvnode*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemzero(tmp2,nnodes*sizeof(int));CHKERRQ(ierr);

  j = 0;
  for (i = 0; i < nvnode; i++) {
    node1 = a2l[grid->ivnode[i] - 1];
    if (node1 >= 0) {
     tmp1[nvnodeLoc] = node1;
     tmp2[node1] = nvnodeLoc;
     ftmp[j++] = grid->vxn[i];
     ftmp[j++] = grid->vyn[i];
     ftmp[j++] = grid->vzn[i];
     ftmp[j++] = grid->va[i];
     nvnodeLoc++;
    }
  }
  for (i = 0; i < nvbound; i++) {
   for (j = isurf; j < isurf + grid->nvtet[i]; j++) {
    node1 = a2l[grid->ivnode[grid->f2ntv[j] - 1] - 1];
    node2 = a2l[grid->ivnode[grid->f2ntv[nvfacet + j] - 1] - 1];
    node3 = a2l[grid->ivnode[grid->f2ntv[2*nvfacet + j] - 1] - 1];
    if ((node1 >= 0) && (node2 >= 0) && (node3 >= 0)) {
       nvfacetLoc++;
       nte++;
       tmp[nb++] = tmp2[node1];
       tmp[nb++] = tmp2[node2];
       tmp[nb++] = tmp2[node3];
    }
   }
   isurf += grid->nvtet[i];
   grid->nvtet[i] = nte;
   nte = 0;
  }
  ierr = PetscFree(grid->f2ntv);CHKERRQ(ierr);
  ierr = PetscFree(grid->ivnode);CHKERRQ(ierr);
  ierr = PetscFree(grid->vxn);CHKERRQ(ierr);
  ierr = PetscFree(grid->vyn);CHKERRQ(ierr);
  ierr = PetscFree(grid->vzn);CHKERRQ(ierr);
  ierr = PetscFree(grid->va);CHKERRQ(ierr);
  icalloc(4*nvfacetLoc, &grid->f2ntv);
  icalloc(nvnodeLoc, &grid->ivnode);
  fcalloc(nvnodeLoc, &grid->vxn);
  fcalloc(nvnodeLoc, &grid->vyn);
  fcalloc(nvnodeLoc, &grid->vzn);
  fcalloc(nvnodeLoc, &grid->va);
  j = 0;
  for (i = 0; i < nvnodeLoc; i++) {
   grid->ivnode[i] = tmp1[i] + 1;
   grid->vxn[i] = ftmp[j++];
   grid->vyn[i] = ftmp[j++];
   grid->vzn[i] = ftmp[j++];
   grid->va[i] = ftmp[j++];
  }
  j = 0;
  for (i = 0; i < nvfacetLoc; i++) {
   grid->f2ntv[i] = tmp[j++] + 1;
   grid->f2ntv[nvfacetLoc+i] = tmp[j++] + 1;
   grid->f2ntv[2*nvfacetLoc+i] = tmp[j++] + 1;
  }
 ierr = PetscFree(tmp);CHKERRQ(ierr);
 ierr = PetscFree(tmp1);CHKERRQ(ierr);
 ierr = PetscFree(tmp2);CHKERRQ(ierr);
 ierr = PetscFree(ftmp);CHKERRQ(ierr);

/* Now identify the triangles on which the current proceesor
   would perform force calculation */
  icalloc(nvfacetLoc, &grid->vface_bit);
  ierr = PetscMemzero(grid->vface_bit,nvfacetLoc*sizeof(int));CHKERRQ(ierr);
  for (i = 0; i < nvfacetLoc; i++) {
    node1 = l2a[grid->ivnode[grid->f2ntv[i] - 1] - 1];
    node2 = l2a[grid->ivnode[grid->f2ntv[nvfacetLoc + i] - 1] - 1];
    node3 = l2a[grid->ivnode[grid->f2ntv[2*nvfacetLoc + i] - 1] - 1];
    if (((v2p[node1] >= rank) && (v2p[node2] >= rank)
        && (v2p[node3] >= rank)) &&
        ((v2p[node1] == rank) || (v2p[node2] == rank)
        || (v2p[node3] == rank))) {
         grid->vface_bit[i] = 1;
    }
  }
  ierr = PetscFree(v2p);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Viscous boundaries partitioned\n");
 
/* Now, handle all the free boundaries - things to be done :
 * 1. Identify the nodes belonging to the free
 *    boundaries and count them.
 * 2. Put proper indices into f2ntf array, after making it
 *    of suitable size
 * 3. Remap the normals and areas of free bound. faces (fxn, fyn, fzn, 
 *    and fa arrays). 
 */
 
  icalloc(nfbound,   &grid->nftet);
  icalloc(nfbound,   &grid->nfpts);
  icalloc(4*nffacet, &grid->f2ntf);
  icalloc(nfnode, &grid->ifnode);
  fcalloc(nfnode, &grid->fxn);
  fcalloc(nfnode, &grid->fyn);
  fcalloc(nfnode, &grid->fzn);
  fcalloc(nfnode, &grid->fa);
  if (!rank) {
   ierr = PetscBinaryRead(fdes, (void *) grid->nftet, nfbound, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->nfpts, nfbound, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->f2ntf, 4*nffacet, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->ifnode, nfnode, PETSC_INT);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->fxn, nfnode, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->fyn, nfnode, PETSC_SCALAR);CHKERRQ(ierr);
   ierr = PetscBinaryRead(fdes, (void *) grid->fzn, nfnode, PETSC_SCALAR);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(grid->nftet, nfbound, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->nfpts, nfbound, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->f2ntf, 4*nffacet, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->ifnode, nfnode, MPI_INT, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->fxn, nfnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->fyn, nfnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(grid->fzn, nfnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);CHKERRQ(ierr);

  isurf = 0;
  nfnodeLoc = 0;
  nffacetLoc = 0;
  nb = 0;
  nte = 0;
  icalloc(3*nffacet, &tmp);
  icalloc(nfnode, &tmp1);
  icalloc(nnodes, &tmp2);
  fcalloc(4*nfnode, &ftmp);
  ierr = PetscMemzero(tmp,3*nffacet*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemzero(tmp1,nfnode*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemzero(tmp2,nnodes*sizeof(int));CHKERRQ(ierr);

  j = 0;
  for (i = 0; i < nfnode; i++) {
    node1 = a2l[grid->ifnode[i] - 1];
    if (node1 >= 0) {
     tmp1[nfnodeLoc] = node1;
     tmp2[node1] = nfnodeLoc;
     ftmp[j++] = grid->fxn[i];
     ftmp[j++] = grid->fyn[i];
     ftmp[j++] = grid->fzn[i];
     ftmp[j++] = grid->fa[i];
     nfnodeLoc++;
    }
  }
  for (i = 0; i < nfbound; i++) {
   for (j = isurf; j < isurf + grid->nftet[i]; j++) {
    node1 = a2l[grid->ifnode[grid->f2ntf[j] - 1] - 1];
    node2 = a2l[grid->ifnode[grid->f2ntf[nffacet + j] - 1] - 1];
    node3 = a2l[grid->ifnode[grid->f2ntf[2*nffacet + j] - 1] - 1];
    if ((node1 >= 0) && (node2 >= 0) && (node3 >= 0)) {
       nffacetLoc++;
       nte++;
       tmp[nb++] = tmp2[node1];
       tmp[nb++] = tmp2[node2];
       tmp[nb++] = tmp2[node3];
    }
   }
   isurf += grid->nftet[i];
   grid->nftet[i] = nte;
   nte = 0;
  }
  ierr = PetscFree(grid->f2ntf);CHKERRQ(ierr);
  ierr = PetscFree(grid->ifnode);CHKERRQ(ierr);
  ierr = PetscFree(grid->fxn);CHKERRQ(ierr);
  ierr = PetscFree(grid->fyn);CHKERRQ(ierr);
  ierr = PetscFree(grid->fzn);CHKERRQ(ierr);
  ierr = PetscFree(grid->fa);CHKERRQ(ierr);
  icalloc(4*nffacetLoc, &grid->f2ntf);
  icalloc(nfnodeLoc, &grid->ifnode);
  fcalloc(nfnodeLoc, &grid->fxn);
  fcalloc(nfnodeLoc, &grid->fyn);
  fcalloc(nfnodeLoc, &grid->fzn);
  fcalloc(nfnodeLoc, &grid->fa);
  j = 0;
  for (i = 0; i < nfnodeLoc; i++) {
   grid->ifnode[i] = tmp1[i] + 1;
   grid->fxn[i] = ftmp[j++];
   grid->fyn[i] = ftmp[j++];
   grid->fzn[i] = ftmp[j++];
   grid->fa[i] = ftmp[j++];
  }
  j = 0;
  for (i = 0; i < nffacetLoc; i++) {
   grid->f2ntf[i] = tmp[j++] + 1;
   grid->f2ntf[nffacetLoc+i] = tmp[j++] + 1;
   grid->f2ntf[2*nffacetLoc+i] = tmp[j++] + 1;
  }

 
  ierr = PetscFree(tmp);CHKERRQ(ierr);
  ierr = PetscFree(tmp1);CHKERRQ(ierr);
  ierr = PetscFree(tmp2);CHKERRQ(ierr);
  ierr = PetscFree(ftmp);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Free boundaries partitioned\n");

  ierr = OptionsHasName(0,"-mem_use",&flg);CHKERRQ(ierr);
  if (flg) {
   PLogDouble space, maxSpace;
   ierr = PetscTrSpace(&space,0,&maxSpace);
   PetscPrintf(PETSC_COMM_WORLD,"Space allocated after partionioning is %g\n",
               space);
   PetscPrintf(PETSC_COMM_WORLD,"Max space allocated so far is %g\n",
               maxSpace);
  }
 /* Put different mappings and other info into grid */
  /* icalloc(nvertices, &grid->loc2pet); 
   icalloc(nvertices, &grid->loc2glo);
   PetscMemcpy(grid->loc2pet,l2p,nvertices*sizeof(int));
   PetscMemcpy(grid->loc2glo,l2a,nvertices*sizeof(int));
   ierr = PetscFree(l2a);CHKERRQ(ierr);
   ierr = PetscFree(l2p);CHKERRQ(ierr);*/
  
   grid->nnodesLoc = nnodesLoc;
   grid->nedgeLoc = nedgeLoc;
   grid->nvertices = nvertices;
   grid->nsnodeLoc = nsnodeLoc;
   grid->nvnodeLoc = nvnodeLoc;
   grid->nfnodeLoc = nfnodeLoc;
   grid->nnfacetLoc = nnfacetLoc;
   grid->nvfacetLoc = nvfacetLoc;
   grid->nffacetLoc = nffacetLoc;
/*
 * fcalloc(nvertices*4,  &grid->gradx);
 * fcalloc(nvertices*4,  &grid->grady);
 * fcalloc(nvertices*4,  &grid->gradz);
 */
   fcalloc(nvertices,    &grid->cdt);
   fcalloc(nvertices*4,  &grid->phi);
/*
   fcalloc(nvertices,    &grid->r11);
   fcalloc(nvertices,    &grid->r12);
   fcalloc(nvertices,    &grid->r13);
   fcalloc(nvertices,    &grid->r22);
   fcalloc(nvertices,    &grid->r23);
   fcalloc(nvertices,    &grid->r33);
*/
   fcalloc(7*nnodesLoc,    &grid->rxy);

/* Print the different mappings
 *
 */
 {
  int partLoc[7],partMax[7],partMin[7], partSum[7];
  int nnodesLocMax = 0, nnodesLocMin = 0, nverticesMin = 0, nverticesMax = 0;
  int nedgeLocMax = 0, nedgeLocMin = 0;
  partLoc[0] = nnodesLoc;
  partLoc[1] = nvertices;
  partLoc[2] = nedgeLoc;
  partLoc[3] = nnfacetLoc;
  partLoc[4] = nffacetLoc;
  partLoc[5] = nsnodeLoc;
  partLoc[6] = nfnodeLoc;
  for (i = 0; i < 7; i++) {
   partMin[i] = 0;
   partMax[i] = 0;
   partSum[i] = 0;
  }

  ierr = MPI_Allreduce(partLoc,partMax,7,MPI_INT,MPI_MAX,MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(partLoc,partMin,7,MPI_INT,MPI_MIN,MPI_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(partLoc,partSum,7,MPI_INT,MPI_SUM,MPI_COMM_WORLD);CHKERRQ(ierr);
  PetscPrintf(MPI_COMM_WORLD, "==============================\n");
  PetscPrintf(MPI_COMM_WORLD, "Partitioning quality info ....\n");
  PetscPrintf(MPI_COMM_WORLD, "==============================\n");
  PetscPrintf(MPI_COMM_WORLD, "------------------------------------------------------------\n");
  PetscPrintf(MPI_COMM_WORLD, "Item                    Min        Max    Average      Total\n");
  PetscPrintf(MPI_COMM_WORLD, "------------------------------------------------------------\n");
  PetscPrintf(MPI_COMM_WORLD, "Local Nodes       %9d  %9d  %9d  %9d\n",
              partMin[0], partMax[0], partSum[0]/CommSize, partSum[0]);
  PetscPrintf(MPI_COMM_WORLD, "Local+Ghost Nodes %9d  %9d  %9d  %9d\n",
              partMin[1], partMax[1], partSum[1]/CommSize, partSum[1]);
  PetscPrintf(MPI_COMM_WORLD, "Local Edges       %9d  %9d  %9d  %9d\n",
              partMin[2], partMax[2], partSum[2]/CommSize, partSum[2]);
  PetscPrintf(MPI_COMM_WORLD, "Local solid faces %9d  %9d  %9d  %9d\n",
              partMin[3], partMax[3], partSum[3]/CommSize, partSum[3]);
  PetscPrintf(MPI_COMM_WORLD, "Local free faces  %9d  %9d  %9d  %9d\n",
              partMin[4], partMax[4], partSum[4]/CommSize, partSum[4]);
  PetscPrintf(MPI_COMM_WORLD, "Local solid nodes %9d  %9d  %9d  %9d\n",
              partMin[5], partMax[5], partSum[5]/CommSize, partSum[5]);
  PetscPrintf(MPI_COMM_WORLD, "Local free nodes  %9d  %9d  %9d  %9d\n",
              partMin[6], partMax[6], partSum[6]/CommSize, partSum[6]);
  PetscPrintf(MPI_COMM_WORLD, "------------------------------------------------------------\n");
 }
 ierr = OptionsHasName(0,"-partition_info",&flg);CHKERRQ(ierr);
 if (flg) {
  sprintf(part_name,"output.%d",rank);
  fptr1 = fopen(part_name,"w");

  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Local and Global Grid Parameters are :\n");
  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Local\t\t\t\tGlobal\n");
  fprintf(fptr1, "nnodesLoc = %d\t\tnnodes = %d\n", nnodesLoc, nnodes);
  fprintf(fptr1, "nedgeLoc = %d\t\t\tnedge = %d\n", nedgeLoc, nedge);
  fprintf(fptr1, "nnfacetLoc = %d\t\tnnfacet = %d\n", nnfacetLoc, nnfacet);
  fprintf(fptr1, "nvfacetLoc = %d\t\t\tnvfacet = %d\n", nvfacetLoc, nvfacet);
  fprintf(fptr1, "nffacetLoc = %d\t\t\tnffacet = %d\n", nffacetLoc, nffacet);
  fprintf(fptr1, "nsnodeLoc = %d\t\t\tnsnode = %d\n", nsnodeLoc, nsnode);
  fprintf(fptr1, "nvnodeLoc = %d\t\t\tnvnode = %d\n", nvnodeLoc, nvnode);
  fprintf(fptr1, "nfnodeLoc = %d\t\t\tnfnode = %d\n", nfnodeLoc, nfnode);
  fprintf(fptr1, "\n");
  fprintf(fptr1,"nvertices = %d\n", nvertices);
  fprintf(fptr1, "nnbound = %d\n", nnbound);
  fprintf(fptr1, "nvbound = %d\n", nvbound);
  fprintf(fptr1, "nfbound = %d\n", nfbound);
  fprintf(fptr1, "\n");

  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Different Orderings\n");
  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Local\t\tPETSc\t\tGlobal\n");
  fprintf(fptr1, "---------------------------------------------\n");
  for (i = 0; i < nvertices; i++) {
   fprintf(fptr1, "%d\t\t%d\t\t%d\n", i, grid->loc2pet[i], grid->loc2glo[i]);
  }
  fprintf(fptr1, "\n");

  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Solid Boundary Nodes\n");
  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Local\t\tPETSc\t\tGlobal\n");
  fprintf(fptr1, "---------------------------------------------\n");
  for (i = 0; i < nsnodeLoc; i++) {
   j = grid->isnode[i]-1;
   fprintf(fptr1, "%d\t\t%d\t\t%d\n", j, grid->loc2pet[j], grid->loc2glo[j]);
  }
  fprintf(fptr1, "\n");
  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "f2ntn array\n");
  fprintf(fptr1, "---------------------------------------------\n");
  for (i = 0; i < nnfacetLoc; i++) {
   fprintf(fptr1, "%d\t\t%d\t\t%d\t\t%d\n",i,grid->f2ntn[i],
           grid->f2ntn[nnfacetLoc+i], grid->f2ntn[2*nnfacetLoc+i]);
  }
  fprintf(fptr1, "\n");


  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Viscous Boundary Nodes\n");
  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Local\t\tPETSc\t\tGlobal\n");
  fprintf(fptr1, "---------------------------------------------\n");
  for (i = 0; i < nvnodeLoc; i++) {
   j = grid->ivnode[i]-1;
   fprintf(fptr1, "%d\t\t%d\t\t%d\n", j, grid->loc2pet[j], grid->loc2glo[j]);
  }
  fprintf(fptr1, "\n");
  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "f2ntv array\n");
  fprintf(fptr1, "---------------------------------------------\n");
  for (i = 0; i < nvfacetLoc; i++) {
   fprintf(fptr1, "%d\t\t%d\t\t%d\t\t%d\n",i,grid->f2ntv[i],
           grid->f2ntv[nvfacetLoc+i], grid->f2ntv[2*nvfacetLoc+i]);
  }
  fprintf(fptr1, "\n");

  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Free Boundary Nodes\n");
  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Local\t\tPETSc\t\tGlobal\n");
  fprintf(fptr1, "---------------------------------------------\n");
  for (i = 0; i < nfnodeLoc; i++) {
   j = grid->ifnode[i]-1;
   fprintf(fptr1, "%d\t\t%d\t\t%d\n", j, grid->loc2pet[j], grid->loc2glo[j]);
  }
  fprintf(fptr1, "\n");
  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "f2ntf array\n");
  fprintf(fptr1, "---------------------------------------------\n");
  for (i = 0; i < nffacetLoc; i++) {
   fprintf(fptr1, "%d\t\t%d\t\t%d\t\t%d\n",i,grid->f2ntf[i],
           grid->f2ntf[nffacetLoc+i], grid->f2ntf[2*nffacetLoc+i]);
  }
  fprintf(fptr1, "\n");

  fprintf(fptr1, "---------------------------------------------\n");
  fprintf(fptr1, "Neighborhood Info In Various Ordering\n");
  fprintf(fptr1, "---------------------------------------------\n");
  icalloc(nnodes, &p2l);
  for (i = 0; i < nvertices; i++)
   p2l[grid->loc2pet[i]] = i;
  for (i = 0; i < nnodesLoc; i++) {
   jstart = grid->ia[grid->loc2glo[i]] - 1;
   jend = grid->ia[grid->loc2glo[i]+1] - 1;
   fprintf(fptr1, "Neighbors of Node %d in Local Ordering are :", i);
   for (j = jstart; j < jend; j++) {
    fprintf(fptr1, "%d ", p2l[grid->ja[j]]);
   }
   fprintf(fptr1, "\n");

   fprintf(fptr1, "Neighbors of Node %d in PETSc ordering are :", grid->loc2pet[i]);
   for (j = jstart; j < jend; j++) {
    fprintf(fptr1, "%d ", grid->ja[j]);
   }
   fprintf(fptr1, "\n");

   fprintf(fptr1, "Neighbors of Node %d in Global Ordering are :", grid->loc2glo[i]);
   for (j = jstart; j < jend; j++) {
    fprintf(fptr1, "%d ", grid->loc2glo[p2l[grid->ja[j]]]);
   }
   fprintf(fptr1, "\n");
 
  }
  fprintf(fptr1, "\n");
  ierr = PetscFree(p2l);CHKERRQ(ierr);
  fclose(fptr1);
 }

/* Free the temporary arrays */
   ierr = PetscFree(a2l);CHKERRQ(ierr);
   ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*---------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "SetPetscDS"
int SetPetscDS(GRID *grid, TstepCtx *tsCtx)
/*---------------------------------------------------------------------*/
{
   int                    ierr, i, j, k, row, bs;
   int                    nnodes, nnz, jstart, jend, nbrs_diag, nbrs_offd;
   int                    nnodesLoc, nedgeLoc, nvertices;
   int                    *val_diag, *val_offd, *svertices, *loc2pet, *loc2glo;
   IS                     isglobal,islocal;
   ISLocalToGlobalMapping isl2g;
   PETSCTRUTH             flg;

  PetscFunctionBegin;
   nnodes = grid->nnodes;
   nnodesLoc = grid->nnodesLoc;
   nedgeLoc = grid->nedgeLoc;
   nvertices = grid->nvertices;
   loc2pet = grid->loc2pet;
   loc2glo = grid->loc2glo;
   bs = 4;
/* Set up the PETSc datastructures */
 
   ierr = VecCreateMPI(MPI_COMM_WORLD,bs*nnodesLoc,bs*nnodes,&grid->qnode);CHKERRQ(ierr);
   ierr = VecDuplicate(grid->qnode,&grid->res);CHKERRQ(ierr);
   ierr = VecDuplicate(grid->qnode,&tsCtx->qold);CHKERRQ(ierr);
   ierr = VecDuplicate(grid->qnode,&tsCtx->func);CHKERRQ(ierr);
   ierr = VecCreateSeq(MPI_COMM_SELF,bs*nvertices,&grid->qnodeLoc);CHKERRQ(ierr);
   ierr = VecCreateMPI(MPI_COMM_WORLD,3*bs*nnodesLoc,3*bs*nnodes,&grid->grad);
   ierr = VecCreateSeq(MPI_COMM_SELF,3*bs*nvertices,&grid->gradLoc);
/* Create Scatter between the local and global vectors */
/* First create scatter for qnode */
   ierr = ISCreateStride(MPI_COMM_SELF,bs*nvertices,0,1,&islocal);CHKERRQ(ierr);
#if defined(INTERLACING) 
#if defined(BLOCKING)
   icalloc(nvertices, &svertices);
   for (i=0; i < nvertices; i++) 
       svertices[i] = bs*loc2pet[i];
   ierr = ISCreateBlock(MPI_COMM_SELF,bs,nvertices,svertices,&isglobal);CHKERRQ(ierr);
#else
   icalloc(bs*nvertices, &svertices);
   for (i = 0; i < nvertices; i++)
     for (j = 0; j < bs; j++)
       svertices[j+bs*i] = j + bs*loc2pet[i];
   ierr = ISCreateGeneral(MPI_COMM_SELF,bs*nvertices,svertices,&isglobal);CHKERRQ(ierr);
#endif
#else
   icalloc(bs*nvertices, &svertices);
   for (j = 0; j < bs; j++)
    for (i = 0; i < nvertices; i++)
       svertices[j*nvertices+i] = j*nvertices + loc2pet[i];
   ierr = ISCreateGeneral(MPI_COMM_SELF,bs*nvertices,svertices,&isglobal);CHKERRQ(ierr);
#endif
   ierr = PetscFree(svertices);CHKERRQ(ierr);
   ierr = VecScatterCreate(grid->qnode,isglobal,grid->qnodeLoc,islocal,&grid->scatter);CHKERRQ(ierr);
   ierr = ISDestroy(isglobal);CHKERRQ(ierr);
   ierr = ISDestroy(islocal);CHKERRQ(ierr);

/* Now create scatter for gradient vector of qnode */
   ierr = ISCreateStride(MPI_COMM_SELF,3*bs*nvertices,0,1,&islocal);CHKERRQ(ierr);
#if defined(INTERLACING)
#if defined(BLOCKING)
   icalloc(nvertices, &svertices);
   for (i=0; i < nvertices; i++)
       svertices[i] = 3*bs*loc2pet[i];
   ierr = ISCreateBlock(MPI_COMM_SELF,3*bs,nvertices,svertices,&isglobal);CHKERRQ(ierr);
#else
   icalloc(3*bs*nvertices, &svertices);
   for (i = 0; i < nvertices; i++)
     for (j = 0; j < 3*bs; j++)
       svertices[j+3*bs*i] = j + 3*bs*loc2pet[i];
   ierr = ISCreateGeneral(MPI_COMM_SELF,3*bs*nvertices,svertices,&isglobal);CHKERRQ(ierr);
#endif
#else
   icalloc(3*bs*nvertices, &svertices);
   for (j = 0; j < 3*bs; j++)
    for (i = 0; i < nvertices; i++)
       svertices[j*nvertices+i] = j*nvertices + loc2pet[i];
   ierr = ISCreateGeneral(MPI_COMM_SELF,3*bs*nvertices,svertices,&isglobal);CHKERRQ(ierr);
#endif
   ierr = PetscFree(svertices);
   ierr = VecScatterCreate(grid->grad,isglobal,grid->gradLoc,islocal,&grid->gradScatter);CHKERRQ(ierr);
   ierr = ISDestroy(isglobal);CHKERRQ(ierr);
   ierr = ISDestroy(islocal);CHKERRQ(ierr);

/* Store the number of non-zeroes per row */
#if defined(INTERLACING)
#if defined(BLOCKING)
   icalloc(nnodesLoc, &val_diag);
   icalloc(nnodesLoc, &val_offd);
   for (i = 0; i < nnodesLoc; i++) {
    jstart = grid->ia[i] - 1;
    jend = grid->ia[i+1] - 1;
    nbrs_diag = 0;
    nbrs_offd = 0;
    for (j = jstart; j < jend; j++) {
      if ((grid->ja[j] >= rstart) && (grid->ja[j] < (rstart+nnodesLoc)))
         nbrs_diag++;
      else
         nbrs_offd++;
    }
    val_diag[i] = nbrs_diag; 
    val_offd[i] = nbrs_offd; 
   }
   ierr = MatCreateMPIBAIJ(MPI_COMM_WORLD,bs,bs*nnodesLoc, bs*nnodesLoc,
                             bs*nnodes,bs*nnodes,PETSC_NULL,val_diag,
                             PETSC_NULL,val_offd,&grid->A);CHKERRQ(ierr);
#else
   icalloc(nnodesLoc*4, &val_diag);
   icalloc(nnodesLoc*4, &val_offd);
   for (i = 0; i < nnodesLoc; i++) {
    jstart = grid->ia[i] - 1;
    jend = grid->ia[i+1] - 1;
    nbrs_diag = 0;
    nbrs_offd = 0;
    for (j = jstart; j < jend; j++) {
      if ((grid->ja[j] >= rstart) && (grid->ja[j] < (rstart+nnodesLoc)))
         nbrs_diag++;
      else
         nbrs_offd++;
    }
    for (j = 0; j < 4; j++) {
      row = 4*i + j;
      val_diag[row] = nbrs_diag*4; 
      val_offd[row] = nbrs_offd*4; 
    }
   }
   ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,bs*nnodesLoc, bs*nnodesLoc,
                             bs*nnodes,bs*nnodes,PETSC_NULL,val_diag,
                             PETSC_NULL,val_offd,&grid->A);CHKERRQ(ierr);
#endif
   ierr = PetscFree(val_diag);CHKERRQ(ierr);
   ierr = PetscFree(val_offd);CHKERRQ(ierr);

#else
   if (CommSize > 1) 
     SETERRQ(1,1,"Parallel case not supported in non-interlaced case\n");
   icalloc(nnodes*4, &val_diag);
   icalloc(nnodes*4, &val_offd);
   for (j = 0; j < 4; j++) {
    for (i = 0; i < nnodes; i++) {
      row = i + j*nnodes;
      jstart = grid->ia[i] - 1;
      jend = grid->ia[i+1] - 1;
      nbrs_diag = jend - jstart;
      val_diag[row] = nbrs_diag*4;
      val_offd[row] = 0;
    }
   }
   /* ierr = MatCreateSeqAIJ(MPI_COMM_SELF,nnodes*4,nnodes*4,PETSC_NULL,
                          val,&grid->A);*/
   ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,bs*nnodesLoc, bs*nnodesLoc,
                             bs*nnodes,bs*nnodes,PETSC_NULL,val_diag,
                             PETSC_NULL,val_offd,&grid->A);CHKERRQ(ierr);
   ierr = PetscFree(val_diag);CHKERRQ(ierr);
   ierr = PetscFree(val_offd);CHKERRQ(ierr);
#endif

   ierr = OptionsHasName(0,"-mem_use",&flg);CHKERRQ(ierr);
   if (flg) {
    PLogDouble space, maxSpace;
    ierr = PetscTrSpace(&space,0,&maxSpace);
    PetscPrintf(PETSC_COMM_WORLD,"Space allocated after allocating PETSc ");
    PetscPrintf(PETSC_COMM_WORLD,"data structures is %g\n", space);
    PetscPrintf(PETSC_COMM_WORLD,"Max space allocated so far is %g\n",
                maxSpace);
   }

/* Set local to global mapping for setting the matrix elements in
 * local ordering : first set row by row mapping
 */
#if defined(INTERLACING)
   icalloc(bs*nvertices, &svertices);
   k = 0;
   for (i=0; i < nvertices; i++)
     for (j=0; j < bs; j++)
       svertices[k++] = (bs*loc2pet[i] + j);
   /*ierr = MatSetLocalToGlobalMapping(grid->A,bs*nvertices,svertices);CHKERRQ(ierr);*/
   ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF,bs*nvertices,svertices,&isl2g);
   ierr = MatSetLocalToGlobalMapping(grid->A,isl2g);CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingDestroy(isl2g);CHKERRQ(ierr);

/* Now set the blockwise local to global mapping */
#if defined(BLOCKING)
   /*ierr = MatSetLocalToGlobalMappingBlock(grid->A,nvertices,loc2pet);CHKERRQ(ierr);*/
   ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF,nvertices,loc2pet,&isl2g);
   ierr = MatSetLocalToGlobalMappingBlock(grid->A,isl2g);CHKERRQ(ierr);
   ierr = ISLocalToGlobalMappingDestroy(isl2g);CHKERRQ(ierr);
#endif
   ierr = PetscFree(svertices);CHKERRQ(ierr);
#endif
   /*ierr = MatSetOption(grid->A, MAT_COLUMNS_SORTED);CHKERRQ(ierr);*/

   PetscFunctionReturn(0);
}

/*================================= CLINK ===================================*/
/*                                                                           */
/* Used in establishing the links between FORTRAN common blocks and C        */
/*                                                                           */
/*===========================================================================*/
EXTERN_C_BEGIN
#undef __FUNC__
#define __FUNC__ "f77CLINK"
void f77CLINK(CINFO *p1,CRUNGE *p2,CGMCOM *p3)
{
   c_info  = p1;
   c_runge = p2;
   c_gmcom = p3;
}
EXTERN_C_END

/*========================== SET_UP_GRID====================================*/
/*                                                                          */
/* Allocates the memory for the fine grid                                   */
/*                                                                          */
/*==========================================================================*/
#undef __FUNC__
#define __FUNC__ "set_up_grid"
int set_up_grid(GRID *grid)                           
{
   int nnodes, ncell, nedge;
   int nsface, nvface, nfface, nbface, kvisc;
   int vface, tnode;
   int nsnode, nvnode, nfnode;
   int ileast, lnodes;
   int nsrch, NoEq, ilu0, nnz;
   int mgzero=0; /* Variable so we dont allocate memory for multigrid */
   int valloc;
   int jalloc;  /* If jalloc=1 allocate space for dfp and dfm */
   int ifcn;
   int ierr;
/*
 * stuff to read in dave's grids
 */
   int nnbound,nvbound,nfbound,nnfacet,nvfacet,nffacet,ntte;
/* end of stuff */

   PetscFunctionBegin;
   nnodes = grid->nnodes;
   ncell  = grid->ncell;
   vface  = grid->nedge;
   tnode  = grid->nnodes;
   nedge  = grid->nedge;
   nsface = grid->nsface;
   nvface = grid->nvface;
   nfface = grid->nfface;
   nbface = nsface + nvface + nfface;
   nsnode = grid->nsnode;
   nvnode = grid->nvnode;
   nfnode = grid->nfnode;
   ileast = grid->ileast;
   lnodes = grid->nnodes;
   kvisc  = grid->jvisc;
   nsrch  = c_gmcom->nsrch;
   ilu0   = c_gmcom->ilu0;
   ifcn   = c_gmcom->ifcn;

   jalloc = 0;
   /*if(ilu0 >=1 && ifcn == 1)jalloc=0;*/

/*
 * stuff to read in dave's grids
 */
   nnbound = grid->nnbound;
   nvbound = grid->nvbound;
   nfbound = grid->nfbound;
   nnfacet = grid->nnfacet;
   nvfacet = grid->nvfacet;
   nffacet = grid->nffacet;
   ntte    = grid->ntte;
/* end of stuff */
   

   if (ileast == 0) lnodes = 1;
/*   printf("In set_up_grid->jvisc = %d\n",grid->jvisc); */

   if (grid->jvisc != 2 && grid->jvisc != 4 && grid->jvisc != 6)vface = 1;
/*   printf(" vface = %d \n",vface); */
   if (grid->jvisc < 3) tnode = 1;
   valloc = 1;
   if (grid->jvisc ==  0)valloc = 0;

/*PetscPrintf(MPI_COMM_WORLD, " nsnode= %d nvnode= %d nfnode= %d\n",nsnode,nvnode,nfnode);*/
/*PetscPrintf(MPI_COMM_WORLD, " nsface= %d nvface= %d nfface= %d\n",nsface,nvface,nfface);
PetscPrintf(MPI_COMM_WORLD, " nbface= %d\n", nbface);*/
/* Now allocate memory for the other grid arrays */
/* icalloc(nedge*2,   &grid->eptr); */
   icalloc(nsface,    &grid->isface);
   icalloc(nvface,    &grid->ivface);
   icalloc(nfface,    &grid->ifface);
/* icalloc(nsnode,    &grid->isnode);
   icalloc(nvnode,    &grid->ivnode);
   icalloc(nfnode,    &grid->ifnode);*/
   /*icalloc(nnodes,    &grid->clist);
   icalloc(nnodes,    &grid->iupdate);
   icalloc(nsface*2,  &grid->sface);
   icalloc(nvface*2,  &grid->vface);
   icalloc(nfface*2,  &grid->fface);
   icalloc(lnodes,    &grid->icount);*/
   /*fcalloc(nnodes,    &grid->x);
   fcalloc(nnodes,    &grid->y);
   fcalloc(nnodes,    &grid->z);
   fcalloc(nnodes,    &grid->area);*/
/*
 * fcalloc(nnodes*4,  &grid->gradx);
 * fcalloc(nnodes*4,  &grid->grady);
 * fcalloc(nnodes*4,  &grid->gradz);
 * fcalloc(nnodes,    &grid->cdt);
 */
/*
 * fcalloc(nnodes*4,  &grid->qnode);
 * fcalloc(nnodes*4,  &grid->dq);
 * fcalloc(nnodes*4,  &grid->res);
 * fcalloc(jalloc*nnodes*4*4, &grid->A);
 * fcalloc(nnodes*4,   &grid->B);
 * fcalloc(jalloc*nedge*4*4, &grid->dfp);
 * fcalloc(jalloc*nedge*4*4, &grid->dfm);
 */
   /*fcalloc(nsnode,    &grid->sxn);
   fcalloc(nsnode,    &grid->syn);
   fcalloc(nsnode,    &grid->szn);
   fcalloc(nsnode,    &grid->sa);
   fcalloc(nvnode,    &grid->vxn);
   fcalloc(nvnode,    &grid->vyn);
   fcalloc(nvnode,    &grid->vzn);
   fcalloc(nvnode,    &grid->va);
   fcalloc(nfnode,    &grid->fxn);
   fcalloc(nfnode,    &grid->fyn);
   fcalloc(nfnode,    &grid->fzn);
   fcalloc(nfnode,    &grid->fa);
   fcalloc(nedge,     &grid->xn);
   fcalloc(nedge,     &grid->yn);
   fcalloc(nedge,     &grid->zn);
   fcalloc(nedge,     &grid->rl);*/

   fcalloc(nbface*15, &grid->us);
   fcalloc(nbface*15, &grid->vs);
   fcalloc(nbface*15, &grid->as);
/*
 * fcalloc(nnodes*4,  &grid->phi);
 * fcalloc(nnodes,    &grid->r11);
 * fcalloc(nnodes,    &grid->r12);
 * fcalloc(nnodes,    &grid->r13);
 * fcalloc(nnodes,    &grid->r22);
 * fcalloc(nnodes,    &grid->r23);
 * fcalloc(nnodes,    &grid->r33);
 */
/*
 * Allocate memory for viscous length scale if turbulent 
 */
   if (grid->jvisc >= 3) {
    fcalloc(tnode,   &grid->slen);
    fcalloc(nnodes,  &grid->turbre);
    fcalloc(nnodes,  &grid->amut);
    fcalloc(tnode,   &grid->turbres);
    fcalloc(nedge,   &grid->dft1);
    fcalloc(nedge,   &grid->dft2);
   }
/*
 * Allocate memory for MG transfer 
 */
/*
 * icalloc(mgzero*nsface,    &grid->isford);
 * icalloc(mgzero*nvface,    &grid->ivford);
 * icalloc(mgzero*nfface,    &grid->ifford);
 * icalloc(mgzero*nnodes,    &grid->nflag);
 * icalloc(mgzero*nnodes,    &grid->nnext);
 * icalloc(mgzero*nnodes,    &grid->nneigh);
 * icalloc(mgzero*ncell,     &grid->ctag);
 * icalloc(mgzero*ncell,     &grid->csearch);
 * icalloc(valloc*ncell*4,   &grid->c2n);
 * icalloc(valloc*ncell*6,   &grid->c2e);
 * grid->c2c = (int *)grid->dfp;
 * icalloc(ncell*4,   &grid->c2c); 
 * icalloc(nnodes,    &grid->cenc);
 * if (grid->iup == 1) {
 *    icalloc(mgzero*nnodes*3,  &grid->icoefup);
 *    fcalloc(mgzero*nnodes*3,  &grid->rcoefup);
 * }
 * if (grid->idown == 1) {
 *    icalloc(mgzero*nnodes*3,  &grid->icoefdn);
 *    fcalloc(mgzero*nnodes*3,  &grid->rcoefdn);
 * }
 * fcalloc(nnodes*4,  &grid->ff);
 * fcalloc(tnode,     &grid->turbff);
 */
/*
 * If using GMRES (nsrch>0) allocate memory
 */
/* NoEq = 0;
*  if(nsrch > 0)NoEq = 4*nnodes;
*  if(nsrch < 0)NoEq = nnodes;
*  fcalloc(NoEq,           &grid->AP);
*  fcalloc(NoEq,           &grid->Xgm);
*  fcalloc(NoEq,           &grid->temr);
*  fcalloc((abs(nsrch)+1)*NoEq, &grid->Fgm);
*/
/*
 * stuff to read in dave's grids
 */
/*
 * icalloc(nnbound,   &grid->ncolorn);
 * icalloc(nnbound*100,&grid->countn);
 * icalloc(nvbound,   &grid->ncolorv);
 * icalloc(nvbound*100,&grid->countv);
 * icalloc(nfbound,   &grid->ncolorf);
 * icalloc(nfbound*100,&grid->countf);
 */
   /*icalloc(nnbound,   &grid->nntet);
   icalloc(nnbound,   &grid->nnpts);
   icalloc(nvbound,   &grid->nvtet);
   icalloc(nvbound,   &grid->nvpts);
   icalloc(nfbound,   &grid->nftet);
   icalloc(nfbound,   &grid->nfpts);
   icalloc(nnfacet*4, &grid->f2ntn);
   icalloc(nvfacet*4, &grid->f2ntv);
   icalloc(nffacet*4, &grid->f2ntf);*/
  PetscFunctionReturn(0);
}

 
/*========================== WRITE_FINE_GRID ================================*/
/*                                                                           */
/* Write memory locations and other information for the fine grid            */
/*                                                                           */
/*===========================================================================*/
#undef __FUNC__
#define __FUNC__ "write_fine_grid"
int write_fine_grid(GRID *grid)                  
{
   int  i;
   FILE *output;

  PetscFunctionBegin;
/* open file for output      */
/* call the output frame.out */

   if (!(output = fopen("frame.out","a"))){
      SETERRQ(1,1,"can't open frame.out");
   }
   fprintf(output,"information for fine grid\n"); 
   fprintf(output,"\n");
   fprintf(output," address of fine grid = %p\n", grid);

   fprintf(output,"grid.nnodes  = %d\n", grid->nnodes);
   fprintf(output,"grid.ncell   = %d\n", grid->ncell);
   fprintf(output,"grid.nedge   = %d\n", grid->nedge);
   fprintf(output,"grid.nsface  = %d\n", grid->nsface);
   fprintf(output,"grid.nvface  = %d\n", grid->nvface);
   fprintf(output,"grid.nfface  = %d\n", grid->nfface);
   fprintf(output,"grid.nsnode  = %d\n", grid->nsnode);
   fprintf(output,"grid.nvnode  = %d\n", grid->nvnode);
   fprintf(output,"grid.nfnode  = %d\n", grid->nfnode);

   fprintf(output,"grid.eptr    = %p\n", grid->eptr);
   fprintf(output,"grid.isface  = %p\n", grid->isface);
   fprintf(output,"grid.ivface  = %p\n", grid->ivface);
   fprintf(output,"grid.ifface  = %p\n", grid->ifface);
   fprintf(output,"grid.isnode  = %p\n", grid->isnode);
   fprintf(output,"grid.ivnode  = %p\n", grid->ivnode);
   fprintf(output,"grid.ifnode  = %p\n", grid->ifnode);
   fprintf(output,"grid.c2n     = %p\n", grid->c2n);
   fprintf(output,"grid.c2e     = %p\n", grid->c2e);
   fprintf(output,"grid.xyz     = %p\n", grid->xyz);
   /*fprintf(output,"grid.y       = %p\n", grid->xyz);
     fprintf(output,"grid.z       = %p\n", grid->z);*/
   fprintf(output,"grid.area    = %p\n", grid->area);
   fprintf(output,"grid.qnode   = %p\n", grid->qnode);
/*
   fprintf(output,"grid.gradx   = %p\n", grid->gradx);
   fprintf(output,"grid.grady   = %p\n", grid->grady);
   fprintf(output,"grid.gradz   = %p\n", grid->gradz);
*/
   fprintf(output,"grid.cdt     = %p\n", grid->cdt);
   fprintf(output,"grid.sxn     = %p\n", grid->sxn);
   fprintf(output,"grid.syn     = %p\n", grid->syn);
   fprintf(output,"grid.szn     = %p\n", grid->szn);
   fprintf(output,"grid.vxn     = %p\n", grid->vxn);
   fprintf(output,"grid.vyn     = %p\n", grid->vyn);
   fprintf(output,"grid.vzn     = %p\n", grid->vzn);
   fprintf(output,"grid.fxn     = %p\n", grid->fxn);
   fprintf(output,"grid.fyn     = %p\n", grid->fyn);
   fprintf(output,"grid.fzn     = %p\n", grid->fzn);
   fprintf(output,"grid.xyzn    = %p\n", grid->xyzn);
   /*fprintf(output,"grid.yn      = %p\n", grid->yn);
   fprintf(output,"grid.zn      = %p\n", grid->zn);
   fprintf(output,"grid.rl      = %p\n", grid->rl);*/
/*
 * close output file
 */
   fclose(output);
   PetscFunctionReturn(0);
}


 
/*========================== FCALLOC =======================================*/
/*                                                                          */
/* Allocates memory for REALing point numbers                              */
/*                                                                          */
/*==========================================================================*/
#undef __FUNC__
#define __FUNC__ "fcalloc"
void fcalloc(int size,REAL **p)
{
   int rsize;
 
   rsize = PetscMax(size,1);
   memSize+=rsize*sizeof(REAL);
   *p = (REAL *)PetscMalloc(rsize*sizeof(REAL)); 
   if (!*p) {
    PLogDouble space, maxSpace;
    int        ierr;
    ierr = PetscTrSpace(&space,0,&maxSpace);
    PetscPrintf(PETSC_COMM_WORLD,"Space allocated currently is %g\n",space);
    PetscPrintf(PETSC_COMM_WORLD,"Max space allocated so far is %g\n",maxSpace);
   }
   CHKPTRA(*p);
}
 
 
/*========================== ICALLOC =======================================*/
/*                                                                          */
/* Allocates memory for integers                                            */
/*                                                                          */
/*==========================================================================*/
#undef __FUNC__
#define __FUNC__ "icalloc"
void icalloc(int size,int **p)
{
   int rsize;
   rsize = PetscMax(size,1);
   memSize+=rsize*sizeof(int);
   *p = (int *)PetscMalloc(rsize*sizeof(int)); 
   if (!*p) {
    PLogDouble space, maxSpace;
    int        ierr;
    ierr = PetscTrSpace(&space,0,&maxSpace);
    PetscPrintf(PETSC_COMM_WORLD,"Space allocated currently is %g\n",space);
    PetscPrintf(PETSC_COMM_WORLD,"Max space allocated so far is %g\n",maxSpace);
   }
   CHKPTRA(*p);
}
#if defined (PARCH_IRIX64) && defined(USE_HW_COUNTERS)
int EventCountersBegin(int *gen_start, Scalar* time_start_counters)
{
 int ierr;
 if ((*gen_start = start_counters(event0,event1)) < 0)
   SETERRQ(1,1,"Error in start_counters\n");
 ierr = PetscGetTime(time_start_counters);CHKERRQ(ierr);
 return 0;
}

int EventCountersEnd(int gen_start, Scalar time_start_counters) 
{
 int gen_read, ierr;
 Scalar time_read_counters;
 long long _counter0, _counter1;

 if ((gen_read = read_counters(event0,&_counter0,event1,&_counter1)) < 0)
   SETERRQ(1,1,"Error in read_counter\n");
 ierr = PetscGetTime(&time_read_counters);CHKERRQ(ierr);
 if (gen_read != gen_start) {
   SETERRQ(1,1,"Lost Counters!! Aborting ...\n");
 }
 counter0 += _counter0;
 counter1 += _counter1;
 time_counters += time_read_counters-time_start_counters;
 return 0;
}
#endif
