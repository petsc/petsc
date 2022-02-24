
static char help[] = "FUN3D - 3-D, Unstructured Incompressible Euler Solver.\n\
originally written by W. K. Anderson of NASA Langley, \n\
and ported into PETSc by D. K. Kaushik, ODU and ICASE.\n\n";

#include <petscsnes.h>
#include <petsctime.h>
#include <petscao.h>
#include "user.h"
#if defined(_OPENMP)
#include "omp.h"
#if !defined(HAVE_REDUNDANT_WORK)
#include "metis.h"
#endif
#endif

#define ICALLOC(size,y) CHKERRQ(PetscMalloc1(PetscMax(size,1),y));
#define FCALLOC(size,y) CHKERRQ(PetscMalloc1(PetscMax(size,1),y));

typedef struct {
  Vec    qnew,qold,func;
  double fnorm_ini,dt_ini,cfl_ini;
  double ptime;
  double cfl_max,max_time;
  double fnorm,dt,cfl;
  double fnorm_ratio;
  int    ires,iramp,itstep;
  int    max_steps,print_freq;
  int    LocalTimeStepping;
} TstepCtx;

typedef struct {                               /*============================*/
  GRID      *grid;                                 /* Pointer to Grid info       */
  TstepCtx  *tsCtx;                                /* Pointer to Time Stepping Context */
  PetscBool PreLoading;
} AppCtx;                                      /*============================*/

extern int  FormJacobian(SNES,Vec,Mat,Mat,void*),
            FormFunction(SNES,Vec,Vec,void*),
            FormInitialGuess(SNES,GRID*),
            Update(SNES,void*),
            ComputeTimeStep(SNES,int,void*),
            GetLocalOrdering(GRID*),
            SetPetscDS(GRID *,TstepCtx*);
static PetscErrorCode WritePVTU(AppCtx*,const char*,PetscBool);
#if defined(_OPENMP) && defined(HAVE_EDGE_COLORING)
int EdgeColoring(int nnodes,int nedge,int *e2n,int *eperm,int *ncolor,int *ncount);
#endif
/* Global Variables */

                                               /*============================*/
CINFO  *c_info;                                /* Pointer to COMMON INFO     */
CRUNGE *c_runge;                               /* Pointer to COMMON RUNGE    */
CGMCOM *c_gmcom;                               /* Pointer to COMMON GMCOM    */
                                               /*============================*/
int  rank,size,rstart;
REAL memSize = 0.0,grad_time = 0.0;
#if defined(_OPENMP)
int max_threads = 2,tot_threads,my_thread_id;
#endif

#if defined(PARCH_IRIX64) && defined(USE_HW_COUNTERS)
int       event0,event1;
Scalar    time_counters;
long long counter0,counter1;
#endif
int  ntran[max_nbtran];        /* transition stuff put here to make global */
REAL dxtran[max_nbtran];

/* ======================== MAIN ROUTINE =================================== */
/*                                                                           */
/* Finite volume flux split solver for general polygons                      */
/*                                                                           */
/*===========================================================================*/

int main(int argc,char **args)
{
  AppCtx      user;
  GRID        f_pntr;
  TstepCtx    tsCtx;
  SNES        snes;                    /* nonlinear solver context */
  Mat         Jpc;                     /* Jacobian and Preconditioner matrices */
  PetscScalar *qnode;
  int         ierr;
  PetscBool   flg,write_pvtu,pvtu_base64;
  MPI_Comm    comm;
  PetscInt    maxfails                       = 10000;
  char        pvtu_fname[PETSC_MAX_PATH_LEN] = "incomp";

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscInitializeFortran());
  CHKERRQ(PetscOptionsInsertFile(PETSC_COMM_WORLD,"petsc.opt",PETSC_FALSE));

  comm = PETSC_COMM_WORLD;
  f77FORLINK();                               /* Link FORTRAN and C COMMONS */

  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-mem_use",&flg,NULL));
  if (flg) CHKERRQ(PetscMemorySetGetMaximumUsage());

  /*======================================================================*/
  /* Initilize stuff related to time stepping */
  /*======================================================================*/
  tsCtx.fnorm_ini         = 0.0;  tsCtx.cfl_ini     = 50.0;    tsCtx.cfl_max = 1.0e+05;
  tsCtx.max_steps         = 50;   tsCtx.max_time    = 1.0e+12; tsCtx.iramp   = -50;
  tsCtx.dt                = -5.0; tsCtx.fnorm_ratio = 1.0e+10;
  tsCtx.LocalTimeStepping = 1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-max_st",&tsCtx.max_steps,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,"-ts_rtol",&tsCtx.fnorm_ratio,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,"-cfl_ini",&tsCtx.cfl_ini,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,"-cfl_max",&tsCtx.cfl_max,NULL));
  tsCtx.print_freq        = tsCtx.max_steps;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-print_freq",&tsCtx.print_freq,&flg));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-pvtu",pvtu_fname,sizeof(pvtu_fname),&write_pvtu));
  pvtu_base64             = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-pvtu_base64",&pvtu_base64,NULL));

  c_info->alpha = 3.0;
  c_info->beta  = 15.0;
  c_info->ivisc = 0;

  c_gmcom->ilu0  = 1;
  c_gmcom->nsrch = 10;

  c_runge->nitfo = 0;

  CHKERRQ(PetscMemzero(&f_pntr,sizeof(f_pntr)));
  f_pntr.jvisc  = c_info->ivisc;
  f_pntr.ileast = 4;
  CHKERRQ(PetscOptionsGetReal(NULL,"-alpha",&c_info->alpha,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,"-beta",&c_info->beta,NULL));

  /*======================================================================*/

  /*Set the maximum number of threads for OpenMP */
#if defined(_OPENMP)
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-max_threads",&max_threads,&flg));
  omp_set_num_threads(max_threads);
  CHKERRQ(PetscPrintf(comm,"Using %d threads for each MPI process\n",max_threads));
#endif

  /* Get the grid information into local ordering */
  CHKERRQ(GetLocalOrdering(&f_pntr));

  /* Allocate Memory for Some Other Grid Arrays */
  CHKERRQ(set_up_grid(&f_pntr));

  /* If using least squares for the gradients,calculate the r's */
  if (f_pntr.ileast == 4) f77SUMGS(&f_pntr.nnodesLoc,&f_pntr.nedgeLoc,f_pntr.eptr,f_pntr.xyz,f_pntr.rxy,&rank,&f_pntr.nvertices);

  user.grid  = &f_pntr;
  user.tsCtx = &tsCtx;

  /* SAWs Stuff */

  /*
    Preload the executable to get accurate timings. This runs the following chunk of
    code twice, first to get the executable pages into memory and the second time for
    accurate timings.
  */
  PetscPreLoadBegin(PETSC_TRUE,"Time integration");
  user.PreLoading = PetscPreLoading;

  /* Create nonlinear solver */
  CHKERRQ(SetPetscDS(&f_pntr,&tsCtx));
  CHKERRQ(SNESCreate(comm,&snes));
  CHKERRQ(SNESSetType(snes,"newtonls"));

  /* Set various routines and options */
  CHKERRQ(SNESSetFunction(snes,user.grid->res,FormFunction,&user));
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-matrix_free",&flg,NULL));
  if (flg) {
    /* Use matrix-free to define Newton system; use explicit (approx) Jacobian for preconditioner */
    CHKERRQ(MatCreateSNESMF(snes,&Jpc));
    CHKERRQ(SNESSetJacobian(snes,Jpc,user.grid->A,FormJacobian,&user));
  } else {
    /* Use explicit (approx) Jacobian to define Newton system and preconditioner */
    CHKERRQ(SNESSetJacobian(snes,user.grid->A,user.grid->A,FormJacobian,&user));
  }

  CHKERRQ(SNESSetMaxLinearSolveFailures(snes,maxfails));
  CHKERRQ(SNESSetFromOptions(snes));

  /* Initialize the flowfield */
  CHKERRQ(FormInitialGuess(snes,user.grid));

  /* Solve nonlinear system */
  CHKERRQ(Update(snes,&user));

  /* Write restart file */
  CHKERRQ(VecGetArray(user.grid->qnode,&qnode));
  /*f77WREST(&user.grid->nnodes,qnode,user.grid->turbre,user.grid->amut);*/

  /* Write Tecplot solution file */
#if 0
  if (rank == 0)
    f77TECFLO(&user.grid->nnodes,
              &user.grid->nnbound,&user.grid->nvbound,&user.grid->nfbound,
              &user.grid->nnfacet,&user.grid->nvfacet,&user.grid->nffacet,
              &user.grid->nsnode, &user.grid->nvnode, &user.grid->nfnode,
              c_info->title,
              user.grid->x,       user.grid->y,       user.grid->z,
              qnode,
              user.grid->nnpts,   user.grid->nntet,   user.grid->nvpts,
              user.grid->nvtet,   user.grid->nfpts,   user.grid->nftet,
              user.grid->f2ntn,   user.grid->f2ntv,   user.grid->f2ntf,
              user.grid->isnode,  user.grid->ivnode,  user.grid->ifnode,
              &rank);
#endif
  if (write_pvtu) CHKERRQ(WritePVTU(&user,pvtu_fname,pvtu_base64));

  /* Write residual,lift,drag,and moment history file */
  /*
    if (rank == 0) f77PLLAN(&user.grid->nnodes,&rank);
  */

  CHKERRQ(VecRestoreArray(user.grid->qnode,&qnode));
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-mem_use",&flg,NULL));
  if (flg) {
    CHKERRQ(PetscMemoryView(PETSC_VIEWER_STDOUT_WORLD,"Memory usage before destroying\n"));
  }

  CHKERRQ(VecDestroy(&user.grid->qnode));
  CHKERRQ(VecDestroy(&user.grid->qnodeLoc));
  CHKERRQ(VecDestroy(&user.tsCtx->qold));
  CHKERRQ(VecDestroy(&user.tsCtx->func));
  CHKERRQ(VecDestroy(&user.grid->res));
  CHKERRQ(VecDestroy(&user.grid->grad));
  CHKERRQ(VecDestroy(&user.grid->gradLoc));
  CHKERRQ(MatDestroy(&user.grid->A));
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-matrix_free",&flg,NULL));
  if (flg) CHKERRQ(MatDestroy(&Jpc));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(VecScatterDestroy(&user.grid->scatter));
  CHKERRQ(VecScatterDestroy(&user.grid->gradScatter));
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-mem_use",&flg,NULL));
  if (flg) {
    CHKERRQ(PetscMemoryView(PETSC_VIEWER_STDOUT_WORLD,"Memory usage after destroying\n"));
  }
  PetscPreLoadEnd();

  /* allocated in set_up_grid() */
  CHKERRQ(PetscFree(user.grid->isface));
  CHKERRQ(PetscFree(user.grid->ivface));
  CHKERRQ(PetscFree(user.grid->ifface));
  CHKERRQ(PetscFree(user.grid->us));
  CHKERRQ(PetscFree(user.grid->vs));
  CHKERRQ(PetscFree(user.grid->as));

  /* Allocated in GetLocalOrdering() */
  CHKERRQ(PetscFree(user.grid->eptr));
  CHKERRQ(PetscFree(user.grid->ia));
  CHKERRQ(PetscFree(user.grid->ja));
  CHKERRQ(PetscFree(user.grid->loc2glo));
  CHKERRQ(PetscFree(user.grid->loc2pet));
  CHKERRQ(PetscFree(user.grid->xyzn));
#if defined(_OPENMP)
#  if defined(HAVE_REDUNDANT_WORK)
  CHKERRQ(PetscFree(user.grid->resd));
#  else
  CHKERRQ(PetscFree(user.grid->part_thr));
  CHKERRQ(PetscFree(user.grid->nedge_thr));
  CHKERRQ(PetscFree(user.grid->edge_thr));
  CHKERRQ(PetscFree(user.grid->xyzn_thr));
#  endif
#endif
  CHKERRQ(PetscFree(user.grid->xyz));
  CHKERRQ(PetscFree(user.grid->area));

  CHKERRQ(PetscFree(user.grid->nntet));
  CHKERRQ(PetscFree(user.grid->nnpts));
  CHKERRQ(PetscFree(user.grid->f2ntn));
  CHKERRQ(PetscFree(user.grid->isnode));
  CHKERRQ(PetscFree(user.grid->sxn));
  CHKERRQ(PetscFree(user.grid->syn));
  CHKERRQ(PetscFree(user.grid->szn));
  CHKERRQ(PetscFree(user.grid->sa));
  CHKERRQ(PetscFree(user.grid->sface_bit));

  CHKERRQ(PetscFree(user.grid->nvtet));
  CHKERRQ(PetscFree(user.grid->nvpts));
  CHKERRQ(PetscFree(user.grid->f2ntv));
  CHKERRQ(PetscFree(user.grid->ivnode));
  CHKERRQ(PetscFree(user.grid->vxn));
  CHKERRQ(PetscFree(user.grid->vyn));
  CHKERRQ(PetscFree(user.grid->vzn));
  CHKERRQ(PetscFree(user.grid->va));
  CHKERRQ(PetscFree(user.grid->vface_bit));

  CHKERRQ(PetscFree(user.grid->nftet));
  CHKERRQ(PetscFree(user.grid->nfpts));
  CHKERRQ(PetscFree(user.grid->f2ntf));
  CHKERRQ(PetscFree(user.grid->ifnode));
  CHKERRQ(PetscFree(user.grid->fxn));
  CHKERRQ(PetscFree(user.grid->fyn));
  CHKERRQ(PetscFree(user.grid->fzn));
  CHKERRQ(PetscFree(user.grid->fa));
  CHKERRQ(PetscFree(user.grid->cdt));
  CHKERRQ(PetscFree(user.grid->phi));
  CHKERRQ(PetscFree(user.grid->rxy));

  CHKERRQ(PetscPrintf(comm,"Time taken in gradient calculation %g sec.\n",grad_time));

  ierr = PetscFinalize();
  return ierr;
}

/*---------------------------------------------------------------------*/
/* ---------------------  Form initial approximation ----------------- */
int FormInitialGuess(SNES snes,GRID *grid)
/*---------------------------------------------------------------------*/
{
  int         ierr;
  PetscScalar *qnode;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(grid->qnode,&qnode));
  f77INIT(&grid->nnodesLoc,qnode,grid->turbre,grid->amut,&grid->nvnodeLoc,grid->ivnode,&rank);
  CHKERRQ(VecRestoreArray(grid->qnode,&qnode));
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
/* ---------------------  Evaluate Function F(x) --------------------- */
int FormFunction(SNES snes,Vec x,Vec f,void *dummy)
/*---------------------------------------------------------------------*/
{
  AppCtx      *user  = (AppCtx*) dummy;
  GRID        *grid  = user->grid;
  TstepCtx    *tsCtx = user->tsCtx;
  PetscScalar *qnode,*res,*qold;
  PetscScalar *grad;
  PetscScalar temp;
  VecScatter  scatter     = grid->scatter;
  VecScatter  gradScatter = grid->gradScatter;
  Vec         localX      = grid->qnodeLoc;
  Vec         localGrad   = grid->gradLoc;
  int         i,j,in,ierr;
  int         nbface,ires;
  PetscScalar time_ini,time_fin;

  PetscFunctionBegin;
  /* Get X into the local work vector */
  CHKERRQ(VecScatterBegin(scatter,x,localX,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(scatter,x,localX,INSERT_VALUES,SCATTER_FORWARD));
  /* VecCopy(x,localX); */
  /* access the local work f,grad,and input */
  CHKERRQ(VecGetArray(f,&res));
  CHKERRQ(VecGetArray(grid->grad,&grad));
  CHKERRQ(VecGetArray(localX,&qnode));
  ires = tsCtx->ires;

  CHKERRQ(PetscTime(&time_ini));
  f77LSTGS(&grid->nnodesLoc,&grid->nedgeLoc,grid->eptr,qnode,grad,grid->xyz,grid->rxy,
           &rank,&grid->nvertices);
  CHKERRQ(PetscTime(&time_fin));
  grad_time += time_fin - time_ini;
  CHKERRQ(VecRestoreArray(grid->grad,&grad));

  CHKERRQ(VecScatterBegin(gradScatter,grid->grad,localGrad,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(gradScatter,grid->grad,localGrad,INSERT_VALUES,SCATTER_FORWARD));
  /*VecCopy(grid->grad,localGrad);*/

  CHKERRQ(VecGetArray(localGrad,&grad));
  nbface = grid->nsface + grid->nvface + grid->nfface;
  f77GETRES(&grid->nnodesLoc,&grid->ncell,  &grid->nedgeLoc,  &grid->nsface,
            &grid->nvface,&grid->nfface, &nbface,
            &grid->nsnodeLoc,&grid->nvnodeLoc, &grid->nfnodeLoc,
            grid->isface, grid->ivface,  grid->ifface, &grid->ileast,
            grid->isnode, grid->ivnode,  grid->ifnode,
            &grid->nnfacetLoc,grid->f2ntn,  &grid->nnbound,
            &grid->nvfacetLoc,grid->f2ntv,  &grid->nvbound,
            &grid->nffacetLoc,grid->f2ntf,  &grid->nfbound,
            grid->eptr,
            grid->sxn,    grid->syn,     grid->szn,
            grid->vxn,    grid->vyn,     grid->vzn,
            grid->fxn,    grid->fyn,     grid->fzn,
            grid->xyzn,
            qnode,        grid->cdt,
            grid->xyz,    grid->area,
            grad, res,
            grid->turbre,
            grid->slen,   grid->c2n,
            grid->c2e,
            grid->us,     grid->vs,      grid->as,
            grid->phi,
            grid->amut,   &ires,
#if defined(_OPENMP)
            &max_threads,
#if defined(HAVE_EDGE_COLORING)
            &grid->ncolor, grid->ncount,
#elif defined(HAVE_REDUNDANT_WORK)
            grid->resd,
#else
            &grid->nedgeAllThr,
            grid->part_thr,grid->nedge_thr,grid->edge_thr,grid->xyzn_thr,
#endif
#endif
            &tsCtx->LocalTimeStepping,&rank,&grid->nvertices);

/* Add the contribution due to time stepping */
  if (ires == 1) {
    CHKERRQ(VecGetArray(tsCtx->qold,&qold));
#if defined(INTERLACING)
    for (i = 0; i < grid->nnodesLoc; i++) {
      temp = grid->area[i]/(tsCtx->cfl*grid->cdt[i]);
      for (j = 0; j < 4; j++) {
        in       = 4*i + j;
        res[in] += temp*(qnode[in] - qold[in]);
      }
    }
#else
    for (j = 0; j < 4; j++) {
      for (i = 0; i < grid->nnodesLoc; i++) {
        temp     = grid->area[i]/(tsCtx->cfl*grid->cdt[i]);
        in       = grid->nnodesLoc*j + i;
        res[in] += temp*(qnode[in] - qold[in]);
      }
    }
#endif
    CHKERRQ(VecRestoreArray(tsCtx->qold,&qold));
  }
  CHKERRQ(VecRestoreArray(localX,&qnode));
  CHKERRQ(VecRestoreArray(f,&res));
  CHKERRQ(VecRestoreArray(localGrad,&grad));
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

int FormJacobian(SNES snes,Vec x,Mat Jac,Mat pc_mat,void *dummy)
/*---------------------------------------------------------------------*/
{
  AppCtx      *user  = (AppCtx*) dummy;
  GRID        *grid  = user->grid;
  TstepCtx    *tsCtx = user->tsCtx;
  Vec         localX = grid->qnodeLoc;
  PetscScalar *qnode;
  int         ierr;

  PetscFunctionBegin;
  /*  CHKERRQ(VecScatterBegin(scatter,x,localX,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(scatter,x,localX,INSERT_VALUES,SCATTER_FORWARD)); */
  /* VecCopy(x,localX); */
  CHKERRQ(MatSetUnfactored(pc_mat));

  CHKERRQ(VecGetArray(localX,&qnode));
  f77FILLA(&grid->nnodesLoc,&grid->nedgeLoc,grid->eptr,
           &grid->nsface,
            grid->isface,grid->fxn,grid->fyn,grid->fzn,
            grid->sxn,grid->syn,grid->szn,
           &grid->nsnodeLoc,&grid->nvnodeLoc,&grid->nfnodeLoc,grid->isnode,
            grid->ivnode,grid->ifnode,qnode,&pc_mat,grid->cdt,
            grid->area,grid->xyzn,&tsCtx->cfl,
           &rank,&grid->nvertices);
  CHKERRQ(VecRestoreArray(localX,&qnode));
  CHKERRQ(MatAssemblyBegin(Jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Jac,MAT_FINAL_ASSEMBLY));
#if defined(MATRIX_VIEW)
  if ((tsCtx->itstep != 0) &&(tsCtx->itstep % tsCtx->print_freq) == 0) {
    PetscViewer viewer;
    char mat_file[PETSC_MAX_PATH_LEN];
    sprintf(mat_file,"mat_bin.%d",tsCtx->itstep);
    ierr = PetscViewerBinaryOpen(MPI_COMM_WORLD,mat_file,FILE_MODE_WRITE,&viewer);
    CHKERRQ(MatView(pc_mat,viewer));
    ierr = PetscViewerDestroy(&viewer);
  }
#endif
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
int Update(SNES snes,void *ctx)
/*---------------------------------------------------------------------*/
{
  AppCtx      *user   = (AppCtx*) ctx;
  GRID        *grid   = user->grid;
  TstepCtx    *tsCtx  = user->tsCtx;
  VecScatter  scatter = grid->scatter;
  Vec         localX  = grid->qnodeLoc;
  PetscScalar *qnode,*res;
  PetscScalar clift,cdrag,cmom;
  int         ierr,its;
  PetscScalar fratio;
  PetscScalar time1,time2,cpuloc,cpuglo;
  int         max_steps;
  PetscBool   print_flag = PETSC_FALSE;
  FILE        *fptr      = 0;
  int         nfailsCum  = 0,nfails = 0;
  /*Scalar         cpu_ini,cpu_fin,cpu_time;*/
  /*int            event0 = 14,event1 = 25,gen_start,gen_read;
  PetscScalar    time_start_counters,time_read_counters;
  long long      counter0,counter1;*/

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-print",&print_flag,NULL));
  if (print_flag) {
    CHKERRQ(PetscFOpen(PETSC_COMM_WORLD,"history.out","w",&fptr));
    CHKERRQ(PetscFPrintf(PETSC_COMM_WORLD,fptr,"VARIABLES = iter,cfl,fnorm,clift,cdrag,cmom,cpu\n"));
  }
  if (user->PreLoading) max_steps = 1;
  else max_steps = tsCtx->max_steps;
  fratio = 1.0;
  /*tsCtx->ptime = 0.0;*/
  CHKERRQ(VecCopy(grid->qnode,tsCtx->qold));
  CHKERRQ(PetscTime(&time1));
#if defined(PARCH_IRIX64) && defined(USE_HW_COUNTERS)
  /* if (!user->PreLoading) {
    PetscBool  flg = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-e0",&event0,&flg));
    CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-e1",&event1,&flg));
    CHKERRQ(PetscTime(&time_start_counters));
    if ((gen_start = start_counters(event0,event1)) < 0)
    SETERRQ(PETSC_COMM_SELF,1,>"Error in start_counters");
  }*/
#endif
  /*cpu_ini = PetscGetCPUTime();*/
  for (tsCtx->itstep = 0; (tsCtx->itstep < max_steps) &&
        (fratio <= tsCtx->fnorm_ratio); tsCtx->itstep++) {
    CHKERRQ(ComputeTimeStep(snes,tsCtx->itstep,user));
    /*tsCtx->ptime +=  tsCtx->dt;*/

    CHKERRQ(SNESSolve(snes,NULL,grid->qnode));
    CHKERRQ(SNESGetIterationNumber(snes,&its));

    CHKERRQ(SNESGetNonlinearStepFailures(snes,&nfails));
    nfailsCum += nfails; nfails = 0;
    PetscCheckFalse(nfailsCum >= 2,PETSC_COMM_SELF,1,"Unable to find a Newton Step");
    if (print_flag) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"At Time Step %d cfl = %g and fnorm = %g\n",
                         tsCtx->itstep,tsCtx->cfl,tsCtx->fnorm);CHKERRQ(ierr);
    }
    CHKERRQ(VecCopy(grid->qnode,tsCtx->qold));

    c_info->ntt = tsCtx->itstep+1;
    CHKERRQ(PetscTime(&time2));
    cpuloc      = time2-time1;
    cpuglo      = 0.0;
    CHKERRMPI(MPI_Allreduce(&cpuloc,&cpuglo,1,MPIU_REAL,MPIU_MAX,PETSC_COMM_WORLD));
    c_info->tot = cpuglo;    /* Total CPU time used upto this time step */

    CHKERRQ(VecScatterBegin(scatter,grid->qnode,localX,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(scatter,grid->qnode,localX,INSERT_VALUES,SCATTER_FORWARD));
    /* VecCopy(grid->qnode,localX); */

    CHKERRQ(VecGetArray(grid->res,&res));
    CHKERRQ(VecGetArray(localX,&qnode));

    f77FORCE(&grid->nnodesLoc,&grid->nedgeLoc,
              grid->isnode, grid->ivnode,
             &grid->nnfacetLoc,grid->f2ntn,&grid->nnbound,
             &grid->nvfacetLoc,grid->f2ntv,&grid->nvbound,
              grid->eptr,   qnode,
              grid->xyz,
              grid->sface_bit,grid->vface_bit,
              &clift,&cdrag,&cmom,&rank,&grid->nvertices);
    if (print_flag) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%d\t%g\t%g\t%g\t%g\t%g\n",tsCtx->itstep,
                        tsCtx->cfl,tsCtx->fnorm,clift,cdrag,cmom);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Wall clock time needed %g seconds for %d time steps\n",
                        cpuglo,tsCtx->itstep);CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_WORLD,fptr,"%d\t%g\t%g\t%g\t%g\t%g\t%g\n",
                          tsCtx->itstep,tsCtx->cfl,tsCtx->fnorm,clift,cdrag,cmom,cpuglo);
    }
    CHKERRQ(VecRestoreArray(localX,&qnode));
    CHKERRQ(VecRestoreArray(grid->res,&res));
    fratio = tsCtx->fnorm_ini/tsCtx->fnorm;
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));

  } /* End of time step loop */

#if defined(PARCH_IRIX64) && defined(USE_HW_COUNTERS)
  if (!user->PreLoading) {
    int  eve0,eve1;
    FILE *cfp0,*cfp1;
    char str[256];
    /* if ((gen_read = read_counters(event0,&counter0,event1,&counter1)) < 0)
    SETERRQ(PETSC_COMM_SELF,1,"Error in read_counter");
    CHKERRQ(PetscTime(&time_read_counters));
    if (gen_read != gen_start) {
    SETERRQ(PETSC_COMM_SELF,1,"Lost Counters!! Aborting ...");
    }*/
    /*sprintf(str,"counters%d_and_%d",event0,event1);
    cfp0 = fopen(str,"a");*/
    /*ierr = print_counters(event0,counter0,event1,counter1);*/
    /*fprintf(cfp0,"%lld %lld %g\n",counter0,counter1,
                  time_counters);
    fclose(cfp0);*/
  }
#endif
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Total wall clock time needed %g seconds for %d time steps\n",
                     cpuglo,tsCtx->itstep);CHKERRQ(ierr);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"cfl = %g fnorm = %g\n",tsCtx->cfl,tsCtx->fnorm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"clift = %g cdrag = %g cmom = %g\n",clift,cdrag,cmom));

  if (rank == 0 && print_flag) fclose(fptr);
  if (user->PreLoading) {
    tsCtx->fnorm_ini = 0.0;
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Preloading done ...\n"));
  }
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
int ComputeTimeStep(SNES snes,int iter,void *ctx)
/*---------------------------------------------------------------------*/
{
  AppCtx      *user  = (AppCtx*) ctx;
  TstepCtx    *tsCtx = user->tsCtx;
  Vec         func   = tsCtx->func;
  PetscScalar inc    = 1.1;
  PetscScalar newcfl;
  int         ierr;
  /*int       iramp = tsCtx->iramp;*/

  PetscFunctionBegin;
  tsCtx->ires = 0;
  CHKERRQ(FormFunction(snes,tsCtx->qold,func,user));
  tsCtx->ires = 1;
  CHKERRQ(VecNorm(func,NORM_2,&tsCtx->fnorm));
  /* first time through so compute initial function norm */
  if (tsCtx->fnorm_ini == 0.0) {
    tsCtx->fnorm_ini = tsCtx->fnorm;
    tsCtx->cfl       = tsCtx->cfl_ini;
  } else {
    newcfl     = inc*tsCtx->cfl_ini*tsCtx->fnorm_ini/tsCtx->fnorm;
    tsCtx->cfl = PetscMin(newcfl,tsCtx->cfl_max);
  }

  /* if (iramp < 0) {
   newcfl = inc*tsCtx->cfl_ini*tsCtx->fnorm_ini/tsCtx->fnorm;
  } else {
   if (tsCtx->dt < 0 && iramp > 0)
    if (iter > iramp) newcfl = tsCtx->cfl_max;
    else newcfl = tsCtx->cfl_ini + (tsCtx->cfl_max - tsCtx->cfl_ini)*
                                (double) iter/(double) iramp;
  }
  tsCtx->cfl = MIN(newcfl,tsCtx->cfl_max);*/
  /*printf("In ComputeTime Step - fnorm is %f\n",tsCtx->fnorm);*/
  /*CHKERRQ(VecDestroy(&func));*/
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
int GetLocalOrdering(GRID *grid)
/*---------------------------------------------------------------------*/
{
  int         ierr,i,j,k,inode,isurf,nte,nb,node1,node2,node3;
  int         nnodes,nedge,nnz,jstart,jend;
  int         nnodesLoc,nvertices,nedgeLoc,nnodesLocEst;
  int         nedgeLocEst,remEdges,readEdges,remNodes,readNodes;
  int         nnfacet,nvfacet,nffacet;
  int         nnfacetLoc,nvfacetLoc,nffacetLoc;
  int         nsnode,nvnode,nfnode;
  int         nsnodeLoc,nvnodeLoc,nfnodeLoc;
  int         nnbound,nvbound,nfbound;
  int         bs = 4;
  int         fdes = 0;
  off_t       currentPos  = 0,newPos = 0;
  int         grid_param  = 13;
  int         cross_edges = 0;
  int         *edge_bit,*pordering;
  int         *l2p,*l2a,*p2l,*a2l,*v2p,*eperm;
  int         *tmp,*tmp1,*tmp2;
  PetscScalar time_ini,time_fin;
  PetscScalar *ftmp,*ftmp1;
  char        mesh_file[PETSC_MAX_PATH_LEN] = "";
  AO          ao;
  FILE        *fptr,*fptr1;
  PetscBool   flg;
  MPI_Comm    comm = PETSC_COMM_WORLD;

  PetscFunctionBegin;
  /* Read the integer grid parameters */
  ICALLOC(grid_param,&tmp);
  if (rank == 0) {
    PetscBool exists;
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-mesh",mesh_file,sizeof(mesh_file),&flg));
    CHKERRQ(PetscTestFile(mesh_file,'r',&exists));
    if (!exists) { /* try uns3d.msh as the file name */
      CHKERRQ(PetscStrcpy(mesh_file,"uns3d.msh"));
    }
    CHKERRQ(PetscBinaryOpen(mesh_file,FILE_MODE_READ,&fdes));
  }
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,tmp,grid_param,PETSC_INT));
  grid->ncell   = tmp[0];
  grid->nnodes  = tmp[1];
  grid->nedge   = tmp[2];
  grid->nnbound = tmp[3];
  grid->nvbound = tmp[4];
  grid->nfbound = tmp[5];
  grid->nnfacet = tmp[6];
  grid->nvfacet = tmp[7];
  grid->nffacet = tmp[8];
  grid->nsnode  = tmp[9];
  grid->nvnode  = tmp[10];
  grid->nfnode  = tmp[11];
  grid->ntte    = tmp[12];
  grid->nsface  = 0;
  grid->nvface  = 0;
  grid->nfface  = 0;
  CHKERRQ(PetscFree(tmp));
  ierr          = PetscPrintf(comm,"nnodes = %d,nedge = %d,nnfacet = %d,nsnode = %d,nfnode = %d\n",
                              grid->nnodes,grid->nedge,grid->nnfacet,grid->nsnode,grid->nfnode);CHKERRQ(ierr);

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
  ICALLOC(nnodes,&l2a);
  ICALLOC(nnodes,&v2p);
  ICALLOC(nnodes,&a2l);
  nnodesLoc = 0;

  for (i = 0; i < nnodes; i++) a2l[i] = -1;
  CHKERRQ(PetscTime(&time_ini));

  if (rank == 0) {
    if (size == 1) {
      CHKERRQ(PetscMemzero(v2p,nnodes*sizeof(int)));
    } else {
      char      spart_file[PETSC_MAX_PATH_LEN],part_file[PETSC_MAX_PATH_LEN];
      PetscBool exists;

      CHKERRQ(PetscOptionsGetString(NULL,NULL,"-partition",spart_file,sizeof(spart_file),&flg));
      CHKERRQ(PetscTestFile(spart_file,'r',&exists));
      if (!exists) { /* try appending the number of processors */
        sprintf(part_file,"part_vec.part.%d",size);
        CHKERRQ(PetscStrcpy(spart_file,part_file));
      }
      fptr = fopen(spart_file,"r");
      PetscCheckFalse(!fptr,PETSC_COMM_SELF,1,"Cannot open file %s",part_file);
      for (inode = 0; inode < nnodes; inode++) {
        fscanf(fptr,"%d\n",&node1);
        v2p[inode] = node1;
      }
      fclose(fptr);
    }
  }
  CHKERRMPI(MPI_Bcast(v2p,nnodes,MPI_INT,0,comm));
  for (inode = 0; inode < nnodes; inode++) {
    if (v2p[inode] == rank) {
      l2a[nnodesLoc] = inode;
      a2l[inode]     = nnodesLoc;
      nnodesLoc++;
    }
  }

  CHKERRQ(PetscTime(&time_fin));
  time_fin -= time_ini;
  CHKERRQ(PetscPrintf(comm,"Partition Vector read successfully\n"));
  CHKERRQ(PetscPrintf(comm,"Time taken in this phase was %g\n",time_fin));

  CHKERRMPI(MPI_Scan(&nnodesLoc,&rstart,1,MPI_INT,MPI_SUM,comm));
  rstart -= nnodesLoc;
  ICALLOC(nnodesLoc,&pordering);
  for (i=0; i < nnodesLoc; i++) pordering[i] = rstart + i;
  CHKERRQ(AOCreateBasic(comm,nnodesLoc,l2a,pordering,&ao));
  CHKERRQ(PetscFree(pordering));

  /* Now count the local number of edges - including edges with
   ghost nodes but edges between ghost nodes are NOT counted */
  nedgeLoc  = 0;
  nvertices = nnodesLoc;
  /* Choose an estimated number of local edges. The choice
   nedgeLocEst = 1000000 looks reasonable as it will read
   the edge and edge normal arrays in 8 MB chunks */
  /*nedgeLocEst = nedge/size;*/
  nedgeLocEst = PetscMin(nedge,1000000);
  remEdges    = nedge;
  ICALLOC(2*nedgeLocEst,&tmp);
  CHKERRQ(PetscBinarySynchronizedSeek(comm,fdes,0,PETSC_BINARY_SEEK_CUR,&currentPos));
  CHKERRQ(PetscTime(&time_ini));
  while (remEdges > 0) {
    readEdges = PetscMin(remEdges,nedgeLocEst);
    /*time_ini = PetscTime();*/
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,tmp,readEdges,PETSC_INT));
    CHKERRQ(PetscBinarySynchronizedSeek(comm,fdes,(nedge-readEdges)*PETSC_BINARY_INT_SIZE,PETSC_BINARY_SEEK_CUR,&newPos));
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,tmp+readEdges,readEdges,PETSC_INT));
    CHKERRQ(PetscBinarySynchronizedSeek(comm,fdes,-nedge*PETSC_BINARY_INT_SIZE,PETSC_BINARY_SEEK_CUR,&newPos));
    /*time_fin += PetscTime()-time_ini;*/
    for (j = 0; j < readEdges; j++) {
      node1 = tmp[j]-1;
      node2 = tmp[j+readEdges]-1;
      if ((v2p[node1] == rank) || (v2p[node2] == rank)) {
        nedgeLoc++;
        if (a2l[node1] == -1) {
          l2a[nvertices] = node1;
          a2l[node1]     = nvertices;
          nvertices++;
        }
        if (a2l[node2] == -1) {
          l2a[nvertices] = node2;
          a2l[node2]     = nvertices;
          nvertices++;
        }
      }
    }
    remEdges = remEdges - readEdges;
    ierr     = MPI_Barrier(comm);
  }
  CHKERRQ(PetscTime(&time_fin));
  time_fin -= time_ini;
  CHKERRQ(PetscPrintf(comm,"Local edges counted with MPI_Bcast %d\n",nedgeLoc));
  CHKERRQ(PetscPrintf(comm,"Local vertices counted %d\n",nvertices));
  CHKERRQ(PetscPrintf(comm,"Time taken in this phase was %g\n",time_fin));

  /* Now store the local edges */
  ICALLOC(2*nedgeLoc,&grid->eptr);
  ICALLOC(nedgeLoc,&edge_bit);
  ICALLOC(nedgeLoc,&eperm);
  i = 0; j = 0; k = 0;
  remEdges   = nedge;
  CHKERRQ(PetscBinarySynchronizedSeek(comm,fdes,currentPos,PETSC_BINARY_SEEK_SET,&newPos));
  currentPos = newPos;

  CHKERRQ(PetscTime(&time_ini));
  while (remEdges > 0) {
    readEdges = PetscMin(remEdges,nedgeLocEst);
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,tmp,readEdges,PETSC_INT));
    CHKERRQ(PetscBinarySynchronizedSeek(comm,fdes,(nedge-readEdges)*PETSC_BINARY_INT_SIZE,PETSC_BINARY_SEEK_CUR,&newPos));
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,tmp+readEdges,readEdges,PETSC_INT));
    CHKERRQ(PetscBinarySynchronizedSeek(comm,fdes,-nedge*PETSC_BINARY_INT_SIZE,PETSC_BINARY_SEEK_CUR,&newPos));
    for (j = 0; j < readEdges; j++) {
      node1 = tmp[j]-1;
      node2 = tmp[j+readEdges]-1;
      if ((v2p[node1] == rank) || (v2p[node2] == rank)) {
        grid->eptr[k]          = a2l[node1];
        grid->eptr[k+nedgeLoc] = a2l[node2];
        edge_bit[k]            = i; /* Record global file index of the edge */
        eperm[k]               = k;
        k++;
      }
      i++;
    }
    remEdges = remEdges - readEdges;
    ierr     = MPI_Barrier(comm);
  }
  CHKERRQ(PetscBinarySynchronizedSeek(comm,fdes,currentPos+2*nedge*PETSC_BINARY_INT_SIZE,PETSC_BINARY_SEEK_SET,&newPos));
  CHKERRQ(PetscTime(&time_fin));
  time_fin -= time_ini;
  CHKERRQ(PetscPrintf(comm,"Local edges stored\n"));
  CHKERRQ(PetscPrintf(comm,"Time taken in this phase was %g\n",time_fin));

  CHKERRQ(PetscFree(tmp));
  ICALLOC(2*nedgeLoc,&tmp);
  CHKERRQ(PetscMemcpy(tmp,grid->eptr,2*nedgeLoc*sizeof(int)));
#if defined(_OPENMP) && defined(HAVE_EDGE_COLORING)
  ierr = EdgeColoring(nvertices,nedgeLoc,grid->eptr,eperm,&grid->ncolor,grid->ncount);
#else
  /* Now reorder the edges for better cache locality */
  /*
  tmp[0]=7;tmp[1]=6;tmp[2]=3;tmp[3]=9;tmp[4]=2;tmp[5]=0;
  ierr = PetscSortIntWithPermutation(6,tmp,eperm);
  for (i=0; i<6; i++)
   printf("%d %d %d\n",i,tmp[i],eperm[i]);
  */
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(0,"-no_edge_reordering",&flg,NULL));
  if (!flg) {
    CHKERRQ(PetscSortIntWithPermutation(nedgeLoc,tmp,eperm));
  }
#endif
  CHKERRQ(PetscMallocValidate(__LINE__,PETSC_FUNCTION_NAME,__FILE__));
  k    = 0;
  for (i = 0; i < nedgeLoc; i++) {
    int cross_node=nnodesLoc/2;
    node1 = tmp[eperm[i]] + 1;
    node2 = tmp[nedgeLoc+eperm[i]] + 1;
#if defined(INTERLACING)
    grid->eptr[k++] = node1;
    grid->eptr[k++] = node2;
#else
    grid->eptr[i]          = node1;
    grid->eptr[nedgeLoc+i] = node2;
#endif
    /* if (node1 > node2)
     printf("On processor %d, for edge %d node1 = %d, node2 = %d\n",
            rank,i,node1,node2);CHKERRQ(ierr);*/
    if ((node1 <= cross_node) && (node2 > cross_node)) cross_edges++;
  }
  CHKERRQ(PetscPrintf(comm,"Number of cross edges %d\n", cross_edges));
  CHKERRQ(PetscFree(tmp));
#if defined(_OPENMP) && !defined(HAVE_REDUNDANT_WORK) && !defined(HAVE_EDGE_COLORING)
  /* Now make the local 'ia' and 'ja' arrays */
  ICALLOC(nvertices+1,&grid->ia);
  /* Use tmp for a work array */
  ICALLOC(nvertices,&tmp);
  f77GETIA(&nvertices,&nedgeLoc,grid->eptr,grid->ia,tmp,&rank);
  nnz = grid->ia[nvertices] - 1;
  ICALLOC(nnz,&grid->ja);
  f77GETJA(&nvertices,&nedgeLoc,grid->eptr,grid->ia,grid->ja,tmp,&rank);
  CHKERRQ(PetscFree(tmp));
#else
  /* Now make the local 'ia' and 'ja' arrays */
  ICALLOC(nnodesLoc+1,&grid->ia);
  /* Use tmp for a work array */
  ICALLOC(nnodesLoc,&tmp);
  f77GETIA(&nnodesLoc,&nedgeLoc,grid->eptr,grid->ia,tmp,&rank);
  nnz = grid->ia[nnodesLoc] - 1;
#if defined(BLOCKING)
  CHKERRQ(PetscPrintf(comm,"The Jacobian has %d non-zero blocks with block size = %d\n",nnz,bs));
#else
  CHKERRQ(PetscPrintf(comm,"The Jacobian has %d non-zeros\n",nnz));
#endif
  ICALLOC(nnz,&grid->ja);
  f77GETJA(&nnodesLoc,&nedgeLoc,grid->eptr,grid->ia,grid->ja,tmp,&rank);
  CHKERRQ(PetscFree(tmp));
#endif
  ICALLOC(nvertices,&grid->loc2glo);
  CHKERRQ(PetscMemcpy(grid->loc2glo,l2a,nvertices*sizeof(int)));
  CHKERRQ(PetscFree(l2a));
  l2a  = grid->loc2glo;
  ICALLOC(nvertices,&grid->loc2pet);
  l2p  = grid->loc2pet;
  CHKERRQ(PetscMemcpy(l2p,l2a,nvertices*sizeof(int)));
  CHKERRQ(AOApplicationToPetsc(ao,nvertices,l2p));

  /* Renumber unit normals of dual face (from node1 to node2)
      and the area of the dual mesh face */
  FCALLOC(nedgeLocEst,&ftmp);
  FCALLOC(nedgeLoc,&ftmp1);
  FCALLOC(4*nedgeLoc,&grid->xyzn);
  /* Do the x-component */
  i = 0; k = 0;
  remEdges = nedge;
  CHKERRQ(PetscTime(&time_ini));
  while (remEdges > 0) {
    readEdges = PetscMin(remEdges,nedgeLocEst);
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,ftmp,readEdges,PETSC_SCALAR));
    for (j = 0; j < readEdges; j++)
      if (edge_bit[k] == (i+j)) {
        ftmp1[k] = ftmp[j];
        k++;
      }
    i       += readEdges;
    remEdges = remEdges - readEdges;
    CHKERRMPI(MPI_Barrier(comm));
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
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,ftmp,readEdges,PETSC_SCALAR));
    for (j = 0; j < readEdges; j++)
      if (edge_bit[k] == (i+j)) {
        ftmp1[k] = ftmp[j];
        k++;
      }
    i       += readEdges;
    remEdges = remEdges - readEdges;
    CHKERRMPI(MPI_Barrier(comm));
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
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,ftmp,readEdges,PETSC_SCALAR));
    for (j = 0; j < readEdges; j++)
      if (edge_bit[k] == (i+j)) {
        ftmp1[k] = ftmp[j];
        k++;
      }
    i       += readEdges;
    remEdges = remEdges - readEdges;
    CHKERRMPI(MPI_Barrier(comm));
  }
  for (i = 0; i < nedgeLoc; i++)
#if defined(INTERLACING)
    grid->xyzn[4*i+2] = ftmp1[eperm[i]];
#else
    grid->xyzn[2*nedgeLoc+i] = ftmp1[eperm[i]];
#endif
  /* Do the area */
  i = 0; k = 0;
  remEdges = nedge;
  while (remEdges > 0) {
    readEdges = PetscMin(remEdges,nedgeLocEst);
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,ftmp,readEdges,PETSC_SCALAR));
    for (j = 0; j < readEdges; j++)
      if (edge_bit[k] == (i+j)) {
        ftmp1[k] = ftmp[j];
        k++;
      }
    i       += readEdges;
    remEdges = remEdges - readEdges;
    CHKERRMPI(MPI_Barrier(comm));
  }
  for (i = 0; i < nedgeLoc; i++)
#if defined(INTERLACING)
    grid->xyzn[4*i+3] = ftmp1[eperm[i]];
#else
    grid->xyzn[3*nedgeLoc+i] = ftmp1[eperm[i]];
#endif

  CHKERRQ(PetscFree(edge_bit));
  CHKERRQ(PetscFree(eperm));
  CHKERRQ(PetscFree(ftmp));
  CHKERRQ(PetscFree(ftmp1));
  CHKERRQ(PetscTime(&time_fin));
  time_fin -= time_ini;
  CHKERRQ(PetscPrintf(comm,"Edge normals partitioned\n"));
  CHKERRQ(PetscPrintf(comm,"Time taken in this phase was %g\n",time_fin));
#if defined(_OPENMP)
  /*Arrange for the division of work among threads*/
#if defined(HAVE_EDGE_COLORING)
#elif defined(HAVE_REDUNDANT_WORK)
  FCALLOC(4*nnodesLoc,   &grid->resd);
#else
  {
    /* Get the local adjacency structure of the graph for partitioning the local
      graph into max_threads pieces */
    int *ia,*ja,*vwtg=0,*adjwgt=0,options[5];
    int numflag = 0, wgtflag = 0, edgecut;
    int thr1,thr2,nedgeAllThreads,ned1,ned2;
    ICALLOC((nvertices+1),&ia);
    ICALLOC((2*nedgeLoc),&ja);
    ia[0] = 0;
    for (i = 1; i <= nvertices; i++) ia[i] = grid->ia[i]-i-1;
    for (i = 0; i < nvertices; i++) {
      int jstart,jend;
      jstart = grid->ia[i]-1;
      jend   = grid->ia[i+1]-1;
      k      = ia[i];
      for (j=jstart; j < jend; j++) {
        inode = grid->ja[j]-1;
        if (inode != i) ja[k++] = inode;
      }
    }
    ICALLOC(nvertices,&grid->part_thr);
    CHKERRQ(PetscMemzero(grid->part_thr,nvertices*sizeof(int)));
    options[0] = 0;
    /* Call the pmetis library routine */
    if (max_threads > 1)
      METIS_PartGraphRecursive(&nvertices,ia,ja,vwtg,adjwgt,
                               &wgtflag,&numflag,&max_threads,options,&edgecut,grid->part_thr);
    PetscPrintf(MPI_COMM_WORLD,"The number of cut edges is %d\n", edgecut);
    /* Write the partition vector to disk */
    flg  = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(0,"-omp_partitioning",&flg,NULL));
    if (flg) {
      int  *partv_loc, *partv_glo;
      int  *disp,*counts,*loc2glo_glo;
      char part_file[PETSC_MAX_PATH_LEN];
      FILE *fp;

      ICALLOC(nnodes, &partv_glo);
      ICALLOC(nnodesLoc, &partv_loc);
      for (i = 0; i < nnodesLoc; i++)
        /*partv_loc[i] = grid->part_thr[i]*size + rank;*/
        partv_loc[i] = grid->part_thr[i] + max_threads*rank;
      ICALLOC(size,&disp);
      ICALLOC(size,&counts);
      MPI_Allgather(&nnodesLoc,1,MPI_INT,counts,1,MPI_INT,MPI_COMM_WORLD);
      disp[0] = 0;
      for (i = 1; i < size; i++) disp[i] = counts[i-1] + disp[i-1];
      ICALLOC(nnodes, &loc2glo_glo);
      MPI_Gatherv(grid->loc2glo,nnodesLoc,MPI_INT,loc2glo_glo,counts,disp,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Gatherv(partv_loc,nnodesLoc,MPI_INT,partv_glo,counts,disp,MPI_INT,0,MPI_COMM_WORLD);
      if (rank == 0) {
        CHKERRQ(PetscSortIntWithArray(nnodes,loc2glo_glo,partv_glo));
        sprintf(part_file,"hyb_part_vec.%d",2*size);
        fp = fopen(part_file,"w");
        for (i = 0; i < nnodes; i++) fprintf(fp,"%d\n",partv_glo[i]);
        fclose(fp);
      }
      PetscFree(partv_loc);
      PetscFree(partv_glo);
      PetscFree(disp);
      PetscFree(counts);
      PetscFree(loc2glo_glo);
    }

    /* Divide the work among threads */
    k = 0;
    ICALLOC((max_threads+1),&grid->nedge_thr);
    CHKERRQ(PetscMemzero(grid->nedge_thr,(max_threads+1)*sizeof(int)));
    cross_edges = 0;
    for (i = 0; i < nedgeLoc; i++) {
      node1 = grid->eptr[k++]-1;
      node2 = grid->eptr[k++]-1;
      thr1  = grid->part_thr[node1];
      thr2  = grid->part_thr[node2];
      grid->nedge_thr[thr1]+=1;
      if (thr1 != thr2) {
        grid->nedge_thr[thr2]+=1;
        cross_edges++;
      }
    }
    PetscPrintf(MPI_COMM_WORLD,"The number of cross edges after Metis partitioning is %d\n",cross_edges);
    ned1 = grid->nedge_thr[0];
    grid->nedge_thr[0] = 1;
    for (i = 1; i <= max_threads; i++) {
      ned2 = grid->nedge_thr[i];
      grid->nedge_thr[i] = grid->nedge_thr[i-1]+ned1;
      ned1 = ned2;
    }
    /* Allocate a shared edge array. Note that a cut edge is evaluated
        by both the threads but updates are done only for the locally
        owned node */
    grid->nedgeAllThr = nedgeAllThreads = grid->nedge_thr[max_threads]-1;
    ICALLOC(2*nedgeAllThreads, &grid->edge_thr);
    ICALLOC(max_threads,&tmp);
    FCALLOC(4*nedgeAllThreads,&grid->xyzn_thr);
    for (i = 0; i < max_threads; i++) tmp[i] = grid->nedge_thr[i]-1;
    k = 0;
    for (i = 0; i < nedgeLoc; i++) {
      int ie1,ie2,ie3;
      node1 = grid->eptr[k++];
      node2 = grid->eptr[k++];
      thr1  = grid->part_thr[node1-1];
      thr2  = grid->part_thr[node2-1];
      ie1   = 2*tmp[thr1];
      ie2   = 4*tmp[thr1];
      ie3   = 4*i;

      grid->edge_thr[ie1]   = node1;
      grid->edge_thr[ie1+1] = node2;
      grid->xyzn_thr[ie2]   = grid->xyzn[ie3];
      grid->xyzn_thr[ie2+1] = grid->xyzn[ie3+1];
      grid->xyzn_thr[ie2+2] = grid->xyzn[ie3+2];
      grid->xyzn_thr[ie2+3] = grid->xyzn[ie3+3];

      tmp[thr1]+=1;
      if (thr1 != thr2) {
        ie1 = 2*tmp[thr2];
        ie2 = 4*tmp[thr2];

        grid->edge_thr[ie1]   = node1;
        grid->edge_thr[ie1+1] = node2;
        grid->xyzn_thr[ie2]   = grid->xyzn[ie3];
        grid->xyzn_thr[ie2+1] = grid->xyzn[ie3+1];
        grid->xyzn_thr[ie2+2] = grid->xyzn[ie3+2];
        grid->xyzn_thr[ie2+3] = grid->xyzn[ie3+3];

        tmp[thr2]+=1;
      }
    }
  }
#endif
#endif

  /* Remap coordinates */
  /*nnodesLocEst = nnodes/size;*/
  nnodesLocEst = PetscMin(nnodes,500000);
  FCALLOC(nnodesLocEst,&ftmp);
  FCALLOC(3*nvertices,&grid->xyz);
  remNodes = nnodes;
  i        = 0;
  CHKERRQ(PetscTime(&time_ini));
  while (remNodes > 0) {
    readNodes = PetscMin(remNodes,nnodesLocEst);
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,ftmp,readNodes,PETSC_SCALAR));
    for (j = 0; j < readNodes; j++) {
      if (a2l[i+j] >= 0) {
#if defined(INTERLACING)
        grid->xyz[3*a2l[i+j]] = ftmp[j];
#else
        grid->xyz[a2l[i+j]] = ftmp[j];
#endif
      }
    }
    i        += nnodesLocEst;
    remNodes -= nnodesLocEst;
    CHKERRMPI(MPI_Barrier(comm));
  }

  remNodes = nnodes;
  i = 0;
  while (remNodes > 0) {
    readNodes = PetscMin(remNodes,nnodesLocEst);
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,ftmp,readNodes,PETSC_SCALAR));
    for (j = 0; j < readNodes; j++) {
      if (a2l[i+j] >= 0) {
#if defined(INTERLACING)
        grid->xyz[3*a2l[i+j]+1] = ftmp[j];
#else
        grid->xyz[nnodesLoc+a2l[i+j]] = ftmp[j];
#endif
      }
    }
    i        += nnodesLocEst;
    remNodes -= nnodesLocEst;
    CHKERRMPI(MPI_Barrier(comm));
  }

  remNodes = nnodes;
  i        = 0;
  while (remNodes > 0) {
    readNodes = PetscMin(remNodes,nnodesLocEst);
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,ftmp,readNodes,PETSC_SCALAR));
    for (j = 0; j < readNodes; j++) {
      if (a2l[i+j] >= 0) {
#if defined(INTERLACING)
        grid->xyz[3*a2l[i+j]+2] = ftmp[j];
#else
        grid->xyz[2*nnodesLoc+a2l[i+j]] = ftmp[j];
#endif
      }
    }
    i        += nnodesLocEst;
    remNodes -= nnodesLocEst;
    CHKERRMPI(MPI_Barrier(comm));
  }

  /* Renumber dual volume "area" */
  FCALLOC(nvertices,&grid->area);
  remNodes = nnodes;
  i        = 0;
  while (remNodes > 0) {
    readNodes = PetscMin(remNodes,nnodesLocEst);
    CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,ftmp,readNodes,PETSC_SCALAR));
    for (j = 0; j < readNodes; j++)
      if (a2l[i+j] >= 0)
        grid->area[a2l[i+j]] = ftmp[j];
    i        += nnodesLocEst;
    remNodes -= nnodesLocEst;
    CHKERRMPI(MPI_Barrier(comm));
  }

  CHKERRQ(PetscFree(ftmp));
  CHKERRQ(PetscTime(&time_fin));
  time_fin -= time_ini;
  CHKERRQ(PetscPrintf(comm,"Coordinates remapped\n"));
  CHKERRQ(PetscPrintf(comm,"Time taken in this phase was %g\n",time_fin));

/* Now,handle all the solid boundaries - things to be done :
 * 1. Identify the nodes belonging to the solid
 *    boundaries and count them.
 * 2. Put proper indices into f2ntn array,after making it
 *    of suitable size.
 * 3. Remap the normals and areas of solid faces (sxn,syn,szn,
 *    and sa arrays).
 */
  ICALLOC(nnbound,  &grid->nntet);
  ICALLOC(nnbound,  &grid->nnpts);
  ICALLOC(4*nnfacet,&grid->f2ntn);
  ICALLOC(nsnode,&grid->isnode);
  FCALLOC(nsnode,&grid->sxn);
  FCALLOC(nsnode,&grid->syn);
  FCALLOC(nsnode,&grid->szn);
  FCALLOC(nsnode,&grid->sa);
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->nntet,nnbound,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->nnpts,nnbound,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->f2ntn,4*nnfacet,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->isnode,nsnode,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->sxn,nsnode,PETSC_SCALAR));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->syn,nsnode,PETSC_SCALAR));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->szn,nsnode,PETSC_SCALAR));

  isurf      = 0;
  nsnodeLoc  = 0;
  nnfacetLoc = 0;
  nb         = 0;
  nte        = 0;
  ICALLOC(3*nnfacet,&tmp);
  ICALLOC(nsnode,&tmp1);
  ICALLOC(nnodes,&tmp2);
  FCALLOC(4*nsnode,&ftmp);
  CHKERRQ(PetscMemzero(tmp,3*nnfacet*sizeof(int)));
  CHKERRQ(PetscMemzero(tmp1,nsnode*sizeof(int)));
  CHKERRQ(PetscMemzero(tmp2,nnodes*sizeof(int)));

  j = 0;
  for (i = 0; i < nsnode; i++) {
    node1 = a2l[grid->isnode[i] - 1];
    if (node1 >= 0) {
      tmp1[nsnodeLoc] = node1;
      tmp2[node1]     = nsnodeLoc;
      ftmp[j++]       = grid->sxn[i];
      ftmp[j++]       = grid->syn[i];
      ftmp[j++]       = grid->szn[i];
      ftmp[j++]       = grid->sa[i];
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
  CHKERRQ(PetscFree(grid->f2ntn));
  CHKERRQ(PetscFree(grid->isnode));
  CHKERRQ(PetscFree(grid->sxn));
  CHKERRQ(PetscFree(grid->syn));
  CHKERRQ(PetscFree(grid->szn));
  CHKERRQ(PetscFree(grid->sa));
  ICALLOC(4*nnfacetLoc,&grid->f2ntn);
  ICALLOC(nsnodeLoc,&grid->isnode);
  FCALLOC(nsnodeLoc,&grid->sxn);
  FCALLOC(nsnodeLoc,&grid->syn);
  FCALLOC(nsnodeLoc,&grid->szn);
  FCALLOC(nsnodeLoc,&grid->sa);
  j = 0;
  for (i = 0; i < nsnodeLoc; i++) {
    grid->isnode[i] = tmp1[i] + 1;
    grid->sxn[i]    = ftmp[j++];
    grid->syn[i]    = ftmp[j++];
    grid->szn[i]    = ftmp[j++];
    grid->sa[i]     = ftmp[j++];
  }
  j = 0;
  for (i = 0; i < nnfacetLoc; i++) {
    grid->f2ntn[i]              = tmp[j++] + 1;
    grid->f2ntn[nnfacetLoc+i]   = tmp[j++] + 1;
    grid->f2ntn[2*nnfacetLoc+i] = tmp[j++] + 1;
  }
  CHKERRQ(PetscFree(tmp));
  CHKERRQ(PetscFree(tmp1));
  CHKERRQ(PetscFree(tmp2));
  CHKERRQ(PetscFree(ftmp));

/* Now identify the triangles on which the current proceesor
   would perform force calculation */
  ICALLOC(nnfacetLoc,&grid->sface_bit);
  PetscMemzero(grid->sface_bit,nnfacetLoc*sizeof(int));
  for (i = 0; i < nnfacetLoc; i++) {
    node1 = l2a[grid->isnode[grid->f2ntn[i] - 1] - 1];
    node2 = l2a[grid->isnode[grid->f2ntn[nnfacetLoc + i] - 1] - 1];
    node3 = l2a[grid->isnode[grid->f2ntn[2*nnfacetLoc + i] - 1] - 1];
    if (((v2p[node1] >= rank) && (v2p[node2] >= rank)
         && (v2p[node3] >= rank)) &&
        ((v2p[node1] == rank) || (v2p[node2] == rank)
         || (v2p[node3] == rank)))
      grid->sface_bit[i] = 1;
  }
  /*printf("On processor %d total solid triangles = %d,locally owned = %d alpha = %d\n",rank,totTr,myTr,alpha);*/
  CHKERRQ(PetscPrintf(comm,"Solid boundaries partitioned\n"));

/* Now,handle all the viscous boundaries - things to be done :
 * 1. Identify the nodes belonging to the viscous
 *    boundaries and count them.
 * 2. Put proper indices into f2ntv array,after making it
 *    of suitable size
 * 3. Remap the normals and areas of viscous faces (vxn,vyn,vzn,
 *    and va arrays).
 */
  ICALLOC(nvbound,  &grid->nvtet);
  ICALLOC(nvbound,  &grid->nvpts);
  ICALLOC(4*nvfacet,&grid->f2ntv);
  ICALLOC(nvnode,&grid->ivnode);
  FCALLOC(nvnode,&grid->vxn);
  FCALLOC(nvnode,&grid->vyn);
  FCALLOC(nvnode,&grid->vzn);
  FCALLOC(nvnode,&grid->va);
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->nvtet,nvbound,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->nvpts,nvbound,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->f2ntv,4*nvfacet,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->ivnode,nvnode,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->vxn,nvnode,PETSC_SCALAR));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->vyn,nvnode,PETSC_SCALAR));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->vzn,nvnode,PETSC_SCALAR));

  isurf      = 0;
  nvnodeLoc  = 0;
  nvfacetLoc = 0;
  nb         = 0;
  nte        = 0;
  ICALLOC(3*nvfacet,&tmp);
  ICALLOC(nvnode,&tmp1);
  ICALLOC(nnodes,&tmp2);
  FCALLOC(4*nvnode,&ftmp);
  CHKERRQ(PetscMemzero(tmp,3*nvfacet*sizeof(int)));
  CHKERRQ(PetscMemzero(tmp1,nvnode*sizeof(int)));
  CHKERRQ(PetscMemzero(tmp2,nnodes*sizeof(int)));

  j = 0;
  for (i = 0; i < nvnode; i++) {
    node1 = a2l[grid->ivnode[i] - 1];
    if (node1 >= 0) {
      tmp1[nvnodeLoc] = node1;
      tmp2[node1]     = nvnodeLoc;
      ftmp[j++]       = grid->vxn[i];
      ftmp[j++]       = grid->vyn[i];
      ftmp[j++]       = grid->vzn[i];
      ftmp[j++]       = grid->va[i];
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
    isurf         += grid->nvtet[i];
    grid->nvtet[i] = nte;
    nte            = 0;
  }
  CHKERRQ(PetscFree(grid->f2ntv));
  CHKERRQ(PetscFree(grid->ivnode));
  CHKERRQ(PetscFree(grid->vxn));
  CHKERRQ(PetscFree(grid->vyn));
  CHKERRQ(PetscFree(grid->vzn));
  CHKERRQ(PetscFree(grid->va));
  ICALLOC(4*nvfacetLoc,&grid->f2ntv);
  ICALLOC(nvnodeLoc,&grid->ivnode);
  FCALLOC(nvnodeLoc,&grid->vxn);
  FCALLOC(nvnodeLoc,&grid->vyn);
  FCALLOC(nvnodeLoc,&grid->vzn);
  FCALLOC(nvnodeLoc,&grid->va);
  j = 0;
  for (i = 0; i < nvnodeLoc; i++) {
    grid->ivnode[i] = tmp1[i] + 1;
    grid->vxn[i]    = ftmp[j++];
    grid->vyn[i]    = ftmp[j++];
    grid->vzn[i]    = ftmp[j++];
    grid->va[i]     = ftmp[j++];
  }
  j = 0;
  for (i = 0; i < nvfacetLoc; i++) {
    grid->f2ntv[i]              = tmp[j++] + 1;
    grid->f2ntv[nvfacetLoc+i]   = tmp[j++] + 1;
    grid->f2ntv[2*nvfacetLoc+i] = tmp[j++] + 1;
  }
  CHKERRQ(PetscFree(tmp));
  CHKERRQ(PetscFree(tmp1));
  CHKERRQ(PetscFree(tmp2));
  CHKERRQ(PetscFree(ftmp));

/* Now identify the triangles on which the current proceesor
   would perform force calculation */
  ICALLOC(nvfacetLoc,&grid->vface_bit);
  CHKERRQ(PetscMemzero(grid->vface_bit,nvfacetLoc*sizeof(int)));
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
  CHKERRQ(PetscFree(v2p));
  CHKERRQ(PetscPrintf(comm,"Viscous boundaries partitioned\n"));

/* Now,handle all the free boundaries - things to be done :
 * 1. Identify the nodes belonging to the free
 *    boundaries and count them.
 * 2. Put proper indices into f2ntf array,after making it
 *    of suitable size
 * 3. Remap the normals and areas of free bound. faces (fxn,fyn,fzn,
 *    and fa arrays).
 */

  ICALLOC(nfbound,  &grid->nftet);
  ICALLOC(nfbound,  &grid->nfpts);
  ICALLOC(4*nffacet,&grid->f2ntf);
  ICALLOC(nfnode,&grid->ifnode);
  FCALLOC(nfnode,&grid->fxn);
  FCALLOC(nfnode,&grid->fyn);
  FCALLOC(nfnode,&grid->fzn);
  FCALLOC(nfnode,&grid->fa);
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->nftet,nfbound,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->nfpts,nfbound,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->f2ntf,4*nffacet,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->ifnode,nfnode,PETSC_INT));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->fxn,nfnode,PETSC_SCALAR));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->fyn,nfnode,PETSC_SCALAR));
  CHKERRQ(PetscBinarySynchronizedRead(comm,fdes,grid->fzn,nfnode,PETSC_SCALAR));

  isurf      = 0;
  nfnodeLoc  = 0;
  nffacetLoc = 0;
  nb         = 0;
  nte        = 0;
  ICALLOC(3*nffacet,&tmp);
  ICALLOC(nfnode,&tmp1);
  ICALLOC(nnodes,&tmp2);
  FCALLOC(4*nfnode,&ftmp);
  CHKERRQ(PetscMemzero(tmp,3*nffacet*sizeof(int)));
  CHKERRQ(PetscMemzero(tmp1,nfnode*sizeof(int)));
  CHKERRQ(PetscMemzero(tmp2,nnodes*sizeof(int)));

  j = 0;
  for (i = 0; i < nfnode; i++) {
    node1 = a2l[grid->ifnode[i] - 1];
    if (node1 >= 0) {
      tmp1[nfnodeLoc] = node1;
      tmp2[node1]     = nfnodeLoc;
      ftmp[j++]       = grid->fxn[i];
      ftmp[j++]       = grid->fyn[i];
      ftmp[j++]       = grid->fzn[i];
      ftmp[j++]       = grid->fa[i];
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
    isurf         += grid->nftet[i];
    grid->nftet[i] = nte;
    nte            = 0;
  }
  CHKERRQ(PetscFree(grid->f2ntf));
  CHKERRQ(PetscFree(grid->ifnode));
  CHKERRQ(PetscFree(grid->fxn));
  CHKERRQ(PetscFree(grid->fyn));
  CHKERRQ(PetscFree(grid->fzn));
  CHKERRQ(PetscFree(grid->fa));
  ICALLOC(4*nffacetLoc,&grid->f2ntf);
  ICALLOC(nfnodeLoc,&grid->ifnode);
  FCALLOC(nfnodeLoc,&grid->fxn);
  FCALLOC(nfnodeLoc,&grid->fyn);
  FCALLOC(nfnodeLoc,&grid->fzn);
  FCALLOC(nfnodeLoc,&grid->fa);
  j = 0;
  for (i = 0; i < nfnodeLoc; i++) {
    grid->ifnode[i] = tmp1[i] + 1;
    grid->fxn[i]    = ftmp[j++];
    grid->fyn[i]    = ftmp[j++];
    grid->fzn[i]    = ftmp[j++];
    grid->fa[i]     = ftmp[j++];
  }
  j = 0;
  for (i = 0; i < nffacetLoc; i++) {
    grid->f2ntf[i]              = tmp[j++] + 1;
    grid->f2ntf[nffacetLoc+i]   = tmp[j++] + 1;
    grid->f2ntf[2*nffacetLoc+i] = tmp[j++] + 1;
  }

  CHKERRQ(PetscFree(tmp));
  CHKERRQ(PetscFree(tmp1));
  CHKERRQ(PetscFree(tmp2));
  CHKERRQ(PetscFree(ftmp));
  CHKERRQ(PetscPrintf(comm,"Free boundaries partitioned\n"));

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(0,"-mem_use",&flg,NULL));
  if (flg) {
    CHKERRQ(PetscMemoryView(PETSC_VIEWER_STDOUT_WORLD,"Memory usage after partitioning\n"));
  }

  /* Put different mappings and other info into grid */
  /* ICALLOC(nvertices,&grid->loc2pet);
   ICALLOC(nvertices,&grid->loc2glo);
   PetscMemcpy(grid->loc2pet,l2p,nvertices*sizeof(int));
   PetscMemcpy(grid->loc2glo,l2a,nvertices*sizeof(int));
   CHKERRQ(PetscFree(l2a));
   CHKERRQ(PetscFree(l2p));*/

  grid->nnodesLoc  = nnodesLoc;
  grid->nedgeLoc   = nedgeLoc;
  grid->nvertices  = nvertices;
  grid->nsnodeLoc  = nsnodeLoc;
  grid->nvnodeLoc  = nvnodeLoc;
  grid->nfnodeLoc  = nfnodeLoc;
  grid->nnfacetLoc = nnfacetLoc;
  grid->nvfacetLoc = nvfacetLoc;
  grid->nffacetLoc = nffacetLoc;
/*
 * FCALLOC(nvertices*4, &grid->gradx);
 * FCALLOC(nvertices*4, &grid->grady);
 * FCALLOC(nvertices*4, &grid->gradz);
 */
  FCALLOC(nvertices,   &grid->cdt);
  FCALLOC(nvertices*4, &grid->phi);
/*
   FCALLOC(nvertices,   &grid->r11);
   FCALLOC(nvertices,   &grid->r12);
   FCALLOC(nvertices,   &grid->r13);
   FCALLOC(nvertices,   &grid->r22);
   FCALLOC(nvertices,   &grid->r23);
   FCALLOC(nvertices,   &grid->r33);
*/
  FCALLOC(7*nnodesLoc,   &grid->rxy);

/* Map the 'ja' array in petsc ordering */
  for (i = 0; i < nnz; i++) grid->ja[i] = l2a[grid->ja[i] - 1];
  CHKERRQ(AOApplicationToPetsc(ao,nnz,grid->ja));
  CHKERRQ(AODestroy(&ao));

/* Print the different mappings
 *
 */
  {
    int partLoc[7],partMax[7],partMin[7],partSum[7];
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

    CHKERRMPI(MPI_Allreduce(partLoc,partMax,7,MPI_INT,MPI_MAX,comm));
    CHKERRMPI(MPI_Allreduce(partLoc,partMin,7,MPI_INT,MPI_MIN,comm));
    CHKERRMPI(MPI_Allreduce(partLoc,partSum,7,MPI_INT,MPI_SUM,comm));
    CHKERRQ(PetscPrintf(comm,"==============================\n"));
    CHKERRQ(PetscPrintf(comm,"Partitioning quality info ....\n"));
    CHKERRQ(PetscPrintf(comm,"==============================\n"));
    CHKERRQ(PetscPrintf(comm,"------------------------------------------------------------\n"));
    CHKERRQ(PetscPrintf(comm,"Item                    Min        Max    Average      Total\n"));
    CHKERRQ(PetscPrintf(comm,"------------------------------------------------------------\n"));
    ierr = PetscPrintf(comm,"Local Nodes       %9d  %9d  %9d  %9d\n",
                       partMin[0],partMax[0],partSum[0]/size,partSum[0]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local+Ghost Nodes %9d  %9d  %9d  %9d\n",
                       partMin[1],partMax[1],partSum[1]/size,partSum[1]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local Edges       %9d  %9d  %9d  %9d\n",
                       partMin[2],partMax[2],partSum[2]/size,partSum[2]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local solid faces %9d  %9d  %9d  %9d\n",
                       partMin[3],partMax[3],partSum[3]/size,partSum[3]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local free faces  %9d  %9d  %9d  %9d\n",
                       partMin[4],partMax[4],partSum[4]/size,partSum[4]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local solid nodes %9d  %9d  %9d  %9d\n",
                       partMin[5],partMax[5],partSum[5]/size,partSum[5]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local free nodes  %9d  %9d  %9d  %9d\n",
                       partMin[6],partMax[6],partSum[6]/size,partSum[6]);CHKERRQ(ierr);
    CHKERRQ(PetscPrintf(comm,"------------------------------------------------------------\n"));
  }
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(0,"-partition_info",&flg,NULL));
  if (flg) {
    char part_file[PETSC_MAX_PATH_LEN];
    sprintf(part_file,"output.%d",rank);
    fptr1 = fopen(part_file,"w");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local and Global Grid Parameters are :\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local\t\t\t\tGlobal\n");
    fprintf(fptr1,"nnodesLoc = %d\t\tnnodes = %d\n",nnodesLoc,nnodes);
    fprintf(fptr1,"nedgeLoc = %d\t\t\tnedge = %d\n",nedgeLoc,nedge);
    fprintf(fptr1,"nnfacetLoc = %d\t\tnnfacet = %d\n",nnfacetLoc,nnfacet);
    fprintf(fptr1,"nvfacetLoc = %d\t\t\tnvfacet = %d\n",nvfacetLoc,nvfacet);
    fprintf(fptr1,"nffacetLoc = %d\t\t\tnffacet = %d\n",nffacetLoc,nffacet);
    fprintf(fptr1,"nsnodeLoc = %d\t\t\tnsnode = %d\n",nsnodeLoc,nsnode);
    fprintf(fptr1,"nvnodeLoc = %d\t\t\tnvnode = %d\n",nvnodeLoc,nvnode);
    fprintf(fptr1,"nfnodeLoc = %d\t\t\tnfnode = %d\n",nfnodeLoc,nfnode);
    fprintf(fptr1,"\n");
    fprintf(fptr1,"nvertices = %d\n",nvertices);
    fprintf(fptr1,"nnbound = %d\n",nnbound);
    fprintf(fptr1,"nvbound = %d\n",nvbound);
    fprintf(fptr1,"nfbound = %d\n",nfbound);
    fprintf(fptr1,"\n");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Different Orderings\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local\t\tPETSc\t\tGlobal\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nvertices; i++) fprintf(fptr1,"%d\t\t%d\t\t%d\n",i,grid->loc2pet[i],grid->loc2glo[i]);
    fprintf(fptr1,"\n");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Solid Boundary Nodes\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local\t\tPETSc\t\tGlobal\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nsnodeLoc; i++) {
      j = grid->isnode[i]-1;
      fprintf(fptr1,"%d\t\t%d\t\t%d\n",j,grid->loc2pet[j],grid->loc2glo[j]);
    }
    fprintf(fptr1,"\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"f2ntn array\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nnfacetLoc; i++) {
      fprintf(fptr1,"%d\t\t%d\t\t%d\t\t%d\n",i,grid->f2ntn[i],
              grid->f2ntn[nnfacetLoc+i],grid->f2ntn[2*nnfacetLoc+i]);
    }
    fprintf(fptr1,"\n");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Viscous Boundary Nodes\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local\t\tPETSc\t\tGlobal\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nvnodeLoc; i++) {
      j = grid->ivnode[i]-1;
      fprintf(fptr1,"%d\t\t%d\t\t%d\n",j,grid->loc2pet[j],grid->loc2glo[j]);
    }
    fprintf(fptr1,"\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"f2ntv array\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nvfacetLoc; i++) {
      fprintf(fptr1,"%d\t\t%d\t\t%d\t\t%d\n",i,grid->f2ntv[i],
              grid->f2ntv[nvfacetLoc+i],grid->f2ntv[2*nvfacetLoc+i]);
    }
    fprintf(fptr1,"\n");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Free Boundary Nodes\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local\t\tPETSc\t\tGlobal\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nfnodeLoc; i++) {
      j = grid->ifnode[i]-1;
      fprintf(fptr1,"%d\t\t%d\t\t%d\n",j,grid->loc2pet[j],grid->loc2glo[j]);
    }
    fprintf(fptr1,"\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"f2ntf array\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nffacetLoc; i++) {
      fprintf(fptr1,"%d\t\t%d\t\t%d\t\t%d\n",i,grid->f2ntf[i],
              grid->f2ntf[nffacetLoc+i],grid->f2ntf[2*nffacetLoc+i]);
    }
    fprintf(fptr1,"\n");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Neighborhood Info In Various Ordering\n");
    fprintf(fptr1,"---------------------------------------------\n");
    ICALLOC(nnodes,&p2l);
    for (i = 0; i < nvertices; i++) p2l[grid->loc2pet[i]] = i;
    for (i = 0; i < nnodesLoc; i++) {
      jstart = grid->ia[grid->loc2glo[i]] - 1;
      jend   = grid->ia[grid->loc2glo[i]+1] - 1;
      fprintf(fptr1,"Neighbors of Node %d in Local Ordering are :",i);
      for (j = jstart; j < jend; j++) fprintf(fptr1,"%d ",p2l[grid->ja[j]]);
      fprintf(fptr1,"\n");

      fprintf(fptr1,"Neighbors of Node %d in PETSc ordering are :",grid->loc2pet[i]);
      for (j = jstart; j < jend; j++) fprintf(fptr1,"%d ",grid->ja[j]);
      fprintf(fptr1,"\n");

      fprintf(fptr1,"Neighbors of Node %d in Global Ordering are :",grid->loc2glo[i]);
      for (j = jstart; j < jend; j++) fprintf(fptr1,"%d ",grid->loc2glo[p2l[grid->ja[j]]]);
      fprintf(fptr1,"\n");

    }
    fprintf(fptr1,"\n");
    CHKERRQ(PetscFree(p2l));
    fclose(fptr1);
  }

/* Free the temporary arrays */
  CHKERRQ(PetscFree(a2l));
  CHKERRMPI(MPI_Barrier(comm));
  PetscFunctionReturn(0);
}

/*
  encode 3 8-bit binary bytes as 4 '6-bit' characters, len is the number of bytes remaining, at most 3 are used
*/
void *base64_encodeblock(void *vout,const void *vin,int len)
{
  unsigned char *out = (unsigned char*)vout,in[3] = {0,0,0};
  /* Translation Table as described in RFC1113 */
  static const char cb64[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  memcpy(in,vin,PetscMin(len,3));
  out[0] = cb64[in[0] >> 2];
  out[1] = cb64[((in[0] & 0x03) << 4) | ((in[1] & 0xf0) >> 4)];
  out[2] = (unsigned char) (len > 1 ? cb64[((in[1] & 0x0f) << 2) | ((in[2] & 0xc0) >> 6)] : '=');
  out[3] = (unsigned char) (len > 2 ? cb64[in[2] & 0x3f] : '=');
  return (void*)(out+4);
}

/* Write binary data, does not do byte swapping. */
static PetscErrorCode PetscFWrite_FUN3D(MPI_Comm comm,FILE *fp,void *data,PetscInt n,PetscDataType dtype,PetscBool base64)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCheckFalse(n < 0,comm,PETSC_ERR_ARG_OUTOFRANGE,"Trying to write a negative amount of data %" PetscInt_FMT,n);
  if (!n) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  if (rank == 0) {
    size_t count;
    int    bytes;
    switch (dtype) {
    case PETSC_DOUBLE:
      size = sizeof(double);
      break;
    case PETSC_FLOAT:
      size = sizeof(float);
      break;
    case PETSC_INT:
      size = sizeof(PetscInt);
      break;
    case PETSC_CHAR:
      size = sizeof(char);
      break;
    default: SETERRQ(comm,PETSC_ERR_SUP,"Data type not supported");
    }
    bytes = size*n;
    if (base64) {
      unsigned char *buf,*ptr;
      int           i;
      size_t        b64alloc = 9 + (n*size*4) / 3 + (n*size*4) % 3;
      CHKERRQ(PetscMalloc(b64alloc,&buf));
      ptr  = buf;
      ptr  = (unsigned char*)base64_encodeblock(ptr,&bytes,3);
      ptr  = (unsigned char*)base64_encodeblock(ptr,((char*)&bytes)+3,1);
      for (i=0; i<bytes; i+=3) {
        int left = bytes - i;
        ptr = (unsigned char*)base64_encodeblock(ptr,((char*)data)+i,left);
      }
      *ptr++ = '\n';
      /* printf("encoded 4+%d raw bytes in %zd base64 chars, allocated for %zd\n",bytes,ptr-buf,b64alloc); */
      count = fwrite(buf,1,ptr-buf,fp);
      if (count < (size_t)(ptr-buf)) {
        perror("");
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"Wrote %" PetscInt_FMT " of %" PetscInt_FMT " bytes",(PetscInt)count,(PetscInt)(ptr-buf));
      }
      CHKERRQ(PetscFree(buf));
    } else {
      count = fwrite(&bytes,sizeof(int),1,fp);
      if (count != 1) {
        perror("");
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"Error writing byte count");
      }
      count = fwrite(data,size,(size_t)n,fp);
      if ((int)count != n) {
        perror("");
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_WRITE,"Wrote %" PetscInt_FMT "/%" PetscInt_FMT " array members of size %" PetscInt_FMT,(PetscInt)count,n,(PetscInt)size);
      }
    }
  }
  PetscFunctionReturn(0);
}

static void SortInt2(PetscInt *a,PetscInt *b)
{
  if (*b < *a) {
    PetscInt c = *b;
    *b = *a;
    *a = c;
  }
}

/* b = intersection(a,b) */
static PetscErrorCode IntersectInt(PetscInt na,const PetscInt *a,PetscInt *nb,PetscInt *b)
{
  PetscInt i,n,j;

  PetscFunctionBegin;
  j = 0;
  n = 0;
  for (i=0; i<*nb; i++) {
    while (j<na && a[j]<b[i]) j++;
    if (j<na && a[j]==b[i]) {
      b[n++] = b[i];
      j++;
    }
  }
  *nb = n;
  PetscFunctionReturn(0);
}

/*
  This function currently has a semantic bug: it only produces cells containing all local edges.  Since the local mesh
  does not even store edges between unowned nodes, primal cells that are effectively shared between processes will not
  be constructed. This causes visualization artifacts.

  This issue could be resolved by either (a) storing more edges from the original mesh or (b) communicating an extra
  layer of edges in this function.
*/
static PetscErrorCode InferLocalCellConnectivity(PetscInt nnodes,PetscInt nedge,const PetscInt *eptr,PetscInt *incell,PetscInt **iconn)
{
  PetscErrorCode ierr;
  PetscInt       ncell,acell,(*conn)[4],node0,node1,node2,node3,i,j,k,l,rowmax;
  PetscInt       *ui,*uj,*utmp,*tmp1,*tmp2,*tmp3,ntmp1,ntmp2,ntmp3;
#if defined(INTERLACING)
#  define GetEdge(eptr,i,n1,n2) do { n1 = eptr[i*2+0]-1; n2 = eptr[i*2+1]-1; } while (0)
#else
#  define GetEdge(eptr,i,n1,n2) do { n1 = eptr[i+0*nedge]-1; n2 = eptr[i+1*nedge]-1; } while (0)
#endif

  PetscFunctionBegin;
  *incell = -1;
  *iconn  = NULL;
  acell   = 100000;              /* allocate for this many cells */
  CHKERRQ(PetscMalloc1(acell,&conn));
  CHKERRQ(PetscMalloc2(nnodes+1,&ui,nedge,&uj));
  CHKERRQ(PetscCalloc1(nnodes,&utmp));
  /* count the number of edges in the upper-triangular matrix u */
  for (i=0; i<nedge; i++) {     /* count number of nonzeros in upper triangular matrix */
    GetEdge(eptr,i,node0,node1);
    utmp[PetscMin(node0,node1)]++;
  }
  rowmax = 0;
  ui[0]  = 0;
  for (i=0; i<nnodes; i++) {
    rowmax  = PetscMax(rowmax,utmp[i]);
    ui[i+1] = ui[i] + utmp[i]; /* convert from count to row offsets */
    utmp[i] = 0;
  }
  for (i=0; i<nedge; i++) {     /* assemble upper triangular matrix U */
    GetEdge(eptr,i,node0,node1);
    SortInt2(&node0,&node1);
    uj[ui[node0] + utmp[node0]++] = node1;
  }
  CHKERRQ(PetscFree(utmp));
  for (i=0; i<nnodes; i++) {    /* sort every row */
    PetscInt n = ui[i+1] - ui[i];
    CHKERRQ(PetscSortInt(n,&uj[ui[i]]));
  }

  /* Infer cells */
  ncell = 0;
  CHKERRQ(PetscMalloc3(rowmax,&tmp1,rowmax,&tmp2,rowmax,&tmp3));
  for (i=0; i<nnodes; i++) {
    node0 = i;
    ntmp1 = ui[node0+1] - ui[node0]; /* Number of candidates for node1 */
    CHKERRQ(PetscMemcpy(tmp1,&uj[ui[node0]],ntmp1*sizeof(PetscInt)));
    for (j=0; j<ntmp1; j++) {
      node1 = tmp1[j];
      PetscCheckFalse(node1 < 0 || nnodes <= node1,PETSC_COMM_SELF,1,"node index %" PetscInt_FMT " out of range [0,%" PetscInt_FMT ")",node1,nnodes);
      PetscCheckFalse(node1 <= node0,PETSC_COMM_SELF,1,"forward neighbor of %" PetscInt_FMT " is %" PetscInt_FMT ", should be larger",node0,node1);
      ntmp2 = ui[node1+1] - ui[node1];
      CHKERRQ(PetscMemcpy(tmp2,&uj[ui[node1]],ntmp2*sizeof(PetscInt)));
      CHKERRQ(IntersectInt(ntmp1,tmp1,&ntmp2,tmp2));
      for (k=0; k<ntmp2; k++) {
        node2 = tmp2[k];
        PetscCheckFalse(node2 < 0 || nnodes <= node2,PETSC_COMM_SELF,1,"node index %" PetscInt_FMT " out of range [0,%" PetscInt_FMT ")",node2,nnodes);
        PetscCheckFalse(node2 <= node1,PETSC_COMM_SELF,1,"forward neighbor of %" PetscInt_FMT " is %" PetscInt_FMT ", should be larger",node1,node2);
        ntmp3 = ui[node2+1] - ui[node2];
        CHKERRQ(PetscMemcpy(tmp3,&uj[ui[node2]],ntmp3*sizeof(PetscInt)));
        CHKERRQ(IntersectInt(ntmp2,tmp2,&ntmp3,tmp3));
        for (l=0; l<ntmp3; l++) {
          node3 = tmp3[l];
          PetscCheckFalse(node3 < 0 || nnodes <= node3,PETSC_COMM_SELF,1,"node index %" PetscInt_FMT " out of range [0,%" PetscInt_FMT ")",node3,nnodes);
          PetscCheckFalse(node3 <= node2,PETSC_COMM_SELF,1,"forward neighbor of %" PetscInt_FMT " is %" PetscInt_FMT ", should be larger",node2,node3);
          PetscCheckFalse(ncell > acell,PETSC_COMM_SELF,PETSC_ERR_SUP,"buffer exceeded");
          if (ntmp3 < 3) continue;
          conn[ncell][0] = node0;
          conn[ncell][1] = node1;
          conn[ncell][2] = node2;
          conn[ncell][3] = node3;
          if (0) {
            PetscViewer viewer = PETSC_VIEWER_STDOUT_WORLD;
            PetscViewerASCIIPrintf(viewer,"created cell %d: %d %d %d %d\n",ncell,node0,node1,node2,node3);
            PetscIntView(ntmp1,tmp1,viewer);
            PetscIntView(ntmp2,tmp2,viewer);
            PetscIntView(ntmp3,tmp3,viewer);
            /* uns3d.msh has a handful of "tetrahedra" that overlap by violating the following condition. As far as I
             * can tell, that means it is an invalid mesh. I don't know what the intent was. */
            PetscCheckFalse(ntmp3 > 2,PETSC_COMM_SELF,1,"More than two ways to complete a tetrahedron using a common triangle");
          }
          ncell++;
        }
      }
    }
  }
  CHKERRQ(PetscFree3(tmp1,tmp2,tmp3));
  CHKERRQ(PetscFree2(ui,uj));

  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Inferred %" PetscInt_FMT " cells with nnodes=%" PetscInt_FMT " nedge=%" PetscInt_FMT "\n",ncell,nnodes,nedge));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  *incell = ncell;
  *iconn  = (PetscInt*)conn;
  PetscFunctionReturn(0);
}

static PetscErrorCode GridCompleteOverlap(GRID *grid,PetscInt *invertices,PetscInt *inedgeOv,PetscInt **ieptrOv)
{
  PetscErrorCode ierr;
  PetscInt       nedgeLoc,nedgeOv,i,j,cnt,node0,node1,node0p,node1p,nnodes,nnodesLoc,nvertices,rstart,nodeEdgeCountAll,nodeEdgeRstart;
  PetscInt       *nodeEdgeCount,*nodeEdgeOffset,*eIdxOv,*p2l,*eptrOv;
  Vec            VNodeEdge,VNodeEdgeInfo,VNodeEdgeInfoOv,VNodeEdgeOv;
  PetscScalar    *vne,*vnei;
  IS             isglobal,isedgeOv;
  VecScatter     nescat,neiscat;
  PetscBool      flg;

  PetscFunctionBegin;
  nnodes    = grid->nnodes;     /* Total number of global nodes */
  nnodesLoc = grid->nnodesLoc;  /* Number of owned nodes */
  nvertices = grid->nvertices;  /* Number of owned+ghosted nodes */
  nedgeLoc  = grid->nedgeLoc;   /* Number of edges connected to owned nodes */

  /* Count the number of neighbors of each owned node */
  CHKERRMPI(MPI_Scan(&nnodesLoc,&rstart,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  rstart -= nnodesLoc;
  CHKERRQ(PetscMalloc2(nnodesLoc,&nodeEdgeCount,nnodesLoc,&nodeEdgeOffset));
  CHKERRQ(PetscMemzero(nodeEdgeCount,nnodesLoc*sizeof(*nodeEdgeCount)));
  for (i=0; i<nedgeLoc; i++) {
    GetEdge(grid->eptr,i,node0,node1);
    node0p = grid->loc2pet[node0];
    node1p = grid->loc2pet[node1];
    if (rstart <= node0p && node0p < rstart+nnodesLoc) nodeEdgeCount[node0p-rstart]++;
    if (rstart <= node1p && node1p < rstart+nnodesLoc) nodeEdgeCount[node1p-rstart]++;
  }
  /* Get the offset in the node-based edge array */
  nodeEdgeOffset[0] = 0;
  for (i=0; i<nnodesLoc-1; i++) nodeEdgeOffset[i+1] = nodeEdgeOffset[i] + nodeEdgeCount[i];
  nodeEdgeCountAll = nodeEdgeCount[nnodesLoc-1] + nodeEdgeOffset[nnodesLoc-1];

  /* Pack a Vec by node of all the edges for that node. The nodes are stored by global index */
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,nodeEdgeCountAll,PETSC_DETERMINE,&VNodeEdge));
  CHKERRQ(PetscMemzero(nodeEdgeCount,nnodesLoc*sizeof(*nodeEdgeCount)));
  CHKERRQ(VecGetArray(VNodeEdge,&vne));
  for (i=0; i<nedgeLoc; i++) {
    GetEdge(grid->eptr,i,node0,node1);
    node0p = grid->loc2pet[node0];
    node1p = grid->loc2pet[node1];
    if (rstart <= node0p && node0p < rstart+nnodesLoc) vne[nodeEdgeOffset[node0p-rstart] + nodeEdgeCount[node0p-rstart]++] = node1p;
    if (rstart <= node1p && node1p < rstart+nnodesLoc) vne[nodeEdgeOffset[node1p-rstart] + nodeEdgeCount[node1p-rstart]++] = node0p;
  }
  CHKERRQ(VecRestoreArray(VNodeEdge,&vne));
  CHKERRQ(VecGetOwnershipRange(VNodeEdge,&nodeEdgeRstart,NULL));

  /* Move the count and offset into a Vec so that we can use VecScatter, translating offset from local to global */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&VNodeEdgeInfo));
  CHKERRQ(VecSetSizes(VNodeEdgeInfo,2*nnodesLoc,2*nnodes));
  CHKERRQ(VecSetBlockSize(VNodeEdgeInfo,2));
  CHKERRQ(VecSetType(VNodeEdgeInfo,VECMPI));

  CHKERRQ(VecGetArray(VNodeEdgeInfo,&vnei));
  for (i=0; i<nnodesLoc; i++) {
    vnei[i*2+0] = nodeEdgeCount[i];                   /* Total number of edges from this vertex */
    vnei[i*2+1] = nodeEdgeOffset[i] + nodeEdgeRstart; /* Now the global index in the next comm round */
  }
  CHKERRQ(VecRestoreArray(VNodeEdgeInfo,&vnei));
  CHKERRQ(PetscFree2(nodeEdgeCount,nodeEdgeOffset));

  /* Create a Vec to receive the edge count and global offset for each node in owned+ghosted, get them, and clean up */
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&VNodeEdgeInfoOv));
  CHKERRQ(VecSetSizes(VNodeEdgeInfoOv,2*nvertices,2*nvertices));
  CHKERRQ(VecSetBlockSize(VNodeEdgeInfoOv,2));
  CHKERRQ(VecSetType(VNodeEdgeInfoOv,VECSEQ));

  CHKERRQ(ISCreateBlock(PETSC_COMM_WORLD,2,nvertices,grid->loc2pet,PETSC_COPY_VALUES,&isglobal)); /* Address the nodes in overlap to get info from */
  CHKERRQ(VecScatterCreate(VNodeEdgeInfo,isglobal,VNodeEdgeInfoOv,NULL,&neiscat));
  CHKERRQ(VecScatterBegin(neiscat,VNodeEdgeInfo,VNodeEdgeInfoOv,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(neiscat,VNodeEdgeInfo,VNodeEdgeInfoOv,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&neiscat));
  CHKERRQ(VecDestroy(&VNodeEdgeInfo));
  CHKERRQ(ISDestroy(&isglobal));

  /* Create a Vec to receive the actual edges for all nodes (owned and ghosted), execute the scatter */
  nedgeOv = 0;                  /* First count the number of edges in the complete overlap */
  CHKERRQ(VecGetArray(VNodeEdgeInfoOv,&vnei));
  for (i=0; i<nvertices; i++) nedgeOv += (PetscInt)vnei[2*i+0];
  /* Allocate for the global indices in VNodeEdge of the edges to receive */
  CHKERRQ(PetscMalloc1(nedgeOv,&eIdxOv));
  for (i=0,cnt=0; i<nvertices; i++) {
    for (j=0; j<(PetscInt)vnei[2*i+0]; j++) eIdxOv[cnt++] = (PetscInt)vnei[2*i+1] + j;
  }
  CHKERRQ(VecRestoreArray(VNodeEdgeInfoOv,&vnei));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,nedgeOv,eIdxOv,PETSC_USE_POINTER,&isedgeOv));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,nedgeOv,&VNodeEdgeOv));
  CHKERRQ(VecScatterCreate(VNodeEdge,isedgeOv,VNodeEdgeOv,NULL,&nescat));
  CHKERRQ(VecScatterBegin(nescat,VNodeEdge,VNodeEdgeOv,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(nescat,VNodeEdge,VNodeEdgeOv,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&nescat));
  CHKERRQ(VecDestroy(&VNodeEdge));
  CHKERRQ(ISDestroy(&isedgeOv));
  CHKERRQ(PetscFree(eIdxOv));

  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] %s: number of edges before pruning: %" PetscInt_FMT ", half=%" PetscInt_FMT "\n",rank,PETSC_FUNCTION_NAME,nedgeOv,nedgeOv/2));

  /* Create the non-scalable global-to-local index map. Yuck, but it has already been used elsewhere. */
  CHKERRQ(PetscMalloc1(nnodes,&p2l));
  for (i=0; i<nnodes; i++) p2l[i] = -1;
  for (i=0; i<nvertices; i++) p2l[grid->loc2pet[i]] = i;
  if (1) {
    PetscInt m = 0;
    for (i=0; i<nnodes; i++) m += (p2l[i] >= 0);
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] %s: number of global indices that map to local indices: %" PetscInt_FMT "; nvertices=%" PetscInt_FMT " nnodesLoc=%" PetscInt_FMT " nnodes=%" PetscInt_FMT "\n",rank,PETSC_FUNCTION_NAME,m,nvertices,nnodesLoc,nnodes));
  }

  /* Log each edge connecting nodes in owned+ghosted exactly once */
  CHKERRQ(VecGetArray(VNodeEdgeInfoOv,&vnei));
  CHKERRQ(VecGetArray(VNodeEdgeOv,&vne));
  /* First count the number of edges to keep */
  nedgeOv = 0;
  for (i=0,cnt=0; i<nvertices; i++) {
    PetscInt n = (PetscInt)vnei[2*i+0]; /* number of nodes connected to i */
    node0 = i;
    for (j=0; j<n; j++) {
      node1p = vne[cnt++];
      node1  = p2l[node1p];
      if (node0 < node1) nedgeOv++;
    }
  }
  /* Array of edges to keep */
  CHKERRQ(PetscMalloc1(2*nedgeOv,&eptrOv));
  nedgeOv = 0;
  for (i=0,cnt=0; i<nvertices; i++) {
    PetscInt n = (PetscInt)vnei[2*i+0]; /* number of nodes connected to i */
    node0 = i;
    for (j=0; j<n; j++) {
      node1p = vne[cnt++];
      node1  = p2l[node1p];
      if (node0 < node1) {
        eptrOv[2*nedgeOv+0] = node0;
        eptrOv[2*nedgeOv+1] = node1;
        nedgeOv++;
      }
    }
  }
  CHKERRQ(VecRestoreArray(VNodeEdgeInfoOv,&vnei));
  CHKERRQ(VecRestoreArray(VNodeEdgeOv,&vne));
  CHKERRQ(VecDestroy(&VNodeEdgeInfoOv));
  CHKERRQ(VecDestroy(&VNodeEdgeOv));
  CHKERRQ(PetscFree(p2l));

  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] %s: nedgeLoc=%" PetscInt_FMT " nedgeOv=%" PetscInt_FMT "\n",rank,PETSC_FUNCTION_NAME,nedgeLoc,nedgeOv));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  flg  = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-complete_overlap",&flg,NULL));
  if (flg) {
    *invertices = grid->nvertices; /* We did not change the number of vertices */
    *inedgeOv   = nedgeOv;
    *ieptrOv    = eptrOv;
  } else {
    *invertices = grid->nvertices;
    *inedgeOv   = nedgeLoc;
    CHKERRQ(PetscFree(eptrOv));
    CHKERRQ(PetscMalloc1(2*nedgeLoc,&eptrOv));
    CHKERRQ(PetscMemcpy(eptrOv,grid->eptr,2*nedgeLoc*sizeof(PetscInt)));
    *ieptrOv    = eptrOv;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode WritePVTU(AppCtx *user,const char *fname,PetscBool base64)
{
  GRID              *grid  = user->grid;
  TstepCtx          *tsCtx = user->tsCtx;
  FILE              *vtu,*pvtu;
  char              pvtu_fname[PETSC_MAX_PATH_LEN],vtu_fname[PETSC_MAX_PATH_LEN];
  MPI_Comm          comm;
  PetscMPIInt       rank,size;
  PetscInt          i,nvertices = 0,nedgeLoc = 0,ncells,bs,nloc,boffset = 0,*eptr = NULL;
  PetscErrorCode    ierr;
  Vec               Xloc,Xploc,Xuloc;
  unsigned char     *celltype;
  int               *celloffset,*conn,*cellrank;
  const PetscScalar *x;
  PetscScalar       *xu,*xp;
  const char        *byte_order = PetscBinaryBigEndian() ? "BigEndian" : "LittleEndian";

  PetscFunctionBegin;
  CHKERRQ(GridCompleteOverlap(user->grid,&nvertices,&nedgeLoc,&eptr));
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRMPI(MPI_Comm_size(comm,&size));
#if defined(PETSC_USE_COMPLEX) || !defined(PETSC_USE_REAL_DOUBLE) || defined(PETSC_USE_64BIT_INDICES)
  SETERRQ(comm,PETSC_ERR_SUP,"This function is only implemented for scalar-type=real precision=double, 32-bit indices");
#endif
  CHKERRQ(PetscSNPrintf(pvtu_fname,sizeof(pvtu_fname),"%s-%" PetscInt_FMT ".pvtu",fname,tsCtx->itstep));
  CHKERRQ(PetscSNPrintf(vtu_fname,sizeof(vtu_fname),"%s-%" PetscInt_FMT "-%" PetscInt_FMT ".vtu",fname,tsCtx->itstep,rank));
  CHKERRQ(PetscFOpen(comm,pvtu_fname,"w",&pvtu));
  CHKERRQ(PetscFPrintf(comm,pvtu,"<?xml version=\"1.0\"?>\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"%s\">\n",byte_order));
  CHKERRQ(PetscFPrintf(comm,pvtu," <PUnstructuredGrid GhostLevel=\"0\">\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"  <PPointData Scalars=\"Pressure\" Vectors=\"Velocity\">\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"   <PDataArray type=\"Float64\" Name=\"Pressure\" NumberOfComponents=\"1\" />\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"   <PDataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\" />\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"  </PPointData>\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"  <PCellData>\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"   <PDataArray type=\"Int32\" Name=\"Rank\" NumberOfComponents=\"1\" />\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"  </PCellData>\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"  <PPoints>\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"   <PDataArray type=\"Float64\" Name=\"Position\" NumberOfComponents=\"3\" />\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"  </PPoints>\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"  <PCells>\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"   <PDataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" />\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"   <PDataArray type=\"Int32\" Name=\"offsets\"      NumberOfComponents=\"1\" />\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"   <PDataArray type=\"UInt8\" Name=\"types\"        NumberOfComponents=\"1\" />\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"  </PCells>\n"));
  for (i=0; i<size; i++) {
    CHKERRQ(PetscFPrintf(comm,pvtu,"  <Piece Source=\"%s-%" PetscInt_FMT "-%" PetscInt_FMT ".vtu\" />\n",fname,tsCtx->itstep,i));
  }
  CHKERRQ(PetscFPrintf(comm,pvtu," </PUnstructuredGrid>\n"));
  CHKERRQ(PetscFPrintf(comm,pvtu,"</VTKFile>\n"));
  CHKERRQ(PetscFClose(comm,pvtu));

  Xloc = grid->qnodeLoc;
  CHKERRQ(VecScatterBegin(grid->scatter,grid->qnode,Xloc,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(grid->scatter,grid->qnode,Xloc,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecGetBlockSize(Xloc,&bs));
  PetscCheckFalse(bs != 4,PETSC_COMM_WORLD,PETSC_ERR_ARG_INCOMP,"expected block size 4, got %" PetscInt_FMT,bs);
  CHKERRQ(VecGetSize(Xloc,&nloc));
  PetscCheckFalse(nloc/bs != nvertices,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"expected nloc/bs=%" PetscInt_FMT " to match nvertices=%" PetscInt_FMT,nloc/bs,nvertices);
  PetscCheckFalse(nvertices != grid->nvertices,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"expected nvertices=%" PetscInt_FMT " to match grid->nvertices=%" PetscInt_FMT,nvertices,grid->nvertices);
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,nvertices,&Xploc));

  CHKERRQ(VecCreate(PETSC_COMM_SELF,&Xuloc));
  CHKERRQ(VecSetSizes(Xuloc,3*nvertices,3*nvertices));
  CHKERRQ(VecSetBlockSize(Xuloc,3));
  CHKERRQ(VecSetType(Xuloc,VECSEQ));

  CHKERRQ(VecGetArrayRead(Xloc,&x));
  CHKERRQ(VecGetArray(Xploc,&xp));
  CHKERRQ(VecGetArray(Xuloc,&xu));
  for (i=0; i<nvertices; i++) {
    xp[i]     = x[i*4+0];
    xu[i*3+0] = x[i*4+1];
    xu[i*3+1] = x[i*4+2];
    xu[i*3+2] = x[i*4+3];
  }
  CHKERRQ(VecRestoreArrayRead(Xloc,&x));

  CHKERRQ(InferLocalCellConnectivity(nvertices,nedgeLoc,eptr,&ncells,&conn));

  CHKERRQ(PetscFOpen(PETSC_COMM_SELF,vtu_fname,"w",&vtu));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"<?xml version=\"1.0\"?>\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"%s\">\n",byte_order));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu," <UnstructuredGrid>\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"  <Piece NumberOfPoints=\"%" PetscInt_FMT "\" NumberOfCells=\"%" PetscInt_FMT "\">\n",nvertices,ncells));

  /* Solution fields */
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"   <PointData Scalars=\"Pressure\" Vectors=\"Velocity\">\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"    <DataArray type=\"Float64\" Name=\"Pressure\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n",boffset));
  boffset += nvertices*sizeof(PetscScalar) + sizeof(int);
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"    <DataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n",boffset));
  boffset += nvertices*3*sizeof(PetscScalar) + sizeof(int);
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"   </PointData>\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"   <CellData>\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"    <DataArray type=\"Int32\" Name=\"Rank\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n",boffset));
  boffset += ncells*sizeof(int) + sizeof(int);
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"   </CellData>\n"));
  /* Coordinate positions */
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"   <Points>\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"    <DataArray type=\"Float64\" Name=\"Position\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n",boffset));
  boffset += nvertices*3*sizeof(double) + sizeof(int);
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"   </Points>\n"));
  /* Cell connectivity */
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"   <Cells>\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"    <DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n",boffset));
  boffset += ncells*4*sizeof(int) + sizeof(int);
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"    <DataArray type=\"Int32\" Name=\"offsets\"      NumberOfComponents=\"1\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n",boffset));
  boffset += ncells*sizeof(int) + sizeof(int);
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"    <DataArray type=\"UInt8\" Name=\"types\"        NumberOfComponents=\"1\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n",boffset));
  boffset += ncells*sizeof(unsigned char) + sizeof(int);
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"   </Cells>\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"  </Piece>\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu," </UnstructuredGrid>\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu," <AppendedData encoding=\"%s\">\n",base64 ? "base64" : "raw"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"_"));

  /* Write pressure */
  CHKERRQ(PetscFWrite_FUN3D(PETSC_COMM_SELF,vtu,xp,nvertices,PETSC_SCALAR,base64));

  /* Write velocity */
  CHKERRQ(PetscFWrite_FUN3D(PETSC_COMM_SELF,vtu,xu,nvertices*3,PETSC_SCALAR,base64));

  /* Label cell rank, not a measure of computation because nothing is actually computed at cells.  This is written
   * primarily to aid in debugging. The partition for computation should label vertices. */
  CHKERRQ(PetscMalloc1(ncells,&cellrank));
  for (i=0; i<ncells; i++) cellrank[i] = rank;
  CHKERRQ(PetscFWrite_FUN3D(PETSC_COMM_SELF,vtu,cellrank,ncells,PETSC_INT,base64));
  CHKERRQ(PetscFree(cellrank));

  CHKERRQ(PetscFWrite_FUN3D(PETSC_COMM_SELF,vtu,grid->xyz,nvertices*3,PETSC_DOUBLE,base64));
  CHKERRQ(PetscFWrite_FUN3D(PETSC_COMM_SELF,vtu,conn,ncells*4,PETSC_INT,base64));
  CHKERRQ(PetscFree(conn));

  CHKERRQ(PetscMalloc1(ncells,&celloffset));
  for (i=0; i<ncells; i++) celloffset[i] = 4*(i+1);
  CHKERRQ(PetscFWrite_FUN3D(PETSC_COMM_SELF,vtu,celloffset,ncells,PETSC_INT,base64));
  CHKERRQ(PetscFree(celloffset));

  CHKERRQ(PetscMalloc1(ncells,&celltype));
  for (i=0; i<ncells; i++) celltype[i] = 10; /* VTK_TETRA */
  CHKERRQ(PetscFWrite_FUN3D(PETSC_COMM_SELF,vtu,celltype,ncells,PETSC_CHAR,base64));
  CHKERRQ(PetscFree(celltype));

  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"\n </AppendedData>\n"));
  CHKERRQ(PetscFPrintf(PETSC_COMM_SELF,vtu,"</VTKFile>\n"));
  CHKERRQ(PetscFClose(PETSC_COMM_SELF,vtu));

  CHKERRQ(VecRestoreArray(Xploc,&xp));
  CHKERRQ(VecRestoreArray(Xuloc,&xu));
  CHKERRQ(VecDestroy(&Xploc));
  CHKERRQ(VecDestroy(&Xuloc));
  CHKERRQ(PetscFree(eptr));
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
int SetPetscDS(GRID *grid,TstepCtx *tsCtx)
/*---------------------------------------------------------------------*/
{
  int                    ierr,i,j,bs;
  int                    nnodes,jstart,jend,nbrs_diag,nbrs_offd;
  int                    nnodesLoc,nvertices;
  int                    *val_diag,*val_offd,*svertices,*loc2pet;
  IS                     isglobal,islocal;
  ISLocalToGlobalMapping isl2g;
  PetscBool              flg;
  MPI_Comm               comm = PETSC_COMM_WORLD;

  PetscFunctionBegin;
  nnodes    = grid->nnodes;
  nnodesLoc = grid->nnodesLoc;
  nvertices = grid->nvertices;
  loc2pet   = grid->loc2pet;
  bs        = 4;

  /* Set up the PETSc datastructures */

  CHKERRQ(VecCreate(comm,&grid->qnode));
  CHKERRQ(VecSetSizes(grid->qnode,bs*nnodesLoc,bs*nnodes));
  CHKERRQ(VecSetBlockSize(grid->qnode,bs));
  CHKERRQ(VecSetType(grid->qnode,VECMPI));

  CHKERRQ(VecDuplicate(grid->qnode,&grid->res));
  CHKERRQ(VecDuplicate(grid->qnode,&tsCtx->qold));
  CHKERRQ(VecDuplicate(grid->qnode,&tsCtx->func));

  CHKERRQ(VecCreate(MPI_COMM_SELF,&grid->qnodeLoc));
  CHKERRQ(VecSetSizes(grid->qnodeLoc,bs*nvertices,bs*nvertices));
  CHKERRQ(VecSetBlockSize(grid->qnodeLoc,bs));
  CHKERRQ(VecSetType(grid->qnodeLoc,VECSEQ));

  ierr = VecCreate(comm,&grid->grad);
  CHKERRQ(VecSetSizes(grid->grad,3*bs*nnodesLoc,3*bs*nnodes));
  CHKERRQ(VecSetBlockSize(grid->grad,3*bs));
  CHKERRQ(VecSetType(grid->grad,VECMPI));

  ierr = VecCreate(MPI_COMM_SELF,&grid->gradLoc);
  CHKERRQ(VecSetSizes(grid->gradLoc,3*bs*nvertices,3*bs*nvertices));
  CHKERRQ(VecSetBlockSize(grid->gradLoc,3*bs));
  CHKERRQ(VecSetType(grid->gradLoc,VECSEQ));

/* Create Scatter between the local and global vectors */
/* First create scatter for qnode */
  CHKERRQ(ISCreateStride(MPI_COMM_SELF,bs*nvertices,0,1,&islocal));
#if defined(INTERLACING)
#if defined(BLOCKING)
  ICALLOC(nvertices,&svertices);
  for (i=0; i < nvertices; i++) svertices[i] = loc2pet[i];
  CHKERRQ(ISCreateBlock(MPI_COMM_SELF,bs,nvertices,svertices,PETSC_COPY_VALUES,&isglobal));
#else
  ICALLOC(bs*nvertices,&svertices);
  for (i = 0; i < nvertices; i++)
    for (j = 0; j < bs; j++) svertices[j+bs*i] = j + bs*loc2pet[i];
  CHKERRQ(ISCreateGeneral(MPI_COMM_SELF,bs*nvertices,svertices,PETSC_COPY_VALUES,&isglobal));
#endif
#else
  ICALLOC(bs*nvertices,&svertices);
  for (j = 0; j < bs; j++)
    for (i = 0; i < nvertices; i++) svertices[j*nvertices+i] = j*nvertices + loc2pet[i];
  CHKERRQ(ISCreateGeneral(MPI_COMM_SELF,bs*nvertices,svertices,PETSC_COPY_VALUES,&isglobal));
#endif
  CHKERRQ(PetscFree(svertices));
  CHKERRQ(VecScatterCreate(grid->qnode,isglobal,grid->qnodeLoc,islocal,&grid->scatter));
  CHKERRQ(ISDestroy(&isglobal));
  CHKERRQ(ISDestroy(&islocal));

/* Now create scatter for gradient vector of qnode */
  CHKERRQ(ISCreateStride(MPI_COMM_SELF,3*bs*nvertices,0,1,&islocal));
#if defined(INTERLACING)
#if defined(BLOCKING)
  ICALLOC(nvertices,&svertices);
  for (i=0; i < nvertices; i++) svertices[i] = loc2pet[i];
  CHKERRQ(ISCreateBlock(MPI_COMM_SELF,3*bs,nvertices,svertices,PETSC_COPY_VALUES,&isglobal));
#else
  ICALLOC(3*bs*nvertices,&svertices);
  for (i = 0; i < nvertices; i++)
    for (j = 0; j < 3*bs; j++) svertices[j+3*bs*i] = j + 3*bs*loc2pet[i];
  CHKERRQ(ISCreateGeneral(MPI_COMM_SELF,3*bs*nvertices,svertices,PETSC_COPY_VALUES,&isglobal));
#endif
#else
  ICALLOC(3*bs*nvertices,&svertices);
  for (j = 0; j < 3*bs; j++)
    for (i = 0; i < nvertices; i++) svertices[j*nvertices+i] = j*nvertices + loc2pet[i];
  CHKERRQ(ISCreateGeneral(MPI_COMM_SELF,3*bs*nvertices,svertices,PETSC_COPY_VALUES,&isglobal));
#endif
  ierr = PetscFree(svertices);
  CHKERRQ(VecScatterCreate(grid->grad,isglobal,grid->gradLoc,islocal,&grid->gradScatter));
  CHKERRQ(ISDestroy(&isglobal));
  CHKERRQ(ISDestroy(&islocal));

/* Store the number of non-zeroes per row */
#if defined(INTERLACING)
#if defined(BLOCKING)
  ICALLOC(nnodesLoc,&val_diag);
  ICALLOC(nnodesLoc,&val_offd);
  for (i = 0; i < nnodesLoc; i++) {
    jstart    = grid->ia[i] - 1;
    jend      = grid->ia[i+1] - 1;
    nbrs_diag = 0;
    nbrs_offd = 0;
    for (j = jstart; j < jend; j++) {
      if ((grid->ja[j] >= rstart) && (grid->ja[j] < (rstart+nnodesLoc))) nbrs_diag++;
      else nbrs_offd++;
    }
    val_diag[i] = nbrs_diag;
    val_offd[i] = nbrs_offd;
  }
  ierr = MatCreateBAIJ(comm,bs,bs*nnodesLoc,bs*nnodesLoc,
                       bs*nnodes,bs*nnodes,PETSC_DEFAULT,val_diag,
                       PETSC_DEFAULT,val_offd,&grid->A);CHKERRQ(ierr);
#else
  ICALLOC(nnodesLoc*4,&val_diag);
  ICALLOC(nnodesLoc*4,&val_offd);
  for (i = 0; i < nnodesLoc; i++) {
    jstart    = grid->ia[i] - 1;
    jend      = grid->ia[i+1] - 1;
    nbrs_diag = 0;
    nbrs_offd = 0;
    for (j = jstart; j < jend; j++) {
      if ((grid->ja[j] >= rstart) && (grid->ja[j] < (rstart+nnodesLoc))) nbrs_diag++;
      else nbrs_offd++;
    }
    for (j = 0; j < 4; j++) {
      row           = 4*i + j;
      val_diag[row] = nbrs_diag*4;
      val_offd[row] = nbrs_offd*4;
    }
  }
  ierr = MatCreateAIJ(comm,bs*nnodesLoc,bs*nnodesLoc,
                      bs*nnodes,bs*nnodes,NULL,val_diag,
                      NULL,val_offd,&grid->A);CHKERRQ(ierr);
#endif
  CHKERRQ(PetscFree(val_diag));
  CHKERRQ(PetscFree(val_offd));

#else
  PetscCheckFalse(size > 1,PETSC_COMM_SELF,1,"Parallel case not supported in non-interlaced case");
  ICALLOC(nnodes*4,&val_diag);
  ICALLOC(nnodes*4,&val_offd);
  for (j = 0; j < 4; j++)
    for (i = 0; i < nnodes; i++) {
      int row;
      row           = i + j*nnodes;
      jstart        = grid->ia[i] - 1;
      jend          = grid->ia[i+1] - 1;
      nbrs_diag     = jend - jstart;
      val_diag[row] = nbrs_diag*4;
      val_offd[row] = 0;
    }
  /* ierr = MatCreateSeqAIJ(MPI_COMM_SELF,nnodes*4,nnodes*4,NULL,
                        val,&grid->A);*/
  ierr = MatCreateAIJ(comm,bs*nnodesLoc,bs*nnodesLoc,
                      bs*nnodes,bs*nnodes,NULL,val_diag,
                      NULL,val_offd,&grid->A);CHKERRQ(ierr);
  CHKERRQ(MatSetBlockSize(grid->A,bs));
  CHKERRQ(PetscFree(val_diag));
  CHKERRQ(PetscFree(val_offd));
#endif

  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(0,"-mem_use",&flg,NULL));
  if (flg) {
    CHKERRQ(PetscMemoryView(PETSC_VIEWER_STDOUT_WORLD,"Memory usage after allocating PETSc data structures\n"));
  }

/* Set local to global mapping for setting the matrix elements in
   local ordering : first set row by row mapping
*/
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF,bs,nvertices,loc2pet,PETSC_COPY_VALUES,&isl2g);
  CHKERRQ(MatSetLocalToGlobalMapping(grid->A,isl2g,isl2g));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&isl2g));
  PetscFunctionReturn(0);
}

/*================================= CLINK ===================================*/
/*                                                                           */
/* Used in establishing the links between FORTRAN common blocks and C        */
/*                                                                           */
/*===========================================================================*/
EXTERN_C_BEGIN
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
int set_up_grid(GRID *grid)
{
  int nnodes,nedge;
  int nsface,nvface,nfface,nbface;
  int tnode,ierr;
  /*int vface,lnodes,nnz,ncell,kvisc,ilu0,nsrch,ileast,ifcn,valloc;*/
  /*int nsnode,nvnode,nfnode; */
  /*int mgzero=0;*/ /* Variable so we dont allocate memory for multigrid */
  /*int jalloc;*/  /* If jalloc=1 allocate space for dfp and dfm */
  /*
  * stuff to read in dave's grids
  */
  /*int nnbound,nvbound,nfbound,nnfacet,nvfacet,nffacet,ntte;*/
  /* end of stuff */

  PetscFunctionBegin;
  nnodes = grid->nnodes;
  tnode  = grid->nnodes;
  nedge  = grid->nedge;
  nsface = grid->nsface;
  nvface = grid->nvface;
  nfface = grid->nfface;
  nbface = nsface + nvface + nfface;

  /*ncell  = grid->ncell;
  vface  = grid->nedge;
  lnodes = grid->nnodes;
  nsnode = grid->nsnode;
  nvnode = grid->nvnode;
  nfnode = grid->nfnode;
  nsrch  = c_gmcom->nsrch;
  ilu0   = c_gmcom->ilu0;
  ileast = grid->ileast;
  ifcn   = c_gmcom->ifcn;
  jalloc = 0;
  kvisc  = grid->jvisc;*/

  /* if (ilu0 >=1 && ifcn == 1) jalloc=0;*/

  /*
  * stuff to read in dave's grids
  */
  /*nnbound = grid->nnbound;
  nvbound = grid->nvbound;
  nfbound = grid->nfbound;
  nnfacet = grid->nnfacet;
  nvfacet = grid->nvfacet;
  nffacet = grid->nffacet;
  ntte    = grid->ntte;*/
  /* end of stuff */

  /* if (!ileast) lnodes = 1;
    printf("In set_up_grid->jvisc = %d\n",grid->jvisc);

  if (grid->jvisc != 2 && grid->jvisc != 4 && grid->jvisc != 6)vface = 1;
  printf(" vface = %d \n",vface);
  if (grid->jvisc < 3) tnode = 1;
  valloc = 1;
  if (grid->jvisc ==  0)valloc = 0;*/

  /*PetscPrintf(PETSC_COMM_WORLD," nsnode= %d nvnode= %d nfnode= %d\n",nsnode,nvnode,nfnode);*/
  /*PetscPrintf(PETSC_COMM_WORLD," nsface= %d nvface= %d nfface= %d\n",nsface,nvface,nfface);
  PetscPrintf(PETSC_COMM_WORLD," nbface= %d\n",nbface);*/
  /* Now allocate memory for the other grid arrays */
  /* ICALLOC(nedge*2,  &grid->eptr); */
  ICALLOC(nsface,   &grid->isface);
  ICALLOC(nvface,   &grid->ivface);
  ICALLOC(nfface,   &grid->ifface);
  /* ICALLOC(nsnode,   &grid->isnode);
    ICALLOC(nvnode,   &grid->ivnode);
    ICALLOC(nfnode,   &grid->ifnode);*/
  /*ICALLOC(nnodes,   &grid->clist);
  ICALLOC(nnodes,   &grid->iupdate);
  ICALLOC(nsface*2, &grid->sface);
  ICALLOC(nvface*2, &grid->vface);
  ICALLOC(nfface*2, &grid->fface);
  ICALLOC(lnodes,   &grid->icount);*/
  /*FCALLOC(nnodes,   &grid->x);
  FCALLOC(nnodes,   &grid->y);
  FCALLOC(nnodes,   &grid->z);
  FCALLOC(nnodes,   &grid->area);*/
  /*
  * FCALLOC(nnodes*4, &grid->gradx);
  * FCALLOC(nnodes*4, &grid->grady);
  * FCALLOC(nnodes*4, &grid->gradz);
  * FCALLOC(nnodes,   &grid->cdt);
  */
  /*
  * FCALLOC(nnodes*4, &grid->qnode);
  * FCALLOC(nnodes*4, &grid->dq);
  * FCALLOC(nnodes*4, &grid->res);
  * FCALLOC(jalloc*nnodes*4*4,&grid->A);
  * FCALLOC(nnodes*4,  &grid->B);
  * FCALLOC(jalloc*nedge*4*4,&grid->dfp);
  * FCALLOC(jalloc*nedge*4*4,&grid->dfm);
  */
  /*FCALLOC(nsnode,   &grid->sxn);
  FCALLOC(nsnode,   &grid->syn);
  FCALLOC(nsnode,   &grid->szn);
  FCALLOC(nsnode,   &grid->sa);
  FCALLOC(nvnode,   &grid->vxn);
  FCALLOC(nvnode,   &grid->vyn);
  FCALLOC(nvnode,   &grid->vzn);
  FCALLOC(nvnode,   &grid->va);
  FCALLOC(nfnode,   &grid->fxn);
  FCALLOC(nfnode,   &grid->fyn);
  FCALLOC(nfnode,   &grid->fzn);
  FCALLOC(nfnode,   &grid->fa);
  FCALLOC(nedge,    &grid->xn);
  FCALLOC(nedge,    &grid->yn);
  FCALLOC(nedge,    &grid->zn);
  FCALLOC(nedge,    &grid->rl);*/

  FCALLOC(nbface*15,&grid->us);
  FCALLOC(nbface*15,&grid->vs);
  FCALLOC(nbface*15,&grid->as);
  /*
  * FCALLOC(nnodes*4, &grid->phi);
  * FCALLOC(nnodes,   &grid->r11);
  * FCALLOC(nnodes,   &grid->r12);
  * FCALLOC(nnodes,   &grid->r13);
  * FCALLOC(nnodes,   &grid->r22);
  * FCALLOC(nnodes,   &grid->r23);
  * FCALLOC(nnodes,   &grid->r33);
  */
  /*
  * Allocate memory for viscous length scale if turbulent
  */
  if (grid->jvisc >= 3) {
    FCALLOC(tnode,  &grid->slen);
    FCALLOC(nnodes, &grid->turbre);
    FCALLOC(nnodes, &grid->amut);
    FCALLOC(tnode,  &grid->turbres);
    FCALLOC(nedge,  &grid->dft1);
    FCALLOC(nedge,  &grid->dft2);
  }
  /*
  * Allocate memory for MG transfer
  */
  /*
  ICALLOC(mgzero*nsface,   &grid->isford);
  ICALLOC(mgzero*nvface,   &grid->ivford);
  ICALLOC(mgzero*nfface,   &grid->ifford);
  ICALLOC(mgzero*nnodes,   &grid->nflag);
  ICALLOC(mgzero*nnodes,   &grid->nnext);
  ICALLOC(mgzero*nnodes,   &grid->nneigh);
  ICALLOC(mgzero*ncell,    &grid->ctag);
  ICALLOC(mgzero*ncell,    &grid->csearch);
  ICALLOC(valloc*ncell*4,  &grid->c2n);
  ICALLOC(valloc*ncell*6,  &grid->c2e);
  grid->c2c = (int*)grid->dfp;
  ICALLOC(ncell*4,  &grid->c2c);
  ICALLOC(nnodes,   &grid->cenc);
  if (grid->iup == 1) {
      ICALLOC(mgzero*nnodes*3, &grid->icoefup);
      FCALLOC(mgzero*nnodes*3, &grid->rcoefup);
  }
  if (grid->idown == 1) {
      ICALLOC(mgzero*nnodes*3, &grid->icoefdn);
      FCALLOC(mgzero*nnodes*3, &grid->rcoefdn);
  }
  FCALLOC(nnodes*4, &grid->ff);
  FCALLOC(tnode,    &grid->turbff);
  */
  /*
  * If using GMRES (nsrch>0) allocate memory
  */
  /* NoEq = 0;
  if (nsrch > 0)NoEq = 4*nnodes;
  if (nsrch < 0)NoEq = nnodes;
  FCALLOC(NoEq,          &grid->AP);
  FCALLOC(NoEq,          &grid->Xgm);
  FCALLOC(NoEq,          &grid->temr);
  FCALLOC((abs(nsrch)+1)*NoEq,&grid->Fgm);
  */
  /*
  * stuff to read in dave's grids
  */
  /*
  ICALLOC(nnbound,  &grid->ncolorn);
  ICALLOC(nnbound*100,&grid->countn);
  ICALLOC(nvbound,  &grid->ncolorv);
  ICALLOC(nvbound*100,&grid->countv);
  ICALLOC(nfbound,  &grid->ncolorf);
  ICALLOC(nfbound*100,&grid->countf);
  */
  /*ICALLOC(nnbound,  &grid->nntet);
  ICALLOC(nnbound,  &grid->nnpts);
  ICALLOC(nvbound,  &grid->nvtet);
  ICALLOC(nvbound,  &grid->nvpts);
  ICALLOC(nfbound,  &grid->nftet);
  ICALLOC(nfbound,  &grid->nfpts);
  ICALLOC(nnfacet*4,&grid->f2ntn);
  ICALLOC(nvfacet*4,&grid->f2ntv);
  ICALLOC(nffacet*4,&grid->f2ntf);*/
  PetscFunctionReturn(0);
}

/*========================== WRITE_FINE_GRID ================================*/
/*                                                                           */
/* Write memory locations and other information for the fine grid            */
/*                                                                           */
/*===========================================================================*/
int write_fine_grid(GRID *grid)
{
  FILE *output;

  PetscFunctionBegin;
/* open file for output      */
/* call the output frame.out */

  PetscCheckFalse(!(output = fopen("frame.out","a")),PETSC_COMM_SELF,1,"can't open frame.out");
  fprintf(output,"information for fine grid\n");
  fprintf(output,"\n");
  fprintf(output," address of fine grid = %p\n",(void*)grid);

  fprintf(output,"grid.nnodes  = %d\n",grid->nnodes);
  fprintf(output,"grid.ncell   = %d\n",grid->ncell);
  fprintf(output,"grid.nedge   = %d\n",grid->nedge);
  fprintf(output,"grid.nsface  = %d\n",grid->nsface);
  fprintf(output,"grid.nvface  = %d\n",grid->nvface);
  fprintf(output,"grid.nfface  = %d\n",grid->nfface);
  fprintf(output,"grid.nsnode  = %d\n",grid->nsnode);
  fprintf(output,"grid.nvnode  = %d\n",grid->nvnode);
  fprintf(output,"grid.nfnode  = %d\n",grid->nfnode);
  /*
  fprintf(output,"grid.eptr    = %p\n",grid->eptr);
  fprintf(output,"grid.isface  = %p\n",grid->isface);
  fprintf(output,"grid.ivface  = %p\n",grid->ivface);
  fprintf(output,"grid.ifface  = %p\n",grid->ifface);
  fprintf(output,"grid.isnode  = %p\n",grid->isnode);
  fprintf(output,"grid.ivnode  = %p\n",grid->ivnode);
  fprintf(output,"grid.ifnode  = %p\n",grid->ifnode);
  fprintf(output,"grid.c2n     = %p\n",grid->c2n);
  fprintf(output,"grid.c2e     = %p\n",grid->c2e);
  fprintf(output,"grid.xyz     = %p\n",grid->xyz);
   */
  /*fprintf(output,"grid.y       = %p\n",grid->xyz);
    fprintf(output,"grid.z       = %p\n",grid->z);*/
  /*
  fprintf(output,"grid.area    = %p\n",grid->area);
  fprintf(output,"grid.qnode   = %p\n",grid->qnode);
   */
/*
  fprintf(output,"grid.gradx   = %p\n",grid->gradx);
  fprintf(output,"grid.grady   = %p\n",grid->grady);
  fprintf(output,"grid.gradz   = %p\n",grid->gradz);
*/
  /*
  fprintf(output,"grid.cdt     = %p\n",grid->cdt);
  fprintf(output,"grid.sxn     = %p\n",grid->sxn);
  fprintf(output,"grid.syn     = %p\n",grid->syn);
  fprintf(output,"grid.szn     = %p\n",grid->szn);
  fprintf(output,"grid.vxn     = %p\n",grid->vxn);
  fprintf(output,"grid.vyn     = %p\n",grid->vyn);
  fprintf(output,"grid.vzn     = %p\n",grid->vzn);
  fprintf(output,"grid.fxn     = %p\n",grid->fxn);
  fprintf(output,"grid.fyn     = %p\n",grid->fyn);
  fprintf(output,"grid.fzn     = %p\n",grid->fzn);
  fprintf(output,"grid.xyzn    = %p\n",grid->xyzn);
   */
  /*fprintf(output,"grid.yn      = %p\n",grid->yn);
  fprintf(output,"grid.zn      = %p\n",grid->zn);
  fprintf(output,"grid.rl      = %p\n",grid->rl);*/
  fclose(output);
  PetscFunctionReturn(0);
}

#if defined(_OPENMP) && defined(HAVE_EDGE_COLORING)
int EdgeColoring(int nnodes,int nedge,int *e2n,int *eperm,int *ncle,int *counte)
{
  int ncolore = *ncle = 0;
  int iedg    = 0,ib = 0,ie = nedge,tagcount;
  int i,n1,n2;
  int *tag;
  ICALLOC(nnodes,&tag);
  while (ib < ie) {
    for (i = 0; i < nnodes; i++) tag[i] = 0;
    counte[ncolore] = 0;
    for (i = ib; i < ie; i++) {
      n1       = e2n[i];
      n2       = e2n[i+nedge];
      tagcount = tag[n1]+tag[n2];
      /* If tagcount = 0 then this edge belongs in this color */
      if (!tagcount) {
        tag[n1]         = 1;
        tag[n2]         = 1;
        e2n[i]          = e2n[iedg];
        e2n[i+nedge]    = e2n[iedg+nedge];
        e2n[iedg]       = n1;
        e2n[iedg+nedge] = n2;
        n1              = eperm[i];
        eperm[i]        = eperm[iedg];
        eperm[iedg]     = n1;
        iedg++;
        counte[ncolore]+= 1;
      }
    }
    ib = iedg;
    ncolore++;
  }
  *ncle = ncolore;
  return 0;
}
#endif
#if defined(PARCH_IRIX64) && defined(USE_HW_COUNTERS)
int EventCountersBegin(int *gen_start,PetscScalar *time_start_counters)
{
  PetscErrorCode ierr;
  PetscCheckFalse((*gen_start = start_counters(event0,event1)) < 0,PETSC_COMM_SELF,1,"Error in start_counters");
  CHKERRQ(PetscTime(&time_start_counters));
  return 0;
}

int EventCountersEnd(int gen_start,PetscScalar time_start_counters)
{
  int         gen_read,ierr;
  PetscScalar time_read_counters;
  long long   _counter0,_counter1;

  PetscCheckFalse((gen_read = read_counters(event0,&_counter0,event1,&_counter1)) < 0,PETSC_COMM_SELF,1,"Error in read_counter");
  CHKERRQ(PetscTime(&&time_read_counters));
  PetscCheckFalse(gen_read != gen_start,PETSC_COMM_SELF,1,"Lost Counters!! Aborting ...");
  counter0      += _counter0;
  counter1      += _counter1;
  time_counters += time_read_counters-time_start_counters;
  return 0;
}
#endif
