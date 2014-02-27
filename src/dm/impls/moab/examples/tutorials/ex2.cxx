static char help[] = "Solves C_t =  -D*C_xx + F(C) + R(C) + D(C) from Brian Wirth's SciDAC project.\n";

/*
        C_t =  -D*C_xx + F(C) + R(C) + D(C) from Brian Wirth's SciDAC project. (A DMMoab fork of Barry's advection-diffusion-reaction/ex10.c)

        D*C_xx  - diffusion of He[1-5] and V[1] and I[1]
        F(C)  -   forcing function; He being created.
        R(C)  -   reaction terms   (clusters combining)
        D(C)  -   dissociation terms (cluster breaking up)

        Sample Options:
          -ts_monitor_draw_solution               -- plot the solution for each concentration as a function of x each in a separate 1d graph
              -draw_fields_by_name 1-He-2-V,1-He  -- only plot the solution for these two concentrations
          -mymonitor                              -- plot the concentrations of He and V as a function of x and cluster size (2d contour plot)
          -da_refine <n=1,2,...>                  -- run on a finer grid
          -ts_max_steps maxsteps                  -- maximum number of time-steps to take
          -ts_final_time time                     -- maximum time to compute to

    Rules for maximum number of He allowed for V in cluster

    Usage: ./ex2 -n 10 -snes_mf -mymonitor -ts_monitor
           ./ex2 -file <$XOLOTL_DIR/benchmarks/tungsten.txt> -snes_mf -mymonitor -ts_monitor
*/

#include <petscdmmoab.h>
#include <petscts.h>

#define TINY_CLUSTER_CONFIG

#ifndef TINY_CLUSTER_CONFIG
/*    Hard wire the number of cluster sizes for He, V, and I, and He-V */
#define  NHe          9
#define  NV           10   /* 50 */
#define  NI           2
#define  MHeV         10   /* 50 */  /* maximum V size in He-V */
PetscInt NHeV[MHeV+1];     /* maximum He size in an He-V with given V */
#define  MNHeV        451  /* 6778 */
#define  DOF          (NHe + NV + NI + MNHeV)
#else
/*    Hard wire the number of cluster sizes for He, V, and I, and He-V */
#define  NHe          5
#define  NV           1   /* 50 */
#define  NI           2
#define  MHeV         1   /* 50 */  /* maximum V size in He-V */
PetscInt NHeV[MHeV+1];     /* maximum He size in an He-V with given V */
#define  MNHeV        3  /* 6778 */
#define  DOF          (NHe + NV + NI + MNHeV)
#endif

/*
     Define all the concentrations (there is one of these unions at each grid point)

      He[He] represents the clusters of pure Helium of size He
      V[V] the Vacencies of size V,
      I[I] represents the clusters of Interstials of size I,  and
      HeV[He][V]  the mixed Helium-Vacancy clusters of size He and V

      The variables He, V, I are always used to index into the concentrations of He, V, and I respectively
      Note that unlike in traditional C code the indices for He[], V[] and I[] run from 1 to N, NOT 0 to N-1
      (the use of the union below "tricks" the C compiler to allow the indices to start at 1.)

*/
typedef struct {
  PetscScalar He[NHe];
  PetscScalar V[NV];
  PetscScalar I[NI];
  PetscScalar HeV[MNHeV];
} Concentrations;

/*
     Holds problem specific options and data
*/
typedef struct {
  PetscScalar HeDiffusion[6];
  PetscScalar VDiffusion[2];
  PetscScalar IDiffusion[2];
  PetscScalar forcingScale;
  PetscScalar reactionScale;
  PetscScalar dissociationScale;
  PetscInt    nelements;
} AppCtx;

extern PetscErrorCode RHSFunction(TS ts,PetscReal ftime,Vec C,Vec F,void *ptr);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode InitialConditions(DM,Vec);
extern PetscErrorCode InitializeVariableSystem(DM);
extern PetscErrorCode MyMonitorSetUp(TS);
extern PetscErrorCode GetDfill(PetscInt*,void*);
extern PetscErrorCode MyLoadData(MPI_Comm,const char*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;                  /* nonlinear solver */
  Vec            C;                   /* solution */
  PetscErrorCode ierr;
  DM             dm;                  /* manages the grid data */
  AppCtx         ctx;                 /* holds problem specific paramters */
  /* PetscInt       He,*ofill,*dfill; */
  char           filename[PETSC_MAX_PATH_LEN];
  const PetscReal   bounds[2] = {0.0,8.0};
  PetscBool      flg;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char*)0,help);

  PetscFunctionBeginUser;
  ierr = PetscOptionsGetString(NULL,"-file",filename,PETSC_MAX_PATH_LEN,&flg);
  if (flg) {
    ierr = MyLoadData(PETSC_COMM_WORLD,filename);CHKERRQ(ierr);
  }

  ctx.nelements      = 10;
  ierr = PetscOptionsGetInt(NULL,"-n",&ctx.nelements,NULL);CHKERRQ(ierr);

  ctx.HeDiffusion[1]    = 1000*2.95e-4; /* From Tibo's notes times 1,000 */
  ctx.HeDiffusion[2]    = 1000*3.24e-4;
  ctx.HeDiffusion[3]    = 1000*2.26e-4;
  ctx.HeDiffusion[4]    = 1000*1.68e-4;
  ctx.HeDiffusion[5]    = 1000*5.20e-5;
  ctx.VDiffusion[1]     = 1000*2.71e-3;
  ctx.IDiffusion[1]     = 1000*2.13e-4;
  ctx.forcingScale      = 100.;         /* made up numbers */
  ctx.reactionScale     = .001;
  ctx.dissociationScale = .0001;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMoabCreateBoxMesh(PETSC_COMM_WORLD, 1, PETSC_FALSE, bounds, ctx.nelements, 1, &dm);CHKERRQ(ierr);
  ierr = InitializeVariableSystem(dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  /* The only spatial coupling in the Jacobian (diffusion) is for the first 5 He, the first V, and the first I.
     The ofill (thought of as a DOF by DOF 2d (row-oriented) array) represents the nonzero coupling between degrees
     of freedom at one point with degrees of freedom on the adjacent point to the left or right. A 1 at i,j in the
     ofill array indicates that the degree of freedom i at a point is coupled to degree of freedom j at the
     adjacent point. In this case ofill has only a few diagonal entries since the only spatial coupling is regular diffusion */
  /* TODO: DMMoab does not have the capability to do DOF-Coupling as of now.. Assume full block coupling */
  /*
  ierr = PetscMalloc(DOF*DOF*sizeof(PetscInt),&ofill);CHKERRQ(ierr);
  ierr = PetscMemzero(ofill,DOF*DOF*sizeof(PetscInt));CHKERRQ(ierr);
  for (He=0; He<PetscMin(NHe,5); He++) ofill[He*DOF + He] = 1;
  ofill[NHe*DOF + NHe] = ofill[(NHe+NV)*DOF + NHe + NV] = 1;
  */

  /*
    dfil (thought of as a DOF by DOF 2d (row-oriented) array) repesents the nonzero coupling between degrees of
   freedom within a single grid point, i.e. the reaction and dissassociation interactions. */
  /*
  ierr = PetscMalloc(DOF*DOF*sizeof(PetscInt),&dfill);CHKERRQ(ierr);
  ierr = PetscMemzero(dfill,DOF*DOF*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = GetDfill(dfill,&ctx);CHKERRQ(ierr);
  ierr = DMDASetBlockFills(da,dfill,ofill);CHKERRQ(ierr);
  ierr = PetscFree(ofill);CHKERRQ(ierr);
  ierr = PetscFree(dfill);CHKERRQ(ierr);
  */
 

  /*  Extract global vector to hold solution */  
  ierr = DMCreateGlobalVector(dm,&C);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobian,&ctx);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,C);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetInitialTimeStep(ts,0.0,.001);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,100,50.0);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = MyMonitorSetUp(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(dm,C);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,C);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&C);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

/*
   cHeV is "trick" to allow easy accessing of the values in the HeV portion of the Concentrations.
   cHeV[i] points to the beginning of each row of HeV[] with V indexing starting a 1.

*/
#undef __FUNCT__
#define __FUNCT__ "cHeVCreate"
PetscErrorCode cHeVCreate(PetscReal ***cHeV)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(MHeV*sizeof(PetscScalar),cHeV);CHKERRQ(ierr);
  (*cHeV)--;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "cHeVInitialize"
PetscErrorCode cHeVInitialize(const PetscScalar *start,PetscReal **cHeV)
{
  PetscInt       i;

  PetscFunctionBegin;
  cHeV[1] = ((PetscScalar*) start) - 1 + NHe + NV + NI;
  for (i=1; i<MHeV; i++) {
    cHeV[i+1] = cHeV[i] + NHeV[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "cHeVDestroy"
PetscErrorCode cHeVDestroy(PetscReal **cHeV)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  cHeV++;
  ierr = PetscFree(cHeV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "InitializeVariableSystem"
PetscErrorCode InitializeVariableSystem(DM dm)
{
  PetscErrorCode ierr;
  PetscInt       I,He,V,cnt = 0;
  char           field[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  ierr = DMMoabSetFieldNames(dm, DOF, NULL);CHKERRQ(ierr);
  ierr = DMMoabSetBlockSize(dm, DOF);CHKERRQ(ierr);

  /* Name each of the concentrations */
  for (He=1; He<NHe+1; He++) {
    ierr = PetscSNPrintf(field,PETSC_MAX_PATH_LEN,"%d-He",He);CHKERRQ(ierr);
    ierr = DMMoabSetFieldName(dm,cnt++,field);CHKERRQ(ierr);
  }
  for (V=1; V<NV+1; V++) {
    ierr = PetscSNPrintf(field,PETSC_MAX_PATH_LEN,"%d-V",V);CHKERRQ(ierr);
    ierr = DMMoabSetFieldName(dm,cnt++,field);CHKERRQ(ierr);
  }
  for (I=1; I<NI+1; I++) {
    ierr = PetscSNPrintf(field,PETSC_MAX_PATH_LEN,"%d-I",I);CHKERRQ(ierr);
    ierr = DMMoabSetFieldName(dm,cnt++,field);CHKERRQ(ierr);
  }

  for (He=1; He<MHeV+1; He++) {
    for (V=1; V<NHeV[He]+1; V++) {
      ierr = PetscSNPrintf(field,PETSC_MAX_PATH_LEN,"%d-He-%d-V",He,V);CHKERRQ(ierr);
      ierr = DMMoabSetFieldName(dm,cnt++,field);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "InitialConditions"
PetscErrorCode InitialConditions(DM dm,Vec C)
{
  PetscErrorCode    ierr;
  PetscInt          i,I,He,V;
  Concentrations    *c;
  PetscReal         **cHeV;
  const moab::Range *vowned;

  PetscFunctionBeginUser;
  /*
     Get pointer to vector data
  */
  ierr = DMMoabVecGetArray(dm,C,&c);CHKERRQ(ierr);
  /* Shift the c pointer to allow accessing with index of 1, instead of 0 */
  c = (Concentrations*)(((PetscScalar*)c)-1);

  /*
     Get local grid vertices
  */
  ierr = DMMoabGetLocalVertices(dm, &vowned, NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  ierr = cHeVCreate(&cHeV);CHKERRQ(ierr);
  for(moab::Range::iterator iter = vowned->begin(); iter != vowned->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    ierr = DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &i);CHKERRQ(ierr);

    for (He=1; He<NHe+1; He++) c[i].He[He] = 0.0;
    for (V=1;  V<NV+1;   V++)  c[i].V[V]   = 1.0;
    for (I=1; I <NI+1;   I++)  c[i].I[I]   = 1.0;
    ierr = cHeVInitialize(&c[i].He[1],cHeV);CHKERRQ(ierr);
    for (V=1; V<MHeV+1; V++) {
      for (He=1; He<NHeV[V]+1; He++)  cHeV[V][He] = 0.0;
    }
  }
  ierr = cHeVDestroy(cHeV);CHKERRQ(ierr);

  /*
     Restore vectors
  */
  c    = (Concentrations*)(((PetscScalar*)c)+1);
  ierr = DMMoabVecRestoreArray(dm,C,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
/*
   RHSFunction - Evaluates nonlinear function that defines the ODE

   Input Parameters:
.  ts - the TS context
.  U - input vector
.  ptr - optional user-defined context

   Output Parameter:
.  F - function values
 */
PetscErrorCode RHSFunction(TS ts,PetscReal ftime,Vec C,Vec F,void *ptr)
{
  AppCtx            *ctx = (AppCtx*) ptr;
  DM                dm;
  PetscErrorCode    ierr;
  PetscInt          xi,Mx,He,he,V,v,I,i;
  PetscReal         hx,sx,x,vpos[3],**cHeV,**fHeV;
  Concentrations    *c,*f;
  Vec               localC;
  const moab::Range *vowned,*elocal;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&localC);CHKERRQ(ierr);
  ierr = DMMoabGetSize(dm,&Mx,PETSC_IGNORE);CHKERRQ(ierr);
  hx   = 8.0/(PetscReal)(Mx); sx = 1.0/(hx*hx);
  ierr = cHeVCreate(&cHeV);CHKERRQ(ierr);
  ierr = cHeVCreate(&fHeV);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(dm,C,INSERT_VALUES,localC);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,C,INSERT_VALUES,localC);CHKERRQ(ierr);

  ierr = VecSet(F,0.0);CHKERRQ(ierr);

  /*
    Get pointers to vector data
  */
  ierr = DMMoabVecGetArrayRead(dm,localC,&c);CHKERRQ(ierr);
  /* Shift the c pointer to allow accessing with index of 1, instead of 0 */
  c    = (Concentrations*)(((PetscScalar*)c)-1);
  ierr = DMMoabVecGetArray(dm,F,&f);CHKERRQ(ierr);
  f    = (Concentrations*)(((PetscScalar*)f)-1);

  /*
     Get local grid boundaries
  */
  ierr = DMMoabGetLocalVertices(dm, &vowned, NULL);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);

  /*
     Loop over grid points computing ODE terms for each grid point
  */
  for(moab::Range::iterator iter = vowned->begin(); iter != vowned->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    ierr = DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &xi);CHKERRQ(ierr);

    ierr = DMMoabGetVertexCoordinates(dm,1,&vhandle,vpos);CHKERRQ(ierr);
    x = vpos[0];

    /* -------------------------------------------------------------
     ---- Compute diffusion over the locally owned part of the grid
    */
    /* He clusters larger than 5 do not diffuse -- are immobile */
    for (He=1; He<PetscMin(NHe+1,6); He++) {
      f[xi].He[He] +=  ctx->HeDiffusion[He]*(-2.0*c[xi].He[He] + c[xi-1].He[He] + c[xi+1].He[He])*sx;
    }

    /* V and I clusters ONLY of size 1 diffuse */
    f[xi].V[1] +=  ctx->VDiffusion[1]*(-2.0*c[xi].V[1] + c[xi-1].V[1] + c[xi+1].V[1])*sx;
    f[xi].I[1] +=  ctx->IDiffusion[1]*(-2.0*c[xi].I[1] + c[xi-1].I[1] + c[xi+1].I[1])*sx;

    /* Mixed He - V clusters are immobile  */

    /* ----------------------------------------------------------------
     ---- Compute forcing that produces He of cluster size 1
          Crude cubic approximation of graph from Tibo's notes
    */
    f[xi].He[1] +=  ctx->forcingScale*PetscMax(0.0,0.0006*x*x*x  - 0.0087*x*x + 0.0300*x);

    ierr = cHeVInitialize(&c[xi].He[1],cHeV);CHKERRQ(ierr);
    ierr = cHeVInitialize(&f[xi].He[1],fHeV);CHKERRQ(ierr);

    /* -------------------------------------------------------------------------
     ---- Compute dissociation terms that removes an item from a cluster
          I assume dissociation means losing only a single item from a cluster
          I cannot tell from the notes if clusters can break up into any sub-size.
    */
    /*   He[He] ->  He[He-1] + He[1] */
    for (He=2; He<NHe+1; He++) {
      f[xi].He[He-1] += ctx->dissociationScale*c[xi].He[He];
      f[xi].He[1]    += ctx->dissociationScale*c[xi].He[He];
      f[xi].He[He]   -= ctx->dissociationScale*c[xi].He[He];
    }

    /*   V[V] ->  V[V-1] + V[1] */
    for (V=2; V<NV+1; V++) {
      f[xi].V[V-1] += ctx->dissociationScale*c[xi].V[V];
      f[xi].V[1]   += ctx->dissociationScale*c[xi].V[V];
      f[xi].V[V]   -= ctx->dissociationScale*c[xi].V[V];
    }

    /*   I[I] ->  I[I-1] + I[1] */
    for (I=2; I<NI+1; I++) {
      f[xi].I[I-1] += ctx->dissociationScale*c[xi].I[I];
      f[xi].I[1]   += ctx->dissociationScale*c[xi].I[I];
      f[xi].I[I]   -= ctx->dissociationScale*c[xi].I[I];
    }

    /*   He[He]-V[1] ->  He[He] + V[1]  */
    for (He=1; He<NHeV[1]+1; He++) {
      f[xi].He[He] += 1000*ctx->dissociationScale*cHeV[1][He];
      f[xi].V[1]   += 1000*ctx->dissociationScale*cHeV[1][He];
      fHeV[1][He]  -= 1000*ctx->dissociationScale*cHeV[1][He];
    }

    /*   He[1]-V[V] ->  He[1] + V[V]  */
    for (V=2; V<MHeV+1; V++) {
      f[xi].He[1]  += 1000*ctx->dissociationScale*cHeV[V][1];
      f[xi].V[V]   += 1000*ctx->dissociationScale*cHeV[V][1];
      fHeV[V][1]   -= 1000*ctx->dissociationScale*cHeV[V][1];
    }

    /*   He[He]-V[V] ->  He[He-1]-V[V] + He[1]  */
    for (V=2; V<MHeV+1; V++) {
      for (He=2; He<NHeV[V]+1; He++) {
        f[xi].He[1]   += 1000*ctx->dissociationScale*cHeV[V][He];
        fHeV[V][He-1] += 1000*ctx->dissociationScale*cHeV[V][He];
        fHeV[V][He]   -= 1000*ctx->dissociationScale*cHeV[V][He];
      }
    }

    /*   He[He]-V[V] ->  He[He]-V[V-1] + V[1]  */
    for (V=2; V<MHeV+1; V++) {
      for (He=2; He<NHeV[V-1]+1; He++) {
        f[xi].V[1]    += 1000*ctx->dissociationScale*cHeV[V][He];
        fHeV[V-1][He] += 1000*ctx->dissociationScale*cHeV[V][He];
        fHeV[V][He]   -= 1000*ctx->dissociationScale*cHeV[V][He];
      }
    }

    /*   He[He]-V[V] ->  He[He]-V[V+1] + I[1]  */
    for (V=1; V<MHeV; V++) {
      for (He=1; He<NHeV[V]+1; He++) {
        fHeV[V+1][He] += 1000*ctx->dissociationScale*cHeV[V][He];
        f[xi].I[1]    += 1000*ctx->dissociationScale*cHeV[V][He];
        fHeV[V][He]   -= 1000*ctx->dissociationScale*cHeV[V][He];
      }
    }

    /* ----------------------------------------------------------------
     ---- Compute reaction terms that can create a cluster of given size
    */
    /*   He[He] + He[he] -> He[He+he]  */
    for (He=2; He<NHe+1; He++) {
      /* compute all pairs of clusters of smaller size that can combine to create a cluster of size He,
         remove the upper half since they are symmetric to the lower half of the pairs. For example
              when He = 5 (cluster size 5) the pairs are
                 1   4
                 2   2
                 3   2  these last two are not needed in the sum since they repeat from above
                 4   1  this is why he < (He/2) + 1            */
      for (he=1; he<(He/2)+1; he++) {
        f[xi].He[He] += ctx->reactionScale*c[xi].He[he]*c[xi].He[He-he];

        /* remove the two clusters that merged to form the larger cluster */
        f[xi].He[he]    -= ctx->reactionScale*c[xi].He[he]*c[xi].He[He-he];
        f[xi].He[He-he] -= ctx->reactionScale*c[xi].He[he]*c[xi].He[He-he];
      }
    }

    /*   V[V]  +  V[v] ->  V[V+v]  */
    for (V=2; V<NV+1; V++) {
      for (v=1; v<(V/2)+1; v++) {
        f[xi].V[V]   += ctx->reactionScale*c[xi].V[v]*c[xi].V[V-v];
        f[xi].V[v]   -= ctx->reactionScale*c[xi].V[v]*c[xi].V[V-v];
        f[xi].V[V-v] -= ctx->reactionScale*c[xi].V[v]*c[xi].V[V-v];
      }
    }

    /*   I[I] +  I[i] -> I[I+i] */
    for (I=2; I<NI+1; I++) {
      for (i=1; i<(I/2)+1; i++) {
        f[xi].I[I]   += ctx->reactionScale*c[xi].I[i]*c[xi].I[I-i];
        f[xi].I[i]   -= ctx->reactionScale*c[xi].I[i]*c[xi].I[I-i];
        f[xi].I[I-i] -= ctx->reactionScale*c[xi].I[i]*c[xi].I[I-i];
      }
    }

    /* He[1] +  V[1]  ->  He[1]-V[1] */
    fHeV[1][1]  += 1000*ctx->reactionScale*c[xi].He[1]*c[xi].V[1];
    f[xi].He[1] -= 1000*ctx->reactionScale*c[xi].He[1]*c[xi].V[1];
    f[xi].V[1]  -= 1000*ctx->reactionScale*c[xi].He[1]*c[xi].V[1];

    /*  He[He]-V[V] + He[he] -> He[He+he]-V[V]  */
    for (V=1; V<MHeV+1; V++) {
      for (He=1; He<NHeV[V]; He++) {
        for (he=1; he+He<NHeV[V]+1; he++) {
          fHeV[V][He+he] += ctx->reactionScale*cHeV[V][He]*c[xi].He[he];
          f[xi].He[he]   -= ctx->reactionScale*cHeV[V][He]*c[xi].He[he];
          fHeV[V][He]    -= ctx->reactionScale*cHeV[V][He]*c[xi].He[he];
        }
      }
    }

    /*  He[He]-V[V] + V[1] -> He[He][V+1] */
    for (V=1; V<MHeV; V++) {
      for (He=1; He<NHeV[V+1]; He++) {
          fHeV[V+1][He] += ctx->reactionScale*cHeV[V][He]*c[xi].V[1];
          /* remove the two clusters that merged to form the larger cluster */
          f[xi].V[1]  -= ctx->reactionScale*cHeV[V][He]*c[xi].V[1];
          fHeV[V][He] -= ctx->reactionScale*cHeV[V][He]*c[xi].V[1];
      }
    }

    /*  He[He]-V[V]  + He[he]-V[v] -> He[He+he][V+v]  */
    /*  Currently the reaction rates for this are zero */


    /*  V[V] + I[I]  ->   V[V-I] if V > I else I[I-V] */
    for (V=1; V<NV+1; V++) {
      for (I=1; I<PetscMin(V,NI); I++) {
        f[xi].V[V-I] += ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
        f[xi].V[V]   -= ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
        f[xi].I[I]   -= ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
      }
      for (I=V+1; I<NI+1; I++) {
          f[xi].I[I-V] += ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
          f[xi].V[V]   -= ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
          f[xi].I[I]   -= ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
      }
    }
  }

  /*
     Restore vectors
  */
  c    = (Concentrations*)(((PetscScalar*)c)+1);
  ierr = DMMoabVecRestoreArrayRead(dm,localC,&c);CHKERRQ(ierr);
  f    = (Concentrations*)(((PetscScalar*)f)+1);
  ierr = DMMoabVecRestoreArray(dm,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localC);CHKERRQ(ierr);
  ierr = cHeVDestroy(cHeV);CHKERRQ(ierr);
  ierr = cHeVDestroy(fHeV);CHKERRQ(ierr);
//  VecView(F,0);
//  std::cin.get();
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */

#define BARRY_IMPL

#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
/*
    Compute the Jacobian entries based on IFuction() and insert them into the matrix
*/
PetscErrorCode RHSJacobian(TS ts,PetscReal ftime,Vec C,Mat A,Mat J,void *ptr)
{
  AppCtx               *ctx = (AppCtx*) ptr;
  DM                   dm;
  PetscErrorCode       ierr;
  PetscInt             xi,Mx,/*xs,xm,*/He,he,V,v,I,i;
  PetscInt             row[3],col[3],dof_indices[2*DOF],lcols[2],rcols[2],left,right;
  PetscReal            hx,sx,x,val[6],vpos[3],lvals[2],rvals[2];
  const Concentrations *c,*f;
  Vec                  localC;
  const moab::EntityHandle *connect;
  PetscInt             vpere=2;
  const moab::Range    *elocal;
  const PetscReal      *rowstart,*colstart;
  const PetscReal      **cHeV,**fHeV;
  static PetscBool     initialized = PETSC_FALSE;

  PetscFunctionBeginUser;
  ierr = cHeVCreate((PetscScalar***)&cHeV);CHKERRQ(ierr);
  ierr = cHeVCreate((PetscScalar***)&fHeV);CHKERRQ(ierr);
  ierr = MatZeroEntries(J);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&localC);CHKERRQ(ierr);
  ierr = DMMoabGetSize(dm,&Mx,PETSC_IGNORE);CHKERRQ(ierr);
  hx   = 8.0/(PetscReal)(Mx); sx = 1.0/(hx*hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(dm,C,INSERT_VALUES,localC);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,C,INSERT_VALUES,localC);CHKERRQ(ierr);

  /*
    Get pointers to vector data

    The f[] is dummy, values are never set into it. It is only used to determine the 
    local row for the entries in the Jacobian
  */
  ierr = DMMoabVecGetArrayRead(dm,localC,&c);CHKERRQ(ierr);
  /* Shift the c pointer to allow accessing with index of 1, instead of 0 */
  c    = (Concentrations*)(((PetscScalar*)c)-1);
  ierr = DMMoabVecGetArray(dm,C,&f);CHKERRQ(ierr);
  f    = (Concentrations*)(((PetscScalar*)f)-1);

  /*
     Get local grid boundaries
  */
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);
//  ierr = DMMoabGetLocalVertices(dm, &vowned, NULL);CHKERRQ(ierr);
//  ierr = DMDAGetCorners(dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
//  rowstart = &f[xs].He[1] -  N*N - 3*N;
//  colstart = &c[xs-1].He[1];

#ifdef BARRY_IMPL
  rowstart = &f[1].He[1] - DOF;
  colstart = &c[0].He[1];
#else
  rowstart = &f[-1].He[1] - DOF*0;
  colstart = &c[-1].He[1];
#endif

  if (!initialized) {
    /* Compute ODE terms over the locally owned part of the grid 
      Assemble the operator by looping over edges and computing
      contribution for each vertex dof                         */
    for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
      const moab::EntityHandle ehandle = *iter;

      // Get connectivity information in canonical order
      ierr = DMMoabGetElementConnectivity(dm, ehandle, &vpere, &connect);CHKERRQ(ierr);
      if (vpere != 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only EDGE2 element bases are supported in the current example. n(Connectivity)=%D.\n", vpere);

      ierr = DMMoabGetDofsBlockedLocal(dm, 1, &connect[0], &xi);CHKERRQ(ierr);
      ierr = DMMoabGetDofsBlockedLocal(dm, vpere, connect, dof_indices);CHKERRQ(ierr);
//      ierr = DMMoabGetDofsLocal(dm, vpere, connect, dof_indices);CHKERRQ(ierr);

//      const PetscInt lcols[] = {idl,idr}, rcols[] = {idr, idl};

//      const PetscInt lcols[] = {idl,idr}, rcols[] = {idr, idl};
//      ierr = MatSetValuesLocal(*Jpre,1,&idr,2,rcols,&e_vals[0][0][0],ADD_VALUES);CHKERRQ(ierr);
//      ierr = MatSetValuesLocal(*Jpre,1,&idl,2,lcols,&e_vals[0][0][0],ADD_VALUES);CHKERRQ(ierr);

      const int& idl = dof_indices[0]-1;
      const int& idr = dof_indices[1]-1;
      PetscPrintf(PETSC_COMM_WORLD,"\t [%D, %D] dof_indices [%D %D]\n", idl,idr,dof_indices[0],dof_indices[1]);

      ierr = cHeVInitialize(&c[xi].He[1],(PetscScalar**)cHeV);CHKERRQ(ierr);
      ierr = cHeVInitialize(&f[xi].He[1],(PetscScalar**)fHeV);CHKERRQ(ierr);

      /* -------------------------------------------------------------
         ---- Compute diffusion over the locally owned part of the grid
       */
      /* He clusters larger than 5 do not diffuse -- are immobile */
      for (He=1; He<PetscMin(NHe+1,6); He++) {
#ifdef BARRY_IMPL
        row[0] = &f[xi].He[He] - rowstart;
        col[0] = &c[xi-1].He[He] - colstart;
        col[1] = &c[xi].He[He] - colstart;
        col[2] = &c[xi+1].He[He] - colstart;
        val[0] = ctx->HeDiffusion[He]*sx;
        val[1] = -2.0*ctx->HeDiffusion[He]*sx;
        val[2] = ctx->HeDiffusion[He]*sx;
        PetscPrintf(PETSC_COMM_WORLD,"[1] Setting row %D: Col = [%D, %D, %D]\n", row[0],col[0],col[1],col[2]);
        ierr = MatSetValuesLocal(J,1,row,3,col,val,ADD_VALUES);CHKERRQ(ierr);
#else
        left = &f[idl].He[He] - rowstart; right = &f[idr].He[He] - rowstart;
        lcols[0] = &c[idl].He[He] - colstart; lcols[1] = &c[idr].He[He] - colstart;
        rcols[0] = lcols[1]; rcols[1] = lcols[0];
        lvals[0] = ctx->HeDiffusion[He]*sx; lvals[1] = -ctx->HeDiffusion[He]*sx;
        rvals[0] = lvals[1]; rvals[1] = lvals[0];
        PetscPrintf(PETSC_COMM_WORLD,"\t [1] row [%D %D]: Left Cols = [%D, %D]  Right Cols = [%D, %D]\n", left,right,lcols[0],lcols[1],rcols[0],rcols[1]);
        ierr = MatSetValuesLocal(J,1,&left,2,lcols,lvals,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatSetValuesLocal(J,1,&right,2,rcols,rvals,ADD_VALUES);CHKERRQ(ierr);
#endif
      }

      /* V and I clusters ONLY of size 1 diffuse */
#ifdef BARRY_IMPL
      row[0] = &f[xi].V[1] - rowstart;
      col[0] = &c[xi-1].V[1] - colstart;
      col[1] = &c[xi].V[1] - colstart;
      col[2] = &c[xi+1].V[1] - colstart;
      val[0] = ctx->VDiffusion[1]*sx;
      val[1] = -2.0*ctx->VDiffusion[1]*sx;
      val[2] = ctx->VDiffusion[1]*sx;
      PetscPrintf(PETSC_COMM_WORLD,"[2] Setting row %D: Col = [%D, %D, %D]\n", row[0],col[0],col[1],col[2]);
      ierr = MatSetValuesLocal(J,1,row,3,col,val,ADD_VALUES);CHKERRQ(ierr);
#else
      left = &f[idl].V[1] - rowstart; right = &f[idr].V[1] - rowstart;
      lcols[0] = &c[idl].V[1] - colstart; lcols[1] = &c[idr].V[1] - colstart;
      rcols[0] = lcols[1]; rcols[1] = lcols[0];
      lvals[0] = ctx->VDiffusion[1]*sx; lvals[1] = -ctx->VDiffusion[1]*sx;
      rvals[0] = lvals[1]; rvals[1] = lvals[0];
      PetscPrintf(PETSC_COMM_WORLD,"\t [2] row [%D %D]: Left Cols = [%D, %D]  Right Cols = [%D, %D]\n", left,right,lcols[0],lcols[1],rcols[0],rcols[1]);
      ierr = MatSetValuesLocal(J,1,&left,2,lcols,lvals,ADD_VALUES);CHKERRQ(ierr);
      //ierr = MatSetValuesLocal(*J,1,&right,2,rcols,rvals,ADD_VALUES);CHKERRQ(ierr);
#endif

#ifdef BARRY_IMPL
      row[0] = &f[xi].I[1] - rowstart;
      col[0] = &c[xi-1].I[1] - colstart;
      col[1] = &c[xi].I[1] - colstart;
      col[2] = &c[xi+1].I[1] - colstart;
      val[0] = ctx->IDiffusion[1]*sx;
      val[1] = -2.0*ctx->IDiffusion[1]*sx;
      val[2] = ctx->IDiffusion[1]*sx;
      PetscPrintf(PETSC_COMM_WORLD,"[3] Setting row %D: Col = [%D, %D, %D]\n", row[0],col[0],col[1],col[2]);
      ierr = MatSetValuesLocal(J,1,row,3,col,val,ADD_VALUES);CHKERRQ(ierr);
#else
      left = &f[idl].I[1] - rowstart; right = &f[idr].I[1] - rowstart;
      lcols[0] = &c[idl].I[1] - colstart; lcols[1] = &c[idr].I[1] - colstart;
      rcols[0] = lcols[1]; rcols[1] = lcols[0];
      lvals[0] = ctx->IDiffusion[1]*sx; lvals[1] = -ctx->IDiffusion[1]*sx;
      rvals[0] = lvals[1]; rvals[1] = lvals[0];
      PetscPrintf(PETSC_COMM_WORLD,"\t [3] row [%D %D]: Left Cols = [%D, %D]  Right Cols = [%D, %D]\n", left,right,lcols[0],lcols[1],rcols[0],rcols[1]);
      ierr = MatSetValuesLocal(J,1,&left,2,lcols,lvals,ADD_VALUES);CHKERRQ(ierr);
      //ierr = MatSetValuesLocal(J,1,&right,2,rcols,rvals,ADD_VALUES);CHKERRQ(ierr);
#endif

      /* Mixed He - V clusters are immobile  */
      
      /* -------------------------------------------------------------------------
       ---- Compute dissociation terms that removes an item from a cluster
       I assume dissociation means losing only a single item from a cluster
       I cannot tell from the notes if clusters can break up into any sub-size.
       */
      
      /*   He[He] ->  He[He-1] + He[1] */
      for (He=2; He<NHe+1; He++) {
        row[0] = &f[xi].He[He-1] - rowstart;
        row[1] = &f[xi].He[1] - rowstart;
        row[2] = &f[xi].He[He] - rowstart;
        col[0] = &c[xi].He[He] - colstart;
        val[0] = ctx->dissociationScale;
        val[1] = ctx->dissociationScale;
        val[2] = -ctx->dissociationScale;
        ierr = MatSetValuesLocal(J,3,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
      }
      
      /*   V[V] ->  V[V-1] + V[1] */
      for (V=2; V<NV+1; V++) {
        row[0] = &f[xi].V[V-1] - rowstart;
        row[1] = &f[xi].V[1] - rowstart;
        row[2] = &f[xi].V[V] - rowstart;
        col[0] = &c[xi].V[V] - colstart;
        val[0] = ctx->dissociationScale;
        val[1] = ctx->dissociationScale;
        val[2] = -ctx->dissociationScale;
        ierr = MatSetValuesLocal(J,3,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
      }
      
      /*   I[I] ->  I[I-1] + I[1] */
      for (I=2; I<NI+1; I++) {
        row[0] = &f[xi].I[I-1] - rowstart;
        row[1] = &f[xi].I[1] - rowstart;
        row[2] = &f[xi].I[I] - rowstart;
        col[0] = &c[xi].I[I] - colstart;
        val[0] = ctx->dissociationScale;
        val[1] = ctx->dissociationScale;
        val[2] = -ctx->dissociationScale;
        ierr = MatSetValuesLocal(J,3,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
      }
      
      /*   He[He]-V[1] ->  He[He] + V[1]  */
      for (He=1; He<NHeV[1]+1; He++) {
        row[0] = &f[xi].He[He] - rowstart;
        row[1] = &f[xi].V[1] - rowstart;
        row[2] = &fHeV[1][He] - rowstart;
        col[0] = &cHeV[1][He] - colstart;
        val[0] = 1000*ctx->dissociationScale;
        val[1] = 1000*ctx->dissociationScale;
        val[2] = -1000*ctx->dissociationScale;
        ierr = MatSetValuesLocal(J,3,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
      }
      

      /*   He[1]-V[V] ->  He[1] + V[V]  */
      for (V=2; V<MHeV+1; V++) {
        row[0] = &f[xi].He[1] - rowstart;
        row[1] = &f[xi].V[V] - rowstart;
        row[2] = &fHeV[V][1] - rowstart;
        col[0] = &cHeV[V][1] - colstart;
        val[0] = 1000*ctx->dissociationScale;
        val[1] = 1000*ctx->dissociationScale;
        val[2] = -1000*ctx->dissociationScale;
        ierr = MatSetValuesLocal(J,3,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
      }
      
      /*   He[He]-V[V] ->  He[He-1]-V[V] + He[1]  */
      for (V=2; V<MHeV+1; V++) {
        for (He=2; He<NHeV[V]+1; He++) {
          row[0] = &f[xi].He[1] - rowstart;
          row[1] = &fHeV[V][He-1] - rowstart;
          row[2] = &fHeV[V][He] - rowstart;
          col[0] = &cHeV[V][He] - colstart;
          val[0] = 1000*ctx->dissociationScale;
          val[1] = 1000*ctx->dissociationScale;
          val[2] = -1000*ctx->dissociationScale;
          ierr = MatSetValuesLocal(J,3,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
      
      /*   He[He]-V[V] ->  He[He]-V[V-1] + V[1]  */
      for (V=2; V<MHeV+1; V++) {
        for (He=2; He<NHeV[V-1]+1; He++) {
          row[0] = &f[xi].V[1] - rowstart;
          row[1] = &fHeV[V-1][He] - rowstart;
          row[2] = &fHeV[V][He] - rowstart;
          col[0] = &cHeV[V][He] - colstart;
          val[0] = 1000*ctx->dissociationScale;
          val[1] = 1000*ctx->dissociationScale;
          val[2] = -1000*ctx->dissociationScale;
          ierr = MatSetValuesLocal(J,3,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
      
      /*   He[He]-V[V] ->  He[He]-V[V+1] + I[1]  */
      for (V=1; V<MHeV; V++) {
        for (He=1; He<NHeV[V]+1; He++) {
          row[0] = &fHeV[V+1][He] - rowstart;
          row[1] = &f[xi].I[1] - rowstart;
          row[2] = &fHeV[V][He] - rowstart;
          col[0] = &cHeV[V][He] - colstart;
          val[0] = 1000*ctx->dissociationScale;
          val[1] = 1000*ctx->dissociationScale;
          val[2] = -1000*ctx->dissociationScale;
          ierr = MatSetValuesLocal(J,3,row,1,col,val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatStoreValues(J);CHKERRQ(ierr);
    MatSetFromOptions(J);
    initialized = PETSC_TRUE;
  } else {
    ierr = MatRetrieveValues(J);CHKERRQ(ierr);
  }


  /*
     Loop over grid points computing Jacobian terms for each grid point for reaction terms
  */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information in canonical order
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &vpere, &connect);CHKERRQ(ierr);
    if (vpere != 2) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only EDGE2 element bases are supported in the current example. n(Connectivity)=%D.\n", vpere);

//      ierr = DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &xi);CHKERRQ(ierr);
    ierr = DMMoabGetDofsLocal(dm, vpere, connect, dof_indices);CHKERRQ(ierr);

    ierr = cHeVInitialize(&c[xi].He[1],(PetscScalar**)cHeV);CHKERRQ(ierr);
    ierr = cHeVInitialize(&f[xi].He[1],(PetscScalar**)fHeV);CHKERRQ(ierr);
    /* ----------------------------------------------------------------
     ---- Compute reaction terms that can create a cluster of given size
    */
    /*   He[He] + He[he] -> He[He+he]  */
    for (He=2; He<NHe+1; He++) {
      /* compute all pairs of clusters of smaller size that can combine to create a cluster of size He,
         remove the upper half since they are symmetric to the lower half of the pairs. For example
              when He = 5 (cluster size 5) the pairs are
                 1   4
                 2   2
                 3   2  these last two are not needed in the sum since they repeat from above
                 4   1  this is why he < (He/2) + 1            */
      for (he=1; he<(He/2)+1; he++) {
        row[0] = &f[xi].He[He] - rowstart;
        row[1] = &f[xi].He[he] - rowstart;
        row[2] = &f[xi].He[He-he] - rowstart;
        col[0] = &c[xi].He[he] - colstart;
        col[1] = &c[xi].He[He-he] - colstart;
        val[0] = ctx->reactionScale*c[xi].He[He-he];
        val[1] = ctx->reactionScale*c[xi].He[he];
        val[2] = -ctx->reactionScale*c[xi].He[He-he];
        val[3] = -ctx->reactionScale*c[xi].He[he];
        val[4] = -ctx->reactionScale*c[xi].He[He-he];
        val[5] = -ctx->reactionScale*c[xi].He[he];
        ierr = MatSetValuesLocal(J,3,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
      }
    }

    /*   V[V]  +  V[v] ->  V[V+v]  */
    for (V=2; V<NV+1; V++) {
      for (v=1; v<(V/2)+1; v++) {
        row[0] = &f[xi].V[V] - rowstart;
        row[1] = &f[xi].V[v] - rowstart;
        row[2] = &f[xi].V[V-v] - rowstart;
        col[0] = &c[xi].V[v] - colstart;
        col[1] = &c[xi].V[V-v] - colstart;
        val[0] = ctx->reactionScale*c[xi].V[V-v];
        val[1] = ctx->reactionScale*c[xi].V[v];
        val[2] = -ctx->reactionScale*c[xi].V[V-v];
        val[3] = -ctx->reactionScale*c[xi].V[v];
        val[4] = -ctx->reactionScale*c[xi].V[V-v];
        val[5] = -ctx->reactionScale*c[xi].V[v];
        ierr = MatSetValuesLocal(J,3,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
      }
    }

    /*   I[I] +  I[i] -> I[I+i] */
    for (I=2; I<NI+1; I++) {
      for (i=1; i<(I/2)+1; i++) {
        row[0] = &f[xi].I[I] - rowstart;
        row[1] = &f[xi].I[i] - rowstart;
        row[2] = &f[xi].I[I-i] - rowstart;
        col[0] = &c[xi].I[i] - colstart;
        col[1] = &c[xi].I[I-i] - colstart;
        val[0] = ctx->reactionScale*c[xi].I[I-i];
        val[1] = ctx->reactionScale*c[xi].I[i];
        val[2] = -ctx->reactionScale*c[xi].I[I-i];
        val[3] = -ctx->reactionScale*c[xi].I[i];
        val[4] = -ctx->reactionScale*c[xi].I[I-i];
        val[5] = -ctx->reactionScale*c[xi].I[i];
        ierr = MatSetValuesLocal(J,3,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
      }
    }

    /* He[1] +  V[1]  ->  He[1]-V[1] */
    row[0] = &fHeV[1][1] - rowstart;
    row[1] = &f[xi].He[1] - rowstart;
    row[2] = &f[xi].V[1] - rowstart;
    col[0] = &c[xi].He[1] - colstart;
    col[1] = &c[xi].V[1] - colstart;
    val[0] = 1000*ctx->reactionScale*c[xi].V[1];
    val[1] = 1000*ctx->reactionScale*c[xi].He[1];
    val[2] = -1000*ctx->reactionScale*c[xi].V[1];
    val[3] = -1000*ctx->reactionScale*c[xi].He[1];
    val[4] = -1000*ctx->reactionScale*c[xi].V[1];
    val[5] = -1000*ctx->reactionScale*c[xi].He[1];
    ierr = MatSetValuesLocal(J,3,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);

    /*  He[He]-V[V] + He[he] -> He[He+he]-V[V]  */
   for (V=1; V<MHeV+1; V++) {
      for (He=1; He<NHeV[V]; He++) {
         for (he=1; he+He<NHeV[V]+1; he++) {
          row[0] = &fHeV[V][He+he] - rowstart;
          row[1] = &f[xi].He[he] - rowstart;
          row[2] = &fHeV[V][He] - rowstart;
          col[0] = &c[xi].He[he] - colstart;
          col[1] = &cHeV[V][He] - colstart;
          val[0] = ctx->reactionScale*cHeV[V][He];
          val[1] = ctx->reactionScale*c[xi].He[he];
          val[2] = -ctx->reactionScale*cHeV[V][He];
          val[3] = -ctx->reactionScale*c[xi].He[he];
          val[4] = -ctx->reactionScale*cHeV[V][He];
          val[5] = -ctx->reactionScale*c[xi].He[he];
          ierr = MatSetValuesLocal(J,3,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }


    /*  He[He]-V[V] + V[1] -> He[He][V+1] */
    for (V=1; V<MHeV; V++) {
      for (He=1; He<NHeV[V+1]; He++) {
        row[0] = &fHeV[V+1][He] - rowstart;
        row[1] = &f[xi].V[1] - rowstart;
        row[2] = &fHeV[V][He] - rowstart;
        col[0] = &c[xi].V[1] - colstart;
        col[1] = &cHeV[V][He] - colstart;
        val[0] = ctx->reactionScale*cHeV[V][He];
        val[1] = ctx->reactionScale*c[xi].V[1];
        val[2] = -ctx->reactionScale*cHeV[V][He];
        val[3] = -ctx->reactionScale*c[xi].V[1];
        val[4] = -ctx->reactionScale*cHeV[V][He];
        val[5] = -ctx->reactionScale*c[xi].V[1];
        ierr = MatSetValuesLocal(J,3,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
     }
    }

    /*  He[He]-V[V]  + He[he]-V[v] -> He[He+he][V+v]  */
    /*  Currently the reaction rates for this are zero */


    /*  V[V] + I[I]  ->   V[V-I] if V > I else I[I-V] */
    for (V=1; V<NV+1; V++) {
      for (I=1; I<PetscMin(V,NI); I++) {
        row[0] = &f[xi].V[V-I] - rowstart;
        row[1] = &f[xi].V[V] - rowstart;
        row[2] = &f[xi].I[I] - rowstart;
        col[0] = &c[xi].V[V] - colstart;
        col[1] = &c[xi].I[I]  - colstart;
        val[0] = ctx->reactionScale*c[xi].I[I];
        val[1] = ctx->reactionScale*c[xi].V[V];
        val[2] = -ctx->reactionScale*c[xi].I[I];
        val[3] = -ctx->reactionScale*c[xi].V[V];
        val[4] = -ctx->reactionScale*c[xi].I[I];
        val[5] = -ctx->reactionScale*c[xi].V[V];
        ierr = MatSetValuesLocal(J,3,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
      }
      for (I=V+1; I<NI+1; I++) {
        row[0] = &f[xi].I[I-V] - rowstart;
        row[1] = &f[xi].V[V] - rowstart;
        row[2] = &f[xi].I[I] - rowstart;
        col[0] = &c[xi].V[V] - colstart;
        col[1] = &c[xi].I[I] - colstart;
        val[0] = ctx->reactionScale*c[xi].I[I];
        val[1] = ctx->reactionScale*c[xi].V[V];
        val[2] = -ctx->reactionScale*c[xi].I[I];
        val[3] = -ctx->reactionScale*c[xi].V[V];
        val[4] = -ctx->reactionScale*c[xi].I[I];
        val[5] = -ctx->reactionScale*c[xi].V[V];
        ierr = MatSetValuesLocal(J,3,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /*
     Restore vectors
  */
  c    = (Concentrations*)(((PetscScalar*)c)+1);
  ierr = DMMoabVecRestoreArrayRead(dm,localC,&c);CHKERRQ(ierr);
  f    = (Concentrations*)(((PetscScalar*)f)+1);
  ierr = DMMoabVecRestoreArray(dm,C,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&localC);CHKERRQ(ierr);
  ierr = cHeVDestroy((PetscScalar**)cHeV);CHKERRQ(ierr);
  ierr = cHeVDestroy((PetscScalar**)fHeV);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != J) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "GetDfill"
/*
    Determines the nonzero structure within the diagonal blocks of the Jacobian that represent coupling resulting from reactions and
    dissasociations of the clusters
*/
PetscErrorCode GetDfill(PetscInt *dfill, void *ptr)
{
  PetscInt       He,he,V,v,I,i,j,k,rows[3],cols[2];
  Concentrations *c;
  PetscScalar    *idxstart,**cHeV;
  PetscErrorCode ierr;

  /* ensure fill for the diagonal of matrix */
  for (i=0; i<(DOF); i++) {
    dfill[i*DOF + i] = 1;
  }

  /*
   c is never used except for computing offsets between variables which are used to fill the non-zero
   structure of the matrix
   */
  ierr     = PetscMalloc(sizeof(Concentrations),&c);CHKERRQ(ierr);
  c        = (Concentrations*)(((PetscScalar*)c)-1);
  ierr     = cHeVCreate(&cHeV);CHKERRQ(ierr);
  ierr     = cHeVInitialize(&c->He[1],cHeV);CHKERRQ(ierr);
  idxstart = (PetscScalar*)&c->He[1];

  /* -------------------------------------------------------------------------
   ---- Compute dissociation terms that removes an item from a cluster
   I assume dissociation means losing only a single item from a cluster
   I cannot tell from the notes if clusters can break up into any sub-size.
   */
  /*   He[He] ->  He[He-1] + He[1] */
  for (He=2; He<NHe+1; He++) {
    rows[0] = &c->He[He-1] - idxstart;
    rows[1] = &c->He[1] - idxstart;
    rows[2] = &c->He[He] - idxstart;
    cols[0] = &c->He[He] - idxstart;
    for (j=0; j<3; j++) {
      dfill[rows[j]*DOF + cols[0]] = 1;
    }
  }

  /*   V[V] ->  V[V-1] + V[1] */
  for (V=2; V<NV+1; V++) {
    rows[0] = &c->V[V] - idxstart;
    rows[1] = &c->V[1] - idxstart;
    rows[2] = &c->V[V-1] - idxstart;
    cols[0] = &c->V[V] - idxstart;
    for (j=0; j<3; j++) {
      dfill[rows[j]*DOF + cols[0]] = 1;
    }
  }
  
  /*   I[I] ->  I[I-1] + I[1] */
  for (I=2; I<NI+1; I++) {
    rows[0] = &c->I[I] - idxstart;
    rows[1] = &c->I[1] - idxstart;
    rows[2] = &c->I[I-1] - idxstart;
    cols[0] = &c->I[I] - idxstart;
    for (j=0; j<3; j++) {
      dfill[rows[j]*DOF + cols[0]] = 1;
    }
  }
  
  /*   He[He]-V[1] ->  He[He] + V[1]  */
  for (He=1; He<NHeV[1]+1; He++) {
    rows[0] = &c->He[He] - idxstart;
    rows[1] = &c->V[1] - idxstart;
    rows[2] = &cHeV[1][He] - idxstart;
    cols[0] = &cHeV[1][He] - idxstart;
    for (j=0; j<3; j++) {
      dfill[rows[j]*DOF + cols[0]] = 1;
    }
  }
  
  /*   He[1]-V[V] ->  He[1] + V[V]  */
  for (V=2; V<MHeV+1; V++) {
    rows[0] = &c->He[1] - idxstart;
    rows[1] = &c->V[V] - idxstart;
    rows[2] = &cHeV[V][1] - idxstart;
    cols[0] = &cHeV[V][1] - idxstart;
    for (j=0; j<3; j++) {
      dfill[rows[j]*DOF + cols[0]] = 1;
    }
  }
  
  /*   He[He]-V[V] ->  He[He-1]-V[V] + He[1]  */
  for (V=2; V<MHeV+1; V++) {
    for (He=2; He<NHeV[V]+1; He++) {
      rows[0] = &c->He[1] - idxstart;
      rows[1] = &cHeV[V][He] - idxstart;
      rows[2] = &cHeV[V][He-1] - idxstart;
      cols[0] = &cHeV[V][He] - idxstart;
      for (j=0; j<3; j++) {
        dfill[rows[j]*DOF + cols[0]] = 1;
      }
    }
  }
  
  /*   He[He]-V[V] ->  He[He]-V[V-1] + V[1]  */
  for (V=2; V<MHeV+1; V++) {
    for (He=2; He<NHeV[V-1]+1; He++) {
      rows[0] = &c->V[1] - idxstart;
      rows[1] = &cHeV[V][He] - idxstart;
      rows[2] = &cHeV[V-1][He] - idxstart;
      cols[0] = &cHeV[V][He] - idxstart;
      for (j=0; j<3; j++) {
        dfill[rows[j]*DOF + cols[0]] = 1;
      }
    }
  }
  
  /*   He[He]-V[V] ->  He[He]-V[V+1] + I[1]  */
  for (V=1; V<MHeV; V++) {
    for (He=1; He<NHeV[V]+1; He++) {
      rows[0] = &c->I[1] - idxstart;
      rows[1] = &cHeV[V+1][He] - idxstart;
      rows[2] = &cHeV[V][He] - idxstart;
      cols[0] = &cHeV[V][He] - idxstart;
      for (j=0; j<3; j++) {
        dfill[rows[j]*DOF + cols[0]] = 1;
      }
    }
  }

  /* These are the reaction terms in the diagonal block */
  for (He=2; He<NHe+1; He++) {
    for (he=1; he<(He/2)+1; he++) {
      rows[0] = &c->He[He] - idxstart;
      rows[1] = &c->He[he] - idxstart;
      rows[2] = &c->He[He-he] - idxstart;
      cols[0] = &c->He[he] - idxstart;
      cols[1] = &c->He[He-he] - idxstart;
      for (j=0; j<3; j++) {
        for (k=0; k<2; k++) {
          dfill[rows[j]*DOF + cols[k]] = 1;
        }
      }
    }
  }

  /*   V[V]  +  V[v] ->  V[V+v]  */
  for (V=2; V<NV+1; V++) {
    for (v=1; v<(V/2)+1; v++) {
      rows[0] = &c->V[V] - idxstart;
      rows[1] = &c->V[v] - idxstart;
      rows[2] = &c->V[V-v] - idxstart;
      cols[0] = &c->V[v] - idxstart;
      cols[1] = &c->V[V-v] - idxstart;
      for (j=0; j<3; j++) {
        for (k=0; k<2; k++) {
          dfill[rows[j]*DOF + cols[k]] = 1;
        }
      }
    }
  }
  
  /*   I[I] +  I[i] -> I[I+i] */
  for (I=2; I<NI+1; I++) {
    for (i=1; i<(I/2)+1; i++) {
      rows[0] = &c->I[I] - idxstart;
      rows[1] = &c->I[i] - idxstart;
      rows[2] = &c->I[I-i] - idxstart;
      cols[0] = &c->I[i] - idxstart;
      cols[1] = &c->I[I-i] - idxstart;
      for (j=0; j<3; j++) {
        for (k=0; k<2; k++) {
          dfill[rows[j]*DOF + cols[k]] = 1;
        }
      }
    }
  }
  
  /* He[1] +  V[1]  ->  He[1]-V[1] */
  rows[0] = &cHeV[1][1] - idxstart;
  rows[1] = &c->He[1] - idxstart;
  rows[2] = &c->V[1] - idxstart;
  cols[0] = &c->He[1] - idxstart;
  cols[1] = &c->V[1] - idxstart;
  for (j=0; j<3; j++) {
    for (k=0; k<2; k++) {
      dfill[rows[j]*DOF + cols[k]] = 1;
    }
  }
  
  /*  He[He]-V[V] + He[he] -> He[He+he]-V[V]  */
  for (V=1; V<MHeV+1; V++) {
    for (He=1; He<NHeV[V]; He++) {
      for (he=1; he+He<NHeV[V]+1; he++) {
        rows[0] = &cHeV[V][He+he] - idxstart;
        rows[1] = &c->He[he] - idxstart;
        rows[2] = &cHeV[V][He] - idxstart;
        cols[0] = &cHeV[V][He] - idxstart;
        cols[1] = &c->He[he] - idxstart;
        for (j=0; j<3; j++) {
          for (k=0; k<2; k++) {
            dfill[rows[j]*DOF + cols[k]] = 1;
          }
        }
      }
    }
  }
  /*  He[He]-V[V] + V[1] -> He[He][V+1] */
  for (V=1; V<MHeV; V++) {
    for (He=1; He<NHeV[V+1]; He++) {
      rows[0] = &cHeV[V+1][He] - idxstart;
      rows[1] = &c->V[1] - idxstart;
      rows[2] = &cHeV[V][He] - idxstart;
      cols[0] = &cHeV[V][He] - idxstart;
      cols[1] = &c->V[1] - idxstart;
      for (j=0; j<3; j++) {
        for (k=0; k<2; k++) {
          dfill[rows[j]*DOF + cols[k]] = 1;
        }
      }
    }
  }

  /*  He[He]-V[V]  + He[he]-V[v] -> He[He+he][V+v]  */
  /*  Currently the reaction rates for this are zero */
  
  /*  V[V] + I[I]  ->   V[V-I] if V > I else I[I-V] */
  for (V=1; V<NV+1; V++) {
    for (I=1; I<PetscMin(V,NI); I++) {
      rows[0] = &c->V[V-I] - idxstart;
      rows[1] = &c->V[V] - idxstart;
      rows[2] = &c->I[I] - idxstart;
      cols[0] = &c->V[V] - idxstart;
      cols[1] = &c->I[I] - idxstart;
      for (j=0; j<3; j++) {
        for (k=0; k<2; k++) {
          dfill[rows[j]*DOF + cols[k]] = 1;
        }
      }
    }
    for (I=V+1; I<NI+1; I++) {
      rows[0] = &c->I[I-V] - idxstart;
      rows[1] = &c->V[V] - idxstart;
      rows[2] = &c->I[I] - idxstart;
      cols[0] = &c->V[V] - idxstart;
      cols[1] = &c->I[I] - idxstart;
      for (j=0; j<3; j++) {
        for (k=0; k<2; k++) {
          dfill[rows[j]*DOF + cols[k]] = 1;
        }
      }
    }
  }

  c    = (Concentrations*)(((PetscScalar*)c)+1);
  ierr = cHeVDestroy(cHeV);CHKERRQ(ierr);
  ierr = PetscFree(c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */

typedef struct {
  DM          Heda,Vda,HeVda;       /* defines the 2d layout of the He subvector */
  Vec         He,V,HeV;
  VecScatter  Hescatter,Vscatter,HeVscatter;
  PetscViewer Heviewer,Vviewer,HeVviewer;
} MyMonitorCtx;

#undef __FUNCT__
#define __FUNCT__ "MyMonitorMonitor"
/*
   Display He as a function of space and cluster size for each time step
*/
PetscErrorCode MyMonitorMonitor(TS ts,PetscInt timestep,PetscReal time,Vec solution, void *ictx)
{
  MyMonitorCtx   *ctx = (MyMonitorCtx*)ictx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecScatterBegin(ctx->Hescatter,solution,ctx->He,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->Hescatter,solution,ctx->He,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecView(ctx->He,ctx->Heviewer);CHKERRQ(ierr);

  ierr = VecScatterBegin(ctx->Vscatter,solution,ctx->V,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->Vscatter,solution,ctx->V,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecView(ctx->V,ctx->Vviewer);CHKERRQ(ierr);

  ierr = VecScatterBegin(ctx->HeVscatter,solution,ctx->HeV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->HeVscatter,solution,ctx->HeV,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecView(ctx->HeV,ctx->HeVviewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyMonitorDestroy"
/*
   Frees all data structures associated with the monitor
*/
PetscErrorCode MyMonitorDestroy(void **ictx)
{
  MyMonitorCtx   **ctx = (MyMonitorCtx**)ictx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecScatterDestroy(&(*ctx)->Hescatter);CHKERRQ(ierr);
  ierr = VecDestroy(&(*ctx)->He);CHKERRQ(ierr);
  ierr = DMDestroy(&(*ctx)->Heda);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&(*ctx)->Heviewer);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&(*ctx)->Vscatter);CHKERRQ(ierr);
  ierr = VecDestroy(&(*ctx)->V);CHKERRQ(ierr);
  ierr = DMDestroy(&(*ctx)->Vda);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&(*ctx)->Vviewer);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&(*ctx)->HeVscatter);CHKERRQ(ierr);
  ierr = VecDestroy(&(*ctx)->HeV);CHKERRQ(ierr);
  ierr = DMDestroy(&(*ctx)->HeVda);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&(*ctx)->HeVviewer);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 1
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "MyMonitorSetUp"
/*
   Sets up a monitor that will display He as a function of space and cluster size for each time step
*/
PetscErrorCode MyMonitorSetUp(TS ts)
{
  DM             dm,da;
  PetscErrorCode ierr;
  PetscInt       xi,xs,xm,*idx,M,xj,cnt = 0;
  const PetscInt *lx;
  Vec            C;
  MyMonitorCtx   *ctx;
  PetscBool      flg;
  IS             is;
  char           ycoor[32];
  PetscReal      valuebounds[4] = {0, 1.2, 0, 1.2};

  PetscFunctionBeginUser;
  ierr = PetscOptionsHasName(NULL,"-mymonitor",&flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = PetscNew(&ctx);CHKERRQ(ierr);

  /* create the replica 1-D DMDA object */
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC,-8,DOF,1,NULL,&da);CHKERRQ(ierr);

  /* setup visualization for He */
  ierr = PetscViewerDrawOpen(PetscObjectComm((PetscObject)da),NULL,"",PETSC_DECIDE,PETSC_DECIDE,600,400,&ctx->Heviewer);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&M,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(da,&lx,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDACreate2d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,NHe,PETSC_DETERMINE,1,1,1,lx,NULL,&ctx->Heda);CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx->Heda,0,"He");CHKERRQ(ierr);
  ierr = DMDASetCoordinateName(ctx->Heda,0,"X coordinate direction");CHKERRQ(ierr);
  ierr = PetscSNPrintf(ycoor,32,"%D ... Cluster size ... 1",NHe);CHKERRQ(ierr);
  ierr = DMDASetCoordinateName(ctx->Heda,1,ycoor);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx->Heda,&ctx->He);CHKERRQ(ierr);
  ierr = PetscMalloc(NHe*xm*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  cnt  = 0;
  for (xj=0; xj<NHe; xj++) {
    for (xi=xs; xi<xs+xm; xi++) {
      idx[cnt++] = DOF*xi + xj;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ts),NHe*xm,idx,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&C);CHKERRQ(ierr);
  ierr = VecScatterCreate(C,is,ctx->He,NULL,&ctx->Hescatter);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  /* sets the bounds on the contour plot values so the colors mean the same thing for different timesteps */
  ierr = PetscViewerDrawSetBounds(ctx->Heviewer,2,valuebounds);CHKERRQ(ierr);

  /* setup visualization for V */
  ierr = PetscViewerDrawOpen(PetscObjectComm((PetscObject)da),NULL,"",PETSC_DECIDE,PETSC_DECIDE,600,400,&ctx->Vviewer);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&M,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(da,&lx,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDACreate2d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,NV,PETSC_DETERMINE,1,1,1,lx,NULL,&ctx->Vda);CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx->Vda,0,"V");CHKERRQ(ierr);
  ierr = DMDASetCoordinateName(ctx->Vda,0,"X coordinate direction");CHKERRQ(ierr);
  ierr = PetscSNPrintf(ycoor,32,"%D ... Cluster size ... 1",NV);CHKERRQ(ierr);
  ierr = DMDASetCoordinateName(ctx->Vda,1,ycoor);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx->Vda,&ctx->V);CHKERRQ(ierr);
  ierr = PetscMalloc(NV*xm*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  cnt  = 0;
  for (xj=0; xj<NV; xj++) {
    for (xi=xs; xi<xs+xm; xi++) {
      idx[cnt++] = NHe + DOF*xi + xj;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ts),NV*xm,idx,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&C);CHKERRQ(ierr);
  ierr = VecScatterCreate(C,is,ctx->V,NULL,&ctx->Vscatter);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  /* sets the bounds on the contour plot values so the colors mean the same thing for different timesteps */
  ierr = PetscViewerDrawSetBounds(ctx->Vviewer,2,valuebounds);CHKERRQ(ierr);

  /* setup visualization for HeV[1][] */
  ierr = PetscViewerDrawOpen(PetscObjectComm((PetscObject)da),NULL,"",PETSC_DECIDE,PETSC_DECIDE,600,400,&ctx->HeVviewer);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&M,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(da,&lx,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDACreate2d(PetscObjectComm((PetscObject)da),DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,NHeV[1],PETSC_DETERMINE,1,1,1,lx,NULL,&ctx->HeVda);CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx->HeVda,0,"HeV[1][]");CHKERRQ(ierr);
  ierr = DMDASetCoordinateName(ctx->HeVda,0,"X coordinate direction");CHKERRQ(ierr);
  ierr = PetscSNPrintf(ycoor,32,"%D ... Cluster size ... 1",NHeV[1]);CHKERRQ(ierr);
  ierr = DMDASetCoordinateName(ctx->HeVda,1,ycoor);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ctx->HeVda,&ctx->HeV);CHKERRQ(ierr);
  ierr = PetscMalloc(NHeV[1]*xm*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  cnt  = 0;
  for (xj=0; xj<NHeV[1]; xj++) {
    for (xi=xs; xi<xs+xm; xi++) {
      idx[cnt++] = NHe + NV + NI + DOF*xi + xj;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ts),NHeV[1]*xm,idx,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&C);CHKERRQ(ierr);
  ierr = VecScatterCreate(C,is,ctx->HeV,NULL,&ctx->HeVscatter);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  /* sets the bounds on the contour plot values so the colors mean the same thing for different timesteps */
  ierr = PetscViewerDrawSetBounds(ctx->HeVviewer,2,valuebounds);CHKERRQ(ierr);

  ierr = TSMonitorSet(ts,MyMonitorMonitor,ctx,MyMonitorDestroy);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "MyLoadData"
PetscErrorCode MyLoadData(MPI_Comm comm,const char *filename)
{
  PetscErrorCode ierr;
  FILE           *fp;
  char           buff[256];
  PetscInt       He,V,I,lc = 0;
  char           Hebindstr[32],Vbindstr[32],Ibindstr[32],trapbindstr[32],*sharp;
  PetscReal      Hebind,Vbind,Ibind,trapbind;

  PetscFunctionBegin;
  ierr = PetscFOpen(comm,filename,"r",&fp);CHKERRQ(ierr);
  ierr = PetscMemzero(buff,sizeof(char)*256);CHKERRQ(ierr);
  ierr = PetscSynchronizedFGets(comm,fp,256,buff);CHKERRQ(ierr);
  while (buff[0]) {
    ierr = PetscStrchr(buff,'#',&sharp);CHKERRQ(ierr);
    if (!sharp) {
      sscanf(buff,"%d %d %d %s %s %s %s",&He,&V,&I,Hebindstr,Vbindstr,Ibindstr,trapbindstr);
      Hebind = strtod(Hebindstr,NULL);
      Vbind = strtod(Vbindstr,NULL);
      Ibind = strtod(Ibindstr,NULL);
      trapbind = strtod(trapbindstr,NULL);
      if (V <= NV) {
        if (He > NHe && V == 0 && I == 0) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Recompile with correct NHe %d %d",He,NHe);
        if (He == 0  && V > NV && I == 0) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Recompile with correct V %d %d",V,NV);
        if (He == 0  && V == 0 && I > NI) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Recompile with correct NI %d %d",I,NI);
        if (lc++ > DOF) SETERRQ6(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Recompile with correct NHe %d NV %d NI %d MNHeV % lc=%D and DOF=%D",NHe,NV,NI,MNHeV,lc,DOF);
        if (He > 0 && V > 0) {  /* assumes the He are sorted in increasing order */
          NHeV[V] = He;
        }
      }
    }
    ierr = PetscMemzero(buff,sizeof(char)*256);CHKERRQ(ierr);
    ierr = PetscSynchronizedFGets(comm,fp,256,buff);CHKERRQ(ierr);
  }
  if (lc != DOF) SETERRQ6(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Recompile with correct NHe %d NV %d NI %d MNHeV %d Actual DOF %d User DOF %d",NHe,NV,NI,MNHeV,lc,DOF);
  PetscFunctionReturn(0);
}

