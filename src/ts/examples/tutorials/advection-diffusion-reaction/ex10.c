static char help[] = "Solves C_t =  -D*C_xx + F(C) + R(C) + D(C) from Brian Wirth's SciDAC project.\n";

/*
        C_t =  -D*C_xx + F(C) + R(C) + D(C) from Brian Wirth's SciDAC project.

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


*/

#include <petscdmda.h>
#include <petscts.h>

/*    Hard wire the number of cluster sizes for He, V, and I */
#define N 15

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
  PetscScalar He[N];
  PetscScalar V[N];
  union {
    PetscScalar I[N];
    PetscScalar HeV[N+1][N]; /* actual size is N by N, the N+1 is there only to "trick" the compiler to have indices start at 1.*/
  };
} Concentrations;

/*
     Holds problem specific options and data
*/
typedef struct {
  PetscBool   noreactions;           /* run without the reaction terms */
  PetscBool   nodissociations;       /* run without the dissociation terms */
  PetscScalar HeDiffusion[6];
  PetscScalar VDiffusion[2];
  PetscScalar IDiffusion[2];
  PetscScalar forcingScale;
  PetscScalar reactionScale;
  PetscScalar dissociationScale;
} AppCtx;

extern PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode InitialConditions(DM,Vec);
extern PetscErrorCode MyMonitorSetUp(TS);
extern PetscErrorCode GetDfill(PetscInt*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;                  /* nonlinear solver */
  Vec            C;                   /* solution */
  PetscErrorCode ierr;
  DM             da;                  /* manages the grid data */
  AppCtx         ctx;                 /* holds problem specific paramters */
  PetscInt       He,dof = 3*N+N*N,*ofill,*dfill;
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char*)0,help);

  PetscFunctionBeginUser;
  ctx.noreactions     = PETSC_FALSE;
  ctx.nodissociations = PETSC_FALSE;

  ierr = PetscOptionsHasName(NULL,"-noreactions",&ctx.noreactions);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,"-nodissociations",&ctx.nodissociations);CHKERRQ(ierr);

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
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_MIRROR,-8,dof,1,NULL,&da);CHKERRQ(ierr);

  /* The only spatial coupling in the Jacobian (diffusion) is for the first 5 He, the first V, and the first I.
     The ofill (thought of as a dof by dof 2d (row-oriented) array represents the nonzero coupling between degrees
     of freedom at one point with degrees of freedom on the adjacent point to the left or right. A 1 at i,j in the
     ofill array indicates that the degree of freedom i at a point is coupled to degree of freedom j at the
     adjacent point. In this case ofill has only a few diagonal entries since the only spatial coupling is regular diffusion. */
  ierr = PetscMalloc(dof*dof*sizeof(PetscInt),&ofill);CHKERRQ(ierr);
  ierr = PetscMalloc(dof*dof*sizeof(PetscInt),&dfill);CHKERRQ(ierr);
  ierr = PetscMemzero(ofill,dof*dof*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(dfill,dof*dof*sizeof(PetscInt));CHKERRQ(ierr);

  for (He=0; He<PetscMin(N,5); He++) ofill[He*dof + He] = 1;
  ofill[N*dof + N] = ofill[2*N*dof + 2*N] = 1;

  ierr = DMDASetBlockFills(da,NULL,ofill);CHKERRQ(ierr);
  ierr = PetscFree(ofill);CHKERRQ(ierr);
  ierr = GetDfill(dfill,&ctx);CHKERRQ(ierr);
  ierr = PetscFree(dfill);CHKERRQ(ierr);

 

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Extract global vector from DMDA to hold solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&C);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&ctx);CHKERRQ(ierr);
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
  ierr = InitialConditions(da,C);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,C);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&C);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "InitialConditions"
PetscErrorCode InitialConditions(DM da,Vec C)
{
  PetscErrorCode ierr;
  PetscInt       i,I,He,V,xs,xm,Mx,cnt = 0;
  Concentrations *c;
  PetscReal      hx,x;
  char           string[16];

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(Mx-1);

  /* Name each of the concentrations */
  for (He=1; He<N+1; He++) {
    ierr = PetscSNPrintf(string,16,"%d-He",He);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
  }
  for (V=1; V<N+1; V++) {
    ierr = PetscSNPrintf(string,16,"%d-V",V);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
  }
  for (I=1; I<N+1; I++) {
    ierr = PetscSNPrintf(string,16,"%d-I",I);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
  }
  for (He=1; He<N+1; He++) {
    for (V=1; V<N+1; V++) {
      ierr = PetscSNPrintf(string,16,"%d-He-%d-V",He,V);CHKERRQ(ierr);
      ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
    }
  }

  /*
     Get pointer to vector data
  */
  ierr = DMDAVecGetArray(da,C,&c);CHKERRQ(ierr);
  /* Shift the c pointer to allow accessing with index of 1, instead of 0 */
  c = (Concentrations*)(((PetscScalar*)c)-1);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;
    for (He=1; He<N+1; He++) c[i].He[He] = 0.0;
    for (V=1; V<N+1; V++)    c[i].V[V]   = 1.0;
    for (I=1; I<N+1; I++)    c[i].I[I]   = 1.0;
    for (He=1; He<N+1; He++) {
      for (V=1; V<N+1; V++)  c[i].HeV[He][V] = 0.0;
    }
  }

  /*
     Restore vectors
  */
  c    = (Concentrations*)(((PetscScalar*)c)+1);
  ierr = DMDAVecRestoreArray(da,C,&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "IFunction"
/*
   IFunction - Evaluates nonlinear function that defines the ODE

   Input Parameters:
.  ts - the TS context
.  U - input vector
.  ptr - optional user-defined context

   Output Parameter:
.  F - function values
 */
PetscErrorCode IFunction(TS ts,PetscReal ftime,Vec C,Vec Cdot,Vec F,void *ptr)
{
  AppCtx         *ctx = (AppCtx*) ptr;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       xi,Mx,xs,xm,He,he,V,v,I,i;
  PetscReal      hx,sx,x;
  Concentrations *c,*f;
  Vec            localC;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localC);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx   = 8.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);

  /*
       F  = Cdot +  all the diffusion and reaction terms added below
  */
  ierr = VecCopy(Cdot,F);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,C,INSERT_VALUES,localC);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,C,INSERT_VALUES,localC);CHKERRQ(ierr);

  /*
    Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,localC,&c);CHKERRQ(ierr);
  /* Shift the c pointer to allow accessing with index of 1, instead of 0 */
  c    = (Concentrations*)(((PetscScalar*)c)-1);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  f    = (Concentrations*)(((PetscScalar*)f)-1);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Loop over grid points computing ODE terms for each grid point
  */
  for (xi=xs; xi<xs+xm; xi++) {
    x = xi*hx;

    /* -------------------------------------------------------------
     ---- Compute diffusion over the locally owned part of the grid
    */
    /* He clusters larger than 5 do not diffuse -- are immobile */
    for (He=1; He<PetscMin(N+1,6); He++) {
      f[xi].He[He] -=  ctx->HeDiffusion[He]*(-2.0*c[xi].He[He] + c[xi-1].He[He] + c[xi+1].He[He])*sx;
    }

    /* V and I clusters ONLY of size 1 diffuse */
    f[xi].V[1] -=  ctx->VDiffusion[1]*(-2.0*c[xi].V[1] + c[xi-1].V[1] + c[xi+1].V[1])*sx;
    f[xi].I[1] -=  ctx->IDiffusion[1]*(-2.0*c[xi].I[1] + c[xi-1].I[1] + c[xi+1].I[1])*sx;

    /* Mixed He - V clusters are immobile  */

    /* ----------------------------------------------------------------
     ---- Compute forcing that produces He of cluster size 1
          Crude cubic approximation of graph from Tibo's notes
    */
    f[xi].He[1] -=  ctx->forcingScale*PetscMax(0.0,0.0006*x*x*x  - 0.0087*x*x + 0.0300*x);
    /* Are V or I produced? */

    if (ctx->noreactions) continue;
    /* ----------------------------------------------------------------
     ---- Compute reaction terms that can create a cluster of given size
    */
    /*   He[He] + He[he] -> He[He+he]  */
    for (He=2; He<N+1; He++) {
      /* compute all pairs of clusters of smaller size that can combine to create a cluster of size He,
         remove the upper half since they are symmetric to the lower half of the pairs. For example
              when He = 5 (cluster size 5) the pairs are
                 1   4
                 2   2
                 3   2  these last two are not needed in the sum since they repeat from above
                 4   1  this is why he < (He/2) + 1            */
      for (he=1; he<(He/2)+1; he++) {
        f[xi].He[He] -= ctx->reactionScale*c[xi].He[he]*c[xi].He[He-he];

        /* remove the two clusters that merged to form the larger cluster */
        f[xi].He[he]    += ctx->reactionScale*c[xi].He[he]*c[xi].He[He-he];
        f[xi].He[He-he] += ctx->reactionScale*c[xi].He[he]*c[xi].He[He-he];
      }
    }
    /*   V[V]  +  V[v] ->  V[V+v]  */
    for (V=2; V<N+1; V++) {
      for (v=1; v<(V/2)+1; v++) {
        f[xi].V[V] -= ctx->reactionScale*c[xi].V[v]*c[xi].V[V-v];
        /* remove the clusters that merged to form the larger cluster */
        f[xi].V[v]   += ctx->reactionScale*c[xi].V[v]*c[xi].V[V-v];
        f[xi].V[V-v] += ctx->reactionScale*c[xi].V[v]*c[xi].V[V-v];
      }
    }
    /*   I[I] +  I[i] -> I[I+i] */
    for (I=2; I<N+1; I++) {
      for (i=1; i<(I/2)+1; i++) {
        f[xi].I[I] -= ctx->reactionScale*c[xi].I[i]*c[xi].I[I-i];
        /* remove the clusters that merged to form the larger cluster */
        f[xi].I[i]   += ctx->reactionScale*c[xi].I[i]*c[xi].I[I-i];
        f[xi].I[I-i] += ctx->reactionScale*c[xi].I[i]*c[xi].I[I-i];
      }
    }
    /* He[1] +  V[1]  ->  He[1]-V[1] */
    f[xi].HeV[1][1] -= 1000*ctx->reactionScale*c[xi].He[1]*c[xi].V[1];
    /* remove the He and V  that merged to form the He-V cluster */
    f[xi].He[1] += 1000*ctx->reactionScale*c[xi].He[1]*c[xi].V[1];
    f[xi].V[1]  += 1000*ctx->reactionScale*c[xi].He[1]*c[xi].V[1];
    /*  He[He]-V[V] + He[he] -> He[He+he]-V[V]  */
    for (He=1; He<N; He++) {
      for (V=1; V<N+1; V++) {
        for (he=1; he<N-He+1; he++) {
          f[xi].HeV[He+he][V] -= ctx->reactionScale*c[xi].HeV[He][V]*c[xi].He[he];
          /* remove the two clusters that merged to form the larger cluster */
          f[xi].He[he]     += ctx->reactionScale*c[xi].HeV[He][V]*c[xi].He[he];
          f[xi].HeV[He][V] += ctx->reactionScale*c[xi].HeV[He][V]*c[xi].He[he];
        }
      }
    }
    /*  He[He]-V[V] + V[v] -> He[He][V+v] */
    for (He=1; He<N+1; He++) {
      for (V=1; V<N; V++) {
        for (v=1; v<N-V+1; v++) {
          f[xi].HeV[He][V+v] -= ctx->reactionScale*c[xi].HeV[He][V]*c[xi].V[v];
          /* remove the two clusters that merged to form the larger cluster */
          f[xi].V[v]       += ctx->reactionScale*c[xi].HeV[He][V]*c[xi].V[v];
          f[xi].HeV[He][V] += ctx->reactionScale*c[xi].HeV[He][V]*c[xi].V[v];
        }
      }
    }
    /*  He[He]-V[V]  + He[he]-V[v] -> He[He+he][V+v]  */
    /*  Currently the reaction rates for this are zero */
    for (He=1; He<N; He++) {
      for (V=1; V<N; V++) {
        for (he=1; he<N-He+1; he++) {
          for (v=1; v<N-V+1; v++) {
            f[xi].HeV[He+he][V+v] -= 0.0*c[xi].HeV[He][V]*c[xi].HeV[he][v];
            /* remove the two clusters that merged to form the larger cluster */
            f[xi].HeV[he][V] += 0.0*c[xi].HeV[He][V]*c[xi].HeV[he][v];
            f[xi].HeV[He][V] += 0.0*c[xi].HeV[He][V]*c[xi].HeV[he][v];
          }
        }
      }
    }
    /*  V[V] + I[I]  ->   V[V-I] if V > I else I[I-V] */
    /*  What should the correct reaction rate should be? */
    for (V=1; V<N+1; V++) {
      for (I=1; I<V; I++) {
        f[xi].V[V-I] -= ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
        f[xi].V[V] += ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
        f[xi].I[I] += ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
      }
      for (I=V+1; I<N+1; I++) {
          f[xi].I[I-V] -= ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
          f[xi].V[V] += ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
          f[xi].I[I] += ctx->reactionScale*c[xi].V[V]*c[xi].I[I];
      }
    }



    if (ctx->nodissociations) continue;
    /* -------------------------------------------------------------------------
     ---- Compute dissociation terms that removes an item from a cluster
          I assume dissociation means losing only a single item from a cluster
          I cannot tell from the notes if clusters can break up into any sub-size.
    */
    /*   He[He] ->  He[He-1] + He[1] */
    for (He=2; He<N+1; He++) {
      f[xi].He[He-1] -= ctx->dissociationScale*c[xi].He[He];
      f[xi].He[1]    -= ctx->dissociationScale*c[xi].He[He];
      f[xi].He[He]   += ctx->dissociationScale*c[xi].He[He];
    }
    /*   V[V] ->  V[V-1] + V[1] */
    for (V=2; V<N+1; V++) {
      f[xi].V[V-1] -= ctx->dissociationScale*c[xi].V[V];
      f[xi].V[1]   -= ctx->dissociationScale*c[xi].V[V];
      f[xi].V[V]   += ctx->dissociationScale*c[xi].V[V];
    }
    /*   I[I] ->  I[I-1] + I[1] */
    for (I=2; I<N+1; I++) {
      f[xi].I[I-1] -= ctx->dissociationScale*c[xi].I[I];
      f[xi].I[1]   -= ctx->dissociationScale*c[xi].I[I];
      f[xi].I[I]   += ctx->dissociationScale*c[xi].I[I];
    }
    /* He[1]-V[1]  ->  He[1] + V[1] */
    f[xi].He[1]     -= 1000*ctx->reactionScale*c[xi].HeV[1][1];
    f[xi].V[1]      -= 1000*ctx->reactionScale*c[xi].HeV[1][1];
    f[xi].HeV[1][1] += 1000*ctx->reactionScale*c[xi].HeV[1][1];
    /*   He[He]-V[1] ->  He[He] + V[1]  */
    for (He=2; He<N+1; He++) {
      f[xi].He[He]     -= 1000*ctx->reactionScale*c[xi].HeV[He][1];
      f[xi].V[1]       -= 1000*ctx->reactionScale*c[xi].HeV[He][1];
      f[xi].HeV[He][1] += 1000*ctx->reactionScale*c[xi].HeV[He][1];
    }
    /*   He[1]-V[V] ->  He[1] + V[V]  */
    for (V=2; V<N+1; V++) {
      f[xi].He[1]     -= 1000*ctx->reactionScale*c[xi].HeV[1][V];
      f[xi].V[V]      -= 1000*ctx->reactionScale*c[xi].HeV[1][V];
      f[xi].HeV[1][V] += 1000*ctx->reactionScale*c[xi].HeV[1][V];
    }
    /*   He[He]-V[V] ->  He[He-1]-V[V] + He[1]  */
    for (He=2; He<N+1; He++) {
      for (V=2; V<N+1; V++) {
        f[xi].He[1]        -= 1000*ctx->reactionScale*c[xi].HeV[He][V];
        f[xi].HeV[He-1][V] -= 1000*ctx->reactionScale*c[xi].HeV[He][V];
        f[xi].HeV[He][V]   += 1000*ctx->reactionScale*c[xi].HeV[He][V];
      }
    }
    /*   He[He]-V[V] ->  He[He]-V[V-1] + V[1]  */
    for (He=2; He<N+1; He++) {
      for (V=2; V<N+1; V++) {
        f[xi].V[1]         -= 1000*ctx->reactionScale*c[xi].HeV[He][V];
        f[xi].HeV[He][V-1] -= 1000*ctx->reactionScale*c[xi].HeV[He][V];
        f[xi].HeV[He][V]   += 1000*ctx->reactionScale*c[xi].HeV[He][V];
      }
    }
    /*   He[He]-V[V] ->  He[He]-V[V+1] + I[1]  */
    /* Again, what is the reasonable dissociation rate? */
    for (He=1; He<N+1; He++) {
      for (V=1; V<N; V++) {
        f[xi].HeV[He][V+1] -= 1000*ctx->reactionScale*c[xi].HeV[He][V];
        f[xi].I[1]         -= 1000*ctx->reactionScale*c[xi].HeV[He][V];
        f[xi].HeV[He][V]   += 1000*ctx->reactionScale*c[xi].HeV[He][V];
      }
    }

  }

  /*
     Restore vectors
  */
  c    = (Concentrations*)(((PetscScalar*)c)+1);
  ierr = DMDAVecRestoreArray(da,localC,&c);CHKERRQ(ierr);
  f    = (Concentrations*)(((PetscScalar*)f)+1);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetDfill"

PetscErrorCode GetDfill(PetscInt *dfill, void *ptr) 
{
  AppCtx         *ctx = (AppCtx*) ptr;
  PetscInt       He,he,V,v,I,i,dof = 3*N+N*N,reactants[3],row,col1,col2,j;

  if (!ctx->noreactions) {
    
    for (He=2; He<N+1; He++) {
      /* compute all pairs of clusters of smaller size that can combine to create a cluster of size He,
       remove the upper half since they are symmetric to the lower half of the pairs. For example
       when He = 5 (cluster size 5) the pairs are
       1   4
       2   2
       3   2  these last two are not needed in the sum since they repeat from above
       4   1  this is why he < (He/2) + 1            */
      for (he=1; he<(He/2)+1; he++) {
        reactants[0] = he, reactants[1] = He-he, reactants[2] = He;
        for (j=0; j<3; j++) {
          row = reactants[j], col1 = reactants[0], col2 = reactants[1];
          dfill[(row-1)*dof + col1 - 1] = 1, dfill[(row-1)*dof + col2 - 1] = 1;
        }
      }
    }
    /*   V[V]  +  V[v] ->  V[V+v]  */
    for (V=2; V<N+1; V++) {
      for (v=1; v<(V/2)+1; v++) {
        reactants[0] = N+v, reactants[1] = N+V-v, reactants[2] = N+V;
        for (j=0; j<3; j++) {
          row = reactants[j], col1 = reactants[0], col2 = reactants[1];
          dfill[(row-1)*dof + col1 - 1] = 1, dfill[(row-1)*dof + col2 - 1] = 1;
        }
      }
    }

    /*   I[I] +  I[i] -> I[I+i] */
    for (I=2; I<N+1; I++) {
      for (i=1; i<(I/2)+1; i++) {
        reactants[0] = 2*N+i, reactants[1] = 2*N+I-i, reactants[2] = 2*N+I;
        for (j=0; j<3; j++) {
          row = reactants[j], col1 = reactants[0], col2 = reactants[1];
          dfill[(row-1)*dof + col1 - 1] = 1, dfill[(row-1)*dof + col2 - 1] = 1;
        }
      }
    }
  
    /* He[1] +  V[1]  ->  He[1]-V[1] */
    reactants[0] = 1, reactants[1] = N+1, reactants[2] = 3*N+1;
    for (j=0; j<3; j++) {
      row = reactants[j], col1 = reactants[0], col2 = reactants[1];
      dfill[(row-1)*dof + col1 - 1] = 1, dfill[(row-1)*dof + col2 - 1] = 1;
    }

    /*  He[He]-V[V] + He[he] -> He[He+he]-V[V]  */
    for (He=1; He<N; He++) {
      for (V=1; V<N+1; V++) {
        for (he=1; he<N-He+1; he++) {
          reactants[0] = 3*N + (He-1)*N + V, reactants[1] = he, reactants[2] = 3*N+(He+he-1)*N+V;
          for (j=0; j<3; j++) {
            row = reactants[j], col1 = reactants[0], col2 = reactants[1];
            dfill[(row-1)*dof + col1 - 1] = 1, dfill[(row-1)*dof + col2 - 1] = 1;
          }
        }
      }
    }
    /*  He[He]-V[V] + V[v] -> He[He][V+v] */
    for (He=1; He<N+1; He++) {
      for (V=1; V<N; V++) {
        for (v=1; v<N-V+1; v++) {
          reactants[0] = 3*N+(He-1)*N+V, reactants[1] = N+v, reactants[2] = 3*N+(He-1)*N+V+v;
          for (j=0; j<3; j++) {
            row = reactants[j], col1 = reactants[0], col2 = reactants[1];
            dfill[(row-1)*dof + col1 - 1] = 1, dfill[(row-1)*dof + col2 - 1] = 1;
          }
        }
      }
    }

    /*  He[He]-V[V]  + He[he]-V[v] -> He[He+he][V+v]  */
    /*  Currently the reaction rates for this are zero */
    for (He=1; He<N; He++) {
      for (V=1; V<N; V++) {
        for (he=1; he<N-He+1; he++) {
          for (v=1; v<N-V+1; v++) {
            reactants[0] = 3*N+(He-1)*N+V, reactants[1] = 3*N+(he-1)*N+V, reactants[2] = 3*N + (He+he-1)*N + V+v;
            for (j=0; j<3; j++) {
              row = reactants[j], col1 = reactants[0], col2 = reactants[1];
              dfill[(row-1)*dof + col1 - 1] = 1, dfill[(row-1)*dof + col2 - 1] = 1;
            }
          }
        }
      }
    }
    /*  V[V] + I[I]  ->   V[V-I] if V > I else I[I-V] */
    /*  What should the correct reaction rate should be? */
    for (V=1; V<N+1; V++) {
      for (I=1; I<V; I++) {
        reactants[0] = N+V, reactants[1] = 2*N+I, reactants[2] = N+V-I;
        for (j=0; j<3; j++) {
          row = reactants[j], col1 = reactants[0], col2 = reactants[1];
          dfill[(row-1)*dof + col1 - 1] = 1, dfill[(row-1)*dof + col2 - 1] = 1;
        }
      }
      for (I=V+1; I<N+1; I++) {
        reactants[0] = N+V, reactants[1] = 2*N+I, reactants[2] = 2*N+I-V;
        for (j=0; j<3; j++) {
          row = reactants[j], col1 = reactants[0], col2 = reactants[1];
          dfill[(row-1)*dof + col1 - 1] = 1, dfill[(row-1)*dof + col2 - 1] = 1;
        }
      }
    }
  }
    /* -------------------------------------------------------------------------
     ---- Compute dissociation terms that removes an item from a cluster
          I assume dissociation means losing only a single item from a cluster
          I cannot tell from the notes if clusters can break up into any sub-size.
    */
  if (!ctx->nodissociations) {
    /*   He[He] ->  He[He-1] + He[1] */
    for (He=2; He<N+1; He++) {
      reactants[0] = He, reactants[1] = He-1, reactants[2] = 1;

      for (j=0; j<3; j++) {
        row = reactants[j], col1 = reactants[0];
        dfill[(row-1)*dof + col1 - 1] = 1;
      }
    }
    /*   V[V] ->  V[V-1] + V[1] */
    for (V=2; V<N+1; V++) {
      reactants[0] = N+V, reactants[1] = N+V-1, reactants[2] = N+1;

      for (j=0; j<3; j++) {
        row = reactants[j], col1 = reactants[0];
        dfill[(row-1)*dof + col1 - 1] = 1;
      }
    }

    /*   I[I] ->  I[I-1] + I[1] */
    for (I=2; I<N+1; I++) {
      reactants[0] = 2*N+I, reactants[1] = 2*N+I-1, reactants[2] = 2*N+1;

      for (j=0; j<3; j++) {
        row = reactants[j], col1 = reactants[0];
        dfill[(row-1)*dof + col1 - 1] = 1;
      }
    }

    /* He[1]-V[1]  ->  He[1] + V[1] */
    reactants[0] = 3*N+1, reactants[1] = 1, reactants[2] = N+1;

    for (j=0; j<3; j++) {
      row = reactants[j], col1 = reactants[0];
      dfill[(row-1)*dof + col1 - 1] = 1;
    }
    
    /*   He[He]-V[1] ->  He[He] + V[1]  */
    for (He=2; He<N+1; He++) {
      reactants[0] = 3*N+(He-1)*N+1, reactants[1] = He, reactants[2] = N+1;

      for (j=0; j<3; j++) {
        row = reactants[j], col1 = reactants[0];
        dfill[(row-1)*dof + col1 - 1] = 1;
      }
    }

    /*   He[1]-V[V] ->  He[1] + V[V]  */
    for (V=2; V<N+1; V++) {
      reactants[0] = 3*N+V, reactants[1] = 1, reactants[2] = N+V;

      for (j=0; j<3; j++) {
        row = reactants[j], col1 = reactants[0];
        dfill[(row-1)*dof + col1 - 1] = 1;
      }
    }

    /*   He[He]-V[V] ->  He[He-1]-V[V] + He[1]  */
    for (He=2; He<N+1; He++) {
      for (V=2; V<N+1; V++) {
        reactants[0] = 3*N+(He-1)*N+V, reactants[1] = 3*N+(He-2)*N+V, reactants[2] = 1;

        for (j=0; j<3; j++) {
          row = reactants[j], col1 = reactants[0];
          dfill[(row-1)*dof + col1 - 1] = 1;
        }
      }
    }
    
    /*   He[He]-V[V] ->  He[He]-V[V-1] + V[1]  */
    for (He=2; He<N+1; He++) {
      for (V=2; V<N+1; V++) {
        reactants[0] = 3*N+(He-1)*N+V, reactants[1] = 3*N+(He-1)*N+V-1, reactants[2] = N+1;

        for (j=0; j<3; j++) {
          row = reactants[j], col1 = reactants[0];
          dfill[(row-1)*dof + col1 - 1] = 1;
        }
      }
    }

    /*   He[He]-V[V] ->  He[He]-V[V+1] + I[1]  */
    /* Again, what is the reasonable dissociation rate? */
    for (He=1; He<N+1; He++) {
      for (V=1; V<N; V++) {
        reactants[0] = 3*N+(He-1)*N+V, reactants[1] = 3*N+(He-1)*N+V+1, reactants[2] = 2*N+1;

        for (j=0; j<3; j++) {
          row = reactants[j], col1 = reactants[0];
          dfill[(row-1)*dof + col1 - 1] = 1;
        }
      }
    }
  }
}
/* ------------------------------------------------------------------- */

typedef struct {
  DM          da;       /* defines the 2d layout of the He subvector */
  Vec         He;
  VecScatter  scatter;
  PetscViewer viewer;
} MyMonitorCtx;

#undef __FUNCT__
#define __FUNCT__ "MyMonitorMonitor"
/*
   Display He and V as a function of space and cluster size for each time step
*/
PetscErrorCode MyMonitorMonitor(TS ts,PetscInt timestep,PetscReal time,Vec solution, void *ictx)
{
  MyMonitorCtx   *ctx = (MyMonitorCtx*)ictx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecScatterBegin(ctx->scatter,solution,ctx->He,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatter,solution,ctx->He,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecView(ctx->He,ctx->viewer);CHKERRQ(ierr);
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
  ierr = VecScatterDestroy(&(*ctx)->scatter);CHKERRQ(ierr);
  ierr = VecDestroy(&(*ctx)->He);CHKERRQ(ierr);
  ierr = DMDestroy(&(*ctx)->da);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&(*ctx)->viewer);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyMonitorSetUp"
/*
   Sets up a monitor that will display He as a function of space and cluster size for each time step
*/
PetscErrorCode MyMonitorSetUp(TS ts)
{
  DM             da;
  PetscErrorCode ierr;
  PetscInt       xi,xs,xm,*idx,M,xj,cnt = 0,dof = 3*N + N*N;
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

  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = PetscNew(MyMonitorCtx,&ctx);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PetscObjectComm((PetscObject)da),NULL,"",PETSC_DECIDE,PETSC_DECIDE,600,400,&ctx->viewer);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&M,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = DMDAGetOwnershipRanges(da,&lx,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDACreate2d(PetscObjectComm((PetscObject)da),DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,PETSC_DETERMINE,1,2,1,lx,NULL,&ctx->da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx->da,0,"He");CHKERRQ(ierr);
  ierr = DMDASetFieldName(ctx->da,1,"V");CHKERRQ(ierr);
  ierr = DMDASetCoordinateName(ctx->da,0,"X coordinate direction");CHKERRQ(ierr);
  ierr = PetscSNPrintf(ycoor,32,"%D ... Cluster size ... 1",N);CHKERRQ(ierr);
  ierr = DMDASetCoordinateName(ctx->da,1,ycoor);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(ctx->da,&ctx->He);CHKERRQ(ierr);
  ierr = PetscMalloc(2*N*xm*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  cnt  = 0;
  for (xj=0; xj<N; xj++) {
    for (xi=xs; xi<xs+xm; xi++) {
      idx[cnt++] = dof*xi + xj;
      idx[cnt++] = dof*xi + xj + N;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ts),2*N*xm,idx,PETSC_OWN_POINTER,&is);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&C);CHKERRQ(ierr);
  ierr = VecScatterCreate(C,is,ctx->He,NULL,&ctx->scatter);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /* sets the bounds on the contour plot values so the colors mean the same thing for different timesteps */
  ierr = PetscViewerDrawSetBounds(ctx->viewer,2,valuebounds);CHKERRQ(ierr);

  ierr = TSMonitorSet(ts,MyMonitorMonitor,ctx,MyMonitorDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

