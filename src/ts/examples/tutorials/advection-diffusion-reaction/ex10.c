
static char help[] = ".\n";

/*

        C_t =  D*C_xx + R(C) from Brian Wirth's SciDAC project.

*/

#include <petscdmda.h>
#include <petscts.h>

#define N 2

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

typedef struct {
  PetscBool   reactions;
  PetscScalar HeDiffusion[6];
  PetscScalar VDiffusion[2];
  PetscScalar IDiffusion[2];
  PetscScalar forcingScale;
  PetscScalar reactionScale;
  PetscScalar dissociationScale;
} AppCtx;

extern PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode InitialConditions(DM,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS              ts;                 /* nonlinear solver */
  Vec             C;                  /* solution, residual vectors */
  PetscErrorCode  ierr;
  DM              da;
  AppCtx          ctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char *)0,help);

  ctx.reactions = PETSC_FALSE;
  ierr = PetscOptionsHasName(PETSC_NULL,"-reactions",&ctx.reactions);CHKERRQ(ierr);
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
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_MIRROR,-8,3*N+N*N,1,PETSC_NULL,&da);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&C);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,PETSC_NULL,IFunction,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetInitialTimeStep(ts,0.0,.001);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,100,50.0);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);


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
  
  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  /* Name each of the concentrations */
  for (He=1; He<N+1; He++) {
    ierr = PetscSNPrintf(string,16,"%d He",He);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
  }
  for (V=1; V<N+1; V++) {
    ierr = PetscSNPrintf(string,16,"%d V",V);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
  }
  for (I=1; I<N+1; I++) {
    ierr = PetscSNPrintf(string,16,"%d I",I);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
  }
  for (He=1; He<N+1; He++) {
    for (V=1; V<N+1; V++) {
      ierr = PetscSNPrintf(string,16,"%d He %d Ve",He,V);CHKERRQ(ierr);
      ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
    }
  }

  hx     = 1.0/(PetscReal)(Mx-1);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,C,&c);CHKERRQ(ierr);
  /* Shift the c pointer to allow accessing with index of 1, instead of 0 */
  c = (Concentrations*)(((PetscScalar*)c)-1);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;
    for (He=1; He<N+1; He++) {
      c[i].He[He] = 0.0;
    }
    for (V=1; V<N+1; V++) {
      c[i].V[V] = 1.0;
    }
    for (I=1; I<N+1; I++) {
      c[i].I[I] = 1.0;
    }
    for (He=1; He<N+1; He++) {
      for (V=1; V<N+1; V++) {
        c[i].HeV[He][V] = 0.0;
      }
    }
  }

  /*
     Restore vectors
  */
  c = (Concentrations*)(((PetscScalar*)c)+1);
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
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
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

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localC);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx     = 8.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);

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
  c = (Concentrations*)(((PetscScalar*)c)-1);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  f = (Concentrations*)(((PetscScalar*)f)-1);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

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

    if (!ctx->reactions) continue;
    /* ----------------------------------------------------------------
     ---- Compute reaction terms that can create a cluster of given size
    */
    for (He=2; He<N+1; He++) {
      /* compute all pairs of clusters of smaller size that can combine to create a cluster of size He,
         remove the upper half since they are symmetric to the lower half of the pairs. For example
              when He = 5 (cluster size 5) the pairs are
                 1   4
                 2   2
                 3   2  these last two are not needed in the sum since they repeat from above
                 4   1  this is why he < (He/2) + 1            */
      for (he=1; he<(He/2)+1; he++) {
        f[xi].He[He]    -= ctx->reactionScale*c[xi].He[he]*c[xi].He[He-he];

        /* remove the two clusters that merged to form the larger cluster */
        f[xi].He[he]    += ctx->reactionScale*c[xi].He[he]*c[xi].He[He-he];
        f[xi].He[He-he] += ctx->reactionScale*c[xi].He[he]*c[xi].He[He-he];
      }
    }
    for (V=2; V<N+1; V++) {
      for (v=1; v<(V/2)+1; v++) {
        f[xi].V[V]    -= ctx->reactionScale*c[xi].V[v]*c[xi].V[V-v];

        /* remove the clusters that merged to form the larger cluster */
        f[xi].V[v]    += ctx->reactionScale*c[xi].V[v]*c[xi].V[V-v];
        f[xi].V[V-v]  += ctx->reactionScale*c[xi].V[v]*c[xi].V[V-v];
      }
    }
    for (I=2; I<N+1; I++) {
      for (i=1; i<(I/2)+1; i++) {
        f[xi].I[I]    -= ctx->reactionScale*c[xi].I[i]*c[xi].I[I-i];

        /* remove the clusters that merged to form the larger cluster */
        f[xi].I[i]    += ctx->reactionScale*c[xi].I[i]*c[xi].I[I-i];
        f[xi].I[I-i]  += ctx->reactionScale*c[xi].I[i]*c[xi].I[I-i];
      }
    }  
    /* creation of He-V of size 1,1 */
    f[xi].HeV[1][1]   -= 1000*ctx->reactionScale*c[xi].He[1]*c[xi].V[1];

     /* remove the He and V  that merged to form the He-V cluster */
    f[xi].He[1]   += 1000*ctx->reactionScale*c[xi].He[1]*c[xi].V[1];
    f[xi].V[1]    += 1000*ctx->reactionScale*c[xi].He[1]*c[xi].V[1];

    /* Need reactions that create larger clusters of He-V */

    /* -------------------------------------------------------------------------
     ---- Compute dissociation terms that removes an item from a cluster
    */
    for (He=2; He<N+1; He++) {
      /* He cluster of size He becomes a cluster of size He-1 and a cluster of size 1 */
      f[xi].He[He-1]  -= ctx->dissociationScale*c[xi].He[He];
      f[xi].He[1]     -= ctx->dissociationScale*c[xi].He[He];
      f[xi].He[He]    += ctx->dissociationScale*c[xi].He[He];
    }
    for (V=2; V<N+1; V++) {
      /* V cluster of size V becomes a cluster of size V-1 and a cluster of size 1 */
      f[xi].V[V-1]  -= ctx->dissociationScale*c[xi].V[V];
      f[xi].V[1]    -= ctx->dissociationScale*c[xi].V[V];
      f[xi].V[V]    += ctx->dissociationScale*c[xi].V[V];
    }
    for (I=2; I<N+1; I++) {
      /* I cluster of size I becomes a cluster of size I-1 and a cluster of size 1 */
      f[xi].I[I-1]   -= ctx->dissociationScale*c[xi].I[I];
      f[xi].I[1]     -= ctx->dissociationScale*c[xi].I[I];
      f[xi].I[I]     += ctx->dissociationScale*c[xi].I[I];
    }
    /* need dissociation of mixed He-V clusters */

  }

  /*
     Restore vectors
  */
  c = (Concentrations*)(((PetscScalar*)c)+1);
  ierr = DMDAVecRestoreArray(da,localC,&c);CHKERRQ(ierr);
  f = (Concentrations*)(((PetscScalar*)f)+1);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


