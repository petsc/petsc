
static char help[] = ".\n";

/*

        u_t =  u_xx + R(u) from Brian Wirth's SciDAC project.

*/

#include <petscdmda.h>
#include <petscts.h>

#define N 2

/*
     Define all the concentrations (there is one of these structs at each grid point)

      He[He] represents the clusters of pure Helium of size He + 1,
      V[V] the Vacencies of size V+1,
      I[I] represents the clusters of Interstials of size I + 1,  and
      HeV[He][V]  the mixed Helium-Vacancy clusters of size He+1 and V+1

      The variables He, V, I are always used to index into the concentrations of He, V, and I respectively

*/
typedef struct {
  PetscScalar He[N];
  PetscScalar V[N];
  PetscScalar I[N];
  PetscScalar HeV[N][N];
} Concentrations;

extern PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode InitialConditions(DM,Vec);
extern PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS              ts;                 /* nonlinear solver */
  Vec             U;                  /* solution, residual vectors */
  Mat             J;                  /* Jacobian matrix */
  PetscInt        maxsteps = 1000;
  PetscErrorCode  ierr;
  DM              da;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char *)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_MIRROR,-8,3*N+N*N,1,PETSC_NULL,&da);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&U);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,MATAIJ,&J);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,PETSC_NULL,IFunction,0);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(da,U);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetInitialTimeStep(ts,0.0,.001);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxsteps,1.0);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "InitialConditions"
PetscErrorCode InitialConditions(DM da,Vec U)
{
  PetscErrorCode ierr;
  PetscInt       i,I,He,V,xs,xm,Mx,cnt = 0;
  Concentrations *u;
  PetscReal      hx,x;
  char           string[16];
  
  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  /* Name each of the concentrations */

  for (He=0; He<N; He++) {
    ierr = PetscSNPrintf(string,16,"%d He",He+1);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
  }
  for (V=0; V<N; V++) {
    ierr = PetscSNPrintf(string,16,"%d V",V+1);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
  }
  for (I=0; I<N; I++) {
    ierr = PetscSNPrintf(string,16,"%d I",I+1);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
  }
  for (He=0; He<N; He++) {
    for (V=0; V<N; V++) {
      ierr = PetscSNPrintf(string,16,"%d He %d Ve",He+1,V+1);CHKERRQ(ierr);
      ierr = DMDASetFieldName(da,cnt++,string);CHKERRQ(ierr);
    }
  }

  hx     = 1.0/(PetscReal)(Mx-1);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;
    for (He=0; He<N; He++) {
      u[i].He[V] = x;
    }
    for (V=0; V<N; V++) {
      u[i].V[V] = x;
    }
    for (I=0; I<N; I++) {
      u[i].I[I] = x;
    }
    for (He=0; He<N; He++) {
      for (V=0; V<N; V++) {
        u[i].HeV[He][V] = 0.0;
      }
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
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
PetscErrorCode IFunction(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr)
{
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,Mx,xs,xm,I,He,V,he;
  PetscReal      hx,sx,x;
  PetscScalar    uxx;
  Concentrations *u,*f;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx     = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);

  /*
       F  = Udot +  all the diffusion and reaction terms added below 
  */
  ierr = VecCopy(Udot,F);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

   /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /*
     Loop over grid points computing ODE terms for each grid point
  */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;

    /*
     ---- Compute diffusion over the locally owned part of the grid
    */
    /* He clusters larger than 5 do not diffuse -- are immobile */
    for (He=0; He<PetscMin(N,5); He++) {
      f[i].He[He] -=  (-2.0*u[i].He[He] + u[i-1].He[He] + u[i+1].He[He])*sx;
    }

    /* V and I clusters ONLY of size 1 diffuse */
    f[i].V[0] -=  - (-2.0*u[i].V[0] + u[i-1].V[0] + u[i+1].V[0])*sx;
    f[i].I[0] -=  - (-2.0*u[i].I[0] + u[i-1].I[0] + u[i+1].I[0])*sx;

    /* Mixed He - V clusters are immobile  */


    /*
     ---- Compute reaction terms that can create a cluster of given size
    */
    for (He=1; He<N; He++) {
      /* compute all pairs of clusters of smaller size that can combine to create a cluster of size He+1,
         remove the upper half since they are symmetric to the lower half of the pairs. For example
              when He = 4 (cluster size 5) the pairs are
                 0   3    (cluster sizes 1 and 4)
                 1   2    (cluster sizes 2 and 3)
                 2   1    (cluster sizes 3 and 2) these last two are not needed in the sum.
                 3   0    (cluster sizes 4 and 1)               */
      for (he=0; he<(He+1)/2; he++) {
        f[i].He[He] += /* coeefficient */ f[i].He[he]*f[i].He[He-he-1];
      }
    }

  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


