
static char help[] = "Transient nonlinear driven cavity in 2d using ASPIN.\n\n";
/* 

This is an example for constructing local subproblems that may be used together to solve a larger problem.  The method
used (by default) is the ASPIN method, which constructs the global Function and Jacobian based upon the solution of
local subproblems.

ex26 has the larger problem as a monolithic solve.

Future improvements to this would involve having Xdot recalculated locally.

*/
#include <petscts.h>
#include <petscdmda.h>

/*
   User-defined routines and data structures
*/
typedef struct {
  PetscScalar u,v,omega,temp;
} Field;

typedef struct {
   /* ASPIN-specific things */
  DM           daoverlap;  /* a DM of the size of the global DM that has a larger overlap */
  DM           dalocal;    /* the local patch DM for the subproblem */
  PetscInt     offset[2];  /* The offset for the local DM in each dimension into the global DM */

  SNES         sneslocal;  /* the local solver for the subproblem */
  PetscReal    ptime;      /* the current time; required to pass the time to the local subproblem */
  PetscInt     overlap;    /* the amount of overlap */

} ASPINCtx;

typedef struct {
  PassiveReal  lidvelocity,prandtl,grashof;  /* physical parameters */
  PetscBool    parabolic;                    /* allow a transient term corresponding roughly to artificial compressibility */
  PetscBool    draw_contours;                /* flag - 1 indicates drawing contours */
  PetscReal    cfl_initial;                  /* CFL for first time step */
  ASPINCtx     *aspin;                       /* Unfortunate, yes, but we need the offset in the formfunction */
} AppCtx;

PetscErrorCode FormIFunctionBlockSNES(SNES,Vec,Vec,void *);
PetscErrorCode FormIFunctionLocal(DMDALocalInfo*,PetscReal,Field**,Field**,Field**,void*);
PetscErrorCode FormIFunctionASPIN(TS,PetscReal,Vec,Vec,Vec,void*);
PetscErrorCode FormInitialSolution(TS,Vec,AppCtx*);


#undef __FUNCT__
#define __FUNCT__ "ASPINCreate"
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Create the ASPIN discretization and solve
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
PetscErrorCode ASPINCreate(ASPINCtx* aspin,DM da,AppCtx* user)
{
  DMDALocalInfo  info;
  SNES           sneslocal;
  DM             dalocal;
  DM             daoverlap;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  aspin->overlap       = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-overlap",&aspin->overlap,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  PetscInt      xm = info.xm,ym=info.ym;

  aspin->offset[0] = info.xs;
  aspin->offset[1] = info.ys;

  /* determine the size of the local DA */
  if (info.xs != 0) {
    xm += aspin->overlap+1;
    aspin->offset[0] -= aspin->overlap+1;
  }
  if (info.xs+info.xm != info.mx) {
    xm += aspin->overlap+1;
  }

  if (info.ys != 0) {
    ym += aspin->overlap+1;
    aspin->offset[1] -= aspin->overlap+1;
  }
  if (info.ys+info.ym != info.my) {
    ym += aspin->overlap+1;
  }

  /* create overlap DM */
  ierr = DMDACreate2d(((PetscObject)da)->comm,
                      DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,
                      DMDA_STENCIL_BOX,
                      info.mx,info.my,
                      PETSC_DECIDE,PETSC_DECIDE,
                      4,
                      aspin->overlap+1,
                      0,0,
                      &daoverlap);CHKERRQ(ierr);
  aspin->daoverlap=daoverlap;

  /* create the local DM */
  ierr = DMDACreate2d(PETSC_COMM_SELF,
                      DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,
                      DMDA_STENCIL_BOX,
                      xm,ym,
                      PETSC_DECIDE,PETSC_DECIDE,
                      4,
                      1,
                      0,0,
                      &dalocal);CHKERRQ(ierr);
  aspin->dalocal=dalocal;

  ierr = SNESCreate(PETSC_COMM_SELF,&sneslocal);CHKERRQ(ierr);
  /* first coordinates of the local problem */
  ierr = SNESSetDM(sneslocal,dalocal);CHKERRQ(ierr);
  ierr = SNESAppendOptionsPrefix(sneslocal,"sub_");CHKERRQ(ierr);
  aspin->sneslocal = sneslocal;
  ierr = SNESSetFunction(sneslocal,PETSC_NULL,FormIFunctionBlockSNES,user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(sneslocal);CHKERRQ(ierr);
  user->aspin = aspin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ASPINDestroy"
PetscErrorCode ASPINDestroy(ASPINCtx* aspin)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = SNESDestroy(&aspin->sneslocal);CHKERRQ(ierr);
  ierr = DMDestroy(&aspin->dalocal);CHKERRQ(ierr);
  ierr = DMDestroy(&aspin->daoverlap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  AppCtx         user;                /* user-defined work context */
  ASPINCtx       aspin;               /* (scale dependent) data for the ASPIN solve */
  PetscInt       mx,my,steps;
  PetscErrorCode ierr;
  TS             ts;
  DM             da;
  Vec            X;
  PetscReal      ftime;
  TSConvergedReason reason;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return(1);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE,4,1,0,0,&da);CHKERRQ(ierr);
  ierr = TSSetDM(ts,(DM)da);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
		   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  /*
     Problem parameters (velocity of lid, prandtl, and grashof numbers)
  */
  user.lidvelocity   = 1.0/(mx*my);
  user.prandtl       = 1.0;
  user.grashof       = 1.0;
  user.draw_contours = PETSC_FALSE;
  user.parabolic     = PETSC_FALSE;
  user.cfl_initial   = 50.;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Driven cavity/natural convection options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lidvelocity","Lid velocity, related to Reynolds number","",user.lidvelocity,&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-prandtl","Ratio of viscous to thermal diffusivity","",user.prandtl,&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-grashof","Ratio of bouyant to viscous forces","",user.grashof,&user.grashof,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw_contours","Plot the solution with contours","",user.draw_contours,&user.draw_contours,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-parabolic","Relax incompressibility to make the system parabolic instead of differential-algebraic","",user.parabolic,&user.parabolic,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cfl_initial","Advective CFL for the first time step","",user.cfl_initial,&user.cfl_initial,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = DMDASetFieldName(da,0,"x-velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"y-velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,2,"Omega");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,3,"temperature");CHKERRQ(ierr);

  ierr = ASPINCreate(&aspin,da,&user);CHKERRQ(ierr);

  ierr = TSSetIFunction(ts,PETSC_NULL,FormIFunctionASPIN,&aspin);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,10000,1e12);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,user.cfl_initial/(user.lidvelocity*mx));CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&X);CHKERRQ(ierr);
  ierr = FormInitialSolution(ts,X,&user);CHKERRQ(ierr);

  ierr = TSSolve(ts,X,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %G after %D steps\n",TSConvergedReasons[reason],ftime,steps);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = ASPINDestroy(&aspin);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

/* ------------------------------------------------------------------- */


#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
/*
   FormInitialSolution - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialSolution(TS ts,Vec X,AppCtx *user)
{
  DM             da;
  PetscInt       i,j,mx,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      grashof,dx;
  Field          **x;

  grashof = user->grashof;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx  = 1.0/(mx-1);

  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x[j][i].u     = 0.0;
      x[j][i].v     = 0.0;
      x[j][i].omega = 0.0;
      x[j][i].temp  = (grashof>0)*i*dx;
    }
  }

  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormIFunctionLocal"
PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info,PetscReal ptime,Field **x,Field **xdot,Field **f,void *ptr)
 {
  AppCtx         *user = (AppCtx*)ptr;
  ASPINCtx       *aspin = user->aspin;
  PetscErrorCode ierr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,udot,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;
  DMDALocalInfo ginfo;
  PetscFunctionBegin;

  ierr = DMDAGetLocalInfo(aspin->daoverlap,&ginfo);CHKERRQ(ierr);
  grashof = user->grashof;
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  dhx = (PetscReal)(ginfo.mx-1);  dhy = (PetscReal)(ginfo.my-1);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints + aspin->offset[1] == 0) {
    j = 0;
    yints += 1;
    /* bottom edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy;
      f[j][i].temp  = x[j][i].temp-x[j+1][i].temp;
    }
  } else {
    yints += 1;
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte + aspin->offset[1] == ginfo.my) {
    j = info->my-1;
    yinte -= 1;
    /* top edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
        f[j][i].u     = x[j][i].u - lid;
        f[j][i].v     = x[j][i].v;
        f[j][i].omega = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy;
	f[j][i].temp  = x[j][i].temp-x[j-1][i].temp;
    }
  } else {
    yinte -= 1;
  }

  /* Test whether we are on the left edge of the global array */
  if (xints + aspin->offset[0] == 0) {
    i = 0;
    xints += 1;
    /* left edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx;
      f[j][i].temp  = x[j][i].temp;
    }
  } else {
    xints += 1;
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte + aspin->offset[0] == ginfo.mx) {
    i = info->mx-1;
    xinte -= 1;
    /* right edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i].v - x[j][i-1].v)*dhx;
      f[j][i].temp  = x[j][i].temp - (PetscReal)(grashof>0);
    }
  } else {
    xinte -= 1;
  }

  //PetscPrintf(PETSC_COMM_SELF,"%d: %d %d, %d %d\n",rank,xints + user->offset[0],xinte + user->offset[0],yints + user->offset[1],yinte + user->offset[1]);

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {

	/*
	  convective coefficients for upwinding
        */
	vx = x[j][i].u; avx = PetscAbsScalar(vx);
        vxp = .5*(vx+avx); vxm = .5*(vx-avx);
	vy = x[j][i].v; avy = PetscAbsScalar(vy);
        vyp = .5*(vy+avy); vym = .5*(vy-avy);

	/* U velocity */
        u          = x[j][i].u;
        udot       = user->parabolic ? xdot[j][i].u : 0.;
        uxx        = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
        uyy        = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
        f[j][i].u  = udot + uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

	/* V velocity */
        u          = x[j][i].v;
        udot       = user->parabolic ? xdot[j][i].v : 0.;
        uxx        = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
        uyy        = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
        f[j][i].v  = udot + uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;

	/* Omega */
        u          = x[j][i].omega;
        uxx        = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
        uyy        = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;
	f[j][i].omega = (xdot[j][i].omega + uxx + uyy
                         + (vxp*(u - x[j][i-1].omega)
                            + vxm*(x[j][i+1].omega - u)) * hy
                         + (vyp*(u - x[j-1][i].omega)
                            + vym*(x[j+1][i].omega - u)) * hx
                         - .5 * grashof * (x[j][i+1].temp - x[j][i-1].temp) * hy);

        /* Temperature */
        u             = x[j][i].temp;
        uxx           = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
        uyy           = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
	f[j][i].temp =  (xdot[j][i].temp + uxx + uyy
                         + prandtl * ((vxp*(u - x[j][i-1].temp)
                                       + vxm*(x[j][i+1].temp - u)) * hy
                                      + (vyp*(u - x[j-1][i].temp)
                                         + vym*(x[j+1][i].temp - u)) * hx));
    }
  }

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* SNES wrapper function for the individual block SNES -- we*/
#undef __FUNCT__
#define __FUNCT__ "FormIFunctionBlockSNES"
PetscErrorCode FormIFunctionBlockSNES(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode   ierr;
  AppCtx           *user = (AppCtx*)ctx;
  ASPINCtx         *aspin = user->aspin;
  DM               dalocal;
  DMDALocalInfo    info;
  Field            **x,**xdot,**f;
  Vec              Xlocal,Xdot;
  PetscReal        ptime = aspin->ptime;
  
  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dalocal);CHKERRQ(ierr);
  ierr = VecSet(F,0.);CHKERRQ(ierr);
  ierr = DMGetNamedGlobalVector(dalocal,"block_Xdot",&Xdot);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dalocal,&Xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dalocal,X,INSERT_VALUES,Xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dalocal,X,INSERT_VALUES,Xlocal);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(dalocal,Xlocal,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dalocal,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dalocal,F,&f);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(dalocal,&info);CHKERRQ(ierr);
  ierr = FormIFunctionLocal(&info,ptime,x,xdot,f,ctx);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(dalocal,Xlocal,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dalocal,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dalocal,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreNamedGlobalVector(dalocal,"block_Xdot",&Xdot);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dalocal,&Xlocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormIFunctionASPIN"
PetscErrorCode FormIFunctionASPIN(TS ts,PetscReal ptime,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  Vec            Xlocal,Xlocalloc,Xgloballoc;
  ASPINCtx       *aspin = (ASPINCtx*)ctx;
  DM             daoverlap=aspin->daoverlap,dalocal=aspin->dalocal;
  SNES           sneslocal=aspin->sneslocal;
  DMDALocalInfo  info,ginfo;
  Vec            Xdotlocal,Xdotlocalloc,Xdotgloballoc;

  PetscFunctionBegin;
  aspin->ptime = ptime;
  ierr = DMGetGlobalVector(dalocal,&Xlocal);CHKERRQ(ierr);
  ierr = DMGetNamedGlobalVector(dalocal,"block_Xdot",&Xdotlocal);CHKERRQ(ierr);

  /* get work vectors for the local and global DAs */
  ierr = DMDAGetLocalInfo(daoverlap,&ginfo);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dalocal,&info);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dalocal,&Xlocalloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dalocal,&Xdotlocalloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(daoverlap,&Xgloballoc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(daoverlap,&Xdotgloballoc);CHKERRQ(ierr);
  ierr = VecSet(Xgloballoc,0.);CHKERRQ(ierr);

  /* get X and Xdot for the overlap */
  ierr = DMGlobalToLocalBegin(daoverlap,X,INSERT_VALUES,Xgloballoc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(daoverlap,X,INSERT_VALUES,Xgloballoc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(daoverlap,Xdot,INSERT_VALUES,Xdotgloballoc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(daoverlap,Xdot,INSERT_VALUES,Xdotgloballoc);CHKERRQ(ierr);

  /* transfer X and Xdot over to the global vectors for the local subproblem */
  ierr = VecCopy(Xgloballoc,Xlocal);CHKERRQ(ierr);
  ierr = VecCopy(Xdotgloballoc,Xdotlocal);CHKERRQ(ierr);

  ierr = DMRestoreNamedGlobalVector(dalocal,"block_Xdot",&Xdotlocal);CHKERRQ(ierr);

  /* local solve */
  ierr = SNESSolve(sneslocal,PETSC_NULL,Xlocal);CHKERRQ(ierr);

  /* copy the local solution back over and redistribute */
  ierr = VecAXPY(Xgloballoc,-1.0,Xlocal);CHKERRQ(ierr);

  /* restrict and subtract */
  ierr = VecSet(F,0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(daoverlap,Xgloballoc,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(daoverlap,Xgloballoc,ADD_VALUES,F);CHKERRQ(ierr);

  /* restore work vectors */
  ierr = DMRestoreLocalVector(dalocal,&Xlocalloc);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(daoverlap,&Xgloballoc);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dalocal,&Xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(daoverlap,&Xdotgloballoc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
