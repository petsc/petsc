#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: monitor.c,v 1.59 1998/07/13 23:48:34 curfman Exp curfman $";
#endif

/*
   This file contains various monitoring routines used by the SNES/Julianne code.
 */
#include "user.h"
#include <math.h>

#undef __FUNC__
#define __FUNC__ "MonitorEuler"
/* 
   MonitorEuler - Routine that is called at the conclusion
   of each successful step of the nonlinear solver.  The user
   sets this routine by calling SNESSetMonitor().

   Input Parameters:
   snes  - SNES context
   its   - current iteration number
   fnorm - current function norm
   dummy - (optional) user-defined application context, as set
           by SNESSetMonitor().

   Notes:
   Depending on runtime options, this routine can
     - write the nonlinear function vector, F, to a file
     - compute a new CFL number and the associated pseudo-transient
       continuation term (for CFL number advancement)
     - call (a slightly modified variant of) the monitoring routine
       within the original Julianne code
   Additional monitoring (such as dumping fields for VRML viewing)
   is done within the routine ComputeFunction().
 */
int MonitorEuler(SNES snes,int its,double fnorm,void *dummy)
{
  MPI_Comm comm;
  Euler      *app = (Euler *)dummy;
  Scalar     negone = -1.0, cfl1, ratio, ratio1, ksprtol, ksprtol1;
  Vec        DX, X;
  Viewer     view1;
  double     t;
  /* PLogDouble t; v2.0.22 */
  char       filename[64], outstring[64];
  int        ierr, lits, overlap, flg;

  PLogEventBegin(app->event_monitor,0,0,0,0);
  PetscObjectGetComm((PetscObject)snes,&comm);

  /* Print the vector F (intended for debugging) */
  if (app->print_vecs) {
    ierr = SNESGetFunction(snes,&app->F); CHKERRQ(ierr);
    sprintf(filename,"res.%d.out",its);
    ierr = ViewerASCIIOpen(app->comm,filename,&view1); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view1,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = VecView(app->F,view1); CHKERRQ(ierr);
    ierr = ViewerDestroy(view1); CHKERRQ(ierr);
  }
  app->flog[its]  = log10(fnorm);
  app->fcfl[its]  = app->cfl; 
  ierr            = PetscGetTime(&t); CHKERRQ(ierr);
  app->ftime[its] = t - app->time_init;
  if (!its) {
    /* Do the following only during the initial call to this routine */
    app->fnorm_init    = app->fnorm_last = fnorm;
    app->lin_its[0]    = 0;
    app->lin_rtol[0]   = 0;
    app->nsup[0]       = 0;
    app->c_lift[0]     = 0;
    app->c_drag[0]     = 0;
    /*    if (!app->no_output) { */
      if (app->cfl_advance == ADVANCE)
        PetscPrintf(comm,"iter=%d, fnorm=%g, fnorm reduction ratio=%g, CFL_init=%g\n",
           its,fnorm,app->f_reduction,app->cfl);
      else PetscPrintf(comm,"iter = %d, Function norm = %g, CFL = %g\n",its,fnorm,app->cfl);
      if (app->rank == 0) {
        overlap = 0;
        ierr = OptionsGetInt(PETSC_NULL,"-pc_asm_overlap",&overlap,&flg); CHKERRQ(ierr);
        if (app->problem == 1) {
          sprintf(filename,"f_m6%s_cc%d_asm%d_p%d.m","c",app->cfl_snes_it,overlap,app->size);
          sprintf(outstring,"zsnes_m6%s_cc%d_asm%d_p%d = [\n","c",app->cfl_snes_it,overlap,app->size);
        }
        else if (app->problem == 2) {
          sprintf(filename,"f_m6%s_cc%d_asm%d_p%d.m","f",app->cfl_snes_it,overlap,app->size);
          sprintf(outstring,"zsnes_m6%s_cc%d_asm%d_p%d = [\n","f",app->cfl_snes_it,overlap,app->size);
        }
        else if (app->problem == 3) {
          sprintf(filename,"f_m6%s_cc%d_asm%d_p%d.m","n",app->cfl_snes_it,overlap,app->size);
          sprintf(outstring,"zsnes_m6%s_cc%d_asm%d_p%d = [\n","n",app->cfl_snes_it,overlap,app->size);
        }
        else if (app->problem == 5) {
          sprintf(filename,"f_duct%s_asm%d_p%d.m","c",overlap,app->size);
          sprintf(outstring,"zsnes_ductc_asm%d_p%d = [\n",overlap,app->size);
        } 
        else if (app->problem == 6) {
          sprintf(filename,"f_duct%s_asm%d_p%d.m","f",overlap,app->size);
          sprintf(outstring,"zsnes_ductf_asm%d_p%d = [\n",overlap,app->size);
        } 
        else SETERRQ(1,0,"No support for this problem number");
        app->fp = fopen(filename,"w"); 
        if (!app->fp) SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open output file");
	fprintf(app->fp,"%% iter, fnorm2, log(fnorm2), CFL#, time, ksp_its, ksp_rtol, c_lift, c_drag, nsup\n");
        fprintf(app->fp,outstring);
	fprintf(app->fp," %5d  %8.4e  %8.4f  %8.1f  %10.2f  %4d  %7.3e  %8.4e  %8.4e  %8d\n",
                its,app->farray[its],app->flog[its],app->fcfl[its],app->ftime[its],app->lin_its[its],
                app->lin_rtol[its],app->c_lift[its],app->c_drag[its],app->nsup[its]);
        }
      /*    } */
    app->sles_tot += app->lin_its[its];
  } else {
    /* For the first iteration and onward we do the following */

    /* Are we transitioning the discretization order from 1 to 2? */
    if (app->order_transition == TRANS_WAIT) {
      if (fnorm/app->fnorm_init < app->order_transition_rtol) {   /* Are we ready to begin? */
        /* simple transition for now */
        app->order_transition_theta = 1.0;
        app->order_transition       = TRANS_START;
        app->order_transition_it    = its;
        app->cfl                    = app->cfl_init*2.0;
        app->cfl_init               = app->cfl;
        app->order                  = 2;
	/*        if (!app->no_output) */
          PetscPrintf(comm,"Discr. transition: 1-2; fnorm/fnorm_init = %g, f_transition tol = %g, cfl = %g\n",
          fnorm/app->fnorm_init,app->order_transition_rtol,app->cfl);
       } 
    } else if (app->order_transition == TRANS_START) {
      app->order_transition = TRANS_DONE;
    }

    /* Compute new CFL number if desired */
    /* Note: BCs change at iter bcswitch, so we defer CFL increase until after this point */
    if (app->cfl_advance == ADVANCE && its > app->bcswitch) {
      if (app->order_transition == TRANS_START
         || (app->order_transition == TRANS_DONE && app->order_transition_it == its-1)) {
         if (!app->no_output)
           PetscPrintf(comm,"Same CFL: discr. transition: cfl = %g\n",app->cfl);
      } else {
          /* Check to see whether we want to begin CFL advancement now if we haven't already */
        if (!app->cfl_begin_advancement) {
          if (fnorm/app->fnorm_init <= app->f_reduction) {
            app->cfl_begin_advancement = 1;
            if (!app->no_output) 
              PetscPrintf(comm,"Beginning CFL advancement: fnorm/fnorm_init = %g, f_reduction ratio = %g\n",
              fnorm/app->fnorm_init,app->f_reduction);
          } else {
            if (!app->no_output)
              PetscPrintf(comm,"Same CFL: fnorm/fnorm_init = %g, f_reduction ratio = %g, cfl = %g\n",
              fnorm/app->fnorm_init,app->f_reduction,app->cfl);
          }
        }
        if (app->cfl_begin_advancement) {
          /* We've already begun CFL advancement */
          if (!(its%app->cfl_snes_it)) {
            if (app->cfl_advance == ADVANCE) {
               if (fnorm != fnorm) {
                 PetscPrintf(comm,"Problem at nonlinear iteration %d!\n",its);
                 SETERRQ(1,0,"NaN detection for fnorm - probably increasing CFL too quickly!");
               }
               ratio1 = app->fnorm_last / fnorm;
               if (ratio1 >= 1.0) ratio = PetscMin(ratio1,app->cfl_max_incr);
               else               ratio = PetscMax(ratio1,app->cfl_max_decr);
               cfl1 = app->cfl * ratio;
            } else SETERRQ(1,1,"Unsupported CFL advancement strategy");
            cfl1     = PetscMin(cfl1,app->cfl_max);
            app->cfl = PetscMax(cfl1,app->cfl_init);
            if (!app->no_output) PetscPrintf(comm,"Next iter CFL: cfl=%g\n",app->cfl);
            /* if (!app->no_output) PetscPrintf(comm,"ratio1=%g, ratio_clipped=%g, cfl1=%g, new cfl=%g\n",
                                             ratio1,ratio,cfl1,app->cfl); */
          } else {
            /* if (!app->no_output) PetscPrintf(comm,"Hold CFL\n"); */
          }
        }
      }
    }
    app->fnorm_last = fnorm;

    /* Calculate new pseudo-transient continuation term, dt */
    /*    if (app->sctype == DT_MULT || next iteration forms Jacobian || matrix-free mult) */
    eigenv_(app->dt,app->xx,app->p,
         app->sadai,app->sadaj,app->sadak,
         app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
         app->aiz,app->ajz,app->akz,&app->ts_type);

    /* Extract solution and update vectors; convert to Julianne format */
    ierr = SNESGetSolutionUpdate(snes,&DX); CHKERRQ(ierr);
    ierr = VecScale(&negone,DX); CHKERRQ(ierr);
    ierr = PackWork(app,app->da,DX,app->localDX,&app->dxx); CHKERRQ(ierr);

    /* Call Julianne monitoring routine and update CFL number */
    ierr = jmonitor_(&app->flog[its],&app->cfl,
           app->work_p,app->xx,app->p,app->dxx,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz,&app->c_lift[its],&app->c_drag[its],&app->nsup[its]); CHKERRQ(ierr); 
    ierr = RestoreWork(app,app->da,DX,app->localDX,&app->dxx); CHKERRQ(ierr);

    /* Get some statistics about the iterative solver */
    ierr = SNESGetNumberLinearIterations(snes,&lits); CHKERRQ(ierr);
    app->lin_its[its] = lits - app->last_its;
    app->last_its     = lits;
    ierr = KSPGetTolerances(app->ksp,&(app->lin_rtol[its]),PETSC_NULL,PETSC_NULL,
           PETSC_NULL); CHKERRQ(ierr);
    ksprtol1 = sqrt(fnorm/app->fnorm_init);
    ksprtol = PetscMin(app->ksp_rtol_max,PetscMax(ksprtol1,app->ksp_rtol_min));
    ierr = KSPSetTolerances(app->ksp,ksprtol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);

    /* temporarily allow this output */
    /*    if (!app->no_output) {  */
      PetscPrintf(comm,"iter = %d, Function norm %g, lin_its = %d\n",
                  its,fnorm,app->lin_its[its]);
      if (app->rank == 0) {
	fprintf(app->fp," %5d  %8.4e  %8.4f  %8.1f  %10.2f  %4d  %7.3e  %8.4e  %8.4e  %8d\n",
                its,app->farray[its],app->flog[its],app->fcfl[its],app->ftime[its],app->lin_its[its],
                app->lin_rtol[its],app->c_lift[its],app->c_drag[its],app->nsup[its]);
        fflush(app->fp);
	/*      } */
    }

    app->sles_tot += app->lin_its[its];

    ierr = OptionsHasName(PETSC_NULL,"-bump_dump_all",&flg); CHKERRQ(ierr);
    if (flg && app->problem >= 5) {
      Scalar *xa;
      ierr = SNESGetSolution(snes,&X); CHKERRQ(ierr);
      ierr = VecGetArray(X,&xa); CHKERRQ(ierr);
      if (!PetscStrcmp(app->mmtype,MMFP)) {
        ierr = VisualizeFP_Matlab(its,app,xa); CHKERRQ(ierr);
      } else SETERRQ(1,0,"Option not supported yet");
      ierr = VecRestoreArray(X,&xa); CHKERRQ(ierr);
    }

    if (!app->no_output) {
      /* Check solution */
      if (app->check_solution && app->bctype == IMPLICIT) {
        ierr = SNESGetSolution(snes,&X); CHKERRQ(ierr);
        ierr = CheckSolution(app,X); CHKERRQ(ierr);
      }
      if (app->print_vecs) {
        ierr = SNESGetSolution(snes,&X); CHKERRQ(ierr);
        sprintf(filename,"x.%d.out",its);
        ierr = ViewerASCIIOpen(app->comm,filename,&view1); CHKERRQ(ierr);
        ierr = ViewerSetFormat(view1,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
        ierr = VecView(X,view1); CHKERRQ(ierr);
        ierr = ViewerDestroy(view1); CHKERRQ(ierr);
      }

      /* Print factored matrix - intended for debugging */
      if (app->print_vecs) {
        SLES   sles;
        PC     pc;
        PCType pctype;
        Mat    fmat;
        Viewer view;
        ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
        ierr = SLESGetPC(sles,&pc); CHKERRQ(ierr);
        ierr = PCGetType(pc,&pctype); CHKERRQ(ierr);
        if (pctype == PCILU) {
          ierr = PCGetFactoredMatrix(pc,&fmat);
          ierr = ViewerASCIIOpen(app->comm,"factor.out",&view); CHKERRQ(ierr);
          ierr = ViewerSetFormat(view,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
          ierr = MatView(fmat,view); CHKERRQ(ierr);
          ierr = ViewerDestroy(view); CHKERRQ(ierr);
        }
      }
    }
  }
  app->iter = its+1;
  PLogEventEnd(app->event_monitor,0,0,0,0);
  return 0;
}
int DFVecFormUniVec_MPIRegular_Private(Vec,Vec*);
#undef __FUNC__
#define __FUNC__ "MonitorDumpGeneral"
/* --------------------------------------------------------------- */
/* 
   MonitorDumpGeneral - Dumps solution fields for later use in visualization.

   Input Parameters:
   snes - nonlinear solver context
   X    - current iterate
   app - user-defined application context
 */
int MonitorDumpGeneral(SNES snes,Vec X,Euler *app)
{
  FILE     *fp;
  int      iter, ierr, i, j, k, ijkx, ni, nj, nk, ni1, nj1, nk1;
  int      istart, iend, jstart, jend, kstart, kend;
  char     filename[64];
  Vec      P_natural, X_natural, F_natural;
  Scalar   *xx, *pp, *ff, *xc = app->xc, *yc = app->yc, *zc = app->zc;
  Scalar   mach, sfluid, ssound, r, yv, xv, xmin, xmax, ymin, ymax, zmin, zmax, gamma1, gm1;
  Scalar   zv, e_av, p_av, f_xv, f_yv, f_zv, f_r, f_e_av, f_tot;

#define xcoord3(i,j,k) xc[(k)*nj*ni + (j)*ni + (i)]
#define ycoord3(i,j,k) yc[(k)*nj*ni + (j)*ni + (i)]
#define zcoord3(i,j,k) zc[(k)*nj*ni + (j)*ni + (i)]
#define den3(i,j,k) xx[5*((k)*nj1*ni1 + (j)*ni1 + (i))]
#define ru3(i,j,k) xx[5*((k)*nj1*ni1 + (j)*ni1 + (i)) + 1]
#define rv3(i,j,k) xx[5*((k)*nj1*ni1 + (j)*ni1 + (i)) + 2]

/* new additions */
#define rw3(i,j,k) xx[5*((k)*nj1*ni1 + (j)*ni1 + (i)) + 3]
#define e3(i,j,k) xx[5*((k)*nj1*ni1 + (j)*ni1 + (i)) + 4]
#define p3(i,j,k) pp[(k)*nj1*ni1 + (j)*ni1 + (i)]

#define f_den(i,j,k) ff[5*((k)*nj1*ni1 + (j)*ni1 + (i))]
#define f_ru(i,j,k) ff[5*((k)*nj1*ni1 + (j)*ni1 + (i)) + 1]
#define f_rv(i,j,k) ff[5*((k)*nj1*ni1 + (j)*ni1 + (i)) + 2]
#define f_rw(i,j,k) ff[5*((k)*nj1*ni1 + (j)*ni1 + (i)) + 3]
#define f_e(i,j,k) ff[5*((k)*nj1*ni1 + (j)*ni1 + (i)) + 4]

  /* Since we call MonitorDumpGeneral() from the routine ComputeFunction(), packing and
     computing the pressure have already been done. */
  /*
  ierr = PackWork(app,app->da,app->X,app->localX,&app->xx); CHKERRA(ierr);
  ierr = jpressure_(app->xx,app->p); CHKERRA(ierr);
  */

  /* If using multiple processors, then assemble the pressure and field vectors on only
     1 processor (in the appropriate ordering) and then view them.  Eventually, we will
     optimize such manipulations and hide them in the viewer routines */

  ierr = SNESGetFunction(snes,&app->F); CHKERRQ(ierr);
  if (app->size != 1) {
    /* Pack pressure and field vectors */
    ierr = UnpackWorkComponent(app,app->p,app->P); CHKERRQ(ierr);
    /*    ierr = DFVecFormUniVec_MPIRegular_Private(app->P,&P_uni); CHKERRQ(ierr);
    ierr = DFVecFormUniVec_MPIRegular_Private(app->X,&X_uni); CHKERRQ(ierr);
    ierr = DFVecFormUniVec_MPIRegular_Private(app->F,&F_uni); CHKERRQ(ierr); */

    ierr = DACreateNaturalVector(app->da,&P_natural);CHKERRQ(ierr);
    ierr = DACreateNaturalVector(app->da,&X_natural);CHKERRQ(ierr);
    ierr = DACreateNaturalVector(app->da,&F_natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(app->da1,app->P,INSERT_VALUES,P_natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(app->da,app->X,INSERT_VALUES,F_natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalBegin(app->da,app->F,INSERT_VALUES,X_natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(app->da1,app->P,INSERT_VALUES,P_natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(app->da,app->X,INSERT_VALUES,X_natural);CHKERRQ(ierr);
    ierr = DAGlobalToNaturalEnd(app->da,app->F,INSERT_VALUES,F_natural);CHKERRQ(ierr);
  }

  /* Dump data from first processor only */
  if (app->size !=1) SETERRQ(1,0,"Data dumping temporarily works for 1 proc only ... We will extend this soon!");
  if (app->rank == 0) {
    if (app->size != 1) {
      ierr = VecGetArray(P_natural,&pp); CHKERRQ(ierr);
      ierr = VecGetArray(X_natural,&xx); CHKERRQ(ierr);
      ierr = VecGetArray(F_natural,&ff); CHKERRQ(ierr);
    } else {
      xx = app->xx;
      pp = app->p;
      ff = app->dxx;
    }
  
    ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
    sprintf(filename,"euler.%d.out",iter);
    fp = fopen(filename,"w"); 
    ni  = app->ni;  nj  = app->nj;  nk = app->nk;
    ni1 = app->ni1; nj1 = app->nj1; nk1 = app->nk1;

    gamma1 = 1.4;
    gm1    = gamma1 - 1.0;
    kstart = 0;
    kend   = app->nk;
    istart = 0;
    iend   = app->ni;
    jstart = 0;
    jend   = app->nj;

    /*
    Restrict data to subset of total domain

    if (app->problem == 1) {
      kstart = 0;
      kend   = app->ktip + 3;
      istart = app->itl - 5;
      iend   = app->itu + 5;
      jstart = 0;
      jend   = app->nj - 3;
    } else if (app->problem == 2) {
      kstart = 0;
      kend   = app->ktip + 5;
      istart = app->itl - 5;
      iend   = app->itu + 5;
      jstart = 0;
      jend   = app->nj - 10;
    } else if (app->problem == 3) {
      kstart = 0;
      kend   = app->nk;
      istart = 0;
      iend   = app->ni;
      jstart = 0;
      jend   = app->nj;
    } else SETERRQ(1,0,"Unsupported problem");
 */
    fprintf(fp,"VARIABLES=x,y,z,pressure,mach,x-momentum,y-momentum,z-momentum,energy,\n\
          f-average,f-density,f-x-momentum,f-y-momentum,f-z-momentum,f-energy\n");
    fprintf(fp,"istart=%d, iend=%d, jstart=%d, jend=%d, kstart=%d, kend=%d\n",
                istart,iend,jstart,jend,kstart,kend);
    xmin = 1000;
    xmax = -1000;
    ymin = 1000;
    ymax = -1000;
    zmin = 1000;
    zmax = -1000;

    for (k=kstart; k<kend; k++) {
      for (j=jstart; j<jend; j++) {
        for (i=istart; i<iend; i++) {
          ijkx = k*nj1*ni1 + j*ni1 + i;
          yv   = 0.25 * (rv3(i,j,k) + rv3(i+1,j,k) + rv3(i,j+1,k) + rv3(i+1,j+1,k));
          xv   = 0.25 * (ru3(i,j,k) + ru3(i+1,j,k) + ru3(i,j+1,k) + ru3(i+1,j+1,k));
          zv   = 0.25 * (rw3(i,j,k) + rw3(i+1,j,k) + rw3(i,j+1,k) + rw3(i+1,j+1,k));
          r    = 0.25 * (den3(i,j,k) + den3(i+1,j,k) + den3(i,j+1,k) + den3(i+1,j+1,k));
          e_av = 0.25 * (e3(i,j,k) + e3(i+1,j,k) + e3(i,j+1,k) + e3(i+1,j+1,k));
          p_av = 0.25 * (p3(i,j,k) + p3(i+1,j,k) + p3(i,j+1,k) + p3(i+1,j+1,k));

          sfluid = sqrt(xv*xv + yv*yv) / r;
          ssound = sqrt(pow(r,gm1));
          mach = sfluid/ssound;

          f_yv   = 0.25 * (f_rv(i,j,k) + f_rv(i+1,j,k) + f_rv(i,j+1,k) + f_rv(i+1,j+1,k));
          f_xv   = 0.25 * (f_ru(i,j,k) + f_ru(i+1,j,k) + f_ru(i,j+1,k) + f_ru(i+1,j+1,k));
          f_zv   = 0.25 * (f_rw(i,j,k) + f_rw(i+1,j,k) + f_rw(i,j+1,k) + f_rw(i+1,j+1,k));
          f_r    = 0.25 * (f_den(i,j,k) + f_den(i+1,j,k) + f_den(i,j+1,k) + f_den(i+1,j+1,k));
          f_e_av = 0.25 * (f_e(i,j,k) + f_e(i+1,j,k) + f_e(i,j+1,k) + f_e(i+1,j+1,k));
          f_tot  = sqrt(f_xv*f_xv + f_yv*f_yv + f_zv*f_zv + f_r*f_r + f_e_av*f_e_av);

          fprintf(fp,"%8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4e  %8.4e  %8.4e  %8.4e  %8.4e  %8.4e\n",
            xcoord3(i,j,k),ycoord3(i,j,k),zcoord3(i,j,k),p_av,mach,xv,yv,zv,e_av,
            f_tot,f_r,f_xv,f_yv,f_zv,f_e_av);
            xmin = PetscMin(xmin,xcoord3(i,j,k));
            xmax = PetscMax(xmax,xcoord3(i,j,k));
            ymin = PetscMin(ymin,ycoord3(i,j,k));
            ymax = PetscMax(ymax,ycoord3(i,j,k));
            zmin = PetscMin(zmin,zcoord3(i,j,k));
            zmax = PetscMax(zmax,zcoord3(i,j,k));
        }
      }
    }

    fprintf(fp,"\nxmin=%g, xmax=%g, ymin=%g, ymax=%g, zmin=%g, zmax=%g\n",
                xmin, xmax, ymin, ymax, zmin, zmax);
    fclose(fp); 
    if (app->size != 1) {
      ierr = VecRestoreArray(P_natural,&pp); CHKERRQ(ierr);
      ierr = VecRestoreArray(X_natural,&xx); CHKERRQ(ierr);
      ierr = VecRestoreArray(F_natural,&xx); CHKERRQ(ierr);
    } 
  }
  if (app->size != 1) {
    ierr = VecDestroy(P_natural); CHKERRQ(ierr);
    ierr = VecDestroy(X_natural); CHKERRQ(ierr);
    ierr = VecDestroy(F_natural); CHKERRQ(ierr);
  }
  return 0;
}
/* --------------------------------------------------------------- */
/*
   VisualizeEuler_Matlab - Computes the mach contour for the duct problem

   Input Parameters:
   app   - user-defined application context
   x     - solution vector
   iter  - iteration number

 */
int VisualizeEuler_Matlab(int iter,Euler *app,Scalar *x)
{
  int    foo, i, j, k, ni1 = app->ni1, nj1 = app->nj1;
  int    ni = app->ni, nj = app->nj;
  int    kj, ijk, jstart, jend, istart, iend;
  Scalar sfluid, ssound, r, gm1, gamma1, xv, yv;
  Scalar *xc = app->xc, *yc = app->yc, *zc = app->zc;
  FILE   *fp2;
  char   filename[64];

  if (app->mmtype != MMEULER) SETERRQ(1,0,"Unsupported model type");
  if (app->size != 1) SETERRQ(1,0,"Currently uniprocessor only!");

  istart = 0;
  iend   = ni;
  jstart = 0;
  jend   = nj;

  gamma1 = 1.4;
  gm1    = gamma1 - 1.0;
  k      = 1;

#define xcoord(i,j) xc[(k)*nj*ni + (j)*ni + (i)]
#define ycoord(i,j) yc[(k)*nj*ni + (j)*ni + (i)]
#define zcoord(i,j) zc[(k)*nj*ni + (j)*ni + (i)]
#define den(i,j) x[5*((k)*nj1*ni1 + (j)*ni1 + (i))]
#define ru(i,j) x[5*((k)*nj1*ni1 + (j)*ni1 + (i)) + 1]
#define rv(i,j) x[5*((k)*nj1*ni1 + (j)*ni1 + (i)) + 2]

  foo = 1;
  if (foo) {

    /* potential and mach data are associated with different physical points */
    if (iter != -1) {
      sprintf(filename,"duct_euler_%d.m",iter);
      fp2 = fopen(filename,"w");
    } else {
      fp2 = fopen("duct_euler.m","w");
    }
    if (!fp2) SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open output file");

    /* Grid and potential data */

    fprintf(fp2,"X = [\n");
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        fprintf(fp2,"%8.4f ", xcoord(i,j));
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");
    fprintf(fp2,"Y = [\n");
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        fprintf(fp2,"%8.4f ", ycoord(i,j));
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");
    fprintf(fp2,"Z = [\n");
    for (j=jstart; j<jend; j++) {
      kj = k*nj*ni + j*ni;
      for (i=istart; i<iend; i++) {
        ijk = kj + i;
        fprintf(fp2,"%8.4f ",zcoord(i,j));
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");
    fprintf(fp2,"density = [\n");
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        fprintf(fp2,"%8.4f ",0.25 * (den(i,j) + den(i+1,j) + den(i,j+1) + den(i+1,j+1)));
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");
    fprintf(fp2,"velocity_v = [\n");
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        fprintf(fp2,"%8.4f ",0.25 * (rv(i,j) + rv(i+1,j) + rv(i,j+1) + rv(i+1,j+1)));
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");

    /* MULTI_MODEL! TO TEST:  Compute average values */ 
    fprintf(fp2,"mach = [\n");
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        /* r  = den(i,j);
           xv = ru(i,j);
           yv = rv(i,j); */
        yv = 0.25 * (rv(i,j) + rv(i+1,j) + rv(i,j+1) + rv(i+1,j+1));
        xv = 0.25 * (ru(i,j) + ru(i+1,j) + ru(i,j+1) + ru(i+1,j+1));
        r  = 0.25 * (den(i,j) + den(i+1,j) + den(i,j+1) + den(i+1,j+1));
        sfluid = sqrt(xv*xv + yv*yv) / r;
        ssound = sqrt(pow(r,gm1));
        fprintf(fp2,"%8.4f ",sfluid/ssound);
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");
    fclose(fp2);
  }

  return 0;
}
/* --------------------------------------------------------------- */
/*
   VisualizeFP_Matlab - Computes the mach contour for the duct problem

   Input Parameters:
   app   - user-defined application context
   x     - solution vector
   iter  - iteration number

 */
int VisualizeFP_Matlab(int iter,Euler *app,Scalar *x)
{
  int    foo, i, j, k, ni1 = app->ni1, nj1 = app->nj1;
  int    ni = app->ni, nj = app->nj;
  int    kj, ijk, jstart, jend, istart, iend;
  Scalar sfluid, ssound, r, gm1, gamma1, xv, yv;
  Scalar *xc = app->xc, *yc = app->yc, *zc = app->zc;
  FILE   *fp2;
  char   filename[64];

  if (app->mmtype != MMFP) SETERRQ(1,0,"Unsupported model type");
  if (app->size != 1) SETERRQ(1,0,"Currently uniprocessor only!");

  istart = 0;
  iend   = ni;
  jstart = 0;
  jend   = nj;

  gamma1 = 1.4;
  gm1    = gamma1 - 1.0;
  k      = 1;

#define xcoord(i,j) xc[(k)*nj*ni + (j)*ni + (i)]
#define ycoord(i,j) yc[(k)*nj*ni + (j)*ni + (i)]
#define zcoord(i,j) zc[(k)*nj*ni + (j)*ni + (i)]
#define pot(i,j) x[(k)*nj1*ni1 + (j)*ni1 + (i)]

  foo = 1;
  if (foo) {

    /* potential and mach data are associated with different physical points */
    if (iter != -1) {
      sprintf(filename,"duct_fp_%d.m",iter);
      fp2 = fopen(filename,"w");
    } else {
      fp2 = fopen("duct_fp.m","w");
    }
    if (!fp2) SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open output file");

    /* Grid and potential data */

    fprintf(fp2,"X = [\n");
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        fprintf(fp2,"%8.4f ", xcoord(i,j));
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");
    fprintf(fp2,"Y = [\n");
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        fprintf(fp2,"%8.4f ", ycoord(i,j));
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");
    fprintf(fp2,"Z = [\n");
    for (j=jstart; j<jend; j++) {
      kj = k*nj*ni + j*ni;
      for (i=istart; i<iend; i++) {
        ijk = kj + i;
        fprintf(fp2,"%8.4f ",zcoord(i,j));
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");
    fprintf(fp2,"potential = [\n");
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        fprintf(fp2,"%8.4f ",0.25 * (pot(i,j) + pot(i+1,j) + pot(i,j+1) + pot(i+1,j+1)));
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");

    fprintf(fp2,"mach = [\n");
    for (j=jstart; j<jend; j++) {
      for (i=istart; i<iend; i++) {
        ijk = k*nj1*ni1 + j*ni1 + i;
        r  = app->den_a[ijk];
        xv = app->xvel_a[ijk];
        yv = app->yvel_a[ijk];
        sfluid = sqrt(xv*xv + yv*yv);
        ssound = sqrt(pow(r,gm1));
        fprintf(fp2,"%8.4f ",sfluid/ssound);
      }
      fprintf(fp2,"\n");
    }
    fprintf(fp2,"];\n\n");
    fclose(fp2);
  }

  return 0;
}
/* --------------------------------------------------------------- */
/*
   ComputeMach - Computes the mach contour on the wing surface.

   Input Parameters:
   app   - user-defined application context
   x     - solution vector

   Output Parameter:
   smach - mach number on wing surface
 */
int ComputeMach(Euler *app,Scalar *x,Scalar *smach)
{
  int    i, j, k, ijkx, ijkxi, ni1 = app->ni1, nj1 = app->nj1;
  int    kstart = 0, kend = app->ktip+1, istart = app->itl, iend = app->itu+1;
  Scalar sfluid, ssound, r, ru, rw, gm1, gamma1;

  kstart = 0;
  kend   = app->ktip+1;
  istart = app->itl;
  iend   = app->itu+1;

  /* temporarily just use grid boundaries even though we only care about values on the surface */
  istart = app->xs;
  iend   = app->xe;
  kstart = app->zs;
  kend   = app->ze;

  gamma1 = 1.4;
  gm1   = gamma1 - 1.0;
  j     = 0;  /* wing surface is j=0 */
  for (k=kstart; k<kend; k++) {
    for (i=istart; i<iend; i++) {
      ijkx  = k*nj1*ni1 + j*ni1 + i;
      ijkxi = ijkx * 5;
      r  = app->xx[ijkxi];
      ru = app->xx[ijkxi+1];
      rw = app->xx[ijkxi+3];
      sfluid = sqrt(ru*ru + rw*rw) / r;
      ssound = sqrt(pow(r,gm1));
      smach[ijkx] = sfluid/ssound;
    }
  }
  return 0;
}

#undef __FUNC__
#define __FUNC__ "ComputeNodalResiduals"
/* ----------------------------------------------------------------------------- */
/*
   ComputeNodalResiduals - Computes nodal residuals (sum of absolute value of
   all residual components at each grid point).  Eventually we should provide 
   additional residual output options.
 */
int ComputeNodalResiduals(Euler *app,Vec X,Vec Xsum)
{
  int    i, j, k, jkx, ijkx, ierr, ijkxt, ndof = app->ndof;
  int    xs = app->xs, ys = app->ys, zs = app->zs;
  int    xe = app->xe, ye = app->ye, ze = app->ze;
  int    xm = app->xm, ym = app->ym;
  Scalar *xa, *xasum;

  ierr = VecGetArray(X,&xa); CHKERRQ(ierr);
  ierr = VecGetArray(Xsum,&xasum); CHKERRQ(ierr);
  for (k=zs; k<ze; k++) {
    for (j=ys; j<ye; j++) {
      jkx = (j-ys)*xm + (k-zs)*xm*ym;
      for (i=xs; i<xe; i++) {
        ijkx   = jkx + i-xs;
        ijkxt  = ndof * ijkx;
        xasum[ijkx] = PetscAbsScalar(xa[ijkxt]) + PetscAbsScalar(xa[ijkxt+1])
                      + PetscAbsScalar(xa[ijkxt+2]) + PetscAbsScalar(xa[ijkxt+3])
                      + PetscAbsScalar(xa[ijkxt+4]);
      }
    }
  }
  ierr = VecRestoreArray(X,&xa); CHKERRQ(ierr);
  ierr = VecRestoreArray(Xsum,&xasum); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------------------ */
#include "src/snes/snesimpl.h"
#undef __FUNC__
#define __FUNC__ "ConvergenceTestEuler"
/*
   ConvergenceTestEuler - We define a convergence test for the Euler code
   that stops only for the following:
      - the function norm satisfies the specified relative decrease
      - stagnation has been detected
      - we're encountering NaNs

   Notes:
   We use this simplistic test because we need to compare timings for
   various methods, and we need a single stopping criterion so that a
   fair comparison is possible.

   We test for stagnation and NaNs only for the implicit BC  versions,
   since these haven't been a problem with explicit BCs.
 */
int ConvergenceTestEuler(SNES snes,double xnorm,double pnorm,double fnorm,void *dummy)
{
  Euler  *app = (Euler *)dummy;
  int    i, last_k, iter = snes->iter, fstagnate = 0;
  Scalar *favg = app->favg, *farray = app->farray;
  Scalar register tmp;

  if (fnorm <= snes->ttol) {
    PLogInfo(snes,
    "ConvergenceTestEuler:Converged due to function norm %g < %g (relative tolerance)\n",fnorm,snes->ttol);
    return 1;
  }
  /* Test for stagnation and NaNs for implicit bcs only */
  if (app->bctype != EXPLICIT) {
    /* Note that NaN != NaN */
    if (fnorm != fnorm) {
      PLogInfo(snes,"ConvergenceTestEuler:Function norm is NaN: %g\n",fnorm);
      return 2;
    }
    if (iter >= 990) {
      /* Compute average fnorm over the past 6 iterations */
      last_k = 5;
      tmp = 0.0;
      for (i=iter-last_k; i<iter+1; i++) tmp += farray[i];
      favg[iter] = tmp/(last_k+1);
      /* printf("   iter = %d, f_avg = %g \n",iter,favg[iter]); */
  
      /* Test for stagnation over the past 10 iterations */
      if (iter >=3000) {
        last_k = 10;
        for (i=iter-last_k; i<iter; i++) {
          if (PetscAbsScalar(favg[i] - favg[iter])/favg[iter] < app->fstagnate_ratio) fstagnate++;
          /* printf("iter = %d, i=%d, ratio = %g, fstg_ratio=%g, fstagnate = %d\n",
              iter,i,ratio,app->fstagnate_ratio,fstagnate); */
        }
        if (fstagnate > 5) {
          PLogInfo(snes,"ConvergenceTestEuler: Stagnation at fnorm = %g\n",fnorm);
          return 3;
        }
      }
    }
  }
  return 0;
}

