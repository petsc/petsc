/*
   This file contains various monitoring routines used by the SNES/Julianne code.
 */
#include "user.h"

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
  Euler    *app = (Euler *)dummy;
  Scalar   negone = -1.0, mfeps;
  Vec      DX;
  Viewer   view1;
  char     filename[64];
  int      ierr, lits;

  PetscObjectGetComm((PetscObject)snes,&comm);

  if (app->matrix_free && app->mf_adaptive) {
    mfeps = fnorm * 1.e-3;
    ierr = UserSetMatrixFreeParameters(snes,mfeps,PETSC_DEFAULT); CHKERRQ(ierr);
    if (!app->no_output) PetscPrintf(comm,"next mf_eps=%g\n",mfeps);
  }
  /* Print the vector F (intended for debugging) */
  if (app->print_vecs) {
    ierr = SNESGetFunction(snes,&app->F); CHKERRQ(ierr);
    sprintf(filename,"res.%d.out",its);
    ierr = ViewerFileOpenASCII(app->comm,filename,&view1); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view1,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = DFVecView(app->F,view1); CHKERRQ(ierr);
    ierr = ViewerDestroy(view1); CHKERRQ(ierr);
  }

  app->flog[its]  = log10(fnorm);
  app->fcfl[its]  = app->cfl; 
  app->ftime[its] = PetscGetTime() - app->time_init;
  if (!its) {
    /* Do the following only during the initial call to this routine */
    app->fnorm_init  = app->fnorm_last = fnorm;
    app->cfl_init    = app->cfl;
    app->lin_its[0]  = 0;
    app->lin_rtol[0] = 0;
    if (!app->no_output) {
      if (app->cfl_advance != CONSTANT)
        PetscPrintf(comm,"iter = %d, Function norm = %g, fnorm reduction ratio = %g, CFL_init = %g\n",
           its,fnorm,app->f_reduction,app->cfl);
      else PetscPrintf(comm,"iter = %d, Function norm = %g, CFL = %g\n",
           its,fnorm,app->f_reduction,app->cfl);
      if (app->rank == 0) {
        app->fp = fopen("fnorm.m","w"); 
        fprintf(app->fp,"zsnes = [\n");
        fprintf(app->fp,"  %d    %8.4f   %12.1f   %10.2f    %d     %g\n",
                its,app->flog[its],app->fcfl[its],app->ftime[its],app->lin_its[its],app->lin_rtol[its]);
      }
    }
  } else {
    /* For the first iteration and onward we do the following */

    /* Get some statistics about the iterative solver */
    ierr = SNESGetNumberLinearIterations(snes,&lits); CHKERRQ(ierr);
    app->lin_its[its] = lits - app->last_its;
    app->last_its     = lits;
    ierr = KSPGetTolerances(app->ksp,&(app->lin_rtol[its]),PETSC_NULL,PETSC_NULL,
           PETSC_NULL); CHKERRQ(ierr);
    if (!app->no_output) {
      PetscPrintf(comm,"iter = %d, Function norm %g, lin_rtol=%g, lin_its = %d\n",
                  its,fnorm,app->lin_rtol[its],app->lin_its[its]);
      if (app->rank == 0) {
        fprintf(app->fp,"  %d    %8.4f   %12.1f   %10.2f    %d     %g\n",
                its,app->flog[its],app->fcfl[its],app->ftime[its],app->lin_its[its],app->lin_rtol[its]);
        fflush(app->fp);
      }
    }

    /* Compute new CFL number if desired */
    /* Note: BCs change at iter 10, so we defer CFL increase until after this point */
    if (app->cfl_advance != CONSTANT && its > 11) {
      /* Check to see if last step was OK ... do we want to increase CFL and DT now? */

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
        /* Modify the CFL if we are past the threshold ratio */
        if (app->cfl_advance == ADVANCE_GLOBAL) {
          app->cfl = app->cfl * app->fnorm_last / fnorm;
          app->fnorm_last = fnorm;
        } else if (app->cfl_advance == ADVANCE_LOCAL) {
          app->cfl = app->cfl_init * app->fnorm_init / fnorm;
        } else SETERRQ(1,1,"Unsupported CFL advancement strategy");
        app->cfl = PetscMin(app->cfl,app->cfl_max);
        app->cfl = PetscMax(app->cfl,app->cfl_init);
        if (!app->no_output) PetscPrintf(comm,"CFL: cfl=%g\n",app->cfl);
      }
    }

    /* Calculate new pseudo-transient continuation term, dt */
    /*    if (app->sctype == DT_MULT || next iteration forms Jacobian || matrix-free mult) */
    eigenv_(app->dt,app->r,app->ru,app->rv,app->rw,app->e,app->p,
         app->sadai,app->sadaj,app->sadak,
         app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
         app->aiz,app->ajz,app->akz);

    /* Extract solution and update vectors; convert to Julianne format */
    ierr = SNESGetSolutionUpdate(snes,&DX); CHKERRQ(ierr);
    ierr = VecScale(&negone,DX); CHKERRQ(ierr);
    ierr = PackWork(app,DX,app->localDX,
                    app->dr,app->dru,app->drv,app->drw,app->de); CHKERRQ(ierr);

    /* Call Julianne monitoring routine */
    ierr = jmonitor_(&app->flog[its],&app->cfl,
             app->work_p,app->r,app->ru,app->rv,app->rw,app->e,
             app->p,app->dr,app->dru,app->drv,app->drw,app->de,
             app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
             app->aiz,app->ajz,app->akz); CHKERRQ(ierr);

    /* Print factored matrix - intended for debugging */
    if (app->print_vecs) {
      SLES   sles;
      PC     pc;
      PCType pctype;
      Mat    fmat;
      Viewer view;
      ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
      ierr = SLESGetPC(sles,&pc); CHKERRQ(ierr);
      ierr = PCGetType(pc,&pctype,PETSC_NULL); CHKERRQ(ierr);
      if (pctype == PCILU) {
        ierr = PCGetFactoredMatrix(pc,&fmat);
        ierr = ViewerFileOpenASCII(app->comm,"factor.out",&view); CHKERRQ(ierr);
        ierr = ViewerSetFormat(view,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
        ierr = MatView(fmat,view); CHKERRQ(ierr);
        ierr = ViewerDestroy(view); CHKERRQ(ierr);
      }
    }
  }
  app->iter = its+1;
  return 0;
}
#undef __FUNC__
#define __FUNC__ "MonitorDumpGeneral"
/* --------------------------------------------------------------- */
/* 
   MonitorDumpGeneral - Dumps solution fields for later use in viewers.

   Input Parameters:
   snes - nonlinear solver context
   X    - current iterate
   app - user-defined application context
 */
int MonitorDumpGeneral(SNES snes,Vec X,Euler *app)
{
  FILE     *fp;
  int      ierr, i, j, k, ijkx, ijkcx, iter, ni, nj, nk, ni1, nj1, nk1;
  char     filename[64];

  /* Since we call MonitorDumpGeneral() from the routine ComputeFunction(), packing and
     computing the pressure have already been done. */
  /*
  ierr = PackWork(app,X,app->localX,
                  app->r,app->ru,app->rv,app->rw,app->e); CHKERRQ(ierr);
  ierr = jpressure_(app->r,app->ru,app->rv,app->rw,app->e,app->p); CHKERRQ(ierr);
  */

  if (app->size != 1) SETERRQ(1,1,"Currently supports uniprocessor use only!")
  ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
  sprintf(filename,"euler.%d.out",iter);
  fp = fopen(filename,"w"); 
  fprintf(fp,"VARIABLES=x,y,z,ru,rv,rw,r,e,p\n");
  ni  = app->ni;  nj  = app->nj;  nk = app->nk;
  ni1 = app->ni1; nj1 = app->nj1; nk1 = app->nk1;
  for (k=0; k<nk; k++) {
    for (j=0; j<nj; j++) {
      for (i=0; i<ni; i++) {
        ijkx  = k*nj1*ni1 + j*ni1 + i;
        ijkcx = k*nj*ni + j*ni + i;
        fprintf(fp,"%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\n",
          app->xc[ijkcx],app->yc[ijkcx],app->zc[ijkcx],app->ru[ijkx],app->rv[ijkx],
          app->rw[ijkx],app->r[ijkx],app->e[ijkx],app->p[ijkx]);
      }
    }
  }
  fclose(fp);
  return 0;
}
#undef __FUNC__
#define __FUNC__ "MonitorDumpGeneralJulianne"
/* --------------------------------------------------------------- */
/* 
   MonitorDumpGeneralJulianne - Dumps solution fields for later use in viewers;
   intended for use with original Julianne solver.

   Input Parameter:
   app - user-defined application context
 */
int MonitorDumpGeneralJulianne(Euler *app)
{
  FILE     *fp;
  int      i, j, k, ijkx, ijkcx, ni, nj, nk, ni1, nj1, nk1;
  char     filename[64];

  sprintf(filename,"julianne.out");
  fp = fopen(filename,"w"); 
  fprintf(fp,"VARIABLES=x,y,z,ru,rv,rw,r,e,p\n");
  ni  = app->ni;  nj  = app->nj;  nk = app->nk;
  ni1 = app->ni1; nj1 = app->nj1; nk1 = app->nk1;
  for (k=0; k<nk; k++) {
    for (j=0; j<nj; j++) {
      for (i=0; i<ni; i++) {
        ijkx  = k*nj1*ni1 + j*ni1 + i;
        ijkcx = k*nj*ni + j*ni + i;
        fprintf(fp,"%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\n",
          app->xc[ijkcx],app->yc[ijkcx],app->zc[ijkcx],app->ru[ijkx],app->rv[ijkx],
          app->rw[ijkx],app->r[ijkx],app->e[ijkx],app->p[ijkx]);
      }
    }
  }
  fclose(fp);
  return 0;
}
/* --------------------------------------------------------------------------- */

extern int DFVecFormUniVec_MPIRegular_Private(DFVec,Vec*);
#undef __FUNC__
#define __FUNC__ "MonitorDumpVRML"
/* 
   MonitorDumpVRML - Outputs fields for use in VRML viewers.  The default
   output is the pressure field.  In addition, the residual field can be
   dumped also.

   Input Parameters:
   snes - nonlinear solver context
   X    - current iterate
   F    - current residual vector
   app - user-defined application context
 */
int MonitorDumpVRML(SNES snes,Vec X,Vec F,Euler *app)
{
  MPI_Comm      comm;
  int           ierr, iter;
  char          filename[64];
  Scalar        *field;
  int           different_files;         /* flag indicating use of different output files for
                                            various iterations */
  Vec           P_uni;                   /* work vector for pressure field */
  Draw          Win;                     /* VRML drawing context */

  PetscObjectGetComm((PetscObject)snes,&comm);
  ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-dump_vrml_different_files",&different_files); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        output pressure field
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (app->dump_vrml_pressure) {

    /* Since we call MonitorDumpVRML() from the routine ComputeFunction(), we've already
       computed the pressure ... so there's no need for the following 2 statements.
    ierr = PackWork(app,X,app->localX,
                    app->r,app->ru,app->rv,app->rw,app->e); CHKERRQ(ierr);
    ierr = jpressure_(app->r,app->ru,app->rv,app->rw,app->e,app->p); CHKERRQ(ierr);
    */

    /* If using multiple processors, then assemble the pressure vector on only 1 processor
       (in the appropriate ordering) and then view it.  Eventually, we will optimize such
       manipulations and hide them in the viewer routines */
    if (app->size == 1) {
      field = app->p;
    } 
    else {
      /* Pack pressure vector */
      ierr = UnpackWorkComponent(app,app->p,app->P); CHKERRQ(ierr);
      ierr = DFVecFormUniVec_MPIRegular_Private(app->P,&P_uni); CHKERRQ(ierr);
      if (app->rank == 0) {ierr = VecGetArray(P_uni,&field); CHKERRQ(ierr);}
    }

    /* Dump VRML images from first processor only */
    if (app->rank == 0) {
      if (different_files) {
        /* Dump all output into different files for later viewing */
        sprintf(filename,"pressure.%d.1.wrl",iter);
      } else {
        /* Dump all output into the same file for continual VRML viewer updates */
        sprintf(filename,"pressure.1.wrl");
      }

      ierr = DrawOpenVRML(MPI_COMM_SELF,filename,"Whitfield pressure field",&Win); CHKERRQ(ierr);
      ierr = DumpField(app,Win,field); CHKERRQ(ierr);
      ierr = DrawDestroy(Win); CHKERRQ(ierr);

      if (app->size != 1) {
        ierr = VecRestoreArray(P_uni,&field); CHKERRQ(ierr);
        ierr = VecDestroy(P_uni); CHKERRQ(ierr);
      }
      /*
       * Now write out a zero-length file that the petsc gw will use for
       * seeing that the file is updated (avoid the send-incomplete-vrml
       * problem.
       *
       * Note from Lois: I moved this inside the processor rank=0 section,
       *                 since we currently only define the filename here.
       */
      {
        FILE *fp;
        char buf[1000];
        sprintf(buf, "%s.ts", filename);
        fp = fopen(buf, "w");
        fprintf(fp, "%d\n", iter);
        fclose(fp);
      }
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        output residual field (sum of absolute value of 
        the 5 residual components at each grid point)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (app->dump_vrml_residual) {
    if (!app->Fvrml) {ierr = VecDuplicate(app->P,&app->Fvrml); CHKERRQ(ierr);}
    ierr = ComputeNodalResiduals(app,F,app->Fvrml); CHKERRQ(ierr);

    /* If using multiple processors, then assemble the nodal residual vector
       on only 1 processor (in the appropriate ordering) and then view it.
       Eventually, we will optimize such manipulations and hide them in the
       viewer routines */
    if (app->size == 1) {
      ierr = VecGetArray(app->Fvrml,&field); CHKERRQ(ierr);
    } 
    else {
      ierr = DFVecFormUniVec_MPIRegular_Private(app->Fvrml,&P_uni); CHKERRQ(ierr);
      if (app->rank == 0) {ierr = VecGetArray(P_uni,&field); CHKERRQ(ierr);}
    }

    /* Dump VRML images from first processor only */
    if (app->rank == 0) {
      if (different_files) {
        /* Dump all output into different files for later viewing */
        sprintf(filename,"residual.%d.1.wrl",iter);
      } else {
        /* Dump all output into the same file for continual VRML viewer updates */
        sprintf(filename,"residual.1.wrl");
      }

      ierr = DrawOpenVRML(MPI_COMM_SELF,filename,"Whitfield residual sums",&Win); CHKERRQ(ierr);
      ierr = DumpField(app,Win,field); CHKERRQ(ierr);
      ierr = DrawDestroy(Win); CHKERRQ(ierr);

      if (app->size != 1) {
        ierr = VecRestoreArray(P_uni,&field); CHKERRQ(ierr);
        ierr = VecDestroy(P_uni); CHKERRQ(ierr);
      }
      /*
       * Now write out a zero-length file that the petsc gw will use for
       * seeing that the file is updated (avoid the send-incomplete-vrml
       * problem.
       *
       * Note from Lois: I moved this inside the processor rank=0 section,
       *                 since we currently only define the filename here.
       */
      {
        FILE *fp;
        char buf[1000];
        sprintf(buf, "%s.ts", filename);
        fp = fopen(buf, "w");
        fprintf(fp, "%d\n", iter);
        fclose(fp);
      }
    }
  }

  return 0;
}
#undef __FUNC__
#define __FUNC__ "DumpField"
/* --------------------------------------------------------------- */
/*
    DumpField - Dumps a field to VRML viewer.  Since the VRML routines are
    all currently uniprocessor only, DumpField() should be called by just
    1 processor, with the complete scalar field over the global domain.
    Eventually, we'll upgrade this for better use in parallel.
 */
int DumpField(Euler *app,Draw Win,Scalar *field)
{
  DrawMesh       mesh;                    /* mesh for VRML viewing */
  VRMLGetHue_fcn color_fcn;               /* color function */
  void           (*huedestroy)( void * ); /* routine for destroying hues */
  void           *hue_ctx;                /* hue context */
  int            evenhue = 0;             /* flag - indicating even hues */
  int            coord_dim;               /* dimension for slicing VRML output */
  int            ycut = 0;                /* cut VRML output in y-planes */
  int            layers;                  /* number of data layers to output */
  int            coord_slice;             /* current coordinate plane slice */
  int            flg, ierr, j, k;
  int            ni = app->ni, nj = app->nj, nk = app->nk;
  Scalar         *x = app->xc, *y = app->yc, *z = app->zc;

  ierr = DrawMeshCreateSimple( &mesh, x, y, z, ni, nj, nk, 1, field, 32 ); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-vrmlevenhue",&evenhue); CHKERRQ(ierr);
  if (evenhue) {
    hue_ctx = VRMLFindHue_setup( mesh, 32 );
    color_fcn = VRMLFindHue;
    huedestroy = VRMLFindHue_destroy;
  }
  else {
    hue_ctx = VRMLGetHue_setup( mesh, 32 );
    color_fcn = VRMLGetHue;
    huedestroy = VRMLGetHue_destroy;
  }
  ierr = OptionsHasName(PETSC_NULL,"-dump_vrml_cut_y",&ycut); CHKERRQ(ierr);
  layers = nk;
  ierr = OptionsGetInt(PETSC_NULL,"-dump_vrml_layers",&layers,&flg); CHKERRQ(ierr);

  if (!ycut) {   /* Dump data, striped by planes in the z-direction */
    layers = PetscMin(layers,nk);
    coord_dim = 2;
    for (k=0; k<layers; k+=1) {
      coord_slice = k;
      ierr = DrawTensorMapSurfaceContour( Win, mesh, 
                           0.0, 0.0, k * 4.0, 
			   coord_slice, coord_dim, 
			   color_fcn, hue_ctx, 32, 0.5 ); CHKERRQ(ierr);
      ierr = DrawTensorMapMesh( Win, mesh, 0.0, 0.0, k * 4.0,
                           coord_slice, coord_dim ); CHKERRQ(ierr);
    }
  }
  else {   /* Dump data, striped by planes in the y-direction */
    coord_dim = 1;
    layers = PetscMin(layers,nj);
    for (j=0; j<layers; j+=1) {
      coord_slice = j;
      ierr = DrawTensorMapSurfaceContour( Win, mesh, 
                           0.0, 0.0, 0.0, 
	                   coord_slice, coord_dim, 
			   color_fcn, hue_ctx, 32, 0.5 ); CHKERRQ(ierr);
      ierr = DrawTensorMapMesh( Win, mesh, 0.0, 0.0, 0.0,
			   coord_slice, coord_dim ); CHKERRQ(ierr);
    }
  }
  (*huedestroy)( hue_ctx );
  ierr = DrawMeshDestroy(&mesh); CHKERRQ(ierr);
  ierr = DrawSyncFlush(Win); CHKERRQ(ierr);

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
  int    i, j, k, jkx, ijkx, ierr, ijkxt, nc = app->nc;
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
        ijkxt  = nc * ijkx;
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
#undef __FUNC__
#define __FUNC__ "TECPLOTMonitor"
/* ------------------------------------------------------------------------------ */
/* 
   TECPLOTMonitor - Monitoring routine for nonlinear solver.
 */
int TECPLOTMonitor(SNES snes,Vec X,Euler *app)
{
  MPI_Comm comm;
  FILE     *fp;
  int      ierr, i, j, k, ik, ijkx, ikc, ijkcx;
  int      gxs = app->gxs, gys = app->gys, gzs = app->gzs;
  int      gxm = app->gxm, gym = app->gym;
  char     filename[64];

  PetscObjectGetComm((PetscObject)snes,&comm);
  ierr = PackWork(app,X,app->localX,
                  app->r,app->ru,app->rv,app->rw,app->e); CHKERRQ(ierr);
  /* Compute pressures */
  ierr = jpressure_(app->r,app->ru,app->rv,app->rw,app->e,app->p); CHKERRQ(ierr);

  for (k=0; k<app->nk; k++) {
    sprintf(filename,"plot.%d.out",k);
    fp = fopen(filename,"w"); 
    fprintf(fp,"VARIABLES=x,y,r,ru,rv,p\n");
    fprintf(fp,"ZONE T=onr, I=%d, J=%d, F=POINT\n",app->nj,app->ni);
    for (i=0; i<app->ni; i++) {
      ik  = (k-gzs)*gxm*gym + i-gxs;
      ikc = k*app->nj*app->ni + i;
      for (j=0; j<app->nj; j++) {
        ijkx  = ik + (j-gys)*gxm;
        ijkcx = ikc + j*app->ni;
        fprintf(fp,"%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\n",
          app->xc[ijkcx],app->yc[ijkcx],app->r[ijkx],app->ru[ijkx],app->rv[ijkx],app->p[ijkx]);
      }
    }
    fclose(fp);
  }
  
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

   This is a simplistic test that we use only because we need to
   compare timings for various methods, and we need a single stopping
   criterion so that a fair comparison is possible.

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
  /* Note that NaN != NaN */
  if (fnorm != fnorm) {
    PLogInfo(snes,"ConvergenceTestEuler:Function norm is NaN: %g\n",fnorm);
    return 2;
  }
  if (iter >= 40) {
    /* Computer average fnorm over the past 6 iterations */
    last_k = 5;
    tmp = 0.0;
    for (i=iter-last_k; i<iter+1; i++) tmp += farray[i];
    favg[iter] = tmp/(last_k+1);
    /* printf("   iter = %d, f_avg = %g \n",iter,favg[iter]); */

    /* Test for stagnation over the past 10 iterations */
    if (iter >=50) {
      last_k = 10;
      for (i=iter-last_k; i<iter; i++) {
        if (PetscAbsScalar(favg[i] - favg[iter])/favg[iter] < app->fstagnate_ratio) fstagnate++;
        /* printf("iter = %d, i=%d, ratio = %g, fstg_ratio=%g, fstagnate = %d\n",iter,i,ratio,app->fstagnate_ratio,fstagnate); */
      }
      if (fstagnate > 5) {
        PLogInfo(snes,"ConvergenceTestEuler: Stagnation at fnorm = %g\n",fnorm);
        return 3;
      }
    }
  }
  return 0;
}

/* ------------------------------------------------------------------------------ */
#include "src/ksp/impls/gmres/gmresp.h"
/* 
   UserConvergenceTestGMRES - 
 */
#undef __FUNC__  
#define __FUNC__ "UserConvergenceTest_GMRES"
/* Nothing for now ... */
/*
int KSPDefaultConverged_GMRES(KSP ksp,int n,double rnorm,void *ptr)
{
  Euler *app = (Euler *)ptr;
  printf("iter = %d\n",n);
  if (!n) app->rinit = rnorm;
  if ( rnorm <= ksp->ttol ) {
    app->lin_rtol[app->iter] = 
    return(1);
  }
  else return(0);
}
*/
