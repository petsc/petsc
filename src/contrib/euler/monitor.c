/*
   This file contains various monitoring routines used by the SNES/Julianne code.
 */
#include "user.h"

#undef __FUNC__
#define __FUNC__ "JulianneMonitor"
/* 
   JulianneMonitor - Routine that is called at the conclusion
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
int JulianneMonitor(SNES snes,int its,double fnorm,void *dummy)
{
  MPI_Comm comm;
  Euler  *app = (Euler *)dummy;
  Scalar   negone = -1.0, ratio;
  Vec      DX;
  int      ierr;
  Viewer   view1;
  char     filename[64];

  PetscObjectGetComm((PetscObject)snes,&comm);
  if (!app->no_output) PetscPrintf(comm,"iter = %d, Function norm %g\n",its,fnorm);

  /* Print the vector F (intended for debugging) */
  if (app->print_vecs) {
    ierr = SNESGetFunction(snes,&app->F); CHKERRQ(ierr);
    sprintf(filename,"res.%d.out",its);
    ierr = ViewerFileOpenASCII(app->comm,filename,&view1); CHKERRQ(ierr);
    ierr = ViewerSetFormat(view1,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
    ierr = DFVecView(app->F,view1); CHKERRQ(ierr);
    ierr = ViewerDestroy(view1); CHKERRQ(ierr);
  }

  if (its) {
    /* Compute new CFL number */
    if (app->cfl_advance) {
      if (fnorm/app->fnorm0 <= app->f_reduction) {
        ratio = app->fnorm_last / fnorm;
        if (ratio < 1.0 || app->cfl <= app->cfl_max) {
          app->cfl = PetscMin(app->cfl * ratio,app->cfl_max);
          if (!app->no_output) PetscPrintf(comm,"New CFL: ratio=%g, cfl=%g\n",ratio,app->cfl);
        } else 
          if (!app->no_output) PetscPrintf(comm,"Same CFL: cfl = %g, cfl_max = %g\n",app->cfl,app->cfl_max);
      } else {
        if (!app->no_output) PetscPrintf(comm,"Same CFL: fnorm/fnorm0 = %g, f_reduction ratio = %g, cfl = %g\n",
          fnorm/app->fnorm0,app->f_reduction,app->cfl);
      }
      app->fnorm_last = fnorm;
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
    ierr = jmonitor_(&fnorm,&app->cfl,app->work_p,app->r,app->ru,app->rv,app->rw,app->e,
             app->p,app->dr,app->dru,app->drv,app->drw,app->de,
             app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
             app->aiz,app->ajz,app->akz); CHKERRQ(ierr);
  } else {
    app->fnorm0 = app->fnorm_last = fnorm;
    if (!app->no_output) {
      if (app->cfl_advance) PetscPrintf(comm,"fnorm0 = %g, fnorm reduction = %g\n",
                             app->fnorm0,app->f_reduction);
      else                   PetscPrintf(comm,"Not advancing CFL.\n");
    }
  }

  /* Print factored matrix - intended for debugging */
  if (its && app->print_vecs) {
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

  if (app->size != 1) SETERRQ(1,1,"MonitorDumpGeneral: Currently supports uniprocessor use only!")
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
#define __FUNC__ "JulianneConvergenceTest"
/*
   JulianneConvergenceTest - We define a convergence test for the Euler code
   that stops only for the following:
      - the function norm satisfies the specified relative decrease
      - stagnation has been detected
      - we're encountering NaNs

   This is a simplistic test that we use only because we need to
   compare timings for various methods, and we need a single stopping
   criterion so that a fair comparison is possible.

 */
int JulianneConvergenceTest(SNES snes,double xnorm,double pnorm,double fnorm,void *dummy)
{
  Euler  *app = (Euler *)dummy;
  int    i, last_k, iter = snes->iter, fstagnate = 0;
  Scalar *favg = app->favg, *farray = app->farray, ratio;
  Scalar register tmp;
  if (fnorm <= snes->ttol) {
    PLogInfo(snes,
    "JulianneConvergenceTest:Converged due to function norm %g < %g (relative tolerance)\n",fnorm,snes->ttol);
    return 1;
  }
  /* Note that NaN != NaN */
  if (fnorm != fnorm) {
    PLogInfo(snes,"JulianneConvergenceTest:Function norm is NaN: %g\n",fnorm);
    return 2;
  }
  if (iter >= 6) {
    /* Computer average fnorm over the past 6 iterations */
    last_k = 5;
    tmp = 0.0;
    for (i=iter-last_k; i<iter+1; i++) {
      tmp += farray[i];
      printf("   iter = %d, i=%d, farray = %g, tmp=%g \n",iter,i,farray[i],tmp);
    }
    favg[iter] = tmp/(last_k+1);
    printf("   iter = %d, f_avg = %g \n",iter,favg[iter]);

    /* Test for stagnation over the past 10 iterations */
    if (iter >= 16) {
      last_k = 10;
      for (i=iter-last_k; i<iter; i++) {
        ratio=PetscAbsScalar(favg[i] - favg[iter])/favg[iter];
        if (ratio < app->fstagnate_ratio) fstagnate++;
        printf("iter = %d, i=%d, ratio = %g, fstg_ratio=%g, fstagnate = %d\n",iter,i,ratio,app->fstagnate_ratio,fstagnate);
      }
      if (fstagnate == last_k) {
        PLogInfo(snes,"JulianneConvergenceTest: Stagnation at fnorm = %g\n",fnorm);
        return 3;
      }
    }
  }
  return 0;
}
