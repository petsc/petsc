/*
   This file contains the routines that control the vector scatters that
   are used to apply certain boundary conditions.  In particular, these
   scatters are needed to communicate data when imposing continuity for
   the C-grid (part of the j=0 boundary).
 */
#include "user.h"

#undef __FUNC__
#define __FUNC__ "BCScatterSetUp"
/* ----------------------------------------------------------------------- */
/*
   BCScatterSetUp - Sets up scatter contexts for certain boundary conditions.

   Input Parameter:
   app - application-defined context

  ----------------
   General notes:
  ----------------
   Vector scatters such as these are an efficient way to handle communication
   of parallel data.  In particular, they enable the application to precompute
   communication patterns and then reuse this information repeatedly in
   during an application.
     
  -----------------------
   Notes for Euler code:
  -----------------------
   The vector scatter contexts are set a single time here and then are
   are reused repeatedly in either BoundaryConditionsExplicit() or
   BoundaryConditionsImplicit(), depending on whether we use explicit 
   or implicit treatment of BC's.

   The scatters are used to communicate data when imposing continuity for
   the C-grid (part of the j=0 boundary).  The original Fortran implementation
   of these boundary conditions (taken from the Julianne code) is as follows:

      DO 2 K=2,NK
      DO 2 I=2,NI
      IF (K.GT.KTIP) GO TO 4
      IF (I.LE.ITL.OR.I.GT.ITU) GO TO 4
      .....
    4 CONTINUE
      ID=NI+1-I
      R(I,1,K)=R(ID,2,K)
      P(I,1,K)=P(ID,2,K)
      E(I,1,K)=E(ID,2,K)
      RU(I,1,K)=RU(ID,2,K)
      RV(I,1,K)=RV(ID,2,K)
      RW(I,1,K)=RW(ID,2,K)
    2 CONTINUE

   The only complication in setting up the parallel scatters is determining 
   the node number on the parallel grid for the point corresponding to
   (id,2,k) on the uniprocessor grid.  Note that the parallel node numbering,
   which is controlled by the distributed arrays (DAs) so that each processor
   locally owns a contiguous section of grid points, in general differs 
   from the uniprocessor ordering.

   To handle this situation, we employ the PETSc utility AOApplicationToPetsc()
   to switch between a app-defined ordering (in this case, the ordering that
   would be used for the grid if only 1 processor were used) and that used
   internally by PETSc matrices and vectors.  Thus, we can easily determine
   precisely what data needs to be communicated, regardless of the processor
   layout.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int BCScatterSetUp(Euler *app)
{
  IS     to, from;
  AO     ao;
  int    jkx_da, jkx_ao, ijkx, idjkx, size, *is0, *is1;
  int    ict, m, k, i, ierr, *ltog, nloc, gxm = app->gxm, gym = app->gym;
  int    zsi = app->zsi, xsi = app->xsi, xei = app->xei, zei = app->zei;
  int    gxs = app->gxs, gys = app->gys, gzs = app->gzs, nc = app->nc;
  int    mx = app->mx, my = app->my, ni = app->ni, j_int, shift;
  int    ktip = app->ktip, itl = app->itl, itu = app->itu;
  char   filename[64];
  Viewer view;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set vector scatter for setting certain pressure BC's.

     The procedure is:
       - Determine indices to which we scatter in the DA ordering
            j=j_int && ((k > ktip) || (i <= itl) || (i > itu))
       - Determine indices from which we scatter in the natural ordering;
         then map these to the DA ordering using AOApplicationToPetsc().
       - Create index sets; then form scatter context.

     This section of code is really needed only when BC's are applied 
     explicitly.  However, even when using the fully implicit formulation, 
     the initial guess computation (routine InitialGuess()) explicitly
     applies the BCs.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create two index sets for building scatter/gather.  For the formulation
     with fully implicit boundary conditions (bctype == IMPLICIT), we retain
     is1 for use in setting some Jacobian terms.  But is0 is used only in this
     routine.
  */
  size = nc * (zei-zsi+1) * (xei-xsi+1);
  is0 = (int *)PetscMalloc(size * sizeof(int)); CHKPTRQ(is0);
  is1 = (int *)PetscMalloc(size * sizeof(int)); CHKPTRQ(is1);
  app->is1 = is1;

  /* Set grid parameters and shifts so that the same code can
     handle both implicit and explicit cases */
  if (app->bctype == EXPLICIT) {j_int = 0; shift = 3;}
  else                         {j_int = 1; shift = 1;}

  ierr = DAGetGlobalIndices(app->da1,&nloc,&ltog); CHKERRQ(ierr);
  ict = 0;
  if (app->ys == 0) {
    for (k=zsi; k<zei; k++) {
      jkx_da = (j_int-gys)*gxm + (k-gzs)*gxm*gym; /* jkx_da = (j-gys)*gxm + (k-gzs)*gxm*gym, j=j_int */
      jkx_ao = j_int*mx + k*mx*my;                /* jkx_ao = j*mx + k*mx*my, j=j_int */
      for (i=xsi; i<xei; i++) {
        if ((k > ktip) || (i <= itl) || (i > itu)) {
          ijkx  = jkx_da + i-gxs;
          is0[ict] = ltog[ijkx];
	  is1[ict] = jkx_ao + ni-shift-i;  /* id = ni-shift-i; */
          ict++;
        }
      }
    }
  }
  /* Get global indices in PETSc ordering.  All processors in the distributed
     array's communicator MUST call these routines to access the parallel
     database!!  DAGetAO() CANNOT be called by a subset of processors or the
     code may hang because the parallel database requires participation of
     all processors. */
  ierr = DAGetAO(app->da1,&ao); CHKERRQ(ierr);

  /* Print debugging info */
  if (app->print_debug) {
    ierr = ViewerFileOpenASCII(app->comm,"aoP.out",&view); CHKERRQ(ierr);
    ierr = AOView(ao,view); CHKERRQ(ierr);
    ierr = ViewerDestroy(view); CHKERRQ(ierr);
  }
  ierr = AOApplicationToPetsc(ao,ict,is1); CHKERRQ(ierr);

  /* Create index sets for use in scatter context.  Note that we use the
     communicator PETSC_COMM_SELF for these index sets.  I.e., each processor
     specifies it's own separate index set when creating the scatter context. */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ict,is1,&from); CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ict,is0,&to); CHKERRQ(ierr);

  /* Print debugging info */
  if (app->print_debug) {
    sprintf(filename,"fromP.%d",app->rank);
    ierr = ViewerFileOpenASCII(PETSC_COMM_SELF,filename,&view); CHKERRQ(ierr);
    ierr = ISView(from,view); CHKERRQ(ierr);
    ierr = ViewerDestroy(view); CHKERRQ(ierr);
    sprintf(filename,"toP.%d",app->rank);
    ierr = ViewerFileOpenASCII(PETSC_COMM_SELF,filename,&view); CHKERRQ(ierr);
    ierr = ISView(to,view); CHKERRQ(ierr);
    ierr = ViewerDestroy(view); CHKERRQ(ierr);
  }

  /* Create scatter context */
  ierr = VecScatterCreate(app->P,from,app->Pbc,to,&app->Pbcscatter); CHKERRQ(ierr);
  ierr = ISDestroy(from); CHKERRQ(ierr);
  ierr = ISDestroy(to); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set vector scatter for setting certain BC's for X.  We use same 
     procedure as above, except with a different DA.  We need this for
     the both the fully implicit and explicit BC formulations. 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DAGetGlobalIndices(app->da,&nloc,&ltog); CHKERRQ(ierr);
  ict = 0;
  if (app->ys == 0) {
    for (k=zsi; k<zei; k++) {
      jkx_da = (j_int-gys)*gxm + (k-gzs)*gxm*gym; /* jkx_da = (j-gys)*gxm + (k-gzs)*gxm*gym, j=j_int */
      jkx_ao = j_int*mx + k*mx*my;                /* jkx_ao = j*mx + k*mx*my, j=j_int */
      for (i=xsi; i<xei; i++) {
        if ((k > ktip) || (i <= itl) || (i > itu)) {
          ijkx  = nc * (jkx_da + i-gxs);
	  idjkx = nc * (jkx_ao + ni-shift-i); /* id = ni-shift-i; */
          for (m=0; m<nc; m++) {
            is0[ict+m] = ltog[ijkx+m];
            is1[ict+m] = idjkx + m;
          }
          ict += nc;
        }
      }
    }
  }
  /* Get global indices in PETSc ordering.  All processors MUST call these routines
     to access the parallel database! */
  ierr = DAGetAO(app->da,&ao); CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,ict,is1); CHKERRQ(ierr);

  /* Print debugging info */
  if (app->print_debug) {
    ierr = ViewerFileOpenASCII(app->comm,"aoX.out",&view); CHKERRQ(ierr);
    ierr = AOView(ao,view); CHKERRQ(ierr);
    ierr = ViewerDestroy(view); CHKERRQ(ierr);
  }

  /* Create scatter context */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ict,is1,&from); CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ict,is0,&to); CHKERRQ(ierr);

  /* Print debugging info */
  if (app->print_debug) {
    sprintf(filename,"fromX.%d",app->rank);
    ierr = ViewerFileOpenASCII(PETSC_COMM_SELF,filename,&view); CHKERRQ(ierr);
    ierr = ISView(from,view); CHKERRQ(ierr);
    ierr = ViewerDestroy(view); CHKERRQ(ierr);
    sprintf(filename,"toX.%d",app->rank);
    ierr = ViewerFileOpenASCII(PETSC_COMM_SELF,filename,&view); CHKERRQ(ierr);
    ierr = ISView(to,view); CHKERRQ(ierr);
    ierr = ViewerDestroy(view); CHKERRQ(ierr);
  }

  ierr = VecScatterCreate(app->X,from,app->Xbc,to,&app->Xbcscatter); CHKERRQ(ierr);
  PetscFree(is0);
  ierr = ISDestroy(from); CHKERRQ(ierr);
  ierr = ISDestroy(to); CHKERRQ(ierr);
  return 0;
}
#undef __FUNC__
#define __FUNC__ "BoundaryConditionsExplicit"
/* ----------------------------------------------------------------------- */
/*
   BoundaryConditionsExplicit - Applies all boundary conditions explicitly.
   Required for parallel case, where vector scatters are used for some BC's.

   Input Parameters:
   app - application-defined context
   X    - current iterate

  -----------------------
   Notes for Euler code:
  -----------------------
   The vector scatter contexts are set a single time in BCScatterSetUp()
   and then are reused repeatedly to apply the boundary conditions.

   Since pressure is not one of the variables in the primary solution vector,
   this scatter occurs separately from the rest of the variables (momentum,
   energy, and density).  The basic procedure used here is:
     - Apply BC's that don't involve vector scatters as usual.  This includes
       all but the C-grid continuity enforcement (part of j=1 boundary)
     - Apply vector scatters for (A) pressure and (B) the primary variables:
         (1) Pack pressure work vector with Julianne work array
         (2) Scatter from this pressure vector to a work vector
         (3) Pack a Julianne work array with the scattered pressure data
         (4) Scatter from X to a work vector
         (5) Pack other Julianne work arrays with the scattered X data
         (6) Apply the boundary conditions using the work arrays that
             contain the newly scattered data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int BoundaryConditionsExplicit(Euler *app,Vec X)
{
  int    ierr;

  /* Uniprocessor boundary conditions, which are now obselete */
  /*  
     bc_uni_(app->xx,app->p,
           app->sadai,app->sadaj,app->sadak,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz);
     return 0; 
  */

  /* Apply explicit bc's that don't involve vector scatters (unless just
     testing for correct BC scatters).  Note that we MUST set i=1 boundary
     first, since the fields at points (i=1,j=2,k=<anything>) are used
     to set the (i=49,j=1,k=<anything>) boundary. */
  if (!app->bc_test) {
    bc_(app->xx,app->p,
           app->sadai,app->sadaj,app->sadak,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz);
  }

  /* Place pressures in work vector */
  if (!app->bc_test) {
    ierr = UnpackWorkComponent(app,app->p,app->P); CHKERRQ(ierr);
  }

  /* Scatter pressure BCs to another work vector */
  ierr = VecScatterBegin(app->P,app->Pbc,INSERT_VALUES,SCATTER_FORWARD,app->Pbcscatter); CHKERRQ(ierr);
  ierr = VecScatterEnd(app->P,app->Pbc,INSERT_VALUES,SCATTER_FORWARD,app->Pbcscatter); CHKERRQ(ierr);

  /* Pack pressure work array */
  ierr = PackWork(app,app->da1,app->Pbc,app->localP,&app->p_bc); CHKERRQ(ierr);

  /* Scatter BC components of X to another work vector */
  ierr = VecScatterBegin(X,app->Xbc,INSERT_VALUES,SCATTER_FORWARD,app->Xbcscatter); CHKERRQ(ierr);
  ierr = VecScatterEnd(X,app->Xbc,INSERT_VALUES,SCATTER_FORWARD,app->Xbcscatter); CHKERRQ(ierr);

  /* Convert BC work vector to Julianne format work arrays */
  ierr = PackWork(app,app->da,app->Xbc,app->localXBC,&app->xx_bc); CHKERRQ(ierr);

  /* Set certain explicit boundary conditions using scattered data */
  bcpart_j1_(app->xx,app->p,app->xx_bc,app->p_bc,
           app->sadai,app->sadaj,app->sadak,
           app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
           app->aiz,app->ajz,app->akz);

  return 0;
}

#undef __FUNC__
#define __FUNC__ "BoundaryConditionsImplicit"
/* ----------------------------------------------------------------------- */
/*
   BoundaryConditionImplicit - Performs scatters needed for computing the
   nonlinear function components for certain implicit boundary conditions.
   Required for parallel case.

   Input Parameters:
   app - application-defined context
   X    - current iterate

  -----------------------
   Notes for Euler code:
  -----------------------
   The vector scatter contexts are set a single time in BCScatterSetUp() and 
   then are reused repeatedly to handle boundary data.

   The basic procedure used here is:
     (1) Scatter from X to another work vector
     (2) Pack another Julianne work array with the scattered data
     (3) Then use this newly scattered data later in computating F(X).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int BoundaryConditionsImplicit(Euler *app,Vec X)
{
  int    ierr;

  /* Scatter BC components of X to another work vector */
  ierr = VecScatterBegin(X,app->Xbc,INSERT_VALUES,SCATTER_FORWARD,app->Xbcscatter); CHKERRQ(ierr);
  ierr = VecScatterEnd(X,app->Xbc,INSERT_VALUES,SCATTER_FORWARD,app->Xbcscatter); CHKERRQ(ierr);

  /* Convert BC work vector to Julianne format work arrays */
  ierr = PackWork(app,app->da,app->Xbc,app->localXBC,&app->xx_bc); CHKERRQ(ierr);

  return 0;
}



