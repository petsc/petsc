/*$Id: dainterp.c,v 1.7 2000/04/09 04:39:49 bsmith Exp bsmith $*/
 
/*
  Code for interpolating between grids represented by DAs
*/

#include "src/dm/da/daimpl.h"    /*I   "da.h"   I*/
#include "mg.h"

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAGetInterpolation"
int DAGetInterpolationScale(DA dac,DA daf,Mat mat,Vec *scale)
{
  int    ierr;
  Vec    fine;
  Scalar one = 1.0;

  PetscFunctionBegin;
  ierr = DACreateGlobalVector(daf,&fine);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(dac,scale);CHKERRQ(ierr);
  ierr = VecSet(&one,fine);CHKERRQ(ierr);
  ierr = MatRestrict(mat,fine,*scale);CHKERRQ(ierr);
  ierr = VecDestroy(fine);CHKERRQ(ierr);
  ierr = VecReciprocal(*scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAGetInterpolation_1D_dof"
int DAGetInterpolation_1D_dof(DA dac,DA daf,Mat *A)
{
  int      ierr,i,i_start,m_f,Mx,*idx;
  int      m_ghost,*idx_c,m_ghost_c,k,ll;
  int      row,col,i_start_ghost,mx,m_c,nc,ratio;
  int      i_c,i_start_c,i_start_ghost_c,cols[2],dof;
  Scalar   v[2],x;
  Mat      mat;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  ratio = (mx-1)/(Mx-1);
  if (ratio*(Mx-1) != mx-1) SETERRQ2(1,1,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %d Mx %d",mx,Mx);

  ierr = DAGetCorners(daf,&i_start,0,0,&m_f,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,0,0,&m_ghost,0,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,0,0,&m_c,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,0,0,&m_ghost_c,0,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = MatCreateMPIAIJ(dac->comm,dof*m_f,dof*m_c,dof*mx,dof*Mx,2*dof,0,0,0,&mat);CHKERRQ(ierr);
  ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for (i=i_start; i<i_start+m_f; i++) {
    /* convert to local "natural" numbering and 
       then to PETSc global numbering */
    row    = idx[dof*(i-i_start_ghost)];

    i_c = (i/ratio);    /* coarse grid node to left of fine grid node */

    /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x direction; since they have no right neighbor
    */
    x  = ((double)(i - i_c*ratio))/((double)ratio);
    /* printf("i j %d %d %g %g\n",i,j,x,y); */
    nc = 0;
      /* one left and below; or we are right on it */
    col      = dof*(i_c-i_start_ghost_c);
    cols[nc] = idx_c[col]; 
    v[nc++]  = - x + 1.0;
    /* one right? */
    if (i_c*ratio != i) { 
      cols[nc] = idx_c[col+dof];
      v[nc++]  = x;
    }
    ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
    for (k=1; k<dof; k++) {
      for (ll=0; ll<nc; ll++) {
        cols[ll]++;
      }
      row++;
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PLogFlops(5*m_f);
  *A = mat;
  PetscFunctionReturn(0);
}


/*   dof degree of freedom per node, nonperiodic */
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAGetInterpolation_2D_dof"
int DAGetInterpolation_2D_dof(DA dac,DA daf,Mat *A)
{
  int      ierr,i,j,i_start,j_start,m_f,n_f,Mx,My,*idx,dof,k;
  int      m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c,l,*dnz,*onz;
  int      row,col,i_start_ghost,j_start_ghost,cols[4],mx,m_c,my,nc,ratio;
  int      i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  Scalar   v[4],x,y;
  Mat      mat;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,&My,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,&my,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  ratio = (mx-1)/(Mx-1);
  if (ratio*(Mx-1) != mx-1) SETERRQ2(1,1,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %d Mx %d",mx,Mx);
  if (ratio != (my-1)/(My-1)) SETERRQ(1,1,"Grid spacing ratio must be same in X and Y direction");

  ierr = DAGetCorners(daf,&i_start,&j_start,0,&m_f,&n_f,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  MatPreallocateInitialize(dac->comm,dof*m_f*n_f,dof*m_c*n_c,dnz,onz);
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

      i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratio);    /* coarse grid node below fine grid node */

      /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      nc = 0;
      /* one left and below; or we are right on it */
      col        = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc++] = idx_c[col]; 
      /* one right and below */
      if (i_c*ratio != i) { 
        cols[nc++] = idx_c[col+dof];
      }
      /* one left and above */
      if (j_c*ratio != j) { 
        cols[nc++] = idx_c[col+m_ghost_c*dof];
      }
      /* one right and above */
      if (j_c*ratio != j && i_c*ratio != i) { 
        cols[nc++] = idx_c[col+(m_ghost_c+1)*dof];
      }
      MatPreallocateSet(row,nc,cols,dnz,onz);
      for (k=1; k<dof; k++) {
        row++;
        MatPreallocateSet(row,nc,cols,dnz,onz);
      }
    }
  }
  ierr = MatCreateMPIAIJ(dac->comm,dof*m_f*n_f,dof*m_c*n_c,dof*mx*my,dof*Mx*My,0,dnz,0,onz,&mat);CHKERRQ(ierr);
  MatPreallocateFinalize(dnz,onz);
  ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

      i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratio);    /* coarse grid node below fine grid node */

      /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      x  = ((double)(i - i_c*ratio))/((double)ratio);
      y  = ((double)(j - j_c*ratio))/((double)ratio);
      /* printf("i j %d %d %g %g\n",i,j,x,y); */
      nc = 0;
      /* one left and below; or we are right on it */
      col      = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc] = idx_c[col]; 
      v[nc++]  = x*y - x - y + 1.0;
      /* one right and below */
      if (i_c*ratio != i) { 
        cols[nc] = idx_c[col+dof];
        v[nc++]  = -x*y + x;
      }
      /* one left and above */
      if (j_c*ratio != j) { 
        cols[nc] = idx_c[col+m_ghost_c*dof];
        v[nc++]  = -x*y + y;
      }
      /* one right and above */
      if (j_c*ratio != j && i_c*ratio != i) { 
        cols[nc] = idx_c[col+(m_ghost_c+1)*dof];
        v[nc++]  = x*y;
      }
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
      for (k=1; k<dof; k++) {
        for (l=0; l<nc; l++) {
          cols[l]++;
        }
        row++;
        ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
      }
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  *A = mat;
  PLogFlops(13*m_f*n_f);
  PetscFunctionReturn(0);
}


/*   dof degree of freedom per node, nonperiodic */
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAGetInterpolation_3D_dof"
int DAGetInterpolation_3D_dof(DA dac,DA daf,Mat *A)
{
  int      ierr,i,j,i_start,j_start,m_f,n_f,Mx,My,*idx,dof,k,l;
  int      m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c,Mz,mz;
  int      row,col,i_start_ghost,j_start_ghost,cols[8],mx,m_c,my,nc,ratio;
  int      i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  int      l_start,p_f,l_start_ghost,p_ghost,l_start_c,p_c;
  int      l_start_ghost_c,p_ghost_c,ll,l_c,*dnz,*onz;
  Scalar   v[8],x,y,z;
  Mat      mat;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,&My,&Mz,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,&my,&mz,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  ratio = (mx-1)/(Mx-1);
  if (ratio*(Mx-1) != mx-1) SETERRQ2(1,1,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %d Mx %d",mx,Mx);
  if (ratio != (my-1)/(My-1)) SETERRQ(1,1,"Grid spacing ratio must be same in X and Y direction");
  if (ratio != (mz-1)/(Mz-1)) SETERRQ(1,1,"Grid spacing ratio must be same in X and Y direction");

  ierr = DAGetCorners(daf,&i_start,&j_start,&l_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&l_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,&j_start_c,&l_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&l_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix, determining exact preallocation */
  MatPreallocateInitialize(dac->comm,dof*m_f*n_f*p_f,dof*m_c*n_c*p_c,dnz,onz);
  /* loop over local fine grid nodes counting interpolating points */
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];
        i_c = (i/ratio);
        j_c = (j/ratio);
        l_c = (l/ratio);

        /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
        */
        nc       = 0;
        col      = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc++] = idx_c[col]; 
        if (i_c*ratio != i) { 
          cols[nc++] = idx_c[col+dof];
        }
        if (j_c*ratio != j) { 
          cols[nc++] = idx_c[col+m_ghost_c*dof];
        }
        if (l_c*ratio != l) { 
          cols[nc++] = idx_c[col+m_ghost_c*n_ghost_c*dof];
        }
        if (j_c*ratio != j && i_c*ratio != i) { 
          cols[nc++] = idx_c[col+(m_ghost_c+1)*dof];
        }
        if (j_c*ratio != j && l_c*ratio != l) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)*dof];
        }
        if (i_c*ratio != i && l_c*ratio != l) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+1)*dof];
        }
        if (i_c*ratio != i && l_c*ratio != l && j_c*ratio != j) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)*dof];
        }
        MatPreallocateSet(row,nc,cols,dnz,onz);
        for (k=1; k<dof; k++) {
          row++;
          MatPreallocateSet(row,nc,cols,dnz,onz);
        }
      }
    }
  }
  ierr = MatCreateMPIAIJ(dac->comm,dof*m_f*n_f*p_f,dof*m_c*n_c*p_c,dof*mx*my*mz,dof*Mx*My*Mz,0,dnz,0,onz,&mat);CHKERRQ(ierr);
  MatPreallocateFinalize(dnz,onz);
  ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

        i_c = (i/ratio);
        j_c = (j/ratio);
        l_c = (l/ratio);

        /* 
           Only include those interpolation points that are truly 
           nonzero. Note this is very important for final grid lines
           in x and y directions; since they have no right/top neighbors
        */
        x  = ((double)(i - i_c*ratio))/((double)ratio);
        y  = ((double)(j - j_c*ratio))/((double)ratio);
        z  = ((double)(l - l_c*ratio))/((double)ratio);
        /* printf("i j l %d %d %d %g %g %g\n",i,j,l,x,y,z); */
        nc = 0;
        /* one left and below; or we are right on it */
        col      = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c)+m_ghost_c*(j_c-j_start_ghost_c)+(i_c-i_start_ghost_c));

        cols[nc] = idx_c[col]; 
        v[nc++]  = .125*(1. - (2.0*x-1.))*(1. - (2.0*y-1.))*(1. - (2.0*z-1.));

        if (i_c*ratio != i) { 
          cols[nc] = idx_c[col+dof];
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. - (2.0*y-1.))*(1. - (2.0*z-1.));
        }

        if (j_c*ratio != j) { 
          cols[nc] = idx_c[col+m_ghost_c*dof];
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. + (2.0*y-1.))*(1. - (2.0*z-1.));
        }

        if (l_c*ratio != l) { 
          cols[nc] = idx_c[col+m_ghost_c*n_ghost_c*dof];
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. - (2.0*y-1.))*(1. + (2.0*z-1.));
        }

        if (j_c*ratio != j && i_c*ratio != i) { 
          cols[nc] = idx_c[col+(m_ghost_c+1)*dof];
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. + (2.0*y-1.))*(1. - (2.0*z-1.));
        }

        if (j_c*ratio != j && l_c*ratio != l) { 
          cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)*dof];
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. + (2.0*y-1.))*(1. + (2.0*z-1.));
        }

        if (i_c*ratio != i && l_c*ratio != l) { 
          cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+1)*dof];
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. - (2.0*y-1.))*(1. + (2.0*z-1.));
        }

        if (i_c*ratio != i && l_c*ratio != l && j_c*ratio != j) { 
          cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)*dof];
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. + (2.0*y-1.))*(1. + (2.0*z-1.));
        }
        ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
        for (k=1; k<dof; k++) {
          for (ll=0; ll<nc; ll++) {
            cols[ll]++;
          }
          row++;
          ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
        }
      }
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  *A = mat;
  PLogFlops(13*m_f*n_f);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DAGetInterpolation"
/*@C
   DAGetInterpolation - Gets and interpolation matrix that maps between 
   grids associated with two DAs.

   Collective on DA

   Input Parameters:
+  dac - the coarse grid DA
-  daf - the fine grid DA

   Output Parameters:
+  A - the interpolation matrix
-  scale - a scaling vector used to scale the coarse grid restricted vector before applying the 
           grid function or grid Jacobian to it.

   Level: intermediate

.keywords: interpolation, restriction, multigrid 

.seealso: DARefine()
@*/
int DAGetInterpolation(DA dac,DA daf,Mat *A,Vec *scale)
{
  int            ierr,dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc,dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DAPeriodicType wrapc,wrapf;
  DAStencilType  stc,stf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DA_COOKIE);
  PetscValidHeaderSpecific(daf,DA_COOKIE);
  PetscCheckSameComm(dac,daf);

  ierr = DAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&wrapc,&stc);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&wrapf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(1,1,"Dimensions of DA do not match %d %d",dimc,dimf);CHKERRQ(ierr);
  if (mc != mf) SETERRQ2(1,1,"Processor dimensions of DA in X %d %d do not match",mc,mf);CHKERRQ(ierr);
  if (nc != nf) SETERRQ2(1,1,"Processor dimensions of DA in Y %d %d do not match",nc,nf);CHKERRQ(ierr);
  if (pc != pf) SETERRQ2(1,1,"Processor dimensions of DA in Z %d %d do not match",pc,pf);CHKERRQ(ierr);
  if (dofc != doff) SETERRQ2(1,1,"DOF of DA do not match %d %d",dofc,doff);CHKERRQ(ierr);
  if (sc != sf) SETERRQ2(1,1,"Stencil width of DA do not match %d %d",sc,sf);CHKERRQ(ierr);
  if (wrapc != wrapf) SETERRQ(1,1,"Periodic type different in two DAs");CHKERRQ(ierr);
  if (stc != stf) SETERRQ(1,1,"Stencil type different in two DAs");CHKERRQ(ierr);

  if (dimc == 1 && wrapc == DA_NONPERIODIC) {
    ierr = DAGetInterpolation_1D_dof(dac,daf,A);CHKERRQ(ierr);
  } else if (dimc == 2 && wrapc == DA_NONPERIODIC) {
    ierr = DAGetInterpolation_2D_dof(dac,daf,A);CHKERRQ(ierr);
  } else if (dimc == 3 && wrapc == DA_NONPERIODIC) {
    ierr = DAGetInterpolation_3D_dof(dac,daf,A);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"No support for this DA yet");
  }

  if (scale) {
    ierr = DAGetInterpolationScale(dac,daf,*A,scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 
