/*$Id: dainterp.c,v 1.25 2001/08/07 03:04:39 balay Exp $*/
 
/*
  Code for interpolating between grids represented by DAs
*/

#include "src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/
#include "petscmg.h"

#undef __FUNCT__  
#define __FUNCT__ "DMGetInterpolationScale"
int DMGetInterpolationScale(DM dac,DM daf,Mat mat,Vec *scale)
{
  int    ierr;
  Vec    fine;
  PetscScalar one = 1.0;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(daf,&fine);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac,scale);CHKERRQ(ierr);
  ierr = VecSet(&one,fine);CHKERRQ(ierr);
  ierr = MatRestrict(mat,fine,*scale);CHKERRQ(ierr);
  ierr = VecDestroy(fine);CHKERRQ(ierr);
  ierr = VecReciprocal(*scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation_1D_Q1"
int DAGetInterpolation_1D_Q1(DA dac,DA daf,Mat *A)
{
  int            ierr,i,i_start,m_f,Mx,*idx_f;
  int            m_ghost,*idx_c,m_ghost_c;
  int            row,col,i_start_ghost,mx,m_c,nc,ratio;
  int            i_c,i_start_c,i_start_ghost_c,cols[2],dof;
  PetscScalar    v[2],x;
  Mat            mat;
  DAPeriodicType pt;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,0,0,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (pt == DA_XPERIODIC) {
    ratio = mx/Mx;
    if (ratio*Mx != mx) SETERRQ2(1,"Ratio between levels: mx/Mx  must be integer: mx %d Mx %d",mx,Mx);
  } else {
    ratio = (mx-1)/(Mx-1);
    if (ratio*(Mx-1) != mx-1) SETERRQ2(1,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %d Mx %d",mx,Mx);
  }

  ierr = DAGetCorners(daf,&i_start,0,0,&m_f,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,0,0,&m_ghost,0,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,0,0,&m_c,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,0,0,&m_ghost_c,0,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = MatCreateMPIAIJ(dac->comm,m_f,m_c,mx,Mx,2,0,0,0,&mat);CHKERRQ(ierr);
  if (!DAXPeriodic(pt)){ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);}

  /* loop over local fine grid nodes setting interpolation for those*/
  for (i=i_start; i<i_start+m_f; i++) {
    /* convert to local "natural" numbering and then to PETSc global numbering */
    row    = idx_f[dof*(i-i_start_ghost)]/dof;

    i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
    if (i_c < i_start_ghost_c) SETERRQ3(1,"Processor's coarse DA must lie over fine DA\n\
    i_start %d i_c %d i_start_ghost_c %d",i_start,i_c,i_start_ghost_c);

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
    cols[nc] = idx_c[col]/dof; 
    v[nc++]  = - x + 1.0;
    /* one right? */
    if (i_c*ratio != i) { 
      cols[nc] = idx_c[col+dof]/dof;
      v[nc++]  = x;
    }
    ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  PetscLogFlops(5*m_f);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation_1D_Q0"
int DAGetInterpolation_1D_Q0(DA dac,DA daf,Mat *A)
{
  int            ierr,i,i_start,m_f,Mx,*idx_f;
  int            m_ghost,*idx_c,m_ghost_c;
  int            row,col,i_start_ghost,mx,m_c,nc,ratio;
  int            i_c,i_start_c,i_start_ghost_c,cols[2],dof;
  PetscScalar    v[2],x;
  Mat            mat;
  DAPeriodicType pt;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,0,0,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,0,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (pt == DA_XPERIODIC) {
    ratio = mx/Mx;
    if (ratio*Mx != mx) SETERRQ2(1,"Ratio between levels: mx/Mx  must be integer: mx %d Mx %d",mx,Mx);
  } else {
    ratio = (mx-1)/(Mx-1);
    if (ratio*(Mx-1) != mx-1) SETERRQ2(1,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %d Mx %d",mx,Mx);
  }

  ierr = DAGetCorners(daf,&i_start,0,0,&m_f,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,0,0,&m_ghost,0,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,0,0,&m_c,0,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,0,0,&m_ghost_c,0,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = MatCreateMPIAIJ(dac->comm,m_f,m_c,mx,Mx,2,0,0,0,&mat);CHKERRQ(ierr);
  if (!DAXPeriodic(pt)) {ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);}

  /* loop over local fine grid nodes setting interpolation for those*/
  for (i=i_start; i<i_start+m_f; i++) {
    /* convert to local "natural" numbering and then to PETSc global numbering */
    row    = idx_f[dof*(i-i_start_ghost)]/dof;

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
    cols[nc] = idx_c[col]/dof; 
    v[nc++]  = - x + 1.0;
    /* one right? */
    if (i_c*ratio != i) { 
      cols[nc] = idx_c[col+dof]/dof;
      v[nc++]  = x;
    }
    ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  PetscLogFlops(5*m_f);
  PetscFunctionReturn(0);
}


/*   dof degree of freedom per node, nonperiodic */
#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation_2D_Q1"
int DAGetInterpolation_2D_Q1(DA dac,DA daf,Mat *A)
{
  int            ierr,i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof;
  int            m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c,*dnz,*onz;
  int            row,col,i_start_ghost,j_start_ghost,cols[4],mx,m_c,my,nc,ratioi,ratioj;
  int            i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  int            size_c,size_f,rank_f,col_shift,col_scale;
  PetscScalar    v[4],x,y;
  Mat            mat;
  DAPeriodicType pt;

  PetscFunctionBegin;

  ierr = DAGetInfo(dac,0,&Mx,&My,0,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,&my,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (DAXPeriodic(pt)){
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(1,"Ratio between levels: mx/Mx  must be integer: mx %d Mx %d",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(1,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %d Mx %d",mx,Mx);
  }
  if (DAYPeriodic(pt)){
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(1,"Ratio between levels: my/My  must be integer: my %d My %d",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(1,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %d My %d",my,My);
  }


  ierr = DAGetCorners(daf,&i_start,&j_start,0,&m_f,&n_f,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /*
     Used for handling a coarse DA that lives on 1/4 the processors of the fine DA.
     The coarse vector is then duplicated 4 times (each time it lives on 1/4 of the 
     processors). It's effective length is hence 4 times its normal length, this is
     why the col_scale is multiplied by the interpolation matrix column sizes.
     sol_shift allows each set of 1/4 processors do its own interpolation using ITS
     copy of the coarse vector. A bit of a hack but you do better.

     In the standard case when size_f == size_c col_scale == 1 and col_shift == 0
  */
  ierr = MPI_Comm_size(dac->comm,&size_c);CHKERRQ(ierr);
  ierr = MPI_Comm_size(daf->comm,&size_f);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(daf->comm,&rank_f);CHKERRQ(ierr);
  col_scale = size_f/size_c;
  col_shift = Mx*My*(rank_f/size_c);

  ierr = MatPreallocateInitialize(daf->comm,m_f*n_f,col_scale*m_c*n_c,dnz,onz);CHKERRQ(ierr);
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */

      if (j_c < j_start_ghost_c) SETERRQ3(1,"Processor's coarse DA must lie over fine DA\n\
    j_start %d j_c %d j_start_ghost_c %d",j_start,j_c,j_start_ghost_c);
      if (i_c < i_start_ghost_c) SETERRQ3(1,"Processor's coarse DA must lie over fine DA\n\
    i_start %d i_c %d i_start_ghost_c %d",i_start,i_c,i_start_ghost_c);

      /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      nc = 0;
      /* one left and below; or we are right on it */
      col        = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc++] = col_shift + idx_c[col]/dof; 
      /* one right and below */
      if (i_c*ratioi != i) { 
        cols[nc++] = col_shift + idx_c[col+dof]/dof;
      }
      /* one left and above */
      if (j_c*ratioj != j) { 
        cols[nc++] = col_shift + idx_c[col+m_ghost_c*dof]/dof;
      }
      /* one right and above */
      if (j_c*ratioi != j && i_c*ratioj != i) { 
        cols[nc++] = col_shift + idx_c[col+(m_ghost_c+1)*dof]/dof;
      }
      ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatCreateMPIAIJ(daf->comm,m_f*n_f,col_scale*m_c*n_c,mx*my,col_scale*Mx*My,0,dnz,0,onz,&mat);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  if (!DAXPeriodic(pt) && !DAYPeriodic(pt)) {ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);}

  /* loop over local fine grid nodes setting interpolation for those*/
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */

      /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      x  = ((double)(i - i_c*ratioi))/((double)ratioi);
      y  = ((double)(j - j_c*ratioj))/((double)ratioj);
      /* printf("i j %d %d %g %g\n",i,j,x,y); */
      nc = 0;
      /* one left and below; or we are right on it */
      col      = dof*(m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc] = col_shift + idx_c[col]/dof; 
      v[nc++]  = x*y - x - y + 1.0;
      /* one right and below */
      if (i_c*ratioi != i) { 
        cols[nc] = col_shift + idx_c[col+dof]/dof;
        v[nc++]  = -x*y + x;
      }
      /* one left and above */
      if (j_c*ratioj != j) { 
        cols[nc] = col_shift + idx_c[col+m_ghost_c*dof]/dof;
        v[nc++]  = -x*y + y;
      }
      /* one right and above */
      if (j_c*ratioj != j && i_c*ratioi != i) { 
        cols[nc] = col_shift + idx_c[col+(m_ghost_c+1)*dof]/dof;
        v[nc++]  = x*y;
      }
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  PetscLogFlops(13*m_f*n_f);
  PetscFunctionReturn(0);
}


/*   dof degree of freedom per node, nonperiodic */
#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation_3D_Q1"
int DAGetInterpolation_3D_Q1(DA dac,DA daf,Mat *A)
{
  int            ierr,i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof,l;
  int            m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c,Mz,mz;
  int            row,col,i_start_ghost,j_start_ghost,cols[8],mx,m_c,my,nc,ratioi,ratioj,ratiok;
  int            i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  int            l_start,p_f,l_start_ghost,p_ghost,l_start_c,p_c;
  int            l_start_ghost_c,p_ghost_c,l_c,*dnz,*onz;
  PetscScalar    v[8],x,y,z;
  Mat            mat;
  DAPeriodicType pt;

  PetscFunctionBegin;
  ierr = DAGetInfo(dac,0,&Mx,&My,&Mz,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,&my,&mz,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (DAXPeriodic(pt)){
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(1,"Ratio between levels: mx/Mx  must be integer: mx %d Mx %d",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(1,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %d Mx %d",mx,Mx);
  }
  if (DAYPeriodic(pt)){
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(1,"Ratio between levels: my/My  must be integer: my %d My %d",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(1,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %d My %d",my,My);
  }
  if (DAZPeriodic(pt)){
    ratiok = mz/Mz;
    if (ratiok*Mz != mz) SETERRQ2(1,"Ratio between levels: mz/Mz  must be integer: mz %d Mz %d",mz,Mz);
  } else {
    ratiok = (mz-1)/(Mz-1);
    if (ratiok*(Mz-1) != mz-1) SETERRQ2(1,"Ratio between levels: (mz - 1)/(Mz - 1) must be integer: mz %d Mz %d",mz,Mz);
  }

  ierr = DAGetCorners(daf,&i_start,&j_start,&l_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&l_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,&j_start_c,&l_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&l_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix, determining exact preallocation */
  ierr = MatPreallocateInitialize(dac->comm,m_f*n_f*p_f,m_c*n_c*p_c,dnz,onz);CHKERRQ(ierr);
  /* loop over local fine grid nodes counting interpolating points */
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;
        i_c = (i/ratioi);
        j_c = (j/ratioj);
        l_c = (l/ratiok);
        if (l_c < l_start_ghost_c) SETERRQ3(1,"Processor's coarse DA must lie over fine DA\n\
          l_start %d l_c %d l_start_ghost_c %d",l_start,l_c,l_start_ghost_c);
        if (j_c < j_start_ghost_c) SETERRQ3(1,"Processor's coarse DA must lie over fine DA\n\
          j_start %d j_c %d j_start_ghost_c %d",j_start,j_c,j_start_ghost_c);
        if (i_c < i_start_ghost_c) SETERRQ3(1,"Processor's coarse DA must lie over fine DA\n\
          i_start %d i_c %d i_start_ghost_c %d",i_start,i_c,i_start_ghost_c);

        /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
        */
        nc       = 0;
        col      = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc++] = idx_c[col]/dof; 
        if (i_c*ratioi != i) { 
          cols[nc++] = idx_c[col+dof]/dof;
        }
        if (j_c*ratioj != j) { 
          cols[nc++] = idx_c[col+m_ghost_c*dof]/dof;
        }
        if (l_c*ratiok != l) { 
          cols[nc++] = idx_c[col+m_ghost_c*n_ghost_c*dof]/dof;
        }
        if (j_c*ratioj != j && i_c*ratioi != i) { 
          cols[nc++] = idx_c[col+(m_ghost_c+1)*dof]/dof;
        }
        if (j_c*ratioj != j && l_c*ratiok != l) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)*dof]/dof;
        }
        if (i_c*ratioi != i && l_c*ratiok != l) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+1)*dof]/dof;
        }
        if (i_c*ratioi != i && l_c*ratiok != l && j_c*ratioj != j) { 
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)*dof]/dof;
        }
        ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatCreateMPIAIJ(dac->comm,m_f*n_f*p_f,m_c*n_c*p_c,mx*my*mz,Mx*My*Mz,0,dnz,0,onz,&mat);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  if (!DAXPeriodic(pt) && !DAYPeriodic(pt) && !DAZPeriodic(pt)) {ierr = MatSetOption(mat,MAT_COLUMNS_SORTED);CHKERRQ(ierr);}

  /* loop over local fine grid nodes setting interpolation for those*/
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[dof*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))]/dof;

        i_c = (i/ratioi);
        j_c = (j/ratioj);
        l_c = (l/ratiok);

        /* 
           Only include those interpolation points that are truly 
           nonzero. Note this is very important for final grid lines
           in x and y directions; since they have no right/top neighbors
        */
        x  = ((double)(i - i_c*ratioi))/((double)ratioi);
        y  = ((double)(j - j_c*ratioj))/((double)ratioj);
        z  = ((double)(l - l_c*ratiok))/((double)ratiok);
        /* printf("i j l %d %d %d %g %g %g\n",i,j,l,x,y,z); */
        nc = 0;
        /* one left and below; or we are right on it */
        col      = dof*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c)+m_ghost_c*(j_c-j_start_ghost_c)+(i_c-i_start_ghost_c));

        cols[nc] = idx_c[col]/dof; 
        v[nc++]  = .125*(1. - (2.0*x-1.))*(1. - (2.0*y-1.))*(1. - (2.0*z-1.));

        if (i_c*ratioi != i) { 
          cols[nc] = idx_c[col+dof]/dof;
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. - (2.0*y-1.))*(1. - (2.0*z-1.));
        }

        if (j_c*ratioj != j) { 
          cols[nc] = idx_c[col+m_ghost_c*dof]/dof;
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. + (2.0*y-1.))*(1. - (2.0*z-1.));
        }

        if (l_c*ratiok != l) { 
          cols[nc] = idx_c[col+m_ghost_c*n_ghost_c*dof]/dof;
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. - (2.0*y-1.))*(1. + (2.0*z-1.));
        }

        if (j_c*ratioj != j && i_c*ratioi != i) { 
          cols[nc] = idx_c[col+(m_ghost_c+1)*dof]/dof;
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. + (2.0*y-1.))*(1. - (2.0*z-1.));
        }

        if (j_c*ratioj != j && l_c*ratiok != l) { 
          cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)*dof]/dof;
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. + (2.0*y-1.))*(1. + (2.0*z-1.));
        }

        if (i_c*ratioi != i && l_c*ratiok != l) { 
          cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+1)*dof]/dof;
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. - (2.0*y-1.))*(1. + (2.0*z-1.));
        }

        if (i_c*ratioi != i && l_c*ratiok != l && j_c*ratioj != j) { 
          cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)*dof]/dof;
          v[nc++]  = .125*(1. + (2.0*x-1.))*(1. + (2.0*y-1.))*(1. + (2.0*z-1.));
        }
        ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
      }
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);
  PetscLogFlops(13*m_f*n_f);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInterpolation"
/*@C
   DAGetInterpolation - Gets an interpolation matrix that maps between 
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

.seealso: DARefine(), DAGetInjection()
@*/
int DAGetInterpolation(DA dac,DA daf,Mat *A,Vec *scale)
{
  int            ierr,dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc,dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DAPeriodicType wrapc,wrapf;
  DAStencilType  stc,stf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DA_COOKIE,1);
  PetscValidHeaderSpecific(daf,DA_COOKIE,2);
  PetscValidPointer(A,3);
  if (scale) PetscValidPointer(scale,4);

  ierr = DAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&wrapc,&stc);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&wrapf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(1,"Dimensions of DA do not match %d %d",dimc,dimf);CHKERRQ(ierr);
  /* if (mc != mf) SETERRQ2(1,"Processor dimensions of DA in X %d %d do not match",mc,mf);CHKERRQ(ierr);
     if (nc != nf) SETERRQ2(1,"Processor dimensions of DA in Y %d %d do not match",nc,nf);CHKERRQ(ierr);
     if (pc != pf) SETERRQ2(1,"Processor dimensions of DA in Z %d %d do not match",pc,pf);CHKERRQ(ierr); */
  if (dofc != doff) SETERRQ2(1,"DOF of DA do not match %d %d",dofc,doff);CHKERRQ(ierr);
  if (sc != sf) SETERRQ2(1,"Stencil width of DA do not match %d %d",sc,sf);CHKERRQ(ierr);
  if (wrapc != wrapf) SETERRQ(1,"Periodic type different in two DAs");CHKERRQ(ierr);
  if (stc != stf) SETERRQ(1,"Stencil type different in two DAs");CHKERRQ(ierr);
  if (Mc < 2) SETERRQ(1,"Coarse grid requires at least 2 points in x direction");
  if (dimc > 1 && Nc < 2) SETERRQ(1,"Coarse grid requires at least 2 points in y direction");
  if (dimc > 2 && Pc < 2) SETERRQ(1,"Coarse grid requires at least 2 points in z direction");

  if (dac->interptype == DA_Q1){
    if (dimc == 1){
      ierr = DAGetInterpolation_1D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 2){
      ierr = DAGetInterpolation_2D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 3){
      ierr = DAGetInterpolation_3D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else {
      SETERRQ2(1,"No support for this DA dimension %d for interpolation type %d",dimc,dac->interptype);
    }
  } else if (dac->interptype == DA_Q0){
    if (dimc == 1){
      ierr = DAGetInterpolation_1D_Q0(dac,daf,A);CHKERRQ(ierr);
    } else {
      SETERRQ2(1,"No support for this DA dimension %d for interpolation type %d",dimc,dac->interptype);
    }
  }
  if (scale) {
    ierr = DMGetInterpolationScale((DM)dac,(DM)daf,*A,scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "DAGetInjection_2D"
int DAGetInjection_2D(DA dac,DA daf,VecScatter *inject)
{
  int            ierr,i,j,i_start,j_start,m_f,n_f,Mx,My,*idx_f,dof;
  int            m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c;
  int            row,i_start_ghost,j_start_ghost,mx,m_c,my,nc,ratioi,ratioj;
  int            i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  int            *cols;
  DAPeriodicType pt;
  Vec            vecf,vecc;
  IS             isf;

  PetscFunctionBegin;

  ierr = DAGetInfo(dac,0,&Mx,&My,0,0,0,0,0,0,&pt,0);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,0,&mx,&my,0,0,0,0,&dof,0,0,0);CHKERRQ(ierr);
  if (DAXPeriodic(pt)){
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(1,"Ratio between levels: mx/Mx  must be integer: mx %d Mx %d",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(1,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %d Mx %d",mx,Mx);
  }
  if (DAYPeriodic(pt)){
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(1,"Ratio between levels: my/My  must be integer: my %d My %d",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(1,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %d My %d",my,My);
  }


  ierr = DAGetCorners(daf,&i_start,&j_start,0,&m_f,&n_f,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(daf,PETSC_NULL,&idx_f);CHKERRQ(ierr);

  ierr = DAGetCorners(dac,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(dac,PETSC_NULL,&idx_c);CHKERRQ(ierr);


  /* loop over local fine grid nodes setting interpolation for those*/
  nc = 0;
  ierr = PetscMalloc(n_f*m_f*sizeof(int),&cols);CHKERRQ(ierr);
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */

      if (i_c*ratioi == i && j_c*ratioj == j) { 
	/* convert to local "natural" numbering and then to PETSc global numbering */
	row    = idx_f[dof*(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];
        cols[nc++] = row; 
      }
    }
  }

  ierr = ISCreateBlock(daf->comm,dof,nc,cols,&isf);CHKERRQ(ierr);
  ierr = DAGetGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DAGetGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = VecScatterCreate(vecf,isf,vecc,PETSC_NULL,inject);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DARestoreGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = ISDestroy(isf);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetInjection"
/*@C
   DAGetInjection - Gets an injection matrix that maps between 
   grids associated with two DAs.

   Collective on DA

   Input Parameters:
+  dac - the coarse grid DA
-  daf - the fine grid DA

   Output Parameters:
.  inject - the injection scatter

   Level: intermediate

.keywords: interpolation, restriction, multigrid, injection 

.seealso: DARefine(), DAGetInterpolation()
@*/
int DAGetInjection(DA dac,DA daf,VecScatter *inject)
{
  int            ierr,dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc,dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DAPeriodicType wrapc,wrapf;
  DAStencilType  stc,stf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DA_COOKIE,1);
  PetscValidHeaderSpecific(daf,DA_COOKIE,2);
  PetscValidPointer(inject,3);

  ierr = DAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&wrapc,&stc);CHKERRQ(ierr);
  ierr = DAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&wrapf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(1,"Dimensions of DA do not match %d %d",dimc,dimf);CHKERRQ(ierr);
  /* if (mc != mf) SETERRQ2(1,"Processor dimensions of DA in X %d %d do not match",mc,mf);CHKERRQ(ierr);
     if (nc != nf) SETERRQ2(1,"Processor dimensions of DA in Y %d %d do not match",nc,nf);CHKERRQ(ierr);
     if (pc != pf) SETERRQ2(1,"Processor dimensions of DA in Z %d %d do not match",pc,pf);CHKERRQ(ierr); */
  if (dofc != doff) SETERRQ2(1,"DOF of DA do not match %d %d",dofc,doff);CHKERRQ(ierr);
  if (sc != sf) SETERRQ2(1,"Stencil width of DA do not match %d %d",sc,sf);CHKERRQ(ierr);
  if (wrapc != wrapf) SETERRQ(1,"Periodic type different in two DAs");CHKERRQ(ierr);
  if (stc != stf) SETERRQ(1,"Stencil type different in two DAs");CHKERRQ(ierr);
  if (Mc < 2) SETERRQ(1,"Coarse grid requires at least 2 points in x direction");
  if (dimc > 1 && Nc < 2) SETERRQ(1,"Coarse grid requires at least 2 points in y direction");
  if (dimc > 2 && Pc < 2) SETERRQ(1,"Coarse grid requires at least 2 points in z direction");

  if (dimc == 2){
    ierr = DAGetInjection_2D(dac,daf,inject);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"No support for this DA dimension %d",dimc);
  }
  PetscFunctionReturn(0);
} 

