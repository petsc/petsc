/*$Id: ex7.c,v 1.19 1999/10/24 14:04:09 bsmith Exp bsmith $*/

static char help[] = "Tests DALocalToLocal().\n\n";

#include "da.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int            rank,M=8,ierr,dof=1,stencil_width=1,i,start,end,P=5;
  PetscTruth     flg,flg2,flg3;
  int            N = 6,m=PETSC_DECIDE,n=PETSC_DECIDE,p=PETSC_DECIDE;
  DAPeriodicType periodic;
  DAStencilType  stencil_type;
  DA             da;
  Vec            local,global,local_copy;
  Scalar         value,mone = -1.0;
  double         norm,work;
  Viewer         viewer;
  char           filename[64];
  FILE           *file;


  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL); CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-stencil_width",&stencil_width,PETSC_NULL); CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-periodic",(int*)&periodic,PETSC_NULL); CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-stencil_type",(int*)&stencil_type,PETSC_NULL); CHKERRA(ierr); 

  ierr = OptionsHasName(PETSC_NULL,"-2d",&flg2);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-3d",&flg3);CHKERRA(ierr);
  if (flg2) {
    ierr = DACreate2d(PETSC_COMM_WORLD,periodic,stencil_type,M,N,m,n,dof,stencil_width,
                      PETSC_NULL,PETSC_NULL,&da);CHKERRA(ierr);
  } else if (flg3) {
    ierr = DACreate3d(PETSC_COMM_WORLD,periodic,stencil_type,M,N,P,m,n,p,dof,stencil_width,
                      PETSC_NULL,PETSC_NULL,PETSC_NULL,&da);CHKERRA(ierr);
  }
  else {
    ierr = DACreate1d(PETSC_COMM_WORLD,periodic,M,dof,stencil_width,PETSC_NULL,&da);CHKERRA(ierr);
  }

  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRA(ierr);
  ierr = VecDuplicate(local,&local_copy);CHKERRA(ierr);

  
  /* zero out vectors so that ghostpoints are zero */
  value = 0;
  ierr = VecSet(&value,local);CHKERRA(ierr);
  ierr = VecSet(&value,local_copy);CHKERRA(ierr);

  ierr = VecGetOwnershipRange(global,&start,&end);CHKERRA(ierr);
  for ( i=start; i<end; i++ ) {
    value = i + 1;
    ierr = VecSetValues(global,1,&i,&value,INSERT_VALUES);CHKERRA(ierr); 
  }
  ierr = VecAssemblyBegin(global);CHKERRA(ierr);
  ierr = VecAssemblyEnd(global);CHKERRA(ierr);

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);


  ierr = DALocalToLocalBegin(da,local,INSERT_VALUES,local_copy);CHKERRA(ierr);
  ierr = DALocalToLocalEnd(da,local,INSERT_VALUES,local_copy);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-save",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
    sprintf(filename,"local.%d",rank);
    ierr = ViewerASCIIOpen(PETSC_COMM_SELF,filename,&viewer);CHKERRA(ierr);
    ierr = ViewerASCIIGetPointer(viewer,&file);CHKERRA(ierr);
    ierr = VecView(local,viewer);CHKERRA(ierr);
    fprintf(file,"Vector with correct ghost points\n");
    ierr = VecView(local_copy,viewer);CHKERRA(ierr);
    ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  }

  ierr = VecAXPY(&mone,local,local_copy);CHKERRA(ierr);
  ierr = VecNorm(local_copy,NORM_MAX,&work);CHKERRA(ierr);
  ierr = MPI_Allreduce( &work, &norm,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD );CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of difference %g should be zero\n",norm);CHKERRA(ierr);
   
  ierr = VecDestroy(local_copy);CHKERRA(ierr);
  ierr = VecDestroy(local);CHKERRA(ierr);
  ierr = VecDestroy(global);CHKERRA(ierr);
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
