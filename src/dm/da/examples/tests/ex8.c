#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex8.c,v 1.12 1999/03/19 21:24:17 bsmith Exp balay $";
#endif
      
static char help[] = "Demonstrates generating a slice from a DA Vector.\n\n";

#include "da.h"
#include "sys.h"
#include "ao.h"

#undef __FUNC__
#define __FUNC__ "GenerateSliceScatter"
/*
    Given a DA generates a VecScatter context that will deliver a slice
  of the global vector to each processor. In this example, each processor
  receives the values i=*, j=*, k=rank, i.e. one z plane.

  Note: This code is written assuming only one degree of freedom per node.
  For multiple degrees of freedom per node use ISCreateBlock()
  instead of ISCreateGeneral().
*/
int GenerateSliceScatter(DA da,VecScatter *scatter,Vec *vslice)
{
  AO       ao;
  int      M,N,P,nslice,rank,*sliceindices,count,ierr,i,j;
  MPI_Comm comm;
  Vec      vglobal;
  IS       isfrom,isto;

  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  MPI_Comm_rank(comm,&rank);

  ierr = DAGetAO(da,&ao); CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&M,&N,&P,0,0,0,0,0,0,0); CHKERRQ(ierr);

  /* 
     nslice is number of degrees of freedom in this processors slice
   if there are more processors then z plans the extra processors get 0
   elements in their slice.
  */
  if (rank < P) {nslice = M*N;} else nslice = 0;

  /* 
     Generate the local vector to hold this processors slice
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,nslice,vslice); CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&vglobal); CHKERRQ(ierr);

  /*
       Generate the indices for the slice in the "natural" global ordering
     Note: this is just an example, one could select any subset of nodes 
    on each processor. Just list them in the global natural ordering.

  */
  sliceindices = (int *) PetscMalloc((nslice+1)*sizeof(int));CHKPTRQ(sliceindices);
  count = 0;
  if (rank < P) {
    for ( j=0; j<N; j++ ) {
      for ( i=0; i<M; i++ ) {
         sliceindices[count++] = rank*M*N + j*M + i;
      }
    }
  }
  /*
      Convert the indices to the "PETSc" global ordering
  */
  ierr = AOApplicationToPetsc(ao,nslice,sliceindices); CHKERRQ(ierr); 
  
  /* Create the "from" and "to" index set */
  /* This is to scatter from the global vector */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nslice,sliceindices,&isfrom);CHKERRQ(ierr);
  /* This is to gather into the local vector */
  ierr = ISCreateStride(PETSC_COMM_SELF,nslice,0,1,&isto);CHKERRQ(ierr);

  ierr = VecScatterCreate(vglobal,isfrom,*vslice,isto,scatter); CHKERRQ(ierr);

  ierr = ISDestroy(isfrom); CHKERRQ(ierr); 
  ierr = ISDestroy(isto); CHKERRQ(ierr);

  PetscFree(sliceindices);
  return 0;
}


#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int            rank, M = 3, N = 5, P=3, s=1, flg;
  int            m = PETSC_DECIDE, n = PETSC_DECIDE, p = PETSC_DECIDE, ierr;
  int            *lx = PETSC_NULL, *ly = PETSC_NULL, *lz = PETSC_NULL;
  DA             da;
  Vec            local, global,vslice;
  Scalar         value;
  DAPeriodicType wrap = DA_XYPERIODIC;
  DAStencilType  stencil_type = DA_STENCIL_BOX;
  VecScatter     scatter;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  /* Read options */  
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-P",&P,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-p",&p,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,&flg); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-star",&flg); CHKERRA(ierr); 
  if (flg) stencil_type =  DA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  ierr = DACreate3d(PETSC_COMM_WORLD,wrap,stencil_type,M,N,P,m,n,p,1,s,
                    lx,ly,lz,&da); CHKERRA(ierr);
  ierr = DAView(da,VIEWER_DRAW_WORLD); CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&global); CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local); CHKERRA(ierr);

  ierr = GenerateSliceScatter(da,&scatter,&vslice); CHKERRA(ierr);

  /* Put the value rank+1 into all locations of vslice and transfer back to global vector */
  value = 1.0 + rank;
  ierr = VecSet(&value,vslice); CHKERRA(ierr);
  ierr = VecScatterBegin(vslice,global,INSERT_VALUES,SCATTER_REVERSE,scatter);CHKERRA(ierr);
  ierr = VecScatterEnd(vslice,global,INSERT_VALUES,SCATTER_REVERSE,scatter);CHKERRA(ierr);

  ierr = VecView(global,VIEWER_DRAW_WORLD);CHKERRA(ierr);

  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
