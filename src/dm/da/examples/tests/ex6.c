#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.20 1996/03/23 00:31:23 curfman Exp curfman $";
#endif
      
static char help[] = "Tests various 3-dimensional DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include "ao.h"
#include <math.h>

int main(int argc,char **argv)
{
  int            rank, M = 3, N = 5, P=3, s=1, w=2, flg, nloc, l, i, j, k, kk;
  int            m = PETSC_DECIDE, n = PETSC_DECIDE, p = PETSC_DECIDE, ierr;
  int            Xs, Xm, Ys, Ym, Zs, Zm, iloc, *ltog, *iglobal, test_order;
  DA             da;
  Viewer         viewer;
  Vec            local, global;
  Scalar         value;
  DAPeriodicType wrap = DA_XYPERIODIC;
  DAStencilType  stencil_type = DA_STENCIL_BOX;
  AO             ao;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerDrawOpenX(MPI_COMM_WORLD,0,"",300,0,400,300,&viewer); CHKERRA(ierr);

  /* Read options */  
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-P",&P,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-p",&p,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-w",&w,&flg); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-star",&flg); CHKERRA(ierr); 
  if (flg) stencil_type =  DA_STENCIL_STAR;
  ierr = OptionsHasName(PETSC_NULL,"-test_order",&test_order); CHKERRA(ierr);

  /* Create distributed array and get vectors */
  ierr = DACreate3d(MPI_COMM_WORLD,wrap,stencil_type,M,N,P,m,n,p,w,s,&da); CHKERRA(ierr);
  ierr = DAView(da,viewer); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  ierr = VecSet(&value,global); CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  /* Scale local vectors according to processor rank; pass to global vector */
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  value = rank;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global); CHKERRA(ierr);

  if (!test_order) { /* turn off printing when testing ordering mappings */
    if (M*N*P<40) {
      PetscPrintf(MPI_COMM_WORLD,"\nGlobal Vector:\n");
      ierr = VecView(global,STDOUT_VIEWER_WORLD); CHKERRA(ierr); 
      PetscPrintf(MPI_COMM_WORLD,"\n");
    }
  }

  /* Send ghost points to local vectors */
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  flg = 0;
  ierr = OptionsHasName(PETSC_NULL,"-local_print",&flg); CHKERRA(ierr);
  if (flg) {
    PetscSequentialPhaseBegin(MPI_COMM_WORLD,1);
    printf("\nLocal Vector: processor %d\n",rank);
    ierr = VecView(local,STDOUT_VIEWER_SELF); CHKERRA(ierr); 
    PetscSequentialPhaseEnd(MPI_COMM_WORLD,1);
  }

  /* Tests mappings betweeen application/PETSc orderings */
  if (test_order) {
    ierr = DAGetGhostCorners(da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm); CHKERRA(ierr);
    ierr = DAGetGlobalIndices(da,&nloc,&ltog); CHKERRQ(ierr);
    ierr = DAGetAO(da,&ao); CHKERRA(ierr);
    /* ierr = AOView(ao,STDOUT_VIEWER_WORLD); CHKERRA(ierr); */
    iglobal = (int *) PetscMalloc( nloc*sizeof(int) ); CHKPTRA(iglobal);

    /* Set iglobal to be global indices for each processor's local and ghost nodes,
       using the DA ordering of grid points */
    kk = 0;
    for (k=Zs; k<Zs+Zm; k++) {
      for (j=Ys; j<Ys+Ym; j++) {
        for (i=Xs; i<Xs+Xm; i++) {
          iloc = w*((k-Zs)*Xm*Ym + (j-Ys)*Xm + i-Xs); 
          for (l=0; l<w; l++) {
            iglobal[kk++] = ltog[iloc+l];
          }
        }
      }
    } 

    /* Map this to the application ordering (which for DAs is just the natural ordering
       that would be used for 1 processor, numbering most rapidly by x, then y, then z) */
    ierr = AOPetscToApplication(ao,nloc,iglobal); CHKERRA(ierr); 

    /* Then map the application ordering back to the PETSc DA ordering */
    ierr = AOApplicationToPetsc(ao,nloc,iglobal); CHKERRA(ierr); 

    /* Verify the mappings */
    kk=0;
    for (k=Zs; k<Zs+Zm; k++) {
      for (j=Ys; j<Ys+Ym; j++) {
        for (i=Xs; i<Xs+Xm; i++) {
          iloc = w*((k-Zs)*Xm*Ym + (j-Ys)*Xm + i-Xs); 
          for (l=0; l<w; l++) {
            if (iglobal[kk] != ltog[iloc+l]) {fprintf(stdout,
              "[%d] Problem with mapping: z=%d, j=%d, i=%d, l=%d, petsc1=%d, petsc2=%d\n",
               rank,k,j,i,l,ltog[iloc+l],iglobal[kk]);}
            kk++;
          }
        }
      }
    }
    PetscFree(iglobal);
  } 

  /* Free memory */
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
  




















