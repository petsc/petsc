/*$Id: ex6.c,v 1.36 1999/11/05 14:47:57 bsmith Exp bsmith $*/
      
static char help[] = "Tests various 3-dimensional DA routines.\n\n";

#include "da.h"
#include "sys.h"
#include "ao.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int            rank,M = 3,N = 5,P=3,s=1,w=2,nloc,l,i,j,k,kk;
  int            m = PETSC_DECIDE,n = PETSC_DECIDE,p = PETSC_DECIDE,ierr;
  int            Xs,Xm,Ys,Ym,Zs,Zm,iloc,*ltog,*iglobal;
  int            *lx = PETSC_NULL,*ly = PETSC_NULL,*lz = PETSC_NULL;
  PetscTruth     test_order;
  DA             da;
  Viewer         viewer;
  Vec            local,global;
  Scalar         value;
  DAPeriodicType wrap = DA_XYPERIODIC;
  DAStencilType  stencil_type = DA_STENCIL_BOX;
  AO             ao;
  PetscTruth     flg;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,400,300,&viewer);CHKERRA(ierr);

  /* Read options */  
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-P",&P,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-w",&w,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-star",&flg);CHKERRA(ierr); 
  if (flg) stencil_type =  DA_STENCIL_STAR;
  ierr = OptionsHasName(PETSC_NULL,"-test_order",&test_order);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-distribute",&flg);CHKERRA(ierr);
  if (flg) {
    if (m == PETSC_DECIDE) SETERRA(1,1,"Must set -m option with -distribute option");
    lx = (int*)PetscMalloc(m*sizeof(int));CHKPTRQ(lx);
    for (i=0; i<m-1; i++) { lx[i] = 4;}
    lx[m-1] = M - 4*(m-1);
    if (n == PETSC_DECIDE) SETERRA(1,1,"Must set -n option with -distribute option");
    ly = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(ly);
    for (i=0; i<n-1; i++) { ly[i] = 2;}
    ly[n-1] = N - 2*(n-1);
    if (p == PETSC_DECIDE) SETERRA(1,1,"Must set -p option with -distribute option");
    lz = (int*)PetscMalloc(p*sizeof(int));CHKPTRQ(lz);
    for (i=0; i<p-1; i++) { lz[i] = 2;}
    lz[p-1] = P - 2*(p-1);
  }

  /* Create distributed array and get vectors */
  ierr = DACreate3d(PETSC_COMM_WORLD,wrap,stencil_type,M,N,P,m,n,p,w,s,
                    lx,ly,lz,&da);CHKERRA(ierr);
  if (lx) {
    ierr = PetscFree(lx);CHKERRA(ierr);
    ierr = PetscFree(ly);CHKERRA(ierr);
    ierr = PetscFree(lz);CHKERRA(ierr);
  }
  ierr = DAView(da,viewer);CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRA(ierr);

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  ierr = VecSet(&value,global);CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);

  /* Scale local vectors according to processor rank; pass to global vector */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  value = rank;
  ierr = VecScale(&value,local);CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global);CHKERRA(ierr);

  if (!test_order) { /* turn off printing when testing ordering mappings */
    if (M*N*P<40) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGlobal Vector:\n");CHKERRA(ierr);
      ierr = VecView(global,VIEWER_STDOUT_WORLD);CHKERRA(ierr); 
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRA(ierr);
    }
  }

  /* Send ghost points to local vectors */
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-local_print",&flg);CHKERRA(ierr);
  if (flg) {
    Viewer sviewer;
    ierr = PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1);CHKERRA(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nLocal Vector: processor %d\n",rank);CHKERRA(ierr);
    ierr = ViewerGetSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
    ierr = VecView(local,sviewer);CHKERRA(ierr); 
    ierr = ViewerRestoreSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);
  }

  /* Tests mappings betweeen application/PETSc orderings */
  if (test_order) {
    ierr = DAGetGhostCorners(da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm);CHKERRA(ierr);
    ierr = DAGetGlobalIndices(da,&nloc,&ltog);CHKERRQ(ierr);
    ierr = DAGetAO(da,&ao);CHKERRA(ierr);
    /* ierr = AOView(ao,VIEWER_STDOUT_WORLD);CHKERRA(ierr); */
    iglobal = (int*)PetscMalloc(nloc*sizeof(int));CHKPTRA(iglobal);

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
    ierr = AOPetscToApplication(ao,nloc,iglobal);CHKERRA(ierr); 

    /* Then map the application ordering back to the PETSc DA ordering */
    ierr = AOApplicationToPetsc(ao,nloc,iglobal);CHKERRA(ierr); 

    /* Verify the mappings */
    kk=0;
    for (k=Zs; k<Zs+Zm; k++) {
      for (j=Ys; j<Ys+Ym; j++) {
        for (i=Xs; i<Xs+Xm; i++) {
          iloc = w*((k-Zs)*Xm*Ym + (j-Ys)*Xm + i-Xs); 
          for (l=0; l<w; l++) {
            if (iglobal[kk] != ltog[iloc+l]) {
              fprintf(stdout,"[%d] Problem with mapping: z=%d, j=%d, i=%d, l=%d, petsc1=%d, petsc2=%d\n",
                      rank,k,j,i,l,ltog[iloc+l],iglobal[kk]);
            }
            kk++;
          }
        }
      }
    }
    ierr = PetscFree(iglobal);CHKERRA(ierr);
  } 

  /* Free memory */
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  ierr = VecDestroy(local);CHKERRA(ierr);
  ierr = VecDestroy(global);CHKERRA(ierr);
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
  




















