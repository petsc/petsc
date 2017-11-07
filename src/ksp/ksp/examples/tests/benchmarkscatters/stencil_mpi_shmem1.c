/*
 * Copyright (c) 2012 Torsten Hoefler. All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@illinois.edu>
 *
 */

#include "./stencil_par.h"
#include <petscksp.h>

int main(int argc, char **argv) {
  PetscErrorCode ierr;

  //MPI_Init(&argc, &argv); 
  ierr = PetscInitialize(&argc,&argv,(char*)0,0);if (ierr) return ierr;
  int r,p;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &p);
  int i,j,iter,n, energy=25, niters=4,iwhile_max=1.e7;

  if (r==0) {
      // argument checking
      if(argc < 2) {
          if(!r) printf("usage: stencil_mpi <n> <energy> <niters>\n");
          MPI_Finalize();
          exit(1);
      }

      n = atoi(argv[1]); // nxn grid
      //energy = atoi(argv[2]); // energy to be injected per iteration
      niters = atoi(argv[2]); // number of iterations
      
      // distribute arguments
      int args[2] = {n,niters};
      MPI_Bcast(args, 2, MPI_INT, 0, comm);
      printf("[%d] n %d, energy %d, niters %d\n",r,n,energy,niters);
  }
  else {
      int args[2];
      MPI_Bcast(args, 2, MPI_INT, 0, comm);
      n=args[0];
      niters = args[1];
  }

  MPI_Comm shmcomm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

  int sr, sp; // rank and size in shmem comm
  MPI_Comm_size(shmcomm, &sp);
  MPI_Comm_rank(shmcomm, &sr);

  // this code works only on comm world!
  if(sp != p) MPI_Abort(comm, 1);

  int pdims[2]={0,0};
  MPI_Dims_create(sp, 2, pdims);
  int px = pdims[0];
  int py = pdims[1];
  if(!r) printf("processor grid: %i x %i\n", px, py);

  // determine my coordinates (x,y) -- r=x*a+y in the 2d processor array
  int rx = r % px;
  int ry = r / px;
  // determine my four neighbors
  int north = (ry-1)*px+rx; if(ry-1 < 0)   north = MPI_PROC_NULL;
  int south = (ry+1)*px+rx; if(ry+1 >= py) south = MPI_PROC_NULL;
  int west= ry*px+rx-1;     if(rx-1 < 0)   west = MPI_PROC_NULL;
  int east = ry*px+rx+1;    if(rx+1 >= px) east = MPI_PROC_NULL;
  // decompose the domain
  int bx = n/px; // block size in x
  int by = n/py; // block size in y
  int offx = rx*bx; // offset in x
  int offy = ry*by; // offset in y

  //printf("%i (%i,%i) - w: %i, e: %i, n: %i, s: %i\n", r, ry,rx,west,east,north,south);

  int size = (bx+2)*(by+2); // process-local grid (including halos (thus +2))
  double *mem;
  MPI_Win win;
  int extrasp = sp;
  MPI_Win_allocate_shared((extrasp+2*size)*sizeof(double), 1, MPI_INFO_NULL, shmcomm, &mem, &win);

  double *tmp;
  double *anew=mem+extrasp; // each rank's offset
  double *aold=mem+size+extrasp; // second half is aold!

  double *northptr, *southptr, *eastptr, *westptr, *myptr;
  double *northptr2, *southptr2, *eastptr2, *westptr2;
  MPI_Aint sz;
  int dsp_unit;
  MPI_Win_shared_query(win, north, &sz, &dsp_unit, &northptr);
  MPI_Win_shared_query(win, south, &sz, &dsp_unit, &southptr);
  MPI_Win_shared_query(win, east, &sz, &dsp_unit, &eastptr);
  MPI_Win_shared_query(win, west, &sz, &dsp_unit, &westptr);
  MPI_Win_shared_query(win, r, &sz, &dsp_unit, &myptr);
  if (myptr != mem) SETERRQ2(PETSC_COMM_SELF,1,"myptr %p != mem %p",myptr,mem);

  double *north_ReadMe = northptr,*south_ReadMe = southptr, *east_ReadMe = eastptr, *west_ReadMe = westptr;
  northptr += extrasp; southptr += extrasp; eastptr += extrasp; westptr += extrasp;

  int nNeighbors = 0;
  for (i=0; i<sp; i++) myptr[i] = -1.0;
  myptr[r] = 0.0;
  if (north != MPI_PROC_NULL) {
    nNeighbors++; myptr[north]=0.0;
  }
  if (south != MPI_PROC_NULL) {
    nNeighbors++; myptr[south]=0.0;
  }
  if (east != MPI_PROC_NULL) {
    nNeighbors++; myptr[east]=0.0;
  }
  if (west != MPI_PROC_NULL) {
    nNeighbors++; myptr[west]=0.0;
  }
  int sum = 0;
  //int icheck;
  for (i=0; i<sp; i++) {
    if (i != r && myptr[i]>-1.0) sum += 1;
  }
  if (sum != nNeighbors)SETERRQ2(PETSC_COMM_SELF,1,"icheck %d != nNeighbors %d",sum,nNeighbors);

  northptr2 = northptr+size;
  southptr2 = southptr+size;
  eastptr2 = eastptr+size;
  westptr2 = westptr+size;

  // initialize three heat sources
  #define nsources 3
  int sources[nsources][2] = {{n/2,n/2}, {n/3,n/3}, {n*4/5,n*8/9}};
  int locnsources=0; // number of sources in my area
  int locsources[nsources][2]; // sources local to my rank
  for (i=0; i<nsources; ++i) { // determine which sources are in my patch
    int locx = sources[i][0] - offx;
    int locy = sources[i][1] - offy;
    if(locx >= 0 && locx < bx && locy >= 0 && locy < by) {
      locsources[locnsources][0] = locx+1; // offset by halo zone
      locsources[locnsources][1] = locy+1; // offset by halo zone
      locnsources++;
    }
  }

  double t=-MPI_Wtime(); // take time
  double heat; // total heat in system
  MPI_Win_lock_all(0, win);
  for (iter=0; iter<niters; iter++) {
    //===============================
    // refresh heat sources
    for(i=0; i<locnsources; ++i) {
      aold[ind(locsources[i][0],locsources[i][1])] += energy; // heat source
    }

    int iwhile = 0;
    while (1) {
      sum=0;
      for (i=0; i<sp; i++) {
        if (i != r) {
          if (myptr[i] == (int)iter) {
            sum += 1;
          } else if (myptr[i] >-1.0) {
            printf("[%d] error iter %d, myptr[%d] %g\n",r,iter, i,myptr[i]);
          }
        }
      }

      if (sum == nNeighbors) {
        //if (iwhile) printf("[%d] iter %d break iwhile %d\n",r,iter,iwhile);
        break; //while-loop
      } else if (sum > nNeighbors) {
        for (i=0; i<sp; i++) {
          if (myptr[i] != 0.0) printf(" (%d,%g), ",i,myptr[i]);
        }
        SETERRQ3(PETSC_COMM_SELF,1,"iter %d sum %d > nNeighbors %d",iter,sum,nNeighbors);
      }
      printf("[%d] iter %d sum %d != nNeighbors %d\n",r,iter,sum,nNeighbors);
      iwhile++;
    }
    if (sum != nNeighbors) SETERRQ2(PETSC_COMM_SELF,1,"sum %d != nNeighbors %d",sum,nNeighbors);
#if 0
    for (i=0; i<sp; i++) {
      if (i != r) {
        myptr[i] = 0.0;
      } else myptr[i] = (double)iter; 
    }
#endif
    //MPI_Win_fence(0,win);
    //MPI_Win_sync(win);
    MPI_Barrier(shmcomm);

    // exchange data with neighbors
    if(north != MPI_PROC_NULL) {
      int iwhile = 0;
      while (iwhile<iwhile_max) {
        if ((int)north_ReadMe[north] == iter) {
          //printf("[%d] north_ReadMe[%d] %g == iter %d, break iwhile %d\n",r,north,north_ReadMe[north],iter,iwhile);
          break;
        } else if ((int)north_ReadMe[north]>iter) {
          SETERRQ2(PETSC_COMM_SELF,1,"north_ReadMe[north] %g > iter %d",north_ReadMe[north],iter );
        } else { //printf("[%d] north_ReadMe[%d] %g != iter %d iwhile %d\n",r,north,north_ReadMe[north],iter,iwhile);
        }
        //ierr = PetscSleep(.1);CHKERRQ(ierr);
        iwhile++;
      }
      if (iwhile == iwhile_max) SETERRQ2(PETSC_COMM_SELF,1,"[%d] iwhile_max %d is reached for north",r,iwhile_max);
      if (north_ReadMe[north] != iter) SETERRQ2(PETSC_COMM_SELF,1,"north_ReadMe %g != iter %d",north_ReadMe[north],iter);
      for(i=0; i<bx; ++i) aold[ind(i+1,0)] = northptr2[ind(i+1,by)]; // pack loop - last valid region
      north_ReadMe[r] += 1.0;
    }
    if(south != MPI_PROC_NULL) {
      int iwhile = 0;
      while (iwhile<iwhile_max) {
        if ((int)south_ReadMe[south] == iter) {
          break;
        } else if ((int)south_ReadMe[south]>iter) {
          SETERRQ2(PETSC_COMM_SELF,1,"south_ReadMe[south] %g > iter %d",south_ReadMe[south],iter );
        } else {//printf("[%d] south_ReadMe[%d] %g != iter %d iwhile %d\n",r,south,south_ReadMe[south],iter,iwhile);
        }
        //ierr = PetscSleep(.1);CHKERRQ(ierr);
        iwhile++;
      }
      if (iwhile == iwhile_max) SETERRQ2(PETSC_COMM_SELF,1,"[%d] iwhile_max %d is reached for south",r,iwhile_max);
      if (south_ReadMe[south] != iter) SETERRQ2(PETSC_COMM_SELF,1,"south_ReadMe %g != iter %d",south_ReadMe[south],iter);
      for(i=0; i<bx; ++i) aold[ind(i+1,by+1)] = southptr2[ind(i+1,1)]; // pack loop
      south_ReadMe[r] += 1.0;
    }
    if(east != MPI_PROC_NULL) {
      int iwhile = 0;
      while (iwhile<iwhile_max) {
        if ((int)east_ReadMe[east] == iter) {
          break;
        } else if ((int)east_ReadMe[east]>iter) {
          SETERRQ2(PETSC_COMM_SELF,1,"south_ReadMe[east] %g > iter %d",east_ReadMe[east],iter );
        } else { //printf("[%d] east_ReadMe[%d] %g != iter %d iwhile %d\n",r,east,east_ReadMe[east],iter,iwhile);
        }
        //ierr = PetscSleep(.1);CHKERRQ(ierr);
        iwhile++;
      }
      if (iwhile == iwhile_max) SETERRQ2(PETSC_COMM_SELF,1,"[%d] iwhile_max %d is reached for east",r,iwhile_max);
      if (east_ReadMe[east] != iter) SETERRQ2(PETSC_COMM_SELF,1,"east_ReadMe %g != iter %d",east_ReadMe[east],iter);
      for(i=0; i<by; ++i) aold[ind(bx+1,i+1)] = eastptr2[ind(1,i+1)]; // pack loop
      east_ReadMe[r] += 1.0;
    }
    if(west != MPI_PROC_NULL) {
      int iwhile = 0;
      while (iwhile<iwhile_max) {
        if ((int)west_ReadMe[west] == iter) {
          break;
        } else if ((int)west_ReadMe[west]>iter) {
          SETERRQ2(PETSC_COMM_SELF,1,"west_ReadMe[west] %g > iter %d",west_ReadMe[west],iter );
        } else {//printf("[%d] west_ReadMe[%d] %g != iter %d iwhile %d\n",r,west,west_ReadMe[west],iter,iwhile);
        }
        //ierr = PetscSleep(.1);CHKERRQ(ierr);
        iwhile++;
      }
      if (iwhile == iwhile_max) SETERRQ2(PETSC_COMM_SELF,1,"[%d] iwhile_max %d is reached for west",r,iwhile_max);
      if (west_ReadMe[west] != iter) SETERRQ2(PETSC_COMM_SELF,1,"west_ReadMe %g != iter %d",west_ReadMe[west],iter);

      for(i=0; i<by; ++i) aold[ind(0,i+1)] = westptr2[ind(bx,i+1)]; // pack loop
      west_ReadMe[r] += 1.0;
    }

    // update grid points
    heat = 0.0;
    for(j=1; j<by+1; ++j) {
      for(i=1; i<bx+1; ++i) {
        anew[ind(i,j)] = aold[ind(i,j)]/2.0 + (aold[ind(i-1,j)] + aold[ind(i+1,j)] + aold[ind(i,j-1)] + aold[ind(i,j+1)])/4.0/2.0;
        heat += anew[ind(i,j)];
      }
    }

    // swap arrays
    tmp=anew; anew=aold; aold=tmp;
    tmp=northptr; northptr=northptr2; northptr2=tmp;
    tmp=southptr; southptr=southptr2; southptr2=tmp;
    tmp=eastptr; eastptr=eastptr2; eastptr2=tmp;
    tmp=westptr; westptr=westptr2; westptr2=tmp;

    // optional - print image
    //if(iter == niters-1) printarr_par(iter, anew, n, px, py, rx, ry, bx, by, offx, offy, comm);
    myptr[r] = iter + 1.0;
  }
  MPI_Win_unlock_all(win);
  t+=MPI_Wtime();

  // get final heat in the system
  double rheat;
  MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, comm);
  if(!r) printf("[%i] last heat: %f time: %f\n", r, rheat, t);

  MPI_Win_free(&win);
  MPI_Comm_free(&shmcomm);

  //MPI_Finalize();
  ierr = PetscFinalize();
  return ierr;
}
