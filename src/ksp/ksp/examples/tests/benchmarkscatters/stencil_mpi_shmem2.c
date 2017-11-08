/*
 * Copyright (c) 2012 Torsten Hoefler. All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@illinois.edu>
 *
 */

#include "./stencil_par.h"

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv); 
  //ierr = PetscInitialize(&argc,&argv,(char*)0,0);if (ierr) return ierr;
  int r,p;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &p);
  int i,j,iter,n, energy=25, niters=4;

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
  MPI_Win_allocate_shared((1+2*size)*sizeof(double), 1, MPI_INFO_NULL, shmcomm, &mem, &win);

  double *tmp;
  double *anew=mem; // each rank's offset
  double *aold=mem+size; // second half is aold!
  int *my_iter = (int*)(aold+size);
  *my_iter = -1;

  double *northptr, *southptr, *eastptr, *westptr;
  double *northptr2, *southptr2, *eastptr2, *westptr2;
  MPI_Aint sz;
  int dsp_unit;
  MPI_Win_shared_query(win, north, &sz, &dsp_unit, &northptr);
  MPI_Win_shared_query(win, south, &sz, &dsp_unit, &southptr);
  MPI_Win_shared_query(win, east, &sz, &dsp_unit, &eastptr);
  MPI_Win_shared_query(win, west, &sz, &dsp_unit, &westptr);
  northptr2 = northptr+size;
  southptr2 = southptr+size;
  eastptr2 = eastptr+size;
  westptr2 = westptr+size;

  // Using volatile to tell compilers to always load data from memory in while-loops below
  volatile int *north_iter = (int*)(northptr2 + size);
  volatile int *south_iter = (int*)(southptr2 + size);
  volatile int *east_iter = (int*)(eastptr2 + size);
  volatile int *west_iter = (int*)(westptr2 + size);

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

  MPI_Barrier(shmcomm); // make sure all procs initialized my_iter to -1

  for(iter=0; iter<niters; ++iter) {
    // refresh heat sources
    for(i=0; i<locnsources; ++i) {
      aold[ind(locsources[i][0],locsources[i][1])] += energy; // heat source
    }

    // make sure my_iter is written only when updated aold[] is visible to other processes
    asm volatile("" ::: "memory");

    *my_iter = iter;

    // exchange data with neighbors
    if(north != MPI_PROC_NULL) {
      // read from north neighbor only when it has advanced beyond iter-th iteration
      while(*north_iter < iter); // Compiler should not store *north_iter in registers
      for(i=0; i<bx; ++i) aold[ind(i+1,0)] = northptr2[ind(i+1,by)]; // pack loop - last valid region
    }
    if(south != MPI_PROC_NULL) {
      while(*south_iter < iter);
      for(i=0; i<bx; ++i) aold[ind(i+1,by+1)] = southptr2[ind(i+1,1)]; // pack loop
    }
    if(east != MPI_PROC_NULL) {
      while(*east_iter < iter);
      for(i=0; i<by; ++i) aold[ind(bx+1,i+1)] = eastptr2[ind(1,i+1)]; // pack loop
    }
    if(west != MPI_PROC_NULL) {
      while(*west_iter < iter);
      for(i=0; i<by; ++i) aold[ind(0,i+1)] = westptr2[ind(bx,i+1)]; // pack loop
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
  }
  t+=MPI_Wtime();

  // get final heat in the system
  double rheat;
  MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, comm);
  if(!r) printf("[%i] last heat: %f time: %f\n", r, rheat, t);

  MPI_Win_free(&win);
  MPI_Comm_free(&shmcomm);

  MPI_Finalize();
  return 0;
}
