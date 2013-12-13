      program calfun
      integer nmax, mmax, nfmax, runmax, lmax
      parameter (nmax=100)
      parameter (mmax=1000)
      parameter (nfmax=1500)
      parameter (runmax=100)
      parameter (lmax=(RUNMAX+11)*(RUNMAX+NMAX)+NMAX*(3*NMAX+11)/2)

      integer np,m,n,i
      double precision fvec(mmax), x(nmax)
      open(201,file='calfun.dat',status='old')
      read(201,*) nprob,n,m
      do i=1,n
         read(201,*) x(i)
      enddo
      close(201)
      call dfovec(m,n,x,fvec,nprob)
      open(202,file='calfun.out',status='unknown')
      do i=1,m
         write(202,*) fvec(i)
      enddo
      close(202)

      end
