      character*30 function dfonm(nprob)
      integer nprob
c     **********
c
c     Subroutine dfonm
c
c     This subroutine specifies the names of the 22 nonlinear 
c     benchmark problems in
c
c     Benchmarking Derivative-Free Optimization Algorithms
c     Jorge J. More' and Stefan M. Wild
c     Mathematics and Computer Science Division
c     Preprint ANL/MCS-P1471-1207, December 2007
c
c     Argonne National Laboratory. 
c     Jorge More' and Stefan Wild. January 2008.
c
c     **********

      if (nprob .eq. 1) then
         dfonm = 'Linear, full rank'
      else if (nprob .eq. 2) then
         dfonm = 'Linear, rank 1'
      else if (nprob .eq. 3) then
         dfonm = 'Linear, rank 1, zero row/cols'
      else if (nprob .eq. 4) then
         dfonm = 'Rosenbrock'
      else if (nprob .eq. 5) then
         dfonm = 'Helical Valley'
      else if (nprob .eq. 6) then
         dfonm = 'Powell singular'
      else if (nprob .eq. 7) then
         dfonm = 'Freudenstein and Roth'
      else if (nprob .eq. 8) then
         dfonm = 'Bard'
      else if (nprob .eq. 9) then
         dfonm = 'Kowalik and Osborne'
      else if (nprob .eq. 10) then
         dfonm = 'Meyer'
      else if (nprob .eq. 11) then
         dfonm = 'Watson'
      else if (nprob .eq. 12) then
         dfonm = 'Box 3-dimensional'
      else if (nprob .eq. 13) then
         dfonm = 'Jenrich and Sampson'
      else if (nprob .eq. 14) then
         dfonm = 'Brown and Dennis'
      else if (nprob .eq. 15) then
         dfonm = 'Chebyquad'
      else if (nprob .eq. 16) then
         dfonm = 'Brown almost-linear'
      else if (nprob .eq. 17) then
         dfonm = 'Osborne 1'
      else if (nprob .eq. 18) then
         dfonm = 'Osborne 2'
      else if (nprob .eq. 19) then
         dfonm = 'Bdqrtic'
      else if (nprob .eq. 20) then
         dfonm = 'Cube'
      else if (nprob .eq. 21) then
         dfonm = 'Mancino'
      else if (nprob .eq. 22) then
         dfonm = 'Heart 8'
      else 
         dfonm = 'Error'
      end if

      end
