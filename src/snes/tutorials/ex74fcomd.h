      integer            probnum, ihod, mx, my, neq, ientro, gorder
      parameter (neq = 3)
      double precision theta, pi, time,  zero
      double precision damfac
      double precision dt, dtmin, dtmax, dtgrow, tfinal, tplot, tcscal, hcscal

      double precision eigval, eigvec, rinv, roestt, fl, fr, deltau, alpha, xnumdif, froe

      double precision dx, xl0, kappa0, kappaa, kappab, visc0, erg0

      logical debug, dampit, wilson, dtcon, pcnew

      common /params/ mx, my, probnum, ihod, ientro, gorder

      common /func/ pi,zero, theta, dx, xl0, damfac, kappa0, kappaa, kappab, visc0, erg0

      common /gudnov/ eigval(neq), eigvec(neq,neq), rinv(neq,neq), roestt(neq), fl(neq), fr(neq), deltau(neq), alpha(neq), xnumdif (neq), froe(neq)

      common /flags/ debug, dampit, wilson, dtcon, pcnew

      common /timcnt/ time, dt, dtmin, dtmax, dtgrow, tfinal, tplot, tcscal, hcscal

