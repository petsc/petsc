c
c  Parallel arrays: Right and left eigenvectors (br and bl) and 
c  eigenvalues (be), computed in the routine rlvecs()
c
      double precision BR(5,5,gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision BL(5,5,gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision BE(5,5,gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)

c  Uniprocessor version
c
c      COMMON /ES/ BR(5,5,NI,NJ,NK),BL(5,5,NI,NJ,NK),BE(2,5,NI,NJ,NK)
c      double precision br, bl, be
c
