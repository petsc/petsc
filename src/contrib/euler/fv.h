c
c  Work arrays used in routines resid(), nd(), and nd2()
c  Space is allocated in UserCreateEuler().
c
      double precision f1(5,gxsf1:xef01,gysf1:yef01,gzsf1:zef01)
      double precision g1(5,gxsf1:xef01,gysf1:yef01,gzsf1:zef01)
      double precision h1(5,gxsf1:xef01,gysf1:yef01,gzsf1:zef01)

c      common /fv/ f1(5,ni,nj,nk),g1(5,ni,nj,nk),h1(5,ni,nj,nk)
c      double precision f1,g1,h1
