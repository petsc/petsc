c
c  Local arrays for function components due to boundary conditions.
c  We could eventually upgrade the routine residbc() to place
c  the function components in the residual vector and thus alleviate
c  the need for these arrays.  
c
      Double fbcrj1(xsf2:xef01,zsf2:zef01)
      Double fbcruj1(xsf2:xef01,zsf2:zef01)
      Double fbcrvj1(xsf2:xef01,zsf2:zef01)
      Double fbcrwj1(xsf2:xef01,zsf2:zef01)
      Double fbcej1(xsf2:xef01,zsf2:zef01)
      Double fbcrj2(xsf2:xef01,zsf2:zef01)
      Double fbcruj2(xsf2:xef01,zsf2:zef01)
      Double fbcrvj2(xsf2:xef01,zsf2:zef01)
      Double fbcrwj2(xsf2:xef01,zsf2:zef01)
      Double fbcej2(xsf2:xef01,zsf2:zef01)
      Double fbcrk1(xsf2:xef01,ysf2:yef01)
      Double fbcruk1(xsf2:xef01,ysf2:yef01)
      Double fbcrvk1(xsf2:xef01,ysf2:yef01)
      Double fbcrwk1(xsf2:xef01,ysf2:yef01)
      Double fbcek1(xsf2:xef01,ysf2:yef01)
      Double fbcrk2(xsf2:xef01,ysf2:yef01)
      Double fbcruk2(xsf2:xef01,ysf2:yef01)
      Double fbcrvk2(xsf2:xef01,ysf2:yef01)
      Double fbcrwk2(xsf2:xef01,ysf2:yef01)
      Double fbcek2(xsf2:xef01,ysf2:yef01)
      Double fbcri1(ysf2:yef01,zsf2:zef01)
      Double fbcrui1(ysf2:yef01,zsf2:zef01)
      Double fbcrvi1(ysf2:yef01,zsf2:zef01)
      Double fbcrwi1(ysf2:yef01,zsf2:zef01)
      Double fbcei1(ysf2:yef01,zsf2:zef01)
      Double fbcri2(ysf2:yef01,zsf2:zef01)
      Double fbcrui2(ysf2:yef01,zsf2:zef01)
      Double fbcrvi2(ysf2:yef01,zsf2:zef01)
      Double fbcrwi2(ysf2:yef01,zsf2:zef01)
      Double fbcei2(ysf2:yef01,zsf2:zef01)

c      COMMON /resbcj1/ fbcrj1(d_ni,1,d_nk),fbcruj1(d_ni,1,d_nk)
c      COMMON /resbcj1/ fbcrvj1(d_ni,1,d_nk)
c      COMMON /resbcj1/ fbcrwj1(d_ni,1,d_nk),fbcej1(d_ni,1,d_nk)
c      COMMON /resbcj2/ fbcrj2(d_ni,1,d_nk),fbcruj2(d_ni,1,d_nk)
c      COMMON /resbcj2/ fbcrvj2(d_ni,1,d_nk)
c      COMMON /resbcj2/ fbcrwj2(d_ni,1,d_nk),fbcej2(d_ni,1,d_nk)
c      COMMON /resbck1/ fbcrk1(d_ni,d_nj,1),fbcruk1(d_ni,d_nj,1)
c      COMMON /resbck1/ fbcrvk1(d_ni,d_nj,1)
c      COMMON /resbck1/ fbcrwk1(d_ni,d_nj,1),fbcek1(d_ni,d_nj,1)
c      COMMON /resbck2/ fbcrk2(d_ni,d_nj,1),fbcruk2(d_ni,d_nj,1)
c      COMMON /resbck2/ fbcrvk2(d_ni,d_nj,1)
c      COMMON /resbck2/ fbcrwk2(d_ni,d_nj,1),fbcek2(d_ni,d_nj,1)
c      COMMON /resbci1/ fbcri1(1,d_nj,d_nk),fbcrui1(1,d_nj,d_nk)
c      COMMON /resbci1/ fbcrvi1(1,d_nj,d_nk)
c      COMMON /resbci1/ fbcrwi1(1,d_nj,d_nk),fbcei1(1,d_nj,d_nk)
c      COMMON /resbci2/ fbcri2(1,d_nj,d_nk),fbcrui2(1,d_nj,d_nk)
c      COMMON /resbci2/ fbcrvi2(1,d_nj,d_nk)
c      COMMON /resbci2/ fbcrwi2(1,d_nj,d_nk),fbcei2(1,d_nj,d_nk)
c      Double fbcrj1,fbcruj1,fbcrvj1,fbcrwj1,fbcej1
c      Double fbcrj2,fbcruj2,fbcrvj2,fbcrwj2,fbcej2
c      Double fbcrk1,fbcruk1,fbcrvk1,fbcrwk1,fbcek1
c      Double fbcrk2,fbcruk2,fbcrvk2,fbcrwk2,fbcek2
c      Double fbcri1,fbcrui1,fbcrvi1,fbcrwi1,fbcei1
c      Double fbcri2,fbcrui2,fbcrvi2,fbcrwi2,fbcei2
