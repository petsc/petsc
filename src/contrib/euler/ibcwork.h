c
c  Local function components due to boundary conditions
c  
      double precision 
     & fff(ndof,xsf1:xefp1,ysf1:yefp1,zsf1:zefp1)

#define fbcrk1(i,j) fff(1,i,j,1)
#define fbcruk1(i,j) fff(2,i,j,1)
#define fbcrvk1(i,j) fff(3,i,j,1)
#define fbcrwk1(i,j) fff(4,i,j,1)
#define fbcek1(i,j) fff(5,i,j,1)

#define fbcfpk1(i,j) fff(ndof,i,j,1)

#define fbcrk2(i,j) fff(1,i,j,nk1_boundary)
#define fbcruk2(i,j) fff(2,i,j,nk1_boundary)
#define fbcrvk2(i,j) fff(3,i,j,nk1_boundary)
#define fbcrwk2(i,j) fff(4,i,j,nk1_boundary)
#define fbcek2(i,j) fff(5,i,j,nk1_boundary)

#define fbcfpk2(i,j) fff(ndof,i,j,nk1_boundary)

#define fbcrj1(i,k) fff(1,i,1,k)
#define fbcruj1(i,k) fff(2,i,1,k)
#define fbcrvj1(i,k) fff(3,i,1,k)
#define fbcrwj1(i,k) fff(4,i,1,k)
#define fbcej1(i,k) fff(5,i,1,k)

#define fbcfpj1(i,k) fff(ndof,i,1,k)

#define fbcrj2(i,k) fff(1,i,nj1,k)
#define fbcruj2(i,k) fff(2,i,nj1,k)
#define fbcrvj2(i,k) fff(3,i,nj1,k)
#define fbcrwj2(i,k) fff(4,i,nj1,k)
#define fbcej2(i,k) fff(5,i,nj1,k)

#define fbcfpj2(i,k) fff(ndof,i,nj1,k)

#define fbcri1(j,k) fff(1,1,j,k)
#define fbcrui1(j,k) fff(2,1,j,k)
#define fbcrvi1(j,k) fff(3,1,j,k)
#define fbcrwi1(j,k) fff(4,1,j,k)
#define fbcei1(j,k) fff(5,1,j,k)

#define fbcfpi1(j,k) fff(ndof,1,j,k)

#define fbcri2(j,k) fff(1,ni1,j,k)
#define fbcrui2(j,k) fff(2,ni1,j,k)
#define fbcrvi2(j,k) fff(3,ni1,j,k)
#define fbcrwi2(j,k) fff(4,ni1,j,k)
#define fbcei2(j,k) fff(5,ni1,j,k)

#define fbcfpi2(j,k) fff(ndof,ni1,j,k)
