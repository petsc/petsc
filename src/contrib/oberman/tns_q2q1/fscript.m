
 % now extract the solution at each point
anlsol1 = zeros(nv,N);
anlsol2 = zeros(nv,N);
anlsol3 = zeros(np,N);

  for i=1:N,
    for j=1:nv
      x = cellx(j,i);
      y = celly(j,i);
      anlsol1(j,i) = s1(x,y);
      anlsol2(j,i) = s2(x,y);
    end
  end
%
  figure(4)
  fill3(cellx,celly,anlsol1,anlsol1)
%
  figure(5)
  fill3(cellx,celly,anlsol2,anlsol2)
%
   for j=1:nv
      x = cellx(j,i);
      y = celly(j,i);
      tt(j) = s1(x,y);
      tt(j) = s2(x,y);
    end