
for i=1:m,
  ss1(:,i) = s1(cellx(:,i), celly(:,i));
  ss2(:,i) = s2(cellx(:,i), celly(:,i));
end

figure(3)
fill3(cellx,celly,ss1,ss1)
%
figure(4)
fill3(cellx,celly,ss2,ss2)

d1 = max(abs(ss1-cellz1));
d2= max(abs(ss2-cellz2));
l1 = max(d1)
l2 = max(d2)
