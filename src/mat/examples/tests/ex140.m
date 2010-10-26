%
%  Tests the results from ex140.c
%
path(path,[getenv('PETSC_DIR') '/bin/matlab'])
d = 2;
[b1,A1,is,b2,A2 ] = PetscBinaryRead('binaryoutput');
D = diag(A1);
A1(:,is) = 0;
A1(is,:) = 0;
for i=1:length(D)
  A1(i,i) = D(i);
end
for i=1:length(is)
  A1(is(i),is(i)) = d;
end
A1-A2
b2(is) - 2


