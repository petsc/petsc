function PetscBinaryWrite(filename,A,b)
%
%  Writes in PETSc binary file a matrix and optionally
%  vector
%
[m,n] = size(A);
if ~issparse(A)
  A = sparse(A);
end
majic = 1.2345678910e-30;
for i=1:min(m,n)
  if A(i,i) == 0
  A(i,i) = majic;
  end
end

fd = fopen(filename,'w','ieee-be');

nz   = nnz(A);
header = fwrite(fd,[1211216,m,n,nz],'int32');
n_nz = sum(A' ~= 0);
fwrite(fd,n_nz,'int32');  %nonzeros per row
[i,j,s] = find(A');
fwrite(fd,i-1,'int32');
for i=1:nz
  if s(i) == majic
    s(i) = 0;
  end
end
fwrite(fd,s,'double');

if nargin == 3
  [m,n] = size(b);
  fwrite(fd,[1211214,m],'int32');
  fwrite(fd,b,'double');
end
fclose(fd);
