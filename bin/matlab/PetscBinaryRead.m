function [A,b] = PetscBinaryRead(filename)
%
%  Reads in PETSc binary file matrix and optionally
%  vector and emits as Matlab sparse matrix (and vector)
%
fd = fopen(filename,'r','ieee-be');
header = fread(fd,4,'int32');
if isempty(header)
  disp('File is empty')
  return
end
if header(1) ~= 1211216
  disp('File does not start with a PETSc binary matrix')
  return
end


m  = header(2);
n  = header(3);
nz = header(4);

nnz = fread(fd,m,'int32');  %nonzeros per row

j   = fread(fd,nz,'int32') + 1;
s   = fread(fd,nz,'double');
i   = ones(nz,1);
cnt = 1;
for k=1:m
  next = cnt+nnz(k)-1;
  i(cnt:next,1) = k*ones(nnz(k),1);
  cnt = next+1;
end
A = sparse(i,j,s,m,n,nz);

if nargout == 2
  header = fread(fd,2,'int32');
  if isempty(header)
    disp('File does not contain PETSc binary vector - 1')
    return
  end
  if header(1) ~= 1211214
    disp('File does not contain PETSc binary vector - 2')
    return
  end
  if header(2) ~= m
    disp('Vector size does not match matrix size')
    return
  end
  b = fread(fd,m,'double');
end
fclose(fd);
