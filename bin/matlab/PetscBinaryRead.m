function [varargout] = PetscBinaryRead(file)
%
%  Reads in PETSc binary file matrices or vectors
%  emits as Matlab sparse matrice or vectors.
%
%  Argument may be file name (string) or matlab
%  file descriptor.
%
   
if ischar(file) fd = fopen(file,'r','ieee-be');
else            fd = file;
end
   
for l=1:nargout
  header = fread(fd,1,'int32');
  if isempty(header)
    disp('File does not have that many items')
    return
  end

  if header == 1211216 % Petsc Mat Object 
    header = fread(fd,3,'int32');
    m      = header(1);
    n      = header(2);
    nz     = header(3);
    nnz = fread(fd,m,'int32');  %nonzeros per row
    sum_nz = sum(nnz);
    if(sum_nz ~=nz)
      str = sprintf('No-Nonzeros sum-rowlengths do not match %d %d',nz,sum_nz);
      error(str);
    end
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
    varargout(l) = {A};
  end
  
  if  header == 1211214 % Petsc Vec Object
    m = fread(fd,1,'int32');
    v = fread(fd,m,'double');
    varargout(l) = {v};
  end

  if header == 1211219 % Petsc Bag Object
     b = PetscBagRead(fd);
     varargout(l) = {b};
  end
end
if ischar(file) fclose(fd); end;
