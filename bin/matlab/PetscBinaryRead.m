function [varargout] = PetscBinaryRead(file,comp)
%
%  Reads in PETSc binary file matrices or vectors
%  emits as Matlab sparse matrice or vectors.
%
%  Argument may be file name (string) or matlab
%  file descriptor.
%
   
if nargin == 1
  comp = 0;
else
  comp = 1;
end

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
    if comp == 1
      s   = fread(fd,2*nz,'double');
    else 
      s   = fread(fd,nz,'double');
    end
    i   = ones(nz,1);
    cnt = 1;
    for k=1:m
      next = cnt+nnz(k)-1;
      i(cnt:next,1) = k*ones(nnz(k),1);
      cnt = next+1;
    end
    if comp == 1
      A = sparse(i,j,complex(s(1:2:2*nz),s(2:2:2*nz)),m,n,nz);
    else
      A = sparse(i,j,s,m,n,nz);
    end
    varargout(l) = {A};
  
  elseif  header == 1211214 % Petsc Vec Object
    m = fread(fd,1,'int32');
    if comp == 1
      v = fread(fd,2*m,'double');
      v = complex(v(1:2:2*m),v(2:2:2*m));
    else
      v = fread(fd,m,'double');
    end
    varargout(l) = {v};

  elseif header == 1211219 % Petsc Bag Object
    b = PetscBagRead(fd);
    varargout(l) = {b};

  else 
    disp('Found unrecongnized header in file. If your file contains complex numbers')
    disp(' then call PetscBinaryRead() with "complex" as the second argument')
    return
  end

end
if ischar(file) fclose(fd); end;
