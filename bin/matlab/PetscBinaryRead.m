function [varargout] = PetscBinaryRead(filename)
%
%  Reads in PETSc binary file matrices or vectors
%  emits as Matlab sparse matrice or vectors
%
fd = fopen(filename,'r','ieee-be');
for l=1:nargout
  header = fread(fd,2,'int32');
  if isempty(header)
    disp('File does not have that many items')
    return
  end

  if header(1) == 1211216

    m      = header(2);
    header = fread(fd,2,'int32');
    n      = header(1);
    nz     = header(2);

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
  if  header(1) == 1211214
    m = header(2);
    v = fread(fd,m,'double');
    varargout(l) = {v};
  end
end
fclose(fd);
