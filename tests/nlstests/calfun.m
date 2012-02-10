function F=calfun(x)
parameters.m;

datfile = fopen('calfun.dat','w')
fprintf(datfile,'%d %d %d\n',np,n,m);  '%
for i=1:n
	fprintf(datfile,'%20.12g\n',x(i)); '%
end
fclose(datfile)

unix('./calfun');

fid = fopen('calfun.out','r');
Fvec=fscanf(fid,'%e');
fclose(fid);

F = Fvec';

