load datafile

c1 = datafile(:,1);
c2 = datafile(:,2);
c3 = datafile(:,3);

r = sin(2*pi*c1).*sin(2*pi*c2);
d = r - c3;

max(abs(d))

figure(1)
plot(d)

figure(2);
plot3(c1,c2,c3,'.');
