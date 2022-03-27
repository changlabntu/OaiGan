clear all
clc

a=double(readNPY('abml.npy'));
b=double(readNPY('bbml.npy'));
pd=readNPY('PD.npy');
pr=readNPY('PR.npy');
pl=readNPY('PL.npy');
p1=max([pl pr],[] ,2);
p2=min([pl pr],[] ,2);

plot(a-b, pd,'.')