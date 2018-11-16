close all;
clear all;
clc;

im_dataT1 = load_untouch_nii('DATA/1/T1.nii');
im_dataT2 = load_untouch_nii('DATA/1/T2_Flair.nii');

imagesT1 = im_dataT1.img;
imagesT2 = im_dataT2.img;

imageT1 = double(imagesT1(:,:,20));
image = imageT1;
imageT2 = double(imagesT2(:,:,20));


imageT1 = imageT1(imageT1(:) > 0);
imageT2 = imageT2(imageT2(:) > 0);

imageT1 = imageT1(:);
imageT2 = imageT2(:);

maxT1 = max(max(imageT1));
maxT2 = max(max(imageT2));

h2d = zeros(maxT1, maxT2);

 for i = 1: size(imageT1,1)
    h2d(imageT1(i), imageT2(i)) = h2d(imageT1(i), imageT2(i)) + 1; 
 end
 
 data = h2d;
 
 num_class = 3;
 idx = kmeans(data(:), 3);
 m1 = mean(data(idx == 1));
 m2 = mean(data(idx == 2));
 m3 = mean(data(idx == 3));

 m = [m1;m2;m3];
 s1 = cov(data(idx == 1));
 s2 = cov(data(idx == 2));
 s3 = cov(data(idx == 3));
 s = [s1; s2; s3];
 
 h = histogram(data(:));
 h = h.Data;
 image_linearized = h;
 
for i = 1:size(image_linearized, 1)
        p(i,1) = exp(-0.5 * (image_linearized(i) - m(1))' * inv(s(1)) * (image_linearized(i) - m(1)))/(2*pi*sqrt(s(1)));
        p(i,2) = exp(-0.5 * (image_linearized(i) - m(2))' * inv(s(2)) * (image_linearized(i) - m(2)))/(2*pi*sqrt(s(2)));
        p(i,3) = exp(-0.5 * (image_linearized(i) - m(3))' * inv(s(3)) * (image_linearized(i) - m(3)))/(2*pi*sqrt(s(3)));
        sump(i) = sum(p(i,:));
end
conv_likelihood(1) = sum(h .* log(sump'));

iter = 1;
while(1)
    iter = iter + 1;
     
     for j=1:num_class
        temp = h .* p(:,j)./sump';
        mem_weights(j) = sum(temp);
        m(j) = sum(h.*temp)/mem_weights(j);
     end
     
     for i = 1:size(image_linearized, 1)
        p(i,1) = exp(-0.5 * (image_linearized(i) - m(1))' * inv(s(1)) * (image_linearized(i) - m(1)))/(2*pi*sqrt(s(1)));
        p(i,2) = exp(-0.5 * (image_linearized(i) - m(2))' * inv(s(2)) * (image_linearized(i) - m(2)))/(2*pi*sqrt(s(2)));
        p(i,3) = exp(-0.5 * (image_linearized(i) - m(3))' * inv(s(3)) * (image_linearized(i) - m(3)))/(2*pi*sqrt(s(3)));
        sump(i) = sum(p(i,:));
     end
     conv_likelihood(iter) = sum(h .* log(sump'));

     convergence_achieved = (conv_likelihood(iter) - conv_likelihood(iter - 1)) < 0.0001;
     if convergence_achieved
         disp(iter);
         break;
     end
end
 %%
mask = zeros(size(image,1),size(image,2));

for i=1:size(image,1)
   for j=1:size(image,2)
      for n = 1:num_class
            point_p(n) = exp(-0.5 * (image(i,j) - m(n))' * inv(s(n)) * (image(i,j) - m(n)))/(2*pi*sqrt(s(n)));
      end
      
      a = find(point_p == max(point_p));
      mask(i,j) = a(1);
   end
end