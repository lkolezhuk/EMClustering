close all;
clear all;
clc;

num_class = 3;

im_dataT1 = load_untouch_nii('DATA/1/T1.nii');
im_dataT2 = load_untouch_nii('DATA/1/T2_Flair.nii');

imagesT1 = im_dataT1.img;
imagesT2 = im_dataT2.img;

imageT1 = double(imagesT1(:,:,20));
imageRAW = imageT1;
imageT2 = double(imagesT2(:,:,20));

 figure; imshow(imageT1,[]);
 imageT1 = imageT1(imageT1(:) > 0);
 
 image = imageT1;
 
 idx = kmeans(imageT1(:), num_class);
 for i=1:num_class
    m(i) = mean(image(idx == i));
    s(i) = cov(image(idx == i));
 end
 
 h = histogram(image(:));
 h = h.Data;
 image_linearized = h;
 
for i = 1:size(image_linearized, 1)
    for c = 1:num_class
        p(i,c) = exp(-0.5 * (image_linearized(i) - m(c))' * inv(s(c)) * (image_linearized(i) - m(c)))/(2*pi*sqrt(s(c)));
    end
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
        for c = 1:num_class
            p(i,c) = exp(-0.5 * (image_linearized(i) - m(c))' * inv(s(c)) * (image_linearized(i) - m(c)))/(2*pi*sqrt(s(c)));
        end 
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
mask = zeros(size(imageRAW,1),size(imageRAW,2));

for i=1:size(imageRAW,1)
   for j=1:size(imageRAW,2)
      for n = 1:num_class
            point_p(n) = exp(-0.5 * (imageRAW(i,j) - m(n))' * inv(s(n)) * (imageRAW(i,j) - m(n)))/(2*pi*sqrt(s(n)));
      end
      
      a = find(point_p == max(point_p));
      mask(i,j) = a(1);
   end
end