close all;
clear all;
clc;

 num_class = 3;

 im_dataT1 = load_untouch_nii('DATA/5/T1.nii');
 im_dataT2 = load_untouch_nii('DATA/5/T2_FLAIR.nii');
 
 tic;
 imagesT1 = im_dataT1.img;
 imageT1 = double(imagesT1(:,:,:));
 imagesT2 = im_dataT2.img;
 imageT2 = double(imagesT2(:,:,:));
 image_labels_data = load_untouch_nii('DATA/5/LabelsForTesting.nii');
 image_labels = image_labels_data.img(:,:,:);
 
 zeros_idx = find(image_labels(:,:,:) == 0);
 imageT1(zeros_idx) = 0;
 imageT2(zeros_idx) = 0;
 
 imageRAW = imageT1(:,:,:);
 imageT1 = imageT1(:,:,15:20);
 imageT2 = imageT2(:,:,15:20);
 
 
 
 imageT1 = imageT1(:);
 imageT2 = imageT2(:);
 
%   figure; imshow(imageT1,[]);
  imageT1 = imageT1(imageT1(:) > 0);
  imageT2 = imageT2(imageT2(:) > 0);  
  image = [imageT1 imageT2];
 
 idx = kmeans(image(:), num_class);
 for i=1:num_class
    m(i) = mean(image(idx == i));
    s(i) = cov(image(idx == i));
 end
 [m sortidx] = sort(m);
 s = s(sortidx);
 
 image_linearized = image;
 h = image;
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

 
mask = zeros(size(imageRAW,1),size(imageRAW,2));
imagesRAW = imageRAW;

for img_ind=1:size(imagesRAW,3)
    imageRAW = imagesRAW(:, :, img_ind);
    for i=1:size(imagesRAW,1)
       for j=1:size(imagesRAW,2)
          for n = 1:num_class
                point_p(n) = exp(-0.5 * (imageRAW(i,j) - m(n))' * inv(s(n)) * (imageRAW(i,j) - m(n)))/(2*pi*sqrt(s(n)));
          end
          if(imageRAW(i,j) == 0)
             mask(i,j) = 0;
          else
              a = find(point_p == max(point_p));
              mask(i,j) = a(1);
          end
       end
    end

    figure; imshow(mask,[]);
    title('Segmentation result');


    %%
    Segmentation_Im = mask;
    Label_Im = double(image_labels(:,:,img_ind));

    VoxelsNumber1=numel(Segmentation_Im); 
    VoxelsNumber2=numel(Label_Im);
    CommonArea=sum(sum(sum(Segmentation_Im == Label_Im))); 
    Dice(img_ind)=(2*CommonArea)./(VoxelsNumber1+VoxelsNumber2);
end
execution_time = toc;