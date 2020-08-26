%% 针对tof影像
% 数据路径：D:\bilateral-solver\others\ConsoleApplication1\ConsoleApplication1\sr_d

%% 1. 制作confidence影像
% 读取disp_sparse.txt
load('disp_sparse.txt');
figure
imagesc(disp_sparse)
max(disp_sparse(:))
min(disp_sparse(:))
[rows,cols] = size(disp_sparse)

confi=zeros(rows,cols);
confi(disp_sparse~=0)=1;
figure
imagesc(confi)
max(confi(:))
min(confi(:))

% 将confi写入txt
fid = fopen('confidence_disp_dot_e4.txt', 'wt');
mat = confi;
mat(mat==0.0)=0.00001;
for i = 1:size(mat, 1)
    fprintf(fid, '%f\t', mat(i,:));
    fprintf(fid, '\n');
end
fclose(fid);

%% 显示等高线
test=imread('chroma_8_luma_8_spatial_8_lam_256.png');
test=test(:,:,2);
minValue=min(test(:));
maxValue=max(test(:));
figure
imagesc(test,[10,130])

figure
contourf(test)

load('disp_sparse.txt');
figure
imshow(disp_sparse)



