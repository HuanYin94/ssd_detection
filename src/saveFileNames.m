file = dir('/home/yh/TL/*.jpg');
fid = fopen('/home/yh/caffe/fileNames.txt', 'w');
for i = 1 : 1 : length(file)
    f = file(i).name;
    d = '/home/yh/TL/';
    fd = [d, f];
    fprintf(fid, '%s\n', fd);    
end
fclose(fid);