tic
positives_dir = "C:\Users\henri\Desktop\Koulu\Gradu\data\Seafile\anom_data\positiiviset";
MRIS = find_images(positives_dir, "nifti_mri", ".img");
PETS = find_images(positives_dir, "nifti_pet", ".img");
MASKS = find_images(positives_dir, "maski", ".img");
data = {};
labels = [];
num_cancer = [];
for idx = 1:length(MRIS)
  mri = niftiread(MRIS(idx));
  pet = niftiread(PETS(idx));
  mask = niftiread(MASKS(idx));
  contains_cancer = [];
  dims = size(mask);
  for slice = 1:dims(3)
    contains_cancer = [contains_cancer, sum(mask(:, :, slice), 'all')];
    data = [data, cat(3, mri(:, :, slice), pet(:, :, slice))];
  end
  labels = [labels, contains_cancer > 0];
  num_cancer = [num_cancer, sum(contains_cancer > 0)];
end

[train, test] = train_test_split(labels, 0.25, true);


for idx = 1:length(train)
    if labels(train(idx)) == 0
      imwrite(imfuse(data{train(idx)}(:,:,1),data{train(idx)}(:,:,2)),fullfile("C:\Users\henri\Desktop\Koulu\Gradu\data\test_data\N",sprintf("mri_pet_%i.png", train(idx))))
    else
      imwrite(imfuse(data{train(idx)}(:,:,1),data{train(idx)}(:,:,2)),fullfile("C:\Users\henri\Desktop\Koulu\Gradu\data\test_data\P",sprintf("mri_pet_%i.png", train(idx))))
    end
end

for idx = 1:length(test)
    imwrite(imfuse(data{train(idx)}(:,:,1),data{train(idx)}(:,:,2)),fullfile("C:\Users\henri\Desktop\Koulu\Gradu\data\test_set",sprintf("mri_pet_%i.png", test(idx))))
end

toc