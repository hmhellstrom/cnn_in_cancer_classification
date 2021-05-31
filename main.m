tic
POSITIVES_DIR = "C:\Users\henri\Desktop\Koulu\Gradu\data\Seafile\anom_data\positiiviset";
MRIS_POS = find_images(POSITIVES_DIR, "nifti_mri", ".img");
PETS_POS = find_images(POSITIVES_DIR, "nifti_pet", ".img");
MASKS = find_images(POSITIVES_DIR, "maski", ".img");

NEGATIVES_DIR = "C:\Users\henri\Desktop\Koulu\Gradu\data\Seafile\anom_data\negatiiviset";
MRIS_NEG = find_images(NEGATIVES_DIR, "nifti_mri", ".img");
PETS_NEG = find_images(NEGATIVES_DIR, "nifti_pet", ".img");

num_cancer = [];
positives_data = {};
slice_nums = [];

for idx = 1:length(MRIS_POS)
  mri = niftiread(MRIS_POS(idx));
  pet = niftiread(PETS_POS(idx));
  mask = niftiread(MASKS(idx));
  contains_cancer = [];
  dims = size(mask);
  for slice = 1:dims(3)
    contains_cancer = [contains_cancer, sum(mask(:, :, slice), 'all')];
    if sum(mask(:, :, slice), 'all') > 0
        positives_data = [positives_data, cat(3, pet(:, :, slice), zscore(double(mri(:, :, slice))), zeros(size(mri(:,:,slice))))];
    end
  end
  slice_nums = [slice_nums, find(contains_cancer)];
  num_cancer = [num_cancer, sum(contains_cancer > 0)];
end

labels = ones(1, length(positives_data));

negatives_data = {};

for idx = 1:length(MRIS_NEG)
    mri = niftiread(MRIS_NEG(idx));
    pet = niftiread(PETS_NEG(idx));
    ROI = round(mean(slice_nums))-round(mean(num_cancer))/2:round(mean(slice_nums))+round(mean(num_cancer))/2;
    selected = ROI(randperm(length(ROI),4));
    for slice = selected
        negatives_data = [negatives_data, cat(3, pet(:, :, slice), zscore(double(mri(:, :, slice))), zeros(size(mri(:,:,slice))))];
    end
end

labels = [labels, zeros(1,length(negatives_data))];
pet_mri_data = [positives_data, negatives_data];

[train, test] = train_test_split(labels, 0.25, true);

for idx = 1:length(train)
    if labels(train(idx)) == 0
      imwrite(imresize(pet_mri_data{train(idx)},[224,224]),fullfile("C:\Users\henri\Desktop\Koulu\Gradu\data\MRI_PET\N",sprintf("pet_mri_%i.png", train(idx))))
    else
      imwrite(imresize(pet_mri_data{train(idx)},[224,224]),fullfile("C:\Users\henri\Desktop\Koulu\Gradu\data\MRI_PET\P",sprintf("pet_mri_%i.png", train(idx))))
    end
end

for idx = 1:length(test)
    imwrite(imresize(pet_mri_data{test(idx)},[224,224]),fullfile("C:\Users\henri\Desktop\Koulu\Gradu\data\MRI_PET_TEST",sprintf("pet_mri_%i.png", train(idx))))
end

toc