function image = find_images(start_dir, file_contains, ext)
image = [];
folders = dir(start_dir);
folders = {folders.name};
folders(1:2) = [];
for patient = folders
  for folder = dir(strcat(start_dir, "\", char(patient)))
    subfolders = {folder.name};
    subfolders(1:2) = [];
    for subfolder = subfolders
      if contains(char(subfolder), file_contains, 'IgnoreCase', true)
        files = dir(strcat(start_dir, "\", char(patient), "\", char(subfolder)));
        files = {files.name};
        files(1:2) = [];
        for file = files
          if contains(file, ext)
            image = [image, strcat(start_dir, "\", char(patient), "\", char(subfolder), "\", file)];
          end
        end
      end
    end
  end
end