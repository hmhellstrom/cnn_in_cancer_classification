
test_labels = labels(test);

predictions = [];
files = dir("C:\Users\henri\Desktop\Koulu\Gradu\data\MRI_PET_TEST");
files = {files.name};
files(1:2) = [];
for item = 1:length(files)
    img = imread(fullfile("C:\Users\henri\Desktop\Koulu\Gradu\data\MRI_PET_TEST",files(item)));
    predictions = [predictions, classify(trainedNetwork_2,img)];
end

new_pred = [];
for label = predictions
    if strcmp(char(label), "N")
        new_pred = [new_pred, 0];
    else
        new_pred = [new_pred, 1];
    end
end
predictions = new_pred;

        

length(predictions)
correct = 0;
for i = 1:length(predictions)
    if predictions(i) == test_labels(i)
        correct = correct + 1;
    end
end

correct/length(predictions)

C = confusionmat(test_labels,predictions)

[X, Y]= perfcurve(test_labels,predictions,1);
plot(X,Y)
trapz(X,Y)