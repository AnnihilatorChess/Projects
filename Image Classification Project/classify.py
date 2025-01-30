import argparse
import torch
from torch.utils.data import DataLoader
from utilities import validate_images
from dataset import ImagesDatasetClassify
import os


labels_dict = {
                0: 'book',
                1: 'bottle',
                2: 'car',
                3: 'cat',
                4: 'chair',
                5: 'computermouse',
                6: 'cup',
                7: 'dog',
                8: 'flower',
                9: 'fork',
                10: 'glass',
                11: 'glasses',
                12: 'headphones',
                13: 'knife',
                14: 'laptop',
                15: 'pen',
                16: 'plate',
                17: 'shoes',
                18: 'spoon',
                19: 'tree'
}


parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, help="directory with image files to classify")
args = parser.parse_args()

n_valid, n_files = validate_images(os.path.abspath(args.img_dir))
print(f"From {n_files} files in your img_dir, {n_valid} are valid for classification\n"
      f"for more information on invalid images check the log_file.")

dataset = ImagesDatasetClassify(os.path.abspath("validated_images"), 100, 100, int)


dl = DataLoader(dataset=dataset, shuffle=False, batch_size=1)

exec(open(f"results\\architecture.py").read())
model.load_state_dict(torch.load(f"results\\model.pth", map_location=torch.device('cpu')))
model.eval()

prediction_lst = []
with torch.no_grad():
    for i, data in enumerate(dl):
        img, path = data
        y_prediction = model(img)
        _, predicted = torch.max(y_prediction.data, 1)
        label_prediction = labels_dict[int(predicted.numpy()[0])]
        new_name = str(i) + "_" + label_prediction + ".jpg"
        try:
            os.rename(src=path[0], dst=os.path.join(os.path.abspath("validated_images"), new_name))
        except FileExistsError:
            print(f"File {new_name} with path {path[0]} exists already")
        prediction_lst.append((os.path.basename(path[0]), label_prediction))
print(prediction_lst)
