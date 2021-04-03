from glob import glob
import random
import argparse
import json
import os

def split_list(curr_list,split_rate):
    sr = int(split_rate*len(curr_list))
    list_1 = curr_list[:sr]
    list_2 = curr_list[sr:]
    return list_1, list_2

def generate_txt_files(args):
    path = args.dataset_path
    out_path = args.txt_files_path
    split_rate =args.split_rate
    ext = args.file_extention
    
    image_list = []
    temp_tr_val = []
    txt_train = []
    txt_val = []
    txt_test = []

    for filename in sorted(glob(path + "/*."+ext)):
        image_list.append(filename)
    random.shuffle(image_list)
    
    print(len(image_list))
    temp_tr_val, txt_test = split_list(image_list,split_rate)
    txt_train,txt_val = split_list(temp_tr_val,split_rate)
    print(len(txt_train))

    with open(os.path.join(out_path , "train.txt"), "w") as trainhandle:
        for line in txt_train:
            trainhandle.write(line)
            trainhandle.write("\n")              
    trainhandle.close()
    
    with open(os.path.join(out_path , "val.txt"), "w") as valhandle:
        for line in txt_val:
            valhandle.write(line)
            valhandle.write("\n")        
    valhandle.close()
    
    with open(os.path.join(out_path , "test.txt"), "w") as testhandle:
        for line in txt_test:
            testhandle.write(line)
            testhandle.write("\n")        
    testhandle.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generating data split txt files")

    # General settings
    parser.add_argument("--dataset_path", default= "/projects/patho1/sparse_segmentation/dataset/cropped/originals",
                        help = "path to the dataset")
    parser.add_argument("--txt_files_path", default=  "/projects/patho1/sparse_segmentation/dataset/split/stage_1/",
                        help = "path to the output txt files")
    parser.add_argument('--split_rate', default=0.8,
                    type=float, help='train split rate')    
    parser.add_argument("--file_extention", default="png",
                        help="file extention without '.' ")
    
    args = parser.parse_args()
    generate_txt_files(args)              

