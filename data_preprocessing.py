from PIL import Image
import os
import random

def image_split(original_path,species,stardand_path,set,inds):
    for img in inds:
        #以原始路径+植物名字+序列号为图片路径打开图片
        plant_path = os.path.join(original_path,species,img)
        in_img = Image,open(plant_path)
        out_img = in_img.resize((224,224))
        #以原始路径+集名+植物名字为该文件集路径
        set_path = os.path.join(stardand_path,set,species)
        if not os.path.exists(set_path):
            os.markdirs(set_path)
        out_img.save(os.path.join(set_path,img))

def main():
    #读取图片
    original_path = './dataset/bjfu_plants'
    stardand_path ='./PlantsData'
    for species in os.listdir(original_path):
        files = sorted(os.listdir((os.path.join(original_path,species))))
        img_num = len(files)
        test_num = int(img_num/10*2)
        vaild_num = int((img_num-test_num)/10*2)
        trains_num = int(img_num-test_num-vaild_num)
        print('%s:%d,train:%d,valid:%d,test:%d'%(species,img_num,trains_num,vaild_num,test_num))

        range.shuffle(files)
        image_split(original_path,species,stardand_path,'train',files[0:trains_num])
        print("%s finish!"% os.path.join(stardand_path,'train',species))
        image_split(original_path,species,stardand_path,'valid',files[trains_num:trains_num+vaild_num])
        print("%s finish!"% os.path.join(stardand_path.join(stardand_path,'valid',species)))
        image_split(original_path,species,stardand_path,'train',files[trains_num+vaild_num:])
        print("%s finish!"%os.path.join(stardand_path,'test',species))


if __name__ == '__main__':
    main()

