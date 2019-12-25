import os


def create_labels(path, set):
    files = os.listdir(os.path.join(path, set))
    # 拍雪后的文件夹名列表
    orderedfiles = sorted(files)
    with open('models/retrained_labels.txt', 'w')as f:
        for i, files in enumerate(orderedfiles):
            # 排除macOS系统文件
            if files == '.DS_Store':
                continue
            else:
                f.writelines(files)

            # 最后一行不加换行
            if i != len(orderedfiles) - 1:
                f.writelines('\n')


if __name__ == '__main__':
    create_labels('PlantsData', 'train')
