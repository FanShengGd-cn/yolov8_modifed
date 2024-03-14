# 思路：
import json, shutil
from decimal import Decimal
import os, shutil


def json_to_txt_v8(path_A, path_B):
    set_A = set(i.split('.')[0] for i in os.listdir(path_A))

    for i in set_A:
        if i == "":
            pass
        else:
            name = os.path.join(path_A, i + '.json')
            # print("name", name)
            save_name = os.path.join(path_B, i + '.txt')
            # print("save_name", save_name)
            with open(name, 'r') as f:
                data = json.load(f)
                # print("" + data.get('version'))
                # print("flags略")
                # print("shapes略")
                # print("" + data.get('imagePath'))
                # print("imageData=略")
                # print("" + str(data['imageHeight']))
                # print("" + str(data['imageWidth']))
                H = data.get('imageHeight')
                W = data.get('imageWidth')
                classes = ["aster tataricus", "chrysanthemum", "digitalis purpurea", "glycyrrhiza uralensis",
                           "mint", "paeonia","salvia", "sanguisorba","saposhnikovia", "spot"]
                locals_items = data['shapes'][0].keys()
                # print("locals_items=", locals_items)
                for k in data['shapes']:
                    # bug  下标不在列表中
                    point_arr = []
                    label_index = list(classes).index(str(k['label']))
                    point_arr.append(str(label_index))
                    # print('标签索引序号',label_index)
                    points_contexts = (k['points'])
                    #   print(points_contexts)
                    #  print("点的个数=", len(points_contexts))
                    for i in range(len(points_contexts)):
                        x_Value = Decimal(str(round(float((points_contexts[i][0]) / (W)), 6))).quantize(
                            Decimal('0.000000'))
                        y_Value = Decimal(str(round(float((points_contexts[i][1]) / (H)), 6))).quantize(
                            Decimal('0.000000'))
                        point_arr.append(str(x_Value))
                        point_arr.append(str(y_Value))
                        # print(points_contexts[i])
                        # print(x_Value)
                        # print(y_Value)
                    # print(point_arr)
                    with open(save_name, 'a+') as ww:
                        for i in range(len(point_arr)):
                            #    print(point_arr[i])
                            ww.write(point_arr[i])
                            ww.write(' ')
                        ww.write('\n')


if __name__ == "__main__":
    root_path = r'D:\selectImg\2024_3_15dataset\val'
    path_A = 'json'
    path_B = 'labels'
    if not os.path.exists(os.path.join(root_path, path_A)):
        print("搞笑吗，需要转的json文件夹不存在，请查看路径")
    else:
        if not os.path.exists(os.path.join(root_path, path_B)):
            os.makedirs(os.path.join(root_path, path_B))
            json_to_txt_v8(os.path.join(root_path, path_A), os.path.join(root_path, path_B))
        else:
            shutil.rmtree(os.path.join(root_path, path_B))
            os.makedirs(os.path.join(root_path, path_B))
            json_to_txt_v8(os.path.join(root_path, path_A), os.path.join(root_path, path_B))
