import tensorflow as tf
import glob
import os.path
import time
from shutil import copy2

# 原始输入数据的目录，这个目录下有五个子目录，每个子目录底下保存属于该类别的所有图片
INPUT_DATA = 'H:/郑伟的空间/OurGroupSummerData/data/'

TRAIN_OUTPUT = 'H:/why_workspace/train/'
TEST_OUTPUT = 'H:/why_workspace/test/'


# 读取数据并将数据分割成训练数据测试数据(未处理的图像地址list)
def create_image_lists(sess):
    # 定义次一级文件夹目录列表
    sub_dirs = []
    # 获取根目录下的所有文件和文件夹目录
    sub_file = os.listdir(INPUT_DATA)
    for file in sub_file:
        m = os.path.join(INPUT_DATA, file)
        # 获取次一级文件夹目录
        if os.path.isdir(m):
            sub_dirs.append(m)

    # 读取所有子目录
    for sub_dir in sub_dirs:
        # 图片文件夹目录
        next_sub_dirs = os.listdir(sub_dir)

        print('next_sub_dirs is ', next_sub_dirs)

        # 获取一个子目录中所有的图片文件
        extensions = ['png']
        for next_sub_dir in next_sub_dirs:
            start_time = time.time()
            print("*****当前文件夹为：", next_sub_dir, " *****")

            file_list = []
            file_list.clear()

            for extension in extensions:
                # print('*****当前图片后缀为：', extension, " *****")
                # os.path.join()合成地址
                file_glob = os.path.join(sub_dir, next_sub_dir, '*.' + extension)
                # print("*****当前图片地址格式为：", file_glob, " *****")
                # glob.glob（）搜索所有符合要求的文件
                file_list.extend(glob.glob(file_glob))
                if not file_list:  # 如果file_list为空list，就判断为False
                    continue

                # 处理图片数据
                num_example = len(file_list)
                num_train = int(num_example * 0.8)
                # print("*****当前file_list的图片数量为：", num_example, "*****")
                # 创建目录文件夹
                os.mkdir(os.path.join(TRAIN_OUTPUT, next_sub_dir))
                os.mkdir(os.path.join(TEST_OUTPUT, next_sub_dir))
                for i in range(0, num_train):
                    png_name1 = os.path.basename(file_list[i])
                    train_output1 = os.path.join(TRAIN_OUTPUT, next_sub_dir, png_name1)
                    copy2(file_list[i], train_output1)

                for n in range(num_example - num_train):
                    png_name2 = os.path.basename(file_list[n + num_train])
                    train_output2 = os.path.join(TEST_OUTPUT, next_sub_dir, png_name2)
                    copy2(file_list[n + num_train], train_output2)
            end_time = time.time()
            print("the folder {0} takes time is {1}".format(next_sub_dir, end_time - start_time))



def main(argv=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    # 设置根据需求增长使用的内存
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        create_image_lists(sess)
    print("all finished")


if __name__ == '__main__':
    tf.app.run()

