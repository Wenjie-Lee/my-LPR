import os, shutil
import string

"""
将车牌的标注从车牌名字中提取出来
"025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
每个图片名字被 “-” 分割为7部分

需要提取的标注有：

Area: Area ratio of license plate area to the entire picture area.
车牌区域相比整个图像的大小：025

Tilt degree: Horizontal tilt degree and vertical tilt degree.
水平、垂直倾斜角度： 95°、113°
*** 若要使用PIL矫正车牌，落地时就要考虑C++；且训练的模型有足够好的泛化能力的话也就不需要矫正

Bounding box coordinates: The coordinates of the left-up and the right-bottom vertices.
标定框的左上角、右下角坐标：(154,383), (386,473)

Four vertices locations: The exact (x, y) coordinates of the four vertices of LP in the whole image. 
    These coordinates start from the right-bottom vertex.
标定框的四个顶点坐标：(386, 473), (177, 454), (154, 383), (363, 402)

License plate number: Each image in CCPD has only one LP
车牌具体号码：0_0_22_27_27_33_16 -> 皖 A Y 3 3 9 S
***We use 'LETTER O' as a sign of "no character" because there is no O in Chinese license plate characters.

Brightness: The brightness of the license plate region.

Blurriness: The Blurriness of the license plate region.


"""
# 车牌中不含字母“O”，只有数字“0”，所以此处字母“O”代表没检测到字符
characters = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", \
                "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", \
                "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", \
                "挂", "使", "港", "澳", "军", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', \
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', \
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y', 'Z', '0', '1', '2', \
            '3', '4', '5', '6', '7', '8', '9', 'O']

ccpd_base_dir = './ccpd_base/'
ccpd_np_dir = './ccpd_np/'
ccpd_fn_dir = './ccpd_fn/'
ccpd_db_dir = './ccpd_db/'
ccpd_rotate_dir = './ccpd_rotate/'
ccpd_tilt_dir = './ccpd_tilt/'
ccpd_weather_dir = './ccpd_weather/'
ccpd_blur_dir = './ccpd_blur/'
ccpd_challenge_dir = './ccpd_challenge/'

# 获取 fields list
def _split(name):
    assert name.endswith('.jpg')
    return name[:-4].split('-')

# 获取所有文件夹中图片的名字到 split.txt
def _walk_folder(srcpath):
    dirs = os.listdir(srcpath)
    for d in dirs:
        path = os.path.join(srcpath, d)
        if os.path.isdir(path):
            txt_dir = os.path.join(path, 'splits.txt')
            f = open(txt_dir, mode='w+')
            pics = os.listdir(path)
            for pic in pics:
                if pic.endswith('.jpg'):
                    output = '{name}\n'.format(name=pic)
                    f.write(output)
            f.close()
            print('{path} split success. \n'.format(path=path))


# 将图片都处理为 704 * 704 大小，都为32的倍数(22 * 22)
def _crop(srcpath, threshold=0.9):
    from PIL import Image

    # srcpath = 'E:/tools/Graduated/license plate/CCPD2019/ccpd_base'
    # srcpath = 'E:/tools/Graduated/license plate/CCPD2019/test/'
    pics = os.listdir(srcpath)
    for pic in pics:
        path = os.path.join(srcpath, pic)
        _, _, bbox, _, _, _, _ = _split(pic)
        bbox = bbox.split('_')
        # x1, y1 = bbox[0].split('&')
        x2, y2 = bbox[1].split('&')
        image = Image.open(path)
        iw, ih = image.size
        if iw==704 and ih==704:
            image.close()
            continue
        # print('{x}, {y}\n'.format(x=x2, y=y2))
        # 将真实框超出704范围，以及太靠近704的图片抛弃，不计入txt中
        if float(x2) < 704 * threshold and float(y2) < 704 * threshold:
            image = image.crop((0, 0, 704, 704))
            image.save(path)
    pass


# 按annotation文件格式，将图像名改为xml文件
def rename_with_annotation(srcpath, annotation_src='./annotations', cls='blue', number=0):
    from PIL import Image

    print('输出annotation文件...')
    files = os.listdir(srcpath)
    folderpath = srcpath.split('/')[-2]
    dst = os.path.join(annotation_src, folderpath)
    if not os.path.isdir(dst):
        os.mkdir(dst)

    i = 1
    length = len(files)
    # 输出annotation的个数
    import random
    random.shuffle(files)
    # number==0, 表示全部输出
    if number == 0 or number > length:
        number = length

    for file in files:
        if file.endswith('jpg'):
            # 裁剪或补足图片大小
            path = os.path.join(srcpath, file)
            image = Image.open(path)
            iw, ih = image.size
            if iw < 704 or ih < 704:
                padding = (0, 0, 704 - iw, 704 - ih)
                image = image.expand(image, padding)
            image.save(path)
            # 解析文件名
            # 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
            # 标定框的左上角、右下角坐标：(154,383), (386,473)
            _, _, bbox, _, _, _, _ = _split(file)
            bbox = bbox.split('_')
            xmin, ymin = bbox[0].split('&')
            xmax, ymax = bbox[1].split('&')

            # 重命名图像文件
            # newpath = os.path.join(srcpath, str(i) + '.jpg')
            # os.rename(path, newpath)

            # 构造xml
            from xml.etree.ElementTree import Element
            from xml.etree.ElementTree import SubElement
            from xml.etree.ElementTree import ElementTree

            from xml.dom import minidom

            anno = Element('annotation')

            folder = SubElement(anno, 'folder')
            folder.text = folderpath
            filename = SubElement(anno, 'filename')
            _filename = file.split('.')[0]
            filename.text = _filename
            

            filepath = SubElement(anno, 'path')
            filepath.text = path

            source = SubElement(anno, 'source')
            database = SubElement(source, 'database')
            database.text = 'Unknown'

            size = SubElement(anno, 'size')
            width = SubElement(size, 'width')
            width.text  = str(iw)
            height = SubElement(size, 'height')
            height.text  = str(ih)
            depth = SubElement(size, 'depth')
            depth.text  = '3'

            segmented = SubElement(anno, 'segmented')
            segmented.text = '0'

            obj = SubElement(anno, 'object')
            label = SubElement(obj, 'name')
            label.text = cls
            pose = SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            truncated = SubElement(obj, 'truncated')
            truncated.text = '0'
            difficult = SubElement(obj, 'difficult')
            difficult.text = '0'
            bndbox = SubElement(obj, 'bndbox')
            _xmin = SubElement(bndbox, 'xmin')
            _xmin.text = xmin
            _ymin = SubElement(bndbox, 'ymin')
            _ymin.text = ymin
            _xmax = SubElement(bndbox, 'xmax')
            _xmax.text = xmax
            _ymax = SubElement(bndbox, 'ymax')
            _ymax.text = ymax

            # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行   
            def prettyXml(element, indent, newline, level = 0): 
                # 判断element是否有子元素
                if element:
                    # 如果element的text没有内容      
                    if element.text == None or element.text.isspace():     
                        element.text = newline + indent * (level + 1)      
                    else:    
                        element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)    
                # 此处两行如果把注释去掉，Element的text也会另起一行 
                #else:     
                    #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level    
                temp = list(element) # 将elemnt转成list    
                for subelement in temp:    
                    # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
                    if temp.index(subelement) < (len(temp) - 1):     
                        subelement.tail = newline + indent * (level + 1)    
                    else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个    
                        subelement.tail = newline + indent * level   
                    # 对子元素进行递归操作 
                    prettyXml(subelement, indent, newline, level = level + 1)   

            prettyXml(anno, '\t', '\n')
            anno_name = 'anno_' + folderpath + '_' + _filename + '.xml'

            # _path = annotation_src
            # if i < number * train_split:
            #     _path = os.path.join(annotation_src, 'train')
            # elif i < number * (train_split + val_split):
            #     _path = os.path.join(annotation_src, 'val')
            # else:
            #     _path = os.path.join(annotation_src, 'test')
            save_path = os.path.join(dst, anno_name)
            tree = ElementTree(anno)
            tree.write(save_path, encoding='utf-8')
            
            i += 1
            # 取够数量就退出
            if i > number:
                break


# 重命名黑白黄车牌的annotation文件
def rename_bwy_annotations(srcpath):
    from xml.etree.ElementTree import ElementTree
    from xml.dom.minidom import parse
    import xml.dom.minidom

    files = os.listdir(srcpath)
    for file in files:
        if file.endswith('.xml'):
            # 解析xml文件
            path = os.path.join(srcpath, file)
            tree = ElementTree()
            tree.parse(path)
            root = tree.getroot()

            folder = srcpath.split('/')[-2]
            foldertab = root.find('folder')
            foldertab.text = folder
            pathtab = root.find('path')
            pathtab.text = path
            save_name = 'anno_' + folder + '_' + file.split('.')[0] + '.xml'
            tree.write(os.path.join(srcpath, save_name), encoding='utf-8')
    

# 移动annotation文件到指定文件夹
def copy_annotations(srcpath, annotation_src='./annotations', number=0):
    files = os.listdir(srcpath)
    length = len(files)

    import random
    random.shuffle(files)

    i = 1
    train_split = 0.8
    val_split = 0.1

    # number==0, 表示全部输出
    if number == 0 or number >= length:
        number = length

    print('move %s annotation files...' % srcpath)
    for file in files:
        if file.endswith('xml'):
            src = os.path.join(srcpath, file)
            dst = annotation_src
            if i <= number * train_split:
                dst = os.path.join(annotation_src, 'train')
            elif i <= number * (train_split + val_split):
                dst = os.path.join(annotation_src, 'val')
            else:
                dst = os.path.join(annotation_src, 'test')

            if not os.path.isdir(dst):
                os.mkdir(dst)
            # shutil.move(src, dst)
            shutil.copy(src, dst)
            i += 1
            if i > number:
                break
    print('move complete...')

def _main():
    _walk_folder(ccpd_base_dir)
    # _walk_folder(ccpd_fn_dir)
    # _walk_folder(ccpd_db_dir)
    # _walk_folder(ccpd_rotate_dir)
    # _walk_folder(ccpd_tilt_dir)
    # _walk_folder(ccpd_weather_dir)
    # _walk_folder(ccpd_blur_dir)
    # _walk_folder(ccpd_challenge_dir)
    pass


if __name__ == '__main__':
    # _main()
    # _crop('./ccpd_base', threshold=0.9)
    #rename_with_annotation(ccpd_base_dir, number=0)
    #rename_with_annotation(ccpd_blur_dir, number=0)
    #rename_with_annotation(ccpd_weather_dir, number=0)
    #rename_with_annotation(ccpd_rotate_dir, number=0)
    #rename_with_annotation(ccpd_tilt_dir, number=0)
    #rename_with_annotation(ccpd_challenge_dir, number=0)
    #rename_with_annotation(ccpd_db_dir, number=0)
    #rename_with_annotation(ccpd_fn_dir, number=0)
    # rename_with_annotation('./ccpd_green/', number=0)
    # rename_bwy_annotations('./ccpd_bp/')
    # rename_bwy_annotations('./ccpd_wp/')
    # rename_bwy_annotations('./ccpd_yp/')
    # copy_annotations('./annotations/ccpd_yp/', number=0)
    # copy_annotations('./annotations/ccpd_wp/', number=0)
    # copy_annotations('./annotations/ccpd_bp/', number=0)
    # copy_annotations('./annotations/ccpd_base/', number=1000)
    # copy_annotations('./annotations/ccpd_green/', number=500)
