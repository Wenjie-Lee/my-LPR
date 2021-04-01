from xml.etree.ElementTree import ElementTree
from os import getcwd

train = './annotations/train.txt'
test = './annotations/test.txt'
train_anno = './annotations/train_anno.txt'
test_anno = './annotations/test_anno.txt'
class_txt = './my-tensorflow-yolo3/model_data/classes.txt'

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def convert_annotation(idx_cls, list_file, filename):
	# in_file = open(filename)

	with open(filename) as f:
		in_file = f.readlines()

	dirs = [c.strip() for c in in_file]

	for d in dirs:
		tree = ElementTree()
		# print(d)
		tree.parse(d)
		root = tree.getroot()
		list_file.write(root.find('path').text)
		print(root.find('path').text)
		for obj in root.iter('object'):
			difficult = obj.find('difficult').text
			cls_id = idx_cls[obj.find('name').text]
			if int(difficult)==1:
				continue
			# cls_id = classes.index(cls)
			xmlbox = obj.find('bndbox')
			b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
			list_file.write(" " + ",".join([str(a) for a in b]) + "," + str(cls_id))
		list_file.write("\n")


wd = getcwd()

class_names = get_classes(class_txt)
num_classes = len(class_names)
idx_cls = dict(zip(class_names, range(num_classes)))
# print(idx_cls)


list_file = open(train, 'w')
convert_annotation(idx_cls, list_file, train_anno)
list_file.close()
list_file = open(test, 'w')
convert_annotation(idx_cls, list_file, test_anno)
list_file.close()