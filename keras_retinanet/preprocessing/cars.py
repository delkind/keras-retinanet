import os


def process_dataset(root):
    train_set = set(open(root + '/ImageSets/train.txt').read().splitlines())
    images = [
        (root + '/Images/' + img, open(root + '/Annotations/' + os.path.splitext(img)[0] + '.txt').read().splitlines())
        for img in os.listdir(root + '/Images') if os.path.splitext(img)[0] in train_set]
    return images


images = process_dataset('../datasets/PUCPR+_devkit/data')
images += process_dataset('../datasets/CARPK_devkit/data')
images = {k: [[int(n) for n in s.split()] for s in v] for k, v in images}

csv = ['{},,,,,'.format(img) for img, v in images.items() if not v] + \
      [','.join([image] + [str(b) for b in box]) for image, boxes in images.items()
       for box in boxes if box[2] - box[0] > 0 and box[3] - box[1] > 0]

with open('annotations.csv', 'wt') as outfile:
    for line in csv:
        print(line, file=outfile)

with open('classes.csv', 'wt') as outfile:
    print('1,1', file=outfile)
