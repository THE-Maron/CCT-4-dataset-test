import os
import pandas as pd
import glob

def main(dir, output):
    df = pd.DataFrame({'img_dir':[],
                        'label':[]})
    #dir = '.' + os.sep + dir + os.sep
    dir = os.path.join('.', dir, '*')
    fonts = glob.glob(dir)
    for font in fonts:
        font = os.path.join(font, '*')
        imgs = glob.glob(font)
        for img in imgs:
            label = img.split(os.sep)[2]
            row = pd.Series([img, label], index=df.columns)
            df = df.append(row, ignore_index=True)
    df.to_csv(output)
    return

if __name__ == '__main__':
    main('Test','Test.csv')
    main('Train','Train.csv')