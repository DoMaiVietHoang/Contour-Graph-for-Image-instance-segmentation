from dataset import TreeCrownGraphDataset

def Dataset_construction():
    Adjacebt_dir = './data/adjacent'
    LPIPS_dir = './data/lpips'
    Shape_dir = './data/shape'
    dataset = TreeCrownGraphDataset(Adjacebt_dir, LPIPS_dir, Shape_dir)
    return dataset
def main(args):
    dataset = Dataset_construction()
    print(dataset.len())
if __name__=="__main__":
    main()