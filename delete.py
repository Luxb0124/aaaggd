import os

def deal_file(file_path):
    pass
    
def deal_dir(chr_path):
    base_dir = os.path.basename(chr_path)
    if base_dir.startswith('.i') or base_dir.startswith('__'):
        cmd = 'rm -rf %s' %(chr_path)
        os.system(cmd)
        print(cmd)
    # elif base_dir == 'datasets':
    #     print('ignore %s' %(base_dir))
    else:        
        iterate_dir(chr_path, deal_dir=deal_dir, deal_file=deal_file)

def iterate_dir(rootdir, deal_dir=None, deal_file=None):
    for dir_or_file in os.listdir(rootdir):
        path = os.path.join(rootdir, dir_or_file)
        if os.path.isfile(path):
            if deal_file == 'pass':
                pass
            elif deal_file:
                deal_file(path)
            else:
                print(path)
        if os.path.isdir(path):
            if deal_dir == 'pass':
                pass
            elif deal_dir:
                deal_dir(path)
            else:
                print(path)

src_dir = '../aaaggd'
iterate_dir(src_dir, deal_dir, deal_file)
