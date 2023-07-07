import os
import shutil
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def backup_code(src_dir, backup_dir):
    if src_dir == config.work_dir:
        for root, dirs, files in os.walk(src_dir):
            if 'output' in dirs:
                dirs.remove('output')
            if root.endswith('output'):
                continue
            for file in files:
                if file.endswith('.py'):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(backup_dir, os.path.relpath(src_path, src_dir))
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copyfile(src_path, dst_path)
        print('backup complete')
    else:
        print('no backup')

