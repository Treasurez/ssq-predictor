import os
import urllib.request
import zipfile
import shutil

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'easyocr_models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_SOURCES = [
    {
        'name': 'craft_mlt_25k.pth',
        'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip',
        'min_size': 500000
    },
    {
        'name': 'zh_sim_g2.pth',
        'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/zh_sim_g2.zip',
        'min_size': 20000000
    },
    {
        'name': 'english_g2.pth',
        'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip',
        'min_size': 80000000
    }
]

def download_file(url, dest_path):
    print(f"下载: {url}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def extract_zip(zip_path, dest_dir):
    print(f"解压: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False

def main():
    print("=" * 60)
    print("EasyOCR 模型下载器")
    print("=" * 60)
    
    for model_info in MODEL_SOURCES:
        model_name = model_info['name']
        url = model_info['url']
        min_size = model_info['min_size']
        
        model_path = os.path.join(MODEL_DIR, model_name)
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            if file_size > min_size:
                print(f"✓ {model_name} 已存在 ({file_size/1024/1024:.1f} MB)")
                continue
            else:
                print(f"⚠ {model_name} 文件太小，重新下载...")
        
        zip_path = os.path.join(MODEL_DIR, model_name.replace('.pth', '.zip'))
        
        if download_file(url, zip_path):
            if extract_zip(zip_path, MODEL_DIR):
                os.remove(zip_path)
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    print(f"✓ {model_name} 下载完成 ({file_size/1024/1024:.1f} MB)")
                else:
                    print(f"✗ 未能找到解压后的 {model_name}")
            else:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
        else:
            print(f"✗ {model_name} 下载失败")
    
    print("\n" + "=" * 60)
    print("创建模型别名 (兼容旧版命名)...")
    print("=" * 60)
    
    aliases = [
        ('zh_sim_g2.pth', 'ch_sim.pth'),
        ('english_g2.pth', 'en.pth'),
        ('craft_mlt_25k.pth', 'detector.pth')
    ]
    
    for source, alias in aliases:
        source_path = os.path.join(MODEL_DIR, source)
        alias_path = os.path.join(MODEL_DIR, alias)
        
        if os.path.exists(source_path):
            if os.path.exists(alias_path):
                os.remove(alias_path)
            os.symlink(source, alias_path)
            print(f"✓ 创建别名: {alias} -> {source}")
        else:
            print(f"✗ 无法创建别名: {source} 不存在")
    
    print("\n" + "=" * 60)
    print("检查模型目录:")
    print("=" * 60)
    for f in os.listdir(MODEL_DIR):
        if f.endswith('.pth'):
            f_path = os.path.join(MODEL_DIR, f)
            if os.path.islink(f_path):
                target = os.readlink(f_path)
                print(f"  {f} -> {target} (符号链接)")
            else:
                size = os.path.getsize(f_path)
                print(f"  {f}: {size/1024/1024:.1f} MB")

if __name__ == '__main__':
    main()