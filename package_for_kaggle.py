import zipfile
import os

def package_project(output_filename='bishe_code.zip'):
    # 1. 定义需要排除的文件夹（大文件夹、环境文件夹、非训练代码）
    exclude_dirs = {
        '.git',              # Git 仓库
        '__pycache__',       # Python 缓存
        '.idea', '.vscode',  # 编辑器配置
        'node_modules',      # 前端依赖
        'Straw6D_Raw',       # 【核心排除】原始数据集文件夹
        'Straw6D_Upload',    # 备份/上传文件夹
        'frontend',          # 前端代码（训练不需要）
        'dist', 'build'      # 编译产物
    }
    
    # 2. 定义需要排除的文件后缀
    exclude_extensions = {
        '.pth', '.ckpt', '.h5',  # 模型权重文件
        '.zip', '.rar', '.7z',    # 压缩包
        '.doc', '.docx', '.pdf',  # 文档
        '.pyc', '.pyo'            # 编译后的 Python 文件
    }

    # 3. 定义需要排除的特定大文件
    exclude_files = {
        'Straw6D_Raw.zip', 
        'Straw6D.zip',
        'darknet_strawberry_checkpoint.pth'
    }

    print(f"开始打包项目到 {output_filename}...")
    total_files_packed = 0
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # 过滤掉不需要的文件夹（通过修改 dirs 列表实现原地过滤）
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                # 检查后缀过滤
                ext_match = any(file.endswith(ext) for ext in exclude_extensions)
                # 检查特定文件过滤
                file_match = file in exclude_files
                
                if ext_match or file_match:
                    continue
                
                file_path = os.path.join(root, file)
                # 相对路径，用于 zip 内的结构保持一致
                arcname = os.path.relpath(file_path, '.')
                
                # 排除脚本自身
                if file == output_filename or file == 'package_for_kaggle.py':
                    continue
                
                # 写入压缩包
                zipf.write(file_path, arcname)
                total_files_packed += 1
                
    if os.path.exists(output_filename):
        print(f"打包完成！")
        print(f">>> 已排除数据集文件夹: Straw6D_Raw")
        print(f">>> 已排除权重文件: *.pth")
        print(f">>> 共打包文件数量: {total_files_packed}")
        print(f">>> 压缩包最终大小: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")
    else:
        print("!!! 打包失败，未生成文件。")

if __name__ == '__main__':
    package_project()
