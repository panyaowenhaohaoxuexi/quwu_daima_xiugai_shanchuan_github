import os

def clean_folder_b(folder_a, folder_b):
    """
    以文件夹A中的文件名为基准，删除文件夹B中多余的图像文件。
    """
    # 获取A、B文件夹中的文件名（不包含路径）
    files_a = set(os.listdir(folder_a))
    files_b = set(os.listdir(folder_b))

    # 找出B中多余的文件
    extra_files = files_b - files_a

    if not extra_files:
        print("文件夹B中没有多余的文件。")
        return

    # 删除多余文件
    for file_name in extra_files:
        file_path = os.path.join(folder_b, file_name)
        try:
            os.remove(file_path)
            print(f"已删除: {file_name}")
        except Exception as e:
            print(f"删除 {file_name} 时出错: {e}")

    print("清理完成。")

if __name__ == "__main__":
    folder_a = input("请输入文件夹A路径（基准文件夹）: ").strip()
    folder_b = input("请输入文件夹B路径（待清理文件夹）: ").strip()
    clean_folder_b(folder_a, folder_b)
