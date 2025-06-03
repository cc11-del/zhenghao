import subprocess
import os
import sys

def main():
    # 步骤1: 重新生成DOT文件，确保使用UTF-8编码
    print("步骤1: 重新生成DOT文件...")
    try:
        # 使用DVC命令生成新的DOT文件
        with open("clean_graph.dot", "w", encoding="utf-8") as f:
            subprocess.run(["dvc", "dag", "--dot"], stdout=f, check=True)
        print("成功生成新的DOT文件: clean_graph.dot")
    except Exception as e:
        print(f"生成DOT文件时出错: {e}")
        return

    # 步骤2: 尝试使用Graphviz的dot命令生成图像
    print("\n步骤2: 尝试使用Graphviz的dot命令生成图像...")
    
    # 可能的Graphviz安装路径
    graphviz_paths = [
        "C:\\Program Files\\Graphviz\\bin",
        "C:\\Graphviz\\bin",
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Graphviz", "bin")
    ]
    
    # 将可能的Graphviz路径添加到环境变量
    for path in graphviz_paths:
        if os.path.exists(path):
            os.environ["PATH"] = os.environ["PATH"] + ";" + path
            print(f"已添加Graphviz路径: {path}")
    
    try:
        # 使用subprocess调用dot命令，不捕获输出（避免编码问题）
        process = subprocess.Popen(
            ["dot", "-Tpng", "clean_graph.dot", "-o", "dvc_pipeline_graph.png"],
            shell=True  # 在Windows上使用shell=True可能更可靠
        )
        process.wait()
        
        if process.returncode == 0:
            print("成功生成图像: dvc_pipeline_graph.png")
        else:
            print(f"dot命令返回错误代码: {process.returncode}")
            
            # 尝试方法3
            print("\n步骤3: 尝试使用Python的pydot库...")
            try:
                import pydot
                graphs = pydot.graph_from_file("clean_graph.dot")
                if graphs:
                    graph = graphs[0]
                    graph.write_png("dvc_pipeline_graph.png")
                    print("成功使用pydot生成图像: dvc_pipeline_graph.png")
                else:
                    print("pydot无法解析DOT文件")
            except Exception as e:
                print(f"使用pydot时出错: {e}")
                
                # 最后的尝试：使用完整路径
                print("\n步骤4: 尝试使用dot的完整路径...")
                for path in graphviz_paths:
                    dot_path = os.path.join(path, "dot.exe")
                    if os.path.exists(dot_path):
                        try:
                            subprocess.run(
                                [dot_path, "-Tpng", "clean_graph.dot", "-o", "dvc_pipeline_graph.png"],
                                check=True
                            )
                            print(f"成功使用完整路径生成图像: {dot_path}")
                            return
                        except Exception as e:
                            print(f"使用完整路径 {dot_path} 时出错: {e}")
                
                print("所有方法都失败了。请确保Graphviz已正确安装。")
    except Exception as e:
        print(f"执行dot命令时出错: {e}")

if __name__ == "__main__":
    main()
