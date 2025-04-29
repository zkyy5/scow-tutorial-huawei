# tutorial

## 贡献指南
欢迎贡献新教程或改进现有教程。请参考现有教程的格式，编写教程后提交pr，仓库管理员会不定期进行合并。

## issue
如果遇到问题，欢迎提交issue。

## 生成web页面

```bash
pip install notebook beautifulsoup4
conda install pandoc
conda install -c conda-forge parallel 
sudo apt-get update
sudo apt-get install texlive-xetex google-chrome-stable pdftk

bash release.sh
```
