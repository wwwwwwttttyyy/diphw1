# Pix2Pix

## 准备数据

进入当前目录：

```powershell
cd .\Pix2Pix
New-Item -ItemType Directory -Force .\datasets | Out-Null
```

### Facades

```powershell
Invoke-WebRequest -Uri "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz" -OutFile ".\datasets\facades.tar.gz"
tar -xzf .\datasets\facades.tar.gz -C .\datasets

Get-ChildItem .\datasets\facades\train -Recurse -Filter *.jpg | Sort-Object FullName | ForEach-Object { $_.FullName } | Set-Content .\train_list.txt
Get-ChildItem .\datasets\facades\val -Recurse -Filter *.jpg | Sort-Object FullName | ForEach-Object { $_.FullName } | Set-Content .\val_list.txt
```

### Maps

如果想换成 `maps`，把上面的 `facades` 改成 `maps` 即可：

```powershell
Invoke-WebRequest -Uri "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz" -OutFile ".\datasets\maps.tar.gz"
tar -xzf .\datasets\maps.tar.gz -C .\datasets

Get-ChildItem .\datasets\maps\train -Recurse -Filter *.jpg | Sort-Object FullName | ForEach-Object { $_.FullName } | Set-Content .\train_list.txt
Get-ChildItem .\datasets\maps\val -Recurse -Filter *.jpg | Sort-Object FullName | ForEach-Object { $_.FullName } | Set-Content .\val_list.txt
```

## 训练

```powershell
python .\train.py
```

## 输出位置

- 训练结果：`train_results/`
- 验证结果：`val_results/`
- 模型权重：`checkpoints/`
