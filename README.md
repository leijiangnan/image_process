# Image Grid Slicer

把九宫格或十二宫格图片切成独立小图。脚本只做裁切，不抠图、不改透明背景、不缩放，尽量保留原图清晰度和所有细节。

每张输出图都会裁成 `1:1` 正方形，并完整包含当前格子的内容。对于非正方形格子，脚本会把画布扩展成正方形；扩展出来的区域使用白底，不会带入相邻表情。若格子边缘本身带到相邻表情的细小残片，也会尽量替换成白底。最后会根据主体重新居中并收缩四周白边，同一组图片会统一输出为相同尺寸。

## 使用

```bash
python3 -m image_grid_slicer.cli "/path/to/source.png" -o output --grid 4x3
```

九宫格：

```bash
python3 -m image_grid_slicer.cli "/path/to/source.png" -o output --grid 3x3
```

也可以自动判断：

```bash
python3 -m image_grid_slicer.cli "/path/to/source.png" -o output
```

## 输出

```text
output/
  slices/
    01.png
    02.png
    ...
  manifest.json
```

`manifest.json` 会记录网格、每张小图尺寸和文件大小。输出小图应全部为正方形。
