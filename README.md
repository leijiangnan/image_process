# Image Grid Slicer

把九宫格或十二宫格图片切成独立小图。脚本只做裁切，不抠图、不改透明背景、不缩放，尽量保留原图清晰度和所有细节。

每张输出图都会裁成 `1:1` 正方形，并完整包含当前格子的内容。对于非正方形格子，脚本会把画布扩展成正方形；扩展出来的区域使用白底，不会带入相邻表情。若格子边缘本身带到相邻表情的细小残片，也会尽量替换成白底。最后会根据主体重新居中并收缩四周白边，同一组图片会统一输出为相同尺寸。

脚本还会额外输出三组 GIF 动图：一组为两帧轻微左右倾斜摆动，一组为人物轻微缩小放大，另一组为人物不动、文字和贴纸轻微缩小放大。

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
  gifs/
    01.gif
    02.gif
    ...
  zoom_gifs/
    01.gif
    02.gif
    ...
  text_blink_gifs/
    01.gif
    02.gif
    ...
  manifest.json
```

`gifs/` 为左右摆动版本，`zoom_gifs/` 为人物缩小放大版本，`text_blink_gifs/` 为人物不动、文字缩小放大版本。

`manifest.json` 会记录网格、每张 PNG、左右摆动 GIF、人物缩放 GIF、文字缩放 GIF 的路径、尺寸和文件大小。输出小图应全部为正方形。
